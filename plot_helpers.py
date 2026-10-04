import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs, cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

# --- helpers ---------------------------------------------------------------
def _get_da(grid, varname="R_mm_per_h"):
    """Return an xarray.DataArray from DataArray or Dataset."""
    # Dataset -> pick named var or first variable
    if hasattr(grid, "data_vars"):
        if varname in grid.data_vars:
            return grid[varname]
        # fallback to first variable
        return next(iter(grid.data_vars.values()))
    # DataArray -> as-is
    if hasattr(grid, "values"):
        return grid
    raise TypeError("R_grid must be an xarray DataArray or Dataset")

def _slice_time(da, t):
    """Return (2D slice, label_time). If no time dim, just return da."""
    if "time" in getattr(da, "dims", ()):
        if t is None:
            # pick middle time as a reasonable default
            t_sel = pd.to_datetime(str(da["time"].values[len(da["time"])//2]))
        else:
            t_sel = pd.Timestamp(t)
        if t_sel.tzinfo is not None:
            t_sel = t_sel.tz_convert("UTC").tz_localize(None)
        sl = da.sel(time=np.datetime64(t_sel), method="nearest")
        return sl, t_sel
    return da, t  # 2-D already

# --- main plotter ----------------------------------------------------------
def plot_slice_cartopy_with_links(
    R_grid,
    meta_df,
    t=None,
    *,
    # classes: either give 'bounds' OR (vmin/vmax + nbins)
    bounds=None, vmin=None, vmax=None, nbins=12, spacing="linear",
    extend="max",
    extent,#=(-3.25, 1.2, 4.8, 11.15),#(-3.5, 1.25, 4.6, 11.25),
    cmap_name="turbo",
    title="",
    title_prefix="time = ",
    # aesthetics
    fig_size=(6.6, 8.2),
    coast_lw=0.7,
    borders_lw=0.6,
    grid_lw=0.45,
    link_color="k",
    link_lw=0.55,
    link_alpha=0.8,
    # colorbar placement
    cbar_side="right",       # "right"|"left"|"bottom"|"top"
    cbar_size="4%",          # width/height of colorbar
    cbar_pad=0.2,            # gap between map and colorbar (in axes fraction)
                # show every Nth class edge as a tick
    label_size=10,
):
    """
    Clean, readable rainfall map with discrete colors and link overlays.
    """

    da = _get_da(R_grid, "R_mm_per_h")
    sl, t_lbl = _slice_time(da, t)

    # ---- class edges (discrete) ----
    if bounds is None:
        data = np.asarray(sl.values, float)
        dmin = float(np.nanmin(data)) if np.isfinite(data).any() else 0.0
        dmax = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
        lo = dmin if vmin is None else float(vmin)
        hi = dmax if vmax is None else float(vmax)
        if spacing.lower().startswith("log"):
            lo = max(lo, 1e-6)
            edges = np.geomspace(lo, max(hi, lo*1.001), int(nbins)+1)
        else:
            edges = np.linspace(lo, hi, int(nbins)+1)
        bounds = np.asarray(edges, float)
    else:
        bounds = np.asarray(bounds, float)

    # ---- map canvas ----
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw={"projection": proj})
    ax.set_extent(extent, crs=proj)

    # subtle background for context
    ax.add_feature(cfeature.OCEAN, facecolor="#e9eef5", zorder=0)
    ax.add_feature(cfeature.LAND,  facecolor="#f7f7f7", zorder=0)
    ax.coastlines(resolution="10m", color="k", linewidth=coast_lw)
    ax.add_feature(cfeature.BORDERS, linewidth=borders_lw, edgecolor="k")

    # gridlines + nicely formatted degree ticks on axes
    lonmin, lonmax, latmin, latmax = extent
    lon_ticks = np.arange(np.floor(lonmin), np.ceil(lonmax)+1, 1.0)
    lat_ticks = np.arange(np.floor(latmin), np.ceil(latmax)+1, 1.0)
    ax.set_xticks(lon_ticks, crs=proj)
    ax.set_yticks(lat_ticks, crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".0f"))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".0f"))
    ax.tick_params(labelsize=label_size-1)
    gl = ax.gridlines(draw_labels=False, linewidth=grid_lw, color="gray",
                      alpha=0.5, linestyle="--")

    # colormap & norm
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_under("white"); cmap.set_bad("white")
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=cmap.N, extend=extend)

    # draw raster
    z = np.ma.masked_invalid(sl.values)
    pc = ax.pcolormesh(sl["lon"], sl["lat"], z, transform=proj,
                       cmap=cmap, norm=norm, shading="nearest", rasterized=True)

    # link overlays
    m = meta_df.copy()
    for c in ["XStart","YStart","XEnd","YEnd"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    segs = np.stack(
        [m[["XStart","YStart"]].to_numpy(),
         m[["XEnd","YEnd"]].to_numpy()], axis=1
    )
    ax.add_collection(LineCollection(segs, colors=link_color,
                                     linewidths=link_lw, alpha=link_alpha,
                                     transform=proj, zorder=3))

    # title
    if title:
        ttl = title
    else:
        t_str = "" if (t_lbl is None or str(t_lbl) == "NaT") else pd.to_datetime(t_lbl).strftime("%Y-%m-%d %H:%M:%S")
        ttl = f"{title_prefix}{t_str}"
    ax.set_title(ttl, fontsize=14, weight="bold", pad=8)

    # colorbar (on a normal Axes, not GeoAxes)
    divider = make_axes_locatable(ax)
    orient = "vertical" if cbar_side in ("right","left") else "horizontal"
    cax = divider.append_axes(cbar_side, size=cbar_size, pad=cbar_pad, axes_class=plt.Axes)
    cb = fig.colorbar(pc, cax=cax, orientation=orient, spacing="proportional")
    # ticks at regular intervals
    tick_min = np.floor(bounds[0])
    tick_max = np.ceil(bounds[-1])
    tick_interval = max(1, int((tick_max - tick_min) / nbins))
    ticks = np.arange(tick_min, tick_max + tick_interval, tick_interval)
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=label_size, direction='out')  # Ensure ticks point outward
    cb.set_ticks(cb.get_ticks())  # Ensure ticks are only placed where labels exist
    cb.set_label("Rainfall Intensity [mm h$^{-1}$]", fontsize=label_size, labelpad=6)

    fig.tight_layout()
    return fig, ax
