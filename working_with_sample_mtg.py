# ================================================================
# MTG FCI CPP daily rainfall accumulation over Ghana
# Compare precip vs precip_ir
# xarray + Cartopy, discrete colorbar
# ================================================================

from pathlib import Path
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ================================================================
# 1. SETTINGS
# ================================================================

base_dir = Path(
    "/home/kkumah/Projects/cml-stuff/mtg-sample/transfer_3602065_files_d301e5bf"
)

out_dir = base_dir / "plots_mtg_precip"
out_dir.mkdir(parents=True, exist_ok=True)

ghana_extent = [-4.0, 2.0, 4.0, 12.0]

precip_vars = ["precip", "precip_ir"]

rain_levels = np.array([
    0, 1, 2, 5, 10, 20, 40, 60, 80, 100,
    125, 150, 175, 200, 250, 300
])

# Discrete colormap with overflow bin for values > 300 mm
cmap = plt.get_cmap("turbo", len(rain_levels))
norm = mcolors.BoundaryNorm(rain_levels, cmap.N, extend="max")

# ================================================================
# 2. HELPER FUNCTIONS
# ================================================================

def open_mtg_file(fp):
    return xr.open_dataset(
        fp,
        engine="h5netcdf",
        decode_cf=True,
        mask_and_scale=True,
    )


def get_duration_hours(ds):
    """
    Use time_bnds to get file duration in hours.
    """
    tb = ds["time_bnds"].isel(time=0)
    start = tb.isel(bnds=0).values
    end = tb.isel(bnds=1).values
    duration_hours = (end - start) / np.timedelta64(1, "h")

    return start, end, float(duration_hours)


def get_ghana_slices(lon, lat, extent):
    """
    Get y/x slices around Ghana using 2D lat/lon.
    """
    lon_min, lon_max, lat_min, lat_max = extent

    mask = (
        (lon >= lon_min) & (lon <= lon_max)
        & (lat >= lat_min) & (lat <= lat_max)
    )

    yy, xx = np.where(mask.values)

    if len(yy) == 0:
        raise ValueError("No pixels found inside Ghana extent.")

    y0, y1 = yy.min(), yy.max() + 1
    x0, x1 = xx.min(), xx.max() + 1

    return slice(y0, y1), slice(x0, x1)


def add_map_features(ax, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.coastlines(resolution="10m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)
    ax.add_feature(cfeature.LAKES, linewidth=0.35, alpha=0.5)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.5)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.25,
        alpha=0.45,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False


def valid_stats(da):
    valid = int(da.notnull().sum().values)

    if valid == 0:
        return valid, np.nan, np.nan, np.nan

    return (
        valid,
        float(da.min(skipna=True).values),
        float(da.mean(skipna=True).values),
        float(da.max(skipna=True).values),
    )


# ================================================================
# 3. FIND FILES
# ================================================================

nc_files = sorted(base_dir.rglob("*.nc"))

if len(nc_files) == 0:
    raise FileNotFoundError(f"No .nc files found under {base_dir}")

print(f"Number of .nc files found: {len(nc_files)}")
print(f"First file: {nc_files[0].name}")
print(f"Last file:  {nc_files[-1].name}")


# ================================================================
# 4. GET GHANA CROP FROM FIRST FILE
# ================================================================

with open_mtg_file(nc_files[0]) as ds0:
    y_slice, x_slice = get_ghana_slices(
        lon=ds0["lon"],
        lat=ds0["lat"],
        extent=ghana_extent,
    )

    lon_g = ds0["lon"].isel(y=y_slice, x=x_slice).load()
    lat_g = ds0["lat"].isel(y=y_slice, x=x_slice).load()

print("Ghana crop shape:", lon_g.shape)


# ================================================================
# 5. DAILY ACCUMULATION OVER GHANA
# ================================================================

daily_accum = {}
valid_counts = {}
valid_hours = {}

file_starts = []
file_ends = []

for i, fp in enumerate(nc_files, start=1):
    try:
        with open_mtg_file(fp) as ds:
            start, end, duration_hours = get_duration_hours(ds)

            file_starts.append(start)
            file_ends.append(end)

            for v in precip_vars:
                if v not in ds:
                    continue

                # Rain rate: kg m-2 h-1, equivalent to mm h-1
                rate_g = ds[v].isel(time=0, y=y_slice, x=x_slice).load()

                # Convert each 10-min rain-rate field to accumulation
                accum_g = rate_g * duration_hours

                valid = accum_g.notnull()

                if v not in daily_accum:
                    daily_accum[v] = xr.zeros_like(accum_g, dtype=np.float32).where(False)
                    valid_counts[v] = xr.zeros_like(accum_g, dtype=np.int16)
                    valid_hours[v] = xr.zeros_like(accum_g, dtype=np.float32)

                daily_accum[v] = daily_accum[v].fillna(0) + accum_g.fillna(0)
                valid_counts[v] = valid_counts[v] + valid.astype(np.int16)
                valid_hours[v] = valid_hours[v] + valid.astype(np.float32) * duration_hours

        if i % 20 == 0 or i == len(nc_files):
            print(f"Processed {i}/{len(nc_files)} files")

    except Exception as e:
        warnings.warn(f"Skipping file:\n{fp}\nReason: {e}")


# Mask pixels where no valid retrieval contributed
for v in daily_accum:
    daily_accum[v] = daily_accum[v].where(valid_counts[v] > 0)


# ================================================================
# 6. PRINT SUMMARY
# ================================================================

print("\nDaily accumulation summary over Ghana:")

for v in precip_vars:
    if v not in daily_accum:
        print(f"\n{v}: not found")
        continue

    count, vmin, vmean, vmax = valid_stats(daily_accum[v])

    print(f"\n{v}")
    print(f"  valid pixels: {count}")
    print(f"  min:  {vmin:.3f} mm")
    print(f"  mean: {vmean:.3f} mm")
    print(f"  max:  {vmax:.3f} mm")


# ================================================================
# 7. PLOT precip VS precip_ir
# Always show both panels, even if one is fully missing
# ================================================================

plot_vars = ["precip", "precip_ir"]

# Cleaner rainfall classes
rain_levels = np.array([
    0, 1, 5, 10, 20, 40, 60, 80, 100,
    150, 200, 250, 300
])

# Need one extra color because extend="max" adds an overflow bin
cmap = plt.get_cmap("turbo", len(rain_levels))
norm = mcolors.BoundaryNorm(rain_levels, cmap.N, extend="max")

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 5.8),
    subplot_kw={"projection": ccrs.PlateCarree()},
    constrained_layout=True,
)

last_mesh = None

for ax, v in zip(axes, plot_vars):
    add_map_features(ax, ghana_extent)

    if v in daily_accum:
        da = daily_accum[v]
        count, vmin, vmean, vmax_val = valid_stats(da)

        last_mesh = ax.pcolormesh(
            lon_g,
            lat_g,
            da,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap=cmap,
            norm=norm,
        )

        if count > 0:
            ax.set_title(
                f"{v}\nmean={vmean:.1f} mm, max={vmax_val:.1f} mm",
                fontsize=11,
            )
        else:
            ax.set_title(
                f"{v}\nno valid data",
                fontsize=11,
            )

    else:
        # If the variable does not exist in the files
        ax.set_title(
            f"{v}\nnot found",
            fontsize=11,
        )

# Shared colorbar
cbar = fig.colorbar(
    last_mesh,
    ax=axes,
    orientation="horizontal",
    fraction=0.045,
    pad=0.06,
    shrink=0.65,
    ticks=[0, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300],
    extend="max",
)

cbar.set_label("Daily rainfall accumulation (mm)", fontsize=10)
cbar.ax.tick_params(labelsize=9)

fig.suptitle(
    "MTG FCI CPP daily precipitation over Ghana",
    fontsize=14,
)

fig_path = out_dir / "daily_precip_vs_precip_ir_ghana.png"
fig.savefig(fig_path, dpi=220, bbox_inches="tight")
plt.show()

print("Saved plot:")
print(fig_path)


#%%
# ================================================================
# Plot MTG FCI CPP precipitation intensity for one time slice
# precip variable only, units: kg m-2 h-1 ≈ mm h-1
# ================================================================

from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ----------------------------------------------------------------
# Settings
# ----------------------------------------------------------------
base_dir = Path(
    "/home/kkumah/Projects/cml-stuff/mtg-sample/transfer_3602065_files_d301e5bf"
)

out_dir = base_dir / "plots_mtg_precip"
out_dir.mkdir(parents=True, exist_ok=True)

# Choose one time slice
target_time = np.datetime64("2026-03-29T07:00")

# Ghana view
ghana_extent = [-4.0, 2.0, 4.0, 12.0]

# Wider Africa view, similar to your reference image
africa_extent = [-20.0, 55.0, -35.0, 38.0]

# Pick which extent to use
plot_extent = ghana_extent
# plot_extent = africa_extent

varname = "precip"


# ----------------------------------------------------------------
# Discrete intensity color levels, mm h-1
# Similar spirit to the reference image
# ----------------------------------------------------------------
intensity_levels = np.array([
    0.3, 1, 2, 4, 8, 16, 32,
])

cmap = plt.get_cmap("magma_r", len(intensity_levels))
norm = mcolors.BoundaryNorm(
    intensity_levels,
    cmap.N,
    extend="max"
)


# ----------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------
def open_mtg_file(fp):
    return xr.open_dataset(
        fp,
        engine="h5netcdf",
        decode_cf=True,
        mask_and_scale=True,
    )


def get_file_start_end(ds):
    tb = ds["time_bnds"].isel(time=0)
    start = tb.isel(bnds=0).values
    end = tb.isel(bnds=1).values
    return start, end


def find_file_for_time(nc_files, target_time):
    """
    Find the MTG granule whose time_bnds contains target_time.
    """
    for fp in nc_files:
        with open_mtg_file(fp) as ds:
            start, end = get_file_start_end(ds)

        if start <= target_time < end:
            return fp, start, end

    raise FileNotFoundError(
        f"No file found containing target_time={target_time}"
    )


def crop_to_extent_2d(da, lon, lat, extent):
    lon_min, lon_max, lat_min, lat_max = extent

    mask = (
        (lon >= lon_min) & (lon <= lon_max)
        & (lat >= lat_min) & (lat <= lat_max)
    )

    yy, xx = np.where(mask.values)

    if len(yy) == 0:
        raise ValueError("No pixels found inside requested extent.")

    y0, y1 = yy.min(), yy.max() + 1
    x0, x1 = xx.min(), xx.max() + 1

    return (
        da.isel(y=slice(y0, y1), x=slice(x0, x1)),
        lon.isel(y=slice(y0, y1), x=slice(x0, x1)),
        lat.isel(y=slice(y0, y1), x=slice(x0, x1)),
    )


def add_map_features(ax, extent):
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    ax.coastlines(resolution="10m", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6)
    ax.add_feature(cfeature.LAKES, linewidth=0.3, alpha=0.45)
    ax.add_feature(cfeature.RIVERS, linewidth=0.25, alpha=0.45)

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.25,
        alpha=0.4,
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False


def valid_stats(da):
    valid = int(da.notnull().sum().values)

    if valid == 0:
        return valid, np.nan, np.nan, np.nan

    return (
        valid,
        float(da.min(skipna=True).values),
        float(da.mean(skipna=True).values),
        float(da.max(skipna=True).values),
    )


# ----------------------------------------------------------------
# Find target file
# ----------------------------------------------------------------
nc_files = sorted(base_dir.rglob("*.nc"))

if len(nc_files) == 0:
    raise FileNotFoundError(f"No .nc files found under {base_dir}")

fp, start, end = find_file_for_time(nc_files, target_time)

print("Selected file:")
print(fp)
print("Time coverage:", start, "to", end)


# ----------------------------------------------------------------
# Read selected file and crop
# ----------------------------------------------------------------
with open_mtg_file(fp) as ds:
    if varname not in ds:
        raise KeyError(f"{varname} not found in selected file.")

    precip_rate = ds[varname].isel(time=0)

    lon = ds["lon"]
    lat = ds["lat"]

    precip_g, lon_g, lat_g = crop_to_extent_2d(
        da=precip_rate,
        lon=lon,
        lat=lat,
        extent=plot_extent,
    )

    precip_g = precip_g.load()
    lon_g = lon_g.load()
    lat_g = lat_g.load()

count, vmin, vmean, vmax = valid_stats(precip_g)

print(f"\n{varname} intensity stats over selected extent")
print("valid pixels:", count)
print("min:", vmin)
print("mean:", vmean)
print("max:", vmax)


# ----------------------------------------------------------------
# Plot
# ----------------------------------------------------------------
fig = plt.figure(figsize=(7, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

add_map_features(ax, plot_extent)
vmin = 0.3
vmax = 32.0 
mesh = ax.pcolormesh(
    lon_g,
    lat_g,
    precip_g,
    transform=ccrs.PlateCarree(),
    shading="auto",
    cmap= cmap,
    norm=norm,
    # vmin=vmin,
    # vmax=vmax
)

title_time = np.datetime_as_string(start, unit="m").replace("T", " ")

ax.set_title(
    f"MTG FCI CPP precipitation intensity\n"
    f"{title_time} UTC",
    fontsize=13,
)

cbar = plt.colorbar(
    mesh,
    ax=ax,
    orientation="vertical",
    fraction=0.035,
    pad=0.035,
    ticks=intensity_levels,
    extend="max",
)

cbar.set_label("Rainfall intensity (mm h$^{-1}$)", fontsize=11)
cbar.ax.tick_params(labelsize=10)

fig_path = out_dir / f"mtg_cpp_precip_intensity_{np.datetime_as_string(start, unit='m').replace(':', '')}.png"
fig.savefig(fig_path, dpi=220, bbox_inches="tight")
plt.show()

print("\nSaved plot:")
print(fig_path)