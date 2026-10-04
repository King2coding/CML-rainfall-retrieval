#%% Package imports
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import gc
import os
import re

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

from datetime import datetime, timedelta

from rasterio.enums import Resampling
import rioxarray
from sklearn.metrics import confusion_matrix

#%% Define working directory
imerg_dir = '/home/kkumah/Projects/cml-stuff/satellite_data/imergv07/data'
era5_dir = '/home/kkumah/Projects/cml-stuff/satellite_data/era5'
cml_sat_15min_dir = r'/home/kkumah/Projects/cml-stuff/new_out_cml_Rain'
# '/home/kkumah/Projects/cml-stuff/out_15min_cml_rain_oper'
cml_sat_daily_dir = r'/home/kkumah/Projects/cml-stuff/out_rain_trials/out_daily_V_May2026'
# r'/home/kkumah/Projects/cml-stuff/out_rain_trials/out_daily_V_May2026'
# r'/home/kkumah/Projects/cml-stuff/out_rain_trials/new_out_daily'
# r'/home/kkumah/Projects/cml-stuff/out_rain_trials/out_daily_no_smooth_strict_lat_params'
# r'/home/kkumah/Projects/cml-stuff/out_rain_trials/out_daily_no_smooth_strict_lat_params'
# r'/home/kkumah/Projects/cml-stuff/out_rain_trials/out_daily'
gauge_dirv = r'/home/kkumah/Projects/cml-stuff/gauge-data'
# '/home/kkumah/Projects/cml-stuff/out_daily_cml_rain_oper' 

plot_dir = '/home/kkumah/Projects/cml-stuff/plots'

#%% Floating varibales and constants
all_imerg_files = sorted([os.path.join(imerg_dir, f) for f in os.listdir(imerg_dir) if f.endswith('.nc4')])

all_cml_sat_files = sorted([os.path.join(cml_sat_daily_dir, f) for f in os.listdir(cml_sat_daily_dir) if f.endswith('.nc')])

all_cml_sat_15min_files = sorted([os.path.join(cml_sat_15min_dir, f) for f in os.listdir(cml_sat_15min_dir) if f.endswith('.nc')])

era5_file = os.path.join(era5_dir, 'ERA5_total_precipitation_2025_09_12_Ghana.nc')
bbox = (-4.0, 1.25, 4.5, 11.25)
cde_run_dte = datetime.today().strftime('%Y%m%d')

days = pd.date_range("2025-09-01", "2025-12-29", freq="D")# "2025-12-29"

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['DejaVu Serif', 'Times', 'serif']
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

#%%Functions
def create_xarray(all_precip, all_time_index, lon, lat, attrs=None):
    """
    Create an xarray DataArray from the list of 2D precipitation arrays and add attributes.

    Parameters:
    - all_precip: List or array of 2D precipitation arrays
    - all_time_index: List of timestamps
    - lon: Array of longitudes
    - lat: Array of latitudes
    - attrs: Dictionary of attributes to add to the DataArray (optional)

    Returns:
    - precip_data: xarray DataArray with the specified attributes
    """
    # Create a pandas DatetimeIndex from the list of timestamps
    time_index = pd.to_datetime(all_time_index)
    
    # Create an xarray DataArray from the list of 2D precipitation arrays
    precip_data = xr.DataArray(
                data=all_precip,
                dims=["time", "lat", "lon"],
                coords={
                    "time": time_index,
                    "lat": lat,
                    "lon": lon
                }
            )

    # Add attributes if provided
    if attrs:
        precip_data.attrs.update(attrs)
    
    return precip_data
def pick_imerg_day_files(files, day_str):
    """
    day_str: 'YYYY-MM-DD'
    Select files whose path/name contains YYYYMMDD.
    """
    ymd = day_str.replace("-", "")
    return [fp for fp in files if ymd in os.path.basename(fp)]

def read_nc_imger_file(file_path):
    imerg_precip_data = xr.open_dataset(file_path)
    precip_aray = imerg_precip_data.precipitation.data     
    precip_aray = np.flip(precip_aray[0,:,:].transpose(), axis=0)
    imerg_time = imerg_precip_data['time'].values[0]
    imerg_time_index = pd.to_datetime(imerg_time,format='%Y-%m-%d')
    lon = imerg_precip_data.coords['lon'].values
    lat = np.flip(imerg_precip_data.coords['lat']).values    
    imerg_precip_data.close()    
    del(imerg_precip_data,imerg_time)    
    return precip_aray, imerg_time_index, lon,lat

def process_imerg(files, product):    

    all_imfn_prcp, all_imfn_tms = [], []

    for imf in files:
        prcp, tms, lon, lat = read_nc_imger_file(imf)
        all_imfn_prcp.append(prcp)
        all_imfn_tms.append(tms)

    img_lon, img_lat = lon, lat

    # sort by time
    sorted_indices = np.argsort(np.array(all_imfn_tms))
    all_imfn_prcp = np.array(all_imfn_prcp)[sorted_indices]
    all_imfn_tms = np.array(all_imfn_tms)[sorted_indices]

    # create half-hourly DataArray
    imerg_xarr_hh_data = create_xarray(
        all_imfn_prcp,
        all_imfn_tms,
        img_lon,
        img_lat,
        attrs={"product": product}
    )

    # daily date
    day = pd.to_datetime(all_imfn_tms[0]).normalize()

    # daily precipitation (mm/day)
    imerg_dd_precip = (
    (imerg_xarr_hh_data.sum(dim="time") * 0.5)
    .expand_dims(time=[day])
    )

    return imerg_xarr_hh_data, imerg_dd_precip

def _standardize_latlon_names(da):
    """Rename common lat/lon variants to x/y for rioxarray."""
    rename_map = {}

    if "lat" in da.dims:
        rename_map["lat"] = "y"
    if "latitude" in da.dims:
        rename_map["latitude"] = "y"

    if "lon" in da.dims:
        rename_map["lon"] = "x"
    if "longitude" in da.dims:
        rename_map["longitude"] = "x"

    if rename_map:
        da = da.rename(rename_map)

    return da

def _ensure_crs(da, crs="EPSG:4326"):
    """Ensure CRS exists."""
    if not da.rio.crs:
        da = da.rio.write_crs(crs, inplace=False)
    return da

def _ensure_monotonic_coords(da):
    """Ensure lat decreases north→south and lon increases west→east."""
    if da.y[0] < da.y[-1]:
        da = da.sortby("y", ascending=False)
    if da.x[0] > da.x[-1]:
        da = da.sortby("x", ascending=True)

    return da

def harmonize_to_era5(
    source_da: xr.DataArray,
    era5_da: xr.DataArray,
    method="average",   # average is correct for precipitation
):
    """
    Harmonize a source DataArray (e.g. IMERG) onto ERA5 grid.
    """

    # --- Standardize dimension names
    src = _standardize_latlon_names(source_da)
    tgt = _standardize_latlon_names(era5_da)

    # --- CRS
    src = _ensure_crs(src)
    tgt = _ensure_crs(tgt)

    # --- Coordinate sanity
    src = _ensure_monotonic_coords(src)
    tgt = _ensure_monotonic_coords(tgt)

    # --- Match grid (NO reprojection needed)
    src_on_era5 = src.rio.reproject_match(
        tgt,
        resampling=Resampling.average if method == "average" else Resampling.bilinear,
    )

    # --- Restore conventional names
    src_on_era5 = src_on_era5.rename({"y": "latitude", "x": "longitude"})

    return src_on_era5

def harmonize_to_era5_v2(
    source_da: xr.DataArray,
    era5_da: xr.DataArray,
    method="average",   # average is correct for precipitation
):
    """
    Harmonize a source DataArray (e.g. IMERG) onto ERA5 grid.
    """

    # --- Standardize dimension names
    src = _standardize_latlon_names(source_da)
    tgt = _standardize_latlon_names(era5_da)

    # --- CRS
    src = _ensure_crs(src)
    tgt = _ensure_crs(tgt)

    # --- Coordinate sanity
    # src = _ensure_monotonic_coords(src)
    # tgt = _ensure_monotonic_coords(tgt)

    # --- Match grid (reprojection needed)
    
    src_on_era5 = src.rio.reproject(src.rio.crs, 
                        shape=tgt.shape[1:],  # set the shape as the ERA5 data
                        resampling=Resampling.nearest,)

    # --- Restore conventional names
    src_on_era5 = src_on_era5.rename({"y": "latitude", "x": "longitude"})

    return src_on_era5

def compute_pdf_elements_clean(data, bins, *, as_percent=True, positive_only=False):
    """
    Compute PDFc and PDFv by rainfall intensity bins.

    PDFc = percentage/fraction of samples in each bin.
    PDFv = percentage/fraction of rainfall volume in each bin.
    """

    if hasattr(data, "values"):
        arr = np.asarray(data.values, dtype=float).ravel()
    else:
        arr = np.asarray(data, dtype=float).ravel()

    arr = arr[np.isfinite(arr)]

    if positive_only:
        arr = arr[arr > 0]

    bins = np.asarray(bins, dtype=float)

    total_count = arr.size
    total_volume = np.nansum(arr)

    rows = []

    for i, upper in enumerate(bins):
        if i == 0:
            mask = arr <= upper
            label = f"{upper:g}"
        else:
            lower = bins[i - 1]
            mask = (arr > lower) & (arr <= upper)
            label = f"{upper:g}"

        vals = arr[mask]
        count = vals.size
        volume = np.nansum(vals)

        pdfc = count / total_count if total_count > 0 else np.nan
        pdfv = volume / total_volume if total_volume > 0 else np.nan

        if as_percent:
            pdfc *= 100.0
            pdfv *= 100.0

        rows.append({
            "bin": upper,
            "bin_label": label,
            "count": count,
            "volume": volume,
            "pdfc": pdfc,
            "pdfv": pdfv,
        })

    return pd.DataFrame(rows)
def plot_pdf_by_intensity_class(
    pdfs,
    *,
    metric="pdfv",
    ylabel="PDF (%)",
    xlabel="Rainfall [mm day$^{-1}$]",
    title=None,
    ylim=None,
):
    """
    Plot PDFc or PDFv using equally spaced rainfall-bin classes,
    not a logarithmic x-axis.
    """

    colors = {
        "IMERG": "k",
        "ERA5": "orange",
        "CML-SAT": "blue",
        "Gauge": "green",
    }

    linestyles = {
        "IMERG": "-",
        "ERA5": "-",
        "CML-SAT": "-",
        "Gauge": "-",
    }

    # Assume all products use the same bins
    first_pdf = next(iter(pdfs.values()))
    x = np.arange(len(first_pdf))
    xlabels = first_pdf["bin_label"].astype(str).values

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)

    for name, pdf in pdfs.items():
        ax.plot(
            x,
            pdf[metric].values,
            label=name,
            lw=3.5,
            color=colors.get(name, None),
            linestyle=linestyles.get(name, "-"),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)

    ax.set_xlabel(xlabel, fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")

    if title is not None:
        ax.set_title(title, fontsize=15, fontweight="bold")

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=12, frameon=False, loc="best", ncol=2)

    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", length=6, width=1.2, labelsize=12)
    ax.tick_params(axis="both", which="minor", length=3, width=0.8)

    plt.tight_layout()
    return fig, ax

def compute_metrics(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 2:
        return np.nan, np.nan, np.nan

    corr = np.corrcoef(x, y)[0, 1]
    # bias = (y.mean() / x.mean()) - 1
    mu_residuals = np.nanmean(y - x)
    mu_obs = np.nanmean(x)

    bias =  (mu_residuals/mu_obs) * 100
    rmse = np.sqrt(np.mean((y - x) ** 2))

    return corr, bias, rmse

def compute_pdf_elements_from_array(data_array, bins):
    """
    Compute PDF elements (PDFc and PDFv) from a 2D numpy array.

    Parameters:
    - data_array: 2D numpy array containing the data.
    - bins: List of bin edges.

    Returns:
    - pdf_df: A dictionary containing bins, PDFc, and PDFv.
    """
    pdfc = []  # PDF by occurrence
    pdfv = []  # PDF by volume
    bin_labels = []  # Bin labels for the output

    # Flatten the 2D array to 1D
    flattened_data = data_array.flatten()

    # Remove NaN values from the flattened data
    flattened_data = flattened_data[~np.isnan(flattened_data)]

    total_count = len(flattened_data)
    total_volume = 0

    # Loop through bins to compute PDFc and PDFv
    for i, bn in enumerate(bins):
        if i == 0:
            bin_data = flattened_data[flattened_data <= bn]
        else:
            bin_data = flattened_data[(flattened_data > bins[i - 1]) & (flattened_data <= bn)]

        # PDFc: Percentage of occurrences in the bin
        bin_count = len(bin_data)
        pdfc.append((bin_count / total_count)) # * 100

        # PDFv: Percentage of volume in the bin
        if bin_count > 0:
            bin_mean = np.mean(bin_data)
            bin_volume = bin_count * bin_mean
        else:
            bin_volume = 0

        total_volume += bin_volume
        pdfv.append(bin_volume)

        # Add bin label
        if i == 0:
            bin_labels.append(f"<= {bn}")
        else:
            bin_labels.append(f"{bins[i - 1]} - {bn}")

    # Normalize PDFv to percentages
    pdfv = [(volume / total_volume) for volume in pdfv] # * 100

    # Create a dictionary with bin, pdfc, and pdfv
    pdf_dict = {
        "bin": bins,
        "pdfc": pdfc,
        "pdfv": pdfv
    }

    del(flattened_data, bin_data
        ,total_count, total_volume
        ,bin_count, bin_mean, bin_volume
        ,bin_labels)

    gc.collect()

    return pdf_dict

def plot_precip_1x2_compare_ghana(
        da_list,
        lons,
        lats,
        titles,
        vmin,
        vmax,
        bbox=None,
        cmap_name="nipy_spectral",
        coast_lw=1.5,
        border_lw=1.5,
        n_colors=21,
        n_ticks=7,
        figsize=(13, 5.8),
        dpi=140,
        cbar_pad=0.055,
        cbar_fraction=0.055,
        wspace=0.08,
        bottom=0.16,
        top=0.90,
        left=0.04,
        right=0.99,
):
    """
    Plot three Ghana precipitation maps with a shared discrete colorbar.

    Key behavior:
    - Uses contourf for smooth-looking fields.
    - Removes the internal white margins by matching the map extent to
      the actual coordinate-center limits used by contourf.
    - If bbox is provided, it is clipped to the actual data extent so the
      axes do not extend beyond the plotted field.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # ------------------------------------------------------------
    # 1. SAFETY CHECKS
    # ------------------------------------------------------------
    if len(da_list) != len(titles):
        raise ValueError("da_list and titles must have the same length.")

    if len(da_list) != 3:
        raise ValueError("This function expects exactly three fields in da_list.")

    lon_vals = np.asarray(lons, dtype=float)
    lat_vals = np.asarray(lats, dtype=float)

    if lon_vals.ndim != 1 or lat_vals.ndim != 1:
        raise ValueError("lons and lats must be 1D coordinate arrays.")

    # Ensure increasing coordinate order
    lon_sort_idx = np.argsort(lon_vals)
    lat_sort_idx = np.argsort(lat_vals)

    lon_vals = lon_vals[lon_sort_idx]
    lat_vals = lat_vals[lat_sort_idx]

    lon_min_data = float(np.nanmin(lon_vals))
    lon_max_data = float(np.nanmax(lon_vals))
    lat_min_data = float(np.nanmin(lat_vals))
    lat_max_data = float(np.nanmax(lat_vals))

    data_extent_center = [
        lon_min_data,
        lon_max_data,
        lat_min_data,
        lat_max_data,
    ]

    # ------------------------------------------------------------
    # 2. MAP EXTENT
    # ------------------------------------------------------------
    # For contourf, use coordinate-center extent, not half-cell-expanded extent.
    if bbox is None:
        map_extent = data_extent_center
    else:
        # Clip requested bbox to actual data coordinate limits.
        # This prevents blank internal strips if bbox is slightly wider
        # than the gridded data.
        map_extent = [
            max(float(bbox[0]), lon_min_data),
            min(float(bbox[1]), lon_max_data),
            max(float(bbox[2]), lat_min_data),
            min(float(bbox[3]), lat_max_data),
        ]

    # ------------------------------------------------------------
    # 3. DISCRETE COLORMAP
    # ------------------------------------------------------------
    def discrete_nipy(vmin, vmax):
        mpl_cm = cm.get_cmap(cmap_name, n_colors)
        ncolors = mpl_cm(np.linspace(0, 1, n_colors))[:n_colors - 1]

        # Preserve your customized low-end colors
        ncolors[0] = [128 / 256, 100 / 256, 128 / 256, 1]

        if len(ncolors) > 2:
            ncolors[2] = ncolors[1].copy()
            ncolors[1] = [0.7, 0.1, 0.7, 1]

        newcmap = ListedColormap(ncolors)

        levels = np.linspace(vmin, vmax, len(ncolors) + 1)
        norm = BoundaryNorm(levels, newcmap.N)

        return newcmap, norm, levels

    cmap, norm, levels = discrete_nipy(vmin, vmax)

    # ------------------------------------------------------------
    # 4. FIGURE + AXES
    # ------------------------------------------------------------
    proj = ccrs.PlateCarree()

    fig, axs = plt.subplots(
        1, 3,
        figsize=figsize,
        dpi=dpi,
        subplot_kw=dict(projection=proj)
    )

    ims = []

    # ------------------------------------------------------------
    # 5. LOOP THROUGH PRODUCTS
    # ------------------------------------------------------------
    for i, (ax, da, title) in enumerate(zip(axs, da_list, titles)):

        # --------------------------------------------------------
        # Convert DataArray / ndarray to clean 2D array
        # --------------------------------------------------------
        if hasattr(da, "values"):
            data_vals = np.asarray(da.values)
        else:
            data_vals = np.asarray(da)

        data_vals = np.squeeze(data_vals)

        if data_vals.shape != (len(lats), len(lons)):
            raise ValueError(
                f"{title}: data shape {data_vals.shape} does not match "
                f"(len(lats), len(lons)) = {(len(lats), len(lons))}."
            )

        # Reorder data to match sorted lats/lons
        data_vals = data_vals[np.argsort(np.asarray(lats)), :]
        data_vals = data_vals[:, np.argsort(np.asarray(lons))]

        # --------------------------------------------------------
        # Map extent
        # --------------------------------------------------------
        ax.set_extent(map_extent, crs=proj)

        # Extra safety: force actual axis limits to the same extent.
        # This removes the visible internal white strips.
        ax.set_xlim(map_extent[0], map_extent[1])
        ax.set_ylim(map_extent[2], map_extent[3])

        # --------------------------------------------------------
        # Geographic outlines
        # --------------------------------------------------------
        ax.coastlines(
            resolution="10m",
            linewidth=coast_lw,
            color="black"
        )

        ax.add_feature(
            cfeature.BORDERS.with_scale("10m"),
            linewidth=border_lw,
            edgecolor="black"
        )

        # --------------------------------------------------------
        # Gridlines and labels
        # --------------------------------------------------------
        gl = ax.gridlines(
            crs=proj,
            draw_labels=True,
            linewidth=0.6,
            linestyle="--",
            color="gray",
            alpha=0.5
        )

        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = (i == 0)
        gl.bottom_labels = True

        gl.xlabel_style = {"size": 12, "weight": "bold"}
        gl.ylabel_style = {"size": 12, "weight": "bold"}

        # --------------------------------------------------------
        # Data plot
        # --------------------------------------------------------
        im = ax.contourf(
            lon_vals,
            lat_vals,
            data_vals,
            levels=levels,
            cmap=cmap,
            norm=norm,
            transform=proj,
            extend="max"
        )

        ax.set_title(
            title,
            fontsize=18,
            fontweight="bold"
        )

        ims.append(im)

    # ------------------------------------------------------------
    # 6. SHARED COLORBAR
    # ------------------------------------------------------------
    cbar = fig.colorbar(
        ims[0],
        ax=axs,
        orientation="horizontal",
        fraction=cbar_fraction,
        pad=cbar_pad,
        extend="max"
    )

    ticks = np.linspace(vmin, vmax, n_ticks)
    cbar.set_ticks(ticks)

    tick_labels = []
    for t in ticks:
        if abs(t - round(t)) < 1e-8:
            tick_labels.append(str(int(round(t))))
        else:
            tick_labels.append(f"{t:.1f}")

    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(labelsize=13)

    cbar.set_label(
        "Rainfall [mm/day]",
        fontsize=15,
        fontweight="bold"
    )

    # ------------------------------------------------------------
    # 7. FINAL LAYOUT
    # ------------------------------------------------------------
    fig.subplots_adjust(
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        wspace=wspace
    )

    return fig, axs

def open_daily_with_time(fname):
    """
    Open a daily CML-SAT file that lacks a time dimension,
    and inject time from filename.
    """
    # Extract YYYYMMDD from filename
    m = re.search(r"(\d{8})", fname)
    if m is None:
        raise ValueError(f"Cannot extract date from {fname}")

    date = pd.to_datetime(m.group(1), format="%Y%m%d")

    ds = xr.open_dataset(fname)

    # Ensure time dimension exists
    if "time" not in ds.dims:
        ds = ds.expand_dims(time=[date])

    # CF-compliant attrs
    ds["time"].attrs.update({
        "standard_name": "time",
        "long_name": "time",
        "axis": "T"
    })

    return ds

def plot_latitude_profiles(
    imerg_da,
    era5_da,
    cml_sat_da,
    lat_name="latitude",
    xlim=None,
    title="Latitudinal Mean Rainfall",
):
    """
    Plot IMERG and ERA5 precipitation profiles vs latitude.
    """

    # --- Extract data safely ---
    lat = imerg_da[lat_name].values
    imerg_vals = imerg_da.values
    era5_vals = era5_da.values
    cml_sat_vals = cml_sat_da.values

    # --- Ensure numpy arrays ---
    lat = np.asarray(lat)
    imerg_vals = np.asarray(imerg_vals)
    era5_vals = np.asarray(era5_vals)
    cml_sat_vals = np.asarray(cml_sat_vals)

    # --- Sort by latitude (south → north for plotting clarity) ---
    sort_idx = np.argsort(lat)
    lat = lat[sort_idx]
    imerg_vals = imerg_vals[sort_idx]
    era5_vals = era5_vals[sort_idx]
    cml_sat_vals = cml_sat_vals[sort_idx]

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(6, 7), dpi=140)

    ax.plot(
        imerg_vals, lat,
        color="k",
        linewidth=3.5,
        label="IMERG",
    )

    ax.plot(
        era5_vals, lat,
        color="orange",
        linewidth=3.5,
        label="ERA5",
    )

    ax.plot(
        cml_sat_vals, lat,
        color="b",
        linewidth=3.5,
        label="CML-SAT",
    )

    # Agro-climatic zone boundaries (°N)
    zone_bounds = [5.5, 7.5, 8.5]

    for z in zone_bounds:
        ax.axhline(
            y=z,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8
        )

    # Zone label positions: (label, y_position)
    zone_labels = [
    ("Coastal Savanna", 5.0),
    ("Forest Zone", 6.6),
    ("Transitional Zone", 8.0),
    ("Guinea Savanna", 10.0),
    ]

    for label, y in zone_labels:
        ax.text(
            0.98, y,
            label,
            fontsize=12,
            fontweight="bold",
            color="dimgray",
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform()
        )

    # --- Labels & title ---
    ax.set_xlabel("Rainfall (mm/day)", fontsize=14)
    ax.set_ylabel("Latitude (°N)", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")

    # --- Limits ---
    if xlim is not None:
        ax.set_xlim(xlim)

    # --- Grid & ticks ---
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", labelsize=12)

    # --- Legend ---
    ax.legend(fontsize=12, loc="best")

    plt.tight_layout()
    plt.show()

def plot_daily_time_series(imerg_dat, era5_dat, cml_sat_dat):
    da1 = imerg_dat          # rename appropriately
    da2 = era5_dat           # rename appropriately
    da3 = cml_sat_dat      # rename appropriately

    fig, ax = plt.subplots(figsize=(10, 4), dpi=140)

    # Plot both on the same axis
    ax.plot(
        da1.time.values,
        da1.values,
        label="IMERG",
        color="k",
        linewidth=3.5
    )

    ax.plot(
        da2.time.values,
        da2.values,
        label="ERA5",
        color="orange",
        linewidth=3.5
    )
    ax.plot(
        da3.time.values,
        da3.values,
        label="CML-SAT",
        color="blue",
        linewidth=3.5
    )

    # --- Axis formatting ---
    # ax.set_xlabel("Time", fontsize=13)
    ax.set_ylabel("Rainfall [mm/day]", fontsize=13)

    # Force identical y-ticks
    yticks = np.arange(0, 21, 5)
    ax.set_yticks(yticks)
    ax.set_ylim(0, 20)

    # Grid & ticks
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=12, length=6, direction='in')
    ax.tick_params(axis='both', which='minor', length=3, direction='in')
    ax.grid(which='major', linestyle='--', linewidth=0.6, alpha=0.6)

    # Legend
    ax.legend(fontsize=12, loc="upper right", frameon=False,ncol=3)

    # Major ticks → months
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))

    # Minor ticks → every 7 days
    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=7))

    ax.xaxis.set_minor_locator(mdates.DayLocator(bymonthday=15))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%d'))   # 15

    # Tick styling
    ax.tick_params(axis='x', which='major', labelsize=16, pad=10)
    ax.tick_params(axis='x', which='minor', labelsize=12, pad=3)

    # Optional: rotate slightly for safety
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    plt.tight_layout()
    # plt.show()
    gc.collect()

def make_cat_performance_diagram(cat_df_res):
    '''
    cat_df_res = should be a dataframe of the categorical metrics containing POD, FAR etc as rown names and
                 product names for which metric was computed as column names

    eval_elements = should be a list containing:
                    columnm names, plot marker marker, product evaluated,colors to use for plot
    '''
    def calculate_csi(pod, success_ratio):
        csi = np.where(pod + success_ratio == 0, 0, pod * success_ratio / (pod + success_ratio - pod * success_ratio))
        return np.nan_to_num(csi)
    # floating variables
    # ['k','k','r','r']
    clrs = {'IMERG': 'k',
            'ERA5': 'orange',
            'CML-SAT': 'b',
            'Gauge': 'g',
            }
    # ['o','*','o','*',]
    mrker = ['o','p','D','^','s','8','*','^', '<', '>',]
    pod_spc = np.linspace(0, 1, 100)
    success_ratio_spc = np.linspace(0, 1, 100)

    X, Y = np.meshgrid(success_ratio_spc, pod_spc)
    CSI = calculate_csi(Y, X)

    # Create the main figure and axis
    fig, ax1 = plt.subplots(figsize=(6,8), dpi=1000)
    plt.subplots_adjust(bottom=0.3)

    # Plot the contour lines for CSI
    csi_levels = np.arange(0.1, 1.0, 0.1)
    CSI_contours = ax1.contour(X, Y, CSI, levels=csi_levels, colors='brown', linestyles='solid')
    # ax1.clabel(CSI_contours, inline=1, fontsize=10, fmt='%1.1f', colors='r')

    # Plotting bias lines
    for fb_level in [0.5, 1, 1.2, 1.5, 2, 4]:
        fb_line = pod_spc * fb_level
        valid = fb_line <= 1
        ax1.plot(success_ratio_spc[valid], fb_line[valid], c='steelblue',ls = '--', lw=2)
        label_x = 1 / fb_level if fb_level > 1 else 0.95
        label_y = fb_line[int(label_x * 100)] if fb_level > 1 else 0.95 * fb_level
        ax1.text(label_x, label_y, str(fb_level), color='steelblue',
                    fontsize=12, ha='center',fontweight='bold')
        if fb_level == 1:  # We'll only write "Bias" near the line where the frequency bias is 1
            ax1.text(0.5, 0.5*fb_level, 'Bias', color='steelblue', fontsize=12, fontweight='bold' ,
                        ha='center', va='bottom', rotation=45) # , backgroundcolor='white'

    # Plotting points for each sim obs eval data points
    for it in enumerate(cat_df_res.columns.to_list()):
        clnme_used = it[1]
        evl_prdt = clnme_used#.split('vrs')[0]

        podd = cat_df_res.loc['POD',clnme_used]
        farr = cat_df_res.loc['FAR',clnme_used]

        succ_ratio = 1 - farr # success ration

        ax1.plot(succ_ratio, podd, marker = mrker[it[0]], color=clrs[evl_prdt], markersize=8, label=evl_prdt)

    ax1.minorticks_on()
    ax1.tick_params(which='major', axis= 'both', direction='in',length=5, top=True, 
                    right=True, bottom=True, left=True)  # Adjust major tick length
    ax1.tick_params(which='minor', axis= 'both', direction='in', length=2.5, top=True, 
                    right=True, bottom=True, left=True) 
    # Customizing the axis
    # ax1.set_title('Performance Diagram')
    ax1.set_xlabel('Success Ratio (1 - FAR)',fontsize=12)
    ax1.set_xticklabels([f'{level:.1f}' for level in [0.0,0.2,0.4,
                        0.6,0.8,1.0]], fontsize=15)
    ax1.set_ylabel('POD', fontsize=12)
    ax1.set_yticklabels([f'{level:.1f}' for level in [0.0,0.2,0.4,
                        0.6,0.8,1.0,1.1]], fontsize=12)
    ax1.grid(True,ls='--',lw=0.5)

    # Secondary y-axis for CSI
    ax2 = ax1.twinx()
    ax2.set_ylabel('CSI', fontsize=15, color='brown')  # Setting the label color to red
    ax2.set_ylim(0, 1)
    ax2.set_yticks(csi_levels)
    ax2.set_yticklabels([f'{level:.1f}' for level in csi_levels], color='brown',
                        fontsize=12)  # Setting the tick labels color to red
    ax2.minorticks_on()
    ax2.tick_params(which='major', axis= 'both', direction='in',length=5, 
                    top=True, right=True, bottom=True, left=True)  # Adjust major tick length
    ax2.tick_params(which='minor', axis= 'both', direction='in', length=2.5, 
                    top=True, right=True, bottom=True, left=True) 

    # Fixing the legend issue
    # Collect labels and handles and keep only unique entries for the legend
    
    handles, labels = ax1.get_legend_handles_labels()
    unique = dict(zip(labels, handles)).items()
    ax1.legend(handles=handles, labels=labels, loc='upper center', frameon=False, 
               bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=15)

def cat_stats_dict_to_df(stats_dict, product_names,
                          metrics=('POD', 'FAR', 'Bias', 'CSI')):
    """
    Convert multiple categorical_stats dict outputs into
    a DataFrame suitable for Roebber / performance diagrams.

    Parameters
    ----------
    stats_dict : dict
        {product_name: stats_dict_from_categorical_stats}
    product_names : list
        Ordered list of product names (columns)
    metrics : tuple
        Metrics to include (row index)

    Returns
    -------
    pandas.DataFrame
        index = metrics
        columns = product_names
    """

    df = pd.DataFrame(index=metrics, columns=product_names, dtype=float)

    for prod in product_names:
        for m in metrics:
            df.loc[m, prod] = stats_dict[prod][m]

    return df

def add_regression_line(ax, x, y, color, xlim):

    # Convert xarray → numpy
    if hasattr(x, "values"):
        x = x.values
    if hasattr(y, "values"):
        y = y.values

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 2:
        return np.nan, np.nan, mask

    x_clean = x[mask]
    y_clean = y[mask]

    model = LinearRegression()
    model.fit(x_clean.reshape(-1, 1), y_clean.reshape(-1, 1))

    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    x_line = np.linspace(xlim[0], xlim[1], 200)
    y_line = slope * x_line + intercept

    ax.plot(
        x_line, y_line,
        color=color, linestyle='-', linewidth=1.5,
        label='Regression Line'
    )

    return slope, intercept, mask


def categorical_stats(forecast, observation, threshold):
    """
    Compute categorical verification statistics following WMO/JWGNE definitions.

    Parameters
    ----------
    forecast : array-like
        Forecast values (e.g., rainfall mm/day)
    observation : array-like
        Observed values (same units as forecast)
    threshold : float
        Event threshold (e.g., rain >= 1 mm/day)

    Returns
    -------
    stats : dict
        Dictionary of categorical statistics
    """

    forecast = np.asarray(forecast)
    observation = np.asarray(observation)

    # Binary events
    f_event = forecast >= threshold
    o_event = observation >= threshold

    # Contingency table
    H = np.sum(f_event & o_event)        # Hits
    M = np.sum(~f_event & o_event)       # Misses
    F = np.sum(f_event & ~o_event)       # False alarms
    C = np.sum(~f_event & ~o_event)      # Correct negatives

    # Avoid division by zero using np.nan
    POD = H / (H + M) if (H + M) > 0 else np.nan
    FAR = F / (H + F) if (H + F) > 0 else np.nan
    CSI = H / (H + M + F) if (H + M + F) > 0 else np.nan
    Bias = (H + F) / (H + M) if (H + M) > 0 else np.nan
    POFD = F / (F + C) if (F + C) > 0 else np.nan
    Accuracy = (H + C) / (H + M + F + C) if (H + M + F + C) > 0 else np.nan

    return {
        "Hits": H,
        "Misses": M,
        "False_Alarms": F,
        "Correct_Negatives": C,
        "POD": POD,           # Probability of Detection
        "FAR": FAR,           # False Alarm Ratio
        "CSI": CSI,           # Critical Success Index
        "Bias": Bias,         # Frequency Bias
        "POFD": POFD,         # Probability of False Detection
        "Accuracy": Accuracy
    }


def read_gauge_daily(fp):
    sid = os.path.splitext(os.path.basename(fp))[0]

    df = pd.read_csv(fp, parse_dates=["timestamp"])
    df = df.set_index("timestamp")

    # rainfall column
    r_cl = [c for c in df.columns if c.startswith("pr")][0]
    pr = df[r_cl].astype(float)

    daily = pr.resample("D").sum(min_count=1)
    daily.name = sid
    return daily

#-------------------------
def extract_point_daily(da, lat, lon):
    """
    da: xarray DataArray (time, y, x)
    returns pandas Series indexed by time
    """
    return (
        da
        .sel(latitude=lat, longitude=lon, method="nearest")
        .to_series()
    )
# -------------------------
def find_grid_indices_from_coords(lat, lon, lats, lons):
    """
    Find nearest grid-cell indices for a given lat/lon
    using actual coordinate vectors.
    """
    row = np.argmin(np.abs(lats - lat))
    col = np.argmin(np.abs(lons - lon))
    return row, col


#%% Main processing
# 1) Read and preprocess IMERG data day by day
daily_xr = []
for day in days:
    day_str = day.strftime("%Y-%m-%d")
    imerg_day_files  = pick_imerg_day_files(all_imerg_files,  day_str)

    _,imerg_day_dat = process_imerg(imerg_day_files, product='IMERG-Late')
    daily_xr.append(imerg_day_dat)

imerg_daily_xarr = xr.concat(daily_xr, dim="time")

# 2) Read and preprocess ERA5 data
# era5_data = xr.open_dataset(era5_file)
# era5_data = era5_data.rename({'valid_time': 'time'})
# era5_daily_data = era5_data['tp'] * 1000  # Convert from meters to mm
# era5_daily_data = era5_daily_data.resample(time='1D').mean() * 24

era5_data = xr.open_dataset(era5_file)
era5_data = era5_data.rename({'valid_time': 'time'})

era5_daily_data = (
    era5_data['tp']
    .groupby('time.date')
    .sum(dim='time')
    .rename({'date': 'time'})
    * 1000.0
)

# 3) Harmonize IMERG data to ERA5 grid
imerg_daily_agg = harmonize_to_era5_v2(
    imerg_daily_xarr,
    era5_daily_data,
)

# Read process cml data
datasets = [open_daily_with_time(f) for f in all_cml_sat_files]

cml_sat_xarr = xr.concat(
    datasets,
    dim="time",
    coords="minimal",
    compat="override"
).sel(y=slice(bbox[2], bbox[3]), x=slice(bbox[0], bbox[1]))

cml_sat_daily_agg = harmonize_to_era5_v2(
    cml_sat_xarr['rain_daily_total'],
    era5_daily_data,
)

# harmonize the data in time ensuring we are comparing same days
# Convert ERA5 time to datetime64[ns]
era5_daily_data = era5_daily_data.assign_coords(
    time=pd.to_datetime(era5_daily_data.time.values)
)

# Normalize IMERG time (safe even if already normalized)
imerg_daily_agg = imerg_daily_agg.assign_coords(
    time=pd.to_datetime(imerg_daily_agg.time.values).normalize()
)

# Normalize CML-Sat time (safe even if already normalized)
cml_sat_daily_agg = cml_sat_daily_agg.assign_coords(
    time=pd.to_datetime(cml_sat_daily_agg.time.values).normalize()
)
# Now intersect safely
common_times = np.intersect1d(
    np.intersect1d(
        imerg_daily_agg.time.values,
        era5_daily_data.time.values
    ),
    cml_sat_daily_agg.time.values
)

# Subset
imerg_daily_agg = imerg_daily_agg.sel(time=common_times)
era5_daily_data = era5_daily_data.sel(time=common_times)
cml_sat_daily_agg = cml_sat_daily_agg.sel(time=common_times)

#%% Begin evaluation and plotting
# 1) a) Spatial mean over Ghana: single value per day time series
imerg_ghana_mean = imerg_daily_agg.mean(dim=['latitude', 'longitude'])
era5_ghana_mean = era5_daily_data.mean(dim=['latitude', 'longitude'])
cml_sat_ghana_mean = cml_sat_daily_agg.mean(dim=['latitude', 'longitude'])

# b) Spatial mean: 2d array
imerg_ghana_2dmean = imerg_daily_agg.mean(dim='time')
era5_ghana_2dmean = era5_daily_data.mean(dim='time')
cml_sat_ghana_2dmean = cml_sat_daily_agg.mean(dim='time', skipna=True)

# c) zonal latitudinal mean
imerg_ghana_zonalmean = imerg_daily_agg.mean(dim=['time', 'longitude'])
era5_ghana_zonalmean = era5_daily_data.mean(dim=['time', 'longitude'])
cml_sat_ghana_zonalmean = cml_sat_daily_agg.mean(dim=['time', 'longitude'], skipna=True)

# ---------------------
# The Mean Daily Rainfall Comparison
# ---------------------
# 1) a) time series
plot_daily_time_series(imerg_ghana_mean, era5_ghana_mean, cml_sat_ghana_mean)

gc.collect()

# 1b) Scatter plot analysis of the mean

# FIGURE SETUP
fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=140)

# Define hemisphere axis limits
xlims = (0., 20)
ylims = (0., 20)

plots = [
    (axs[0], imerg_ghana_mean, cml_sat_ghana_mean, 'r', 'CML-SAT vs IMERG', xlims, ylims),
    (axs[1], era5_ghana_mean, cml_sat_ghana_mean, 'r', 'CML-SAT vs ERA5', xlims, ylims),
    (axs[2], era5_ghana_mean, imerg_ghana_mean, 'r', 'ERA5 vs IMERG', xlims, ylims) # (axs[1, 0], chgps_NH_means['zonal_means'], merra2_NH_means['zonal_means'], 'MERRA2', 'red', 'SH', SH_xlim, SH_ylim),
    
]

for ax, x, y, color, lab, xlim, ylim in plots:

    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ticks = np.arange(0, 21, 5)

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # Scatter
    ax.scatter(x, y, s=45, color=color, edgecolor='black', alpha=0.85,
               label=lab)

    # 1:1 LINE across fixed limits
    ax.plot([xlim[0], xlim[1]], [ylim[0], ylim[1]],
            color='black', linestyle='--', linewidth=1.5) # , label='1:1 Line'

    # Regression line
    slope, intercept, mask = add_regression_line(ax, x, y, color, xlim)

    # Stats using same cleaned subset
    # Ensure numeric
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    mask = np.isfinite(x) & np.isfinite(y)#~(x.isna() | y.isna())
    x_clean = x[mask].astype(float)#.values
    y_clean = y[mask].astype(float)#.values

    # Must be at least 2 values to compute correlation
    if len(x_clean) < 2:
        corr = np.nan
    else:
        corr = np.corrcoef(x_clean, y_clean)[0, 1]
    bias = (y_clean.mean() / x_clean.mean()) - 1

    # Regression equation text
    eq_text = f"y = {slope:.2f}x + {intercept:.2f}"

    # Metrics block
    metrics_text = (
    f"Corr = {corr:.2f}\n"
    f"Bias = {bias:.1%}\n"
    f"{eq_text}\n"
    "--  1:1 line\n"
    "—   Regression line"
    )

    ax.text(
        0.03, 0.97,
        metrics_text,
        transform=ax.transAxes,
        fontsize=14,
        fontweight='bold',
        va='top', ha='left',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
    )
    
    # Axes + ticks
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=15, length=7, direction='in')
    ax.tick_params(axis='both', which='minor', length=4, direction='in')
    ax.grid(which='major', linestyle='--', linewidth=0.6, color='grey')

    # Legend
    # ax.legend(fontsize=12, loc='lower right', frameon=False)

# Axis labels
axs[0].set_ylabel("CML-SAT [mm/day]", fontsize=16)
axs[0].set_xlabel("IMERG [mm/day]", fontsize=16)
axs[1].set_ylabel("CML-SAT [mm/day]", fontsize=16)
axs[1].set_xlabel("ERA5 [mm/day]", fontsize=16)
axs[2].set_ylabel("IMERG [mm/day]", fontsize=16)
axs[2].set_xlabel("ERA5 [mm/day]", fontsize=16)

plt.tight_layout()
plt.show()
gc.collect()


#-------------------------------------------------------------------------------

# 2) a) Spatial distribution of mean dailyrainfall
fig, axs = plot_precip_1x2_compare_ghana(
    da_list=[imerg_ghana_2dmean, 
             cml_sat_ghana_2dmean,
             era5_ghana_2dmean, 
             ],
    lons=imerg_ghana_2dmean.longitude.values,
    lats=imerg_ghana_2dmean.latitude.values,
    titles=["IMERG", "CML-SAT", "ERA5"],
    vmin=0,
    vmax=6,
    bbox=None,
    cbar_pad=0.12,     # pulls colorbar downward
    bottom=0.22,       # gives enough figure space below maps
    cbar_fraction=0.05
)
gc.collect()

# 2) b) Spatial distribution of rainfall on a specific day (2025-09-20)
fig, axs = plot_precip_1x2_compare_ghana(
    da_list=[imerg_daily_agg.sel(time='2025-09-20'),               
             cml_sat_daily_agg.sel(time='2025-09-20'),
             era5_daily_data.sel(time='2025-09-20')],
    lons=imerg_ghana_2dmean.longitude.values,
    lats=imerg_ghana_2dmean.latitude.values,
    titles=["IMERG", "CML-SAT", "ERA5"],
    vmin=0,
    vmax=18,
    bbox=None,
    cbar_pad=0.12,     # pulls colorbar downward
    bottom=0.22,       # gives enough figure space below maps
    cbar_fraction=0.05

)

gc.collect()
#-------------------------------------------------------------------------------

# 2) c) Zonal mean plots
plot_latitude_profiles(
    imerg_ghana_zonalmean,   #  IMERG 1D DataArray
    era5_ghana_zonalmean,    #  ERA5 1D DataArray
    cml_sat_ghana_zonalmean, #  CML-SAT 1D DataArray
    xlim=(1, 6.0),
    title="Latitudinal Mean Rainfall"
)
gc.collect()
# %% 2) DO PDF ANALYSIS

bins = [0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]

pdfs_domain = {
    "IMERG": compute_pdf_elements_clean(imerg_daily_agg, bins, as_percent=True),
    "ERA5": compute_pdf_elements_clean(era5_daily_data, bins, as_percent=True),
    "CML-SAT": compute_pdf_elements_clean(cml_sat_daily_agg, bins, as_percent=True),
}

plot_pdf_by_intensity_class(
    pdfs_domain,
    metric="pdfv",
    ylabel="PDF (%)",
    title="",
    ylim=(0, 45),
)



#%% Categorical metrics computation
thr = 0.5
cml_sat_imerg_stats = categorical_stats(
    forecast=cml_sat_daily_agg.values,
    observation=imerg_daily_agg.values,
    threshold=thr   # 1 mm/day wet-day threshold
)

era5_imerg_stats = categorical_stats(
    forecast=era5_daily_data.values,
    observation=imerg_daily_agg.values,
    threshold=thr   # 1 mm/day wet-day threshold
)

cml_sat_era5_stats = categorical_stats(
    forecast=cml_sat_daily_agg.values,
    observation=era5_daily_data.values,
    threshold=thr   # 1 mm/day wet-day threshold
)

imerg_era5_stats = categorical_stats(
    forecast=imerg_daily_agg.values,
    observation=era5_daily_data.values,
    threshold=thr   # 1 mm/day wet-day threshold
)


stats_imerg_obs = {
    "CML-SAT": cml_sat_imerg_stats,
    "ERA5":    era5_imerg_stats
}

cat_df_imerg_obs = cat_stats_dict_to_df(
    stats_dict=stats_imerg_obs,
    product_names=["CML-SAT", "ERA5"]
)

stats_era5_obs = {
    "CML-SAT": cml_sat_era5_stats,
    "IMERG":   imerg_era5_stats
}

cat_df_era5_obs = cat_stats_dict_to_df(
    stats_dict=stats_era5_obs,
    product_names=["CML-SAT", "IMERG"]
)

make_cat_performance_diagram(cat_df_imerg_obs)

make_cat_performance_diagram(cat_df_era5_obs)

# %% The gauge evaluation
all_gauges_files = [os.path.join(gauge_dirv, f) for f in os.listdir(gauge_dirv) if (f.endswith('.csv') and f.startswith('TA'))]
sheet = "Jan-2023 - Jan-2021"  # or whichever one you want; they share the same station list

station_meta = pd.read_csv(os.path.join(gauge_dirv, 'stations_metadata.csv'),)
# station_meta.columns = ["station_id", "cc", "lat", "lon", "n_total", "n_valid", "n_total2"]

# Ghana stations only
# station_meta = station_meta[station_meta["cc"] == "GH"].copy()
# station_meta.set_index("station_id", inplace=True)

# ids = [os.path.splitext(os.path.basename(f))[0] for f in all_gauges_files]
# sub = station_meta[station_meta.index.isin(ids)].copy()
# sub.shape


#-------------------------
station_meta = station_meta.copy()

rows, cols = [], []
for _, r in station_meta.iterrows():
    row, col = find_grid_indices_from_coords(
        r["latitude"], r["longitude"],
        era5_daily_data['latitude'].values, era5_daily_data['longitude'].values
    )
    rows.append(row)
    cols.append(col)

station_meta["row"] = rows
station_meta["col"] = cols
station_meta["pixel_id"] = list(zip(rows, cols))

pix_gauge_count = (
    station_meta
    .groupby(["row", "col"])["station code"]
    .nunique()
    .rename("n_gauge")
    .reset_index()
)
#-------------------------
# build daily gauge dataframe
daily_gauges = []
for fp in all_gauges_files:
    # sid = os.path.splitext(os.path.basename(fp))[0]
    # if sid in sub.index:
    daily_gauges.append(read_gauge_daily(fp))

gauge_daily_df = pd.concat(daily_gauges, axis=1)

#-------------------------
records = []

for sid, row in station_meta.iterrows():
    lat, lon = row["latitude"], row["longitude"]
    pid = row["pixel_id"]
    st_row, st_col = row["row"], row["col"]

    # gauge daily
    g = gauge_daily_df[row['station code']].dropna()

    # products
    cml   = pd.DataFrame(extract_point_daily(cml_sat_daily_agg, lat, lon))
    imerg = pd.DataFrame(extract_point_daily(imerg_daily_agg, lat, lon))
    era5  = pd.DataFrame(extract_point_daily(era5_daily_data, lat, lon))

    df_merg = pd.merge(g, cml, how="left", left_index=True, right_index=True)
    df_merg = pd.merge(df_merg, imerg, how="left", left_index=True, right_index=True)
    df_merg = pd.merge(df_merg, era5, how="left", left_index=True, right_index=True)

    df_merg = df_merg.dropna(axis=0)
    df_merg.columns = ["gauge", "cml_sat", "imerg", "era5"]

    # df = pd.concat(
    #     [g, cml, imerg, era5],
    #     axis=1,
    #     keys=["gauge", "cml_sat", "imerg", "era5"]
    # ).dropna()

    df_merg["station"] = row['station code']
    df_merg["st_row"] = st_row
    df_merg["st_col"] = st_col
    records.append(df_merg.reset_index())

compare_df = pd.concat(records, ignore_index=True)

compare_df = compare_df.merge(
    pix_gauge_count,
    left_on=["st_row", "st_col"],
    right_on=["row", "col"],
    how="left"
)

compare_df_pix = (
    compare_df
    .groupby(["st_row", "st_col", "timestamp"], as_index=False)
    .agg(
        gauge=("gauge", "mean"),
        cml_sat=("cml_sat", "mean"),
        imerg=("imerg", "mean"),
        era5=("era5", "mean"),
        n_gauge=("n_gauge", "first")
    )
)
# Ensure numeric
for c in ["gauge", "cml_sat", "imerg", "era5"]:
    compare_df_pix[c] = pd.to_numeric(compare_df_pix[c], errors="coerce")

#-------------------------

fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=140)
mzval = 50.0
xlims = (0., mzval)
ylims = (0., mzval)
ticks = np.arange(0, mzval+1, 10)

plots = [
    (axs[0], compare_df_pix["gauge"], compare_df_pix["cml_sat"], "CML-SAT vs Gauge"),
    (axs[1], compare_df_pix["gauge"], compare_df_pix["imerg"],  "IMERG vs Gauge"),
    (axs[2], compare_df_pix["gauge"], compare_df_pix["era5"],   "ERA5 vs Gauge"),
]

for ax, x, y, title in plots:

    # limits & ticks
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # scatter
    ax.scatter(
        x, y,
        s=45, color="r", edgecolor="k", alpha=0.85
    )

    # 1:1 line
    ax.plot(
        [xlims[0], xlims[1]],
        [ylims[0], ylims[1]],
        "b--", lw=1.5
    )

    # regression line
    slope, intercept, mask = add_regression_line(ax, x, y, "k", xlims)

    # metrics
    corr, bias, rmse = compute_metrics(x.values, y.values)

    eq_text = f"y = {slope:.2f}x + {intercept:.2f}"

    metrics_text = (
        f"Corr = {corr:.2f}\n"
        f"Bias = {bias:.1f}%\n"
        f"RMSE = {rmse:.2f}\n"
        f"{eq_text}\n"
        "--  1:1 line\n"
        "—   Regression line"
    )

    ax.text(
        0.03, 0.97,
        metrics_text,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
    )

    # cosmetics
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=14, length=7, direction="in")
    ax.tick_params(axis="both", which="minor", length=4, direction="in")
    ax.grid(which="major", linestyle="--", linewidth=0.6, color="grey")

# axis labels
axs[0].set_xlabel("Gauge [mm/day]", fontsize=15)
axs[1].set_xlabel("Gauge [mm/day]", fontsize=15)
axs[2].set_xlabel("Gauge [mm/day]", fontsize=15)

axs[0].set_ylabel("CML-SAT [mm/day]", fontsize=15)
axs[1].set_ylabel("IMERG [mm/day]", fontsize=15)
axs[2].set_ylabel("ERA5 [mm/day]", fontsize=15)

plt.tight_layout()
plt.show()
gc.collect()

#-------------------------

cat = {}
for prod in ["cml_sat", "imerg", "era5"]:
    cat[prod] = categorical_stats(
        compare_df_pix[prod].values,
        compare_df_pix["gauge"].values,
        threshold=thr
    )

cat_stats_at_gauge = pd.DataFrame(cat)
cat_stats_at_gauge.columns = ["CML-SAT", "IMERG", "ERA5"]
make_cat_performance_diagram(cat_stats_at_gauge)

# -------------------------
# mean daily rainfall at gauge locations (pixel-level mean)
df_mean = (
    compare_df_pix.copy()
    .groupby(["st_row", "st_col"])[["gauge", "cml_sat", "imerg", "era5"]]
    .mean()
    .reset_index()
)

df_mean

fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=140)

xlims = (0., 10.)
ylims = (0., 10.)
ticks = np.arange(0, 11, 2)

plots = [
    (axs[0], df_mean["gauge"], df_mean["cml_sat"], "CML-SAT vs Gauge (Mean Daily)"),
    (axs[1], df_mean["gauge"], df_mean["imerg"],  "IMERG vs Gauge (Mean Daily)"),
    (axs[2], df_mean["gauge"], df_mean["era5"],   "ERA5 vs Gauge (Mean Daily)"),
]

for ax, x, y, title in plots:

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    # scatter (36 points)
    ax.scatter(
        x, y,
        s=80, color="k", edgecolor="black", alpha=0.9
    )

    # 1:1 line
    ax.plot(
        [xlims[0], xlims[1]],
        [ylims[0], ylims[1]],
        "k--", lw=1.5
    )

    # regression
    slope, intercept, mask = add_regression_line(ax, x, y, "k", xlims)

    # metrics
    corr, bias, rmse = compute_metrics(x.values, y.values)

    eq_text = f"y = {slope:.2f}x + {intercept:.2f}"

    metrics_text = (
        f"Corr = {corr:.2f}\n"
        f"Bias = {bias:.1f}%\n"
        f"RMSE = {rmse:.2f}\n"
        f"{eq_text}\n"
        "--  1:1 line\n"
        "—   Regression line"
    )

    ax.text(
        0.03, 0.97,
        metrics_text,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top", ha="left",
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
    )

    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.minorticks_on()
    ax.tick_params(axis="both", which="major", labelsize=14, length=7, direction="in")
    ax.tick_params(axis="both", which="minor", length=4, direction="in")
    ax.grid(which="major", linestyle="--", linewidth=0.6, color="grey")

axs[0].set_xlabel("Gauge mean [mm/day]", fontsize=15)
axs[1].set_xlabel("Gauge mean [mm/day]", fontsize=15)
axs[2].set_xlabel("Gauge mean [mm/day]", fontsize=15)

axs[0].set_ylabel("CML-SAT mean [mm/day]", fontsize=15)
axs[1].set_ylabel("IMERG mean [mm/day]", fontsize=15)
axs[2].set_ylabel("ERA5 mean [mm/day]", fontsize=15)

plt.tight_layout()
plt.show()
gc.collect()


#-----------------------------------------------------------------

pdfs_gauge_matched = {
    "Gauge": compute_pdf_elements_clean(compare_df_pix["gauge"], bins, as_percent=True),
    "IMERG": compute_pdf_elements_clean(compare_df_pix["imerg"], bins, as_percent=True),
    "ERA5": compute_pdf_elements_clean(compare_df_pix["era5"], bins, as_percent=True),
    "CML-SAT": compute_pdf_elements_clean(compare_df_pix["cml_sat"], bins, as_percent=True),
}

plot_pdf_by_intensity_class(
    pdfs_gauge_matched,
    metric="pdfv",
    ylabel="PDF (%)",
    title="",
    ylim=(0, 45),
)

gc.collect()
#%%
#%%
def kstd_by_lat(lat):
    lat = np.asarray(lat)
    k = np.zeros_like(lat, dtype="float32")
    k = np.where(lat < 5, 0.85, k)
    k = np.where((lat >= 5) & (lat < 8), 0.92, k)
    k = np.where(lat >= 8, 0.95, k)
    return k

def kstd_by_lat_xr(lat_da):
    """
    lat_da: xarray.DataArray
        1D (y) or 2D (y, x) latitude field
    Returns
    -------
    kstd : xarray.DataArray
        Same shape as lat_da
    """

    # start with zeros, preserve coords & dims
    k = xr.zeros_like(lat_da, dtype="float32")

    # coastal
    k = k.where(lat_da < 5, 0.85)

    # forest / transition
    k = k.where(~((lat_da >= 5) & (lat_da < 8)), 0.92)

    # savanna
    k = k.where(lat_da >= 8, 0.95)

    return k

def kstd_by_lat_xr(lat_da):

    k = xr.full_like(lat_da, np.nan, dtype="float32")

    k = k.where(lat_da >= 5, 0.85)                    # lat < 5
    k = k.where(lat_da < 8, 0.92)                     # 5 ≤ lat < 8
    k = k.where(lat_da < 1e9, 0.95)                   # lat ≥ 8

    return k

lat_grid = cml_sat_xarr["y"].broadcast_like(cml_sat_xarr["rain_daily_total"][0])
kstd_grid = kstd_by_lat_xr(lat_grid)

def kstd_by_lat_xr(lat_da):

    return xr.where(
        lat_da < 5,
        0.85,
        xr.where(
            lat_da < 8,
            0.92,
            0.95
        )
    ).astype("float32")

lat_grid = cml_sat_xarr["y"].broadcast_like(cml_sat_xarr["rain_daily_total"][0])
kstd_grid = kstd_by_lat_xr(lat_grid)

dlat = float(kstd_grid["y"].diff("y").mean())
win = int(round(0.2 / dlat))  # ≈ 7 for your grid

kstd_smooth = kstd_grid.rolling(
    y=win,
    center=True,
    min_periods=1
).mean()

lat = lat_grid

def kstd_by_lat_xr(lat_grid, dtrans=0.5):
    """
    Latitude-dependent kstd with smooth transitions.

    Parameters
    ----------
    lat_grid : xr.DataArray
        Latitude field (2D or 1D, degrees_north)
    dtrans : float
        Transition width in degrees (default: 0.5)

    Returns
    -------
    kstd : xr.DataArray
        Smooth kstd field
    """

    kstd = xr.where(
        lat_grid < 5 - dtrans / 2, 0.85,
        xr.where(
            lat_grid < 5 + dtrans / 2,
            0.85 + (lat_grid - (5 - dtrans / 2)) / dtrans * (0.92 - 0.85),
            xr.where(
                lat_grid < 8 - dtrans / 2, 0.92,
                xr.where(
                    lat_grid < 8 + dtrans / 2,
                    0.92 + (lat_grid - (8 - dtrans / 2)) / dtrans * (0.95 - 0.92),
                    0.95
                )
            )
        )
    )

    return kstd.astype("float32")

lat_grid = cml_sat_xarr["y"].broadcast_like(cml_sat_xarr["rain_daily_total"][0])
kstd_grid = kstd_by_lat_xr(lat_grid)

def kstd_by_lat_xr(lat_grid, dtrans=0.5):
    """
    Latitude-dependent kstd with smooth transitions.
    Tuned to reduce wet bias while preserving convective cores.
    •	Coastal (<5°): 0.95
	•	Forest / transition (5–8°): 1.00
	•	Savanna (>8°): 1.05

    Parameters
    ----------
    lat_grid : xr.DataArray
        Latitude field (2D or 1D, degrees_north)
    dtrans : float
        Transition width in degrees (default: 0.5)

    Returns
    -------
    kstd : xr.DataArray
        Smooth kstd field
    """

    kstd = xr.where(
        lat_grid < 5 - dtrans / 2, 0.95,  # Coastal Savanna
        xr.where(
            lat_grid < 5 + dtrans / 2,
            0.95 + (lat_grid - (5 - dtrans / 2)) / dtrans * (1.00 - 0.95),
            xr.where(
                lat_grid < 8 - dtrans / 2, 1.00,  # Forest / Transition
                xr.where(
                    lat_grid < 8 + dtrans / 2,
                    1.00 + (lat_grid - (8 - dtrans / 2)) / dtrans * (1.05 - 1.00),
                    1.05  # Guinea Savanna
                )
            )
        )
    )

    return kstd.astype("float32")

lat_grid = cml_sat_xarr["y"].broadcast_like(cml_sat_xarr["rain_daily_total"][0])
kstd_grid = kstd_by_lat_xr(lat_grid)


#%%
# Lets do some plotting of the 15 min maps to be sure
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import shapely.vectorized as svec
import numpy as np
import xarray as xr

def add_geo(ax):
    ax.coastlines(resolution="10m", linewidth=1.1, color="black")
    ax.add_feature(cfeature.BORDERS, linewidth=0.9, edgecolor="black")
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.7, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10, "color": "black"}
    gl.ylabel_style = {"size": 10, "color": "black"}
    gl.xlocator = plt.MultipleLocator(1.0)
    gl.ylocator = plt.MultipleLocator(1.0)


cml = xr.open_dataset(r'/home/kkumah/Projects/cml-stuff/out_rain_trials/out_15min/CML-SAT_Rainfall_Estimates_15min_V1_20250920.nc')
# img = xr.open_dataset(r'/home/kkumah/Projects/cml-stuff/satellite_data/imergv07/data/')
R15_dat = cml['rain_rate']

rntme = np.random.choice(R15_dat.time, size=3, replace=False)

for t in rntme:
    da = R15_dat.sel(time=t)
    # da = Rd  # your DataArray (y=lat, x=lon)
    da = da.sortby("y")  # safety: ensures south->north increasing
    daily_bins = np.arange(0,4.0,0.2)
    # [
    #     0,   1,   2,   5,   10,
    #     20,  30,  40
    # ]

    base_cmap = plt.cm.Spectral_r  # or turbo   # or viridis, plasma, 
    daily_cmap = ListedColormap(
        base_cmap(np.linspace(0.05, 0.95, len(daily_bins) - 1))
    )

    daily_norm = BoundaryNorm(
        boundaries=daily_bins,
        ncolors=daily_cmap.N,
        clip=True
    )

    import cartopy.io.shapereader as shpreader
    import shapely.geometry as sgeom

    shp = shpreader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_0_countries"
    )

    reader = shpreader.Reader(shp)
    ghana_geom = None
    for rec in reader.records():
        if rec.attributes["NAME_LONG"] == "Ghana":
            ghana_geom = rec.geometry
            break

    # Ghana geometry is already loaded as ghana_geom (shapely)
    lon2d, lat2d = np.meshgrid(
        da["x"].values,
        da["y"].values
    )

    ghana_mask = svec.contains(ghana_geom, lon2d, lat2d)

    R_daily_ghana = da.where(ghana_mask)

    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(8,8), dpi=150)
    ax = plt.axes(projection=proj)

    ax.set_extent([-3.5, 1.25, 4.5, 11.2], crs=proj)

    # ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="aqua")
    # ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="aqua", edgecolor="none")
    ax.coastlines(resolution="10m", linewidth=1.0)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.7, edgecolor="0.25")

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="0.6", alpha=0.6, linestyle="--")
    gl.right_labels = False
    gl.top_labels = False

    # ensure y-axis is ordered correctly
    R_daily = R_daily_ghana.sortby("y")

    im = ax.pcolormesh(
        R_daily["x"].values,
        R_daily["y"].values,
        R_daily.values,
        cmap=daily_cmap,
        norm=daily_norm,
        transform=proj,
        shading="auto"
    )

    # Ghana outline on top
    ax.add_geometries(
        [ghana_geom],
        crs=proj,
        facecolor="none",
        edgecolor="black",
        linewidth=1.2,
        zorder=3
    )

    # add_geo(ax)
    t_ = pd.to_datetime(t, format='%Y-%m-%d %H:%M:%S')#.normalize()
    ax.set_title(f"Rain Rate at {str(t_)}", fontsize=16)

    cb = plt.colorbar(
        im, ax=ax, ticks=daily_bins,
        fraction=0.046, pad=0.04,
        extend='max'
    )
    cb.set_label("mm h$^{-1}$")

    plt.show()
