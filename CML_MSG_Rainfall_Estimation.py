#%% Packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from My_program_utils import *
import joblib
import os, re, glob
from collections import OrderedDict
from pathlib import Path
from datetime import datetime

import xarray as xr
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from rasterstats import zonal_stats
import rioxarray
import geopandas as gpd
from shapely.geometry import shape
from rasterio import features
from pyproj import CRS

import xgboost as xgb
from quantnn.quantiles import posterior_quantiles
#%% Paths
path_to_msg_ir_fls = r'/home/kkumah/Projects/cml-stuff/satellite_data/msg_val'
path_to_msg_clm_fls = r'/home/kkumah/Projects/cml-stuff/satellite_data/msg_clm_val'
path_to_put_15min_cml_rainfall_estimates = r'/home/kkumah/Projects/cml-stuff/out_rain_trials/out_15min'
# r'/home/kkumah/Projects/cml-stuff/out_15min_cml_rain_oper'
path_to_put_daily_cml_rainfall_estimates = r'/home/kkumah/Projects/cml-stuff/out_rain_trials/new_out_daily'
# r'/home/kkumah/Projects/cml-stuff/out_rain_trials/out_daily_no_smooth_strict_lat_params'
# r'/home/kkumah/Projects/cml-stuff/out_rain_trials/out_daily'
# r'/home/kkumah/Projects/cml-stuff/out_daily_cml_rain_oper'

LOG_DIR = Path("/home/kkumah/Projects/cml-stuff/out_logs")

path_to_ml_model = r'/home/kkumah/Projects/cml-stuff/out_train_model'
#%% Floating Variables
# load model later
boosters_by_q = joblib.load(os.path.join(
                            path_to_ml_model,
                            'xgb_quantile_models_ghana_oper_20260225.pkl'))

meta = joblib.load(os.path.join(
                   path_to_ml_model, 
                   'xgb_quantile_models_ghana_oper_20260225_meta.pkl'))
qs_dense = meta["qs_dense"]

# 15-min cadence (CML); adjust if you prefer per-minute
TIME_ROUND = '15min'

feat_names = ["BT_IR108", "BT_IR120", "BT_WV062", "IR108_minus_WV062"]

f, invf = np.log1p, np.expm1

res_deg = 0.03

bbox = (-4.0, 1.25, 4.5, 11.25)

# your CRS string (keep as you used)
GEOS_CRS = "+proj=geos +lon_0=0 +h=35785831 +a=6378137 +b=6356752.31414 +sweep=x +no_defs"

BT_VARS  = ["BT_IR108", "BT_IR120", "BT_WV062"]
CLM_VARS = ["cloud_mask"]

lon_min, lon_max, lat_min, lat_max = bbox
target_lats = np.arange(lat_min, lat_max + 1e-6, res_deg)
target_lons = np.arange(lon_min, lon_max + 1e-6, res_deg)

#%% Functions
# -------------------------
# FILE SELECTION (per day)
# -------------------------
def list_nc(root):
    return sorted(glob.glob(os.path.join(root, "**", "*.nc"), recursive=True))
# def list_nc(root):
#     nc_files = []
#     for r, _, files in os.walk(root):
#         for f in files:
#             if f.endswith(".nc"):
#                 nc_files.append(os.path.join(r, f))
#     return sorted(nc_files)
# def pick_files_for_day(files, day_str):
#     """
#     day_str: 'YYYY-MM-DD'
#     Select files whose path/name contains YYYYMMDD.
#     """
#     ymd = day_str.replace("-", "")
#     return [fp for fp in files if ymd in os.path.basename(fp)]

# matches 20251017T123000Z

TS_RE = re.compile(r"\d{8}T\d{6}Z")

def extract_time(fp):
    m = TS_RE.search(fp)
    if not m:
        return None
    return pd.to_datetime(m.group(0), format="%Y%m%dT%H%M%SZ")

def pick_files_for_day_rounded(files, day_str, freq="15min"):
    """
    Keep ONE file per rounded 15-min slot.
    """
    day = pd.to_datetime(day_str)
    out = {}

    for fp in files:
        t = extract_time(fp)
        if t is None:
            continue

        if t.date() != day.date():
            continue

        t_round = t.round(freq)

        # keep first occurrence per rounded slot
        if t_round not in out:
            out[t_round] = fp

    # return files ordered by rounded time
    return [out[k] for k in sorted(out)]

LOG_DIR.mkdir(parents=True, exist_ok=True)

def is_valid_netcdf(path):
    try:
        xr.open_dataset(path, engine="netcdf4").close()
        return True
    except Exception:
        return False


def validate_files(files, product, day_str):
    valid, bad = [], []

    for f in files:
        if is_valid_netcdf(f):
            valid.append(f)
        else:
            bad.append(f)

    if bad:
        logfile = LOG_DIR / f"bad_{product}_files_{day_str.replace('-', '')}.txt"
        with open(logfile, "a") as lf:
            for bf in bad:
                lf.write(f"{datetime.utcnow().isoformat()} | {bf}\n")

    return valid

# -------------------------
# TIME ROUNDING
# -------------------------
def round_time_index(ds, freq=TIME_ROUND):
    if "time" in ds.coords:
        ds = ds.assign_coords(time=ds["time"].dt.round(freq))
        _, idx = np.unique(ds["time"].values, return_index=True)
        ds = ds.isel(time=np.sort(idx))
    return ds
# -------------------------
# RASTER VECTOR OPS
# -------------------------
def array_to_vector(array, mask_values, trns):
    """Return GeoDataFrame of polygons for pixels in mask_values."""
    array = array.astype(np.int16)
    wgs84 = CRS.from_epsg(4326).to_wkt()
    mask = np.isin(array, mask_values)
    recs = [
        {"properties": {"raster_val": int(v)}, "geometry": shape(s)}
        for _, (s, v) in enumerate(features.shapes(array, mask=mask, transform=trns))
    ]
    if not recs:
        return gpd.GeoDataFrame(columns=["raster_val", "geometry"], crs=wgs84)
    return gpd.GeoDataFrame.from_features(recs, crs=wgs84)

def rasterize_me(meta_data, polygon2rasterize, rast_val):
    """Rasterize a field from a GeoDataFrame onto a numpy array."""
    out = np.full((meta_data["height"], meta_data["width"]), np.nan, dtype=meta_data["dtype"])
    if len(polygon2rasterize) == 0:
        return out
    shp_gen = ((gm, vl) for gm, vl in zip(polygon2rasterize.geometry, polygon2rasterize[rast_val]))
    return features.rasterize(shapes=shp_gen,
                              fill=np.nan,
                              out=out,
                              transform=meta_data["transform"])

# def correct_wet_mask(bin_rast, bt_rast, meta, use_std=False, k_std=2.0):
#     # # cold-core filter: keep where BT <= (mean - k_std*std) within each wet patch
#     gdf = array_to_vector(bin_rast, [1], meta["transform"])
#     if len(gdf)==0: 
#         return bin_rast
#     bt_stats = np.where(np.isfinite(bt_rast), bt_rast, -999.0)
#     zs = zonal_stats(gdf.geometry, bt_stats, stats=["mean","std"],
#                      affine=meta["transform"], nodata=-999.0)
#     gdf = pd.concat([gdf.reset_index(drop=True), pd.DataFrame(zs)], axis=1)
#     mean_r = rasterize_me(meta, gdf, "mean"); std_r = rasterize_me(meta, gdf, "std")
#     # choose threshold: mean only (old) or mean - k_std*std
#     if use_std:
#         thr = mean_r - (k_std * std_r)
#     else:
#         thr = mean_r
#     keep = (bin_rast==1) & (bt_rast <= thr)
#     out = np.where(keep, 1, 0).astype(np.int16)
#     out = np.where(np.isfinite(bt_rast), out, np.nan)
#     return out
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
        lat_grid < 5 - dtrans / 2, 1.00,  # Coastal Savanna, 0.95
        xr.where(
            lat_grid < 5 + dtrans / 2,
            1.00 + (lat_grid - (5 - dtrans / 2)) / dtrans * (1.05 - 1.00), # 1.00
            xr.where(
                lat_grid < 8 - dtrans / 2, 0.89,  # Forest / Transition
                xr.where(
                    lat_grid < 8 + dtrans / 2,
                    0.89 + (lat_grid - (8 - dtrans / 2)) / dtrans * (1.0 - 0.89),
                    1.1  # Guinea Savanna and beyond
                )
            )
        )
    )

    return kstd.astype("float32")
# def kstd_by_lat_xr(lat_da):

#     return xr.where(
#         lat_da < 5,
#         0.85,
#         xr.where(
#             lat_da < 8,
#             0.92,
#             0.95
#         )
#     ).astype("float32")

def regime_aware_quantile_mean(
    Yq,
    qs,
    scene_values,
    q_low=0.30,
    q_high=0.70,
    low_band=(0.10, 0.30),
    mid_band=(0.30, 0.70),
    high_band=(0.70, 0.95),
):
    """
    Regime-aware aggregation of dense quantile predictions.

    Parameters
    ----------
    Yq : np.ndarray
        Shape (N, n_q), monotonic quantile predictions
    qs : np.ndarray
        Quantile levels (same length as n_q)
    scene_values : np.ndarray
        First-guess rain estimates (posterior mean AFTER patch correction)
    q_low, q_high : float
        Scene-relative thresholds for regime separation
    low_band, mid_band, high_band : tuple
        Quantile bands to average in each regime

    Returns
    -------
    r_out : np.ndarray
        Regime-adjusted rain estimates (N,)
    """

    # scene-relative thresholds
    t_low  = np.nanquantile(scene_values, q_low)
    t_high = np.nanquantile(scene_values, q_high)

    r_out = np.zeros_like(scene_values, dtype="float32")

    # index masks
    idx_low  = scene_values <= t_low
    idx_mid  = (scene_values > t_low) & (scene_values <= t_high)
    idx_high = scene_values > t_high

    def band_mean(Yq_sub, qmin, qmax):
        m = (qs >= qmin) & (qs <= qmax)
        return Yq_sub[:, m].mean(axis=1)

    # apply regime-specific aggregation
    if idx_low.any():
        r_out[idx_low] = band_mean(Yq[idx_low], *low_band)

    if idx_mid.any():
        r_out[idx_mid] = band_mean(Yq[idx_mid], *mid_band)

    if idx_high.any():
        r_out[idx_high] = band_mean(Yq[idx_high], *high_band)

    return r_out
#---------------------------
def low_or_q70_quantile_reducer(
    Yq,
    qs,
    q_thresh=0.30,
    q_high=0.78,
):
    """
    Minimal pixel-local conditional quantile reducer.

    Rule:
    - If posterior mean < Q(q_thresh): mean of quantiles <= q_thresh
    - Else: use Q(q_high)

    Parameters
    ----------
    Yq : np.ndarray
        Shape (N, n_q), monotonic quantile predictions
    qs : np.ndarray
        Quantile levels

    Returns
    -------
    r_out : np.ndarray
        Reduced rainfall estimate (N,)
    """

    from quantnn.quantiles import posterior_mean

    r_mean = posterior_mean(Yq, quantiles=qs)

    i_qth   = np.searchsorted(qs, q_thresh)
    i_qhigh = np.searchsorted(qs, q_high)

    low_q_mask = qs <= q_thresh

    r_out = np.empty_like(r_mean, dtype="float32")

    low_mask = r_mean < Yq[:, i_qth]

    # low intensities → mean of low quantiles
    if low_mask.any():
        r_out[low_mask] = Yq[low_mask][:, low_q_mask].mean(axis=1)

    # all others → fixed upper-mid quantile
    if (~low_mask).any():
        r_out[~low_mask] = Yq[~low_mask, i_qhigh]

    return r_out
#---------------------------

def regime_aware_quantile_mean_local(
    Yq,
    qs,
    r_mean,
    q_low=0.30,
    q_high=0.70,
):
    """
    Per-pixel regime-aware quantile aggregation.

    Parameters
    ----------
    Yq : ndarray (N, nq)
        Quantile predictions per pixel
    qs : ndarray (nq,)
        Quantile levels (ascending)
    r_mean : ndarray (N,)
        Posterior mean rainfall (mm h-1)
    q_low, q_high : float
        Regime split thresholds in percentile space

    Returns
    -------
    r_regime : ndarray (N,)
        Regime-aware rainfall estimate
    """

    # scene-wide reference (robust)
    p30 = np.percentile(r_mean, 30)
    p70 = np.percentile(r_mean, 70)

    low_mask  = r_mean < p30
    mid_mask  = (r_mean >= p30) & (r_mean <= p70)
    high_mask = r_mean > p70

    r_out = np.zeros_like(r_mean, dtype="float32")

    # LOW: conservative (suppress drizzle inflation)
    m = (qs >= 0.10) & (qs <= 0.30)
    if m.any():
        r_out[low_mask] = Yq[low_mask][:, m].mean(axis=1)

    # MID: posterior-stable
    m = (qs >= 0.30) & (qs <= 0.70)
    if m.any():
        r_out[mid_mask] = Yq[mid_mask][:, m].mean(axis=1)

    # HIGH: allow intensity, but not runaway
    m = (qs >= 0.70) & (qs <= 0.90)
    if m.any():
        r_out[high_mask] = Yq[high_mask][:, m].mean(axis=1)

    return r_out
#--------------------------
def correct_wet_mask(bin_rast, 
                     bt_rast, 
                     meta, 
                     use_std=False, 
                     lat_grid=None,
                     std_min=1.5,
                     std_max=6.0):
    """
    Patch-based wet-mask correction using IR BT statistics.

    Parameters
    ----------
    bin_rast : ndarray
        Binary wet mask (1 = wet)
    bt_rast : ndarray
        Brightness temperature field (K)
    meta : dict
        Raster metadata (includes transform)
    use_std : bool
        Whether to use mean - kstd*std thresholding
    lat_grid : ndarray or xr.DataArray
        Latitude-dependent kstd grid
    std_min, std_max : float
        Min/max bounds for effective BT std (K)

    Returns
    -------
    out : ndarray
        Corrected wet mask
    """

    gdf = array_to_vector(bin_rast, [1], meta["transform"])
    if len(gdf) == 0:
        return bin_rast

    bt_stats = np.where(np.isfinite(bt_rast), bt_rast, -999.0)

    zs = zonal_stats(
        gdf.geometry,
        bt_stats,
        stats=["mean", "std"],
        affine=meta["transform"],
        nodata=-999.0
    )

    gdf = pd.concat([gdf.reset_index(drop=True), pd.DataFrame(zs)], axis=1)

    mean_r = rasterize_me(meta, gdf, "mean")
    std_r  = rasterize_me(meta, gdf, "std")

    if use_std:
        # --- NEW: stabilize BT variability ---
        # std_eff = np.clip(std_r, std_min, std_max)

        # latitude-dependent kstd
        kstd_grid = lat_grid

        # threshold for keeping rain pixels
        thr = mean_r - (kstd_grid * std_r)
    else:
        thr = mean_r

    keep = (bin_rast == 1) & (bt_rast <= thr)

    out = np.where(keep, 1, 0).astype(np.int16)
    out = np.where(np.isfinite(bt_rast), out, np.nan)

    return out

def smooth_da_mean(da, win=3):
    wet = da.where(da > 0)
    num = wet.rolling(y=win, x=win, center=True, min_periods=1).sum()
    den = (~wet.isnull()).rolling(y=win, x=win, center=True, min_periods=1).sum()
    sm  = (num/den).where(da > 0, 0.0).fillna(0.0)
    sm.name = da.name; sm.attrs.update(da.attrs)
    sm.attrs["long_name"] = f"{da.attrs.get('long_name','pred')} (rolling {win}x{win})"
    return sm

def xarray_meta_from_da(da):
    """Build rasterio-style meta dict from a 2D lon/lat DataArray (dims y,x)."""
    height = da.sizes["y"]
    width  = da.sizes["x"]
    xmin, xmax = float(da["x"].values.min()), float(da["x"].values.max())
    ymin, ymax = float(da["y"].values.min()), float(da["y"].values.max())
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    return {"height": height, "width": width, "transform": transform, "dtype": "float32"}

#-------------------------
def apply_rcmq_cap(
    r_rcmq,
    Yq,
    qs,
    r_mean,
    scene_vals,
    q_cap=0.85,
    alpha=1.15,
    beta=1.20,
):
    """
    Apply local + area-aware cap to RCMQ output.
    """

    # quantile-based absolute ceiling
    q_idx = np.searchsorted(qs, q_cap)
    q_idx = min(q_idx, len(qs) - 1)
    q_cap_vals = Yq[:, q_idx]          # (N,)

    # scene-wide reference (scalar)
    scene_med = np.median(scene_vals)

    # broadcast scalar → vector
    scene_cap = np.full_like(r_mean, beta * scene_med)

    # final cap (element-wise)
    cap = np.minimum.reduce([
        q_cap_vals,
        alpha * r_mean,
        scene_cap,
    ])

    return np.minimum(r_rcmq, cap)

#-------------------------

def predict_slice_meanq(time_val, 
                        BT_IR108, BT_IR120, BT_WV062, BT_diff,
                        mask_cloud,
                        win_smooth,
                        apply_patch=True,
                        drizzle_floor=0.05, 
                        use_trimmed=False, 
                        ):
    # --- gather features ---
    b1 = BT_IR108.sel(time=time_val).where(mask_cloud.sel(time=time_val))
    b2 = BT_IR120.sel(time=time_val).where(mask_cloud.sel(time=time_val))
    b3 = BT_WV062.sel(time=time_val).where(mask_cloud.sel(time=time_val))
    bd = BT_diff .sel(time=time_val).where(mask_cloud.sel(time=time_val))
    valid = np.isfinite(b1) & np.isfinite(b2) & np.isfinite(b3) & np.isfinite(bd)

    if valid.sum().item() == 0:
        out = xr.zeros_like(b1).fillna(0.0); out.name = "R_pred_mm_per_h"
        return out, out

    X_t = np.column_stack([b1.values[valid], b2.values[valid],
                           b3.values[valid], bd.values[valid]]).astype("float32")
    dX  = xgb.DMatrix(X_t, feature_names=feat_names, nthread=18)

    # --- DENSE quantile aggregation (Option A) ---
    # predict in log1p-space, then invert; stack (N, n_q), enforce non-crossing
    pred_list = [invf(boosters_by_q[q].predict(dX)) for q in qs_dense]
    Yq = np.column_stack(pred_list).astype("float32")         # shape (Nvalid, n_q)
    Yq = np.maximum.accumulate(Yq, axis=1)                    # enforce monotone

    # aggregator: posterior mean (stable) or trimmed mean over mid-quantiles
    try:
        from quantnn.quantiles import posterior_mean
        r_flat = posterior_mean(Yq, quantiles=qs_dense).astype("float32")
    except Exception:
        # fallback: trimmed mean over 0.30–0.90
        m = (qs_dense >= 0.30) & (qs_dense <= 0.90) if use_trimmed else slice(None)
        r_flat = Yq[:, m].mean(axis=1).astype("float32")

    # --- rebuild map ---
    rain_map = xr.full_like(b1, np.nan, dtype="float32")
    rain_map.values[valid] = np.clip(r_flat, 0.0, None)
    rain_map = rain_map.fillna(0.0)
    rain_map.name = "R_pred_mm_per_h"
    rain_map.attrs["long_name"] = "Rainfall intensity (dense-quantile posterior mean)"
    rain_map.attrs["units"] = "mm h-1"

    # drizzle filter then smoothing
    if drizzle_floor is not None:
        rain_map.values = np.where(rain_map.values < drizzle_floor, 0.0, rain_map.values)
    if (win_smooth[0] == "Yes") and (win_smooth > 1):
        rain_map = smooth_da_mean(rain_map, win=win_smooth)

    rain_map_cor = rain_map.copy()

    # optional patch correction by IR108
    if apply_patch:
        # lat_grid = np.repeat(b1["y"].values[:, None], b1.sizes["x"], axis=1)
        lat_grid = b1["y"].broadcast_like(b1)
         # at each lat use a different patch correction constant
        kstd_grid = kstd_by_lat_xr(lat_grid)

        # smooth patch correction constant to avoid boundary lines in map
        # dlat = float(kstd_grid["y"].diff("y").mean())
        # win = int(round(0.2 / dlat))  # ≈ 7 for your grid        
        # kstd_smooth = kstd_grid.rolling(
        #     y=win,
        #     center=True,
        #     min_periods=1
        # ).mean()

        meta = xarray_meta_from_da(b1)
        wet_grid = (rain_map.values > 0).astype(np.int16)
        corr_wet = correct_wet_mask(wet_grid, 
                                    b1.values.astype("float32"), 
                                    meta, 
                                    use_std=True,
                                    lat_grid=kstd_grid)
        
        rain_map_cor = rain_map_cor.where(corr_wet == 1, 0.0)
    # smoothing
    # rain_smooth = smooth_da_mean(rain_map_cor, win=win_smooth)
    return rain_map_cor,  rain_map, #rain_smooth,

#-------------------------
# Regime aware conditional quantile retrieval version
# def predict_slice_regime_conditional_meanq(
#     time_val,
#     BT_IR108,
#     BT_IR120,
#     BT_WV062,
#     BT_diff,
#     mask_cloud,
#     win_smooth,
#     apply_patch=True,
#     drizzle_floor=0.03,
# ):
#     """
#     Conditional quantile rainfall retrieval with patch correction.

#     Returns
#     -------
#     rain_final : xr.DataArray
#         Final rainfall estimate (mm h-1)
#     rain_mean : xr.DataArray
#         Posterior-mean rainfall (diagnostic only)
#     """

#     # --------------------------------------------------
#     # 1. Gather features
#     # --------------------------------------------------
#     b1 = BT_IR108.sel(time=time_val).where(mask_cloud.sel(time=time_val))
#     b2 = BT_IR120.sel(time=time_val).where(mask_cloud.sel(time=time_val))
#     b3 = BT_WV062.sel(time=time_val).where(mask_cloud.sel(time=time_val))
#     bd = BT_diff .sel(time=time_val).where(mask_cloud.sel(time=time_val))

#     valid = np.isfinite(b1) & np.isfinite(b2) & np.isfinite(b3) & np.isfinite(bd)

#     if valid.sum().item() == 0:
#         out = xr.zeros_like(b1).fillna(0.0)
#         out.attrs["units"] = "mm h-1"
#         return out, out

#     # --------------------------------------------------
#     # 2. Dense quantile prediction
#     # --------------------------------------------------
#     X = np.column_stack([
#         b1.values[valid],
#         b2.values[valid],
#         b3.values[valid],
#         bd.values[valid],
#     ]).astype("float32")

#     dX = xgb.DMatrix(X, feature_names=feat_names, nthread=18)

#     Yq = np.column_stack([
#         invf(boosters_by_q[q].predict(dX)) for q in qs_dense
#     ]).astype("float32")

#     Yq = np.maximum.accumulate(Yq, axis=1)

#     # --------------------------------------------------
#     # 3. Posterior mean (first guess)
#     # --------------------------------------------------
#     from quantnn.quantiles import posterior_mean
#     r_mean_flat = posterior_mean(Yq, quantiles=qs_dense).astype("float32")

#     rain_mean = xr.full_like(b1, 0.0, dtype="float32")
#     rain_mean.values[valid] = np.clip(r_mean_flat, 0.0, None)
#     rain_mean.attrs["units"] = "mm h-1"

#     # --------------------------------------------------
#     # 4. Patch correction (AREA CONTROL)
#     # --------------------------------------------------
#     rain_patch = rain_mean.copy()

#     if apply_patch:
#         lat_grid = b1["y"].broadcast_like(b1)
#         kstd_grid = kstd_by_lat_xr(lat_grid)

#         meta = xarray_meta_from_da(b1)

#         # IMPORTANT: wet mask from PATCH, not from conditional output
#         wet_grid = (rain_patch.values > 0).astype(np.int16)

#         corr_wet = correct_wet_mask(
#             wet_grid,
#             b1.values.astype("float32"),
#             meta,
#             use_std=True,
#             lat_grid=kstd_grid,
#         )

#         rain_patch = rain_patch.where(corr_wet == 1, 0.0)

#     # --------------------------------------------------
#     # 5. Conditional quantile intensity (PIXEL-LOCAL)
#     # --------------------------------------------------
#     r_regime = low_or_q70_quantile_reducer(
#         Yq=Yq,
#         qs=qs_dense,
#         q_thresh=0.30,
#         q_high=0.70,
#     )

#     # --------------------------------------------------
#     # 6. APPLY PATCH MASK (THIS WAS MISSING)
#     # --------------------------------------------------
#     if apply_patch:
#         patch_mask_flat = rain_patch.values[valid] > 0
#         r_regime = np.where(patch_mask_flat, r_regime, 0.0)

#     # --------------------------------------------------
#     # 7. Final map
#     # --------------------------------------------------
#     rain_final = xr.full_like(b1, 0.0, dtype="float32")
#     rain_final.values[valid] = np.clip(r_regime, 0.0, None)
#     rain_final.attrs["units"] = "mm h-1"

#     # drizzle + smoothing
#     if drizzle_floor is not None:
#         rain_final.values[rain_final.values < drizzle_floor] = 0.0

#     if (win_smooth[0] == "Yes") and (win_smooth > 1):
#         rain_final = smooth_da_mean(rain_final, win=win_smooth)

#     return rain_final, rain_mean
#-------------------------
def predict_slice_regime_conditional_meanq(
    time_val,
    BT_IR108,
    BT_IR120,
    BT_WV062,
    BT_diff,
    mask_cloud,
    win_smooth,
    apply_patch=True,
    drizzle_floor=0.03,
):
    """
    Conditional quantile rainfall retrieval with PATCH-CONTROLLED OCCURRENCE
    and UNPATCHED INTENSITY logic.

    Returns
    -------
    rain_final : xr.DataArray
        Final rainfall estimate (mm h-1)  <-- USE THIS
    rain_mean : xr.DataArray
        Posterior-mean rainfall (diagnostic only)
    """

    # --------------------------------------------------
    # 1. Gather features
    # --------------------------------------------------
    b1 = BT_IR108.sel(time=time_val).where(mask_cloud.sel(time=time_val))
    b2 = BT_IR120.sel(time=time_val).where(mask_cloud.sel(time=time_val))
    b3 = BT_WV062.sel(time=time_val).where(mask_cloud.sel(time=time_val))
    bd = BT_diff .sel(time=time_val).where(mask_cloud.sel(time=time_val))

    valid = (
        np.isfinite(b1) &
        np.isfinite(b2) &
        np.isfinite(b3) &
        np.isfinite(bd)
    )

    if valid.sum().item() == 0:
        out = xr.zeros_like(b1).fillna(0.0)
        out.attrs["units"] = "mm h-1"
        return out, out

    # --------------------------------------------------
    # 2. Dense quantile prediction
    # --------------------------------------------------
    X = np.column_stack([
        b1.values[valid],
        b2.values[valid],
        b3.values[valid],
        bd.values[valid],
    ]).astype("float32")

    dX = xgb.DMatrix(X, feature_names=feat_names, nthread=18)

    Yq = np.column_stack([
        invf(boosters_by_q[q].predict(dX)) for q in qs_dense
    ]).astype("float32")

    # enforce monotonicity
    Yq = np.maximum.accumulate(Yq, axis=1)

    # --------------------------------------------------
    # 3. Posterior mean (FIRST GUESS — UNPATCHED)
    # --------------------------------------------------
    from quantnn.quantiles import posterior_mean
    r_mean_flat = posterior_mean(Yq, quantiles=qs_dense).astype("float32")

    rain_mean = xr.full_like(b1, 0.0, dtype="float32")
    rain_mean.values[valid] = np.clip(r_mean_flat, 0.0, None)
    rain_mean.attrs["units"] = "mm h-1"

    # --------------------------------------------------
    # 4. Patch correction (OCCURRENCE ONLY)
    # --------------------------------------------------
    corr_wet = None

    if apply_patch:
        lat_grid = b1["y"].broadcast_like(b1)
        kstd_grid = kstd_by_lat_xr(lat_grid)

        meta = xarray_meta_from_da(b1)

        # IMPORTANT: patch mask derived ONLY from rain_mean
        wet_grid = (rain_mean.values > 0).astype(np.int16)

        corr_wet = correct_wet_mask(
            wet_grid,
            b1.values.astype("float32"),
            meta,
            use_std=True,
            lat_grid=kstd_grid,
        )

    # --------------------------------------------------
    # 5. CONDITIONAL QUANTILE INTENSITY (UNPATCHED LOGIC)
    # --------------------------------------------------
    # NOTE: NO patch influence here — intensity only
    r_regime = low_or_q70_quantile_reducer(
        Yq=Yq,
        qs=qs_dense,
        q_thresh=0.30,   # below this → mean(0–0.3)
        q_high=0.75,     # above → q70
    )

    # --------------------------------------------------
    # 6. APPLY PATCH MASK (FINAL GATE ONLY)
    # --------------------------------------------------
    if apply_patch:
        patch_mask_flat = corr_wet[valid] == 1
        r_regime = np.where(patch_mask_flat, r_regime, 0.0)

    # --------------------------------------------------
    # 7. Final map
    # --------------------------------------------------
    rain_final = xr.full_like(b1, 0.0, dtype="float32")
    rain_final.values[valid] = np.clip(r_regime, 0.0, None)
    rain_final.attrs["units"] = "mm h-1"

    # drizzle floor
    if drizzle_floor is not None:
        rain_final.values[rain_final.values < drizzle_floor] = 0.0

    # smoothing
    if (win_smooth[0] == "Yes") and (win_smooth[1] > 1):
        rain_final = smooth_da_mean(rain_final, win=win_smooth[1])

    return rain_final, rain_mean
# -------------------------
# OPEN + REPROJECT + CLIP + INTERP (per day)
# -------------------------
def open_and_prepare_msg_day(day_str,
    bt_files_all,
    clm_files_all,
    require_clm=True
    ):
    """
    day_str: 'YYYY-MM-DD'
    """

    # -------------------------------------------------
    # 1. Collect files for day
    # -------------------------------------------------
    bt_day  = pick_files_for_day_rounded(bt_files_all, day_str)
    clm_day = pick_files_for_day_rounded(clm_files_all, day_str)

    # -------------------------------------------------
    # 2. Validate + log corrupt files
    # -------------------------------------------------
    bt_day  = validate_files(bt_day,  product="msg_ir",  day_str=day_str)
    clm_day = validate_files(clm_day, product="msg_clm", day_str=day_str)

    # -------------------------------------------------
    # 3. Safety checks
    # -------------------------------------------------
    if len(bt_day) == 0:
        raise RuntimeError(f"[{day_str}] No valid MSG IR files")

    if require_clm and len(clm_day) == 0:
        raise RuntimeError(f"[{day_str}] No valid MSG CLM files")

    # -------------------------------------------------
    # 4. Open datasets safely
    # -------------------------------------------------
    msg_bt = xr.open_mfdataset(
        bt_day,
        combine="by_coords",
        parallel=False
    )

    msg_clm = None
    if clm_day:
        msg_clm = xr.open_mfdataset(
            clm_day,
            combine="by_coords",
            parallel=False
        )

    msg_bt  = round_time_index(msg_bt,  TIME_ROUND)
    msg_clm = round_time_index(msg_clm, TIME_ROUND)

    # align on common times
    common_times = np.intersect1d(msg_bt.time.values, msg_clm.time.values)
    msg_bt  = msg_bt.sel(time=common_times)
    msg_clm = msg_clm.sel(time=common_times)

    # ---- coverage stats ----
    n = len(common_times)
    expected = int(pd.Timedelta(TIME_ROUND) / pd.Timedelta("15min")) * 96  # but TIME_ROUND is 15min, so just 96
    expected = 96
    coverage_frac = n / expected

    # attach CRS then reproject to EPSG:4326
    msg_bt  = msg_bt.rio.write_crs(GEOS_CRS, inplace=False)
    msg_clm = msg_clm.rio.write_crs(GEOS_CRS, inplace=False)

    msg_bt_ll  = msg_bt.rio.reproject("EPSG:4326", resampling=Resampling.nearest)
    msg_clm_ll = msg_clm.rio.reproject("EPSG:4326", resampling=Resampling.mode)

    # IMPORTANT: make sure y is ascending so slice(lat_min, lat_max) works
    if "y" in msg_bt_ll.coords:
        msg_bt_ll = msg_bt_ll.sortby("y")
    if "y" in msg_clm_ll.coords:
        msg_clm_ll = msg_clm_ll.sortby("y")

    # clip to bbox (lon=x, lat=y)
    msg_bt_ll  = msg_bt_ll.sel(x=slice(lon_min, lon_max), y=slice(lat_min, lat_max))
    msg_clm_ll = msg_clm_ll.sel(x=slice(lon_min, lon_max), y=slice(lat_min, lat_max))

    # harmonize to your fixed 0.03° grid
    msg_bt_ll  = msg_bt_ll.interp(y=target_lats, x=target_lons, method="linear")
    msg_clm_ll = msg_clm_ll.interp(y=target_lats, x=target_lons, method="nearest")

    # keep only needed vars
    msg_bt_ll  = msg_bt_ll[ [v for v in BT_VARS if v in msg_bt_ll.data_vars] ]
    msg_clm_ll = msg_clm_ll[ [v for v in CLM_VARS if v in msg_clm_ll.data_vars] ]

    return msg_bt_ll, msg_clm_ll, {"n_common": n, "expected": expected, "coverage_frac": coverage_frac}

# -------------------------
# PREDICT 15-MIN FOR A DAY
# -------------------------
def predict_day_15min(msg_bt_ll, msg_clm_ll, *,
                      win_smooth=('Yes', 3), apply_patch=True, 
                      drizzle_floor):
    # build masks & features (same as your draft)
    ds_bt = msg_bt_ll.copy()
    for v in BT_VARS:
        ds_bt[v] = ds_bt[v].astype("float32")

    ds_clm = msg_clm_ll.copy()
    mask_cloud = (ds_clm["cloud_mask"] == 2)

    BT_IR108 = ds_bt["BT_IR108"].where(mask_cloud)
    BT_IR120 = ds_bt["BT_IR120"].where(mask_cloud)
    BT_WV062 = ds_bt["BT_WV062"].where(mask_cloud)
    BT_diff  = (BT_IR108 - BT_WV062).where(mask_cloud)

    times_day = ds_bt.time.values

    preds = []
    for t in times_day:
        p_corr, _ = predict_slice_regime_conditional_meanq(
        t,
        BT_IR108=BT_IR108,
        BT_IR120=BT_IR120,
        BT_WV062=BT_WV062,
        BT_diff=BT_diff,
        mask_cloud=mask_cloud,
        win_smooth=win_smooth,
        apply_patch=apply_patch,
        drizzle_floor=drizzle_floor,        
        )
        preds.append(p_corr)

    R15 = xr.concat(preds, dim="time").transpose("time", "y", "x")
    R15 = R15.assign_coords(time=times_day)
    R15.name = "rain_rate"
    R15.attrs["units"] = "mm h-1"
    R15.attrs["long_name"] = "Predicted rainfall rate (15-min cadence)"

    msg_bt_ll.close()
    msg_clm_ll.close()

    return R15

# -------------------------
# DAILY TOTAL
# -------------------------
def daily_total_from_15min(R15):
    Rd = R15.mean("time") * 24.0
    # Rd = (R15 * 0.25).sum("time", skipna=True)
    Rd.name = "rain_daily_total"
    Rd.attrs["units"] = "mm day-1"
    Rd.attrs["long_name"] = "Daily total rainfall (mean rate × 24)"
    return Rd

def ensure_time_dim(da, time_value):
    """
    Ensure DataArray has a time dimension (CF-compliant).
    """
    if "time" not in da.dims:
        da = da.expand_dims(time=[pd.to_datetime(time_value)])
    else:
        da = da.assign_coords(time=pd.to_datetime(da.time.values))
    return da
# -------------------------
# SAVE NETCDF (1 file/day)
# -------------------------

def save_day_files(R15, Rd, day_str, alg="V1", producer="K. K. Kumah"):
    os.makedirs(path_to_put_15min_cml_rainfall_estimates, exist_ok=True)
    os.makedirs(path_to_put_daily_cml_rainfall_estimates, exist_ok=True)

    # --------------------------------------------------
    # FIX: ensure time dimension exists
    # --------------------------------------------------
    # R15 = ensure_time_dim(R15, R15.time.values[0])
    Rd  = ensure_time_dim(Rd,  pd.to_datetime(day_str))

    # datasets
    ds15 = R15.to_dataset(name="rain_rate")
    dsd  = Rd.to_dataset(name="rain_daily_total")

    # global attrs (keep concise but useful)
    common_attrs = {
        "title": "Ghana operational rainfall estimates from MSG SEVIRI + CML-trained XGBoost",
        "institution": "TRANS-AFRICAN HYDRO-METEOROLOGICAL OBSERVATORY (TAHMO)",
        "institution_url": "https://tahmo.org/",
        "producer": producer,
        "algorithm": f"XGBoost quantile regression; aggregation=posterior mean; patch_filter=IR108",
        "training_period": meta.get("training_period", "June–Aug 2025 (operational training)"),
        "time_coverage_start": str(pd.to_datetime(R15.time.values[0]).to_pydatetime()),
        "time_coverage_end": str(pd.to_datetime(R15.time.values[-1]).to_pydatetime()),
        "created_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spatial_domain": f"Ghana bbox lon[{lon_min},{lon_max}] lat[{lat_min},{lat_max}]",
        "grid_res_deg": str(res_deg),
        "crs": "EPSG:4326",
        "notes": "Operational forward processing; 15-min rate saved + daily total saved separately.",
        "model_file": os.path.basename("xgb_quantile_models_ghana_oper_20251229.pkl"),
    }
    ds15.attrs.update(common_attrs)
    ds15.attrs["n_time_steps"] = int(ds15.dims["time"])
    ds15.attrs["expected_steps_per_day"] = 96
    ds15.attrs["coverage_fraction"] = float(ds15.dims["time"] / 96)
    dsd.attrs.update(common_attrs)

    # filenames
    ymd = day_str.replace("-", "")
    fn15 = f"CML-SAT_Rainfall_Estimates_15min_{alg}_{ymd}.nc"
    fnd  = f"CML-SAT_Rainfall_Estimates_Daily_{alg}_{ymd}.nc"

    p15 = os.path.join(path_to_put_15min_cml_rainfall_estimates, fn15)
    pd1 = os.path.join(path_to_put_daily_cml_rainfall_estimates, fnd)

    # compression (good operational default)
    enc15 = {v: {"zlib": True, "complevel": 8, "dtype": "float32"} for v in ds15.data_vars}
    encd  = {v: {"zlib": True, "complevel": 8, "dtype": "float32"} for v in dsd.data_vars}

    ds15.to_netcdf(p15, encoding=enc15)
    dsd.to_netcdf(pd1, encoding=encd)

    print("Saved:", p15)
    print("Saved:", pd1)

#%%
# -------------------------
# MAIN OPERATIONAL DRIVER
# -------------------------
bt_all  = sorted(list_nc(path_to_msg_ir_fls))
clm_all = sorted(list_nc(path_to_msg_clm_fls))

# choose your operational period (example: Sep–Dec 2025)
days = pd.date_range("2025-09-03", "2025-12-31", freq="D")

for day in days:
    day_str = day.strftime("%Y-%m-%d")
    bt_day  = pick_files_for_day_rounded(bt_all,  day_str)
    clm_day = pick_files_for_day_rounded(clm_all, day_str)

    if (len(bt_day) == 0) or (len(clm_day) == 0):
        print("Skipping (missing files):", day_str)
        continue

    print("\nProcessing:", day_str, "| BT:", len(bt_day), "| CLM:", len(clm_day))

    msg_bt_ll, msg_clm_ll, cov = open_and_prepare_msg_day(day_str,bt_day, clm_day)

    if cov["n_common"] == 0:
        print("Skipping (no common times):", day_str)
        continue

    min_frac = 0.50  # you choose
    if cov["coverage_frac"] < min_frac:
        print(f"Skipping (low coverage {cov['coverage_frac']:.2%}):", day_str)
        continue

    R15 = predict_day_15min(msg_bt_ll, msg_clm_ll,
                            win_smooth=('Yes', 3), 
                            apply_patch=True, 
                            drizzle_floor=0.03, 
                            )

    Rd = daily_total_from_15min(R15)

    save_day_files(R15, Rd, day_str, alg="V1", producer="K. K. Kumah")

print("Done.")
print('*'*50)

xxx

#%%
# plot using cartopy to show boundary
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
import matplotlib.pyplot as plt
import shapely.geometry as sgeom
import shapely.vectorized as svec
import numpy as np

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


da = Rd  # your DataArray (y=lat, x=lon)
da = da.sortby("y")  # safety: ensures south->north increasing
daily_bins = np.arange(0,70.25,2.5)
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

# # Ghana outline on top
# ax.add_geometries(
#     [ghana_geom],
#     crs=proj,
#     facecolor="none",
#     edgecolor="black",
#     linewidth=1.2,
#     zorder=3
# )

# add_geo(ax)
ax.set_title(f"Daily total predicted rainfall\n{str(day_str)}")

cb = plt.colorbar(
    im, ax=ax, ticks=daily_bins,
    fraction=0.046, pad=0.04,
    extend='max'
)
cb.set_label("mm day$^{-1}$")

plt.show()


#%%
f,xx = plt.subplots()
xx.plot(Rd.mean(dim='x').values,Rd.mean(dim='x')['y'].values,label='CML-SAT')
xx.plot(imerg_daily_xarr[2].mean(dim='lon').values,imerg_daily_xarr[2].mean(dim='lon')['lat'].values,label='IMERG')
xx.plot(era5_daily_data[2].mean(dim='longitude').values,era5_daily_data[2].mean(dim='longitude')['latitude'].values,label='ERA5')
xx.set_title(f"Daily total predicted rainfall\n{str(pd.to_datetime(R15['time'].values[0]).normalize())}")
xx.set_xlabel("mm day$^{-1}$")
xx.set_ylabel("Latitude")
xx.legend()