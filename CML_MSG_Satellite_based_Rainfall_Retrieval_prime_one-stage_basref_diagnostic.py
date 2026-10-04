#%%
# package imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from My_program_utils import *

import xarray as xr

from rasterio.enums import Resampling
from joblib import Parallel, delayed

from pyproj import CRS, Transformer
from scipy.interpolate import griddata
import rioxarray  # enables .rio accessor on xarray objects
from rioxarray.merge import merge_arrays

import xgboost as xgb

from quantnn.quantiles import posterior_quantiles
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

#%%
# define paths
PATH_CML_RAIN = r'/home/kkumah/Projects/cml-stuff/new_out_cml_Rain_bas_stats_training_ref'
# r'/home/kkumah/Projects/cml-stuff/out_cml_rain_dir'
# r'/home/kkumah/Projects/cml-stuff/out_cml_rain_dir_2025-11-17'
# 
PATH_MSG_BT = r'/home/kkumah/Projects/cml-stuff/satellite_data/msg'
# r'/home/kkumah/Projects/cml-stuff/satellite_data/msg/run_20251027_201416'
PATH_MSG_CLM = r'/home/kkumah/Projects/cml-stuff/satellite_data/msg_clm'
# r'/home/kkumah/Projects/cml-stuff/satellite_data/msg_clm/run_20251027_201601'
#%% Constants
# Which MSG vars to resample (present in your examples)
BT_VARS = ['BT_IR108', 'BT_IR120', 'BT_WV062']   # linear interp
CLM_VARS = ['cloud_mask']                        # nearest interp

# 15-min cadence (CML); adjust if you prefer per-minute
TIME_ROUND = '15min'

# wet/dry controls (tune)
WET_THR = 0.10         # mm/h threshold to define "wet"
WET_FRAC = 0.40        # fraction of wet pixels to sample per time

# sampling controls (tune)
N_TIMES_PER_MONTH = 800   # number of timestamps per month
N_PIX_PER_TIME    = 4000  # pixels per timestamp

feat_names = ["BT_IR108", "BT_IR120", "BT_WV062", "IR108_minus_WV062"]

rng = np.random.RandomState(42)

res_deg = 0.03

#%% Define fuctions

# -----------------------------
# Helpers
# -----------------------------
def _list_files(root, suffix='.nc'):
    if os.path.isdir(root):
        return sorted([os.path.join(r, f)
                       for r, _, fs in os.walk(root)
                       for f in fs if f.endswith(suffix)])
    else:
        # allow a single file
        return [root] if root.endswith(suffix) else []

def round_time_index(ds, freq=TIME_ROUND):
    if 'time' in ds.coords:
        ds = ds.assign_coords(time=ds['time'].dt.round(freq))
        # drop duplicate times after rounding (keep first)
        _, idx = np.unique(ds['time'].values, return_index=True)
        ds = ds.isel(time=np.sort(idx))
    return ds

def train_quantile(alpha, dtrain, num_boost_round=600, seed=150):
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": float(alpha),
        "tree_method": "hist",
        "max_depth": 10,
        "eta": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "seed": seed,
        "nthread": 20,
    }
    return xgb.train(params, dtrain, num_boost_round=num_boost_round)

def sample_times_month_balanced(times, n_per_month=N_TIMES_PER_MONTH, seed=42):
    t = pd.to_datetime(times)
    df = pd.DataFrame({"time": t})
    df["month"] = df["time"].dt.to_period("M").astype(str)

    out = []
    for m, g in df.groupby("month"):
        k = min(n_per_month, len(g))
        take = g.sample(n=k, replace=False, random_state=seed)
        out.append(take["time"].values)

    out = np.concatenate(out) if out else np.array([], dtype="datetime64[ns]")
    return np.sort(out)

def sample_pixels_one_time(b1, b2, b3, bd, r,
                           n_pix=N_PIX_PER_TIME, wet_thr=WET_THR, wet_frac=WET_FRAC,
                           seed=None):

    b1 = np.asarray(b1).squeeze()
    b2 = np.asarray(b2).squeeze()
    b3 = np.asarray(b3).squeeze()
    bd = np.asarray(bd).squeeze()
    r  = np.asarray(r).squeeze()

    if not (b1.shape == b2.shape == b3.shape == bd.shape == r.shape):
        raise ValueError(
            "Shape mismatch in sample_pixels_one_time:\n"
            f"b1: {b1.shape}\n"
            f"b2: {b2.shape}\n"
            f"b3: {b3.shape}\n"
            f"bd: {bd.shape}\n"
            f"r : {r.shape}\n"
        )

    valid = np.isfinite(b1) & np.isfinite(b2) & np.isfinite(b3) & np.isfinite(bd) & np.isfinite(r)
    if valid.sum() == 0:
        return None, None

    wet = valid & (r >= wet_thr)
    dry = valid & (r <  wet_thr)

    n_wet = int(n_pix * wet_frac)
    n_dry = n_pix - n_wet

    wet_idx = np.flatnonzero(wet)
    dry_idx = np.flatnonzero(dry)
    any_idx = np.flatnonzero(valid)

    rs = np.random.RandomState(seed) if seed is not None else np.random

    pick = []
    if wet_idx.size > 0:
        k = min(n_wet, wet_idx.size)
        pick.append(rs.choice(wet_idx, size=k, replace=False))
    if dry_idx.size > 0:
        k = min(n_dry, dry_idx.size)
        pick.append(rs.choice(dry_idx, size=k, replace=False))

    pick = np.concatenate(pick) if len(pick) else np.array([], dtype=int)

    # top-up if we didn't reach n_pix
    if pick.size < n_pix and any_idx.size > pick.size:
        need = n_pix - pick.size
        remaining = np.setdiff1d(any_idx, pick, assume_unique=False)
        if remaining.size > 0:
            k = min(need, remaining.size)
            pick = np.concatenate([pick, rs.choice(remaining, size=k, replace=False)])

    if pick.size == 0:
        return None, None

    X = np.column_stack([
        b1.ravel()[pick],
        b2.ravel()[pick],
        b3.ravel()[pick],
        bd.ravel()[pick],
    ]).astype("float32")
    y = r.ravel()[pick].astype("float32")
    return X, y


def preprocess_cml_grid_only(ds):
    """
    Keep only gridded CML variables and remove link-level variables.
    This prevents hidden broadcasting between (time, lat, lon) and (time, link).
    """
    keep = [
        "R_mm_per_h",
        "cml_support_confidence",
        "cml_support_mask",
        "cml_coverage_quality",
    ]
    keep = [v for v in keep if v in ds.data_vars]
    ds = ds[keep]

    rename = {}
    if "lat" in ds.dims:
        rename["lat"] = "y"
    if "lon" in ds.dims:
        rename["lon"] = "x"
    if rename:
        ds = ds.rename(rename)

    return ds

def preprocess_msg_bt(ds):
    keep = [v for v in BT_VARS if v in ds.data_vars]
    ds = ds[keep]

    # Ensure scalar time becomes a real time dimension if needed.
    if "time" in ds.coords and "time" not in ds.dims:
        ds = ds.expand_dims(time=[pd.to_datetime(ds["time"].values)])

    return ds

def preprocess_msg_clm(ds):
    keep = [v for v in CLM_VARS if v in ds.data_vars]
    ds = ds[keep]

    # CLM files often have cloud_mask(y, x) with scalar time.
    # Convert scalar time coordinate into time dimension.
    if "time" in ds.coords and "time" not in ds.dims:
        ds = ds.expand_dims(time=[pd.to_datetime(ds["time"].values)])

    return ds

def get_file_time(path):
    """Extract one timestamp from a single NetCDF file."""
    with xr.open_dataset(path, decode_times=True) as ds:
        if "time" not in ds.coords and "time" not in ds:
            raise ValueError(f"No time coordinate found in {path}")

        t = np.asarray(ds["time"].values).ravel()[0]
        return np.datetime64(pd.Timestamp(t).round(TIME_ROUND), "ns")
#%% Main processing
# 1) Discover files
print("Discovering files...")
cml_files = _list_files(PATH_CML_RAIN)
msg_bt_files  = _list_files(PATH_MSG_BT)
msg_clm_files = _list_files(PATH_MSG_CLM)

if not cml_files:
    raise FileNotFoundError(f"No CML files in {PATH_CML_RAIN}")
if not msg_bt_files:
    raise FileNotFoundError(f"No MSG BT files in {PATH_MSG_BT}")
if not msg_clm_files:
    raise FileNotFoundError(f"No MSG CLM files in {PATH_MSG_CLM}")

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# 2) Open with xarray (safe concat by coords)
#    Tip: chunk lightly if files are large; here we let xarray decide.
print("Opening files...")
cml = xr.open_mfdataset(
    cml_files,
    combine="by_coords",
    preprocess=preprocess_cml_grid_only,
    data_vars="minimal",
    coords="minimal",
    compat="override",
    join="override",
    parallel=False,
)
print("CML files opened:!")


msg_bt = xr.open_mfdataset(
    msg_bt_files,
    combine="nested",
    concat_dim="time",
    preprocess=preprocess_msg_bt,
    data_vars="all",
    coords="minimal",
    compat="override",
    join="override",
    parallel=False,
)
print("MSG BT files opened!")

msg_clm = xr.open_mfdataset(
    msg_clm_files,
    combine="nested",
    concat_dim="time",
    preprocess=preprocess_msg_clm,
    data_vars="all",
    coords="minimal",
    compat="override",
    join="override",
    parallel=False,
)

# Replace fake integer CLM time index with real timestamps from files.
msg_clm_times = np.array(
    [get_file_time(f) for f in msg_clm_files],
    dtype="datetime64[ns]"
)

if msg_clm.sizes["time"] != len(msg_clm_times):
    raise ValueError(
        f"CLM time length mismatch: dataset has {msg_clm.sizes['time']} times, "
        f"but extracted {len(msg_clm_times)} file timestamps."
    )

msg_clm = msg_clm.assign_coords(time=msg_clm_times)

print("Fixed msg_clm time dtype:", msg_clm.time.dtype, msg_clm.time.values[:3])

print("MSG CLM files opened!")

# sanity checks
print("msg_bt dims:", msg_bt.dims)
print("msg_clm dims:", msg_clm.dims)
print("msg_bt time size:", msg_bt.sizes.get("time"))
print("msg_clm time size:", msg_clm.sizes.get("time"))
print("cloud_mask dims:", msg_clm["cloud_mask"].dims)
print("CML dims:", cml['R_mm_per_h'].dims)

# 3) Normalize times (to CML cadence) and drop duplicates
cml   = round_time_index(cml, TIME_ROUND)
msg_bt  = round_time_index(msg_bt, TIME_ROUND)
msg_clm = round_time_index(msg_clm, TIME_ROUND)

# --- 4) Time-align FIRST (minute rounding already done on your side) ---
common_times = np.intersect1d(np.intersect1d(cml.time.values, 
                                             msg_bt.time.values),
                                             msg_clm.time.values)
cml     = cml.sel(time=common_times)
msg_bt  = msg_bt.sel(time=common_times)
msg_clm = msg_clm.sel(time=common_times)

# --- 5) Clip CML to Ghana bbox ---
bbox = (-4.0, 1.25, 4.5, 11.25)  # lon_min, lon_max, lat_min, lat_max

cml = cml.sortby("y").sortby("x")
cml = cml.sel(
    y=slice(bbox[2], bbox[3]),
    x=slice(bbox[0], bbox[1]),
)

# --- 6) Attach CRS and warp MSG to lat/lon ---
# (Do this once per MSG dataset)
# Assign geostationary CRS (CF standard)
msg_bt = msg_bt.rio.write_crs(
                            "+proj=geos +lon_0=0 +h=35785831 +a=6378137 +b=6356752.31414 +sweep=x +no_defs", 
                            inplace=False
                            )
msg_clm = msg_clm.rio.write_crs(
                            "+proj=geos +lon_0=0 +h=35785831 +a=6378137 +b=6356752.31414 +sweep=x +no_defs", 
                            inplace=False
                            )

# Reproject both to EPSG:4326 (lat/lon)
msg_bt_ll  = msg_bt.rio.reproject(
             "EPSG:4326", 
             resampling=Resampling.nearest
             )
msg_clm_ll = msg_clm.rio.reproject(
             "EPSG:4326", 
             resampling=Resampling.mode
             )

# --- 7) Clip warped MSG to Ghana bbox too ---
msg_bt_ll = msg_bt_ll.sortby("y").sortby("x")
msg_clm_ll = msg_clm_ll.sortby("y").sortby("x")

msg_bt_ll = msg_bt_ll.sel(
    x=slice(bbox[0], bbox[1]),
    y=slice(bbox[2], bbox[3]),
)

msg_clm_ll = msg_clm_ll.sel(
    x=slice(bbox[0], bbox[1]),
    y=slice(bbox[2], bbox[3]),
)

# msg_bt_ll_  = msg_bt_ll.sel(x=slice(bbox[0], bbox[2]), y=slice(bbox[1], bbox[3]))

# --- 8) Harmonize to Ghana grid resolution ---
target_lats = np.arange(bbox[2], bbox[3] + 1e-6, res_deg)
target_lons = np.arange(bbox[0], bbox[1] + 1e-6, res_deg)

msg_bt_ll  = msg_bt_ll.interp(y=target_lats, x=target_lons, method="linear")
msg_clm_ll = msg_clm_ll.interp(y=target_lats, x=target_lons, method="nearest")
cml_on_msg = cml.interp(y=target_lats, x=target_lons, method="linear")


# --- 9) Sanity check alignment --- Operational
assert np.allclose(msg_bt_ll.y, cml_on_msg.y)
assert np.allclose(msg_bt_ll.x, cml_on_msg.x)

print("Datasets aligned:")

#%% Operational training (NO validation)

print("Starting operational training...")

# --------------------
# 0) Inputs already in memory
# --------------------
ds_bt   = msg_bt_ll.copy()
for v in ["BT_IR108", "BT_IR120", "BT_WV062"]:
    ds_bt[v] = ds_bt[v].astype("float32")

ds_clm  = msg_clm_ll.copy()
ds_rain = cml_on_msg.copy()

# --------------------
# 1) Build features + target on common grid/time
# --------------------
mask_cloud = (ds_clm["cloud_mask"] == 2)  # cloud pixels only
BT_IR108 = ds_bt["BT_IR108"].where(mask_cloud)
BT_IR120 = ds_bt["BT_IR120"].where(mask_cloud)
BT_WV062 = ds_bt["BT_WV062"].where(mask_cloud)
BT_diff  = (BT_IR108 - BT_WV062).where(mask_cloud)

R = cml_on_msg["R_mm_per_h"].astype("float32")
R = R.where(R >= 0.01, 0.0)
R = R.where(R >= 0.0, 0.0)

R_sample = R.isel(time=slice(0, min(20, R.sizes["time"]))).values

print("Reference target quick-check:")
print("  min/max:", float(np.nanmin(R_sample)), float(np.nanmax(R_sample)))
print("  zero fraction:", float(np.nanmean(np.isfinite(R_sample) & (R_sample == 0))))
print("  wet fraction >= 0.10:", float(np.nanmean(np.isfinite(R_sample) & (R_sample >= WET_THR))))
print("  mean all:", float(np.nanmean(R_sample)))
print("  mean wet:", float(np.nanmean(R_sample[R_sample >= WET_THR])) if np.any(R_sample >= WET_THR) else np.nan)

print("R dims:", R.dims)
print("R shape:", R.shape)
print("BT_IR108 dims:", BT_IR108.dims)
print("BT_IR108 shape:", BT_IR108.shape)
# --------------------
# 2) Sample training data
# --------------------
# month-balanced time subset (operational training period)
times_sub = sample_times_month_balanced(ds_bt.time.values, n_per_month=N_TIMES_PER_MONTH, seed=42)
print("Selected times:", len(times_sub), "from", len(ds_bt.time.values))

# collect samples without stacking the full cube
X_list, y_list = [], []
for i, t in enumerate(times_sub):
    b1 = BT_IR108.sel(time=t).transpose("y", "x").values
    b2 = BT_IR120.sel(time=t).transpose("y", "x").values
    b3 = BT_WV062.sel(time=t).transpose("y", "x").values
    bd = BT_diff.sel(time=t).transpose("y", "x").values
    rr = R.sel(time=t).transpose("y", "x").values

    X_i, y_i = sample_pixels_one_time(
        b1, b2, b3, bd, rr,
        n_pix=N_PIX_PER_TIME, wet_thr=WET_THR, wet_frac=WET_FRAC,
        seed=42 + i
    )
    if X_i is None:
        continue
    X_list.append(X_i)
    y_list.append(y_i)

X_all = np.vstack(X_list).astype("float32")
y_all = np.concatenate(y_list).astype("float32")
print("Operational training samples:", X_all.shape, y_all.shape)

#--------------------
# 3) Train quantile models 
#--------------------
f, invf = np.log1p, np.expm1
dtrain = xgb.DMatrix(X_all, label=f(y_all), feature_names=feat_names, nthread=18)

qs_dense = np.linspace(0.05, 0.95, 19)
boosters_by_q = {q: train_quantile(q, dtrain) for q in qs_dense}

print(" Operational training complete........")
#%% Save models and metadata
import os, joblib

path_loc = "/home/kkumah/Projects/cml-stuff/out_train_model"
os.makedirs(path_loc, exist_ok=True)

model_path = os.path.join(path_loc, f"cmlsat_onestage_xgb_basref_diagnostic_models_{cde_run_dte}.pkl")
joblib.dump(boosters_by_q, model_path, compress=3)

meta = {
    "qs_dense": qs_dense,
    "training_period": "2025-06-14_to_2025-08-24",
    "features": feat_names,
    "wet_thr": WET_THR,
    "wet_frac": WET_FRAC,
    "n_times_per_month": N_TIMES_PER_MONTH,
    "n_pix_per_time": N_PIX_PER_TIME,
    "date_trained": cde_run_dte,
    "transform_y": "log1p",
}
joblib.dump(meta, model_path.replace(".pkl", "_meta.pkl"))
print("Saved:", model_path)

print("All done and trained models saved.")