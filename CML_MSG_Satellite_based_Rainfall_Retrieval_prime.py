#%%
# package imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from datetime import timedelta
from My_program_utils import *

from rasterio.enums import Resampling
from joblib import Parallel, delayed

from pyproj import CRS, Transformer
from scipy.interpolate import griddata
import rioxarray  # enables .rio accessor on xarray objects
from rioxarray.merge import merge_arrays

#%%
# define paths
PATH_CML_RAIN = r'/home/kkumah/Projects/cml-stuff/out_cml_rain_dir'
# r'/home/kkumah/Projects/cml-stuff/out_cml_rain_dir_2025-11-17'
# 
PATH_MSG_BT = r'/home/kkumah/Projects/cml-stuff/satellite_data/msg/run_20251027_201416'
PATH_MSG_CLM = r'/home/kkumah/Projects/cml-stuff/satellite_data/msg_clm/run_20251027_201601'

#%% Constants
# Which MSG vars to resample (present in your examples)
BT_VARS = ['BT_IR108', 'BT_IR120', 'BT_WV062']   # linear interp
CLM_VARS = ['cloud_mask']                        # nearest interp

# 15-min cadence (CML); adjust if you prefer per-minute
TIME_ROUND = '15min'

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

def _find_geos_crs(ds):
    # 1) A data var that names a grid_mapping
    for v in ds.data_vars:
        gm = ds[v].attrs.get("grid_mapping")
        if gm and gm in ds:
            gmv = ds[gm]
            if gmv.attrs.get("grid_mapping_name") == "geostationary":
                wkt = gmv.attrs.get("crs_wkt") or gmv.attrs.get("spatial_ref")
                if wkt:
                    return CRS.from_wkt(wkt)
                # Build from individual attrs if WKT missing
                lon0 = float(gmv.attrs.get("longitude_of_projection_origin", 0.0))
                h    = float(gmv.attrs.get("perspective_point_height", 35785831.0))
                a    = float(gmv.attrs.get("semi_major_axis", 6378137.0))
                rf   = float(gmv.attrs.get("inverse_flattening", 298.257222101))
                sweep= gmv.attrs.get("sweep_angle_axis", "y")
                return CRS.from_proj4(f"+proj=geos +lon_0={lon0} +h={h} +a={a} +rf={rf} +sweep={sweep} +units=m +no_defs")

    # 2) Any variable that already *is* the grid mapping
    for name, var in ds.variables.items():
        if var.attrs.get("grid_mapping_name") == "geostationary":
            wkt = var.attrs.get("crs_wkt") or var.attrs.get("spatial_ref")
            if wkt:
                return CRS.from_wkt(wkt)
            lon0 = float(var.attrs.get("longitude_of_projection_origin", 0.0))
            h    = float(var.attrs.get("perspective_point_height", 35785831.0))
            a    = float(var.attrs.get("semi_major_axis", 6378137.0))
            rf   = float(var.attrs.get("inverse_flattening", 298.257222101))
            sweep= var.attrs.get("sweep_angle_axis", "y")
            return CRS.from_proj4(f"+proj=geos +lon_0={lon0} +h={h} +a={a} +rf={rf} +sweep={sweep} +units=m +no_defs")

    # 3) WKT sitting on coords or globals
    for coord in ("x", "y"):
        if coord in ds.coords:
            wkt = ds.coords[coord].attrs.get("spatial_ref") or ds.coords[coord].attrs.get("crs_wkt")
            if wkt:
                return CRS.from_wkt(wkt)
    for key in ("crs_wkt", "spatial_ref"):
        if key in ds.attrs:
            return CRS.from_wkt(ds.attrs[key])

    # 4) Fallback: MSG SEVIRI default
    return CRS.from_proj4("+proj=geos +lon_0=0 +h=35785831 +a=6378137 +rf=298.257222101 +sweep=y +units=m +no_defs")

def build_transformer_from_geos_attrs(geos_attrs: dict):
    """
    Create pyproj Transformer (GEOS→WGS84) from the 'geostationary' grid_mapping attrs.
    Works with CF-1.5 style attributes present in your netCDF.
    """
    # Required attributes in your file:
    lam = float(geos_attrs.get('longitude_of_projection_origin', 0.0))
    h   = float(geos_attrs.get('perspective_point_height', 35785831.0))
    a   = float(geos_attrs.get('semi_major_axis', 6378137.0))
    rf  = float(geos_attrs.get('inverse_flattening', 298.257222101))
    sweep = geos_attrs.get('sweep_angle_axis', 'y')  # 'x' or 'y'

    # PROJ geostationary CRS string
    # Note: proj uses +h for satellite height above the ellipsoid center.
    proj_geos = CRS.from_proj4(
        f"+proj=geos +lon_0={lam} +h={h} +a={a} +rf={rf} +sweep={sweep} +units=m +no_defs"
    )

    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(proj_geos, wgs84, always_xy=True)
    return transformer

def geos_xy_to_lonlat(ds_msg):
    """Return (lon2d, lat2d) for an MSG dataset with x/y in meters."""
    if "x" not in ds_msg.coords or "y" not in ds_msg.coords:
        raise ValueError("Dataset lacks 'x' and 'y' coordinates (meters in geostationary projection).")

    crs_geos = _find_geos_crs(ds_msg)
    crs_ll   = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(crs_geos, crs_ll, always_xy=True)

    x = ds_msg["x"].values
    y = ds_msg["y"].values
    X, Y = np.meshgrid(x, y)
    lon, lat = transformer.transform(X, Y)
    return lon, lat

def _griddata_resample(src_lon, src_lat, src_field, tgt_lon2d, tgt_lat2d,
                       method='linear', fill_method='nearest'):
    """
    Resample a 2D field (src) defined on irregular lon/lat to the target lon/lat grid.
    Uses scipy.griddata. Returns a 2D array on tgt grid.
    """
    mask = np.isfinite(src_field)
    if mask.sum() < 4:
        # not enough points; just return all-nan
        return np.full_like(tgt_lon2d, np.nan, dtype=float)

    pts = np.column_stack((src_lon[mask].ravel(), src_lat[mask].ravel()))
    vals = src_field[mask].ravel()
    tgt_pts = np.column_stack((tgt_lon2d.ravel(), tgt_lat2d.ravel()))

    out = griddata(pts, vals, tgt_pts, method=method)
    out = out.reshape(tgt_lon2d.shape)

    if np.isnan(out).any() and fill_method:
        # fill holes with a secondary method (usually 'nearest')
        fill = griddata(pts, vals, tgt_pts, method=fill_method).reshape(out.shape)
        nan_idx = np.isnan(out)
        out[nan_idx] = fill[nan_idx]

    return out

def geostationary_to_latlon(ds_msg,
                            target_latitudes,
                            target_longitudes,
                            method="linear",
                            bt_vars=BT_VARS,
                            clm_vars=CLM_VARS,
                            n_jobs=12):
    """
    Warp MSG geostationary 2D fields to a regular lat/lon grid, in parallel.
    - Continuous vars (BT) use `method` with nearest-fill.
    - Categorical vars (cloud_mask) use nearest only.
    Parallelization: per-variable over time with joblib.
    """
    # ---- target grid (compute once) ----
    lat_t = np.asarray(target_latitudes)
    lon_t = np.asarray(target_longitudes)
    Lon_t, Lat_t = np.meshgrid(lon_t, lat_t)

    # ---- source lon/lat (compute once) ----
    src_lon2d, src_lat2d = geos_xy_to_lonlat(ds_msg)

    times = ds_msg["time"].values
    candidate_vars = [v for v in (bt_vars + clm_vars) if v in ds_msg.data_vars]
    if not candidate_vars:
        raise ValueError("No MSG variables found matching BT_VARS/CLM_VARS in this dataset.")

    out = {}

    # helper to resample a single (var, time) slice
    def _resample_one(v, t_i, is_categorical):
        src = ds_msg[v].isel(time=t_i).values  # (y,x)
        if is_categorical:
            return _griddata_resample(src_lon2d, src_lat2d, src,
                                      Lon_t, Lat_t,
                                      method="nearest", fill_method=None).astype("float32")
        else:
            return _griddata_resample(src_lon2d, src_lat2d, src,
                                      Lon_t, Lat_t,
                                      method=method, fill_method="nearest").astype("float32")

    # parallel per variable (each submits a list of time jobs)
    for v in candidate_vars:
        is_cat = v in clm_vars
        # NOTE: use "processes" for best speed with SciPy; change to prefer="threads" if needed.
        frames = Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto")(
            delayed(_resample_one)(v, t_i, is_cat) for t_i in range(len(times))
        )
        out[v] = np.stack(frames, axis=0)

    data_vars = {v: (("time", "lat", "lon"), arr) for v, arr in out.items()}
    return xr.Dataset(data_vars=data_vars,
                      coords={"time": times, "lat": lat_t, "lon": lon_t})

#%% Main processing
# 1) Discover files
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
cml = xr.open_mfdataset(cml_files, combine='by_coords')
msg_bt  = xr.open_mfdataset(msg_bt_files, combine='by_coords')
msg_clm = xr.open_mfdataset(msg_clm_files, combine='by_coords')

# 3) Normalize times (to CML cadence) and drop duplicates
cml   = round_time_index(cml, TIME_ROUND)
msg_bt  = round_time_index(msg_bt, TIME_ROUND)
msg_clm = round_time_index(msg_clm, TIME_ROUND)

# --- 5) Time-align FIRST (minute rounding already done on your side) ---
common_times = np.intersect1d(np.intersect1d(cml.time.values, 
                                             msg_bt.time.values),
                                             msg_clm.time.values)
cml     = cml.sel(time=common_times)
msg_bt  = msg_bt.sel(time=common_times)
msg_clm = msg_clm.sel(time=common_times)


# --- 6) Clip CML to Ghana bbox ---
bbox = (-4.0, 1.5, 4.5, 11.5)
cml = cml.sel(lat=slice(bbox[1], bbox[3]), lon=slice(bbox[0], bbox[2]))

# --- 7) Attach CRS and warp MSG to lat/lon ---
# (Do this once per MSG dataset)
# Assign geostationary CRS (CF standard)
msg_bt = msg_bt.rio.write_crs("+proj=geos +lon_0=0 +h=35785831 +a=6378137 +b=6356752.31414 +sweep=x +no_defs", inplace=False)
msg_clm = msg_clm.rio.write_crs("+proj=geos +lon_0=0 +h=35785831 +a=6378137 +b=6356752.31414 +sweep=x +no_defs", inplace=False)

# Reproject both to EPSG:4326 (lat/lon)
msg_bt_ll  = msg_bt.rio.reproject("EPSG:4326", resampling=Resampling.nearest)
msg_clm_ll = msg_clm.rio.reproject("EPSG:4326", resampling=Resampling.mode)

# --- 3) Clip warped MSG to Ghana bbox too ---
msg_bt_ll  = msg_bt_ll.sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[3], bbox[2]))
msg_clm_ll = msg_clm_ll.sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[3], bbox[2]))

# --- 4) Harmonize to Ghana grid resolution ---
res_deg = 0.03
target_lats = np.arange(bbox[2], bbox[3] + 1e-6, res_deg)
target_lons = np.arange(bbox[0], bbox[1] + 1e-6, res_deg)

msg_bt_ll  = msg_bt_ll.interp(y=target_lats, x=target_lons, method="linear")
msg_clm_ll = msg_clm_ll.interp(y=target_lats, x=target_lons, method="linear")
cml_on_msg = cml.interp(lat=target_lats, lon=target_lons, method="linear")

# --- 5) Sanity check alignment ---
assert np.allclose(msg_bt_ll.y, cml_on_msg.lat)
assert np.allclose(msg_bt_ll.x, cml_on_msg.lon)


#%% Machine Learning Retrieval (XGB quantiles, day-wise splits; no classifier)
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
from quantnn.quantiles import posterior_quantiles
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

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

ds_rain["R_mm_per_h"] = ds_rain["R_mm_per_h"].astype("float32")
if {"lat", "lon"}.issubset(ds_rain["R_mm_per_h"].dims):
    ds_rain = ds_rain.rename({"lat": "y", "lon": "x"})
R = ds_rain["R_mm_per_h"].where(ds_rain["R_mm_per_h"] >= 0.01, 0.0)  # drizzle gate optional


# --------------------
# 2) 70/30 split by *day* (based on ds_bt.time)
# --------------------
days_all = np.asarray(pd.to_datetime(ds_bt.time.values).normalize(), dtype="datetime64[D]").astype(str)
dummy    = np.arange(len(days_all))
gss      = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, val30_idx = next(gss.split(dummy, dummy, groups=days_all))

times_all = ds_bt.time.values
times_70  = times_all[train_idx]
times_30  = times_all[val30_idx]

# Select 70% times
BT_IR108_70 = BT_IR108.sel(time=times_70)
BT_IR120_70 = BT_IR120.sel(time=times_70)
BT_WV062_70 = BT_WV062.sel(time=times_70)
BT_diff_70  = BT_diff .sel(time=times_70)
R_70        = R       .sel(time=times_70)

# Align
BT_IR108_70, BT_IR120_70, BT_WV062_70, BT_diff_70, R_70 = xr.align(
    BT_IR108_70, BT_IR120_70, BT_WV062_70, BT_diff_70, R_70, join="exact"
)

# Features to ("sample","feature"), target to ("sample",)
feat4 = xr.concat(
    [BT_IR108_70, BT_IR120_70, BT_WV062_70, BT_diff_70],
    dim=xr.DataArray(["IR108","IR120","WV062","IR108_minus_WV062"], dims="feature")
).transpose("time","y","x","feature")

feat4_s = feat4.stack(sample=("time","y","x")).transpose("sample","feature")
tgt_s   = R_70.stack(sample=("time","y","x"))

# Validity mask (concrete NumPy)
valid = (np.isfinite(feat4_s).all("feature") & np.isfinite(tgt_s)).compute().values
feat4_s = feat4_s.isel(sample=valid)
tgt_s   = tgt_s.isel(sample=valid)

# --------------------
# 3) Day-wise 70/30 split *over valid samples*
# --------------------
times_s = np.asarray(pd.to_datetime(feat4_s["time"].values).normalize(), dtype="datetime64[D]")
groups_per_sample = times_s.astype(str)

N = feat4_s.sizes["sample"]
idx_all = np.arange(N)
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
idx70, idx30 = next(gss.split(idx_all, np.zeros(N), groups=groups_per_sample))

# Optional cap for RAM
cap = 5_000_000
if idx70.size > cap:
    rng = np.random.RandomState(42)
    idx70 = rng.choice(idx70, size=cap, replace=False)

feat_names = ["BT_IR108", "BT_IR120", "BT_WV062", "IR108_minus_WV062"]
X_70 = feat4_s.isel(sample=idx70).values.astype("float32")
y_70 = tgt_s   .isel(sample=idx70).values.astype("float32")
X_30 = feat4_s.isel(sample=idx30).values.astype("float32")
y_30 = tgt_s   .isel(sample=idx30).values.astype("float32")

print("Shapes → X_70:", X_70.shape, "y_70:", y_70.shape, "| X_30:", X_30.shape, "y_30:", y_30.shape)

# # %%--------------------
# # 4) Inner split INSIDE 70% (for early-stopping selection)
# # --------------------
# times_70 = times_s[idx70]
# uniq_days_70 = np.unique(times_70)
# rng = np.random.RandomState(123)
# rng.shuffle(uniq_days_70)
# inner_split = int(0.9 * len(uniq_days_70))
# days_tr, days_val = set(uniq_days_70[:inner_split]), set(uniq_days_70[inner_split:])
# m_tr  = np.isin(times_70, list(days_tr))
# m_val = np.isin(times_70, list(days_val))

# X_tr,  y_tr  = X_70[m_tr],  y_70[m_tr]
# X_val, y_val = X_70[m_val], y_70[m_val]

# # --------------------
# # 5) Quantile models (q=0.70/0.75/0.80) in log1p space
# # --------------------
# f, invf = np.log1p, np.expm1
# y_tr_t, y_val_t, y_30_t = f(y_tr), f(y_val), f(y_30)

# dtr  = xgb.DMatrix(X_tr,  label=y_tr_t,  feature_names=feat_names, nthread=12)
# dval = xgb.DMatrix(X_val, label=y_val_t, feature_names=feat_names, nthread=12)
# d30  = xgb.DMatrix(X_30,                  feature_names=feat_names, nthread=12)

# def train_quantile_model(alpha, dtrain, dvalid, num_boost_round=500, esr=30, seed=150):
#     params = {
#         "objective": "reg:quantileerror",  # xgboost >= 2.0
#         "quantile_alpha": float(alpha),
#         "tree_method": "hist",
#         "max_depth": 10,
#         "eta": 0.10,
#         "subsample": 0.8,
#         "colsample_bytree": 0.9,
#         "seed": seed,
#         "nthread": 12,
#     }
#     return xgb.train(params, dtrain, num_boost_round=num_boost_round,
#                      early_stopping_rounds=esr, evals=[(dtrain,"train"), (dvalid,"val")],
#                      verbose_eval=50)

# q_list = [0.70, 0.75, 0.80]
# boosters_q = {q: train_quantile_model(q, dtr, dval) for q in q_list}

# # TRANSFORMED preds → invert → enforce monotone → mean of three
# preds_val = {q: invf(boosters_q[q].predict(dval)) for q in q_list}
# preds_30  = {q: invf(boosters_q[q].predict(d30))  for q in q_list}

# # Non-crossing
# qs_sorted = sorted(q_list)
# stack_val = np.maximum.accumulate(np.vstack([preds_val[q] for q in qs_sorted]), axis=0)
# stack_30  = np.maximum.accumulate(np.vstack([preds_30[q]  for q in qs_sorted]), axis=0)

# preds_val_nc = {q: stack_val[i] for i, q in enumerate(qs_sorted)}
# preds_30_nc  = {q: stack_30[i]  for i, q in enumerate(qs_sorted)}

# final_val = (preds_val_nc[0.70] + preds_val_nc[0.75] + preds_val_nc[0.80]) / 3.0
# final_30  = (preds_30_nc [0.70] + preds_30_nc [0.75] + preds_30_nc [0.80]) / 3.0
# final_val = np.clip(final_val, 0.0, None)
# final_30  = np.clip(final_30,  0.0, None)

# # --------------------
# # 6) Metrics on inner and 30%
# # --------------------
# def rel_bias(obs, sim):
#     m = np.isfinite(obs) & np.isfinite(sim)
#     if not m.any(): return np.nan
#     mu_res = np.nanmean(sim[m] - obs[m])
#     mu_obs = np.nanmean(obs[m])
#     return np.nan if mu_obs == 0 else mu_res / mu_obs

# def rmse(obs, sim):
#     m = np.isfinite(obs) & np.isfinite(sim)
#     if not m.any(): return np.nan
#     return np.sqrt(mean_squared_error(obs[m], sim[m]))

# def corr(obs, sim):
#     m = np.isfinite(obs) & np.isfinite(sim)
#     if m.sum() < 2: return np.nan
#     return np.corrcoef(obs[m], sim[m])[0, 1]

# def det_metrics(obs, sim, thr=0.1):
#     m = np.isfinite(obs) & np.isfinite(sim)
#     if not m.any(): return (np.nan,)*4
#     o = obs[m] >= thr; s = sim[m] >= thr
#     TP = np.sum(s & o); FP = np.sum(s & ~o); FN = np.sum(~s & o)
#     pod = TP/(TP+FN) if (TP+FN) else np.nan
#     far = FP/(TP+FP) if (TP+FP) else np.nan
#     csi = TP/(TP+FP+FN) if (TP+FP+FN) else np.nan
#     db  = (TP+FP)/(TP+FN)  if (TP+FN)  else np.nan
#     return pod, far, csi, db

# print("== Inner (mean of q70/75/80) ==")
# print("RelBias:", rel_bias(y_val, final_val))
# print("RMSE:",    rmse(y_val,  final_val))
# print("Corr:",     corr(y_val,  final_val))
# print("POD/FAR/CSI/DetBias:", det_metrics(y_val, final_val, thr=0.1))

# print("== Held-out 30% (mean of q70/75/80) ==")
# print("RelBias:", rel_bias(y_30, final_30))
# print("RMSE:",    rmse(y_30,  final_30))
# print("Corr:",     corr(y_30,  final_30))
# print("POD/FAR/CSI/DetBias:", det_metrics(y_30, final_30, thr=0.1))
#%% Visualization of Inner Validation
# plt scatter of yval and pred values
# plt.figure(figsize=(6,6))
# plt.scatter(y_val, final_val, alpha=0.3, s=5)
# plt.xlabel("Observed Rainfall (mm/h)")
# plt.ylabel("Predicted Rainfall (mm/h)")
# plt.title("XGB Satellite-based Rainfall Retrieval: Inner Validation")
# plt.xlim(0, 25)
# plt.ylim(0, 25)
# plt.plot([0, 25], [0, 25], 'r--')
# # add metrics
# text = f"RelBias: {rel_bias(y_val, final_val):.2f}\nRMSE: {rmse(y_val, final_val):.2f}\nCorr: {corr(y_val, final_val):.2f}"
# plt.text(0.1, 0.9, text, transform=plt.gca().transAxes)

# plt.grid(which='major', linestyle='--', alpha=0.5,lw=0.8)

#%%
# --------------------
# 7) Retrain on FULL 70% and predict maps for 30% times
# --------------------
# Train three quantile models on all 70%
# --- Train dense quantiles on ALL 70% samples (not wet-only) ---
f, invf = np.log1p, np.expm1
d70_all = xgb.DMatrix(X_70, label=f(y_70), feature_names=feat_names, nthread=12)

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

qs_dense = np.linspace(0.05, 0.95, 19)#np.linspace(0, 1, 34)[1:-1]  #             # 5–95% every 5%
boosters_by_q = {q: train_quantile(q, d70_all) for q in qs_dense}

# save the trained model
import os
import joblib

path_loc = "/home/kkumah/Projects/cml-stuff/out_train_model"
os.makedirs(path_loc, exist_ok=True)

model_path = os.path.join(
    path_loc,
    f"xgb_quantile_models_ghana_70pct_{cde_run_dte}.pkl"
)

joblib.dump(boosters_by_q, model_path, compress=3)

meta = {
    "qs_dense": qs_dense,
    "training_split": "70pct",
    "objective": "reg:quantileerror",
    "features": feat_names,        # if available
    "date_trained": cde_run_dte,
}

joblib.dump(meta, model_path.replace(".pkl", "_meta.pkl"))

# load model later
boosters_by_q = joblib.load(os.path.join(
    path_loc,'xgb_quantile_models_ghana_70pct_20251220.pkl'))

meta = joblib.load(model_path.replace(".pkl", "_meta.pkl"))
qs_dense = meta["qs_dense"]
# --- Refit quantiles on WET-ONLY samples in log1p space ---
# thr_wet   = 0.02
# f, invf   = np.log1p, np.expm1
# m_wet70   = (y_70 >= thr_wet)

# X70_wet   = X_70[m_wet70]
# y70_wet_t = f(y_70[m_wet70])

# d70_reg = xgb.DMatrix(X70_wet, label=y70_wet_t, feature_names=feat_names, nthread=12)

# def train_quantile(alpha, dtrain, num_boost_round=500, seed=150):
#     params = {
#         "objective": "reg:quantileerror",
#         "quantile_alpha": float(alpha),
#         "tree_method": "hist",
#         "max_depth": 10,
#         "eta": 0.08,
#         "subsample": 0.8,
#         "colsample_bytree": 0.9,
#         "seed": seed,
#         "nthread": 12,
#     }
#     return xgb.train(params, dtrain, num_boost_round=num_boost_round)

# q_list = [0.70, 0.75, 0.80]
# boosters_final = {q: train_quantile(q, d70_reg) for q in q_list}
# --- after you build d70_reg (the DMatrix for wet training) ---
# def train_quantile(alpha, dtrain, num_boost_round=500, seed=150):
#     params = {
#         "objective": "reg:quantileerror",
#         "quantile_alpha": float(alpha),
#         "tree_method": "hist",
#         "max_depth": 10,
#         "eta": 0.08,
#         "subsample": 0.8,
#         "colsample_bytree": 0.9,
#         "seed": seed,
#         "nthread": 12,
#     }
#     return xgb.train(params, dtrain, num_boost_round=num_boost_round)

# q_list = [0.70, 0.75, 0.80]
# boosters_by_q = {q: train_quantile(q, d70_reg) for q in q_list}
#%%
# Helpers for patch correction (relaxed) & smoothing
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape
from rasterio import features
from rasterstats import zonal_stats
from pyproj import CRS
from rasterio.transform import from_bounds
from scipy.ndimage import binary_opening, generate_binary_structure
from scipy.ndimage import uniform_filter

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

def xarray_meta_from_da(da):
    """Build rasterio-style meta dict from a 2D lon/lat DataArray (dims y,x)."""
    height = da.sizes["y"]
    width  = da.sizes["x"]
    xmin, xmax = float(da["x"].values.min()), float(da["x"].values.max())
    ymin, ymax = float(da["y"].values.min()), float(da["y"].values.max())
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    return {"height": height, "width": width, "transform": transform, "dtype": "float32"}

def correct_wet_dry(
    bin_rast,
    bt_rastdat,
    meta_data,
    *,
    k_std=0.5,             # your suggested penalty: mean + 0.5*std
    p_low=0.30,            # also require IR108 ≤ local 30th percentile of each patch
    abs_bt_max=270.0,      # and IR108 ≤ 270 K overall
    min_patch_px=12,       # drop tiny wet blobs
    morph_open=True        # denoise mask before stats
):
    """
    bin_rast   : 2D int array 0/1 (wet/dry) to be corrected
    bt_rastdat : 2D float IR108 array (same grid)
    meta_data  : dict with keys {height,width,transform,dtype}

    Keep rule (per patch):
        IR108 <= min( mean + k*std, percentile_low, abs_bt_max )
    """
    wet0 = (bin_rast == 1)

    # Optional: denoise small speckles before stats
    if morph_open:
        st = generate_binary_structure(2, 1)  # 3x3 cross
        wet0 = binary_opening(wet0, structure=st)

    # Remove very small patches
    # (cheap approximation via convolution on neighbors count)
    if min_patch_px > 1:
        from scipy.ndimage import uniform_filter
        neigh = uniform_filter(wet0.astype(float), size=3, mode="nearest") * 9
        wet0 = np.where(neigh >= min_patch_px, wet0, 0)

    # Vectorize remaining wet pixels
    gdf = array_to_vector(wet0.astype(np.int16), mask_values=[1], trns=meta_data["transform"])
    if len(gdf) == 0:
        out = np.where(np.isfinite(bt_rastdat), 0, np.nan)
        return out

    # Patch stats
    bt_for_stats = np.where(np.isfinite(bt_rastdat), bt_rastdat, -999.0)
    zs = zonal_stats(
        vectors=gdf.geometry, raster=bt_for_stats,
        stats=["mean", "std", f"percentile_{int(p_low*100)}"],
        affine=meta_data["transform"], nodata=-999.0
    )
    df = pd.DataFrame(zs).rename(columns={f"percentile_{int(p_low*100)}": "p_low"})
    gdf = pd.concat([gdf.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    # Rasterize thresholds back to grid
    mean_r = rasterize_me(meta_data, gdf, "mean")
    std_r  = rasterize_me(meta_data, gdf, "std")
    plo_r  = rasterize_me(meta_data, gdf, "p_low")

    # Composite threshold
    thr = np.nanmin(np.dstack([mean_r + k_std*std_r, plo_r, np.full_like(mean_r, abs_bt_max)]), axis=2)

    # Keep where IR108 is sufficiently cold within wet patches
    keep = (wet0 == 1) & np.isfinite(bt_rastdat) & (bt_rastdat <= thr)

    out = np.where(np.isfinite(bt_rastdat), keep.astype(np.int16), np.nan)
    return out

# def kstd_by_lat(lat):
#     """
#     lat: ndarray (same shape as raster)
#     returns: ndarray of k_std with same shape
#     """
#     lat = np.asarray(lat)

#     kstd = np.empty_like(lat, dtype="float32")

#     kstd[lat < 5.5] = 0.3                     # Coastal savanna
#     kstd[(lat >= 5.5) & (lat < 7.5)] = 0.7     # Forest zone
#     kstd[(lat >= 7.5) & (lat < 9.0)] = 1.0     # Transitional
#     kstd[lat >= 9.0] = 1.3                    # Guinea savanna

#     return kstd      # very strict (anvil-dominated)

def kstd_by_lat(lat):
    """
    lat: array (2D lat grid)
    """
    lat = np.asarray(lat)

    k = np.zeros_like(lat, dtype="float32")

    # coastal → forest
    k = np.where(lat < 5.5, 0.4, k)
    k = np.where((lat >= 5.5) & (lat < 7.5),
                 0.6 + (lat - 5.5) * (0.6 - 0.3) / (7.5 - 5.5),
                 k)

    # forest → savanna
    k = np.where((lat >= 7.5) & (lat < 9.5),
                 0.9 + (lat - 7.5) * (0.9 - 0.6) / (9.5 - 7.5),
                 k)

    k = np.where(lat >= 9.5, 1.3, k)
    return k

def correct_wet_mask(bin_rast, bt_rast, meta, use_std=False, lat_grid=None):
    # relaxed: keep where BT <= mean + 2*std (per wet patch)
    gdf = array_to_vector(bin_rast, [1], meta["transform"])
    if len(gdf)==0: 
        return bin_rast
    bt_stats = np.where(np.isfinite(bt_rast), bt_rast, -999.0)
    zs = zonal_stats(gdf.geometry, bt_stats, stats=["mean","std"],
                     affine=meta["transform"], nodata=-999.0)
    gdf = pd.concat([gdf.reset_index(drop=True), pd.DataFrame(zs)], axis=1)
    mean_r = rasterize_me(meta, gdf, "mean"); std_r = rasterize_me(meta, gdf, "std")
    # choose threshold: mean only (old) or mean + k_std*std
    if use_std:
        
        kstd_grid = kstd_by_lat(lat_grid)

        thr = mean_r - (kstd_grid * std_r)
        # thr = mean_r - (k_std * std_r)
    else:
        thr = mean_r
    keep = (bin_rast==1) & (bt_rast <= thr)
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

def predict_slice_meanq(time_val, win_smooth=('Yes', 3), apply_patch=True,
                        drizzle_floor=0.10, use_trimmed=False, kstd=0.7,):
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
    if win_smooth[0] == "Yes" and win_smooth[1] > 1:
        rain_map = smooth_da_mean(rain_map, win=win_smooth)

    rain_map_cor = rain_map.copy()

    # optional patch correction by IR108
    if apply_patch:
        lat_grid = np.repeat(b1["y"].values[:, None], b1.sizes["x"], axis=1)
        meta = xarray_meta_from_da(b1)
        wet_grid = (rain_map_cor.values > 0).astype(np.int16)
        corr_wet = correct_wet_mask(wet_grid, 
                                    b1.values.astype("float32"), 
                                    meta, use_std=True,                                     
                                    lat_grid=lat_grid)
        
        rain_map_cor = rain_map_cor.where(corr_wet == 1, 0.0)
    # smoothing
    # if win_smooth[0] == "Yes":
    #     rain_smooth = smooth_da_mean(rain_map_cor, win=win_smooth[1])
    return rain_map_cor, rain_map

from quantnn.quantiles import posterior_mean, posterior_quantiles

def predict_slice_denseq_with_patch(
    time_val,
    mode="highq",          # "highq" or "median"
    win_smooth=3,
    drizzle_floor=0.02,
    apply_patch=True,
    highq_range=(0.50, 0.75),  # range for high-q mean
):
    """
    mode = "highq"  → mean over qs_dense in highq_range
    mode = "median" → posterior median via posterior_quantiles
    Returns: (rain_corr, rain_raw)
    """
    # --- gather features ---
    b1 = BT_IR108.sel(time=time_val).where(mask_cloud.sel(time=time_val))
    b2 = BT_IR120.sel(time=time_val).where(mask_cloud.sel(time=time_val))
    b3 = BT_WV062.sel(time=time_val).where(mask_cloud.sel(time=time_val))
    bd = BT_diff .sel(time=time_val).where(mask_cloud.sel(time=time_val))
    valid = np.isfinite(b1) & np.isfinite(b2) & np.isfinite(b3) & np.isfinite(bd)

    if valid.sum().item() == 0:
        out = xr.zeros_like(b1).fillna(0.0)
        out.name = "R_pred_mm_per_h"
        return out, out

    X_t = np.column_stack([
        b1.values[valid], b2.values[valid],
        b3.values[valid], bd.values[valid]
    ]).astype("float32")
    dX = xgb.DMatrix(X_t, feature_names=feat_names, nthread=18)

    # --- dense quantile predictions in log1p space, then invert ---
    pred_list = [invf(boosters_by_q[q].predict(dX)) for q in qs_dense]
    Yq = np.column_stack(pred_list).astype("float32")   # (Nvalid, n_q)
    Yq = np.maximum.accumulate(Yq, axis=1)              # enforce monotone

    # --- aggregation: HIGH-Q MEAN or POSTERIOR MEDIAN ---
    if mode == "highq":
        qmin, qmax = highq_range
        mask_q = (qs_dense >= qmin) & (qs_dense <= qmax)
        if mask_q.sum() == 0:
            raise ValueError(f"No qs_dense within {highq_range}, got {qs_dense}")
        r_flat = Yq[:, mask_q].mean(axis=1).astype("float32")

    elif mode == "median":
        # directly compute posterior median (q=0.5)
        med = posterior_quantiles(
            y_pred=Yq,
            quantiles=qs_dense,
            new_quantiles=[0.5]
        )
        # posterior_quantiles returns shape (Nvalid, 1)
        r_flat = med[:, 0].astype("float32")

    else:
        raise ValueError(f"Unknown mode={mode!r}. Use 'highq' or 'median'.")

    # --- rebuild 2D field ---
    rain_map = xr.full_like(b1, np.nan, dtype="float32")
    rain_map.values[valid] = np.clip(r_flat, 0.0, None)
    rain_map = rain_map.fillna(0.0)
    rain_map.name = "R_pred_mm_per_h"
    rain_map.attrs["long_name"] = f"Rainfall ({mode} aggregation)"
    rain_map.attrs["units"] = "mm h-1"

    # drizzle filter
    if drizzle_floor is not None:
        rain_map.values = np.where(rain_map.values < drizzle_floor, 0.0, rain_map.values)

    # smoothing (mean in a local neighborhood)
    if win_smooth and win_smooth > 1:
        rain_map = smooth_da_mean(rain_map, win=win_smooth)

    rain_raw = rain_map.copy()  # BEFORE patch correction

    # --- IR-based patch correction ---
    if apply_patch:
        meta = xarray_meta_from_da(b1)
        wet_grid = (rain_map.values >= drizzle_floor).astype(np.int16)
        bt_array = b1.values.astype("float32")

        # corr_wet = correct_wet_dry(
        #     wet_grid, bt_array, meta,
        #     k_std=0.5,
        #     p_low=0.30,
        #     abs_bt_max=270.0,
        #     min_patch_px=12,
        #     morph_open=True
        # )

        corr_wet = correct_wet_mask(wet_grid, 
                                    b1.values.astype("float32"), 
                                    meta, use_std=True, k_std=2.0)

        rain_corr = rain_map.where(corr_wet == 1, 0.0)
    else:
        rain_corr = rain_map

    # optional additional smoothing of corrected field
    if win_smooth and win_smooth > 1:
        rain_corr = smooth_da_mean(rain_corr, win=win_smooth)

    rain_corr.name = "R_pred_mm_per_h"
    rain_corr.attrs["long_name"] = (
        f"Rainfall ({mode} aggregation, IR108 patch-corrected, smoothed)"
    )
    rain_corr.attrs["units"] = "mm h-1"

    return rain_corr, rain_raw

def predict_slice_with_correction(time_val, drizzle_floor=0.10, smooth_size=3):
    # === gather features & mask ===
    b1 = BT_IR108.sel(time=time_val).where(mask_cloud.sel(time=time_val))
    b2 = BT_IR120.sel(time=time_val).where(mask_cloud.sel(time=time_val))
    b3 = BT_WV062.sel(time=time_val).where(mask_cloud.sel(time=time_val))
    bd = BT_diff .sel(time=time_val).where(mask_cloud.sel(time=time_val))
    valid = np.isfinite(b1) & np.isfinite(b2) & np.isfinite(b3) & np.isfinite(bd)

    if valid.sum().item() == 0:
        out = xr.zeros_like(b1).fillna(0.0)
        out.name = "R_pred_mm_per_h"
        return out, out

    X_t = np.column_stack([
        b1.values[valid], b2.values[valid],
        b3.values[valid], bd.values[valid]
    ]).astype("float32")

    # Quantile predictions (log1p space → invert)
    d_t = xgb.DMatrix(X_t, feature_names=feat_names, nthread=12)
    q_list = [0.70, 0.75, 0.80]

    preds = [invf(boosters_by_q[q].predict(d_t)) for q in q_list]  # list of (Nvalid,)
    Y     = np.vstack(preds)                                       # (3, Nvalid)
    Y     = np.maximum.accumulate(Y, axis=0)                       # enforce non-crossing
    rain_flat = Y.mean(axis=0).astype("float32")                   # mean of q70/75/80             # mean over q70/75/80

    # Rebuild 2D and apply drizzle floor + smoothing
    rain_map = xr.full_like(b1, np.nan, dtype="float32")
    rain_map.values[valid] = rain_flat
    rain_map = rain_map.fillna(0.0)

    if smooth_size and smooth_size > 1:
        sm = uniform_filter(rain_map.values, size=smooth_size, mode="nearest")
        rain_map.values = sm

    rain_map.values = np.where(rain_map.values < drizzle_floor, 0.0, rain_map.values)

    # === patch-based correction using IR108 (stricter/adaptive) ===
    meta = xarray_meta_from_da(b1)

    # initial “wet” mask from predicted rain after smoothing/floor
    wet_grid = (rain_map.values >= drizzle_floor).astype(np.int16)
    bt_array = b1.values.astype("float32")

    corr_wet = correct_wet_dry(
        wet_grid, bt_array, meta,
        k_std=0.5,        # your new penalty
        p_low=0.30,       # 30th percentile cap
        abs_bt_max=270.0, # absolute cap in K
        min_patch_px=12,
        morph_open=True
    )

    # Zero predicted rain where corrected mask says "dry"
    rain_corr = rain_map.where(corr_wet == 1, 0.0)
    rain_corr.name = "R_pred_mm_per_h"
    rain_corr.attrs["long_name"] = "XGB mean(q70/75/80) rainfall (IR108 patch-corrected, smoothed)"
    rain_corr.attrs["units"] = "mm h-1"
    return rain_corr, rain_map

# Run over held-out 30% times
pred_pairs = [predict_slice_meanq(t, win_smooth=3, apply_patch=True) for t in times_30]
R_pred_30_corr        = xr.concat([p[0] for p in pred_pairs], dim="time").transpose("time","y","x")
R_pred_30_uncorrected = xr.concat([p[1] for p in pred_pairs], dim="time").transpose("time","y","x")
R_pred_30_corr["time"]        = times_30
R_pred_30_uncorrected["time"] = times_30

# --- high-q mean version ---
pairs_highq = [
    predict_slice_denseq_with_patch(t, mode="highq", win_smooth=3, drizzle_floor=0.10, apply_patch=True)
    for t in times_30
]
R_highq_corr = xr.concat([p[0] for p in pairs_highq], dim="time").transpose("time", "y", "x")
R_highq_raw  = xr.concat([p[1] for p in pairs_highq], dim="time").transpose("time", "y", "x")
R_highq_corr["time"] = times_30
R_highq_raw["time"]  = times_30

# --- posterior median version ---
pairs_med = [
    predict_slice_denseq_with_patch(t, mode="median", win_smooth=3, drizzle_floor=0.10, apply_patch=True)
    for t in times_30
]
R_med_corr = xr.concat([p[0] for p in pairs_med], dim="time").transpose("time", "y", "x")
R_med_raw  = xr.concat([p[1] for p in pairs_med], dim="time").transpose("time", "y", "x")
R_med_corr["time"] = times_30
R_med_raw["time"]  = times_30

# pred_maps_corr = [predict_slice_with_correction(t, drizzle_floor=0.10, smooth_size=3) for t in times_30]
# R_pred_30_corr        = xr.concat([p[0] for p in pred_maps_corr], dim="time").transpose("time","y","x")
# R_pred_uncorrected_30 = xr.concat([p[1] for p in pred_maps_corr], dim="time").transpose("time","y","x")
# R_pred_30_corr["time"] = times_30
# R_pred_uncorrected_30["time"] = times_30
#%% # --------------------
# 8) Quick viz of one slice
# --------------------
# --- Discrete rainfall styling (21 colors over [0, 8]) ---
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec

# --- Choose a time slice ---
# np.datetime64('2025-06-18T03:30:00.000000000')
t0 = np.datetime64('2025-06-18T03:30:00.000000000')#times_30[92]
t0_str = np.datetime_as_string(t0, unit='m')

pairs_med = [
    predict_slice_denseq_with_patch(t0, mode="median", win_smooth=3, drizzle_floor=0.05, apply_patch=True)
]
R_med_corr = pairs_med[0][0].transpose("y", "x")
R_med_raw  = pairs_med[0][1].transpose("y", "x")
R_med_corr["time"] = t0
R_med_raw["time"]  = t0

pairs_highq = [
    predict_slice_denseq_with_patch(t0, mode="highq", win_smooth=3, drizzle_floor=0.10, apply_patch=True)
]
R_highq_corr = pairs_highq[0][0].transpose("y", "x")
R_highq_raw  = pairs_highq[0][1].transpose("y", "x")
R_highq_corr["time"] = t0
R_highq_raw["time"]  = t0

pred_pairs = [predict_slice_meanq(t0, win_smooth=3, apply_patch=True)]
R_pred_30_corr        = pred_pairs[0][0].transpose("y","x")
R_pred_30_uncorrected = pred_pairs[0][1].transpose("y","x")
R_pred_30_corr["time"]        = t0
R_pred_30_uncorrected["time"] = t0

# --- Pull inputs for t0 (only IR108 and CLM as requested) ---
bt108 = BT_IR108.sel(time=t0).where(mask_cloud.sel(time=t0))
clm   = ds_clm["cloud_mask"].sel(time=t0)

# Robust limits for BT
bt_vmin, bt_vmax = np.nanpercentile(bt108, [2, 98])

# --- Cloud mask (categorical) styling ---
# 0 Clear water, 1 Clear land, 2 Cloud, 3 No data
clm_bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
clm_norm   = mcolors.BoundaryNorm(clm_bounds, ncolors=4)
clm_cmap   = mcolors.ListedColormap(["#4477aa", "#66aa55", "#ffcc00", "#999999"])
clm_labels = {0: "Clear water", 1: "Clear land", 2: "Cloud", 3: "No data"}
legend_handles = [Patch(color=clm_cmap(i), label=clm_labels[i]) for i in range(4)]

# --- Small helper for geostyling ---
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

# --- Figure layout: 3 panels on top, 2 panels bottom ---
vmin_r, vmax_r = 0.0, 5
n_colors = 21
rain_bounds = np.linspace(vmin_r, vmax_r, n_colors)
rain_norm   = mcolors.BoundaryNorm(rain_bounds, ncolors=256)
rain_cmap   = plt.get_cmap("turbo")
rain_ticks  = np.round(rain_bounds[::2], 1)

fig = plt.figure(figsize=(18, 9), constrained_layout=True)
gs  = GridSpec(2, 3, figure=fig, height_ratios=[1, 1])

proj = ccrs.PlateCarree()

ax_pred   = fig.add_subplot(gs[0, 0], projection=proj)
ax_cml    = fig.add_subplot(gs[0, 1], projection=proj)
ax_uncorr = fig.add_subplot(gs[0, 2], projection=proj)
ax_bt108  = fig.add_subplot(gs[1, 0], projection=proj)
ax_clm    = fig.add_subplot(gs[1, 1], projection=proj)
# Leave gs[1,2] empty for a clean 2-wide bottom row
fig.add_subplot(gs[1, 2]).axis("off")

alg = 'OrigV'
# --- Top row: rain maps (discrete colorbar) ---
# R_pred_30_corr
# im0 = R_med_corr.sel(time=t0).plot(
#     ax=ax_pred, transform=proj, cmap=rain_cmap, norm=rain_norm, add_colorbar=False
# )
im0 = R_pred_30_corr.plot(
    ax=ax_pred, transform=proj, cmap=rain_cmap, norm=rain_norm, add_colorbar=False
)
add_geo(ax_pred)
ax_pred.set_title(f"Predicted rain (smoothed)\n{t0_str}\n{alg}")
cb0 = fig.colorbar(im0, ax=ax_pred, ticks=rain_ticks, fraction=0.046, pad=0.04)
cb0.set_label("mm h$^{-1}$")

im1 = R.sel(time=t0).plot(
    ax=ax_cml, transform=proj, cmap=rain_cmap, norm=rain_norm, add_colorbar=False
)
add_geo(ax_cml)
ax_cml.set_title("CML rain")
cb1 = fig.colorbar(im1, ax=ax_cml, ticks=rain_ticks, fraction=0.046, pad=0.04)
cb1.set_label("mm h$^{-1}$")
# R_pred_30_uncorrected
# im2 = R_pred_30_uncorrected.sel(time=t0).plot(
#     ax=ax_uncorr, transform=proj, cmap=rain_cmap, norm=rain_norm, add_colorbar=False
# )
im2 = R_pred_30_uncorrected.plot(
    ax=ax_uncorr, transform=proj, cmap=rain_cmap, norm=rain_norm, add_colorbar=False
)
add_geo(ax_uncorr)
ax_uncorr.set_title(f"Predicted rain (uncorrected)\n{t0_str}\n{alg}")
cb2 = fig.colorbar(im2, ax=ax_uncorr, ticks=rain_ticks, fraction=0.046, pad=0.04)
cb2.set_label("mm h$^{-1}$")

# --- Bottom row: inputs (IR108 BT and Cloud Mask) ---
im3 = bt108.plot(
    ax=ax_bt108, transform=proj, cmap="viridis", vmin=bt_vmin, vmax=bt_vmax, add_colorbar=False
)
add_geo(ax_bt108)
ax_bt108.set_title("IR108 Brightness Temperature [K]")
cb3 = fig.colorbar(im3, ax=ax_bt108, fraction=0.046, pad=0.04)
cb3.set_label("K")

im5 = clm.plot(
    ax=ax_clm, transform=proj, cmap=clm_cmap, norm=clm_norm, add_colorbar=False
)
add_geo(ax_clm)
ax_clm.set_title("Cloud mask (0/1/2/3)")
ax_clm.legend(handles=legend_handles, loc="lower left", frameon=True)

plt.show()


#%%
# Daily analysis
dte_t = [t for t in times_30 if pd.to_datetime(t).date().strftime('%Y-%m-%d')=='2025-06-18']

pred_pairs = [predict_slice_meanq(t, win_smooth=('Yes', 3), apply_patch=True) for t in dte_t]
pred_pairs_unsmooth = [predict_slice_meanq(t, win_smooth=('No',0), apply_patch=True) for t in dte_t]

R_pred_30_corr        = xr.concat([p[0] for p in pred_pairs], dim="time").transpose("time","y","x")
R_pred_30_corr["time"]        = dte_t

t0 = dte_t[0]

R_daily_mean = R_pred_30_corr.mean(dim='time') * 24
R_daily_mean["time"] = pd.to_datetime(t0).date().strftime('%Y-%m-%d') 
R_daily_mean.name = "R_daily_total_mm_per_day"
R_daily_mean.attrs["units"] = "mm day-1"


R_pred_30_corr_unsmooth        = xr.concat([p[0] for p in pred_pairs_unsmooth], dim="time").transpose("time","y","x")
R_pred_30_corr_unsmooth["time"]        = dte_t

R_daily_mean_unsmooth = R_pred_30_corr_unsmooth.mean(dim='time') * 24
R_daily_mean_unsmooth["time"] = pd.to_datetime(t0).date().strftime('%Y-%m-%d') 
R_daily_mean_unsmooth.name = "R_daily_total_mm_per_day"
R_daily_mean_unsmooth.attrs["units"] = "mm day-1"

imrg = xr.open_dataset(r'/home/kkumah/Projects/cml-stuff/satellite_data/imergv07/g4.timeAvgMap.GPM_3IMERGDL_07_precipitation.20250618-20250618.180W_90S_180E_90N.nc')


fg,ax = plt.subplots(figsize=(8,6), dpi=150)
lat_profile_sm = R_daily_mean.mean(dim='x').plot(y='y',ax=ax, color='b', label='XGB Daily Mean')
lat_profile_unsm = R_daily_mean_unsmooth.mean(dim='x').plot(y='y',ax=ax, color='g', label='XGB Daily Mean (unsmoothed)')

imrg_profile = imrg['GPM_3IMERGDL_07_precipitation'].mean(dim='lon').plot(y='lat',ax=ax, color='r', label='IMERG Daily Mean')

ax.legend()

imrg['GPM_3IMERGDL_07_precipitation'].plot(cmap='Spectral_r', vmin=0, vmax=20)

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

da = R_daily_mean_unsmooth  # your DataArray (y=lat, x=lon)
da = da.sortby("y")  # safety: ensures south->north increasing
daily_bins = np.arange(0,22,2.5)
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
    R_daily_mean["x"].values,
    R_daily_mean["y"].values
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

add_geo(ax)
ax.set_title(f"Daily total predicted rainfall\n{str(R_daily.time.values)}")

cb = plt.colorbar(
    im, ax=ax, ticks=daily_bins,
    fraction=0.046, pad=0.04,
    extend='max'
)
cb.set_label("mm day$^{-1}$")

plt.show()


#%%
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

    # gc.collect()

    return pdf_dict

bins = [0.2 * (2 ** i) for i in range(11)]

rdaily_sm = compute_pdf_elements_from_array(R_daily_mean.data, bins)
rdaily_unsm = compute_pdf_elements_from_array(R_daily_mean_unsmooth.data, bins)
imerg_ = compute_pdf_elements_from_array(imrg['GPM_3IMERGDL_07_precipitation'].values, bins)

fnt_size = 13
fg, axes = plt.subplots(figsize=(5, 5), sharey=True, constrained_layout=True)
lw = 3.5
# plot global pdfs
axes.plot(bins,imerg_['pdfv'], 
             label='IMERG', lw=lw, c='k', ls='-')
axes.plot(bins,rdaily_sm['pdfv'], 
             label='Sm', lw=lw, c='orange', ls='-')
axes.plot(bins,rdaily_unsm['pdfv'], 
             label='Unsm', lw=lw, c='b',ls='-')
axes.set_xscale('log')

axes.legend(fontsize=fnt_size, frameon=False, loc='best')

axes.set_ylabel('PDFv', fontsize=fnt_size)

axes.set_xlabel('Rainfall [mm/day]', fontsize=fnt_size)
