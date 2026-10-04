"""
Function
"""

#%%
# import packages
import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd

import os, math

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


from pykrige.ok import OrdinaryKriging

#%%
# Constants and floating variables
# cadence (kept at native 15-min)
CADENCE_MIN = 15

# Config knobs (you can tweak later)
RSL_VALID_RANGE = (-150.0, -40.0)
MAX_JUMP_DB     = 10.0     # per 15-min step (TSL_AVG proxy)
FLATLINE_RUN    = 4        # consecutive steps with RSL_MIN==RSL_MAX==TSL_AVG
ATPC_STEP_DB    = 1.2
ATPC_GUARD_STEPS= 2

# Wet-dry config (used later)
TAU_FLOOR_DB    = 0.7
Z_MAD           = 3.0
NN_RADIUS_KM    = 30.0
NN_K_NEIGHBORS  = 3
NN_MIN_FRAC     = 0.4
NN_TIME_HALF_W  = 1        # ±1 step (15 min)

# Baseline config (used later)
BASELINE_HOURS      = 48
BASELINE_MIN_POINTS = 24
BASELINE_EXPAND_HRS = 72

# WAA (used later)
WAA_CONSTANT_DB = 1.5

# ITU a,b table (rounded GHz → (a,b) by pol H/V). Adjust if needed.
AB_TABLE = {
    ("13","H"):(0.045,1.10), ("13","V"):(0.055,1.12),
    ("15","H"):(0.060,1.10), ("15","V"):(0.073,1.12),
    ("18","H"):(0.110,1.08), ("18","V"):(0.130,1.10),
    ("23","H"):(0.220,1.05), ("23","V"):(0.260,1.07),
    ("38","H"):(0.900,1.00), ("38","V"):(1.100,1.02),
}


# Tunables (adjust after a first run if needed)
HARD_ABS_P        = 100.0   # gross sanity rail for |P|
Z_MAD_VAL         = 8.0     # value-domain rail: |P - med| <= Z * MAD
Z_MAD_DIFF        = 6.0     # diff-domain rail (spike filter) in units of MAD(ΔP)
DIFF_WIN_STEPS    = 9       # window for MAD of diffs (e.g., 9*15min = ~2.25h)
CADENCE_MIN       = 15      # used to scale thresholds with gaps
TAU_FLOOR_DB      = 0.3     # minimum per-step spike threshold in dB
DIVERGE_MAX_DB    = 12.0    # if (Pmax - Pmin) > this → mark Pmin bad
DIVERGE_MIN_DB    = -0.5    # if (Pmax - Pmin) < this (i.e. Pmin>Pmax) → mark Pmin bad
FLATLINE_RUN      = 6       # if BOTH Pmin&Pmax nearly unchanged for N steps → mask
FLAT_TOL          = 0.05    # ~0.05 dB is "no change" tolerance

ghana_bbox = (-3.5, 1.5, 4.5, 11.5)  # (min_lon, max_lon, min_lat, max_lat)

#%% Various functions

def _mad(series):
    s = np.asarray(series, dtype=float)
    med = np.nanmedian(s)
    return 1.4826 * np.nanmedian(np.abs(s - med))
#--------------------------------------------------------------------------

def _robust_clip(s: pd.Series, z=8.0):
    med = np.nanmedian(s)
    mad = _mad(s)
    if not np.isfinite(mad) or mad == 0:
        lo, hi = med - 40.0, med + 40.0
    else:
        lo, hi = med - z * mad, med + z * mad
    mask = (s < lo) | (s > hi)
    return mask, dict(median=float(med), mad=float(mad), lo=float(lo), hi=float(hi))
#--------------------------------------------------------------------------

def _gap_scale(dt_min):
    """Scale factor for thresholds vs actual delta time."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (dt_min / CADENCE_MIN).clip(lower=0)
#--------------------------------------------------------------------------

def clean_pmin_pmax(df_link: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a single link's precomputed Pmin/Pmax series.
    Returns a copy with:
      - Pmin_clean, Pmax_clean, P_used
      - flags: hard_min/max, val_min/max, diff_min/max, invert, diverge, flat
      - dt_min for debugging
    Expected columns: timestamp, link_id, Pmin, Pmax
    """
    g = df_link.sort_values("timestamp").copy()
    g["timestamp"] = pd.to_datetime(g["timestamp"], utc=True, errors="coerce")
    g = g.dropna(subset=["timestamp"])

    # ---------- 0) gross rails ----------
    g["hard_min"] = (g["Pmin"].abs() > HARD_ABS_P)
    g["hard_max"] = (g["Pmax"].abs() > HARD_ABS_P)
    g.loc[g["hard_min"], "Pmin"] = np.nan
    g.loc[g["hard_max"], "Pmax"] = np.nan

    # ---------- 1) robust value rails (per link, separately for Pmin/Pmax) ----------
    val_mask_min, stats_min = _robust_clip(g["Pmin"], z=Z_MAD_VAL)
    val_mask_max, stats_max = _robust_clip(g["Pmax"], z=Z_MAD_VAL)
    g["val_min"] = val_mask_min
    g["val_max"] = val_mask_max
    g.loc[g["val_min"], "Pmin"] = np.nan
    g.loc[g["val_max"], "Pmax"] = np.nan

    # ---------- 2) enforce Pmin <= Pmax (inversion check) ----------
    g["invert"] = (g["Pmin"] > g["Pmax"])
    # If inverted, Pmin is the likely culprit → drop Pmin
    g.loc[g["invert"], "Pmin"] = np.nan

    # ---------- 3) divergence constraint (Pmax - Pmin too big/small) ----------
    diff_pm = g["Pmax"] - g["Pmin"]
    g["diverge"] = (diff_pm > DIVERGE_MAX_DB) | (diff_pm < DIVERGE_MIN_DB)
    # When diverging, prefer to keep Pmax (stable) and drop Pmin
    g.loc[g["diverge"], "Pmin"] = np.nan

    # ---------- 4) gap-aware spike filter using rolling MAD of diffs ----------
    ts = g["timestamp"]
    dt_min = ts.diff().dt.total_seconds().div(60.0)
    scale = _gap_scale(dt_min)             # 0 for first row/NaN
    for col in ["Pmin", "Pmax"]:
        dcol = g[col].diff()
        # rolling MAD of diffs
        mad_d = dcol.rolling(DIFF_WIN_STEPS, min_periods=max(5, DIFF_WIN_STEPS//3)) \
                   .apply(lambda x: _mad(x), raw=False)
        tau = np.maximum(TAU_FLOOR_DB, Z_MAD_DIFF * mad_d) * scale
        spike = (dcol.abs() > tau) & (scale > 0)
        g[f"diff_flag_{col}"] = spike.fillna(False)
        g.loc[spike, col] = np.nan

    # ---------- 5) optional flatline (both Pmin & Pmax flat) ----------
    def _nochange(a, b):  # both within tolerance
        return (np.isfinite(a) and np.isfinite(b)
                and abs(a - b) <= FLAT_TOL)
    same_min = g["Pmin"].diff().abs() <= FLAT_TOL
    same_max = g["Pmax"].diff().abs() <= FLAT_TOL
    flat = (same_min & same_max).rolling(FLATLINE_RUN, min_periods=FLATLINE_RUN) \
                                .apply(lambda x: 1.0 if np.all(x) else 0.0, raw=False) \
                                .astype(bool)
    g["flat"] = flat.fillna(False)
    # If you want to remove long “frozen” patches entirely, uncomment:
    # g.loc[g["flat"], ["Pmin","Pmax"]] = np.nan

    # ---------- 6) finalize clean series and choose P_used ----------
    g["Pmin_clean"] = g["Pmin"]
    g["Pmax_clean"] = g["Pmax"]

    # Choose P_used: prefer Pmax_clean; fallback to Pmin_clean
    g["P_used"] = g["Pmax_clean"].where(g["Pmax_clean"].notna(), g["Pmin_clean"])
    g["P_source"] = np.where(g["Pmax_clean"].notna(), "Pmax", "Pmin")

    # Housekeeping
    g["dt_min"] = dt_min

    # Optional: small forward/back-fill to bridge isolated NaNs (comment out if you prefer raw)
    # g["P_used"] = g["P_used"].interpolate(limit=1).ffill().bfill()

    # Debug print for one link
    # print("Stats Pmin:", stats_min)
    # print("Stats Pmax:", stats_max)

    return g

#--------------------------------------------------------------------------
def to_datetime_utc(s):
    # Assuming the DateTime column is in the format 'YYYYMMDDHHMM' and the timezone is Ghana (UTC+0)
    return pd.to_datetime(s, format='%Y%m%d%H%M', utc=True, errors="coerce").dt.tz_localize(None)
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def ensure_length_km(meta):
    if 'PathLength' not in meta.columns or meta['PathLength'].isna().any():
        meta = meta.copy()
        meta['PathLength'] = haversine_km(meta['tx_lat'],meta['tx_lon'],meta['rx_lat'],meta['rx_lon'])
    return meta
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

def clean_timeseries_robust(
    df_link: pd.DataFrame,
    *,
    # hard but very wide sanity rails per column (just to catch typos/engineering limits)
    hard_rails = {
        "RSL_MIN": (-200.0,  10.0),
        "RSL_MAX": (-200.0,  10.0),
        "TSL_AVG": (-200.0,  20.0),
    },
    # robust per-link rails = median ± z*MAD
    z_mad = {"RSL_MIN": 8.0, "RSL_MAX": 8.0, "TSL_AVG": 10.0},
    # flatline detection
    flatline_run = 4,
    equal_tol = 1e-6,
    # gap-aware jump detection (per 15 min)
    max_jump_db_per_15 = 10.0,
    atpc_enabled = False,             # your data: keep False
    atpc_step_db_per_15 = 1.2,
    atpc_guard_steps = 2,
):
    """
    Returns (df_clean, stats) where:
      df_clean: cleaned dataframe with NaNs where invalid
      stats:    dict of per-link med/MAD and rail values actually used
    Expects columns: timestamp, RSL_MIN, RSL_MAX, TSL_AVG
    """
    df = df_link.sort_values("timestamp").copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # 0) wide hard rails (gross outliers only)
    for col in ["RSL_MIN","RSL_MAX","TSL_AVG"]:
        lo, hi = hard_rails[col]
        bad = (df[col] < lo) | (df[col] > hi)
        df.loc[bad, col] = np.nan

    # 1) robust per-link rails from distribution (median ± z*MAD)
    stats = {}
    for col in ["RSL_MIN","RSL_MAX","TSL_AVG"]:
        s = df[col].astype(float)
        med = np.nanmedian(s)
        mad = 1.4826 * np.nanmedian(np.abs(s - med))
        z = z_mad[col]
        lo = med - z * mad
        hi = med + z * mad
        # If MAD collapses (all equal), keep very wide rails around median
        if not np.isfinite(mad) or mad == 0:
            lo = med - 40.0
            hi = med + 40.0
        clip_mask = (s < lo) | (s > hi)
        df.loc[clip_mask, col] = np.nan
        stats[col] = {"median": float(med), "mad": float(mad), "lo": float(lo), "hi": float(hi)}

    # 2) physical consistency: RSL_MAX should not be lower than RSL_MIN
    #    If inverted, mask that row (safer than swapping)
    inv = (df["RSL_MAX"] < df["RSL_MIN"])
    df.loc[inv, ["RSL_MIN","RSL_MAX","TSL_AVG"]] = np.nan

    # 3) flatlines: all three nearly equal for >= flatline_run consecutive steps
    def _all_equal_row(row):
        a,b,c = row
        return (np.isfinite(a) and np.isfinite(b) and np.isfinite(c)
                and abs(a-b) <= equal_tol and abs(a-c) <= equal_tol)
    same = df[["RSL_MIN","RSL_MAX","TSL_AVG"]].apply(_all_equal_row, axis=1)
    flat = same.rolling(flatline_run, min_periods=flatline_run).apply(
        lambda x: 1.0 if np.all(x) else 0.0, raw=False
    ).astype(bool)
    df.loc[flat, ["RSL_MIN","RSL_MAX","TSL_AVG"]] = np.nan

    # 4) gap-aware jump detection on TSL_AVG
    ts = df["timestamp"]
    dt_min = ts.diff().dt.total_seconds().div(60.0)
    d_db = df["TSL_AVG"].diff().abs()
    scale = (dt_min / 15.0).clip(lower=0)    # scale threshold by elapsed time
    thresh = max_jump_db_per_15 * scale
    big_jump = (d_db > thresh) & (scale > 0)
    df.loc[big_jump, ["RSL_MIN","RSL_MAX","TSL_AVG"]] = np.nan
    stats["jump"] = {"max_per_15": max_jump_db_per_15, "flagged": int(big_jump.sum())}

    # 5) ATPC masking (disabled unless you flip the flag)
    if atpc_enabled:
        atpc_thr = atpc_step_db_per_15 * scale
        step = (d_db >= atpc_thr) & (scale > 0)
        mask = step.copy()
        for k in range(1, int(atpc_guard_steps)+1):
            mask |= step.shift(k, fill_value=False) | step.shift(-k, fill_value=False)
        df.loc[mask, ["RSL_MIN","RSL_MAX","TSL_AVG"]] = np.nan
        stats["atpc"] = {"enabled": True, "flagged": int(mask.sum())}
    else:
        stats["atpc"] = {"enabled": False, "flagged": 0}

    # 6) compute Pmin/Pmax for downstream (RAINLINK convention)
    df["Pmin"] = df["RSL_MIN"] - df["TSL_AVG"]
    df["Pmax"] = df["RSL_MAX"] - df["TSL_AVG"]

    return df, stats


# --- helpers --------------------------------------------------------------

def _ensure_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure columns 'lon_c' and 'lat_c' (link midpoints) exist.
    If missing, compute them from XStart/YStart and XEnd/YEnd (in lon/lat).
    """
    out = df.copy()
    if "lon_c" not in out.columns or "lat_c" not in out.columns:
        if not {"XStart","YStart","XEnd","YEnd"}.issubset(out.columns):
            raise ValueError("Centroids missing and cannot be computed: need XStart,YStart,XEnd,YEnd.")
        out["lon_c"] = 0.5 * (out["XStart"].astype(float) + out["XEnd"].astype(float))
        out["lat_c"] = 0.5 * (out["YStart"].astype(float) + out["YEnd"].astype(float))
    return out

def _make_grid(bbox, res_deg):
    """
    bbox = (min_lon, max_lon, min_lat, max_lat), res_deg is grid spacing in degrees.
    Returns: LON(2D), LAT(2D), lons(1D), lats(1D)
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    lons = np.arange(lon_min, lon_max + 1e-9, res_deg)
    lats = np.arange(lat_min, lat_max + 1e-9, res_deg)
    LON, LAT = np.meshgrid(lons, lats)
    return LON, LAT, lons, lats

# --- generic single-time gridding ----------------------------------------

def grid_one_timeslice_generic(
    df_ts: pd.DataFrame,
    grid_func,                 # callable: (lon_pts, lat_pts, values, LON_grid, LAT_grid, **kwargs) -> 2D array
    grid_kwargs=None,          # dict of kwargs forwarded to grid_func
    value_col="R_mm_h",
    bbox=None,
    res_deg=0.05,
    min_pts_required=3,        # skip gridding if fewer valid points than this
):
    """
    Interpolate a *single* timestamp using a pluggable gridding function.

    Returns dict:
      {
        'timestamp': pandas.Timestamp,
        'grid'     : 2D np.array (rain rate on grid),
        'lon'      : 1D lons,
        'lat'      : 1D lats,
        'bbox'     : (min_lon, max_lon, min_lat, max_lat),
        'n_points' : number of valid input points used
      }
    """
    grid_kwargs = {} if grid_kwargs is None else dict(grid_kwargs)
    df_ts = _ensure_centroids(df_ts)

    # Choose bounds automatically if not supplied (pad a touch for neat plotting)
    if bbox is None:
        lon_min = float(np.nanmin(df_ts["lon_c"])) - 0.05
        lon_max = float(np.nanmax(df_ts["lon_c"])) + 0.05
        lat_min = float(np.nanmin(df_ts["lat_c"])) - 0.05
        lat_max = float(np.nanmax(df_ts["lat_c"])) + 0.05
        bbox = (lon_min, lon_max, lat_min, lat_max)

    LON, LAT, lons, lats = _make_grid(bbox, res_deg)

    # Pull valid points
    pts = df_ts[["lon_c", "lat_c", value_col]].dropna()
    n_points = len(pts)
    ts = pd.to_datetime(df_ts["timestamp"].iloc[0])

    # if n_points < min_pts_required:
    #     # Not enough data — return an all-NaN grid for this timestep
    #     return {"timestamp": ts, "grid": np.full_like(LON, np.nan, dtype=float),
    #             "lon": lons, "lat": lats, "bbox": bbox, "n_points": n_points}

    # Call the chosen gridding routine
    Z = grid_func(
        lon_pts=pts["lon_c"].to_numpy(),
        lat_pts=pts["lat_c"].to_numpy(),
        values =pts[value_col].to_numpy(),
        LON_grid=LON, LAT_grid=LAT,
        **grid_kwargs
    )

    return {"timestamp": ts, "grid": Z, "lon": lons, "lat": lats, "bbox": bbox, "n_points": n_points}

# --- generic batch gridding over all timestamps --------------------------

def grid_all_times_generic(
    df: pd.DataFrame,
    grid_func,
    grid_kwargs=None,
    value_col="R_mm_h",
    bbox=None,
    res_deg=0.05,
    min_pts_required=3,
):
    """
    Group by 'timestamp' and grid each slice using a pluggable gridding function.

    Returns: list of dicts (same format as grid_one_timeslice_generic output).
    """
    out = []
    dff = df.copy()
    dff["timestamp"] = pd.to_datetime(dff["timestamp"])
    for ts, g in dff.sort_values("timestamp").groupby("timestamp"):
        res = grid_one_timeslice_generic(
            g, grid_func=grid_func, grid_kwargs=grid_kwargs,
            value_col=value_col, bbox=bbox, res_deg=res_deg,
            min_pts_required=min_pts_required,
        )
        out.append(res)
    return out

#- - - - - - - - -------------------------------------------------------------
# -----------------------------
# Ordinary Kriging (robust, no SciPy)
# -----------------------------
def kriging_grid_ok(lon_pts, lat_pts, values,
                    LON_grid, LAT_grid,
                    variogram_model="spherical",
                    variogram_parameters=None,
                    nlags=6,
                    enable_plotting=False,
                    exact_values=False,
                    fill_value=np.nan):
    

    """
    Ordinary Kriging on a lon/lat grid.

    Parameters
    ----------
    lon_pts, lat_pts, values : 1D arrays
        Station longitudes, latitudes, and rain rate values (mm/h).
    LON_grid, LAT_grid : 2D arrays
        Target grid.
    variogram_model : str
        'linear', 'power', 'gaussian', 'spherical', 'exponential', 'hole-effect'
    variogram_parameters : dict or tuple or None
        If None, PyKrige will estimate from data. Example dict:
        {'sill': 20.0, 'range': 1.0, 'nugget': 0.5}
        (range in degrees; start with ~0.3–0.8 for Ghana-scale, tune later)
    nlags : int
        Number of lags to build the variogram (used if params are None).
    exact_values : bool
        If True, kriging honors points exactly (like nugget≈0).

    Returns
    -------
    grid : 2D array of interpolated values (same shape as LON_grid)
    """
    lon_pts = np.asarray(lon_pts, float)
    lat_pts = np.asarray(lat_pts, float)
    values  = np.asarray(values,  float)

    mask = np.isfinite(lon_pts) & np.isfinite(lat_pts) & np.isfinite(values)
    if mask.sum() < 3:
        return np.full_like(LON_grid, fill_value, dtype=float)

    OK = OrdinaryKriging(
        lon_pts[mask], lat_pts[mask], values[mask],
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nlags=nlags,
        enable_plotting=enable_plotting,
        exact_values=exact_values,
        coordinates_type="geographic"  # treat inputs as lon/lat
    )

    z, ss = OK.execute("grid", LON_grid[0, :], LAT_grid[:, 0])  # (ny, nx)
    z = np.asarray(z, dtype=float)

    # Optional: mask negative tiny values back to 0
    z[z < 0] = 0.0
    return z

# -----------------------------
# IDW core (robust, no SciPy)
# -----------------------------
def idw_grid(lon_pts, lat_pts, values,
             LON_grid, LAT_grid,
             power=2.0, radius_km=30.0, min_pts=3, fill_value=np.nan):
    """
    IDW interpolation on a lon/lat grid (km-based search radius).
    - lon_pts, lat_pts, values: 1D arrays (same length)
    - LON_grid, LAT_grid: 2D arrays defining grid nodes
    Returns 2D array (same shape as LON_grid)
    """
    ny, nx = LON_grid.shape
    out = np.full((ny, nx), fill_value, dtype=float)

    # Pre-stack station arrays for vectorized distance calc per row (lat-major loops for cache locality)
    lon_pts = np.asarray(lon_pts, dtype=float)
    lat_pts = np.asarray(lat_pts, dtype=float)
    values = np.asarray(values, dtype=float)

    # Mask invalid stations
    mask = np.isfinite(lon_pts) & np.isfinite(lat_pts) & np.isfinite(values)
    lon_pts = lon_pts[mask]; lat_pts = lat_pts[mask]; values = values[mask]

    if len(values) == 0:
        return out  # nothing to interpolate

    # Loop over rows (lat lines) for memory balance; vectorize across x
    for j in range(ny):
        lon_row = LON_grid[j, :]
        lat_row = LAT_grid[j, :]
        # distances to all points for every x in this row (broadcast)
        # We'll compute per x since N is moderate and this stays readable/robust.
        row_vals = np.full(nx, fill_value, dtype=float)
        for i in range(nx):
            d = _haversine_km(lon_pts, lat_pts, lon_row[i], lat_row[i])  # km
            # neighbors within radius
            neigh = d <= radius_km
            if np.count_nonzero(neigh) >= max(1, min_pts):
                dd = d[neigh]
                vv = values[neigh]
                # exact match or extremely close → take the point value
                if dd.min() < 1e-6:
                    row_vals[i] = vv[dd.argmin()]
                else:
                    w = 1.0 / np.power(dd, power)
                    row_vals[i] = np.sum(w * vv) / np.sum(w)
            else:
                # Fallback to nearest neighbor (optional). Comment this block out if you prefer NaN.
                if len(d) > 0:
                    k = np.argmin(d)
                    if d[k] <= radius_km * 1.5:  # soft fallback radius
                        row_vals[i] = values[k]
                    else:
                        row_vals[i] = fill_value
                else:
                    row_vals[i] = fill_value
        out[j, :] = row_vals
    return out

# -----------------------------
# Haversine distance (km)
def _haversine_km(lon1, lat1, lon2, lat2):
    """Great-circle distance in km."""
    R = 6371.0
    lon1r, lat1r, lon2r, lat2r = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2r - lon1r
    dlat = lat2r - lat1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    return 2*R*np.arcsin(np.sqrt(a))

# -----------------------------
# Gaussian Process regression with Matérn kernel 
# (SciPy-based)
# -----------------------------
def gp_grid_matern(lon_pts, lat_pts, values,
                   LON_grid, LAT_grid,
                   length_scale_km=50.0, nu=1.5,
                   noise_level=0.05,  # small white noise (mm/h^2)
                   normalize_y=True,
                   fill_value=np.nan):
    """
    Gaussian Process regression with Matérn kernel on a lon/lat grid.
    lon/lat are internally converted to km using a crude scaling at Ghana’s latitude.

    Returns 2D array of interpolated values.
    """
    lon_pts = np.asarray(lon_pts, float)
    lat_pts = np.asarray(lat_pts, float)
    values  = np.asarray(values,  float)

    mask = np.isfinite(lon_pts) & np.isfinite(lat_pts) & np.isfinite(values)
    if mask.sum() < 3:
        return np.full_like(LON_grid, fill_value, dtype=float)

    # Rough lon/lat → km scaling around ~7–8°N (OK for Ghana-scale interpolation)
    lat0 = 7.5 * np.pi/180.0
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(lat0)

    X = np.column_stack([(lon_pts[mask]) * km_per_deg_lon,
                         (lat_pts[mask]) * km_per_deg_lat])
    y = values[mask].astype(float)

    # Build target grid coords in km
    Xg = np.column_stack([LON_grid.ravel() * km_per_deg_lon,
                          LAT_grid.ravel() * km_per_deg_lat])

    # Kernel: amplitude * Matern + white noise
    k = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(length_scale=length_scale_km,
                                                  length_scale_bounds=(5.0, 300.0),
                                                  nu=nu) \
        + WhiteKernel(noise_level=noise_level, noise_level_bounds=(1e-6, 1.0))

    gp = GaussianProcessRegressor(kernel=k, alpha=0.0, normalize_y=normalize_y, n_restarts_optimizer=2)
    gp.fit(X, y)
    z_pred, z_std = gp.predict(Xg, return_std=True)
    Z = z_pred.reshape(LON_grid.shape)

    Z[Z < 0] = 0.0
    return Z  # (optionally also return z_std.reshape(...) for uncertainty)

# -----------------------------
from pykrige.ok import OrdinaryKriging

from pykrige.ok import OrdinaryKriging

def krige_exponential(inst, xx, yy, range_km=60.0, sill=None, nugget=0.3,
                      max_neighbors=16, enable_mask=True, max_dist_km=90.0,
                      backend_when_mw="loop"):
    x = inst["lon_mid"].values.astype(float)
    y = inst["lat_mid"].values.astype(float)
    z = inst["R_mm_h"].values.astype(float)

    if sill is None:
        v = np.nanvar(z)
        sill = float(v if np.isfinite(v) and v > 0 else 1.0)

    # km -> deg (isotropic average at mean lat)
    km_lon, km_lat = km_per_deg(np.nanmean(y))
    deg_per_km = 0.5*(1.0/km_lon + 1.0/km_lat)
    range_deg  = range_km * deg_per_km

    OK = OrdinaryKriging(
        x, y, z,
        variogram_model="exponential",
        variogram_parameters=[sill, range_deg, nugget],
        enable_plotting=False, nlags=12, verbose=False
    )

    # Choose backend: moving-window needs 'loop' (or 'c' if your install supports it)
    if max_neighbors is None:
        Z, ss = OK.execute("grid", xx[0, :], yy[:, 0], backend="vectorized")
    else:
        Z, ss = OK.execute("grid", xx[0, :], yy[:, 0],
                           backend=backend_when_mw,           # "loop" or "c"
                           n_closest_points=max_neighbors)

    Z = np.asarray(Z)
    ss = np.asarray(ss)

    if enable_mask and (max_dist_km is not None):
        far_mask = distance_mask(xx, yy, x, y, max_dist_km)
        Z = Z.copy();  Z[far_mask]  = np.nan
        ss = ss.copy(); ss[far_mask] = np.nan

    return Z, ss, dict(sill=sill, range_deg=range_deg, nugget=nugget)

# -----------------------------
def suggest_range_km(inst, test_ranges_km=(40, 60, 80, 100, 140)):
    """Very light LOO CV on a random subset (speed). Returns (best_range, table)"""
    rngs = []
    rmses = []
    x = inst["lon_mid"].values.astype(float)
    y = inst["lat_mid"].values.astype(float)
    z = inst["R_mm_h"].values.astype(float)
    n = len(z)
    # sample up to 150 for speed
    idx = np.random.default_rng(0).choice(n, size=min(n,150), replace=False)
    km_lon, km_lat = km_per_deg(np.nanmean(y))
    for rkm in test_ranges_km:
        deg_per_km = 0.5*(1/km_lon + 1/km_lat)
        rdeg = rkm*deg_per_km
        errs = []
        for i in idx:
            xi = np.delete(x, i); yi = np.delete(y, i); zi = np.delete(z, i)
            OK = OrdinaryKriging(xi, yi, zi, variogram_model="exponential",
                                 variogram_parameters=[np.var(zi), rdeg, 0.2],
                                 enable_plotting=False, nlags=12, verbose=False)
            zhat, _ = OK.execute("points", np.array([x[i]]), np.array([y[i]]))
            errs.append(float(zhat) - z[i])
        rmse = float(np.sqrt(np.mean(np.square(errs))))
        rngs.append(rkm); rmses.append(rmse)
    best = float(rngs[int(np.nanargmin(rmses))])
    return best, list(zip(rngs, rmses))

# -----------------------------
import numpy as np
from sklearn.neighbors import KDTree

def km_per_deg(lat_deg: float) -> tuple[float, float]:
    """Return (km_per_deg_lon, km_per_deg_lat) at a representative latitude."""
    lat_rad = np.deg2rad(lat_deg)
    km_lat = 111.32
    km_lon = 111.32 * np.cos(lat_rad)
    return km_lon, km_lat

def distance_mask(xx, yy, pts_lon, pts_lat, max_dist_km):
    """Mask grid cells farther than max_dist_km from nearest obs point."""
    km_lon, km_lat = km_per_deg(np.nanmean(pts_lat))
    # scale lon/lat to km so KDTree uses Euclidean in ~km
    obs_xy = np.c_[ (pts_lon * km_lon), (pts_lat * km_lat) ]
    grid_xy = np.c_[ (xx.ravel() * km_lon), (yy.ravel() * km_lat) ]
    tree = KDTree(obs_xy)
    d_km, _ = tree.query(grid_xy, k=1)
    far = (d_km.ravel() > max_dist_km)
    return far.reshape(xx.shape)