# step6_grid_ok.py
# -------------------------------------------------------------
# 15-min rain gridding via Ordinary Kriging (OK) on log(R+eps).
# If PyKrige is not available, falls back to LinearNDInterpolator
# (triangulation-based; much less blobby than IDW).
# -------------------------------------------------------------

from __future__ import annotations
from typing import Iterable, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from pykrige.ok import OrdinaryKriging
    _HAVE_PYKRIGE = True
except Exception:
    _HAVE_PYKRIGE = False

from scipy.interpolate import LinearNDInterpolator

# ---------- utilities ---------------------------------------------------------

def link_midpoints(df_like: pd.DataFrame) -> pd.DataFrame:
    """
    One row per link with midpoint lon/lat.
    Requires columns: ID, XStart, YStart, XEnd, YEnd
    """
    meta = (df_like.reset_index()[["ID","XStart","YStart","XEnd","YEnd"]]
                   .drop_duplicates("ID").copy())
    for c in ["XStart","YStart","XEnd","YEnd"]:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")
    meta["lon"] = 0.5 * (meta["XStart"] + meta["XEnd"])
    meta["lat"] = 0.5 * (meta["YStart"] + meta["YEnd"])
    return meta[["ID","lon","lat"]].dropna()

# ---------- core kriging ------------------------------------------------------

def ok_grid_one_time(
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    r_obs: np.ndarray,
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    *,
    variogram_model: str = "exponential",
    variogram_model_parameters: Optional[Tuple[float, float, float]] = None,  # (sill, range, nugget)
    nlags: int = 12,
    eps: float = 0.1,
    min_pts: int = 5,
) -> np.ndarray:
    """
    Ordinary kriging of log(R+eps) → grid. Returns 2D array (lat, lon).
    If PyKrige fit fails, fall back to LinearNDInterpolator.
    """
    ok = np.isfinite(lon_obs) & np.isfinite(lat_obs) & np.isfinite(r_obs) & (r_obs >= 0)
    lon_obs, lat_obs, r_obs = lon_obs[ok], lat_obs[ok], r_obs[ok]
    if lon_obs.size < min_pts:
        return np.full((len(grid_lat), len(grid_lon)), np.nan, dtype="float32")

    z = np.log(r_obs + eps)

    # if the sample is nearly flat, kriging fit is ill-posed → go linear
    if np.nanstd(z) < 1e-6 or not _HAVE_PYKRIGE:
        xi, yi = np.meshgrid(grid_lon, grid_lat)
        interp = LinearNDInterpolator(np.c_[lon_obs, lat_obs], z, fill_value=np.nan)
        zhat = interp(xi, yi)
        return (np.exp(zhat) - eps).clip(min=0).astype("float32")

    # Heuristic defaults if user didn’t pass parameters (skip the fit)
    params = variogram_model_parameters
    if params is None:
        sill = float(np.nanvar(z)) if np.nanvar(z) > 0 else 0.05
        # rough geographic range ~ half of domain diagonal
        rng = 0.5 * float(np.hypot(grid_lon.max()-grid_lon.min(),
                                   grid_lat.max()-grid_lat.min()))
        rng = max(rng, 0.05)
        nug = 0.1 * sill
        params = (sill, rng, nug)

    try:
        OK = OrdinaryKriging(
            lon_obs, lat_obs, z,
            variogram_model=variogram_model,
            variogram_model_parameters=params,       # <-- correct keyword
            enable_plotting=False,
            coordinates_type="geographic",
            nlags=nlags,
        )
        zhat, _ = OK.execute("grid", grid_lon, grid_lat)
    except Exception:
        # Safe fallback per timestamp
        xi, yi = np.meshgrid(grid_lon, grid_lat)
        interp = LinearNDInterpolator(np.c_[lon_obs, lat_obs], z, fill_value=np.nan)
        zhat = interp(xi, yi)

    return (np.exp(zhat) - eps).clip(min=0).astype("float32")
