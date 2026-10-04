# step6_grid_ked.py
# -------------------------------------------------------------
# Kriging with External Drift (KED): use a background/drift field
# (e.g., MSG proxy) to shape storms; krige residuals.
# If PyKrige's UniversalKriging is unavailable, we do a simple
# regression + OK of residuals (needs step6_grid_ok.ok_grid_one_time).
# -------------------------------------------------------------

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

try:
    from pykrige.uk import UniversalKriging
    _HAVE_PYKRIGE = True
except Exception:
    _HAVE_PYKRIGE = False

from step6_grid_ok import ok_grid_one_time

def ked_grid_one_time(
    lon_obs: np.ndarray,
    lat_obs: np.ndarray,
    r_obs: np.ndarray,
    drift_obs: np.ndarray,              # MSG at obs points (same length)
    grid_lon: np.ndarray,
    grid_lat: np.ndarray,
    drift_grid: np.ndarray,             # MSG on grid (2D: lat x lon)
    *,
    variogram_model: str = "exponential",
    variogram_parameters: Optional[Tuple[float,float,float]] = None,
    nlags: int = 12,
    eps: float = 0.1,
    min_pts: int = 5,
) -> np.ndarray:
    """KED on log(R+eps) with specified drift arrays."""
    ok = (np.isfinite(lon_obs) & np.isfinite(lat_obs) &
          np.isfinite(r_obs) & (r_obs >= 0) & np.isfinite(drift_obs))
    lon_obs, lat_obs, r_obs, drift_obs = lon_obs[ok], lat_obs[ok], r_obs[ok], drift_obs[ok]
    if lon_obs.size < min_pts:
        return np.full((len(grid_lat), len(grid_lon)), np.nan, dtype="float32")

    y = np.log(r_obs + eps)

    if _HAVE_PYKRIGE:
        UK = UniversalKriging(
            lon_obs, lat_obs, y,
            variogram_model=variogram_model,
            variogram_parameters=variogram_parameters,
            drift_terms=["specified"], specified_drift=drift_obs,
            coordinates_type="geographic", enable_plotting=False, nlags=nlags
        )
        y_grid, _ = UK.execute("grid", grid_lon, grid_lat, specified_drift=drift_grid)
        R = np.exp(y_grid) - eps
        return np.clip(R, 0, None).astype("float32")

    # ---- fallback: regress on drift, OK residuals ---------------------------
    X = np.c_[np.ones_like(drift_obs), drift_obs]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = (beta[0] + beta[1] * drift_obs)
    res = y - y_hat

    # OK residuals on grid
    res_grid = ok_grid_one_time(
        lon_obs, lat_obs, res,
        grid_lon, grid_lat,
        variogram_model=variogram_model,
        variogram_parameters=variogram_parameters,
        nlags=nlags, eps=0.0,  # residuals, no transform
        min_pts=min_pts
    )
    drift_grid_flat = drift_grid  # already 2D
    y_grid = beta[0] + beta[1] * drift_grid_flat + res_grid
    R = np.exp(y_grid) - eps
    return np.clip(R, 0, None).astype("float32")
