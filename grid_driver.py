# grid_driver.py
# -------------------------------------------------------------
# Orchestrates per-timestamp gridding (OK or KED) into an
# xarray DataArray (time, lat, lon), keeping your 15-min cadence.
# -------------------------------------------------------------

from __future__ import annotations
from typing import Callable, Literal, Optional, Tuple
import numpy as np
import pandas as pd
import xarray as xr

from joblib import Parallel, delayed

from step6_grid_ok import link_midpoints, ok_grid_one_time
from step6_grid_ked import ked_grid_one_time

# ---- optional drift provider signature --------------------------------------
#   def drift_provider(t, lon_obs, lat_obs, lon_grid, lat_grid) -> (drift_obs, drift_grid)
# Default below returns zeros (i.e., falls back to OK-like behavior when KED chosen).

def _zeros_drift(t, lon_obs, lat_obs, lon_grid, lat_grid):
    return (np.zeros_like(lon_obs, dtype="float32"),
            np.zeros((len(lat_grid), len(lon_grid)), dtype="float32"))

# ---- main driver -------------------------------------------------------------

def grid_rain_15min(
    df_s5: pd.DataFrame,
    df_meta_for_xy: pd.DataFrame,
    *,
    method: Literal["ok","ked"] = "ok",
    domain_pad_deg: float = 0.2,
    grid_res_deg: float = 0.05,
    times: Optional[pd.DatetimeIndex] = None,
    drift_provider: Optional[Callable] = None,
    n_jobs: int = 8,
    variogram_model: str = "exponential",
    variogram_model_parameters: Optional[Tuple[float, float, float]] = None,
    nlags: int = 12,
    eps: float = 0.1,
    min_pts: int = 5,
) -> xr.DataArray:
    """
    Parameters
    ----------
    df_s5 : DataFrame with index=time, columns ['ID','R_mm_per_h'].
    df_meta_for_xy : DataFrame with link geometry (ID,XStart,YStart,XEnd,YEnd) or ['ID','lon','lat'].
    method : 'ok' or 'ked'
    drift_provider : function returning (drift_obs, drift_grid) for time t (required for 'ked')
    """

    # --- midpoints
    if {'lon','lat','ID'}.issubset(df_meta_for_xy.columns):
        meta = df_meta_for_xy[['ID','lon','lat']].copy()
    else:
        meta = link_midpoints(df_meta_for_xy)

    id_to_xy = dict(zip(meta['ID'], zip(meta['lon'].to_numpy(), meta['lat'].to_numpy())))

    # --- domain/grid
    lon_all, lat_all = meta['lon'].to_numpy(), meta['lat'].to_numpy()
    lon0, lon1 = lon_all.min()-domain_pad_deg, lon_all.max()+domain_pad_deg
    lat0, lat1 = lat_all.min()-domain_pad_deg, lat_all.max()+domain_pad_deg
    grid_lon = np.arange(lon0, lon1 + grid_res_deg/2, grid_res_deg, dtype="float32")
    grid_lat = np.arange(lat0, lat1 + grid_res_deg/2, grid_res_deg, dtype="float32")

    # --- times
    if times is None:
        times = pd.DatetimeIndex(df_s5.index.unique()).sort_values()
    times = times.tz_convert("UTC") if times.tz is not None else times.tz_localize("UTC")

    if method == "ked" and drift_provider is None:
        drift_provider = _zeros_drift  # safe default; replace when MSG is ready

    # --- helper to extract obs per time
    def _obs_at(t):
        d = df_s5.loc[df_s5.index == t, ["ID","R_mm_per_h"]].dropna()
        if d.empty:
            return np.array([]), np.array([]), np.array([])
        xy = np.array([id_to_xy.get(i, (np.nan, np.nan)) for i in d["ID"]], dtype="float32")
        return xy[:,0], xy[:,1], d["R_mm_per_h"].to_numpy("float32")

    # --- per-time worker
    def _one(t):
        x, y, r = _obs_at(t)
        if r.size < min_pts:
            return np.full((len(grid_lat), len(grid_lon)), np.nan, dtype="float32")
        if method == "ok":
            return ok_grid_one_time(
                x, y, r, grid_lon, grid_lat,
                variogram_model=variogram_model,
                variogram_model_parameters=variogram_model_parameters,  # <-- pass through
                nlags=nlags, eps=eps, min_pts=min_pts
            )
        drift_obs, drift_grid = drift_provider(t, x, y, grid_lon, grid_lat)
        return ked_grid_one_time(
            x, y, r, drift_obs, grid_lon, grid_lat, drift_grid,
            variogram_model=variogram_model,
            variogram_parameters=variogram_model_parameters,  # KED wrapper accepts either
            nlags=nlags, eps=eps, min_pts=min_pts
        )

    # --- collect 15-min timestamps we will grid
    if times is None:
        if isinstance(df_s5.index, pd.MultiIndex):
            times = df_s5.index.get_level_values(0).unique().sort_values()
        else:
            times = pd.DatetimeIndex(df_s5.index).unique().sort_values()
    else:
        times = pd.DatetimeIndex(times).sort_values()

    # make them naive-UTC and make sure the name is 'time'
    if times.tz is not None:
        times = times.tz_convert("UTC").tz_localize(None)
    times = times.rename("time")

    # --- run (parallel over time) -> list of 2D arrays (lat, lon)
    grids = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_one)(t) for t in times)

    # --- to xarray (explicit coordinate dims so names match)
    da = xr.DataArray(
        np.stack(grids),
        coords=[("time", times.values), ("lat", grid_lat), ("lon", grid_lon)],
        name="R_mm_per_h",
    )
    da.attrs.update(
        dict(units="mm h-1", method=method, variogram_model=variogram_model,
            grid_res_deg=float(grid_res_deg))
    )
    return da

