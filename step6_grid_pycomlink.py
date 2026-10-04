# step6_grid_pycomlink.py
# 15-min gridding for CML rain rates (R_mm_per_h) using:
#   1) pycomlink Ordinary Kriging (preferred)
#   2) pycomlink IDW
#   3) robust local IDW fallback
#
# Input:
#   df_s5: DataFrame indexed by time, with columns ['ID','R_mm_per_h']
#   df_meta_xy: per-link geometry (ID, XStart, YStart, XEnd, YEnd)
#
# Output:
#   xarray.DataArray with dims ("time","lat","lon") named "R_mm_per_h"
#   and an optional diagnostics dict.

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

# optional dependencies (graceful if missing)
try:
    import pycomlink as pcm
    _HAVE_PYCOMLINK = True
except Exception:
    pcm = None
    _HAVE_PYCOMLINK = False

# If pycomlink OK is missing or fails, we use PyKrige directly
try:
    from pykrige.ok import OrdinaryKriging as PKOrdinaryKriging
    _HAVE_PYKRIGE = True
except Exception:
    PKOrdinaryKriging = None
    _HAVE_PYKRIGE = False


# ----------------------- helpers -----------------------

def _midpoints(meta):
    m = meta.copy()
    m["lon"] = 0.5 * (pd.to_numeric(m["XStart"], errors="coerce") +
                      pd.to_numeric(m["XEnd"],   errors="coerce"))
    m["lat"] = 0.5 * (pd.to_numeric(m["YStart"], errors="coerce") +
                      pd.to_numeric(m["YEnd"],   errors="coerce"))
    return m[["ID","lon","lat"]]

def _make_grid(meta_xy, res_deg=0.05, pad_deg=0.2):
    lon0, lon1 = meta_xy["lon"].min(), meta_xy["lon"].max()
    lat0, lat1 = meta_xy["lat"].min(), meta_xy["lat"].max()
    lon = np.arange(lon0 - pad_deg, lon1 + pad_deg + 1e-9, res_deg)
    lat = np.arange(lat0 - pad_deg, lat1 + pad_deg + 1e-9, res_deg)
    return lon, lat

def _safe_time_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx

def _local_idw_on_grid(x, y, z, grid_lon, grid_lat, power=2):
    # simple, robust IDW fallback (works with any number of points)
    Xg, Yg = np.meshgrid(grid_lon, grid_lat)
    Zi = np.full_like(Xg, np.nan, float)
    # vectorized along grid rows
    for i in range(Xg.shape[0]):
        dx = Xg[i, :][None, :] - x[:, None]
        dy = Yg[i, :][None, :] - y[:, None]
        d2 = dx*dx + dy*dy
        w = 1.0 / np.maximum(d2, 1e-12)**(power/2)
        Zi[i, :] = np.sum(w * z[:, None], axis=0) / np.sum(w, axis=0)
    return np.maximum(Zi, 0.0)

def _pycomlink_handles():
    """Try to find pycomlink's IDW and OK helpers under different versions."""
    if not _HAVE_PYCOMLINK:
        return None, None
    idw = ok = None
    # IDW possibilities
    for dotted in (
        "spatial.interpolation.inverse_distance",
        "spatial.interp.inverse_distance",
        "spatial.interpolators.inverse_distance",
    ):
        try:
            obj = pcm
            for name in dotted.split("."):
                obj = getattr(obj, name)
            idw = obj; break
        except Exception:
            pass
    # OK possibilities (pycomlink may wrap pykrige)
    for dotted in (
        "spatial.interpolation.ordinary_kriging",
        "spatial.interp.ordinary_kriging",
        "spatial.interpolators.ordinary_kriging",
    ):
        try:
            obj = pcm
            for name in dotted.split("."):
                obj = getattr(obj, name)
            ok = obj; break
        except Exception:
            pass
    return idw, ok


# --------------------- main API ------------------------

def grid_rain_15min(
    df_s5: pd.DataFrame,
    df_meta_for_xy: pd.DataFrame,
    *,
    # domain & grid
    grid_res_deg: float = 0.05,
    domain_pad_deg: float = 0.2,
    # OK controls
    min_pts_ok: int = 3,                 # minimum points to attempt OK
    variogram_model: str = "exponential",
    variogram_params: tuple | None = None,
    nlags: int = 12,
    # IDW controls
    idw_power: int = 2,
    # parallel
    n_jobs: int = 1
) -> tuple[xr.DataArray, dict]:
    """
    Produce a 15-min grid of rain rate (mm h-1) by preferring pycomlink OK,
    falling back to pycomlink IDW, then to a robust local IDW.

    Returns
    -------
    da : xr.DataArray
        dims: (time, lat, lon), name='R_mm_per_h'
    diag : dict
        simple diagnostics (counts of OK vs IDW uses, failures, etc.)
    """
    # --- inputs & meta ---
    need_cols = {"ID","R_mm_per_h"}
    if not need_cols.issubset(df_s5.columns):
        raise ValueError(f"df_s5 must have columns {need_cols}")

    meta_xy = _midpoints(df_meta_for_xy)
    lon, lat = _make_grid(meta_xy, res_deg=grid_res_deg, pad_deg=domain_pad_deg)

    df = df_s5.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df_s5 index must be a DatetimeIndex (UTC preferred).")
    df.index = _safe_time_index(df.index)
    # enforce numeric & nonnegative
    df["R_mm_per_h"] = pd.to_numeric(df["R_mm_per_h"], errors="coerce").clip(lower=0.0)

    times = pd.Index(df.index).unique().sort_values()

    # pycomlink handles (may be None)
    pcm_idw, pcm_ok = _pycomlink_handles()

    # per-time function
    def _one_time(t):
        g = df.loc[df.index == t, ["ID","R_mm_per_h"]].merge(meta_xy, on="ID", how="left")
        x = pd.to_numeric(g["lon"], errors="coerce").to_numpy()
        y = pd.to_numeric(g["lat"], errors="coerce").to_numpy()
        z = pd.to_numeric(g["R_mm_per_h"], errors="coerce").to_numpy()
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        x, y, z = x[m], y[m], z[m]

        method_used = "none"
        Zi = np.full((len(lat), len(lon)), np.nan, float)

        if x.size == 0:
            return Zi, method_used

        # try OK (pycomlink first, else PyKrige), only if enough points & some variance
        if (x.size >= min_pts_ok) and (np.nanvar(z) > 1e-12):
            # pycomlink OK wrapper
            if pcm_ok is not None:
                try:
                    Zi = pcm_ok(x, y, z, lon, lat,
                                variogram_model=variogram_model,
                                variogram_parameters=variogram_params,
                                nlags=nlags)
                    Zi = np.asarray(Zi, float)
                    Zi = np.maximum(Zi, 0.0)
                    return Zi, "ok_pcm"
                except Exception:
                    pass
            # direct PyKrige OK
            if _HAVE_PYKRIGE:
                try:
                    OK = PKOrdinaryKriging(
                        x, y, z,
                        variogram_model=variogram_model,
                        variogram_parameters=variogram_params,
                        nlags=nlags,
                        enable_statistics=False,
                    )
                    Zi, _ = OK.execute("grid", lon, lat)
                    Zi = np.asarray(Zi, float)
                    Zi = np.maximum(Zi, 0.0)
                    return Zi, "ok_pykrige"
                except Exception:
                    pass

        # pycomlink IDW
        if pcm_idw is not None:
            try:
                # versions differ in signature; try with power, else without
                try:
                    Zi = pcm_idw(x, y, z, lon, lat, power=idw_power)
                except TypeError:
                    Zi = pcm_idw(x, y, z, lon, lat)
                Zi = np.asarray(Zi, float)
                Zi = np.maximum(Zi, 0.0)
                return Zi, "idw_pcm"
            except Exception:
                pass

        # robust local IDW fallback
        Zi = _local_idw_on_grid(x, y, z, lon, lat, power=idw_power)
        return Zi, "idw_fallback"

    # run (parallel or serial)
    if (n_jobs != 1) and _HAVE_JOBLIB:
        results = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(_one_time)(t) for t in times)
    else:
        results = [_one_time(t) for t in times]

    grids = [r[0] for r in results]
    methods = [r[1] for r in results]

    times = pd.DatetimeIndex(pd.Index(times)).tz_localize(None)
    times = pd.DatetimeIndex(times.values, name="time")  # force name 'time'

    da = xr.DataArray(
        np.stack(grids),
        coords=[("time", times.values.astype("datetime64[ns]")),
                ("lat",  np.asarray(lat, float)),
                ("lon",  np.asarray(lon, float))],
        dims=("time", "lat", "lon"),
        name="R_mm_per_h",
    )
    da.attrs.update(dict(units="mm h-1",
                         method="pycomlink-OK → pycomlink-IDW → local-IDW",
                         grid_res_deg=float(grid_res_deg)))

    diag = {
        "n_times": int(len(times)),
        "counts": {m: methods.count(m) for m in set(methods)},
        "pycomlink_available": _HAVE_PYCOMLINK,
        "pykrige_available": _HAVE_PYKRIGE,
    }
    return da, diag