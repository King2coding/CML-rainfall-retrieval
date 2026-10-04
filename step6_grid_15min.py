# step6_grid_15min.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

@dataclass(frozen=True)
class Grid15Config:
    # grid
    res_deg: float = 0.05           # ~5 km near equator
    pad_deg: float = 0.10           # bbox padding around points
    # IDW
    k: int = 12                     # neighbors per grid cell
    max_radius_km: float = 50.0
    power: float = 2.0              # IDW power
    # geometry
    earth_km_per_deg: float = 111.32
    # which value to grid (we will compute P15 if missing)
    value_col: str = "P15_mm"       # "P15_mm" or "R_mm_per_h"

def _to_utc_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is None: 
        return idx.tz_localize("UTC")
    return idx.tz_convert("UTC")

def _midpoints(meta: pd.DataFrame) -> pd.DataFrame:
    # expects XStart,YStart,XEnd,YEnd
    m = meta.copy()
    for c in ["XStart","YStart","XEnd","YEnd"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["lon"] = 0.5*(m["XStart"] + m["XEnd"])
    m["lat"] = 0.5*(m["YStart"] + m["YEnd"])
    return m[["ID","lon","lat"]].drop_duplicates("ID").set_index("ID")

def _bbox_from_points(pts: pd.DataFrame, pad: float) -> Tuple[float,float,float,float]:
    lon_min, lon_max = np.nanmin(pts["lon"]), np.nanmax(pts["lon"])
    lat_min, lat_max = np.nanmin(pts["lat"]), np.nanmax(pts["lat"])
    return lon_min-pad, lon_max+pad, lat_min-pad, lat_max+pad

def _lonlat_to_xy_km(lon, lat, lon0, lat0, km_per_deg=111.32):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    x = (lon - lon0) * km_per_deg * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * km_per_deg
    return x, y

def _infer_dt_minutes(t: pd.DatetimeIndex) -> float:
    if len(t) < 2: 
        return 15.0
    # try pandas freq first
    try:
        freq = pd.infer_freq(t)
        if freq:
            return (pd.Timedelta(freq).total_seconds()/60.0)
    except Exception:
        pass
    return float((t[1] - t[0]).total_seconds()/60.0)

def _ensure_P15(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = _to_utc_index(out.index)
    if "P15_mm" in out.columns:
        return out
    if "R_mm_per_h" not in out.columns:
        raise ValueError("Need either 'P15_mm' or 'R_mm_per_h' to compute 15-min rainfall.")
    # compute per-row Δt from index spacing (assumes regular grid per link)
    # do it once globally:
    t = pd.DatetimeIndex(out.index.unique()).sort_values()
    dt_min = _infer_dt_minutes(t)
    out["P15_mm"] = pd.to_numeric(out["R_mm_per_h"], errors="coerce") * (dt_min/60.0)
    return out

def _points_from(df_values: pd.DataFrame,
                 df_geom: Optional[pd.DataFrame]) -> pd.DataFrame:
    # Get static midpoints per ID (from df_values if present; else from df_geom)
    if {"XStart","YStart","XEnd","YEnd"}.issubset(df_values.columns):
        meta = df_values[["ID","XStart","YStart","XEnd","YEnd"]].drop_duplicates("ID")
    elif df_geom is not None and {"XStart","YStart","XEnd","YEnd"}.issubset(df_geom.columns):
        meta = df_geom[["ID","XStart","YStart","XEnd","YEnd"]].drop_duplicates("ID")
    else:
        raise ValueError("Need XStart/YStart/XEnd/YEnd in df_values or df_geom to build geometry.")
    return _midpoints(meta)

def grid_15min_idw(df_s5: pd.DataFrame,
                   df_geom: Optional[pd.DataFrame],
                   cfg: Grid15Config):
    """
    IDW grid the 15-min rainfall. Returns (cube, meta).
    cube: float32 array [time, lat, lon] in mm per 15 min (or chosen value_col)
    meta: dict with 'times','lats','lons','units','value_col'
    """
    # 1) prepare values & geometry
    dfv = _ensure_P15(df_s5)
    pts = _points_from(dfv, df_geom)   # index: ID → lon/lat
    ids = pts.index.to_list()
    M = len(ids)

    # 2) pick bbox & grid
    lon0, lon1, lat0, lat1 = _bbox_from_points(pts, cfg.pad_deg)
    lons = np.arange(lon0, lon1 + 1e-9, cfg.res_deg)
    lats = np.arange(lat0, lat1 + 1e-9, cfg.res_deg)
    nx, ny = len(lons), len(lats)

    # 3) static neighbor search (grid→points)
    lon_ref = 0.5*(lon0+lon1); lat_ref = 0.5*(lat0+lat1)
    px, py = _lonlat_to_xy_km(pts["lon"].values, pts["lat"].values, lon_ref, lat_ref, cfg.earth_km_per_deg)
    tree = cKDTree(np.c_[px, py])

    gx, gy = np.meshgrid(lons, lats)   # lon/lat
    gxk, gyk = _lonlat_to_xy_km(gx, gy, lon_ref, lat_ref, cfg.earth_km_per_deg)
    # query once (k & radius)
    dists, nn = tree.query(np.c_[gxk.ravel(), gyk.ravel()],
                           k=min(cfg.k, M), distance_upper_bound=cfg.max_radius_km)
    nn = nn.reshape(ny, nx, -1)         # int indices into point list (0..M-1 or M for 'no hit')
    dists = dists.reshape(ny, nx, -1).astype("float64")

    # 4) prep time axis
    dfv = dfv[dfv["ID"].isin(ids)]
    dfv = dfv.sort_index()
    times = pd.DatetimeIndex(dfv.index.unique()).sort_values()
    T = len(times)

    # 5) fast mapping: ID → position in pts
    id_to_pos = {id_: i for i, id_ in enumerate(ids)}
    id_pos = dfv["ID"].map(id_to_pos).to_numpy()

    # 6) allocate cube
    cube = np.full((T, ny, nx), np.nan, dtype="float32")

    # 7) run time loop (reasonably fast; neighbors precomputed)
    val_col = cfg.value_col if cfg.value_col in dfv.columns else "P15_mm"
    for ti, t in enumerate(times):
        block = dfv.loc[dfv.index == t, ["ID", val_col]].copy()
        # dense vector of point values at this time
        v = np.full(M, np.nan, dtype="float64")
        pos = block["ID"].map(id_to_pos).to_numpy()
        v[pos] = pd.to_numeric(block[val_col].to_numpy(), errors="coerce")
        # gather neighbors
        nn_idx = nn.copy()
        bad = (nn_idx >= M)
        nn_idx[bad] = 0  # placeholder
        vals = v[nn_idx]          # [ny, nx, k]
        vals[bad] = np.nan

        # IDW weights
        w = 1.0 / np.maximum(dists, 1e-6) ** cfg.power
        w[bad] = 0.0
        wsum = np.nansum(w, axis=2)
        with np.errstate(invalid="ignore"):
            grid = np.nansum(w * np.nan_to_num(vals), axis=2) / np.where(wsum > 0, wsum, np.nan)
        cube[ti] = grid.astype("float32")

    meta = dict(
        times=times.tz_convert("UTC") if times.tz is not None else times.tz_localize("UTC"),
        lons=lons, lats=lats,
        units="mm per 15 min" if val_col == "P15_mm" else "mm/h",
        value_col=val_col
    )
    return cube, meta

def save_grid_npz(path: str, cube: np.ndarray, meta: Dict):
    np.savez_compressed(
        path,
        cube=cube,
        times=meta["times"].astype("datetime64[ns]"),
        lons=meta["lons"].astype("float32"),
        lats=meta["lats"].astype("float32"),
        units=np.array([meta["units"]], dtype=object),
        value_col=np.array([meta["value_col"]], dtype=object),
    )

def quick_map(cube: np.ndarray, meta: Dict, t_index: int = 0, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8,5))
    im = ax.imshow(
        cube[t_index], origin="lower",
        extent=[meta["lons"][0], meta["lons"][-1], meta["lats"][0], meta["lats"][-1]],
        vmin=vmin, vmax=vmax, aspect="equal"
    )
    ts = pd.to_datetime(meta["times"][t_index]).strftime("%Y-%m-%d %H:%M UTC")
    plt.colorbar(im, ax=ax, label=meta["units"])
    ax.set_title(f"{meta['value_col']} @ {ts}")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.tight_layout(); plt.show()
