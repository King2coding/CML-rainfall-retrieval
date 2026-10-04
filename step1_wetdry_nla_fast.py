# step1_wetdry_nla_fast.py
# Fast neighbor-link wet/dry classification with:
# - spatial KDTree neighbor search
# - optional ±time pooling of neighbor deltas (storm motion slack)
#
# Adds columns: ['dA_self_db','dA_self_db_per_km','dA_self_pool','dA_self_pool_per_km',
#                'nb_count','dA_nb_med_db','dA_nb_med_db_per_km','is_wet','src_present']
#
# Assumes 15-min cadence but is agnostic to actual frequency.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class NLAConfig:
    # spatial neighbor search
    radius_km: float = 30.0          # try 25–35 km for Ghana
    min_neighbors: int = 3
    # thresholds
    thr_self_db: float = 1.2
    thr_self_db_per_km: float = 0.6
    thr_nb_db: float = 1.4
    thr_nb_db_per_km: float = 0.7
    # temporal slack for neighbors: with 15-min data, 2 => ±30 minutes
    temporal_pad_bins: int = 2
    neighbor_pool: str = "max"       # 'max' or 'p95'
    # misc
    earth_km_per_deg: float = 111.32

def _lonlat_to_km(lon, lat, lon0, lat0, km_per_deg=111.32):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    x = (lon - lon0) * km_per_deg * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * km_per_deg
    return x, y

def _build_neighbor_index(cat: pd.DataFrame, cfg: NLAConfig) -> Dict[str, List[str]]:
    """Return adjacency: link_id -> list of neighbor IDs within radius_km (centroid-to-centroid)."""
    lon0 = np.nanmean(np.r_[cat["XStart"].values, cat["XEnd"].values])
    lat0 = np.nanmean(np.r_[cat["YStart"].values, cat["YEnd"].values])
    xs, ys = _lonlat_to_km(cat["XStart"], cat["YStart"], lon0, lat0, cfg.earth_km_per_deg)
    xe, ye = _lonlat_to_km(cat["XEnd"],   cat["YEnd"],   lon0, lat0, cfg.earth_km_per_deg)
    xc = 0.5*(xs+xe); yc = 0.5*(ys+ye)
    tree = cKDTree(np.c_[xc, yc])
    nn = tree.query_ball_point(np.c_[xc, yc], r=cfg.radius_km)
    ids = cat["ID"].to_list()
    return {ids[i]: [ids[j] for j in idx if j != i] for i, idx in enumerate(nn)}

def _ensure_deltas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Abar" not in out.columns:
        if "Pbar" in out.columns:
            out["Abar"] = -pd.to_numeric(out["Pbar"], errors="coerce")
        elif {"Pmin","Pmax"}.issubset(out.columns):
            out["Abar"] = -0.5*(pd.to_numeric(out["Pmin"], errors="coerce") +
                                 pd.to_numeric(out["Pmax"], errors="coerce"))
        else:
            raise ValueError("Need Abar or enough to derive it (Pbar or Pmin/Pmax).")
    if "dA_self_db" not in out.columns:
        out["dA_self_db"] = out.groupby("ID")["Abar"].diff().abs()
    if "PathLength" in out.columns and "dA_self_db_per_km" not in out.columns:
        L = pd.to_numeric(out["PathLength"], errors="coerce")
        out["dA_self_db_per_km"] = out["dA_self_db"] / L.replace(0, np.nan)
    return out

def _pool_temporal(s: pd.Series, pad_bins: int, how: str) -> pd.Series:
    if pad_bins <= 0:
        return s
    win = 2*pad_bins + 1
    if how == "max":
        return s.rolling(window=win, center=True, min_periods=1).max()
    elif how.lower() in ("p95","q95","quant95","quantile95"):
        return s.rolling(window=win, center=True, min_periods=1).quantile(0.95)
    else:
        return s.rolling(window=win, center=True, min_periods=1).max()

def wetdry_classify(df_clean: pd.DataFrame, cfg: NLAConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    need = ["ID","XStart","YStart","XEnd","YEnd","PathLength"]
    miss = [c for c in need if c not in df_clean.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    df = _ensure_deltas(df_clean)
    df["src_present"] = np.isfinite(df["Abar"].to_numpy())

    # temporal pooling of each link's own delta (used only by neighbors) — assign, no join
    pooled = (df.groupby("ID", sort=False, group_keys=False)["dA_self_db"]
                .apply(lambda s: _pool_temporal(s, cfg.temporal_pad_bins, cfg.neighbor_pool)))
    df["dA_self_pool"] = pooled

    # per-km pooled
    L = pd.to_numeric(df["PathLength"], errors="coerce")
    df["dA_self_pool_per_km"] = df["dA_self_pool"] / L.replace(0, np.nan)

    # neighbor index (static per link)
    cat = df.drop_duplicates("ID")[["ID","XStart","YStart","XEnd","YEnd"]].reset_index(drop=True)
    adjacency = _build_neighbor_index(cat, cfg)

    # fast lookups (time, ID) -> columns
    dmat = df.copy()
    dmat = dmat.set_index([df.index.rename("time"), "ID"])
    dmat.index.names = ["time", "ID"]

    # output buffers
    n = len(df)
    nb_count  = np.zeros(n, dtype="int16")
    nb_med    = np.full(n, np.nan, dtype="float64")
    nb_med_km = np.full(n, np.nan, dtype="float64")

    # --- time-slice loop using *positional* indices ---
    # dict: {Timestamp -> ndarray[int positions]}
    time_pos = df.groupby(df.index).indices
    id_arr = df["ID"].to_numpy()

    for t, idx_pos in time_pos.items():
        idx_pos = np.asarray(idx_pos, dtype=np.intp)
        ids_block = id_arr[idx_pos]

        # neighbor pooled deltas at time t
        pools, counts = [], []
        for lid in ids_block:
            nbs = adjacency.get(lid, [])
            if not nbs:
                pools.append(np.empty(0, dtype="float64")); counts.append(0); continue
            try:
                sel = dmat.loc[(t, nbs)]
                vv = sel["dA_self_pool"].to_numpy(dtype="float64")
                pools.append(vv[np.isfinite(vv)])
                counts.append(len(vv))
            except KeyError:
                pools.append(np.empty(0, dtype="float64")); counts.append(0)

        med_vals = np.array([np.nan if p.size == 0 else np.median(p) for p in pools], dtype="float64")
        nb_med[idx_pos]   = med_vals
        nb_count[idx_pos] = np.asarray(counts, dtype="int16")

        # per-km neighbor medians
        pools_km = []
        for lid in ids_block:
            nbs = adjacency.get(lid, [])
            if not nbs:
                pools_km.append(np.empty(0, dtype="float64")); continue
            try:
                sel = dmat.loc[(t, nbs)]
                vv = sel["dA_self_pool_per_km"].to_numpy(dtype="float64")
                pools_km.append(vv[np.isfinite(vv)])
            except KeyError:
                pools_km.append(np.empty(0, dtype="float64"))
        med_vals_km = np.array([np.nan if p.size == 0 else np.median(p) for p in pools_km], dtype="float64")
        nb_med_km[idx_pos] = med_vals_km

    df["nb_count"] = nb_count
    df["dA_nb_med_db"] = nb_med
    df["dA_nb_med_db_per_km"] = nb_med_km

    # triggers
    self_trig = (
        (df["dA_self_db"] >= cfg.thr_self_db) |
        ((df["dA_self_db_per_km"]) >= cfg.thr_self_db_per_km)
    )
    nb_ok = (
        (df["dA_nb_med_db"] >= cfg.thr_nb_db) |
        (df["dA_nb_med_db_per_km"] >= cfg.thr_nb_db_per_km)
    ) & (df["nb_count"] >= cfg.min_neighbors)

    df["is_wet"] = (self_trig & nb_ok).fillna(False)

    # summary per link
    summ = []
    for lid, g in df.groupby("ID", sort=False):
        src = g[g["src_present"]]
        summ.append({
            "ID": lid,
            "n_src": int(len(src)),
            "n_with_nb": int((src["nb_count"] >= cfg.min_neighbors).sum()),
            "wet_frac_src": float(src["is_wet"].mean()),
            "med_nb_count": float(np.nanmedian(src["nb_count"])),
        })
    s1_summary = pd.DataFrame(summ).sort_values("ID").reset_index(drop=True)
    return df, s1_summary
