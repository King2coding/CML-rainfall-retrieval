# step1_wetdry_nla_fast2.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

@dataclass(frozen=True)
class NLAConfig:
    radius_km: float = 30.0
    min_neighbors: int = 3
    thr_self_db: float = 1.2
    thr_self_db_per_km: float = 0.6
    thr_nb_db: float = 1.4
    thr_nb_db_per_km: float = 0.7
    temporal_pad_bins: int = 2
    neighbor_pool: str = "max"
    earth_km_per_deg: float = 111.32
    # NEW: parallelism
    n_jobs: int = 1  # set to 8 for your runs

def _lonlat_to_km(lon, lat, lon0, lat0, km_per_deg=111.32):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    x = (lon - lon0) * km_per_deg * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * km_per_deg
    return x, y

def _build_neighbor_index(cat: pd.DataFrame, cfg: NLAConfig) -> Dict[int, np.ndarray]:
    """Return adjacency as integer indices (fast)."""
    lon0 = np.nanmean(np.r_[cat["XStart"].values, cat["XEnd"].values])
    lat0 = np.nanmean(np.r_[cat["YStart"].values, cat["YEnd"].values])
    xs, ys = _lonlat_to_km(cat["XStart"], cat["YStart"], lon0, lat0, cfg.earth_km_per_deg)
    xe, ye = _lonlat_to_km(cat["XEnd"],   cat["YEnd"],   lon0, lat0, cfg.earth_km_per_deg)
    xc = 0.5*(xs+xe); yc = 0.5*(ys+ye)
    tree = cKDTree(np.c_[xc, yc])
    nn = tree.query_ball_point(np.c_[xc, yc], r=cfg.radius_km)
    # consistent link order [0..M-1]
    ids = cat["ID"].to_list()
    id_to_idx = {lid:i for i,lid in enumerate(ids)}
    adj_idx = {i: np.array([j for j in idx if j != i], dtype=np.int32) for i, idx in enumerate(nn)}
    return adj_idx

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

def wetdry_classify_fast(df_clean: pd.DataFrame, cfg: NLAConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Accelerated NLA with integer neighbor indices + optional parallelism."""
    need = ["ID","XStart","YStart","XEnd","YEnd","PathLength"]
    miss = [c for c in need if c not in df_clean.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    df = _ensure_deltas(df_clean)
    df["src_present"] = np.isfinite(df["Abar"].to_numpy())

    # temporal pooling (assign back; no join)
    pooled = (df.groupby("ID", sort=False, group_keys=False)["dA_self_db"]
                .apply(lambda s: _pool_temporal(s, cfg.temporal_pad_bins, cfg.neighbor_pool)))
    df["dA_self_pool"] = pooled
    L = pd.to_numeric(df["PathLength"], errors="coerce")
    df["dA_self_pool_per_km"] = df["dA_self_pool"] / L.replace(0, np.nan)

    # static catalog and neighbor indices (int)
    cat = df.drop_duplicates("ID")[["ID","XStart","YStart","XEnd","YEnd"]].reset_index(drop=True)
    link_ids = cat["ID"].tolist()
    link_to_idx = {lid:i for i,lid in enumerate(link_ids)}
    adj_idx = _build_neighbor_index(cat, cfg)
    M = len(link_ids)

    # precompute per-row link indices (fast mapping)
    link_idx_arr = df["ID"].map(link_to_idx).to_numpy()

    # buffers
    n = len(df)
    nb_count  = np.zeros(n, dtype="int16")
    nb_med    = np.full(n, np.nan, dtype="float64")
    nb_med_km = np.full(n, np.nan, dtype="float64")

    # time groups as integer positions
    time_pos: Dict[pd.Timestamp, np.ndarray] = df.groupby(df.index).indices

    # function to process one time slice using only NumPy
    dA_pool_all = df["dA_self_pool"].to_numpy(dtype="float64")
    dA_pool_km_all = df["dA_self_pool_per_km"].to_numpy(dtype="float64")

    def _process_one(idx_pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx_pos = np.asarray(idx_pos, dtype=np.intp)
        # build dense vectors (length M) for this time slice
        vec_pool   = np.full(M, np.nan, dtype="float64")
        vec_pool_k = np.full(M, np.nan, dtype="float64")
        li = link_idx_arr[idx_pos]
        vec_pool[li]   = dA_pool_all[idx_pos]
        vec_pool_k[li] = dA_pool_km_all[idx_pos]

        # per-row neighbor median & count
        med  = np.empty(idx_pos.size, dtype="float64")
        cnt  = np.empty(idx_pos.size, dtype="int16")
        medk = np.empty(idx_pos.size, dtype="float64")
        for j, lidx in enumerate(li):
            nb = adj_idx.get(lidx, np.empty(0, dtype=np.int32))
            if nb.size == 0:
                med[j] = np.nan; cnt[j] = 0; medk[j] = np.nan; continue
            vals  = vec_pool[nb]
            valsk = vec_pool_k[nb]
            good = np.isfinite(vals)
            cnt[j] = np.int16(good.sum())
            med[j] = np.nan if cnt[j]==0 else np.nanmedian(vals)
            goodk = np.isfinite(valsk)
            medk[j] = np.nan if not goodk.any() else np.nanmedian(valsk)
        return idx_pos, med, cnt, medk

    # run (parallel or serial)
    items = list(time_pos.values())
    if cfg.n_jobs != 1 and _HAVE_JOBLIB:
        results = Parallel(n_jobs=cfg.n_jobs, backend="loky", prefer="processes")(
            delayed(_process_one)(idx) for idx in items
        )
        for pos, med, cnt, medk in results:
            nb_med[pos] = med; nb_count[pos] = cnt; nb_med_km[pos] = medk
    else:
        for idx in items:
            pos, med, cnt, medk = _process_one(idx)
            nb_med[pos] = med; nb_count[pos] = cnt; nb_med_km[pos] = medk

    df["nb_count"] = nb_count
    df["dA_nb_med_db"] = nb_med
    df["dA_nb_med_db_per_km"] = nb_med_km

    # triggers
    self_trig = (
        (df["dA_self_db"] >= cfg.thr_self_db) |
        (df["dA_self_db_per_km"] >= cfg.thr_self_db_per_km)
    )
    nb_ok = (
        (df["dA_nb_med_db"] >= cfg.thr_nb_db) |
        (df["dA_nb_med_db_per_km"] >= cfg.thr_nb_db_per_km)
    ) & (df["nb_count"] >= cfg.min_neighbors)

    df["is_wet"] = (self_trig & nb_ok).fillna(False)

    # summary
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

# aliases for backwards compatibility
wetdry_classify = wetdry_classify_fast
wetdry_delta    = wetdry_classify_fast

# --- add at the very end of the file ---------------------------------------

# Presets for "strict" (Rainlink-like) vs "relaxed" (your current style)
RAINLINK_NLA_PRESET = NLAConfig(
    radius_km=20.0,          # tighter neighborhood
    min_neighbors=3,
    thr_self_db=1.0,         # a bit stricter self trigger
    thr_self_db_per_km=0.5,
    thr_nb_db=1.0,
    thr_nb_db_per_km=0.5,
    temporal_pad_bins=0,     # <-- no temporal pooling (strict)
    neighbor_pool="max",
    n_jobs=1
)

RELAXED_NLA_PRESET = NLAConfig(
    radius_km=30.0,
    min_neighbors=3,
    thr_self_db=1.2,
    thr_self_db_per_km=0.6,
    thr_nb_db=1.4,
    thr_nb_db_per_km=0.7,
    temporal_pad_bins=2,     # <-- your current pooling
    neighbor_pool="max",
    n_jobs=1
)

def wetdry_classify_with_mode(df_clean: pd.DataFrame,
                              mode: str = "strict",
                              override: dict | None = None):
    import dataclasses
    """
    Dispatch to wetdry_classify_fast with mode-based presets.
    override: optional dict of fields to tweak in the chosen preset.
    """
    m = str(mode).lower()
    base = RAINLINK_NLA_PRESET if m == "strict" else RELAXED_NLA_PRESET
    cfg = dataclasses.replace(base, **(override or {}))
    return wetdry_classify_fast(df_clean, cfg)
