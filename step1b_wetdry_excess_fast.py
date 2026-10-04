# step1b_wetdry_excess_fast.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

# ---------- Base config (unchanged) ----------
@dataclass(frozen=True)
class ExcessNLAConfig:
    radius_km: float = 30.0
    min_neighbors: int = 3
    earth_km_per_deg: float = 111.32
    temporal_pad_bins: int = 2
    pool_stat: str = "max"         # "max", "p75", "median"
    thr_self_db_per_km: float = 0.08
    thr_nb_db_per_km:   float = 0.10
    n_jobs: int = 1

# ---------- NEW: spatio-temporal consensus + hysteresis ----------
@dataclass(frozen=True)
class ExcessNLAConfigV2(ExcessNLAConfig):
    consensus_pad_bins: int = 2          # neighbors allowed within ±2 bins
    consensus_min_neighbors: int = 3
    hysteresis_bins: int = 2             # keep wet for N bins after last ON
    max_excess_db_per_km: float | None = 1.2

# ---------- helpers ----------
def _lonlat_to_km(lon, lat, lon0, lat0, km_per_deg=111.32):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    x = (lon - lon0) * km_per_deg * np.cos(np.deg2rad(lat0))
    y = (lat - lat0) * km_per_deg
    return x, y

def _build_neighbor_index(cat: pd.DataFrame, cfg: ExcessNLAConfig) -> Dict[int, np.ndarray]:
    lon0 = np.nanmean(np.r_[cat["XStart"].values, cat["XEnd"].values])
    lat0 = np.nanmean(np.r_[cat["YStart"].values, cat["YEnd"].values])
    xs, ys = _lonlat_to_km(cat["XStart"], cat["YStart"], lon0, lat0, cfg.earth_km_per_deg)
    xe, ye = _lonlat_to_km(cat["XEnd"],   cat["YEnd"],   lon0, lat0, cfg.earth_km_per_deg)
    xc = 0.5*(xs+xe); yc = 0.5*(ys+ye)
    tree = cKDTree(np.c_[xc, yc])
    nn = tree.query_ball_point(np.c_[xc, yc], r=cfg.radius_km)
    return {i: np.array([j for j in idx if j != i], dtype=np.int32) for i, idx in enumerate(nn)}

def _pool_series(s: pd.Series, pad_bins: int, how: str) -> pd.Series:
    if pad_bins <= 0: return s
    win = 2*pad_bins + 1
    how = str(how).lower()
    if how == "max":
        return s.rolling(win, center=True, min_periods=1).max()
    if how in ("p75","q75","quant75","quantile75"):
        return s.rolling(win, center=True, min_periods=1).quantile(0.75)
    if how == "median":
        return s.rolling(win, center=True, min_periods=1).median()
    return s.rolling(win, center=True, min_periods=1).max()

# ---------- Original fast function (kept) ----------
def wetdry_from_excess_fast(df_step2: pd.DataFrame, cfg: ExcessNLAConfig):
    need = ["ID","XStart","YStart","XEnd","YEnd","PathLength","A_excess_db"]
    miss = [c for c in need if c not in df_step2.columns]
    if miss: raise ValueError(f"Missing required columns: {miss}")

    df = df_step2.copy()
    L = pd.to_numeric(df["PathLength"], errors="coerce").replace(0, np.nan)

    pooled = (df.groupby("ID", sort=False, group_keys=False)["A_excess_db"]
                .apply(lambda s: _pool_series(s, cfg.temporal_pad_bins, cfg.pool_stat)))
    df["A_ex_pool"] = pooled
    df["A_ex_pool_per_km"] = df["A_ex_pool"] / L

    cat = df.drop_duplicates("ID")[["ID","XStart","YStart","XEnd","YEnd"]].reset_index(drop=True)
    link_ids = cat["ID"].tolist()
    link_to_idx = {lid:i for i,lid in enumerate(link_ids)}
    adj = _build_neighbor_index(cat, cfg)
    M = len(link_ids)

    link_idx_arr = df["ID"].map(link_to_idx).to_numpy()
    time_pos = df.groupby(df.index).indices

    n = len(df)
    nb_cnt  = np.zeros(n, dtype="int16")
    nb_medk = np.full(n, np.nan, dtype="float64")

    ax_all = df["A_ex_pool_per_km"].to_numpy(dtype="float64")

    def _one(idx_pos):
        idx_pos = np.asarray(idx_pos, dtype=np.intp)
        vec = np.full(M, np.nan, dtype="float64")
        li = link_idx_arr[idx_pos]
        vec[li] = ax_all[idx_pos]
        medk = np.empty(idx_pos.size, dtype="float64")
        cnt  = np.empty(idx_pos.size, dtype="int16")
        for j, lidx in enumerate(li):
            nb = adj.get(lidx, np.empty(0, dtype=np.int32))
            if nb.size == 0:
                medk[j] = np.nan; cnt[j] = 0; continue
            vv = vec[nb]
            good = np.isfinite(vv)
            cnt[j] = np.int16(good.sum())
            medk[j] = np.nan if cnt[j]==0 else np.nanmedian(vv[good])
        return idx_pos, medk, cnt

    items = list(time_pos.values())
    if cfg.n_jobs != 1 and _HAVE_JOBLIB:
        results = Parallel(n_jobs=cfg.n_jobs, backend="loky", prefer="processes")(
            delayed(_one)(idx) for idx in items
        )
        for pos, medk, cnt in results:
            nb_medk[pos] = medk; nb_cnt[pos] = cnt
    else:
        for idx in items:
            pos, medk, cnt = _one(idx)
            nb_medk[pos] = medk; nb_cnt[pos] = cnt

    df["nb_count_ex"] = nb_cnt
    df["nb_med_A_ex_per_km"] = nb_medk

    df["is_wet_excess"] = (
        (df["A_ex_pool_per_km"] >= cfg.thr_self_db_per_km) &
        (df["nb_med_A_ex_per_km"] >= cfg.thr_nb_db_per_km) &
        (df["nb_count_ex"] >= cfg.min_neighbors)
    ).fillna(False)

    rows = []
    for lid, g in df.groupby("ID", sort=False):
        src = g[np.isfinite(g["A_ex_pool_per_km"])]
        rows.append({
            "ID": lid,
            "n_src": int(len(src)),
            "wet_frac_src_ex": float(src["is_wet_excess"].mean()),
            "med_nb_count_ex": float(np.nanmedian(src["nb_count_ex"])),
        })
    s_summary = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    return df, s_summary

# ---------- NEW: consensus + hysteresis variant ----------
def wetdry_from_excess_consensus(df_step2: pd.DataFrame, cfg: ExcessNLAConfigV2):
    need = ["ID","XStart","YStart","XEnd","YEnd","PathLength","A_excess_db"]
    if any(c not in df_step2.columns for c in need):
        raise ValueError("Missing columns for excess-based gating.")

    base_cfg = ExcessNLAConfig(
        radius_km=cfg.radius_km, min_neighbors=cfg.min_neighbors,
        earth_km_per_deg=cfg.earth_km_per_deg,
        temporal_pad_bins=cfg.temporal_pad_bins, pool_stat=cfg.pool_stat,
        thr_self_db_per_km=cfg.thr_self_db_per_km, thr_nb_db_per_km=cfg.thr_nb_db_per_km,
        n_jobs=cfg.n_jobs
    )
    df0, _ = wetdry_from_excess_fast(df_step2, base_cfg)  # gives A_ex_pool_per_km etc.

    df = df0.copy()
    idx = df.index
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")

    if cfg.max_excess_db_per_km is not None and "A_ex_pool_per_km" in df.columns:
        df["A_ex_pool_per_km"] = df["A_ex_pool_per_km"].clip(upper=cfg.max_excess_db_per_km)

    cat = df.drop_duplicates("ID")[["ID","XStart","YStart","XEnd","YEnd"]].reset_index(drop=True)
    link_ids = cat["ID"].tolist()
    link_to_idx = {lid:i for i,lid in enumerate(link_ids)}
    adj = _build_neighbor_index(cat, base_cfg)
    M = len(link_ids)

    t_groups = df.groupby(df.index).indices
    times = np.array(sorted(t_groups.keys()))

    link_idx = df["ID"].map(link_to_idx).to_numpy()
    Ax_all = df["A_ex_pool_per_km"].to_numpy(dtype="float64")
    self_over = df["A_ex_pool_per_km"].to_numpy() >= base_cfg.thr_self_db_per_km

    T = len(times)
    self_mat = np.zeros((T, M), dtype=bool)

    for ti, t in enumerate(times):
        pos = t_groups[t]
        li = link_idx[pos]
        self_mat[ti, li] = self_over[pos]

    pad = int(cfg.consensus_pad_bins)
    nb_need = int(cfg.consensus_min_neighbors)

    adj_lists = [adj.get(i, np.array([], dtype=np.int32)) for i in range(M)]
    nb_consensus = np.zeros_like(self_mat, dtype=bool)
    for ti in range(T):
        t0 = max(0, ti - pad)
        t1 = min(T, ti + pad + 1)
        win = self_mat[t0:t1, :]
        any_nb_in_win = np.zeros(M, dtype=int)
        for i in range(M):
            nb = adj_lists[i]
            if nb.size:
                any_nb_in_win[i] = int(win[:, nb].any(axis=0).sum())
        nb_consensus[ti, :] = any_nb_in_win >= nb_need

    wet_now = self_mat & nb_consensus

    keep = np.copy(wet_now)
    hold = int(cfg.hysteresis_bins)
    if hold > 0:
        for i in range(M):
            last_on = -999
            for ti in range(T):
                if wet_now[ti, i]:
                    keep[ti, i] = True
                    last_on = ti
                else:
                    keep[ti, i] = (ti - last_on) <= hold

    out = df.copy()
    out["is_wet_excess"] = False
    for ti, t in enumerate(times):
        pos = t_groups[t]
        li = link_idx[pos]
        out.loc[pos, "is_wet_excess"] = keep[ti, li]

    rows = []
    for lid, g in out.groupby("ID", sort=False):
        src = g[np.isfinite(g["A_ex_pool_per_km"])]
        rows.append({
            "ID": lid,
            "n_src": int(len(src)),
            "wet_frac_src_ex": float(src["is_wet_excess"].mean()),
            "med_nb_count_ex": float(nb_need),
        })
    s_summary = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    return out, s_summary


# --- add near the bottom of the file ---------------------------------------

# Presets for strict vs relaxed
STRICT_EXCESS_PRESET = ExcessNLAConfig(
    radius_km=20.0,          # tighter neighborhood
    min_neighbors=3,
    earth_km_per_deg=111.32,
    temporal_pad_bins=0,     # <-- no temporal pooling (strict)
    pool_stat="max",
    thr_self_db_per_km=0.08,
    thr_nb_db_per_km=0.10,
    n_jobs=1
)

RELAXED_EXCESS_PRESET = ExcessNLAConfigV2(
    radius_km=30.0,
    min_neighbors=3,
    earth_km_per_deg=111.32,
    temporal_pad_bins=2,     # pooled like you do now
    pool_stat="max",
    thr_self_db_per_km=0.08,
    thr_nb_db_per_km=0.10,
    n_jobs=1,
    # consensus/hysteresis (spatio-temporal smoothing)
    consensus_pad_bins=2,
    consensus_min_neighbors=3,
    hysteresis_bins=2,
    max_excess_db_per_km=1.2
)

def wetdry_from_excess_with_mode(df_step2: pd.DataFrame,
                                 mode: str = "strict",
                                 override: dict | None = None):
    import dataclasses
    """
    strict  -> wetdry_from_excess_fast(ExcessNLAConfig)
    relaxed -> wetdry_from_excess_consensus(ExcessNLAConfigV2)
    """
    m = str(mode).lower()
    if m == "strict":
        cfg = dataclasses.replace(STRICT_EXCESS_PRESET, **(override or {}))
        return wetdry_from_excess_fast(df_step2, cfg)
    else:
        cfg = dataclasses.replace(RELAXED_EXCESS_PRESET, **(override or {}))
        return wetdry_from_excess_consensus(df_step2, cfg)