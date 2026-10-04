# step1_wetdry_nla.py
# Step 1 — Wet–Dry classification (RAINLINK-style NLA) on top of the cleaned df (R0).
# Expects df_clean from r0_clean_minmax_auto.clean_minmax_auto(...), i.e., with columns:
#   ['ID', 'Abar', 'Pmin', 'Pmax', 'PathLength', 'XStart','YStart','XEnd','YEnd','src_present', ...]
# Notes:
#   - Uses Abar so "rain" ≈ positive increase regardless of RSL/TL semantics
#   - Uses neighbors within radius_km and a short-term delta on Abar
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NLAConfig:
    # Neighbor search
    radius_km: float = 15.0
    min_neighbors: int = 3          # need at least this many with data at time t
    require_majority_frac: float = 0.5  # fraction of neighbors that must pass self threshold

    # Short-term delta baseline (robust)
    delta_window_samples: int = 4    # median over the previous N samples (~1 hour for 15-min data)
    min_hist_samples: int = 2        # minimum historical samples to compute the baseline

    # Thresholds (tuned for Abar; can adjust for local network)
    thr_self_db: float = 1.4         # link's own delta threshold in dB
    thr_self_db_per_km: float = 0.7  # link delta normalized by length (dB/km)
    thr_nb_db: float = 1.4           # neighbor median delta (dB)
    thr_nb_db_per_km: float = 0.7    # neighbor median delta/length (dB/km)

    # How to handle missing path length
    compute_length_from_coords: bool = True


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    to_rad = np.pi / 180.0
    dlat = (lat2 - lat1) * to_rad
    dlon = (lon2 - lon1) * to_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * to_rad) * np.cos(lat2 * to_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def _midpoint(lat1, lon1, lat2, lon2) -> Tuple[float, float]:
    return (lat1 + lat2) / 2.0, (lon1 + lon2) / 2.0


def _build_link_catalog(df: pd.DataFrame, cfg: NLAConfig) -> pd.DataFrame:
    """One row per ID: midpoint, length (km), and geometry present flags."""
    meta_cols = ["ID", "PathLength", "XStart", "YStart", "XEnd", "YEnd"]
    missing = [c for c in meta_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) for NLA: {missing}")

    # Take first non-null meta per ID
    meta = (df.reset_index()[["ID", "PathLength", "XStart", "YStart", "XEnd", "YEnd"]]
            .drop_duplicates(subset=["ID"]).copy())
    meta["PathLength"] = pd.to_numeric(meta["PathLength"], errors="coerce")
    for c in ["XStart", "YStart", "XEnd", "YEnd"]:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")

    # Midpoint and length sanity
    meta["ym"], meta["xm"] = _midpoint(meta["YStart"], meta["XStart"], meta["YEnd"], meta["XEnd"])
    # Compute length from coords if missing/zero and allowed
    need_len = meta["PathLength"].isna() | (meta["PathLength"] <= 0)
    if need_len.any() and cfg.compute_length_from_coords:
        est = []
        for _, r in meta.iterrows():
            if np.isfinite(r["YStart"]) and np.isfinite(r["XStart"]) and np.isfinite(r["YEnd"]) and np.isfinite(r["XEnd"]):
                est.append(_haversine_km(r["YStart"], r["XStart"], r["YEnd"], r["XEnd"]))
            else:
                est.append(np.nan)
        meta.loc[need_len, "PathLength"] = np.array(est)[need_len.values]

    return meta.set_index("ID")


def _compute_neighbors(meta: pd.DataFrame, radius_km: float) -> Dict[str, List[str]]:
    ids = meta.index.tolist()
    ym = meta["ym"].values
    xm = meta["xm"].values
    neighbors: Dict[str, List[str]] = {i: [] for i in ids}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            d = _haversine_km(ym[i], xm[i], ym[j], xm[j])
            if d <= radius_km:
                neighbors[ids[i]].append(ids[j])
                neighbors[ids[j]].append(ids[i])
    return neighbors


def _rolling_baseline(A: pd.Series, win: int, minp: int) -> pd.Series:
    """Rolling median baseline using only PAST samples (closed='left')."""
    return A.rolling(window=win, min_periods=minp, closed="left").median()


def _prep_deltas(df: pd.DataFrame, cfg: NLAConfig) -> pd.DataFrame:
    """Add: Abar_baseline, delta_db, delta_db_per_km."""
    out = df.copy()
    # per link rolling baseline
    out["Abar_baseline"] = (
        out.groupby("ID", group_keys=False)["Abar"]
        .apply(lambda s: _rolling_baseline(s, cfg.delta_window_samples, cfg.min_hist_samples))
    )
    out["delta_db"] = out["Abar"] - out["Abar_baseline"]
    # normalize by path length
    out["PathLength"] = pd.to_numeric(out["PathLength"], errors="coerce")
    out["delta_db_per_km"] = out["delta_db"] / out["PathLength"]
    return out


def wetdry_classify(df_clean: pd.DataFrame, cfg: NLAConfig = NLAConfig()
                    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Classify wet/dry per (ID, time) using neighbor-link approach.
    Returns:
      df_out: original frame + ['delta_db','delta_db_per_km','nb_med_delta_db','nb_med_delta_db_per_km',
                               'nb_wet_frac','nb_count','is_wet']
      summary: per-ID coverage and wet fractions
    """
    # Ensure needed columns
    need = ["ID", "Abar", "PathLength", "XStart", "YStart", "XEnd", "YEnd"]
    miss = [c for c in need if c not in df_clean.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    # Build catalog and neighbor graph
    cat = _build_link_catalog(df_clean, cfg)
    nbrs = _compute_neighbors(cat, cfg.radius_km)

    # Prepare deltas
    df = _prep_deltas(df_clean, cfg)

    # Neighborhood stats per time step
    # We'll iterate time slices (clear & readable; optimize later if needed)
    nb_med_delta = []
    nb_med_delta_norm = []
    nb_frac_wet = []
    nb_count = []
    is_wet_list = []

    # Precompute “self wet” condition (independent of neighbors)
    self_wet = (df["delta_db"] >= cfg.thr_self_db) & (df["delta_db_per_km"] >= cfg.thr_self_db_per_km)

    # Group by time
    for t, g in df.groupby(level=0, sort=False):
        # quick dicts for this time
        d_by_id = g["delta_db"].to_dict()
        dn_by_id = g["delta_db_per_km"].to_dict()

        # which neighbors have data at this time
        nb_med = {}
        nb_med_norm = {}
        nb_frac = {}
        nb_ct = {}
        wet_now = {}

        # Determine which neighbors pass "self" wet at this time
        self_wet_ids = set(g.index.get_level_values("DateTimeUTC") if isinstance(g.index, pd.MultiIndex) else [])
        # we can just refer to self_wet aligned with df:
        wet_mask_this_time = self_wet.loc[g.index]

        # For each link present at time t:
        for link_id in g["ID"].unique():
            nlist = nbrs.get(link_id, [])
            if not nlist:
                nb_med[link_id] = np.nan
                nb_med_norm[link_id] = np.nan
                nb_frac[link_id] = np.nan
                nb_ct[link_id] = 0
                wet_now[link_id] = False
                continue

            # Collect neighbor deltas if neighbor has data at t
            vals = []
            valsn = []
            wet_votes = []
            c = 0
            for nb in nlist:
                # look up neighbor delta at this time
                # Find row mask for (t, nb) quickly:
                # because we have g limited to time t, we can filter by ID
                nb_row = g[g["ID"] == nb]
                if nb_row.empty:
                    continue
                dval = nb_row["delta_db"].iloc[0]
                dnval = nb_row["delta_db_per_km"].iloc[0]
                if np.isfinite(dval) and np.isfinite(dnval):
                    vals.append(dval)
                    valsn.append(dnval)
                    c += 1
                    # neighbor own-wet vote
                    nb_wet_vote = (dval >= cfg.thr_self_db) and (dnval >= cfg.thr_self_db_per_km)
                    wet_votes.append(nb_wet_vote)

            if c < cfg.min_neighbors:
                nb_med[link_id] = np.nan
                nb_med_norm[link_id] = np.nan
                nb_frac[link_id] = np.nan
                nb_ct[link_id] = c
                wet_now[link_id] = False  # insufficient neighborhood evidence
                continue

            nb_med_val = float(np.median(vals))
            nb_med_norm_val = float(np.median(valsn))
            frac_wet = float(np.mean(wet_votes)) if wet_votes else np.nan

            nb_med[link_id] = nb_med_val
            nb_med_norm[link_id] = nb_med_norm_val
            nb_frac[link_id] = frac_wet
            nb_ct[link_id] = c

            # Neighborhood condition
            nb_ok = (nb_med_val >= cfg.thr_nb_db) and (nb_med_norm_val >= cfg.thr_nb_db_per_km)
            vote_ok = (frac_wet >= cfg.require_majority_frac)

            # Self condition at this time for this link
            self_row = g[g["ID"] == link_id]
            self_ok = False
            if not self_row.empty:
                self_ok = bool(
                    (self_row["delta_db"].iloc[0] >= cfg.thr_self_db) and
                    (self_row["delta_db_per_km"].iloc[0] >= cfg.thr_self_db_per_km)
                )

            wet_now[link_id] = bool(self_ok and (nb_ok or vote_ok))

        # align back to g (rows at time t)
        nb_med_delta.extend([nb_med[i] for i in g["ID"].values])
        nb_med_delta_norm.extend([nb_med_norm[i] for i in g["ID"].values])
        nb_frac_wet.extend([nb_frac[i] for i in g["ID"].values])
        nb_count.extend([nb_ct[i] for i in g["ID"].values])
        is_wet_list.extend([wet_now[i] for i in g["ID"].values])

    df = df.copy()
    df["nb_med_delta_db"] = nb_med_delta
    df["nb_med_delta_db_per_km"] = nb_med_delta_norm
    df["nb_wet_frac"] = nb_frac_wet
    df["nb_count"] = nb_count
    df["is_wet"] = is_wet_list

    # Summary per link
    summ = []
    for link_id, g in df.groupby("ID"):
        src = g[g.get("src_present", True)]
        summ.append({
            "ID": link_id,
            "n_src": int(len(src)),
            "n_with_nb": int((src["nb_count"] >= cfg.min_neighbors).sum()),
            "wet_frac_src": float(src["is_wet"].mean()) if len(src) else np.nan,
            "med_nb_count": float(src["nb_count"].median()) if len(src) else 0.0,
        })
    summary = pd.DataFrame(summ).sort_values("ID").reset_index(drop=True)
    return df, summary
