# step1c_temporal_rescue.py

import dataclasses
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class TemporalRescueConfig:
    """
    Config for temporal rescue of wet flags when neighbours are insufficient.
    Operates per-link on a time-indexed DataFrame.
    """
    gamma_col: str = "A_ex_pool_per_km"   # per-km excess/gamma
    nb_col: str = "nb_count_ex"           # neighbour count column
    wet_col: str = "is_wet_final"         # primary wet mask (from S2b)

    # Only consider bins where neighbour info is weak:
    max_nb_for_rescue: int = 2            # "strict" NLA uses 3; rescue when nb < 3

    # Attenuation threshold signalling possible rain:
    gamma_thr_db_per_km: float = 0.03

    # Temporal coherence requirement (within each link’s time series):
    min_run_bins: int = 2                 # contiguous high-γ bins required

    # Network-wide anchor: only rescue when some links are already wet
    require_network_anchor: bool = True
    min_network_wet_frac: float = 0.05    # e.g. 5% of links wet

    # Output column names
    out_temporal_flag: str = "is_wet_temporal"
    out_final_col: str = "is_wet_final"   # will be overwritten
    out_orig_col: str = "is_wet_final_orig"


def _utcify_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    return out


def apply_temporal_rescue(df_in: pd.DataFrame,
                          cfg: Optional[TemporalRescueConfig] = None):
    """
    Add a temporal-rescue wet flag and update the final wet mask.

    Parameters
    ----------
    df_in : DataFrame
        Must have:
          - DatetimeIndex (UTC-able)
          - columns: ["ID", cfg.wet_col, cfg.gamma_col, cfg.nb_col]
    cfg : TemporalRescueConfig, optional

    Returns
    -------
    df_out : DataFrame
        Copy of df_in with extra columns:
          - cfg.out_temporal_flag  (bool)
          - cfg.out_orig_col       (bool)
          - cfg.out_final_col      (bool, updated)
    summary : dict
        Small dict with counts and basic stats.
    """
    if cfg is None:
        cfg = TemporalRescueConfig()

    required = ["ID", cfg.wet_col, cfg.gamma_col]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        raise ValueError(f"apply_temporal_rescue: missing required columns: {missing}")

    if cfg.nb_col not in df_in.columns:
        # If neighbour count is missing, treat as 0 (most conservative)
        df = df_in.copy()
        df[cfg.nb_col] = 0
    else:
        df = df_in.copy()

    # Ensure UTC index and sorted
    df = _utcify_index(df)
    df = df.sort_index()

    n_rows = len(df)
    rescued = np.zeros(n_rows, dtype=bool)

    # Network-wide wet fraction per time (from original mask)
    net_wet_frac: Dict[pd.Timestamp, float]
    if cfg.require_network_anchor:
        net_wet = (
            df[cfg.wet_col]
            .groupby(df.index)
            .mean()
        )
        net_wet_frac = net_wet.to_dict()
    else:
        net_wet_frac = {}

    # Process per link for temporal coherence
    # Keep track of global row positions for each group to write back flags
    df_reset = df.reset_index()
    idx_arr = df_reset.index.to_numpy()

    for link_id, sub in df_reset.groupby("ID", sort=False):
        # Row positions in the original df
        pos = sub.index.to_numpy()

        gamma = pd.to_numeric(sub[cfg.gamma_col], errors="coerce").to_numpy(float)
        nb = pd.to_numeric(sub[cfg.nb_col], errors="coerce").to_numpy(float)
        wet0 = sub[cfg.wet_col].astype(bool).to_numpy()

        # Candidate bins: not already wet, low neighbour count, gamma above thr
        cand = (~wet0) & (nb <= cfg.max_nb_for_rescue) & np.isfinite(gamma)
        high = cand & (gamma >= cfg.gamma_thr_db_per_km)

        if not high.any():
            continue

        # Find contiguous runs where high == True
        # Simple run-length encoding
        idx_high = np.where(high)[0]
        # Split into contiguous blocks
        splits = np.where(np.diff(idx_high) > 1)[0] + 1
        blocks = np.split(idx_high, splits)

        for block in blocks:
            if len(block) < cfg.min_run_bins:
                continue

            # Network anchor check (optional)
            if cfg.require_network_anchor:
                times_block = sub["index"].iloc[block]  # this "index" column is original DatetimeIndex
                ok_anchor = False
                for t in times_block:
                    f = net_wet_frac.get(t, 0.0)
                    if f >= cfg.min_network_wet_frac:
                        ok_anchor = True
                        break
                if not ok_anchor:
                    continue  # skip this block

            # If we reach here, rescue the whole block
            rescued[pos[block]] = True

    df[cfg.out_temporal_flag] = rescued.astype(bool)

    # Preserve original mask, then update
    df[cfg.out_orig_col] = df[cfg.wet_col].astype(bool)
    df[cfg.out_final_col] = df[cfg.out_orig_col] | df[cfg.out_temporal_flag]

    summary: Dict[str, Any] = {
        "n_rows": n_rows,
        "n_rescued": int(rescued.sum()),
        "frac_rescued": float(rescued.mean()),
        "gamma_thr_db_per_km": cfg.gamma_thr_db_per_km,
        "max_nb_for_rescue": cfg.max_nb_for_rescue,
        "min_run_bins": cfg.min_run_bins,
        "require_network_anchor": cfg.require_network_anchor,
        "min_network_wet_frac": cfg.min_network_wet_frac,
    }
    return df, summary