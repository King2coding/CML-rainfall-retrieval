# step1d_singlelink_fallback.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd


@dataclass
class SingleLinkConfig:
    """
    Single-link wet/dry fallback for sparse networks.

    Idea:
      - If a link shows persistent excess attenuation A_ex above a threshold
        BUT has too few neighbours for NLA to decide,
        we classify it as wet based on its own time series.

    Parameters
    ----------
    gamma_col : str
        Column with per-km excess attenuation (e.g. 'A_ex_pool_per_km').
    nb_col : Optional[str]
        Column with neighbour count (e.g. 'nb_count_ex'). If None, no
        neighbour constraint is applied.
    wet_col : str
        Name of the final wet mask column to update (e.g. 'is_wet_final').
    thr_db_per_km : float
        Threshold for "significant" excess attenuation [dB/km].
    min_run_bins : int
        Minimum number of consecutive 15-min bins above threshold to
        accept as rain (e.g. 2 → at least 30 minutes).
    max_nb_for_fallback : int
        Only apply single-link fallback where neighbour count is
        <= this value (sparse neighbourhood).
    """

    gamma_col: str = "A_ex_pool_per_km"
    nb_col: Optional[str] = "nb_count_ex"
    wet_col: str = "is_wet_final"
    thr_db_per_km: float = 0.05
    min_run_bins: int = 2
    max_nb_for_fallback: int = 2


def _find_runs(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given a boolean 1D array, return start and end indices (inclusive)
    of all contiguous True runs.
    """
    if mask.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # True where a run starts / ends
    starts = (mask & ~np.r_[False, mask[:-1]]).nonzero()[0]
    ends   = (mask & ~np.r_[mask[1:], False]).nonzero()[0]
    return starts, ends


def apply_singlelink_fallback(
    df_in: pd.DataFrame,
    cfg: SingleLinkConfig,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply single-link fallback wet classification.

    Returns
    -------
    df_out : DataFrame
        Same as input but with:
          - 'is_wet_single' (bool) column added
          - cfg.wet_col updated as OR(original, is_wet_single)
    summary : dict
        Diagnostics (n_rescued, thresholds, etc.).
    """
    required = [cfg.gamma_col, cfg.wet_col]
    missing = [c for c in required if c not in df_in.columns]
    if missing:
        raise ValueError(
            f"apply_singlelink_fallback: missing required columns: {missing}"
        )

    df = df_in.copy()
    # Ensure boolean
    df[cfg.wet_col] = df[cfg.wet_col].astype(bool)

    # Prepare gamma
    gamma = pd.to_numeric(df[cfg.gamma_col], errors="coerce")

    # Neighbour count (optional)
    if cfg.nb_col is not None and cfg.nb_col in df.columns:
        nb = pd.to_numeric(df[cfg.nb_col], errors="coerce").fillna(0).astype(int)
    else:
        nb = None

    # Output mask
    is_wet_single = pd.Series(False, index=df.index)

    # Work group-by-link for contiguous runs
    for lid, g in df.groupby("ID"):
        idx = g.index

        g_gamma = gamma.loc[idx]
        cond = g_gamma > cfg.thr_db_per_km
        cond &= g_gamma.notna()

        if nb is not None:
            g_nb = nb.loc[idx]
            cond &= (g_nb <= cfg.max_nb_for_fallback)

        if not cond.any():
            continue

        arr = cond.to_numpy()
        starts, ends = _find_runs(arr)

        for s, e in zip(starts, ends):
            run_len = e - s + 1
            if run_len >= cfg.min_run_bins:
                # mark this whole block as single-link wet
                run_idx = idx[s : e + 1]
                is_wet_single.loc[run_idx] = True

    # How many wet flags were added relative to original?
    already_wet = df[cfg.wet_col].to_numpy()
    rescued = is_wet_single & ~df[cfg.wet_col]

    df["is_wet_single"] = is_wet_single
    df[cfg.wet_col] = df[cfg.wet_col] | is_wet_single

    summary = dict(
        n_rows=int(len(df)),
        n_single_wet=int(is_wet_single.sum()),
        n_rescued=int(rescued.sum()),
        frac_rescued=float(rescued.sum()) / float(len(df)) if len(df) else 0.0,
        thr_db_per_km=cfg.thr_db_per_km,
        min_run_bins=cfg.min_run_bins,
        max_nb_for_fallback=cfg.max_nb_for_fallback,
        gamma_col=cfg.gamma_col,
        nb_col=cfg.nb_col,
        wet_col=cfg.wet_col,
    )

    return df, summary