# cml_rainfall_retrieval_core_bas_202605.py
#
# Core CML rainfall retrieval and gridding functions for Bas review.
#
# This file was distilled from the current Ghana CML rainfall pipeline used in the
# TAHMO/Rainboo workflow. It intentionally removes the API/service wrapper and
# keeps the technical method: CML signal cleaning, wet/dry classification, dry
# baseline estimation, wet-antenna correction, rainfall-rate conversion, and
# gridding/support layers.
#
# Notes for review:
# - Comments are kept where they explain scientific or operational choices.
# - The support/confidence layer is not a formal uncertainty estimate; it is a
#   practical representation of CML network support and wet/dry consistency.
# - Directory paths are handled in the separate run script, not here.

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from datetime import datetime, timezone, timedelta
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

from joblib import Parallel, delayed, parallel_backend
from scipy.spatial import cKDTree
from scipy.ndimage import (
    uniform_filter,
    binary_opening,
    generate_binary_structure,
    distance_transform_edt,
    binary_closing,
    binary_fill_holes,
    label,
)

from pycomlink.processing.wet_antenna import waa_leijnse_2008_from_A_obs
from pycomlink.processing.k_R_relation import calc_R_from_A

try:
    from pykrige.ok import OrdinaryKriging
    _PYKRIGE_AVAILABLE = True
except Exception:
    _PYKRIGE_AVAILABLE = False

try:
    from sklearn.neighbors import BallTree
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")

_EPS = 1e-12
_KM_PER_DEG = 111.0
_EARTH_R_KM = 6371.0

# Optional lightweight pipeline logging. This is retained because some functions
# append diagnostic messages, but it is not required for the rainfall retrieval.
PIPELINE_BASE_DIR = Path(__file__).resolve().parent
PIPELINE_LOG_DIR = PIPELINE_BASE_DIR / "rainfall_logs"
PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 1. R0 signal cleaning configuration and utilities
# =============================================================================

@dataclass

class R0AutoConfig:
    # Time
    cadence_minutes: int = 15
    snap_tolerance: str = "2min"        # snap to nearest 15-min if within this tolerance
    regularize_grid: bool = True        # reindex to exact 15-min grid (gaps -> NaN)
    source_tz: str = "Africa/Accra"     # Ghana is UTC year-round

    # Bounds for RSL (dBm) and TL (dB)
    rsl_min_dbm: float = -130.0
    rsl_max_dbm: float = -20.0
    tl_min_db: float = 0.0
    tl_max_db: float = 80.0

    # Within-bin dynamic range (15-min data)
    max_dyn_range_db: float = 12.0

    # Outage heuristics (semantics-aware)
    rsl_outage_floor_dbm: float = -115.0
    tl_outage_high_db: float = 75.0
    outage_min_consec: int = 2

    # Hampel (spike) on Pbar = (Pmin+Pmax)/2  [relaxed]
    hampel_window: int = 7
    hampel_nsigma: float = 6.0

    # Plateau (flag-only)  [relaxed]
    plateau_run_len: int = 20           # 5 h
    plateau_tol_db: float = 0.05

    # Unpaired-jump rule
    unpaired_spread_db: float = 6.0
    unpaired_delta_db: float = 2.0

    # Semantics control
    semantics: str = "auto"             # "auto" | "rsl" | "tl"
    # Auto-retry trigger
    retry_oob_frac: float = 0.50        # if >50% OOB …
    retry_valid_frac: float = 0.05      # … and <5% valid pairs → flip semantics & retry

def _parse_dt(series: pd.Series, cfg: R0AutoConfig) -> pd.DatetimeIndex:
    dt = pd.to_datetime(series.astype(str), format="%Y%m%d%H%M", errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(cfg.source_tz, nonexistent="shift_forward", ambiguous="NaT")
    return dt.dt.tz_convert("UTC")

def _snap_to_grid(dt: pd.Series, minutes: int, tol: pd.Timedelta) -> pd.Series:
    base = dt.dt.floor(f"{minutes}min")
    offs = (dt - base)
    up = offs >= pd.Timedelta(minutes=minutes/2)
    anchor = base.where(~up, base + pd.Timedelta(minutes=minutes))
    diff = (dt - anchor).abs()
    return anchor.where(diff <= tol)  # otherwise NaT

def _hampel_mask(x: pd.Series, window: int, nsigma: float) -> pd.Series:
    med = x.rolling(window, center=True, min_periods=3).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=3).median()
    sigma = 1.4826 * mad
    return ((x - med).abs() > nsigma * sigma).fillna(False)

def _flag_plateaus(x: pd.Series, run_len: int, tol_db: float) -> pd.Series:
    if x.isna().all():
        return pd.Series(False, index=x.index)
    d = x.diff().abs().fillna(0.0) <= tol_db
    gid = (~d).cumsum()
    counts = pd.Series(gid).map(pd.Series(gid).value_counts())
    return (d & (counts.values >= run_len)).reindex_like(x).fillna(False)

def _consec_true(mask: pd.Series, min_len: int) -> pd.Series:
    if mask.empty:
        return mask
    gid = (mask != mask.shift(1, fill_value=False)).cumsum()
    run_len = gid.map(gid.value_counts())
    return mask & (run_len >= min_len)

def _detect_semantics_robust(pmin: pd.Series, pmax: pd.Series, cfg: R0AutoConfig) -> str:
    """Decide RSL vs TL by counting finite samples that fall inside each range."""
    vals = pd.concat([pmin, pmax], axis=0).dropna().values
    if vals.size == 0:
        return "rsl"
    in_rsl = ((vals >= cfg.rsl_min_dbm) & (vals <= cfg.rsl_max_dbm)).sum()
    in_tl  = ((vals >= cfg.tl_min_db)    & (vals <= cfg.tl_max_db)).sum()
    if in_rsl > in_tl:
        return "rsl"
    if in_tl > in_rsl:
        return "tl"
    # tie-breaker: sign of median
    return "rsl" if np.nanmedian(vals) < 0 else "tl"

def _clean_one_link(g: pd.DataFrame, semantics: str, cfg: R0AutoConfig) -> Tuple[pd.DataFrame, Dict]:
    """Return cleaned df (after reindex) + metrics computed on src_present only."""
    s = g.copy()

    # Flags container
    flags = {k: np.zeros(len(s), dtype=bool) for k in
             ["OOB", "SWAP", "DYN", "OUTAGE", "UNPAIRED", "SPIKE", "PLAT", "SEM_RSL", "SEM_TL"]}
    flags["SEM_RSL"][:] = (semantics == "rsl")
    flags["SEM_TL"][:] = (semantics == "tl")

    # Bounds by semantics
    if semantics == "rsl":
        bad_min = ~s["Pmin"].between(cfg.rsl_min_dbm, cfg.rsl_max_dbm, inclusive="both")
        bad_max = ~s["Pmax"].between(cfg.rsl_min_dbm, cfg.rsl_max_dbm, inclusive="both")
    else:
        bad_min = ~s["Pmin"].between(cfg.tl_min_db, cfg.tl_max_db, inclusive="both")
        bad_max = ~s["Pmax"].between(cfg.tl_min_db, cfg.tl_max_db, inclusive="both")
    oob = bad_min | bad_max
    flags["OOB"] = oob.values
    s.loc[oob, ["Pmin", "Pmax"]] = np.nan

    # Order: Pmax >= Pmin
    need_swap = (s["Pmin"].notna() & s["Pmax"].notna() & (s["Pmin"] > s["Pmax"]))
    s.loc[need_swap, ["Pmin", "Pmax"]] = s.loc[need_swap, ["Pmax", "Pmin"]].values
    flags["SWAP"] = need_swap.values

    # Dynamic range
    spread = s["Pmax"] - s["Pmin"]
    dyn_bad = (spread > cfg.max_dyn_range_db)
    flags["DYN"] = dyn_bad.fillna(False).values
    s.loc[dyn_bad, ["Pmin", "Pmax"]] = np.nan

    # Outage (semantics-aware)
    if semantics == "rsl":
        low_min = s["Pmin"] <= cfg.rsl_outage_floor_dbm
        low_max = s["Pmax"] <= cfg.rsl_outage_floor_dbm
        outage = _consec_true(low_min | low_max, cfg.outage_min_consec)
    else:
        high_min = s["Pmin"] >= cfg.tl_outage_high_db
        high_max = s["Pmax"] >= cfg.tl_outage_high_db
        outage = _consec_true(high_min | high_max, cfg.outage_min_consec)
    flags["OUTAGE"] = outage.values
    s.loc[outage, ["Pmin", "Pmax"]] = np.nan

    # Unpaired jump
    delta_min = s["Pmin"].diff().abs()
    delta_max = s["Pmax"].diff().abs()
    unpaired = (
        (spread > cfg.unpaired_spread_db) &
        (
            ((delta_min > cfg.unpaired_delta_db) & (delta_max <= cfg.unpaired_delta_db)) |
            ((delta_max > cfg.unpaired_delta_db) & (delta_min <= cfg.unpaired_delta_db))
        )
    )
    flags["UNPAIRED"] = unpaired.fillna(False).values
    s.loc[unpaired, ["Pmin", "Pmax"]] = np.nan

    # -------------------------------------------------------------------------
    # Signal convention used by the retrieval
    # -------------------------------------------------------------------------
    # The AT Ghana raw files provide Pmin/Pmax as the available signal pair.
    # We use their midpoint as the representative signal for each logging interval:
    #     Pbar = (Pmin + Pmax) / 2
    # This is a conservative operational choice that reduces sensitivity to
    # short-lived extrema/noise within the interval.
    # For each link, Pmin/Pmax may behave either like received signal level
    # values (RSL; typically negative dBm) or like total/path-loss values
    # (TL; typically positive dB). We therefore convert Pbar into sig_db,
    # an internal standardized signal where larger values always mean
    # stronger/drier signal:
    #     sig_db =  Pbar    for RSL-like links
    #     sig_db = -Pbar    for TL/path-loss-like links
    # This allows the same attenuation calculation to be applied downstream:
    #     A_obs = dry_baseline_sig_db - current_sig_db
    # -------------------------------------------------------------------------

    # Hampel spikes
    pbar = (s["Pmin"] + s["Pmax"]) / 2.0
    spike = _hampel_mask(pbar, cfg.hampel_window, cfg.hampel_nsigma)
    flags["SPIKE"] = spike.values
    s.loc[spike, ["Pmin", "Pmax"]] = np.nan

    # Plateau (flag-only)
    pbar_after = (s["Pmin"] + s["Pmax"]) / 2.0
    plat = _flag_plateaus(pbar_after, cfg.plateau_run_len, cfg.plateau_tol_db)
    flags["PLAT"] = plat.values

    # Record source presence BEFORE reindex
    src_present = pd.Series(True, index=s.index, name="src_present")

    # Regularize to exact 15-min grid
    if cfg.regularize_grid and not s.empty:
        full_idx = pd.date_range(s.index.min(), s.index.max(),
                                 freq=f"{cfg.cadence_minutes}min", tz="UTC")
        s = s.reindex(full_idx)

    # Build QC string on the original index, then reindex
    def pack(i: int) -> str:
        labs = [k for k, v in flags.items() if i < len(v) and v[i]]
        return ",".join(labs) if labs else ""

    qc_series = pd.Series([pack(i) for i in range(len(g))], index=g.index)
    s["qc_flags"] = qc_series.reindex(s.index).fillna("")
    s["src_present"] = src_present.reindex(s.index).fillna(False)

    # Convenience fields
    # ------------------------------------------------------------
    # Convenience fields (semantics-aware signal definition)
    # ------------------------------------------------------------
    # Pbar is the mid-point of min/max (either RSL [dBm] or TL [dB])
    s["Pbar"] = (s["Pmin"] + s["Pmax"]) / 2.0
    s["Pspread"] = s["Pmax"] - s["Pmin"]
    s["semantics"] = semantics

    # IMPORTANT:
    # We define a single "signal" time series to baseline against later.
    # - If semantics == "rsl": Pbar is already RSL in dBm (higher = stronger signal).
    # - If semantics == "tl" : Pbar is TL in dB (higher = more loss), so we flip sign.
    #   This makes sig_db behave like received power (higher = stronger signal),
    #   so baseline logic (baseline - signal) produces positive attenuation-like values.
    s["sig_db"] = np.where(s["semantics"] == "rsl", s["Pbar"], -s["Pbar"])

    # Metrics (only over true source samples)
    src = s[s["src_present"]]
    metrics = {
        "n_src": int(src.shape[0]),
        "n_valid_pairs_src": int(src[["Pmin", "Pmax"]].dropna().shape[0]),
        "frac_oob_src": float(src["qc_flags"].str.contains("OOB").mean()) if len(src) else 0.0,
        "frac_valid_src": float(src[["Pmin", "Pmax"]].dropna().shape[0] / max(len(src), 1))
    }
    return s, metrics

def clean_minmax_auto(df_raw: pd.DataFrame, cfg: R0AutoConfig = R0AutoConfig()
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean 15-min Pmin/Pmax per link (ID) with robust semantics detection + auto-retry.

    Required columns: ['ID','DateTime','Pmin','Pmax']
    Other columns are passed through untouched.

    Returns
    -------
    df_out : per-timestamp rows with cleaned Pmin/Pmax, Pbar, Pspread, Abar, qc_flags, semantics, src_present
    df_summary : per-ID summary stats (computed on src_present only)
    """
    need = ["ID", "DateTime", "Pmin", "Pmax"]
    missing = [c for c in need if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df_raw.copy()

    # Parse & snap to 15-min
    dt = _parse_dt(df["DateTime"], cfg)
    snap = _snap_to_grid(dt, cfg.cadence_minutes, pd.Timedelta(cfg.snap_tolerance))
    df["DateTimeUTC"] = snap
    df = df.dropna(subset=["DateTimeUTC"]).sort_values(["ID", "DateTimeUTC"])
    df = df.drop_duplicates(subset=["ID", "DateTimeUTC"], keep="first")
    df.set_index("DateTimeUTC", inplace=True)
    # Keep a clean, human-readable time stamp column for debugging/exports
    # (prevents scientific notation like 2.025061e+11 in df_clean.head()).
    df["DateTime_str"] = df.index.strftime("%Y%m%d%H%M")

    # Numeric Pmin/Pmax
    df["Pmin"] = pd.to_numeric(df["Pmin"], errors="coerce")
    df["Pmax"] = pd.to_numeric(df["Pmax"], errors="coerce")

    out_frames: List[pd.DataFrame] = []
    summaries: List[Dict] = []

    for link_id, g in df.groupby("ID", sort=False):
        # Decide semantics
        sem0 = cfg.semantics if cfg.semantics in ("rsl", "tl") else _detect_semantics_robust(g["Pmin"], g["Pmax"], cfg)

        # First pass
        s0, m0 = _clean_one_link(g, sem0, cfg)

        # Retry logic
        use_retry = (m0["frac_oob_src"] > cfg.retry_oob_frac) and (m0["frac_valid_src"] < cfg.retry_valid_frac)
        if use_retry:
            sem1 = "tl" if sem0 == "rsl" else "rsl"
            s1, m1 = _clean_one_link(g, sem1, cfg)
            # Keep the better (more valid) result
            if m1["n_valid_pairs_src"] > m0["n_valid_pairs_src"]:
                s_keep, m_keep, sem_keep = s1, m1, sem1
            else:
                s_keep, m_keep, sem_keep = s0, m0, sem0
        else:
            s_keep, m_keep, sem_keep = s0, m0, sem0

        # Attach ID
        s_keep["ID"] = link_id

        # Summary (src domain)
        summaries.append({
            "ID": link_id,
            "semantics": sem_keep,
            "n_rows_src": m_keep["n_src"],
            "n_valid_pairs_src": m_keep["n_valid_pairs_src"],
            "frac_oob_src": m_keep["frac_oob_src"],
            "frac_valid_src": m_keep["frac_valid_src"],
        })

        out_frames.append(s_keep)

    df_out = pd.concat(out_frames, axis=0).sort_index()
    df_out["DateTime"] = df_out.index.strftime("%Y%m%d%H%M")  # ensure always present
    # Optional: standardize output time label column name
    # Keeps compatibility with your raw schema while remaining readable.
    if "DateTime" in df_out.columns:
        df_out = df_out.drop(columns=["DateTime"])
    df_out = df_out.rename(columns={"DateTime_str": "DateTime"})
    df_summary = pd.DataFrame(summaries).sort_values(["ID"]).reset_index(drop=True)
    return df_out, df_summary


# =============================================================================
# 2. RainLINK-style 15-min baseline, wet/dry, and rainfall retrieval
# =============================================================================

def _baseline_q90_past_only(
    rsl_series: pd.Series,
    win: str = "24H",
    q: float = 0.9,
    min_past_bins: int = 8,        # 8 bins = 2 hours @ 15-min
    ffill_limit_bins: int = 32,    # carry forward baseline <= 8 hours
    bfill_limit_bins: int = 4      # tiny backfill at start of record
) -> pd.Series:
    """
    Rolling quantile baseline using ONLY past samples (closed='left').

    Why:
      - operationally stable
      - avoids "peeking" into the future
      - quantile baseline is robust to occasional dips

    Notes:
      - set min_past_bins high enough to reduce noisy baseline early in the day
      - ffill_limit prevents baseline drifting forever across missing data
    """
    base = rsl_series.rolling(window=win, min_periods=min_past_bins, closed="left").quantile(q)
    base = base.ffill(limit=ffill_limit_bins).bfill(limit=bfill_limit_bins)
    return base

def build_15min_timeseries(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build a strict 15-min, one-row-per-(link,time) table for the PRIME pipeline.

    Key idea (IMPORTANT):
      - We DO NOT assume Pmin/Pmax are always RSL (dBm). Some links are TL (dB).
      - R0 already creates `sig_db` which is semantics-safe:
          * semantics == "rsl" -> sig_db =  Pbar   (higher = stronger)
          * semantics == "tl"  -> sig_db = -Pbar   (higher = stronger)
      - Prime baseline should run on a single, consistent “stronger = larger” series.
        So we baseline `sig_db` (not the old RainLINK RSL formula).

    Inputs expected (at least):
      ['ID','DateTime' (or datetime index), 'Pmin','Pmax','Frequency','PathLength',
       'XStart','YStart','XEnd','YEnd','Polarization']
    Recommended from R0:
      ['src_present','qc_flags','semantics','sig_db']

    Output columns (core):
      ['link_id','time','sig_db','Frequency','PathLength','XStart','YStart','XEnd','YEnd','pol']
    Plus QA passthrough when present:
      ['src_present','qc_flags','semantics']
    """
    r = df_clean.copy()

    # ------------------------------------------------------------
    # 1) Time handling (prefer DatetimeIndex from R0 output)
    # ------------------------------------------------------------
    if isinstance(r.index, pd.DatetimeIndex):
        t = r.index
        if t.tz is None:
            t = t.tz_localize("UTC")
        else:
            t = t.tz_convert("UTC")
        r["time"] = t
    else:
        # fallback to DateTime column (digits-only)
        s = r["DateTime"].astype(str).str.strip()
        s = s.str.replace(r"\.0+$", "", regex=True)
        s = s.str.replace(r"[^0-9]", "", regex=True)

        L = s.str.len()
        t = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")

        m12 = L == 12  # YYYYMMDDHHMM
        m14 = L == 14  # YYYYMMDDHHMMSS
        if m12.any():
            t.loc[m12] = pd.to_datetime(s[m12], format="%Y%m%d%H%M", utc=True, errors="coerce")
        if m14.any():
            t.loc[m14] = pd.to_datetime(s[m14], format="%Y%m%d%H%M%S", utc=True, errors="coerce")

        other = ~(m12 | m14)
        if other.any():
            t.loc[other] = pd.to_datetime(s[other], utc=True, errors="coerce")

        r["time"] = t

    # strict 15-min bins (floor)
    r["time15"] = pd.to_datetime(r["time"], utc=True, errors="coerce").dt.floor("15min")

    # ------------------------------------------------------------
    # 2) Link identity (must match gridding/meta exactly)
    # ------------------------------------------------------------
    r["link_id"] = r["ID"].astype(str)

    # ------------------------------------------------------------
    # 3) Numeric conversion for meta + signal pieces
    # ------------------------------------------------------------
    for c in ["Pmin", "Pmax", "Frequency", "PathLength", "XStart", "YStart", "XEnd", "YEnd"]:
        if c in r.columns:
            r[c] = pd.to_numeric(r[c], errors="coerce")

    # polarization single char
    r["pol"] = r["Polarization"].astype(str).str.upper().str[0]

    # ------------------------------------------------------------
    # 4) Build the semantics-safe signal used for baseline
    # ------------------------------------------------------------
    if "sig_db" in r.columns:
        # Best case: produced by R0 (already semantics-safe)
        r["sig_db"] = pd.to_numeric(r["sig_db"], errors="coerce")
    else:
        # Fallback: reconstruct from Pmin/Pmax (+ semantics if present)
        r["Pbar"] = (r["Pmin"] + r["Pmax"]) / 2.0
        sem = r["semantics"] if "semantics" in r.columns else "rsl"
        # If sem is missing, default to rsl assumption (negative values typical)
        r["sig_db"] = np.where(sem == "tl", -r["Pbar"], r["Pbar"])

    # ------------------------------------------------------------
    # 5) Aggregate to one value per (link_id, time15)
    # ------------------------------------------------------------
    passthrough = {}
    if "src_present" in r.columns:
        passthrough["src_present"] = "max"  # True if any src in that bin
    if "qc_flags" in r.columns:
        passthrough["qc_flags"] = "first"
    if "semantics" in r.columns:
        passthrough["semantics"] = "first"

    agg_map = {
        "sig_db": "median",
        "Frequency": "median",
        "PathLength": "median",
        "XStart": "median",
        "YStart": "median",
        "XEnd": "median",
        "YEnd": "median",
        "pol": "first",
        **passthrough,
    }

    ts = (
        r.groupby(["link_id", "time15"], as_index=False)
         .agg(agg_map)
         .rename(columns={"time15": "time"})
         .sort_values(["link_id", "time"])
         .reset_index(drop=True)
    )

    # guard
    bad = ts["time"].isna().sum()
    if bad:
        raise ValueError(
            f"{bad} timestamps are NaT after parsing. Example:\n{ts[ts['time'].isna()].head()}"
        )

    return ts

# Uses a past-only dry baseline so the method can be applied in near-real-time
# without peeking into future CML observations.
# NOTE ON WET/DRY CLASSIFICATION
# Several CML rainfall retrieval methods first classify wet/dry periods using
# neighbor-link consistency, signal statistics, or external references such as
# gauges/radar/satellite data. In this operational version, wet/dry is derived
# internally from a first-pass attenuation estimate:
#
#   1. estimate a past-only high-quantile baseline,
#   2. compute first-pass attenuation,
#   3. flag wet periods where A_obs exceeds wet_thr_db,
#   4. recompute the baseline after excluding initially wet samples.
#
# This two-pass approach keeps the method self-contained and near-real-time,
# but it may be sensitive to false wet detections. A useful optimization task
# is to test an explicit wet/dry classifier based on neighboring links and/or
# short-term signal statistics before the final baseline step.
def rainlink_strict_Aobs(
    ts_15: pd.DataFrame,
    wet_thr_db: float = 3.0,
    win: str = "24H",
    q: float = 0.99,
    min_past_bins: int = 8,
    ffill_limit_bins: int = 32,
    two_pass: bool = True,
    require_src_present: bool = True,
    # NEW: guard to avoid bad pass-2 baselines
    use_drycount_guard: bool = True,
    min_dry_bins: int = 8,             # min dry samples required within `win` to accept base2
    dry_mask_source: str = "wet1",     # "wet1" (from pass-1) or "wet_final" (from final Aobs)
    guard_behavior: str = "fallback",  # "fallback" -> use base1, "zero" -> force Aobs=0 when unreliable
) -> pd.DataFrame:
    """
    Compute:
      baseline_rsl (past-only q90),
      A_obs_dB = max(0, baseline - sig_db),
      wet_rl   = A_obs_dB > wet_thr_db

    NEW (optional) dry-count guard:
      - recompute baseline using dry-only samples (pass-2),
      - BUT only trust base2 where we have enough dry samples within the rolling window.
      - Else either fall back to base1 or set Aobs=0 (user-controlled).
    """
    if guard_behavior not in ("fallback", "zero"):
        raise ValueError("guard_behavior must be 'fallback' or 'zero'")
    if dry_mask_source not in ("wet1", "wet_final"):
        raise ValueError("dry_mask_source must be 'wet1' or 'wet_final'")

    out = []
    win_td = pd.Timedelta(win)

    for lid, g in ts_15.groupby("link_id", sort=False):
        g = g.sort_values("time").copy()
        idx = pd.DatetimeIndex(pd.to_datetime(g["time"], utc=True))

        sig = pd.Series(pd.to_numeric(g["sig_db"], errors="coerce").astype(float).values, index=idx)

        # optionally mask non-source rows (avoid regularized filler influencing baseline)
        if require_src_present and ("src_present" in g.columns):
            src_mask = g["src_present"].astype(bool).values
            sig_src = sig.mask(~src_mask)
        else:
            sig_src = sig

        # -------------------
        # pass 1 (all src samples)
        # -------------------
        base1 = _baseline_q90_past_only(
            sig_src, win=win, q=q,
            min_past_bins=min_past_bins,
            ffill_limit_bins=ffill_limit_bins
        )
        A1 = np.maximum(0.0, base1.values - sig.values)
        wet1 = (A1 > wet_thr_db) & np.isfinite(A1)

        # default: pass-1 outputs
        base = base1.copy()

        # -------------------
        # pass 2 (dry-only recompute)
        # -------------------
        if two_pass:
            # choose which wet mask defines "dry"
            wet_mask_for_dry = wet1

            # dry-only series used for pass-2 baseline
            sig_dry = sig_src.mask(wet_mask_for_dry)

            base2 = _baseline_q90_past_only(
                sig_dry, win=win, q=q,
                min_past_bins=min_past_bins,
                ffill_limit_bins=ffill_limit_bins
            )

            if use_drycount_guard:
                # rolling dry sample count over the SAME window
                # (count finite samples in sig_dry)
                dry_count = sig_dry.notna().rolling(window=win_td, closed="left").sum()

                # accept base2 only where enough dry samples exist
                ok2 = (dry_count.values >= float(min_dry_bins))

                # where base2 isn't OK, either fallback to base1 or mark as unreliable
                if guard_behavior == "fallback":
                    base = base1.copy()
                    base.values[ok2] = base2.values[ok2]
                else:  # guard_behavior == "zero"
                    # still use base2 where ok2, else we'll zero out Aobs later
                    base = base1.copy()
                    base.values[ok2] = base2.values[ok2]

                # store diagnostics (handy for debugging north links)
                g["dry_count_win"] = dry_count.values
                g["base2_ok"] = ok2
            else:
                # original behavior: prefer base2 where available, else base1
                base = base2.fillna(base1)

        # -------------------
        # final Aobs/wet
        # -------------------
        Aobs = np.maximum(0.0, base.values - sig.values)
        wet = (Aobs > wet_thr_db) & np.isfinite(Aobs)

        # If user requested guard based on final wet (rarely needed), you can rerun mask:
        # (kept simple: use wet1 by default; "wet_final" option would require a second recompute.)

        if two_pass and use_drycount_guard and guard_behavior == "zero":
            # wherever pass-2 is unreliable, force Aobs=0 and wet=False
            # (only applies if we actually created base2_ok)
            bad2 = ~g.get("base2_ok", pd.Series(True, index=g.index)).to_numpy(bool)
            Aobs = np.where(bad2, 0.0, Aobs)
            wet  = np.where(bad2, False, wet)

        g["baseline_rsl"] = base.values
        g["A_obs_dB"] = Aobs
        g["wet_rl"] = wet

        out.append(g)

    return pd.concat(out, ignore_index=True)

def rainlink_strict_R(
    dfA: pd.DataFrame,
    R_min: float = 0.0,
    wet_col: str = "wet_rl",
    set_dry_to_zero: bool = True
) -> pd.DataFrame:
    """
    Convert A_obs -> R using:
      - Leijnse (2008) wet-antenna correction
      - ITU(2005) k–α conversion (via pycomlink calc_R_from_A)

    HARD rule (recommended for prime mode):
      if not wet => A_waa = 0, A_rain = 0, gamma = 0, R = 0

    This is the key fix that prevents the "2–5 mm/h carpet" from tiny positive noise.
    """
    parts = []
    for lid, g in dfA.groupby("link_id", sort=False):
        g = g.sort_values("time").copy()

        L_km = float(pd.to_numeric(g["PathLength"], errors="coerce").iloc[0])
        f_GHz = float(pd.to_numeric(g["Frequency"], errors="coerce").iloc[0])
        pol = str(g["pol"].iloc[0]).upper()[0]  # 'H'/'V'

        A_obs = pd.to_numeric(g["A_obs_dB"], errors="coerce").fillna(0.0).values
        wet = g[wet_col].fillna(False).astype(bool).values if wet_col in g.columns else np.isfinite(A_obs)

        # Wet-antenna ONLY where wet (else force 0)
        waa = waa_leijnse_2008_from_A_obs(A_obs=A_obs, f_Hz=f_GHz * 1e9, pol=pol, L_km=L_km)
        if set_dry_to_zero:
            waa = np.where(wet, waa, 0.0)

        A_rain = np.maximum(0.0, A_obs - waa)
        if set_dry_to_zero:
            A_rain = np.where(wet, A_rain, 0.0)

        # Attenuation to rain rate
        R = calc_R_from_A(
            A=A_rain,
            L_km=L_km,
            f_GHz=f_GHz,
            pol=pol,
            a_b_approximation="ITU_2005",
            R_min=R_min
        )
        if set_dry_to_zero:
            R = np.where(wet, R, 0.0)

        g["A_waa_dB"] = waa
        g["A_rain_dB"] = A_rain
        g["gamma_for_R"] = A_rain / max(L_km, 1e-6)
        g["R_mm_per_h"] = R

        parts.append(g)

    return pd.concat(parts, ignore_index=True)


# =============================================================================
# 3. Prepare link-level rainfall for gridding
# =============================================================================

def prepare_inputs_for_gridding(df_rate: pd.DataFrame, ts_15: pd.DataFrame):
    # --- meta (one row per link) ---
    meta = (ts_15.drop_duplicates("link_id")[["link_id","XStart","YStart","XEnd","YEnd"]]
                 .rename(columns={"link_id":"ID"})
                 .copy())
    for c in ["XStart","YStart","XEnd","YEnd"]:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")

    # --- rates (time-indexed, tz-naive UTC) ---
    s5 = df_rate[["link_id","time","R_mm_per_h"]].rename(columns={"link_id":"ID"}).copy()
    t = pd.to_datetime(s5["time"], utc=True, errors="coerce")

    ok = t.notna()
    s5 = s5.loc[ok].drop(columns=["time"])
    s5.index = t.loc[ok].dt.tz_convert("UTC").dt.tz_localize(None)
    s5.index.name = "time"
    s5 = s5.sort_index()

    # sanity
    miss = set(s5["ID"].unique()) - set(meta["ID"].unique())
    if miss:
        raise RuntimeError(f"{len(miss)} IDs in rates missing from meta. Example: {list(miss)[:5]}")

    return s5, meta


# =============================================================================
# 4. Gridding, support masks, confidence layers, and output fields
# =============================================================================

def _km_factors(lat0_deg: float) -> tuple[float, float]:
    lat0r = np.deg2rad(float(lat0_deg))
    kx = 111.0 * max(0.2, np.cos(lat0r))
    ky = 111.0
    return kx, ky

def _lonlat_to_km(lon, lat, lon0, lat0):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    kx, ky = _km_factors(lat0)
    return (lon - lon0) * kx, (lat - lat0) * ky

def _nearest_distance_mask(
    lon_vec, lat_vec, x_obs, y_obs, max_dist_km: float, workers: int | None = None
) -> np.ndarray:
    """True where nearest observed point within max_dist_km (euclidean in km)."""
    XX, YY = np.meshgrid(lon_vec, lat_vec)
    lon0 = float((lon_vec.min() + lon_vec.max()) / 2.0)
    lat0 = float((lat_vec.min() + lat_vec.max()) / 2.0)
    Pkx, Pky = _lonlat_to_km(XX, YY, lon0, lat0)

    if len(x_obs) == 0:
        return np.zeros_like(XX, dtype=bool)

    Qkx, Qky = _lonlat_to_km(x_obs, y_obs, lon0, lat0)
    tree = cKDTree(np.c_[Qkx.ravel(), Qky.ravel()])
    kw = {} if workers is None else {"workers": int(workers)}
    dmin, _ = tree.query(np.c_[Pkx.ravel(), Pky.ravel()], k=1, **kw)
    return (dmin.reshape(XX.shape) <= float(max_dist_km))

def _clean_wet_mask(mask: np.ndarray) -> np.ndarray:
    st = generate_binary_structure(2, 1)  # 4-neighborhood
    return binary_opening(mask, structure=st)

def _smooth_in_mask(Z, mask, kernel_px: int = 3) -> np.ndarray:
    """Box-filter smoothing ONLY where mask is True; no bleed outside."""
    if kernel_px is None or int(kernel_px) <= 1:
        return Z
    Z = np.asarray(Z, float); mask = mask.astype(bool)
    Z0 = np.nan_to_num(Z, nan=0.0); W = mask.astype(float)
    num = uniform_filter(Z0, size=int(kernel_px), mode="nearest")
    den = uniform_filter(W,  size=int(kernel_px), mode="nearest")
    Zs = np.where(den > 0, num / np.maximum(den, _EPS), np.nan)
    return np.where(mask, Zs, np.nan)

def _clean_support_mask(mask, closing_iters=1, fill_holes=True, max_hole_px=25):
    """
    Light cleanup of support mask:
    - binary closing to connect tiny gaps
    - optional filling of only small interior holes
    """
    m = np.asarray(mask, dtype=bool)

    if closing_iters and closing_iters > 0:
        m = binary_closing(m, iterations=int(closing_iters))

    if fill_holes:
        filled = binary_fill_holes(m)
        holes = filled & (~m)

        if holes.any():
            lab, nlab = label(holes)
            keep = np.zeros_like(m, dtype=bool)
            for i in range(1, nlab + 1):
                region = (lab == i)
                if region.sum() <= int(max_hole_px):
                    keep |= region
            m = m | keep

    return m

def _idw_on_grid_km_weighted(
    xkm, ykm, z, Xkm, Ykm, nnear, power, maxdist_km, w_pts=None, workers: int | None = None
):
    """KNN-IDW in km space with optional per-point weights (e.g., sqrt(PathLength))."""
    wet = np.isfinite(z) & (z > 0.0)
    if wet.sum() == 0:
        return np.full_like(Xkm, np.nan, dtype=float)

    pts  = np.c_[xkm[wet], ykm[wet]]
    vals = z[wet].astype(float)
    wp   = np.ones_like(vals) if w_pts is None else np.asarray(w_pts, float)[wet]

    tree = cKDTree(pts)
    k = int(max(1, nnear))
    kw = {"distance_upper_bound": float(maxdist_km)}
    if workers is not None:
        kw["workers"] = int(workers)

    d, idx = tree.query(np.c_[Xkm.ravel(), Ykm.ravel()], k=k, **kw)
    if k == 1:
        d = d[:, None]; idx = idx[:, None]

    valid = np.isfinite(d)
    w_spatial = np.where(valid, 1.0 / np.maximum(d, _EPS) ** float(power), 0.0)

    # pad arrays to handle "no neighbor within radius" (idx == len(vals))
    vals_pad = np.concatenate([vals, [0.0]])
    wp_pad   = np.concatenate([wp,   [0.0]])
    idx = np.where(idx == len(vals), len(vals_pad) - 1, idx)

    num = np.sum(w_spatial * (vals_pad[idx] * wp_pad[idx]), axis=1)
    den = np.sum(w_spatial * wp_pad[idx], axis=1)
    out = np.where(den > 0, num / den, np.nan)
    return out.reshape(Xkm.shape)

def _try_ok_on_grid_km(
    xkm, ykm, z, Xkm, Ykm, *, nlags=10, range_km=25.0, nugget_frac=0.05
):
    """Ordinary Kriging in Euclidean km space with simple exponential variogram."""
    if not _PYKRIGE_AVAILABLE:
        raise RuntimeError("PyKrige not available")

    wet = np.isfinite(z) & (z > 0.0)
    if wet.sum() < 3:
        raise RuntimeError("not enough wet points for OK")

    z_w = z[wet]
    var = float(np.nanvar(z_w))
    if var <= 0:
        raise RuntimeError("zero variance – OK ill-posed")

    nugget = max(1e-6, float(nugget_frac) * var)
    sill   = max(1e-6, var)

    ok = OrdinaryKriging(
        xkm[wet], ykm[wet], z_w,
        variogram_model="exponential",
        variogram_parameters={"nugget": nugget, "sill": sill, "range": float(range_km)},
        nlags=int(nlags), enable_plotting=False, coordinates_type="euclidean",
    )
    xvec = np.unique(Xkm[0, :]); yvec = np.unique(Ykm[:, 0])
    zhat, _ = ok.execute("grid", xvec, yvec)
    return np.asarray(zhat, float)

def _edge_taper_from_mask(
    support_mask,
    *,
    taper_pixels: int = 4,
    min_edge_weight: float = 0.15,
):

    """
    Build a smooth edge-taper weight inside a binary support mask.
    Parameters
    ----------
    support_mask : 2D bool array
        True where CML support exists.
    taper_pixels : int
        Number of grid pixels over which rainfall ramps up from the edge
        toward the interior. With grid_res_deg=0.03, 4 pixels is roughly
        ~12 km north-south.
    min_edge_weight : float
        Minimum multiplier at the support-mask boundary. Use 0.0 for full
        decay to zero at the edge, or 0.10-0.25 to avoid over-suppressing
        edge rainfall.
    Returns
    -------
    taper : 2D float array
        Values in [0, 1]. Outside support is 0. Interior approaches 1.
    """
    m = np.asarray(support_mask, dtype=bool)
    if not np.any(m):
        return np.zeros_like(m, dtype=float)
    if taper_pixels is None or int(taper_pixels) <= 0:
        return m.astype(float)
    # distance_transform_edt gives distance from each True pixel
    # to the nearest False pixel, in pixel units.
    dist_inside = distance_transform_edt(m)
    taper = np.clip(dist_inside / float(taper_pixels), 0.0, 1.0)
    # keep a small value at the mask edge instead of dropping too harshly
    taper = min_edge_weight + (1.0 - min_edge_weight) * taper
    taper = np.where(m, taper, 0.0)
    return taper

def _cosmetic_smooth_rain_for_display(

    Z,
    support_mask,
    *,
    kernel_px: int = 2,
    fill_holes: bool = False,
    drizzle_to_zero: float | None = 0.10,
    # NEW: edge taper controls
    apply_edge_taper: bool = True,
    edge_taper_pixels: int = 4,
    edge_taper_min_weight: float = 0.15,
):

    """
    Cosmetic/display-only smoothing for rainfall maps.
    IMPORTANT:
    - This should NOT replace the scientific rainfall field.
    - It is only meant to reduce harsh circular/blocky visual artifacts.
    - Smoothing is restricted to the support mask.
    - Outside support remains NaN.
    - Optional edge taper softens the hard cliff at support-mask boundaries.
    Parameters
    ----------
    Z : 2D array
        Scientific rainfall field.
    support_mask : 2D bool array
        Valid CML support mask.
    kernel_px : int
        Box smoothing kernel size in pixels.
    fill_holes : bool
        If True, smoothed values can fill supported NaN holes.
        If False, only existing finite rainfall pixels are smoothed.
    drizzle_to_zero : float or None
        Small finite values below this threshold are set to 0.
    apply_edge_taper : bool
        Whether to taper display rainfall near support-mask edges.
    edge_taper_pixels : int
        Width of taper zone in pixels.
    edge_taper_min_weight : float
        Minimum edge multiplier inside support.
    Returns
    -------
    Z_display : 2D array
        Cosmetic rainfall field for plotting/display only.
    """

    Z = np.asarray(Z, dtype=float)
    support_mask = np.asarray(support_mask, dtype=bool)
    if kernel_px is None or int(kernel_px) <= 1:
        Z_display = Z.copy()
    else:
        Zs = _smooth_normalized(
            Z,
            write_mask=support_mask,
            kernel_px=int(kernel_px),
        )

        if fill_holes:
            Z_display = np.where(support_mask, Zs, np.nan)
        else:
            Z_display = np.where(support_mask & np.isfinite(Z), Zs, np.nan)

    # NEW: soften hard support-mask edges for display only
    if apply_edge_taper:
        taper = _edge_taper_from_mask(
            support_mask,
            taper_pixels=int(edge_taper_pixels),
            min_edge_weight=float(edge_taper_min_weight),
        )

        Z_display = np.where(
            support_mask & np.isfinite(Z_display),
            Z_display * taper,
            np.nan,
        )
    if drizzle_to_zero is not None:
        Z_display = np.where(
            np.isfinite(Z_display) & (Z_display < float(drizzle_to_zero)),
            0.0,
            Z_display,
        )
    return Z_display

def _coverage_quality_from_confidence(
    confidence,
    support_mask=None,
    *,
    med_thr: float = 0.50,
    high_thr: float = 0.75,
):
    """
    Classes:
      0 = unsupported
      1 = low confidence
      2 = moderate confidence
      3 = high confidence
    """
    conf = np.asarray(confidence, dtype=float)

    if support_mask is None:
        mask = np.isfinite(conf) & (conf > 0)
    else:
        mask = np.asarray(support_mask, dtype=bool)

    q = np.zeros(conf.shape, dtype=np.int8)
    q[mask & (conf > 0)] = 1
    q[mask & (conf >= med_thr)] = 2
    q[mask & (conf >= high_thr)] = 3

    return q

def grid_rain_15min(
    df_s5,
    df_meta_for_xy,
    *,
    grid_res_deg: float = 0.03,
    domain_pad_deg: float = 0.2,
    drizzle_to_zero: float = 0.1,
    use_ok: bool = True,
    min_pts_ok: int = 20,
    nlags: int = 10,
    ok_range_km: float = 25.0,
    ok_nugget_frac: float = 0.05,
    ok_max_train: int | None = None,
    idw_power: float = 2.0,
    idw_nnear: int = 15,
    idw_maxdist_km: float = 25.0,
    max_dist_km_mask: float = 28.0,
    smooth_kernel_px: int = 3,
    n_jobs: int = 1,
    times_sel=None,
    parallel_backend_name: str = "processes",
    kdtree_workers: int | None = 1,
    collocate_bin_km: float = 2.0,
    use_pathlength_weights: bool = True,
    # NEW fill controls (defaults = NaN)
    outside_support_fill=np.nan,           # value outside wet-footprint support
    interior_no_neighbor_fill=np.nan,      # value for interior “holes” (no neighbors within radius)
    insufficient_training_fill=np.nan,     # whole grid when no wet pts or cannot train
):
    """
    df_s5: index=DatetimeIndex (naive UTC), columns: ["ID","R_mm_per_h", ...]
    df_meta_for_xy: columns: ID, XStart, YStart, XEnd, YEnd

    NOTE: We no longer force NaNs to 0.0. Small finite drizzle values are floored to 0.0,
    but unknowns remain NaN unless you override the *_fill parameters.
    """
    df_s5 = df_s5.copy()
    df_s5["R_mm_per_h"] = pd.to_numeric(df_s5["R_mm_per_h"], errors="coerce")

    # Drizzle → 0.0 (only where finite and positive)
    if drizzle_to_zero is not None:
        v = df_s5["R_mm_per_h"].to_numpy(float)
        mask = np.isfinite(v) & (v > 0.0) & (v < float(drizzle_to_zero))
        v[mask] = 0.0
        df_s5["R_mm_per_h"] = v

    # index hygiene
    idx = df_s5.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("df_s5.index must be a DatetimeIndex")
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df_s5.index = idx
    df_s5.index.name = "time"

    # time selection
    all_times = df_s5.index.unique().sort_values()
    if times_sel is not None:
        req = pd.to_datetime(np.atleast_1d(times_sel))
        req_naive = []
        for tt in req:
            tt = pd.Timestamp(tt)
            if tt.tzinfo is not None:
                tt = tt.tz_convert("UTC").tz_localize(None)
            req_naive.append(tt)
        times = pd.DatetimeIndex(req_naive).intersection(all_times)
        if len(times) == 0:
            raise ValueError("times_sel did not match any times in df_s5.index.")
    else:
        times = all_times

    # meta → midpoints
    m = df_meta_for_xy.dropna(subset=["XStart","YStart","XEnd","YEnd"]).copy()
    for c in ["XStart","YStart","XEnd","YEnd","PathLength"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    m["lon"] = 0.5*(m["XStart"] + m["XEnd"])
    m["lat"] = 0.5*(m["YStart"] + m["YEnd"])

    # grid extent
    lon_min = float(np.nanmin(m["lon"])) - float(domain_pad_deg)
    lon_max = float(np.nanmax(m["lon"])) + float(domain_pad_deg)
    lat_min = float(np.nanmin(m["lat"])) - float(domain_pad_deg)
    lat_max = float(np.nanmax(m["lat"])) + float(domain_pad_deg)

    lon = np.arange(lon_min, lon_max + 1e-9, float(grid_res_deg))
    lat = np.arange(lat_min, lat_max + 1e-9, float(grid_res_deg))

    lon0 = float((lon.min() + lon.max())/2.0)
    lat0 = float((lat.min() + lat.max())/2.0)
    LON, LAT = np.meshgrid(lon, lat)
    Xkm, Ykm = _lonlat_to_km(LON, LAT, lon0, lat0)

    methods_used = []
    n_wet_list  = []

    def _do_time(t):
        g = df_s5.loc[df_s5.index == t, ["ID","R_mm_per_h"]].copy()
        g = g.merge(m[["ID","lon","lat","PathLength"]] if "PathLength" in m.columns else m[["ID","lon","lat"]],
                    on="ID", how="inner")
        g["lon"] = pd.to_numeric(g["lon"], errors="coerce")
        g["lat"] = pd.to_numeric(g["lat"], errors="coerce")
        g["R_mm_per_h"] = pd.to_numeric(g["R_mm_per_h"], errors="coerce")
        g = g.dropna(subset=["lon","lat","R_mm_per_h"])

        x = g["lon"].to_numpy(float); y = g["lat"].to_numpy(float); z = g["R_mm_per_h"].to_numpy(float)
        xkm, ykm = _lonlat_to_km(x, y, lon0, lat0)

        # If no usable points at all → whole grid = insufficient_training_fill
        if len(xkm) == 0:
            Z = np.full_like(LON, insufficient_training_fill, float)
            return Z, "no_info", 0

        # deduplicate near-collocated links (median)
        if len(xkm):
            dfp = pd.DataFrame({"x": xkm, "y": ykm, "z": z})
            bin_km = float(collocate_bin_km)
            bx = np.round(dfp["x"]/bin_km).astype(int)
            by = np.round(dfp["y"]/bin_km).astype(int)
            dfp["_bin"] = list(zip(bx, by))
            gmed = dfp.groupby("_bin", as_index=False).median(numeric_only=True)
            xkm_d = gmed["x"].to_numpy(); ykm_d = gmed["y"].to_numpy(); z_d = gmed["z"].to_numpy()
        else:
            xkm_d = xkm; ykm_d = ykm; z_d = z

        # wet-only footprint (strictly positive)
        wet_pts = np.isfinite(z) & (z > 0.0)
        if wet_pts.sum() < 1:
            # no wet info → whole grid = insufficient_training_fill
            Z = np.full_like(LON, insufficient_training_fill, float)
            return Z, "no_wet", 0

        x_w, y_w = x[wet_pts], y[wet_pts]
        covmask_wet = _nearest_distance_mask(
            lon, lat, x_w, y_w, max_dist_km=max_dist_km_mask, workers=kdtree_workers
        )
        covmask_wet = _clean_wet_mask(covmask_wet)

        # per-point weights for IDW (sqrt(PathLength) if available)
        w_pts = None
        if use_pathlength_weights and "PathLength" in g.columns:
            w_pts = np.sqrt(np.clip(pd.to_numeric(g["PathLength"], errors="coerce").to_numpy(float), 1.0, None))

        used = "idw_knn"
        Z = np.full_like(LON, np.nan, dtype=float)
        n_wet = int((z_d > 0.0).sum())

        if use_ok and _PYKRIGE_AVAILABLE and (n_wet >= int(min_pts_ok)):
            try:
                Z = _try_ok_on_grid_km(
                    xkm_d, ykm_d, z_d, Xkm, Ykm,
                    nlags=int(nlags), range_km=float(ok_range_km), nugget_frac=float(ok_nugget_frac)
                )
                used = "ok_pykrige"
            except Exception:
                Z = _idw_on_grid_km_weighted(
                    xkm_d, ykm_d, z_d, Xkm, Ykm,
                    nnear=int(idw_nnear), power=float(idw_power),
                    maxdist_km=float(idw_maxdist_km), w_pts=w_pts, workers=kdtree_workers
                )
                used = "idw_knn"
        else:
            Z = _idw_on_grid_km_weighted(
                xkm_d, ykm_d, z_d, Xkm, Ykm,
                nnear=int(idw_nnear), power=float(idw_power),
                maxdist_km=float(idw_maxdist_km), w_pts=w_pts, workers=kdtree_workers
            )
            used = "idw_knn"

        # Outside support → fill (default NaN)
        if np.isnan(outside_support_fill):
            Z[~covmask_wet] = np.nan
        else:
            Z[~covmask_wet] = float(outside_support_fill)

        # Smooth ONLY where we have values (no bleed)
        valid_mask = np.isfinite(Z)
        Z = _smooth_in_mask(Z, valid_mask, kernel_px=smooth_kernel_px)

        # Interior “holes” (inside support but still NaN) → fill (default NaN)
        inside_holes = covmask_wet & ~np.isfinite(Z)
        if np.isnan(interior_no_neighbor_fill):
            # leave as NaN
            pass
        else:
            Z[inside_holes] = float(interior_no_neighbor_fill)

        return Z, used, n_wet

    backend = "loky" if parallel_backend_name == "processes" else "threading"
    with parallel_backend(backend):
        res = Parallel(n_jobs=int(n_jobs))(delayed(_do_time)(t) for t in times)

    grids    = [r[0] for r in res]
    methods  = [r[1] for r in res]
    n_wet_l  = [r[2] for r in res]

    da = xr.DataArray(
        np.stack(grids),
        coords={"time": times.values.astype("datetime64[ns]"), "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="R_mm_per_h",
    )
    da.attrs.update(dict(
        units="mm h-1",
        method="OK→IDW (gated) + cleaned wet-footprint + box-mean smooth",
        grid_res_deg=float(grid_res_deg),
    ))

    counts = {mth: methods.count(mth) for mth in set(methods)}
    first5 = []
    for i in range(min(5, len(times))):
        sl = grids[i]
        first5.append(dict(method=methods[i], n_wet=int(n_wet_l[i]),
                           min=float(np.nanmin(sl)), max=float(np.nanmax(sl))))

    diag = dict(
        n_times=int(len(times)),
        counts=counts,
        first5=first5,
        ok_times=[str(t) for t, mth in zip(times, methods) if mth == "ok_pykrige"],
        pykrige_available=_PYKRIGE_AVAILABLE,
        config=dict(
            grid_res_deg=grid_res_deg, domain_pad_deg=domain_pad_deg,
            drizzle_to_zero=drizzle_to_zero, use_ok=bool(use_ok),
            min_pts_ok=min_pts_ok, nlags=nlags, ok_range_km=ok_range_km,
            ok_nugget_frac=ok_nugget_frac, ok_max_train=ok_max_train,
            idw_power=idw_power, idw_nnear=idw_nnear, idw_maxdist_km=idw_maxdist_km,
            max_dist_km_mask=max_dist_km_mask, smooth_kernel_px=smooth_kernel_px,
            n_jobs=n_jobs, parallel_backend_name=parallel_backend_name,
            kdtree_workers=kdtree_workers, collocate_bin_km=collocate_bin_km,
            use_pathlength_weights=use_pathlength_weights,
            outside_support_fill=outside_support_fill,
            interior_no_neighbor_fill=interior_no_neighbor_fill,
            insufficient_training_fill=insufficient_training_fill,
        ),
        times_used=[str(t) for t in times[:10]],
    )
    return da, diag

def grid_rain_at_time(df_s5, df_meta_for_xy, t, **kwargs):
    """Convenience wrapper to grid a single timestamp."""
    return grid_rain_15min(df_s5=df_s5, df_meta_for_xy=df_meta_for_xy, times_sel=[t], **kwargs)

def _midpoints(dfm):
    m = dfm.copy()
    for c in ["XStart","YStart","XEnd","YEnd"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["lon_mid"] = (m["XStart"] + m["XEnd"]) / 2.0
    m["lat_mid"] = (m["YStart"] + m["YEnd"]) / 2.0
    return m

def _grid_from_meta(meta_xy, grid_res_deg=0.03, pad_deg=0.20):
    mm = _midpoints(meta_xy)
    x0, x1 = float(mm["lon_mid"].min()), float(mm["lon_mid"].max())
    y0, y1 = float(mm["lat_mid"].min()), float(mm["lat_mid"].max())
    xv = np.arange(x0 - pad_deg, x1 + pad_deg + grid_res_deg, grid_res_deg)
    yv = np.arange(y0 - pad_deg, y1 + pad_deg + grid_res_deg, grid_res_deg)
    lon, lat = np.meshgrid(xv, yv)
    return lon, lat, xv, yv

def _haversine_distance_km(lon1, lat1, lon2, lat2):
    """
    Great-circle distance between lon/lat points in km.
    Inputs can be scalars or arrays.
    """
    lon1 = np.asarray(lon1, dtype=float)
    lat1 = np.asarray(lat1, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)

    rlon1 = np.deg2rad(lon1)
    rlat1 = np.deg2rad(lat1)
    rlon2 = np.deg2rad(lon2)
    rlat2 = np.deg2rad(lat2)

    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return _EARTH_R_KM * c

def _n_support_points_from_length(
    length_km,
    *,
    spacing_km=2.0,
    min_points=3,
    max_points=7,
):
    """
    Choose number of along-link support points from link length.

    Examples with defaults:
      length 1 km  -> 3 points
      length 8 km  -> 5 points
      length 15 km -> 7 points

    The returned points include start and end.
    """
    if not np.isfinite(length_km) or length_km <= 0:
        return int(min_points)

    n = int(np.ceil(float(length_km) / float(spacing_km))) + 1
    n = max(int(min_points), n)
    n = min(int(max_points), n)
    return int(n)

def _expand_links_to_path_support_points(
    pts,
    *,
    support_geometry="link_path",
    n_support_points_per_link=5,
    support_point_spacing_km=2.0,
    min_support_points_per_link=3,
    max_support_points_per_link=7,
    use_length_conditioned_points=True,
):
    """
    Expand link observations into geometry support points.

    Parameters
    ----------
    pts : pd.DataFrame
        Must contain:
          ID, R_mm_per_h, XStart, YStart, XEnd, YEnd, lon_mid, lat_mid

    support_geometry : {"midpoint", "start_mid_end", "link_path"}
        midpoint:
            One support point per link at midpoint.
        start_mid_end:
            Three support points per link: start, midpoint, end.
        link_path:
            Multiple support points along the link path.

    Returns
    -------
    out : pd.DataFrame
        Columns:
          ID, R_mm_per_h, lon_support, lat_support, support_frac

    Notes
    -----
    All support points from the same link carry the same R_mm_per_h.
    This does not imply rainfall varies along the link; it only improves support geometry.
    """
    if support_geometry not in ("midpoint", "start_mid_end", "link_path"):
        raise ValueError(
            "support_geometry must be one of: 'midpoint', 'start_mid_end', 'link_path'"
        )

    rows = []

    for _, row in pts.iterrows():
        lid = str(row["ID"])
        r = pd.to_numeric(row["R_mm_per_h"], errors="coerce")

        x0 = pd.to_numeric(row["XStart"], errors="coerce")
        y0 = pd.to_numeric(row["YStart"], errors="coerce")
        x1 = pd.to_numeric(row["XEnd"], errors="coerce")
        y1 = pd.to_numeric(row["YEnd"], errors="coerce")
        xm = pd.to_numeric(row["lon_mid"], errors="coerce")
        ym = pd.to_numeric(row["lat_mid"], errors="coerce")

        if not (
            np.isfinite(r)
            and np.isfinite(x0)
            and np.isfinite(y0)
            and np.isfinite(x1)
            and np.isfinite(y1)
            and np.isfinite(xm)
            and np.isfinite(ym)
        ):
            continue

        if support_geometry == "midpoint":
            fracs = np.array([0.5], dtype=float)

        elif support_geometry == "start_mid_end":
            fracs = np.array([0.0, 0.5, 1.0], dtype=float)

        else:
            # support_geometry == "link_path"
            if use_length_conditioned_points:
                L_km = _haversine_distance_km(x0, y0, x1, y1)
                npts = _n_support_points_from_length(
                    L_km,
                    spacing_km=support_point_spacing_km,
                    min_points=min_support_points_per_link,
                    max_points=max_support_points_per_link,
                )
            else:
                npts = int(n_support_points_per_link)
                npts = max(2, npts)

            fracs = np.linspace(0.0, 1.0, int(npts))

        for f in fracs:
            lon_s = x0 + float(f) * (x1 - x0)
            lat_s = y0 + float(f) * (y1 - y0)

            rows.append(
                {
                    "ID": lid,
                    "R_mm_per_h": float(r),
                    "lon_support": float(lon_s),
                    "lat_support": float(lat_s),
                    "support_frac": float(f),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["ID", "R_mm_per_h", "lon_support", "lat_support", "support_frac"]
        )

    return pd.DataFrame(rows)

def _kth_distance_km_haversine(lon_grid, lat_grid, lon_pts, lat_pts, k=1):
    """
    Return distance [km] to the k-th nearest point for each grid cell.
    If insufficient points, returns +inf everywhere.
    """
    if (not _SKLEARN_AVAILABLE) or (len(lon_pts) < max(1, int(k))):
        return np.full(lon_grid.shape, np.inf, float)

    tree = BallTree(np.deg2rad(np.c_[lat_pts, lon_pts]), metric="haversine")
    pts = np.deg2rad(np.c_[lat_grid.ravel(), lon_grid.ravel()])
    d_rad, _ = tree.query(pts, k=int(k))
    d_km = d_rad[:, -1] * _EARTH_R_KM
    return d_km.reshape(lon_grid.shape)

def _support_mask_wet_dry_haversine(
    lon_grid, lat_grid,
    lon_wet, lat_wet,
    lon_dry, lat_dry,
    *,
    wet_k=2,
    wet_radius_km=25.0,
    dry_radius_km=12.0,
    dry_deactivate_if_wet_k_within=1,
):
    """
    Support mask that:
      - requires wet support nearby
      - suppresses rainfall where dry links are very close,
        unless a strong wet neighbor is also very close

    Returns
    -------
    mask : bool array
    """
    wet_d = _kth_distance_km_haversine(
        lon_grid, lat_grid, lon_wet, lat_wet, k=max(1, int(wet_k))
    )
    wet_mask = wet_d <= float(wet_radius_km)

    if len(lon_dry) == 0:
        return wet_mask

    dry_d = _kth_distance_km_haversine(
        lon_grid, lat_grid, lon_dry, lat_dry, k=1
    )

    if len(lon_wet) >= max(1, int(dry_deactivate_if_wet_k_within)):
        wet1_d = _kth_distance_km_haversine(
            lon_grid, lat_grid, lon_wet, lat_wet, k=max(1, int(dry_deactivate_if_wet_k_within))
        )
        strong_wet_near = wet1_d <= max(6.0, 0.35 * float(wet_radius_km))
    else:
        strong_wet_near = np.zeros_like(wet_mask, dtype=bool)

    dry_suppression = (dry_d <= float(dry_radius_km)) & (~strong_wet_near)
    mask = wet_mask & (~dry_suppression)
    return mask

def _support_confidence_from_wet_dry_v2(
    lon_grid,
    lat_grid,
    lon_wet,
    lat_wet,
    lon_dry,
    lat_dry,
    *,
    wet_k=2,
    wet_radius_km=25.0,
    dry_radius_km=12.0,
    dry_penalty_weight=0.50,
    conf_power=1.2,
):
    """
    Support/confidence field in [0, 1].

    Logic:
    - Wet confidence is based on distance to the k-th nearest wet support point.
      This rewards local wet-link/path clustering, not just a single point.
    - Dry links gently reduce confidence where they are very close.
    - This is a support/geometry indicator, not formal uncertainty.
    """
    wet_d = _kth_distance_km_haversine(
        lon_grid,
        lat_grid,
        lon_wet,
        lat_wet,
        k=max(1, int(wet_k)),
    )

    wet_score = np.clip(
        1.0 - wet_d / max(float(wet_radius_km), 1e-6),
        0.0,
        1.0,
    )

    if len(lon_dry) == 0:
        conf = wet_score
    else:
        dry_d = _kth_distance_km_haversine(
            lon_grid,
            lat_grid,
            lon_dry,
            lat_dry,
            k=1,
        )

        dry_score = np.clip(
            1.0 - dry_d / max(float(dry_radius_km), 1e-6),
            0.0,
            1.0,
        )

        dry_factor = 1.0 - float(dry_penalty_weight) * dry_score
        dry_factor = np.clip(dry_factor, 0.0, 1.0)

        conf = wet_score * dry_factor

    conf = np.clip(conf, 0.0, 1.0)

    if conf_power is not None:
        conf = conf ** float(conf_power)

    return conf

def _grid_from_meta_or_fixed(
    meta_xy,
    grid_res_deg=0.03,
    pad_deg=0.20,
    fixed_extent=None,
):
    """
    Build grid either from:
      - dynamic link footprint in metadata, or
      - a fixed extent (lon_min, lon_max, lat_min, lat_max)

    Returns
    -------
    lon2d, lat2d, xv, yv
    """
    if fixed_extent is not None:
        lon_min, lon_max, lat_min, lat_max = fixed_extent
        xv = np.arange(float(lon_min), float(lon_max) + float(grid_res_deg), float(grid_res_deg))
        yv = np.arange(float(lat_min), float(lat_max) + float(grid_res_deg), float(grid_res_deg))
        lon, lat = np.meshgrid(xv, yv)
        return lon, lat, xv, yv

    mm = _midpoints(meta_xy)
    x0, x1 = float(mm["lon_mid"].min()), float(mm["lon_mid"].max())
    y0, y1 = float(mm["lat_mid"].min()), float(mm["lat_mid"].max())
    xv = np.arange(x0 - pad_deg, x1 + pad_deg + grid_res_deg, grid_res_deg)
    yv = np.arange(y0 - pad_deg, y1 + pad_deg + grid_res_deg, grid_res_deg)
    lon, lat = np.meshgrid(xv, yv)
    return lon, lat, xv, yv

def _support_mask_wet_haversine(lon_grid, lat_grid, lon_wet, lat_wet, k=2, radius_km=25.0):
    """True where the k-th nearest WET link is within radius_km (haversine)."""
    if not _SKLEARN_AVAILABLE or len(lon_wet) < k:
        return np.zeros_like(lon_grid, dtype=bool)
    tree = BallTree(np.deg2rad(np.c_[lat_wet, lon_wet]), metric="haversine")
    pts  = np.deg2rad(np.c_[lat_grid.ravel(), lon_grid.ravel()])
    d_rad, _ = tree.query(pts, k=int(k))
    d_km = d_rad[:, -1] * _EARTH_R_KM
    return (d_km <= float(radius_km)).reshape(lon_grid.shape)

def _estimate_variogram_params(values, range_km=25.0, nugget_frac=0.4):
    """Simple robust parameters; sill from variance of training values."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    sill = float(np.nanvar(v)) if v.size else 1e-6
    sill = max(sill, 1e-6)
    return {"sill": sill, "range": float(range_km), "nugget": float(nugget_frac) * sill}

def grid_rain_15min_rainlink_ok(
    df_s5,                    # index: time (naive UTC); cols: ID, R_mm_per_h
    df_meta_for_xy,           # cols: ID, XStart, YStart, XEnd, YEnd
    *,
    grid_res_deg=0.03, domain_pad_deg=0.20,
    wet_thr=0.8, dry_thr=0.05,               # define wet/dry points for training
    ok_model="exponential", ok_range_km=25.0, ok_nugget_frac=0.4,
    min_pts_ok=12,                            # need at least this many (wet+dry0) to run OK
    support_k=2, support_radius_km=25.0,     # mask requires ≥k wet links within radius
    drizzle_to_zero=0.10,                     # floor tiny positives to 0.0 (doesn't touch NaNs)
    times_sel=None,    
    # NEW:
    n_jobs: int = 1,
    parallel_backend_name: str = "processes",
    # NEW controls for “no information”
    outside_support_fill=np.nan,             # what to put OUTSIDE strict wet support (np.nan or 0.0)
    insufficient_training_fill=np.nan,
    smooth_kernel_px: int | None = None,
    smooth_fill_holes: bool = True,
):
    """
    RAINLINK-like: OK trained on wet values + dry zeros; apply strict wet support mask.

    outside_support_fill:
        Value for cells outside the wet-link support mask (default NaN).
    insufficient_training_fill:
        Value for the entire grid when there were no usable points or too few to train (default NaN).
    """
    if not _PYKRIGE_AVAILABLE:
        raise RuntimeError("PyKrige not available for RainLINK-style gridding.")

    # 1) grid axes
    LON, LAT, xv, yv = _grid_from_meta(df_meta_for_xy, grid_res_deg, domain_pad_deg)

    # 2) outputs
    all_times = pd.Index(sorted(df_s5.index.unique()))
    times = all_times if times_sel is None else pd.Index(pd.to_datetime(times_sel))
    out = np.full((len(times), LAT.shape[0], LAT.shape[1]), np.nan, float)

    # precompute midpoints by ID
    mid = _midpoints(df_meta_for_xy[["ID","XStart","YStart","XEnd","YEnd"]].drop_duplicates("ID"))
    id2xy = mid.set_index("ID")[["lon_mid","lat_mid"]]

    # diagnostics
    diag = {"counts": {"ok": 0, "fallback": 0}, "wet_counts": [], "train_counts": []}

    def _do_one(it, t):
        # slice points and attach coords
        try:
            pts = (df_s5.loc[t].merge(id2xy, on="ID", how="inner"))
        except KeyError:
            # no rows at this timestamp → whole grid is "no info"
            Z = np.full_like(LON, insufficient_training_fill, float)
            return it, Z, 0, 0, False

        vals = pd.to_numeric(pts["R_mm_per_h"], errors="coerce").values
        lon  = pd.to_numeric(pts["lon_mid"], errors="coerce").values
        lat  = pd.to_numeric(pts["lat_mid"], errors="coerce").values

        good = np.isfinite(vals) & np.isfinite(lon) & np.isfinite(lat)
        if not good.any():
            Z = np.full_like(LON, insufficient_training_fill, float)
            return it, Z, 0, 0, False

        vals, lon, lat = vals[good], lon[good], lat[good]

        # classify wet/dry for training
        wet = vals >= float(wet_thr)
        dry = vals <= float(dry_thr)

        lon_wet, lat_wet = lon[wet], lat[wet]
        # training set = wet values + dry zeros
        lon_tr = np.concatenate([lon[wet], lon[dry]])
        lat_tr = np.concatenate([lat[wet], lat[dry]])
        val_tr = np.concatenate([vals[wet], np.zeros(np.count_nonzero(dry), float)])        

        if len(val_tr) < max(3, int(min_pts_ok)):
            Z = np.full_like(LON, insufficient_training_fill, float)
            return it, Z, int(wet.sum()), int(len(val_tr)), False

        # OK on geographic coords
        vparam = _estimate_variogram_params(val_tr, range_km=ok_range_km, nugget_frac=ok_nugget_frac)
        try:
            OK = OrdinaryKriging(
                lon_tr, lat_tr, val_tr,
                variogram_model=ok_model,
                variogram_parameters=vparam,
                coordinates_type="geographic",
                enable_plotting=False, verbose=False
            )
            Z, _ = OK.execute("grid", xv, yv)   # (ny, nx)
            Z = np.asarray(Z, float)

            # strict support mask based on WET links only
            if _SKLEARN_AVAILABLE and len(lon_wet) >= max(1, int(support_k)):
                mask = _support_mask_wet_haversine(LON, LAT, lon_wet, lat_wet,
                                                   k=int(support_k), radius_km=float(support_radius_km))
            else:
                # fallback: nearest-distance footprint from wet points in degrees (~approx km)
                mask = _nearest_distance_mask(xv, yv, lon_wet, lat_wet,
                                              max_dist_km=support_radius_km, workers=1)

            # Outside support → fill (NaN by default)
            if np.isnan(outside_support_fill):
                Z = np.where(mask, Z, np.nan)
            else:
                Z = np.where(mask, Z, float(outside_support_fill))
            
            # --- optional smoothing inside strict support (gap-closing) ---
            if smooth_kernel_px is not None and int(smooth_kernel_px) > 1:
                if smooth_fill_holes:
                    # fill holes: overwrite ALL pixels inside support with smoothed field
                    Zs = _smooth_normalized(Z, write_mask=mask, kernel_px=int(smooth_kernel_px))
                    Z = np.where(mask, Zs, Z)  # outside already handled
                else:
                    # no hole fill: smooth only where Z is already finite
                    Zs = _smooth_normalized(Z, write_mask=mask, kernel_px=int(smooth_kernel_px))
                    Z = np.where(mask & np.isfinite(Z), Zs, Z)

            # Drizzle floor to zero (does not touch NaNs)
            if drizzle_to_zero is not None:
                Z = np.where(np.isfinite(Z) & (Z < float(drizzle_to_zero)), 0.0, Z)

            return it, Z, int(wet.sum()), int(len(val_tr)), True
        except Exception:
            Z = np.full_like(LON, insufficient_training_fill, float)
            return it, Z, int(wet.sum()), int(len(val_tr)), False

    backend = "loky" if parallel_backend_name == "processes" else "threading"
    with parallel_backend(backend):
        results = Parallel(n_jobs=int(n_jobs))(
            delayed(_do_one)(i, t) for i, t in enumerate(times)
        )

    for it, Z, nwet, ntrain, ok_flag in results:
        out[it, :, :] = Z
        diag["wet_counts"].append(nwet)
        diag["train_counts"].append(ntrain)
        diag["counts"]["ok" if ok_flag else "fallback"] += 1

    da = xr.DataArray(
        out, coords={"time": times.tz_localize(None), "lat": yv, "lon": xv},
        dims=("time", "lat", "lon"), name="R_mm_per_h", attrs={"units": "mm h-1"}
    )
    return da, diag

# Main operational gridding routine used in the recent Ghana/Rainboo workflow.
# OK is used when there are enough local wet links; otherwise pixels are left
# missing or supported through conservative fallback logic to avoid overextending
# sparse CML information.
def grid_rain_15min_rainlink_ok_full_ghana(
    df_s5,
    df_meta_for_xy,
    *,
    grid_res_deg=0.03,
    domain_pad_deg=0.20,
    fixed_extent=(-3.5, 1.5, 4.5, 11.5),   # Ghana AOI: lon_min, lon_max, lat_min, lat_max

    # wet/dry classification
    wet_thr=0.8,
    dry_thr=0.05,

    # OK / variogram
    ok_model="exponential",
    ok_range_km=25.0,
    ok_nugget_frac=0.4,
    min_pts_ok=12,

    # support mask
    support_k=2,
    support_radius_km=20.0,
    dry_radius_km=10.0,

    # NEW: support/confidence geometry
    support_geometry: str = "link_path",             # "midpoint" | "start_mid_end" | "link_path"
    n_support_points_per_link: int = 5,
    support_point_spacing_km: float = 2.0,
    min_support_points_per_link: int = 3,
    max_support_points_per_link: int = 7,
    use_length_conditioned_support_points: bool = True,

    # support/confidence controls
    use_dry_constraint=True,
    use_soft_confidence=True,
    confidence_floor=0.15,
    confidence_power=1.5,
    confidence_dry_penalty_weight=0.50,

    # rainfall handling
    drizzle_to_zero=0.10,
    times_sel=None,
    n_jobs: int = 1,
    parallel_backend_name: str = "processes",
    outside_support_fill=np.nan,
    insufficient_training_fill=np.nan,

    # scientific-field smoothing controls
    smooth_kernel_px: int | None = 1,
    smooth_fill_holes: bool = False,

    # support mask cleanup controls
    clean_support: bool = True,
    support_closing_iters: int = 1,
    support_fill_holes: bool = True,
    support_max_hole_px: int = 20,

    # display/cosmetic layer controls
    make_display_field: bool = True,
    display_smooth_kernel_px: int | None = 2,
    display_smooth_fill_holes: bool = False,

    # display edge taper controls
    apply_display_edge_taper: bool = True,
    display_edge_taper_pixels: int = 4,
    display_edge_taper_min_weight: float = 0.15,

    # coverage quality controls
    make_coverage_quality: bool = True,
    coverage_quality_med_thr: float = 0.50,
    coverage_quality_high_thr: float = 0.75,

    # return type
    return_dataset: bool = True,
):
    """
    RainLINK-like OK gridding on a fixed Ghana-wide grid, with optional
    link-path support geometry and support/confidence outputs.

    Returns
    -------
    If return_dataset=True:
        ds : xr.Dataset
            Contains:
              - R_mm_per_h(time, lat, lon)
              - R_display_mm_per_h(time, lat, lon)
              - cml_support_confidence(time, lat, lon)
              - cml_support_mask(time, lat, lon)
              - cml_coverage_quality(time, lat, lon)
        diag : dict

    If return_dataset=False:
        R_da : xr.DataArray
            Scientific rainfall field only, for backward compatibility.
        diag : dict

    Notes
    -----
    R_mm_per_h:
        Scientific gridded rainfall field.

    R_display_mm_per_h:
        Cosmetic/display-only rainfall field. Use for map visualization,
        not scientific validation or quantitative blending.

    cml_support_confidence:
        Practical support/confidence indicator in [0, 1], based on local
        CML network geometry and wet/dry consistency. This is not a formal
        probabilistic uncertainty estimate.

    cml_support_mask:
        Binary supported/unsupported CML rainfall mask.

    cml_coverage_quality:
        Categorical support/coverage quality layer derived from confidence
        and support mask. This is intended to help downstream users decide
        how strongly to use or blend the CML rainfall product.

    support_geometry:
        "midpoint":
            support/confidence uses one point per CML link, at midpoint.
        "start_mid_end":
            support/confidence uses start, midpoint, and end of each link.
        "link_path":
            support/confidence uses several points along each link path.

    Important implementation choice:
        OK training remains based on link midpoints only.
        Along-link points are used only for support/confidence geometry.
        This avoids artificially duplicating rainfall values from long links
        in the kriging training.
    """

    if not _PYKRIGE_AVAILABLE:
        raise RuntimeError("PyKrige not available for RainLINK-style gridding.")

    if support_geometry not in ("midpoint", "start_mid_end", "link_path"):
        raise ValueError(
            "support_geometry must be one of: 'midpoint', 'start_mid_end', 'link_path'"
        )

    # ------------------------------------------------------------
    # 1) fixed Ghana grid
    # ------------------------------------------------------------
    LON, LAT, xv, yv = _grid_from_meta_or_fixed(
        df_meta_for_xy,
        grid_res_deg=grid_res_deg,
        pad_deg=domain_pad_deg,
        fixed_extent=fixed_extent,
    )

    ny, nx = LAT.shape

    # ------------------------------------------------------------
    # 2) time axis
    # ------------------------------------------------------------
    all_times = pd.Index(sorted(df_s5.index.unique()))

    if times_sel is None:
        times = all_times
    else:
        times_raw = pd.Index(pd.to_datetime(times_sel))
        fixed_times = []

        for tt in times_raw:
            tt = pd.Timestamp(tt)
            if tt.tzinfo is not None:
                tt = tt.tz_convert("UTC").tz_localize(None)
            else:
                tt = tt.tz_localize(None)
            fixed_times.append(tt)

        times = pd.Index(fixed_times)

    # ------------------------------------------------------------
    # 3) output arrays
    # ------------------------------------------------------------
    rain_out = np.full((len(times), ny, nx), np.nan, dtype=float)
    display_out = np.full((len(times), ny, nx), np.nan, dtype=float)
    conf_out = np.full((len(times), ny, nx), 0.0, dtype=float)
    mask_out = np.zeros((len(times), ny, nx), dtype=np.int8)

    # NEW: categorical coverage/support quality output
    coverage_quality_out = np.zeros((len(times), ny, nx), dtype=np.int8)

    # ------------------------------------------------------------
    # 4) precompute link geometry
    # ------------------------------------------------------------
    mid = _midpoints(
        df_meta_for_xy[["ID", "XStart", "YStart", "XEnd", "YEnd"]]
        .drop_duplicates("ID")
        .copy()
    )

    # Keep full geometry because link-path support needs start/end points.
    id2geom = mid.set_index("ID")[["XStart", "YStart", "XEnd", "YEnd", "lon_mid", "lat_mid"]]

    # ------------------------------------------------------------
    # 5) diagnostics container
    # ------------------------------------------------------------
    diag = {
        "counts": {"ok": 0, "failed_or_skipped": 0},
        "wet_counts": [],
        "dry_counts": [],
        "train_counts": [],
        "support_point_counts": [],
        "supported_pixel_counts": [],
        "mean_confidence_supported": [],
        "max_confidence": [],
        "coverage_quality_counts": [],
        "fixed_extent": {
            "lon_min": float(fixed_extent[0]),
            "lon_max": float(fixed_extent[1]),
            "lat_min": float(fixed_extent[2]),
            "lat_max": float(fixed_extent[3]),
        },
        "grid_shape": (int(ny), int(nx)),
        "grid_res_deg": float(grid_res_deg),
        "config": {
            "wet_thr": float(wet_thr),
            "dry_thr": float(dry_thr),
            "ok_model": ok_model,
            "ok_range_km": float(ok_range_km),
            "ok_nugget_frac": float(ok_nugget_frac),
            "min_pts_ok": int(min_pts_ok),

            "support_k": int(support_k),
            "support_radius_km": float(support_radius_km),
            "dry_radius_km": float(dry_radius_km),

            "support_geometry": support_geometry,
            "n_support_points_per_link": int(n_support_points_per_link),
            "support_point_spacing_km": float(support_point_spacing_km),
            "min_support_points_per_link": int(min_support_points_per_link),
            "max_support_points_per_link": int(max_support_points_per_link),
            "use_length_conditioned_support_points": bool(use_length_conditioned_support_points),

            "use_dry_constraint": bool(use_dry_constraint),
            "use_soft_confidence": bool(use_soft_confidence),
            "confidence_floor": float(confidence_floor),
            "confidence_power": float(confidence_power),
            "confidence_dry_penalty_weight": float(confidence_dry_penalty_weight),

            "drizzle_to_zero": drizzle_to_zero,
            "outside_support_fill": outside_support_fill,
            "insufficient_training_fill": insufficient_training_fill,

            "smooth_kernel_px": smooth_kernel_px,
            "smooth_fill_holes": bool(smooth_fill_holes),

            "clean_support": bool(clean_support),
            "support_closing_iters": int(support_closing_iters),
            "support_fill_holes": bool(support_fill_holes),
            "support_max_hole_px": int(support_max_hole_px),

            "make_display_field": bool(make_display_field),
            "display_smooth_kernel_px": display_smooth_kernel_px,
            "display_smooth_fill_holes": bool(display_smooth_fill_holes),

            # NEW: display-only edge taper settings
            "apply_display_edge_taper": bool(apply_display_edge_taper),
            "display_edge_taper_pixels": int(display_edge_taper_pixels),
            "display_edge_taper_min_weight": float(display_edge_taper_min_weight),

            # NEW: categorical coverage quality settings
            "make_coverage_quality": bool(make_coverage_quality),
            "coverage_quality_med_thr": float(coverage_quality_med_thr),
            "coverage_quality_high_thr": float(coverage_quality_high_thr),
        },
    }

    # ------------------------------------------------------------
    # 6) per-time gridding worker
    # ------------------------------------------------------------
    def _do_one(it, t):
        # default outputs for failed/no-info cases
        Z_fail = np.full_like(LON, insufficient_training_fill, dtype=float)
        M_fail = np.zeros_like(LON, dtype=bool)
        C_fail = np.zeros_like(LON, dtype=float)
        D_fail = np.full_like(LON, np.nan, dtype=float)
        Q_fail = np.zeros_like(LON, dtype=np.int8)

        # --------------------------------------------------------
        # A) slice points and attach geometry
        # --------------------------------------------------------
        try:
            pts_raw = df_s5.loc[t]
        except KeyError:
            return it, Z_fail, D_fail, C_fail, M_fail, Q_fail, 0, 0, 0, 0, False

        if isinstance(pts_raw, pd.Series):
            pts_raw = pts_raw.to_frame().T

        pts = pts_raw.merge(id2geom, on="ID", how="inner")

        if pts.empty:
            return it, Z_fail, D_fail, C_fail, M_fail, Q_fail, 0, 0, 0, 0, False

        # --------------------------------------------------------
        # B) midpoint observations for OK training
        # --------------------------------------------------------
        vals_mid = pd.to_numeric(pts["R_mm_per_h"], errors="coerce").to_numpy(float)
        lon_mid = pd.to_numeric(pts["lon_mid"], errors="coerce").to_numpy(float)
        lat_mid = pd.to_numeric(pts["lat_mid"], errors="coerce").to_numpy(float)

        good_mid = np.isfinite(vals_mid) & np.isfinite(lon_mid) & np.isfinite(lat_mid)

        if not good_mid.any():
            return it, Z_fail, D_fail, C_fail, M_fail, Q_fail, 0, 0, 0, 0, False

        pts_good = pts.loc[good_mid].copy()

        vals_mid = vals_mid[good_mid]
        lon_mid = lon_mid[good_mid]
        lat_mid = lat_mid[good_mid]

        # Wet/dry classification at link/midpoint level.
        wet_mid = vals_mid >= float(wet_thr)
        dry_mid = vals_mid <= float(dry_thr)

        nwet = int(np.count_nonzero(wet_mid))
        ndry = int(np.count_nonzero(dry_mid))

        if nwet < 2:
            return it, Z_fail, D_fail, C_fail, M_fail, Q_fail, nwet, ndry, 0, 0, False

        # --------------------------------------------------------
        # C) expanded along-link support points
        # --------------------------------------------------------
        support_pts = _expand_links_to_path_support_points(
            pts_good,
            support_geometry=support_geometry,
            n_support_points_per_link=n_support_points_per_link,
            support_point_spacing_km=support_point_spacing_km,
            min_support_points_per_link=min_support_points_per_link,
            max_support_points_per_link=max_support_points_per_link,
            use_length_conditioned_points=use_length_conditioned_support_points,
        )

        if support_pts.empty:
            return it, Z_fail, D_fail, C_fail, M_fail, Q_fail, nwet, ndry, 0, 0, False

        svals = pd.to_numeric(support_pts["R_mm_per_h"], errors="coerce").to_numpy(float)
        slon = pd.to_numeric(support_pts["lon_support"], errors="coerce").to_numpy(float)
        slat = pd.to_numeric(support_pts["lat_support"], errors="coerce").to_numpy(float)

        good_s = np.isfinite(svals) & np.isfinite(slon) & np.isfinite(slat)

        svals = svals[good_s]
        slon = slon[good_s]
        slat = slat[good_s]

        if svals.size == 0:
            return it, Z_fail, D_fail, C_fail, M_fail, Q_fail, nwet, ndry, 0, 0, False

        wet_s = svals >= float(wet_thr)
        dry_s = svals <= float(dry_thr)

        lon_wet, lat_wet = slon[wet_s], slat[wet_s]
        lon_dry, lat_dry = slon[dry_s], slat[dry_s]

        nsupport = int(len(svals))

        # --------------------------------------------------------
        # D) OK training set
        # --------------------------------------------------------
        # Keep OK training at link midpoint level.
        # Along-link points are used only for support/confidence geometry.
        lon_tr = np.concatenate([lon_mid[wet_mid], lon_mid[dry_mid]])
        lat_tr = np.concatenate([lat_mid[wet_mid], lat_mid[dry_mid]])
        val_tr = np.concatenate([
            vals_mid[wet_mid],
            np.zeros(np.count_nonzero(dry_mid), dtype=float),
        ])

        ntrain = int(len(val_tr))

        if ntrain < max(3, int(min_pts_ok)):
            return it, Z_fail, D_fail, C_fail, M_fail, Q_fail, nwet, ndry, ntrain, nsupport, False

        # --------------------------------------------------------
        # E) Ordinary Kriging
        # --------------------------------------------------------
        vparam = _estimate_variogram_params(
            val_tr,
            range_km=ok_range_km,
            nugget_frac=ok_nugget_frac,
        )

        try:
            OK = OrdinaryKriging(
                lon_tr,
                lat_tr,
                val_tr,
                variogram_model=ok_model,
                variogram_parameters=vparam,
                coordinates_type="geographic",
                enable_plotting=False,
                verbose=False,
            )

            Z, _ = OK.execute("grid", xv, yv)
            Z = np.asarray(Z, dtype=float)

            # ----------------------------------------------------
            # F) support mask from link-path support points
            # ----------------------------------------------------
            if use_dry_constraint:
                mask = _support_mask_wet_dry_haversine(
                    LON,
                    LAT,
                    lon_wet,
                    lat_wet,
                    lon_dry,
                    lat_dry,
                    wet_k=int(support_k),
                    wet_radius_km=float(support_radius_km),
                    dry_radius_km=float(dry_radius_km),
                    dry_deactivate_if_wet_k_within=1,
                )
            else:
                if _SKLEARN_AVAILABLE and len(lon_wet) >= max(1, int(support_k)):
                    mask = _support_mask_wet_haversine(
                        LON,
                        LAT,
                        lon_wet,
                        lat_wet,
                        k=int(support_k),
                        radius_km=float(support_radius_km),
                    )
                else:
                    mask = _nearest_distance_mask(
                        xv,
                        yv,
                        lon_wet,
                        lat_wet,
                        max_dist_km=support_radius_km,
                        workers=1,
                    )

            mask = np.asarray(mask, dtype=bool)

            if clean_support and np.any(mask):
                mask = _clean_support_mask(
                    mask,
                    closing_iters=int(support_closing_iters),
                    fill_holes=bool(support_fill_holes),
                    max_hole_px=int(support_max_hole_px),
                )

            # ----------------------------------------------------
            # G) confidence/support layer from link-path support points
            # ----------------------------------------------------
            if use_soft_confidence and np.any(mask) and len(lon_wet) > 0:
                conf = _support_confidence_from_wet_dry_v2(
                    LON,
                    LAT,
                    lon_wet,
                    lat_wet,
                    lon_dry,
                    lat_dry,
                    wet_k=int(support_k),
                    wet_radius_km=float(support_radius_km),
                    dry_radius_km=float(dry_radius_km),
                    dry_penalty_weight=float(confidence_dry_penalty_weight),
                    conf_power=float(confidence_power),
                )

                conf = np.where(mask, conf, 0.0)

                # Apply confidence floor only inside support.
                conf = np.where(
                    mask,
                    np.clip(conf, float(confidence_floor), 1.0),
                    0.0,
                )
            else:
                conf = np.where(mask, 1.0, 0.0)

            # ----------------------------------------------------
            # H) scientific rainfall field
            # ----------------------------------------------------
            if np.isnan(outside_support_fill):
                Z = np.where(mask, Z, np.nan)
            else:
                Z = np.where(mask, Z, float(outside_support_fill))

            # Soft taper scientific rainfall by confidence.
            # This reduces weak-support pixels without abruptly deleting them.
            if use_soft_confidence:
                Z = np.where(np.isfinite(Z) & mask, Z * conf, Z)

            # Optional smoothing of scientific field.
            # Recommendation: keep this low/disabled for scientific output.
            if smooth_kernel_px is not None and int(smooth_kernel_px) > 1:
                Zs = _smooth_normalized(
                    Z,
                    write_mask=mask,
                    kernel_px=int(smooth_kernel_px),
                )

                if smooth_fill_holes:
                    Z = np.where(mask, Zs, Z)
                else:
                    Z = np.where(mask & np.isfinite(Z), Zs, Z)

            # Drizzle floor to zero, preserving NaNs.
            if drizzle_to_zero is not None:
                Z = np.where(
                    np.isfinite(Z) & (Z < float(drizzle_to_zero)),
                    0.0,
                    Z,
                )

            # ----------------------------------------------------
            # I) display/cosmetic rainfall field
            # ----------------------------------------------------
            if make_display_field:
                Z_display = _cosmetic_smooth_rain_for_display(
                    Z,
                    mask,
                    kernel_px=display_smooth_kernel_px,
                    fill_holes=display_smooth_fill_holes,
                    drizzle_to_zero=drizzle_to_zero,
                    apply_edge_taper=apply_display_edge_taper,
                    edge_taper_pixels=display_edge_taper_pixels,
                    edge_taper_min_weight=display_edge_taper_min_weight,
                )
            else:
                Z_display = Z.copy()

            # ----------------------------------------------------
            # J) categorical coverage quality layer
            # ----------------------------------------------------
            if make_coverage_quality:
                coverage_quality = _coverage_quality_from_confidence(
                    conf,
                    support_mask=mask,
                    med_thr=float(coverage_quality_med_thr),
                    high_thr=float(coverage_quality_high_thr),
                )
            else:
                coverage_quality = np.zeros_like(mask, dtype=np.int8)

            return it, Z, Z_display, conf, mask, coverage_quality, nwet, ndry, ntrain, nsupport, True

        except Exception:
            return it, Z_fail, D_fail, C_fail, M_fail, Q_fail, nwet, ndry, ntrain, nsupport, False

    # ------------------------------------------------------------
    # 7) parallel execution
    # ------------------------------------------------------------
    backend = "loky" if parallel_backend_name == "processes" else "threading"

    with parallel_backend(backend):
        results = Parallel(n_jobs=int(n_jobs))(
            delayed(_do_one)(i, t) for i, t in enumerate(times)
        )

    # ------------------------------------------------------------
    # 8) collect outputs
    # ------------------------------------------------------------
    for it, Z, Z_display, conf, mask, coverage_quality, nwet, ndry, ntrain, nsupport, ok_flag in results:
        rain_out[it, :, :] = Z
        display_out[it, :, :] = Z_display
        conf_out[it, :, :] = conf
        mask_out[it, :, :] = mask.astype(np.int8)
        coverage_quality_out[it, :, :] = coverage_quality.astype(np.int8)

        diag["wet_counts"].append(int(nwet))
        diag["dry_counts"].append(int(ndry))
        diag["train_counts"].append(int(ntrain))
        diag["support_point_counts"].append(int(nsupport))
        diag["supported_pixel_counts"].append(int(np.count_nonzero(mask)))

        if np.any(mask):
            diag["mean_confidence_supported"].append(float(np.nanmean(conf[mask])))
        else:
            diag["mean_confidence_supported"].append(float("nan"))

        if np.isfinite(conf).any():
            diag["max_confidence"].append(float(np.nanmax(conf)))
        else:
            diag["max_confidence"].append(float("nan"))

        # Store per-time quality class counts for troubleshooting.
        uq, ct = np.unique(coverage_quality.astype(np.int8), return_counts=True)
        diag["coverage_quality_counts"].append(
            {int(k): int(v) for k, v in zip(uq, ct)}
        )

        diag["counts"]["ok" if ok_flag else "failed_or_skipped"] += 1

    # ------------------------------------------------------------
    # 9) build DataArray/Dataset
    # ------------------------------------------------------------
    time_values = pd.DatetimeIndex(times).tz_localize(None).values.astype("datetime64[ns]")

    R_da = xr.DataArray(
        rain_out,
        coords={"time": time_values, "lat": yv, "lon": xv},
        dims=("time", "lat", "lon"),
        name="R_mm_per_h",
        attrs={
            "long_name": "Primary gridded CML rainfall rate",
            "standard_name": "rainfall_rate",
            "units": "mm h-1",
            "grid_method": "RainLINK-style Ordinary Kriging on fixed Ghana grid",
            "description": (
                "Primary scientific gridded rainfall field derived from CML link-level "
                "rainfall estimates. This is the operational rainfall variable intended "
                "for downstream use, blending, and ingestion. Unsupported pixels are NaN "
                "by default. Support/confidence geometry can be computed from midpoint, "
                "start-mid-end, or along-link support points."
            ),
            "operational_use": (
                "Use this variable as the primary gridded CML rainfall estimate."
            ),
            "grid_res_deg": float(grid_res_deg),
            "fixed_extent": str(tuple(fixed_extent)),
            "support_geometry": support_geometry,
        },
    )

    if not return_dataset:
        return R_da, diag

    R_display_da = xr.DataArray(
        display_out,
        coords={"time": time_values, "lat": yv, "lon": xv},
        dims=("time", "lat", "lon"),
        name="R_display_mm_per_h",
        attrs={
            "long_name": "Display-only gridded CML rainfall rate",
            "units": "mm h-1",
            "description": (
                "Cosmetically smoothed rainfall field for visualization and diagnostic "
                "plotting only. This variable may include display smoothing and/or edge "
                "tapering and can differ numerically from R_mm_per_h. Do not use this "
                "variable for operational blending, scientific validation, quantitative "
                "forecasting, or API ingestion unless explicitly requested."
            ),
            "operational_use": (
                "Do not export/use in the main operational NetCDF product. "
                "Use R_mm_per_h instead."
            ),
            "support_geometry": support_geometry,
            "edge_taper_applied": bool(apply_display_edge_taper),
            "edge_taper_pixels": int(display_edge_taper_pixels),
            "edge_taper_min_weight": float(display_edge_taper_min_weight),
        },
    )

    conf_da = xr.DataArray(
        conf_out,
        coords={"time": time_values, "lat": yv, "lon": xv},
        dims=("time", "lat", "lon"),
        name="cml_support_confidence",
        attrs={
            "long_name": "CML rainfall support confidence",
            "units": "1",
            "valid_min": 0.0,
            "valid_max": 1.0,
            "description": (
                "Practical support/confidence indicator based on local CML network geometry "
                "and wet/dry consistency. This is not a formal probabilistic uncertainty estimate. "
                "When support_geometry='link_path', confidence is computed from sampled points along "
                "each CML path, not only from link midpoints."
            ),
            "support_geometry": support_geometry,
        },
    )

    mask_da = xr.DataArray(
        mask_out,
        coords={"time": time_values, "lat": yv, "lon": xv},
        dims=("time", "lat", "lon"),
        name="cml_support_mask",
        attrs={
            "long_name": "CML rainfall support mask",
            "units": "1",
            "flag_values": "0, 1",
            "flag_meanings": "unsupported supported",
            "description": (
                "Binary mask identifying grid cells supported by nearby wet CML geometry, "
                "optionally constrained by nearby dry CML geometry. When support_geometry='link_path', "
                "the support mask is computed from sampled points along each CML path."
            ),
            "support_geometry": support_geometry,
        },
    )

    coverage_quality_da = xr.DataArray(
        coverage_quality_out,
        coords={"time": time_values, "lat": yv, "lon": xv},
        dims=("time", "lat", "lon"),
        name="cml_coverage_quality",
        attrs={
            "long_name": "CML rainfall coverage quality class",
            "units": "1",
            "flag_values": "0, 1, 2, 3",
            "flag_meanings": (
                "unsupported low_confidence moderate_confidence high_confidence"
            ),
            "description": (
                "Categorical CML coverage/support quality derived from "
                "cml_support_confidence and cml_support_mask. "
                "This layer is intended to help downstream users decide how strongly "
                "to use or blend the CML rainfall field."
            ),
            "class_0": "unsupported",
            "class_1": "low confidence / weak CML support",
            "class_2": "moderate confidence / usable with caution",
            "class_3": "high confidence / strong CML support",
            "medium_threshold": float(coverage_quality_med_thr),
            "high_threshold": float(coverage_quality_high_thr),
            "support_geometry": support_geometry,
        },
    )

    ds = xr.Dataset(
        data_vars={
            "R_mm_per_h": R_da,
            "R_display_mm_per_h": R_display_da,
            "cml_support_confidence": conf_da,
            "cml_support_mask": mask_da,
            "cml_coverage_quality": coverage_quality_da,
        },
        coords={
            "time": time_values,
            "lat": yv,
            "lon": xv,
        },
        attrs={
            "title": "Ghana CML rainfall gridded product with support/confidence layers",
            "summary": (
                "Dataset containing the primary scientific CML gridded rainfall field, "
                "CML support confidence, CML support mask, categorical CML coverage quality, "
                "and an optional internal display-only rainfall layer."
            ),
            "grid_method": "RainLINK-style OK on fixed Ghana grid",
            "grid_res_deg": float(grid_res_deg),
            "fixed_extent": str(tuple(fixed_extent)),
            "support_geometry": support_geometry,
            "support_note": (
                "Link-path support points are used only to improve support/confidence geometry. "
                "The exported link-level point rainfall variable should remain assigned to link midpoints."
            ),
            "display_note": (
                "R_display_mm_per_h is intended for internal visualization/diagnostics only. "
                "It may differ numerically from R_mm_per_h and should not be used as the "
                "main operational rainfall variable."
            ),
            "operational_export_note": (
                "For operational NetCDF export to TAHMO/Rainboo, include R_mm_per_h, "
                "cml_support_confidence, cml_support_mask, cml_coverage_quality, "
                "R_point_mm_per_h, link_lon, link_lat, and link_id. Do not include "
                "R_display_mm_per_h in the main operational product unless explicitly requested."
            ),
            "coverage_quality_note": (
                "cml_coverage_quality is derived from cml_support_confidence and cml_support_mask "
                "to provide a simple downstream decision layer for blending or filtering."
            ),
        },
    )

    ds["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    ds["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    ds["time"].attrs.update({"standard_name": "time"})

    return ds, diag

def _smooth_normalized(Z, write_mask, kernel_px: int):
    """
    Normalized box smoothing that ignores NaNs:
      Z_smooth = box(Z0) / box(W)
    where Z0=0 at NaNs and W=1 where finite else 0.
    Returns NaN where denominator is 0.
    """
    if kernel_px is None or int(kernel_px) <= 1:
        return Z

    Z = np.asarray(Z, float)
    W = np.isfinite(Z).astype(float)
    Z0 = np.nan_to_num(Z, nan=0.0)

    num = uniform_filter(Z0, size=int(kernel_px), mode="nearest")
    den = uniform_filter(W,  size=int(kernel_px), mode="nearest")
    Zs  = np.where(den > 0, num / np.maximum(den, _EPS), np.nan)

    # only write where requested (e.g., strict support)
    out = np.where(write_mask, Zs, np.nan)
    return out

def grid_rain_at_time_rainlink(df_s5, df_meta_for_xy, t, **kwargs):
    return grid_rain_15min_rainlink_ok(
        df_s5=df_s5, df_meta_for_xy=df_meta_for_xy, times_sel=[t], **kwargs
    )

def get_da(Res, name="R_mm_per_h"):
    """Accepts DataArray, Dataset, or dict and returns the rain DataArray."""
    if isinstance(Res, xr.DataArray):
        return Res
    if isinstance(Res, xr.Dataset):
        if name in Res: return Res[name]
        raise KeyError(f"{name} not in Dataset vars: {list(Res.data_vars)}")
    if isinstance(Res, dict):
        if name in Res: return Res[name]
        raise KeyError(f"{name} not in dict keys: {list(Res.keys())}")
    raise TypeError(f"Unsupported Res type: {type(Res)}")

def slice_time(da, t=None):
    """
    Return a 2-D (lat,lon) DataArray and a label timestamp.
    - If 'time' in dims: select nearest t (if t is None -> use first time).
    - If no 'time' dim: just return da and label from attr if present.
    """
    if "time" in da.dims:
        if t is None:
            t_sel = pd.to_datetime(da["time"].values[0]).to_pydatetime()
        else:
            t_sel = pd.Timestamp(t)
            if t_sel.tzinfo is not None:
                t_sel = t_sel.tz_convert("UTC").tz_localize(None)
        out = da.sel(time=np.datetime64(t_sel), method="nearest")
        return out, pd.to_datetime(out["time"].item())
    # 2-D already
    label = pd.to_datetime(da.attrs.get("time", "NaT"))
    return da, label

def apply_support_mask(Res, df_s5, meta_xy, t=None, *, k=3, km=35.0, eps=0.1, name="R_mm_per_h"):
    """
    Keep cells with ≥k links having R>eps within km (great-circle).
    df_s5: time-indexed with ['ID','R_mm_per_h'].
    meta_xy: ['ID','XStart','YStart','XEnd','YEnd'] numeric.
    """
    da = get_da(Res, name)
    sl, t_used = slice_time(da, t)

    # points at that time
    pts = (df_s5.loc[pd.Timestamp(t_used)]
           .merge(meta_xy, on="ID", how="inner")
           .assign(lon_mid=lambda d: (pd.to_numeric(d.XStart)+pd.to_numeric(d.XEnd))/2,
                   lat_mid=lambda d: (pd.to_numeric(d.YStart)+pd.to_numeric(d.YEnd))/2))
    wet = (pts["R_mm_per_h"].to_numpy(float) > eps)
    if wet.sum() < max(3, k):
        # nothing to mask—return slice unchanged
        sl2 = sl.copy()
        sl2.attrs["time"] = str(t_used)
        return sl2

    tree = BallTree(np.deg2rad(np.c_[pts["lat_mid"], pts["lon_mid"]]), metric="haversine")
    gx, gy = np.meshgrid(sl["lon"].values, sl["lat"].values)
    q = np.deg2rad(np.c_[gy.ravel(), gx.ravel()])
    neigh = tree.query_radius(q, r=km/6371.0)
    kcnt = np.fromiter((wet[idx].sum() if len(idx) else 0 for idx in neigh), int).reshape(gy.shape)

    Z = sl.values.copy()
    Z[kcnt < k] = np.nan
    out = sl.copy(data=Z)
    out.attrs["time"] = str(t_used)
    return out


# =============================================================================
# 5. NetCDF output helpers
# =============================================================================

def save_each_time_to_netcdf(
    data,                           # xr.DataArray OR xr.Dataset (with var_name)
    out_dir,
    base_name="ghana_cml_R",
    *,
    var_name="R_mm_per_h",          # ignored if data is a DataArray
    engine="netcdf4",               # or "h5netcdf"
    complevel=9,
    dtype="float32",
    fill_value=np.nan,              # what to write for NaNs
    chunks_lat=256,
    chunks_lon=256,
    keep_time_dim=True,             # keep a size-1 time dimension in each file
):
    """
    Writes one .nc per timestamp with high compression and correct chunking.
    Returns list of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Normalize to a DataArray
    if isinstance(data, xr.Dataset):
        if var_name not in data.data_vars and len(data.data_vars) == 1:
            var_name = list(data.data_vars)[0]
        da = data[var_name]
    elif isinstance(data, xr.DataArray):
        var_name = data.name or var_name
        da = data
    else:
        raise TypeError("data must be an xarray DataArray or Dataset")

    # If no time dimension, write a single file and return
    if "time" not in da.dims:
        d2 = da.astype(dtype)
        ds = d2.to_dataset(name=var_name)
        # chunks: match dims exactly
        dims = ds[var_name].dims
        sizes = ds[var_name].sizes
        chunks = []
        for d in dims:
            if d == "lat":
                chunks.append(min(int(chunks_lat), sizes[d]))
            elif d == "lon":
                chunks.append(min(int(chunks_lon), sizes[d]))
            else:
                chunks.append(min(1, sizes[d]))
        enc = {
            var_name: {
                "zlib": True, "complevel": int(complevel), "shuffle": True,
                "dtype": dtype, "_FillValue": fill_value,
                "chunksizes": tuple(chunks),
            }
        }
        fn = os.path.join(out_dir, f"{base_name}.nc")
        ds.to_netcdf(fn, engine=engine, encoding=enc)
        return [fn]

    # Otherwise iterate times
    times = pd.to_datetime(da["time"].values)
    out_paths = []

    for t in times:
        # 2-D slice (lat,lon)
        sl = da.sel(time=np.datetime64(t)).astype(dtype)

        # Keep time dim? -> expand to (time,lat,lon) of length 1
        if keep_time_dim:
            sl = sl.expand_dims(time=[np.datetime64(t)])
        ds = sl.to_dataset(name=var_name)

        # Build chunks tuple that matches EXACT dims order
        dims = ds[var_name].dims             # e.g., ('time','lat','lon') or ('lat','lon')
        sizes = ds[var_name].sizes
        chunks = []
        for d in dims:
            if d == "time":
                chunks.append(1)
            elif d == "lat":
                chunks.append(min(int(chunks_lat), sizes[d]))
            elif d == "lon":
                chunks.append(min(int(chunks_lon), sizes[d]))
            else:
                # Unknown dim: just chunk by its full size
                chunks.append(sizes[d])

        enc = {
            var_name: {
                "zlib": True, "complevel": int(complevel), "shuffle": True,
                "dtype": dtype, "_FillValue": fill_value,
                "chunksizes": tuple(chunks),
            },
            # coords (don’t compress tiny arrays)
            "lat": {"zlib": False},
            "lon": {"zlib": False},
            "time": {"zlib": False},
        }

        # Nice filename stamp: YYYYmmddTHHMMSS
        t_str = pd.Timestamp(t).strftime("%Y%m%dT%H%M%S")
        fn = os.path.join(out_dir, f"{base_name}_{t_str}.nc")

        # Use unlimited time when present
        unlimited = {"time"} if keep_time_dim else None
        ds.to_netcdf(fn, engine=engine, encoding=enc, unlimited_dims=unlimited)
        out_paths.append(fn)

    return out_paths

def save_daily_grid_and_points_netcdf(
    R_da_day: xr.DataArray,          # (time, lat, lon) for one day
    df_s5_day: pd.DataFrame,         # time-indexed; cols: ID, R_mm_per_h
    meta_xy: pd.DataFrame,           # cols: ID, XStart, YStart, XEnd, YEnd
    out_dir: str,
    day: pd.Timestamp | str,         # e.g. "2025-06-19"
    base_name: str = "ghana_cml_R",
    *,
    var_grid: str = "R_mm_per_h",
    var_point: str = "R_point_mm_per_h",
    engine: str = "netcdf4",
    complevel: int = 5,
    dtype: str = "float32",
    fill_value: float = -9999.0,
    chunks_time: int = 1,
    chunks_lat: int = 256,
    chunks_lon: int = 256,
    chunks_link: int = 2048,

    # ---------------------------
    # NEW: metadata knobs
    # ---------------------------
    version: str = "V1",
    title: str | None = None,
    summary: str | None = None,
    producer_name: str = "Trans-African Hydro-Meteorological Observatory (TAHMO)",
    institution: str = "TAHMO",
    creator_name: str  = "Kingsley Kumah",          # e.g., "Kingsley Kumah"
    creator_email: str | None = None,
    project: str = "PRIME Ghana CML rainfall retrieval",
    source: str = "Commercial Microwave Links (CML); RainLINK-like processing; Ordinary Kriging gridding",
    references: str | None = None,            # DOI / repo URL / paper
    comment: str | None = None,
    conventions: str = "CF-1.8",
):
    os.makedirs(out_dir, exist_ok=True)
    day_date = pd.Timestamp(day).date()

    # --- ensure day consistency ---
    # Grid times
    t_grid = pd.to_datetime(R_da_day["time"].values)
    t_grid = pd.DatetimeIndex(t_grid).tz_localize(None)

    # Points times from df_s5_day index
    idx = df_s5_day.index
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df_s5_day = df_s5_day.copy()
    df_s5_day.index = idx

    # Restrict to day (just in case)
    df_s5_day = df_s5_day[df_s5_day.index.date == day_date]

    # --- build link geometry (midpoints) ---
    m = meta_xy.drop_duplicates("ID").copy()
    for c in ["XStart", "YStart", "XEnd", "YEnd"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["XStart", "YStart", "XEnd", "YEnd"])

    m["lon_mid"] = 0.5 * (m["XStart"] + m["XEnd"])
    m["lat_mid"] = 0.5 * (m["YStart"] + m["YEnd"])

    link_ids = m["ID"].astype(str).values
    n_link = len(link_ids)

    # --- align points to (time, link) matrix ---
    times = t_grid  # master axis to match grid exactly
    Rpt = np.full((len(times), n_link), np.nan, dtype=np.float32)
    id2j = {lid: j for j, lid in enumerate(link_ids)}

    for i, tt in enumerate(times):
        g = df_s5_day.loc[df_s5_day.index == tt]
        if g is None or len(g) == 0:
            continue
        if isinstance(g, pd.Series):
            g = g.to_frame().T
        for _, row in g.iterrows():
            lid = str(row["ID"])
            j = id2j.get(lid, None)
            if j is None:
                continue
            val = pd.to_numeric(row["R_mm_per_h"], errors="coerce")
            if np.isfinite(val):
                Rpt[i, j] = float(val)

    # --- build dataset ---
    grid = R_da_day.astype(dtype).rename(var_grid)

    ds = xr.Dataset(
        data_vars={
            var_grid: grid,
            var_point: (("time", "link"), Rpt),
            "link_lon": (("link",), m["lon_mid"].to_numpy(float)),
            "link_lat": (("link",), m["lat_mid"].to_numpy(float)),
            "link_id":  (("link",), link_ids),
        },
        coords={
            "time": times.values.astype("datetime64[ns]"),
            "lat": grid["lat"].values,
            "lon": grid["lon"].values,
            "link": np.arange(n_link, dtype=int),
        }
    )

    # ---------------------------
    # NEW: variable-level attrs
    # ---------------------------
    ds[var_grid].attrs.update({
        "long_name": "Gridded rainfall rate from CML",
        "units": "mm h-1",
        "description": (
            "Spatially continuous rainfall field on a regular lat/lon grid. "
            "Derived from link-level CML rainfall estimates mapped via RainLINK-like OK with strict wet support."
        ),
    })
    ds[var_point].attrs.update({
        "long_name": "Link midpoint rainfall rate",
        "units": "mm h-1",
        "description": (
            "Rainfall rate estimated per CML link and assigned to the link midpoint. "
            "Array is aligned to the grid time axis; missing link/time pairs are fill_value."
        ),
    })
    ds["link_lon"].attrs.update({"long_name": "Link midpoint longitude", "units": "degrees_east"})
    ds["link_lat"].attrs.update({"long_name": "Link midpoint latitude", "units": "degrees_north"})
    ds["link_id"].attrs.update({"long_name": "CML link identifier"})

    ds["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    ds["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
    ds["time"].attrs.update({"standard_name": "time"})

    # ---------------------------
    # NEW: global attrs (file metadata)
    # ---------------------------
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day_str_iso = pd.Timestamp(day_date).strftime("%Y-%m-%d")

    if title is None:
        title = f"Ghana CML rainfall (gridded + link midpoints) — {day_str_iso}"
    if summary is None:
        summary = (
            "Daily NetCDF containing (1) 15-min gridded rainfall maps over Ghana and "
            "(2) corresponding 15-min link-level rainfall rates assigned to link midpoints."
        )

    ds.attrs.update({
        "Conventions": conventions,
        "title": title,
        "summary": summary,
        "product_version": version,
        "institution": institution,
        "producer_name": producer_name,
        "project": project,
        "source": source,
        "history": f"{created}: created daily grid+points NetCDF",
        "date_created": created,
        "time_coverage_start": str(times.min()),
        "time_coverage_end": str(times.max()),
        "geospatial_lat_min": float(np.nanmin(ds["lat"].values)),
        "geospatial_lat_max": float(np.nanmax(ds["lat"].values)),
        "geospatial_lon_min": float(np.nanmin(ds["lon"].values)),
        "geospatial_lon_max": float(np.nanmax(ds["lon"].values)),
    })

    # optional creator/contact fields
    if creator_name is not None:
        ds.attrs["creator_name"] = creator_name
    if creator_email is not None:
        ds.attrs["creator_email"] = creator_email
    if references is not None:
        ds.attrs["references"] = references
    if comment is not None:
        ds.attrs["comment"] = comment

    # --- encoding / chunking ---
    enc = {
        var_grid: {
            "zlib": True, "complevel": int(complevel), "shuffle": True,
            "dtype": dtype, "_FillValue": fill_value,
            "chunksizes": (
                min(chunks_time, len(times)),
                min(chunks_lat, ds.sizes["lat"]),
                min(chunks_lon, ds.sizes["lon"]),
            ),
        },
        var_point: {
            "zlib": True, "complevel": int(complevel), "shuffle": True,
            "dtype": dtype, "_FillValue": fill_value,
            "chunksizes": (min(chunks_time, len(times)), min(chunks_link, n_link)),
        },
        "lat": {"zlib": False},
        "lon": {"zlib": False},
        "time": {"zlib": False},
        "link": {"zlib": False},
        "link_lon": {"zlib": False},
        "link_lat": {"zlib": False},
        "link_id": {"zlib": False},
    }

    day_str = pd.Timestamp(day_date).strftime("%Y%m%d")
    fn = os.path.join(out_dir, f"{base_name}_{day_str}.nc")

    ds.to_netcdf(fn, engine=engine, encoding=enc, unlimited_dims={"time"})
    return fn

def save_15min_grid_and_points_netcdf(
    R_da_day: xr.DataArray,          # (time, lat, lon) for one day (or any time range)
    df_s5_day: pd.DataFrame,         # time-indexed; cols: ID, R_mm_per_h
    meta_xy: pd.DataFrame,           # cols: ID, XStart, YStart, XEnd, YEnd
    out_dir: str,
    day: pd.Timestamp | str,         # e.g. "2025-06-19" (used for folder naming / filtering)
    base_name: str = "ghana_cml_R_15min",
    *,
    var_grid: str = "R_mm_per_h",
    var_point: str = "R_point_mm_per_h",
    engine: str = "netcdf4",
    complevel: int = 5,
    dtype: str = "float32",
    fill_value: float = -9999.0,
    chunks_lat: int = 256,
    chunks_lon: int = 256,
    chunks_link: int = 2048,

    # metadata knobs
    version: str = "V1",
    title: str | None = None,
    summary: str | None = None,
    producer_name: str = "Trans-African Hydro-Meteorological Observatory (TAHMO)",
    institution: str = "TAHMO",
    creator_name: str  = "Kingsley Kumah",
    creator_email: str | None = None,
    project: str = "PRIME Ghana CML rainfall retrieval",
    source: str = "Commercial Microwave Links (CML); RainLINK-like processing; Ordinary Kriging gridding",
    references: str | None = None,
    comment: str | None = None,
    conventions: str = "CF-1.8",

    # file naming
    ts_fmt: str = "%Y%m%dT%H%M%SZ",   # e.g., 20250619T161500Z
):
    """
    Writes ONE NetCDF PER TIME STEP (e.g., 15-min).
    Each file contains:
      - var_grid(lat, lon): rainfall map at that timestamp
      - var_point(link): link midpoint rainfall at that timestamp
      - link geometry variables (link_lon, link_lat, link_id)
      - time as a scalar coordinate (or length-1 time dim, depending on preference)
    Returns list of written filenames.
    """

    os.makedirs(out_dir, exist_ok=True)
    day_date = pd.Timestamp(day).date()

    # --- grid times ---
    t_grid = pd.to_datetime(R_da_day["time"].values)
    t_grid = pd.DatetimeIndex(t_grid).tz_localize(None)

    # --- points times from df index ---
    df = df_s5_day.copy()
    idx = df.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df.index = pd.DatetimeIndex(idx)

    # Restrict to day (optional but consistent with your old function)
    df = df[df.index.date == day_date]

    # --- link midpoints ---
    m = meta_xy.drop_duplicates("ID").copy()
    for c in ["XStart", "YStart", "XEnd", "YEnd"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m = m.dropna(subset=["XStart", "YStart", "XEnd", "YEnd"])

    m["lon_mid"] = 0.5 * (m["XStart"] + m["XEnd"])
    m["lat_mid"] = 0.5 * (m["YStart"] + m["YEnd"])

    link_ids = m["ID"].astype(str).values
    n_link = len(link_ids)
    id2j = {lid: j for j, lid in enumerate(link_ids)}

    # pre-extract geometry vectors
    link_lon = m["lon_mid"].to_numpy(float)
    link_lat = m["lat_mid"].to_numpy(float)

    # created stamp (global)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day_str_iso = pd.Timestamp(day_date).strftime("%Y-%m-%d")

    if title is None:
        title = f"Ghana CML rainfall (single time-step files) — {day_str_iso}"
    if summary is None:
        summary = (
            "NetCDF per 15-min time step containing (1) gridded rainfall map over Ghana and "
            "(2) corresponding link-level rainfall rates assigned to link midpoints."
        )

    written = []

    # --- iterate over each time ---
    for tt in t_grid:
        # 1) grid slice (lat, lon)
        grid_t = R_da_day.sel(time=tt).astype(dtype).rename(var_grid)

        # 2) link vector (link,)
        rlink = np.full((n_link,), np.nan, dtype=np.float32)
        g = df.loc[df.index == tt]
        if isinstance(g, pd.Series):
            g = g.to_frame().T
        if g is not None and len(g) > 0:
            for _, row in g.iterrows():
                lid = str(row.get("ID"))
                j = id2j.get(lid, None)
                if j is None:
                    continue
                val = pd.to_numeric(row.get("R_mm_per_h"), errors="coerce")
                if np.isfinite(val):
                    rlink[j] = float(val)

        # 3) dataset (single step)
        # keep a length-1 time dimension for CF-friendliness
        ds = xr.Dataset(
            data_vars={
                var_grid: grid_t.expand_dims(time=[np.datetime64(tt)]),
                var_point: (("time", "link"), rlink.reshape(1, -1)),
                "link_lon": (("link",), link_lon),
                "link_lat": (("link",), link_lat),
                "link_id":  (("link",), link_ids),
            },
            coords={
                "time": np.array([np.datetime64(tt)], dtype="datetime64[ns]"),
                "lat": grid_t["lat"].values,
                "lon": grid_t["lon"].values,
                "link": np.arange(n_link, dtype=int),
            }
        )

        # variable attrs
        ds[var_grid].attrs.update({
            "long_name": "Gridded rainfall rate from CML",
            "units": "mm h-1",
        })
        ds[var_point].attrs.update({
            "long_name": "Link midpoint rainfall rate",
            "units": "mm h-1",
        })
        ds["link_lon"].attrs.update({"long_name": "Link midpoint longitude", "units": "degrees_east"})
        ds["link_lat"].attrs.update({"long_name": "Link midpoint latitude", "units": "degrees_north"})
        ds["link_id"].attrs.update({"long_name": "CML link identifier"})
        ds["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
        ds["lon"].attrs.update({"standard_name": "longitude", "units": "degrees_east"})
        ds["time"].attrs.update({"standard_name": "time"})

        # global attrs (include timestamp-specific coverage)
        ds.attrs.update({
            "Conventions": conventions,
            "title": title,
            "summary": summary,
            "product_version": version,
            "institution": institution,
            "producer_name": producer_name,
            "project": project,
            "source": source,
            "history": f"{created}: created 15-min grid+points NetCDF",
            "date_created": created,
            "time_coverage_start": str(pd.Timestamp(tt)),
            "time_coverage_end": str(pd.Timestamp(tt)),
            "geospatial_lat_min": float(np.nanmin(ds["lat"].values)),
            "geospatial_lat_max": float(np.nanmax(ds["lat"].values)),
            "geospatial_lon_min": float(np.nanmin(ds["lon"].values)),
            "geospatial_lon_max": float(np.nanmax(ds["lon"].values)),
            "creator_name": creator_name,
        })
        if creator_email is not None:
            ds.attrs["creator_email"] = creator_email
        if references is not None:
            ds.attrs["references"] = references
        if comment is not None:
            ds.attrs["comment"] = comment

        # encoding / compression
        enc = {
            var_grid: {
                "zlib": True, "complevel": int(complevel), "shuffle": True,
                "dtype": dtype, "_FillValue": fill_value,
                "chunksizes": (1, min(chunks_lat, ds.sizes["lat"]), min(chunks_lon, ds.sizes["lon"])),
            },
            var_point: {
                "zlib": True, "complevel": int(complevel), "shuffle": True,
                "dtype": dtype, "_FillValue": fill_value,
                "chunksizes": (1, min(chunks_link, n_link)),
            },
            "lat": {"zlib": False},
            "lon": {"zlib": False},
            "time": {"zlib": False},
            "link": {"zlib": False},
            "link_lon": {"zlib": False},
            "link_lat": {"zlib": False},
            "link_id": {"zlib": False},
        }

        # filename per timestamp
        ts = pd.Timestamp(tt).tz_localize("UTC")  # label as Z for name
        fn = os.path.join(out_dir, f"{base_name}_{ts.strftime(ts_fmt)}.nc")

        ds.to_netcdf(fn, engine=engine, encoding=enc)
        written.append(fn)

        ds.close()

    return written

def save_15min_grid_and_points_netcdf_for_day(
    grid_data: xr.Dataset | xr.DataArray,   # preferred: grid_ds from gridding function
    df_s5: pd.DataFrame,                    # can be multi-day; we will filter to `day`
    meta_xy: pd.DataFrame,
    out_dir: str,
    day: pd.Timestamp | str,
    base_name: str = "ghana_cml_R_15min",
    *,
    # operational variable names
    var_grid: str = "R_mm_per_h",
    var_point: str = "R_point_mm_per_h",

    # optional grid support variables to export if available
    support_conf_var: str = "cml_support_confidence",
    support_mask_var: str = "cml_support_mask",
    coverage_quality_var: str = "cml_coverage_quality",

    # IMPORTANT:
    # R_display_mm_per_h is intentionally excluded by default to avoid
    # operational confusion. Keep it for internal diagnostics/plots only.
    include_display_field: bool = False,
    display_var: str = "R_display_mm_per_h",

    engine: str = "netcdf4",
    complevel: int = 5,
    dtype: str = "float32",
    fill_value: float = -9999.0,
    chunks_lat: int = 256,
    chunks_lon: int = 256,
    chunks_link: int = 2048,
    version: str = "V1",
    conventions: str = "CF-1.8",
    creator_name: str = "Kingsley Kumah",
    creator_email: str | None = None,
    institution: str = "TAHMO",
    producer_name: str = "Trans-African Hydro-Meteorological Observatory (TAHMO)",
    project: str = "PRIME Ghana CML rainfall retrieval",
    source: str = (
        "Commercial Microwave Links (CML); RainLINK-like processing; "
        "Ordinary Kriging gridding with CML support/confidence layers"
    ),
    references: str | None = None,
    comment: str | None = None,
    ts_fmt: str = "%Y%m%dT%H%M%SZ",
):
    """
    Write one operational NetCDF file per 15-min timestamp for one day.

    Preferred input
    ---------------
    grid_data : xr.Dataset
        Dataset returned by grid_rain_15min_rainlink_ok_full_ghana(), containing:
          - R_mm_per_h(time, lat, lon)
          - cml_support_confidence(time, lat, lon)
          - cml_support_mask(time, lat, lon)
          - cml_coverage_quality(time, lat, lon)

        R_display_mm_per_h may exist in grid_data but is not exported by default.

    Backward-compatible input
    -------------------------
    grid_data : xr.DataArray
        Single gridded rainfall DataArray. In this case only var_grid and point
        variables are written.

    Operational NetCDF contents
    ---------------------------
    Main gridded variables:
      - R_mm_per_h(time, lat, lon)
      - cml_support_confidence(time, lat, lon), if available
      - cml_support_mask(time, lat, lon), if available
      - cml_coverage_quality(time, lat, lon), if available

    Link-point variables:
      - R_point_mm_per_h(time, link)
      - link_lon(link)
      - link_lat(link)
      - link_id(link)

    Notes
    -----
    R_display_mm_per_h is excluded by default to avoid confusing downstream
    operational users. It is a display/diagnostic field, not the primary
    rainfall estimate.

    Important dtype behavior
    ------------------------
    Continuous variables:
      - R_mm_per_h
      - cml_support_confidence
      - R_point_mm_per_h
    are written as float32.

    Categorical variables:
      - cml_support_mask
      - cml_coverage_quality
    are written as int8 without _FillValue so xarray does not decode them
    back to float.
    """

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # 0) Normalize requested day
    # ------------------------------------------------------------
    day_ts = pd.Timestamp(day)

    if day_ts.tzinfo is not None:
        day_ts = day_ts.tz_convert("UTC").tz_localize(None)
    else:
        day_ts = day_ts.tz_localize(None)

    day_date = day_ts.date()

    # ------------------------------------------------------------
    # 1) Normalize grid input
    # ------------------------------------------------------------
    if isinstance(grid_data, xr.DataArray):
        # Backward compatibility: single rainfall DataArray.
        R_da = grid_data

        if R_da.name is None:
            R_da = R_da.rename(var_grid)

        grid_ds_all = R_da.to_dataset(name=var_grid)

    elif isinstance(grid_data, xr.Dataset):
        grid_ds_all = grid_data

        if var_grid not in grid_ds_all.data_vars:
            raise KeyError(
                f"Primary rainfall variable '{var_grid}' not found in grid_data. "
                f"Available variables: {list(grid_ds_all.data_vars)}"
            )

    else:
        raise TypeError("grid_data must be an xarray Dataset or DataArray")

    if "time" not in grid_ds_all.coords:
        raise ValueError("grid_data must contain a 'time' coordinate.")

    if "lat" not in grid_ds_all.coords or "lon" not in grid_ds_all.coords:
        raise ValueError("grid_data must contain 'lat' and 'lon' coordinates.")

    # ------------------------------------------------------------
    # 2) Filter grid variables to requested day
    # ------------------------------------------------------------
    t_all = pd.to_datetime(grid_ds_all["time"].values)
    t_all = pd.DatetimeIndex(t_all).tz_localize(None)

    mask_day = np.array([t.date() == day_date for t in t_all])

    if mask_day.sum() == 0:
        raise ValueError(f"No grid times found for day={day_date} in grid_data.time")

    selected_times = t_all[mask_day]
    grid_day = grid_ds_all.sel(time=selected_times.values)

    # Master time axis for requested day only
    times = pd.to_datetime(grid_day["time"].values)
    times = pd.DatetimeIndex(times).tz_localize(None)

    # ------------------------------------------------------------
    # 3) Select operational grid variables to write
    # ------------------------------------------------------------
    grid_vars_to_write = [var_grid]

    for v in [support_conf_var, support_mask_var, coverage_quality_var]:
        if v in grid_day.data_vars:
            grid_vars_to_write.append(v)

    if include_display_field and display_var in grid_day.data_vars:
        grid_vars_to_write.append(display_var)

    # ------------------------------------------------------------
    # 4) Filter points/link rainfall to requested day
    # ------------------------------------------------------------
    df = df_s5.copy()

    idx = pd.to_datetime(df.index)
    idx = pd.DatetimeIndex(idx)

    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    else:
        idx = idx.tz_localize(None)

    df.index = idx
    df = df[df.index.date == day_date]

    # ------------------------------------------------------------
    # 5) Link midpoint geometry
    # ------------------------------------------------------------
    m = meta_xy.drop_duplicates("ID").copy()

    for c in ["XStart", "YStart", "XEnd", "YEnd"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")

    m = m.dropna(subset=["XStart", "YStart", "XEnd", "YEnd"])

    m["lon_mid"] = 0.5 * (m["XStart"] + m["XEnd"])
    m["lat_mid"] = 0.5 * (m["YStart"] + m["YEnd"])

    link_ids = m["ID"].astype(str).values
    n_link = len(link_ids)

    id2j = {lid: j for j, lid in enumerate(link_ids)}

    link_lon = m["lon_mid"].to_numpy(float)
    link_lat = m["lat_mid"].to_numpy(float)

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    day_str_iso = pd.Timestamp(day_date).strftime("%Y-%m-%d")

    written = []

    # ------------------------------------------------------------
    # 6) Write one file per timestamp
    # ------------------------------------------------------------
    for tt in times:
        tt_np = np.datetime64(tt)

        # --------------------------------------------------------
        # A) Build link midpoint rainfall vector for this timestamp
        # --------------------------------------------------------
        rlink = np.full((n_link,), np.nan, dtype=np.float32)

        g = df.loc[df.index == tt]  # exact timestamp match

        if isinstance(g, pd.Series):
            g = g.to_frame().T

        if g is not None and len(g) > 0:
            for _, row in g.iterrows():
                lid = str(row.get("ID"))
                j = id2j.get(lid, None)

                if j is None:
                    continue

                val = pd.to_numeric(row.get("R_mm_per_h"), errors="coerce")

                if np.isfinite(val):
                    rlink[j] = float(val)

        # --------------------------------------------------------
        # B) Build data variables for this timestamp
        # --------------------------------------------------------
        data_vars = {}

        for v in grid_vars_to_write:
            da_t = grid_day[v].sel(time=tt_np).expand_dims(time=[tt_np])

            # Keep mask/quality as compact integer variables.
            if v in [support_mask_var, coverage_quality_var]:
                da_t = da_t.astype("int8")
            else:
                da_t = da_t.astype(dtype)

            data_vars[v] = da_t

        data_vars[var_point] = (("time", "link"), rlink.reshape(1, -1))
        data_vars["link_lon"] = (("link",), link_lon)
        data_vars["link_lat"] = (("link",), link_lat)
        data_vars["link_id"] = (("link",), link_ids)

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "time": np.array([tt_np], dtype="datetime64[ns]"),
                "lat": grid_day["lat"].values,
                "lon": grid_day["lon"].values,
                "link": np.arange(n_link, dtype=int),
            },
        )

        # --------------------------------------------------------
        # C) Variable attributes
        # --------------------------------------------------------
        ds[var_grid].attrs.update({
            "long_name": "Primary gridded CML rainfall rate",
            "standard_name": "rainfall_rate",
            "units": "mm h-1",
            "description": (
                "Primary operational gridded rainfall estimate derived from CML "
                "link-level rainfall rates. This is the rainfall variable intended "
                "for downstream ingestion, blending, and operational use."
            ),
        })

        if support_conf_var in ds.data_vars:
            ds[support_conf_var].attrs.update({
                "long_name": "CML rainfall support confidence",
                "units": "1",
                "valid_min": 0.0,
                "valid_max": 1.0,
                "description": (
                    "Practical CML support/confidence indicator from 0 to 1, based on "
                    "local CML network geometry and wet/dry consistency. This is not a "
                    "formal probabilistic uncertainty estimate. Use this layer to "
                    "downweight or filter rainfall in weakly supported areas."
                ),
            })

        if support_mask_var in ds.data_vars:
            ds[support_mask_var].attrs.update({
                "long_name": "CML rainfall support mask",
                "units": "1",
                "flag_values": "0, 1",
                "flag_meanings": "unsupported supported",
                "description": (
                    "Binary mask identifying grid cells supported by nearby wet CML "
                    "geometry. Cells marked unsupported should generally not be treated "
                    "as valid CML rainfall estimates."
                ),
            })

        if coverage_quality_var in ds.data_vars:
            ds[coverage_quality_var].attrs.update({
                "long_name": "CML rainfall coverage quality class",
                "units": "1",
                "flag_values": "0, 1, 2, 3",
                "flag_meanings": (
                    "unsupported low_confidence moderate_confidence high_confidence"
                ),
                "description": (
                    "Categorical CML coverage/support quality derived from support "
                    "confidence and support mask. Intended to provide a simple "
                    "downstream decision layer for blending or filtering."
                ),
                "class_0": "unsupported",
                "class_1": "low confidence / weak CML support",
                "class_2": "moderate confidence / usable with caution",
                "class_3": "high confidence / strong CML support",
            })

        if include_display_field and display_var in ds.data_vars:
            ds[display_var].attrs.update({
                "long_name": "Display-only gridded CML rainfall rate",
                "units": "mm h-1",
                "description": (
                    "Display/diagnostic rainfall field only. This variable may include "
                    "cosmetic smoothing and should not be used as the primary operational "
                    "rainfall estimate."
                ),
                "operational_use": (
                    "Do not use as primary rainfall variable; use R_mm_per_h."
                ),
            })

        ds[var_point].attrs.update({
            "long_name": "Link midpoint CML rainfall rate",
            "units": "mm h-1",
            "description": (
                "CML link-level rainfall rate assigned to the link midpoint. This point "
                "variable is aligned with the file time coordinate and link dimension."
            ),
        })

        ds["link_lon"].attrs.update({
            "long_name": "CML link midpoint longitude",
            "units": "degrees_east",
        })

        ds["link_lat"].attrs.update({
            "long_name": "CML link midpoint latitude",
            "units": "degrees_north",
        })

        ds["link_id"].attrs.update({
            "long_name": "CML link identifier",
            "description": (
                "Unique identifier for each CML link/path used in the rainfall retrieval."
            ),
        })

        ds["lat"].attrs.update({
            "standard_name": "latitude",
            "units": "degrees_north",
        })

        ds["lon"].attrs.update({
            "standard_name": "longitude",
            "units": "degrees_east",
        })

        ds["time"].attrs.update({
            "standard_name": "time",
        })

        # --------------------------------------------------------
        # D) Global attributes
        # --------------------------------------------------------
        ds.attrs.update({
            "Conventions": conventions,
            "title": f"Ghana CML rainfall operational product — {day_str_iso}",
            "summary": (
                "Single time-step NetCDF containing the primary gridded CML rainfall "
                "estimate, CML support/confidence layers, categorical coverage quality, "
                "and corresponding link-midpoint rainfall values."
            ),
            "product_version": version,
            "institution": institution,
            "producer_name": producer_name,
            "creator_name": creator_name,
            "project": project,
            "source": source,
            "history": f"{created}: created 15-min operational CML rainfall NetCDF",
            "date_created": created,
            "time_coverage_start": str(pd.Timestamp(tt)),
            "time_coverage_end": str(pd.Timestamp(tt)),
            "geospatial_lat_min": float(np.nanmin(ds["lat"].values)),
            "geospatial_lat_max": float(np.nanmax(ds["lat"].values)),
            "geospatial_lon_min": float(np.nanmin(ds["lon"].values)),
            "geospatial_lon_max": float(np.nanmax(ds["lon"].values)),
            "primary_rainfall_variable": var_grid,
            "point_rainfall_variable": var_point,
            "operational_note": (
                "R_mm_per_h is the primary gridded rainfall variable for operational use. "
                "CML support confidence, support mask, and coverage quality should be used "
                "to filter, downweight, or blend the rainfall field in weakly supported areas."
            ),
            "display_field_note": (
                "Display-only rainfall fields are excluded from this operational file by default "
                "to avoid confusion with the primary rainfall estimate."
            ),
        })

        if creator_email is not None:
            ds.attrs["creator_email"] = creator_email

        if references is not None:
            ds.attrs["references"] = references

        if comment is not None:
            ds.attrs["comment"] = comment

        # --------------------------------------------------------
        # E) Encoding / compression
        # --------------------------------------------------------
        enc = {}

        for v in grid_vars_to_write:
            if v in [support_mask_var, coverage_quality_var]:
                # Categorical variables:
                # keep as int8 and do NOT assign _FillValue.
                # They are fully defined everywhere:
                #   cml_support_mask: 0/1
                #   cml_coverage_quality: 0/1/2/3
                #
                # Not assigning _FillValue avoids xarray decoding these
                # categorical layers back into float arrays when reading.
                enc[v] = {
                    "zlib": True,
                    "complevel": int(complevel),
                    "shuffle": True,
                    "dtype": "int8",
                    "chunksizes": (
                        1,
                        min(chunks_lat, ds.sizes["lat"]),
                        min(chunks_lon, ds.sizes["lon"]),
                    ),
                }
            else:
                # Continuous variables:
                # rainfall and confidence remain float32.
                enc[v] = {
                    "zlib": True,
                    "complevel": int(complevel),
                    "shuffle": True,
                    "dtype": dtype,
                    "_FillValue": fill_value,
                    "chunksizes": (
                        1,
                        min(chunks_lat, ds.sizes["lat"]),
                        min(chunks_lon, ds.sizes["lon"]),
                    ),
                }

        enc[var_point] = {
            "zlib": True,
            "complevel": int(complevel),
            "shuffle": True,
            "dtype": dtype,
            "_FillValue": fill_value,
            "chunksizes": (
                1,
                min(chunks_link, n_link),
            ),
        }

        # Do not compress small coordinate/geometry variables.
        enc["lat"] = {"zlib": False}
        enc["lon"] = {"zlib": False}
        enc["time"] = {"zlib": False}
        enc["link"] = {"zlib": False}
        enc["link_lon"] = {"zlib": False}
        enc["link_lat"] = {"zlib": False}
        enc["link_id"] = {"zlib": False}

        # --------------------------------------------------------
        # F) Write file
        # --------------------------------------------------------
        ts = pd.Timestamp(tt).tz_localize("UTC")
        fn = os.path.join(out_dir, f"{base_name}_{ts.strftime(ts_fmt)}.nc")

        ds.to_netcdf(fn, engine=engine, encoding=enc)

        written.append(fn)
        ds.close()

    return written


# =============================================================================
# 6. Raw CML file timestamp extraction and metadata coupling
# =============================================================================

def extract_datetime_from_filename(fname):
    # Example: Schedule_pfm_SDH_20250812004105281472818770368_1
    parts = fname.split("_")
    if len(parts) < 4:
        return None  # unexpected filename
    
    timestamp = parts[3]  # "20250812004105281472818770368"
    try:
        # Take first 10 chars: YYYYMMDDHH
        dt = datetime.strptime(timestamp[:10], "%Y%m%d%H")
        return dt
    except ValueError:
        return None

def _append_pipeline_log(prefix, message):
    log_path = PIPELINE_LOG_DIR / f"{prefix}_{cde_run_dte}.txt"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def extract_polarization(x):
    if "MODU-" in x:
        modu_part = x.split("MODU-")[1]  
        modu_number = modu_part[0]      
        return {'1': 'V', '2': 'H'}.get(modu_number, None)
    return None

def get_event_value_or_return_nan(group, event_name):
    """Get the value from the correct column based on the event name."""
    row = group[group['EventName'] == event_name]
    return row['Value'].values[0] if not row.empty else np.nan

def cml2metadata_coupling_framework(cml, metadat):
    """
    Function to couple signal level data (CML) with metadata and return the processed data in RAINLINK format.

    Parameters:
    cml (str): Path to the signal level data file.
    metadat (pd.DataFrame): Metadata dataframe.

    Returns:
    pd.DataFrame: Coupled data in RAINLINK format.
    """
    # print(f'Processing CML file: {os.path.basename(cml)}')
    cml_dat_df = pd.read_csv(cml, header=0, sep='\t')  # assumes headers are present and identical

    # Construct an ID to distinguish between sublinks, antennas, and polarizations across a single path
    cml_dat_df['Monitored_ID'] = (cml_dat_df['NEName'].astype(str) + '-' +
                                  cml_dat_df['BrdID'].astype(str) + '-' +
                                  cml_dat_df['BrdName'].astype(str) + '-' +
                                  cml_dat_df['PortNO'].astype(str) + '(' +
                                  cml_dat_df['PortName'].astype(str) + ')-' +
                                  cml_dat_df['PathID'].astype(str))

    # Add polarization column
    cml_dat_df['Polarization'] = cml_dat_df['Monitored_ID'].apply(extract_polarization)

    # Merge CML data with metadata
    cml_data = pd.merge(cml_dat_df, metadat, on=['Monitored_ID'], how='inner')

    # Check if the merge resulted in an empty dataframe
    if cml_data.empty:
        error_message = f"No common 'Monitored_ID' found for file: {os.path.basename(cml)}"
        print(error_message)
        _append_pipeline_log("unmatched_cml_files", error_message)
        return None  # Return None to indicate skipping further processing

    # Handle unmatched polarizations
    cml_data['Polarization'] = np.where((cml_data['Polarization_x'] != cml_data['Polarization_y']) &
                                        ~(cml_data['Polarization_x'].isin(['H', 'V'])),
                                        cml_data['Polarization_y'],
                                        cml_data['Polarization_x'])

    # Drop unnecessary columns
    cml_data = cml_data.drop(columns=['ONEID', 'ONEName', 'NEID', 'NEType',
                                      'NEName', 'BrdID', 'BrdName', 'PortNO', 'PortName', 'PathID',
                                      'ShelfID', 'BrdType', 'PortID', 'MOType',
                                      'FBName', 'EventID', 'PMParameterName', 'PMLocationID',
                                      'PMLocation', 'UpLevel', 'DownLevel', 'ResultOfLevel',
                                      'Unnamed: 27', 'Polarization_x', 'Temp_ID', 'Polarization_y',
                                      'ATPC'])

    # Columns to group by
    group_columns = ['Monitored_ID', 'Far_end_ID', 'Polarization', 'Period', 'EndTime']

    # Process TSL data
    df_tsl = cml_data.copy()
    flattened_tsl_rows = []
    for group_keys, group_data in df_tsl.groupby(group_columns):
        row = dict(zip(group_columns, group_keys))
        for event in ['TSL_MIN', 'TSL_MAX', 'TSL_CUR', 'TSL_AVG']:
            row[event] = get_event_value_or_return_nan(group_data, event)
        flattened_tsl_rows.append(row)
    tsl_flattened = pd.DataFrame(flattened_tsl_rows)

    # Process RSL data
    df_rsl = cml_data.copy()
    df_rsl = df_rsl.rename(columns={'Monitored_ID': 'Far_end_ID', 'Far_end_ID': 'Monitored_ID'})
    flattened_rsl_rows = []
    for group_keys, group_data in df_rsl.groupby(group_columns):
        row = dict(zip(group_columns, group_keys))
        for event in ['RSL_MIN', 'RSL_MAX', 'RSL_CUR', 'RSL_AVG']:
            row[event] = get_event_value_or_return_nan(group_data, event)
        flattened_rsl_rows.append(row)
    rsl_flattened = pd.DataFrame(flattened_rsl_rows)

    # Merge TSL and RSL dataframes
    if tsl_flattened.empty or rsl_flattened.empty:
        error_message = f"Skipping file {os.path.basename(cml)} due to insufficient data after processing."
        print(error_message)
        _append_pipeline_log("insufficient_data", error_message)
        return None  # Skip further processing if any intermediate result is empty

    cml_data_flattened = pd.merge(tsl_flattened, rsl_flattened, on=group_columns)

    # Add metadata columns back
    metadata_cols = ['Frequency', 'XStart', 'YStart', 'XEnd', 'YEnd', 'PathLength']
    cml_data_unique_metadata = cml_data[group_columns + metadata_cols].drop_duplicates()
    cml_data_flattened = pd.merge(cml_data_flattened, cml_data_unique_metadata, how='left', on=group_columns)

    # Filter and process data
    linkdata = cml_data_flattened.copy()
    linkdata = linkdata.dropna(subset=['TSL_MIN', 'TSL_MAX', 'TSL_AVG', 'TSL_CUR'])
    try:
        linkdata['RSL_MIN'] = pd.to_numeric(linkdata['RSL_MIN'], errors='coerce')
        linkdata['TSL_AVG'] = pd.to_numeric(linkdata['TSL_AVG'], errors='coerce')
        linkdata['RSL_MAX'] = pd.to_numeric(linkdata['RSL_MAX'], errors='coerce')
        
        linkdata['Pmin'] = linkdata['RSL_MIN'] - linkdata['TSL_AVG']
        linkdata['Pmax'] = linkdata['RSL_MAX'] - linkdata['TSL_AVG']
    except Exception as e:
        error_message = f"Error processing Pmin/Pmax of file {os.path.basename(cml)}: {e}"
        print(error_message)
        _append_pipeline_log("error_log", error_message)
        return None
    linkdata['ID'] = linkdata['Monitored_ID'] + '>>' + linkdata['Far_end_ID']
    linkdata = linkdata.drop(columns=['Monitored_ID', 'Far_end_ID', 'Period'])
    linkdata = linkdata.rename(columns={'EndTime': 'DateTime'})
    linkdata['Frequency'] = linkdata['Frequency'] / 1000  # convert to GHz
    linkdata['DateTime'] = pd.to_datetime(linkdata['DateTime'], utc=True).dt.strftime('%Y%m%d%H%M')

    # Reorder columns
    order_columns = ['Frequency', 'DateTime', 'Pmin', 'Pmax', 'XStart', 'YStart', 'XEnd', 'YEnd', 'ID',
                     'Polarization', 'PathLength', 'TSL_AVG']
    linkdata = linkdata[order_columns]

    return linkdata

def cml2metadata_coupling_framework_fast(cml, metadat):
    """
    Faster version of cml2metadata_coupling_framework.

    Purpose
    -------
    Couple signal-level CML data with metadata and return RAINLINK-style link data.

    Main speed improvement
    ----------------------
    Replaces slow Python groupby loops with vectorized pivot_table operations for
    TSL_* and RSL_* values.

    Core logic preserved
    --------------------
    - Builds Monitored_ID the same way.
    - Extracts polarization the same way.
    - Merges with metadata on Monitored_ID.
    - Builds reverse-direction RSL table by swapping Monitored_ID and Far_end_ID.
    - Merges TSL and RSL on:
        Monitored_ID, Far_end_ID, Polarization, Period, EndTime
    - Computes:
        Pmin = RSL_MIN - TSL_AVG
        Pmax = RSL_MAX - TSL_AVG
    - Creates:
        ID = Monitored_ID + '>>' + Far_end_ID
    """

    fname = os.path.basename(cml)

    # ------------------------------------------------------------
    # 1) Read CML file
    # ------------------------------------------------------------
    cml_dat_df = pd.read_csv(cml, header=0, sep="\t")

    # ------------------------------------------------------------
    # 2) Construct Monitored_ID exactly as before
    # ------------------------------------------------------------
    cml_dat_df["Monitored_ID"] = (
        cml_dat_df["NEName"].astype(str) + "-"
        + cml_dat_df["BrdID"].astype(str) + "-"
        + cml_dat_df["BrdName"].astype(str) + "-"
        + cml_dat_df["PortNO"].astype(str) + "("
        + cml_dat_df["PortName"].astype(str) + ")-"
        + cml_dat_df["PathID"].astype(str)
    )

    # Add polarization column
    cml_dat_df["Polarization"] = cml_dat_df["Monitored_ID"].apply(extract_polarization)

    # ------------------------------------------------------------
    # 3) Merge CML data with metadata
    # ------------------------------------------------------------
    cml_data = pd.merge(cml_dat_df, metadat, on=["Monitored_ID"], how="inner")

    if cml_data.empty:
        error_message = f"No common 'Monitored_ID' found for file: {fname}"
        print(error_message)
        log_path = f"/home/kkumah/Projects/cml-stuff/data-cml/metadata/unmatched_cml_files_{cde_run_dte}.txt"
        with open(log_path, "a") as f:
            f.write(error_message + "\n")
        return None

    # ------------------------------------------------------------
    # 4) Resolve polarization exactly as before
    # ------------------------------------------------------------
    cml_data["Polarization"] = np.where(
        (cml_data["Polarization_x"] != cml_data["Polarization_y"])
        & ~(cml_data["Polarization_x"].isin(["H", "V"])),
        cml_data["Polarization_y"],
        cml_data["Polarization_x"],
    )

    # ------------------------------------------------------------
    # 5) Drop unnecessary columns, but ignore missing columns safely
    # ------------------------------------------------------------
    drop_cols = [
        "ONEID", "ONEName", "NEID", "NEType",
        "NEName", "BrdID", "BrdName", "PortNO", "PortName", "PathID",
        "ShelfID", "BrdType", "PortID", "MOType",
        "FBName", "EventID", "PMParameterName", "PMLocationID",
        "PMLocation", "UpLevel", "DownLevel", "ResultOfLevel",
        "Unnamed: 27", "Polarization_x", "Temp_ID", "Polarization_y",
        "ATPC",
    ]

    cml_data = cml_data.drop(columns=[c for c in drop_cols if c in cml_data.columns])

    # ------------------------------------------------------------
    # 6) Define grouping columns and required event names
    # ------------------------------------------------------------
    group_columns = ["Monitored_ID", "Far_end_ID", "Polarization", "Period", "EndTime"]

    tsl_events = ["TSL_MIN", "TSL_MAX", "TSL_CUR", "TSL_AVG"]
    rsl_events = ["RSL_MIN", "RSL_MAX", "RSL_CUR", "RSL_AVG"]

    # ------------------------------------------------------------
    # 7) Fast TSL flattening using pivot_table
    # ------------------------------------------------------------
    # This replaces:
    #   groupby(group_columns) + get_event_value_or_return_nan(...)
    #
    # aggfunc="first" mimics your old behavior of taking the first matching value.
    # If duplicate event records occur within a group, this keeps the first one.
    # ------------------------------------------------------------
    df_tsl = cml_data[cml_data["EventName"].isin(tsl_events)].copy()

    tsl_flattened = (
        df_tsl
        .pivot_table(
            index=group_columns,
            columns="EventName",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
    )

    # Remove the column index name that pivot_table creates
    tsl_flattened.columns.name = None

    # Ensure all expected TSL columns exist
    for col in tsl_events:
        if col not in tsl_flattened.columns:
            tsl_flattened[col] = np.nan

    # ------------------------------------------------------------
    # 8) Fast RSL flattening using swapped IDs + pivot_table
    # ------------------------------------------------------------
    df_rsl = cml_data[cml_data["EventName"].isin(rsl_events)].copy()

    # Same as your original:
    # df_rsl = df_rsl.rename(columns={'Monitored_ID': 'Far_end_ID', 'Far_end_ID': 'Monitored_ID'})
    df_rsl = df_rsl.rename(
        columns={
            "Monitored_ID": "_tmp_Monitored_ID",
            "Far_end_ID": "Monitored_ID",
        }
    )
    df_rsl = df_rsl.rename(columns={"_tmp_Monitored_ID": "Far_end_ID"})

    rsl_flattened = (
        df_rsl
        .pivot_table(
            index=group_columns,
            columns="EventName",
            values="Value",
            aggfunc="first",
        )
        .reset_index()
    )

    rsl_flattened.columns.name = None

    # Ensure all expected RSL columns exist
    for col in rsl_events:
        if col not in rsl_flattened.columns:
            rsl_flattened[col] = np.nan

    # ------------------------------------------------------------
    # 9) Check empty TSL/RSL results
    # ------------------------------------------------------------
    if tsl_flattened.empty or rsl_flattened.empty:
        error_message = f"Skipping file {fname} due to insufficient data after processing."
        print(error_message)
        log_path = f"/home/kkumah/Projects/cml-stuff/data-cml/metadata/insufficient_data_{cde_run_dte}.txt"
        with open(log_path, "a") as f:
            f.write(error_message + "\n")
        return None

    # ------------------------------------------------------------
    # 10) Merge TSL and RSL flattened tables
    # ------------------------------------------------------------
    cml_data_flattened = pd.merge(
        tsl_flattened,
        rsl_flattened,
        on=group_columns,
        how="inner",
    )

    if cml_data_flattened.empty:
        error_message = f"Skipping file {fname}: no matching TSL/RSL directional pairs after flattening."
        print(error_message)
        log_path = f"/home/kkumah/Projects/cml-stuff/data-cml/metadata/insufficient_data_{cde_run_dte}.txt"
        with open(log_path, "a") as f:
            f.write(error_message + "\n")
        return None

    # ------------------------------------------------------------
    # 11) Add metadata columns back
    # ------------------------------------------------------------
    metadata_cols = ["Frequency", "XStart", "YStart", "XEnd", "YEnd", "PathLength"]

    available_metadata_cols = [c for c in metadata_cols if c in cml_data.columns]

    cml_data_unique_metadata = (
        cml_data[group_columns + available_metadata_cols]
        .drop_duplicates(subset=group_columns)
        .copy()
    )

    cml_data_flattened = pd.merge(
        cml_data_flattened,
        cml_data_unique_metadata,
        how="left",
        on=group_columns,
    )

    # ------------------------------------------------------------
    # 12) Filter and compute Pmin/Pmax
    # ------------------------------------------------------------
    linkdata = cml_data_flattened.copy()

    # Same filter as original
    linkdata = linkdata.dropna(subset=["TSL_MIN", "TSL_MAX", "TSL_AVG", "TSL_CUR"])

    if linkdata.empty:
        error_message = f"Skipping file {fname}: no valid TSL rows after required TSL filtering."
        print(error_message)
        log_path = f"/home/kkumah/Projects/cml-stuff/data-cml/metadata/insufficient_data_{cde_run_dte}.txt"
        with open(log_path, "a") as f:
            f.write(error_message + "\n")
        return None

    try:
        linkdata["RSL_MIN"] = pd.to_numeric(linkdata["RSL_MIN"], errors="coerce")
        linkdata["TSL_AVG"] = pd.to_numeric(linkdata["TSL_AVG"], errors="coerce")
        linkdata["RSL_MAX"] = pd.to_numeric(linkdata["RSL_MAX"], errors="coerce")

        linkdata["Pmin"] = linkdata["RSL_MIN"] - linkdata["TSL_AVG"]
        linkdata["Pmax"] = linkdata["RSL_MAX"] - linkdata["TSL_AVG"]

    except Exception as e:
        error_message = f"Error processing Pmin/Pmax of file {fname}: {e}"
        print(error_message)
        log_path = f"/home/kkumah/Projects/cml-stuff/data-cml/metadata/error_log_{cde_run_dte}.txt"
        with open(log_path, "a") as f:
            f.write(error_message + "\n")
        return None

    # ------------------------------------------------------------
    # 13) Build ID, format time, frequency
    # ------------------------------------------------------------
    linkdata["ID"] = linkdata["Monitored_ID"].astype(str) + ">>" + linkdata["Far_end_ID"].astype(str)

    linkdata = linkdata.drop(columns=["Monitored_ID", "Far_end_ID", "Period"])
    linkdata = linkdata.rename(columns={"EndTime": "DateTime"})

    linkdata["Frequency"] = pd.to_numeric(linkdata["Frequency"], errors="coerce") / 1000.0

    linkdata["DateTime"] = (
        pd.to_datetime(linkdata["DateTime"], utc=True, errors="coerce")
        .dt.strftime("%Y%m%d%H%M")
    )

    # ------------------------------------------------------------
    # 14) Reorder columns exactly as before
    # ------------------------------------------------------------
    order_columns = [
        "Frequency", "DateTime", "Pmin", "Pmax",
        "XStart", "YStart", "XEnd", "YEnd",
        "ID", "Polarization", "PathLength", "TSL_AVG",
    ]

    # Make sure missing expected columns exist as NaN
    for col in order_columns:
        if col not in linkdata.columns:
            linkdata[col] = np.nan

    linkdata = linkdata[order_columns]

    return linkdata
