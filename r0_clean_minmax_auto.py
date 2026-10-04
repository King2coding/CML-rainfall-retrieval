# r0_clean_minmax_auto.py
# R0 — Prior cleaning for 15-min Pmin/Pmax with per-link semantics (RSL dBm vs TL dB)
# Upgrades:
#   - Robust semantics detection using in-range counts
#   - Auto-retry with flipped semantics if first pass nukes a link
#   - src_present flag for QA (distinguish true samples from regularized grid)
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Dict
import numpy as np
import pandas as pd


@dataclass(frozen=True)
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


# ---------- helpers ----------
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


# ---------- single-link cleaning ----------
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
    s["Pbar"] = (s["Pmin"] + s["Pmax"]) / 2.0
    s["Pspread"] = s["Pmax"] - s["Pmin"]
    s["semantics"] = semantics
    s["Abar"] = np.where(s["semantics"] == "tl", s["Pbar"], -s["Pbar"])

    # Metrics (only over true source samples)
    src = s[s["src_present"]]
    metrics = {
        "n_src": int(src.shape[0]),
        "n_valid_pairs_src": int(src[["Pmin", "Pmax"]].dropna().shape[0]),
        "frac_oob_src": float(src["qc_flags"].str.contains("OOB").mean()) if len(src) else 0.0,
        "frac_valid_src": float(src[["Pmin", "Pmax"]].dropna().shape[0] / max(len(src), 1))
    }
    return s, metrics


# ---------- public API ----------
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
    df_summary = pd.DataFrame(summaries).sort_values(["ID"]).reset_index(drop=True)
    return df_out, df_summary


if __name__ == "__main__":
    pass
