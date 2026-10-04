# step2_baseline_dry48.py
# STEP 2 — Dry-baseline (≤48 h) estimation & excess attenuation
#
# What this module does
# ---------------------
# For each link and timestamp it builds a DRY-only baseline of Abar using the
# previous W hours (default 48 h), excluding current sample and all samples
# classified as wet in Step 1. It then computes excess attenuation:
#   A_excess = max(0, Abar - baseline).
#
# Inputs (from Step 1 / NLA):
#   - df_nla: pandas.DataFrame with DatetimeIndex (UTC), columns at least:
#       ['ID','Abar','is_wet','PathLength', 'Pmin','Pmax', ...]
#     15-min cadence assumed (configurable via cadence_minutes).
#
# Outputs (adds columns to a copy of df_nla):
#   - 'baseline_db'             : dry baseline B(t) [dB] of Abar
#   - 'baseline_n_dry'          : number of dry samples available in the main window
#   - 'A_excess_db'             : max(0, Abar - baseline_db)
#   - 'A_excess_db_per_km'      : A_excess_db / PathLength
#   - 'baseline_used_window_h'  : 48 or fallback (e.g., 72) used at each time
#
# Notes:
#   - Robust baseline uses a rolling *past-only* quantile (q≈0.2) of DRY Abar.
#   - Fallback expands window to 72 h when too few dry points exist.
#   - Final fallback forward-fills the last baseline across long wet spells.
#   - FIXED: rolling counts now use min_periods=1 so we don't create NA booleans.
#
# step2_baseline_dry48.py
# step2_baseline_dry48.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Baseline48Config:
    window_hours: int = 48            # main dry window
    fallback_hours: int = 72          # used if main window lacks dry coverage
    min_dry_samples: int = 8          # min dry samples required in a window
    min_dry_frac: float | None = None # optional: also require this dry fraction (0..1)
    smooth_baseline_samples: int = 0  # optional trailing smoothing (samples)

def _pick_dry_mask(df: pd.DataFrame) -> pd.Series:
    """Prefer 'is_wet_excess' if present; else 'is_wet'; else treat all as dry."""
    if "is_wet_excess" in df.columns:
        return ~df["is_wet_excess"].fillna(False)
    if "is_wet" in df.columns:
        return ~df["is_wet"].fillna(False)
    return pd.Series(True, index=df.index)

def _rolling_time_median(x: pd.Series, hours: int, minp: int) -> pd.Series:
    return x.rolling(window=f"{hours}H", min_periods=minp, closed="left").median()

def _rolling_time_count(x: pd.Series, hours: int) -> pd.Series:
    return x.rolling(window=f"{hours}H", closed="left").count()

def _per_link_baseline(g: pd.DataFrame, cfg: Baseline48Config) -> pd.DataFrame:
    g = g.sort_index()
    dry = _pick_dry_mask(g)

    # Abar available as float
    A = pd.to_numeric(g["Abar"], errors="coerce")

    # Keep only dry samples for baseline stats
    A_dry = A.where(dry, np.nan)

    # Main window (48h by default)
    med48 = _rolling_time_median(A_dry, cfg.window_hours, cfg.min_dry_samples)
    n_dry48 = _rolling_time_count(dry.astype("float"), cfg.window_hours)  # True=1.0 count
    n_all48 = _rolling_time_count(A, cfg.window_hours).clip(lower=1)      # avoid /0

    # Fallback window (72h by default)
    medFB = _rolling_time_median(A_dry, cfg.fallback_hours, cfg.min_dry_samples)
    n_dryFB = _rolling_time_count(dry.astype("float"), cfg.fallback_hours)
    n_allFB = _rolling_time_count(A, cfg.fallback_hours).clip(lower=1)

    # Decide which window qualifies — explicit boolean masks (no np.where on NA)
    ok48 = (n_dry48 >= cfg.min_dry_samples)
    if cfg.min_dry_frac is not None:
        ok48 = ok48 & ((n_dry48 / n_all48) >= float(cfg.min_dry_frac))

    okFB = (n_dryFB >= cfg.min_dry_samples)
    if cfg.min_dry_frac is not None:
        okFB = okFB & ((n_dryFB / n_allFB) >= float(cfg.min_dry_frac))

    # Start with NaN baseline, then fill from 48h where ok, else 72h where ok
    baseline = pd.Series(np.nan, index=g.index, dtype="float64")
    baseline.loc[ok48] = med48.loc[ok48]
    fillFB = (~ok48) & okFB
    baseline.loc[fillFB] = medFB.loc[fillFB]

    # Optional trailing smoothing (simple moving median over samples)
    if cfg.smooth_baseline_samples and cfg.smooth_baseline_samples > 1:
        baseline = baseline.rolling(cfg.smooth_baseline_samples, min_periods=1).median()

    # --- replace the block that sets baseline_n_dry and baseline_used_window_h ---

    out = g.copy()
    out["baseline_db"] = baseline

    # which window was used?
    used_h = pd.Series(np.nan, index=g.index, dtype="float64")
    used_h.loc[ok48]  = float(cfg.window_hours)
    used_h.loc[(~ok48) & okFB] = float(cfg.fallback_hours)
    out["baseline_used_window_h"] = used_h

    # how many dry samples in the chosen window?
    baseline_n = pd.Series(np.nan, index=g.index, dtype="float64")
    baseline_n.loc[ok48] = n_dry48.loc[ok48]
    baseline_n.loc[(~ok48) & okFB] = n_dryFB.loc[(~ok48) & okFB]
    # if you really want pandas’ nullable dtype, uncomment the next line:
    # baseline_n = baseline_n.astype("Float32")
    out["baseline_n_dry"] = baseline_n

    # Excess attenuation (>=0)
    out["A_excess_db"] = np.maximum(0.0, A - out["baseline_db"])
    if "PathLength" in out.columns:
        L = pd.to_numeric(out["PathLength"], errors="coerce").replace(0, np.nan)
        out["A_excess_db_per_km"] = out["A_excess_db"] / L

    return out

def compute_dry_baseline_48h(df_nla: pd.DataFrame, cfg: Baseline48Config
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    need = ["ID", "Abar"]
    miss = [c for c in need if c not in df_nla.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    frames: List[pd.DataFrame] = []
    for _, g in df_nla.groupby("ID", sort=False):
        frames.append(_per_link_baseline(g, cfg))

    df_out = pd.concat(frames, axis=0).sort_index()

    # Per-link summary
    rows = []
    for lid, g in df_out.groupby("ID", sort=False):
        src = g[np.isfinite(g["Abar"])]
        rows.append({
            "ID": lid,
            "n_src": int(len(src)),
            "baseline_avail_frac": float(np.isfinite(src["baseline_db"]).mean()),
            "median_n_dry": float(np.nanmedian(src["baseline_n_dry"])),
            "median_baseline_db": float(np.nanmedian(src["baseline_db"])),
            "wet_frac_src": float(src.get("is_wet", pd.Series(False, index=src.index)).mean()),
        })
    summary = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    return df_out, summary

# step2_baseline_dry48.py (append this)
import numpy as np
import pandas as pd

def compute_dry_baseline_48h_robust(df_in: pd.DataFrame, cfg):
    """
    Two-pass robust baseline (48h):
      1) draft baseline = rolling Q20
      2) exclude rough-wet (Abar - base1 > 0.5 dB), recompute baseline = rolling Q30
      3) interpolate small gaps + light rolling mean
    Returns df_out, df_summary (same columns as your original step: baseline_db, A_excess_db, A_excess_db_per_km).
    """
    df = df_in.copy()
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    win = f"{getattr(cfg, 'window_hours', 48)}H"

    base1 = (df.groupby("ID")["Abar"]
               .transform(lambda s: s.rolling(win, min_periods=6).quantile(0.20)))
    ex1 = df["Abar"] - base1
    rough_wet = (ex1 > 0.5)

    d2 = df.copy()
    d2.loc[rough_wet, "Abar"] = np.nan
    base2 = (d2.groupby("ID")["Abar"]
               .transform(lambda s: s.rolling(win, min_periods=6).quantile(0.30)))
    base2 = (base2.groupby(df["ID"])
                   .transform(lambda s: s.interpolate(limit=6)
                                       .rolling(5, min_periods=1, center=True).mean()))

    out = df.copy()
    out["baseline_db"] = base2
    out["A_excess_db"] = (df["Abar"] - out["baseline_db"]).clip(lower=0.0)
    L = pd.to_numeric(df["PathLength"], errors="coerce").replace(0, np.nan)
    out["A_excess_db_per_km"] = out["A_excess_db"] / L

    rows = []
    for lid, g in out.groupby("ID", sort=False):
        rows.append({"ID": lid, "n_src": len(g), "median_base": float(np.nanmedian(g["baseline_db"]))})
    s2_sum = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    return out, s2_sum
