#%%
'''
CML2Rainfall_program_utils_20250927.py program utilities and functions
'''

#%%
# Import packages

import os, re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

#%%
# Global constants, parameters and variables
# --- Window parameters ---
REQUIRED_HOURS         = 51.0   # strict Rainlink-style history length
FREQ_MINUTES           = 15     # native resolution of dumps
MAX_FILE_LOOKBACK_H    = 7.0    # how far the earliest file may sit *after* window start
SOFT_MIN_HOURS         = 36.0   # if STRICT_MODE=False we can allow >= this
STRICT_MODE            = True   # True => 51 h required by archive timestamps

# --- Coverage targets (used later in R3/R4 after reading) ---
PER_LINK_MIN_COVER_PCT = 70.0   # per-link min coverage inside the window
GLOBAL_MIN_COVER_PCT   = 85.0   # fraction of links that must pass per-link coverage
WRITE_INTERMEDIATE     = True   # save diags even if coverage fails (R3/R4 stage)

# --- Optional: force a single target datetime (UTC) ---
# If set, we plan for *only this* target; otherwise we plan for all possible targets.
MANUAL_TARGET_DT_UTC = None  # e.g., datetime(2025,8,24,6,0,tzinfo=timezone.utc)


# ---------- tuning for post-read (in-file) coverage ----------
POSTREAD_STRICT             = False     # moderate tolerance (as discussed)
POSTREAD_EDGE_SLACK_MIN     = 60        # allow up to 60 min slack at each edge
POSTREAD_MIN_SPAN_H         = 50.0      # require at least 50.0 h real span

#%%

def extract_dt_from_filename(fname: str):
    """
    Parse YYYYMMDDHH from 'Schedule_pfm_SDH_<TIMESTAMP>_*.txt' and return a tz-aware UTC datetime.
    We use the first 10 digits of the long timestamp field.
    """
    # Examples: Schedule_pfm_SDH_20250824064454281472028357056_1.txt
    m = re.search(r"Schedule_pfm_SDH_(\d{10})", fname)
    if not m:
        return None
    try:
        dt_naive = datetime.strptime(m.group(1), "%Y%m%d%H")
        return dt_naive.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
# ---------- list all valid files in a directory ----------
def list_files_with_dt(root_dir: str):
    """
    Return a sorted list of (dt_utc, filename) for all *.txt files that match the scheme.
    Sorted ascending by datetime.
    """
    items = []
    for fn in os.listdir(root_dir):
        if not fn.endswith(".txt"):
            continue
        dt = extract_dt_from_filename(fn)
        if dt is not None:
            items.append((dt, fn))
    items.sort(key=lambda x: x[0])
    return items
# ---------- select files for a single window ----------
def select_files_for_window(all_files,
                            target_dt_utc: pd.Timestamp,
                            required_hours: float = REQUIRED_HOURS,
                            max_file_lookback_h: float = MAX_FILE_LOOKBACK_H,
                            strict_mode: bool = STRICT_MODE,
                            soft_min_hours: float = SOFT_MIN_HOURS):
    """
    Given a sorted list of (dt_utc, filename), choose the minimal set of files that
    *should* cover [target-51h, target] based on file timestamps alone.

    Returns:
        dict with keys:
          - ok (bool)
          - reason (str)
          - window_start_utc (Timestamp)
          - archive_span_h (float)
          - selected (list[(dt_utc, filename)])
    """
    if not all_files:
        return dict(ok=False, reason="Archive empty.", window_start_utc=None,
                    archive_span_h=0.0, selected=[])

    target_dt_utc = pd.to_datetime(target_dt_utc, utc=True)
    window_start_utc = target_dt_utc - timedelta(hours=required_hours)

    # Walk backward from newest until we pass window_start
    selected = []
    accum_start = target_dt_utc
    for dt, fname in reversed(all_files):
        if dt <= target_dt_utc:
            selected.append((dt, fname))
            if dt < accum_start:
                accum_start = dt
            if accum_start <= window_start_utc:
                break
    selected.sort(key=lambda x: x[0])

    if not selected:
        return dict(ok=False, reason="No files at or before target.", window_start_utc=window_start_utc,
                    archive_span_h=0.0, selected=[])

    earliest = selected[0][0]
    latest   = selected[-1][0]
    archive_span_h = (latest - earliest) / pd.Timedelta(hours=1)

    # Quick check: latest must be >= target
    if latest < target_dt_utc:
        return dict(ok=False,
                    reason=f"Earliest-latest span ends before target ({latest} < {target_dt_utc}).",
                    window_start_utc=window_start_utc, archive_span_h=archive_span_h, selected=selected)

    # How far after the desired window start does the earliest file begin?
    gap_h = (earliest - window_start_utc) / pd.Timedelta(hours=1)
    if gap_h > max_file_lookback_h:
        reason = (f"Archive gap near window start too large: earliest {earliest}, start {window_start_utc}, "
                  f"gap ≈ {gap_h:.2f} h > MAX_FILE_LOOKBACK_H={max_file_lookback_h} h.")
        if strict_mode:
            return dict(ok=False, reason=reason, window_start_utc=window_start_utc,
                        archive_span_h=archive_span_h, selected=selected)
        else:
            # Soft mode: allow if we still have enough span for Rainlink-ish behavior
            if archive_span_h >= soft_min_hours:
                return dict(ok=True,  reason="SOFT_OK: start gap but span >= soft minimum.",
                            window_start_utc=window_start_utc, archive_span_h=archive_span_h, selected=selected)
            return dict(ok=False, reason=reason + f" Span {archive_span_h:.2f} h < soft_min {soft_min_hours} h.",
                        window_start_utc=window_start_utc, archive_span_h=archive_span_h, selected=selected)

    # Otherwise we’re good (strict 51h by filenames)
    return dict(ok=True, reason="STRICT_OK", window_start_utc=window_start_utc,
                archive_span_h=archive_span_h, selected=selected)

# ---------- plan all possible windows ----------
def plan_batch_windows(all_files,
                       required_hours: float = REQUIRED_HOURS,
                       max_file_lookback_h: float = MAX_FILE_LOOKBACK_H,
                       strict_mode: bool = STRICT_MODE,
                       soft_min_hours: float = SOFT_MIN_HOURS,
                       dedup_windows: bool = True):
    """
    Evaluate *every* file timestamp as a potential target and return:
      - jobs_ok:   list of job dicts (each contains target_dt_utc and selected files)
      - jobs_fail: list of (target_dt_utc, reason)

    If dedup_windows=True, we avoid yielding multiple jobs that would read the exact same
    [earliest→latest] file set (useful when files are hours apart but windows overlap perfectly).
    """
    jobs_ok, jobs_fail = [], []
    seen_windows = set()

    for target_dt_utc, _fname in all_files:
        sel = select_files_for_window(
            all_files=all_files,
            target_dt_utc=target_dt_utc,
            required_hours=required_hours,
            max_file_lookback_h=max_file_lookback_h,
            strict_mode=strict_mode,
            soft_min_hours=soft_min_hours
        )
        if sel["ok"]:
            # Optional de-dup: fingerprint by (earliest_dt, latest_dt)
            earliest = sel["selected"][0][0]
            latest   = sel["selected"][-1][0]
            fp = (earliest, latest)
            if (not dedup_windows) or (fp not in seen_windows):
                seen_windows.add(fp)
                jobs_ok.append({
                    "target_dt_utc": target_dt_utc,
                    "window_start_utc": sel["window_start_utc"],
                    "archive_span_h": sel["archive_span_h"],
                    "selected": sel["selected"],
                    "reason": sel["reason"],
                })
        else:
            jobs_fail.append((target_dt_utc, sel["reason"]))

    return jobs_ok, jobs_fail

# ---------- helpers (reuse from R3 single) ----------
def ensure_utc(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, errors="coerce")
    if getattr(t.dt, "tz", None) is None:
        t = t.dt.tz_localize("UTC")
    else:
        t = t.dt.tz_convert("UTC")
    return t
# ---------- read a single dump file ----------
def read_txt_dump(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t", header=0, low_memory=False)
    except Exception as e:
        print(f"⚠️ Could not read {os.path.basename(path)}: {e}")
        return pd.DataFrame()
    df.columns = [c.strip() for c in df.columns]
    df["source_file"] = os.path.basename(path)
    return df
# ---------- post-read coverage check ----------
def postread_ok(tmin, tmax, window_start_utc, target_dt_utc) -> (bool, str):
    """Moderate tolerance rule."""
    span_h = (tmax - tmin) / pd.Timedelta(hours=1)

    # slack windows
    lo_ok = (tmin <= window_start_utc) or ((tmin - window_start_utc) <= pd.Timedelta(minutes=POSTREAD_EDGE_SLACK_MIN))
    hi_ok = (tmax >= target_dt_utc)     or ((target_dt_utc - tmax) <= pd.Timedelta(minutes=POSTREAD_EDGE_SLACK_MIN))
    span_ok = span_h >= POSTREAD_MIN_SPAN_H if not POSTREAD_STRICT else span_h >= REQUIRED_HOURS - 1e-9

    if lo_ok and hi_ok and span_ok:
        return True, f"OK (span={span_h:.2f} h, edge slack ≤ {POSTREAD_EDGE_SLACK_MIN} min)"
    # build reason
    reasons = []
    if not lo_ok:
        reasons.append(f"start short by {(tmin - window_start_utc) / pd.Timedelta(minutes=1):.0f} min")
    if not hi_ok:
        reasons.append(f"end short by {(target_dt_utc - tmax) / pd.Timedelta(minutes=1):.0f} min")
    if not span_ok:
        reasons.append(f"span {span_h:.2f} h < min {POSTREAD_MIN_SPAN_H:.2f} h")
    return False, " & ".join(reasons)

