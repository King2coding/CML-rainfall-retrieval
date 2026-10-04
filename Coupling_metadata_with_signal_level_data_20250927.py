'''
This script demonstrates the coupling of metadata with signal-level data for enhanced analysis and interpretation.
'''

#%%
# import necessary libraries
from CML2Rainfall_program_utils_20250927 import *

#%% 
# ================================================================
# SECTION R0: Imports & Configuration  (Batch-aware)
# ================================================================
from datetime import datetime, timedelta, timezone
import os
import re
import pandas as pd

# --- Paths ---
CML_DIR   = r'/home/kkumah/Projects/cml-stuff/data-cml/rsl'   # folder with *.txt dumps
META_CSV  = r'/home/kkumah/Projects/cml-stuff/data-cml/outs/matched_metadata_kkk_20250527.csv'
OUT_DIR   = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'


# ================================================================
# SECTION R1: Discover available files and extract datetimes
# ================================================================

# --- Run R1 ---
all_files = list_files_with_dt(CML_DIR)
print(f"Discovered {len(all_files)} files")
if all_files:
    print("First 3:", all_files[:3])
    print("Last 3:", all_files[-3:])


# ================================================================
# SECTION R2: Strict window selector (single-target) + Batch planner
# ================================================================

# ---- Single-target mode (if MANUAL_TARGET_DT_UTC is set) ----
if MANUAL_TARGET_DT_UTC:
    one = select_files_for_window(all_files, MANUAL_TARGET_DT_UTC,
                                  required_hours=REQUIRED_HOURS,
                                  max_file_lookback_h=MAX_FILE_LOOKBACK_H,
                                  strict_mode=STRICT_MODE,
                                  soft_min_hours=SOFT_MIN_HOURS)
    print("\n[Single-target selection]")
    print("Target datetime (UTC):", pd.to_datetime(MANUAL_TARGET_DT_UTC, utc=True))
    if one["ok"]:
        print(f"Window start (UTC):   {one['window_start_utc']}")
        print(f"Selected {len(one['selected'])} files covering {one['selected'][0][0]} → {one['selected'][-1][0]}  (~{one['archive_span_h']:.2f} h by filenames)")
        print("Status:", one["reason"])
    else:
        print("Status: FAIL")
        print("Reason:", one["reason"])
else:
    # ---- Batch planner over *all* file timestamps ----
    jobs_ok, jobs_fail = plan_batch_windows(
        all_files,
        required_hours=REQUIRED_HOURS,
        max_file_lookback_h=MAX_FILE_LOOKBACK_H,
        strict_mode=STRICT_MODE,
        soft_min_hours=SOFT_MIN_HOURS,
        dedup_windows=True  # change to False to see every target even if identical window
    )

    print("\n[Batch window planning]")
    print(f"Total candidate targets: {len(all_files)}")
    print(f"Windows ready (ok):      {len(jobs_ok)}")
    print(f"Windows rejected:        {len(jobs_fail)}")

    # Preview a few OK jobs
    for j in jobs_ok[:3]:
        sel = j["selected"]
        print("\nOK job → target:", j["target_dt_utc"])
        print("  window start:", j["window_start_utc"])
        print(f"  files: {len(sel)} covering {sel[0][0]} → {sel[-1][0]}  (~{j['archive_span_h']:.2f} h)")
        print("  status:", j["reason"])

    # Preview a few FAIL reasons
    for t, reason in jobs_fail[:3]:
        print("\nFAIL job → target:", t)
        print("  reason:", reason)

# NOTE:
# - Next sections (R3+) will consume a chosen job (from jobs_ok)
#   and actually read the files, run the *post-read* coverage & gap checks,
#   link with metadata, and write the RAINLINK-ready .dat for that target.

#%%
# ================================================================
# SECTION R3B: Batch read + post-read coverage checks + persist
# ================================================================
import os
import pandas as pd
import numpy as np


# ---------- batch processor ----------
if "jobs_ok" not in globals() or not jobs_ok:
    raise RuntimeError("R3B: no OK jobs from R2. Nothing to process.")

os.makedirs(OUT_DIR, exist_ok=True)
manifest = []  # rows of results

print(f"\n[R3B] Processing {len(jobs_ok)} planned windows…")
for idx, job in enumerate(jobs_ok, 1):
    tgt   = pd.to_datetime(job["target_dt_utc"], utc=True)
    start = pd.to_datetime(job["window_start_utc"], utc=True)
    sel   = job["selected"]

    # read & combine
    parts = []
    for dt, fname in sel:
        path = os.path.join(CML_DIR, fname)
        df = read_txt_dump(path)
        if not df.empty:
            parts.append(df)

    if not parts:
        reason = "all selected files failed to read"
        print(f"[{idx:04d}] {tgt}  FAIL: {reason}")
        manifest.append(dict(target_dt_utc=tgt, window_start_utc=start, n_files=len(sel),
                             n_rows=0, ok=False, reason=reason, out_path=None))
        continue

    raw = pd.concat(parts, ignore_index=True)
    if "EndTime" not in raw.columns:
        reason = "missing EndTime column"
        print(f"[{idx:04d}] {tgt}  FAIL: {reason}")
        manifest.append(dict(target_dt_utc=tgt, window_start_utc=start, n_files=len(sel),
                             n_rows=0, ok=False, reason=reason, out_path=None))
        continue

    raw["EndTime"] = ensure_utc(raw["EndTime"])
    raw = raw.drop_duplicates()

    # strict trim to target window (defensive)
    mask = (raw["EndTime"] >= start) & (raw["EndTime"] <= tgt)
    win = raw.loc[mask].copy()

    if win.empty:
        reason = "no rows inside window after trim"
        print(f"[{idx:04d}] {tgt}  FAIL: {reason}")
        manifest.append(dict(target_dt_utc=tgt, window_start_utc=start, n_files=len(sel),
                             n_rows=0, ok=False, reason=reason, out_path=None))
        continue

    tmin, tmax = win["EndTime"].min(), win["EndTime"].max()
    ok, why = postread_ok(tmin, tmax, start, tgt)

    if not ok:
        print(f"[{idx:04d}] {tgt}  FAIL: {why}  "
              f"(have {tmin} → {tmax})")
        manifest.append(dict(target_dt_utc=tgt, window_start_utc=start, n_files=len(sel),
                             n_rows=len(win), ok=False, reason=why, out_path=None))
        continue

    # persist per-target window (compact parquet + small csv preview)
    stamp = tgt.strftime("%Y%m%d%H%M")
    out_parquet = os.path.join(OUT_DIR, f"raw_window_{stamp}.parquet")
    out_csv     = os.path.join(OUT_DIR, f"raw_window_{stamp}_head.csv")

    try:
        win.to_parquet(out_parquet, index=False)
        win.head(2000).to_csv(out_csv, index=False)  # small peek for quick inspection
        print(f"[{idx:04d}] {tgt}  ✅ SAVED  rows={len(win)}  "
              f"span≈{(tmax-tmin)/pd.Timedelta(hours=1):.2f} h  → {os.path.basename(out_parquet)}")
        manifest.append(dict(target_dt_utc=tgt, window_start_utc=start, n_files=len(sel),
                             n_rows=len(win), ok=True, reason=why, out_path=out_parquet))
    except Exception as e:
        reason = f"save error: {e}"
        print(f"[{idx:04d}] {tgt}  FAIL: {reason}")
        manifest.append(dict(target_dt_utc=tgt, window_start_utc=start, n_files=len(sel),
                             n_rows=len(win), ok=False, reason=reason, out_path=None))

# write manifest
man_df = pd.DataFrame(manifest).sort_values(["ok","target_dt_utc"], ascending=[False, True])
man_path = os.path.join(OUT_DIR, "R3B_manifest.csv")
man_df.to_csv(man_path, index=False)

print("\n[R3B] Done.")
print("Summary:")
print("  OK windows:   ", int(man_df["ok"].sum()))
print("  FAILED windows:", int((~man_df["ok"]).sum()))
print(f"  Manifest:      {man_path}")

# - - - - - - 
man_df = pd.read_csv(man_path)
# Load all valid windows
valid_jobs = man_df[man_df.ok]

# Pick one window
row = valid_jobs.iloc[0]
print("Target:", row.target_dt_utc, "File:", row.out_path)

# Read its concatenated dataframe
df = pd.read_parquet(row.out_path)
print(df.shape, df["EndTime"].min(), "→", df["EndTime"].max())
df.columns
df.head(3)
# For downstream sections:
# - R4 can iterate man_df[man_df.ok] and for each row read the parquet at out_path,
#   then do metadata linking, flattening, and Rainlink formatting.

#%%
#%%
# ================================================================
# SECTION R4: Load R3 manifest and prepare window(s) for next steps
#   - Robust to old/new manifest schemas
#   - Robust to NaN/missing paths
#   - Handles Parquet or CSV seamlessly
#   - No wet/dry yet — this is staging & sanity
# ================================================================

import os
import pandas as pd

MANIFEST = os.path.join(OUT_DIR, "R3B_manifest.csv")

# ---------------------------
# Helpers
# ---------------------------
def _normalize_path(p):
    """
    Return absolute path for a manifest path entry, or None if missing/invalid.
    - Accepts NaN/None/non-str -> coerces or returns None
    - Resolves relative paths against OUT_DIR
    """
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return None
    if not isinstance(p, str):
        try:
            p = str(p)
        except Exception:
            return None
    p = p.strip()
    if not p:
        return None
    return p if os.path.isabs(p) else os.path.join(OUT_DIR, p)

def _safe_load_window(path):
    """
    Load a concatenated raw window produced in R3.
    - If *.parquet -> read with pyarrow engine
    - Else treat as CSV (utf-8-sig; fallback to latin-1)
    Always parses EndTime to datetime (UTC) when present.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".parquet":
        df = pd.read_parquet(path, engine="pyarrow")
    else:
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")

    # Normalize EndTime if present
    if "EndTime" in df.columns:
        df["EndTime"] = pd.to_datetime(df["EndTime"], utc=True, errors="coerce")
        df = df.dropna(subset=["EndTime"])
    return df

def _read_time_span_only(path):
    """
    Fast span calculator for a raw window file without loading everything.
    Works for parquet or CSV; only reads EndTime.
    Returns span (hours) or NaN if not computable.
    """
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".parquet":
            # Read just EndTime column (pyarrow supports column projection)
            df = pd.read_parquet(path, columns=["EndTime"], engine="pyarrow")
        else:
            df = pd.read_csv(path, usecols=["EndTime"], encoding="utf-8-sig")
        tt = pd.to_datetime(df["EndTime"], utc=True, errors="coerce").dropna()
        if len(tt) == 0:
            return float("nan")
        return (tt.max() - tt.min()) / pd.Timedelta(hours=1)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(path, usecols=["EndTime"], encoding="latin-1")
            tt = pd.to_datetime(df["EndTime"], utc=True, errors="coerce").dropna()
            if len(tt) == 0:
                return float("nan")
            return (tt.max() - tt.min()) / pd.Timedelta(hours=1)
        except Exception:
            return float("nan")
    except Exception:
        return float("nan")

# ---------------------------
# 1) Read & normalize manifest
# ---------------------------
man_df = pd.read_csv(MANIFEST)

# (a) Map older column names → new names we expect downstream
if "raw_csv_path" not in man_df.columns and "out_path" in man_df.columns:
    man_df = man_df.rename(columns={"out_path": "raw_csv_path"})

# (b) Ensure tz-aware datetimes
for col in ["target_dt_utc", "window_start_utc"]:
    if col in man_df.columns:
        man_df[col] = pd.to_datetime(man_df[col], utc=True, errors="coerce")

# (c) Ensure archive_span_h exists and fill where missing/NaN using file content
if "archive_span_h" not in man_df.columns:
    man_df["archive_span_h"] = pd.NA

mask_need_span = man_df["archive_span_h"].isna()
abs_paths = man_df.get("raw_csv_path", pd.Series([pd.NA]*len(man_df))).apply(_normalize_path)

spans = []
for need, p in zip(mask_need_span.tolist(), abs_paths.tolist()):
    if not need:
        spans.append(pd.NA)                 # keep pre-existing values as-is
        continue
    if p is None or not os.path.exists(p):
        spans.append(float("nan"))          # cannot compute
        continue
    spans.append(_read_time_span_only(p))

man_df.loc[mask_need_span, "archive_span_h"] = spans

# (d) Final sanity on required columns
req_cols = {
    "target_dt_utc", "window_start_utc",
    "n_files", "n_rows", "ok", "reason",
    "raw_csv_path", "archive_span_h"
}
missing = req_cols - set(man_df.columns)
if missing:
    raise RuntimeError(f"Manifest missing columns after normalization: {missing}")

print(f"[R4] Loaded manifest with {len(man_df)} windows")
print(man_df.head(3)[["target_dt_utc","window_start_utc","n_files","archive_span_h","reason"]])

# ---------------------------
# 2) Choose which windows to process here
#    (process all for now; filter later if needed)
# ---------------------------
to_process = man_df.copy()

# ---------------------------
# 3) Iterate windows and load the staged raw data
# ---------------------------
for _, row in to_process.iterrows():
    target_dt = row["target_dt_utc"]
    win_start = row["window_start_utc"]
    raw_path  = _normalize_path(row["raw_csv_path"])
    status    = row["reason"]
    n_files   = int(row["n_files"])
    span_h    = float(row["archive_span_h"]) if pd.notna(row["archive_span_h"]) else float("nan")

    if raw_path is None or not os.path.exists(raw_path):
        print(f"⚠️  Skipping {target_dt} — raw window path missing or not found.")
        continue

    print("\n[R4] Loading window:")
    print(f"  target: {target_dt}   window_start: {win_start}")
    print(f"  files:  {n_files}      span≈{span_h:.2f} h   status: {status}")
    print(f"  raw:    {raw_path}")

    df = _safe_load_window(raw_path)

    if df.empty:
        print("  ⚠️ dataframe is EMPTY")
        continue

    # If span was NaN in manifest, compute from loaded data now
    if not pd.notna(span_h) and "EndTime" in df.columns:
        span_h = (df["EndTime"].max() - df["EndTime"].min()) / pd.Timedelta(hours=1)

    # Quick context
    tmin = df["EndTime"].min() if "EndTime" in df.columns else None
    tmax = df["EndTime"].max() if "EndTime" in df.columns else None
    link_col = "LinkID" if "LinkID" in df.columns else ("ID" if "ID" in df.columns else None)
    n_links = (df[link_col].nunique() if link_col else None)
    print(f"  rows: {len(df)}   time-span: {tmin} → {tmax}  "
          f"{f'links: {n_links}' if n_links is not None else ''}")

    # Optional: frequency snapshot
    if "Frequency" in df.columns:
        try:
            freq_ghz = (pd.to_numeric(df["Frequency"], errors="coerce") / 1000.0).round(1)
            top = freq_ghz.value_counts().head(5)
            if len(top):
                print("  top freqs (GHz):")
                for f, c in top.items():
                    print(f"    {f:>5}: {c}")
        except Exception:
            pass

    # ------------------------------------------------------------
    # TODO (R4b/R5): per-link coverage check + wet/dry/baseline prep
    # NOTE to future self:
    #   Each 15-min in the *target* 3 h block should have baseline
    #   estimated from the previous 48 h *dry* minutes. Our 51 h
    #   slice is built to provide those 48 h + the final 3 h block.
    # ------------------------------------------------------------

print("\n[R4] Done. Windows are loaded and summarized. Ready for the next preprocessing step.")

#%%
#%%
#%%
# ================================================================
# SECTION R5: Per-link coverage & gap checks; write cleaned windows
#   - Input:  R3B_manifest.csv (from R3) or man_df already loaded in R4
#   - Output: Cleaned parquet per window + R5_manifest.csv
# ================================================================


# ---- Inputs / paths
MANIFEST_R4 = os.path.join(OUT_DIR, "R3B_manifest.csv")  # same file used by R4
MANIFEST_R5 = os.path.join(OUT_DIR, "R5_manifest.csv")

# ---- Parameters (uses globals from R0)
EXPECTED_FREQ = f"{FREQ_MINUTES}min"       # '15min'
STEP_PER_HOUR = 60 // FREQ_MINUTES
EXPECTED_STEPS_51H = int(round(REQUIRED_HOURS * STEP_PER_HOUR))  # 51h * (60/15)=204

# ---- Helpers
def _coerce_utc(ts):
    t = pd.to_datetime(ts, utc=True, errors="coerce")
    return t

def _expected_grid(win_start_utc, target_dt_utc, freq=EXPECTED_FREQ):
    """
    Build the nominal time grid of EndTime stamps for the window [start, target],
    inclusive at the right edge.
    """
    win_start_utc = _coerce_utc(win_start_utc)
    target_dt_utc = _coerce_utc(target_dt_utc)
    if pd.isna(win_start_utc) or pd.isna(target_dt_utc):
        return pd.DatetimeIndex([], tz="UTC")
    # Align to the grid
    win_start_utc = win_start_utc.floor(freq)
    target_dt_utc = target_dt_utc.floor(freq)
    return pd.date_range(win_start_utc, target_dt_utc, freq=freq, tz="UTC")

def _link_coverage_diag(df_link, expected_idx):
    """
    Compute coverage % and longest internal gap (in minutes) for one link.
    df_link: rows for a single LinkID (must have EndTime).
    expected_idx: DatetimeIndex of expected bins (EndTime values).
    """
    # unique observed EndTime on the grid
    obs = _coerce_utc(df_link["EndTime"]).dropna().dt.floor(EXPECTED_FREQ).unique()
    obs_idx = pd.DatetimeIndex(obs) if len(obs) else pd.DatetimeIndex([], tz="UTC")
    # normalize tz to UTC safely
    if len(obs_idx):
        if obs_idx.tz is None:
            obs_idx = obs_idx.tz_localize("UTC")
        else:
            obs_idx = obs_idx.tz_convert("UTC")

    # coverage
    expected_count = len(expected_idx)
    observed_count = len(expected_idx.intersection(obs_idx))
    cover_pct = 100.0 * observed_count / expected_count if expected_count else 0.0

    # longest internal gap (in minutes) computed on the expected grid
    present = pd.Series(False, index=expected_idx)
    if len(obs_idx):
        present.loc[present.index.intersection(obs_idx)] = True

    if not present.any():
        longest_gap_min = (expected_count - 1) * FREQ_MINUTES if expected_count > 0 else 0
    else:
        true_pos = np.flatnonzero(present.values)
        longest_gap_steps = 0
        if len(true_pos) >= 2:
            diffs = np.diff(true_pos) - 1  # missing bins between consecutive present points
            longest_gap_steps = int(diffs.max()) if len(diffs) else 0
        longest_gap_min = longest_gap_steps * FREQ_MINUTES

    return cover_pct, longest_gap_min, int(observed_count), int(expected_count)

def _safe_read_parquet(p):
    try:
        df = pd.read_parquet(p)
        # Standardize EndTime to UTC
        if "EndTime" in df.columns:
            df["EndTime"] = _coerce_utc(df["EndTime"])
        return df
    except Exception as e:
        print(f"  ✖️ failed to read parquet {p}: {e}")
        return pd.DataFrame()

def _write_cleaned_window(df_clean, target_dt):
    fn = f"clean_window_{pd.to_datetime(target_dt, utc=True).strftime('%Y%m%d%H%M')}.parquet"
    path = os.path.join(OUT_DIR, fn)
    try:
        df_clean.to_parquet(path, index=False)
    except Exception as e:
        print(f"  ✖️ failed to write cleaned parquet: {e}")
        path = ""
    return path

def _write_link_diag(diag_df, target_dt):
    fn = f"r5_links_{pd.to_datetime(target_dt, utc=True).strftime('%Y%m%d%H%M')}.csv"
    path = os.path.join(OUT_DIR, fn)
    try:
        diag_df.to_csv(path, index=False)
    except Exception as e:
        print(f"  ✖️ failed to write link diagnostics: {e}")
        path = ""
    return path

# ---- Load manifest from R4 step (if not already in memory)
if 'man_df' not in globals() or man_df is None:
    man_df = pd.read_csv(MANIFEST_R4)
# harmonize columns
if "raw_csv_path" not in man_df.columns and "out_path" in man_df.columns:
    man_df = man_df.rename(columns={"out_path": "raw_csv_path"})
for col in ["target_dt_utc", "window_start_utc"]:
    if col in man_df.columns:
        man_df[col] = pd.to_datetime(man_df[col], utc=True)

print(f"\n[R5] Starting coverage checks over {len(man_df)} windows")

# ---- R5 manifest rows
r5_rows = []

# ---- Iterate windows
for i, row in man_df.iterrows():
    target_dt   = row["target_dt_utc"]
    win_start   = row["window_start_utc"]
    raw_path    = row["raw_csv_path"]
    status_r4   = str(row.get("reason", ""))
    n_files     = int(row.get("n_files", 0))
    n_rows_raw  = int(row.get("n_rows", -1)) if "n_rows" in row and not pd.isna(row["n_rows"]) else None

    if not isinstance(raw_path, str) or (not os.path.isabs(raw_path)):
        raw_path = os.path.join(OUT_DIR, str(raw_path))

    if not os.path.exists(raw_path):
        print(f"\n[R5] ⚠️  Missing raw window for {target_dt}: {raw_path}")
        r5_rows.append({
            "target_dt_utc": target_dt, "window_start_utc": win_start,
            "n_files": n_files, "n_rows_raw": n_rows_raw,
            "n_links_total": 0, "n_links_good": 0,
            "pct_links_good": 0.0, "mean_cover_pct": np.nan, "median_cover_pct": np.nan,
            "global_ok": False, "reason": "RAW_MISSING", "clean_path": "", "link_diag_csv": ""
        })
        continue

    print(f"\n[R5] Window target={target_dt}  start={win_start}  raw={os.path.basename(raw_path)}")
    df = _safe_read_parquet(raw_path)
    if df.empty:
        print("  ⚠️ raw dataframe is EMPTY")
        r5_rows.append({
            "target_dt_utc": target_dt, "window_start_utc": win_start,
            "n_files": n_files, "n_rows_raw": 0,
            "n_links_total": 0, "n_links_good": 0,
            "pct_links_good": 0.0, "mean_cover_pct": np.nan, "median_cover_pct": np.nan,
            "global_ok": False, "reason": "RAW_EMPTY", "clean_path": "", "link_diag_csv": ""
        })
        continue

    # Ensure EndTime present
    if "EndTime" not in df.columns:
        print("  ✖️ EndTime column missing — skipping window")
        r5_rows.append({
            "target_dt_utc": target_dt, "window_start_utc": win_start,
            "n_files": n_files, "n_rows_raw": len(df),
            "n_links_total": df["LinkID"].nunique() if "LinkID" in df.columns else 0,
            "n_links_good": 0, "pct_links_good": 0.0,
            "mean_cover_pct": np.nan, "median_cover_pct": np.nan,
            "global_ok": False, "reason": "NO_ENDTIME", "clean_path": "", "link_diag_csv": ""
        })
        continue

    # Build expected 15-min grid for the window
    expected_idx = _expected_grid(win_start, target_dt, EXPECTED_FREQ)
    expected_steps = len(expected_idx)
    # Note: coverage denominator uses expected_steps, which exactly matches the realized window length.

    # Per-link diagnostics
    if "LinkID" not in df.columns:
        df["LinkID"] = "__NO_LINK_ID__"

    link_groups = df.groupby("LinkID", sort=False)
    diag_rows = []
    for link_id, df_link in link_groups:
        cover_pct, longest_gap_min, obs_count, exp_count = _link_coverage_diag(df_link, expected_idx)
        diag_rows.append({
            "LinkID": link_id,
            "cover_pct": cover_pct,
            "observed_steps": obs_count,
            "expected_steps": exp_count,
            "longest_gap_min": longest_gap_min,
            "first_time": df_link["EndTime"].min(),
            "last_time":  df_link["EndTime"].max()
        })
    link_diag = pd.DataFrame(diag_rows).sort_values("cover_pct", ascending=False)

    n_links_total = len(link_diag)
    # determine "good" links
    good_mask = link_diag["cover_pct"] >= PER_LINK_MIN_COVER_PCT
    n_links_good = int(good_mask.sum())
    pct_links_good = (100.0 * n_links_good / n_links_total) if n_links_total else 0.0
    mean_cover = float(link_diag["cover_pct"].mean()) if n_links_total else np.nan
    median_cover = float(link_diag["cover_pct"].median()) if n_links_total else np.nan

    # Filter df to keep only good links
    good_links = set(link_diag.loc[good_mask, "LinkID"])
    df_clean = df[df["LinkID"].isin(good_links)].copy()

    # Global requirement
    global_ok = pct_links_good >= GLOBAL_MIN_COVER_PCT
    reason = "OK" if global_ok else f"FEW_GOOD_LINKS ({pct_links_good:.1f}% < {GLOBAL_MIN_COVER_PCT:.0f}%)"
    # Always write cleaned and diagnostics (so we can inspect)
    cleaned_path = _write_cleaned_window(df_clean, target_dt)
    link_diag_csv = _write_link_diag(link_diag, target_dt)

    print(f"  links: total={n_links_total}  good={n_links_good} ({pct_links_good:.1f}%)  "
          f"mean/median cover={mean_cover:.1f}/{median_cover:.1f}%  -> {reason}")
    print(f"  wrote: {os.path.basename(cleaned_path)}  &  {os.path.basename(link_diag_csv)}")

    r5_rows.append({
        "target_dt_utc": target_dt,
        "window_start_utc": win_start,
        "n_files": n_files,
        "n_rows_raw": len(df),
        "n_links_total": n_links_total,
        "n_links_good": n_links_good,
        "pct_links_good": pct_links_good,
        "mean_cover_pct": mean_cover,
        "median_cover_pct": median_cover,
        "global_ok": bool(global_ok),
        "reason": reason,
        "clean_path": cleaned_path,
        "link_diag_csv": link_diag_csv,
        "status_r4": status_r4
    })

# ---- Write R5 manifest
r5_df = pd.DataFrame(r5_rows)
cols_order = ["target_dt_utc","window_start_utc","status_r4","reason","global_ok",
              "n_files","n_rows_raw","n_links_total","n_links_good","pct_links_good",
              "mean_cover_pct","median_cover_pct","clean_path","link_diag_csv"]
r5_df = r5_df.reindex(columns=cols_order)
r5_df.to_csv(MANIFEST_R5, index=False)
print(f"\n[R5] Done. Wrote window diagnostics to: {MANIFEST_R5}")

ok_count = int(r5_df["global_ok"].sum()) if "global_ok" in r5_df else 0
print(f"[R5] Windows passing global link coverage: {ok_count}/{len(r5_df)}")


#%% Sanity check
from pathlib import Path
import pandas as pd

# If OUT_DIR is already a Path, this is a no-op; if it's a str, we wrap it.
OUT = Path(OUT_DIR) if not isinstance(OUT_DIR, Path) else OUT_DIR

if not OUT.exists():
    raise FileNotFoundError(f"OUT_DIR not found: {OUT}")

# Helper readers
def read_linkids_parquet(p, cand_cols=("LinkID","link_id","linkID","id_link","Link")):
    for c in cand_cols:
        try:
            return pd.read_parquet(p, columns=[c]).rename(columns={c:"LinkID"})
        except Exception:
            pass
    return pd.DataFrame(columns=["LinkID"])

def read_linkids_csv(p, cand_cols=("LinkID","link_id","Link","id_link")):
    try:
        df = pd.read_csv(p)
        for c in cand_cols:
            if c in df.columns:
                return df[[c]].rename(columns={c:"LinkID"})
    except Exception:
        pass
    return pd.DataFrame(columns=["LinkID"])

# Find files (recursive in case outputs are in subfolders)
pq_files = sorted(OUT.rglob("raw_window_*.parquet"))
csv_files = sorted(OUT.rglob("r5_links_*.csv"))

# Collect LinkIDs
dfs = []
for f in pq_files:
    df = read_linkids_parquet(f)
    if not df.empty:
        dfs.append(df)

for f in csv_files:
    df = read_linkids_csv(f)
    if not df.empty:
        dfs.append(df)

if dfs:
    all_ids = (pd.concat(dfs, ignore_index=True)
                 .dropna(subset=["LinkID"])
                 .assign(LinkID=lambda x: x["LinkID"].astype(str))
                 .drop_duplicates())
    print(f"Distinct LinkIDs across OUT_DIR (raw + r5_links): {all_ids['LinkID'].nunique()}")
else:
    print("No LinkIDs found. (No matching files or no LinkID-like columns present.)")

print(f"Found {len(pq_files)} raw_window parquet files and {len(csv_files)} r5_links CSV files.")