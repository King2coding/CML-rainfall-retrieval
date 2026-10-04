# run_cml_rainfall_retrieval_bas_202605.py
#
# Example application script for the Ghana CML rainfall retrieval and gridding
# workflow prepared for Bas review.
#
# This file intentionally avoids the TAHMO API/service layer. It shows the
# technical sequence only: raw CML files + metadata -> cleaned link signals ->
# link-level rainfall rates -> gridded rainfall/support NetCDF outputs.
#
# Bas: the paths below are Kingsley's local/operational paths. Please replace
# them with your own local directories before running the script.

from __future__ import annotations

import gc
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from cml_rainfall_retrieval_core_bas_202605 import (
    R0AutoConfig,
    cml2metadata_coupling_framework_fast,
    clean_minmax_auto,
    build_15min_timeseries,
    rainlink_strict_Aobs,
    rainlink_strict_R,
    prepare_inputs_for_gridding,
    grid_rain_15min_rainlink_ok_full_ghana,
    save_15min_grid_and_points_netcdf_for_day,
    extract_datetime_from_filename,
)


# =============================================================================
# 0. User paths and run controls
# =============================================================================

# Kingsley's current path to the matched CML metadata file.
# Bas: replace this with the directory containing your metadata CSV.
metadata_dir = Path(r"/home/kkumah/Projects/cml-stuff/data-cml/outs")
metadata_file = metadata_dir / "matched_metadata_kkk_20250527.csv"

# Kingsley's current path to raw AT Ghana CML text files.
# Bas: replace this with your directory containing the raw CML files.
raw_cml_dir = Path(r"/home/kkumah/Projects/cml-stuff/data-cml/rsl")

# Kingsley's current output directory for rainfall/gridded products.
# Bas: replace this with your preferred output directory.
output_dir = Path(r"/home/kkumah/Projects/cml-stuff/misc")
output_dir.mkdir(parents=True, exist_ok=True)

# Use a fixed target date for reproducible testing, or set to None to use the
# latest available timestamp in raw_cml_dir.
target_date = pd.to_datetime("2025-06-15 23:59:00")

# Recent history window used to build a dry baseline before the target block.
# The operational API has used a shorter lookback, while development tests used
# up to 72 h. Keep this explicit so Bas can test sensitivity.
lookback_hours = 72

# We save files only for this day after gridding. By default this follows the
# target_date/latest date selected below.
base_output_name = "ghana_cml_R_15min"


# =============================================================================
# 1. Select recent CML files
# =============================================================================

matched_metadata = pd.read_csv(metadata_file)

file_names = sorted(os.listdir(raw_cml_dir))
file_datetimes = [(extract_datetime_from_filename(f), f) for f in file_names]
file_datetimes = [(dt, f) for dt, f in file_datetimes if dt is not None]

if not file_datetimes:
    raise RuntimeError(f"No timestamped raw CML files were found in {raw_cml_dir}")

if target_date is not None:
    latest_dt = datetime(
        int(target_date.year),
        int(target_date.month),
        int(target_date.day),
        int(target_date.hour),
        int(target_date.minute),
    )
else:
    latest_dt, _ = max(file_datetimes, key=lambda x: x[0])

cutoff_time = latest_dt - timedelta(hours=lookback_hours)
recent_files = [(dt, f) for dt, f in file_datetimes if cutoff_time <= dt <= latest_dt]
recent_files.sort(key=lambda x: x[0])

if not recent_files:
    raise RuntimeError("No raw CML files were found within the requested lookback window.")

print("First file:", recent_files[0])
print("Last file: ", recent_files[-1])
print("Number of files:", len(recent_files))


# =============================================================================
# 2. Couple raw CML files with link metadata
# =============================================================================

coupled_frames = []
for idx, (_, fname) in enumerate(recent_files):
    if idx % 5 == 0:
        print(f"Coupled {idx} / {len(recent_files)} files")

    raw_file = raw_cml_dir / fname
    frame = cml2metadata_coupling_framework_fast(str(raw_file), matched_metadata)
    if frame is not None and not frame.empty:
        coupled_frames.append(frame)

if not coupled_frames:
    raise RuntimeError("No raw CML files could be coupled to metadata.")

df_raw = pd.concat(coupled_frames, ignore_index=True)
print("Raw coupled rows:", len(df_raw))


# =============================================================================
# 3. Clean min/max received-signal values and build strict 15-min series
# =============================================================================

# semantics="auto" lets the cleaner decide whether each link behaves like RSL
# in dBm or path-loss/total-loss in dB. This is important because network files
# can mix conventions or contain ambiguous signal columns.
cfg = R0AutoConfig(semantics="auto", regularize_grid=True)

df_clean, df_summary = clean_minmax_auto(df_raw, cfg)
print(df_summary.head(10))

ts_15 = build_15min_timeseries(df_clean)
gc.collect()

# =============================================================================
# Optional diagnostic plot: inspect cleaned 15-min signal behavior
# =============================================================================
# Internal self-check only. This checks the semantics-safe signal used for
# baseline estimation. Higher sig_db means a stronger/drier signal after
# handling RSL vs TL conventions.

import matplotlib.pyplot as plt

print("ts_15 columns:")
print(ts_15.columns.tolist())

# Select a few links with enough valid 15-min samples
valid_counts = (
    ts_15
    .groupby("link_id")["sig_db"]
    .apply(lambda x: x.notna().sum())
    .sort_values(ascending=False)
)

sample_links = valid_counts.head(4).index.tolist()

for link_id in sample_links:
    d = ts_15[ts_15["link_id"] == link_id].copy()
    d = d.sort_values("time")

    plt.figure(figsize=(12, 4))
    plt.plot(d["time"], d["sig_db"], marker=".", linewidth=1, label="sig_db")

    plt.title(f"Cleaned 15-min semantics-safe signal: {link_id}")
    plt.xlabel("Time")
    plt.ylabel("sig_db; stronger/drier signal is larger")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 4. RainLINK-style wet/dry detection, baseline estimation, and rain retrieval
# =============================================================================

# The baseline is past-only to remain valid for near-real-time operation.
# two_pass=True refines the dry baseline after an initial wet/dry classification.
# The dry-count guard is currently off in the operational run, but it is useful
# to keep here as a clearly documented tuning option for Bas to evaluate.
df_A = rainlink_strict_Aobs(
    ts_15,
    two_pass=True,
    use_drycount_guard=False,
    # Alternative sensitivity test:
    # use_drycount_guard=True,
    # min_dry_bins=8,
    # guard_behavior="fallback",
)

# Convert observed attenuation to rain rate using wet-antenna correction and the
# pycomlink k-R conversion. R_min=0.0 keeps true zeros after wet/dry gating.
df_rate = rainlink_strict_R(df_A, R_min=0.0)

# Save the link-level rainfall table for diagnostics/review.
link_rain_csv = output_dir / f"link_level_cml_rainfall_{latest_dt.date().isoformat()}.csv"
df_rate.to_csv(link_rain_csv, index=False)
print("Saved link-level rainfall:", link_rain_csv)

# =============================================================================
# Optional diagnostic plot: signal, baseline, wet/dry, attenuation, rainfall
# =============================================================================
# Internal self-check only. This helps inspect whether the baseline, wet/dry
# classification, attenuation, and final rainfall are behaving consistently
# for one selected link.

import matplotlib.pyplot as plt
import numpy as np

rain_col = "R_mm_per_h"

# Pick a link with the largest retrieved rainfall total
rain_by_link = (
    df_rate
    .groupby("link_id")[rain_col]
    .sum(min_count=1)
    .sort_values(ascending=False)
)

print("Top rainfall links:")
print(rain_by_link.head(10))

link_id = rain_by_link.index[0]
print("Selected diagnostic link:", link_id)

d = df_rate[df_rate["link_id"] == link_id].copy()
d = d.sort_values("time")

fig, axes = plt.subplots(
    4, 1,
    figsize=(14, 10),
    sharex=True,
    gridspec_kw={"height_ratios": [2.0, 0.8, 1.2, 1.2]}
)

# -------------------------------------------------------------------------
# 1. Signal and dry baseline
# -------------------------------------------------------------------------
baseline_col = "baseline_sig_db" if "baseline_sig_db" in d.columns else "baseline_rsl"

axes[0].plot(
    d["time"], d["sig_db"],
    marker=".", linewidth=1,
    label="sig_db"
)

axes[0].plot(
    d["time"], d[baseline_col],
    linewidth=2,
    label=baseline_col
)

axes[0].set_ylabel("Signal / baseline (dB)")
axes[0].set_title(f"CML retrieval diagnostic: {link_id}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# -------------------------------------------------------------------------
# 2. Wet/dry classification
# -------------------------------------------------------------------------
axes[1].step(
    d["time"], d["wet_rl"].astype(int),
    where="post",
    linewidth=1.5,
    label="wet_rl"
)

axes[1].set_ylabel("Wet flag")
axes[1].set_yticks([0, 1])
axes[1].set_yticklabels(["dry", "wet"])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# -------------------------------------------------------------------------
# 3. Observed attenuation
# -------------------------------------------------------------------------
axes[2].plot(
    d["time"], d["A_obs_dB"],
    marker=".", linewidth=1,
    label="A_obs_dB"
)

axes[2].axhline(
    3.0,
    linestyle="--",
    linewidth=1,
    label="wet threshold = 3 dB"
)

axes[2].set_ylabel("Attenuation (dB)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# -------------------------------------------------------------------------
# 4. Retrieved rain rate
# -------------------------------------------------------------------------
axes[3].bar(
    d["time"], d[rain_col],
    width=0.008,
    label=rain_col
)

axes[3].set_ylabel("Rain rate (mm/h)")
axes[3].set_xlabel("Time")
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# =============================================================================
# 5. Prepare link rainfall for gridding
# =============================================================================

df_s5, meta_xy_grid = prepare_inputs_for_gridding(df_rate, ts_15)

# For the gridded product, process only the selected output day. The preceding
# lookback period is still important because it supports baseline estimation.
output_day = latest_dt.date()
df_s5_day = df_s5[df_s5.index.date == output_day]

if df_s5_day.empty:
    raise RuntimeError(f"No link rainfall data are available for {output_day} after retrieval.")


# =============================================================================
# 6. CML rainfall gridding and support/confidence layers
# =============================================================================

# Main gridding logic:
# - Training values are link-level rainfall rates.
# - Link-path support points are used for support/confidence geometry so the
#   coverage follows CML path corridors better than pure midpoint circles.
# - Ordinary Kriging is used only when enough nearby wet links exist.
# - Dry-link constraints and support masks reduce unrealistic spatial spreading.
# - The confidence layer is a practical support/consistency layer, not a formal
#   probabilistic uncertainty estimate.
grid_ds, diag = grid_rain_15min_rainlink_ok_full_ghana(
    df_s5=df_s5_day,
    df_meta_for_xy=meta_xy_grid,
    fixed_extent=(-3.5, 1.5, 4.5, 11.5),  # lon_min, lon_max, lat_min, lat_max
    grid_res_deg=0.03,

    # Wet/dry thresholds for gridding support and training selection.
    wet_thr=0.8,
    dry_thr=0.05,

    # Ordinary Kriging / variogram controls.
    ok_model="exponential",
    ok_range_km=22.0,
    ok_nugget_frac=0.4,
    min_pts_ok=15,

    # Link-path support geometry.
    support_geometry="link_path",
    n_support_points_per_link=5,
    support_point_spacing_km=2.0,
    min_support_points_per_link=3,
    max_support_points_per_link=7,
    use_length_conditioned_support_points=True,

    # CML support mask and dry constraint.
    support_k=2,
    support_radius_km=30.0,
    dry_radius_km=3.0,
    use_dry_constraint=True,

    # Soft support/confidence layer used by downstream consumers.
    use_soft_confidence=True,
    confidence_floor=0.03,
    confidence_power=1.1,
    confidence_dry_penalty_weight=0.50,

    # Rainfall handling.
    drizzle_to_zero=0.10,
    outside_support_fill=np.nan,
    insufficient_training_fill=np.nan,

    # Keep the scientific field conservative; avoid excessive smoothing.
    smooth_kernel_px=1,
    smooth_fill_holes=False,

    # Support-mask cleanup.
    clean_support=True,
    support_closing_iters=1,
    support_fill_holes=True,
    support_max_hole_px=20,

    # Display/cosmetic field is useful for diagnostics but is not exported by
    # default in the operational Rainboo/TAHMO NetCDF files.
    make_display_field=True,
    display_smooth=True,
    display_smooth_kernel_px=3,
    display_smooth_fill_holes=False,
    apply_display_edge_taper=True,
    display_edge_taper_pixels=6,
    display_edge_taper_min_weight=0.05,

    # Simple categorical layer for downstream usage decisions.
    make_coverage_quality=True,
    coverage_quality_med_thr=0.50,
    coverage_quality_high_thr=0.75,

    n_jobs=20,
    parallel_backend_name="processes",
    return_dataset=True,
)

print("Gridding counts:", diag.get("counts"))
print("Coverage-quality counts:", diag.get("coverage_quality_counts"))
print("Generated timestamps:", grid_ds.sizes.get("time"))


# =============================================================================
# 7. Save 15-min NetCDF outputs
# =============================================================================

written_files = save_15min_grid_and_points_netcdf_for_day(
    grid_data=grid_ds,
    df_s5=df_s5_day,
    meta_xy=meta_xy_grid,
    out_dir=str(output_dir),
    day=output_day.strftime("%Y-%m-%d"),
    base_name=base_output_name,
    version="V1",

    # Operational recommendation: export the scientific rainfall field plus
    # support/confidence/quality and link-midpoint values. Keep the display field
    # out of the formal product unless explicitly needed for diagnostics.
    include_display_field=False,
)

print(f"Saved {len(written_files)} NetCDF files")
for f in written_files[:5]:
    print("  ", f)

# Quick output integrity check on the first file.
if written_files:
    with xr.open_dataset(written_files[0]) as ds_check:
        print("Variables:", list(ds_check.data_vars))
        print("R dtype:", ds_check["R_mm_per_h"].dtype)
        print("Support-confidence dtype:", ds_check["cml_support_confidence"].dtype)
        print("Support-mask values:", np.unique(ds_check["cml_support_mask"].values))
        print("Coverage-quality values:", np.unique(ds_check["cml_coverage_quality"].values))
