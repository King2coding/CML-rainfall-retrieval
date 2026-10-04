# run_cml_rainfall_retrieval_stats_wetdry_bas_202605.py
#
# Example application script for the revised Ghana CML rainfall retrieval
# workflow prepared for Bas review. This version uses a statistics-based
# wet/dry-first retrieval before baseline estimation.
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

from cml_rainfall_retrieval_stats_wetdry_core_bas_202605 import (
    R0AutoConfig,
    cml2metadata_coupling_framework_fast,
    clean_minmax_auto,
    build_15min_timeseries,
    stats_wetdry_Aobs,
    add_spatial_wetdry_support,
    rainlink_strict_R,
    plot_cml_retrieval_diagnostic,
    prepare_inputs_for_gridding,
    grid_rain_15min_rainlink_ok_full_ghana,
    save_15min_grid_and_points_netcdf_for_day,
    extract_datetime_from_filename,
    plot_cml_grid_diagnostics,
    plot_daily_mean_cml_rain_rate,
    plot_daily_equivalent_cml_rainfall,
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

# Diagnostic plots are saved here so they can be copied into Kingsley's
# comparison Google Slides.
diagnostics_dir = output_dir / "cml_bas_method_diagnostics"
diagnostics_dir.mkdir(parents=True, exist_ok=True)

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
# 4. Statistics-based wet/dry-first retrieval
# =============================================================================

# Why this replaces the old baseline-first/high-quantile method:
# - The old method could allow short high-signal jumps to inflate the dry
#   baseline and produce long false rainfall periods.
# - Here we first classify wet/dry periods from link-level signal variability.
# - The dry baseline is then estimated only from dry-classified intervals.
# - Frequency-family thresholds are used because rain sensitivity depends on
#   microwave frequency, but exact-frequency groups can be too sparse.

# Candidate settings after diagnostics:
# - window_bins=10 means 150 minutes for 15-min data.
# - std_percentile=95 is a stricter frequency-family threshold that reduces
#   false wet classifications from noisy/unstable link-level signal variability.
# - baseline_win="48H" and baseline_q=0.50 use a 48-hour dry-only median
#   reference, following the same general idea of estimating dry reference levels
#   from stable dry periods over a longer history window.
# - require_temporal_support_for_wet remains False inside the function defaults:
#   short convective events are not removed immediately, but temporal support is
#   retained as a diagnostic/confidence flag.
df_A, wetdry_thresholds = stats_wetdry_Aobs(
    ts_15,
    window_bins=10,                  # 150-min rolling std
    std_percentile=95.0,             # stricter wet/dry threshold
    baseline_win="48H",              # RainLINK-inspired long dry-reference window
    baseline_q=0.50,                 # dry-only median baseline
    min_dry_bins=8,
    ffill_limit_bins=32,
    require_src_present=True,
)

print("Wet/dry rolling-std thresholds by frequency family:")
for k, v in wetdry_thresholds.items():
    print(f"  {k}: {v:.4f} dB")

# Optional spatial support check. This does not automatically reject isolated
# links; it adds neighbor_count, neighbor_wet_fraction, wet_spatial_support, and
# updates retrieval_confidence. This is useful for gridding and for diagnosing
# sparse-network false alarms.
df_A = add_spatial_wetdry_support(
    df_A,
    radius_km=25.0,
    min_neighbors=3,
    min_neighbor_wet_fraction=0.50,
    wet_col="wet_final",
)

# Convert observed attenuation to rain rate using wet-antenna correction and the
# pycomlink k-R conversion. The wet_col is wet_final from the new method.
df_rate = rainlink_strict_R(
    df_A,
    R_min=0.0,
    wet_col="wet_final",
    set_dry_to_zero=True,
)

# Preserve retrieval confidence after rain-rate conversion.
# rainlink_strict_R copies all input columns, so these should remain available.
print("Rain-rate output columns include:")
print([c for c in [
    "wet_stat_initial", "wet_temporal_support", "wet_spatial_support",
    "wet_final", "baseline_sig_db", "A_obs_dB", "R_mm_per_h",
    "retrieval_confidence"
] if c in df_rate.columns])

# Save the link-level rainfall table for diagnostics/review.
link_rain_csv = output_dir / f"link_level_cml_rainfall_stats_wetdry_{latest_dt.date().isoformat()}.csv"
df_rate.to_csv(link_rain_csv, index=False)
print("Saved link-level rainfall:", link_rain_csv)

# Diagnostic plots for the problematic link identified during the baseline-first
# review. Additional links can be added to this list after inspection.
diagnostic_links = [
    # Old baseline-first failure case
    "GH0064_RTN950A-22-MODU-1(RTNRF-1)-1>>GH0329_NE_1-22-MODU-1(RTNRF-1)-1",

    # New-method top-rainfall / important diagnostic case
    "KPNDAE_RTN950A-24-MODU-1(RTNRF-1)-1>>WALD_RTN950A-23-MODU-1(RTNRF-1)-1",
]

for link_id in diagnostic_links:
    try:
        safe_name = (
            link_id.replace(">>", "__")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
        )[:160]
        plot_cml_retrieval_diagnostic(
            df_rate,
            link_id=link_id,
            title_suffix="Stats wet/dry-first: win=150min, p95, baseline=48H dry-median",
            wet_thr_db=None,
            save_path=str(diagnostics_dir / f"stats_wetdry_win150_p95_48H_median_{safe_name}.png"),
        )
    except Exception as exc:
        print(f"Could not make diagnostic plot for {link_id}: {exc}")

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

#==============================================================================
# 6b. Some grid level diagnostic plot for context
# ==============================================================================
# Find timestamp with largest gridded rainfall maximum
rain_max_by_time = grid_ds["R_mm_per_h"].max(dim=("lat", "lon"), skipna=True)
imax = int(rain_max_by_time.argmax(dim="time").values)
t_peak = pd.to_datetime(grid_ds["time"].values[imax])

print("Peak grid rainfall time:", t_peak)
print("Peak max rainfall:", float(rain_max_by_time.isel(time=imax).values))

plot_cml_grid_diagnostics(
    grid_ds,
    meta_xy_grid,
    t_peak,
    rainfall_vmin=0.0,
    rainfall_vmax=12.0,
    save_path=diagnostics_dir / f"grid_diagnostic_peak_{pd.Timestamp(t_peak).strftime('%Y%m%dT%H%M')}.png",
)

daily_mean_rate = plot_daily_mean_cml_rain_rate(
    grid_ds,
    meta_xy_grid,
    day=output_day,
    vmin=0.0,
    vmax=4.0,
    save_path=diagnostics_dir / f"daily_mean_rain_rate_{output_day}.png",
)

daily_equiv_mm_day = plot_daily_equivalent_cml_rainfall(
    grid_ds,
    meta_xy_grid,
    day=output_day,
    vmin=0.0,
    vmax=80.0,   # similar broad range to IMERG/Giovanni
    save_path=diagnostics_dir / f"daily_equiv_rainfall_mm_day_{output_day}.png",
)
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
