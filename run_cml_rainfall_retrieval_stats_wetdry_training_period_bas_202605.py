# run_cml_rainfall_retrieval_stats_wetdry_training_period_bas_202605.py
#
# Multi-day / training-period application script for the revised Ghana CML
# rainfall retrieval workflow prepared for Bas review.
#
# Purpose:
#   Generate CML-only rainfall reference files for the MSG + CML / CML-SAT
#   machine-learning training period using the statistics-based wet/dry-first
#   method. The generated NetCDF files can then be used as the rainfall target
#   for MSG wet/dry classification and wet-only rainfall-intensity training.
#
# This script reuses the Bas-review core functions without changing them.
# It loops over a user-defined date range and, for each output day, uses a
# preceding lookback window for dry-baseline estimation before gridding/saving
# the selected day.


from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated*", category=UserWarning)

import gc
import os
import traceback
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

metadata_dir = Path(r"/home/kkumah/Projects/cml-stuff/data-cml/outs")
metadata_file = metadata_dir / "matched_metadata_kkk_20250527.csv"

raw_cml_dir = Path(r"/home/kkumah/Projects/cml-stuff/data-cml/rsl")

# Use a dedicated output directory for the CML-only reference used by MSG training.
# This avoids overwriting older/API-test products.
output_dir = Path(r"/home/kkumah/Projects/cml-stuff/new_out_cml_Rain_bas_stats_training_ref_sens_q75_p90_wet03")
# output_dir = Path(r"/home/kkumah/Projects/cml-stuff/new_out_cml_Rain_bas_stats_training_ref")
output_dir.mkdir(parents=True, exist_ok=True)

diagnostics_dir = output_dir / "diagnostics"
diagnostics_dir.mkdir(parents=True, exist_ok=True)

log_dir = output_dir / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

# MSG-CML training period. Adjust as needed.
training_start = "2025-06-14"
training_end   = "2025-08-24"

# Baseline/lookback history before each output day.
lookback_hours = 72

# Daily target timestamp used to select the lookback window.
# 23:59 includes all files on that output day up to the last available 15-min slot.
target_hour = 23
target_minute = 59

version = "V2_q75_p90_wet03"
base_output_name = "ghana_cml_R_15min_bas_stats_ref_q75_p90_wet03"
# If True, days with at least one existing output NetCDF are skipped.
skip_existing_days = False

# Make expensive diagnostics for the peak day/timestamp. Keep False for long runs.
make_daily_diagnostics = False

# Optional: set to a small integer for testing, e.g., 3.
max_days_to_process = None


# =============================================================================
# 1. Helper functions
# =============================================================================

def discover_raw_files(raw_dir: Path) -> list[tuple[datetime, str]]:
    file_names = sorted(os.listdir(raw_dir))
    file_datetimes = [(extract_datetime_from_filename(f), f) for f in file_names]
    file_datetimes = [(dt, f) for dt, f in file_datetimes if dt is not None]
    file_datetimes.sort(key=lambda x: x[0])
    return file_datetimes


def day_has_existing_outputs(out_dir: Path, base_name: str, day: pd.Timestamp) -> bool:
    ymd = day.strftime("%Y%m%d")
    # save_15min_grid_and_points_netcdf_for_day normally includes base_name/version/date/time.
    return any(out_dir.glob(f"{base_name}*{ymd}*.nc"))


def write_day_failure(log_path: Path, day_str: str, exc: BaseException) -> None:
    with open(log_path, "a") as f:
        f.write("\n" + "=" * 90 + "\n")
        f.write(f"FAILED DAY: {day_str}\n")
        f.write(f"ERROR: {repr(exc)}\n")
        f.write(traceback.format_exc())
        f.write("\n")

def quick_Aobs_diagnostics(df_A, label="df_A"):
    a = pd.to_numeric(df_A["A_obs_dB"], errors="coerce").to_numpy(float)
    wet = df_A["wet_final"].fillna(False).astype(bool).to_numpy() if "wet_final" in df_A else df_A["wet_rl"].fillna(False).astype(bool).to_numpy()

    finite = np.isfinite(a)

    print("\n" + "=" * 80)
    print(f"{label} attenuation diagnostics")
    print("=" * 80)
    print("finite count:", int(finite.sum()))
    print("wet fraction:", float(np.nanmean(wet)))
    if finite.any():
        print("A_obs min/max:", float(np.nanmin(a[finite])), float(np.nanmax(a[finite])))
        print("A_obs mean all:", float(np.nanmean(a[finite])))
        print("A_obs > 0 fraction:", float(np.nanmean((a[finite] > 0))))
        print("A_obs percentiles all:",
            np.nanpercentile(a[finite], [50, 75, 90, 95, 99, 99.5, 99.9]))
    else:
        print("No finite A_obs values.")
        return

    if np.any(wet & finite):
        print("A_obs wet mean:", float(np.nanmean(a[wet & finite])))
        print("A_obs wet percentiles:",
              np.nanpercentile(a[wet & finite], [50, 75, 90, 95, 99]))
    print("=" * 80)

def run_one_output_day(
    *,
    output_day: pd.Timestamp,
    matched_metadata: pd.DataFrame,
    file_datetimes: list[tuple[datetime, str]],
) -> list[str]:
    """
    Run the Bas/statistics wet-dry CML rainfall workflow for one output day.

    The retrieval uses the lookback window for dry-baseline estimation, but
    gridding and NetCDF export are restricted to output_day only.
    """

    latest_dt = datetime(
        int(output_day.year),
        int(output_day.month),
        int(output_day.day),
        target_hour,
        target_minute,
    )

    cutoff_time = latest_dt - timedelta(hours=lookback_hours)
    recent_files = [(dt, f) for dt, f in file_datetimes if cutoff_time <= dt <= latest_dt]
    recent_files.sort(key=lambda x: x[0])

    if not recent_files:
        raise RuntimeError("No raw CML files were found within the requested lookback window.")

    print("First file:", recent_files[0])
    print("Last file: ", recent_files[-1])
    print("Number of files:", len(recent_files))

    # -------------------------------------------------------------------------
    # 2. Couple raw CML files with metadata
    # -------------------------------------------------------------------------
    coupled_frames = []
    for idx, (_, fname) in enumerate(recent_files):
        if idx % 10 == 0:
            print(f"Coupled {idx} / {len(recent_files)} files")

        raw_file = raw_cml_dir / fname
        frame = cml2metadata_coupling_framework_fast(str(raw_file), matched_metadata)
        if frame is not None and not frame.empty:
            coupled_frames.append(frame)

    if not coupled_frames:
        raise RuntimeError("No raw CML files could be coupled to metadata.")

    df_raw = pd.concat(coupled_frames, ignore_index=True)
    print("Raw coupled rows:", len(df_raw))

    # -------------------------------------------------------------------------
    # 3. Clean signal and build strict 15-min link-time table
    # -------------------------------------------------------------------------
    cfg = R0AutoConfig(semantics="auto", regularize_grid=True)
    df_clean, df_summary = clean_minmax_auto(df_raw, cfg)
    print(df_summary.head(10))

    ts_15 = build_15min_timeseries(df_clean)
    gc.collect()

    # -------------------------------------------------------------------------
    # 4. Statistics-based wet/dry-first retrieval
    # -------------------------------------------------------------------------
    df_A, wetdry_thresholds = stats_wetdry_Aobs(
        ts_15,
        window_bins=10,
        std_percentile=90.0,
        baseline_win="48H",
        baseline_q=0.75,
        min_dry_bins=8,
        ffill_limit_bins=32,
        require_src_present=True,
    )

    print("Wet/dry rolling-std thresholds by frequency family:")
    for k, v in wetdry_thresholds.items():
        print(f"  {k}: {v:.4f} dB")

    df_A = add_spatial_wetdry_support(
        df_A,
        radius_km=25.0,
        min_neighbors=3,
        min_neighbor_wet_fraction=0.50,
        wet_col="wet_final",
    )

    quick_Aobs_diagnostics(df_A, label="Bas method before k-alpha")

    df_rate = rainlink_strict_R(
        df_A,
        R_min=0.0,
        wet_col="wet_final",
        set_dry_to_zero=True,
    )

    link_rain_csv = output_dir / f"link_level_cml_rainfall_stats_wetdry_{output_day.strftime('%Y%m%d')}.csv"
    df_rate.to_csv(link_rain_csv, index=False)
    print("Saved link-level rainfall:", link_rain_csv)

    def quick_link_rain_diagnostics(df_rate, label="df_rate"):
        r = pd.to_numeric(df_rate["R_mm_per_h"], errors="coerce").to_numpy(float)
        finite = np.isfinite(r)
        wet01 = finite & (r >= 0.10)
        wet03 = finite & (r >= 0.30)
        wet08 = finite & (r >= 0.80)

        print("\n" + "=" * 80)
        print(f"{label} link-rain diagnostics")
        print("=" * 80)
        print("finite count:", int(finite.sum()))
        if finite.any():
            print("min/max:", float(np.nanmin(r[finite])), float(np.nanmax(r[finite])))
            print("mean all:", float(np.nanmean(r[finite])))
        else:
            print("No finite rain-rate values.")
            return
        print("zero fraction:", float(np.nanmean(r[finite] == 0.0)))
        print("wet fraction >= 0.10:", float(np.nanmean(wet01[finite])))
        print("wet fraction >= 0.30:", float(np.nanmean(wet03[finite])))
        print("wet fraction >= 0.80:", float(np.nanmean(wet08[finite])))

        if wet01.any():
            print("mean wet >= 0.10:", float(np.nanmean(r[wet01])))
            print("wet percentiles >= 0.10:",
                np.nanpercentile(r[wet01], [50, 75, 90, 95, 99]))

        print("all percentiles:",
            np.nanpercentile(r[finite], [50, 75, 90, 95, 99, 99.5, 99.9]))
        print("=" * 80)
    quick_link_rain_diagnostics(df_rate, label="Bas method before gridding")

    # -------------------------------------------------------------------------
    # 5. Prepare selected output day for gridding
    # -------------------------------------------------------------------------
    df_s5, meta_xy_grid = prepare_inputs_for_gridding(df_rate, ts_15)
    df_s5_day = df_s5[df_s5.index.date == output_day.date()]

    if df_s5_day.empty:
        raise RuntimeError(f"No link rainfall data are available for {output_day.date()} after retrieval.")

    # -------------------------------------------------------------------------
    # 6. Gridding and support/confidence layers
    # -------------------------------------------------------------------------
    grid_ds, diag = grid_rain_15min_rainlink_ok_full_ghana(
        df_s5=df_s5_day,
        df_meta_for_xy=meta_xy_grid,
        fixed_extent=(-3.5, 1.5, 4.5, 11.5),
        grid_res_deg=0.03,

        wet_thr=0.3,
        dry_thr=0.05,

        ok_model="exponential",
        ok_range_km=22.0,
        ok_nugget_frac=0.4,
        min_pts_ok=15,

        support_geometry="link_path",
        n_support_points_per_link=5,
        support_point_spacing_km=2.0,
        min_support_points_per_link=3,
        max_support_points_per_link=7,
        use_length_conditioned_support_points=True,

        support_k=2,
        support_radius_km=30.0,
        dry_radius_km=3.0,
        use_dry_constraint=True,

        use_soft_confidence=True,
        confidence_floor=0.03,
        confidence_power=1.1,
        confidence_dry_penalty_weight=0.50,

        drizzle_to_zero=0.1,
        outside_support_fill=np.nan,
        insufficient_training_fill=np.nan,

        smooth_kernel_px=1,
        smooth_fill_holes=False,

        clean_support=True,
        support_closing_iters=1,
        support_fill_holes=True,
        support_max_hole_px=20,

        make_display_field=True,
        display_smooth_kernel_px=3,
        display_smooth_fill_holes=False,
        apply_display_edge_taper=True,
        display_edge_taper_pixels=6,
        display_edge_taper_min_weight=0.05,

        make_coverage_quality=True,
        coverage_quality_med_thr=0.50,
        coverage_quality_high_thr=0.75,

        n_jobs=20,
        parallel_backend_name="processes",
        return_dataset=True,
    )

    print("Gridding counts:", diag.get("counts"))
    cov_counts = diag.get("coverage_quality_counts", [])
    print("Coverage-quality counts first 3:", cov_counts[:3])
    print("Coverage-quality counts last 3:", cov_counts[-3:])

    n_grid_times = int(grid_ds.sizes["time"])
    print("Generated timestamps:", n_grid_times)

    if n_grid_times == 0:
        raise RuntimeError("Gridding produced zero timestamps; nothing to save.")

    # -------------------------------------------------------------------------
    # 6b. Optional diagnostics
    # -------------------------------------------------------------------------
    if make_daily_diagnostics and grid_ds.sizes["time"] > 0:
        rain_max_by_time = grid_ds["R_mm_per_h"].max(dim=("lat", "lon"), skipna=True)
        imax = int(rain_max_by_time.argmax(dim="time").values)
        t_peak = pd.to_datetime(grid_ds["time"].values[imax])

        plot_cml_grid_diagnostics(
            grid_ds,
            meta_xy_grid,
            t_peak,
            rainfall_vmin=0.0,
            rainfall_vmax=12.0,
            save_path=diagnostics_dir / f"grid_diagnostic_peak_{pd.Timestamp(t_peak).strftime('%Y%m%dT%H%M')}.png",
        )

        plot_daily_mean_cml_rain_rate(
            grid_ds,
            meta_xy_grid,
            day=output_day.date(),
            vmin=0.0,
            vmax=4.0,
            save_path=diagnostics_dir / f"daily_mean_rain_rate_{output_day.strftime('%Y%m%d')}.png",
        )

        plot_daily_equivalent_cml_rainfall(
            grid_ds,
            meta_xy_grid,
            day=output_day.date(),
            vmin=0.0,
            vmax=80.0,
            save_path=diagnostics_dir / f"daily_equiv_rainfall_mm_day_{output_day.strftime('%Y%m%d')}.png",
        )

    # -------------------------------------------------------------------------
    # 7. Save one NetCDF per 15-min timestamp for this output day
    # -------------------------------------------------------------------------
    written_files = save_15min_grid_and_points_netcdf_for_day(
        grid_data=grid_ds,
        df_s5=df_s5_day,
        meta_xy=meta_xy_grid,
        out_dir=str(output_dir),
        day=output_day.strftime("%Y-%m-%d"),
        base_name=base_output_name,
        version=version,
        include_display_field=False,
    )

    print(f"Saved {len(written_files)} NetCDF files")
    for f in written_files[:5]:
        print("  ", f)

    if written_files:
        with xr.open_dataset(written_files[0]) as ds_check:
            print("Variables:", list(ds_check.data_vars))
            print("R dtype:", ds_check["R_mm_per_h"].dtype)
            print("Support-confidence dtype:", ds_check["cml_support_confidence"].dtype)
            print("Support-mask values:", np.unique(ds_check["cml_support_mask"].values))
            print("Coverage-quality values:", np.unique(ds_check["cml_coverage_quality"].values))

    # Release memory before next day.
    del df_raw, df_clean, ts_15, df_A, df_rate, df_s5, df_s5_day, grid_ds
    gc.collect()

    return written_files


# =============================================================================
# 2. Main multi-day driver
# =============================================================================

if __name__ == "__main__":
    matched_metadata = pd.read_csv(metadata_file)

    file_datetimes = discover_raw_files(raw_cml_dir)
    if not file_datetimes:
        raise RuntimeError(f"No timestamped raw CML files were found in {raw_cml_dir}")

    days = pd.date_range(training_start, training_end, freq="D")
    if max_days_to_process is not None:
        days = days[:max_days_to_process]

    success_rows = []
    failure_log = log_dir / "failed_training_period_days.txt"

    print("=" * 80)
    print("CML-only Bas/statistics wet-dry reference generation")
    print("Training/reference period:", training_start, "to", training_end)
    print("Number of output days:", len(days))
    print("Output directory:", output_dir)
    print("=" * 80)

    for i, day in enumerate(days, start=1):
        day = pd.Timestamp(day)
        day_str = day.strftime("%Y-%m-%d")

        print("\n" + "#" * 80)
        print(f"Processing day {i}/{len(days)}: {day_str}")
        print("#" * 80)

        if skip_existing_days and day_has_existing_outputs(output_dir, base_output_name, day):
            print(f"Skipping {day_str}: existing NetCDF outputs found.")
            success_rows.append({"day": day_str, "status": "skipped_existing", "n_files": np.nan})
            continue

        try:
            written_files = run_one_output_day(
                output_day=day,
                matched_metadata=matched_metadata,
                file_datetimes=file_datetimes,
            )
            success_rows.append({"day": day_str, "status": "success", "n_files": len(written_files)})

        except Exception as exc:
            print(f"FAILED {day_str}: {exc}")
            write_day_failure(failure_log, day_str, exc)
            success_rows.append({"day": day_str, "status": "failed", "n_files": 0})
            continue

    summary = pd.DataFrame(success_rows)
    summary_file = log_dir / f"training_period_generation_summary_{training_start.replace('-', '')}_{training_end.replace('-', '')}.csv"
    summary.to_csv(summary_file, index=False)

    print("\n" + "=" * 80)
    print("Finished training-period reference generation")
    print(summary["status"].value_counts(dropna=False))
    print("Summary saved:", summary_file)
    print("=" * 80)


#%%
# plot_one_cml_nc_ghana_map.py
#
# Plot one 15-min CML rainfall NetCDF file on a Ghana border map.

# import numpy as np
# import pandas as pd
# import xarray as xr
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature


# # ============================================================
# # 1. USER SETTINGS
# # ============================================================

# nc_file = (
#     "/home/kkumah/Projects/cml-stuff/"
#     "new_out_cml_Rain_bas_stats_training_ref_sens_q75_p90_wet01/"
#     "ghana_cml_R_15min_bas_stats_ref_q75_p90_wet01_20250615T001500Z.nc"
# )

# rain_var = "R_mm_per_h"

# # Ghana map extent: lon_min, lon_max, lat_min, lat_max
# extent = (-3.5, 1.5, 4.5, 11.5)

# # Plot controls
# vmin = 0.0
# vmax = 8.0
# nbins = 15
# cmap = "Spectral_r"

# plot_link_points = True
# plot_support_mask_outline = True

# save_plot = True
# out_png = "ghana_cml_rainfall_one_file.png"


# # ============================================================
# # 2. OPEN DATA
# # ============================================================

# ds = xr.open_dataset(nc_file)

# # print(ds)
# print("Variables:", list(ds.data_vars))

# if rain_var not in ds:
#     raise KeyError(f"{rain_var} not found. Available variables: {list(ds.data_vars)}")

# R = ds[rain_var]

# # If time dimension exists with length 1, select the first/only time.
# if "time" in R.dims:
#     R2d = R.isel(time=0)
#     plot_time = pd.to_datetime(ds["time"].values[0])
# else:
#     R2d = R
#     plot_time = "unknown time"

# # Convert encoded fill values to NaN if needed.
# R2d = R2d.where(np.isfinite(R2d))
# R2d = R2d.where(R2d > -9990)

# # Also mask unsupported pixels if support mask exists.
# if "cml_support_mask" in ds:
#     support = ds["cml_support_mask"]
#     if "time" in support.dims:
#         support2d = support.isel(time=0)
#     else:
#         support2d = support

#     # Keep rainfall only where support mask == 1
#     R2d = R2d.where(support2d == 1)
# else:
#     support2d = None


# # ============================================================
# # 3. COLOR LEVELS
# # ============================================================

# levels = np.linspace(vmin, vmax, nbins + 1)
# norm = mcolors.BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=False)


# # ============================================================
# # 4. PLOT GHANA MAP
# # ============================================================

# fig = plt.figure(figsize=(8, 9))
# ax = plt.axes(projection=ccrs.PlateCarree())

# ax.set_extent(extent, crs=ccrs.PlateCarree())

# # Background map features
# ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.25)
# ax.add_feature(cfeature.OCEAN, facecolor="white")
# ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
# ax.add_feature(cfeature.BORDERS, linewidth=1.1)
# ax.add_feature(cfeature.LAKES, linewidth=0.4, edgecolor="black", facecolor="none")
# ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.5)

# # Rainfall field
# pcm = ax.pcolormesh(
#     R2d["lon"].values,
#     R2d["lat"].values,
#     R2d.values,
#     transform=ccrs.PlateCarree(),
#     cmap=cmap,
#     norm=norm,
#     shading="auto",
# )

# # Optional support-mask contour
# if plot_support_mask_outline and support2d is not None:
#     ax.contour(
#         support2d["lon"].values,
#         support2d["lat"].values,
#         support2d.values,
#         levels=[0.5],
#         colors="black",
#         linewidths=0.6,
#         transform=ccrs.PlateCarree(),
#     )

# # Optional link midpoint overlay
# if plot_link_points:
#     if ("link_lon" in ds) and ("link_lat" in ds):
#         link_lon = ds["link_lon"].values
#         link_lat = ds["link_lat"].values

#         if "R_point_mm_per_h" in ds:
#             Rpt = ds["R_point_mm_per_h"]
#             if "time" in Rpt.dims:
#                 Rpt = Rpt.isel(time=0)

#             rpoint = Rpt.values
#             wet_links = np.isfinite(rpoint) & (rpoint >= 0.10)
#             dry_links = np.isfinite(rpoint) & (rpoint < 0.10)

#             ax.scatter(
#                 link_lon[dry_links],
#                 link_lat[dry_links],
#                 s=8,
#                 marker="o",
#                 color="gray",
#                 alpha=0.35,
#                 transform=ccrs.PlateCarree(),
#                 label="dry/weak links",
#                 zorder=5,
#             )

#             ax.scatter(
#                 link_lon[wet_links],
#                 link_lat[wet_links],
#                 s=18,
#                 marker="o",
#                 color="black",
#                 alpha=0.85,
#                 transform=ccrs.PlateCarree(),
#                 label="wet links",
#                 zorder=6,
#             )
#         else:
#             ax.scatter(
#                 link_lon,
#                 link_lat,
#                 s=8,
#                 color="black",
#                 alpha=0.5,
#                 transform=ccrs.PlateCarree(),
#                 label="CML link midpoints",
#                 zorder=5,
#             )

# # Gridlines
# gl = ax.gridlines(
#     draw_labels=True,
#     linewidth=0.4,
#     color="gray",
#     alpha=0.5,
#     linestyle="--",
# )
# gl.top_labels = False
# gl.right_labels = False

# # Colorbar
# cbar = plt.colorbar(
#     pcm,
#     ax=ax,
#     orientation="vertical",
#     pad=0.03,
#     shrink=0.82,
#     boundaries=levels,
#     ticks=levels[::2],
#     extend="max",
# )
# cbar.set_label("CML rainfall rate (mm h$^{-1}$)")

# # Title
# ax.set_title(
#     f"Ghana CML rainfall\n{plot_time}",
#     fontsize=14,
# )

# if plot_link_points:
#     ax.legend(loc="lower left", fontsize=8, frameon=True)

# plt.tight_layout()

# if save_plot:
#     plt.savefig(out_png, dpi=200, bbox_inches="tight")
#     print("Saved plot:", out_png)

# plt.show()

# ds.close()

#%%
# plot_daily_cml_equivalent_rainfall_ghana.py
#
# Daily CML rainfall diagnostic:
#   daily_equivalent_mm_day = mean(15-min R_mm_per_h maps over day) * 24
#
# This is daily-equivalent rainfall in mm/day.
# It is not strict accumulation unless all 96 15-min maps are present.

# import os
# import glob
# import numpy as np
# import pandas as pd
# import xarray as xr
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

# import cartopy.crs as ccrs
# import cartopy.feature as cfeature


# # ============================================================
# # 1. USER SETTINGS
# # ============================================================

# cml_nc_dir = (
#     "/home/kkumah/Projects/cml-stuff/"
#     "new_out_cml_Rain_bas_stats_training_ref_sens_q75_p90_wet01"
# )

# target_day = "2025-06-15"

# file_pattern = "ghana_cml_R_15min_bas_stats_ref_q75_p90_wet01_*.nc"

# rain_var = "R_mm_per_h"
# support_var = "cml_support_mask"

# # Ghana map extent: lon_min, lon_max, lat_min, lat_max
# extent = (-3.5, 1.5, 4.5, 11.5)

# # Plot controls
# vmin = 0.0
# vmax = 70.0
# nbins = 20
# cmap = "Spectral_r"

# plot_link_points = True
# plot_support_outline = True

# save_plot = True
# out_png = f"ghana_cml_daily_equivalent_{target_day}.png"


# # ============================================================
# # 2. FIND FILES FOR SELECTED DAY
# # ============================================================

# day_tag = pd.Timestamp(target_day).strftime("%Y%m%d")

# all_files = sorted(glob.glob(os.path.join(cml_nc_dir, file_pattern)))

# day_files = [
#     f for f in all_files
#     if day_tag in os.path.basename(f)
# ]

# if len(day_files) == 0:
#     raise FileNotFoundError(
#         f"No NetCDF files found for {target_day} in:\n{cml_nc_dir}"
#     )

# print(f"Found {len(day_files)} files for {target_day}")
# print("First file:", os.path.basename(day_files[0]))
# print("Last file :", os.path.basename(day_files[-1]))


# # ============================================================
# # 3. OPEN AND COMBINE DAILY FILES
# # ============================================================

# ds = xr.open_mfdataset(
#     day_files,
#     combine="nested",
#     concat_dim="time",
#     data_vars="all",
#     coords="minimal",
#     compat="override",
#     join="override",
#     parallel=False,
# )

# # Ensure chronological order
# ds = ds.sortby("time")

# # print(ds)
# print("Time coverage:", pd.to_datetime(ds.time.values[0]), "to", pd.to_datetime(ds.time.values[-1]))
# print("Number of 15-min maps:", ds.sizes["time"])


# # ============================================================
# # 4. CLEAN RAINFALL FIELD
# # ============================================================

# R = ds[rain_var].astype("float32")

# # Convert encoded missing values to NaN
# R = R.where(np.isfinite(R))
# R = R.where(R > -9990)

# # Optional: mask unsupported pixels before daily averaging
# if support_var in ds:
#     support = ds[support_var]
#     R = R.where(support == 1)

# # Optional: remove negative values if any slipped through
# R = R.where(R >= 0.0)


# # ============================================================
# # 5. DAILY-EQUIVALENT RAINFALL
# # ============================================================
# # Since R is mm/h:
# # daily-equivalent mm/day = mean rain rate over available 15-min maps * 24

# daily_equiv = R.mean(dim="time", skipna=True) * 24.0
# daily_equiv.name = "R_daily_equiv_mm_day"
# daily_equiv.attrs["units"] = "mm day-1"

# # Useful diagnostics
# n_times = ds.sizes["time"]
# coverage_fraction = n_times / 96.0

# print("\nDaily diagnostics")
# print("-----------------")
# print("Expected full-day 15-min maps:", 96)
# print("Available maps:", n_times)
# print("Temporal coverage fraction:", coverage_fraction)
# print("Daily-equivalent min/max:", float(daily_equiv.min(skipna=True)), float(daily_equiv.max(skipna=True)))
# print("Daily-equivalent mean:", float(daily_equiv.mean(skipna=True)))

# if n_times < 96:
#     print(
#         "\nNOTE: fewer than 96 maps are available. "
#         "This is mean rain rate × 24, not strict observed accumulation."
#     )


# # ============================================================
# # 6. DAILY SUPPORT MASK / VALID COVERAGE
# # ============================================================

# # Fraction of available timesteps where each pixel had valid CML support.
# valid_fraction = R.notnull().mean(dim="time")
# valid_fraction.name = "daily_valid_fraction"

# # A loose daily support outline: pixels valid at least 10% of available times.
# daily_support_mask = valid_fraction >= 0.10


# # ============================================================
# # 7. LINK MIDPOINT SUMMARY FOR DAILY OVERLAY
# # ============================================================

# link_lon = None
# link_lat = None
# wet_link_daily = None

# if plot_link_points and ("link_lon" in ds) and ("link_lat" in ds):
#     link_lon = ds["link_lon"].isel(time=0).values if "time" in ds["link_lon"].dims else ds["link_lon"].values
#     link_lat = ds["link_lat"].isel(time=0).values if "time" in ds["link_lat"].dims else ds["link_lat"].values

#     if "R_point_mm_per_h" in ds:
#         Rpt = ds["R_point_mm_per_h"].astype("float32")
#         Rpt = Rpt.where(np.isfinite(Rpt))
#         Rpt = Rpt.where(Rpt > -9990)
#         Rpt = Rpt.where(Rpt >= 0.0)

#         # Daily-equivalent link rainfall: mean link rain rate × 24
#         Rpt_daily = Rpt.mean(dim="time", skipna=True) * 24.0
#         wet_link_daily = Rpt_daily.values >= 1.0  # mm/day threshold for display only


# # ============================================================
# # 8. PLOT DAILY MAP
# # ============================================================

# levels = np.linspace(vmin, vmax, nbins + 1)
# norm = mcolors.BoundaryNorm(levels, ncolors=plt.get_cmap(cmap).N, clip=False)

# fig = plt.figure(figsize=(8.5, 9.5))
# ax = plt.axes(projection=ccrs.PlateCarree())

# ax.set_extent(extent, crs=ccrs.PlateCarree())

# # Background features
# ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.25)
# ax.add_feature(cfeature.OCEAN, facecolor="white")
# ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
# ax.add_feature(cfeature.BORDERS, linewidth=1.1)
# ax.add_feature(cfeature.LAKES, linewidth=0.4, edgecolor="black", facecolor="none")
# ax.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.5)

# # Rain field
# pcm = ax.pcolormesh(
#     daily_equiv["lon"].values,
#     daily_equiv["lat"].values,
#     daily_equiv.values,
#     transform=ccrs.PlateCarree(),
#     cmap=cmap,
#     norm=norm,
#     shading="auto",
# )

# # Daily support outline
# if plot_support_outline:
#     ax.contour(
#         daily_support_mask["lon"].values,
#         daily_support_mask["lat"].values,
#         daily_support_mask.values.astype(int),
#         levels=[0.5],
#         colors="black",
#         linewidths=0.7,
#         transform=ccrs.PlateCarree(),
#     )

# # Link midpoint overlay
# if plot_link_points and link_lon is not None and link_lat is not None:
#     if wet_link_daily is not None:
#         ax.scatter(
#             link_lon[~wet_link_daily],
#             link_lat[~wet_link_daily],
#             s=8,
#             color="gray",
#             alpha=0.35,
#             transform=ccrs.PlateCarree(),
#             label="low daily link rain",
#             zorder=5,
#         )

#         ax.scatter(
#             link_lon[wet_link_daily],
#             link_lat[wet_link_daily],
#             s=18,
#             color="black",
#             alpha=0.85,
#             transform=ccrs.PlateCarree(),
#             label="wet daily links",
#             zorder=6,
#         )
#     else:
#         ax.scatter(
#             link_lon,
#             link_lat,
#             s=8,
#             color="black",
#             alpha=0.5,
#             transform=ccrs.PlateCarree(),
#             label="CML link midpoints",
#             zorder=5,
#         )

# # Gridlines
# gl = ax.gridlines(
#     draw_labels=True,
#     linewidth=0.4,
#     color="gray",
#     alpha=0.5,
#     linestyle="--",
# )
# gl.top_labels = False
# gl.right_labels = False

# # Colorbar
# cbar = plt.colorbar(
#     pcm,
#     ax=ax,
#     orientation="vertical",
#     pad=0.03,
#     shrink=0.82,
#     boundaries=levels,
#     ticks=levels[::2],
#     extend="max",
# )
# cbar.set_label("Daily-equivalent CML rainfall (mm day$^{-1}$)")

# ax.set_title(
#     f"Ghana CML daily-equivalent rainfall\n"
#     f"{target_day} | mean of {n_times} × 15-min maps × 24 h",
#     fontsize=14,
# )

# if plot_link_points:
#     ax.legend(loc="lower left", fontsize=8, frameon=True)

# plt.tight_layout()

# if save_plot:
#     plt.savefig(out_png, dpi=220, bbox_inches="tight")
#     print("Saved plot:", out_png)

# plt.show()


# # ============================================================
# # 9. OPTIONAL: SAVE DAILY FIELD TO NETCDF
# # ============================================================

# save_daily_nc = True

# if save_daily_nc:
#     out_nc = f"ghana_cml_daily_equivalent_{day_tag}.nc"

#     ds_daily = xr.Dataset(
#         data_vars={
#             "R_daily_equiv_mm_day": daily_equiv.astype("float32"),
#             "daily_valid_fraction": valid_fraction.astype("float32"),
#         },
#         coords={
#             "lat": daily_equiv["lat"],
#             "lon": daily_equiv["lon"],
#         },
#         attrs={
#             "title": "Ghana CML daily-equivalent rainfall",
#             "day": target_day,
#             "method": "mean 15-min CML rain rate multiplied by 24",
#             "note": (
#                 "This is daily-equivalent rainfall in mm/day. "
#                 "If fewer than 96 15-min maps are available, it is not a strict accumulation."
#             ),
#             "n_15min_maps": int(n_times),
#             "expected_full_day_maps": 96,
#             "temporal_coverage_fraction": float(coverage_fraction),
#         },
#     )

#     ds_daily["R_daily_equiv_mm_day"].attrs.update({
#         "long_name": "Daily-equivalent CML rainfall",
#         "units": "mm day-1",
#     })

#     ds_daily["daily_valid_fraction"].attrs.update({
#         "long_name": "Fraction of available 15-min maps with valid supported CML rainfall",
#         "units": "1",
#     })

#     enc = {
#         "R_daily_equiv_mm_day": {
#             "zlib": True,
#             "complevel": 5,
#             "shuffle": True,
#             "dtype": "float32",
#             "_FillValue": -9999.0,
#         },
#         "daily_valid_fraction": {
#             "zlib": True,
#             "complevel": 5,
#             "shuffle": True,
#             "dtype": "float32",
#             "_FillValue": -9999.0,
#         },
#     }

#     ds_daily.to_netcdf(out_nc, encoding=enc)
#     print("Saved daily NetCDF:", out_nc)

# ds.close()