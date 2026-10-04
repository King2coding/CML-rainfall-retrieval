#%%
import gc
from CML_Rainfall_Retrieval_Pipeline_modes_TAHMO_20260418 import *

#%% Define paths
metadata_path = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'
raw_cml_path = r'/home/kkumah/Projects/cml-stuff/data-cml/rsl'
# r'/home/kkumah/Projects/cml-stuff/data-cml/rsl_2026'
# 
out_CML_R_path = r'/home/kkumah/Projects/cml-stuff/misc'
# r'/home/kkumah/Projects/cml-stuff/new_out_15min_oper'
# r'/home/kkumah/Projects/cml-stuff/new_out_cml_Rain'
#%% Block 0 — Inputs: Gathering and Linking CML Data and Metadata
# e.g Schedule_pfm_SDH_20250919181726281472160571840_1.txt
matched_metadata = pd.read_csv(os.path.join(metadata_path, 'matched_metadata_kkk_20250527.csv'))

target_date = pd.to_datetime("2025-06-15 23:59:00") # or set to None to use latest available
if target_date is not None:
    dt_args = (
        int(target_date.year),
        int(target_date.month),
        int(target_date.day),
        int(target_date.hour),
        int(target_date.minute),
    )
    manual_latest_dt = datetime(*dt_args)
else:
    dt_args = None
    manual_latest_dt = None

file_names = sorted(os.listdir(raw_cml_path))
file_datetimes = [(extract_datetime_from_filename(f), f) for f in file_names]
file_datetimes = [(dt, f) for dt, f in file_datetimes if dt is not None]

if manual_latest_dt is not None:
    latest_dt = manual_latest_dt
else:
    latest_dt, latest_file = max(file_datetimes, key=lambda x: x[0])

cutoff_time = latest_dt - timedelta(hours=72)

recent_files = [(dt, f) for dt, f in file_datetimes if cutoff_time <= dt <= latest_dt]
recent_files.sort(key=lambda x: x[0])
print("First file:", recent_files[0])
print("Last file: ",  recent_files[-1][1])
print("Number of files: " + str(len(recent_files)))
#%% Block 1 — Coupling raw CML to metadata

coupled_dat = []

for idx, f in enumerate(recent_files):

    filename = os.path.join(raw_cml_path, f[1])

    if idx % 5 == 0:
        print(f'Processed {idx} files')
    
    coupled_dat.append(cml2metadata_coupling_framework_fast(cml=filename, metadat=matched_metadata))

df_raw = pd.concat(coupled_dat, ignore_index=True)
#%% Block 2 — R0 Cleaning (quality control): The Preprocessing and Cleaning Pipeline
# 0) Load your linked dataframe (must contain ID, DateTime, Pmin, Pmax)
# df_raw = pd.read_csv(r'/home/kkumah/Projects/cml-stuff/data-cml/outs/Multi-Link-Multi-Timestamp_coupled_linkdata_kkk_20251006.csv')  # or CSV
# path_to_put_output = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'

# 1) Run R0 cleaner (with the new defaults)
cfg = R0AutoConfig(
    semantics="auto",          # let it decide per link
    regularize_grid=True,      # keep exact 15-min grid
)

df_clean, df_sum = clean_minmax_auto(df_raw, cfg)
print(df_sum.head(10))
gc.collect()

#%% The CML-Rainfall-Retrieval Pipeline

# Block 3 — Build strict 15-minute time series: strict 15-min series with Rainlink-style RSL
ts_15 = build_15min_timeseries(df_clean)

# Block 4 — Baseline + wet/dry classification: strict past-only baseline and observed attenuation
# dfA = rainlink_strict_Aobs(ts_15, wet_thr_db=0.5)
dfA = rainlink_strict_Aobs(ts_15, two_pass=True, use_drycount_guard=False)
# dfA = rainlink_strict_Aobs(ts_15, two_pass=True, use_drycount_guard=True, min_dry_bins=8, guard_behavior="fallback")

# Block 5 — Convert attenuation to rainfall rate (per link): Leijnse WA + ITU(2005) k–α → *allow true zeros*
df_rate = rainlink_strict_R(dfA, R_min=0.0)
# Prepare data for gridding
df_s5, meta_xy_grid = prepare_inputs_for_gridding(df_rate, ts_15)
gc.collect()
#%% Block 6 — Gridding (maps): The Gridding Pipeline
df_s5_20250615 = df_s5[df_s5.index.date ==  latest_dt.date()] # pd.to_datetime("2025-06-19").date()
# R_da_rl, diag_rl = grid_rain_15min_rainlink_ok(
#     df_s5, 
#     meta_xy_grid,
#     grid_res_deg=0.03,
#     domain_pad_deg=0.20,
#     wet_thr=1.0,
#     dry_thr=0.0,
#     ok_model="exponential",
#     ok_range_km=15.0,
#     ok_nugget_frac=0.5,
#     min_pts_ok=50,
#     support_k=4,
#     support_radius_km=40.0,
#     drizzle_to_zero=0.5,     # you can change from default 0.10 if you like
#     n_jobs=15,                 # or >1 if you want parallel
#     parallel_backend_name="processes",
#     outside_support_fill=np.nan,
#     insufficient_training_fill=np.nan,
#     smooth_kernel_px=3,
#     smooth_fill_holes=True,
# )
# print(diag_rl)
# df_meta = meta_xy_grid.copy()
# R_da, diag = grid_rain_15min_rainlink_ok_full_ghana(
#     df_s5=df_s5_20250619,
#     df_meta_for_xy=meta_xy_grid,
#     fixed_extent=(-3.5, 1.5, 4.5, 11.5),
#     grid_res_deg=0.03,
#     wet_thr=0.8,
#     dry_thr=0.05,
#     ok_model="exponential",
#     ok_range_km=22.0, # 22.0
#     ok_nugget_frac=0.4,
#     min_pts_ok=15,
#     support_k=2,
#     support_radius_km=40.0,
#     dry_radius_km=3.0,
#     use_dry_constraint=True,
#     use_soft_confidence=True,
#     confidence_floor=0.03,# 0.03
#     confidence_power=1.1, # 1.1
#     drizzle_to_zero=0.10,
#     smooth_kernel_px=2, # 
#     smooth_fill_holes=False,
#     outside_support_fill=np.nan,
#     insufficient_training_fill=np.nan,
#     n_jobs=20,
# )
# print(diag)

grid_ds, diag = grid_rain_15min_rainlink_ok_full_ghana(
    df_s5=df_s5_20250615,
    df_meta_for_xy=meta_xy_grid,
    fixed_extent=(-3.5, 1.5, 4.5, 11.5),
    grid_res_deg=0.03,

    wet_thr=0.8,
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

    drizzle_to_zero=0.10,

    smooth_kernel_px=1,
    smooth_fill_holes=False,

    clean_support=True,
    support_closing_iters=1,
    support_fill_holes=True,
    support_max_hole_px=20,

    make_display_field=True,
    display_smooth_kernel_px=4,
    display_smooth_fill_holes=False,

    apply_display_edge_taper=True,
    display_edge_taper_pixels=8,
    display_edge_taper_min_weight=0.02,

    make_coverage_quality=True,
    coverage_quality_med_thr=0.50,
    coverage_quality_high_thr=0.75,

    outside_support_fill=np.nan,
    insufficient_training_fill=np.nan,

    n_jobs=20,
    return_dataset=True,
)

print(diag["counts"])
print(diag["coverage_quality_counts"][:3])
#%% Block 7 — Saving outputs: Save slices

written_files = save_15min_grid_and_points_netcdf_for_day(
    grid_data=grid_ds,
    df_s5=df_s5_20250615,
    meta_xy=meta_xy_grid,
    out_dir=out_CML_R_path,
    day="2025-06-15",
    base_name="ghana_cml_R_15min",
    include_display_field=False,   # recommended for Rainboo/TAHMO operational files
)

ds_check = xr.open_dataset(written_files[0])

print(ds_check["R_mm_per_h"].dtype)
print(ds_check["cml_support_confidence"].dtype)
print(ds_check["cml_support_mask"].dtype)
print(ds_check["cml_coverage_quality"].dtype)

print(np.unique(ds_check["cml_support_mask"].values))
print(np.unique(ds_check["cml_coverage_quality"].values))

print(list(ds_check.data_vars))
#%%

plot_cml_grid_diagnostics(
    grid_ds,
    meta_xy_grid,
    t0="2025-06-15 03:45:00",
    extent=(-3.5, 1.5, 4.5, 11.5),
)
gc.collect()


plot_cml_grid_diagnostics(
    grid_ds,
    meta_xy_grid,
    t0="2025-06-15 03:45:00",
    vars_to_plot=[
        "R_mm_per_h",
        "R_display_mm_per_h",
        "cml_support_confidence",
        "cml_support_mask",
    ],
)
gc.collect()

plot_cml_grid_diagnostics(
    grid_ds,
    meta_xy_grid,
    t0="2025-06-15 03:45:00",
    vars_to_plot=[
        "R_mm_per_h",
        "R_display_mm_per_h",
    ],
    figsize=(13, 6),
)

gc.collect()