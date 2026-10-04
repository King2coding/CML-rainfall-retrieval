#%%
import gc
from CML_Rainfall_Retrieval_Pipeline_modes_TAHMO_20260330 import *

#%% Define paths
metadata_path = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'
raw_cml_path = r'/home/kkumah/Projects/cml-stuff/data-cml/rsl_2026'
# r'/home/kkumah/Projects/cml-stuff/data-cml/rsl'
out_CML_R_path = r'/home/kkumah/Projects/cml-stuff/new_out_15min_oper'
# r'/home/kkumah/Projects/cml-stuff/new_out_cml_Rain'
#%% Block 0 — Inputs: Gathering and Linking CML Data and Metadata
# e.g Schedule_pfm_SDH_20250919181726281472160571840_1.txt
matched_metadata = pd.read_csv(os.path.join(metadata_path, 'matched_metadata_kkk_20250527.csv'))

target_date = None#pd.to_datetime("2025-06-25 23:59:00") # or set to None to use latest available
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

#%% Block 1 — Coupling raw CML to metadata

coupled_dat = []

for idx, f in enumerate(recent_files):

    filename = os.path.join(raw_cml_path, f[1])

    if idx % 5 == 0:
        print(f'Processed {idx} files')
    
    coupled_dat.append(cml2metadata_coupling_framework(cml=filename, metadat=matched_metadata))

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
df_s5_20250619 = df_s5[df_s5.index.date ==  latest_dt.date()] # pd.to_datetime("2025-06-19").date()
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

R_da_rl, diag_rl = grid_rain_15min_rainlink_ok_full_ghana(
    df_s5,
    meta_xy_grid,
    grid_res_deg=0.03,
    domain_pad_deg=0.20,                     # ignored when fixed_extent is used
    fixed_extent=(-4.0, 1.25, 4.5, 11.25),  # (lon_min, lon_max, lat_min, lat_max)
    wet_thr=1.0,
    dry_thr=0.0,
    ok_model="exponential",
    ok_range_km=15.0,
    ok_nugget_frac=0.5,
    min_pts_ok=50,
    support_k=4,
    support_radius_km=40.0,
    drizzle_to_zero=0.5,
    n_jobs=15,
    parallel_backend_name="processes",
    outside_support_fill=np.nan,
    insufficient_training_fill=np.nan,
    smooth_kernel_px=3,
    smooth_fill_holes=True,
)
print(diag_rl)
#%% Block 7 — Saving outputs: Save slices

out_files = save_15min_grid_and_points_netcdf_for_day(
    R_da=R_da_rl,          # your (time, lat, lon) rainfall
    df_s5=df_s5,   # your link table for that day
    meta_xy=meta_xy_grid,       # link endpoints
    out_dir=out_CML_R_path,
    day=latest_dt.date().strftime('%Y-%m-%d'),
    base_name="Ghana_cml_R",
    version="V1",
)
print("Wrote", len(out_files), "files for", latest_dt.date().strftime('%Y-%m-%d'))
