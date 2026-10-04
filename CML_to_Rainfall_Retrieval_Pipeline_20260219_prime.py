#%%
import gc
from CML_Rainfall_Retrieval_Pipeline_modes import *

#%% Define paths
metadata_path = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'
raw_cml_path = r'/home/kkumah/Projects/cml-stuff/data-cml/rsl'
out_CML_R_path = r'/home/kkumah/Projects/cml-stuff/new_out_15min_oper'
# r'/home/kkumah/Projects/cml-stuff/new_out_cml_Rain'
#%% Block 0 — Inputs: Gathering and Linking CML Data and Metadata
# e.g Schedule_pfm_SDH_20250919181726281472160571840_1.txt
matched_metadata = pd.read_csv(os.path.join(metadata_path, 'matched_metadata_kkk_20250527.csv'))

target_date = pd.to_datetime("2025-06-25 23:59:00") # or set to None to use latest available
dt_args = (int(target_date.year), int(target_date.month), int(target_date.day), int(target_date.hour), int(target_date.minute))

manual_latest_dt = datetime(*dt_args) if target_date else None # Example: datetime(2025, 8, 29, 12, 0)

file_names = os.listdir(raw_cml_path)
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
R_da_rl, diag_rl = grid_rain_15min_rainlink_ok(
    df_s5, 
    meta_xy_grid,
    grid_res_deg=0.03,
    domain_pad_deg=0.20,
    wet_thr=1.0,
    dry_thr=0.0,
    ok_model="exponential",
    ok_range_km=15.0,
    ok_nugget_frac=0.5,
    min_pts_ok=50,
    support_k=4,
    support_radius_km=40.0,
    drizzle_to_zero=0.5,     # you can change from default 0.10 if you like
    n_jobs=15,                 # or >1 if you want parallel
    parallel_backend_name="processes",
    outside_support_fill=np.nan,
    insufficient_training_fill=np.nan,
    smooth_kernel_px=3,
    smooth_fill_holes=True,
)
print(diag_rl)

#%% Block 7 — Saving outputs: Save slices
# from pipeline_modes import save_each_time_to_netcdf

# out_paths = save_each_time_to_netcdf(
#     R_da_rl,
#     out_dir=out_CML_R_path,
#     base_name="ghana_cml_R",
#     engine="netcdf4",
#     complevel=5,
#     dtype="float32",
#     fill_value=-9999.0,     # or np.nan if you prefer
#     chunks_lat=256,
#     chunks_lon=256,
#     keep_time_dim=True
# )
# print(f"Wrote {len(out_paths)} files. First:\n", out_paths[:3])

# 3) write daily file with both products
# out_file = save_daily_grid_and_points_netcdf(
#     R_da_day=R_da_rl,
#     df_s5_day=df_s5[["ID", "R_mm_per_h"]],
#     meta_xy=meta_xy_grid,                       # needs ID, XStart,YStart,XEnd,YEnd
#     out_dir=out_CML_R_path,
#     day=latest_dt.date(),
#     base_name="Ghana_cml_R",

#     # metadata
#     version="V1",
#     creator_name="TAHMO",
#     creator_email="kkumahkwabena@gmail.com",
#     project="PRIME Ghana CML rainfall retrieval",
#     references=None,
#     comment="Daily file contains gridded rainfall + link-midpoint rainfall points aligned on the same 15-min time axis.",
# )

# print("Wrote:", out_file)

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

# # Optional quick sanity check
# ds = xr.open_dataset(out_file)
# print(ds)
# print("Grid var:", ds["R_mm_per_h"].shape, "Point var:", ds["R_point_mm_per_h"].shape)

#%% Some diagnostics
# %%
# import matplotlib.pyplot as plt
# m = meta_xy_grid.copy()
# for c in ["XStart","YStart","XEnd","YEnd"]:
#     m[c] = pd.to_numeric(m[c], errors="coerce")

# m["lon_mid"] = 0.5*(m["XStart"] + m["XEnd"])
# m["lat_mid"] = 0.5*(m["YStart"] + m["YEnd"])

# lat_cut = 9.0  # tune
# north_ids = m.loc[m["lat_mid"] >= lat_cut, "ID"].unique()
# print("north links:", len(north_ids))

# t = pd.Timestamp("2025-06-19 16:15:00")

# pts_t = df_s5.loc[t].copy()
# if isinstance(pts_t, pd.Series):
#     pts_t = pts_t.to_frame().T

# pts_t = pts_t[pts_t["ID"].isin(north_ids)].copy()
# pts_t["R_mm_per_h"] = pd.to_numeric(pts_t["R_mm_per_h"], errors="coerce").fillna(0.0)

# wet_thr_mmph = 0.1
# north_wet = pts_t.loc[pts_t["R_mm_per_h"] > wet_thr_mmph].sort_values("R_mm_per_h", ascending=False)

# print("north wet links:", len(north_wet))
# north_wet.head(20)
# north_wet_ids = north_wet["ID"].unique()



# def plot_link_prime_diagnostics(link_id, t0=None, t1=None):
#     # link_id in dfA/df_rate uses link_id == ID
#     a = dfA[dfA["link_id"] == link_id].copy()
#     r = df_rate[df_rate["link_id"] == link_id].copy()

#     if a.empty:
#         print("No dfA rows for", link_id); return
#     a["time"] = pd.to_datetime(a["time"], utc=True, errors="coerce").dt.tz_localize(None)
#     r["time"] = pd.to_datetime(r["time"], utc=True, errors="coerce").dt.tz_localize(None)

#     if t0 is not None:
#         t0 = pd.Timestamp(t0)
#         a = a[a["time"] >= t0]; r = r[r["time"] >= t0]
#     if t1 is not None:
#         t1 = pd.Timestamp(t1)
#         a = a[a["time"] <= t1]; r = r[r["time"] <= t1]

#     fig, ax = plt.subplots(4, 1, figsize=(14, 9), sharex=True)

#     ax[0].plot(a["time"], a["sig_db"], lw=1, label="sig_db")
#     ax[0].plot(a["time"], a["baseline_rsl"], lw=2, label="baseline (q90 past-only)")
#     ax[0].set_ylabel("dB-ish"); ax[0].legend(); ax[0].grid(alpha=0.2)

#     ax[1].plot(a["time"], a["A_obs_dB"], lw=1, label="A_obs_dB")
#     ax[1].axhline(0.5, ls="--", lw=1, label="wet_thr_db=0.5")  # change if needed
#     ax[1].set_ylabel("dB"); ax[1].legend(); ax[1].grid(alpha=0.2)

#     ax[2].plot(a["time"], a["wet_rl"].astype(int), lw=1, label="wet_rl (0/1)")
#     ax[2].set_ylabel("wet"); ax[2].set_yticks([0,1]); ax[2].legend(); ax[2].grid(alpha=0.2)

#     ax[3].plot(r["time"], r["R_mm_per_h"], lw=1, label="R_mm_per_h")
#     ax[3].set_ylabel("mm/h"); ax[3].legend(); ax[3].grid(alpha=0.2)

#     ax[0].set_title(link_id)
#     plt.tight_layout(); plt.show()

# # run on top 3 suspicious north links:
# for lid in north_wet_ids[:3]:
#     plot_link_prime_diagnostics(lid, t0="2025-06-18", t1="2025-06-20")


#%%
# %%
# %%
# import numpy as np
# import xarray as xr
# import pandas as pd
# import matplotlib.pyplot as plt

# # --- load ---
# ds = xr.open_dataset(out_file)

# # --- red-spot definition (same as before) ---
# t0 = pd.Timestamp("2025-06-19 16:15:00")
# lon0, lat0 = -2.25, 6.20
# R_km = 25.0

# # --- find links within radius (same logic as before) ---
# lon = ds["link_lon"].values.astype(float)
# lat = ds["link_lat"].values.astype(float)
# ids = ds["link_id"].values

# km_per_deg_lat = 111.0
# km_per_deg_lon = 111.0 * np.cos(np.deg2rad(lat0))
# dx = (lon - lon0) * km_per_deg_lon
# dy = (lat - lat0) * km_per_deg_lat
# dist_km = np.sqrt(dx * dx + dy * dy)

# idx = np.where(dist_km <= R_km)[0]  # link indices in radius

# # --- rank links by intensity at t0 and take top N ---
# rpt_t0 = ds["R_point_mm_per_h"].sel(time=t0).isel(link=idx).values.astype(float)
# good = np.isfinite(rpt_t0)
# idx2 = idx[good]
# rpt2 = rpt_t0[good]

# order = np.argsort(rpt2)[::-1]  # high -> low
# topN = 5  # change to 3, 5, 10...
# top_idx = idx2[order[:topN]]

# # --- time window around t0 ---
# t1 = t0 - pd.Timedelta("75min")   # e.g., 15:00 if t0=16:15
# t2 = t0 + pd.Timedelta("45min")   # e.g., 17:00
# da = ds["R_point_mm_per_h"].sel(time=slice(t1, t2)).isel(link=top_idx)
# dist = dist_km[L]
# lbl = f"{str(ids[L])[:35]}...  ({dist:.1f} km)"


# # --- plot time series for top links ---
# plt.figure(figsize=(11, 4.2))
# for j, L in enumerate(top_idx):
#     y = da.isel(link=j).values.astype(float)
#     dist = dist_km[L]
#     lbl = f"{str(ids[L])[:35]}...  ({dist:.1f} km)"
#     plt.plot(da["time"].values, y, marker="o", lw=2, label=lbl)

# plt.grid(alpha=0.25)
# plt.ylabel("R_point_mm_per_h [mm h$^{-1}$]")
# plt.xlabel("time (UTC)")
# plt.title(f"Top {topN} link midpoint rain rates within {R_km} km of (lon={lon0}, lat={lat0})")
# plt.legend(loc="upper right", fontsize=8)
# plt.tight_layout()
# plt.show()

#%%
# t = pd.Timestamp("2025-06-19 16:15:00")

# meta_xy = (
#     df_clean.reset_index()[["ID","XStart","YStart","XEnd","YEnd"]]
#     .drop_duplicates("ID")
# )

# R1, d1 = grid_rain_at_time_rainlink(
#     df_s5=df_s5[["ID", "R_mm_per_h"]],
#     df_meta_for_xy=meta_xy,
#     t=t,                      # naive UTC timestamp present in df_s5.index

#     # RainLINK-OK parameters:
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

# print(d1)
# R1.plot(cmap='Spectral_r',vmax=15)

# # If you already have a 2-D slice:
# R2d = R1  # (lat,lon)

# # meta_xy_grid returned by prepare_inputs_for_gridding, OR meta_xy you built
# plot_grid_with_wetdry_midpoints_discrete_linear(
#     R2d[0], df_s5, meta_xy_grid, t,
#     wet_thr_mmph=0.1,
#     extent=(-3.25, 1.2, 4.8, 11.15),
#     n_bins=15,   # 12 linear discrete steps
#     vmin=0.0,
#     vmax=15,  
#     cmap="Spectral_r",
# )

#%%
df_raw_trbs = df_raw.copy()
df_raw_trbs['time'] = pd.to_datetime(df_raw_trbs['DateTime'])

df_raw_trbs_sub = df_raw_trbs[df_raw_trbs['time'].dt.date == target_date.date()]

len(df_raw_trbs_sub['ID'].unique())

len(df_raw_trbs_sub['time'].unique())