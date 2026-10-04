#%%
import gc
from My_program_utils import *
# R0 — Prior processing & data hygiene (per-link, unified method)
# Import necessary libraries
import pandas as pd
from r0_clean_minmax_auto import clean_minmax_auto, R0AutoConfig
from r0_qc_helpers import masking_report, plot_flags, plot_minmax_basic
import matplotlib.dates as mdates
from persist_utils import save_r0_outputs
# 0) Load your linked dataframe (must contain ID, DateTime, Pmin, Pmax)
# out_fname = os.path.join(path_to_put_output, f'Multi-Link-Multi-Timestamp_coupled_linkdata_kkk_{cde_run_dte}.csv')
df_raw = pd.read_csv(r'/home/kkumah/Projects/cml-stuff/data-cml/outs/Multi-Link-Multi-Timestamp_coupled_linkdata_kkk_20250929.csv')  # or CSV
path_to_put_output = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'

# 1) Run R0 cleaner (with the new defaults)
cfg = R0AutoConfig(
    semantics="auto",          # let it decide per link
    regularize_grid=True,      # keep exact 15-min grid
)

df_clean, df_sum = clean_minmax_auto(df_raw, cfg)
print(df_sum.head(10))

# 2) QA & sanity checks
rep = masking_report(df_clean)
print(rep.head(15))

# 3) Pick a link and plot (your plotting recipe)
link_id = df_sum.loc[0, "ID"]
one_link = df_clean[df_clean["ID"] == link_id]
plot_minmax_basic(one_link)         # your basic plot

# 4) Overlay flags to see what got masked/flagged
plot_flags(df_clean, link_id)

# 5) Save outputs
paths_r0 = save_r0_outputs(df_clean, df_sum, cfg, out_dir="outputs")
print("R0 saved:", paths_r0)

#%% Step 1 Wet/Dry Classification
import pandas as pd
import matplotlib.dates as mdates
# ==== CLEAN, LINEAR PIPELINE ====
# R0 -> S1 (Δ-based NLA) -> S2 (48h baseline) -> S1b (excess NLA)
# -> S2b (integrate masks) -> S2c (wet-antenna correction) -> S3 (γ→R)
# + QA plots

import numpy as np, pandas as pd, importlib

# ---------- helpers ----------
def _utcify_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = out.index.tz_localize("UTC") if out.index.tz is None else out.index.tz_convert("UTC")
    return out

def _join_on_id_time(left: pd.DataFrame, right: pd.DataFrame, cols):
    L = left.set_index(["ID", left.index])             # (ID, time)
    R = right.set_index(["ID", right.index])[list(cols)]
    out = (L.join(R, how="left")
             .reset_index(level=0, drop=False)
             .sort_index())
    return out

# --- drop-in replacement helper (put near the top with the other helpers) ---
def _merge_on_id_time(left: pd.DataFrame, right: pd.DataFrame, cols, how="left"):
    """
    Robust merge on (ID, time). Works even if 'right' has duplicate (ID,time).
    Keeps the *last* occurrence per (ID,time) on the right.
    """
    L = _utcify_index(left).reset_index().rename(columns={"index": "time_utc"})
    R = _utcify_index(right).reset_index().rename(columns={"index": "time_utc"})
    R = R[["ID", "time_utc", *cols]].drop_duplicates(["ID", "time_utc"], keep="last")
    out = (L.merge(R, on=["ID", "time_utc"], how=how)
             .set_index("time_utc")
             .sort_index())
    return out

# ====== ONE KNOB ======
WETDRY_MODE = "strict"   # choose: "strict" (Rainlink-like) or "relaxed" (yours)

import importlib
import step1_wetdry_nla_fast2 as s1; importlib.reload(s1)
import step1b_wetdry_excess_fast as s1b; importlib.reload(s1b)

# --- Step 1: Δ-based NLA (strict/relaxed)
df_nla, s1_sum = s1.wetdry_classify_with_mode(_utcify_index(df_clean),
                                              mode=WETDRY_MODE,
                                              override={"n_jobs": 12})

# --- Step 2: 48h baseline (unchanged)
import step2_baseline_dry48 as s2; importlib.reload(s2)
from step2_baseline_dry48 import Baseline48Config, compute_dry_baseline_48h
cfg2 = Baseline48Config(window_hours=48, fallback_hours=72,
                        min_dry_samples=8, smooth_baseline_samples=0)
df_step2, s2_sum = compute_dry_baseline_48h(df_nla, cfg2)
df_step2 = _utcify_index(df_step2)

# --- Step 1b: Excess-based NLA (strict/relaxed)
df_ex, s1b_sum = s1b.wetdry_from_excess_with_mode(df_step2,
                                                  mode=WETDRY_MODE,
                                                  override={"n_jobs": 12})
df_ex = _utcify_index(df_ex)

# --- Step 2b: Merge masks (toggle onset rule by mode)
import step2b_integrate_masks as s2b; importlib.reload(s2b)
from step2b_integrate_masks import integrate_wetdry_and_excess

df_s12 = integrate_wetdry_and_excess(
    df_step2, df_nla, df_ex,
    onset_allow_if_excess_pos=(WETDRY_MODE != "strict")   # strict: OFF, relaxed: ON
).pipe(_utcify_index)
assert "is_wet_final" in df_s12.columns

# ---- ensure meta + pooled-excess-per-km are on df_s12 ----
# meta come from Step 2; pooled excess per-km comes from Step 1b
need_meta = ["PathLength", "Frequency", "Polarization"]
if not set(need_meta).issubset(df_s12.columns):
    df_s12 = _merge_on_id_time(df_s12, df_step2, need_meta)

if "A_ex_pool_per_km" not in df_s12.columns and "A_ex_pool_per_km" in df_ex.columns:
    df_s12 = _merge_on_id_time(df_s12, df_ex, ["A_ex_pool_per_km"])

# pick WA source automatically
wa_src = "A_ex_pool_per_km" if "A_ex_pool_per_km" in df_s12.columns else (
         "A_excess_db_per_km" if "A_excess_db_per_km" in df_s12.columns else None)
if wa_src is None:
    raise ValueError("No per-km excess column found. Need 'A_ex_pool_per_km' (preferred) or 'A_excess_db_per_km'.")

print("Wet-antenna γ source:", wa_src)

# ---------- STEP 2c (Wet-antenna correction → γ_corr with decay) ----------
import importlib
import step2b_wet_antenna as wa; importlib.reload(wa)
from step2b_wet_antenna import WAConfigV2, apply_wet_antenna_decay

cfg_wa = WAConfigV2(
    gamma_source_col=wa_src,         # auto-chosen earlier
    wet_mask_col="is_wet_final",
    wa_db_per_terminal=0.08,
    assume_wet_terminals=2,
    dt_minutes=15, tau_on_min=30, tau_off_min=60,
    out_raw_col="gamma_raw_db_per_km",
    out_corr_col="gamma_corr_db_per_km",
    floor_zero=True,
)
df_attn = apply_wet_antenna_decay(df_s12, cfg_wa)
df_attn = _utcify_index(df_attn)

# ---------- STEP 3 (γ → R with ITU k–α, gated) ----------
import step3_kalpha as s3; importlib.reload(s3)
from step3_kalpha import KAlphaConfigV2, gamma_to_r_gated

cfg_k = KAlphaConfigV2(
    lut=lut,
    pol_col="Polarization",
    freq_col="Frequency",
    gamma_col="gamma_corr_db_per_km",  # from WA V2
    gamma_gate_db_per_km=0.02,         # filter tiny γ
    use_wet_mask_col="is_wet_final",
    r_cap_mmph_by_band={6: 60, 8: 80, 19: 120},  # optional
)
df_s5, s5_sum = gamma_to_r_gated(df_attn, cfg_k)
df_s5 = _utcify_index(df_s5)
#%%

# # ---------- QA PLOTS ----------
# import importlib, nla_qc_helpers as qh
# importlib.reload(qh)

# qh.plot_wetdry_overlay(
#     df_nla, df_sum.loc[0,"ID"],
#     t0="2025-06-12", t1="2025-06-14",
#     thr_self_db=cfg1.thr_self_db, thr_nb_db=cfg1.thr_nb_db,
#     show_per_km=False
# )

# # Excess-based overlay (per-km thresholds)
# link_id = df_sum.loc[0, "ID"]
# dplot = df_ex[df_ex["ID"] == link_id].copy()
# dplot["is_wet"] = dplot["is_wet_excess"]  # helper expects 'is_wet'
# qh.plot_excess_overlay(
#     dplot, link_id,
#     t0="2025-06-12", t1="2025-06-14",
#     thr_self_db_per_km=cfg_ex.thr_self_db_per_km,
#     thr_nb_db_per_km=cfg_ex.thr_nb_db_per_km
# )

# # γ & R plot
# from step5_plot_helpers import plot_rainrate

# link_id = df_sum.loc[0, "ID"]
# plot_rainrate(df_s5, link_id, t0="2025-06-12", t1="2025-06-14",
#               wet_mask_col="is_wet_final")  # or "is_wet_excess" if that’s what you want to shade

# # ---------- tiny sanity printouts ----------
# print(s1_sum.head())
# print(s2_sum.head())
# print(s1b_sum.head())
# print(s5_sum.head())


#%%
# 6) 15-min gridding (IDW)
# import importlib, step6_grid_15min as g6
# importlib.reload(g6)
# from step6_grid_15min import Grid15Config, grid_15min_idw, save_grid_npz, quick_map

# # df_s5 = output from gamma_to_r (has R_mm_per_h) and a UTC DateTime index.
# # df_clean (or df_nla) has the link geometry. Either one is fine.

# cfg6 = Grid15Config(
#     res_deg=0.05,        # ≈5 km; tune to match MSG footprint if you like
#     pad_deg=0.10,        # small bbox padding
#     k=12, max_radius_km=50.0, power=2.0,
#     value_col="R_mm_per_h"   # grid 15-min totals to match MSG cadence
# )

# cube, meta = grid_15min_idw(df_s5, df_clean, cfg6)

# # # Save compactly (easy to reload later and pair with MSG):
# # save_grid_npz("outputs/rain_15min_idw.npz", cube, meta)

# # Quick look at a time slice (e.g., the big event peak):
# t_idx = int(np.argmax(cube.max(axis=(1,2))))  # frame with max rain
# quick_map(cube, meta, t_index=t_idx, vmin=0, vmax=2)

#%%
# ==== STEP 6 — 15-min Rain Gridding (OK/KED) ====
# import importlib, step6_grid_ok as s6ok, grid_driver as gd
# importlib.reload(s6ok); importlib.reload(gd)

# meta_xy = (df_clean.reset_index()[["ID","XStart","YStart","XEnd","YEnd"]]
#                      .drop_duplicates("ID"))

# R_ok = gd.grid_rain_15min(
#     df_s5=df_s5,
#     df_meta_for_xy=meta_xy,
#     method="ok",
#     grid_res_deg=0.05,
#     domain_pad_deg=0.2,
#     n_jobs=8,
#     variogram_model="exponential",
#     variogram_model_parameters=None,   # heuristic defaults from the earlier patch
#     nlags=12, eps=0.1, min_pts=5
# )
# for t in R_ok.time.values:
#     r2plt = R_ok.sel(time=t)
#     if np.nanmean(r2plt.values) > 0:
#         print("Max R at", pd.to_datetime(t), "is", np.nanmax(r2plt.values))
#         r2plt.plot(size=5, cmap="jet")


#%%
# import importlib, step6_grid_pycomlink as s6pcm
# importlib.reload(s6pcm)

# # midpoints from your R0-clean meta
# meta_xy = (df_clean.reset_index()[["ID","XStart","YStart","XEnd","YEnd"]]
#                     .drop_duplicates("ID"))

# R_grid, diag = s6pcm.grid_rain_15min(
#     df_s5=df_s5[["ID","R_mm_per_h"]],   # df_s5 index = UTC time
#     df_meta_for_xy=meta_xy,
#     grid_res_deg=0.05,
#     domain_pad_deg=0.2,
#     min_pts_ok=3,
#     variogram_model="exponential",
#     variogram_params=None,              # let library choose; robust fallbacks enabled
#     nlags=12,
#     idw_power=2,
#     n_jobs=8
# )

# print(diag)  # see how many slices used OK vs IDW

# # sanity: list rainy times
# has_rain = [pd.to_datetime(t) for t in R_grid.time.values
#             if np.nanmax(R_grid.sel(time=t).values) > 0]
# print("First rainy times:", has_rain[:10])

# # plot first rainy slice
# if has_rain:
#     R_grid.sel(time=has_rain[100]).plot(size=5, cmap="jet", vmin=0, vmax=8)

# wet_times = (df_s5.groupby(df_s5.index)["R_mm_per_h"].max() > 0.05)
# times = wet_times.index[wet_times]
# # pass times=times to the gridding call if your wrapper exposes it

#%%
# import importlib, step6_grid_ok_pcm as s6pcm
# importlib.reload(s6pcm)

# # meta for XY (one row per ID)
# meta_xy = (df_clean.reset_index()[["ID","XStart","YStart","XEnd","YEnd"]]
#            .drop_duplicates("ID"))

# R_grid, diag = s6pcm.grid_rain_15min(
#     df_s5=df_s5[["ID","R_mm_per_h"]],
#     df_meta_for_xy=meta_xy,
#     grid_res_deg=0.03,
#     domain_pad_deg=0.2,
#     idw_power=2.0,
#     idw_nnear=15,
#     idw_maxdist_km=35.0,     # hard IDW radius ~ like pycomlink example
#     max_dist_km_mask=40.0,   # coverage mask (same/slightly larger)
#     use_ok=True,
#     n_jobs=15,
# )

# print(diag)

import importlib
import step6_grid_ok_pcm as s6pcm
import numpy as np, pandas as pd
from plot_helpers import plot_slice_cartopy_with_links

# reload the gridding module after edits
importlib.reload(s6pcm)

# --- meta midpoints (one row per ID) ---
meta_xy = (df_clean.reset_index()[["ID","XStart","YStart","XEnd","YEnd"]]
                     .drop_duplicates("ID"))

# OPTIONAL: if you want to do drizzle mapping yourself (not required; the function can do it)
# thr = 0.010
# df_s5_in = df_s5.copy()
# df_s5_in["R_mm_per_h"] = df_s5_in["R_mm_per_h"].where(
#     (df_s5_in["R_mm_per_h"] >= thr) | (df_s5_in["R_mm_per_h"] == 0.0), 0.0
# )
# Otherwise, just pass df_s5 and set drizzle_to_zero below:
df_s5_in = df_s5

# t0 = "2025-06-11 12:00:00Z"
# Unique times with some rain signal
rr_max = df_s5.groupby(df_s5.index)["R_mm_per_h"].max()
rr_med = df_s5.groupby(df_s5.index)["R_mm_per_h"].median()
# choose frames with >0.5 mm/h median or >2 mm/h max (tune as needed)
times_to_try = rr_max[(rr_max > 2.0) | (rr_med > 0.5)].index.to_list()

for t in times_to_try[:5]:
    R1, d1 = s6pcm.grid_rain_at_time(
    df_s5=df_s5[["ID","R_mm_per_h"]],
    df_meta_for_xy=meta_xy,
    t=t,
    grid_res_deg=0.03, domain_pad_deg=0.2,
    drizzle_to_zero=0.02,   # instead of 0.10
    idw_power=2.0,          # instead of 3.0
    idw_nnear=15,           # back up a bit
    idw_maxdist_km=30.0,    # give it more reach
    max_dist_km_mask=35.0,  # slightly larger mask than idw radius
    smooth_kernel_px=3,     # gentle blur to remove blockiness      # was 3
    n_jobs=15,
    )
    print(d1["first5"][0])  # has method, n_wet, min, max for the time you gridded

    print("counts:", d1["counts"])
    print("ok_times:", d1["ok_times"][:5])

    # choose time to plot (OK time if available, else the one we gridded)
    if d1.get("ok_times"):
        t_plot = pd.Timestamp(d1["ok_times"][0])
    else:
        t_plot = pd.Timestamp(R1.time.values[0])

    # make sure it's tz-naive (selector expects naive)
    if t_plot.tzinfo is not None:
        t_plot = t_plot.tz_convert("UTC").tz_localize(None)

    fig, ax = plot_slice_cartopy_with_links(
        R1, meta_xy, t_plot,
        bounds=[0.1, 0.2, 0.5, 1, 2, 4, 7, 10, 20, 40],
        extent=(-4, 1.5, 4.5, 11.5),
        cmap_name="turbo",
    )

# plot a frame
# t_with_vals = [t for t in R_grid.time.values if np.nanmax(R_grid.sel(time=t).values) > 20]
# t = "2025-06-12 12:00:00+00:00" # t_with_vals[0]
# from plot_helpers import plot_slice_cartopy_with_links
# _ = plot_slice_cartopy_with_links(R_grid, df_clean, t,cmap_name='Blues')


# Res, diag = s6pcm.grid_rain_15min(
#     df_s5=df_s5[["ID","R_mm_per_h"]],
#     df_meta_for_xy=meta_xy,    
#     grid_res_deg=0.03, domain_pad_deg=0.2,
#     drizzle_to_zero=0.02,   # instead of 0.10
#     idw_power=2.0,          # instead of 3.0
#     idw_nnear=15,           # back up a bit
#     idw_maxdist_km=30.0,    # give it more reach
#     max_dist_km_mask=35.0,  # slightly larger mask than idw radius
#     smooth_kernel_px=3,     # gentle blur to remove blockiness      # was 3
#     n_jobs=15,
#     )
# print(diag)

# find times that could qualify for OK

# import pandas as pd
# import numpy as np

# # 1) Count "wet" links per 15-min time (same drizzle gate as your gridding call)
# thr = 0.02
# wet_counts = (df_s5["R_mm_per_h"].fillna(0) > thr).groupby(df_s5.index).sum().sort_values(ascending=False)
# print("Top 20 times by wet-link count:\n", wet_counts.head(20))

# # 2) Pick a candidate that has enough wet points for OK (>= min_pts_ok)
# min_pts_ok = 20
# candidates = wet_counts[wet_counts >= min_pts_ok]
# print("OK-eligible times:", list(candidates.index[:10]))

# # 3) Try one candidate with OK (relax min_pts_ok if needed)
# if len(candidates):
#     t_ok = candidates.index[0]
# else:
#     # no time meets 20 wet links → relax gate
#     min_pts_ok = 10
#     t_ok = wet_counts.index[0]
#     print(f"No time >=20 wet links. Trying {t_ok} with min_pts_ok={min_pts_ok}")

# R1, d1 = s6pcm.grid_rain_at_time(
#     df_s5=df_s5[["ID","R_mm_per_h"]],
#     df_meta_for_xy=meta_xy,
#     t=t_ok,
#     grid_res_deg=0.03, domain_pad_deg=0.2,
#     drizzle_to_zero=0.02,
#     use_ok=True, min_pts_ok=min_pts_ok, nlags=10, ok_range_km=25.0,
#     idw_power=2.0, idw_nnear=10, idw_maxdist_km=25.0,
#     max_dist_km_mask=50.0, smooth_kernel_px=3, n_jobs=8,
# )

# print("method counts:", d1["counts"])
# print("ok_times:", d1["ok_times"])

# fig, ax = plot_slice_cartopy_with_links(
#         R1, meta_xy, t_ok,
#         bounds=[0.1, 0.2, 0.5, 1, 2, 4, 7, 10, 20, 40],
#         extent=(-4, 1.5, 4.5, 11.5),
#         cmap_name="turbo",
#     )


#%%
# your module with robust time parsing
import importlib
import pipeline_modes as pm
importlib.reload(pm)
# from plot_helpers import plot_helpers as ph
# 1) strict 15-min series with Rainlink-style RSL
ts_15 = pm.build_15min_timeseries(df_clean)

# 2) strict past-only baseline and observed attenuation
dfA = pm.rainlink_strict_Aobs(ts_15, wet_thr_db=0.5)

# 3) Leijnse WA + ITU(2005) k–α → *allow true zeros*
df_rate = pm.rainlink_strict_R(dfA, R_min=0.0)

# 4) gate by strict wet mask & apply a drizzle cut (default 0.20 mm/h)
df_rate_gated = pm.apply_wet_gate_and_drizzle(df_rate, dfA, drizzle=0.20)
print(df_rate_gated[["link_id","time","R_mm_per_h"]].head())

# --- 0) Ensure we use the SAME key on both tables ---------------------------
# ts_15 and df_rate came from pipeline_modes (they both have `link_id`)

# A. metadata with the SAME link_id key used in df_rate/df_s5
meta_xy_grid = (
    ts_15.drop_duplicates("link_id")
        [["link_id","XStart","YStart","XEnd","YEnd"]]
        .rename(columns={"link_id":"ID"})
        .copy()
)
for c in ["XStart","YStart","XEnd","YEnd"]:
    meta_xy_grid[c] = pd.to_numeric(meta_xy_grid[c], errors="coerce")

# B. df_s5 with the SAME key and a proper tz-naive DatetimeIndex
tt = pd.to_datetime(df_rate["time"], utc=True)
df_s5 = (df_rate[["link_id","R_mm_per_h"]]
         .rename(columns={"link_id":"ID"})
         .set_index(tt)
         .drop(columns=[]))  # keep shape explicit

df_s5.index = df_s5.index.tz_convert("UTC").tz_localize(None)

# C. sanity: every ID in df_s5 must exist in meta_xy_grid
missing = set(df_s5["ID"].unique()) - set(meta_xy_grid["ID"].unique())
print("IDs missing from meta:", len(missing))
if missing:
    print("Example missing:", list(sorted(missing))[:5])
    raise RuntimeError("Fix meta/id mismatch before gridding.")

# --- 1) Run gridding (OK should fire now) -----------------------------------
import importlib, step6_grid_ok_pcm as s6pcm
importlib.reload(s6pcm)

# 1) run gridding (kill drizzle + enable OK)
Res, diag = s6pcm.grid_rain_15min(
    df_s5=df_s5[["ID","R_mm_per_h"]],
    df_meta_for_xy=meta_xy_grid,          # from your ts_15
    grid_res_deg=0.03, domain_pad_deg=0.2,
    drizzle_to_zero=0.10,                 # kills the faint 0–0.2 mm/h carpet
    use_ok=True, min_pts_ok=12, nlags=10, ok_range_km=35.0,
    idw_power=1.7, idw_nnear=15, idw_maxdist_km=35.0,
    max_dist_km_mask=40.0, smooth_kernel_px=2,
    n_jobs=20,
)
print(diag["counts"])
ok_times = pd.to_datetime(diag["ok_times"])
print("OK slices:", len(ok_times), "e.g.", ok_times[:5])

# --- 2) Plot an OK slice -----------------------------------------------------
ok_times = [pd.Timestamp(t) for t in diag.get("ok_times", [])]
t_plot = ok_times[0] if ok_times else pd.Timestamp(diag["times_used"][0])

# compute max rain for every OK time and pick the peak
mx = []
for t in ok_times:
    a = Res.sel(time=np.datetime64(t), method="nearest").values
    mx.append(np.nanmax(a))
t_peak = ok_times[int(np.nanargmax(mx))]

t_plot = t_peak

plot_slice_cartopy_with_links(
    Res, meta_xy_grid.rename(columns={"ID":"link_id"}),  # function expects link columns as in your meta
    t_plot,
    vmin=0.0, vmax=15.0, nbins=16,
    extent=(-4, 1.5, 4.5, 11.5), cmap_name="turbo"
)
#%%

import numpy as np, pandas as pd, xarray as xr
from sklearn.neighbors import BallTree

def get_da(Res, name="R_mm_per_h"):
    """Accepts DataArray, Dataset, or dict and returns the rain DataArray."""
    if isinstance(Res, xr.DataArray):
        return Res
    if isinstance(Res, xr.Dataset):
        if name in Res: return Res[name]
        raise KeyError(f"{name} not in Dataset vars: {list(Res.data_vars)}")
    if isinstance(Res, dict):
        if name in Res: return Res[name]
        raise KeyError(f"{name} not in dict keys: {list(Res.keys())}")
    raise TypeError(f"Unsupported Res type: {type(Res)}")

def slice_time(da, t=None):
    """
    Return a 2-D (lat,lon) DataArray and a label timestamp.
    - If 'time' in dims: select nearest t (if t is None -> use first time).
    - If no 'time' dim: just return da and label from attr if present.
    """
    if "time" in da.dims:
        if t is None:
            t_sel = pd.to_datetime(da["time"].values[0]).to_pydatetime()
        else:
            t_sel = pd.Timestamp(t)
            if t_sel.tzinfo is not None:
                t_sel = t_sel.tz_convert("UTC").tz_localize(None)
        out = da.sel(time=np.datetime64(t_sel), method="nearest")
        return out, pd.to_datetime(out["time"].item())
    # 2-D already
    label = pd.to_datetime(da.attrs.get("time", "NaT"))
    return da, label

def apply_support_mask(Res, df_s5, meta_xy, t=None, *, k=3, km=35.0, eps=0.1, name="R_mm_per_h"):
    """
    Keep cells with ≥k links having R>eps within km (great-circle).
    df_s5: time-indexed with ['ID','R_mm_per_h'].
    meta_xy: ['ID','XStart','YStart','XEnd','YEnd'] numeric.
    """
    da = get_da(Res, name)
    sl, t_used = slice_time(da, t)

    # points at that time
    pts = (df_s5.loc[pd.Timestamp(t_used)]
           .merge(meta_xy, on="ID", how="inner")
           .assign(lon_mid=lambda d: (pd.to_numeric(d.XStart)+pd.to_numeric(d.XEnd))/2,
                   lat_mid=lambda d: (pd.to_numeric(d.YStart)+pd.to_numeric(d.YEnd))/2))
    wet = (pts["R_mm_per_h"].to_numpy(float) > eps)
    if wet.sum() < max(3, k):
        # nothing to mask—return slice unchanged
        sl2 = sl.copy()
        sl2.attrs["time"] = str(t_used)
        return sl2

    tree = BallTree(np.deg2rad(np.c_[pts["lat_mid"], pts["lon_mid"]]), metric="haversine")
    gx, gy = np.meshgrid(sl["lon"].values, sl["lat"].values)
    q = np.deg2rad(np.c_[gy.ravel(), gx.ravel()])
    neigh = tree.query_radius(q, r=km/6371.0)
    kcnt = np.fromiter((wet[idx].sum() if len(idx) else 0 for idx in neigh), int).reshape(gy.shape)

    Z = sl.values.copy()
    Z[kcnt < k] = np.nan
    out = sl.copy(data=Z)
    out.attrs["time"] = str(t_used)
    return out


# 2-D masked slice (no time dim) -> just pass t for the title
# sl_masked = apply_support_mask(Res, df_s5, meta_xy_grid, t=t_plot,
#                                k=3, km=35.0, eps=0.10)

# plot_slice_cartopy_with_links(
#     sl_masked.to_dataset(name="R_mm_per_h"),
#     meta_xy_grid,                      # has columns ID, X/YStart/End
#     t= t_plot, #"2025-06-11 07:00:00",
#     vmin=0.0, vmax=15.0, nbins=16,     # or bounds=(...)
#     extent=(-4, 1.5, 4.5, 11.5), cmap_name="turbo"
# )


# df_s5: time-indexed with columns ["ID","R_mm_per_h"]  (naive UTC index)
# meta_xy_grid: columns ["ID","XStart","YStart","XEnd","YEnd"]

# R_da, D = s6pcm.grid_rain_15min(
#     df_s5=df_s5, df_meta_for_xy=meta_xy_grid,
#     # OK gating
#     use_ok=True, min_pts_ok=20, nlags=10, ok_range_km=25.0, ok_nugget_frac=0.05, ok_max_train=180,
#     # IDW fallback
#     idw_power=1.7, idw_nnear=15, idw_maxdist_km=35.0,
#     # support & smoothing
#     max_dist_km_mask=40.0, smooth_kernel_px=2,
#     # grid
#     grid_res_deg=0.03, domain_pad_deg=0.2,
#     # parallel
#     n_jobs=-15, parallel_backend_name="processes", kdtree_workers=1,
#     # collocation
#     collocate_bin_km=2.0, use_pathlength_weights=True,
# )
# print(D)

# plot_slice_cartopy_with_links(
#     R_da,
#     meta_xy_grid,
#     t=t_plot,
#     vmin=0.0, vmax=15.0, nbins=16,
#     extent=(-4, 1.5, 4.5, 11.5),
#     cmap_name="turbo",
#     # optional niceties:
#     cbar_side="right", cbar_size="4%", cbar_pad=0.05, tick_every=2
# )



R_da_rl, diag_rl = s6pcm.grid_rain_15min_rainlink_ok(
    df_s5, meta_xy_grid,
    grid_res_deg=0.03, domain_pad_deg=0.20,
    wet_thr=0.8, dry_thr=0.05,
    ok_model="exponential", ok_range_km=25.0, ok_nugget_frac=0.45,
    min_pts_ok=15, support_k=3, support_radius_km=25.0,
    drizzle_to_zero=0.15,
    n_jobs=15, parallel_backend_name="processes",
)
print(diag_rl)

ok_times = [pd.Timestamp(t) for t in diag_rl.get("ok_times", [])]
t_plot = ok_times[0] if ok_times else pd.Timestamp(diag_rl["times_used"][0])

# compute max rain for every OK time and pick the peak
mx = []
for t in R_da_rl["time"].values:
    a = R_da_rl.sel(time=np.datetime64(t), method="nearest").values
    mx.append(np.nanmax(a))
t_peak = ok_times[int(np.nanargmax(mx))]

t_plot = pd.Timestamp(R_da_rl["time"][int(np.nanargmax(mx))].values)#t_peak, '2025-06-19 16:15:00'

plot_slice_cartopy_with_links(
    R_da_rl,
    meta_xy_grid,
    t='2025-06-19 16:15:00', #t_plot,
    vmin=1.0, vmax=15.0, nbins=16,
    extent=(-3.25, 1.2, 4.8, 11.15),
    cmap_name="Blues",
    # optional niceties:
    cbar_side="right", cbar_size="4%", cbar_pad=0.05
)
#%%  Step 2 Baseline Estimation
import importlib, step2_baseline_dry48
importlib.reload(step2_baseline_dry48)
from step2_plot_helpers import plot_baseline_overlay

from step2_baseline_dry48 import compute_dry_baseline_48h, Baseline48Config
cfg2 = Baseline48Config(window_hours=48, fallback_hours=72,
                        min_dry_samples=12, q_baseline=0.20,
                        smooth_baseline_samples=0)

df_step2, s2_sum = compute_dry_baseline_48h(df_nla, cfg2)
print(s2_sum.head())

# Visual check
plot_baseline_overlay(df_step2, df_sum.loc[0, "ID"], t0="2025-06-12", t1="2025-06-14")

#%% Step 3  and Step 4 Compute path attenuation and apply wet antenna

from step3_attenuation import compute_path_attenuation, AttenuationConfig
from step4_wet_antenna import apply_wet_antenna_correction, WetAntennaConfig

# Step 3
cfg3 = AttenuationConfig(min_len_km=0.5, max_gamma_db_per_km=None)
df_s3, s3_sum = compute_path_attenuation(df_step2, cfg3)
print(s3_sum.head(10))

# Step 4 (choose either a fixed Aw or the frequency rule)
cfg4 = WetAntennaConfig(fixed_aw_per_end_db=None,  # or e.g. 0.5
                        n_wet_ends=2,
                        apply_only_when_wet=True)
df_s4, s4_sum = apply_wet_antenna_correction(df_s3, cfg4)
print(s4_sum.head(10))
# 


#%% Step 5 Rainfall Retrieval
import pandas as pd
from step5_rainrate import make_itu_provider_from_lut, gamma_to_rainrate, RainrateConfig

# your 'lut' DataFrame already has Frequency_GHz, kH, αH, kV, αV
itu_provider = make_itu_provider_from_lut(lut)

cfg5 = RainrateConfig(
    use_corrected_gamma=True,
    zero_when_dry=True,
    k_alpha_provider=itu_provider,
    max_R_mm_per_h=400.0,
)

df_s5, s5_summary = gamma_to_rainrate(df_s4, cfg5)
print(s5_summary.head())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def _to_utc(ts):
    if ts is None: return None
    t = pd.to_datetime(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

def plot_rainrate(df_s5, link_id, t0=None, t1=None):
    d = df_s5[df_s5["ID"] == link_id].copy()
    if d.empty:
        print("No rows for", link_id); return

    # ensure UTC index
    if isinstance(d.index, pd.DatetimeIndex):
        d.index = d.index.tz_localize("UTC") if d.index.tz is None else d.index.tz_convert("UTC")

    # optional time window (tz-safe)
    t0u, t1u = _to_utc(t0), _to_utc(t1)
    if t0u is not None: d = d[d.index >= t0u]
    if t1u is not None: d = d[d.index <= t1u]
    if d.empty:
        print("No rows in requested window."); return

    # decide which gamma column to plot
    gcol = None
    if "used_gamma" in d.columns:
        used_vals = d["used_gamma"].dropna().unique()
        if len(used_vals) == 1:
            if used_vals[0] == "corr" and "gamma_corr_db_per_km" in d.columns:
                gcol = "gamma_corr_db_per_km"
            elif used_vals[0] == "raw" and "gamma_raw_db_per_km" in d.columns:
                gcol = "gamma_raw_db_per_km"
    if gcol is None:
        gcol = "gamma_corr_db_per_km" if "gamma_corr_db_per_km" in d.columns else "gamma_raw_db_per_km"

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # γ used
    ax[0].plot(d.index, d[gcol], label=gcol)
    ax[0].set_ylabel("γ [dB/km]")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

    # Rain rate
    ax[1].plot(d.index, d["R_mm_per_h"], label="R [mm/h]")
    ax[1].set_ylabel("R [mm/h]")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.suptitle(f"γ → R — {link_id}")
    plt.tight_layout()
    plt.show()

plot_rainrate(df_s5, df_sum.loc[0, "ID"], t0="2025-06-12", t1="2025-06-14")
