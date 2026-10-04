# CML_to_Rainfall_Retrieval_Pipeline_20251116.py
# R0 → S1 (Δ-based NLA) → S2 (48h baseline) → S1b (excess NLA)
# → S2b (integrate masks) → WA v2 → k–α → df_s5 → OK+IDW gridding

# %%
import os
import gc
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.dates as mdates

from My_program_utils import *

from r0_clean_minmax_auto import clean_minmax_auto, R0AutoConfig
from r0_qc_helpers import masking_report, plot_flags, plot_minmax_basic
from persist_utils import save_r0_outputs

# ---------------------------------------------------------------------------
# 0) Load raw linked data (must contain ID, DateTime, Pmin, Pmax, X/YStart/End)
# ---------------------------------------------------------------------------
df_raw = pd.read_csv(
    r"/home/kkumah/Projects/cml-stuff/data-cml/outs/Multi-Link-Multi-Timestamp_coupled_linkdata_kkk_20250929.csv"
)
path_to_put_output = r"/home/kkumah/Projects/cml-stuff/data-cml/outs"

# ---------------------------------------------------------------------------
# 1) R0 cleaner (unified semantics + 15-min grid)
# ---------------------------------------------------------------------------
cfg_r0 = R0AutoConfig(
    semantics="auto",      # auto-detect min/max semantics per link
    regularize_grid=True,  # enforce exact 15-min grid
)

df_clean, df_sum = clean_minmax_auto(df_raw, cfg_r0)
print("R0 summary:")
print(df_sum.head(10))

# QA & sanity checks
rep = masking_report(df_clean)
print("Masking report:")
print(rep.head(15))

# One link quick-look
link0 = df_sum.loc[0, "ID"]
one_link = df_clean[df_clean["ID"] == link0]
plot_minmax_basic(one_link)
plot_flags(df_clean, link0)

# Save R0 outputs
paths_r0 = save_r0_outputs(df_clean, df_sum, cfg_r0, out_dir="outputs")
print("R0 saved:", paths_r0)

# %%
# =========================
# Helpers for S1–S5 pipeline
# =========================

def _utcify_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    return out


def _merge_on_id_time(left: pd.DataFrame, right: pd.DataFrame, cols, how="left"):
    """
    Robust merge on (ID, time). Works even if 'right' has duplicate (ID,time).
    Keeps the *last* occurrence per (ID,time) on the right.
    """
    L = _utcify_index(left).reset_index().rename(columns={"index": "time_utc"})
    R = _utcify_index(right).reset_index().rename(columns={"index": "time_utc"})
    R = R[["ID", "time_utc", *cols]].drop_duplicates(["ID", "time_utc"], keep="last")
    out = (
        L.merge(R, on=["ID", "time_utc"], how=how)
         .set_index("time_utc")
         .sort_index()
    )
    return out


# ONE knob for NLA aggressiveness
WETDRY_MODE = "strict"   # "strict" (Rainlink-like) or "relaxed"

# %%
# =====================
# Step 1: Δ-based NLA
# =====================
import importlib
import step1_wetdry_nla_fast2 as s1
import step1b_wetdry_excess_fast as s1b
importlib.reload(s1)
importlib.reload(s1b)

# IMPORTANT: feed R0 output (has 'ID'), not ts_15
df_nla, s1_sum = s1.wetdry_classify_with_mode(
    _utcify_index(df_clean),
    mode=WETDRY_MODE,
    override={"n_jobs": 15},
)
df_nla = _utcify_index(df_nla)
print("Step 1 (Δ-based NLA) done. Columns:", df_nla.columns.tolist())

# %%
# ============================
# Step 2: 48h dry-only baseline
# ============================
import importlib, step2_baseline_dry48
importlib.reload(step2_baseline_dry48)
from step2_baseline_dry48 import compute_dry_baseline_48h, Baseline48Config

# Use only arguments that Baseline48Config actually supports
cfg2 = Baseline48Config(
    window_hours=48,
    fallback_hours=72,
    min_dry_samples=12,          # or 12, if you prefer stricter baseline
    smooth_baseline_samples=0,
)

df_step2, s2_sum = compute_dry_baseline_48h(df_nla, cfg2)
df_step2 = _utcify_index(df_step2)
print(s2_sum.head())
# %%
# ===============================
# Step 1b: Excess-based NLA (A_ex)
# ===============================
df_ex, s1b_sum = s1b.wetdry_from_excess_with_mode(
    df_step2,
    mode=WETDRY_MODE,
    override={"n_jobs": 15},
)
df_ex = _utcify_index(df_ex)
print("Step 1b (excess NLA) done. Columns:", df_ex.columns.tolist())

# %%
# ===============================
# Step 2b: Integrate masks → is_wet_final
# ===============================
import step2b_integrate_masks as s2b
importlib.reload(s2b)
from step2b_integrate_masks import integrate_wetdry_and_excess

df_s12 = integrate_wetdry_and_excess(
    df_step2,
    df_nla,
    df_ex,
    onset_allow_if_excess_pos=(WETDRY_MODE != "strict"),  # strict: OFF, relaxed: ON
)
df_s12 = _utcify_index(df_s12)
assert "is_wet_final" in df_s12.columns

# Keep original Rainlink-style mask for diagnostics
df_s12["is_wet_final_orig"] = df_s12["is_wet_final"].astype("uint8")

print("Step 2b (integrated wet mask) done. Columns:", df_s12.columns.tolist())

# ==============================
# Attach A_ex_pool_per_km + neighbour counts from S1b
# ==============================
cols_from_ex = []
if "A_ex_pool_per_km" in df_ex.columns:
    cols_from_ex.append("A_ex_pool_per_km")
if "nb_count_ex" in df_ex.columns:
    cols_from_ex.append("nb_count_ex")

if cols_from_ex:
    df_s12 = _merge_on_id_time(df_s12, df_ex, cols_from_ex)

# At this point df_s12 should contain:
# ['ID', ..., 'is_wet_final', 'is_wet_final_orig',
#  'A_ex_pool_per_km', 'nb_count_ex', ...]

# -----------------------------
# (Optional) temporal rescue (1c)
# -----------------------------
import importlib, step1c_temporal_rescue as s1c
importlib.reload(s1c)
from step1c_temporal_rescue import TemporalRescueConfig, apply_temporal_rescue

# Work on a copy with time as a normal column
df_tr_in = df_s12.reset_index()     # first column = time col (e.g. 'time_utc')
time_col = df_tr_in.columns[0]

# Robust choice of gamma-like column for temporal rescue
gamma_col = None
for cand in ["A_ex_pool_per_km", "A_excess_db_per_km"]:
    if cand in df_tr_in.columns:
        gamma_col = cand
        break

if gamma_col is None:
    # Nothing suitable to drive temporal rescue → skip cleanly
    print("Temporal rescue: SKIPPED (no A_ex_* column found). "
          "Available columns:", list(df_tr_in.columns))
    tr_sum = {"skipped": True}
else:
    # neighbour count column (optional)
    nb_col = "nb_count_ex" if "nb_count_ex" in df_tr_in.columns else None

    tr_cfg = TemporalRescueConfig(
        gamma_col=gamma_col,
        nb_col=nb_col,
        wet_col="is_wet_final",
        max_nb_for_rescue=2,
        gamma_thr_db_per_km=0.01,
        min_run_bins=2,
        require_network_anchor=True,
        min_network_wet_frac=0.05,
    )

    df_tr_out, tr_sum = apply_temporal_rescue(df_tr_in, tr_cfg)
    print("Temporal rescue summary:", tr_sum)

    # Restore DatetimeIndex from the time column
    df_tr_out = df_tr_out.set_index(time_col).sort_index()

    # ensure UTC tz-aware index (same convention as elsewhere)
    if df_tr_out.index.tz is None:
        df_tr_out.index = df_tr_out.index.tz_localize("UTC")
    else:
        df_tr_out.index = df_tr_out.index.tz_convert("UTC")

    # Drop any stray unnamed column
    if None in df_tr_out.columns:
        df_tr_out = df_tr_out.drop(columns=[None])

    df_s12 = df_tr_out

# -----------------------------
# NEW: Step 1d — single-link fallback (Option B: 0.05 dB/km)
# -----------------------------
import step1d_singlelink_fallback as s1d
importlib.reload(s1d)
from step1d_singlelink_fallback import SingleLinkConfig, apply_singlelink_fallback

# Decide which gamma-like column is present
if "A_ex_pool_per_km" in df_s12.columns:
    gamma_col = "A_ex_pool_per_km"
elif "A_excess_db_per_km" in df_s12.columns:
    gamma_col = "A_excess_db_per_km"
else:
    raise ValueError(
        "Single-link fallback: need one of 'A_ex_pool_per_km' or 'A_excess_db_per_km'. "
        f"Available columns: {list(df_s12.columns)}"
    )

nb_col = "nb_count_ex" if "nb_count_ex" in df_s12.columns else None

sl_cfg = SingleLinkConfig(
    gamma_col=gamma_col,
    nb_col=nb_col,
    wet_col="is_wet_final",
    thr_db_per_km=0.02,     # <<< Option B
    min_run_bins=2,
    max_nb_for_fallback=2,
)

df_s12, sl_sum = apply_singlelink_fallback(df_s12, sl_cfg)
print("Single-link fallback summary:", sl_sum)

# -----------------------------
# Attach metadata if still missing (for WA and k–α)
# -----------------------------
need_meta = ["PathLength", "Frequency", "Polarization"]
if not set(need_meta).issubset(df_s12.columns):
    df_s12 = _merge_on_id_time(df_s12, df_step2, need_meta)

# Pick WA source automatically
wa_src = "A_ex_pool_per_km" if "A_ex_pool_per_km" in df_s12.columns else (
         "A_excess_db_per_km" if "A_excess_db_per_km" in df_s12.columns else None
)
if wa_src is None:
    raise ValueError(
        "No per-km excess column found. Need 'A_ex_pool_per_km' (preferred) "
        "or 'A_excess_db_per_km'."
    )

print("Wet-antenna γ source:", wa_src)
#%%
# ========= Sanity check: temporal rescue vs original mask =========
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 0) Basic safety checks
for col in ["is_wet_final_orig", "is_wet_final"]:
    if col not in df_s12.columns:
        raise RuntimeError(
            f"Column '{col}' not found in df_s12. "
            "Make sure you set df_s12['is_wet_final_orig'] "
            "before calling apply_temporal_rescue."
        )

if "A_ex_pool_per_km" not in df_s12.columns:
    print("WARNING: 'A_ex_pool_per_km' missing in df_s12; "
          "temporal rescue sanity plots will only use masks, not gamma-like values.")

# 1) Focus day: 2025-06-19 (tz-aware UTC to match df_s12.index)
day0 = pd.Timestamp("2025-06-19 00:00:00", tz="UTC")
day1 = pd.Timestamp("2025-06-19 23:59:59", tz="UTC")

idx = df_s12.index
if not isinstance(idx, pd.DatetimeIndex):
    raise TypeError("df_s12.index must be a DatetimeIndex for this sanity check.")

# This will now work because both sides are tz-aware UTC
mask_day = (idx >= day0) & (idx <= day1)
day = df_s12.loc[mask_day].copy()

print("Rows on 2025-06-19:", len(day))

# 2) Count how many wet flags before/after rescue
orig_wet = day["is_wet_final_orig"].sum()
new_wet  = day["is_wet_final"].sum()
print(f"2025-06-19 — wet flags orig: {orig_wet}, after rescue: {new_wet}, diff: {new_wet - orig_wet}")

# 3) Optional: histogram comparison of wet/dry counts
plt.figure(figsize=(6,4))
plt.bar(["orig_wet", "rescued_wet"], [orig_wet, new_wet])
plt.title("Number of wet flags on 2025-06-19")
plt.ylabel("count of (ID, time) flagged wet")
plt.grid(axis="y", alpha=0.3)
plt.show()

# 4) Pick a link with changes and plot γ + masks around 16:15 UTC
changed = day.groupby("ID").apply(
    lambda d: (d["is_wet_final_orig"] != d["is_wet_final"]).any()
)
changed_ids = changed[changed].index.tolist()
print("Links with any changed wet/dry on that day:", changed_ids[:10])

if changed_ids:
    link_id = changed_ids[0]
    print("Plotting link:", link_id)

    dlink = day[day["ID"] == link_id].copy()

    # focus window around 16:15 UTC
    t0 = pd.Timestamp("2025-06-19 12:00:00", tz="UTC")
    t1 = pd.Timestamp("2025-06-19 20:00:00", tz="UTC")
    dlink = dlink[(dlink.index >= t0) & (dlink.index <= t1)]

    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

    # γ-like signal if available
    if "A_ex_pool_per_km" in dlink.columns:
        ax[0].plot(dlink.index, dlink["A_ex_pool_per_km"], label="A_ex_pool_per_km [dB/km]")
        ax[0].set_ylabel("A_ex_pool_per_km")
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)
    else:
        ax[0].text(0.5, 0.5, "A_ex_pool_per_km not available",
                   transform=ax[0].transAxes, ha="center", va="center")
        ax[0].set_axis_off()

    # Wet masks
    ax[1].step(dlink.index, dlink["is_wet_final_orig"], where="mid", label="orig wet", alpha=0.7)
    ax[1].step(dlink.index, dlink["is_wet_final"], where="mid", label="rescued wet", alpha=0.7)
    ax[1].set_ylabel("wet mask")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    # Rain rate (if already computed)
    if "R_mm_per_h" in dlink.columns:
        ax[2].plot(dlink.index, dlink["R_mm_per_h"], label="R [mm/h]")
        ax[2].set_ylabel("R [mm/h]")
        ax[2].legend()
        ax[2].grid(True, alpha=0.3)
    else:
        ax[2].text(0.5, 0.5, "R_mm_per_h not in df_s12 (that is fine at S2b stage)",
                   transform=ax[2].transAxes, ha="center", va="center")
        ax[2].set_axis_off()

    ax[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M", tz=day.index.tz))
    plt.suptitle(f"Temporal rescue sanity — link {link_id} on 2025-06-19")
    plt.tight_layout()
    plt.show()
else: 
    print("No links with changed wet/dry flags on 2025-06-19.")

# %%
# ================================
# Step 2c: Wet-antenna V2 (decay)
# ================================
import step2b_wet_antenna as wa
importlib.reload(wa)
from step2b_wet_antenna import WAConfigV2, apply_wet_antenna_decay

# 1) Make sure meta is present
need_meta = ["PathLength", "Frequency", "Polarization"]
if not set(need_meta).issubset(df_s12.columns):
    df_s12 = _merge_on_id_time(df_s12, df_step2, need_meta)
    print("Merged meta into df_s12. Now has:", [c for c in need_meta if c in df_s12.columns])

# 2) pick WA source automatically from df_s12
wa_src = "A_ex_pool_per_km" if "A_ex_pool_per_km" in df_s12.columns else (
         "A_excess_db_per_km" if "A_excess_db_per_km" in df_s12.columns else None
)
if wa_src is None:
    raise ValueError(
        "No per-km excess column found. Need 'A_ex_pool_per_km' (preferred) "
        "or 'A_excess_db_per_km'. "
        f"Available columns: {list(df_s12.columns)}"
    )
print("Wet-antenna γ source:", wa_src)

cfg_wa = WAConfigV2(
    gamma_source_col=wa_src,        # from above
    wet_mask_col="is_wet_final",
    wa_db_per_terminal=0.08,
    assume_wet_terminals=2,
    dt_minutes=15,
    tau_on_min=30,
    tau_off_min=60,
    out_raw_col="gamma_raw_db_per_km",
    out_corr_col="gamma_corr_db_per_km",
    floor_zero=True,
)

df_attn = apply_wet_antenna_decay(df_s12, cfg_wa)
df_attn = _utcify_index(df_attn)
print("Step 2c (WA V2) done. Columns:", df_attn.columns.tolist())
# # %%
# =============================
# Step 3: k–α (γ → R, gated)
# =============================
import step3_kalpha as s3
importlib.reload(s3)
from step3_kalpha import KAlphaConfigV2, gamma_to_r_gated

# 'lut' must exist in your environment (ITU coeffs table)
cfg_k = KAlphaConfigV2(
    lut=lut,                        # DataFrame with k, α by freq/pol
    pol_col="Polarization",
    freq_col="Frequency",
    gamma_col="gamma_corr_db_per_km",
    gamma_gate_db_per_km=0.01,
    use_wet_mask_col="is_wet_final",
    r_cap_mmph_by_band={6: 60, 8: 80, 19: 120},
)

df_s5, s5_sum = gamma_to_r_gated(df_attn, cfg_k)
df_s5 = _utcify_index(df_s5)
# make index tz-naive UTC for gridding
df_s5.index = df_s5.index.tz_convert("UTC").tz_localize(None)
df_s5.index.name = "time"

print("Step 3 (k–α) done. df_s5 columns:", df_s5.columns.tolist())
print("df_s5 time range:", df_s5.index.min(), "→", df_s5.index.max())

# Quick per-link QA plot (optional)
def _to_utc(ts):
    if ts is None:
        return None
    t = pd.to_datetime(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def plot_rainrate(df_s5, link_id, t0=None, t1=None):
    import matplotlib.pyplot as plt

    d = df_s5[df_s5["ID"] == link_id].copy()
    if d.empty:
        print("No rows for", link_id)
        return

    # Ensure UTC index
    if isinstance(d.index, pd.DatetimeIndex):
        d.index = d.index.tz_localize("UTC") if d.index.tz is None else d.index.tz_convert("UTC")

    t0u, t1u = _to_utc(t0), _to_utc(t1)
    if t0u is not None:
        d = d[d.index >= t0u]
    if t1u is not None:
        d = d[d.index <= t1u]
    if d.empty:
        print("No rows in requested window.")
        return

    gcol = "gamma_corr_db_per_km" if "gamma_corr_db_per_km" in d.columns else "gamma_raw_db_per_km"

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax[0].plot(d.index, d[gcol], label=gcol)
    ax[0].set_ylabel("γ [dB/km]")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend()

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


# Example QA for first link
plot_rainrate(df_s5, df_sum.loc[0, "ID"], t0="2025-06-12", t1="2025-06-14")
# %%
# =========================
# Gridding: OK → IDW combo
# =========================
from sklearn.neighbors import BallTree
from plot_helpers import plot_slice_cartopy_with_links
import step6_grid_ok_pcm as s6pcm
importlib.reload(s6pcm)

# Meta midpoints (one row per ID) from R0 output
meta_xy_grid = (
    df_clean.reset_index()[["ID", "XStart", "YStart", "XEnd", "YEnd"]]
    .drop_duplicates("ID")
)

for c in ["XStart", "YStart", "XEnd", "YEnd"]:
    meta_xy_grid[c] = pd.to_numeric(meta_xy_grid[c], errors="coerce")

# Sanity: all IDs in df_s5 must exist in meta_xy_grid
missing = set(df_s5["ID"].unique()) - set(meta_xy_grid["ID"].unique())
print("IDs missing from meta:", len(missing))
if missing:
    print("Example missing:", list(sorted(missing))[:5])
    raise RuntimeError("Fix meta/id mismatch before gridding.")

# Main gridding call using consensus df_s5
R_da, diag_rl = s6pcm.grid_rain_15min_rainlink_ok(
    df_s5=df_s5[["ID", "R_mm_per_h"]],
    df_meta_for_xy=meta_xy_grid,
    grid_res_deg=0.03,
    domain_pad_deg=0.20,
    wet_thr=0.8,
    dry_thr=0.05,
    ok_model="exponential",
    ok_range_km=25.0,
    ok_nugget_frac=0.45,
    min_pts_ok=15,
    support_k=4,              # stricter spatial support
    support_radius_km=25.0,
    drizzle_to_zero=0.10,     # kill sub-0.3 mm/h drizzle
    n_jobs=18,
    parallel_backend_name="processes",
)
print("Gridding diagnostics:")
print(diag_rl)

# Choose a time to plot (peak rain)
mx = []
for t in R_da["time"].values:
    a = R_da.sel(time=np.datetime64(t), method="nearest").values
    mx.append(np.nanmax(a))

imax = int(np.nanargmax(mx))
t_peak = pd.Timestamp(R_da["time"][imax].values)
print("Peak rain at:", t_peak, "max =", mx[imax])

# Plot that slice
plot_slice_cartopy_with_links(
    R_da,
    meta_xy_grid,
    t=t_peak,
    vmin=1.0,
    vmax=15.0,
    nbins=16,
    extent=(-3.25, 1.2, 4.8, 11.15),
    cmap_name="Blues",
    cbar_side="right",
    cbar_size="4%",
    cbar_pad=0.05,
)

t_peak = pd.Timestamp(np.datetime64('2025-06-19T16:15:00.000000000'))

plot_slice_cartopy_with_links(
    R_da,
    meta_xy_grid,
    t=t_peak,
    vmin=0.0,
    vmax=15.0,
    nbins=16,
    extent=(-3.25, 1.2, 4.8, 11.15),
    cmap_name="Blues",
    cbar_side="right",
    cbar_size="4%",
    cbar_pad=0.05,
)


#=============================================================================
import step6_grid_ok_pcm as s6pcm
import importlib
importlib.reload(s6pcm)

t = pd.Timestamp("2025-06-19 16:15:00")

meta_xy = (
    df_clean.reset_index()[["ID","XStart","YStart","XEnd","YEnd"]]
    .drop_duplicates("ID")
)

R1, d1 = s6pcm.grid_rain_at_time_rainlink(
    df_s5=df_s5[["ID", "R_mm_per_h"]],
    df_meta_for_xy=meta_xy,
    t=t,                      # naive UTC timestamp present in df_s5.index

    # RainLINK-OK parameters:
    grid_res_deg=0.03,
    domain_pad_deg=0.20,
    wet_thr=0.15,
    dry_thr=0.15,
    ok_model="exponential",
    ok_range_km=25.0,
    ok_nugget_frac=0.4,
    min_pts_ok=4,
    support_k=2,
    support_radius_km=25.0,
    drizzle_to_zero=0.05,     # you can change from default 0.10 if you like
    n_jobs=5,                 # or >1 if you want parallel
    parallel_backend_name="processes",
    outside_support_fill=np.nan,
    insufficient_training_fill=np.nan,
)

print(d1)

plot_slice_cartopy_with_links(
    R1,
    meta_xy,
    t=t,
    vmin=0.5, vmax=15.0, nbins=16,
    extent=(-3.25, 1.2, 4.8, 11.15),
    cmap_name="Blues",
    cbar_side="right", cbar_size="4%", cbar_pad=0.05,
)
# %%
# ===============================
# Save each time slice to NetCDF
# ===============================
from pipeline_modes import save_each_time_to_netcdf

out_paths = save_each_time_to_netcdf(
    R_da,
    out_dir="/home/kkumah/Projects/cml-stuff/out_cml_rain_dir_2025-11-17",
    base_name="ghana_cml_R_consensus",
    engine="netcdf4",
    complevel=5,
    dtype="float32",
    fill_value=-9999.0,
    chunks_lat=256,
    chunks_lon=256,
    keep_time_dim=True,
)
print(f"Wrote {len(out_paths)} files. First:\n", out_paths[:3])