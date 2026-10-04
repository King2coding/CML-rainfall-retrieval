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
# df_raw = pd.read_csv(r'/home/kkumah/Projects/cml-stuff/data-cml/outs/Multi-Link-Multi-Timestamp_coupled_linkdata_kkk_20250929.csv')  # or CSV
df_raw = pd.read_csv(r'/home/kkumah/Projects/cml-stuff/data-cml/outs/Multi-Link-Multi-Timestamp_coupled_linkdata_kkk_20251006.csv')  # or CSV
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
# paths_r0 = save_r0_outputs(df_clean, df_sum, cfg, out_dir="outputs")
# print("R0 saved:", paths_r0)

#%%
# your module with robust time parsing
import importlib
import pipeline_modes as pm
importlib.reload(pm)
# from plot_helpers import plot_helpers as ph
# 1) strict 15-min series with Rainlink-style RSL
ts_15 = pm.build_15min_timeseries(df_clean)

# 2) strict past-only baseline and observed attenuation
dfA = pm.rainlink_strict_Aobs(ts_15, wet_thr_db=3.0)

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

#%%

import numpy as np, pandas as pd, xarray as xr
from sklearn.neighbors import BallTree
from plot_helpers import plot_slice_cartopy_with_links
import importlib, step6_grid_ok_pcm as s6pcm
importlib.reload(s6pcm)

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


R_da_rl, diag_rl = s6pcm.grid_rain_15min_rainlink_ok(
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
    support_radius_km=25.0,
    drizzle_to_zero=0.5,
    n_jobs=20, 
    parallel_backend_name="processes",
)
print(diag_rl)

#%%
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
    wet_thr=1.0,
    dry_thr=0.0,
    ok_model="exponential",
    ok_range_km=15.0,
    ok_nugget_frac=0.5,
    min_pts_ok=50,
    support_k=6,
    support_radius_km=25.0,
    drizzle_to_zero=0.5,     # you can change from default 0.10 if you like
    n_jobs=20,                 # or >1 if you want parallel
    parallel_backend_name="processes",
    outside_support_fill=np.nan,
    insufficient_training_fill=np.nan,
)

print(d1)
#-----------------------------------------

plot_slice_cartopy_with_links(
    R1,
    meta_xy_grid,
    t='2025-06-19 15:00:00', #t_plot,
    vmin=0.0, vmax=15.0, nbins=16,
    extent=(-3.25, 1.2, 4.8, 11.15),
    cmap_name="Blues",
    # optional niceties:
    cbar_side="right", cbar_size="4%", cbar_pad=0.05
)

#%% Save slices
from pipeline_modes import save_each_time_to_netcdf

out_paths = save_each_time_to_netcdf(
    R_da_rl,
    out_dir="/home/kkumah/Projects/cml-stuff/out_cml_rain_dir",
    base_name="ghana_cml_R",
    engine="netcdf4",
    complevel=5,
    dtype="float32",
    fill_value=-9999.0,     # or np.nan if you prefer
    chunks_lat=256,
    chunks_lon=256,
    keep_time_dim=True
)
print(f"Wrote {len(out_paths)} files. First:\n", out_paths[:3])