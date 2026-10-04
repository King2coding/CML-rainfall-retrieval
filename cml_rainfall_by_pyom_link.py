#%%
import os
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt

from pycomlink.processing.wet_dry import stft as wd_stft
from pycomlink.processing.baseline import baseline_linear
from pycomlink.processing.k_R_relation import calc_R_from_A
from pycomlink.processing.wet_antenna import waa_leijnse_2008_from_A_obs
from pycomlink.spatial.interpolator import IdwKdtreeInterpolator

#%%
# --- CONFIG: point to your files ---
path_to_put_output = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'

META_CSV  = os.path.join(path_to_put_output, 'matched_metadata_kkk_20250527.csv')         # your meta_req
# 'Linkdata_AT_20250824_processed_on_20250527.dat'
LINK_DATA = os.path.join(path_to_put_output, 'Multi-Link-Multi-Timestamp_coupled_linkdata_kkk_20250929.csv')   # your link_raw

# pipeline mode: "rainlink_compat" or "pycomlink_default"
MODE = "rainlink_compat"

# ghana grid resolution (~1.1 km near equator)
GRID_RES_DEG = 0.01

#%%
# --- meta ---
meta = pd.read_csv(META_CSV)

# --- link_raw ---
raw = pd.read_csv(LINK_DATA, sep=None, engine="python")

# === SECTION 1: pycomlink-ready metadata from your `meta` ===
# work on a copy
m = meta.copy()

# enforce numeric types where appropriate
for c in ["Frequency","XStart","YStart","XEnd","YEnd","PathLength"]:
    m[c] = pd.to_numeric(m[c], errors="coerce")

# build path_id consistent with your convention (Monitored_ID ~~ Far_end_ID)
m["path_id"] = (
    m["Monitored_ID"].astype(str).str.strip()
    + "~~" +
    m["Far_end_ID"].astype(str).str.strip()
)

# polarization → single char H/V
m["pol"] = m["Polarization"].astype(str).str.upper().str[0]

# Frequency in meta is MHz (e.g., 8103.495) → convert to GHz
m["freq_GHz"] = m["Frequency"] / 1000.0

# coordinates and length (keep as provided)
m["tx_lon"] = m["XStart"];  m["tx_lat"] = m["YStart"]
m["rx_lon"] = m["XEnd"];    m["rx_lat"] = m["YEnd"]
m["length_km"] = m["PathLength"]

# stable per-channel identifier (direction-agnostic path + pol + rounded freq)
m["link_id"] = m["path_id"] + "|" + m["pol"] + "|" + m["freq_GHz"].round(3).astype(str)

# final pycomlink-style metadata table
meta_pc = m[[
    "link_id","path_id","pol","freq_GHz",
    "tx_lat","tx_lon","rx_lat","rx_lon","length_km"
]].copy()

# quick sanity prints (purely informational)
print("meta_pc shape:", meta_pc.shape)
print(meta_pc.head(5))
print("freq_GHz range:", (meta_pc["freq_GHz"].min(), meta_pc["freq_GHz"].max()))
print("pol values:", sorted(meta_pc["pol"].unique()))

#%%
# === SECTION 2: build pycomlink-ready time series from your `raw` ===
r = raw.copy()

# A) keys & fields (exactly from your columns)
r["path_id"] = r["ID"].astype(str).str.replace(">>", "~~", regex=False).str.strip()
r["pol"]      = r["Polarization"].astype(str).str.upper().str[0]
r["freq_GHz"] = pd.to_numeric(r["Frequency"], errors="coerce")

# RAINLINK-style RSL (you computed Pmin/Pmax that way)
r["RSL_dBm"] = pd.to_numeric(r["TSL_AVG"], errors="coerce") + (
    pd.to_numeric(r["Pmin"], errors="coerce") + pd.to_numeric(r["Pmax"], errors="coerce")
) / 2.0

# timestamps (UTC) from YYYYMMDDHHMM
r["time"] = pd.to_datetime(r["DateTime"].astype(str), format="%Y%m%d%H%M", utc=True)

# channel id aligned with meta_pc["link_id"]
r["link_id"] = r["path_id"] + "|" + r["pol"] + "|" + r["freq_GHz"].round(3).astype(str)

# B) keep only channels known in meta_pc (no merge, just intersection)
keep = set(meta_pc["link_id"].unique())
r = r[r["link_id"].isin(keep)].copy()

# C) ensure numeric dtypes for fields we’ll aggregate numerically
num_cols = ["RSL_dBm","freq_GHz","PathLength","XStart","YStart","XEnd","YEnd"]
for c in num_cols:
    r[c] = pd.to_numeric(r[c], errors="coerce")

# D) align to strict 15-minute bins (handles any slight timestamp jitter)
r["time15"] = r["time"].dt.floor("15min")

# E) aggregate to one row per (link_id, 15-min)
#    - numeric columns: median
#    - non-numeric columns we need: take first
agg_numeric = {c: "median" for c in num_cols}
agg_other = {"pol": "first"}  # stays constant within link_id
ts_15min = (
    r.groupby(["link_id", "time15"], as_index=False)
     .agg({**agg_numeric, **agg_other})
     .rename(columns={"time15": "time"})
     .sort_values(["link_id","time"])
)

# --- sanity prints ---
print("ts_15min shape:", ts_15min.shape)
print("unique channels:", ts_15min["link_id"].nunique())
print("time span:", ts_15min["time"].min(), "→", ts_15min["time"].max())
print(ts_15min.head(8))
#%%
# === SECTION 3: baseline (RAINLINK-style), wet mask, observed attenuation ===
def rolling_upper_quantile_past_only(series_with_time_index, q=0.9, min_periods=2):
    """
    Past-only rolling upper-quantile baseline; no future leakage.
    Requires a DatetimeIndex (we use 24h window with closed='left').
    """
    base = (series_with_time_index
            .rolling(window="24H", min_periods=min_periods, closed="left")
            .quantile(q))
    # Fill very early gaps so we can proceed
    return base.fillna(method="backfill", limit=4)

out = []
for lid, g in ts_15min.groupby("link_id"):
    g = g.sort_values("time").copy()

    # make a Series with a proper DatetimeIndex
    rsl = pd.Series(g["RSL_dBm"].astype(float).values, index=pd.DatetimeIndex(g["time"]))

    # Pass 1
    base1 = rolling_upper_quantile_past_only(rsl, q=0.9, min_periods=2)
    A1 = np.maximum(0.0, base1 - rsl)
    wet1 = A1 > 0.5  # dB threshold; tune later if needed

    # Pass 2: recompute baseline using "dry-ish" only
    rsl_dry = rsl.mask(wet1)
    base2 = (rsl_dry
             .rolling(window="24H", min_periods=2, closed="left")
             .quantile(0.9))
    base = base2.fillna(base1)

    # Final wet & observed attenuation
    A_obs = np.maximum(0.0, base - rsl)
    wet = A_obs > 0.5

    g["baseline_rsl"] = base.values
    g["A_obs_dB"] = A_obs.values
    g["wet"] = wet.values
    out.append(g)

dfp = pd.concat(out, ignore_index=True)

# --- sanity prints ---
print("dfp shape:", dfp.shape)
print(dfp[["link_id","time","RSL_dBm","baseline_rsl","A_obs_dB","wet"]].head(12))
print("A_obs stats (dB):")
print(dfp["A_obs_dB"].describe())
print("wet fraction:", float(dfp["wet"].mean()))
#%%
# === SECTION 4: wet-antenna correction + attenuation → rain rate (mm/h) ===


from pycomlink.processing.wet_antenna import waa_leijnse_2008_from_A_obs
from pycomlink.processing.k_R_relation import calc_R_from_A

cols_needed = ["link_id","time","A_obs_dB","freq_GHz","pol","PathLength"]
df4 = dfp[cols_needed].copy()

# dtypes
for c in ["A_obs_dB","freq_GHz","PathLength"]:
    df4[c] = pd.to_numeric(df4[c], errors="coerce")
df4["pol"] = df4["pol"].astype(str).str.upper().str[0]

parts = []
for lid, g in df4.groupby("link_id", sort=False):
    g = g.sort_values("time").copy()

    # Scalars per link (constant along time)
    L_km       = float(g["PathLength"].iloc[0])
    f_Hz_link  = float(g["freq_GHz"].iloc[0]) * 1e9      # scalar
    pol_link   = str(g["pol"].iloc[0])                   # scalar 'H' or 'V'

    # Time series arrays
    A_obs = g["A_obs_dB"].values

    # Wet-antenna correction (requires scalars for L_km, f_Hz, pol)
    waa = waa_leijnse_2008_from_A_obs(
        A_obs=A_obs,
        f_Hz=f_Hz_link,
        pol=pol_link,
        L_km=L_km
    )
    g["A_rain_dB"] = np.maximum(0.0, A_obs - waa)

    # Attenuation → rain rate (accepts scalars for f_GHz & pol as well)
    g["R_mm_h"] = calc_R_from_A(
        A=g["A_rain_dB"].values,
        L_km=L_km,
        f_GHz=f_Hz_link / 1e9,
        pol=pol_link,
        a_b_approximation="ITU_2005",
        R_min=0.05
    )

    parts.append(g)

df_rate = pd.concat(parts, ignore_index=True)

# QC prints
print("df_rate shape:", df_rate.shape)
print(df_rate[["link_id","time","A_obs_dB","A_rain_dB","R_mm_h"]].head(12))
print("R stats (mm/h):")
print(df_rate["R_mm_h"].describe())


#%%import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pycomlink.spatial.interpolator import IdwKdtreeInterpolator

# midpoints per link (using same channels as df_rate)
mid = (ts_15min.drop_duplicates(subset=["link_id"])[
    ["link_id","XStart","YStart","XEnd","YEnd"]
].copy())
for c in ["XStart","YStart","XEnd","YEnd"]:
    mid[c] = pd.to_numeric(mid[c], errors="coerce")
mid["lon_mid"] = (mid["XStart"] + mid["XEnd"]) / 2.0
mid["lat_mid"] = (mid["YStart"] + mid["YEnd"]) / 2.0

# pick a mapping time with most wet links (or change to a specific timestamp)
wet_counts = (dfp[dfp["wet"]].groupby("time")["link_id"].nunique().sort_values(ascending=False))
time_map = wet_counts.index[0] if len(wet_counts) else df_rate["time"].min()
print("Mapping time:", time_map)

inst = (df_rate[df_rate["time"] == time_map]
        .merge(mid[["link_id","lon_mid","lat_mid"]], on="link_id", how="inner")
        .dropna(subset=["R_mm_h","lon_mid","lat_mid"]))

print("Instant points:", len(inst))

# grid (start coarse; refine later)
GRID_RES_DEG = 0.03  # ~3 km
pad = 0.25
min_lon, max_lon = inst["lon_mid"].min()-pad, inst["lon_mid"].max()+pad
min_lat, max_lat = inst["lat_mid"].min()-pad, inst["lat_mid"].max()+pad
xv = np.arange(min_lon, max_lon + GRID_RES_DEG, GRID_RES_DEG)
yv = np.arange(min_lat, max_lat + GRID_RES_DEG, GRID_RES_DEG)
xx, yy = np.meshgrid(xv, yv)

# IDW
# IDW with pycomlink (expects x, y, z, xgrid, ygrid)
interp = IdwKdtreeInterpolator(nnear=15, p=1.5, 
                               exclude_nan=True, 
                                max_distance=0.3)
Z = interp(
    inst["lon_mid"].values,           # x: 1D
    inst["lat_mid"].values,           # y: 1D
    inst["R_mm_h"].values,            # z: 1D
    xx,                               # xgrid: 2D meshgrid
    yy                                # ygrid: 2D meshgrid
)
# Z already has the shape of xx/yy; no need to reshape

# plot
plt.figure(figsize=(8,7))
im = plt.pcolormesh(xx, yy, Z, shading="auto")
plt.scatter(inst["lon_mid"], inst["lat_mid"], s=8, c="k", alpha=0.6, label="CML midpoints")
plt.colorbar(im, label="Rain rate (mm/h)")
plt.title(f"CML rain rate — {pd.to_datetime(time_map).strftime('%Y-%m-%d %H:%M UTC')}")
plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.legend()
plt.tight_layout(); plt.show()


#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# accumulation from 15-min rates
rate = df_rate.copy()
rate["mm_per_15min"] = rate["R_mm_h"] * 0.25  # 15/60

hourly = (rate.groupby(["link_id", pd.Grouper(key="time", freq="1H")])["mm_per_15min"]
               .sum()
               .reset_index(name="mm_hour"))

# choose hour with widest coverage
hcounts = hourly.groupby("time")["link_id"].nunique().sort_values(ascending=False)
hour_sel = hcounts.index[0]
print("Accumulation hour:", hour_sel)

hh = (hourly[hourly["time"] == hour_sel]
      .merge(mid[["link_id","lon_mid","lat_mid"]], on="link_id", how="inner")
      .dropna(subset=["mm_hour","lon_mid","lat_mid"]))

print("Hourly points:", len(hh))

Zhr = interp(
    hh["lon_mid"].values,
    hh["lat_mid"].values,
    hh["mm_hour"].values,
    xx,
    yy
)

#- - -- - - - - - - - - - - -- - - - - - - - - - - -- - - - - - - - - - - 
# --- Ordinary Kriging (PyKrige) on lon/lat grid -----------------------------
import numpy as np
from pykrige.ok import OrdinaryKriging
from sklearn.neighbors import NearestNeighbors

# Inputs:
# inst: DataFrame with columns ["lon_mid","lat_mid","R_mm_h"] at one timestamp
# xv, yv: 1D lon/lat axes (same as you already built)
# Returns: Zk (kriged rain rate), Vk (kriging variance), both shaped (len(yv), len(xv))

def estimate_variogram_params(lon, lat, val, frac_nugget=0.2, k_nn=8):
    """Heuristic: sill = var(val), range ~ 1.5 * median k-NN spacing (km), nugget = frac*sill."""
    v = np.asarray(val, float)
    sill = float(np.nanvar(v)) if np.isfinite(v).any() else 1.0e-3

    # robust spatial scale (km) from k-NN distances on lon/lat
    XY = np.c_[lon, lat]
    nn = NearestNeighbors(n_neighbors=min(k_nn, len(XY))).fit(XY)
    dists_deg, _ = nn.kneighbors(XY)   # degrees
    # median distance to k-th neighbor, convert to km (approx 111 km per degree)
    med_deg = np.nanmedian(dists_deg[:, -1]) if dists_deg.size else 0.2
    range_km = max(30.0, 1.5 * med_deg * 111.0)  # keep >= ~30 km to avoid overly short ranges

    nugget = float(frac_nugget * sill)
    return {"sill": max(sill, 1e-6), "range": range_km, "nugget": nugget}

def kriging_grid(inst, xv, yv, model="exponential", frac_nugget=0.2):
    lon = inst["lon_mid"].values.astype(float)
    lat = inst["lat_mid"].values.astype(float)
    val = inst["R_mm_h"].values.astype(float)

    # keep finite points only
    good = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(val)
    lon, lat, val = lon[good], lat[good], val[good]

    # no data guard
    if lon.size < 3:
        Zk = np.full((len(yv), len(xv)), np.nan, float)
        Vk = np.full_like(Zk, np.nan)
        return Zk, Vk

    # first-cut variogram parameters
    vparam = estimate_variogram_params(lon, lat, val, frac_nugget=frac_nugget)

    # OK in geographic coordinates (PyKrige handles lon/lat with great-circle metric)
    OK = OrdinaryKriging(
        lon, lat, val,
        variogram_model=model,
        variogram_parameters=vparam,       # dict with range (km), sill, nugget
        coordinates_type="geographic",     # IMPORTANT for lon/lat
        verbose=False, enable_plotting=False
    )

    gridx, gridy = np.asarray(xv), np.asarray(yv)
    # PyKrige returns (z, ss) with shape (len(gridy), len(gridx))
    Zk, Vk = OK.execute("grid", gridx, gridy)    # kriged value & variance
    Zk = np.asarray(Zk, float)
    Vk = np.asarray(Vk, float)
    # Negative tiny values → 0
    Zk[Zk < 0] = 0.0
    return Zk, Vk

# ---- use it ----
Zk, Vk = kriging_grid(inst, xv, yv, model="exponential", frac_nugget=0.2)

# If you want a coverage mask (recommended), reuse the same k-NN mask you used for IDW:
# ... build 'support_mask' ... then:
# Zk = np.where(support_mask, Zk, np.nan)

# Plot with your existing plot function (xx, yy from np.meshgrid(xv, yv))

plt.figure(figsize=(8,7))
im = plt.pcolormesh(xx, yy, Zk, shading="auto")
plt.scatter(inst["lon_mid"], inst["lat_mid"], s=8, c="k", alpha=0.6, label="CML midpoints")
plt.colorbar(im, label="Rain rate (mm/h)")
plt.title(f"Exp:CML rain rate — {pd.to_datetime(time_map).strftime('%Y-%m-%d %H:%M UTC')}")
plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.legend()
plt.tight_layout(); plt.show()

Zk, Vk = kriging_grid(inst, xv, yv, model="spherical", frac_nugget=0.2)

# If you want a coverage mask (recommended), reuse the same k-NN mask you used for IDW:
# ... build 'support_mask' ... then:
# Zk = np.where(support_mask, Zk, np.nan)

# Plot with your existing plot function (xx, yy from np.meshgrid(xv, yv))

plt.figure(figsize=(8,7))
im = plt.pcolormesh(xx, yy, Zk, shading="auto")
plt.scatter(inst["lon_mid"], inst["lat_mid"], s=8, c="k", alpha=0.6, label="CML midpoints")
plt.colorbar(im, label="Rain rate (mm/h)")
plt.title(f"Sphr:CML rain rate — {pd.to_datetime(time_map).strftime('%Y-%m-%d %H:%M UTC')}")
plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.legend()
plt.tight_layout(); plt.show()

Zk, Vk = kriging_grid(inst, xv, yv, model="gaussian", frac_nugget=0.2)

# If you want a coverage mask (recommended), reuse the same k-NN mask you used for IDW:
# ... build 'support_mask' ... then:
# Zk = np.where(support_mask, Zk, np.nan)

# Plot with your existing plot function (xx, yy from np.meshgrid(xv, yv))

plt.figure(figsize=(8,7))
im = plt.pcolormesh(xx, yy, Zk, shading="auto")
plt.scatter(inst["lon_mid"], inst["lat_mid"], s=8, c="k", alpha=0.6, label="CML midpoints")
plt.colorbar(im, label="Rain rate (mm/h)")
plt.title(f"Gaus:CML rain rate — {pd.to_datetime(time_map).strftime('%Y-%m-%d %H:%M UTC')}")
plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.legend()
plt.tight_layout(); plt.show()


# Choose a more local correlation length & tighter mask
Z_exp, SS_exp, pars = krige_exponential(
    inst, xx, yy,
    range_km=60.0,
    nugget=0.3,
    max_neighbors=None,      # moving window
    enable_mask=True,
    max_dist_km=90.0,
    backend_when_mw="loop" # or "c" if available on your build
)
plt.figure(figsize=(8,7))
im = plt.pcolormesh(xx, yy, Z_exp, shading="auto")
plt.scatter(inst["lon_mid"], inst["lat_mid"], s=8, c="k", alpha=0.6, label="CML midpoints")
plt.colorbar(im, label="Rain rate (mm/h)")
plt.title(f"Exp-max-neig=16:CML rain rate — {pd.to_datetime(time_map).strftime('%Y-%m-%d %H:%M UTC')}")
plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.legend()
plt.tight_layout(); plt.show()
#%% some plotting
# midpoints
# --- imports you need ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_rain_rate_grid_ghana(
    item,
    vmin,
    vmax,
    levels=12,                    # number of discrete steps
    cbar_ticks=None,              # e.g. [0,2,5,10,20,50]
    title="Rain Rate [mm/h]",
    extent=(-5, 2.5, 4, 12),      # Ghana bounding box
    ocean_color="silver",
    ocean_alpha=0.5,
    cmap_name="jet",
    mask_threshold=0.0,           # <= this value will be shown as white
):
    """
    Plot a single gridded rainfall field over Ghana using PlateCarree projection.
    All values <= mask_threshold are masked (white). Remaining values use a
    discrete colormap between vmin and vmax.

    item: dict with keys
      - "grid": 2D array (rain field: mm/h or mm)
      - "lon" : 2D array (meshgrid longitudes)
      - "lat" : 2D array (meshgrid latitudes)
      - "timestamp" (optional): pd.Timestamp or str
      - "units" (optional): colorbar label override
    """
    z   = item["grid"]
    lon = item["lon"]
    lat = item["lat"]
    ts  = item.get("timestamp", "")
    units = item.get("units", "Rainfall Intensity [mm/h]")

    # mask NaNs and values <= threshold
    z_plot = np.ma.masked_invalid(z)
    z_plot = np.ma.masked_where(z_plot <= mask_threshold, z_plot)

    # discrete colormap with white for "under"
    levels_arr = np.linspace(vmin, vmax, levels)
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_under("white")  # anything < vmin (including masked) shows white
    norm = BoundaryNorm(levels_arr, cmap.N)

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": proj})
    ax.set_extent(extent, crs=proj)

    # base map features
    ax.add_feature(cfeature.OCEAN, zorder=1, color=ocean_color, alpha=ocean_alpha)
    ax.coastlines(resolution="10m", color="k", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="k")

    # gridlines
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                      linestyle="--", color="gray", linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}

    # plot field
    im = ax.pcolormesh(lon, lat, z_plot,
                       cmap=cmap, norm=norm,
                       shading="auto", transform=proj)

    # title
    ts_str = f" ({ts})" if ts != "" else ""
    ax.set_title(f"{title}{ts_str}", fontsize=14)

    # colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                      fraction=0.03, pad=0.08, extend="max")
    cb.set_label(units, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)

    plt.tight_layout()
    return fig, ax

# `xx`, `yy`, `Z` came from Section 5A after the IDW call
Z_ = Z.copy()
# Z_[Z_ <=0.05] = 0.0  # mask very low values to white
item_inst = {
    "grid": Z,
    "lon": xx,
    "lat": yy,
    "timestamp": pd.to_datetime(time_map).strftime("%Y-%m-%d %H:%M UTC"),
    "units": "Rainfall Intensity [mm/h]"
}

# pick ranges that fit your event — tweak as needed
fig, ax = plot_rain_rate_grid_ghana(
    item_inst,
    vmin=0.0,
    vmax=50.0,
    levels=12,
    cbar_ticks=[0, 2, 5, 10, 15, 20, 25, 30, 40, 50],
    title="CML Rain Rate [mm/h]",
    extent=(-3.5, 1.5, 4.5, 11.5)  # optional tighter box if you prefer
)
plt.show()

item_inst = {
    "grid": Zhr,
    "lon": xx,
    "lat": yy,
    "timestamp": pd.to_datetime(time_map).strftime("%Y-%m-%d %H:%M UTC"),
    "units": "Rainfall Intensity [mm/h]"
}

fig, ax = plot_rain_rate_grid_ghana(
    item_inst,
    vmin=0.0,
    vmax=25.0,
    levels=12,
    cbar_ticks=[0, 2, 5, 10, 15, 20, 25],
    title="CML Rain accumulation [mm/h]",
    extent=(-3.5, 1.5, 4.5, 11.5)  # optional tighter box if you prefer
)
plt.show()


#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_rain_rate_grid_ghana(
    item,
    vmin,
    vmax,
    levels=12,
    cbar_ticks=None,
    title="Rain Rate [mm/h]",
    extent=(-5, 2.5, 4, 12),
    ocean_color="silver",
    ocean_alpha=0.5,
    cmap_name="viridis",
    mask_threshold=0.0,
    # NEW: optional point overlay
    points_df=None,              # DataFrame with columns: lon, lat, (optional) value
    point_mode="edge",           # "edge" (uniform color) or "value" (color by value)
    point_size=10,
    point_alpha=0.9,
    point_color="k",
    point_cmap="gray",           # used when point_mode="value"
    point_vmin=None,             # used when point_mode="value"
    point_vmax=None,             # used when point_mode="value"
    point_edgecolor="white",
    point_linewidth=0.2,
    point_zorder=7
):
    """
    Plot a gridded rainfall field over Ghana (PlateCarree) and optionally overlay
    the CML midpoint locations used for interpolation.

    item: dict with keys
      - "grid": 2D array (rain field: mm/h or mm)
      - "lon" : 2D array (meshgrid longitudes)
      - "lat" : 2D array (meshgrid latitudes)
      - "timestamp" (optional): pd.Timestamp or str
      - "units" (optional): colorbar label override
    """
    z   = item["grid"]
    lon = item["lon"]
    lat = item["lat"]
    ts  = item.get("timestamp", "")
    units = item.get("units", "Rainfall Intensity [mm/h]")

    # mask NaNs and values <= threshold
    z_plot = z.copy()
    # z_plot = np.ma.masked_invalid(z)
    # z_plot = np.ma.masked_where(z_plot <= mask_threshold, z_plot)

    # discrete colormap with white for "under"
    levels_arr = np.linspace(vmin, vmax, levels)
    cmap = plt.get_cmap(cmap_name).copy()
    # cmap.set_under("white")
    norm = BoundaryNorm(levels_arr, cmap.N)

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": proj})
    ax.set_extent(extent, crs=proj)

    # base map
    ax.add_feature(cfeature.OCEAN, zorder=1, color=ocean_color, alpha=ocean_alpha)
    ax.coastlines(resolution="10m", color="k", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.6, edgecolor="k")

    # gridlines
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                      linestyle="--", color="gray", linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}

    # field
    im = ax.pcolormesh(lon, lat, z_plot,
                       cmap=cmap, norm=norm,
                       shading="auto", transform=proj)

    # points overlay (optional)
    if points_df is not None and len(points_df):
        if point_mode == "value" and "value" in points_df.columns:
            sc = ax.scatter(points_df["lon"], points_df["lat"],
                            c=points_df["value"],
                            s=point_size,
                            cmap=point_cmap,
                            vmin=point_vmin, vmax=point_vmax,
                            alpha=point_alpha,
                            edgecolor=point_edgecolor, linewidth=point_linewidth,
                            transform=proj, zorder=point_zorder)
            # small colorbar for points if using value mode
            cbp = fig.colorbar(sc, ax=ax, orientation="vertical",
                               fraction=0.030, pad=0.01)
            cbp.set_label("Point value", fontsize=9)
            cbp.ax.tick_params(labelsize=8)
        else:
            ax.scatter(points_df["lon"], points_df["lat"],
                       s=point_size,
                       c=point_color,
                       alpha=point_alpha,
                       edgecolor=point_edgecolor, linewidth=point_linewidth,
                       transform=proj, zorder=point_zorder)

    # title & colorbar for field
    ts_str = f" ({ts})" if ts != "" else ""
    ax.set_title(f"{title}{ts_str}", fontsize=14)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                      fraction=0.03, pad=0.08, extend="max")
    cb.set_label(units, fontsize=12)
    cb.ax.tick_params(labelsize=10)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)

    plt.tight_layout()
    return fig, ax


# midpoints you already built earlier:
mid = (ts_15min.drop_duplicates(subset=["link_id"])[
    ["link_id","XStart","YStart","XEnd","YEnd"]
].copy())
mid["lon_mid"] = (mid["XStart"] + mid["XEnd"]) / 2.0
mid["lat_mid"] = (mid["YStart"] + mid["YEnd"]) / 2.0

inst = (df_rate[df_rate["time"] == time_map]
        .merge(mid[["link_id","lon_mid","lat_mid"]], on="link_id", how="inner")
        .dropna(subset=["R_mm_h","lon_mid","lat_mid"]))

# build a points dataframe
pts_inst = inst.rename(columns={"lon_mid":"lon","lat_mid":"lat"})[["lon","lat","R_mm_h"]]
pts_inst = pts_inst.rename(columns={"R_mm_h":"value"})  # value column is optional

item_inst = {
    "grid": Z_exp, "lon": xx, "lat": yy,
    "timestamp": pd.to_datetime(time_map).strftime("%Y-%m-%d %H:%M UTC"),
    "units": "Rainfall Intensity [mm/h]"
}

fig, ax = plot_rain_rate_grid_ghana(
    item_inst, vmin=0, vmax=25, levels=12,
    cbar_ticks=np.linspace(0,25,10),#[0,2,5,10,15,20,25],
    title="CML Rain Rate [mm/h]",
    extent=(-3.5, 1.5, 4.5, 11.5),
    mask_threshold=0.0,
    points_df=pts_inst,
    point_mode="edge",         # uniform black dots
    point_size=8,
    point_alpha=0.9,
    point_color="k",
    point_edgecolor="white",
    point_linewidth=0.2
)
plt.show()