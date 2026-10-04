# If needed (only once per env):
# !conda install -y -c conda-forge cartopy matplotlib pandas
#%%
KML_PATH = "/home/kkumah/Projects/cml-stuff/data-cml/AT_MWL_NETWORK.kml"   # change if needed
EXTENT = (-3.5, 1.25, 4.6, 11.25) # (lon_min, lon_max, lat_min, lat_max)

import os, math, xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D
#%%
#%% Parse KML LineStrings
def parse_kml_lines(kml_path):
    tree = ET.parse(kml_path); root = tree.getroot()
    def findall_anyns(elem, tag): return elem.findall(f".//{{*}}{tag}")
    links=[]
    for pm in findall_anyns(root, "Placemark"):
        for ls in findall_anyns(pm, "LineString"):
            ce = ls.find(".//{*}coordinates")
            if ce is None or not ce.text: 
                continue
            coords=[]
            for c in ce.text.strip().replace("\n"," ").split():
                p=c.split(",")
                if len(p)>=2:
                    try:
                        coords.append((float(p[0]), float(p[1])))
                    except:
                        pass
            if len(coords)>=2:
                links.append({"coords": coords})
    return links

#%% Haversine distance (km)
def hav_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlmb  = math.radians(lon2 - lon1)
    a = (math.sin(dphi/2)**2
         + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

#%% Load links + compute length per link
links = parse_kml_lines(KML_PATH)
for L in links:
    coords = L["coords"]
    length = 0.0
    for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
        length += hav_km(lon1, lat1, lon2, lat2)
    L["length_km"] = length

#%% Plot
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(8,8), dpi=150)
ax = plt.axes(projection=proj)
ax.set_title("Airtel-Tigo CML Network based on KML file data", fontsize=14, pad=12)
ax.set_extent((EXTENT[0], EXTENT[1], EXTENT[2], EXTENT[3]), crs=proj)

# Aqua water + simple context
ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="aqua")
ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="aqua", edgecolor="none")
ax.coastlines(resolution="10m", linewidth=1.0)
ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.7, edgecolor="0.25")

gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.4, color="0.6", alpha=0.6, linestyle="--")
gl.right_labels = False
gl.top_labels = False

# Color rules
def pick_color(len_km):
    if len_km < 5.0:
        return "r"        # < 5 km
    elif len_km <= 10.0:
        return "b"        # 5–10 km
    else:
        return "k"        # > 10 km

for L in links:
    xs = [p[0] for p in L["coords"]]
    ys = [p[1] for p in L["coords"]]
    ax.plot(xs, ys, transform=proj, color=pick_color(L["length_km"]),
            linewidth=1.0, alpha=0.95)

# Legend
handles = [
    Line2D([0],[0], color="r",  lw=2, label="< 5 km"),
    Line2D([0],[0], color="b",  lw=2, label="5–10 km"),
    Line2D([0],[0], color="k",  lw=2, label="> 10 km"),
]
ax.legend(handles=handles, loc="lower right", frameon=True)

plt.show()



#%%
import os, re, math, xml.etree.ElementTree as ET
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

KML_PATH = "/home/kkumah/Projects/cml-stuff/data-cml/AT_MWL_NETWORK.kml"

def findall_anyns(elem, tag): return elem.findall(f".//{{*}}{tag}")

def hav_km(lon1, lat1, lon2, lat2):
    R=6371.0
    import math
    phi1,phi2=math.radians(lat1),math.radians(lat2)
    dphi=math.radians(lat2-lat1); dlmb=math.radians(lon2-lon1)
    a=(math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2)
    return R*2*math.atan2(math.sqrt(a), math.sqrt(1-a))

# Parse links + lengths
tree = ET.parse(KML_PATH); root = tree.getroot()
links=[]
for pm in findall_anyns(root,"Placemark"):
    name_el = pm.find(".//{*}name")
    name = name_el.text.strip() if (name_el is not None and name_el.text) else None
    for ls in findall_anyns(pm,"LineString"):
        ce = ls.find(".//{*}coordinates")
        if ce is None or not ce.text: continue
        coords=[]
        for c in ce.text.strip().replace("\n"," ").split():
            p=c.split(",")
            if len(p)>=2:
                try: coords.append((float(p[0]), float(p[1])))
                except: pass
        if len(coords)<2: continue
        L=0.0
        for (x1,y1),(x2,y2) in zip(coords[:-1], coords[1:]):
            L+=hav_km(x1,y1,x2,y2)
        links.append({"name":name,"length_km":L})

df = pd.DataFrame(links)
print("Links total:", len(df))
print("Length stats (km):", df["length_km"].describe())

# Length histogram with your bins
bins = [0,5,10,20,40,60,80]
plt.figure(figsize=(7,4), dpi=140)
plt.hist(df["length_km"], bins=bins, edgecolor="none")
plt.xlabel("Path length [km]"); plt.ylabel("Count"); plt.grid(axis="y", ls=":", alpha=0.6)
plt.title("CML link-length distribution (from KML)")
plt.show()


#%%
rsl_files = pd.read_csv('/home/kkumah/Projects/cml-stuff/data-cml/outs/Multi-Link-Multi-Timestamp_coupled_linkdata_kkk_20251006.csv')
print(rsl_files.head(10))
#%%
# cml_density_and_stats.py
# ------------------------------------------------------------
# Input: a DataFrame `df` already in memory OR a CSV path with columns:
# ['Frequency','DateTime','Pmin','Pmax','XStart','YStart','XEnd','YEnd',
#  'ID','Polarization','PathLength','TSL_AVG']
# Output:
#   /mnt/data/cml_density_map.png
#   /mnt/data/cml_freq_len_stats.png
# ------------------------------------------------------------

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

# ======= CONFIG =======
CSV_PATH = None  # e.g., "/home/kkumah/Projects/cml-stuff/outs/links_15min.csv"; leave None if df is already loaded
# EXTENT = (-3.5, 1.25, 4.8, 11.25)  # (min_lon, max_lon, min_lat, max_lat)
OUT_DENSITY = "/home/kkumah/Projects/cml-stuff/data-cml/cml_density_map.png"
OUT_STATS   = "/home/kkumah/Projects/cml-stuff/data-cml/cml_freq_len_stats.png"
GRID_SIZE   = 55  # hexbin gridsize (tune as you like)
# Length bins/colors
def pick_color(L):
    if L < 5:   return "r"
    if L <= 10: return "b"
    return "k"
# ======================

# --- Load data (either from CSV or assume `df` already exists) ---
if CSV_PATH is not None:
    df = rsl_files#pd.read_csv(CSV_PATH)
else:
    # If you're running in a notebook where `df` is already defined, this will use it.
    # Otherwise, raise a helpful error.
    try:
        df  # noqa: F821
    except NameError:
        raise RuntimeError("No CSV_PATH provided and no `df` found in memory.")

# --- Basic cleaning/coercion ---
for col in ["XStart","YStart","XEnd","YEnd","PathLength","Frequency"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# --- Collapse 15-min rows to unique links ---
# We treat each unique 'ID' as one link; keep the first row's geometry & metadata.
# (If you prefer, use groupby().agg(...) to pick medians for Frequency/PathLength.)
keep_cols = ["ID","Frequency","XStart","YStart","XEnd","YEnd","PathLength","Polarization"]
links = (df[keep_cols]
         .dropna(subset=["XStart","YStart","XEnd","YEnd"])
         .drop_duplicates(subset=["ID"])
         .reset_index(drop=True))

# Safety: fill lengths if missing by haversine on endpoints
def hav_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlmb/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

if "PathLength" in links.columns:
    missing_len = links["PathLength"].isna()
    if missing_len.any():
        links.loc[missing_len, "PathLength"] = [
            hav_km(lon1, lat1, lon2, lat2)
            for lon1, lat1, lon2, lat2 in zip(
                links.loc[missing_len, "XStart"],
                links.loc[missing_len, "YStart"],
                links.loc[missing_len, "XEnd"],
                links.loc[missing_len, "YEnd"],
            )
        ]

# Midpoints (for density)
links["lon_mid"] = (links["XStart"] + links["XEnd"]) / 2.0
links["lat_mid"] = (links["YStart"] + links["YEnd"]) / 2.0

# --- DENSITY MAP (hexbin of midpoints) + link overlay colored by length ---
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(9.5, 10), dpi=160)
ax = plt.axes(projection=proj)
ax.set_title("Current CML Network", fontsize=16, pad=12)
ax.set_extent((EXTENT[0], EXTENT[1], EXTENT[2], EXTENT[3]), crs=proj)

# Aqua water + boundaries
ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="aqua")
ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="aqua", edgecolor="none")
ax.coastlines(resolution="10m", linewidth=1.0)
ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.7, edgecolor="0.25")
gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.4, color="0.6", alpha=0.6, linestyle="--")
gl.right_labels = False
gl.top_labels = False

# Hexbin density of midpoints
# hb = ax.hexbin(
#     links["lon_mid"], links["lat_mid"],
#     gridsize=GRID_SIZE, transform=proj, cmap="Greys", bins="log", alpha=0.8
# )
# cb = plt.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
# cb.set_label("log10(count) of links (midpoint)")

# Overlay links colored by length bins
for _, r in links.iterrows():
    ax.plot(
        [r["XStart"], r["XEnd"]],
        [r["YStart"], r["YEnd"]],
        transform=proj,
        color=pick_color(float(r["PathLength"])),
        linewidth=0.9,
        alpha=0.95,
    )

# Legend for length bins
legend_items = [
    Line2D([0],[0], color="r",  lw=2, label="< 5 km"),
    Line2D([0],[0], color="b",  lw=2, label="5–10 km"),
    Line2D([0],[0], color="k",  lw=2, label="> 10 km"),
]
ax.legend(handles=legend_items, loc="lower right", frameon=False)
plt.savefig(OUT_DENSITY, bbox_inches="tight")

# --- STATS PLOT: Frequency vs Path length with marginal histograms ---
df_stats = links.dropna(subset=["Frequency","PathLength"]).copy()
colors = df_stats["PathLength"].apply(lambda L: pick_color(float(L))).values

fig = plt.figure(figsize=(8.5, 8.5), dpi=160)
gs = GridSpec(4, 4, figure=fig, wspace=0.15, hspace=0.1)

ax_scatter = fig.add_subplot(gs[1:, :3])
ax_histx   = fig.add_subplot(gs[0,  :3], sharex=ax_scatter)
ax_histy   = fig.add_subplot(gs[1:,  3], sharey=ax_scatter)

ax_scatter.scatter(df_stats["PathLength"], df_stats["Frequency"],
                   s=12, c=colors, alpha=0.9, edgecolors="none")
ax_scatter.set_xlabel("Path length [km]")
ax_scatter.set_ylabel("Frequency [GHz]")
ax_scatter.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

# histX (length)
bins_len = np.histogram_bin_edges(df_stats["PathLength"], bins="auto")
ax_histx.hist(df_stats["PathLength"], bins=bins_len, edgecolor="none", color="0.5")
ax_histx.set_ylabel("count")
ax_histx.grid(True, axis="y", linestyle=":", alpha=0.5)
ax_histx.tick_params(labelbottom=False)

# histY (frequency)
bins_f = np.histogram_bin_edges(df_stats["Frequency"], bins="auto")
ax_histy.hist(df_stats["Frequency"], bins=bins_f, orientation="horizontal", edgecolor="none", color="0.5")
ax_histy.set_xlabel("count")
ax_histy.grid(True, axis="x", linestyle=":", alpha=0.5)
ax_histy.tick_params(labelleft=False)

# Length-bin legend
ax_scatter.legend(legend_items, loc="lower right", frameon=True)

plt.savefig(OUT_STATS, bbox_inches="tight")

print(f"Saved density map  → {OUT_DENSITY}")
print(f"Saved stats figure → {OUT_STATS}")

# --- (Optional) quick counts by length bin ---
len_bin = pd.cut(
    links["PathLength"],
    bins=[-np.inf, 5, 10, np.inf],
    labels=["<5 km", "5–10 km", ">10 km"]
)
counts = len_bin.value_counts().sort_index()
print("\nLinks per length bin:")
print(counts.to_string())


#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ---- constants (EUM/MSG expect per-wavenumber radiance in mW m^-2 sr^-1 (cm^-1)^-1)
C1, C2 = 1.19104e-5, 1.43877
BANDS = {
    "IR108": {"var": "channel_9",  "attr": "ch09_cal", "nu_c": 931.122, "alpha": 0.9983, "beta": 0.6256, "lambda_um": 10.8},
    "IR120": {"var": "channel_10", "attr": "ch10_cal", "nu_c": 839.113, "alpha": 0.9988, "beta": 0.4002, "lambda_um": 12.0},
}

def parse_offset_slope(val):
    parts = str(val).strip().replace(",", " ").split()
    if len(parts) < 2:
        raise ValueError(f"Cannot parse offset/slope from {val!r}")
    return float(parts[0]), float(parts[1])

def looks_like_radiance(da):
    u = (da.attrs.get("units") or "").lower()
    ln = (da.attrs.get("long_name") or "").lower()
    if "radiance" in u or "radiance" in ln:
        return True, "per-wavenumber" if "cm-1" in u or "(cm-1)" in u else ("per-wavelength" if "um-1" in u or "µm-1" in u else "unknown")
    # Heuristic on values
    med = float(np.nanmedian(da.values))
    if 0.5 <= med <= 100:  # ~radiance scale in mW m^-2 sr^-1 (cm^-1)^-1
        return True, "unknown"
    if med > 4095:  # not DN
        return True, "unknown"
    return False, "unknown"

def ensure_per_wavenumber_mW(rad_da, band_info):
    """Convert to mW m^-2 sr^-1 (cm^-1)^-1 if units say per-wavelength."""
    units = (rad_da.attrs.get("units") or "").lower()
    if "w m-2 sr-1 um-1" in units or "w m^-2 sr^-1 µm^-1" in units or "w/m^2/sr/um" in units:
        factor = 0.1 * (band_info["lambda_um"] ** 2)  # Lσ = 0.1*λ^2 * Lλ
        out = rad_da.astype("float32") * factor
        out.attrs["units"] = "mW m-2 sr-1 (cm-1)-1"
        out.attrs["note"] = f"converted from W m-2 sr-1 um-1 using 0.1*λ^2, λ={band_info['lambda_um']} µm"
        return out
    return rad_da  # assume already per-wavenumber or unspecified

def dn_to_radiance(da, ds, band_info):
    # Prefer CF scale/offset if present
    sc, off = da.attrs.get("scale_factor"), da.attrs.get("add_offset")
    if sc is not None or off is not None:
        arr = da.astype("float32")
        if sc is not None: arr = arr * float(sc)
        if off is not None: arr = arr + float(off)
        out = arr
        out.attrs["units"] = "mW m-2 sr-1 (cm-1)-1"
        out.attrs["note"] = "CF scale/offset"
        return out

    # Fallback: global offset+slope from ch##_cal
    attr = band_info["attr"]
    if attr in ds.attrs:
        offset, slope = parse_offset_slope(ds.attrs[attr])
        out = offset + da.astype("float32") * slope
        out.attrs["units"] = "mW m-2 sr-1 (cm-1)-1"
        out.attrs["note"] = f"offset+slope from {attr}"
        return out

    return None  # nothing we can do

def bt_from_radiance(rad, band_info):
    p = band_info
    Le = xr.where(rad > 1e-9, rad, np.nan)  # guard log
    term = C1 * (p["nu_c"]**3) / Le + 1.0
    return (C2 * p["nu_c"]) / (p["alpha"] * np.log(term)) - (p["beta"] / p["alpha"])

# ---------- choose band ----------
BAND = "IR108"  # or "IR120"
info = BANDS[BAND]

PATH = "/home/kkumah/Projects/cml-stuff/satellite_data/msg/run_20251014_014430/20250619_153010/HRSEVIRI_20250619T153010Z_20250619T154243Z_epct_c670a0b7_FC_dup1760406351_CROP.nc"
ds = xr.open_dataset(PATH)
da = ds[info["var"]].astype("float32")

# Step 1: detect/calibrate
is_rad, rad_kind = looks_like_radiance(da)
if is_rad:
    rad = ensure_per_wavenumber_mW(da, info)
else:
    rad = dn_to_radiance(da, ds, info)
    if rad is None:
        raise RuntimeError("Could not find calibration to convert DN -> radiance.")

# Step 2: BT
Tb = bt_from_radiance(rad, info)
Tb.attrs.update(units="K", long_name=f"Brightness Temperature {BAND}")

print(
    f"{BAND} BT stats (K): "
    f"min={float(np.nanmin(Tb)):.1f}, mean={float(np.nanmean(Tb)):.1f}, max={float(np.nanmax(Tb)):.1f}"
)

# Step 3: plot in geostationary projection
proj = ccrs.Geostationary(satellite_height=35785831.0, central_longitude=0.0, sweep_axis="y")
fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection=proj)
ax.set_title(Tb.long_name + " [K]")
ax.coastlines(linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.6)
im = Tb.plot(ax=ax, transform=proj, cmap="inferno")
plt.show()

ds.close()

#%%
PATH = "/home/kkumah/Projects/cml-stuff/satellite_data/msg/run_20251014_014430/20250619_153010/HRSEVIRI_20250619T153010Z_20250619T154243Z_epct_c670a0b7_FC_dup1760406351_CROP.nc"
ds = xr.open_dataset(PATH)

c1 = 1.191044e-5   # mW/(m²·sr·cm⁻⁴) - First radiation constant
c2 = 1.4387752     # K·cm - Second radiation constant

bt = (c2 * 931.122) / np.log(1 + (c1 * 931.122**3) / ds['channel_9'])
bt.attrs.update(units="K", long_name=f"Brightness Temperature {BAND}")

BAND = "IR108"  # or "IR120"
info = BANDS[BAND]
term = c1 * (info["nu_c"]**3) / ds['channel_9'] + 1.0
bt_ = ((c2 * info["nu_c"]) / np.log(term)- info["beta"]) / info["alpha"]
bt_.attrs.update(units="K", long_name=f"Brightness Temperature {BAND}")




proj = ccrs.Geostationary(satellite_height=35785831.0, central_longitude=0.0, sweep_axis="y")
fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection=proj)
ax.coastlines(linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.6)
im = bt.plot(ax=ax, transform=proj, cmap="jet")
ax.set_title(bt.long_name + " [K] from bt")
plt.show()

plt.scatter(bt.values,bt_.values)
plt.xlabel("bt values")
plt.ylabel("bt_ values")
plt.title("Scatter plot of bt vs bt_")
plt.show()


#%%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
C1, C2 = 1.19104e-5, 1.43877
BANDS = {
    "IR108": {"var": "channel_9",  "attr": "ch09_cal", "nu_c": 931.122, "alpha": 0.9983, "beta": 0.6256, "lambda_um": 10.8},
    "IR120": {"var": "channel_10", "attr": "ch10_cal", "nu_c": 839.113, "alpha": 0.9988, "beta": 0.4002, "lambda_um": 12.0},
}

# BT_COEFF = {
#     "WV062": {"nu": 1596.080, "alpha": 0.9959, "beta": 2.0780},  # channel_5
#     "IR108": {"nu":  931.122, "alpha": 0.9983, "beta": 0.6256},  # channel_9
#     "IR120": {"nu":  839.113, "alpha": 0.9988, "beta": 0.4002},  # channel_10
# }

ds_path = r'/home/kkumah/Projects/cml-stuff/satellite_data/msg/run_20251025_222108/20250619_161511/HRSEVIRI_20250619T161511Z_20250619T162743Z_epct_cbaff216_FC_dup1761430899_BT.nc'

ds = xr.open_dataset(ds_path)

ds_ir108 = ds['BT_IR108']
ds_ir120 = ds['BT_IR120']
proj = ccrs.Geostationary(satellite_height=35785831.0, central_longitude=0.0, sweep_axis="y")
fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection=proj)
ax.coastlines(linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.6)
im = ds_ir108.plot(ax=ax, transform=proj, cmap="jet")
ax.set_title(ds_ir108.long_name + " [K] from bt")
# --- Projection (assuming same as your dataset) ---
# --- Add gridlines & labeled ticks ---
gl = ax.gridlines(draw_labels=True, linewidth=1.5, color='gray', alpha=0.6, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 9}
gl.ylabel_style = {'size': 9}
proj = ccrs.PlateCarree()
# Optional: specify tick spacing like your rainfall maps
ax.set_extent([-4, 1.5, 4.5, 11.5], crs=proj)
gl.xlocator = mticker.FixedLocator(np.arange(-4, 3, 1))
gl.ylocator = mticker.FixedLocator(np.arange(5, 13, 1))

plt.show()

BAND = "IR108"  # or "IR120"
info = BANDS[BAND]
ds_chan_108 = ds['channel_9']
term = C1 * (info["nu_c"]**3) / ds['channel_9'] + 1.0
bt_108 = ((C2 * info["nu_c"]) / np.log(term)- info["beta"]) / info["alpha"]


proj = ccrs.Geostationary(satellite_height=35785831.0, central_longitude=0.0, sweep_axis="y")
fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection=proj)
ax.coastlines(linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.6)
im = bt_108.plot(ax=ax, transform=proj, cmap="jet")
ax.set_title("BT_IR108 [K]")
plt.show()


BAND = "IR120"  # or "IR120"
info = BANDS[BAND]
ds_chan_120 = ds['channel_10']
term = C1 * (info["nu_c"]**3) / ds['channel_10'] + 1.0
bt_120 = ((C2 * info["nu_c"]) / np.log(term)- info["beta"]) / info["alpha"]

proj = ccrs.Geostationary(satellite_height=35785831.0, central_longitude=0.0, sweep_axis="y")
fig = plt.figure(figsize=(6,6))
ax = plt.axes(projection=proj)
ax.coastlines(linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.6)
im = bt_120.plot(ax=ax, transform=proj, cmap="jet")
ax.set_title("BT_IR120 [K]")
plt.show()


#%%
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib.colors as mcolors

clm_path = r'/home/kkumah/Projects/cml-stuff/satellite_data/msg_clm/run_20251025_221705/20250619_161500/MSGCLMK_20250619T161500Z_20250619T161500Z_epct_b0b17714_C_dup1761430652_CLM.nc'
ds_clm = xr.open_dataset(clm_path)
cmask = ds_clm["cloud_mask"].squeeze()

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors

# --- Cloud mask label mapping ---
label_map = {0: "Clear water", 1: "Clear land", 2: "Cloud", 3: "No data"}
cmap = mcolors.ListedColormap(["#6EC5FF", "#E6D96A", "#FFFFFF", "#A0A0A0"])
norm = mcolors.BoundaryNorm([0,1,2,3,4], cmap.N)

# --- CRS definitions ---
data_crs = ccrs.Geostationary(satellite_height=35785831.0,
                              central_longitude=0.0, sweep_axis="y")
map_crs = ccrs.PlateCarree()

# --- Mesh grid for plotting ---
X, Y = np.meshgrid(cmask["x"].values, cmask["y"].values)

# --- Plot setup ---
fig = plt.figure(figsize=(6, 6))
ax = plt.axes(projection=map_crs)
ax.set_extent([-4, 1.5, 4.5, 11.5], crs=map_crs)

# --- Background and borders ---
ax.coastlines(linewidth=0.8)
ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.6)
ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor='none', edgecolor='gray', linewidth=0.5)
ax.add_feature(cfeature.RIVERS.with_scale("50m"), edgecolor='lightgray', linewidth=0.4)

# --- Cloud mask layer ---
im = ax.pcolormesh(X, Y, cmask.values, transform=data_crs,
                   cmap=cmap, norm=norm, shading="nearest")

# --- Colorbar ---
cbar = plt.colorbar(im, ticks=np.arange(0.5, 4.5))
cbar.ax.set_yticklabels([label_map[i] for i in range(4)])
cbar.set_label("Cloud Mask Classes")

# --- Gridlines and ticks (as in rainfall map) ---
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 9}
gl.ylabel_style = {'size': 9}
gl.xlocator = plt.FixedLocator(np.arange(-4, 3, 1))
gl.ylocator = plt.FixedLocator(np.arange(5, 13, 1))

# --- Title ---
plt.title("MSG Cloud Mask Classification\n" + str(cmask.attrs.get("time", "")), fontsize=12)

plt.tight_layout()
plt.show()