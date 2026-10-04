#%% --- STEP 0: Setup ---
from Utils import *
#%% <<< EDIT THESE >>>
path_to_put_output = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'
META_PATH = os.path.join(path_to_put_output, 'matched_metadata_kkk_20250527.csv')
RSL_PATH  = os.path.join(path_to_put_output, 'Linkdata_AT_20250824_processed_on_20250527.dat')
OUT_DIR   = path_to_put_output
# os.makedirs(OUT_DIR, exist_ok=True)


#%% Load data
meta = pd.read_csv(META_PATH)
meta['Freq_GHz'] = meta['Frequency'] / 1000 # convert to GHz

link = pd.read_csv(RSL_PATH)
link['timestamp'] = to_datetime_utc(link['DateTime'])
# link = link.dropna(subset=['timestamp'])

# Basic sanity
need_meta = {'Temp_ID','XStart','YStart','XEnd','YEnd','Freq_GHz','Polarization'}
need_rsl  = {'timestamp','ID','RSL_MIN','RSL_MAX','TSL_AVG'}
missing_meta = need_meta - set(meta.columns)
missing_rsl  = need_rsl  - set(link.columns)
if missing_meta:
    raise ValueError(f"metadata missing: {sorted(missing_meta)}")
if missing_rsl:
    raise ValueError(f"link_raw missing: {sorted(missing_rsl)}")

meta = ensure_length_km(meta)

print("Loaded meta:", meta.shape, " | columns:", list(meta.columns))
print("Loaded link:", link.shape, " | columns:", list(link.columns))
print(link.head(3))

#%% ===== STEP 1: CML Preprocessing & QC =====
print(f'--- Apply cleaning per link (keeps 15-min cadence) ---')
blocks, qcstats = [], {}
for lid, g in link.groupby("ID", sort=False):
    gg, st = clean_timeseries_robust(g, atpc_enabled=False)
    gg["link_id"] = lid
    blocks.append(gg)
    qcstats[lid] = st

df1 = pd.concat(blocks, ignore_index=True)
print("QC sample:", df1[["timestamp","link_id","RSL_MIN","RSL_MAX","TSL_AVG","Pmin","Pmax"]].head())

# optional: inspect one link’s stats
one = next(iter(qcstats))
print("Example stats for", one, "→", qcstats[one])

link[link['ID'] == lid].head()


# for pid in link['ID'].unique()[:15]:
#     subset = df1[df1['link_id'] == pid]
#     # print(subset.shape)
#     print(pid)
#     # plt.figure(figsize=(12, 4))
#     # plt.plot(subset['timestamp'], subset['RSL_MIN'], label='RSL_MIN', color='blue', marker='o', markersize=3)
#     # plt.plot(subset['timestamp'], subset['RSL_MAX'], label='RSL_MAX', color='orange', marker='o', markersize=3)
#     # plt.plot(subset['timestamp'], subset['TSL_AVG'], label='TSL_AVG', color='green', marker='o', markersize=3)


#     # plt.title(f'Signal Levels for Link ID: {pid}')
#     # plt.xlabel('Timestamp')
#     # plt.ylabel('Signal Level (dB)')
#     # plt.legend()
#     # plt.grid()
#     # plt.tight_layout()
#     # plt.show()

#     if not subset['Pmin'].isnull().all() and not subset['Pmax'].isnull().all():
#         plt.figure(figsize=(12, 4))
#         plt.plot(subset['timestamp'], subset['Pmin'], label='Pmin', color='red', marker='o', markersize=3)
#         plt.plot(subset['timestamp'], subset['Pmax'], label='Pmax', color='purple', marker='o', markersize=3)
#         plt.title(f'Path Attenuation for Link ID: {pid}')
#         plt.xlabel('Timestamp')
#         plt.ylabel('Path Attenuation (dB)')
#         plt.legend()
#         plt.grid()
#         plt.tight_layout()
#         plt.show()


#%%

df_with_p = link
clean_blocks = []
for lid, g in df_with_p.groupby("ID", sort=False):
    gg = clean_pmin_pmax(g)
    clean_blocks.append(gg.assign(link_id=lid))
df_qc = pd.concat(clean_blocks, ignore_index=True)


# for pid in df_qc['ID'].unique()[:20]:
#     g_raw = df_with_p[df_with_p.ID==pid].copy()
#     g_qc  = df_qc[df_qc.ID==pid].copy()

#     if not g_raw['Pmin'].isnull().all() and not g_raw['Pmax'].isnull().all():

#         print(f"Before QC: min/max/mean _{pid}")
#         # print(g_raw[["Pmin","Pmax"]].describe())

#         plt.figure(figsize=(12, 4))
#         plt.plot(g_raw['timestamp'], g_raw['Pmin'], label='Pmin', color='red', marker='o', markersize=3)
#         plt.plot(g_raw['timestamp'], g_raw['Pmax'], label='Pmax', color='purple', marker='o', markersize=3)
#         plt.title(f'Path Attenuation for Link ID: {pid}-raw')
#         plt.xlabel('Timestamp')
#         plt.ylabel('Path Attenuation (dB)')
#         plt.legend()
#         plt.grid()
#         plt.tight_layout()
#         plt.show()

#         print(f"\nAfter QC: min/max/mean_{pid}")
#         # print(g_qc[["Pmin_clean","Pmax_clean","P_used"]].describe())

#         plt.figure(figsize=(12, 4))
#         plt.plot(g_qc['timestamp'], g_qc['Pmin_clean'], label='Pmin', color='red', marker='o', markersize=3)
#         plt.plot(g_qc['timestamp'], g_qc['Pmax_clean'], label='Pmax', color='purple', marker='o', markersize=3)
#         plt.plot(g_qc['timestamp'], g_qc['P_used'], label='P_used', color='black', marker='o', markersize=3)
#         plt.title(f'Path Attenuation for Link ID: {pid}-cleaned')
#         plt.xlabel('Timestamp')
#         plt.ylabel('Path Attenuation (dB)')
#         plt.legend()
#         plt.grid()
#         plt.tight_layout()
#         plt.show()


#%%  Step 2: Wet–Dry Classification/ Baseline / Attenuation Estimation
# ---- knobs you can tune ----
CADENCE_MIN       = 15
BASELINE_HOURS    = 48              # time-based rolling median
BASELINE_MIN_PTS  = 24              # min samples within window
ATT_THRESH_DB     = 1.0             # attenuation threshold for "wet"
NN_RADIUS_KM      = 30.0            # neighbour search radius
NN_MIN_NEIGHBORS  = 2               # need at least K neighbours
NN_MIN_FRACTION   = 0.4             # OR at least this fraction wet among neighbours
WAA_CONST_DB      = 0.05             # (optional) constant wet-antenna subtraction later

# ---- helpers ----
def to_datetime_utc(s): return pd.to_datetime(s, utc=True, errors="coerce")

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2-lat1, lon2-lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def link_centers(meta):
    # Accept either (XStart,YStart,XEnd,YEnd) or (tx_lon,tx_lat,rx_lon,rx_lat)
    cols_xy = ['XStart','YStart','XEnd','YEnd']
    cols_ll = ['tx_lon','tx_lat','rx_lon','rx_lat']
    if set(cols_xy).issubset(meta.columns):
        cx = (meta['XStart'] + meta['XEnd'])/2.0
        cy = (meta['YStart'] + meta['YEnd'])/2.0
        return pd.DataFrame({'ID':meta['ID'], 'lon_c':cx, 'lat_c':cy})
    elif set(cols_ll).issubset(meta.columns):
        clat = (meta['tx_lat'] + meta['rx_lat'])/2.0
        clon = (meta['tx_lon'] + meta['rx_lon'])/2.0
        return pd.DataFrame({'ID':meta['ID'], 'lon_c':clon, 'lat_c':clat})
    else:
        raise ValueError("Metadata must have either X/Y start/end or tx/rx lat/lon columns.")

# ---- 1) Initial baseline (time-window rolling median) ----
def initial_baseline(df_qc):
    # time-based rolling median per link (robust to gaps)
    out = []
    for pid, g in df_qc.groupby('ID', sort=False):
        gg = g.sort_values('timestamp').copy()
        gg['timestamp'] = to_datetime_utc(gg['timestamp'])
        gg = gg.set_index('timestamp')
        # 48h rolling window on time index
        base = gg['P_used'].rolling(f'{BASELINE_HOURS}H', min_periods=BASELINE_MIN_PTS).median()
        gg['Baseline0'] = base.ffill().bfill()
        out.append(gg.reset_index())
    return pd.concat(out, ignore_index=True)

# ---- 2) Provisional attenuation & wet flag (local) ----
def provisional_att_wet(df_bl):
    df = df_bl.copy()
    df['A0_db'] = (df['Baseline0'] - df['P_used']).clip(lower=0.0)
    df['wet0']  = df['A0_db'] > ATT_THRESH_DB
    return df

# ---- 3) Neighbour voting at each timestamp ----
def neighbour_vote(df_att, meta):
    meta_ = meta.copy()
    meta_.rename(columns={'Temp_ID':'ID'}, inplace=True)
    centers = link_centers(meta_)
    df = df_att.merge(centers, on='ID', how='left')
    final = []
    for ts, g in df.groupby('timestamp', sort=False):
        g = g.copy()
        lat = g['lat_c'].values; lon = g['lon_c'].values
        wet0 = g['wet0'].values
        ids  = g['ID'].values
        n = len(g)
        wet_final = np.zeros(n, dtype=bool)
        if n == 0:
            continue
        # distance matrix (n x n)
        D = haversine_km(lat[:,None], lon[:,None], lat[None,:], lon[None,:])
        for i in range(n):
            if not wet0[i]:
                wet_final[i] = False
                continue
            nbr_mask = (D[i] > 0) & (D[i] <= NN_RADIUS_KM)
            nbr_count = int(nbr_mask.sum())
            if nbr_count == 0:
                wet_final[i] = True  # keep if isolated (conservative)
                continue
            nbr_wet = int(wet0[nbr_mask].sum())
            condK   = (nbr_wet >= NN_MIN_NEIGHBORS)
            condFrac= (nbr_count > 0) and ((nbr_wet / nbr_count) >= NN_MIN_FRACTION)
            wet_final[i] = bool(condK or condFrac)
        g['wet'] = wet_final
        final.append(g)
    return pd.concat(final, ignore_index=True)

# ---- 4) Recompute dry-only baseline and final attenuation ----
def dry_baseline_and_attenuation(df_voted):
    out = []
    for pid, g in df_voted.groupby('ID', sort=False):
        gg = g.sort_values('timestamp').copy().set_index('timestamp')
        # dry-only samples for baseline
        s = gg['P_used'].where(~gg['wet'])
        base = s.rolling(f'{BASELINE_HOURS}H', min_periods=max(6, BASELINE_MIN_PTS//2)).median()
        base = base.ffill().bfill()
        gg['Baseline'] = base
        gg['A_db']     = (gg['Baseline'] - gg['P_used']).clip(lower=0.0)
        out.append(gg.reset_index())
    return pd.concat(out, ignore_index=True)

# ======= RUN THE LOGIC =======
# df_qc must have: ['timestamp','ID','P_used']; meta must have coords per ID
df0 = initial_baseline(df_qc) # rolling median over the last 48 h (time-based, so gaps are okay).
df1 = provisional_att_wet(df0) # first-pass attenuation = Baseline0 − P_used.
df2 = neighbour_vote(df1, meta) # at each timestamp, a link stays “wet” only if at least one nearby link (within 30 km) is also wet (≥ K neighbours or ≥ fraction).
df3 = dry_baseline_and_attenuation(df2)

# Optional: Wet Antenna Attenuation (constant)
df3['A_eff_db'] = np.where(df3['wet'], np.maximum(df3['A_db'] - WAA_CONST_DB, 0.0), 0.0)

# What you get:
#   - df3['wet']       : final wet/dry flag
#   - df3['Baseline']  : dry-only baseline (per link)
#   - df3['A_db']      : final path attenuation (dB)
#   - df3['A_eff_db']  : attenuation after WAA subtraction (if applied)

# for pid in df_qc['ID'].unique()[:20]:
#     g_raw = df3[df3.ID==pid].copy()
#     print(pid)
#     # g_qc  = df_qc[df_qc.ID==pid].copy()

#     if not g_raw['Pmin'].isnull().all() and not g_raw['Pmax'].isnull().all():

#         # print(f"Before QC: min/max/mean _{pid}")
        
#         # print(g_raw[["Pmin","Pmax"]].describe())

#         plt.figure(figsize=(12, 4))
#         plt.plot(g_raw['timestamp'], g_raw['A_eff_db'], label='A_eff_db', color='red', marker='o', markersize=3)
#         plt.plot(g_raw['timestamp'], g_raw['A_db'], label='A_db', color='purple', marker='o', markersize=3)
#         plt.title(f'Path Attenuation for Link ID: {pid}-raw')
#         plt.xlabel('Timestamp')
#         plt.ylabel('Path Attenuation (dB)')
#         plt.legend()
#         plt.grid()
#         plt.tight_layout()
#         plt.show()


#%% Rainfall Rate from ITU Power Law

# Partial extract of ITU Table 5 (up to 50 GHz). You can extend if needed.
data = {
    "Frequency_GHz": [
         1,   1.5,   2,   2.5,   3,   3.5,   4,   4.5,   5,   5.5,
         6,   7,   8,   9,   10,  11,  12,  13,  14,  15,
        16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
        36,  37,  38,  39,  40,  41,  42,  43,  44,  45,
        46,  47,  48,  49,  50
    ],
    "kH": [0.0000259,0.0000443,0.0000847,0.0001321,0.0001390,0.0001155,0.0001071,0.0001340,0.0002162,0.0003909,
            0.0007056,0.001915,0.004115,0.007535,0.01217,0.01772,0.02386,0.03041,0.03738,0.04481,
            0.05282,0.06146,0.07078,0.08084,0.09164,0.1032,0.1155,0.1286,0.1425,0.1571,
            0.1724,0.1884,0.2051,0.2224,0.2403,0.2588,0.2778,0.2972,0.3171,0.3374,
            0.3580,0.3789,0.4001,0.4215,0.4431,0.4647,0.4865,0.5084,0.5302,0.5521,
            0.5738,0.5956,0.6172,0.6386,0.6600],
    "αH": [0.9691,1.0185,1.0664,1.1209,1.2322,1.4189,1.6009,1.6948,1.6969,1.6499,
            1.5900,1.4810,1.3905,1.3155,1.2571,1.2140,1.1825,1.1586,1.1396,1.1233,
            1.1086,1.0949,1.0818,1.0691,1.0568,1.0447,1.0329,1.0214,1.0101,0.9991,
            0.9884,0.9780,0.9679,0.9580,0.9485,0.9392,0.9302,0.9214,0.9129,0.9047,
            0.8967,0.8890,0.8816,0.8743,0.8673,0.8605,0.8539,0.8476,0.8414,0.8355,
            0.8297,0.8241,0.8187,0.8134,0.8084],
    "kV": [0.0000308,0.0000574,0.0000998,0.0001464,0.0001942,0.0002346,0.0002461,0.0002347,0.0002428,0.0003115,
            0.0004878,0.001425,0.003450,0.006691,0.01129,0.01731,0.02455,0.03266,0.04126,0.05008,
            0.05899,0.06797,0.07708,0.08642,0.09611,0.1063,0.1170,0.1284,0.1404,0.1533,
            0.1669,0.1813,0.1964,0.2124,0.2291,0.2465,0.2646,0.2833,0.3026,0.3224,
            0.3427,0.3633,0.3844,0.4058,0.4274,0.4492,0.4712,0.4932,0.5153,0.5375,
            0.5596,0.5817,0.6037,0.6255,0.6472],
    "αV": [0.8592,0.8957,0.9490,1.0085,1.0688,1.1387,1.2476,1.3987,1.5317,1.5882,
            1.5728,1.4745,1.3797,1.2895,1.2156,1.1617,1.1216,1.0901,1.0646,1.0440,
            1.0273,1.0137,1.0025,0.9930,0.9847,0.9771,0.9700,0.9630,0.9561,0.9491,
            0.9421,0.9349,0.9277,0.9203,0.9129,0.9055,0.8981,0.8907,0.8834,0.8761,
            0.8690,0.8621,0.8552,0.8486,0.8421,0.8357,0.8296,0.8236,0.8179,0.8123,
            0.8069,0.8017,0.7967,0.7918,0.7871]
}

lut = pd.DataFrame(data)

def get_k_alpha(freq, pol="H"):
    """
    Lookup k and alpha from ITU P.838-3 Table 5 (up to 50 GHz).
    If freq is not exact, it interpolates.
    
    pol: "H" (horizontal) or "V" (vertical)
    """
    # interpolate to nearest frequency
    if pol.upper() == "H":
        k = np.interp(freq, lut["Frequency_GHz"], lut["kH"])
        alpha = np.interp(freq, lut["Frequency_GHz"], lut["αH"])
    else:
        k = np.interp(freq, lut["Frequency_GHz"], lut["kV"])
        alpha = np.interp(freq, lut["Frequency_GHz"], lut["αV"])
    return k, alpha


# -----------------------------
# 2) A → gamma → Rain rate
# -----------------------------
A_COL = "A_eff_db"    # or "A_db" if you disabled WAA
GATE_BY_WET = True    # False to ignore the wet flag

def compute_rain_mm_per_hr(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # specific attenuation γ_R [dB/km]
    # (avoid divide-by-zero if PathLength has zeros)
    L = out["PathLength"].astype(float).replace(0, np.nan)
    out["gamma_R"] = out[A_COL].astype(float) / L

    # look up k, alpha row-wise and compute rain rate
    def row_to_R(row):
        # If gating by wet/dry: no rain when wet=False
        if GATE_BY_WET and (not bool(row.get("wet", False))):
            return 0.0
        gamma = row["gamma_R"]
        if not np.isfinite(gamma) or gamma <= 0:
            return 0.0
        k, a = get_k_alpha(float(row["Frequency"]), str(row["Polarization"]))
        # R = (gamma / k) ** (1/alpha)
        if k <= 0:
            return 0.0
        R = (gamma / k) ** (1.0 / a)
        # sanity cap
        if (R < 0) or (R > 3000):  # very high cap; you can tighten to 200 if you like
            return np.nan
        return float(R)

    out["R_mm_h"] = out.apply(row_to_R, axis=1)

    # 15-min accumulation (mm) if you want it right away
    # Change factor if your cadence differs
    out["acc_mm"] = out["R_mm_h"] * (15.0 / 60.0)

    return out

df_rain = compute_rain_mm_per_hr(df3)
# df_rain now has: gamma_R, R_mm_h, acc_mm

# for pid in df_qc['ID'].unique():
#     g_raw = df_rain[df_rain.ID == pid].copy()
#     # print(pid)

#     # Filter for instances where R_mm_h is greater than 0
#     # g_raw = g_raw[g_raw['R_mm_h'] > 0]

#     if g_raw[g_raw['R_mm_h'].sum() == 0].empty:
#         continue

#     # Plot only if there are non-zero rainfall values
#     if not g_raw.empty:
#         plt.figure(figsize=(12, 4))
#         plt.plot(g_raw['timestamp'], g_raw['R_mm_h'], label='R_mm_h', color='red', marker='o', markersize=3)
#         plt.title(f'Path ACC Rain for Link ID: {pid}-raw')
#         plt.xlabel('Timestamp')
#         plt.ylabel('Path ACC Rain (mm)')
#         plt.legend()
#         plt.grid()
#         plt.tight_layout()
#         plt.show()

#%% Gridding

import numpy as np
import pandas as pd

# -----------------------------
# Helpers
# -----------------------------
def _ensure_centroids(df: pd.DataFrame) -> pd.DataFrame:
    """Fill lon_c/lat_c from link endpoints if missing."""
    out = df.copy()
    if 'lon_c' not in out.columns or out['lon_c'].isna().all():
        out['lon_c'] = (out['XStart'].astype(float) + out['XEnd'].astype(float)) / 2.0
    if 'lat_c' not in out.columns or out['lat_c'].isna().all():
        out['lat_c'] = (out['YStart'].astype(float) + out['YEnd'].astype(float)) / 2.0
    return out


def _make_grid(bbox, res_deg):
    """Create lon/lat 2D grids from (min_lon,max_lon,min_lat,max_lat) and resolution in degrees."""
    min_lon, max_lon, min_lat, max_lat = bbox
    lons = np.arange(min_lon, max_lon + 1e-12, res_deg)
    lats = np.arange(min_lat, max_lat + 1e-12, res_deg)
    # meshgrid as (ny, nx)
    LON, LAT = np.meshgrid(lons, lats)
    return LON, LAT, lons, lats


# -----------------------------
# Public API: grid one timestep
# -----------------------------
def grid_one_timeslice(df_ts: pd.DataFrame,
                       bbox=None,
                       res_deg=0.05,
                       power=2.0,
                       radius_km=30.0,
                       min_pts=3,
                       value_col="R_mm_h"):
    """
    Interpolate rain rate (mm/h) for a *single* timestamp.
    Returns: dict with 'timestamp','grid','lon','lat'
    """
    df_ts = _ensure_centroids(df_ts)

    # auto-bounds from data if not supplied; pad slightly
    if bbox is None:
        min_lon = float(np.nanmin(df_ts["lon_c"])) - 0.05
        max_lon = float(np.nanmax(df_ts["lon_c"])) + 0.05
        min_lat = float(np.nanmin(df_ts["lat_c"])) - 0.05
        max_lat = float(np.nanmax(df_ts["lat_c"])) + 0.05
        bbox = (min_lon, max_lon, min_lat, max_lat)

    LON, LAT, lons, lats = _make_grid(bbox, res_deg)

    # pull valid points
    pts = df_ts[["lon_c", "lat_c", value_col]].dropna()
    z = idw_grid(
        lon_pts=pts["lon_c"].values,
        lat_pts=pts["lat_c"].values,
        values =pts[value_col].values,
        LON_grid=LON, LAT_grid=LAT,
        power=power, radius_km=radius_km, min_pts=min_pts,
        fill_value=np.nan
    )

    ts = pd.to_datetime(df_ts["timestamp"].iloc[0])
    return {"timestamp": ts, "grid": z, "lon": lons, "lat": lats}

# -----------------------------
# Batch over all 15-min steps
# -----------------------------
def grid_all_times(df: pd.DataFrame,
                   res_deg=0.05,
                   power=2.0,
                   radius_km=30.0,
                   min_pts=3,
                   value_col="R_mm_h",
                   bbox=None):
    """
    Group by timestamp and grid each 15-min slice.
    Returns: list of dicts (each like grid_one_timeslice output).
    """
    out = []
    # make sure timestamp is datetime and sorted
    dff = df.copy()
    dff["timestamp"] = pd.to_datetime(dff["timestamp"])
    for ts, g in dff.sort_values("timestamp").groupby("timestamp"):
        res = grid_one_timeslice(
            g, bbox=bbox, res_deg=res_deg,
            power=power, radius_km=radius_km,
            min_pts=min_pts, value_col=value_col
        )
        out.append(res)
    return out


#-----------------------------
# import numpy as np


# -----------------------------
import numpy as np


# -----------------------------
# Example usage
# -----------------------------
# 1) Work only with rows that actually have a finite rain rate (or keep zeros to visualize dry areas)
df_map = df_rain.copy()  # <- this is your dataframe with R_mm_h from the previous step
# Option A (recommended): keep zeros too, but drop NaN
df_map = df_map[np.isfinite(df_map["R_mm_h"])]

# 2) Define a bbox (optional). If you omit it, it's derived from data.
# Ghana-ish bounds; tweak as you like:


# assume idw_grid(...) already defined by you
idw_kwargs = dict(power=2.0, radius_km=30.0, min_pts=3, fill_value=np.nan)
grids_idw = grid_all_times_generic(df_map, grid_func=idw_grid, grid_kwargs=idw_kwargs,
                                   value_col="R_mm_h", bbox=ghana_bbox, res_deg=0.05)

# make sure kriging function from earlier is in scope: kriging_grid_ok(...)
ok_kwargs = dict(variogram_model="spherical", variogram_parameters=None, nlags=6, exact_values=False)
grids_ok = grid_all_times_generic(df_map, grid_func=kriging_grid_ok, grid_kwargs=ok_kwargs,
                                  value_col="R_mm_h", bbox=ghana_bbox, res_deg=0.05)

# gp_grid_matern(...) from above
gp_kwargs = dict(length_scale_km=50.0, nu=1.5, noise_level=0.05)
grids_gp = grid_all_times_generic(df_map, grid_func=gp_grid_matern, grid_kwargs=gp_kwargs,
                                  value_col="R_mm_h", bbox=ghana_bbox, res_deg=0.05)

# 3) Grid all timesteps
grids = grid_all_times(
    df_map,
    res_deg=0.05,      # ~5–6 km in lon; ~5.5 km in lat near the equator. Set smaller (e.g., 0.025) for finer maps.
    power=2.0,
    radius_km=40.0,    # enlarge radius if point density is low
    min_pts=3,
    value_col="R_mm_h",
    bbox=ghana_bbox
)

# 'grids' is a list of dicts:
#   item = {
#     "timestamp": pd.Timestamp,
#     "grid": 2D np.array (lat x lon),
#     "lon": 1D np.array of longitude centers,
#     "lat": 1D np.array of latitude centers
#   }
# You can visualize with matplotlib imshow / pcolormesh later, or save to NetCDF/GeoTIFF in a next step.
print(f"Generated {len(grids)} 15-min grids.")


#%% plot one example grid
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def plot_grid_discrete_map(item,vmin, vmax,
                           bins=None,
                           cmap="jet",
                           title=None,
                           savepath=None):
    z   = item["grid"]
    lon = item["lon"]
    lat = item["lat"]
    ts  = item["timestamp"]

    # mask zeros → white background
    z_masked = np.ma.masked_where(z == 0, z)

    # default bins (mm/h thresholds)
    if bins is None:
        bins = [0.1, 1, 2, 5, 10, 20, 50]  # adjustable

    # build discrete cmap and norm
    base_cmap = plt.get_cmap(cmap, len(bins)-1)
    colors = base_cmap(np.arange(len(bins)-1))
    cmap_disc = ListedColormap(colors)
    norm = BoundaryNorm(bins, cmap_disc.N, clip=True)

    # edges for pcolormesh
    lon_edges = np.r_[lon - np.diff(lon[:2])/2, lon[-1] + np.diff(lon[-2:])/2]
    lat_edges = np.r_[lat - np.diff(lat[:2])/2, lat[-1] + np.diff(lat[-2:])/2]

    # plot
    fig, ax = plt.subplots(figsize=(7, 7),
                           subplot_kw={'projection': ccrs.PlateCarree()})
    pm = ax.pcolormesh(lon_edges, lat_edges, z_masked,
                       cmap=cmap_disc, norm=norm,
                       shading="auto", transform=ccrs.PlateCarree())

    # features: coastlines, borders, gridlines
    ax.coastlines(resolution='10m', linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                      color='gray', alpha=0.5, linestyle='--')
    gl.right_labels = False
    gl.top_labels = False

    # title
    if title is None:
        title = f"Rain rate (mm/h) — {ts}"
    ax.set_title(title)

    # discrete colorbar
    cb = fig.colorbar(pm, ax=ax, fraction=0.046, pad=0.04, ticks=bins)
    cb.ax.set_yticklabels([f"{b}" for b in bins])
    cb.set_label("mm/h")

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    else:
        plt.show()

plot_grid_discrete_map(grids[25], 0, 20, bins=[0, 1.5, 3, 6, 9, 20,], title="Example Rain Rate Grid")



#%%
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import ScalarMappable

def plot_rain_rate_grid(lon, lat, rain_rate, title="Rainfall Intensity Grid"):
    """
    Plot gridded rainfall intensities with discrete colors, white for zeros.

    Parameters
    ----------
    lon, lat : 2D arrays
        Longitude and latitude grid.
    rain_rate : 2D array
        Rainfall intensity in mm/h.
    title : str
        Plot title.
    """
    projc = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(6, 7), subplot_kw={'projection': projc}, dpi=200)

    # --- Mask zeros so they appear white ---
    rain_masked = np.ma.masked_equal(rain_rate, 0.0)

    # --- Define bins (discrete thresholds for rainfall intensity) ---
    bins = [0.1, 1, 2, 5, 10, 20, 50]  # mm/h
    cmap = plt.get_cmap("jet", len(bins) - 1)
    norm = BoundaryNorm(bins, cmap.N, extend="both")

    # --- Plot rainfall grid ---
    im = ax.pcolormesh(lon, lat, rain_masked, cmap=cmap, norm=norm, shading="auto", transform=projc)

    # --- Add coastlines, borders, and gridlines ---
    ax.coastlines(resolution="10m", color="black", linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                      linestyle="--", color="gray", linewidth=0.4)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"fontsize": 8}
    gl.ylabel_style = {"fontsize": 8}

    # --- Colorbar with discrete ticks ---
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8, extend="both")
    cbar.set_label("Rainfall Intensity [mm/h]", fontsize=10)
    cbar.set_ticks(bins)
    cbar.ax.set_yticklabels([f"{b}" for b in bins])

    ax.set_title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.show()


def plot_rain_rate_grid_ghana(item,
                              vmin,
                              vmax,
                              levels=12,                    # fewer discrete steps for clarity
                              cbar_ticks=None,              # e.g. [0,2,5,10,20,50]
                              title="Rain Rate [mm/h]",
                              extent=(-5, 2.5, 4, 12),      # Ghana bounding box
                              ocean_color="silver",
                              ocean_alpha=0.5):
    """
    Plot a single gridded rainfall field over Ghana using PlateCarree projection,
    discrete 'jet' colormap, and a horizontal colorbar.

    Parameters
    ----------
    item : dict
        {"timestamp": pd.Timestamp, "grid": 2D array, "lon": 2D array, "lat": 2D array}
    vmin, vmax : float
        Color range for precipitation.
    levels : int
        Number of discrete color steps between vmin and vmax.
    cbar_ticks : list[float] or None
        Colorbar tick positions. If None, ticks are auto-generated.
    title : str
        Title string (timestamp is appended automatically).
    extent : tuple
        (lon_min, lon_max, lat_min, lat_max).
    """

    # Unpack
    z   = item["grid"]
    lon = item["lon"]
    lat = item["lat"]
    ts  = item.get("timestamp", "")

    # Mask zeros → show as white
    z_plot = np.ma.masked_where(z == 0, z)

    # Discrete colormap
    cmap   = plt.cm.jet
    levels = np.linspace(vmin, vmax, levels)
    norm   = BoundaryNorm(levels, cmap.N)

    # Setup figure
    proj   = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": proj})

    # Map extent
    ax.set_extent(extent, crs=proj)

    # Plot rainfall field
    im = ax.pcolormesh(lon, lat, z_plot,
                       cmap=cmap, norm=norm,
                       shading="auto", transform=proj)

    # Map features
    ax.coastlines(resolution="10m", color="k", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor="k")
    ax.add_feature(cfeature.OCEAN, zorder=1, color=ocean_color, alpha=ocean_alpha)

    # Gridlines
    gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False,
                      linestyle="--", color="gray", linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 10}
    gl.ylabel_style = {"size": 10}

    # Title
    ax.set_title(f"{title} ({ts})", fontsize=14)

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                      fraction=0.03, pad=0.08, extend="max")
    cb.set_label("Rainfall Intensity [mm/h]", fontsize=12)
    cb.ax.tick_params(labelsize=10)
    if cbar_ticks is not None:
        cb.set_ticks(cbar_ticks)

    plt.tight_layout()
    return fig, ax


# Example usage of plot_rain_rate_grid_ghana

# Fake test grid around Ghana
lon = np.linspace(-5, 3, 50)
lat = np.linspace(4, 12, 40)
lon2d, lat2d = np.meshgrid(lon, lat)

# Create synthetic rainfall data (Gaussian rain cell)
rain_grid = 20 * np.exp(-((lon2d - 0.5)**2 + (lat2d - 7.5)**2) / 2.0)
# Build item dict (like your 15-min snapshot)
idw_item = grids_idw[2]
ok_item = grids_ok[2]
gp_item = grids_gp[2]

gr_items = gr[5]

# Plot idw
fig, ax = plot_rain_rate_grid_ghana(
    idw_item,
    vmin=0, vmax=15,
    levels=12,                      # number of discrete colors
    cbar_ticks=np.linspace(0, 15, 7), #[0, 5, 10, 15, 20, 25],
    title="IDW-based Rainfall Intensity"
)

plt.show()

# plot ok
fig, ax = plot_rain_rate_grid_ghana(
    ok_item,
    vmin=0, vmax=15,
    levels=12,                      # number of discrete colors
    cbar_ticks=np.linspace(0, 15, 7), #[0, 5, 10, 15, 20, 25],
    title="OK-based Rainfall Intensity"
)

plt.show()

# plot gp
fig, ax = plot_rain_rate_grid_ghana(
    gp_item,
    vmin=0, vmax=8,
    levels=12,                      # number of discrete colors
    cbar_ticks=np.linspace(0, 8, 7), #[0, 5, 10, 15, 20, 25],
    title="GP-basedRainfall Intensity"
)

plt.show()


# plot gp
fig, ax = plot_rain_rate_grid_ghana(
    gr_items,
    vmin=5, vmax=13,
    levels=12,                      # number of discrete colors
    cbar_ticks=np.linspace(5, 13, 2), #[0, 5, 10, 15, 20, 25],
    title="GP-basedRainfall Intensity"
)

plt.show()