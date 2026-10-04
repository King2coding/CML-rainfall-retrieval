#%%
'''
CML2Rainfall_retrieval_20250927.py
'''

#%%
# Import packages
from CML2Rainfall_program_utils_20250927 import *


#%% Define paths and parameters
path_to_put_output = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'

META_CSV  = os.path.join(path_to_put_output, 'matched_metadata_kkk_20250527.csv')         # your meta_req
LINK_DATA = os.path.join(path_to_put_output, 'Linkdata_AT_20250824_processed_on_20250527.dat')   # your link_raw


#%% Define floating variables
cde_run_dte = datetime.datetime.today().strftime('%Y%m%d')


#%%
# ================================================================
# SECTION 1: Prepare Metadata for PyComLink
# ================================================================

# --- Load metadata ---
meta = pd.read_csv(META_CSV)  # path to matched_metadata_kkk.csv or similar

# Work on a copy
m = meta.copy()

# ------------------------------------------------
# 1. Enforce numeric types
# (frequency in MHz, coordinates in deg, length in km)
# ------------------------------------------------
for c in ["Frequency", "XStart", "YStart", "XEnd", "YEnd", "PathLength"]:
    m[c] = pd.to_numeric(m[c], errors="coerce")

# ------------------------------------------------
# 2. Build path_id
#    Combine Monitored_ID and Far_end_ID to uniquely
#    represent the directional path
# ------------------------------------------------
m["path_id"] = (
    m["Monitored_ID"].astype(str).str.strip()
    + "~~" +
    m["Far_end_ID"].astype(str).str.strip()
)

# ------------------------------------------------
# 3. Polarization: standardize to single char (H/V)
# ------------------------------------------------
m["pol"] = m["Polarization"].astype(str).str.upper().str[0]

# ------------------------------------------------
# 4. Frequency: convert MHz → GHz
# ------------------------------------------------
m["freq_GHz"] = m["Frequency"] / 1000.0

# ------------------------------------------------
# 5. Coordinates & path length
#    Assign Tx and Rx separately for clarity
# ------------------------------------------------
m["tx_lon"] = m["XStart"]
m["tx_lat"] = m["YStart"]
m["rx_lon"] = m["XEnd"]
m["rx_lat"] = m["YEnd"]
m["length_km"] = m["PathLength"]

# ------------------------------------------------
# 6. Stable link_id
#    Unique per physical path + polarization + rounded frequency
# ------------------------------------------------
m["link_id"] = (
    m["path_id"] + "|" + m["pol"] + "|" + m["freq_GHz"].round(3).astype(str)
)

# ------------------------------------------------
# 7. Final PyComLink-style metadata table
# ------------------------------------------------
meta_pc = m[[
    "link_id", "path_id", "pol", "freq_GHz",
    "tx_lat", "tx_lon", "rx_lat", "rx_lon", "length_km"
]].copy()

# ------------------------------------------------
# 8. Sanity checks
# ------------------------------------------------
print("meta_pc shape:", meta_pc.shape)
print(meta_pc.head(5))
print("Frequency range (GHz):", (meta_pc["freq_GHz"].min(), meta_pc["freq_GHz"].max()))
print("Polarizations present:", sorted(meta_pc["pol"].unique()))

#%%
# ================================================================
# SECTION 2: Prepare Raw Link Data for PyComLink
# ================================================================

# --- Load raw data ---
raw = pd.read_csv(LINK_DATA, sep=None, engine="python")  
# This is the processed "RAINLINK-style" linkdata file (.dat or .csv)

# Work on a copy
r = raw.copy()

# ------------------------------------------------
# 1. Convert DateTime string → pandas datetime
#    Input format: YYYYMMDDHHMM (e.g. 202508232330)
# ------------------------------------------------
r["time"] = pd.to_datetime(r["DateTime"].astype(str), 
                           format="%Y%m%d%H%M", utc=True)
r = r.drop(columns=["DateTime"])

# ------------------------------------------------
# 2. Build path_id from ID
#    Raw `ID` looks like: Monitored_ID >> Far_end_ID
#    → Replace >> with ~~ for consistency with metadata
# ------------------------------------------------
r["path_id"] = r["ID"].str.replace(">>", "~~", regex=False)

# ------------------------------------------------
# 3. Build link_id
#    Match the convention in Section 1:
#    path_id | pol | rounded frequency
# ------------------------------------------------
r["link_id"] = (
    r["path_id"] + "|" +
    r["Polarization"].astype(str).str.upper().str[0] + "|" +
    r["Frequency"].round(3).astype(str)
)

# ------------------------------------------------
# 4. Observed attenuation (A_obs_dB)
#    From Pmin / Pmax. Midpoint is commonly used.
# ------------------------------------------------
r["A_obs_dB"] = (r["Pmax"] + r["Pmin"]) / 2.0

# ------------------------------------------------
# 5. Frequency (already in GHz in your raw file)
#    Ensure it’s stored as numeric GHz
# ------------------------------------------------
r["freq_GHz"] = pd.to_numeric(r["Frequency"], errors="coerce")

# ------------------------------------------------
# 6. Final PyComLink-ready raw table
# ------------------------------------------------
raw_pc = r[[
    "time", "link_id", "freq_GHz", "PathLength", "A_obs_dB"
]].copy()
raw_pc.rename(columns={"PathLength": "length_km"}, inplace=True)

# ------------------------------------------------
# 7. Sanity checks
# ------------------------------------------------
print("raw_pc shape:", raw_pc.shape)
print(raw_pc.head(5))
print("Time span:", raw_pc["time"].min(), "→", raw_pc["time"].max())
print("Unique links:", raw_pc["link_id"].nunique())

# get one link to plot
example_link = raw_pc["link_id"].unique()[0]
print("Example link_id:", example_link)
example_data = raw_pc[raw_pc["link_id"] == example_link]
print(example_data.head(5))
print("Example data time span:", example_data["time"].min(), "→", example_data["time"].max())
print("Example data A_obs_dB range:", (example_data["A_obs_dB"].min(), example_data["A_obs_dB"].max()))
# ================================================================
# plot the timeseries to see its behavior


fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(example_data["time"], example_data["A_obs_dB"], ".-")
ax.set_xlabel("Time")
ax.set_ylabel("Attenuation (dB)")
ax.set_title("Example Attenuation Time Series")
ax.set_xlim(example_data["time"].min(), example_data["time"].max())
ax.set_ylim(example_data["A_obs_dB"].min(), example_data["A_obs_dB"].max())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
ax.grid(True)
plt.show()
