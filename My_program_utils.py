#%%
# Import necessary libraries
from __future__ import annotations

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

import os


from dataclasses import dataclass
from typing import Tuple, List, Dict


#%%
# Gloabal variables
cde_run_dte = datetime.today().strftime('%Y%m%d')


#------------------------------------------------------------------------------
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


#%%
# Functions
# Function to extract datetime from filename
def extract_datetime_from_filename(fname):
    """
    Extracts a datetime object from a given filename.

    Args:
        fname (str): The filename to extract the datetime from. 
                     Expected format: "Schedule_pfm_SDH_<timestamp>_<other_info>".

    Returns:
        datetime or None: A datetime object if extraction is successful, 
                          otherwise None for unexpected filename or parsing errors.
    """
    # Split the filename into parts using "_" as the delimiter
    parts = fname.split("_")
    
    # Check if the filename has at least 4 parts to ensure it matches the expected format
    if len(parts) < 4:
        return None  # Return None for unexpected filename format
    
    # Extract the timestamp part (4th part of the filename)
    timestamp = parts[3]  # Example: "20250812004105281472818770368"
    
    try:
        # Parse the first 10 characters of the timestamp as a datetime object
        # Format: YYYYMMDDHH (Year, Month, Day, Hour)
        dt = datetime.strptime(timestamp[:10], "%Y%m%d%H")
        return dt
    except ValueError:
        # Return None if the timestamp cannot be parsed into a datetime object
        return None
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Function to extract polarization
def extract_polarization(x):
    if "MODU-" in x:
        modu_part = x.split("MODU-")[1]  
        modu_number = modu_part[0]      
        return {'1': 'V', '2': 'H'}.get(modu_number, None)
    return None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Function to get event value or return NaN
def get_event_value_or_return_nan(group, event_name):
    """Get the value from the correct column based on the event name."""
    row = group[group['EventName'] == event_name]
    return row['Value'].values[0] if not row.empty else np.nan



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Main coupling function
def cml2metadata_coupling_framework(cml, metadat):
    """
    Function to couple signal level data (CML) with metadata and return the processed data in RAINLINK format.

    Parameters:
    cml (str): Path to the signal level data file.
    metadat (pd.DataFrame): Metadata dataframe.

    Returns:
    pd.DataFrame: Coupled data in RAINLINK format.
    """
    # print(f'Processing CML file: {os.path.basename(cml)}')
    cml_dat_df = pd.read_csv(cml, header=0, sep='\t')  # assumes headers are present and identical

    # Construct an ID to distinguish between sublinks, antennas, and polarizations across a single path
    cml_dat_df['Monitored_ID'] = (cml_dat_df['NEName'].astype(str) + '-' +
                                  cml_dat_df['BrdID'].astype(str) + '-' +
                                  cml_dat_df['BrdName'].astype(str) + '-' +
                                  cml_dat_df['PortNO'].astype(str) + '(' +
                                  cml_dat_df['PortName'].astype(str) + ')-' +
                                  cml_dat_df['PathID'].astype(str))

    # Add polarization column
    cml_dat_df['Polarization'] = cml_dat_df['Monitored_ID'].apply(extract_polarization)

    # Merge CML data with metadata
    cml_data = pd.merge(cml_dat_df, metadat, on=['Monitored_ID'], how='inner')

    # Check if the merge resulted in an empty dataframe
    if cml_data.empty:
        error_message = f"No common 'Monitored_ID' found for file: {os.path.basename(cml)}"
        print(error_message)
        log_path = f"/home/kkumah/Projects/cml-stuff/data-cml/metadata/unmatched_cml_files_{cde_run_dte}.txt"
        with open(log_path, "a") as f:
            f.write(error_message + "\n")
        return None  # Return None to indicate skipping further processing

    # Handle unmatched polarizations
    cml_data['Polarization'] = np.where((cml_data['Polarization_x'] != cml_data['Polarization_y']) &
                                        ~(cml_data['Polarization_x'].isin(['H', 'V'])),
                                        cml_data['Polarization_y'],
                                        cml_data['Polarization_x'])

    # Drop unnecessary columns
    cml_data = cml_data.drop(columns=['ONEID', 'ONEName', 'NEID', 'NEType',
                                      'NEName', 'BrdID', 'BrdName', 'PortNO', 'PortName', 'PathID',
                                      'ShelfID', 'BrdType', 'PortID', 'MOType',
                                      'FBName', 'EventID', 'PMParameterName', 'PMLocationID',
                                      'PMLocation', 'UpLevel', 'DownLevel', 'ResultOfLevel',
                                      'Unnamed: 27', 'Polarization_x', 'Temp_ID', 'Polarization_y',
                                      'ATPC'])

    # Columns to group by
    group_columns = ['Monitored_ID', 'Far_end_ID', 'Polarization', 'Period', 'EndTime']

    # Process TSL data
    df_tsl = cml_data.copy()
    flattened_tsl_rows = []
    for group_keys, group_data in df_tsl.groupby(group_columns):
        row = dict(zip(group_columns, group_keys))
        for event in ['TSL_MIN', 'TSL_MAX', 'TSL_CUR', 'TSL_AVG']:
            row[event] = get_event_value_or_return_nan(group_data, event)
        flattened_tsl_rows.append(row)
    tsl_flattened = pd.DataFrame(flattened_tsl_rows)

    # Process RSL data
    df_rsl = cml_data.copy()
    df_rsl = df_rsl.rename(columns={'Monitored_ID': 'Far_end_ID', 'Far_end_ID': 'Monitored_ID'})
    flattened_rsl_rows = []
    for group_keys, group_data in df_rsl.groupby(group_columns):
        row = dict(zip(group_columns, group_keys))
        for event in ['RSL_MIN', 'RSL_MAX', 'RSL_CUR', 'RSL_AVG']:
            row[event] = get_event_value_or_return_nan(group_data, event)
        flattened_rsl_rows.append(row)
    rsl_flattened = pd.DataFrame(flattened_rsl_rows)

    # Merge TSL and RSL dataframes
    if tsl_flattened.empty or rsl_flattened.empty:
        error_message = f"Skipping file {os.path.basename(cml)} due to insufficient data after processing."
        print(error_message)
        log_path = f"/home/kkumah/Projects/cml-stuff/data-cml/metadata/insufficient_data_{cde_run_dte}.txt"
        with open(log_path, "a") as f:
            f.write(error_message + "\n")
        return None  # Skip further processing if any intermediate result is empty

    cml_data_flattened = pd.merge(tsl_flattened, rsl_flattened, on=group_columns)

    # Add metadata columns back
    metadata_cols = ['Frequency', 'XStart', 'YStart', 'XEnd', 'YEnd', 'PathLength']
    cml_data_unique_metadata = cml_data[group_columns + metadata_cols].drop_duplicates()
    cml_data_flattened = pd.merge(cml_data_flattened, cml_data_unique_metadata, how='left', on=group_columns)

    # Filter and process data
    linkdata = cml_data_flattened.copy()
    linkdata = linkdata.dropna(subset=['TSL_MIN', 'TSL_MAX', 'TSL_AVG', 'TSL_CUR'])
    try:
        linkdata['RSL_MIN'] = pd.to_numeric(linkdata['RSL_MIN'], errors='coerce')
        linkdata['TSL_AVG'] = pd.to_numeric(linkdata['TSL_AVG'], errors='coerce')
        linkdata['RSL_MAX'] = pd.to_numeric(linkdata['RSL_MAX'], errors='coerce')
        
        linkdata['Pmin'] = linkdata['RSL_MIN'] - linkdata['TSL_AVG']
        linkdata['Pmax'] = linkdata['RSL_MAX'] - linkdata['TSL_AVG']
    except Exception as e:
        error_message = f"Error processing Pmin/Pmax of file {os.path.basename(cml)}: {e}"
        print(error_message)
        log_path = f"/home/kkumah/Projects/cml-stuff/data-cml/metadata/error_log_{cde_run_dte}.txt"
        with open(log_path, "a") as f:
            f.write(error_message + "\n")
        return None
    linkdata['ID'] = linkdata['Monitored_ID'] + '>>' + linkdata['Far_end_ID']
    linkdata = linkdata.drop(columns=['Monitored_ID', 'Far_end_ID', 'Period'])
    linkdata = linkdata.rename(columns={'EndTime': 'DateTime'})
    linkdata['Frequency'] = linkdata['Frequency'] / 1000  # convert to GHz
    linkdata['DateTime'] = pd.to_datetime(linkdata['DateTime'], utc=True).dt.strftime('%Y%m%d%H%M')

    # Reorder columns
    order_columns = ['Frequency', 'DateTime', 'Pmin', 'Pmax', 'XStart', 'YStart', 'XEnd', 'YEnd', 'ID',
                     'Polarization', 'PathLength', 'TSL_AVG']
    linkdata = linkdata[order_columns]

    return linkdata

# - - - - - - - - - - - - - - - - - - -v - - - - - - - - - - - - - - - - 


@dataclass(frozen=True)
class R0AutoConfig:
    # Time
    cadence_minutes: int = 15
    snap_tolerance: str = "2min"       # snap timestamps to nearest 15 min if within this tolerance
    regularize_grid: bool = True       # reindex to exact 15-min grid (gaps -> NaN)
    source_tz: str = "Africa/Accra"    # Ghana is UTC year-round

    # Bounds for RSL (dBm) and TL (dB)
    rsl_min_dbm: float = -130.0
    rsl_max_dbm: float = -20.0
    tl_min_db: float = 0.0
    tl_max_db: float = 80.0

    # Within-bin dynamic range for 15-min data
    max_dyn_range_db: float = 12.0

    # Outage heuristics (different for RSL vs TL)
    rsl_outage_floor_dbm: float = -115.0   # near receiver floor
    tl_outage_high_db: float = 75.0        # near saturation
    outage_min_consec: int = 2             # require consecutive steps

    # Hampel (spike) detection on Pbar = (Pmin+Pmax)/2
    hampel_window: int = 5
    hampel_nsigma: float = 5.0

    # Plateau (frozen sensor) detection: flag only (do NOT drop)
    plateau_run_len: int = 16              # 4 h at 15-min cadence
    plateau_tol_db: float = 0.02           # variation tolerated to call it flat

    # Force semantics? ("auto", "rsl", "tl")
    semantics: str = "auto"


# ---------------------- helpers ----------------------
def _parse_dt(series: pd.Series, cfg: R0AutoConfig) -> pd.DatetimeIndex:
    dt = pd.to_datetime(series.astype(str), format="%Y%m%d%H%M", errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize(cfg.source_tz, nonexistent="shift_forward", ambiguous="NaT")
    return dt.dt.tz_convert("UTC")


def _snap_to_grid(dt: pd.Series, minutes: int, tol: pd.Timedelta) -> pd.Series:
    base = dt.dt.floor(f"{minutes}min")
    offs = (dt - base)
    up = offs >= pd.Timedelta(minutes=minutes/2)
    anchor = base.where(~up, base + pd.Timedelta(minutes=minutes))
    diff = (dt - anchor).abs()
    return anchor.where(diff <= tol)  # otherwise NaT


def _hampel_mask(x: pd.Series, window: int, nsigma: float) -> pd.Series:
    med = x.rolling(window, center=True, min_periods=3).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=3).median()
    sigma = 1.4826 * mad
    return ((x - med).abs() > nsigma * sigma).fillna(False)


def _flag_plateaus(x: pd.Series, run_len: int, tol_db: float) -> pd.Series:
    if x.isna().all():
        return pd.Series(False, index=x.index)
    d = x.diff().abs().fillna(0.0) <= tol_db
    gid = (~d).cumsum()
    counts = pd.Series(gid).map(pd.Series(gid).value_counts())
    return (d & (counts.values >= run_len)).reindex_like(x).fillna(False)


def _consec_true(mask: pd.Series, min_len: int) -> pd.Series:
    if mask.empty:
        return mask
    gid = (mask != mask.shift(1, fill_value=False)).cumsum()
    run_len = gid.map(gid.value_counts())
    return mask & (run_len >= min_len)


def _detect_semantics(pmin: pd.Series, pmax: pd.Series, cfg: R0AutoConfig) -> str:
    # Use medians, ignoring NaNs
    med_min = pmin.median(skipna=True)
    med_max = pmax.median(skipna=True)
    # If both look RSL
    if (cfg.rsl_min_dbm <= med_min <= cfg.rsl_max_dbm) and (cfg.rsl_min_dbm <= med_max <= cfg.rsl_max_dbm):
        return "rsl"
    # If both look TL
    if (cfg.tl_min_db <= med_min <= cfg.tl_max_db) and (cfg.tl_min_db <= med_max <= cfg.tl_max_db):
        return "tl"
    # Fall back: if most finite samples are negative, call it RSL; else TL
    finite = pd.concat([pmin, pmax], axis=1).dropna().values.ravel()
    if finite.size == 0:
        return "rsl"  # harmless default; will be NaN anyway
    return "rsl" if (np.nanmedian(finite) < 0) else "tl"


# ---------------------- core cleaner ----------------------
def clean_minmax_auto(df_raw: pd.DataFrame, cfg: R0AutoConfig = R0AutoConfig()
                      ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean 15-min Pmin/Pmax per link (ID) with per-link semantics detection (RSL vs TL).
    Required columns: ['ID','DateTime','Pmin','Pmax']
    Other columns are passed through untouched.

    Returns
    -------
    df_out : per-timestamp rows with cleaned Pmin/Pmax, Pbar, Pspread, qc_flags, semantics
    df_summary : per-ID summary stats
    """
    need = ["ID", "DateTime", "Pmin", "Pmax"]
    missing = [c for c in need if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df_raw.copy()

    # 1) Parse & snap time to 15-min
    dt = _parse_dt(df["DateTime"], cfg)
    snap = _snap_to_grid(dt, cfg.cadence_minutes, pd.Timedelta(cfg.snap_tolerance))
    df["DateTimeUTC"] = snap
    df = df.dropna(subset=["DateTimeUTC"]).sort_values(["ID", "DateTimeUTC"])
    df = df.drop_duplicates(subset=["ID", "DateTimeUTC"], keep="first")
    df.set_index("DateTimeUTC", inplace=True)

    # 2) Numeric Pmin/Pmax
    df["Pmin"] = pd.to_numeric(df["Pmin"], errors="coerce")
    df["Pmax"] = pd.to_numeric(df["Pmax"], errors="coerce")

    out_frames: List[pd.DataFrame] = []
    summaries: List[Dict] = []

    for link_id, g in df.groupby("ID", sort=False):
        s = g.copy()

        # Decide semantics
        semantics = cfg.semantics if cfg.semantics in ("rsl", "tl") else _detect_semantics(s["Pmin"], s["Pmax"], cfg)

        # Init QC flags
        flags = {k: np.zeros(len(s), dtype=bool) for k in [
            "OOB", "SWAP", "DYN", "OUTAGE", "SPIKE", "PLAT", "SEM_RSL", "SEM_TL"
        ]}
        flags["SEM_RSL"][:] = (semantics == "rsl")
        flags["SEM_TL"][:] = (semantics == "tl")

        # 3) Bounds by semantics
        if semantics == "rsl":
            bad_min = ~s["Pmin"].between(cfg.rsl_min_dbm, cfg.rsl_max_dbm, inclusive="both")
            bad_max = ~s["Pmax"].between(cfg.rsl_min_dbm, cfg.rsl_max_dbm, inclusive="both")
        else:  # tl
            bad_min = ~s["Pmin"].between(cfg.tl_min_db, cfg.tl_max_db, inclusive="both")
            bad_max = ~s["Pmax"].between(cfg.tl_min_db, cfg.tl_max_db, inclusive="both")
        oob = bad_min | bad_max
        flags["OOB"] = oob.values
        s.loc[oob, ["Pmin", "Pmax"]] = np.nan

        # 4) Order: expect Pmax >= Pmin
        need_swap = (s["Pmin"].notna() & s["Pmax"].notna() & (s["Pmin"] > s["Pmax"]))
        s.loc[need_swap, ["Pmin", "Pmax"]] = s.loc[need_swap, ["Pmax", "Pmin"]].values
        flags["SWAP"] = need_swap.values

        # 5) Within-bin dynamic range
        spread = s["Pmax"] - s["Pmin"]
        dyn_bad = (spread > cfg.max_dyn_range_db)
        flags["DYN"] = dyn_bad.fillna(False).values
        s.loc[dyn_bad, ["Pmin", "Pmax"]] = np.nan

        # 6) Outage heuristics (semantics-aware)
        if semantics == "rsl":
            low_min = s["Pmin"] <= cfg.rsl_outage_floor_dbm
            low_max = s["Pmax"] <= cfg.rsl_outage_floor_dbm
            outage = _consec_true(low_min | low_max, cfg.outage_min_consec)
        else:
            high_min = s["Pmin"] >= cfg.tl_outage_high_db
            high_max = s["Pmax"] >= cfg.tl_outage_high_db
            outage = _consec_true(high_min | high_max, cfg.outage_min_consec)
        flags["OUTAGE"] = outage.values
        s.loc[outage, ["Pmin", "Pmax"]] = np.nan

        # 7) Hampel spikes on Pbar
        pbar = (s["Pmin"] + s["Pmax"]) / 2.0
        spike = _hampel_mask(pbar, cfg.hampel_window, cfg.hampel_nsigma)
        flags["SPIKE"] = spike.values
        s.loc[spike, ["Pmin", "Pmax"]] = np.nan

        # 8) Plateau — flag only
        plat = _flag_plateaus(pbar, cfg.plateau_run_len, cfg.plateau_tol_db)
        flags["PLAT"] = plat.values

        # 9) Regularize to exact 15-min grid
        if cfg.regularize_grid and not s.empty:
            full_idx = pd.date_range(s.index.min(), s.index.max(),
                                     freq=f"{cfg.cadence_minutes}min", tz="UTC")
            s = s.reindex(full_idx)

        # 10) QC string & convenience fields
        def pack(i: int) -> str:
            labs = [k for k, v in flags.items() if i < len(v) and v[i]]
            return ",".join(labs) if labs else ""

        qc_series = pd.Series([pack(i) for i in range(len(g))], index=g.index)
        s["qc_flags"] = qc_series.reindex(s.index).fillna("")
        s["Pbar"] = (s["Pmin"] + s["Pmax"]) / 2.0
        s["Pspread"] = s["Pmax"] - s["Pmin"]
        s["ID"] = link_id
        s["semantics"] = semantics

        # Summary
        summaries.append({
            "ID": link_id,
            "semantics": semantics,
            "n_rows": len(s),
            "n_valid_pairs": s[["Pmin", "Pmax"]].dropna().shape[0],
            "n_swap": int(need_swap.sum()),
            "n_oob": int(flags["OOB"].sum()),
            "n_dyn": int(flags["DYN"].sum()),
            "n_outage": int(flags["OUTAGE"].sum()),
            "n_spike": int(flags["SPIKE"].sum()),
            "frac_plateau_flag": float((s["qc_flags"].str.contains("PLAT")).mean()),
        })

        out_frames.append(s)

    df_out = pd.concat(out_frames, axis=0).sort_index()
    df_summary = pd.DataFrame(summaries).sort_values(["ID"]).reset_index(drop=True)
    return df_out, df_summary

#%%
# import requests
# from requests.auth import HTTPBasicAuth

# CONSUMER_KEY = "your_consumer_key"
# CONSUMER_SECRET = "your_consumer_secret"

# def get_eumetsat_token():
#     url = "https://api.eumetsat.int/token"
#     response = requests.post(
#         url,
#         data={"grant_type": "client_credentials"},
#         auth=HTTPBasicAuth(CONSUMER_KEY, CONSUMER_SECRET),
#     )
#     response.raise_for_status()
#     token = response.json()["access_token"]
#     print("Token retrieved successfully.")
#     return token

# # Example
# token = get_eumetsat_token()
# headers = {"Authorization": f"Bearer {token}"}




