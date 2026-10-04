#%%
#import packages 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from ftplib import FTP
import os
from difflib import get_close_matches
import re


#%%
# define global variables
matched_metadata = pd.read_csv(r'/home/kkumah/Projects/cml-stuff/data-cml/outs/matched_metadata_kkk_20250527.csv')
cml_data_path = r'/home/kkumah/Projects/cml-stuff/data-cml/rsl'

cml_data_files = [os.path.join(cml_data_path, f) for f in os.listdir(cml_data_path) if f.endswith(".txt")]

path_to_put_output = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'

cde_run_dte = datetime.today().strftime('%Y%m%d')#datetime(2025, 5, 27).strftime("%Y%m%d")


'''
# loading signal level data from ftp
ftp_host = '136.144.230.152'
ftp_user = os.environ.get("TAHMO_FTP_USER")
ftp_pass = os.environ.get("TAHMO_FTP_PASSWORD")

ftp = FTP(ftp_host)
ftp.login(user=ftp_user, passwd=ftp_pass)
ftp.cwd("/ATGhana")

# List files
files = sorted(ftp.nlst())
print(files)

ftp.quit()
'''
#%%

# Optional: set a manual datetime from which to count back, or None to use newest file
manual_latest_dt = None#datetime(2025,6,23,0,0) #datetime(2025, 8, 13, 0, 0)  # Example: datetime(2025, 8, 29, 12, 0)

# Function to extract datetime from filename
def extract_datetime_from_filename(fname):
    # Example: Schedule_pfm_SDH_20250812004105281472818770368_1
    parts = fname.split("_")
    if len(parts) < 4:
        return None  # unexpected filename
    
    timestamp = parts[3]  # "20250812004105281472818770368"
    try:
        # Take first 10 chars: YYYYMMDDHH
        dt = datetime.strptime(timestamp[:10], "%Y%m%d%H")
        return dt
    except ValueError:
        return None
# - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - 
# Step 1: Get file timestamps
file_names = os.listdir(cml_data_path)
file_datetimes = [(extract_datetime_from_filename(f), f) for f in file_names]
file_datetimes = [(dt, f) for dt, f in file_datetimes if dt is not None]
# - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - - - - - - - - - - -- - - - - 
# Step 2: Determine “latest” datetime
if manual_latest_dt is not None:
    latest_dt = manual_latest_dt
else:
    latest_dt, latest_file = max(file_datetimes, key=lambda x: x[0])


# Step 3: Define cutoff time
cutoff_time = latest_dt - timedelta(hours=51)

# Step 4: Filter files between cutoff_time and latest_dt and sort chronologically
recent_files = [(dt, f) for dt, f in file_datetimes if cutoff_time <= dt <= latest_dt]
recent_files.sort(key=lambda x: x[0])

# Step 5: Read and combine into one dataframe
df_list = []
for _, fname in recent_files:
    fpath = os.path.join(cml_data_path, fname)
    try:
        # df = pd.read_csv(fpath)  
        df = pd.read_csv(fpath, header=0, sep='\t') # assumes headers are present and identical
        df["source_file"] = fname   # optional: track origin file
        df_list.append(df)
    except Exception as e:
        print(f"⚠️ Could not read {fname}: {e}")

if df_list:
    ftp_data = pd.concat(df_list, ignore_index=True)
    print(f"✅ Combined dataframe shape: {ftp_data.shape} "
          f"(from {len(df_list)} files)")
else:
    ftp_data = pd.DataFrame()
    print("⚠️ No valid files found to combine.")

    #%%
# Construct an ID in such a format to be able to distinguish between sublinks, antennas and polarizations across a single path
ftp_data['Monitored_ID'] = (ftp_data['NEName'].astype(str) + '-' + 
                            ftp_data['BrdID'].astype(str) + '-' + 
                            ftp_data['BrdName'].astype(str) + '-' + 
                            ftp_data['PortNO'].astype(str) + '(' + 
                            ftp_data['PortName'].astype(str) + ')-' + 
                            ftp_data['PathID'].astype(str)
)

# We add a polarization column to the signal level data to use as a 
# double check when coupling the metadata. We know from the file *Microwave_Link_Report_* that 
# ```MODU-1(RTNRF-1)``` refers to vertically polarized signals and ```MODU-2(RTNRF-2)``` to horizontally polarized signals. 
def extract_polarization(x):
    if "MODU-" in x:
        modu_part = x.split("MODU-")[1]  
        modu_number = modu_part[0]      
        return {'1': 'V', '2': 'H'}.get(modu_number, None)
    return None

ftp_data['Polarization'] = ftp_data['Monitored_ID'].apply(extract_polarization)
# %% The coupling
# Not all polarizations between the metadata and the CML data match because in some cases the metadata misses 
# polarization data because it was not in the metadata file *Microwave_Link_Report_*.
#  However, since we can deduce the polarization from the ```Monitored_ID``` with a fair amount of certainty, 
# to minimize the number of links discarded, when the polarization is missing from the metadata we use the polarization 
# deduced in the CML data file. 

# Standardize Monitored_ID and Polarization columns before merging
# Standardize Monitored_IDs: remove all occurrences of '.0' only when they appear before '(' or at the end

def clean_monitored_id(mid):
    # Remove '.0' before '('
    mid = re.sub(r'\.0\(', '(', mid)
    # Remove '.0' at the end
    mid = re.sub(r'\.0$', '', mid)
    return mid.strip()

ftp_data['Monitored_ID'] = ftp_data['Monitored_ID'].astype(str).apply(clean_monitored_id)
#matched_metadata['Monitored_ID'] = matched_metadata['Monitored_ID'].astype(str).str.strip()
#print(ftp_data['Monitored_ID'].drop_duplicates().head(10).to_list())
#print("Sample Monitored_IDs from matched_metadata:")
#print(matched_metadata['Monitored_ID'].drop_duplicates().head(10).to_list())

# Try to find closest matches using fuzzy matching (optional, for debug)
'''
try:
    ftp_sample = ftp_data['Monitored_ID'].drop_duplicates().head(10).to_list()
    meta_sample = matched_metadata['Monitored_ID'].drop_duplicates().to_list()
    for id_ in ftp_sample:
        matches = get_close_matches(id_, meta_sample, n=3, cutoff=0.6)
        print(f"Closest matches for '{id_}': {matches}")
except ImportError:
    pass
'''
# Also standardize Polarization columns (upper-case, strip)
#ftp_data['Polarization'] = ftp_data['Polarization'].astype(str).str.upper().str.strip()
# matched_metadata['Polarization'] = matched_metadata['Polarization'].astype(str).str.upper().str.strip()

# Check for overlap before merging
# common_ids = set(ftp_data['Monitored_ID']).intersection(set(matched_metadata['Monitored_ID']))
# print(f"Number of common Monitored_IDs: {len(common_ids)}")

# Merge on both Monitored_ID and Polarization (if you want to be strict)
# cml_data = pd.merge(ftp_data, matched_metadata, on=['Monitored_ID', 'Polarization'])

# Or merge only on Monitored_ID (if Polarization may be missing or unreliable)
cml_data = pd.merge(ftp_data, matched_metadata, on=['Monitored_ID'], how='inner', suffixes=('_x', '_y'))

print(f"Merged dataframe shape: {cml_data.shape}")
print('Percentage of rows with unmatched polarizations:', 
    len(cml_data[cml_data['Polarization_x'] != cml_data['Polarization_y']]) / len(cml_data) * 100 if len(cml_data) > 0 else 0)
# Not all polarizations between the metadata and the CML data match because in 
# some cases the metadata misses polarization data because it was not in the metadata file 
# *Microwave_Link_Report_*. However, since we can deduce the polarization from the
# ```Monitored_ID``` with a fair amount of certainty, to minimize the number of links discarded, 
# when the polarization is missing from the metadata we use the polarization deduced in the CML data file. 

cml_data['Polarization'] = np.where((cml_data['Polarization_x'] != cml_data['Polarization_y']) & ~(cml_data['Polarization_x'].isin(['H', 'V'])), 
                                      cml_data['Polarization_y'],
                                      cml_data['Polarization_x'])

print(cml_data.columns.to_list())

cml_data = cml_data.drop(columns=['ONEID', 'ONEName', 'NEID', 'NEType', 
                                  'NEName', 'BrdID', 'BrdName', 'PortNO', 'PortName', 'PathID',
                                  'ShelfID', 'BrdType','PortID', 'MOType', 
                                  'FBName', 'EventID', 'PMParameterName', 'PMLocationID', 
                                  'PMLocation', 'UpLevel', 'DownLevel', 'ResultOfLevel', 
                                  'Unnamed: 27', 'Polarization_x', 'Temp_ID', 'Polarization_y', 
                                  'ATPC'])

#%%
# Columns to group by
group_columns = [
    'Monitored_ID', 'Far_end_ID', 'Polarization', 'Period', 'EndTime'
]

#########################################
def get_event_value_or_return_nan(group, event_name):
    """Get the value from the correct column based on the event name."""
    row = group[group['EventName'] == event_name]
    return row['Value'].values[0] if not row.empty else np.nan
#########################################

# TSL
df_tsl = cml_data.copy()
flattened_tsl_rows = []

for group_keys, group_data in df_tsl.groupby(group_columns):
    row = dict(zip(group_columns, group_keys))

    for event in ['TSL_MIN', 'TSL_MAX', 'TSL_CUR', 'TSL_AVG']:
        row[event] = get_event_value_or_return_nan(group_data, event)

    flattened_tsl_rows.append(row)

tsl_flattened = pd.DataFrame(flattened_tsl_rows)

# RSL
# 'swap' the Monitored and Far End IDs first
df_rsl = cml_data.copy()
df_rsl = df_rsl.rename(columns={
    'Monitored_ID': 'Far_end_ID',
    'Far_end_ID': 'Monitored_ID',
})

flattened_rsl_rows = []

for group_keys, group_data in df_rsl.groupby(group_columns):
    row = dict(zip(group_columns, group_keys))

    for event in ['RSL_MIN', 'RSL_MAX', 'RSL_CUR', 'RSL_AVG']:
        row[event] = get_event_value_or_return_nan(group_data, event)

    flattened_rsl_rows.append(row)

rsl_flattened = pd.DataFrame(flattened_rsl_rows)
#%%

# Merge the TSL and RSL dataframes
cml_data_flattened = pd.merge(tsl_flattened, rsl_flattened, on=group_columns)

# Add the other metadata columns back to the flattened dataframe
metadata_cols = ['Frequency', 'XStart', 'YStart', 'XEnd', 'YEnd', 'PathLength']
cml_data_unique_metadata = cml_data[group_columns + metadata_cols].drop_duplicates()
cml_data_flattened = pd.merge(cml_data_flattened, cml_data_unique_metadata, how='left', on=group_columns)

print('Number of sub-links for which we have (part of) the signal level data:', len(ftp_data.drop_duplicates(subset=['Monitored_ID'])))
print('Number of sub-links for which we have (part of) the signal level data and metadata:', len(cml_data.drop_duplicates(subset=['Monitored_ID', 'Far_end_ID'])))
print('Number of sub-links for which we have missing TSL data for some timestamps:', len(cml_data_flattened[cml_data_flattened[['TSL_MAX', 'TSL_CUR', 'TSL_MIN', 'TSL_AVG']].isna().all(axis=1)].drop_duplicates(subset=['Monitored_ID', 'Far_end_ID'])))
print('Number of sub-links for which we have missing RSL data for some timestamps:', len(cml_data_flattened[cml_data_flattened[['RSL_MAX', 'RSL_CUR', 'RSL_MIN', 'RSL_AVG']].isna().all(axis=1)].drop_duplicates(subset=['Monitored_ID', 'Far_end_ID'])))
print('Number of sub-links for which we have all signal level data for some timestamps:', len(cml_data_flattened.dropna().drop_duplicates(subset=['Monitored_ID', 'Far_end_ID'])))
print('Number of sub-links for which we have no TSL data for some timestamps:', cml_data_flattened.groupby(['Monitored_ID', 'Far_end_ID'])[['TSL_MIN', 'TSL_MAX', 'TSL_AVG', 'TSL_CUR']].apply(lambda g: g.isna().all().all()).sum())
print('Number of sub-links for which we have no RSL data for some timestamps:', cml_data_flattened.groupby(['Monitored_ID', 'Far_end_ID'])[['RSL_MIN', 'RSL_MAX', 'RSL_AVG', 'RSL_CUR']].apply(lambda g: g.isna().all().all()).sum())


linkdata = cml_data_flattened.copy()


#%%
### TSL filter 
# Fluctuations in TSL are generally small compared to fluctuations in RSL. 
# Nevertheless, since a fluctuation in TSL does influence the specific attenuation along the path, 
# if it is not constant, it has to be known, and if it is not known, the specific attenuation along the 
# path cannot be determined, and the sub-link can thus not be used. <br>
# Note: if fluctuations in TSL are generally small, and removing sub-links without TSL data leads to 
# high data loss, it is also possible to not discard these. Alternatively also a TSL filter could be 
# applied that only removes timestamps if the TSL values fluctuate too much 
# (<0.25 dB used in previous work from Sri Lanka).

# Additionally, a TSL(min, max, avg and cur) of -55 dBm often points to a link in "stand-by" mode. 
# What this means exactly is unclear (perhaps that there is no bandwidth across the signal?). 
#                                     Occasionally the value of -55 dBm only appears for the TSL_min, 
#                                     and not all the other variables (max/avg/cur). 
# This could possibly point to the device having gone in some form of "stand-by" mode sometime 
# in the previous 15 minutes. Despite this "stand-by" mode there is often still a signal being transmitted 
# as most of these links do record an RSL at the other end. Hence, for
# now the choice is made to keep these links with a constant TSL of -55 dBm.

# To be safe we also best drop rows with no TSL data
linkdata = linkdata.dropna(subset=['TSL_MIN', 'TSL_MAX', 'TSL_AVG', 'TSL_CUR'])
print('Number of sub-links for which we have all the required signal level data and metadata (but not necessarily for all timestamps):', len(linkdata.drop_duplicates(subset=['Monitored_ID', 'Far_end_ID'])))

print('Percentage of rows for which all TSL variables are -55:', len(linkdata[linkdata[['TSL_MIN', 'TSL_MAX', 'TSL_AVG', 'TSL_CUR']].isin([-55]).all(axis=1)]) / len(linkdata) * 100)
print('Percentage of rows for which only TSL_MIN is -55:', len(linkdata[(linkdata['TSL_MIN'] == -55) & (linkdata['TSL_MAX'] != -55) & (linkdata['TSL_AVG'] != -55) & (linkdata['TSL_CUR'] != -55)]) / len(linkdata) * 100)


#%%
# TURN THE DATA INTO RAINLINK FORMAT
# Since TSL is not constant, we will need to subtract TSL from RSL before running RAINLINK to get the actual values of attenuation along the path! Because the minimum and maximum TSL do not necessarily coincide with the minimum and maximum RSL in the previous 15 minutes, we use the average TSL to subtract 
# from the RSL, as this is the most representative for the 15 minutes. 

linkdata['Pmin'] = linkdata['RSL_MIN'] - linkdata['TSL_AVG']
linkdata['Pmax'] = linkdata['RSL_MAX'] - linkdata['TSL_AVG']

# RAINLINK only takes one ID variable. To ensure that each ID is unique we join 
# the ```Monitored_ID``` with the ```Far_end_ID``` using '>>' 
# to create a unique ID for each and every connection between two sites. 
linkdata['ID'] = linkdata['Monitored_ID'] + '>>' + linkdata['Far_end_ID']

# Show ID with any character that is not "a letter, digit, -, _, ( or )"
special_character_ids = matched_metadata.loc[matched_metadata["Monitored_ID"].str.contains(r"[^A-Za-z0-9_()-]", na=False), "Monitored_ID"].unique()
# display(special_character_ids)


# Drop rows that are not needed for RAINLINK
linkdata = linkdata.drop(columns=['Monitored_ID',
                                  'Far_end_ID',
                                  'Period'])

# change names of the remaining columns to RAINLINK format
linkdata = linkdata.rename(columns={
    'EndTime': 'DateTime'
})

# put the frequency and datetime in RAINLINK format
linkdata['Frequency'] = linkdata['Frequency'] / 1000 # convert to GHz
linkdata['DateTime'] = pd.to_datetime(linkdata['DateTime']).dt.strftime('%Y%m%d%H%M')

# and re-order columns and keep only PathLength and TSL_AVG as extra columns
order_columns = ['Frequency', 'DateTime', 'Pmin', 'Pmax', 'XStart', 'YStart', 'XEnd', 'YEnd', 'ID',  
                 'Polarization', 'PathLength', 'RSL_MIN', 'RSL_MAX', 'TSL_AVG']
linkdata = linkdata[order_columns]

print(len(linkdata['ID'].unique()))

# save to file
linkdata.to_csv(os.path.join(path_to_put_output, f'Linkdata_AT_{latest_dt.strftime("%Y%m%d")}_processed_on_{cde_run_dte}.dat'), sep=',', index=False)
# %%
