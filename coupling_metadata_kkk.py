#%%
#import packages 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import os


#%%
# LOADING METADATA
metadata_file_path = r'/home/kkumah/Projects/cml-stuff/data-cml/metadata/Microwave_Link_Report_27-05-2025_20-49-04.xlsx'
metadata_raw = pd.read_excel(metadata_file_path, skiprows=6, header=1, engine="calamine")

coordinates_file_path = r'/home/kkumah/Projects/cml-stuff/data-cml/metadata/consolidated_data_modified_bas.xlsx'
coordinates_raw = pd.read_excel(coordinates_file_path, header=1)

path_to_put_output = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'
#%%
# define global variables
cde_run_dte = datetime.today().strftime('%Y%m%d')
metadata_vars = ['Source NE Name', 
                 'Sink NE Name',
                 'Source XPIC Polarization Direction',
                 'Sink XPIC Polarization Direction',
                 'Source NE Frequency (MHz)',
                 'Sink NE Frequency (MHz)',
                 'Source ODU Board/Port/Path',
                 'Sink ODU Board/Port/Path',]

coordinates_vars = ['Link ID', 
                    'Link Name',
                    'Site 1 ID',
                    'Site 2 ID', 
                    'Site 1 Latitude',
                    'Site 2 Latitude', 
                    'Site 1 Longitude', 
                    'Site 2 Longitude',
                    'Distance(km)',
                    'ATPC(dB)', # not used explicitly in RAINLINK but important to know which links ATCP is applied on
                    'Site 1 TX Freq(MHz)', # to double check the frequencies are the same in both metadata files
                    'Site 2 TX Freq(MHz)' # to double check the frequencies are the same in both metadata files
                    ]


metadata = metadata_raw[metadata_vars]
coordinates = coordinates_raw[coordinates_vars]
# %% OMBINING THE METADT FILES
# Fix inconsistency 1: 
# Capitalize Site IDs and take only the first part of the string
coordinates.loc[:, 'Site 1 ID'] = coordinates['Site 1 ID'].str.extract(r'^([A-Za-z0-9]+)')[0].str.upper() 
coordinates.loc[:, 'Site 2 ID'] = coordinates['Site 2 ID'].str.extract(r'^([A-Za-z0-9]+)')[0].str.upper()
# display(coordinates[['Site 1 ID', 'Site 2 ID']].drop_duplicates())

#%%
sublink1 = pd.DataFrame()
sublink1['Temp_ID'] = metadata['Source NE Name'].str.extract(r'^([A-Z0-9]+)') + '_' + metadata['Sink NE Name'].str.extract(r'^([A-Z0-9]+)') # dummy ID to match with coordinates
sublink1['Monitored_ID'] = metadata['Source NE Name'] + '-' + metadata['Source ODU Board/Port/Path']
sublink1['Far_end_ID'] = metadata['Sink NE Name'] + '-' + metadata['Sink ODU Board/Port/Path']
sublink1['Frequency'] = metadata['Source NE Frequency (MHz)']
sublink1['Polarization'] = metadata['Source XPIC Polarization Direction']

sublink2 = pd.DataFrame()
sublink2['Temp_ID'] = metadata['Sink NE Name'].str.extract(r'^([A-Z0-9]+)') + '_' + metadata['Source NE Name'].str.extract(r'^([A-Z0-9]+)')
sublink2['Monitored_ID'] = metadata['Sink NE Name'] + '-' + metadata['Sink ODU Board/Port/Path']
sublink2['Far_end_ID'] = metadata['Source NE Name'] + '-' + metadata['Source ODU Board/Port/Path']
sublink2['Frequency'] = metadata['Sink NE Frequency (MHz)']
sublink2['Polarization'] = metadata['Sink XPIC Polarization Direction']

sublink1_coordinates = pd.DataFrame()
sublink1_coordinates['Temp_ID'] = coordinates['Site 1 ID'] + '_' + coordinates['Site 2 ID']
sublink1_coordinates['XStart'] = coordinates['Site 1 Longitude']
sublink1_coordinates['YStart'] = coordinates['Site 1 Latitude']
sublink1_coordinates['XEnd'] = coordinates['Site 2 Longitude']
sublink1_coordinates['YEnd'] = coordinates['Site 2 Latitude']
sublink1_coordinates['PathLength'] = coordinates['Distance(km)']
sublink1_coordinates['ATPC'] = coordinates['ATPC(dB)']
sublink1_coordinates['Frequency'] = coordinates['Site 1 TX Freq(MHz)'].apply(lambda x: x.split(',')[0].split('_')[0]) # to double check the frequencies are the same in both metadata files

sublink2_coordinates = pd.DataFrame()
sublink2_coordinates['Temp_ID'] = coordinates['Site 2 ID'] + '_' + coordinates['Site 1 ID']
sublink2_coordinates['XStart'] = coordinates['Site 2 Longitude']
sublink2_coordinates['YStart'] = coordinates['Site 2 Latitude']
sublink2_coordinates['XEnd'] = coordinates['Site 1 Longitude']
sublink2_coordinates['YEnd'] = coordinates['Site 1 Latitude']
sublink2_coordinates['PathLength'] = coordinates['Distance(km)']
sublink2_coordinates['ATPC'] = coordinates['ATPC(dB)']
sublink2_coordinates['Frequency'] = coordinates['Site 2 TX Freq(MHz)'].apply(lambda x: x.split(',')[0].split('_')[0]) # to double check the frequencies are the same in both metadata files

# join the dataframes for the two sublinks
all_sublinks = pd.concat([sublink1, sublink2], ignore_index=True)
all_sublink_coordinates = pd.concat([sublink1_coordinates, sublink2_coordinates], ignore_index=True)

# enforce data types to be the same before merging
all_sublinks['Temp_ID'] = all_sublinks['Temp_ID'].astype(str)
all_sublinks['Frequency'] = pd.to_numeric(all_sublinks['Frequency'], errors='coerce')
all_sublink_coordinates['Temp_ID'] = all_sublink_coordinates['Temp_ID'].astype(str)
all_sublink_coordinates['Frequency'] = pd.to_numeric(all_sublink_coordinates['Frequency'], errors='coerce')

all_metadata = pd.merge(all_sublinks, all_sublink_coordinates, on=['Temp_ID'], how='outer', indicator=True)

# check how many rows match and mismatch in frequency
frequency_match = all_metadata[all_metadata['Frequency_x'] == all_metadata['Frequency_y']]
frequency_mismatch = all_metadata[all_metadata['Frequency_x'] != all_metadata['Frequency_y']]

# display(frequency_match)
# display(frequency_mismatch)

#%%
# Fix inconsistency 2:
# Check if there are CMLs where the frequencies between Site 1 and Site 2 are swapped
swapped_frequencies = frequency_mismatch[frequency_mismatch['_merge'] == 'both'].copy()
id_frequency_pairs = set(zip(swapped_frequencies['Temp_ID'], swapped_frequencies['Frequency_x'], swapped_frequencies['Frequency_y']))

for idx, row in swapped_frequencies.iterrows():
    swapped_pairs = (row['Temp_ID'], row['Frequency_y'], row['Frequency_x'])  # swapped frequency_y/x
    if swapped_pairs in id_frequency_pairs and row['Frequency_x'] != row['Frequency_y']:
        swapped_frequencies.at[idx, 'Frequency_y'] = row['Frequency_x']

swapped_frequencies = swapped_frequencies[swapped_frequencies['Frequency_x'] == swapped_frequencies['Frequency_y']]

# update the matched and mismatched frequencies
frequency_match = pd.concat([frequency_match, swapped_frequencies], ignore_index=True)
frequency_mismatch = frequency_mismatch.drop(swapped_frequencies.index)

#%%
# Fix inconsistency 3:
# Allow for 500 MHz difference in frequencies
frequency_buffer = frequency_mismatch[(frequency_mismatch['_merge'] == 'both') & (abs(frequency_mismatch['Frequency_x'] - frequency_mismatch['Frequency_y']) <= 500)]

# update the matched and mismatched frequencies
frequency_match = pd.concat([frequency_match, frequency_buffer], ignore_index=True)
frequency_mismatch = frequency_mismatch.drop(frequency_buffer.index)


#%%
# Drop column Frequncy_y and change name of Frequency_x to Frequency
frequency_match = frequency_match.drop(columns=['Frequency_y', '_merge'])
frequency_match = frequency_match.rename(columns={'Frequency_x': 'Frequency'})

# Summary
print('The total number of IDs in both metadata files together is:', len(all_metadata))
print('Based on matching IDs in the two metadata files, the number of available sub-links is:', len(all_metadata[all_metadata['_merge'] == 'both']))
print('Based on a more strict matching of IDs and frequencies, the number of available sub-links is:', len(frequency_match))

print('Number of sub-links with matching IDs but mismatching frequencies (excl. missing frequency data):', len(frequency_mismatch[frequency_mismatch['_merge'] == 'both'].dropna(subset=['Frequency_x', 'Frequency_y'])))

#%%
# cleaning the metadata
matched_metadata = frequency_match.copy()
# Convert ATPC to numeric, coerce errors to NaN
matched_metadata['ATPC'] = pd.to_numeric(matched_metadata['ATPC'], errors='coerce')
print('Number of sub-links with ATPC:', len(matched_metadata[matched_metadata['ATPC'] != 0.0].drop_duplicates(subset=['Monitored_ID', 'Far_end_ID'])))
matched_metadata = matched_metadata[matched_metadata['ATPC'] == 0.0]

# remove non-numeric values from numeric columns
num_cols = ['Frequency', 'XStart', 'YStart', 'XEnd', 'YEnd', 'PathLength']
matched_metadata.loc[:, num_cols] = matched_metadata[num_cols].apply(pd.to_numeric, errors='coerce')
drop_rows = matched_metadata[num_cols].isna().any(axis=1)
print(f"Drop {matched_metadata[drop_rows].index.to_list()} rows because these contain non-numeric values.")
matched_metadata = matched_metadata[~drop_rows]

# drop rows that have NaNs in required columns
matched_metadata = matched_metadata.dropna(subset=['Frequency', 'Polarization', 'XStart', 'YStart', 'XEnd', 'YEnd'])
print('Number of sub-links for which we have all the required signal level data and metadata:', len(matched_metadata.drop_duplicates(subset=['Monitored_ID', 'Far_end_ID'])))

# remove zero frequencies or path lengths
matched_metadata = matched_metadata[matched_metadata['Frequency'] > 0]
matched_metadata = matched_metadata[matched_metadata['PathLength'] > 0]

# optionally, save to file
matched_metadata.to_csv(os.path.join(path_to_put_output, f'matched_metadata_kkk_{cde_run_dte}.csv'), index=False)
# %%
