#%%
# import necessary libraries and program utils
from My_program_utils import *

#%%
# Gloabal variables
print('Starting the coupling process ...')
matched_metadata = pd.read_csv(r'/home/kkumah/Projects/cml-stuff/data-cml/outs/matched_metadata_kkk_20250527.csv')
cml_data_path = r'/home/kkumah/Projects/cml-stuff/data-cml/rsl'

cml_data_files = sorted([os.path.join(cml_data_path, f) for f in os.listdir(cml_data_path) if f.endswith(".txt")])

path_to_put_output = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'

# we want to couple all data in cml_data_files and concatenate into one dataframe and siave to disk

coupled_dat = []

for idx, f in enumerate(cml_data_files):

    if idx % 100 == 0:
        print(f'Processed {idx} files')
    
    coupled_dat.append(cml2metadata_coupling_framework(cml=f, metadat=matched_metadata))

coupled_linkdata = pd.concat(coupled_dat, ignore_index=True)

# coupled_linkdata = pd.concat([cml2metadata_coupling_framework(cml=f, 
#                                                               metadat=matched_metadata) \
#                                                               for f in cml_data_files], 
#                                                               ignore_index=True)
# save data to disk
out_fname = os.path.join(path_to_put_output, f'Multi-Link-Multi-Timestamp_coupled_linkdata_kkk_{cde_run_dte}.csv')
coupled_linkdata.to_csv(out_fname, index=False)

print('Done!')
#%%
cml = os.path.join(cml_data_path, 'Schedule_pfm_SDH_20250613095545281472135393728_1.txt')
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
cml_data = pd.merge(cml_dat_df, matched_metadata, on=['Monitored_ID'])

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
cml_data_flattened = pd.merge(tsl_flattened, rsl_flattened, on=group_columns)

# Add metadata columns back
metadata_cols = ['Frequency', 'XStart', 'YStart', 'XEnd', 'YEnd', 'PathLength']
cml_data_unique_metadata = cml_data[group_columns + metadata_cols].drop_duplicates()
cml_data_flattened = pd.merge(cml_data_flattened, cml_data_unique_metadata, how='left', on=group_columns)

# Filter and process data
linkdata = cml_data_flattened.copy()
linkdata = linkdata.dropna(subset=['TSL_MIN', 'TSL_MAX', 'TSL_AVG', 'TSL_CUR'])
linkdata['Pmin'] = linkdata['RSL_MIN'] - linkdata['TSL_AVG']
linkdata['Pmax'] = linkdata['RSL_MAX'] - linkdata['TSL_AVG']
linkdata['ID'] = linkdata['Monitored_ID'] + '>>' + linkdata['Far_end_ID']
linkdata = linkdata.drop(columns=['Monitored_ID', 'Far_end_ID', 'Period'])
linkdata = linkdata.rename(columns={'EndTime': 'DateTime'})
linkdata['Frequency'] = linkdata['Frequency'] / 1000  # convert to GHz
linkdata['DateTime'] = pd.to_datetime(linkdata['DateTime'], utc=True).dt.strftime('%Y%m%d%H%M')

# Reorder columns
order_columns = ['Frequency', 'DateTime', 'Pmin', 'Pmax', 'XStart', 'YStart', 'XEnd', 'YEnd', 'ID',
                    'Polarization', 'PathLength', 'TSL_AVG']
linkdata = linkdata[order_columns]

#%%
# some sanity check
# out_linkdata = cml2metadata_coupling_framework(cml = cml_data_files[0], 
#                                                metadat=matched_metadata)

# out_linkdata = coupled_linkdata.copy()
# # get one link file
# one_link = out_linkdata[out_linkdata['ID'] == out_linkdata['ID'].unique()[0]]
# one_link['DateTime'] = pd.to_datetime(one_link['DateTime'].astype(str), format='%Y%m%d%H%M', utc=True)
# print('Number of timestamps for this link:', len(one_link))
# # print(one_link.head())
# # print(one_link.describe())
# # print(one_link.info())

# # plot pmin and pmax separately


# fig, ax = plt.subplots(1, 2, figsize=(15, 5))
# ax[0].plot(one_link['DateTime'], one_link['Pmin'])
# ax[0].set_title('Pmin')
# ax[1].plot(one_link['DateTime'], one_link['Pmax'])
# ax[1].set_title('Pmax')
# ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
# ax[0].xaxis.set_major_locator(mdates.AutoDateLocator())
# ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
# ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
# plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45, ha="right")
# plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
# ax[0].grid(True)
# ax[1].grid(True)
# plt.show()


