
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import os
from typing import Optional, Tuple, List

# -----------------------------
# Helpers
# -----------------------------
def _extract_caps_prefix(s: pd.Series) -> pd.Series:
    """Take first alnum token and uppercase."""
    return s.astype(str).str.extract(r'^([A-Za-z0-9]+)')[0].str.upper()

def _latest_from_ftp(ftp_dir: Path, manual_latest_dt: Optional[datetime]) -> Tuple[Optional[datetime], List[Tuple[datetime,str]]]:
    def extract_dt(fname: str) -> Optional[datetime]:
        # Schedule_pfm_SDH_2025081200410528147..._1.xlsx
        parts = fname.split("_")
        if len(parts) < 4:
            return None
        ts = parts[3]
        try:
            return datetime.strptime(ts[:10], "%Y%m%d%H")
        except Exception:
            return None
    files = [(extract_dt(f.name), f.name) for f in ftp_dir.iterdir() if f.name.startswith("Schedule_pfm_")]
    files = [(dt, fn) for dt, fn in files if dt is not None]
    if not files and manual_latest_dt is None:
        return None, []
    if manual_latest_dt is not None:
        return manual_latest_dt, files
    latest_dt, _ = max(files, key=lambda x: x[0])
    return latest_dt, files

def _read_ftp_file(path: Path) -> Optional[pd.DataFrame]:
    # Try tab-separated text first, then CSV, then Excel
    for reader, kwargs in (
        (pd.read_csv, {"sep":"\t"}),
        (pd.read_csv, {}),
        (pd.read_excel, {}),
    ):
        try:
            df = reader(path, **kwargs)
            df["source_file"] = path.name
            return df
        except Exception:
            continue
    return None

def _ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _extract_polarization(monitored_id: str) -> Optional[str]:
    if isinstance(monitored_id, str) and "MODU-" in monitored_id:
        try:
            modu_number = monitored_id.split("MODU-")[1][0]
            return {"1":"V","2":"H"}.get(modu_number)
        except Exception:
            return None
    return None

def _flatten_events(df: pd.DataFrame, group_cols: List[str], event_prefix: str) -> pd.DataFrame:
    out_rows = []
    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for ev in [f"{event_prefix}_MIN", f"{event_prefix}_MAX", f"{event_prefix}_CUR", f"{event_prefix}_AVG"]:
            sub = grp.loc[grp["EventName"]==ev, "Value"]
            row[ev] = sub.iloc[0] if not sub.empty else np.nan
        out_rows.append(row)
    return pd.DataFrame(out_rows)

# -----------------------------
# Core pipeline (Bas logic)
# -----------------------------
def run_coupling(
    metadata_path: Path,
    coordinates_path: Path,
    ftp_dir: Path,
    out_dir: Path,
    manual_latest_dt: Optional[datetime]=None,
    lookback_hours: int = 51
) -> Path:
    metadata_path = Path(metadata_path)
    coordinates_path = Path(coordinates_path)
    ftp_dir = Path(ftp_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_vars = [
        'Source NE Name','Sink NE Name',
        'Source XPIC Polarization Direction','Sink XPIC Polarization Direction',
        'Source NE Frequency (MHz)','Sink NE Frequency (MHz)',
        'Source ODU Board/Port/Path','Sink ODU Board/Port/Path'
    ]
    coordinates_vars = [
        'Link ID','Link Name','Site 1 ID','Site 2 ID',
        'Site 1 Latitude','Site 2 Latitude','Site 1 Longitude','Site 2 Longitude',
        'Distance(km)','ATPC(dB)','Site 1 TX Freq(MHz)','Site 2 TX Freq(MHz)'
    ]

    md = pd.read_excel(metadata_path, skiprows=6, header=1)
    md = md[metadata_vars].copy()
    cr = pd.read_excel(coordinates_path, header=1)
    cr = cr[coordinates_vars].copy()

    # Clean coordinate IDs
    cr.loc[:, 'Site 1 ID'] = _extract_caps_prefix(cr['Site 1 ID'])
    cr.loc[:, 'Site 2 ID'] = _extract_caps_prefix(cr['Site 2 ID'])

    # Build sublink tables
    sub1 = pd.DataFrame({
        "Temp_ID": md['Source NE Name'].str.extract(r'^([A-Z0-9]+)')[0] + '_' + md['Sink NE Name'].str.extract(r'^([A-Z0-9]+)')[0],
        "Monitored_ID": md['Source NE Name'] + '-' + md['Source ODU Board/Port/Path'],
        "Far_end_ID": md['Sink NE Name'] + '-' + md['Sink ODU Board/Port/Path'],
        "Frequency": pd.to_numeric(md['Source NE Frequency (MHz)'], errors='coerce'),
        "Polarization": md['Source XPIC Polarization Direction']
    })
    sub2 = pd.DataFrame({
        "Temp_ID": md['Sink NE Name'].str.extract(r'^([A-Z0-9]+)')[0] + '_' + md['Source NE Name'].str.extract(r'^([A-Z0-9]+)')[0],
        "Monitored_ID": md['Sink NE Name'] + '-' + md['Sink ODU Board/Port/Path'],
        "Far_end_ID": md['Source NE Name'] + '-' + md['Source ODU Board/Port/Path'],
        "Frequency": pd.to_numeric(md['Sink NE Frequency (MHz)'], errors='coerce'),
        "Polarization": md['Sink XPIC Polarization Direction']
    })
    sub = pd.concat([sub1, sub2], ignore_index=True)

    cr1 = pd.DataFrame({
        "Temp_ID": cr['Site 1 ID'] + '_' + cr['Site 2 ID'],
        "XStart": cr['Site 1 Longitude'],
        "YStart": cr['Site 1 Latitude'],
        "XEnd": cr['Site 2 Longitude'],
        "YEnd": cr['Site 2 Latitude'],
        "PathLength": cr['Distance(km)'],
        "ATPC": cr['ATPC(dB)'],
        "Frequency": pd.to_numeric(cr['Site 1 TX Freq(MHz)'].astype(str).str.split(',').str[0].str.split('_').str[0], errors='coerce')
    })
    cr2 = pd.DataFrame({
        "Temp_ID": cr['Site 2 ID'] + '_' + cr['Site 1 ID'],
        "XStart": cr['Site 2 Longitude'],
        "YStart": cr['Site 2 Latitude'],
        "XEnd": cr['Site 1 Longitude'],
        "YEnd": cr['Site 1 Latitude'],
        "PathLength": cr['Distance(km)'],
        "ATPC": cr['ATPC(dB)'],
        "Frequency": pd.to_numeric(cr['Site 2 TX Freq(MHz)'].astype(str).str.split(',').str[0].str.split('_').str[0], errors='coerce')
    })
    cr_links = pd.concat([cr1, cr2], ignore_index=True)

    # Join (outer) and handle frequency mismatches
    all_md = pd.merge(sub, cr_links, on='Temp_ID', how='outer', indicator=True, suffixes=('_x','_y'))
    freq_match = all_md[all_md['Frequency_x'] == all_md['Frequency_y']].copy()
    freq_mismatch = all_md[all_md['Frequency_x'] != all_md['Frequency_y']].copy()

    # Frequency swap fix
    swapped = freq_mismatch[freq_mismatch['_merge'] == 'both'].copy()
    pairs = set(zip(swapped['Temp_ID'], swapped['Frequency_x'], swapped['Frequency_y']))
    for idx, row in swapped.iterrows():
        if (row['Temp_ID'], row['Frequency_y'], row['Frequency_x']) in pairs and pd.notna(row['Frequency_x']) and pd.notna(row['Frequency_y']) and row['Frequency_x'] != row['Frequency_y']:
            swapped.at[idx, 'Frequency_y'] = row['Frequency_x']
    swapped = swapped[swapped['Frequency_x'] == swapped['Frequency_y']]
    freq_match = pd.concat([freq_match, swapped], ignore_index=True)
    freq_mismatch = freq_mismatch.drop(swapped.index)

    # Frequency buffer <= 500 MHz
    buf = freq_mismatch[(freq_mismatch['_merge']=='both') & (np.abs(freq_mismatch['Frequency_x'] - freq_mismatch['Frequency_y']) <= 500)].copy()
    freq_match = pd.concat([freq_match, buf], ignore_index=True)
    freq_mismatch = freq_mismatch.drop(buf.index)

    # Finalize metadata
    freq_match = freq_match.drop(columns=['Frequency_y','_merge']).rename(columns={'Frequency_x':'Frequency'})
    # Drop ATPC != 0 and ensure numeric
    freq_match = _ensure_numeric(freq_match, ['Frequency','XStart','YStart','XEnd','YEnd','PathLength','ATPC'])
    freq_match = freq_match[(freq_match['ATPC']==0) & (freq_match['Frequency']>0) & (freq_match['PathLength']>0)]
    freq_match = freq_match.dropna(subset=['Frequency','Polarization','XStart','YStart','XEnd','YEnd'])

    matched_metadata = freq_match.copy()

    # -----------------------------
    # Load FTP data window
    # -----------------------------
    latest_dt, all_files = _latest_from_ftp(ftp_dir, manual_latest_dt)
    if latest_dt is None:
        raise FileNotFoundError("No suitable Schedule_pfm_* files found in ftp_dir and no manual_latest_dt provided.")
    cutoff = latest_dt - timedelta(hours=lookback_hours)
    window_files = [fn for dt, fn in all_files if cutoff <= dt <= latest_dt]
    window_files.sort()

    df_list = []
    for fn in window_files:
        fpath = ftp_dir / fn
        df = _read_ftp_file(fpath)
        if df is None:
            continue
        # Expect standard column names present in Airtel-Tigo dumps
        # Normalize minimal set used later
        needed = ['NEName','BrdID','BrdName','PortNO','PortName','PathID','EventName','Period','EndTime','Value']
        missing = [c for c in needed if c not in df.columns]
        if missing:
            # Some dumps use slightly different headers; skip those
            continue
        df_list.append(df[needed + ['source_file']])

    if not df_list:
        raise FileNotFoundError("No readable FTP data files in the selected time window.")

    ftp_data = pd.concat(df_list, ignore_index=True)

    # Build Monitored_ID
    ftp_data['Monitored_ID'] = (ftp_data['NEName'].astype(str) + '-' +
                                ftp_data['BrdID'].astype(str) + '-' +
                                ftp_data['BrdName'].astype(str) + '-' +
                                ftp_data['PortNO'].astype(str) + '(' +
                                ftp_data['PortName'].astype(str) + ')-' +
                                ftp_data['PathID'].astype(str))
    # Extract polarization from Monitored_ID
    ftp_data['Polarization'] = ftp_data['Monitored_ID'].apply(_extract_polarization)

    # Join with metadata on Monitored_ID
    cml = pd.merge(ftp_data, matched_metadata, on='Monitored_ID')

    # Fix polarization using FTP-derived if metadata missing/mismatch
    if 'Polarization_x' in cml.columns and 'Polarization_y' in cml.columns:
        cml['Polarization'] = np.where(
            (cml['Polarization_x'] != cml['Polarization_y']) & ~(cml['Polarization_x'].isin(['H','V'])),
            cml['Polarization_y'],
            cml['Polarization_x']
        )
        cml = cml.drop(columns=['Polarization_x','Polarization_y'])
    else:
        # if only one polarization column present, keep it as 'Polarization'
        if 'Polarization_x' in cml.columns:
            cml = cml.rename(columns={'Polarization_x':'Polarization'})
        if 'Polarization_y' in cml.columns:
            cml = cml.rename(columns={'Polarization_y':'Polarization'})

    # Keep only columns needed later
    drop_cols = ['Period','source_file']
    keep_cols = ['Monitored_ID','Far_end_ID','Polarization','EventName','EndTime','Value',
                 'Frequency','XStart','YStart','XEnd','YEnd','PathLength']
    cml = cml[[c for c in keep_cols if c in cml.columns] + [c for c in drop_cols if c in cml.columns]]

    # Split to TSL and RSL flattened
    group_cols = ['Monitored_ID','Far_end_ID','Polarization','Period','EndTime']
    # ensure Period exists for grouping (if not present, create dummy 15-min string)
    if 'Period' not in cml.columns:
        cml['Period'] = '15'
    tsl_flat = _flatten_events(cml, group_cols, 'TSL')

    # RSL requires swapping ends
    rsl_tmp = cml.rename(columns={'Monitored_ID':'Far_end_ID','Far_end_ID':'Monitored_ID'}).copy()
    rsl_flat = _flatten_events(rsl_tmp, group_cols, 'RSL')

    # Merge flat
    flat = pd.merge(tsl_flat, rsl_flat, on=group_cols, how='outer')

    # Add back unique metadata per group
    meta_cols = ['Monitored_ID','Far_end_ID','Polarization','Period','EndTime','Frequency','XStart','YStart','XEnd','YEnd','PathLength']
    uniq_meta = cml[meta_cols].drop_duplicates()
    flat = pd.merge(flat, uniq_meta, on=group_cols, how='left')

    # Drop rows without any TSL (Bas choice in notebook)
    flat = flat.dropna(subset=['TSL_MIN','TSL_MAX','TSL_AVG','TSL_CUR'], how='any')

    # Compute attenuation
    flat['Pmin'] = flat['RSL_MIN'] - flat['TSL_AVG']
    flat['Pmax'] = flat['RSL_MAX'] - flat['TSL_AVG']

    # Build RAINLINK output
    out = flat.copy()
    out['ID'] = out['Monitored_ID'] + '>>' + out['Far_end_ID']
    out = out.drop(columns=['Monitored_ID','Far_end_ID','Period'], errors='ignore')
    out = out.rename(columns={'EndTime':'DateTime'})
    out['Frequency'] = out['Frequency'] / 1000.0  # MHz -> GHz
    out['DateTime'] = pd.to_datetime(out['DateTime'], errors='coerce').dt.strftime('%Y%m%d%H%M')
    out = out[['Frequency','DateTime','Pmin','Pmax','XStart','YStart','XEnd','YEnd','ID','Polarization','PathLength','TSL_AVG']]

    # Write
    date_tag = latest_dt.strftime("%Y%m%d")
    out_file = out_dir / f"Linkdata_AT_{date_tag}.dat"
    out.to_csv(out_file, index=False)

    return out_file

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    p.add_argument("--coordinates", required=True)
    p.add_argument("--ftp-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--latest", default=None, help="YYYYMMDDHH (optional pin)")
    p.add_argument("--lookback-hours", type=int, default=51)
    args = p.parse_args()

    manual_dt = datetime.strptime(args.latest, "%Y%m%d%H") if args.latest else None
    outp = run_coupling(
        metadata_path=Path(args.metadata),
        coordinates_path=Path(args.coordinates),
        ftp_dir=Path(args.ftp_dir),
        out_dir=Path(args.out_dir),
        manual_latest_dt=manual_dt,
        lookback_hours=args.lookback_hours
    )
    print(str(outp))
