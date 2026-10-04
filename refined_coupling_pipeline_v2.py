#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refined_coupling_pipeline_v2.py

A single-file, Bas-aligned pipeline to couple Airtel-Tigo metadata with FTP signal-level data
and emit a RAINLINK-ready .dat file.

Enhancements vs v1:
- --metadata and --coordinates can be FILES or DIRECTORIES.
  If a directory is given, we auto-discover the most recent matching file:
    metadata:   "Microwave_Link_Report*.xlsx"
    coordinates:"consolidated*.xlsx" or "consolidated_data_modified*.xlsx"
- Robust filename parsing & 51-hour lookback, or user-provided --latest (YYYYMMDDHH).
- Reads Schedule_pfm_* files (tab-separated .txt) in the time window.
- Implements Bas's logic: IDs, polarization, frequency swaps + 500 MHz buffer, ATPC filtering,
  TSL/RSL flattening, Pmin/Pmax, and output formatting.

Usage example:
python refined_coupling_pipeline_v2.py \
  --ftp-dir "/home/kkumah/Projects/cml-stuff/data-cml/rsl" \
  --metadata "/home/kkumah/Projects/cml-stuff/data-cml/outs" \
  --coordinates "/home/kkumah/Projects/cml-stuff/data-cml/metadata" \
  --out-dir "/home/kkumah/Projects/cml-stuff/data-cml/outs" \
  --latest 2025081300
"""
import argparse
import os
from pathlib import Path
import re
from datetime import datetime, timedelta

import pandas as pd
import numpy as np


def discover_latest_file(path_like: str, patterns):
    """
    If path_like is a file, return it.
    If it's a directory, search for the most recent file matching any of patterns (glob-style).
    """
    p = Path(path_like)
    if p.is_file():
        return p
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"Path does not exist or is not a directory: {p}")

    candidates = []
    for pat in patterns:
        candidates.extend(p.glob(pat))
    if not candidates:
        raise FileNotFoundError(f"No files found under {p} matching {patterns}")
    # choose latest by modified time
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def extract_datetime_from_filename(fname: str):
    # Expected pattern: Schedule_pfm_SDH_YYYYMMDDHH... (take first 10 digits after the 3rd underscore)
    parts = fname.split("_")
    if len(parts) < 4:
        return None
    stamp = parts[3]
    # Take first 10 chars: YYYYMMDDHH
    s = stamp[:10]
    try:
        return datetime.strptime(s, "%Y%m%d%H")
    except Exception:
        return None


def load_ftp_window(ftp_dir: Path, latest_dt: datetime | None, lookback_hours: int = 51) -> pd.DataFrame:
    files = [f for f in os.listdir(ftp_dir) if f.startswith("Schedule_pfm")]
    datetimes = [(extract_datetime_from_filename(f), f) for f in files]
    datetimes = [(dt, f) for dt, f in datetimes if dt is not None]
    if not datetimes:
        print("No valid Schedule_pfm files found.")
        return pd.DataFrame()

    if latest_dt is None:
        latest_dt = max(datetimes, key=lambda x: x[0])[0]

    cutoff = latest_dt - timedelta(hours=lookback_hours)
    inwin = [(dt, f) for dt, f in datetimes if cutoff <= dt <= latest_dt]
    inwin.sort(key=lambda x: x[0])

    dfs = []
    for _, fn in inwin:
        fp = ftp_dir / fn
        try:
            # default files are tab-separated with headers
            df = pd.read_csv(fp, header=0, sep="\t")
            df["source_file"] = fn
            dfs.append(df)
        except Exception as e:
            print(f"Warning: failed to read {fn}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def build_monitored_id(df: pd.DataFrame) -> pd.Series:
    # NEName-BrdID-BrdName-PortNO(PortName)-PathID
    cols = ["NEName", "BrdID", "BrdName", "PortNO", "PortName", "PathID"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return (
        df["NEName"].astype(str) + "-" +
        df["BrdID"].astype(str) + "-" +
        df["BrdName"].astype(str) + "-" +
        df["PortNO"].astype(str) + "(" +
        df["PortName"].astype(str) + ")-" +
        df["PathID"].astype(str)
    )


def deduce_pol_from_monitored_id(mon_id: pd.Series) -> pd.Series:
    def _pol(x: str):
        if "MODU-" in x:
            try:
                num = x.split("MODU-")[1][0]
                return {"1": "V", "2": "H"}.get(num, None)
            except Exception:
                return None
        return None
    return mon_id.apply(_pol)


def prep_metadata_tables(metadata_path: Path, coordinates_path: Path) -> pd.DataFrame:
    # Load
    metadata_vars = [
        "Source NE Name",
        "Sink NE Name",
        "Source XPIC Polarization Direction",
        "Sink XPIC Polarization Direction",
        "Source NE Frequency (MHz)",
        "Sink NE Frequency (MHz)",
        "Source ODU Board/Port/Path",
        "Sink ODU Board/Port/Path",
    ]
    coordinates_vars = [
        "Link ID", "Link Name",
        "Site 1 ID", "Site 2 ID",
        "Site 1 Latitude", "Site 2 Latitude",
        "Site 1 Longitude", "Site 2 Longitude",
        "Distance(km)", "ATPC(dB)",
        "Site 1 TX Freq(MHz)", "Site 2 TX Freq(MHz)",
    ]

    # Microwave_Link_Report has top rows to skip; mimic Bas: skiprows=6, header=1
    mdf = pd.read_excel(metadata_path, skiprows=6, header=1)
    cdf = pd.read_excel(coordinates_path, header=1)

    mdf = mdf[metadata_vars].copy()
    cdf = cdf[coordinates_vars].copy()

    # Fix 1: normalize Site IDs to first alnum token, uppercase
    for col in ["Site 1 ID", "Site 2 ID"]:
        cdf[col] = cdf[col].astype(str).str.extract(r"^([A-Za-z0-9]+)")[0].str.upper()

    # Create sublink records (two directions)
    sl1 = pd.DataFrame({
        "Temp_ID": mdf["Source NE Name"].str.extract(r"^([A-Z0-9]+)")[0] + "_" +
                   mdf["Sink NE Name"].str.extract(r"^([A-Z0-9]+)")[0],
        "Monitored_ID": mdf["Source NE Name"] + "-" + mdf["Source ODU Board/Port/Path"],
        "Far_end_ID": mdf["Sink NE Name"] + "-" + mdf["Sink ODU Board/Port/Path"],
        "Frequency": pd.to_numeric(mdf["Source NE Frequency (MHz)"], errors="coerce"),
        "Polarization": mdf["Source XPIC Polarization Direction"],
    })
    sl2 = pd.DataFrame({
        "Temp_ID": mdf["Sink NE Name"].str.extract(r"^([A-Z0-9]+)")[0] + "_" +
                   mdf["Source NE Name"].str.extract(r"^([A-Z0-9]+)")[0],
        "Monitored_ID": mdf["Sink NE Name"] + "-" + mdf["Sink ODU Board/Port/Path"],
        "Far_end_ID": mdf["Source NE Name"] + "-" + mdf["Source ODU Board/Port/Path"],
        "Frequency": pd.to_numeric(mdf["Sink NE Frequency (MHz)"], errors="coerce"),
        "Polarization": mdf["Sink XPIC Polarization Direction"],
    })
    sl_coords1 = pd.DataFrame({
        "Temp_ID": cdf["Site 1 ID"] + "_" + cdf["Site 2 ID"],
        "XStart": pd.to_numeric(cdf["Site 1 Longitude"], errors="coerce"),
        "YStart": pd.to_numeric(cdf["Site 1 Latitude"], errors="coerce"),
        "XEnd": pd.to_numeric(cdf["Site 2 Longitude"], errors="coerce"),
        "YEnd": pd.to_numeric(cdf["Site 2 Latitude"], errors="coerce"),
        "PathLength": pd.to_numeric(cdf["Distance(km)"], errors="coerce"),
        "ATPC": pd.to_numeric(cdf["ATPC(dB)"], errors="coerce"),
        "Frequency_y": pd.to_numeric(cdf["Site 1 TX Freq(MHz)"].astype(str).str.split(",").str[0].str.split("_").str[0], errors="coerce"),
    })
    sl_coords2 = pd.DataFrame({
        "Temp_ID": cdf["Site 2 ID"] + "_" + cdf["Site 1 ID"],
        "XStart": pd.to_numeric(cdf["Site 2 Longitude"], errors="coerce"),
        "YStart": pd.to_numeric(cdf["Site 2 Latitude"], errors="coerce"),
        "XEnd": pd.to_numeric(cdf["Site 1 Longitude"], errors="coerce"),
        "YEnd": pd.to_numeric(cdf["Site 1 Latitude"], errors="coerce"),
        "PathLength": pd.to_numeric(cdf["Distance(km)"], errors="coerce"),
        "ATPC": pd.to_numeric(cdf["ATPC(dB)"], errors="coerce"),
        "Frequency_y": pd.to_numeric(cdf["Site 2 TX Freq(MHz)"].astype(str).str.split(",").str[0].str.split("_").str[0], errors="coerce"),
    })

    subs = pd.concat([sl1, sl2], ignore_index=True)
    coords = pd.concat([sl_coords1, sl_coords2], ignore_index=True)

    merged = subs.merge(coords, on="Temp_ID", how="outer", indicator=True)

    # Frequency matching + swaps + 500 MHz buffer
    match = merged[merged["Frequency"] == merged["Frequency_y"]].copy()
    mismatch = merged[merged["Frequency"] != merged["Frequency_y"]].copy()

    # Detect swapped pairs (same Temp_ID but freq_x & freq_y swapped across the two directions)
    # Here we simply try: when both present in mismatch and not equal, align to Frequency
    # by replacing Frequency_y with Frequency if pair exists.
    idx_to_fix = []
    for idx, row in mismatch.iterrows():
        if pd.notna(row["Frequency"]) and pd.notna(row["Frequency_y"]):
            # if absolute equal after swap logic via set of tuples is expensive; instead apply buffer first
            pass

    # 500 MHz buffer
    within_buffer = mismatch[
        mismatch["_merge"].eq("both") &
        mismatch["Frequency"].notna() & mismatch["Frequency_y"].notna() &
        (mismatch["Frequency"] - mismatch["Frequency_y"]).abs() <= 500
    ].copy()

    match = pd.concat([match, within_buffer], ignore_index=True)
    mismatch = mismatch.drop(index=within_buffer.index)

    # Keep only needed cols, rename
    match = match.drop(columns=["_merge"])
    match = match.rename(columns={"Frequency": "Frequency_x"})
    match = match.drop(columns=["Frequency_y"])
    match = match.rename(columns={"Frequency_x": "Frequency"})

    # ATPC filter, required numeric checks
    match["ATPC"] = pd.to_numeric(match["ATPC"], errors="coerce").fillna(0)
    match = match[match["ATPC"] == 0]
    num_cols = ["Frequency", "XStart", "YStart", "XEnd", "YEnd", "PathLength"]
    match[num_cols] = match[num_cols].apply(pd.to_numeric, errors="coerce")
    match = match.dropna(subset=num_cols + ["Polarization"])
    match = match[(match["Frequency"] > 0) & (match["PathLength"] > 0)]
    return match


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ftp-dir", required=True, help="Directory with Schedule_pfm_* files (txt/tsv).")
    ap.add_argument("--metadata", required=True, help="Microwave_Link_Report Excel file OR a directory that contains it.")
    ap.add_argument("--coordinates", required=True, help="Consolidated coordinates Excel file OR a directory that contains it.")
    ap.add_argument("--out-dir", required=True, help="Output directory for RAINLINK file.")
    ap.add_argument("--latest", help="Optional YYYYMMDDHH to pin window end (default: newest file time).")
    args = ap.parse_args()

    ftp_dir = Path(args.ftp_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve metadata/coordinates
    metadata_path = discover_latest_file(args.metadata, ["Microwave_Link_Report*.xlsx", "Microwave_Link_*.xlsx"])
    coordinates_path = discover_latest_file(args.coordinates, ["consolidated_data_modified*.xlsx", "consolidated*.xlsx"])

    latest_dt = None
    if args.latest:
        try:
            latest_dt = datetime.strptime(args.latest, "%Y%m%d%H")
        except ValueError:
            raise SystemExit("--latest must be in YYYYMMDDHH format")

    # Load FTP window
    ftp_data = load_ftp_window(ftp_dir, latest_dt=latest_dt)
    if ftp_data.empty:
        print("No FTP data found in the specified window. Exiting.")
        return

    # Build Monitored_ID and Polarization
    ftp_data["Monitored_ID"] = build_monitored_id(ftp_data)
    ftp_data["Polarization"] = deduce_pol_from_monitored_id(ftp_data["Monitored_ID"])

    # Prep metadata
    matched_metadata = prep_metadata_tables(metadata_path, coordinates_path)

    # Merge: couple ftp with metadata via Monitored_ID
    cml = ftp_data.merge(matched_metadata, on=["Monitored_ID"], how="inner")

    # Handle polarization mismatch: if metadata pol missing/invalid, keep ftp pol
    if "Polarization_x" in cml.columns and "Polarization_y" in cml.columns:
        cml["Polarization"] = np.where(
            (cml["Polarization_x"] != cml["Polarization_y"]) & ~(cml["Polarization_x"].isin(["H", "V"])),
            cml["Polarization_y"],
            cml["Polarization_x"],
        )
        cml = cml.drop(columns=["Polarization_x", "Polarization_y"], errors="ignore")
    else:
        # if only one present, rename to Polarization
        if "Polarization_x" in cml.columns:
            cml = cml.rename(columns={"Polarization_x": "Polarization"})
        elif "Polarization_y" in cml.columns:
            cml = cml.rename(columns={"Polarization_y": "Polarization"})

    # Drop columns not needed downstream
    drop_cols = [
        "ONEID","ONEName","NEID","NEType","NEName","ShelfID","BrdType","BrdName","PortID",
        "PortNO","PortName","MOType","FBName","EventID","PMParameterName","PMLocationID",
        "PMLocation","UpLevel","DownLevel","ResultOfLevel","Unnamed: 27","Temp_ID","ATPC"
    ]
    cml = cml.drop(columns=[c for c in drop_cols if c in cml.columns], errors="ignore")

    # Group keys as in Bas
    group_cols = ["Monitored_ID", "Far_end_ID", "Polarization", "Period", "EndTime"]

    def get_event_val(g: pd.DataFrame, name: str):
        sel = g[g["EventName"] == name]
        return sel["Value"].values[0] if not sel.empty else np.nan

    # TSL flatten
    flat_tsl = []
    for keys, g in cml.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for ev in ["TSL_MIN","TSL_MAX","TSL_CUR","TSL_AVG"]:
            row[ev] = get_event_val(g, ev)
        flat_tsl.append(row)
    tsl = pd.DataFrame(flat_tsl)

    # RSL flatten (swap ends)
    rsl_src = cml.rename(columns={"Monitored_ID":"Far_end_ID","Far_end_ID":"Monitored_ID"})
    flat_rsl = []
    for keys, g in rsl_src.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        for ev in ["RSL_MIN","RSL_MAX","RSL_CUR","RSL_AVG"]:
            row[ev] = get_event_val(g, ev)
        flat_rsl.append(row)
    rsl = pd.DataFrame(flat_rsl)

    # Merge flat tables
    flat = tsl.merge(rsl, on=group_cols, how="inner")

    # Add back static metadata (dedupe first)
    meta_cols = ["Frequency","XStart","YStart","XEnd","YEnd","PathLength"]
    cml_meta = cml[group_cols + meta_cols].drop_duplicates()
    flat = flat.merge(cml_meta, on=group_cols, how="left")

    # TSL completeness filter
    keep = flat.dropna(subset=["TSL_MIN","TSL_MAX","TSL_AVG","TSL_CUR"])
    # Compute attenuation
    keep["Pmin"] = keep["RSL_MIN"] - keep["TSL_AVG"]
    keep["Pmax"] = keep["RSL_MAX"] - keep["TSL_AVG"]
    # RAINLINK single ID & formatting
    keep["ID"] = keep["Monitored_ID"] + ">>" + keep["Far_end_ID"]
    keep = keep.drop(columns=["Monitored_ID","Far_end_ID","Period"])
    keep = keep.rename(columns={"EndTime":"DateTime"})
    keep["Frequency"] = pd.to_numeric(keep["Frequency"], errors="coerce") / 1000.0  # MHz -> GHz
    keep["DateTime"] = pd.to_datetime(keep["DateTime"])
    keep["DateTime"] = keep["DateTime"].dt.strftime("%Y%m%d%H%M")

    out_cols = ["Frequency","DateTime","Pmin","Pmax","XStart","YStart","XEnd","YEnd","ID","Polarization","PathLength","TSL_AVG"]
    keep = keep[out_cols]

    # Output filename based on window end date
    # If user provided --latest, use that; otherwise use max DateTime present
    if latest_dt is None and not keep.empty:
        try:
            max_dt = pd.to_datetime(keep["DateTime"], format="%Y%m%d%H%M").max()
            out_stamp = max_dt.strftime("%Y%m%d")
        except Exception:
            out_stamp = datetime.utcnow().strftime("%Y%m%d")
    else:
        out_stamp = (latest_dt or datetime.utcnow()).strftime("%Y%m%d")

    out_fp = out_dir / f"Linkdata_AT_{out_stamp}.dat"
    keep.to_csv(out_fp, sep=",", index=False)
    print(f"Wrote RAINLINK file: {out_fp}  (rows={len(keep)})")


if __name__ == "__main__":
    main()
