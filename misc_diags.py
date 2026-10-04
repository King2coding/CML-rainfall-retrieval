
#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from contextlib import redirect_stdout

def extract_polarization(x):
    if "MODU-" in str(x):
        modu_part = str(x).split("MODU-")[1]
        modu_number = modu_part[0]
        return {'1': 'V', '2': 'H'}.get(modu_number, None)
    return None

def build_monitored_id(df):
    out = df.copy()
    out["Monitored_ID"] = (
        out["NEName"].astype(str) + "-" +
        out["BrdID"].astype(str) + "-" +
        out["BrdName"].astype(str) + "-" +
        out["PortNO"].astype(str) + "(" +
        out["PortName"].astype(str) + ")-" +
        out["PathID"].astype(str)
    )
    out["Polarization"] = out["Monitored_ID"].apply(extract_polarization)
    return out

def load_metadata(metadata):
    if isinstance(metadata, pd.DataFrame):
        return metadata.copy()
    elif isinstance(metadata, str):
        return pd.read_csv(metadata)
    else:
        raise TypeError("metadata must be either a pandas DataFrame or a file path string")

def get_matched_midpoints(txt_path, metadata):
    raw = pd.read_csv(txt_path, sep="\t")
    raw = build_monitored_id(raw)
    meta = load_metadata(metadata)

    matched = pd.merge(raw, meta, on=["Monitored_ID"], how="inner")

    coord_cols = {"XStart", "YStart", "XEnd", "YEnd"}
    if not coord_cols.issubset(matched.columns):
        raise ValueError("Metadata must contain XStart, YStart, XEnd, YEnd")

    pts = matched[["Monitored_ID", "XStart", "YStart", "XEnd", "YEnd"]].drop_duplicates().copy()

    for c in ["XStart", "YStart", "XEnd", "YEnd"]:
        pts[c] = pd.to_numeric(pts[c], errors="coerce")

    pts["lon_mid"] = 0.5 * (pts["XStart"] + pts["XEnd"])
    pts["lat_mid"] = 0.5 * (pts["YStart"] + pts["YEnd"])
    pts = pts.dropna(subset=["lon_mid", "lat_mid"]).copy()

    return pts

def diagnose_file(txt_path, metadata, label="FILE"):
    raw = pd.read_csv(txt_path, sep="\t")
    raw = build_monitored_id(raw)
    meta = load_metadata(metadata)

    raw_ids = set(raw["Monitored_ID"].dropna().unique())

    matched = pd.merge(raw, meta, on=["Monitored_ID"], how="inner")
    matched_ids = set(matched["Monitored_ID"].dropna().unique())
    unmatched_ids = raw_ids - matched_ids

    print("=" * 80)
    print(f"{label}: {txt_path}")
    print("=" * 80)
    print(f"Raw rows: {len(raw):,}")
    print(f"Raw unique NEName: {raw['NEName'].nunique():,}")
    print(f"Raw unique Monitored_ID: {len(raw_ids):,}")
    print(f"Matched rows: {len(matched):,}")
    print(f"Matched unique Monitored_ID: {len(matched_ids):,}")
    print(f"Unmatched unique Monitored_ID: {len(unmatched_ids):,}")

    if len(raw_ids) > 0:
        frac = 100 * len(matched_ids) / len(raw_ids)
        print(f"Metadata match rate: {frac:.2f}%")

    print("\nTop 50 unmatched Monitored_ID examples:")
    for x in sorted(list(unmatched_ids))[:50]:
        print(f"  {x}")

    coord_cols = {"XStart", "YStart", "XEnd", "YEnd"}
    if coord_cols.issubset(matched.columns):
        m = matched[["Monitored_ID", "XStart", "YStart", "XEnd", "YEnd"]].drop_duplicates().copy()
        for c in ["XStart", "YStart", "XEnd", "YEnd"]:
            m[c] = pd.to_numeric(m[c], errors="coerce")

        m["lon_mid"] = 0.5 * (m["XStart"] + m["XEnd"])
        m["lat_mid"] = 0.5 * (m["YStart"] + m["YEnd"])
        m = m.dropna(subset=["lon_mid", "lat_mid"])

        if not m.empty:
            print("\nMatched-link spatial extent:")
            print(f"  lon: {m['lon_mid'].min():.4f} to {m['lon_mid'].max():.4f}")
            print(f"  lat: {m['lat_mid'].min():.4f} to {m['lat_mid'].max():.4f}")
            print(f"  Unique matched links with coords: {m['Monitored_ID'].nunique():,}")
        else:
            print("\nMatched-link spatial extent: no valid coordinates found.")
    else:
        print("\nMatched-link spatial extent: required coordinate columns not found in metadata.")

    print("\n")

def plot_old_new_midpoints(old_txt_path, new_txt_path, metadata, out_png_path,
                           ghana_bbox=(-3.5, 1.5, 4.5, 11.5)):
    old_pts = get_matched_midpoints(old_txt_path, metadata)
    new_pts = get_matched_midpoints(new_txt_path, metadata)

    xmin, xmax, ymin, ymax = ghana_bbox

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # Panel 1: old
    axes[0].scatter(old_pts["lon_mid"], old_pts["lat_mid"], s=12, alpha=0.8)
    axes[0].set_title(f"Old matched CML midpoints\nN={old_pts['Monitored_ID'].nunique()}")
    axes[0].set_xlim(xmin, xmax)
    axes[0].set_ylim(ymin, ymax)
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    axes[0].grid(True, alpha=0.3)

    # Panel 2: new
    axes[1].scatter(new_pts["lon_mid"], new_pts["lat_mid"], s=12, alpha=0.8)
    axes[1].set_title(f"New matched CML midpoints\nN={new_pts['Monitored_ID'].nunique()}")
    axes[1].set_xlim(xmin, xmax)
    axes[1].set_ylim(ymin, ymax)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: overlay
    axes[2].scatter(old_pts["lon_mid"], old_pts["lat_mid"], s=12, alpha=0.5, label="Old")
    axes[2].scatter(new_pts["lon_mid"], new_pts["lat_mid"], s=12, alpha=0.5, label="New")
    axes[2].set_title("Overlay of matched CML midpoints")
    axes[2].set_xlim(xmin, xmax)
    axes[2].set_ylim(ymin, ymax)
    axes[2].set_xlabel("Longitude")
    axes[2].set_ylabel("Latitude")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.suptitle("Old vs New Matched CML Spatial Coverage", fontsize=14)

    out_dir = os.path.dirname(out_png_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(out_png_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return old_pts, new_pts

def compare_old_vs_new(old_txt_path, new_txt_path, metadata, out_txt_path, out_png_path=None):
    out_dir = os.path.dirname(out_txt_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_txt_path, "w", encoding="utf-8") as f:
        with redirect_stdout(f):
            print("CML RAW INPUT / METADATA COUPLING DIAGNOSTIC REPORT")
            print("\n")
            print(f"Old file     : {old_txt_path}")
            print(f"New file     : {new_txt_path}")
            print(f"Metadata src : {'DataFrame in memory' if isinstance(metadata, pd.DataFrame) else metadata}")
            print(f"Report file  : {out_txt_path}")
            if out_png_path is not None:
                print(f"Plot file    : {out_png_path}")
            print("\n")

            diagnose_file(old_txt_path, metadata, label="OLD FILE")
            diagnose_file(new_txt_path, metadata, label="NEW FILE")

            old_raw = build_monitored_id(pd.read_csv(old_txt_path, sep="\t"))
            new_raw = build_monitored_id(pd.read_csv(new_txt_path, sep="\t"))

            old_ids = set(old_raw["Monitored_ID"].dropna().unique())
            new_ids = set(new_raw["Monitored_ID"].dropna().unique())

            shared_ids = old_ids.intersection(new_ids)
            only_old = old_ids - new_ids
            only_new = new_ids - old_ids

            old_sites = set(old_raw["NEName"].dropna().unique())
            new_sites = set(new_raw["NEName"].dropna().unique())
            shared_sites = old_sites.intersection(new_sites)

            print("=" * 80)
            print("OLD VS NEW DIRECT COMPARISON")
            print("=" * 80)
            print(f"Old unique NEName: {len(old_sites):,}")
            print(f"New unique NEName: {len(new_sites):,}")
            print(f"Shared unique NEName: {len(shared_sites):,}")
            print(f"Old unique Monitored_ID: {len(old_ids):,}")
            print(f"New unique Monitored_ID: {len(new_ids):,}")
            print(f"Shared unique Monitored_ID: {len(shared_ids):,}")
            print(f"Only in old: {len(only_old):,}")
            print(f"Only in new: {len(only_new):,}")

            print("\nTop 50 Monitored_ID only in OLD:")
            for x in sorted(list(only_old))[:50]:
                print(f"  {x}")

            print("\nTop 50 Monitored_ID only in NEW:")
            for x in sorted(list(only_new))[:50]:
                print(f"  {x}")

    print(f"Diagnostic report saved to:\n{out_txt_path}")

    if out_png_path is not None:
        plot_old_new_midpoints(old_txt_path, new_txt_path, metadata, out_png_path)
        print(f"Coverage plot saved to:\n{out_png_path}")
#%%
# ------------------------------------------------------------------
# EXAMPLE USAGE
# ------------------------------------------------------------------
metadata_dir = r'/home/kkumah/Projects/cml-stuff/data-cml/outs'
diags_path = r'/home/kkumah/Projects/cml-stuff/misc'

matched_metadata = pd.read_csv(
    os.path.join(metadata_dir, 'matched_metadata_kkk_20250527.csv')
)

old_file = os.path.join(diags_path, "Schedule_pfm_SDH_20250919032247281472860733888_1.txt")
new_file = os.path.join(diags_path, "Schedule_pfm_SDH_20260326121152281472355889600_1.txt")

out_txt_path = os.path.join(diags_path, "cml_diagnostic_report.txt")
out_png_path = os.path.join(diags_path, "cml_old_vs_new_coverage.png")

compare_old_vs_new(
    old_file,
    new_file,
    matched_metadata,
    out_txt_path,
    out_png_path=out_png_path
)