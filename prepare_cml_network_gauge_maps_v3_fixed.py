#!/usr/bin/env python3
"""
Prepare one-day Ghana CML network maps for discussion with Bas.

Outputs
-------
1. cml_network_daily_availability_<YYYY-MM-DD>.png
2. cml_network_frequency_<YYYY-MM-DD>.png
3. cml_network_length_<YYYY-MM-DD>.png
4. cml_network_three_panel_<YYYY-MM-DD>.png
5. daily_cml_link_inventory_<YYYY-MM-DD>.csv
6. daily_cml_summary_<YYYY-MM-DD>.txt

The script follows the Monitored_ID construction used in
cml_rainfall_retrieval_core_bas_202605.py, then couples the IDs observed on a
selected day to the matched metadata CSV. Opposite-direction sub-links can be
collapsed to one physical link for a clearer density map.
"""

from __future__ import annotations

import argparse
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Ghana-focused plotting extent: lon_min, lon_max, lat_min, lat_max
DEFAULT_EXTENT = (-3.5, 1.25, 4.5, 11.25)

# Length classes chosen to align with the concern raised in the email exchange.
LENGTH_BINS_KM = (0.0, 5.0, 10.0, np.inf)
LENGTH_LABELS = ("Short: <5 km", "Medium: 5–10 km", "Long: >10 km")

# Set explicit boundaries here, e.g. (0, 7, 8, np.inf), if preferred.
# None means derive low/medium/high classes from the observed unique frequencies.
FREQUENCY_BINS_GHZ: tuple[float, float, float, float] | None = None

# Matplotlib default qualitative colors, intentionally explicit for stable figures.
CATEGORY_COLORS = ("tab:blue", "tab:orange", "tab:red")
GAUGE_COLOR = "black"


def haversine_km(lon1, lat1, lon2, lat2):
    """Vectorized great-circle distance in kilometres."""
    lon1, lat1, lon2, lat2 = map(lambda x: np.asarray(x, dtype=float),
                                 (lon1, lat1, lon2, lat2))
    r = 6371.0088
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dp = np.deg2rad(lat2 - lat1)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))


def construct_monitored_id(df: pd.DataFrame) -> pd.Series:
    """Reproduce the ID construction in the supplied CML processing code."""
    required = ["NEName", "BrdID", "BrdName", "PortNO", "PortName", "PathID"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Raw CML file is missing columns required for Monitored_ID: {missing}")

    return (
        df["NEName"].astype(str) + "-"
        + df["BrdID"].astype(str) + "-"
        + df["BrdName"].astype(str) + "-"
        + df["PortNO"].astype(str) + "("
        + df["PortName"].astype(str) + ")-"
        + df["PathID"].astype(str)
    )


def iter_raw_files(raw_dir: Path, patterns: Sequence[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(raw_dir.glob(pattern))
    return sorted(set(p for p in files if p.is_file()))


def extract_datetime_from_filename(fname: str) -> datetime | None:
    """Extract the nominal file hour from an AT Schedule filename.

    Example
    -------
    Schedule_pfm_SDH_20250812004105281472818770368_1.txt
                         ^^^^^^^^^^
                         YYYYMMDDHH
    """
    parts = Path(fname).name.split("_")
    if len(parts) < 4:
        return None
    timestamp = parts[3]
    try:
        return datetime.strptime(timestamp[:10], "%Y%m%d%H")
    except (ValueError, TypeError):
        return None


def _read_raw_cml_file(path: Path, usecols: Sequence[str]) -> pd.DataFrame:
    """Read one AT PM file without the incompatible low_memory/python pairing."""
    # IMPORTANT: low_memory is supported only by pandas' C parser. It must not
    # be passed together with engine="python".
    try:
        return pd.read_csv(
            path,
            sep="\t",
            usecols=lambda c: str(c).strip() in usecols,
            engine="python",
            on_bad_lines="warn",
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            sep="\t",
            usecols=lambda c: str(c).strip() in usecols,
            engine="python",
            on_bad_lines="warn",
            encoding="latin-1",
        )


def load_observed_ids_for_day(
    raw_dir: Path,
    target_date: str,
    patterns: Sequence[str] = ("*.txt", "*.tsv", "*.csv"),
) -> tuple[pd.DataFrame, list[Path]]:
    """Load all CML observations belonging to one UTC calendar day.

    Files are first filtered using the timestamp embedded in their names. The
    actual ``EndTime`` column is then used for the final day selection. A
    one-hour margin is included on either side so a file whose nominal hour is
    close to midnight is not accidentally omitted.
    """
    target = pd.Timestamp(target_date)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    else:
        target = target.tz_convert("UTC")

    day_start = target.normalize()
    day_end = day_start + pd.Timedelta(days=1)

    all_files = iter_raw_files(raw_dir, patterns)
    if not all_files:
        raise FileNotFoundError(
            f"No raw CML files found in {raw_dir} for patterns {patterns}"
        )

    # Follow the same filename-selection logic as the operational workflow.
    dated_files = [
        (extract_datetime_from_filename(path.name), path) for path in all_files
    ]
    dated_files = [(dt, path) for dt, path in dated_files if dt is not None]

    if not dated_files:
        raise RuntimeError(
            f"No timestamped Schedule files were found in {raw_dir}. "
            "Expected names such as Schedule_pfm_SDH_YYYYMMDDHH..._1.txt"
        )

    naive_start = day_start.tz_localize(None).to_pydatetime() - timedelta(hours=1)
    naive_end = day_end.tz_localize(None).to_pydatetime() + timedelta(hours=1)
    candidate_files = [
        path for dt, path in dated_files if naive_start <= dt < naive_end
    ]

    available_min = min(dt for dt, _ in dated_files)
    available_max = max(dt for dt, _ in dated_files)
    print(f"Timestamped raw files available: {available_min} to {available_max}")
    print(f"Files selected for {day_start.date()} UTC: {len(candidate_files)}")

    if not candidate_files:
        raise RuntimeError(
            f"No filenames overlap {day_start.date()} UTC. "
            f"Available filename range is {available_min} to {available_max}."
        )

    chunks: list[pd.DataFrame] = []
    used_files: list[Path] = []
    usecols = [
        "NEName", "BrdID", "BrdName", "PortNO", "PortName", "PathID",
        "EndTime", "EventName", "Value",
    ]

    for path in candidate_files:
        try:
            df = _read_raw_cml_file(path, usecols)
            # Protect against hidden whitespace in column names.
            df.columns = [str(c).strip() for c in df.columns]
        except Exception as exc:
            print(f"WARNING: skipped {path.name}: {exc}")
            continue

        if "EndTime" not in df.columns:
            print(f"WARNING: skipped {path.name}: no EndTime column")
            continue

        t = pd.to_datetime(df["EndTime"], utc=True, errors="coerce")
        keep = (t >= day_start) & (t < day_end)
        if not keep.any():
            continue

        d = df.loc[keep].copy()
        d["time15"] = t.loc[keep].dt.floor("15min")
        d["Monitored_ID"] = construct_monitored_id(d)

        if "EventName" in d.columns:
            signal_event = d["EventName"].astype(str).str.startswith(("RSL_", "TSL_"))
            d = d.loc[signal_event]

        if not d.empty:
            chunks.append(d[["Monitored_ID", "time15"]])
            used_files.append(path)

    if not chunks:
        raise RuntimeError(
            f"Files were found for {day_start.date()}, but none contained valid "
            "RSL/TSL observations with EndTime inside that UTC day. Check the "
            "file delimiter and EndTime values."
        )

    observed = pd.concat(chunks, ignore_index=True).drop_duplicates()
    inventory = (
        observed.groupby("Monitored_ID", as_index=False)
        .agg(
            n_15min_bins=("time15", "nunique"),
            first_time=("time15", "min"),
            last_time=("time15", "max"),
        )
    )
    inventory["availability_fraction"] = inventory["n_15min_bins"] / 96.0
    return inventory, used_files


def standardize_metadata(metadata_csv: Path) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv, low_memory=False)
    required = ["Monitored_ID", "XStart", "YStart", "XEnd", "YEnd"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Metadata file is missing required columns: {missing}")

    for c in ["XStart", "YStart", "XEnd", "YEnd", "Frequency", "PathLength"]:
        if c in meta.columns:
            meta[c] = pd.to_numeric(meta[c], errors="coerce")

    # Use geometry-derived length for a consistent map diagnostic. Preserve the
    # source PathLength for comparison in the exported inventory.
    meta["length_geometry_km"] = haversine_km(
        meta["XStart"], meta["YStart"], meta["XEnd"], meta["YEnd"]
    )

    if "Frequency" not in meta.columns:
        meta["Frequency"] = np.nan

    freq = meta["Frequency"].astype(float)
    finite = freq[np.isfinite(freq)]
    # The supplied coupling code divides metadata frequency by 1000, indicating
    # MHz input. This heuristic also accepts metadata already expressed in GHz.
    if not finite.empty and finite.median() > 100.0:
        meta["frequency_ghz"] = freq / 1000.0
    else:
        meta["frequency_ghz"] = freq

    return meta


def physical_link_key(row: pd.Series, decimals: int = 5) -> str:
    """Direction-independent key from sorted endpoint coordinate pairs."""
    a = (round(float(row.XStart), decimals), round(float(row.YStart), decimals))
    b = (round(float(row.XEnd), decimals), round(float(row.YEnd), decimals))
    p0, p1 = sorted((a, b))
    return f"{p0[0]:.{decimals}f},{p0[1]:.{decimals}f}|{p1[0]:.{decimals}f},{p1[1]:.{decimals}f}"


def build_daily_links(
    observed_ids: pd.DataFrame,
    metadata: pd.DataFrame,
    collapse_directions: bool = True,
) -> pd.DataFrame:
    """Couple observed IDs to metadata and optionally collapse opposite directions."""
    links = observed_ids.merge(metadata, on="Monitored_ID", how="left", indicator=True)
    unmatched = links["_merge"] != "both"
    if unmatched.any():
        print(f"WARNING: {unmatched.sum()} observed Monitored_ID values lack metadata")
    links = links.loc[~unmatched].drop(columns="_merge").copy()
    links = links.dropna(subset=["XStart", "YStart", "XEnd", "YEnd"])

    links["physical_link_id"] = links.apply(physical_link_key, axis=1)

    if collapse_directions:
        # Keep one geometric line; aggregate properties across directional
        # sub-links. Median frequency is used only for visualization where the
        # two directions overlap spatially.
        agg = {
            "XStart": "first", "YStart": "first", "XEnd": "first", "YEnd": "first",
            "length_geometry_km": "median", "frequency_ghz": "median",
            "n_15min_bins": "max", "availability_fraction": "max",
            "first_time": "min", "last_time": "max",
            "Monitored_ID": lambda x: ";".join(sorted(set(map(str, x)))),
        }
        if "PathLength" in links.columns:
            agg["PathLength"] = "median"
        links = links.groupby("physical_link_id", as_index=False).agg(agg)
        links["n_directional_sublinks"] = links["Monitored_ID"].str.count(";") + 1
    else:
        links["n_directional_sublinks"] = 1

    return links


def load_tahmo_gauges(gauge_path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Load TAHMO station metadata from CSV or the Gold Standard workbook.

    Preferred project format
    ------------------------
    ``stations_metadata.csv`` with columns such as::

        station code, latitude, longitude

    Optional country columns (``cc``, ``country``, ``country_code``) are used
    to retain Ghana stations only. If no country column exists, all rows with
    coordinates inside the Ghana plotting domain are retained.

    The older headerless Gold Standard Excel workbook remains supported.
    """
    gauge_path = Path(gauge_path)
    if not gauge_path.exists():
        raise FileNotFoundError(f"Gauge metadata file not found: {gauge_path}")

    suffix = gauge_path.suffix.lower()

    if suffix == ".csv":
        g = pd.read_csv(gauge_path)
        original_columns = list(g.columns)
        normalized = {
            c: re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
            for c in g.columns
        }
        g = g.rename(columns=normalized)

        def first_existing(candidates):
            for c in candidates:
                if c in g.columns:
                    return c
            return None

        id_col = first_existing([
            "station_code", "station_id", "station", "code", "id"
        ])
        lat_col = first_existing(["latitude", "lat", "station_latitude"])
        lon_col = first_existing(["longitude", "lon", "lng", "station_longitude"])
        country_col = first_existing([
            "cc", "country", "country_code", "iso2", "iso_code"
        ])

        if lat_col is None or lon_col is None:
            raise ValueError(
                "Gauge CSV must contain latitude and longitude columns. "
                f"Found columns: {original_columns}"
            )

        if id_col is None:
            g["station_id"] = [f"GAUGE_{i:04d}" for i in range(len(g))]
            id_col = "station_id"

        keep = [id_col, lat_col, lon_col]
        if country_col is not None:
            keep.append(country_col)
        g = g[keep].copy()

        rename = {
            id_col: "station_id",
            lat_col: "latitude",
            lon_col: "longitude",
        }
        if country_col is not None:
            rename[country_col] = "country"
        g = g.rename(columns=rename)

        if "country" in g.columns:
            country = g["country"].astype(str).str.upper().str.strip()
            gh_values = {"GH", "GHA", "GHANA"}
            # Apply the filter only when recognizable country values exist.
            if country.isin(gh_values).any():
                g = g[country.isin(gh_values)].copy()

        g.attrs["source"] = str(gauge_path)
        g.attrs["sheet_name"] = "CSV"

    elif suffix in {".xlsx", ".xls"}:
        excel = pd.ExcelFile(gauge_path)
        selected = sheet_name or excel.sheet_names[0]
        g = pd.read_excel(gauge_path, sheet_name=selected, header=None)
        if g.shape[1] < 4:
            raise ValueError(f"Gauge worksheet {selected!r} has fewer than four columns")

        g = g.iloc[:, :4].copy()
        g.columns = ["station_id", "country", "latitude", "longitude"]
        country = g["country"].astype(str).str.upper().str.strip()
        g = g[country.isin({"GH", "GHA", "GHANA"})].copy()
        g.attrs["source"] = str(gauge_path)
        g.attrs["sheet_name"] = selected

    else:
        raise ValueError(
            f"Unsupported gauge metadata format {suffix!r}; use .csv, .xlsx, or .xls"
        )

    g["station_id"] = g["station_id"].astype(str).str.strip()
    g["latitude"] = pd.to_numeric(g["latitude"], errors="coerce")
    g["longitude"] = pd.to_numeric(g["longitude"], errors="coerce")
    g = g.dropna(subset=["latitude", "longitude"])

    # Restrict to the Ghana map domain. This is especially useful when the CSV
    # has no country-code column or contains stations from several countries.
    lon_min, lon_max, lat_min, lat_max = DEFAULT_EXTENT
    g = g[
        g["longitude"].between(lon_min, lon_max)
        & g["latitude"].between(lat_min, lat_max)
    ].copy()

    g = g.drop_duplicates(subset=["station_id", "latitude", "longitude"])
    g = g.reset_index(drop=True)

    if g.empty:
        raise RuntimeError(
            f"No valid Ghana gauge coordinates were found in {gauge_path}"
        )

    print(f"TAHMO gauges loaded: {len(g)} from {gauge_path}")
    return g


def derive_frequency_bins(values: pd.Series):
    finite = np.sort(pd.unique(values[np.isfinite(values)].astype(float)))
    if len(finite) == 0:
        return (0.0, 7.0, 8.0, np.inf), ("Low", "Medium", "High")

    if FREQUENCY_BINS_GHZ is not None:
        edges = FREQUENCY_BINS_GHZ
    elif len(finite) >= 3:
        q1, q2 = np.quantile(finite, [1 / 3, 2 / 3])
        if np.isclose(q1, q2):
            # Equal-width fallback for a heavily discrete frequency set.
            lo, hi = float(finite.min()), float(finite.max())
            q1, q2 = lo + (hi - lo) / 3.0, lo + 2.0 * (hi - lo) / 3.0
        edges = (-np.inf, float(q1), float(q2), np.inf)
    else:
        center = float(np.median(finite))
        edges = (-np.inf, center - 0.25, center + 0.25, np.inf)

    labels = (
        f"Low: ≤{edges[1]:.2f} GHz",
        f"Medium: {edges[1]:.2f}–{edges[2]:.2f} GHz",
        f"High: >{edges[2]:.2f} GHz",
    )
    return edges, labels


def add_base_map(ax, extent=DEFAULT_EXTENT):
    proj = ccrs.PlateCarree()
    ax.set_extent(extent, crs=proj)
    ax.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="0.92", zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("10m"), facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale("10m"), facecolor="0.92", edgecolor="0.6", zorder=0)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.8, edgecolor="0.25")
    ax.coastlines(resolution="10m", linewidth=0.9)

    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0.4,
                      color="0.55", alpha=0.65, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.FixedLocator(np.arange(-4, 2, 1))
    gl.ylocator = mticker.FixedLocator(np.arange(5, 12, 1))
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}
    return proj


def plot_gauges(ax, gauges: pd.DataFrame, proj, label=True):
    ax.scatter(gauges["longitude"], gauges["latitude"], transform=proj,
               marker="^", s=28, facecolor="white", edgecolor=GAUGE_COLOR,
               linewidth=0.9, zorder=5, label="TAHMO gauges" if label else None)


def draw_lines(ax, links, proj, color_values=None, categories=None,
               colors=CATEGORY_COLORS, linewidth=1.1, alpha=0.85):
    if color_values is None:
        for row in links.itertuples(index=False):
            ax.plot([row.XStart, row.XEnd], [row.YStart, row.YEnd],
                    transform=proj, color="0.25", linewidth=linewidth,
                    alpha=alpha, zorder=3)
        return

    for cat, color in zip(categories, colors):
        sel = links[color_values == cat]
        for row in sel.itertuples(index=False):
            ax.plot([row.XStart, row.XEnd], [row.YStart, row.YEnd],
                    transform=proj, color=color, linewidth=linewidth,
                    alpha=alpha, zorder=3)


def plot_availability(ax, links, gauges, target_date):
    proj = add_base_map(ax)
    edges = [-0.001, 0.50, 0.90, 1.001]
    labels = ["Low: <50%", "Medium: 50–90%", "High: >90%"]
    cat = pd.cut(links["availability_fraction"], bins=edges, labels=labels,
                 include_lowest=True)
    draw_lines(ax, links, proj, cat, labels)
    plot_gauges(ax, gauges, proj)
    handles = [Line2D([0], [0], color=c, lw=2, label=l)
               for c, l in zip(CATEGORY_COLORS, labels)]
    handles.append(Line2D([0], [0], marker="^", linestyle="none", markersize=6,
                          markerfacecolor="white", markeredgecolor=GAUGE_COLOR,
                          label="TAHMO gauges"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)
    ax.set_title(f"Daily CML network and data availability\n{target_date} UTC", fontsize=12)


def plot_frequency(ax, links, gauges, target_date):
    proj = add_base_map(ax)
    edges, labels = derive_frequency_bins(links["frequency_ghz"])
    cat = pd.cut(links["frequency_ghz"], bins=edges, labels=labels,
                 include_lowest=True)
    draw_lines(ax, links, proj, cat, labels)
    plot_gauges(ax, gauges, proj)
    handles = [Line2D([0], [0], color=c, lw=2, label=l)
               for c, l in zip(CATEGORY_COLORS, labels)]
    handles.append(Line2D([0], [0], marker="^", linestyle="none", markersize=6,
                          markerfacecolor="white", markeredgecolor=GAUGE_COLOR,
                          label="TAHMO gauges"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)
    ax.set_title(f"CML operating-frequency classes\n{target_date} UTC", fontsize=12)


def plot_length(ax, links, gauges, target_date):
    proj = add_base_map(ax)
    cat = pd.cut(links["length_geometry_km"], bins=LENGTH_BINS_KM,
                 labels=LENGTH_LABELS, include_lowest=True, right=True)
    draw_lines(ax, links, proj, cat, LENGTH_LABELS)
    plot_gauges(ax, gauges, proj)
    handles = [Line2D([0], [0], color=c, lw=2, label=l)
               for c, l in zip(CATEGORY_COLORS, LENGTH_LABELS)]
    handles.append(Line2D([0], [0], marker="^", linestyle="none", markersize=6,
                          markerfacecolor="white", markeredgecolor=GAUGE_COLOR,
                          label="TAHMO gauges"))
    ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=True)
    ax.set_title(f"CML path-length classes\n{target_date} UTC", fontsize=12)


def save_single_plot(plotter, links, gauges, target_date, path):
    fig = plt.figure(figsize=(8.0, 8.0), dpi=180)
    ax = plt.axes(projection=ccrs.PlateCarree())
    plotter(ax, links, gauges, target_date)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def write_summary(path: Path, links: pd.DataFrame, gauges: pd.DataFrame,
                  target_date: str, used_files: Iterable[Path]):
    n_phys = len(links)
    n_sub = int(links["n_directional_sublinks"].sum())
    length_counts = pd.cut(links["length_geometry_km"], bins=LENGTH_BINS_KM,
                           labels=LENGTH_LABELS, include_lowest=True).value_counts().sort_index()
    freq_edges, freq_labels = derive_frequency_bins(links["frequency_ghz"])
    freq_counts = pd.cut(links["frequency_ghz"], bins=freq_edges,
                         labels=freq_labels, include_lowest=True).value_counts().sort_index()

    lines = [
        f"Target day (UTC): {target_date}",
        f"Raw files contributing data: {len(list(used_files))}",
        f"Unique physical links: {n_phys}",
        f"Directional sub-links represented: {n_sub}",
        f"TAHMO Ghana gauges plotted: {len(gauges)}",
        f"Median link length: {links['length_geometry_km'].median():.2f} km",
        f"Links >10 km: {(links['length_geometry_km'] > 10).sum()} ({100*(links['length_geometry_km'] > 10).mean():.1f}%)",
        f"Median daily availability: {100*links['availability_fraction'].median():.1f}% of 96 bins",
        "",
        "Length-class counts:",
        *[f"  {k}: {int(v)}" for k, v in length_counts.items()],
        "",
        "Frequency-class counts:",
        *[f"  {k}: {int(v)}" for k, v in freq_counts.items()],
        "",
        f"Gauge metadata source: {gauges.attrs.get('source', 'unknown')}",
        f"Gauge sheet/type: {gauges.attrs.get('sheet_name', 'unknown')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="Directory containing AT Schedule_*.txt files")
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Matched CML metadata CSV containing Monitored_ID and endpoint coordinates")
    parser.add_argument("--gauges", type=Path, required=True,
                        help="TAHMO station metadata (.csv preferred; .xlsx also supported)")
    parser.add_argument("--date", required=True, help="Target UTC date, YYYY-MM-DD")
    parser.add_argument("--output-dir", type=Path, default=Path("./cml_map_outputs"))
    parser.add_argument("--gauge-sheet", default=None,
                        help="Worksheet name when --gauges is an Excel workbook")
    parser.add_argument("--keep-directions", action="store_true",
                        help="Keep directional sub-links rather than collapsing identical endpoint pairs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.Timestamp(args.date)  # validate date

    observed, used_files = load_observed_ids_for_day(args.raw_dir, args.date)
    metadata = standardize_metadata(args.metadata)
    links = build_daily_links(observed, metadata,
                              collapse_directions=not args.keep_directions)
    gauges = load_tahmo_gauges(args.gauges, args.gauge_sheet)

    if links.empty:
        raise RuntimeError("No observed CML IDs could be coupled to valid link metadata")

    date_tag = pd.Timestamp(args.date).strftime("%Y-%m-%d")
    save_single_plot(plot_availability, links, gauges, date_tag,
                     args.output_dir / f"cml_network_daily_availability_{date_tag}.png")
    save_single_plot(plot_frequency, links, gauges, date_tag,
                     args.output_dir / f"cml_network_frequency_{date_tag}.png")
    save_single_plot(plot_length, links, gauges, date_tag,
                     args.output_dir / f"cml_network_length_{date_tag}.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 7), dpi=180,
                             subplot_kw={"projection": ccrs.PlateCarree()})
    plot_availability(axes[0], links, gauges, date_tag)
    plot_frequency(axes[1], links, gauges, date_tag)
    plot_length(axes[2], links, gauges, date_tag)
    fig.suptitle("AirtelTigo Ghana CML network diagnostics with TAHMO gauges",
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.output_dir / f"cml_network_three_panel_{date_tag}.png",
                bbox_inches="tight")
    plt.close(fig)

    export_cols = [
        "physical_link_id", "Monitored_ID", "XStart", "YStart", "XEnd", "YEnd",
        "frequency_ghz", "length_geometry_km", "PathLength",
        "n_15min_bins", "availability_fraction", "first_time", "last_time",
        "n_directional_sublinks"
    ]
    export_cols = [c for c in export_cols if c in links.columns]
    links[export_cols].to_csv(
        args.output_dir / f"daily_cml_link_inventory_{date_tag}.csv", index=False
    )
    write_summary(args.output_dir / f"daily_cml_summary_{date_tag}.txt",
                  links, gauges, date_tag, used_files)

    print(f"Completed. Outputs written to: {args.output_dir.resolve()}")
    print(f"Physical links: {len(links)}")
    print(f"Directional sub-links represented: {int(links['n_directional_sublinks'].sum())}")
    print(f"TAHMO Ghana gauges: {len(gauges)}")


if __name__ == "__main__":
    main()
