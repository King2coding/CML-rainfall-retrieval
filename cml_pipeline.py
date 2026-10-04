
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CML → Rainfall → Map (15‑min, Pycomlink‑style cleaning, RAINLINK‑like logic)

Author: (you)
Date: 2025-09-23

USAGE (quick start):
    1) Update the CONFIG section below or point to a YAML via --config cml_config.yaml
    2) Run:
        python cml_pipeline.py --meta path/to/metadata.csv --rsl path/to/link_raw.csv --out ./out_dir
    3) Outputs:
        - cleaned_perlink.parquet           (per-link, per-15min cleaned signals + Pmin/Pmax)
        - retrieval_perlink.parquet         (per-link rainfall, attenuation, flags)
        - retrieval_points_15min.parquet    (geo points for mapping at each 15-min time)
        - grid_15min_hourly/                (NetCDF grids for instantaneous and hourly accumulations)

NOTES:
    - Native cadence = 15 minutes (kept).
    - Inputs expected columns:
        * metadata.csv: link_id, tx_lat, tx_lon, rx_lat, rx_lon, freq_GHz, pol, length_km
        * link_raw.csv: timestamp, link_id, RSL_MIN, RSL_MAX, TSL_AVG
    - Duplex links are detected by matching reversed endpoints (approx) & same (freq, pol).
      When both directions valid → average rain rate. If only one valid → use single.
    - Wet–dry: provisional threshold on Pmin using rolling MAD; then neighbor voting (same timestamp).
      (Time ±1-step voting can be added later if needed.)

DEPENDENCIES (pip):
    pip install pandas numpy scipy xarray pyproj pyyaml pyarrow

Adapt / extend as needed. This script aims for clarity over micro-optimizations.
"""

import argparse
import sys
import math
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xarray as xr
import yaml

# -----------------------------
# CONFIG (editable defaults)
# -----------------------------

DEFAULTS = dict(
    cadence_minutes=15,
    cleaning=dict(
        rsl_valid_range=[-150.0, -40.0],
        max_jump_db=10.0,          # max allowed consecutive-step jump; larger → mask
        flatline_run=4,            # if RSL_MIN == RSL_MAX == TSL_AVG for >= N steps → mask
        atpc_step_db=1.2,          # detect step changes ≥ this; mask a guard window
        atpc_guard_steps=2,        # steps to mask before/after a detected step
    ),
    baseline=dict(
        window_hours=48,           # rolling window size for dry-only median baseline
        min_points=24,             # minimum dry points required within window (for 15‑min cadence)
        expand_to_hours=72,        # fallback window if not enough dry samples
    ),
    wetdry=dict(
        tau_floor_db=0.7,          # minimum excess attenuation to consider wet
        z_mad=3.0,                 # MAD multiplier for dynamic tau
        neighbor_radius_km=30.0,   # neighbor search radius (link centers)
        min_neighbors=3,           # K
        min_neighbor_frac=0.4      # p
    ),
    waa=dict(
        mode="constant",           # "constant" (recommended initially)
        constant_db=1.5
    ),
    mapping=dict(
        use_triplet=False,         # if True → assign 1/3 R to tx, rx, center
        grid_res_deg=0.1,          # grid resolution for maps
        idw_radius_km=60.0,
        idw_power=2.0,
        min_points=3
    ),
    output=dict(
        save_intermediate=True
    )
)


# -----------------------------
# Utility functions
# -----------------------------

def to_datetime_utc(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, utc=True, errors="coerce")
    return out


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    # Vectorized is elsewhere; here a scalar fallback
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def approx_equal(a: float, b: float, tol: float = 1e-4) -> bool:
    return abs(a - b) <= tol


# -----------------------------
# ITU-R P.838-3 coefficients (approx table; edit if you have the exact set)
# keys: (freq_GHz_rounded, pol) -> (a, b)
# These are representative; please update with your exact bands if needed.
# -----------------------------

AB_TABLE = {
    (13.0, 'H'): (0.045, 1.10), (13.0, 'V'): (0.055, 1.12),
    (15.0, 'H'): (0.060, 1.10), (15.0, 'V'): (0.073, 1.12),
    (18.0, 'H'): (0.110, 1.08), (18.0, 'V'): (0.130, 1.10),
    (23.0, 'H'): (0.220, 1.05), (23.0, 'V'): (0.260, 1.07),
    (38.0, 'H'): (0.900, 1.00), (38.0, 'V'): (1.100, 1.02),
}

def ab_from_freq_pol(f_GHz: float, pol: str) -> Tuple[float, float]:
    # Map to nearest defined freq; you may refine to interpolate by frequency.
    pol = (pol or "H").upper()[0]
    freqs = sorted({f for f,_ in AB_TABLE.keys()})
    f_rounded = min(freqs, key=lambda fr: abs(fr - float(f_GHz)))
    return AB_TABLE.get((f_rounded, pol), (0.060, 1.10))


# -----------------------------
# Cleaning: Pycomlink-like filters
# -----------------------------

def clean_link_timeseries(df_link: pd.DataFrame, conf: Dict) -> pd.DataFrame:
    """
    Input df_link columns: timestamp, RSL_MIN, RSL_MAX, TSL_AVG
    Returns same index with 'valid' boolean mask and optionally masked values.
    """
    df = df_link.sort_values("timestamp").copy()
    rng = conf["cleaning"]["rsl_valid_range"]
    max_jump = conf["cleaning"]["max_jump_db"]
    flat_run = int(conf["cleaning"]["flatline_run"])
    step_db = conf["cleaning"]["atpc_step_db"]
    guard = int(conf["cleaning"]["atpc_guard_steps"])

    # 1) Range checks
    valid = (
        df["RSL_MIN"].between(rng[0], rng[1], inclusive="both") &
        df["RSL_MAX"].between(rng[0], rng[1], inclusive="both") &
        df["TSL_AVG"].between(rng[0], rng[1], inclusive="both")
    )

    # 2) Flatline detection
    flat = (df["RSL_MIN"] == df["RSL_MAX"]) & (df["RSL_MIN"] == df["TSL_AVG"])
    # rolling count of consecutive flat states
    flat_count = flat.rolling(flat_run, min_periods=flat_run).sum().fillna(0) >= flat_run
    valid = valid & ~flat_count

    # 3) Large jumps between consecutive steps (on TSL_AVG as proxy)
    d = df["TSL_AVG"].diff().abs()
    large_jump = d > max_jump
    valid = valid & ~large_jump

    # 4) ATPC step detection (persistent step)
    # Simple heuristic: where diff exceeds threshold; mask ±guard steps
    step_idx = np.where(d >= step_db)[0]
    atpc_mask = np.zeros(len(df), dtype=bool)
    for idx in step_idx:
        lo = max(0, idx - guard)
        hi = min(len(df)-1, idx + guard)
        atpc_mask[lo:hi+1] = True
    valid = valid & ~atpc_mask

    out = df.copy()
    out["valid"] = valid.values
    # (Optionally) mask values themselves
    out.loc[~out["valid"], ["RSL_MIN", "RSL_MAX", "TSL_AVG"]] = np.nan
    return out


# -----------------------------
# Pmin / Pmax and provisional baseline
# -----------------------------

def compute_Pmin_Pmax(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Pmin"] = out["RSL_MIN"] - out["TSL_AVG"]
    out["Pmax"] = out["RSL_MAX"] - out["TSL_AVG"]
    return out


def rolling_MAD(series: pd.Series, window_steps: int, min_periods: int = 12) -> pd.Series:
    def mad(x):
        med = np.nanmedian(x)
        return 1.4826 * np.nanmedian(np.abs(x - med))
    return series.rolling(window_steps, min_periods=min_periods).apply(mad, raw=False)


def initial_baseline_Pmin(df_link: pd.DataFrame, steps_48h: int) -> pd.Series:
    """
    A first baseline guess using rolling median of Pmin (including all points).
    This will be refined using dry-only in the next pass.
    """
    return df_link["Pmin"].rolling(steps_48h, min_periods=12).median().ffill()


# -----------------------------
# Wet–dry classification (two-pass)
# -----------------------------

def provisional_wet_mask(df_link: pd.DataFrame, conf: Dict, steps_48h: int) -> pd.Series:
    """
    Create a provisional wet/dry based on excess attenuation vs dynamic tau.
    """
    tau_floor = conf["wetdry"]["tau_floor_db"]
    z = conf["wetdry"]["z_mad"]

    base0 = initial_baseline_Pmin(df_link, steps_48h)
    A0 = (base0 - df_link["Pmin"]).clip(lower=0)
    noise = rolling_MAD(df_link["Pmin"].diff(), window_steps=steps_48h//8 or 6, min_periods=6)
    tau = np.maximum(tau_floor, z * noise.fillna(noise.median()))
    wet0 = A0 > tau
    return wet0.fillna(False)


def refine_baseline_dry_only(df_link: pd.DataFrame, wet_mask: pd.Series, conf: Dict) -> pd.Series:
    # rolling dry-only median with fallback to extended window
    step_minutes = conf["cadence_minutes"]
    steps_48h = int(conf["baseline"]["window_hours"] * 60 / step_minutes)
    steps_72h = int(conf["baseline"]["expand_to_hours"] * 60 / step_minutes)
    min_pts = int(conf["baseline"]["min_points"])

    dry_vals = df_link["Pmin"].where(~wet_mask)
    base = dry_vals.rolling(steps_48h, min_periods=min_pts).median()
    if base.isna().any():
        base2 = dry_vals.rolling(steps_72h, min_periods=min_pts//2 or 6).median()
        base = base.combine_first(base2)
    return base.ffill()


def neighbor_graph(meta: pd.DataFrame, radius_km: float) -> List[List[int]]:
    """
    Build neighbor index list based on link center coordinates.
    Returns list of neighbor indices per link index in `meta`.
    """
    R = 6371.0
    latc = (meta["tx_lat"].values + meta["rx_lat"].values) / 2.0
    lonc = (meta["tx_lon"].values + meta["rx_lon"].values) / 2.0

    # Simple equirectangular projection around mean latitude for KD
    lat0 = np.nanmean(latc)
    x = np.deg2rad(lonc) * R * np.cos(np.deg2rad(lat0))
    y = np.deg2rad(latc) * R
    coords = np.c_[x, y]

    tree = cKDTree(coords)
    all_nbrs = []
    for i, pt in enumerate(coords):
        idx = tree.query_ball_point(pt, r=radius_km)
        idx = [j for j in idx if j != i]
        all_nbrs.append(idx)
    return all_nbrs


def neighbor_vote_at_time(df_time: pd.DataFrame, meta_idx_map: Dict[str, int],
                          nbrs: List[List[int]], conf: Dict) -> pd.Series:
    """
    Apply neighbor voting to provisional wet mask at a single timestamp.
    df_time: columns include link_id, wet0 (provisional, bool).
    Returns final wet mask for these link_ids at this time.
    """
    K = int(conf["wetdry"]["min_neighbors"])
    p = float(conf["wetdry"]["min_neighbor_frac"])

    wet0 = df_time.set_index("link_id")["wet0"]
    out = {}
    for link_id, is_wet in wet0.items():
        if not bool(is_wet):
            out[link_id] = False
            continue
        i = meta_idx_map.get(link_id, None)
        if i is None:
            out[link_id] = bool(is_wet)
            continue
        neighbors_i = nbrs[i]
        if not neighbors_i:
            out[link_id] = bool(is_wet)
            continue
        # count neighbor wets in this time slice
        neighbor_ids = [df_time.iloc[j]["link_id"] for j in range(len(df_time))]  # list present this time
        # map neighbor indices to ids present
        present_nbr_ids = []
        for j in neighbors_i:
            # find link_id for meta index j
            # build a reverse map meta_index->link_id first (outside) for speed; here simple approach:
            pass
        # We'll do a faster approach: build a set of link_ids present
        present_set = set(df_time["link_id"])
        # We need a mapping from meta idx to link_id; so we will pre-build it outside and pass it in.
        # For simplicity, re-derive here from df_time:
        # BUT to keep performance, let's accept a simpler rule:
        # compute frac of present neighbors (by geo radius) that are wet, approximated by those in df_time & neighbor center distance precomputed later.
        out[link_id] = True  # provisional; we will refine shortly
    # The above placeholder needs a robust implementation.
    # Let's implement a simpler, data-driven neighbor fraction based on great-circle distance threshold computed on-the-fly.

    # Re-implement neighbor voting robustly with geodesic distance filter
    # Precompute centers for df_time only:
    centers = df_time[["link_id", "lat_c", "lon_c", "wet0"]].set_index("link_id")
    ids = centers.index.tolist()
    lat = centers["lat_c"].values
    lon = centers["lon_c"].values
    wet0v = centers["wet0"].values.astype(bool)

    # Distance matrix (small, only links present in this time slice)
    n = len(ids)
    km = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i+1, n):
            km_ij = haversine_km(lat[i], lon[i], lat[j], lon[j])
            km[i, j] = km[j, i] = km_ij

    Rkm = float(conf["wetdry"]["neighbor_radius_km"])
    final = []
    for i in range(n):
        if not wet0v[i]:
            final.append(False)
            continue
        # neighbors within Rkm (excluding self)
        mask = (km[i, :] > 0) & (km[i, :] <= Rkm)
        if not np.any(mask):
            final.append(True)   # keep wet if no neighbors known
            continue
        nbr_wets = np.sum(wet0v[mask])
        nbr_cnt = np.sum(mask)
        cond = (nbr_wets >= K) or (nbr_cnt > 0 and (nbr_wets / nbr_cnt) >= p)
        final.append(bool(cond))
    return pd.Series(final, index=ids)


# -----------------------------
# Wet-antenna & rain retrieval
# -----------------------------

def apply_WAA(A: pd.Series, wet: pd.Series, conf: Dict) -> pd.Series:
    mode = conf["waa"]["mode"]
    if mode == "constant":
        W = float(conf["waa"]["constant_db"])
        Aeff = np.where(wet.values, np.clip(A.values - W, 0, None), 0.0)
        return pd.Series(Aeff, index=A.index)
    else:
        # Placeholder for dynamic WAA
        W = float(conf["waa"]["constant_db"])
        Aeff = np.where(wet.values, np.clip(A.values - W, 0, None), 0.0)
        return pd.Series(Aeff, index=A.index)


def rain_from_Aeff(Aeff: pd.Series, length_km: float, a: float, b: float,
                   rmin: float = 0.1, rmax: float = 200.0) -> pd.Series:
    L = max(length_km, 0.1)
    k = Aeff.values / L   # dB/km
    with np.errstate(divide='ignore', invalid='ignore'):
        R = np.power(np.maximum(k / max(a, 1e-6), 0), 1.0/b)
    R = np.clip(R, rmin, rmax)
    R[np.isnan(R)] = 0.0
    return pd.Series(R, index=Aeff.index)


# -----------------------------
# Duplex pairing & averaging
# -----------------------------

def detect_duplex_pairs(meta: pd.DataFrame) -> Dict[str, str]:
    """
    Attempt to pair links that are the same path in opposite direction & same (freq, pol).
    Returns a mapping primary_id -> mate_id (symmetric), with link_id strings.
    Heuristic: endpoints within small spatial tol and swapped, and |freq diff| < 0.2 GHz and same pol.
    """
    tol_km = 0.2  # spatial tolerance
    freq_tol = 0.2

    # Build index keyed by (rounded coords, freq_rounded, pol)
    def round_coord(x): return round(float(x), 5)

    records = []
    for _, r in meta.iterrows():
        key_f = round(float(r.get("freq_GHz", np.nan)), 1)
        records.append(dict(
            link_id=r["link_id"],
            tx_lat=float(r["tx_lat"]), tx_lon=float(r["tx_lon"]),
            rx_lat=float(r["rx_lat"]), rx_lon=float(r["rx_lon"]),
            freq_r=key_f,
            pol=str(r.get("pol", "H"))[:1].upper()
        ))
    M = pd.DataFrame(records)
    pairs = {}

    # O(N^2) but metadata size is manageable; optimize later if needed
    for i in range(len(M)):
        li = M.iloc[i]
        for j in range(i+1, len(M)):
            lj = M.iloc[j]
            # freq & pol match
            if abs(li["freq_r"] - lj["freq_r"]) > freq_tol: 
                continue
            if li["pol"] != lj["pol"]: 
                continue
            # endpoints swapped?
            d1 = haversine_km(li["tx_lat"], li["tx_lon"], lj["rx_lat"], lj["rx_lon"])
            d2 = haversine_km(li["rx_lat"], li["rx_lon"], lj["tx_lat"], lj["tx_lon"])
            if d1 <= tol_km and d2 <= tol_km:
                pairs[li["link_id"]] = lj["link_id"]
                pairs[lj["link_id"]] = li["link_id"]
    return pairs


# -----------------------------
# Mapping (IDW) & outputs
# -----------------------------

def build_grid(extent: Tuple[float,float,float,float], res_deg: float):
    lat_min, lat_max, lon_min, lon_max = extent
    lats = np.arange(lat_min, lat_max + 1e-9, res_deg)
    lons = np.arange(lon_min, lon_max + 1e-9, res_deg)
    LON, LAT = np.meshgrid(lons, lats)
    return LAT, LON


def idw_grid(points_lon: np.ndarray, points_lat: np.ndarray, values: np.ndarray,
             grid_lon: np.ndarray, grid_lat: np.ndarray,
             radius_km: float, power: float, min_points: int) -> np.ndarray:
    out = np.full(grid_lon.shape, np.nan, dtype=float)
    if points_lon.size == 0:
        return out
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(np.nanmean(points_lat) if points_lat.size else 0.0))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            dx = (points_lon - grid_lon[i, j]) * km_per_deg_lon
            dy = (points_lat - grid_lat[i, j]) * km_per_deg_lat
            d = np.hypot(dx, dy)
            m = (d > 0) & (d <= radius_km) & np.isfinite(values)
            if np.count_nonzero(m) < min_points:
                continue
            w = 1.0 / np.power(d[m], power)
            out[i, j] = np.sum(w * values[m]) / np.sum(w)
    return out


def infer_extent(meta: pd.DataFrame, margin_deg: float = 0.5) -> Tuple[float,float,float,float]:
    latc = (meta["tx_lat"].values + meta["rx_lat"].values) / 2.0
    lonc = (meta["tx_lon"].values + meta["rx_lon"].values) / 2.0
    return (float(np.nanmin(latc) - margin_deg),
            float(np.nanmax(latc) + margin_deg),
            float(np.nanmin(lonc) - margin_deg),
            float(np.nanmax(lonc) + margin_deg))


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="CML → Rainfall → Map (15‑min pipeline)")
    parser.add_argument("--meta", required=True, help="Path to metadata CSV")
    parser.add_argument("--rsl", required=True, help="Path to link_raw CSV (15‑min)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--config", default=None, help="YAML config (optional)")
    args = parser.parse_args()

    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    # Load config
    conf = DEFAULTS.copy()
    if args.config:
        with open(args.config, "r") as f:
            user_conf = yaml.safe_load(f) or {}
        # deep-merge
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k, {}), dict):
                    d[k] = deep_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        conf = deep_update(conf, user_conf)

    # 1) Read inputs
    meta = pd.read_csv(args.meta)
    link = pd.read_csv(args.rsl)

    # Basic sanity
    req_meta = {"link_id", "tx_lat", "tx_lon", "rx_lat", "rx_lon", "freq_GHz", "pol", "length_km"}
    req_rsl = {"timestamp", "link_id", "RSL_MIN", "RSL_MAX", "TSL_AVG"}
    if not req_meta.issubset(meta.columns):
        raise ValueError(f"metadata is missing columns: {sorted(list(req_meta - set(meta.columns)))}")
    if not req_rsl.issubset(link.columns):
        raise ValueError(f"link_raw is missing columns: {sorted(list(req_rsl - set(link.columns)))}")

    # Timestamp to UTC
    link["timestamp"] = to_datetime_utc(link["timestamp"])
    # Drop rows with invalid timestamps
    link = link.dropna(subset=["timestamp"])

    # Optionally filter to cadence minutes grid (e.g., exact :00, :15, :30, :45)
    # (Skip snapping here; assume upstream already aligned.)

    # 2) Cleaning per-link (Pycomlink-ish)
    cleaned = []
    for lid, dfL in link.groupby("link_id", sort=False):
        dfc = clean_link_timeseries(dfL, conf)
        dfc["link_id"] = lid
        cleaned.append(dfc)
    cleaned = pd.concat(cleaned, ignore_index=True)
    if conf["output"]["save_intermediate"]:
        cleaned.to_parquet(outdir / "cleaned_perlink.parquet", index=False)

    # 3) Compute Pmin/Pmax
    cleaned = compute_Pmin_Pmax(cleaned)

    # Merge metadata onto per-record rows
    meta_keys = ["link_id", "tx_lat", "tx_lon", "rx_lat", "rx_lon", "freq_GHz", "pol", "length_km"]
    cleaned = cleaned.merge(meta[meta_keys], on="link_id", how="left")

    # Precompute centers
    cleaned["lat_c"] = (cleaned["tx_lat"] + cleaned["rx_lat"]) / 2.0
    cleaned["lon_c"] = (cleaned["tx_lon"] + cleaned["rx_lon"]) / 2.0

    # 4) Provisional wet mask & refine baseline (two-pass) per-link
    step_minutes = int(conf["cadence_minutes"])
    steps_48h = int(conf["baseline"]["window_hours"] * 60 / step_minutes)

    records = []
    for lid, dfL in cleaned.groupby("link_id", sort=False):
        dfL = dfL.sort_values("timestamp")
        wet0 = provisional_wet_mask(dfL, conf, steps_48h)
        base = refine_baseline_dry_only(dfL, wet0, conf)
        # Compute attenuation against refined baseline
        A = (base - dfL["Pmin"]).clip(lower=0)
        dfL2 = dfL.copy()
        dfL2["wet0"] = wet0.values
        dfL2["Baseline"] = base.values
        dfL2["A"] = A.values
        records.append(dfL2)
    perlink = pd.concat(records, ignore_index=True)

    # 5) Neighbor voting per timestamp
    # Build neighbor graph convenience via distance matrix at each time (robust version already in function).
    voted = []
    for t, dfT in perlink.groupby("timestamp", sort=False):
        if len(dfT) == 0:
            continue
        wet_final = neighbor_vote_at_time(dfT, meta_idx_map={}, nbrs=[], conf=conf)
        dfX = dfT.set_index("link_id")
        dfX["wet"] = wet_final
        voted.append(dfX.reset_index())
    perlink = pd.concat(voted, ignore_index=True)

    # 6) WAA & rain retrieval
    out_rows = []
    for lid, dfL in perlink.groupby("link_id", sort=False):
        a, b = ab_from_freq_pol(dfL["freq_GHz"].iloc[0], dfL["pol"].iloc[0])
        Aeff = apply_WAA(dfL["A"], dfL["wet"], conf)
        R = rain_from_Aeff(Aeff, float(dfL["length_km"].iloc[0]), a, b)
        rec = dfL.copy()
        rec["Aeff"] = Aeff.values
        rec["R_mmph"] = R.values
        out_rows.append(rec)
    retrieval = pd.concat(out_rows, ignore_index=True)

    # 7) Duplex averaging
    pairs = detect_duplex_pairs(meta)
    # Build canonical representative: for each pair, keep smallest link_id as primary
    used = set()
    averaged = []
    for t, dfT in retrieval.groupby("timestamp", sort=False):
        # map of link_id to mate
        dfT = dfT.copy()
        dfT["mate_id"] = dfT["link_id"].map(pairs).fillna("")
        # Identify pairs present
        done_ids = set()
        rows = []
        for _, r in dfT.iterrows():
            lid = r["link_id"]
            if lid in done_ids:
                continue
            mate = r["mate_id"]
            if mate and mate in dfT["link_id"].values:
                # average the two
                r2 = dfT[dfT["link_id"] == mate].iloc[0]
                # simple mean of rain; wet if any wet; Aeff mean
                R_mean = np.nanmean([r["R_mmph"], r2["R_mmph"]])
                Aeff_mean = np.nanmean([r["Aeff"], r2["Aeff"]])
                wet_any = bool(r["wet"] or r2["wet"])
                # choose canonical id (sorted)
                cid = min(str(lid), str(mate))
                row = r.copy()
                row["link_id"] = cid + "|duplex"
                row["R_mmph"] = R_mean
                row["Aeff"] = Aeff_mean
                row["wet"] = wet_any
                rows.append(row)
                done_ids.add(lid); done_ids.add(mate)
            else:
                rows.append(r)
                done_ids.add(lid)
        averaged.append(pd.DataFrame(rows))
    retrieval2 = pd.concat(averaged, ignore_index=True)

    if conf["output"]["save_intermediate"]:
        retrieval2.to_parquet(outdir / "retrieval_perlink.parquet", index=False)

    # 8) Build mapping points (center, or triplet)
    pts = []
    if conf["mapping"]["use_triplet"]:
        for _, r in retrieval2.iterrows():
            for which, lat, lon in [("tx", r["tx_lat"], r["tx_lon"]),
                                    ("rx", r["rx_lat"], r["rx_lon"]),
                                    ("c",  r["lat_c"], r["lon_c"])]:
                pts.append(dict(timestamp=r["timestamp"], link_id=f"{r['link_id']}:{which}",
                                lat=lat, lon=lon, R_mmph=float(r["R_mmph"])/3.0))
    else:
        pts = retrieval2.rename(columns={"lat_c": "lat", "lon_c": "lon"})[
            ["timestamp", "link_id", "lat", "lon", "R_mmph"]
        ].to_dict("records")
    pts = pd.DataFrame(pts)
    pts.to_parquet(outdir / "retrieval_points_15min.parquet", index=False)

    # 9) Gridding (IDW) per time → NetCDF; and hourly accumulations
    extent = infer_extent(retrieval2.drop_duplicates("link_id"))
    LAT, LON = build_grid(extent, float(conf["mapping"]["grid_res_deg"]))
    times = sorted(pts["timestamp"].dropna().unique())
    data_15 = []
    for t in times:
        dfT = pts[pts["timestamp"] == t]
        grid = idw_grid(
            dfT["lon"].values, dfT["lat"].values, dfT["R_mmph"].values,
            LON, LAT,
            radius_km=float(conf["mapping"]["idw_radius_km"]),
            power=float(conf["mapping"]["idw_power"]),
            min_points=int(conf["mapping"]["min_points"])
        )
        data_15.append(grid[np.newaxis, ...])
    if data_15:
        arr15 = np.concatenate(data_15, axis=0)  # time, y, x
        ds15 = xr.Dataset(
            {
                "rain_rate": (("time", "lat", "lon"), arr15)
            },
            coords={
                "time": pd.to_datetime(times),
                "lat": LAT[:,0],
                "lon": LON[0,:]
            },
            attrs={"units": "mm/h", "description": "IDW gridded rain rate (15‑min cadence)"}
        )
        grid_dir = outdir / "grid_15min_hourly"; grid_dir.mkdir(exist_ok=True, parents=True)
        ds15.to_netcdf(grid_dir / "grid_15min.nc")

        # Hourly accumulation (mm): sum(R * 0.25) over each hour (since 15‑min → 0.25 h)
        ds15_hour = (ds15.resample(time="1H").sum(skipna=True) * 0.25)
        ds15_hour.rain_rate.attrs["units"] = "mm"
        ds15_hour.rain_rate.attrs["description"] = "Hourly accumulation (sum of 15‑min rates × 0.25 h)"
        ds15_hour.to_netcdf(grid_dir / "grid_hourly_accum.nc")

    print("Done. Outputs in:", str(outdir))


if __name__ == "__main__":
    main()
