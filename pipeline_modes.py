# pipeline_modes.py
import numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Optional

# --- pycomlink bits (assumed installed) ---
from pycomlink.processing.wet_antenna import waa_leijnse_2008_from_A_obs
from pycomlink.processing.k_R_relation import calc_R_from_A


# ---------- A) RAINLINK-strict utilities ----------
def _baseline_q90_past_only(
    rsl_series: pd.Series,
    win="24H", q=0.9,
    min_past_bins:int = 4,     # need at least this many past 15-min bins
    ffill_limit_bins:int = 32, # carry baseline forward ≤8h across gaps
    bfill_limit_bins:int = 4   # small early backfill
):
    base = rsl_series.rolling(window=win, min_periods=min_past_bins, closed="left").quantile(q)
    return base.ffill(limit=ffill_limit_bins).bfill(limit=bfill_limit_bins)



def build_15min_timeseries(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Build a strict 15-min series (RAINLINK style):
      - robust time parsing from df_clean["DateTime"]
      - RSL_dBm = TSL_AVG + (Pmin + Pmax)/2
      - link_id := ID  (so it matches your meta exactly)
      - median aggregate per (link_id, 15 min)
    Output columns include:
      ['link_id','time','RSL_dBm','Frequency','PathLength','XStart','YStart','XEnd','YEnd','pol']
    """
    r = df_clean.copy()

    # ---- robust time parsing ----
    if "DateTime" in r.columns:
        s = r["DateTime"].astype(str).str.strip()
        s = s.str.replace(r"\.0+$", "", regex=True)      # drop trailing ".0"
        s = s.str.replace(r"[^0-9]", "", regex=True)     # keep digits only

        L = s.str.len()
        t = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")

        m12 = L == 12   # YYYYMMDDHHMM
        m14 = L == 14   # YYYYMMDDHHMMSS

        if m12.any():
            t.loc[m12] = pd.to_datetime(s[m12], format="%Y%m%d%H%M", utc=True, errors="coerce")
        if m14.any():
            t.loc[m14] = pd.to_datetime(s[m14], format="%Y%m%d%H%M%S", utc=True, errors="coerce")

        # fallback for anything else (ISO8601 etc.)
        other = ~(m12 | m14)
        if other.any():
            t.loc[other] = pd.to_datetime(s[other], utc=True, errors="coerce")

        r["time"] = t
    else:
        # no DateTime column → try index
        if isinstance(r.index, pd.DatetimeIndex):
            r["time"] = r.index.tz_localize("UTC") if r.index.tz is None else r.index.tz_convert("UTC")
        else:
            raise ValueError("build_15min_timeseries: need 'DateTime' column or a DatetimeIndex.")

    # strict 15-min bins
    r["time15"] = r["time"].dt.floor("15min")

    # RSL (RainLINK-style) from your columns
    r["RSL_dBm"] = (
        pd.to_numeric(r.get("TSL_AVG"), errors="coerce") +
        (pd.to_numeric(r.get("Pmin"), errors="coerce") + pd.to_numeric(r.get("Pmax"), errors="coerce")) / 2.0
    )

    # --- IDs / fields
    # IMPORTANT: make link_id == ID so it matches your gridding meta
    r["link_id"] = r["ID"].astype(str)

    num_cols = ["RSL_dBm","Frequency","PathLength","XStart","YStart","XEnd","YEnd"]
    for c in num_cols:
        r[c] = pd.to_numeric(r[c], errors="coerce")

    # Polarization to single char
    r["pol"] = r["Polarization"].astype(str).str.upper().str[0]

    ts = (
        r.groupby(["link_id","time15"], as_index=False)
         .agg({**{c: "median" for c in num_cols}, **{"pol":"first"}})
         .rename(columns={"time15": "time"})
         .sort_values(["link_id","time"])
         .reset_index(drop=True)
    )

    # quick guard so you notice bad rows early
    bad = ts["time"].isna().sum()
    if bad:
        raise ValueError(f"{bad} timestamps are NaT after parsing. Example bad rows:\n{ts[ts['time'].isna()].head()}")

    return ts


def rainlink_strict_Aobs(ts_15: pd.DataFrame, wet_thr_db: float = 0.5,
                         min_past_bins:int = 4, ffill_limit_bins:int = 32,
                         two_pass: bool = True) -> pd.DataFrame:
    """
    Past-only Q90 baseline -> A_obs and wet mask.
    If two_pass=True: recompute baseline on 'dry-only' (mask out first-pass wet).
    """
    out = []
    for lid, g in ts_15.groupby("link_id", sort=False):
        g = g.sort_values("time").copy()
        rsl = pd.Series(g["RSL_dBm"].astype(float).values,
                        index=pd.DatetimeIndex(g["time"]))

        # pass 1
        base1 = _baseline_q90_past_only(rsl, win="24H", q=0.9,
                                        min_past_bins=min_past_bins,
                                        ffill_limit_bins=ffill_limit_bins)
        A1 = np.maximum(0.0, base1.values - rsl.values)
        wet1 = (A1 > wet_thr_db) & np.isfinite(A1)

        if two_pass:
            # mask wet, recompute baseline only from 'dry' samples
            rsl_dry = rsl.mask(wet1)
            base2 = _baseline_q90_past_only(rsl_dry, win="24H", q=0.9,
                                            min_past_bins=min_past_bins,
                                            ffill_limit_bins=ffill_limit_bins)
            base = base2.fillna(base1)  # keep pass-1 where pass-2 lacks support
        else:
            base = base1

        Aobs = np.maximum(0.0, base.values - rsl.values)
        g["baseline_rsl"] = base.values
        g["A_obs_dB"]     = Aobs
        g["wet_rl"]       = (Aobs > wet_thr_db) & np.isfinite(Aobs)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def rainlink_strict_R(dfA: pd.DataFrame, R_min: float = 0.05) -> pd.DataFrame:
    """
    Leijnse (2008) wet-antenna + ITU(2005) k–α to rain rate (mm/h).
    Requires columns: ['link_id','time','A_obs_dB','Frequency','PathLength','pol']
    """
    parts = []
    for lid, g in dfA.groupby("link_id", sort=False):
        g = g.sort_values("time").copy()
        L_km  = float(g["PathLength"].iloc[0])
        f_GHz = float(g["Frequency"].iloc[0])     # your Frequency is already in GHz
        pol   = str(g["pol"].iloc[0])             # 'H' or 'V'
        A_obs = g["A_obs_dB"].astype(float).values

        # Wet-antenna attenuation (vectorized)
        waa = waa_leijnse_2008_from_A_obs(A_obs=A_obs, f_Hz=f_GHz*1e9, pol=pol, L_km=L_km)
        g["A_waa_dB"]  = waa
        g["A_rain_dB"] = np.maximum(0.0, A_obs - waa)

        # Attenuation -> R (mm/h)
        g["R_mm_per_h"] = calc_R_from_A(
            A=g["A_rain_dB"].values,
            L_km=L_km,
            f_GHz=f_GHz,
            pol=pol,
            a_b_approximation="ITU_2005",
            R_min=R_min
        )
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


# ---------- B) Our “consensus/hysteresis” pipeline wrapper (optional) ----------
@dataclass
class ConsensusParams:
    thr_self_db: float = 1.2
    thr_self_db_per_km: float = 0.6
    thr_nb_db: float = 1.4
    thr_nb_db_per_km: float = 0.7
    radius_km: float = 30.0
    min_neighbors: int = 3
    temporal_pad_bins: int = 2
    use_leijnse_WA: bool = True
    gamma_gate_db_per_km: Optional[float] = None
    drizzle_floor: float = 0.0


def run_consensus_pipeline(df_clean: pd.DataFrame, lut, params: ConsensusParams, n_jobs: int = 8) -> pd.DataFrame:
    """
    Calls your existing step1/2/1b/2b/WA/3 with safer defaults and returns df_s5 with 'R_mm_per_h'.
    Keep as before; unchanged from your previous version.
    """
    import importlib
    # S1: Δ-based
    import step1_wetdry_nla_fast2 as s1; importlib.reload(s1)
    cfg1 = s1.NLAConfig(
        radius_km=params.radius_km, min_neighbors=params.min_neighbors,
        thr_self_db=params.thr_self_db, thr_self_db_per_km=params.thr_self_db_per_km,
        thr_nb_db=params.thr_nb_db, thr_nb_db_per_km=params.thr_nb_db_per_km,
        temporal_pad_bins=params.temporal_pad_bins, neighbor_pool="max",
        n_jobs=n_jobs
    )
    df_nla, _ = s1.wetdry_classify_fast(df_clean, cfg1)

    # S2: 48h baseline
    import step2_baseline_dry48 as s2; importlib.reload(s2)
    df_s2, _ = s2.compute_dry_baseline_48h(df_nla, s2.Baseline48Config())

    # S1b: excess NLA
    import step1b_wetdry_excess_fast as s1b; importlib.reload(s1b)
    df_ex, _ = s1b.wetdry_from_excess_fast(df_s2, s1b.ExcessNLAConfig(n_jobs=n_jobs))

    # integrate masks
    import step2b_integrate_masks as s2b; importlib.reload(s2b)
    df_s12 = s2b.integrate_wetdry_and_excess(df_s2, df_nla, df_ex, onset_allow_if_excess_pos=True)

    # A_obs for Leijnse WA from Step 2 output
    if "A_excess_db" in df_s12.columns:
        df_s12["A_obs_dB"] = df_s12["A_excess_db"].clip(lower=0)
    else:
        raise ValueError("Need A_excess_db from Step2 to compute A_obs for Leijnse WA.")

    # Apply Leijnse per link
    out = []
    for lid, g in df_s12.groupby("ID", sort=False):
        g = g.sort_index().copy()
        L_km   = float(pd.to_numeric(g["PathLength"]).iloc[0])
        f_GHz  = float(pd.to_numeric(g["Frequency"]).iloc[0])
        pol    = str(g["Polarization"]).upper()[0]
        A_obs  = pd.to_numeric(g["A_obs_dB"]).fillna(0).values
        waa    = waa_leijnse_2008_from_A_obs(A_obs=A_obs, f_Hz=f_GHz*1e9, pol=pol, L_km=L_km)
        g["A_rain_dB"] = np.maximum(0.0, A_obs - waa)
        g["gamma_for_R"] = g["A_rain_dB"] / max(L_km, 1e-6)
        out.append(g)
    df_wa = pd.concat(out, axis=0)

    # k–α to R (gated disabled by default)
    import step3_kalpha as s3; importlib.reload(s3)
    cfg_k = s3.KAlphaConfigV2(
        lut=lut, pol_col="Polarization", freq_col="Frequency",
        gamma_col="gamma_for_R",
        gamma_gate_db_per_km=None, use_wet_mask_col=None, r_cap_mmph_by_band=None
    )
    df_s5, _ = s3.gamma_to_r_gated(df_wa, cfg_k)
    return df_s5


# ---------- C) Helper to prepare inputs for step6 (ID/time alignment) ----------
def make_grid_inputs_for_s6(df_rate: pd.DataFrame, df_clean: pd.DataFrame):
    """
    Produce the exact inputs step6_grid_ok_pcm.grid_rain_15min expects:
      - df_s5_grid: index = naive UTC DatetimeIndex, cols ['ID','R_mm_per_h']
      - meta_for_grid: one row per ID with ['ID','XStart','YStart','XEnd','YEnd']
    """
    # Rates table → ['ID','R_mm_per_h'] with naive-UTC index named 'time'
    s5 = (df_rate[["link_id","time","R_mm_per_h"]].rename(columns={"link_id":"ID"})).copy()

    t = pd.to_datetime(s5["time"], utc=True, errors="coerce")   # tz-aware UTC
    t = t.dt.tz_convert("UTC").dt.tz_localize(None)             # make it naive-UTC

    s5 = s5.drop(columns=["time"])
    s5.index = t
    s5.index.name = "time"
    s5 = s5.sort_index()

    # Geometry from df_clean (one row per ID)
    meta = df_clean.drop_duplicates("ID")[["ID","XStart","YStart","XEnd","YEnd"]].copy()
    for c in ["XStart","YStart","XEnd","YEnd"]:
        meta[c] = pd.to_numeric(meta[c], errors="coerce")

    return s5, meta


# ---- link_id maker that matches build_15min_timeseries ----
def make_link_id(df: pd.DataFrame) -> pd.Series:
    return (
        df["ID"].astype(str).str.replace(">>","~~", regex=False).str.strip() + "|" +
        df["Polarization"].astype(str).str.upper().str[0] + "|" +
        pd.to_numeric(df["Frequency"], errors="coerce").round(3).astype(str)
    )

# ---- gate by strict wet mask & drizzle floor (zeros become real zeros) ----
def apply_wet_gate_and_drizzle(df_rate: pd.DataFrame,
                               dfA: pd.DataFrame,
                               drizzle: float = 0.20,
                               wet_col: str = "wet_rl") -> pd.DataFrame:
    z = dfA[["link_id","time",wet_col]].rename(columns={wet_col:"wet"})
    out = df_rate.merge(z, on=["link_id","time"], how="left")

    # treat NaN rain as 0 unless explicitly wet AND finite
    R = pd.to_numeric(out["R_mm_per_h"], errors="coerce")
    is_wet = out["wet"] == True
    R_safe = R.where(is_wet & R.notna(), 0.0)

    out["R_mm_per_h"] = np.where(R_safe < float(drizzle), 0.0, R_safe)
    return out.drop(columns=["wet"])

# --- save_slices.py (you can paste this near your step6 code or in a utils file) ---
import os
import numpy as np
import pandas as pd
import xarray as xr

def save_each_time_to_netcdf(
    data,                           # xr.DataArray OR xr.Dataset (with var_name)
    out_dir,
    base_name="ghana_cml_R",
    *,
    var_name="R_mm_per_h",          # ignored if data is a DataArray
    engine="netcdf4",               # or "h5netcdf"
    complevel=9,
    dtype="float32",
    fill_value=np.nan,              # what to write for NaNs
    chunks_lat=256,
    chunks_lon=256,
    keep_time_dim=True,             # keep a size-1 time dimension in each file
):
    """
    Writes one .nc per timestamp with high compression and correct chunking.
    Returns list of written file paths.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Normalize to a DataArray
    if isinstance(data, xr.Dataset):
        if var_name not in data.data_vars and len(data.data_vars) == 1:
            var_name = list(data.data_vars)[0]
        da = data[var_name]
    elif isinstance(data, xr.DataArray):
        var_name = data.name or var_name
        da = data
    else:
        raise TypeError("data must be an xarray DataArray or Dataset")

    # If no time dimension, write a single file and return
    if "time" not in da.dims:
        d2 = da.astype(dtype)
        ds = d2.to_dataset(name=var_name)
        # chunks: match dims exactly
        dims = ds[var_name].dims
        sizes = ds[var_name].sizes
        chunks = []
        for d in dims:
            if d == "lat":
                chunks.append(min(int(chunks_lat), sizes[d]))
            elif d == "lon":
                chunks.append(min(int(chunks_lon), sizes[d]))
            else:
                chunks.append(min(1, sizes[d]))
        enc = {
            var_name: {
                "zlib": True, "complevel": int(complevel), "shuffle": True,
                "dtype": dtype, "_FillValue": fill_value,
                "chunksizes": tuple(chunks),
            }
        }
        fn = os.path.join(out_dir, f"{base_name}.nc")
        ds.to_netcdf(fn, engine=engine, encoding=enc)
        return [fn]

    # Otherwise iterate times
    times = pd.to_datetime(da["time"].values)
    out_paths = []

    for t in times:
        # 2-D slice (lat,lon)
        sl = da.sel(time=np.datetime64(t)).astype(dtype)

        # Keep time dim? -> expand to (time,lat,lon) of length 1
        if keep_time_dim:
            sl = sl.expand_dims(time=[np.datetime64(t)])
        ds = sl.to_dataset(name=var_name)

        # Build chunks tuple that matches EXACT dims order
        dims = ds[var_name].dims             # e.g., ('time','lat','lon') or ('lat','lon')
        sizes = ds[var_name].sizes
        chunks = []
        for d in dims:
            if d == "time":
                chunks.append(1)
            elif d == "lat":
                chunks.append(min(int(chunks_lat), sizes[d]))
            elif d == "lon":
                chunks.append(min(int(chunks_lon), sizes[d]))
            else:
                # Unknown dim: just chunk by its full size
                chunks.append(sizes[d])

        enc = {
            var_name: {
                "zlib": True, "complevel": int(complevel), "shuffle": True,
                "dtype": dtype, "_FillValue": fill_value,
                "chunksizes": tuple(chunks),
            },
            # coords (don’t compress tiny arrays)
            "lat": {"zlib": False},
            "lon": {"zlib": False},
            "time": {"zlib": False},
        }

        # Nice filename stamp: YYYYmmddTHHMMSS
        t_str = pd.Timestamp(t).strftime("%Y%m%dT%H%M%S")
        fn = os.path.join(out_dir, f"{base_name}_{t_str}.nc")

        # Use unlimited time when present
        unlimited = {"time"} if keep_time_dim else None
        ds.to_netcdf(fn, engine=engine, encoding=enc, unlimited_dims=unlimited)
        out_paths.append(fn)

    return out_paths