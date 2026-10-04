#!/usr/bin/env python3
"""
Automated HRSEVIRI (MSG) pipeline using eumdac 3.x + Data Tailor

Pipeline
1) Search the Data Store for HRSEVIRI products in a time window.
2) For each product, submit a Data Tailor job with your server-side chain
   (e.g., 'ghana_config') to generate a tailored “FC” NetCDF.
3) Poll until the CLI reports the job is finished (accepts 'DONE' and synonyms).
4) Download the tailored output and extract *_FC.nc.
5) Science post-processing on *_FC.nc:
     - Try DN → Radiance (CF scale/offset OR global (offset, slope) per band)
     - Radiance → Brightness Temperature for WV062 / IR108 / IR120
     - Radiance → Reflectance for VIS006 / VIS008 / NIR016 (when Esun present)
     - Crop by BBOX (only if lon/lat exist; FC often provides x/y only)
6) Write *_CROP.nc (+ *_BT.nc, *_REFL.nc when produced).

Notes
- Satpy is intentionally not used here (FC is not native L1.5).
- The critical fix vs your previous version is the counts_to_radiance() call:
  we now pass ONLY ds.attrs (not ds_crop.attrs).
"""

# ------------------------- stdlib & 3p imports --------------------------------
import os, re, time, zipfile, shutil, subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from eumdac import AccessToken, DataStore

# ------------------------- USER SETTINGS --------------------------------------
OUTDIR = Path("/home/kkumah/Projects/cml-stuff/satellite_data/msg")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Time window (UTC ISO8601)
START = "2025-06-19T15:00:00Z"
END   = "2025-06-19T18:00:00Z"

# Crop box (lon_min, lat_min, lon_max, lat_max); only used when lon/lat present
BBOX: Tuple[float, float, float, float] = (-4.0, 1.5, 4.5, 11.5)

# Tailor chain (you already created/validated it)
TAILOR_CHAIN = "ghana_config" # ghana_clm_config

# Throttle how many Tailor jobs run in parallel
MAX_INFLIGHT = 2

# ------------------------- SEVIRI constants -----------------------------------
# Planck-like constants for EUM SEVIRI usage (radiance in mW m^-2 sr^-1 (cm^-1)^-1)
C1, C2 = 1.19104e-5, 1.43877
SEVIRI_BT_COEFF = {
    "WV062": {"nu_c": 1596.080, "alpha": 0.9959, "beta": 2.0780},  # ch5
    "IR108": {"nu_c": 931.122,  "alpha": 0.9983, "beta": 0.6256},  # ch9
    "IR120": {"nu_c": 839.113,  "alpha": 0.9988, "beta": 0.4002},  # ch10
}

# Tailored FC variable names typically present
INDEX_TO_VAR = {
    1:"channel_1", 2:"channel_2", 3:"channel_3", 4:"channel_4", 5:"channel_5",
    6:"channel_6", 7:"channel_7", 8:"channel_8", 9:"channel_9", 10:"channel_10", 11:"channel_11",
}
INDEX_TO_CODE = {
    1:"VIS006", 2:"VIS008", 3:"NIR016",
    4:"IR039",  5:"WV062",  6:"WV073",
    7:"IR087",  8:"IR097",  9:"IR108", 10:"IR120", 11:"IR134",
}

# ========================= HELPER UTILITIES ===================================
@dataclass
class ProductMini:
    """Minimal product record we pass around."""
    pid: str
    start: Optional[datetime]
    end: Optional[datetime]

def _load_env():
    """Optionally pull credentials from a .env file anywhere above cwd."""
    try:
        from dotenv import load_dotenv
        here = Path.cwd().resolve()
        for p in [here] + list(here.parents):
            f = p / ".env"
            if f.exists():
                load_dotenv(f); print("[env] loaded:", f); break
    except Exception:
        pass

def to_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Return timezone-naive UTC (DataStore search expects naive times)."""
    if dt is None: return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt

def run(args: List[str], timeout=1800, check=True) -> subprocess.CompletedProcess:
    """Run a CLI command with logging; raise if non-zero when check=True."""
    print("[CMD]", " ".join(args))
    res = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if res.stdout.strip(): print(res.stdout.strip())
    if res.stderr.strip(): print(res.stderr.strip())
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode})")
    return res

def load_credentials() -> Tuple[str, str]:
    """Load EUMETSAT consumer key/secret from env or ~/.eumdac/credentials."""
    key = os.environ.get("EUMETSAT_CONSUMER_KEY")
    sec = os.environ.get("EUMETSAT_CONSUMER_SECRET")
    if key and sec: return key, sec
    cred = Path.home()/".eumdac"/"credentials"
    if cred.exists():
        lines = [ln.strip() for ln in cred.read_text().splitlines() if ln.strip()]
        if len(lines) >= 2:
            print(f"[auth] loaded {cred}"); return lines[0], lines[1]
    raise RuntimeError("Missing credentials. Set env or run `eumdac set-credentials`.")

def _safe_times(prod) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Normalize product time attributes across potential naming variants."""
    start = getattr(prod, "start_time", None) or getattr(prod, "begin_time", None) \
         or getattr(prod, "sensing_start", None) or getattr(prod, "dtstart", None)
    end   = getattr(prod, "stop_time", None) or getattr(prod, "end_time", None) \
         or getattr(prod, "sensing_end", None) or getattr(prod, "dtend", None)
    return to_naive_utc(start), to_naive_utc(end)

def search_products(store: DataStore, start_iso: str, end_iso: str) -> List[ProductMini]:
    """Find HRSEVIRI products between START and END (inclusive)."""
    coll = store.get_collection("EO:EUM:DAT:MSG:HRSEVIRI")
    t0 = datetime.fromisoformat(start_iso.replace("Z","+00:00")).astimezone(timezone.utc)
    t1 = datetime.fromisoformat(end_iso.replace("Z","+00:00")).astimezone(timezone.utc)
    found = list(coll.search(dtstart=t0.replace(tzinfo=None), dtend=t1.replace(tzinfo=None)))
    out: List[ProductMini] = []
    for p in found:
        pid = getattr(p, "_id", None) or getattr(p, "id", None) or str(p)
        st, en = _safe_times(p)
        out.append(ProductMini(pid, st, en))
    out.sort(key=lambda k: k.start or datetime.min.replace(tzinfo=None))
    return out

# ========================= DATA TAILOR (CLI) ==================================
def submit_tailor_job(product_id: str, chain: str) -> str:
    """Submit a Data Tailor job. Return the 8-hex job id printed by the CLI."""
    res = run(["eumdac", "tailor", "post",
               "-c", "EO:EUM:DAT:MSG:HRSEVIRI",
               "-p", product_id,
               "--chain", chain], timeout=120)
    m = re.findall(r"\b([a-f0-9]{8})\b", (res.stdout + "\n" + res.stderr))
    if not m:
        raise RuntimeError("Could not find tailor job id in output.")
    job = m[-1]
    print(f"[tailor] submitted job: {job}")
    return job

def poll_tailor(job_id: str, wait_s=8, max_wait=3600):
    """
    Poll a Tailor job until it finishes.
    The eumdac CLI prints 'DONE' when a job is finished — accept that,
    as well as FINISHED/SUCCEEDED/COMPLETED. Fail fast on FAILED/ERROR.
    """
    t0 = time.time()
    done_tokens = ("DONE", "FINISHED", "SUCCEEDED", "COMPLETED")
    fail_tokens = ("FAILED", "ERROR")

    while True:
        res = run(["eumdac", "tailor", "status", job_id], timeout=60, check=False)
        text = (res.stdout + "\n" + res.stderr).upper()

        if any(tok in text for tok in fail_tokens):
            raise RuntimeError(f"[tailor] job {job_id} failed:\n{text}")

        if any(tok in text for tok in done_tokens):
            print(f"[tailor] job {job_id} finished.")
            return

        if time.time() - t0 > max_wait:
            raise RuntimeError(f"[tailor] job {job_id} timed out after {max_wait}s.")
        time.sleep(wait_s)

def download_tailored(job_id: str, outdir: Path) -> Path:
    """
    Download Tailor outputs for a job into outdir.
    Returns the newest file (typically a .zip produced by the chain).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    before = {p for p in outdir.glob("*")}
    run(["eumdac", "tailor", "download", job_id, "-o", str(outdir)], timeout=600)
    time.sleep(1)
    after = {p for p in outdir.glob("*")}
    new = sorted(after - before, key=lambda p: p.stat().st_mtime if p.exists() else 0.0)
    if not new:
        raise RuntimeError(f"[tailor] no files downloaded for job {job_id}")
    print(f"[tailor] new files: {[n.name for n in new]}")
    return new[-1]

def extract_fc_nc(archive: Path, outdir: Path) -> Optional[Path]:
    """
    Extract *_FC.nc out of a Tailor zip (or copy if already .nc).
    Returns the path to the *_FC.nc kept in outdir, or None if not found.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    fc_path: Optional[Path] = None

    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            for n in z.namelist():
                if n.endswith("_FC.nc"):
                    target = outdir / Path(n).name
                    if target.exists():
                        target = target.with_name(f"{target.stem}_dup{int(time.time())}{target.suffix}")
                    z.extract(n, path=outdir)
                    (outdir / n).rename(target)  # move to outdir root
                    fc_path = target
                elif n.lower().endswith((".xml", ".nat", ".nc", ".txt")):
                    try: z.extract(n, path=outdir)
                    except Exception: pass

    elif archive.suffix.lower() == ".nc" and archive.name.endswith("_FC.nc"):
        target = outdir / archive.name
        if target.exists():
            target = target.with_name(f"{target.stem}_dup{int(time.time())}{target.suffix}")
        shutil.copy2(archive, target)
        fc_path = target

    return fc_path

# ========================= SCIENCE HELPERS ====================================
def crop_bbox(ds: xr.Dataset, bbox: Tuple[float,float,float,float]) -> xr.Dataset:
    """
    Crop by BBOX *only* if lon/lat exist (FC often only has x/y).
    Falls back to returning the dataset unchanged.
    """
    lon_name = next((c for c in ds.coords if c.lower() in ("lon","longitude","xlon")), None)
    lat_name = next((c for c in ds.coords if c.lower() in ("lat","latitude","xlat")), None)
    if lon_name and lat_name:
        try:
            lo0, la0, lo1, la1 = bbox
            return ds.sel({lon_name: slice(lo0, lo1), lat_name: slice(la0, la1)})
        except Exception:
            pass
    return ds

def _is_radiance(da: xr.DataArray) -> bool:
    """Heuristic: does this already look like radiance?"""
    u = (da.attrs.get("units") or "").lower()
    ln = (da.attrs.get("long_name") or "").lower()
    return ("radiance" in u or "radiance" in ln
            or ("mw" in u and "sr" in u and ("cm" in u or "um" in u or "µm" in u)))

def _parse_offset_slope(val: object) -> Optional[Tuple[float, float]]:
    """
    Parse strings like '-1.1337868e+01 2.2231114e-01' into (offset, slope).
    Accepts str/bytes/iterables; returns None if it can't parse two floats.
    """
    try:
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return float(val[0]), float(val[1])
        s = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
        parts = s.strip().replace(",", " ").split()
        if len(parts) >= 2:
            return float(parts[0]), float(parts[1])
    except Exception:
        pass
    return None

def counts_to_radiance(
    da: xr.DataArray,
    gattrs: Dict,
    debug_prefix: str = ""
) -> Optional[xr.DataArray]:
    """
    DN -> radiance using (in order):
      1) Per-variable CF scale_factor/add_offset (preferred).
      2) Per-band (offset, slope) from global attrs, e.g. ch10_cal with
         radiometric_parameters_format == 'offset slope'.
      3) (Optional) other single-gain styles (not seen in your FCs).

    Returns a DataArray with radiance-like units, or None if nothing found.
    """
    # 1) CF per-variable
    sc = da.attrs.get("scale_factor")
    off = da.attrs.get("add_offset")
    if sc is not None or off is not None:
        arr = da.data.astype("float32")
        if sc is not None: arr = arr * float(sc)
        if off is not None: arr = arr + float(off)
        return xr.DataArray(
            arr, dims=da.dims, coords=da.coords,
            attrs={"units": "mW m-2 sr-1 (cm-1)-1", "note": "CF scale/offset"}
        )

    # 2) Global (offset, slope) by channel index from keys like ch10_cal
    idx = None
    try:
        idx = int(str(da.name).split("_")[-1])  # "channel_10" -> 10
    except Exception:
        pass

    if idx is not None:
        key_exact = f"ch{idx:02d}_cal"
        key_loose = f"ch{idx}_cal"
        cand = gattrs.get(key_exact, None) if key_exact in gattrs else gattrs.get(key_loose, None)
        if cand is not None:
            os_pair = _parse_offset_slope(cand)
            if os_pair is not None:
                offset, slope = os_pair
                dn = da.data.astype("float32")
                rad = offset + slope * dn
                # (Optional) clamp small negatives if desired:
                # rad = np.maximum(rad, 0.0)
                return xr.DataArray(
                    rad, dims=da.dims, coords=da.coords,
                    attrs={"units": "mW m-2 sr-1 (cm-1)-1",
                           "note": f"offset+slope (ch{idx:02d}_cal)"}
                )

    # Nothing we can use: log a short hint once
    if debug_prefix:
        keys = list(gattrs.keys())[:12]
        print(f"{debug_prefix}no CF scale/offset and no (offset,slope) found. "
              f"Sample global attrs: {[str(k) for k in keys]}")
    return None

def bt_from_radiance(L: np.ndarray, band: str) -> np.ndarray:
    """Radiance -> Brightness Temperature [K] for SEVIRI thermal/WV bands."""
    p = SEVIRI_BT_COEFF[band]
    Le = np.maximum(L, 1e-12)  # guard against non-positive radiance
    term = C1 * (p["nu_c"]**3) / Le + 1.0
    return (C2 * p["nu_c"]) / (p["alpha"] * np.log(term)) - (p["beta"] / p["alpha"])

def earth_sun_distance_au(dt: datetime) -> float:
    """Seasonal approximation for Earth–Sun distance in AU (for reflectance)."""
    doy = int(dt.strftime("%j"))
    return (1.000110 + 0.034221*np.cos(2*np.pi*doy/365.0) + 0.001280*np.sin(2*np.pi*doy/365.0)
            + 0.000719*np.cos(4*np.pi*doy/365.0) + 0.000077*np.sin(4*np.pi*doy/365.0))

def radiance_to_reflectance(rad: xr.DataArray, Esun: Optional[float], dt: Optional[datetime]) -> Optional[xr.DataArray]:
    """TOA reflectance ≈ π * L * d^2 / Esun; only computed when Esun & time are available."""
    if Esun is None or dt is None: return None
    d = earth_sun_distance_au(dt)
    refl = np.pi * np.asarray(rad.data, dtype="float32") * (d**2) / float(Esun)
    return xr.DataArray(refl, dims=rad.dims, coords=rad.coords,
                        attrs={"units":"1", "long_name":"Reflectance"})

def process_fc_nc(fc_nc: Path, bbox: Tuple[float,float,float,float], start_time: Optional[datetime]) -> Dict[str, Path]:
    """
    Open the tailored FC file, crop, compute BT/Reflectance, and write outputs.
    Returns a dict with keys present: crop, bt, refl.
    """
    out: Dict[str, Path] = {}
    ds = xr.open_dataset(fc_nc)

    # 1) Crop (no-op if lon/lat missing)
    ds_crop = crop_bbox(ds, bbox)
    out_crop = fc_nc.with_name(fc_nc.stem + "_CROP.nc")
    ds_crop.to_netcdf(out_crop, encoding={v: {"zlib": True, "complevel": 4} for v in ds_crop.data_vars})
    out["crop"] = out_crop
    print(f"[fc] wrote crop: {out_crop.name}")

    # 2) Brightness Temperature
    bt_vars = {}
    for idx, code in [(5,"WV062"), (9,"IR108"), (10,"IR120")]:
        v = INDEX_TO_VAR[idx]
        if v not in ds_crop:
            print(f"[bt] missing variable {v}; skipped.")
            continue
        da = ds_crop[v]
        if _is_radiance(da):
            rad = da
            print(f"[bt] {v} already looks like radiance (units='{da.attrs.get('units','')}').")
        else:
            # -------- FIXED: pass only ds.attrs and a debug prefix ----------
            rad = counts_to_radiance(da, ds.attrs, debug_prefix=f"[bt] {v} -> ")
        if rad is None:
            print(f"[bt] {v} ({code}): no radiance info; skipped.")
            continue
        bt = bt_from_radiance(rad.data, code)
        bt_vars[f"BT_{code}"] = xr.DataArray(bt, dims=rad.dims, coords=rad.coords,
                                             attrs={"units":"K","long_name":f"Brightness Temperature {code}"})
    if bt_vars:
        out_bt = fc_nc.with_name(fc_nc.stem + "_BT.nc")
        xr.Dataset(bt_vars).to_netcdf(out_bt, encoding={k: {"zlib": True, "complevel": 4} for k in bt_vars})
        out["bt"] = out_bt
        print(f"[fc] wrote BT: {out_bt.name}")
    else:
        print("[bt] none written.")

    # 3) Reflectance (VIS/NIR), only if Esun available
    refl_vars = {}
    for idx, code in [(1,"VIS006"), (2,"VIS008"), (3,"NIR016")]:
        v = INDEX_TO_VAR[idx]
        if v not in ds_crop:
            print(f"[refl] missing variable {v}; skipped.")
            continue
        da = ds_crop[v]
        rad = da if _is_radiance(da) else counts_to_radiance(da, ds.attrs, debug_prefix=f"[refl] {v} -> ")
        if rad is None:
            print(f"[refl] {v} ({code}): no radiance info; skipped.")
            continue
        # Try several places for Esun (FC may not include these; then we skip)
        Esun = None
        for k in ("solar_irradiance","Esun", f"Esun_{idx:02d}", f"Esun_{code.lower()}"):
            if k in da.attrs: Esun = float(da.attrs[k]); break
            if k in ds.attrs: Esun = float(ds.attrs[k]); break
            if k in ds_crop.attrs: Esun = float(ds_crop.attrs[k]); break
        rr = radiance_to_reflectance(rad, Esun, start_time)
        if rr is not None:
            refl_vars[f"REFL_{code}"] = rr
        else:
            print(f"[refl] {v} ({code}): missing Esun/time; skipped.")
    if refl_vars:
        out_refl = fc_nc.with_name(fc_nc.stem + "_REFL.nc")
        xr.Dataset(refl_vars).to_netcdf(out_refl, encoding={k: {"zlib": True, "complevel": 4} for k in refl_vars})
        out["refl"] = out_refl
        print(f"[fc] wrote REFL: {out_refl.name}")

    ds.close()
    return out

# ========================= MAIN ORCHESTRATION =================================
def main():
    _load_env()
    key, sec = load_credentials()
    token = AccessToken((key, sec))
    print("Token OK; expires:", token.expiration)
    store = DataStore(token)

    products = search_products(store, START, END)
    if not products:
        print("[search] No HRSEVIRI products in window.")
        return
    print(f"[search] {len(products)} product(s) in window.")

    run_root = OUTDIR / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)

    inflight: Dict[str, ProductMini] = {}
    queue = products[:]

    while queue or inflight:
        # Top up the queue
        while queue and len(inflight) < MAX_INFLIGHT:
            p = queue.pop(0)
            try:
                job = submit_tailor_job(p.pid, TAILOR_CHAIN)
                inflight[job] = p
            except Exception as e:
                print(f"[tailor] submit failed for {p.pid}: {e}")

        # Poll and process completed jobs
        for job_id, prod in list(inflight.items()):
            try:
                poll_tailor(job_id, wait_s=8)
                tsdir = run_root / (prod.start.strftime("%Y%m%d_%H%M%S") if prod.start else "unknown_time")
                tsdir.mkdir(parents=True, exist_ok=True)
                archive = download_tailored(job_id, tsdir)
                fc = extract_fc_nc(archive, tsdir)
                if fc is None:
                    print(f"[tailor] no *_FC.nc found for job {job_id}")
                else:
                    print("[fc] using:", fc.name)
                    process_fc_nc(fc, BBOX, prod.start)
            except Exception as e:
                print(f"[tailor] job {job_id} failed: {e}")
            finally:
                inflight.pop(job_id, None)

        if inflight:
            time.sleep(5)  # brief breather before next poll loop

if __name__ == "__main__":
    main()