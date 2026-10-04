#!/usr/bin/env python3
"""
Ghana MSG HRSEVIRI → BT/REFL NetCDF pipeline (radiance-only, with time coord)

Flow
1) Find HRSEVIRI products in a time window.
2) Tailor with your server-side chain (e.g., 'ghana_config').
3) Download & extract *_FC.nc (already in radiance).
4) Science:
   - BT for WV062/IR108/IR120 from radiance:
       T = ((C2 * ν) / ln(1 + (C1 * ν^3) / L) - β) / α
     (no clamp; L≤0 → NaN if present)
   - Optional reflectance (VIS/NIR) when Esun+time exist:
       ρ ≈ π * L * d^2 / Esun
   - Crop to bbox (no-op if lon/lat not present)
5) Write *_BT.nc (and *_REFL.nc if requested) with a 1-step `time` dimension.
6) Delete intermediates unless --keep-intermediate.
"""

import argparse
import os, re, time, zipfile, shutil, subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from eumdac import AccessToken, DataStore

# ---------------------------- Constants ---------------------------------------
# Radiation constants (mW/(m^2·sr·cm^-4) and K·cm)
C1, C2 = 1.191044e-5, 1.4387752

# MSG @ 0° (Meteosat-11/MSG-4) – central wavenumber ν [cm^-1] + regression (α, β)
BT_COEFF = {
    "WV062": {"nu": 1596.080, "alpha": 0.9959, "beta": 2.0780},  # ch5
    "IR108": {"nu":  931.122, "alpha": 0.9983, "beta": 0.6256},  # ch9
    "IR120": {"nu":  839.113, "alpha": 0.9988, "beta": 0.4002},  # ch10
}

INDEX_TO_VAR = {i: f"channel_{i}" for i in range(1, 12)}
BAND_LIST = [("WV062", 5), ("IR108", 9), ("IR120", 10)]
VIS_LIST  = [("VIS006", 1), ("VIS008", 2), ("NIR016", 3)]

# ---------------------------- CLI ---------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Download MSG HRSEVIRI (ghana_config), compute BT/reflectance from radiance, crop, and save compact NetCDFs."
    )
    p.add_argument("--start", required=True, help="UTC ISO8601, e.g. 2025-06-19T15:00:00Z")
    p.add_argument("--end",   required=True, help="UTC ISO8601, e.g. 2025-06-19T18:00:00Z")
    p.add_argument("--bbox",  default="-4.0,1.5,4.5,11.5",
                   help="lon_min,lat_min,lon_max,lat_max (used only if lon/lat coords exist)")
    p.add_argument("--outdir", default="/home/kkumah/Projects/cml-stuff/satellite_data/msg",
                   help="Root output directory")
    p.add_argument("--chain", default="ghana_config", help="Data Tailor chain")
    p.add_argument("--max-inflight", type=int, default=2, help="Max concurrent tailor jobs")
    p.add_argument("--reflectance", action="store_true",
                   help="Also compute VIS/NIR reflectance (requires Esun+time)")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="Keep FC/ZIP/NAT intermediates")
    return p.parse_args()

# ---------------------------- Utility -----------------------------------------
@dataclass
class ProductMini:
    pid: str
    start: Optional[datetime]
    end: Optional[datetime]

def _load_env():
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
    if dt is None: return None
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt

def run(args: List[str], timeout=1800, check=True) -> subprocess.CompletedProcess:
    print("[CMD]", " ".join(args))
    res = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if res.stdout.strip(): print(res.stdout.strip())
    if res.stderr.strip(): print(res.stderr.strip())
    if check and res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode})")
    return res

def load_credentials() -> Tuple[str, str]:
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
    start = getattr(prod, "start_time", None) or getattr(prod, "begin_time", None) \
         or getattr(prod, "sensing_start", None) or getattr(prod, "dtstart", None)
    end   = getattr(prod, "stop_time", None) or getattr(prod, "end_time", None) \
         or getattr(prod, "sensing_end", None) or getattr(prod, "dtend", None)
    return to_naive_utc(start), to_naive_utc(end)

def search_products(store: DataStore, start_iso: str, end_iso: str) -> List[ProductMini]:
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

# ---------------------------- Data Tailor CLI ---------------------------------
def submit_tailor_job(product_id: str, chain: str) -> str:
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
                    (outdir / n).rename(target)
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

# ---------------------------- Science -----------------------------------------
def parse_fc_time(ds: xr.Dataset, fallback: Optional[datetime]) -> Optional[datetime]:
    """Prefer FC attribute EPCT_start_sensing_time like 20250619T150010Z; else fallback."""
    val = ds.attrs.get("EPCT_start_sensing_time")
    if isinstance(val, str) and val:
        try:
            # Example format: 20250619T150010Z
            return datetime.strptime(val, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return fallback

def add_time_dim(ds: xr.Dataset, t: Optional[datetime]) -> xr.Dataset:
    """Return dataset with a length-1 time dimension (CF-compliant encoding)."""
    if t is None:
        return ds  # if time is unknown, leave as is
    t64 = np.datetime64(t.astimezone(timezone.utc).replace(tzinfo=None))
    ds2 = ds.expand_dims(time=[t64])
    if "time" in ds2.coords:
        ds2["time"].attrs.update(standard_name="time", long_name="Time")
    return ds2

def crop_bbox(ds: xr.Dataset, bbox: Tuple[float,float,float,float]) -> xr.Dataset:
    lon_name = next((c for c in ds.coords if c.lower() in ("lon","longitude","xlon")), None)
    lat_name = next((c for c in ds.coords if c.lower() in ("lat","latitude","xlat")), None)
    if lon_name and lat_name:
        lo0, la0, lo1, la1 = bbox
        try:
            return ds.sel({lon_name: slice(lo0, lo1), lat_name: slice(la0, la1)})
        except Exception:
            pass
    return ds

def radiance_to_bt(L: np.ndarray, band_code: str) -> np.ndarray:
    """Inverse Planck with MSG regression (no guards/clamps)."""
    p = BT_COEFF[band_code]
    nu = p["nu"]; alpha, beta = p["alpha"], p["beta"]
    term = C1 * (nu**3) / L + 1.0
    return (C2 * nu / np.log(term) - beta) / alpha

def earth_sun_distance_au(dt: datetime) -> float:
    doy = int(dt.strftime("%j"))
    return (1.000110 + 0.034221*np.cos(2*np.pi*doy/365.0) + 0.001280*np.sin(2*np.pi*doy/365.0)
            + 0.000719*np.cos(4*np.pi*doy/365.0) + 0.000077*np.sin(4*np.pi*doy/365.0))

def radiance_to_reflectance(rad: xr.DataArray, Esun: Optional[float], dt: Optional[datetime]) -> Optional[xr.DataArray]:
    if Esun is None or dt is None: return None
    d = earth_sun_distance_au(dt)
    refl = np.pi * np.asarray(rad.data, dtype="float32") * (d**2) / float(Esun)
    return xr.DataArray(refl, dims=rad.dims, coords=rad.coords,
                        attrs={"units":"1", "long_name":"Reflectance"})

# ---------------------------- Processing --------------------------------------
def process_fc_nc(fc_nc: Path, bbox: Tuple[float,float,float,float],
                  start_time_hint: Optional[datetime],
                  want_reflectance: bool) -> Dict[str, Path]:
    """
    Open FC (radiance), crop, add time, compute BT (+ optional REFL), and write.
    Returns dict with keys present: {'bt': Path, 'refl': Path}
    """
    out: Dict[str, Path] = {}
    ds = xr.open_dataset(fc_nc)

    # Resolve time
    fc_time = parse_fc_time(ds, start_time_hint)

    # Crop (no-op if lon/lat missing)
    ds_crop = crop_bbox(ds, bbox)

    # ---- Brightness Temperature (WV062/IR108/IR120) ----
    bt_vars = {}
    for band, ch in BAND_LIST:
        varname = INDEX_TO_VAR[ch]
        if varname not in ds_crop:
            print(f"[bt] missing {varname}; skipped.")
            continue
        rad = ds_crop[varname]  # already radiance
        bt = radiance_to_bt(rad.data.astype("float32"), band)
        bt_vars[f"BT_{band}"] = xr.DataArray(
            bt, dims=rad.dims, coords=rad.coords,
            attrs={"units": "K", "long_name": f"Brightness Temperature {band}"}
        )
    if bt_vars:
        ds_bt = xr.Dataset(bt_vars)
        ds_bt = add_time_dim(ds_bt, fc_time)
        out_bt = fc_nc.with_name(fc_nc.stem + "_BT.nc")
        enc = {k: {"zlib": True, "complevel": 4} for k in ds_bt.data_vars}
        ds_bt.to_netcdf(out_bt, encoding=enc)
        out["bt"] = out_bt
        print(f"[fc] wrote BT: {out_bt.name}")
    else:
        print("[bt] none written.")

    # ---- Reflectance (VIS/NIR) optional ----
    if want_reflectance:
        refl_vars = {}
        for band, ch in VIS_LIST:
            varname = INDEX_TO_VAR[ch]
            if varname not in ds_crop:
                print(f"[refl] missing {varname}; skipped.")
                continue
            rad = ds_crop[varname]  # radiance
            # Try several places for Esun
            Esun = None
            for k in ("solar_irradiance", "Esun", f"Esun_{ch:02d}", f"Esun_{band.lower()}"):
                if k in rad.attrs: Esun = float(rad.attrs[k]); break
                if k in ds.attrs:  Esun = float(ds.attrs[k]);  break
                if k in ds_crop.attrs: Esun = float(ds_crop.attrs[k]); break
            rr = radiance_to_reflectance(rad, Esun, fc_time)
            if rr is not None:
                refl_vars[f"REFL_{band}"] = rr
            else:
                print(f"[refl] {varname}: missing Esun/time; skipped.")
        if refl_vars:
            ds_refl = xr.Dataset(refl_vars)
            ds_refl = add_time_dim(ds_refl, fc_time)
            out_refl = fc_nc.with_name(fc_nc.stem + "_REFL.nc")
            enc = {k: {"zlib": True, "complevel": 4} for k in ds_refl.data_vars}
            ds_refl.to_netcdf(out_refl, encoding=enc)
            out["refl"] = out_refl
            print(f"[fc] wrote REFL: {out_refl.name}")

    ds.close()
    return out

# ---------------------------- Main --------------------------------------------
def main():
    args = parse_args()

    lo0, la0, lo1, la1 = map(float, args.bbox.split(","))
    bbox = (lo0, la0, lo1, la1)

    outroot = Path(args.outdir); outroot.mkdir(parents=True, exist_ok=True)

    _load_env()
    key, sec = load_credentials()
    token = AccessToken((key, sec))
    print("Token OK; expires:", token.expiration)
    store = DataStore(token)

    products = search_products(store, args.start, args.end)
    if not products:
        print("[search] No HRSEVIRI products in window."); return
    print(f"[search] {len(products)} product(s) in window.")

    run_root = outroot / f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)

    inflight: Dict[str, ProductMini] = {}
    queue = products[:]

    while queue or inflight:
        while queue and len(inflight) < args.max_inflight:
            p = queue.pop(0)
            try:
                job = submit_tailor_job(p.pid, args.chain)
                inflight[job] = p
            except Exception as e:
                print(f"[tailor] submit failed for {p.pid}: {e}")

        for job_id, prod in list(inflight.items()):
            tsdir = run_root / (prod.start.strftime("%Y%m%d_%H%M%S") if prod.start else "unknown_time")
            try:
                poll_tailor(job_id, wait_s=8)
                tsdir.mkdir(parents=True, exist_ok=True)
                archive = download_tailored(job_id, tsdir)
                fc = extract_fc_nc(archive, tsdir)
                if fc is None:
                    print(f"[tailor] no *_FC.nc found for job {job_id}")
                else:
                    print("[fc] using:", fc.name)
                    _ = process_fc_nc(fc, bbox, prod.start, args.reflectance)

                    if not args.keep_intermediate:
                        try:
                            if fc.exists(): fc.unlink()
                            if archive.exists(): archive.unlink()
                            for side in tsdir.glob("*.nat"): side.unlink()
                            for side in tsdir.glob("*.xml"): side.unlink()
                            for side in tsdir.glob("*.txt"): side.unlink()
                        except Exception as e:
                            print("[clean] warning:", e)
                    else:
                        print("[clean] keeping intermediate FC/ZIP/NAT.")
            except Exception as e:
                print(f"[tailor] job {job_id} failed: {e}")
            finally:
                inflight.pop(job_id, None)

        if inflight:
            time.sleep(5)

if __name__ == "__main__":
    main()