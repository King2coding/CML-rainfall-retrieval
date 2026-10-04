#!/usr/bin/env python3
"""
Ghana MSG Cloud Mask (CLM) → cropped NetCDF pipeline (eumdac 3.x + Data Tailor)

What it does
1) Searches EO:EUM:DAT:MSG:CLM for products in a UTC time window.
2) Submits EUMETSAT Data Tailor jobs using your saved chain (e.g., "ghana_clm_config").
3) Polls until done, downloads results (ZIP or .nc).
4) Extracts the CLM NetCDF, optionally crops to Ghana bbox (if lon/lat are present),
   adds a 'time' coordinate from metadata, and saves a compact *_CLM.nc file.
5) Cleans intermediate files unless --keep-intermediate is set.

No radiance/BT/reflection conversions are done—CLM is used as delivered.

Usage example
python Ghana_cloud_mask_data_download_pipeline.py \
  --start 2025-06-19T15:00:00Z \
  --end   2025-06-19T18:00:00Z \
  --bbox "-4.0,1.5,4.5,11.5" \
  --outdir /home/kkumah/Projects/cml-stuff/satellite_data/msg_clm \
  --chain ghana_clm_config \
  --max-inflight 2 \
  --keep-intermediate
"""

import argparse
import os, re, time, zipfile, shutil, subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import xarray as xr
from eumdac import AccessToken, DataStore

# ---------------------------- CLI ---------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Download MSG Cloud Mask (CLM) with a Data Tailor chain, crop to Ghana, and save compact NetCDFs."
    )
    p.add_argument("--start", required=True, help="UTC ISO8601, e.g. 2025-06-19T15:00:00Z")
    p.add_argument("--end",   required=True, help="UTC ISO8601, e.g. 2025-06-19T18:00:00Z")
    p.add_argument("--bbox", default="-4.0,1.5,4.5,11.5",
                   help="lon_min,lat_min,lon_max,lat_max (used only if lon/lat coords exist)")
    p.add_argument("--outdir", default="/home/kkumah/Projects/cml-stuff/satellite_data/msg_clm",
                   help="Root output directory")
    p.add_argument("--chain", default="ghana_clm_config", help="Data Tailor chain name")
    p.add_argument("--max-inflight", type=int, default=2, help="Max concurrent Data Tailor jobs")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="Keep ZIP/.nat/.xml (and any raw .nc) next to outputs")
    return p.parse_args()

# ---------------------------- Utils -------------------------------------------
@dataclass
class ProductMini:
    pid: str
    start: Optional[datetime]
    end: Optional[datetime]

def _load_env():
    """Optionally load .env with EUMETSAT creds."""
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
    coll = store.get_collection("EO:EUM:DAT:MSG:CLM")
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

# ---------------------------- Data Tailor -------------------------------------
def submit_tailor_job(product_id: str, chain: str) -> str:
    res = run(["eumdac", "tailor", "post",
               "-c", "EO:EUM:DAT:MSG:CLM",
               "-p", product_id,
               "--chain", chain], timeout=120)
    # job ids are short hex tokens printed by the CLI
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

def extract_first_nc(archive: Path, outdir: Path) -> Optional[Path]:
    """Return the first .nc inside archive (ZIP) or copy if already .nc."""
    outdir.mkdir(parents=True, exist_ok=True)
    nc_path: Optional[Path] = None
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as z:
            # prefer *_CLM.nc if present
            names = z.namelist()
            preferred = [n for n in names if n.lower().endswith("_clm.nc")]
            candidates = preferred or [n for n in names if n.lower().endswith(".nc")]
            for n in candidates:
                target = outdir / Path(n).name
                if target.exists():
                    target = target.with_name(f"{target.stem}_dup{int(time.time())}{target.suffix}")
                z.extract(n, path=outdir)
                (outdir / n).rename(target)
                nc_path = target
                break
            # optionally extract small sidecar files for provenance
            for n in names:
                if n.lower().endswith((".xml", ".nat", ".txt")):
                    try: z.extract(n, path=outdir)
                    except Exception: pass
    elif archive.suffix.lower() == ".nc":
        target = outdir / archive.name
        if target.exists():
            target = target.with_name(f"{target.stem}_dup{int(time.time())}{target.suffix}")
        shutil.copy2(archive, target)
        nc_path = target
    return nc_path

# ---------------------------- Science-lite ------------------------------------
def crop_bbox(ds: xr.Dataset, bbox: Tuple[float,float,float,float]) -> xr.Dataset:
    lon_name = next((c for c in ds.coords if c.lower() in ("lon","longitude","xlon")), None)
    lat_name = next((c for c in ds.coords if c.lower() in ("lat","latitude","xlat")), None)
    if lon_name and lat_name:
        try:
            lo0, la0, lo1, la1 = bbox
            return ds.sel({lon_name: slice(lo0, lo1), lat_name: slice(la0, la1)})
        except Exception:
            pass
    return ds

def add_time_coord(ds: xr.Dataset) -> xr.Dataset:
    # Try common attrs added by Data Tailor; fall back to None
    # Examples: 'EPCT_start_sensing_time' like '20250619T150010Z' or 'date_time' like '20250619/15:12'
    t = None
    for k in ("EPCT_start_sensing_time", "EPCT_sensing_start", "date_time"):
        if k in ds.attrs and ds.attrs[k]:
            raw = str(ds.attrs[k]).strip()
            try:
                if "T" in raw and raw.endswith("Z"):
                    t = datetime.strptime(raw, "%Y%m%dT%H%M%SZ")
                elif "/" in raw:  # e.g. 20250619/15:12
                    t = datetime.strptime(raw, "%Y%m%d/%H:%M")
                else:
                    t = pd.to_datetime(raw, utc=True).to_pydatetime()
                break
            except Exception:
                continue
    if t is None:
        return ds
    # Insert a scalar time coordinate (DataArray with one element)
    t64 = np.datetime64(pd.to_datetime(t, utc=True))
    return ds.assign_coords(time=("time", [t64]))

# ---------------------------- Processing --------------------------------------
import numpy as np

def process_clm_nc(nc_path: Path, bbox: Tuple[float,float,float,float]) -> Path:
    """
    Open CLM NetCDF, crop if lon/lat exist, add a scalar 'time' coordinate if available,
    and write *_CLM.nc alongside.
    """
    ds = xr.open_dataset(nc_path)
    ds2 = crop_bbox(ds, bbox)
    ds2 = add_time_coord(ds2)

    out_nc = nc_path.with_name(nc_path.stem + "_CLM.nc")
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds2.data_vars}
    # Make sure we also compress coordinates if they are large
    for c in ds2.coords:
        if c not in encoding: encoding[c] = {"zlib": True, "complevel": 4}
    ds2.to_netcdf(out_nc, encoding=encoding)
    ds.close(); ds2.close()
    print(f"[clm] wrote: {out_nc.name}")
    return out_nc

# ---------------------------- Main --------------------------------------------
def main():
    args = parse_args()

    # parse bbox string
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
        print("[search] No CLM products in window."); return
    print(f"[search] {len(products)} CLM product(s) in window.")

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
                nc = extract_first_nc(archive, tsdir)
                if nc is None:
                    print(f"[tailor] no .nc found for job {job_id}")
                else:
                    print("[clm] using:", nc.name)
                    _ = process_clm_nc(nc, bbox)

                    if not args.keep_intermediate:
                        try:
                            if nc.exists(): nc.unlink()  # keep only the compressed *_CLM.nc
                            if archive.exists(): archive.unlink()
                            for ext in ("*.nat", "*.xml", "*.txt"):
                                for f in tsdir.glob(ext): f.unlink()
                        except Exception as e:
                            print("[clean] warning:", e)
                    else:
                        print("[clean] keeping intermediate ZIP/NAT/XML/TXT/raw .nc.")
            except Exception as e:
                print(f"[tailor] job {job_id} failed: {e}")
            finally:
                inflight.pop(job_id, None)

        if inflight:
            time.sleep(5)

if __name__ == "__main__":
    main()