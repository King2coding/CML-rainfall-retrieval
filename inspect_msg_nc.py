#!/usr/bin/env python3
import xarray as xr
import numpy as np
from pathlib import Path

# === Your Meteosat-9 (MSG FM-2) wavenumbers (cm^-1), inverse-Planck only ===
C1 = 1.191044e-5   # mW/(m^2·sr·cm^-4)
C2 = 1.4387752     # K·cm
WAVENUM = {
    4: 2568.832,    # IR3.9
    5: 1600.548,    # WV6.2
    6: 1360.330,    # WV7.3
    9: 931.700,     # IR10.8
    10: 836.445,    # IR12.0
    11: 751.792,    # IR13.4
}
INDEX_TO_VAR = {5:"channel_5", 9:"channel_9", 10:"channel_10"}

def radiance_to_bt(channel: int, radiance):
    """Inverse Planck: T = (C2*ν) / ln(1 + (C1*ν^3)/L)."""
    nu = WAVENUM[channel]
    L = np.asarray(radiance, np.float64)
    # L = np.where(L > 0, L, np.nan)  # avoid log of <= 0
    return (C2 * nu) / np.log(1.0 + (C1 * (nu**3)) / L)

def parse_offset_slope(val):
    """Parse 'offset slope' string -> (offset, slope) floats."""
    if val is None:
        return None
    s = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
    parts = s.replace(",", " ").split()
    if len(parts) < 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except Exception:
        return None

def looks_like_radiance(da: xr.DataArray) -> bool:
    u = (da.attrs.get("units") or "").lower()
    ln = (da.attrs.get("long_name") or "").lower()
    return ("radiance" in u) or ("mw" in u and "sr" in u and ("cm" in u or "µm" in u or "um" in u))

def get_radiance(da: xr.DataArray, ds_attrs: dict, ch: int) -> xr.DataArray:
    """Return radiance DataArray. If DN, use (offset + slope*DN) from chXX_cal."""
    if looks_like_radiance(da):
        return da.astype("float32")
    key = f"ch{ch:02d}_cal"
    alt = f"ch{ch}_cal"
    pair = parse_offset_slope(ds_attrs.get(key) or ds_attrs.get(alt))
    if pair:
        offset, slope = pair
        rad = offset + slope * da.astype("float32")
        rad.attrs.update(units="mW m-2 sr-1 (cm-1)^-1", note=f"offset+slope from {key if key in ds_attrs else alt}")
        return rad
    # If we get here, we don't know how to convert
    return None

# === CHANGE THIS PATH to your test file ===
nc_path = "/home/kkumah/Projects/RCSP/misc_dat/seviri/HRSEVIRI_20250624T131510Z_20250624T132740Z_epct_43a33a5a_FPC.nc"

ds = xr.open_dataset(nc_path)
print("File:", Path(nc_path).name)
print("radiometric_parameters_format:", ds.attrs.get("radiometric_parameters_format"))

for ch in (9, 10):  # IR10.8 and IR12.0
    vname = INDEX_TO_VAR[ch]
    if vname not in ds:
        print(f" - {vname} missing; skipping.")
        continue
    da = ds[vname]
    rad = get_radiance(da, ds.attrs, ch)
    if rad is None:
        print(f" - {vname}: cannot determine radiance (no units & no ch{ch:02d}_cal).")
        continue

    # Compute BT via your formula
    Tb = xr.DataArray(radiance_to_bt(ch, rad.values), dims=rad.dims, coords=rad.coords)
    Tb.attrs.update(units="K", long_name=f"Brightness Temperature ch{ch}")

    # Basic sanity stats
    def s(a): 
        a = np.asarray(a); return np.nanmin(a), np.nanmean(a), np.nanmax(a)
    bt_min, bt_mean, bt_max = s(Tb)
    r_min, r_mean, r_max = s(rad)

    print(f"Channel {ch} ({vname}):")
    print(f"   Radiance [mW m^-2 sr^-1 (cm^-1)^-1]: min={r_min:.3f}, mean={r_mean:.3f}, max={r_max:.3f}")
    print(f"   BT [K] (inverse Planck):            min={bt_min:.2f}, mean={bt_mean:.2f}, max={bt_max:.2f}")

# OPTIONAL quicklook (comment out if you don’t want plots)
try:
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    for ch in (9, 10):
        vname = INDEX_TO_VAR[ch]
        if vname not in ds: 
            continue
        da = ds[vname]
        rad = get_radiance(da, ds.attrs, ch)
        if rad is None:
            continue
        Tb = xr.DataArray(radiance_to_bt(ch, rad.values), dims=rad.dims, coords=rad.coords)
        ax = plt.axes(projection=ccrs.Geostationary(satellite_height=35785831))
        ax.coastlines(); ax.add_feature(cfeature.BORDERS, linewidth=0.5)
        Tb.plot(ax=ax, transform=ccrs.Geostationary(satellite_height=35785831), cmap="jet")
        plt.title(f"MSG Meteosat-9 BT (ch{ch})")
        plt.show()
except Exception as e:
    print("Plot skipped:", e)

ds.close()