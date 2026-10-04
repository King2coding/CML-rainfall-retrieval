# step6_grid_ok_pcm.py
# ===============================================================
# Gridding 15-min CML rain to lon/lat
# - Keep zeros; map drizzle to 0.0
# - Gate Ordinary Kriging (OK) and fallback to KNN-IDW (hard radius)
# - Wet-only footprint + coverage cleaning (binary opening)
# - Optional link-length weights in IDW
# - Gentle in-mask smoothing (no bleed)
# - RainLINK-style OK variant (wet+dry0 training & strict support)
# - Process/Thread parallelism with controllable KDTree workers
# - Single-time helper and diagnostics
# ===============================================================

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr

from joblib import Parallel, delayed, parallel_backend
from scipy.spatial import cKDTree
from scipy.ndimage import uniform_filter, binary_opening, generate_binary_structure

try:
    from pykrige.ok import OrdinaryKriging
    _PYKRIGE_AVAILABLE = True
except Exception:
    _PYKRIGE_AVAILABLE = False

# For RainLINK-like support mask
try:
    from sklearn.neighbors import BallTree
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

_EPS = 1e-12
_KM_PER_DEG = 111.0
_EARTH_R_KM = 6371.0


# ---------------- geometry ----------------
def _km_factors(lat0_deg: float) -> tuple[float, float]:
    lat0r = np.deg2rad(float(lat0_deg))
    kx = 111.0 * max(0.2, np.cos(lat0r))
    ky = 111.0
    return kx, ky

def _lonlat_to_km(lon, lat, lon0, lat0):
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    kx, ky = _km_factors(lat0)
    return (lon - lon0) * kx, (lat - lat0) * ky


# ---------------- masks & smoothing ----------------
def _nearest_distance_mask(
    lon_vec, lat_vec, x_obs, y_obs, max_dist_km: float, workers: int | None = None
) -> np.ndarray:
    """True where nearest observed point within max_dist_km (euclidean in km)."""
    XX, YY = np.meshgrid(lon_vec, lat_vec)
    lon0 = float((lon_vec.min() + lon_vec.max()) / 2.0)
    lat0 = float((lat_vec.min() + lat_vec.max()) / 2.0)
    Pkx, Pky = _lonlat_to_km(XX, YY, lon0, lat0)

    if len(x_obs) == 0:
        return np.zeros_like(XX, dtype=bool)

    Qkx, Qky = _lonlat_to_km(x_obs, y_obs, lon0, lat0)
    tree = cKDTree(np.c_[Qkx.ravel(), Qky.ravel()])
    kw = {} if workers is None else {"workers": int(workers)}
    dmin, _ = tree.query(np.c_[Pkx.ravel(), Pky.ravel()], k=1, **kw)
    return (dmin.reshape(XX.shape) <= float(max_dist_km))


def _clean_wet_mask(mask: np.ndarray) -> np.ndarray:
    st = generate_binary_structure(2, 1)  # 4-neighborhood
    return binary_opening(mask, structure=st)


def _smooth_in_mask(Z, mask, kernel_px: int = 3) -> np.ndarray:
    """Box-filter smoothing ONLY where mask is True; no bleed outside."""
    if kernel_px is None or int(kernel_px) <= 1:
        return Z
    Z = np.asarray(Z, float); mask = mask.astype(bool)
    Z0 = np.nan_to_num(Z, nan=0.0); W = mask.astype(float)
    num = uniform_filter(Z0, size=int(kernel_px), mode="nearest")
    den = uniform_filter(W,  size=int(kernel_px), mode="nearest")
    Zs = np.where(den > 0, num / np.maximum(den, _EPS), np.nan)
    return np.where(mask, Zs, np.nan)


# ---------------- interpolators ----------------
def _idw_on_grid_km_weighted(
    xkm, ykm, z, Xkm, Ykm, nnear, power, maxdist_km, w_pts=None, workers: int | None = None
):
    """KNN-IDW in km space with optional per-point weights (e.g., sqrt(PathLength))."""
    wet = np.isfinite(z) & (z > 0.0)
    if wet.sum() == 0:
        return np.full_like(Xkm, np.nan, dtype=float)

    pts  = np.c_[xkm[wet], ykm[wet]]
    vals = z[wet].astype(float)
    wp   = np.ones_like(vals) if w_pts is None else np.asarray(w_pts, float)[wet]

    tree = cKDTree(pts)
    k = int(max(1, nnear))
    kw = {"distance_upper_bound": float(maxdist_km)}
    if workers is not None:
        kw["workers"] = int(workers)

    d, idx = tree.query(np.c_[Xkm.ravel(), Ykm.ravel()], k=k, **kw)
    if k == 1:
        d = d[:, None]; idx = idx[:, None]

    valid = np.isfinite(d)
    w_spatial = np.where(valid, 1.0 / np.maximum(d, _EPS) ** float(power), 0.0)

    # pad arrays to handle "no neighbor within radius" (idx == len(vals))
    vals_pad = np.concatenate([vals, [0.0]])
    wp_pad   = np.concatenate([wp,   [0.0]])
    idx = np.where(idx == len(vals), len(vals_pad) - 1, idx)

    num = np.sum(w_spatial * (vals_pad[idx] * wp_pad[idx]), axis=1)
    den = np.sum(w_spatial * wp_pad[idx], axis=1)
    out = np.where(den > 0, num / den, np.nan)
    return out.reshape(Xkm.shape)


def _try_ok_on_grid_km(
    xkm, ykm, z, Xkm, Ykm, *, nlags=10, range_km=25.0, nugget_frac=0.05
):
    """Ordinary Kriging in Euclidean km space with simple exponential variogram."""
    if not _PYKRIGE_AVAILABLE:
        raise RuntimeError("PyKrige not available")

    wet = np.isfinite(z) & (z > 0.0)
    if wet.sum() < 3:
        raise RuntimeError("not enough wet points for OK")

    z_w = z[wet]
    var = float(np.nanvar(z_w))
    if var <= 0:
        raise RuntimeError("zero variance – OK ill-posed")

    nugget = max(1e-6, float(nugget_frac) * var)
    sill   = max(1e-6, var)

    ok = OrdinaryKriging(
        xkm[wet], ykm[wet], z_w,
        variogram_model="exponential",
        variogram_parameters={"nugget": nugget, "sill": sill, "range": float(range_km)},
        nlags=int(nlags), enable_plotting=False, coordinates_type="euclidean",
    )
    xvec = np.unique(Xkm[0, :]); yvec = np.unique(Ykm[:, 0])
    zhat, _ = ok.execute("grid", xvec, yvec)
    return np.asarray(zhat, float)


# ---------------- main API (OK-gated + IDW fallback) ----------------
def grid_rain_15min(
    df_s5,
    df_meta_for_xy,
    *,
    grid_res_deg: float = 0.03,
    domain_pad_deg: float = 0.2,
    drizzle_to_zero: float = 0.1,
    use_ok: bool = True,
    min_pts_ok: int = 20,
    nlags: int = 10,
    ok_range_km: float = 25.0,
    ok_nugget_frac: float = 0.05,
    ok_max_train: int | None = None,
    idw_power: float = 2.0,
    idw_nnear: int = 15,
    idw_maxdist_km: float = 25.0,
    max_dist_km_mask: float = 28.0,
    smooth_kernel_px: int = 3,
    n_jobs: int = 1,
    times_sel=None,
    parallel_backend_name: str = "processes",
    kdtree_workers: int | None = 1,
    collocate_bin_km: float = 2.0,
    use_pathlength_weights: bool = True,
    # NEW fill controls (defaults = NaN)
    outside_support_fill=np.nan,           # value outside wet-footprint support
    interior_no_neighbor_fill=np.nan,      # value for interior “holes” (no neighbors within radius)
    insufficient_training_fill=np.nan,     # whole grid when no wet pts or cannot train
):
    """
    df_s5: index=DatetimeIndex (naive UTC), columns: ["ID","R_mm_per_h", ...]
    df_meta_for_xy: columns: ID, XStart, YStart, XEnd, YEnd

    NOTE: We no longer force NaNs to 0.0. Small finite drizzle values are floored to 0.0,
    but unknowns remain NaN unless you override the *_fill parameters.
    """
    df_s5 = df_s5.copy()
    df_s5["R_mm_per_h"] = pd.to_numeric(df_s5["R_mm_per_h"], errors="coerce")

    # Drizzle → 0.0 (only where finite and positive)
    if drizzle_to_zero is not None:
        v = df_s5["R_mm_per_h"].to_numpy(float)
        mask = np.isfinite(v) & (v > 0.0) & (v < float(drizzle_to_zero))
        v[mask] = 0.0
        df_s5["R_mm_per_h"] = v

    # index hygiene
    idx = df_s5.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError("df_s5.index must be a DatetimeIndex")
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    df_s5.index = idx
    df_s5.index.name = "time"

    # time selection
    all_times = df_s5.index.unique().sort_values()
    if times_sel is not None:
        req = pd.to_datetime(np.atleast_1d(times_sel))
        req_naive = []
        for tt in req:
            tt = pd.Timestamp(tt)
            if tt.tzinfo is not None:
                tt = tt.tz_convert("UTC").tz_localize(None)
            req_naive.append(tt)
        times = pd.DatetimeIndex(req_naive).intersection(all_times)
        if len(times) == 0:
            raise ValueError("times_sel did not match any times in df_s5.index.")
    else:
        times = all_times

    # meta → midpoints
    m = df_meta_for_xy.dropna(subset=["XStart","YStart","XEnd","YEnd"]).copy()
    for c in ["XStart","YStart","XEnd","YEnd","PathLength"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    m["lon"] = 0.5*(m["XStart"] + m["XEnd"])
    m["lat"] = 0.5*(m["YStart"] + m["YEnd"])

    # grid extent
    lon_min = float(np.nanmin(m["lon"])) - float(domain_pad_deg)
    lon_max = float(np.nanmax(m["lon"])) + float(domain_pad_deg)
    lat_min = float(np.nanmin(m["lat"])) - float(domain_pad_deg)
    lat_max = float(np.nanmax(m["lat"])) + float(domain_pad_deg)

    lon = np.arange(lon_min, lon_max + 1e-9, float(grid_res_deg))
    lat = np.arange(lat_min, lat_max + 1e-9, float(grid_res_deg))

    lon0 = float((lon.min() + lon.max())/2.0)
    lat0 = float((lat.min() + lat.max())/2.0)
    LON, LAT = np.meshgrid(lon, lat)
    Xkm, Ykm = _lonlat_to_km(LON, LAT, lon0, lat0)

    methods_used = []
    n_wet_list  = []

    def _do_time(t):
        g = df_s5.loc[df_s5.index == t, ["ID","R_mm_per_h"]].copy()
        g = g.merge(m[["ID","lon","lat","PathLength"]] if "PathLength" in m.columns else m[["ID","lon","lat"]],
                    on="ID", how="inner")
        g["lon"] = pd.to_numeric(g["lon"], errors="coerce")
        g["lat"] = pd.to_numeric(g["lat"], errors="coerce")
        g["R_mm_per_h"] = pd.to_numeric(g["R_mm_per_h"], errors="coerce")
        g = g.dropna(subset=["lon","lat","R_mm_per_h"])

        x = g["lon"].to_numpy(float); y = g["lat"].to_numpy(float); z = g["R_mm_per_h"].to_numpy(float)
        xkm, ykm = _lonlat_to_km(x, y, lon0, lat0)

        # If no usable points at all → whole grid = insufficient_training_fill
        if len(xkm) == 0:
            Z = np.full_like(LON, insufficient_training_fill, float)
            return Z, "no_info", 0

        # deduplicate near-collocated links (median)
        if len(xkm):
            dfp = pd.DataFrame({"x": xkm, "y": ykm, "z": z})
            bin_km = float(collocate_bin_km)
            bx = np.round(dfp["x"]/bin_km).astype(int)
            by = np.round(dfp["y"]/bin_km).astype(int)
            dfp["_bin"] = list(zip(bx, by))
            gmed = dfp.groupby("_bin", as_index=False).median(numeric_only=True)
            xkm_d = gmed["x"].to_numpy(); ykm_d = gmed["y"].to_numpy(); z_d = gmed["z"].to_numpy()
        else:
            xkm_d = xkm; ykm_d = ykm; z_d = z

        # wet-only footprint (strictly positive)
        wet_pts = np.isfinite(z) & (z > 0.0)
        if wet_pts.sum() < 1:
            # no wet info → whole grid = insufficient_training_fill
            Z = np.full_like(LON, insufficient_training_fill, float)
            return Z, "no_wet", 0

        x_w, y_w = x[wet_pts], y[wet_pts]
        covmask_wet = _nearest_distance_mask(
            lon, lat, x_w, y_w, max_dist_km=max_dist_km_mask, workers=kdtree_workers
        )
        covmask_wet = _clean_wet_mask(covmask_wet)

        # per-point weights for IDW (sqrt(PathLength) if available)
        w_pts = None
        if use_pathlength_weights and "PathLength" in g.columns:
            w_pts = np.sqrt(np.clip(pd.to_numeric(g["PathLength"], errors="coerce").to_numpy(float), 1.0, None))

        used = "idw_knn"
        Z = np.full_like(LON, np.nan, dtype=float)
        n_wet = int((z_d > 0.0).sum())

        if use_ok and _PYKRIGE_AVAILABLE and (n_wet >= int(min_pts_ok)):
            try:
                Z = _try_ok_on_grid_km(
                    xkm_d, ykm_d, z_d, Xkm, Ykm,
                    nlags=int(nlags), range_km=float(ok_range_km), nugget_frac=float(ok_nugget_frac)
                )
                used = "ok_pykrige"
            except Exception:
                Z = _idw_on_grid_km_weighted(
                    xkm_d, ykm_d, z_d, Xkm, Ykm,
                    nnear=int(idw_nnear), power=float(idw_power),
                    maxdist_km=float(idw_maxdist_km), w_pts=w_pts, workers=kdtree_workers
                )
                used = "idw_knn"
        else:
            Z = _idw_on_grid_km_weighted(
                xkm_d, ykm_d, z_d, Xkm, Ykm,
                nnear=int(idw_nnear), power=float(idw_power),
                maxdist_km=float(idw_maxdist_km), w_pts=w_pts, workers=kdtree_workers
            )
            used = "idw_knn"

        # Outside support → fill (default NaN)
        if np.isnan(outside_support_fill):
            Z[~covmask_wet] = np.nan
        else:
            Z[~covmask_wet] = float(outside_support_fill)

        # Smooth ONLY where we have values (no bleed)
        valid_mask = np.isfinite(Z)
        Z = _smooth_in_mask(Z, valid_mask, kernel_px=smooth_kernel_px)

        # Interior “holes” (inside support but still NaN) → fill (default NaN)
        inside_holes = covmask_wet & ~np.isfinite(Z)
        if np.isnan(interior_no_neighbor_fill):
            # leave as NaN
            pass
        else:
            Z[inside_holes] = float(interior_no_neighbor_fill)

        return Z, used, n_wet

    backend = "loky" if parallel_backend_name == "processes" else "threading"
    with parallel_backend(backend):
        res = Parallel(n_jobs=int(n_jobs))(delayed(_do_time)(t) for t in times)

    grids    = [r[0] for r in res]
    methods  = [r[1] for r in res]
    n_wet_l  = [r[2] for r in res]

    da = xr.DataArray(
        np.stack(grids),
        coords={"time": times.values.astype("datetime64[ns]"), "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="R_mm_per_h",
    )
    da.attrs.update(dict(
        units="mm h-1",
        method="OK→IDW (gated) + cleaned wet-footprint + box-mean smooth",
        grid_res_deg=float(grid_res_deg),
    ))

    counts = {mth: methods.count(mth) for mth in set(methods)}
    first5 = []
    for i in range(min(5, len(times))):
        sl = grids[i]
        first5.append(dict(method=methods[i], n_wet=int(n_wet_l[i]),
                           min=float(np.nanmin(sl)), max=float(np.nanmax(sl))))

    diag = dict(
        n_times=int(len(times)),
        counts=counts,
        first5=first5,
        ok_times=[str(t) for t, mth in zip(times, methods) if mth == "ok_pykrige"],
        pykrige_available=_PYKRIGE_AVAILABLE,
        config=dict(
            grid_res_deg=grid_res_deg, domain_pad_deg=domain_pad_deg,
            drizzle_to_zero=drizzle_to_zero, use_ok=bool(use_ok),
            min_pts_ok=min_pts_ok, nlags=nlags, ok_range_km=ok_range_km,
            ok_nugget_frac=ok_nugget_frac, ok_max_train=ok_max_train,
            idw_power=idw_power, idw_nnear=idw_nnear, idw_maxdist_km=idw_maxdist_km,
            max_dist_km_mask=max_dist_km_mask, smooth_kernel_px=smooth_kernel_px,
            n_jobs=n_jobs, parallel_backend_name=parallel_backend_name,
            kdtree_workers=kdtree_workers, collocate_bin_km=collocate_bin_km,
            use_pathlength_weights=use_pathlength_weights,
            outside_support_fill=outside_support_fill,
            interior_no_neighbor_fill=interior_no_neighbor_fill,
            insufficient_training_fill=insufficient_training_fill,
        ),
        times_used=[str(t) for t in times[:10]],
    )
    return da, diag


def grid_rain_at_time(df_s5, df_meta_for_xy, t, **kwargs):
    """Convenience wrapper to grid a single timestamp."""
    return grid_rain_15min(df_s5=df_s5, df_meta_for_xy=df_meta_for_xy, times_sel=[t], **kwargs)


# ---------------- RainLINK-like OK variant (wet+dry0 training & strict support) ----------------
def _midpoints(dfm):
    m = dfm.copy()
    for c in ["XStart","YStart","XEnd","YEnd"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["lon_mid"] = (m["XStart"] + m["XEnd"]) / 2.0
    m["lat_mid"] = (m["YStart"] + m["YEnd"]) / 2.0
    return m

def _grid_from_meta(meta_xy, grid_res_deg=0.03, pad_deg=0.20):
    mm = _midpoints(meta_xy)
    x0, x1 = float(mm["lon_mid"].min()), float(mm["lon_mid"].max())
    y0, y1 = float(mm["lat_mid"].min()), float(mm["lat_mid"].max())
    xv = np.arange(x0 - pad_deg, x1 + pad_deg + grid_res_deg, grid_res_deg)
    yv = np.arange(y0 - pad_deg, y1 + pad_deg + grid_res_deg, grid_res_deg)
    lon, lat = np.meshgrid(xv, yv)
    return lon, lat, xv, yv

def _support_mask_wet_haversine(lon_grid, lat_grid, lon_wet, lat_wet, k=2, radius_km=25.0):
    """True where the k-th nearest WET link is within radius_km (haversine)."""
    if not _SKLEARN_AVAILABLE or len(lon_wet) < k:
        return np.zeros_like(lon_grid, dtype=bool)
    tree = BallTree(np.deg2rad(np.c_[lat_wet, lon_wet]), metric="haversine")
    pts  = np.deg2rad(np.c_[lat_grid.ravel(), lon_grid.ravel()])
    d_rad, _ = tree.query(pts, k=int(k))
    d_km = d_rad[:, -1] * _EARTH_R_KM
    return (d_km <= float(radius_km)).reshape(lon_grid.shape)

def _estimate_variogram_params(values, range_km=25.0, nugget_frac=0.4):
    """Simple robust parameters; sill from variance of training values."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    sill = float(np.nanvar(v)) if v.size else 1e-6
    sill = max(sill, 1e-6)
    return {"sill": sill, "range": float(range_km), "nugget": float(nugget_frac) * sill}

def grid_rain_15min_rainlink_ok(
    df_s5,                    # index: time (naive UTC); cols: ID, R_mm_per_h
    df_meta_for_xy,           # cols: ID, XStart, YStart, XEnd, YEnd
    *,
    grid_res_deg=0.03, domain_pad_deg=0.20,
    wet_thr=0.8, dry_thr=0.05,               # define wet/dry points for training
    ok_model="exponential", ok_range_km=25.0, ok_nugget_frac=0.4,
    min_pts_ok=12,                            # need at least this many (wet+dry0) to run OK
    support_k=2, support_radius_km=25.0,     # mask requires ≥k wet links within radius
    drizzle_to_zero=0.10,                     # floor tiny positives to 0.0 (doesn't touch NaNs)
    times_sel=None,
    # NEW:
    n_jobs: int = 1,
    parallel_backend_name: str = "processes",
    # NEW controls for “no information”
    outside_support_fill=np.nan,             # what to put OUTSIDE strict wet support (np.nan or 0.0)
    insufficient_training_fill=np.nan,       # what to put when we cannot train/interpolate at all
):
    """
    RAINLINK-like: OK trained on wet values + dry zeros; apply strict wet support mask.

    outside_support_fill:
        Value for cells outside the wet-link support mask (default NaN).
    insufficient_training_fill:
        Value for the entire grid when there were no usable points or too few to train (default NaN).
    """
    if not _PYKRIGE_AVAILABLE:
        raise RuntimeError("PyKrige not available for RainLINK-style gridding.")

    # 1) grid axes
    LON, LAT, xv, yv = _grid_from_meta(df_meta_for_xy, grid_res_deg, domain_pad_deg)

    # 2) outputs
    all_times = pd.Index(sorted(df_s5.index.unique()))
    times = all_times if times_sel is None else pd.Index(pd.to_datetime(times_sel))
    out = np.full((len(times), LAT.shape[0], LAT.shape[1]), np.nan, float)

    # precompute midpoints by ID
    mid = _midpoints(df_meta_for_xy[["ID","XStart","YStart","XEnd","YEnd"]].drop_duplicates("ID"))
    id2xy = mid.set_index("ID")[["lon_mid","lat_mid"]]

    # diagnostics
    diag = {"counts": {"ok": 0, "fallback": 0}, "wet_counts": [], "train_counts": []}

    def _do_one(it, t):
        # slice points and attach coords
        try:
            pts = (df_s5.loc[t].merge(id2xy, on="ID", how="inner"))
        except KeyError:
            # no rows at this timestamp → whole grid is "no info"
            Z = np.full_like(LON, insufficient_training_fill, float)
            return it, Z, 0, 0, False

        vals = pd.to_numeric(pts["R_mm_per_h"], errors="coerce").values
        lon  = pd.to_numeric(pts["lon_mid"], errors="coerce").values
        lat  = pd.to_numeric(pts["lat_mid"], errors="coerce").values

        good = np.isfinite(vals) & np.isfinite(lon) & np.isfinite(lat)
        if not good.any():
            Z = np.full_like(LON, insufficient_training_fill, float)
            return it, Z, 0, 0, False

        vals, lon, lat = vals[good], lon[good], lat[good]

        # classify wet/dry for training
        wet = vals >= float(wet_thr)
        dry = vals <= float(dry_thr)

        lon_wet, lat_wet = lon[wet], lat[wet]
        # training set = wet values + dry zeros
        lon_tr = np.concatenate([lon[wet], lon[dry]])
        lat_tr = np.concatenate([lat[wet], lat[dry]])
        val_tr = np.concatenate([vals[wet], np.zeros(np.count_nonzero(dry), float)])

        if len(val_tr) < max(3, int(min_pts_ok)):
            Z = np.full_like(LON, insufficient_training_fill, float)
            return it, Z, int(wet.sum()), int(len(val_tr)), False

        # OK on geographic coords
        vparam = _estimate_variogram_params(val_tr, range_km=ok_range_km, nugget_frac=ok_nugget_frac)
        try:
            OK = OrdinaryKriging(
                lon_tr, lat_tr, val_tr,
                variogram_model=ok_model,
                variogram_parameters=vparam,
                coordinates_type="geographic",
                enable_plotting=False, verbose=False
            )
            Z, _ = OK.execute("grid", xv, yv)   # (ny, nx)
            Z = np.asarray(Z, float)

            # strict support mask based on WET links only
            if _SKLEARN_AVAILABLE and len(lon_wet) >= max(1, int(support_k)):
                mask = _support_mask_wet_haversine(LON, LAT, lon_wet, lat_wet,
                                                   k=int(support_k), radius_km=float(support_radius_km))
            else:
                # fallback: nearest-distance footprint from wet points in degrees (~approx km)
                mask = _nearest_distance_mask(xv, yv, lon_wet, lat_wet,
                                              max_dist_km=support_radius_km, workers=1)

            # Outside support → fill (NaN by default)
            if np.isnan(outside_support_fill):
                Z = np.where(mask, Z, np.nan)
            else:
                Z = np.where(mask, Z, float(outside_support_fill))

            # Drizzle floor to zero (does not touch NaNs)
            if drizzle_to_zero is not None:
                Z = np.where(np.isfinite(Z) & (Z < float(drizzle_to_zero)), 0.0, Z)

            return it, Z, int(wet.sum()), int(len(val_tr)), True
        except Exception:
            Z = np.full_like(LON, insufficient_training_fill, float)
            return it, Z, int(wet.sum()), int(len(val_tr)), False

    backend = "loky" if parallel_backend_name == "processes" else "threading"
    with parallel_backend(backend):
        results = Parallel(n_jobs=int(n_jobs))(
            delayed(_do_one)(i, t) for i, t in enumerate(times)
        )

    for it, Z, nwet, ntrain, ok_flag in results:
        out[it, :, :] = Z
        diag["wet_counts"].append(nwet)
        diag["train_counts"].append(ntrain)
        diag["counts"]["ok" if ok_flag else "fallback"] += 1

    da = xr.DataArray(
        out, coords={"time": times.tz_localize(None), "lat": yv, "lon": xv},
        dims=("time", "lat", "lon"), name="R_mm_per_h", attrs={"units": "mm h-1"}
    )
    return da, diag


def grid_rain_at_time_rainlink(df_s5, df_meta_for_xy, t, **kwargs):
    return grid_rain_15min_rainlink_ok(
        df_s5=df_s5, df_meta_for_xy=df_meta_for_xy, times_sel=[t], **kwargs
    )