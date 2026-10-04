#%%
"""
CML-SAT two-stage MSG + CML rainfall training using Bas wet/dry-first CML reference.

Purpose
-------
Train two MSG-based ML models over Ghana:
  1) Wet/dry occurrence classifier from MSG SEVIRI predictors.
  2) Rainfall-intensity quantile regressors trained only on wet CML-reference pixels.

This script is adapted from the earlier one-stage CML-SAT quantile-regression training code,
but now uses the Bas-review CML-only statistics-based wet/dry-first rainfall outputs as the
reference target.

Main decisions preserved from previous CML-SAT workflow
------------------------------------------------------
- MSG predictors: BT_IR108, BT_IR120, BT_WV062, IR108_minus_WV062.
- Cloud-pixel mask: cloud_mask == 2.
- 15-min cadence/time rounding.
- Ghana fixed grid at 0.03 degrees.
- CML rainfall reference variable: R_mm_per_h.
- CML reference is clipped/gated at tiny drizzle values before defining wet/dry.

Note
----
The latitude/zonal IR108 patch correction is NOT part of this training script. It should remain
in the application/mapping script as the final occurrence refinement after classifier + intensity
prediction.
"""

#%% Imports
import warnings
warnings.filterwarnings("ignore")
import os
import json
import gc
from datetime import date

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401; enables .rio accessor
from rasterio.enums import Resampling

import joblib
import xgboost as xgb

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    from quantnn.quantiles import posterior_mean, posterior_quantiles
except Exception:
    posterior_mean = None
    posterior_quantiles = None

#%% Paths
# New Bas/statistics wet-dry CML-only reference rainfall output.
# PATH_CML_RAIN = r"/home/kkumah/Projects/cml-stuff/new_out_cml_Rain_bas_stats_training_ref"
PATH_CML_RAIN = r"/home/kkumah/Projects/cml-stuff/new_out_cml_Rain_bas_stats_training_ref_sens_q75_p90_wet03"
# MSG training-period files.
PATH_MSG_BT  = r"/home/kkumah/Projects/cml-stuff/satellite_data/msg"
PATH_MSG_CLM = r"/home/kkumah/Projects/cml-stuff/satellite_data/msg_clm"

# Model output.
PATH_OUT_MODEL = r"/home/kkumah/Projects/cml-stuff/out_train_model_twostage_basref"
os.makedirs(PATH_OUT_MODEL, exist_ok=True)

# Internal holdout QA plots/statistics output.
PATH_OUT_QA = os.path.join(PATH_OUT_MODEL, "internal_holdout_QA")
os.makedirs(PATH_OUT_QA, exist_ok=True)

#%% Constants and controls
BT_VARS = ["BT_IR108", "BT_IR120", "BT_WV062"]
CLM_VARS = ["cloud_mask"]

TIME_ROUND = "15min"

# Ghana domain used in current CML-SAT code.
LON_MIN, LON_MAX = -4.0, 1.25
LAT_MIN, LAT_MAX = 4.5, 11.25
RES_DEG = 0.03

# Training period. June 14 gives a 72-hour warm-up if raw CML began on June 11.
TRAIN_START = "2025-06-14"
TRAIN_END   = "2025-08-24"

# Wet/dry target threshold for CML reference rain rate.
# With the Bas method we now have true zeros, but 0.10 mm/h avoids training on tiny residual/noise rates.
WET_THR = 0.10  # mm h-1
DRIZZLE_FLOOR_FOR_TARGET = 0.01  # values below this become 0 before wet/dry target construction

# Sampling controls.
N_TIMES_PER_MONTH = 800
N_PIX_PER_TIME = 4000

# Balanced sampling fraction for the classifier/training table.
# For occurrence training, use a stronger wet fraction than natural occurrence to avoid dry domination.
WET_FRAC_SAMPLE = 0.50

# Optional internal diagnostic split by time groups. Final models are still trained on all sampled data.
DO_INTERNAL_HOLDOUT_QA = True
HOLDOUT_TEST_SIZE = 0.20
RANDOM_SEED = 42
NTHREAD = 20

FEAT_NAMES = ["BT_IR108", "BT_IR120", "BT_WV062", "IR108_minus_WV062"]

# Dense quantile intensity models trained on wet-only samples.
QS_DENSE = np.linspace(0.05, 0.95, 19)

GEOS_CRS = "+proj=geos +lon_0=0 +h=35785831 +a=6378137 +b=6356752.31414 +sweep=x +no_defs"

CDE_RUN_DTE = date.today().strftime("%Y%m%d")

#%% Helpers
def list_files(root, suffix=".nc"):
    if os.path.isdir(root):
        return sorted(
            os.path.join(r, f)
            for r, _, fs in os.walk(root)
            for f in fs
            if f.endswith(suffix)
        )
    return [root] if str(root).endswith(suffix) else []


def round_time_index(ds, freq=TIME_ROUND):
    if "time" in ds.coords:
        ds = ds.assign_coords(time=ds["time"].dt.round(freq))
        _, idx = np.unique(ds["time"].values, return_index=True)
        ds = ds.isel(time=np.sort(idx))
    return ds


def subset_time(ds, start=TRAIN_START, end=TRAIN_END):
    if "time" not in ds.coords:
        return ds
    start_ts = np.datetime64(pd.Timestamp(start))
    end_ts = np.datetime64(pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    return ds.sel(time=slice(start_ts, end_ts))


def standardize_cml_grid(ds):
    """Ensure CML rainfall uses y/x dimensions for later alignment."""
    rename = {}
    if "lat" in ds.dims:
        rename["lat"] = "y"
    if "lon" in ds.dims:
        rename["lon"] = "x"
    if rename:
        ds = ds.rename(rename)
    return ds


def sort_latlon_yx(ds):
    if "y" in ds.coords:
        ds = ds.sortby("y")
    if "x" in ds.coords:
        ds = ds.sortby("x")
    return ds


def sample_times_month_balanced(times, n_per_month=N_TIMES_PER_MONTH, seed=RANDOM_SEED):
    t = pd.to_datetime(times)
    df = pd.DataFrame({"time": t})
    df["month"] = df["time"].dt.to_period("M").astype(str)

    out = []
    for _, g in df.groupby("month"):
        k = min(n_per_month, len(g))
        if k > 0:
            out.append(g.sample(n=k, replace=False, random_state=seed)["time"].values)

    if not out:
        return np.array([], dtype="datetime64[ns]")
    return np.sort(np.concatenate(out))


def sample_pixels_one_time_twostage(
    b1, b2, b3, bd, r,
    n_pix=N_PIX_PER_TIME,
    wet_thr=WET_THR,
    wet_frac=WET_FRAC_SAMPLE,
    seed=None,
):
    """Return sampled features, rain intensity, wet/dry target for one time slice."""
    valid = (
        np.isfinite(b1) & np.isfinite(b2) & np.isfinite(b3) &
        np.isfinite(bd) & np.isfinite(r)
    )
    if valid.sum() == 0:
        return None, None, None

    wet = valid & (r >= wet_thr)
    dry = valid & (r < wet_thr)

    n_wet = int(n_pix * wet_frac)
    n_dry = n_pix - n_wet

    wet_idx = np.flatnonzero(wet)
    dry_idx = np.flatnonzero(dry)
    valid_idx = np.flatnonzero(valid)

    rs = np.random.RandomState(seed) if seed is not None else np.random
    pick = []

    if wet_idx.size > 0:
        pick.append(rs.choice(wet_idx, size=min(n_wet, wet_idx.size), replace=False))
    if dry_idx.size > 0:
        pick.append(rs.choice(dry_idx, size=min(n_dry, dry_idx.size), replace=False))

    pick = np.concatenate(pick) if pick else np.array([], dtype=int)

    # Top up from all valid pixels if a scene has too few wet or dry pixels.
    if pick.size < n_pix and valid_idx.size > pick.size:
        remaining = np.setdiff1d(valid_idx, pick, assume_unique=False)
        if remaining.size > 0:
            pick = np.concatenate([
                pick,
                rs.choice(remaining, size=min(n_pix - pick.size, remaining.size), replace=False),
            ])

    if pick.size == 0:
        return None, None, None

    X = np.column_stack([
        b1.ravel()[pick],
        b2.ravel()[pick],
        b3.ravel()[pick],
        bd.ravel()[pick],
    ]).astype("float32")

    y_rate = r.ravel()[pick].astype("float32")
    y_wet = (y_rate >= wet_thr).astype("int8")

    return X, y_rate, y_wet


def train_xgb_classifier(dtrain, dvalid=None, scale_pos_weight=None, seed=150):
    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr", "error"],
        "tree_method": "hist",
        "max_depth": 8,
        "eta": 0.06,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "seed": seed,
        "nthread": NTHREAD,
    }
    if scale_pos_weight is not None and np.isfinite(scale_pos_weight) and scale_pos_weight > 0:
        params["scale_pos_weight"] = float(scale_pos_weight)

    evals = [(dtrain, "Training")]
    kwargs = {}
    if dvalid is not None:
        evals.append((dvalid, "Validation"))
        kwargs.update({"early_stopping_rounds": 30})

    return xgb.train(params, dtrain, num_boost_round=500, evals=evals, **kwargs)


def train_quantile(alpha, dtrain, dvalid=None, seed=150):
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": float(alpha),
        "tree_method": "hist",
        "max_depth": 10,
        "eta": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.9,
        "seed": seed,
        "nthread": NTHREAD,
    }
    evals = [(dtrain, "Training")]
    kwargs = {}
    if dvalid is not None:
        evals.append((dvalid, "Validation"))
        kwargs.update({"early_stopping_rounds": 30})
    return xgb.train(params, dtrain, num_boost_round=600, evals=evals, **kwargs)


def binary_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1]).ravel()
    pod = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    sr = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    far = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    bias = (tp + fp) / (tp + fn) if (tp + fn) > 0 else np.nan
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else np.nan
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    return {
        "threshold": float(threshold),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Accuracy": float(acc),
        "POD": float(pod), "SR": float(sr), "FAR": float(far),
        "Bias": float(bias), "CSI": float(csi),
    }


def choose_threshold_by_csi(y_true, y_prob):
    """Choose wet-probability threshold maximizing CSI on diagnostic holdout."""
    thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for thr in thresholds:
        m = binary_metrics(y_true, y_prob, threshold=thr)
        rows.append(m)
    df = pd.DataFrame(rows)
    if df["CSI"].notna().any():
        best = df.loc[df["CSI"].idxmax()].to_dict()
    else:
        best = binary_metrics(y_true, y_prob, threshold=0.5)
    return best, df


def predict_quantile_stack(boosters_by_q, X, qs):
    dX = xgb.DMatrix(X.astype("float32"), feature_names=FEAT_NAMES, nthread=NTHREAD)
    arr = np.column_stack([np.expm1(boosters_by_q[q].predict(dX)) for q in qs]).astype("float32")
    return np.maximum.accumulate(arr, axis=1)


def quantile_point_estimates(Yq, qs):
    if posterior_mean is not None:
        qmean = posterior_mean(Yq, quantiles=qs).astype("float32")
    else:
        qmean = Yq.mean(axis=1).astype("float32")

    if posterior_quantiles is not None:
        qmedian = posterior_quantiles(Yq, quantiles=qs, new_quantiles=[0.5])[..., 0].astype("float32")
    else:
        qmedian = Yq[:, np.argmin(np.abs(qs - 0.5))].astype("float32")
    return qmean, qmedian


def preprocess_cml(ds):
    keep = [
        "R_mm_per_h",
        "cml_support_confidence",
        "cml_support_mask",
        "cml_coverage_quality",
    ]
    keep = [v for v in keep if v in ds.data_vars]

    ds = ds[keep]

    rename = {}
    if "lat" in ds.dims:
        rename["lat"] = "y"
    if "lon" in ds.dims:
        rename["lon"] = "x"
    if rename:
        ds = ds.rename(rename)

    return ds


def continuous_metrics(y_true, y_pred, label="all"):
    """
    Continuous rainfall evaluation metrics.
    y_true and y_pred should both be in mm/h.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return {
            "subset": label,
            "n": int(valid.sum()),
            "mean_obs": np.nan,
            "mean_pred": np.nan,
            "CC": np.nan,
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "RelBias_pct": np.nan,
        }

    obs = y_true[valid]
    pred = y_pred[valid]

    cc = np.corrcoef(obs, pred)[0, 1] if np.nanstd(obs) > 0 and np.nanstd(pred) > 0 else np.nan
    rmse = np.sqrt(np.nanmean((pred - obs) ** 2))
    mae = np.nanmean(np.abs(pred - obs))
    bias = np.nanmean(pred - obs)
    relbias = 100.0 * bias / max(np.nanmean(obs), 1e-6)

    return {
        "subset": label,
        "n": int(valid.sum()),
        "mean_obs": float(np.nanmean(obs)),
        "mean_pred": float(np.nanmean(pred)),
        "CC": float(cc),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "Bias": float(bias),
        "RelBias_pct": float(relbias),
    }


def save_confusion_matrix_plot(y_true, y_prob, threshold, out_png, title):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true.astype(int), y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred dry", "Pred wet"])
    ax.set_yticklabels(["Obs dry", "Obs wet"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12)

    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Count")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_threshold_curve_plot(threshold_curve, out_png):
    fig, ax = plt.subplots(figsize=(8, 5))

    for col in ["POD", "SR", "FAR", "CSI", "Bias"]:
        if col in threshold_curve.columns:
            ax.plot(threshold_curve["threshold"], threshold_curve[col], label=col)

    ax.axhline(1.0, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Wet-probability threshold")
    ax.set_ylabel("Metric value")
    ax.set_title("Internal holdout wet/dry threshold sensitivity")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_precision_recall_plot(y_true, y_prob, out_png):
    precision, recall, _ = precision_recall_curve(y_true.astype(int), y_prob)
    ap = average_precision_score(y_true.astype(int), y_prob)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall / POD")
    ax.set_ylabel("Precision / Success Ratio")
    ax.set_title("Internal holdout precision–recall curve")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_probability_histogram(y_true, y_prob, out_png):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        y_prob[y_true == 0],
        bins=np.linspace(0, 1, 51),
        alpha=0.55,
        density=True,
        label="Obs dry",
    )
    ax.hist(
        y_prob[y_true == 1],
        bins=np.linspace(0, 1, 51),
        alpha=0.55,
        density=True,
        label="Obs wet",
    )

    ax.set_xlabel("Predicted wet probability")
    ax.set_ylabel("Density")
    ax.set_title("Wet-probability separation: dry vs wet reference pixels")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_obs_pred_density_plot(y_true, y_pred, out_png, title, max_plot=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    obs = y_true[valid]
    pred = y_pred[valid]

    if obs.size == 0:
        return

    if max_plot is None:
        max_plot = np.nanpercentile(np.concatenate([obs, pred]), 99.5)
        max_plot = max(max_plot, 1.0)

    fig, ax = plt.subplots(figsize=(6.5, 5.8))

    hb = ax.hexbin(
        obs,
        pred,
        gridsize=70,
        mincnt=1,
        bins="log",
    )

    ax.plot([0, max_plot], [0, max_plot], "k--", linewidth=1.0)
    ax.set_xlim(0, max_plot)
    ax.set_ylim(0, max_plot)

    ax.set_xlabel("Observed CML reference rainfall (mm h$^{-1}$)")
    ax.set_ylabel("Predicted rainfall (mm h$^{-1}$)")
    ax.set_title(title)

    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("log10(count)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_error_histogram(y_true, y_pred, out_png):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    err = y_pred[valid] - y_true[valid]

    if err.size == 0:
        return

    lim = np.nanpercentile(np.abs(err), 99)
    lim = max(lim, 1.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(err, bins=np.linspace(-lim, lim, 80), alpha=0.85)
    ax.axvline(0, color="k", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Prediction error: pred - obs (mm h$^{-1}$)")
    ax.set_ylabel("Count")
    ax.set_title("Internal holdout rainfall error distribution")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_residual_vs_obs_plot(y_true, y_pred, out_png):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    obs = y_true[valid]
    err = y_pred[valid] - y_true[valid]

    if obs.size == 0:
        return

    xmax = np.nanpercentile(obs, 99.5)
    ymax = np.nanpercentile(np.abs(err), 99)
    xmax = max(xmax, 1.0)
    ymax = max(ymax, 1.0)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    hb = ax.hexbin(
        obs,
        err,
        gridsize=70,
        mincnt=1,
        bins="log",
    )

    ax.axhline(0, color="k", linewidth=1.0, linestyle="--")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-ymax, ymax)

    ax.set_xlabel("Observed CML reference rainfall (mm h$^{-1}$)")
    ax.set_ylabel("Prediction error: pred - obs (mm h$^{-1}$)")
    ax.set_title("Residuals versus observed rainfall")

    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("log10(count)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=240, bbox_inches="tight")
    plt.close(fig)

# Sampled training-target diagnostics
def print_training_rain_target_diagnostics(y_rate_all, y_wet_all, wet_thr=WET_THR):
    y = np.asarray(y_rate_all, dtype=float)
    wet = np.asarray(y_wet_all).astype(bool)

    finite = np.isfinite(y)
    positive = finite & (y > 0)
    weak_positive = finite & (y > 0) & (y < wet_thr)
    wet_by_rate = finite & (y >= wet_thr)
    dry_by_rate = finite & (y < wet_thr)

    print("\n" + "=" * 90)
    print("SAMPLED TRAINING RAINFALL TARGET DIAGNOSTICS")
    print("=" * 90)

    print("Total sampled pixels:", int(y.size))
    print("Finite sampled pixels:", int(finite.sum()))
    print("NaN/invalid sampled pixels:", int((~finite).sum()))

    if finite.sum() == 0:
        print("No finite rainfall targets found.")
        print("=" * 90)
        return

    print("\nAll finite sampled rainfall values")
    print("----------------------------------")
    print("min:", float(np.nanmin(y[finite])))
    print("max:", float(np.nanmax(y[finite])))
    print("mean:", float(np.nanmean(y[finite])))
    print("median:", float(np.nanmedian(y[finite])))
    print("std:", float(np.nanstd(y[finite])))

    print("percentiles [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]:")
    print(np.nanpercentile(
        y[finite],
        [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
    ))

    print("\nOccurrence fractions")
    print("--------------------")
    print("zero fraction:", float(np.mean(y[finite] == 0.0)))
    print("positive fraction > 0:", float(np.mean(y[finite] > 0.0)))
    print(f"weak-positive dry fraction 0 < R < {wet_thr}:", float(np.mean((y[finite] > 0.0) & (y[finite] < wet_thr))))
    print(f"wet fraction R >= {wet_thr}:", float(np.mean(y[finite] >= wet_thr)))
    print("wet fraction from y_wet_all:", float(np.mean(wet[finite])))

    print("\nDry samples by y_wet_all == 0")
    print("-----------------------------")
    dry_mask = finite & (~wet)
    print("count:", int(dry_mask.sum()))
    if dry_mask.sum() > 0:
        print("min:", float(np.nanmin(y[dry_mask])))
        print("max:", float(np.nanmax(y[dry_mask])))
        print("mean:", float(np.nanmean(y[dry_mask])))
        print("percentiles [50, 75, 90, 95, 99, 100]:")
        print(np.nanpercentile(y[dry_mask], [50, 75, 90, 95, 99, 100]))

    print("\nWet samples by y_wet_all == 1")
    print("-----------------------------")
    wet_mask = finite & wet
    print("count:", int(wet_mask.sum()))
    if wet_mask.sum() > 0:
        print("min:", float(np.nanmin(y[wet_mask])))
        print("max:", float(np.nanmax(y[wet_mask])))
        print("mean:", float(np.nanmean(y[wet_mask])))
        print("median:", float(np.nanmedian(y[wet_mask])))
        print("std:", float(np.nanstd(y[wet_mask])))
        print("percentiles [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]:")
        print(np.nanpercentile(
            y[wet_mask],
            [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.5, 99.9, 100]
        ))

    print("\nConsistency checks")
    print("------------------")
    print("Count where y_wet_all == 1 but R < wet_thr:", int(np.sum(wet & finite & (y < wet_thr))))
    print("Count where y_wet_all == 0 but R >= wet_thr:", int(np.sum((~wet) & finite & (y >= wet_thr))))
    print("Count of weak-positive dry samples:", int(weak_positive.sum()))
    print("=" * 90 + "\n")


#%% Open and align datasets
print("Discovering files...")
cml_files = [
    f for f in list_files(PATH_CML_RAIN)
    if "ghana_cml_R_15min_bas_stats_ref_q75_p90_wet03_" in os.path.basename(f)
]
msg_bt_files = list_files(PATH_MSG_BT)
msg_clm_files = list_files(PATH_MSG_CLM)

if not cml_files:
    raise FileNotFoundError(f"No CML reference NetCDF files found in {PATH_CML_RAIN}")
if not msg_bt_files:
    raise FileNotFoundError(f"No MSG BT NetCDF files found in {PATH_MSG_BT}")
if not msg_clm_files:
    raise FileNotFoundError(f"No MSG CLM NetCDF files found in {PATH_MSG_CLM}")

print(f"CML files: {len(cml_files)}")
print(f"MSG BT files: {len(msg_bt_files)}")
print(f"MSG CLM files: {len(msg_clm_files)}")

print("Opening files with xarray...")
cml = xr.open_mfdataset(
    cml_files,
    combine="by_coords",
    preprocess=preprocess_cml,
    parallel=False,
    data_vars="minimal",
    coords="minimal",
    compat="override",
    join="override",
)
msg_bt = xr.open_mfdataset(
        msg_bt_files,
        combine="by_coords",
        parallel=False,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
    )
msg_clm = xr.open_mfdataset(
        msg_clm_files,
        combine="by_coords",
        parallel=False,
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
    )

# cml = standardize_cml_grid(cml)

# Keep only training period before expensive reprojection/interpolation.
cml = subset_time(round_time_index(cml, TIME_ROUND))
msg_bt = subset_time(round_time_index(msg_bt, TIME_ROUND))
msg_clm = subset_time(round_time_index(msg_clm, TIME_ROUND))

common_times = np.intersect1d(np.intersect1d(cml.time.values, msg_bt.time.values), msg_clm.time.values)
if common_times.size == 0:
    raise RuntimeError("No common 15-min timestamps among CML reference, MSG BT, and MSG CLM.")

cml = cml.sel(time=common_times)
msg_bt = msg_bt.sel(time=common_times)
msg_clm = msg_clm.sel(time=common_times)
print("Common aligned times:", len(common_times), pd.to_datetime(common_times[0]), "to", pd.to_datetime(common_times[-1]))

# Clip CML reference to Ghana bbox.
cml = sort_latlon_yx(cml)
cml = cml.sel(y=slice(LAT_MIN, LAT_MAX), x=slice(LON_MIN, LON_MAX))

# Reproject MSG from geostationary CRS to EPSG:4326.
print("Reprojecting MSG to EPSG:4326...")
msg_bt = msg_bt.rio.write_crs(GEOS_CRS, inplace=False)
msg_clm = msg_clm.rio.write_crs(GEOS_CRS, inplace=False)

msg_bt_ll = msg_bt.rio.reproject("EPSG:4326", resampling=Resampling.nearest)
msg_clm_ll = msg_clm.rio.reproject("EPSG:4326", resampling=Resampling.mode)

msg_bt_ll = sort_latlon_yx(msg_bt_ll)
msg_clm_ll = sort_latlon_yx(msg_clm_ll)

msg_bt_ll = msg_bt_ll.sel(x=slice(LON_MIN, LON_MAX), y=slice(LAT_MIN, LAT_MAX))
msg_clm_ll = msg_clm_ll.sel(x=slice(LON_MIN, LON_MAX), y=slice(LAT_MIN, LAT_MAX))

# Fixed training grid.
target_lats = np.arange(LAT_MIN, LAT_MAX + 1e-6, RES_DEG)
target_lons = np.arange(LON_MIN, LON_MAX + 1e-6, RES_DEG)

print("Interpolating MSG and CML reference to fixed Ghana grid...")
msg_bt_ll = msg_bt_ll.interp(y=target_lats, x=target_lons, method="linear")
msg_clm_ll = msg_clm_ll.interp(y=target_lats, x=target_lons, method="nearest")
cml_on_msg = cml.interp(y=target_lats, x=target_lons, method="linear")


assert np.allclose(msg_bt_ll.y, cml_on_msg.y)
assert np.allclose(msg_bt_ll.x, cml_on_msg.x)
print("Datasets aligned on fixed grid.")

#%% Build feature/target arrays lazily and sample
print("Building features and sampling training table...")

for v in BT_VARS:
    msg_bt_ll[v] = msg_bt_ll[v].astype("float32")

mask_cloud = (msg_clm_ll["cloud_mask"] == 2)
BT_IR108 = msg_bt_ll["BT_IR108"].where(mask_cloud)
BT_IR120 = msg_bt_ll["BT_IR120"].where(mask_cloud)
BT_WV062 = msg_bt_ll["BT_WV062"].where(mask_cloud)
BT_DIFF = (BT_IR108 - BT_WV062).where(mask_cloud)

if "R_mm_per_h" not in cml_on_msg.data_vars:
    raise KeyError(f"R_mm_per_h not found in CML reference. Available: {list(cml_on_msg.data_vars)}")

R = cml_on_msg["R_mm_per_h"].astype("float32")

# Keep only physically valid CML-supported pixels.
R = R.where(np.isfinite(R))
R = R.where(R > -9990)

# Optional but strongly recommended: keep only pixels supported by CML geometry.
if "cml_support_mask" in cml_on_msg:
    support_on_grid = cml_on_msg["cml_support_mask"]
    R = R.where(support_on_grid >= 0.5)

# Optional: require at least low/medium CML coverage quality.
# 0 = no/poor support; 1/2/3 = increasing quality, based on your saved layer.
if "cml_coverage_quality" in cml_on_msg:
    covq_on_grid = cml_on_msg["cml_coverage_quality"]
    R = R.where(covq_on_grid >= 1)

# Preserve NaNs outside support; only convert tiny finite drizzle to zero.
R = R.where(~np.isfinite(R) | (R >= DRIZZLE_FLOOR_FOR_TARGET), 0.0)

# Protect against negative finite values only.
R = R.where(~np.isfinite(R) | (R >= 0.0), 0.0)

print("After interpolation:")
print("  R dims:", R.dims)
print("  R shape:", R.shape)
print("  BT_IR108 dims:", BT_IR108.dims)
print("  BT_IR108 shape:", BT_IR108.shape)

# Quick target diagnostics before sampling.
R_vals_sample = R.isel(time=slice(0, min(20, R.sizes["time"]))).values
print("Initial CML reference quick-check:")
print("  sample min/max:", np.nanmin(R_vals_sample), np.nanmax(R_vals_sample))
print("  sample zero fraction:", np.nanmean(np.isfinite(R_vals_sample) & (R_vals_sample == 0)))
print("  sample wet fraction >= WET_THR:", np.nanmean(np.isfinite(R_vals_sample) & (R_vals_sample >= WET_THR)))

times_sub = sample_times_month_balanced(msg_bt_ll.time.values, n_per_month=N_TIMES_PER_MONTH, seed=RANDOM_SEED)
print("Selected training times:", len(times_sub), "from", len(msg_bt_ll.time.values))

X_list, y_rate_list, y_wet_list, group_list = [], [], [], []

for i, t in enumerate(times_sub):
    if i % 25 == 0:
        print(f"Sampling time {i+1}/{len(times_sub)}: {pd.to_datetime(t)}", flush=True)

    b1 = BT_IR108.sel(time=t).transpose("y", "x").values
    b2 = BT_IR120.sel(time=t).transpose("y", "x").values
    b3 = BT_WV062.sel(time=t).transpose("y", "x").values
    bd = BT_DIFF.sel(time=t).transpose("y", "x").values
    rr = R.sel(time=t).transpose("y", "x").values

    X_i, y_rate_i, y_wet_i = sample_pixels_one_time_twostage(
        b1, b2, b3, bd, rr,
        n_pix=N_PIX_PER_TIME,
        wet_thr=WET_THR,
        wet_frac=WET_FRAC_SAMPLE,
        seed=RANDOM_SEED + i,
    )

    if X_i is None:
        continue

    X_list.append(X_i)
    y_rate_list.append(y_rate_i)
    y_wet_list.append(y_wet_i)
    group_list.append(np.full(y_wet_i.shape, i, dtype="int32"))

if not X_list:
    raise RuntimeError("No valid training samples were collected.")

X_all = np.vstack(X_list).astype("float32")
y_rate_all = np.concatenate(y_rate_list).astype("float32")
y_wet_all = np.concatenate(y_wet_list).astype("int8")
groups_all = np.concatenate(group_list).astype("int32")

print("Training table:", X_all.shape)
print("Wet fraction in sampled table:", float(y_wet_all.mean()))
print("Rain target min/mean/max:", float(np.nanmin(y_rate_all)), float(np.nanmean(y_rate_all)), float(np.nanmax(y_rate_all)))

print_training_rain_target_diagnostics(y_rate_all, y_wet_all, wet_thr=WET_THR)

# Clean memory from large lists.
del X_list, y_rate_list, y_wet_list, group_list
gc.collect()

#%% Internal holdout QA by time group
qa_summary = {}
threshold_best = {"threshold": 0.5}

if DO_INTERNAL_HOLDOUT_QA and np.unique(groups_all).size >= 5:
    print("Running internal group holdout QA...")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=HOLDOUT_TEST_SIZE,
        random_state=RANDOM_SEED,
    )
    train_idx, test_idx = next(splitter.split(X_all, y_wet_all, groups=groups_all))

    X_tr, X_te = X_all[train_idx], X_all[test_idx]
    yw_tr, yw_te = y_wet_all[train_idx], y_wet_all[test_idx]
    yr_tr, yr_te = y_rate_all[train_idx], y_rate_all[test_idx]

    print("Holdout split:")
    print("  Train samples:", len(train_idx))
    print("  Test samples :", len(test_idx))
    print("  Train wet fraction:", float(yw_tr.mean()))
    print("  Test wet fraction :", float(yw_te.mean()))

    neg = np.sum(yw_tr == 0)
    pos = np.sum(yw_tr == 1)
    spw = neg / pos if pos > 0 else None

    dcls_tr = xgb.DMatrix(X_tr, label=yw_tr, feature_names=FEAT_NAMES, nthread=NTHREAD)
    dcls_te = xgb.DMatrix(X_te, label=yw_te, feature_names=FEAT_NAMES, nthread=NTHREAD)

    cls_qa = train_xgb_classifier(
        dcls_tr,
        dvalid=dcls_te,
        scale_pos_weight=spw,
        seed=150,
    )

    wet_prob_te = cls_qa.predict(dcls_te)

    metrics_05 = binary_metrics(yw_te, wet_prob_te, threshold=0.5)
    threshold_best, threshold_curve = choose_threshold_by_csi(yw_te, wet_prob_te)
    best_thr = float(threshold_best["threshold"])

    te_pred_wet = wet_prob_te >= best_thr

    qa_summary["classifier_threshold_0p5"] = metrics_05
    qa_summary["classifier_best_csi_threshold"] = threshold_best
    qa_summary["holdout_n"] = int(len(test_idx))
    qa_summary["holdout_reference_wet_fraction"] = float(yw_te.mean())
    qa_summary["holdout_predicted_wet_fraction_best_thr"] = float(te_pred_wet.mean())

    # ---------------------------------------------------------------------
    # Save categorical QA tables and plots
    # ---------------------------------------------------------------------
    threshold_curve_path = os.path.join(
        PATH_OUT_QA,
        f"twostage_threshold_curve_basref_{CDE_RUN_DTE}.csv",
    )
    threshold_curve.to_csv(threshold_curve_path, index=False)

    save_confusion_matrix_plot(
        yw_te,
        wet_prob_te,
        threshold=0.5,
        out_png=os.path.join(PATH_OUT_QA, f"confusion_matrix_thr0p50_{CDE_RUN_DTE}.png"),
        title="Wet/dry confusion matrix | threshold = 0.50",
    )

    save_confusion_matrix_plot(
        yw_te,
        wet_prob_te,
        threshold=best_thr,
        out_png=os.path.join(PATH_OUT_QA, f"confusion_matrix_bestCSI_{CDE_RUN_DTE}.png"),
        title=f"Wet/dry confusion matrix | best CSI threshold = {best_thr:.2f}",
    )

    save_threshold_curve_plot(
        threshold_curve,
        out_png=os.path.join(PATH_OUT_QA, f"threshold_metrics_curve_{CDE_RUN_DTE}.png"),
    )

    save_precision_recall_plot(
        yw_te,
        wet_prob_te,
        out_png=os.path.join(PATH_OUT_QA, f"precision_recall_curve_{CDE_RUN_DTE}.png"),
    )

    save_probability_histogram(
        yw_te,
        wet_prob_te,
        out_png=os.path.join(PATH_OUT_QA, f"wet_probability_histogram_{CDE_RUN_DTE}.png"),
    )

    # ---------------------------------------------------------------------
    # Wet-only intensity QA using true wet train pixels
    # ---------------------------------------------------------------------
    tr_wet = yw_tr == 1

    if tr_wet.sum() > 100 and te_pred_wet.sum() > 10:
        print("Training holdout wet-only quantile models for internal intensity QA...")

        dreg_tr = xgb.DMatrix(
            X_tr[tr_wet],
            label=np.log1p(yr_tr[tr_wet]),
            feature_names=FEAT_NAMES,
            nthread=NTHREAD,
        )

        reg_qa = {
            q: train_quantile(q, dreg_tr, dvalid=None, seed=150)
            for q in QS_DENSE
        }

        # Predict intensity only where classifier says wet.
        Yq_te_predwet = predict_quantile_stack(reg_qa, X_te[te_pred_wet], QS_DENSE)
        pred_mean_predwet, pred_median_predwet = quantile_point_estimates(
            Yq_te_predwet,
            QS_DENSE,
        )

        # Full hurdle prediction: dry-classified pixels are zero.
        pred_full_mean = np.zeros_like(yr_te, dtype="float32")
        pred_full_median = np.zeros_like(yr_te, dtype="float32")

        pred_full_mean[te_pred_wet] = pred_mean_predwet
        pred_full_median[te_pred_wet] = pred_median_predwet

        # Also evaluate intensity model on observed-wet holdout pixels only.
        te_obs_wet = yw_te == 1
        if te_obs_wet.sum() > 10:
            Yq_te_obswet = predict_quantile_stack(reg_qa, X_te[te_obs_wet], QS_DENSE)
            pred_mean_obswet, pred_median_obswet = quantile_point_estimates(
                Yq_te_obswet,
                QS_DENSE,
            )
        else:
            pred_mean_obswet = np.array([], dtype="float32")
            pred_median_obswet = np.array([], dtype="float32")

        # Metrics
        metric_rows = []
        metric_rows.append(continuous_metrics(yr_te, pred_full_mean, label="full_hurdle_mean_all_pixels"))
        metric_rows.append(continuous_metrics(yr_te, pred_full_median, label="full_hurdle_median_all_pixels"))

        if te_obs_wet.sum() > 10:
            metric_rows.append(
                continuous_metrics(
                    yr_te[te_obs_wet],
                    pred_mean_obswet,
                    label="intensity_mean_obs_wet_pixels_only",
                )
            )
            metric_rows.append(
                continuous_metrics(
                    yr_te[te_obs_wet],
                    pred_median_obswet,
                    label="intensity_median_obs_wet_pixels_only",
                )
            )

        metrics_df = pd.DataFrame(metric_rows)
        metrics_csv = os.path.join(PATH_OUT_QA, f"continuous_holdout_metrics_{CDE_RUN_DTE}.csv")
        metrics_df.to_csv(metrics_csv, index=False)

        qa_summary["continuous_metrics_csv"] = metrics_csv
        for _, row in metrics_df.iterrows():
            qa_summary[row["subset"]] = {
                "n": int(row["n"]),
                "mean_obs": float(row["mean_obs"]),
                "mean_pred": float(row["mean_pred"]),
                "CC": float(row["CC"]),
                "RMSE": float(row["RMSE"]),
                "MAE": float(row["MAE"]),
                "Bias": float(row["Bias"]),
                "RelBias_pct": float(row["RelBias_pct"]),
            }

        # Quantitative plots: full hurdle
        save_obs_pred_density_plot(
            yr_te,
            pred_full_mean,
            out_png=os.path.join(PATH_OUT_QA, f"obs_vs_pred_full_hurdle_mean_{CDE_RUN_DTE}.png"),
            title="Internal holdout: observed vs predicted rainfall | full hurdle mean",
        )

        save_obs_pred_density_plot(
            yr_te,
            pred_full_median,
            out_png=os.path.join(PATH_OUT_QA, f"obs_vs_pred_full_hurdle_median_{CDE_RUN_DTE}.png"),
            title="Internal holdout: observed vs predicted rainfall | full hurdle median",
        )

        save_error_histogram(
            yr_te,
            pred_full_mean,
            out_png=os.path.join(PATH_OUT_QA, f"error_histogram_full_hurdle_mean_{CDE_RUN_DTE}.png"),
        )

        save_residual_vs_obs_plot(
            yr_te,
            pred_full_mean,
            out_png=os.path.join(PATH_OUT_QA, f"residual_vs_obs_full_hurdle_mean_{CDE_RUN_DTE}.png"),
        )

        # Quantitative plots: observed-wet only
        if te_obs_wet.sum() > 10:
            save_obs_pred_density_plot(
                yr_te[te_obs_wet],
                pred_mean_obswet,
                out_png=os.path.join(PATH_OUT_QA, f"obs_vs_pred_obswet_mean_{CDE_RUN_DTE}.png"),
                title="Internal holdout: observed wet pixels only | quantile mean",
            )

            save_error_histogram(
                yr_te[te_obs_wet],
                pred_mean_obswet,
                out_png=os.path.join(PATH_OUT_QA, f"error_histogram_obswet_mean_{CDE_RUN_DTE}.png"),
            )

            save_residual_vs_obs_plot(
                yr_te[te_obs_wet],
                pred_mean_obswet,
                out_png=os.path.join(PATH_OUT_QA, f"residual_vs_obs_obswet_mean_{CDE_RUN_DTE}.png"),
            )

        # Save holdout point predictions for later inspection.
        holdout_pred_df = pd.DataFrame({
            "obs_R_mm_h": yr_te.astype("float32"),
            "obs_wet": yw_te.astype("int8"),
            "pred_wet_prob": wet_prob_te.astype("float32"),
            "pred_wet_best_thr": te_pred_wet.astype("int8"),
            "pred_R_full_mean_mm_h": pred_full_mean.astype("float32"),
            "pred_R_full_median_mm_h": pred_full_median.astype("float32"),
            "group_id": groups_all[test_idx].astype("int32"),
        })

        holdout_pred_csv = os.path.join(PATH_OUT_QA, f"holdout_predictions_{CDE_RUN_DTE}.csv")
        holdout_pred_df.to_csv(holdout_pred_csv, index=False)
        qa_summary["holdout_predictions_csv"] = holdout_pred_csv

        print("Saved continuous metrics:", metrics_csv)
        print("Saved holdout predictions:", holdout_pred_csv)

    else:
        print("Skipping intensity QA because wet train or predicted-wet holdout sample is too small.")

    # Save QA summary JSON
    qa_json = os.path.join(PATH_OUT_QA, f"internal_holdout_QA_summary_{CDE_RUN_DTE}.json")
    with open(qa_json, "w") as f:
        json.dump(qa_summary, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

    print("Saved threshold curve:", threshold_curve_path)
    print("Saved QA summary:", qa_json)
    print("Internal QA summary:")
    print(json.dumps(qa_summary, indent=2))

else:
    print("Skipping internal holdout QA; using default classifier threshold 0.5.")

#%% Train final models on all sampled data
print("Training final classifier on all sampled data...")
neg_all = np.sum(y_wet_all == 0)
pos_all = np.sum(y_wet_all == 1)
scale_pos_weight_all = neg_all / pos_all if pos_all > 0 else None

dcls_all = xgb.DMatrix(X_all, label=y_wet_all, feature_names=FEAT_NAMES, nthread=NTHREAD)
classifier = train_xgb_classifier(dcls_all, dvalid=None, scale_pos_weight=scale_pos_weight_all, seed=150)

print("Training final wet-only quantile intensity models...")
wet_all = y_wet_all == 1
if wet_all.sum() < 100:
    raise RuntimeError(f"Too few wet samples for intensity training: {wet_all.sum()}")

dreg_all = xgb.DMatrix(X_all[wet_all], label=np.log1p(y_rate_all[wet_all]), feature_names=FEAT_NAMES, nthread=NTHREAD)
boosters_by_q = {q: train_quantile(q, dreg_all, dvalid=None, seed=150) for q in QS_DENSE}

print("Final model training complete.")

#%% Save models and metadata
model_bundle = {
    "classifier": classifier,
    "intensity_quantile_boosters": boosters_by_q,
}

model_path = os.path.join(PATH_OUT_MODEL, f"cmlsat_twostage_xgb_basref_models_{CDE_RUN_DTE}.pkl")
joblib.dump(model_bundle, model_path, compress=3)

meta = {
    "model_type": "two_stage_hurdle_xgb",
    "description": "MSG wet/dry classifier + wet-only XGBoost quantile rainfall intensity using Bas CML-only reference",
    "cml_reference_path": PATH_CML_RAIN,
    "msg_bt_path": PATH_MSG_BT,
    "msg_clm_path": PATH_MSG_CLM,
    "training_period": f"{TRAIN_START}_to_{TRAIN_END}",
    "features": FEAT_NAMES,
    "cloud_mask_rule": "cloud_mask == 2",
    "bbox": {
        "lon_min": LON_MIN, "lon_max": LON_MAX,
        "lat_min": LAT_MIN, "lat_max": LAT_MAX,
    },
    "grid_res_deg": RES_DEG,
    "time_round": TIME_ROUND,
    "wet_thr_mm_h": WET_THR,
    "drizzle_floor_for_target_mm_h": DRIZZLE_FLOOR_FOR_TARGET,
    "wet_frac_sample": WET_FRAC_SAMPLE,
    "n_times_per_month": N_TIMES_PER_MONTH,
    "n_pix_per_time": N_PIX_PER_TIME,
    "qs_dense": QS_DENSE,
    "transform_intensity_target": "log1p",
    "classifier_default_threshold": 0.5,
    "classifier_recommended_threshold": float(threshold_best.get("threshold", 0.5)),
    "scale_pos_weight_final": float(scale_pos_weight_all) if scale_pos_weight_all is not None else None,
    "n_samples_total": int(X_all.shape[0]),
    "n_samples_wet": int(wet_all.sum()),
    "sampled_wet_fraction": float(y_wet_all.mean()),
    "date_trained": CDE_RUN_DTE,
    "qa_summary": qa_summary,
    "application_note": (
        "During mapping, apply classifier first, estimate intensity only where wet probability exceeds threshold, "
        "set dry-classified pixels to zero, then retain existing latitude/zonal IR108 patch correction as final occurrence refinement."
    ),
}

meta_path = model_path.replace(".pkl", "_meta.pkl")
json_path = model_path.replace(".pkl", "_meta.json")
joblib.dump(meta, meta_path)
with open(json_path, "w") as f:
    json.dump(meta, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

print("Saved model bundle:", model_path)
print("Saved metadata:", meta_path)
print("Saved metadata JSON:", json_path)
print("All done.")
