# step2_plot_helpers.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def _runs(index: pd.DatetimeIndex, mask: np.ndarray, fallback_minutes=15):
    if len(index) <= 1: return []
    dt = pd.Series(index).diff().median()
    if pd.isna(dt): dt = pd.Timedelta(minutes=fallback_minutes)
    runs, start, in_run = [], None, False
    for i, f in enumerate(mask):
        if f and not in_run: in_run, start = True, index[i]
        elif (not f) and in_run: runs.append((start, index[i])); in_run = False
    if in_run: runs.append((start, index[-1] + dt))
    return runs

def plot_baseline_overlay(df_step2: pd.DataFrame, link_id: str, t0=None, t1=None):
    d = df_step2[df_step2["ID"] == link_id].copy()
    if d.empty: print("No rows for", link_id); return
    # ensure UTC awareness for safe slicing
    if isinstance(d.index, pd.DatetimeIndex):
        d.index = d.index.tz_localize("UTC") if d.index.tz is None else d.index.tz_convert("UTC")
    if t0 is not None: d = d[d.index >= pd.to_datetime(t0).tz_localize("UTC")]
    if t1 is not None: d = d[d.index <= pd.to_datetime(t1).tz_localize("UTC")]
    if d.empty: print("No rows in window"); return

    wet_mask = d["is_wet"].fillna(False).to_numpy()
    spans = _runs(d.index, wet_mask)

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Top: Abar + baseline (shade wet)
    ax[0].plot(d.index, d["Abar"], lw=1, label="Abar")
    ax[0].plot(d.index, d["baseline_db"], lw=1.5, label="baseline (dry, ≤48h)")
    for s, e in spans: ax[0].axvspan(s, e, alpha=0.12, lw=0)
    ax[0].set_title(f"Abar & dry baseline — {link_id}")
    ax[0].legend(); ax[0].grid(True, alpha=0.3)

    # Bottom: Excess attenuation
    ax[1].plot(d.index, d["A_excess_db"], lw=1, label="A_excess (dB)")
    ax[1].plot(d.index, d["A_excess_db_per_km"], lw=1, label="A_excess (dB/km)")
    for s, e in spans: ax[1].axvspan(s, e, alpha=0.12, lw=0)
    ax[1].axhline(0.0, lw=0.8)
    ax[1].legend(ncol=2); ax[1].grid(True, alpha=0.3)

    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout(); plt.show()