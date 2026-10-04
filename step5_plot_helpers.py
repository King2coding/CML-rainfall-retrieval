# step5_plot_helpers.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def _to_utc(ts):
    ts = pd.to_datetime(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

def _spans_from_mask(idx: pd.DatetimeIndex, mask: np.ndarray):
    spans = []
    if not mask.any():
        return spans
    on = np.flatnonzero(mask)
    start = on[0]
    for i in range(1, len(on)):
        if on[i] != on[i-1] + 1:
            spans.append((idx[start], idx[on[i-1]]))
            start = on[i]
    spans.append((idx[start], idx[on[-1]]))
    return spans

def plot_rainrate(df_like: pd.DataFrame, link_id: str, t0=None, t1=None,
                  wet_mask_col: str | None = None, title: str | None = None):
    """
    Quick QA plot for a link:
      Top: chosen γ (dB/km)
      Bottom: rain rate R (mm/h) if available
    Picks γ in this order of preference: gamma_corr_db_per_km, gamma_raw_db_per_km, A_ex_pool_per_km.
    If wet_mask_col is provided and present, shades wet spans.

    Parameters
    ----------
    df_like : DataFrame containing columns:
        'ID', time index (DatetimeIndex), and one of:
          - 'gamma_corr_db_per_km' (preferred if present)
          - 'gamma_raw_db_per_km'
          - 'A_ex_pool_per_km' (as a fallback visual)
        Optional: 'R_mm_per_h', 'used_gamma'
    link_id : str
    t0, t1 : str | Timestamp | None  (window)
    wet_mask_col : str | None  (e.g., 'is_wet_final' or 'is_wet_excess')
    title : str | None
    """
    d = df_like[df_like["ID"] == link_id].copy()
    if d.empty:
        print("No rows for", link_id); return

    # Ensure UTC-aware index
    if isinstance(d.index, pd.DatetimeIndex):
        d.index = d.index.tz_localize("UTC") if d.index.tz is None else d.index.tz_convert("UTC")

    # Window
    if t0 is not None: d = d[d.index >= _to_utc(t0)]
    if t1 is not None: d = d[d.index <= _to_utc(t1)]
    if d.empty:
        print("No rows in requested window."); return

    cols = set(d.columns)

    # Choose γ column
    gcol = None
    if "gamma_corr_db_per_km" in cols:
        # If 'used_gamma' exists and contains 'corr' anywhere, prefer corr;
        # otherwise still prefer corr if raw is missing.
        if "used_gamma" in cols:
            if "corr" in pd.Series(d["used_gamma"]).astype(str).unique():
                gcol = "gamma_corr_db_per_km"
        if gcol is None:
            gcol = "gamma_corr_db_per_km"
    elif "gamma_raw_db_per_km" in cols:
        gcol = "gamma_raw_db_per_km"
    elif "A_ex_pool_per_km" in cols:
        gcol = "A_ex_pool_per_km"
    else:
        raise ValueError("No γ-like column found (expected one of: "
                         "'gamma_corr_db_per_km', 'gamma_raw_db_per_km', 'A_ex_pool_per_km').")

    # Prepare wet spans if mask is provided
    spans = []
    if wet_mask_col and wet_mask_col in cols:
        wet_mask = d[wet_mask_col].fillna(False).to_numpy(bool)
        spans = _spans_from_mask(d.index, wet_mask)

    # Build figure
    have_R = "R_mm_per_h" in cols
    nrows = 2 if have_R else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 5 if nrows==1 else 6), sharex=True)
    ax = axes if nrows == 2 else [axes]

    # Top: γ
    ax[0].plot(d.index, d[gcol], label=gcol)
    ax[0].set_ylabel("γ [dB/km]" if "per_km" in gcol else "A_ex per km [dB/km]")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=9)
    for s,e in spans: ax[0].axvspan(s, e, alpha=0.15, lw=0)

    # Bottom: R (if present)
    if have_R:
        ax[1].plot(d.index, d["R_mm_per_h"], label="R [mm/h]")
        ax[1].set_ylabel("R [mm/h]"); ax[1].grid(True, alpha=0.3); ax[1].legend(fontsize=9)
        for s,e in spans: ax[1].axvspan(s, e, alpha=0.15, lw=0)

    # X axis
    ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Title
    ttl = title or f"γ and R — {link_id}"
    fig.suptitle(ttl)
    plt.tight_layout()
    plt.show()
