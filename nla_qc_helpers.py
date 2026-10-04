# nla_qc_helpers.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def wetdry_report(df_nla: pd.DataFrame) -> pd.DataFrame:
    """Compact per-link wet/dry coverage report (over src_present only)."""
    out = []
    for link, g in df_nla.groupby("ID"):
        src = g[g.get("src_present", True)]
        row = {
            "ID": link,
            "n_src": len(src),
            "n_nb_ok": int((src["nb_count"] >= 3).sum()),
            "wet_frac_src": float(src["is_wet"].mean()) if len(src) else 0.0,
        }
        out.append(row)
    return pd.DataFrame(out).sort_values("wet_frac_src", ascending=False)

def _to_utc(x):
    ts = pd.to_datetime(x)
    return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")

def _contiguous_true_runs(index: pd.DatetimeIndex, mask: np.ndarray):
    """Return list of (start_ts, end_ts) for contiguous True runs."""
    if not mask.any():
        return []
    idx = np.where(mask)[0]
    # split where gaps > 1
    splits = np.where(np.diff(idx) > 1)[0] + 1
    groups = np.split(idx, splits)
    spans = []
    for g in groups:
        s = index[g[0]]
        e = index[g[-1]]
        spans.append((s, e))
    return spans

def plot_wetdry_overlay(df_like, link_id, t0=None, t1=None,
                        thr_self_db=None, thr_nb_db=None,
                        thr_self_db_per_km=None, thr_nb_db_per_km=None,
                        show_per_km=False, shade_alpha=0.15):
    import matplotlib.pyplot as plt, matplotlib.dates as mdates
    import pandas as pd, numpy as np

    d = df_like[df_like["ID"] == link_id].copy()
    if d.empty:
        print("No rows for", link_id); return

    # UTC index
    if isinstance(d.index, pd.DatetimeIndex):
        d.index = d.index.tz_localize("UTC") if d.index.tz is None else d.index.tz_convert("UTC")
    if t0 is not None: d = d[d.index >= pd.to_datetime(t0, utc=True)]
    if t1 is not None: d = d[d.index <= pd.to_datetime(t1, utc=True)]
    if d.empty:
        print("No rows in requested window."); return

    # Which mode?
    has_delta_new = {"dA_self_db","dA_nb_med_db"}.issubset(d.columns)
    has_delta_old = {"delta_db","nb_med_delta_db"}.issubset(d.columns)
    has_excess    = {"A_ex_pool_per_km","nb_med_A_ex_per_km"}.issubset(d.columns)

    mode = None
    if has_delta_new or has_delta_old:
        mode = "delta"
        x_self = "dA_self_db" if has_delta_new else "delta_db"
        x_nb   = "dA_nb_med_db" if has_delta_new else "nb_med_delta_db"
    elif has_excess:
        mode = "excess"
        x_self = "A_ex_pool_per_km"
        x_nb   = "nb_med_A_ex_per_km"
        show_per_km = True
    else:
        raise ValueError("Neither Δ nor Excess columns are present to plot.")

    # wet mask + spans
    wet_mask = d.get("is_wet", pd.Series(False, index=d.index)).fillna(False).to_numpy(bool)
    def _spans(idx, mask):
        spans = []
        if mask.any():
            on = np.flatnonzero(mask)
            starts = [on[0]]
            for i in range(1, len(on)):
                if on[i] != on[i-1] + 1:
                    spans.append((idx[starts[-1]], idx[on[i-1]]))
                    starts.append(on[i])
            spans.append((idx[starts[-1]], idx[on[-1]]))
        return spans
    wet_spans = _spans(d.index, wet_mask)

    # plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # top: Pmin/Pmax if available
    if {"Pmin","Pmax"}.issubset(d.columns):
        ax[0].plot(d.index, d["Pmin"], lw=1, label="Pmin")
        ax[0].plot(d.index, d["Pmax"], lw=1, label="Pmax")
    else:
        ax[0].plot(d.index, d[x_self], lw=1, label=x_self)

    if wet_mask.any():
        wet_idx = d.index[wet_mask]
        if "Pmin" in d:
            ax[0].scatter(wet_idx, d.loc[wet_idx, "Pmin"], s=18, label="WET", marker="o")
        for s, e in wet_spans:
            ax[0].axvspan(s, e, alpha=shade_alpha, lw=0)
    ax[0].set_title(f"Pmin/Pmax — {link_id}")
    ax[0].grid(True, alpha=0.3); ax[0].legend(ncol=3, fontsize=8)

    # bottom: diagnostics
    ax[1].plot(d.index, d[x_self], lw=1, label=("Δ self (dB)" if mode=="delta" else "A_excess per-km (dB/km)"))
    ax[1].plot(d.index, d[x_nb],   lw=1, label=("median Δ neighbors (dB)" if mode=="delta" else "median neighbors (per-km)"))
    ax[1].axhline(0.0, lw=0.8, alpha=0.6)

    if mode == "delta" and (thr_self_db is not None or thr_nb_db is not None):
        if thr_self_db is not None: ax[1].axhline(thr_self_db, ls="--", lw=1, label="self thr")
        if thr_nb_db  is not None: ax[1].axhline(thr_nb_db,  ls=":",  lw=1, label="neighbor thr")
    if mode == "excess" and (thr_self_db_per_km is not None or thr_nb_db_per_km is not None):
        if thr_self_db_per_km is not None: ax[1].axhline(thr_self_db_per_km, ls="--", lw=1, label="self thr/km")
        if thr_nb_db_per_km  is not None: ax[1].axhline(thr_nb_db_per_km,  ls=":",  lw=1, label="neighbor thr/km")

    if wet_mask.any():
        for s, e in wet_spans:
            ax[1].axvspan(s, e, alpha=shade_alpha, lw=0)
    ax[1].set_ylabel("dB" if mode=="delta" else "dB/km")
    ax[1].grid(True, alpha=0.3); ax[1].legend(ncol=3, fontsize=8)

    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout(); plt.show()



# --- quick excess overlay plotter (no file edits needed) ---
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib.dates as mdates

def plot_excess_overlay(df_like, link_id, t0=None, t1=None,
                        thr_self_db_per_km=None, thr_nb_db_per_km=None,
                        shade_alpha=0.15):
    """
    Overlay wet periods from the *excess-based* classifier on top of the per-km excess series.
    Expects columns: ['ID','A_ex_pool_per_km','nb_med_A_ex_per_km','is_wet_excess'].
    Uses Pmin/Pmax for context if present; otherwise shows the per-km series on top.
    """
    d = df_like[df_like["ID"] == link_id].copy()
    if d.empty:
        print("No rows for", link_id); return

    # Ensure UTC-aware time index and window filter
    if isinstance(d.index, pd.DatetimeIndex):
        d.index = d.index.tz_localize("UTC") if d.index.tz is None else d.index.tz_convert("UTC")
    to_utc = lambda t: pd.to_datetime(t).tz_localize("UTC") if pd.to_datetime(t).tzinfo is None else pd.to_datetime(t).tz_convert("UTC")
    if t0 is not None: d = d[d.index >= to_utc(t0)]
    if t1 is not None: d = d[d.index <= to_utc(t1)]
    if d.empty:
        print("No rows in requested window."); return

    need = {"A_ex_pool_per_km","nb_med_A_ex_per_km","is_wet_excess"}
    missing = [c for c in need if c not in d.columns]
    if missing:
        raise ValueError(f"Missing columns for excess plot: {missing}")

    # wet spans
    wet_mask = d["is_wet_excess"].fillna(False).to_numpy(bool)
    spans = []
    if wet_mask.any():
        on = np.flatnonzero(wet_mask)
        start = on[0]
        for i in range(1, len(on)):
            if on[i] != on[i-1] + 1:
                spans.append((d.index[start], d.index[on[i-1]]))
                start = on[i]
        spans.append((d.index[start], d.index[on[-1]]))

    # figure
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Top panel: Pmin/Pmax if available; else plot A_ex_pool_per_km
    if {"Pmin","Pmax"}.issubset(d.columns):
        ax[0].plot(d.index, d["Pmin"], lw=1, label="Pmin")
        ax[0].plot(d.index, d["Pmax"], lw=1, label="Pmax")
        if wet_mask.any():
            wet_idx = d.index[wet_mask]
            ax[0].scatter(wet_idx, d.loc[wet_idx, "Pmin"], s=18, label="WET", marker="o")
        ax[0].set_title(f"Pmin/Pmax — {link_id}")
        ax[0].legend(ncol=3, fontsize=8)
    else:
        ax[0].plot(d.index, d["A_ex_pool_per_km"], lw=1, label="A_excess per km")
        ax[0].set_title(f"A_excess per km — {link_id}")
        ax[0].legend(fontsize=8)
    for s,e in spans: ax[0].axvspan(s, e, alpha=shade_alpha, lw=0)
    ax[0].grid(True, alpha=0.3)

    # Bottom panel: per-km excess + neighbor median
    ax[1].plot(d.index, d["A_ex_pool_per_km"], lw=1, label="A_excess per km")
    ax[1].plot(d.index, d["nb_med_A_ex_per_km"], lw=1, label="median neighbors (per km)")
    if thr_self_db_per_km is not None: ax[1].axhline(thr_self_db_per_km, ls="--", lw=1, label="self thr/km")
    if thr_nb_db_per_km  is not None: ax[1].axhline(thr_nb_db_per_km,  ls=":",  lw=1, label="neighbor thr/km")
    for s,e in spans: ax[1].axvspan(s, e, alpha=shade_alpha, lw=0)
    ax[1].set_ylabel("dB/km"); ax[1].grid(True, alpha=0.3); ax[1].legend(ncol=3, fontsize=8)

    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.tight_layout(); plt.show()
