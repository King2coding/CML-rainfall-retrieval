# r0_qc_helpers.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def masking_report(df_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Rates per link computed over the true source samples only (src_present==True),
    so regularized filler rows don't dilute the stats.
    """
    tags = ["OOB","DYN","OUTAGE","UNPAIRED","SPIKE","PLAT"]
    out = []
    for link, g in df_clean.groupby("ID"):
        src = g[g.get("src_present", False)]
        row = {"ID": link, "n_src": len(src)}
        qcs = src["qc_flags"].fillna("")
        for t in tags:
            row[f"frac_{t.lower()}_src"] = float(qcs.str.contains(t).mean()) if len(src) else 0.0
        row["valid_pairs_frac_src"] = float(src[["Pmin","Pmax"]].dropna().shape[0] / max(len(src),1))
        out.append(row)
    return pd.DataFrame(out).sort_values("valid_pairs_frac_src")

def plot_flags(df_clean: pd.DataFrame, link_id: str, t0=None, t1=None) -> None:
    """Overlay QC flags on Pmin/Pmax for a chosen link."""
    d = df_clean[df_clean["ID"] == link_id].copy()
    if d.empty:
        print("No rows for", link_id); return
    if t0 is not None: d = d[d.index >= pd.to_datetime(t0)]
    if t1 is not None: d = d[d.index <= pd.to_datetime(t1)]
    qc = d["qc_flags"].fillna("")
    fig, ax = plt.subplots(1,2, figsize=(15,4), sharex=True)
    for j, col in enumerate(["Pmin","Pmax"]):
        ax[j].plot(d.index, d[col], lw=1, label=col)
        for tag in ["OOB","DYN","OUTAGE","UNPAIRED","SPIKE","PLAT"]:
            mk = qc.str.contains(tag)
            if mk.any():
                ax[j].scatter(d.index[mk], d[col][mk], s=18, label=tag)
        ax[j].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        ax[j].xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax[j].xaxis.get_majorticklabels(), rotation=45, ha="right")
        ax[j].grid(True, alpha=0.3)
        ax[j].legend(ncol=3, fontsize=8)
        ax[j].set_title(f"{col} — {link_id}")
    plt.tight_layout(); plt.show()

def plot_minmax_basic(one_link: pd.DataFrame) -> None:
    """Your quick plotting recipe for a single link subset."""
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(one_link.index, one_link['Pmin'])
    ax[0].set_title('Pmin')
    ax[1].plot(one_link.index, one_link['Pmax'])
    ax[1].set_title('Pmax')
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax[0].xaxis.set_major_locator(mdates.AutoDateLocator())
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax[1].xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax[0].xaxis.get_majorticklabels(), rotation=45, ha="right")
    plt.setp(ax[1].xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax[0].grid(True); ax[1].grid(True)
    plt.show()
