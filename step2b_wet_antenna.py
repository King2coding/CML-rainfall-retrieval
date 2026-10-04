# step2b_wet_antenna.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

# ---------------- Original (no-decay) WA ----------------
@dataclass(frozen=True)
class WAConfig:
    gamma_source_col: str = "A_ex_pool_per_km"   # from Step 2/1b
    wet_mask_col: str = "is_wet_final"           # or "is_wet_excess"
    wa_db_per_terminal: float = 0.08             # 0.05–0.12 dB typical
    assume_wet_terminals: int = 2                # 1 or 2
    out_raw_col: str = "gamma_raw_db_per_km"
    out_corr_col: str = "gamma_corr_db_per_km"
    floor_zero: bool = True

def apply_wet_antenna(df_in: pd.DataFrame, cfg: WAConfig) -> pd.DataFrame:
    """
    Simple WA: subtract fixed per-km WA when wet; 0 when dry.
    Adds: cfg.out_raw_col, cfg.out_corr_col, 'used_gamma'
    """
    need = {"PathLength", cfg.gamma_source_col}
    miss = [c for c in need if c not in df_in.columns]
    if miss:
        raise ValueError(f"Missing columns for WA correction: {miss}")

    df = df_in.copy()
    g_raw = pd.to_numeric(df[cfg.gamma_source_col], errors="coerce")
    df[cfg.out_raw_col] = g_raw

    L = pd.to_numeric(df["PathLength"], errors="coerce").replace(0, np.nan)
    wa_per_km = (cfg.wa_db_per_terminal * cfg.assume_wet_terminals) / L

    wet = df[cfg.wet_mask_col].fillna(False).to_numpy(bool) if cfg.wet_mask_col in df.columns else np.zeros(len(df), bool)
    g_corr = g_raw - np.where(wet, wa_per_km, 0.0)
    if cfg.floor_zero:
        g_corr = g_corr.clip(lower=0.0)

    df[cfg.out_corr_col] = g_corr
    df["used_gamma"] = np.where(wet, "corr", "raw")
    return df

# ---------------- New (decay) WA V2 ---------------------
@dataclass(frozen=True)
class WAConfigV2:
    gamma_source_col: str = "A_ex_pool_per_km"    # per-km “rain-ish” signal
    wet_mask_col: str = "is_wet_final"            # boolean per-sample
    wa_db_per_terminal: float = 0.08              # dB per wet terminal
    assume_wet_terminals: int = 2                 # 1 or 2
    dt_minutes: int = 15                          # sampling period
    tau_on_min: int = 30                          # rise time constant (min)
    tau_off_min: int = 60                         # decay time constant (min)
    out_raw_col: str = "gamma_raw_db_per_km"
    out_corr_col: str = "gamma_corr_db_per_km"
    floor_zero: bool = True

def apply_wet_antenna_decay(df_in: pd.DataFrame, cfg: WAConfigV2) -> pd.DataFrame:
    """
    EWMA WA per link:
      w_t = w_{t-1} + α*(target - w_{t-1}), α = 1-exp(-Δt/τ)
      target = (wa_db_per_terminal*terminals)/L when wet; else 0
    Adds:
      cfg.out_raw_col, cfg.out_corr_col, 'wa_applied_per_km', 'used_gamma'
    """
    need = {"PathLength", cfg.gamma_source_col}
    miss = [c for c in need if c not in df_in.columns]
    if miss:
        raise ValueError(f"WA V2 missing columns: {miss}")

    df = df_in.copy()
    g_raw = pd.to_numeric(df[cfg.gamma_source_col], errors="coerce")
    df[cfg.out_raw_col] = g_raw

    # static WA magnitude per sample
    L = pd.to_numeric(df["PathLength"], errors="coerce").replace(0, np.nan)
    wa_per_km = (cfg.wa_db_per_terminal * cfg.assume_wet_terminals) / L

    # EWMA coefficients
    dt = max(1e-6, float(cfg.dt_minutes))
    a_on  = 1.0 - np.exp(-dt / max(1e-3, float(cfg.tau_on_min)))
    a_off = 1.0 - np.exp(-dt / max(1e-3, float(cfg.tau_off_min)))

    df["wa_applied_per_km"] = 0.0

    # process each link independently (avoid global indexing)
    for lid, g in df.groupby("ID", sort=False):
        g = g.sort_index()  # ensure chronological
        ix = g.index

        # per-group wet flags
        if cfg.wet_mask_col and cfg.wet_mask_col in g.columns:
            wet_g = g[cfg.wet_mask_col].fillna(False).to_numpy(bool)
        else:
            wet_g = np.zeros(len(g), dtype=bool)

        # per-group WA targets (finite only)
        wa_g = wa_per_km.loc[ix].to_numpy(float)
        wa_g = np.where(np.isfinite(wa_g), wa_g, 0.0)

        w = 0.0
        for j, irow in enumerate(ix):
            target = wa_g[j] if wet_g[j] else 0.0
            alpha  = a_on if target > w else a_off
            w = w + alpha * (target - w)
            df.at[irow, "wa_applied_per_km"] = max(0.0, w)

    gamma_corr = g_raw - df["wa_applied_per_km"]
    if cfg.floor_zero:
        gamma_corr = gamma_corr.clip(lower=0.0)

    df[cfg.out_corr_col] = gamma_corr
    # mark which path we used; this is informational
    df["used_gamma"] = np.where(
        (df.get(cfg.wet_mask_col, False)).fillna(False).to_numpy(bool),
        "corr_decay", "raw_decay"
    )
    return df