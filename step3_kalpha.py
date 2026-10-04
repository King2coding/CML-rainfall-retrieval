# step3_kalpha.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

# ---------------- Original (ungated) k–α ----------------
@dataclass(frozen=True)
class KAlphaConfig:
    lut: pd.DataFrame                  # columns: Frequency_GHz,kH,αH,kV,αV
    pol_col: str = "Polarization"      # 'H' or 'V'
    freq_col: str = "Frequency"        # GHz
    gamma_col: str = "gamma_corr_db_per_km"
    r_cap_mmph: float | None = None    # optional cap

def _interp_coeffs(freq_ghz: np.ndarray, pol: np.ndarray, lut: pd.DataFrame):
    fgrid = lut["Frequency_GHz"].to_numpy()
    kH, aH = lut["kH"].to_numpy(), lut["αH"].to_numpy()
    kV, aV = lut["kV"].to_numpy(), lut["αV"].to_numpy()
    k_hi = np.interp(freq_ghz, fgrid, kH); a_hi = np.interp(freq_ghz, fgrid, aH)
    k_vi = np.interp(freq_ghz, fgrid, kV); a_vi = np.interp(freq_ghz, fgrid, aV)
    pol_up = np.char.upper(pol.astype(str))
    return np.where(pol_up=="H", k_hi, k_vi), np.where(pol_up=="H", a_hi, a_vi)

def gamma_to_r(df_attn: pd.DataFrame, cfg: KAlphaConfig):
    need = [cfg.gamma_col, cfg.freq_col, cfg.pol_col, "ID"]
    miss = [c for c in need if c not in df_attn.columns]
    if miss:
        raise ValueError(f"Missing columns for k–α conversion: {miss}")
    df = df_attn.copy()

    gamma = pd.to_numeric(df[cfg.gamma_col], errors="coerce").to_numpy(float)
    freq  = pd.to_numeric(df[cfg.freq_col], errors="coerce").to_numpy(float)
    pol   = df[cfg.pol_col].to_numpy()

    k, a = _interp_coeffs(freq, pol, cfg.lut)
    good = np.isfinite(gamma) & (gamma > 0) & np.isfinite(k) & (k > 0) & np.isfinite(a) & (a > 0)
    R = np.zeros_like(gamma, float)
    R[good] = (gamma[good] / k[good]) ** (1.0 / a[good])
    if cfg.r_cap_mmph is not None:
        R = np.minimum(R, float(cfg.r_cap_mmph))

    df["k_used"] = k
    df["alpha_used"] = a
    df["R_mm_per_h"] = R

    rows = []
    for lid, g in df.groupby("ID", sort=False):
        gg = g[np.isfinite(g["R_mm_per_h"])]
        rows.append({
            "ID": lid,
            "n_src": int(len(g)),
            "frac_R_pos": float((g["R_mm_per_h"] > 0).mean()),
            "R50": float(np.nanpercentile(gg["R_mm_per_h"], 50)) if len(gg) else 0.0,
            "R95": float(np.nanpercentile(gg["R_mm_per_h"], 95)) if len(gg) else 0.0,
        })
    s = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    return df, s

# ---------------- New (gated) k–α V2 --------------------
@dataclass(frozen=True)
class KAlphaConfigV2:
    lut: pd.DataFrame
    pol_col: str = "Polarization"
    freq_col: str = "Frequency"
    gamma_col: str = "gamma_corr_db_per_km"   # typically from WA V2
    gamma_gate_db_per_km: float = 0.02        # ignore tiny γ
    use_wet_mask_col: str | None = "is_wet_final"
    r_cap_mmph_by_band: dict | None = None    # e.g., {6:60, 8:80, 19:120}

def _interp_coeffs_v2(freq_ghz: np.ndarray, pol: np.ndarray, lut: pd.DataFrame):
    return _interp_coeffs(freq_ghz, pol, lut)

def gamma_to_r_gated(df_attn: pd.DataFrame, cfg: KAlphaConfigV2):
    need = [cfg.gamma_col, cfg.freq_col, cfg.pol_col, "ID"]
    miss = [c for c in need if c not in df_attn.columns]
    if miss:
        raise ValueError(f"Missing columns for k–α V2: {miss}")

    df = df_attn.copy()
    gamma = pd.to_numeric(df[cfg.gamma_col], errors="coerce").to_numpy(float)
    freq  = pd.to_numeric(df[cfg.freq_col],  errors="coerce").to_numpy(float)
    pol   = df[cfg.pol_col].to_numpy()

    wetmask = np.ones_like(gamma, bool)
    if cfg.use_wet_mask_col and cfg.use_wet_mask_col in df.columns:
        wetmask = df[cfg.use_wet_mask_col].fillna(False).to_numpy(bool)
    g_used = np.where(wetmask & np.isfinite(gamma) & (gamma >= float(cfg.gamma_gate_db_per_km)),
                      gamma, 0.0)

    k, a = _interp_coeffs_v2(freq, pol, cfg.lut)
    ok = (g_used > 0) & np.isfinite(k) & (k > 0) & np.isfinite(a) & (a > 0)
    R = np.zeros_like(gamma, float)
    R[ok] = (g_used[ok] / k[ok]) ** (1.0 / a[ok])

    if cfg.r_cap_mmph_by_band:
        cap = np.full_like(freq, np.nan, float)
        for fkey, rcap in sorted(cfg.r_cap_mmph_by_band.items()):
            cap[np.abs(freq - float(fkey)) <= 1.0] = float(rcap)
        use_cap = np.isfinite(cap)
        R[use_cap] = np.minimum(R[use_cap], cap[use_cap])

    df["gamma_for_R"] = g_used
    df["R_mm_per_h"]  = R

    rows = []
    for lid, g in df.groupby("ID", sort=False):
        gg = g[np.isfinite(g["R_mm_per_h"])]
        rows.append({
            "ID": lid,
            "n_src": int(len(g)),
            "frac_R_pos": float((g["R_mm_per_h"] > 0).mean()),
            "R50": float(np.nanpercentile(gg["R_mm_per_h"], 50)) if len(gg) else 0.0,
            "R95": float(np.nanpercentile(gg["R_mm_per_h"], 95)) if len(gg) else 0.0,
        })
    s = pd.DataFrame(rows).sort_values("ID").reset_index(drop=True)
    return df, s