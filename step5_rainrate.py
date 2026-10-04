# step5_rainrate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
import pandas as pd

# ---------------- ITU-LUT provider ----------------
def make_itu_provider_from_lut(lut: pd.DataFrame) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Build a (k, alpha) provider from an ITU Table 5 LUT.
    Expected columns: ['Frequency_GHz','kH','αH','kV','αV']  (alpha column name can be 'aH'/'aV' too).
    Interpolation: log10-linear for k, linear for alpha. Clamped to table range.
    """
    lut = lut.copy()
    # normalize alpha column names if they came in as 'αH'/'αV' or 'aH'/'aV'
    for want, alts in {"αH": ["αH", "aH", "alphaH", "alpha_H"],
                       "αV": ["αV", "aV", "alphaV", "alpha_V"]}.items():
        for c in alts:
            if c in lut.columns:
                lut.rename(columns={c: want}, inplace=True)
                break

    ftab = lut["Frequency_GHz"].to_numpy(dtype="float64")
    kH = lut["kH"].to_numpy(dtype="float64")
    kV = lut["kV"].to_numpy(dtype="float64")
    aH = lut["αH"].to_numpy(dtype="float64")
    aV = lut["αV"].to_numpy(dtype="float64")

    logkH = np.log10(kH)
    logkV = np.log10(kV)

    def provider(freq_GHz: np.ndarray, pol: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        f = np.asarray(freq_GHz, dtype="float64")
        p = np.asarray(pol)
        # default to 'H' if unspecified
        p = np.where(pd.isna(p), "H", p).astype(str)

        # Interpolate per polarization
        logk_out = np.where(
            np.char.upper(p) == "V",
            np.interp(f, ftab, logkV, left=logkV[0], right=logkV[-1]),
            np.interp(f, ftab, logkH, left=logkH[0], right=logkH[-1]),
        )
        a_out = np.where(
            np.char.upper(p) == "V",
            np.interp(f, ftab, aV, left=aV[0], right=aV[-1]),
            np.interp(f, ftab, aH, left=aH[0], right=aH[-1]),
        )
        k_out = np.power(10.0, logk_out)
        return k_out, a_out

    return provider

# ---------------- core conversion ----------------
@dataclass(frozen=True)
class RainrateConfig:
    use_corrected_gamma: bool = True   # use 'gamma_corr_db_per_km' if True else 'gamma_raw_db_per_km'
    zero_when_dry: bool = True
    k_alpha_provider: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] = None
    max_R_mm_per_h: float | None = 400.0
    min_gamma_db_per_km: float = 0.0
    eps_k: float = 1e-9

def gamma_to_rainrate(df: pd.DataFrame, cfg: RainrateConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    need_gamma = "gamma_corr_db_per_km" if cfg.use_corrected_gamma else "gamma_raw_db_per_km"
    need = ["ID", "Frequency", "Polarization", "is_wet", need_gamma]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")
    if cfg.k_alpha_provider is None:
        raise ValueError("RainrateConfig.k_alpha_provider must be set (use make_itu_provider_from_lut(lut)).")

    out = df.copy()

    # choose γ
    gamma = pd.to_numeric(out[need_gamma], errors="coerce").astype("float64")
    gamma = np.maximum(cfg.min_gamma_db_per_km, gamma)
    used = "corr" if cfg.use_corrected_gamma else "raw"

    # k, α from provider
    f = pd.to_numeric(out["Frequency"], errors="coerce").to_numpy(dtype="float64")
    pol = out["Polarization"].astype("string").to_numpy()
    k, alpha = cfg.k_alpha_provider(f, pol)

    # R = (γ/k)^(1/α)
    R = np.power(np.maximum(0.0, gamma) / np.maximum(cfg.eps_k, k), 1.0 / np.maximum(1e-6, alpha))

    if cfg.zero_when_dry:
        wet = out["is_wet"].fillna(False).to_numpy()
        R = np.where(wet, R, 0.0)
    if cfg.max_R_mm_per_h is not None:
        R = np.minimum(R, float(cfg.max_R_mm_per_h))

    out["R_mm_per_h"] = R.astype("float32")
    out["k_coeff"] = k.astype("float32")
    out["alpha_coeff"] = alpha.astype("float32")
    out["used_gamma"] = used

    # per-link QA
    summ = []
    for link_id, g in out.groupby("ID"):
        src = g[g.get("src_present", True)]
        wet = src["is_wet"].fillna(False)
        summ.append({
            "ID": link_id,
            "n_src": int(len(src)),
            "wet_frac_src": float(wet.mean()),
            "median_R_wet": float(np.nanmedian(src.loc[wet, "R_mm_per_h"])) if wet.any() else 0.0,
            "p95_R_wet": float(np.nanpercentile(src.loc[wet, "R_mm_per_h"], 95)) if wet.any() else 0.0,
        })
    return out, pd.DataFrame(summ).sort_values("ID").reset_index(drop=True)
