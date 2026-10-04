# step4_wet_antenna.py
# STEP 4 — Wet-antenna correction
#
# Inputs: df_s3 (from Step 3) with at least:
#   ['ID','A_path_db_raw','gamma_raw_db_per_km','is_wet','Frequency','PathLength']
#
# Outputs (added):
#   'Aw_per_end_db'        : wet-antenna attenuation per end used [dB]
#   'Aw_total_db'          : total wet-antenna attenuation applied (n_ends * Aw_per_end_db) [dB]
#   'A_path_db_corr'       : corrected path attenuation [dB]
#   'gamma_corr_db_per_km' : corrected specific attenuation [dB/km]
#   'wa_applied'           : bool (True when correction used)
#
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple
import numpy as np
import pandas as pd


# --- default frequency rule (editable): very rough, conservative values
def default_aw_rule(f_ghz: float, pol: str | None = None) -> float:
    """
    Return a per-antenna wet attenuation [dB] as a function of frequency and polarization.
    These are placeholders; tune with local calibration if available.
    """
    if np.isnan(f_ghz):
        return 0.5
    if f_ghz < 10:    aw = 0.25
    elif f_ghz < 20: aw = 0.5
    elif f_ghz < 40: aw = 1.0
    else:            aw = 1.5
    # small tweak for polarization if desired (typically minor):
    if pol == "V":
        aw *= 1.0
    elif pol == "H":
        aw *= 1.0
    return float(aw)


@dataclass(frozen=True)
class WetAntennaConfig:
    # Either provide a fixed per-end Aw, or a callable to compute it from (freq, pol).
    fixed_aw_per_end_db: float | None = None
    aw_rule: Callable[[float, str | None], float] = default_aw_rule
    # Number of ends assumed wet when is_wet is True. 2 means both antennas wet.
    n_wet_ends: int = 2
    # Only apply correction when is_wet==True
    apply_only_when_wet: bool = True
    # Do not allow negative corrected attenuation
    clip_negative: bool = True


def apply_wet_antenna_correction(df_s3: pd.DataFrame,
                                 cfg: WetAntennaConfig = WetAntennaConfig()
                                 ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    need = ["ID","A_path_db_raw","gamma_raw_db_per_km","PathLength","is_wet"]
    miss = [c for c in need if c not in df_s3.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    out = df_s3.copy()

    # Resolve per-end Aw for each row
    if cfg.fixed_aw_per_end_db is not None:
        aw_per_end = np.full(len(out), float(cfg.fixed_aw_per_end_db), dtype="float64")
    else:
        f = pd.to_numeric(out.get("Frequency", np.nan), errors="coerce").astype("float64")
        pol = out.get("Polarization", None)
        if pol is not None:
            pol = pol.astype("string")
        aw_per_end = np.array([cfg.aw_rule(fi, (pi if isinstance(pi, str) else (None if pd.isna(pi) else str(pi))))
                               for fi, pi in zip(f, (pol if pol is not None else [None]*len(f)))],
                              dtype="float64")

    n_ends = int(cfg.n_wet_ends)
    Aw_total = n_ends * aw_per_end  # dB

    # Apply only when wet (else zero)
    if cfg.apply_only_when_wet:
        wet_mask = out["is_wet"].fillna(False).to_numpy()
        Aw_total = np.where(wet_mask, Aw_total, 0.0)

    # Correct path attenuation
    A_corr = out["A_path_db_raw"].astype("float64") - Aw_total
    if cfg.clip_negative:
        A_corr = np.maximum(0.0, A_corr)

    # Correct specific attenuation (per km)
    L = pd.to_numeric(out["PathLength"], errors="coerce").astype("float64")
    gamma_corr = np.where(L > 0, A_corr / L, np.nan)

    out["Aw_per_end_db"] = aw_per_end.astype("float32")
    out["Aw_total_db"] = Aw_total.astype("float32")
    out["A_path_db_corr"] = A_corr.astype("float32")
    out["gamma_corr_db_per_km"] = gamma_corr.astype("float32")
    out["wa_applied"] = (Aw_total > 0).astype(bool)

    # Summary
    summ = []
    for link_id, g in out.groupby("ID"):
        src = g[g.get("src_present", True)]
        summ.append({
            "ID": link_id,
            "n_src": int(len(src)),
            "frac_wa_applied": float(src["wa_applied"].mean()),
            "median_A_raw": float(np.nanmedian(src["A_path_db_raw"])),
            "median_A_corr": float(np.nanmedian(src["A_path_db_corr"])),
        })
    s4_summary = pd.DataFrame(summ).sort_values("ID").reset_index(drop=True)
    return out, s4_summary
