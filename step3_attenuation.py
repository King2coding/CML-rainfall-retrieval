# step3_attenuation.py
# STEP 3 — Compute path attenuation from dry-baseline excess
#
# Inputs: df_step2 (from Step 2) with columns at least:
#   ['ID','A_excess_db','A_excess_db_per_km','PathLength','is_wet','Frequency','Polarization']
#   DatetimeIndex (UTC), 15-min cadence.
#
# Outputs (added):
#   'A_path_db_raw'        : path attenuation above dry baseline [dB]
#   'gamma_raw_db_per_km'  : specific attenuation [dB/km] = A_path_db_raw / PathLength
#   'attn_valid'           : boolean flag (finite, length ok, etc.)
#
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AttenuationConfig:
    # Minimum plausible electrical path length (km) to trust per-km normalization.
    # Values below this will be treated as NaN for per-km quantities.
    min_len_km: float = 0.5
    # Optional guard: cap absurd per-km values (e.g., glitches on very short links)
    max_gamma_db_per_km: float | None = None   # e.g., 15.0 for low bands, or None for no cap
    # Clip negatives (should already be 0 from Step 2, but keep defensive)
    clip_negative: bool = True


def compute_path_attenuation(df_step2: pd.DataFrame,
                             cfg: AttenuationConfig = AttenuationConfig()
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    need = ["ID","A_excess_db","PathLength"]
    miss = [c for c in need if c not in df_step2.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}")

    out = df_step2.copy()

    # 1) Path attenuation above dry baseline (dB)
    A = out["A_excess_db"].astype("float64")
    if cfg.clip_negative:
        A = np.maximum(0.0, A)

    # 2) Specific attenuation (dB/km) — guard tiny/invalid lengths
    L = pd.to_numeric(out["PathLength"], errors="coerce").astype("float64")
    L_ok = (L >= cfg.min_len_km)
    gamma = np.full(len(out), np.nan, dtype="float64")
    gamma[L_ok.values] = A[L_ok.values] / L[L_ok.values]
    if cfg.max_gamma_db_per_km is not None:
        gamma = np.minimum(gamma, cfg.max_gamma_db_per_km)

    out["A_path_db_raw"] = A.astype("float32")
    out["gamma_raw_db_per_km"] = gamma.astype("float32")
    out["attn_valid"] = np.isfinite(out["A_path_db_raw"]) & np.isfinite(out["gamma_raw_db_per_km"]) & L_ok.values

    # Per-link QA summary
    summ = []
    for link_id, g in out.groupby("ID"):
        src = g[g.get("src_present", True)]
        summ.append({
            "ID": link_id,
            "n_src": int(len(src)),
            "attn_valid_frac": float(src["attn_valid"].mean()),
            "median_A_path_db": float(np.nanmedian(src["A_path_db_raw"])),
            "median_gamma_db_per_km": float(np.nanmedian(src["gamma_raw_db_per_km"])),
        })
    s3_summary = pd.DataFrame(summ).sort_values("ID").reset_index(drop=True)
    return out, s3_summary
