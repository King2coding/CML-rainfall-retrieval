# step2b_integrate_masks.py

import pandas as pd
import numpy as np

def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a tz-aware UTC DatetimeIndex (no name assumptions)."""
    out = df.copy()
    idx = out.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise ValueError("Expected a DatetimeIndex.")
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    out.index = idx
    return out

def integrate_wetdry_and_excess(df_step2: pd.DataFrame,
                                df_nla: pd.DataFrame,
                                df_ex: pd.DataFrame,
                                keep_cols_step2=("Abar","baseline_db","A_excess_db","A_excess_db_per_km"),
                                keep_cols_nla=("is_wet","dA_self_db","dA_nb_med_db","nb_count"),
                                keep_cols_ex=("is_wet_excess","A_ex_pool_per_km","nb_med_A_ex_per_km","nb_count_ex"),
                                onset_allow_if_excess_pos=True) -> pd.DataFrame:
    """
    Combine wet/dry (Δ-based) with excess-based mask on a common (ID, time) index.
    Returns a single dataframe with a final boolean mask `is_wet_final`.

    Arguments
    ---------
    df_step2 : output of compute_dry_baseline_48h (must contain A_excess columns)
    df_nla   : output of step1 wet/dry (Δ-based); must contain `is_wet`
    df_ex    : output of wetdry_from_excess_fast; must contain `is_wet_excess`

    Strategy
    --------
    Primary gate is the EXCESS mask (`is_wet_excess`).
    Optionally allow very early onset when Δ says wet but excess is tiny/just starting
    (`onset_allow_if_excess_pos=True`): `is_wet_final = is_wet_excess | (is_wet & A_ex_pool_per_km>0)`.
    """

    # --- sanity & index harmonization ---
    for name, df in [("df_step2", df_step2), ("df_nla", df_nla), ("df_ex", df_ex)]:
        if "ID" not in df.columns:
            raise ValueError(f"{name} is missing column 'ID'")
    df_step2 = _to_utc_index(df_step2)
    df_nla   = _to_utc_index(df_nla)
    df_ex    = _to_utc_index(df_ex)

    # Materialize a real column for time to make a clean (ID, time) index, avoids IDE warnings
    s2 = df_step2.copy()
    s2["time_utc"] = s2.index
    nla = df_nla.copy()
    nla["time_utc"] = nla.index
    ex  = df_ex.copy()
    ex["time_utc"] = ex.index

    # Minimal, named subsets for merging
    left = s2.set_index(["ID","time_utc"])[list(keep_cols_step2)].sort_index()
    right_delta = nla.set_index(["ID","time_utc"])[list(keep_cols_nla)].rename(
        columns={c: f"{c}_delta" for c in keep_cols_nla}
    )
    right_ex = ex.set_index(["ID","time_utc"])[list(keep_cols_ex)]

    # --- merge ---
    df_s12 = (
        left.join(right_delta, how="left")
            .join(right_ex,    how="left")
            .reset_index()               # bring time_utc back to index
            .set_index("time_utc")
            .sort_index()
    )

    # --- final mask ---
    # Primary signal: excess
    wet_ex = df_s12["is_wet_excess"].fillna(False)

    if onset_allow_if_excess_pos and "A_ex_pool_per_km" in df_s12:
        onset = df_s12.get("is_wet_delta", pd.Series(False, index=df_s12.index)).fillna(False) & \
                (df_s12["A_ex_pool_per_km"].fillna(0) > 0)
        df_s12["is_wet_final"] = (wet_ex | onset)
    else:
        df_s12["is_wet_final"] = wet_ex

    return df_s12
