# persist_utils.py
from __future__ import annotations
import json, os
from dataclasses import asdict, is_dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

# ---- dtype optimization ----
def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Floats → float32 (keep DatetimeIndex as-is)
    for c in out.select_dtypes(include=["float64"]).columns:
        out[c] = out[c].astype("float32")
    # Ints → int32 (preserve NaNs by using pandas nullable Int32)
    for c in out.select_dtypes(include=["int64"]).columns:
        if out[c].isna().any():
            out[c] = out[c].astype("Int32")
        else:
            out[c] = out[c].astype("int32")
    # Objecty low-cardinality → category
    for c in out.select_dtypes(include=["object"]).columns:
        # only convert long string columns, not arbitrary blobs
        uniq = out[c].nunique(dropna=True)
        if 0 < uniq <= max(2000, int(0.2 * len(out))):
            out[c] = out[c].astype("category")
    return out

# ---- parquet helpers (pyarrow recommended) ----
def _save_parquet(df: pd.DataFrame, path: str, *, index: bool = True) -> None:
    # Use zstd for compact size + speed; requires pyarrow
    df.to_parquet(path, engine="pyarrow", compression="zstd", index=index)

def _load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")

def _save_json(meta: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

def _config_to_dict(cfg) -> Dict[str, Any]:
    if cfg is None:
        return {}
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, dict):
        return cfg
    # last resort
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_") and not callable(getattr(cfg, k))}

def _daterange_tag(df: pd.DataFrame) -> str:
    idx = df.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx):
        return f"{idx.min().strftime('%Y%m%d_%H%M')}-{idx.max().strftime('%Y%m%d_%H%M')}"
    return "na"

# ---- public API ----
def save_r0_outputs(df_clean: pd.DataFrame, df_summary: pd.DataFrame, cfg, out_dir: str) -> Dict[str,str]:
    os.makedirs(out_dir, exist_ok=True)
    tag = _daterange_tag(df_clean)
    # optimize copies for disk
    dfc = optimize_dtypes(df_clean)
    dfs = optimize_dtypes(df_summary)

    p_clean = os.path.join(out_dir, f"R0_clean_{tag}.parquet")
    p_sum   = os.path.join(out_dir, f"R0_summary_{tag}.parquet")
    p_meta  = os.path.join(out_dir, f"R0_config_{tag}.json")

    _save_parquet(dfc, p_clean, index=True)
    _save_parquet(dfs, p_sum, index=False)
    _save_json(_config_to_dict(cfg), p_meta)
    return {"clean": p_clean, "summary": p_sum, "meta": p_meta}

def save_step1_outputs(df_nla: pd.DataFrame, df_summary: pd.DataFrame, cfg, out_dir: str) -> Dict[str,str]:
    os.makedirs(out_dir, exist_ok=True)
    tag = _daterange_tag(df_nla)
    dfn = optimize_dtypes(df_nla)
    dfs = optimize_dtypes(df_summary)

    p_nla  = os.path.join(out_dir, f"S1_wetdry_{tag}.parquet")
    p_sum  = os.path.join(out_dir, f"S1_summary_{tag}.parquet")
    p_meta = os.path.join(out_dir, f"S1_config_{tag}.json")

    _save_parquet(dfn, p_nla, index=True)
    _save_parquet(dfs, p_sum, index=False)
    _save_json(_config_to_dict(cfg), p_meta)
    return {"wetdry": p_nla, "summary": p_sum, "meta": p_meta}

def load_parquet(path: str) -> pd.DataFrame:
    return _load_parquet(path)

# convenience: resume if already saved
def maybe_load_or_run(save_path: str, run_fn, *args, **kwargs):
    """If save_path exists, load and return; otherwise run_fn(*args, **kwargs)."""
    if os.path.exists(save_path):
        return _load_parquet(save_path)
    return run_fn(*args, **kwargs)
