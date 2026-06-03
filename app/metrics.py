"""Shared holdout-test metrics — one source of truth for 'which model is best'.

Both the Next Race model-comparison card and the Methodology page read these,
so the app can never contradict itself about the winning model or its Brier.
Reads the sealed holdout predictions (predictions/mvp_test_{target}.parquet),
the same file the Backtest page reports — so all three views always agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]

_TARGET_COL = {"podium": "target_podium", "teammate": "target_beat_teammate"}


@st.cache_data(show_spinner=False)
def _load(target_short: str) -> pd.DataFrame:
    path = REPO / "predictions" / f"mvp_test_{target_short}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def holdout_brier(target_short: str) -> dict[str, float]:
    """Per-model calibrated Brier on the holdout test (lower = better). {} if absent."""
    h = _load(target_short)
    y_col = _TARGET_COL.get(target_short)
    if h.empty or y_col is None or y_col not in h.columns:
        return {}
    y = h[y_col].astype(int).to_numpy()
    out: dict[str, float] = {}
    for col in h.columns:
        if col.startswith("prob_") and col.endswith("_cal"):
            name = col.removeprefix("prob_").removesuffix("_cal")
            if name == "constantrate":  # base rate — not a real contender
                continue
            out[name] = float(np.mean((h[col].to_numpy() - y) ** 2))
    return out


def best_model(target_short: str) -> tuple[str, float] | None:
    """(raw_model_name, brier) of the best holdout model, or None if unavailable."""
    briers = holdout_brier(target_short)
    if not briers:
        return None
    name = min(briers, key=briers.get)
    return name, briers[name]


def holdout_shape(target_short: str) -> tuple[int, int]:
    """(row_count, race_count) of the holdout test for this target. (0, 0) if absent."""
    h = _load(target_short)
    if h.empty:
        return 0, 0
    races = int(h["race_id"].nunique()) if "race_id" in h.columns else 0
    return len(h), races
