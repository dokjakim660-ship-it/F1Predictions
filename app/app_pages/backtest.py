"""Backtest History page — calibrated Brier per model + per-race drilldown on the holdout test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[2]

target_short = st.radio("Target", ["podium", "teammate"], horizontal=True, key="target_radio")
target_col = "target_podium" if target_short == "podium" else "target_beat_teammate"

preds_path = REPO / "predictions" / f"mvp_test_{target_short}.parquet"
if not preds_path.exists():
    st.error(
        f"Predictions file missing: `{preds_path.relative_to(REPO)}`. "
        f"Run `just final-eval TARGET={target_short}` first."
    )
    st.stop()

preds = pd.read_parquet(preds_path)

c1, c2, c3 = st.columns(3)
c1.metric("Holdout rows", f"{len(preds):,}")
c2.metric("Holdout races", int(preds["race_id"].nunique()))
c3.metric("Base rate", f"{preds[target_col].mean():.1%}")

st.subheader("Calibrated Brier per model")
prob_cols_cal = sorted(c for c in preds.columns if c.startswith("prob_") and c.endswith("_cal"))
y_true = preds[target_col].astype(int).to_numpy()
brier_rows = []
for col in prob_cols_cal:
    name = col.removeprefix("prob_").removesuffix("_cal")
    y_prob = preds[col].to_numpy()
    brier_rows.append(
        {
            "model": name,
            "brier_cal": float(np.mean((y_prob - y_true) ** 2)),
        }
    )
brier_df = pd.DataFrame(brier_rows).sort_values("brier_cal").reset_index(drop=True)
st.dataframe(brier_df, hide_index=True, width="stretch")

plot_path = REPO / "models" / f"reliability_mvp_{target_short}.png"
if plot_path.exists():
    st.subheader("Reliability diagram (holdout test, calibrated)")
    st.image(str(plot_path), width="stretch")

st.subheader("Per-race predictions")
races = sorted(preds["race_id"].unique(), reverse=True)
race_id = st.selectbox("Pick a race", races)
race_df = preds[preds["race_id"] == race_id].copy()
sort_col = prob_cols_cal[0] if prob_cols_cal else None
show_cols = ["driver_id", target_col] + prob_cols_cal
table = race_df[show_cols]
if sort_col:
    table = table.sort_values(sort_col, ascending=False)
st.dataframe(table, hide_index=True, width="stretch")
