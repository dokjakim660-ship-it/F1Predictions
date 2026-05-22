"""F1 Predictions — Streamlit app skeleton (Phase 2).

Reads the holdout-test predictions committed by `just final-eval TARGET=...`
and renders a quick methodology + backtest dashboard. Single file with two
tabs for now; the Phase 2 roadmap turns this into a multi-page app with
Next-Race + Feature-Importance pages once the pre-race inference pipeline
and SHAP step land.

Run locally:  `just app`
              `uv run --group app streamlit run app/streamlit_app.py`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parent.parent

st.set_page_config(page_title="F1 Predictions", page_icon="🏎️", layout="wide")
st.title("🏎️ F1 Predictions")
st.caption(
    "Pre-race podium + teammate-H2H probabilities from FastF1 telemetry. "
    "Built from scratch as a learning project — also trying to beat Tipico."
)

methodology_tab, backtest_tab = st.tabs(["📋 Methodology", "📈 Backtest History"])

with methodology_tab:
    st.markdown(
        """
### What the model predicts

Two betting markets per F1 race weekend, **after qualifying**, before the race:

- **Podium** — P(driver finishes P1–P3). Imbalanced base rate ~15 %.
  Best model: **XGBoost (calibrated)**, Brier **0.0616** on the holdout test.
- **Teammate H2H** — P(driver finishes ahead of their constructor team-mate).
  Balanced ~50 %. Best model: **LogisticRegression**, Brier **0.1958**.

### Where the signal comes from

A 31-feature table joined from three L2 sources (one row = one driver × one race):

- **FastF1 telemetry** — quali gap to pole, FP2 long-run pace gap
  (the secret weapon), FP2 short-run pace.
- **Jolpica results** — grid, rolling driver & constructor form (DNF rate,
  points, finishes — all `.shift(1)`-lagged so race N sees only races strictly
  before N).
- **Track attributes** — length, corners, DRS zones, street vs permanent.
- Plus driver × track history, season progress, an era flag for the 2022
  ground-effect reglement bump.

### How it is evaluated

- **Walk-forward** validation, four expanding-window folds across 2018 →
  mid-2024. Optuna sees these.
- **Holdout test** = 2024-07 onward (41 races, ~820 rows per target).
  Sealed; Optuna never touches it.
- Headline metric is **Brier**, with **Isotonic calibration** before
  reporting. Paired bootstrap (1000×, races resampled) for model comparisons.

### What is next

Phase 2 brings this app to HuggingFace Spaces and adds a Next-Race page
once the pre-race inference pipeline is built. Phase 4 brings real
bookmaker odds and the ROI backtest — the actual "beat Tipico?" answer.
        """
    )

with backtest_tab:
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
    st.dataframe(brier_df, hide_index=True, use_container_width=True)

    plot_path = REPO / "models" / f"reliability_mvp_{target_short}.png"
    if plot_path.exists():
        st.subheader("Reliability diagram (holdout test, calibrated)")
        st.image(str(plot_path), use_container_width=True)

    st.subheader("Per-race predictions")
    races = sorted(preds["race_id"].unique(), reverse=True)
    race_id = st.selectbox("Pick a race", races)
    race_df = preds[preds["race_id"] == race_id].copy()
    sort_col = prob_cols_cal[0] if prob_cols_cal else None
    show_cols = ["driver_id", target_col] + prob_cols_cal
    table = race_df[show_cols]
    if sort_col:
        table = table.sort_values(sort_col, ascending=False)
    st.dataframe(table, hide_index=True, use_container_width=True)
