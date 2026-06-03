"""Methodology page — what the model does, how it's trained, what's next."""

from __future__ import annotations

import streamlit as st

import ui

ui.page_header(
    "Methodology",
    eyebrow="About · How it works",
    desc="What the models predict, where the signal comes from, and how it is "
    "evaluated.",
)

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

Phase 3 added the **Next Race** page (default tab) so each Saturday-after-quali
the calibrated podium + teammate-H2H probabilities are one `just predict-next`
away. Phase 4 brings real bookmaker odds and the ROI backtest — the actual
"beat Tipico?" answer.
    """
)
