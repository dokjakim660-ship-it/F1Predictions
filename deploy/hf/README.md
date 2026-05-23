---
title: F1 Predictions
emoji: 🏎️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Pre-race podium + teammate-H2H probabilities from FastF1.
---

# F1 Predictions

Pre-race podium + teammate-H2H probabilities for Formula 1, built from scratch
as a learning project — also trying to beat Tipico long-run.

The Streamlit app is **inference-only**: it reads parquets and reliability
diagrams that were trained, calibrated, and evaluated locally and committed
into this Space. No FastF1 / xgboost / shap dependencies at runtime.

## What you get on each page

- **Methodology** — what the two targets are, how features are built and
  evaluated, what comes next.
- **Backtest History** — per-model calibrated Brier with 95 % bootstrap CIs,
  the reliability diagram, per-race Brier trend across the holdout window,
  and a Grand-Prix drilldown with per-race Brier callouts.
- **Feature Importance** — top features per target (mean(|SHAP|) for the
  XGBoost podium model, |standardised coefficient| for the LogReg teammate
  model), bar-charted with direction colouring.

## Source

The full training + evaluation pipeline (FastF1 ingest, walk-forward Optuna,
isotonic calibration, paired bootstrap CI tests) lives in the GitHub repo:

<https://github.com/dokjakim660-ship-it/F1Predictions>

This Space holds only the runtime artefacts and the Streamlit code that
renders them.
