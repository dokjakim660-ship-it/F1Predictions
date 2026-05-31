# F1 Predictions - Pipeline Runner
# Run `just` (no args) to list all available targets.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# Force UTF-8 stdout so non-ASCII race names (Sao Paulo etc.) don't break prints.
export PYTHONIOENCODING := "utf-8"

default:
    @just --list

# --- Setup ---

# Install/sync all dependencies (incl. dev group) via uv
setup:
    uv sync --all-groups

# Quick env info
info:
    uv run python --version
    uv run python -c "import fastf1, pandas, xgboost, lightgbm, sklearn, optuna; print('all imports OK')"

# --- Quality ---

# Lint + format check (no writes)
lint:
    uv run ruff check src tests
    uv run ruff format --check src tests

# Auto-fix lint + format
fix:
    uv run ruff check --fix src tests
    uv run ruff format src tests

# Run unit tests
test:
    uv run pytest

# --- Phase 0 Smoke Tests ---

# Verify FastF1 can fetch + cache one race
smoke-fastf1:
    uv run python -m src.ingest.fastf1_ingest smoke

# Verify Jolpica-F1 API reachable
smoke-jolpica:
    uv run python -m src.ingest.jolpica_ingest smoke

# Verify Open-Meteo API reachable
smoke-meteo:
    uv run python -m src.ingest.openmeteo_ingest smoke

# All three smoke tests in sequence
smoke: smoke-fastf1 smoke-jolpica smoke-meteo

# --- Phase 1 Ingest ---

# Refresh per-season schedules (Jolpica) + rebuild race inventory
ingest-schedule:
    uv run python -m src.ingest.jolpica_ingest schedule --years 2018 2019 2020 2021 2022 2023 2024 2025 2026
    uv run python -m src.utils.race_inventory build

# Print summary of the cached race inventory
inventory-show:
    uv run python -m src.utils.race_inventory show

# Pull all three sources for one race weekend
ingest-race YEAR ROUND:
    uv run python -m src.ingest.fastf1_ingest race --year {{YEAR}} --round {{ROUND}}
    uv run python -m src.ingest.jolpica_ingest results --year {{YEAR}} --round {{ROUND}}
    uv run python -m src.ingest.openmeteo_ingest race --year {{YEAR}} --round {{ROUND}}

# Pull all three sources for every completed race in a season
ingest-season YEAR:
    uv run python -m src.ingest.fastf1_ingest season --year {{YEAR}}
    uv run python -m src.ingest.jolpica_ingest results --year {{YEAR}}
    uv run python -m src.ingest.openmeteo_ingest season --year {{YEAR}}

# Phase 3.1 next-race ingest: refresh schedule + pull Q/FP2 + weather forecast for ONE future race.
# Runs Sa-Abend after qualifying. No R session (race hasn't happened); forecast lands in
# `{year}_{round:02d}.forecast.json` so it doesn't clobber the historical archive later.
ingest-next YEAR ROUND:
    uv run python -m src.ingest.jolpica_ingest schedule --years {{YEAR}} --refresh
    uv run python -m src.utils.race_inventory build
    uv run python -m src.ingest.fastf1_ingest next --year {{YEAR}} --round {{ROUND}}
    uv run python -m src.ingest.openmeteo_ingest next --year {{YEAR}} --round {{ROUND}}

# Phase 3.2 next-race feature build. Rebuilds the FastF1 sessions + weather L2
# first so freshly ingested Q/FP2 and the new forecast land in their parquets,
# then synthesizes a pseudo-row per driver for the target race and runs the full
# feature pipeline on top. Writes data/features/next_race.parquet.
# Run AFTER `just ingest-next YEAR ROUND`.
build-next-features YEAR ROUND:
    uv run python -m src.process.fastf1 build
    uv run python -m src.process.openmeteo build
    uv run python -m src.features.next_race build --year {{YEAR}} --round {{ROUND}}

# Phase 3.3 next-race inference: re-trains XGB+LGBM+LogReg on the full historical
# feature table, fits isotonic calibrators on dev OOF predictions, predicts the
# next race for both podium and teammate-H2H targets. Writes
# `predictions/next_race_{podium,teammate}.parquet`. Run AFTER build-next-features.
predict-next YEAR ROUND:
    uv run python -m src.models.predict_next run --year {{YEAR}} --round {{ROUND}} --target podium
    uv run python -m src.models.predict_next run --year {{YEAR}} --round {{ROUND}} --target teammate

# --- Phase 4.2 Pre-quali (predict BEFORE qualifying) ---

# Phase 4.2.4b pre-quali feature build. Rebuilds the FastF1 sessions + weather L2
# (so freshly ingested FP2 + forecast land), then synthesizes a driver row WITHOUT
# qualifying for the target race. Writes data/features/next_race_prequali.parquet.
# Run AFTER `just ingest-next YEAR ROUND` (typically Friday night, post-FP2).
build-prequali-features YEAR ROUND:
    uv run python -m src.process.fastf1 build
    uv run python -m src.process.openmeteo build
    uv run python -m src.features.next_race_prequali build --year {{YEAR}} --round {{ROUND}}

# Phase 4.2.4c pre-quali inference: trains the pre-quali stack on the full history
# and predicts all four quali targets (pole, top3, top10, teammate) in both timing
# modes (pre_weekend / post_fp2). Writes predictions/next_race_prequali_{target}.parquet.
# Run AFTER build-prequali-features.
predict-pre-quali YEAR ROUND:
    uv run python -m src.models.predict_prequali run --year {{YEAR}} --round {{ROUND}} --target all

# Phase 4.2.8 composition A/B: holdout backtest of a pre-race podium model that
# replaces the real quali/grid features with out-of-fold predicted-quali probs,
# vs the real-grid podium model (+ ConstantRate / Top3Quali anchors). Eval-only;
# measures how much podium signal survives without real qualifying.
compose-prequali:
    uv run python -m src.models.compose_prequali eval

# --- Phase 5 ranking (predict the full grid order: quali + race) ---

# Holdout A/B per task: position-regression vs LambdaMART ranker vs the trivial
# baseline (grid order for race, recent quali form for quali). Scored on rank
# metrics (position MAE, Spearman, top-k), not Brier. Eval-only, no betting.
eval-rank TASK="both":
    uv run python -m src.models.rank eval --task {{TASK}}

# Dev-set model selection (walk-forward position MAE; holdout never touched).
select-rank TASK="both":
    uv run python -m src.models.rank select --task {{TASK}}

# Next-race ranking inference: predicted qualifying order + predicted race order
# for one upcoming race. Needs next_race_prequali.parquet (build-prequali-features)
# for quali and next_race.parquet (build-next-features) for race. Writes
# predictions/next_race_rank_{quali,race}.parquet.
predict-rank YEAR ROUND:
    uv run python -m src.models.predict_rank run --year {{YEAR}} --round {{ROUND}} --task all

# --- Phase 5.2 DNF sub-model (P(driver does not finish)) ---

# Holdout eval per timing mode (pre_weekend / post_fp2 / race): LogReg/XGB/LGBM/
# Ensemble vs ConstantRate + TeamReliability baselines. Walk-forward OOF ->
# isotonic calibration -> Brier on the sealed holdout, with the Go/No-Go verdict.
eval-dnf MODE="all":
    uv run python -m src.models.dnf eval --mode {{MODE}}

# Dev-set model selection (walk-forward Brier; holdout never touched).
select-dnf MODE="all":
    uv run python -m src.models.dnf select --mode {{MODE}}

# Next-race DNF inference: trains the DNF stack on the full history, calibrates on
# dev OOF, predicts P(DNF) for one upcoming race in every timing mode. race mode
# reads next_race.parquet; pre_weekend/post_fp2 read next_race_prequali.parquet.
# Writes predictions/next_race_dnf.parquet.
predict-dnf YEAR ROUND:
    uv run python -m src.models.predict_dnf run --year {{YEAR}} --round {{ROUND}}

# Phase 5.2 C1 ablation: does adding a leak-free P(DNF) feature improve the podium
# model on the holdout? Paired-bootstrap CI vs the plain podium model. Eval-only.
compose-dnf:
    uv run python -m src.models.compose_dnf eval

# Full backfill across all sources. Lightweight sources first (Open-Meteo, Jolpica)
# so they finish even if FastF1 rate-limits us (500 calls/h - see prune-rate-limit).
ingest-all:
    uv run python -m src.ingest.openmeteo_ingest all
    uv run python -m src.ingest.jolpica_ingest results --all
    uv run python -m src.ingest.fastf1_ingest all

# --- Phase 1.1 Process (RAW -> L2 processed parquet) ---

# Build all three L2 processed parquets (results, weather, sessions)
build-l2:
    uv run python -m src.process.jolpica build
    uv run python -m src.process.openmeteo build
    uv run python -m src.process.fastf1 build

# Print summaries of all three L2 parquets
show-l2:
    uv run python -m src.process.jolpica show
    uv run python -m src.process.openmeteo show
    uv run python -m src.process.fastf1 show

# --- Phase 1.3 Baseline model (Brier latte vor MVP) ---

# Build baseline feature table from L2 results.parquet
build-baseline-features:
    uv run python -m src.features.baseline build

# Train + evaluate baseline models, save predictions + pickled LogReg
train-baseline: build-baseline-features
    uv run python -m src.models.baseline train

# --- Phase 1.4 MVP model (TARGET = podium | teammate) ---

# Walk-forward Brier: XGBoost + LightGBM vs baselines (no tuning, no calibration)
eval-mvp TARGET="podium":
    uv run python -m src.models.mvp evaluate --target {{TARGET}}

# Optuna tuning for XGBoost + LightGBM on the walk-forward objective
tune-mvp TRIALS="50" TARGET="podium":
    uv run python -m src.models.tune xgboost --trials {{TRIALS}} --target {{TARGET}}
    uv run python -m src.models.tune lightgbm --trials {{TRIALS}} --target {{TARGET}}
    uv run python -m src.models.tune show --target {{TARGET}}

# Final eval on the holdout test set: calibration + MLflow + CHANGELOG (needs tune-mvp first)
final-eval TARGET="podium":
    uv run python -m src.models.final_eval run --target {{TARGET}}

# Feature importance for the best model (XGB+SHAP for podium, LogReg coefs for teammate).
# Writes predictions/importance_{TARGET}.parquet, consumed by the Streamlit app.
importance TARGET="podium":
    uv run python -m src.eval.importance run --target {{TARGET}}

# --- Phase 1.2 Feature table (L2 processed -> L3 model-ready) ---

# Build the rich MVP feature table -> data/features/mvp.parquet
build:
    uv run python -m src.features.build build

# Print summary of the current MVP feature table
show-features:
    uv run python -m src.features.build show

# --- Phase 4 Post-race ROI ---

# Sunday-evening workflow: ingest race results + evaluate ROI against saved odds.
# Requires that odds were saved via the Stakes page before the race.
post-race YEAR ROUND:
    uv run python -m src.ingest.jolpica_ingest results --year {{YEAR}} --round {{ROUND}}
    uv run python -m src.eval.roi run --year {{YEAR}} --round {{ROUND}}

# Phase 4.2.7 quali-market ROI: evaluate the four pre-quali bets (pole/top3/top10/
# teammate-Q) against odds saved on the Pre-Quali Stakes page. Rebuilds the L2 +
# feature table first so the realised quali targets (from mvp.parquet) are current,
# then settles the bets. Run after the weekend (qualifying must be ingested+built).
post-quali YEAR ROUND: build-l2 build
    uv run python -m src.eval.roi run-quali --year {{YEAR}} --round {{ROUND}}

# Print ROI log summary
roi-show:
    uv run python -m src.eval.roi show

# --- Phase 2 Streamlit app ---

# Run the Streamlit app locally (browser opens automatically).
app:
    uv run --group app streamlit run app/streamlit_app.py

# --- Phase 2.4 HuggingFace Spaces deploy ---

# Copy the deployable subset into a local HF Space clone (then commit+push there).
# HF_PATH must point at an existing git clone of the HF Space repo.
deploy-hf HF_PATH:
    uv run python scripts/sync_to_hf.py {{HF_PATH}}

# --- Phase 1 placeholders (filled later) ---

train:
    @echo "Not implemented yet (Phase 1): train MVP models"

predict RACE:
    @echo "Not implemented yet (Phase 1): predict {{RACE}}"
