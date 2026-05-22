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

# --- Phase 1.2 Feature table (L2 processed -> L3 model-ready) ---

# Build the rich MVP feature table -> data/features/mvp.parquet
build:
    uv run python -m src.features.build build

# Print summary of the current MVP feature table
show-features:
    uv run python -m src.features.build show

# --- Phase 1 placeholders (filled later) ---

train:
    @echo "Not implemented yet (Phase 1): train MVP models"

predict RACE:
    @echo "Not implemented yet (Phase 1): predict {{RACE}}"
