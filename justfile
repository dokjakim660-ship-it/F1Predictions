# F1 Predictions — Pipeline Runner
# Run `just` (no args) to list all available targets.

set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

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
    uv run python -m src.ingest.fastf1_ingest

# Verify Jolpica-F1 API reachable
smoke-jolpica:
    uv run python -m src.ingest.jolpica_ingest

# Verify Open-Meteo API reachable
smoke-meteo:
    uv run python -m src.ingest.openmeteo_ingest

# All three smoke tests in sequence
smoke: smoke-fastf1 smoke-jolpica smoke-meteo

# --- Phase 1 placeholders (filled later) ---

fetch RACE:
    @echo "Not implemented yet (Phase 1): fetch {{RACE}}"

build:
    @echo "Not implemented yet (Phase 1): build feature table"

train:
    @echo "Not implemented yet (Phase 1): train models"

predict RACE:
    @echo "Not implemented yet (Phase 1): predict {{RACE}}"
