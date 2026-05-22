from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
REFERENCE_DIR = DATA_DIR / "reference"

FASTF1_CACHE = RAW_DIR / "fastf1_cache"

MODELS_DIR = REPO_ROOT / "models"
PREDICTIONS_DIR = REPO_ROOT / "predictions"

# Optuna study store (gitignored via *.db) and MLflow tracking dir.
OPTUNA_DB = MODELS_DIR / "optuna.db"
MLRUNS_DIR = REPO_ROOT / "mlruns"
