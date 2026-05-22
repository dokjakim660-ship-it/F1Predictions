"""MVP podium models: XGBoost + LightGBM on the rich L3 feature table.

Increment B of the Phase 1.4 pipeline, re-pointed in Phase 1.2 onto the full
feature table (data/features/mvp.parquet -- 31 numeric features + track_id)
instead of the nine thin baseline columns. This is the first honest "rich
features vs. baseline" Brier comparison through the walk-forward harness.

No Optuna tuning (increment C) and no calibration (increment D) yet: models
use library defaults and natural class balance, so predict_proba stays roughly
calibrated and Brier is a fair comparison. scale_pos_weight is deliberately
left off here -- it improves ranking but inflates probabilities, which hurts
Brier until the calibration layer lands.

track_id is the one categorical feature: XGBoost and LightGBM consume it
natively (pandas category dtype), while the LogisticRegression reference
one-hot encodes it. NaN rolling features (rookies, no-FP2, first track visit)
go to the tree models as-is; only the LogisticRegression needs imputation.

Run: `python -m src.models.mvp evaluate`
"""

from __future__ import annotations

import argparse
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.eval.walk_forward import FitPredictFn, WalkForwardResult, evaluate, split_dev_test
from src.features.build import (
    CATEGORICAL_COLUMNS,
    TARGET_PODIUM,
    load_features,
)
from src.features.build import FEATURE_COLUMNS as NUMERIC_FEATURES

# Numeric features go to every model; track_id (categorical) is appended for
# the tree models, which handle it natively. LogReg one-hot encodes it instead.
TREE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_COLUMNS
TARGET = TARGET_PODIUM

_RANDOM_STATE = 42

# BaselineLogistic on the year-split test set (src/models/baseline.py). Rough
# reference only -- not the same split as the walk-forward folds below.
_BASELINE_TARGET = 0.0620


def load_model_frame() -> pd.DataFrame:
    """Load the rich feature table with track_id as a categorical column.

    The category dtype is set once here, on the full frame, so every
    walk-forward slice shares the same category set -- per-fold one-hot
    encoding (LogReg) and native categorical handling (XGB/LGBM) stay aligned.
    """
    df = load_features()
    df["track_id"] = df["track_id"].astype("category")
    return df


def constant_fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    return np.full(len(val), float(train[TARGET].mean()))


def _logreg_matrix(df: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    """Median-imputed numeric features + one-hot track_id as a float ndarray.

    track_id is a category dtype, so get_dummies emits identical columns in
    identical order for any slice -- train and val matrices stay aligned.
    """
    numeric = df[NUMERIC_FEATURES].fillna(medians)
    dummies = pd.get_dummies(df["track_id"], prefix="trk").astype(float)
    return pd.concat([numeric, dummies], axis=1).to_numpy()


def logreg_fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    medians = train[NUMERIC_FEATURES].median(numeric_only=True)
    x_train = _logreg_matrix(train, medians)
    x_val = _logreg_matrix(val, medians)
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=1000).fit(
        scaler.transform(x_train), train[TARGET].astype(int)
    )
    return model.predict_proba(scaler.transform(x_val))[:, 1]


def xgb_fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        enable_categorical=True,
        random_state=_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train[TREE_FEATURES], train[TARGET].astype(int))
    return model.predict_proba(val[TREE_FEATURES])[:, 1]


def lgbm_fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        num_leaves=15,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=_RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    # LightGBM auto-detects the pandas category column as a categorical feature.
    model.fit(train[TREE_FEATURES], train[TARGET].astype(int))
    return model.predict_proba(val[TREE_FEATURES])[:, 1]


_MODELS: dict[str, FitPredictFn] = {
    "ConstantRate": constant_fit_predict,
    "LogisticRegression": logreg_fit_predict,
    "XGBoost": xgb_fit_predict,
    "LightGBM": lgbm_fit_predict,
}


def evaluate_all(dev: pd.DataFrame) -> list[tuple[str, WalkForwardResult]]:
    return [(name, evaluate(dev, fn)) for name, fn in _MODELS.items()]


def _print_results(dev: pd.DataFrame, results: list[tuple[str, WalkForwardResult]]) -> None:
    print()
    print("=" * 92)
    print(
        f"MVP models -- walk-forward Brier  |  dev set: {len(dev)} rows, "
        f"4 folds 2018..2024-06  |  {len(TREE_FEATURES)} features"
    )
    print("=" * 92)
    for name, res in results:
        per_fold = " ".join(f"{b:.4f}" for b in res.fold_briers)
        print(
            f"{name:<20s}  folds[{per_fold}]  "
            f"mean={res.mean_brier:.4f}  std={res.std_brier:.4f}  obj={res.objective:.4f}"
        )
    print("=" * 92)
    print(f"Reference: BaselineLogistic (year-split test) brier {_BASELINE_TARGET:.4f}")
    print("Lower brier/obj = better. obj = mean + 0.2*std across folds.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate", help="Walk-forward Brier for all MVP models vs baselines.")
    args = p.parse_args(argv)

    if args.cmd == "evaluate":
        dev, _ = split_dev_test(load_model_frame())
        results = evaluate_all(dev)
        _print_results(dev, results)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
