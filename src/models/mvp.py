"""MVP podium models: XGBoost + LightGBM on the baseline feature set.

Increment B of the Phase 1.4 pipeline skeleton. Trains both gradient-boosted
tree models on the same nine features the baseline LogisticRegression uses and
scores them through the walk-forward harness -- a first honest "trees vs.
linear" Brier number.

No Optuna tuning (increment C) and no calibration (increment D) yet: models
use library defaults and natural class balance, so predict_proba stays roughly
calibrated and Brier is a fair comparison. scale_pos_weight is deliberately
left off here -- it improves ranking but inflates probabilities, which hurts
Brier until the calibration layer lands.

Tree models receive NaN rolling features as-is (both XGBoost and LightGBM learn
a default split direction for missing values); only the LogisticRegression
reference needs median imputation + scaling.

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
from src.features.baseline import load_features

# The nine baseline features. Tree models also justify their existence against
# exactly these -- the rich FastF1 feature table is a later increment.
FEATURE_COLUMNS = [
    "grid_effective",
    "grid_log",
    "is_pole",
    "is_top3_grid",
    "con_rolling_pts_l5",
    "driver_rolling_avg_pos_l5",
    "driver_rolling_dnf_rate_l5",
    "driver_career_race_count",
    "driver_age_years",
]
TARGET = "target_podium"

_RANDOM_STATE = 42

# BaselineLogistic on the year-split test set (src/models/baseline.py). Rough
# reference only -- not the same split as the walk-forward folds below.
_BASELINE_TARGET = 0.0620


def constant_fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    return np.full(len(val), float(train[TARGET].mean()))


def logreg_fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    medians = train[FEATURE_COLUMNS].median(numeric_only=True)
    x_train = train[FEATURE_COLUMNS].fillna(medians)
    x_val = val[FEATURE_COLUMNS].fillna(medians)
    scaler = StandardScaler().fit(x_train.values)
    model = LogisticRegression(max_iter=1000).fit(
        scaler.transform(x_train.values), train[TARGET].astype(int)
    )
    return model.predict_proba(scaler.transform(x_val.values))[:, 1]


def xgb_fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(train[FEATURE_COLUMNS], train[TARGET].astype(int))
    return model.predict_proba(val[FEATURE_COLUMNS])[:, 1]


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
    model.fit(train[FEATURE_COLUMNS], train[TARGET].astype(int))
    return model.predict_proba(val[FEATURE_COLUMNS])[:, 1]


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
    print(f"MVP models -- walk-forward Brier  |  dev set: {len(dev)} rows, 4 folds 2018..2024-06")
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
        df = load_features()
        dev, _ = split_dev_test(df)
        results = evaluate_all(dev)
        _print_results(dev, results)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
