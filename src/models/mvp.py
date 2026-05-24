"""MVP models on the rich L3 feature table -- podium AND teammate H2H.

Increment B of the Phase 1.4 pipeline, parameterised in Phase 1.5 to drive
both MVP targets through the same walk-forward harness and the same feature
table (data/features/mvp.parquet -- 31 numeric features + track_id):

- target_podium       -- P(driver finishes P1-P3), imbalanced ~15%
- target_beat_teammate -- P(driver finishes ahead of their constructor team-mate),
                          balanced ~50%, NaN for the rare single-car or DNF-tie
                          row (~0.6%) which prepare_dev_test drops upstream.

No Optuna tuning (increment C) and no calibration (increment D) here: models
use library defaults and natural class balance, so predict_proba stays roughly
calibrated and Brier is a fair comparison. scale_pos_weight is deliberately
left off -- it helps ranking but inflates probabilities, which hurts Brier
until the calibration layer lands.

track_id is the one categorical feature: XGBoost and LightGBM consume it
natively (pandas category dtype), while the LogisticRegression reference
one-hot encodes it. NaN rolling features (rookies, no-FP2, first track visit)
go to the tree models as-is; only the LogisticRegression needs imputation.

Run: `python -m src.models.mvp evaluate --target podium`
     `python -m src.models.mvp evaluate --target teammate`
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
    TARGET_TEAMMATE,
    load_features,
)
from src.features.build import FEATURE_COLUMNS as NUMERIC_FEATURES

# Numeric features go to every model; track_id (categorical) is appended for
# the tree models, which handle it natively. LogReg one-hot encodes it instead.
TREE_FEATURES = NUMERIC_FEATURES + CATEGORICAL_COLUMNS

# CLI short names mapped to the actual target column. The short name flows end
# to end -- study names, MLflow experiments, artefact filenames all use it.
TARGETS: dict[str, str] = {
    "podium": TARGET_PODIUM,
    "teammate": TARGET_TEAMMATE,
}
DEFAULT_TARGET = "podium"

# Back-compat alias for callers that still expect a single TARGET constant.
TARGET = TARGET_PODIUM

_RANDOM_STATE = 42

# Default monthly decay rate for time-decay sample weights (Phase 3.6.5).
# 0.97/month -> 24mo old race weighted 0.48, 60mo old race weighted 0.16.
# Treated as opt-in: pass decay_per_month=DEFAULT_DECAY_PER_MONTH explicitly,
# `None` keeps the historical (unweighted) behaviour for clean ablation.
DEFAULT_DECAY_PER_MONTH = 0.97

# BaselineLogistic on the year-split test set (src/models/baseline.py). Rough
# reference for the podium target only -- not the same split as the walk-forward
# folds below, and meaningless for the teammate target.
_PODIUM_BASELINE_TARGET = 0.0620


def compute_time_decay_weights(
    race_dates: pd.Series,
    decay_per_month: float = DEFAULT_DECAY_PER_MONTH,
    ref_date: pd.Timestamp | None = None,
) -> np.ndarray:
    """Per-row exponential decay weight by race age in months.

    weight = decay_per_month ** months_back. Newer races -> ~1.0, older races
    -> exponentially smaller. ref_date defaults to today. Returns 1.0 for any
    row with a missing race_date so the row stays in the training set
    untouched.
    """
    if ref_date is None:
        ref_date = pd.Timestamp.today().normalize()
    days_back = (pd.Timestamp(ref_date) - pd.to_datetime(race_dates)).dt.days.astype(float)
    months_back = days_back / 30.44
    weights = np.power(decay_per_month, months_back.fillna(0.0).to_numpy())
    return weights


def load_model_frame() -> pd.DataFrame:
    """Load the rich feature table with track_id as a categorical column.

    The category dtype is set once here, on the full frame, so every
    walk-forward slice shares the same category set -- per-fold one-hot
    encoding (LogReg) and native categorical handling (XGB/LGBM) stay aligned.
    """
    df = load_features()
    df["track_id"] = df["track_id"].astype("category")
    return df


def prepare_dev_test(target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the rich frame, drop NaN-target rows, return (dev, test).

    target_beat_teammate is NaN for ~0.6 % of rows (single-car constructors,
    same-lap DNF ties) -- no defined outcome, would corrupt training and
    Brier. target_podium is never NaN, so this is a no-op there.
    """
    df = load_model_frame()
    df = df[df[target_col].notna()].copy()
    return split_dev_test(df)


def _logreg_matrix(df: pd.DataFrame, medians: pd.Series) -> np.ndarray:
    """Median-imputed numeric features + one-hot track_id as a float ndarray.

    track_id is a category dtype, so get_dummies emits identical columns in
    identical order for any slice -- train and val matrices stay aligned.
    """
    numeric = df[NUMERIC_FEATURES].fillna(medians)
    dummies = pd.get_dummies(df["track_id"], prefix="trk").astype(float)
    return pd.concat([numeric, dummies], axis=1).to_numpy()


def make_constant_fit_predict(target_col: str = TARGET_PODIUM) -> FitPredictFn:
    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        return np.full(len(val), float(train[target_col].mean()))

    return fit_predict


def make_top3_quali_fit_predict(target_col: str = TARGET_PODIUM) -> FitPredictFn:
    """The "Top-3-Quali = Podium" F1-domain baseline (PLANNING.md §6, §10).

    Probabilistic version: predict P(target | is_top3_grid), estimated as the
    empirical rate on the train slice. Brier-comparable -- a hard deterministic
    1/0 prediction would blow up Brier on every misclassified row.
    """

    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        top3_mask = train["is_top3_grid"] == 1
        top3_rate = (
            float(train.loc[top3_mask, target_col].mean())
            if top3_mask.any()
            else float(train[target_col].mean())
        )
        rest_rate = (
            float(train.loc[~top3_mask, target_col].mean())
            if (~top3_mask).any()
            else float(train[target_col].mean())
        )
        return np.where(val["is_top3_grid"] == 1, top3_rate, rest_rate)

    return fit_predict


def make_logreg_fit_predict(
    target_col: str = TARGET_PODIUM,
    *,
    decay_per_month: float | None = None,
) -> FitPredictFn:
    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        medians = train[NUMERIC_FEATURES].median(numeric_only=True)
        x_train = _logreg_matrix(train, medians)
        x_val = _logreg_matrix(val, medians)
        scaler = StandardScaler().fit(x_train)
        weights = (
            compute_time_decay_weights(train["race_date"], decay_per_month)
            if decay_per_month is not None
            else None
        )
        model = LogisticRegression(max_iter=1000).fit(
            scaler.transform(x_train),
            train[target_col].astype(int),
            sample_weight=weights,
        )
        return model.predict_proba(scaler.transform(x_val))[:, 1]

    return fit_predict


_XGB_DEFAULT_PARAMS: dict = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
)


def make_default_xgb_fit_predict(
    target_col: str = TARGET_PODIUM,
    *,
    decay_per_month: float | None = None,
) -> FitPredictFn:
    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        model = xgb.XGBClassifier(
            **_XGB_DEFAULT_PARAMS,
            eval_metric="logloss",
            enable_categorical=True,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
        )
        weights = (
            compute_time_decay_weights(train["race_date"], decay_per_month)
            if decay_per_month is not None
            else None
        )
        model.fit(train[TREE_FEATURES], train[target_col].astype(int), sample_weight=weights)
        return model.predict_proba(val[TREE_FEATURES])[:, 1]

    return fit_predict


_LGBM_DEFAULT_PARAMS: dict = dict(
    n_estimators=300,
    max_depth=4,
    num_leaves=15,
    learning_rate=0.05,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
)


def make_default_lgbm_fit_predict(
    target_col: str = TARGET_PODIUM,
    *,
    decay_per_month: float | None = None,
) -> FitPredictFn:
    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        model = lgb.LGBMClassifier(
            **_LGBM_DEFAULT_PARAMS,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        weights = (
            compute_time_decay_weights(train["race_date"], decay_per_month)
            if decay_per_month is not None
            else None
        )
        # LightGBM auto-detects the pandas category column as a categorical feature.
        model.fit(train[TREE_FEATURES], train[target_col].astype(int), sample_weight=weights)
        return model.predict_proba(val[TREE_FEATURES])[:, 1]

    return fit_predict


# Back-compat default-target bindings: final_eval imports these by name.
constant_fit_predict: FitPredictFn = make_constant_fit_predict(TARGET_PODIUM)
logreg_fit_predict: FitPredictFn = make_logreg_fit_predict(TARGET_PODIUM)
xgb_fit_predict: FitPredictFn = make_default_xgb_fit_predict(TARGET_PODIUM)
lgbm_fit_predict: FitPredictFn = make_default_lgbm_fit_predict(TARGET_PODIUM)


def models_for(target_col: str) -> dict[str, FitPredictFn]:
    return {
        "ConstantRate": make_constant_fit_predict(target_col),
        "LogisticRegression": make_logreg_fit_predict(target_col),
        "XGBoost": make_default_xgb_fit_predict(target_col),
        "LightGBM": make_default_lgbm_fit_predict(target_col),
    }


def evaluate_all(
    dev: pd.DataFrame, target_col: str = TARGET_PODIUM
) -> list[tuple[str, WalkForwardResult]]:
    return [
        (name, evaluate(dev, fn, target_col=target_col))
        for name, fn in models_for(target_col).items()
    ]


def _print_results(
    dev: pd.DataFrame,
    results: list[tuple[str, WalkForwardResult]],
    target_short: str,
) -> None:
    print()
    print("=" * 95)
    print(
        f"MVP models ({target_short}) -- walk-forward Brier  |  "
        f"dev: {len(dev)} rows, 4 folds 2018..2024-06  |  {len(TREE_FEATURES)} features"
    )
    print("=" * 95)
    for name, res in results:
        per_fold = " ".join(f"{b:.4f}" for b in res.fold_briers)
        print(
            f"{name:<20s}  folds[{per_fold}]  "
            f"mean={res.mean_brier:.4f}  std={res.std_brier:.4f}  obj={res.objective:.4f}"
        )
    print("=" * 95)
    if target_short == "podium":
        print(f"Reference: BaselineLogistic (year-split test) brier {_PODIUM_BASELINE_TARGET:.4f}")
    print("Lower brier/obj = better. obj = mean + 0.2*std across folds.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate", help="Walk-forward Brier for all MVP models vs baselines.")
    ev.add_argument(
        "--target",
        choices=tuple(TARGETS),
        default=DEFAULT_TARGET,
        help=f"Which target to model (default: {DEFAULT_TARGET}).",
    )
    args = p.parse_args(argv)

    if args.cmd == "evaluate":
        target_col = TARGETS[args.target]
        dev, _ = prepare_dev_test(target_col)
        results = evaluate_all(dev, target_col=target_col)
        _print_results(dev, results, args.target)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
