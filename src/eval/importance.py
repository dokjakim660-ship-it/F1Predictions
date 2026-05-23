"""Feature importance for the Phase-1 best models (podium + teammate H2H).

Trains the same final model used by final_eval (same dev/test split, same
hyperparameters loaded from the persisted Optuna study) and computes one
attribution number per feature on the holdout test set:

- Podium   -> XGBoost.            mean(|SHAP|) per feature, plus mean(SHAP)
                                  signed for direction.
- Teammate -> LogisticRegression. |standardised coefficient| per feature,
                                  signed coefficient for direction.
                                  track_id one-hots collapse to a single
                                  row via L2 norm; direction is NaN there
                                  because levels point in different ways.

Output is one parquet per target -- columns `feature`, `importance` (>=0,
ranked descending), `direction` (signed; NaN for aggregated one-hot bundles).
The Streamlit feature-importance page reads it directly. shap / numba are
never imported on HF Spaces; this module runs locally as part of the
`just importance` pipeline, the parquet ships with the repo.

Run: `python -m src.eval.importance run --target podium`
     `python -m src.eval.importance run --target teammate`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.features.build import FEATURE_COLUMNS as NUMERIC_FEATURES
from src.models.mvp import (
    DEFAULT_TARGET,
    TARGETS,
    TREE_FEATURES,
    _logreg_matrix,
    prepare_dev_test,
)
from src.models.tune import best_params
from src.utils.paths import PREDICTIONS_DIR

_RANDOM_STATE = 42


def _importance_path(target_short: str) -> Path:
    return PREDICTIONS_DIR / f"importance_{target_short}.parquet"


def _xgb_importance(dev: pd.DataFrame, test: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Fit the tuned XGB on dev, return mean(|SHAP|) + mean(SHAP) per feature on test."""
    params = best_params("xgboost", "podium")
    model = xgb.XGBClassifier(
        **params,
        eval_metric="logloss",
        enable_categorical=True,
        random_state=_RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(dev[TREE_FEATURES], dev[target_col].astype(int))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test[TREE_FEATURES])

    return pd.DataFrame(
        {
            "feature": list(TREE_FEATURES),
            "importance": np.abs(shap_values).mean(axis=0),
            "direction": shap_values.mean(axis=0),
        }
    )


def _logreg_importance(dev: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Fit the LogReg pipeline on dev, return |coef| + signed coef per feature.

    track_id one-hot levels collapse to a single 'track_id' row via L2 norm,
    matching how the Streamlit page wants to render it. Their individual
    directions disagree, so direction is set to NaN for that aggregated row.
    """
    medians = dev[NUMERIC_FEATURES].median(numeric_only=True)
    x_dev = _logreg_matrix(dev, medians)
    scaler = StandardScaler().fit(x_dev)
    model = LogisticRegression(max_iter=1000).fit(
        scaler.transform(x_dev), dev[target_col].astype(int)
    )
    coefs = model.coef_[0]

    dummy_cols = pd.get_dummies(dev["track_id"], prefix="trk").columns.tolist()
    n_numeric = len(NUMERIC_FEATURES)
    numeric_coefs = coefs[:n_numeric]
    track_coefs = coefs[n_numeric : n_numeric + len(dummy_cols)]

    rows = [
        {"feature": name, "importance": float(abs(c)), "direction": float(c)}
        for name, c in zip(NUMERIC_FEATURES, numeric_coefs, strict=True)
    ]
    rows.append(
        {
            "feature": "track_id",
            "importance": float(np.sqrt(np.sum(np.square(track_coefs)))),
            "direction": float("nan"),
        }
    )
    return pd.DataFrame(rows)


def compute(target_short: str = DEFAULT_TARGET) -> pd.DataFrame:
    if target_short not in TARGETS:
        raise ValueError(f"unknown target {target_short!r}; choose from {tuple(TARGETS)}")
    target_col = TARGETS[target_short]
    dev, test = prepare_dev_test(target_col)
    if dev.empty or test.empty:
        raise RuntimeError("empty dev/test split -- run `just build` first.")

    if target_short == "podium":
        df = _xgb_importance(dev, test, target_col)
    else:
        df = _logreg_importance(dev, target_col)

    return df.sort_values("importance", ascending=False, ignore_index=True)


def _print_top(df: pd.DataFrame, target_short: str, model_name: str, source: str) -> None:
    n_show = min(15, len(df))
    print()
    print("=" * 78)
    print(f"Feature importance ({target_short}) -- {model_name}, {source}")
    print("=" * 78)
    print(f"{'feature':<32s}  {'importance':>12s}  {'direction':>12s}")
    print("-" * 78)
    for _, row in df.head(n_show).iterrows():
        d = row["direction"]
        direction_str = f"{d:>+12.4f}" if np.isfinite(d) else f"{'nan':>12s}"
        print(f"{row['feature']:<32s}  {row['importance']:>12.4f}  {direction_str}")
    if len(df) > n_show:
        print(f"... and {len(df) - n_show} more (full table -> parquet).")
    print("=" * 78)


def run(target_short: str = DEFAULT_TARGET) -> int:
    try:
        df = compute(target_short)
    except (RuntimeError, ValueError) as exc:
        print(f"[importance] {exc}", file=sys.stderr)
        return 1

    out_path = _importance_path(target_short)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    model_name = "XGBoost" if target_short == "podium" else "LogisticRegression"
    source = "mean(|SHAP|) on holdout test" if target_short == "podium" else "|standardised coef|"
    _print_top(df, target_short, model_name, source)
    print(f"Wrote {len(df)} rows -> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Compute feature importance for one target.")
    r.add_argument(
        "--target",
        choices=tuple(TARGETS),
        default=DEFAULT_TARGET,
        help=f"Which target to attribute (default: {DEFAULT_TARGET}).",
    )
    args = p.parse_args(argv)
    if args.cmd == "run":
        return run(args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
