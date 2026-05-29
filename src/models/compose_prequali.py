"""Phase 4.2.8 composition A/B: predicted qualifying as a pre-race podium input.

The pre-race podium model normally sees the REAL qualifying result (grid slot,
q_position, is_top3_grid, ...). This module asks an honest counterfactual: if we
only had PREDICTED qualifying (the Phase 4.2 pre-quali stack, run Friday night
post-FP2), how much podium signal survives?

Three feature variants are scored on the SAME sealed holdout test set
(2024-07-01 on), all predicting target_podium:

- real_grid -- the deployed pre-race set (full FEATURE_COLUMNS, real quali/grid).
- composed  -- FEATURE_COLUMNS_POST_FP2 (no quali, keeps FP2) PLUS four predicted
               quali probabilities (pole / top-3 / top-10 / beat-teammate) fed in
               as stacked features.

Plus two anchors: ConstantRate (Brier floor) and Top3Quali (the real-grid domain
rule "top-3 on the grid => podium").

Leakage discipline mirrors final_eval/prequali:
- The pre-quali features on the holdout are produced by pre-quali models trained
  ONLY on dev, so no test row ever sees its own quali outcome.
- On dev, the pre-quali feature is out-of-fold where the walk-forward windows
  cover it (2022-07-01 on); the earlier dev rows fall back to an in-sample fit
  (those rows are the oldest and least recency-weighted, and in-sample pq only
  makes the composed model MORE optimistic at train time -- a conservative bias
  for the A/B, never an inflation of the holdout number that decides it).

Run: `python -m src.models.compose_prequali eval`
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.eval.calibration import IsotonicCalibrator, ece
from src.eval.metrics import brier, paired_bootstrap_brier_ci
from src.eval.walk_forward import FitPredictFn, make_folds, oof_predictions
from src.features.build import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_POST_FP2,
    PRE_QUALI_FEATURE_SETS,
    TARGET_PODIUM,
    TARGET_POLE,
    TARGET_QUALI_BEAT_TEAMMATE,
    TARGET_TOP3_QUALI,
    TARGET_TOP10_QUALI,
)
from src.models.mvp import (
    make_constant_fit_predict,
    make_default_lgbm_fit_predict,
    make_default_xgb_fit_predict,
    make_logreg_fit_predict,
    make_top3_quali_fit_predict,
    prepare_dev_test,
)

# Predicted-quali feature column -> the quali target it estimates. Built with the
# post_fp2 pre-quali feature set (Friday-night realistic, matches the deployed
# pre-quali pipeline).
_PQ_TARGETS: dict[str, str] = {
    "pq_pole": TARGET_POLE,
    "pq_top3_quali": TARGET_TOP3_QUALI,
    "pq_top10_quali": TARGET_TOP10_QUALI,
    "pq_teammate_quali": TARGET_QUALI_BEAT_TEAMMATE,
}
_PQ_FEATURES = list(_PQ_TARGETS)
_PQ_MODE = "post_fp2"

# Composed pre-race set: drop the real quali/grid block, add the predicted one.
COMPOSED_FEATURES = FEATURE_COLUMNS_POST_FP2 + _PQ_FEATURES

_STACK_MODELS = ("LogisticRegression", "XGBoost", "LightGBM", "Ensemble")
_HEADLINE = "Ensemble"


def _stack_specs(numeric_features: list[str]) -> list[tuple[str, FitPredictFn]]:
    return [
        (
            "LogisticRegression",
            make_logreg_fit_predict(TARGET_PODIUM, numeric_features=numeric_features),
        ),
        ("XGBoost", make_default_xgb_fit_predict(TARGET_PODIUM, numeric_features=numeric_features)),
        (
            "LightGBM",
            make_default_lgbm_fit_predict(TARGET_PODIUM, numeric_features=numeric_features),
        ),
    ]


def _pq_fit_predict(target_col: str, numeric_features: list[str]) -> FitPredictFn:
    """Pre-quali logreg that drops NaN-target train rows.

    target_quali_beat_teammate is NaN for single-car / same-lap-DNF rows; the
    podium dev/test split keeps them (podium is always defined), so the pre-quali
    fit must skip them itself.
    """
    base = make_logreg_fit_predict(target_col, numeric_features=numeric_features)

    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        train = train[train[target_col].notna()]
        return base(train, val)

    return fit_predict


def _oof_feature(dev: pd.DataFrame, fit_predict: FitPredictFn, target_col: str) -> pd.Series:
    """Out-of-fold pre-quali prediction per dev row, aligned to dev.index.

    The walk-forward val windows only cover 2022-07-01 on, so earlier rows stay
    NaN here and are filled by an in-sample fit afterwards.
    """
    out = pd.Series(np.nan, index=dev.index, dtype=float)
    for train, val in make_folds(dev):
        if train[train[target_col].notna()].empty or val.empty:
            continue
        out.loc[val.index] = np.asarray(fit_predict(train, val), dtype=float)
    missing = out.isna()
    if missing.any():
        insample = np.asarray(fit_predict(dev, dev.loc[missing]), dtype=float)
        out.loc[missing] = insample
    return out


def _augment(dev: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach the four pq_* predicted-quali feature columns to dev + test."""
    numeric_features = PRE_QUALI_FEATURE_SETS[_PQ_MODE]
    dev = dev.copy()
    test = test.copy()
    for col, target_col in _PQ_TARGETS.items():
        fit_predict = _pq_fit_predict(target_col, numeric_features)
        dev[col] = _oof_feature(dev, fit_predict, target_col)
        test[col] = np.asarray(fit_predict(dev, test), dtype=float)
    return dev, test


@dataclass
class VariantEval:
    variant: str
    name: str
    brier_raw: float
    brier_cal: float
    ece_cal: float
    raw_prob: np.ndarray


def _eval_variant(
    variant: str, numeric_features: list[str], dev: pd.DataFrame, test: pd.DataFrame
) -> list[VariantEval]:
    """Train + isotonic-calibrate the podium stack on one feature variant."""
    y_true = test[TARGET_PODIUM].astype(int).to_numpy()

    raw_by: dict[str, np.ndarray] = {}
    cal_by: dict[str, np.ndarray] = {}
    for name, fit_predict in _stack_specs(numeric_features):
        raw = np.asarray(fit_predict(dev, test), dtype=float)
        oof = oof_predictions(dev, fit_predict, target_col=TARGET_PODIUM)
        cal = IsotonicCalibrator.fit(oof.y_prob, oof.y_true).transform(raw)
        raw_by[name] = raw
        cal_by[name] = cal
    raw_by["Ensemble"] = (raw_by["XGBoost"] + raw_by["LightGBM"]) / 2.0
    cal_by["Ensemble"] = (cal_by["XGBoost"] + cal_by["LightGBM"]) / 2.0

    return [
        VariantEval(
            variant=variant,
            name=name,
            brier_raw=brier(y_true, raw_by[name]),
            brier_cal=brier(y_true, cal_by[name]),
            ece_cal=ece(y_true, cal_by[name]),
            raw_prob=raw_by[name],
        )
        for name in _STACK_MODELS
    ]


def _eval_anchor(
    name: str, fit_predict: FitPredictFn, dev: pd.DataFrame, test: pd.DataFrame
) -> VariantEval:
    y_true = test[TARGET_PODIUM].astype(int).to_numpy()
    raw = np.asarray(fit_predict(dev, test), dtype=float)
    return VariantEval(
        variant="anchor",
        name=name,
        brier_raw=brier(y_true, raw),
        brier_cal=brier(y_true, raw),
        ece_cal=ece(y_true, raw),
        raw_prob=raw,
    )


def evaluate() -> tuple[dict[str, list[VariantEval]], dict[str, VariantEval], pd.DataFrame]:
    dev, test = prepare_dev_test(TARGET_PODIUM)
    if dev.empty or test.empty:
        raise RuntimeError("empty dev/test split -- run `just build` first")

    dev_pq, test_pq = _augment(dev, test)

    variants = {
        "real_grid": _eval_variant("real_grid", FEATURE_COLUMNS, dev, test),
        "composed": _eval_variant("composed", COMPOSED_FEATURES, dev_pq, test_pq),
    }
    anchors = {
        "ConstantRate": _eval_anchor(
            "ConstantRate", make_constant_fit_predict(TARGET_PODIUM), dev, test
        ),
        "Top3Quali": _eval_anchor(
            "Top3Quali", make_top3_quali_fit_predict(TARGET_PODIUM), dev, test
        ),
    }
    return variants, anchors, test


def run() -> int:
    variants, anchors, test = evaluate()
    y_true = test[TARGET_PODIUM].astype(int).to_numpy()
    race_ids = test["race_id"]

    print()
    print("=" * 90)
    print(
        f"Phase 4.2.8 composition A/B -- target = podium  "
        f"(test: {len(test)} rows, {race_ids.nunique()} races, base rate {y_true.mean():.3f})"
    )
    print("=" * 90)
    print(
        f"{'variant':<12s}  {'model':<20s}  {'brier_raw':>9s}  {'brier_cal':>9s}  {'ece_cal':>8s}"
    )
    print("-" * 90)
    for variant in ("real_grid", "composed"):
        for ev in variants[variant]:
            print(
                f"{variant:<12s}  {ev.name:<20s}  {ev.brier_raw:>9.4f}  "
                f"{ev.brier_cal:>9.4f}  {ev.ece_cal:>8.4f}"
            )
        print("-" * 90)
    for ev in anchors.values():
        print(
            f"{'anchor':<12s}  {ev.name:<20s}  {ev.brier_raw:>9.4f}  "
            f"{ev.brier_cal:>9.4f}  {ev.ece_cal:>8.4f}"
        )
    print("=" * 90)

    by_real = {e.name: e for e in variants["real_grid"]}
    by_comp = {e.name: e for e in variants["composed"]}
    composed = by_comp[_HEADLINE]
    real = by_real[_HEADLINE]

    # How much podium signal survives without real qualifying.
    lo, hi = paired_bootstrap_brier_ci(race_ids, y_true, composed.raw_prob, real.raw_prob)
    diff = composed.brier_raw - real.brier_raw
    if lo > 0:
        verdict = "real quali significantly better -- predicted quali loses signal"
    elif hi < 0:
        verdict = "composed significantly better (surprising -- inspect)"
    else:
        verdict = "no significant gap -- predicted quali recovers the podium signal"
    print(
        f"[{_HEADLINE}] composed brier_raw={composed.brier_raw:.4f} vs "
        f"real_grid={real.brier_raw:.4f}  diff={diff:+.4f}  "
        f"95% CI [{lo:+.4f}, {hi:+.4f}]  ->  {verdict}"
    )

    # Is the composed stack at least worth more than the trivial real-grid rule?
    top3 = anchors["Top3Quali"]
    lo2, hi2 = paired_bootstrap_brier_ci(race_ids, y_true, composed.raw_prob, top3.raw_prob)
    diff2 = composed.brier_raw - top3.brier_raw
    if hi2 < 0:
        verdict2 = "composed beats the real-grid Top3 rule at 95%"
    elif diff2 < 0:
        verdict2 = "composed ahead of Top3 but NOT significant"
    else:
        verdict2 = "composed does NOT beat the real-grid Top3 rule"
    print(
        f"[{_HEADLINE}] composed vs Top3Quali(real grid)  diff={diff2:+.4f}  "
        f"95% CI [{lo2:+.4f}, {hi2:+.4f}]  ->  {verdict2}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("eval", help="Holdout A/B: composed (predicted quali) vs real-grid podium.")
    args = p.parse_args(argv)
    if args.cmd == "eval":
        return run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
