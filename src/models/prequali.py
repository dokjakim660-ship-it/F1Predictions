"""Phase 4.2.2/4.2.3 pre-quali model driver + holdout evaluation.

Predicts four qualifying outcomes (pole, top-3, top-10/Q3, beat-teammate)
BEFORE qualifying has happened, in two timing modes:

- pre_weekend: nothing from the race weekend (Thursday / Friday morning).
- post_fp2:    pre_weekend + FP2 long-run pace (Friday night, normal weekends).

Reuses the Phase 1.5 stack (LogReg + XGB + LGBM, default params) through the
feature-set-parametrised factories in mvp.py -- each model trains on the reduced
pre-quali feature set instead of the full pre-race one. Same eval harness as
final_eval: walk-forward OOF -> isotonic calibration -> brier on the sealed
holdout test set (2024-07-01 on).

Two baselines anchor the Go/No-Go question:
- ConstantRate    -- the trivial Brier floor (predict the base rate).
- RecentQualiForm -- a 1-feature logistic regression on the driver's recent
  average qualifying position. The pre-quali analogue of the Top-3-Quali=Podium
  domain baseline: "you qualify roughly where you've been qualifying lately".
  If the full model can't beat this, the extra features add nothing.

Run: `python -m src.models.prequali eval`
     `python -m src.models.prequali eval --mode post_fp2 --target top10_quali`
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.eval.calibration import IsotonicCalibrator, ece, pair_normalize_teammate
from src.eval.metrics import brier, paired_bootstrap_brier_ci
from src.eval.walk_forward import FitPredictFn, oof_predictions
from src.features.build import (
    PRE_QUALI_FEATURE_SETS,
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
    prepare_dev_test,
)

# CLI short name -> target column. Mirrors mvp.TARGETS for the pre-quali targets.
PRE_QUALI_TARGETS: dict[str, str] = {
    "pole": TARGET_POLE,
    "top3_quali": TARGET_TOP3_QUALI,
    "top10_quali": TARGET_TOP10_QUALI,
    "teammate_quali": TARGET_QUALI_BEAT_TEAMMATE,
}
DEFAULT_TARGET = "top10_quali"
DEFAULT_MODE = "both"

# Order shown in the report. ConstantRate + RecentQualiForm are baselines, the
# rest are the real stack; Ensemble is XGB+LGBM averaged after calibration.
_MODEL_ORDER = (
    "ConstantRate",
    "RecentQualiForm",
    "LogisticRegression",
    "XGBoost",
    "LightGBM",
    "Ensemble",
)
_BASELINES = ("ConstantRate", "RecentQualiForm")

_RECENT_QUALI_FEATURE = "driver_form_quali_pos_l5"


def make_recent_quali_form_fit_predict(target_col: str) -> FitPredictFn:
    """Domain baseline: 1-feature logistic regression on the driver's recent
    average qualifying position. Median-imputes rookies (no history)."""

    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        med = train[_RECENT_QUALI_FEATURE].median()
        x_train = train[[_RECENT_QUALI_FEATURE]].fillna(med).to_numpy()
        x_val = val[[_RECENT_QUALI_FEATURE]].fillna(med).to_numpy()
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(max_iter=1000).fit(
            scaler.transform(x_train), train[target_col].astype(int)
        )
        return model.predict_proba(scaler.transform(x_val))[:, 1]

    return fit_predict


def _specs(target_col: str, numeric_features: list[str]) -> list[tuple[str, FitPredictFn]]:
    return [
        ("ConstantRate", make_constant_fit_predict(target_col)),
        ("RecentQualiForm", make_recent_quali_form_fit_predict(target_col)),
        (
            "LogisticRegression",
            make_logreg_fit_predict(target_col, numeric_features=numeric_features),
        ),
        ("XGBoost", make_default_xgb_fit_predict(target_col, numeric_features=numeric_features)),
        ("LightGBM", make_default_lgbm_fit_predict(target_col, numeric_features=numeric_features)),
    ]


@dataclass
class PQEval:
    mode: str
    target: str
    name: str
    brier_raw: float
    brier_cal: float
    ece_cal: float
    raw_prob: np.ndarray


def evaluate_mode_target(mode: str, target_short: str) -> list[PQEval]:
    """Train + calibrate every model for one (mode, target) and score the holdout."""
    target_col = PRE_QUALI_TARGETS[target_short]
    numeric_features = PRE_QUALI_FEATURE_SETS[mode]
    is_teammate = target_short == "teammate_quali"

    dev, test = prepare_dev_test(target_col)
    if dev.empty or test.empty:
        raise RuntimeError("empty dev/test split -- run `just build` first")

    y_true = test[target_col].astype(int).to_numpy()
    constructor_ids = test["constructor_id"].to_numpy()

    raw_by: dict[str, np.ndarray] = {}
    cal_by: dict[str, np.ndarray] = {}
    for name, fit_predict in _specs(target_col, numeric_features):
        raw = np.asarray(fit_predict(dev, test), dtype=float)
        if name == "ConstantRate":
            cal = raw  # flat prob -> calibrator degenerate; cal == raw
        else:
            oof = oof_predictions(dev, fit_predict, target_col=target_col)
            calibrator = IsotonicCalibrator.fit(oof.y_prob, oof.y_true)
            cal = calibrator.transform(raw)
            if is_teammate:
                cal = pair_normalize_teammate(cal, constructor_ids)
        raw_by[name] = raw
        cal_by[name] = cal

    # Ensemble = equal-weight XGB+LGBM, mirroring final_eval._build_ensemble.
    raw_by["Ensemble"] = (raw_by["XGBoost"] + raw_by["LightGBM"]) / 2.0
    cal_by["Ensemble"] = (cal_by["XGBoost"] + cal_by["LightGBM"]) / 2.0

    return [
        PQEval(
            mode=mode,
            target=target_short,
            name=name,
            brier_raw=brier(y_true, raw_by[name]),
            brier_cal=brier(y_true, cal_by[name]),
            ece_cal=ece(y_true, cal_by[name]),
            raw_prob=raw_by[name],
        )
        for name in _MODEL_ORDER
    ]


def _print_target_block(target_short: str, modes: list[str], test_meta: pd.DataFrame) -> None:
    target_col = PRE_QUALI_TARGETS[target_short]
    y_true = test_meta[target_col].astype(int).to_numpy()
    race_ids = test_meta["race_id"]

    print()
    print("=" * 88)
    print(
        f"Pre-quali holdout eval -- target = {target_short}  "
        f"(test: {len(test_meta)} rows, {race_ids.nunique()} races, "
        f"base rate {y_true.mean():.3f})"
    )
    print("=" * 88)
    print(f"{'mode':<12s}  {'model':<18s}  {'brier_raw':>9s}  {'brier_cal':>9s}  {'ece_cal':>8s}")
    print("-" * 88)

    best_per_mode: dict[str, PQEval] = {}
    baseline_per_mode: dict[str, PQEval] = {}
    for mode in modes:
        evals = evaluate_mode_target(mode, target_short)
        by_name = {ev.name: ev for ev in evals}
        for ev in evals:
            tag = "  <- baseline" if ev.name in _BASELINES else ""
            print(
                f"{mode:<12s}  {ev.name:<18s}  {ev.brier_raw:>9.4f}  "
                f"{ev.brier_cal:>9.4f}  {ev.ece_cal:>8.4f}{tag}"
            )
        print("-" * 88)
        real = [e for e in evals if e.name not in _BASELINES]
        best_per_mode[mode] = min(real, key=lambda e: e.brier_raw)
        baseline_per_mode[mode] = by_name["RecentQualiForm"]

    # Go/No-Go: best real model vs the RecentQualiForm domain baseline, per mode.
    for mode in modes:
        best = best_per_mode[mode]
        base = baseline_per_mode[mode]
        lo, hi = paired_bootstrap_brier_ci(race_ids, y_true, best.raw_prob, base.raw_prob)
        diff = best.brier_raw - base.brier_raw
        if hi < 0:
            verdict = f"{best.name} beats RecentQualiForm at 95%"
        elif diff < 0:
            verdict = f"{best.name} ahead but NOT significant (CI straddles 0)"
        else:
            verdict = f"{best.name} does NOT beat RecentQualiForm"
        print(
            f"[{mode}] best={best.name} brier_raw={best.brier_raw:.4f} vs "
            f"RecentQualiForm={base.brier_raw:.4f}  diff={diff:+.4f}  "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]  ->  {verdict}"
        )

    # FP2 lift: does post_fp2 improve on pre_weekend for the best model?
    if "pre_weekend" in best_per_mode and "post_fp2" in best_per_mode:
        delta = best_per_mode["post_fp2"].brier_raw - best_per_mode["pre_weekend"].brier_raw
        arrow = "helps" if delta < 0 else "hurts/no change"
        print(f"FP2 lift (best model): post_fp2 - pre_weekend brier = {delta:+.4f}  ({arrow})")


def run(mode: str, target_short: str) -> int:
    modes = ["pre_weekend", "post_fp2"] if mode == "both" else [mode]
    targets = list(PRE_QUALI_TARGETS) if target_short == "all" else [target_short]
    for t in targets:
        # One shared test slice per target for the summary stats (same split all
        # models see inside evaluate_mode_target).
        _, test_meta = prepare_dev_test(PRE_QUALI_TARGETS[t])
        _print_target_block(t, modes, test_meta)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval", help="Walk-forward + holdout Brier for pre-quali models.")
    ev.add_argument(
        "--mode",
        choices=("pre_weekend", "post_fp2", "both"),
        default=DEFAULT_MODE,
        help=f"Timing mode (default: {DEFAULT_MODE}).",
    )
    ev.add_argument(
        "--target",
        choices=(*PRE_QUALI_TARGETS, "all"),
        default="all",
        help="Quali target (default: all).",
    )
    args = p.parse_args(argv)

    if args.cmd == "eval":
        return run(args.mode, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
