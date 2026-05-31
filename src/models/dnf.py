"""Phase 5.2 DNF sub-model driver + holdout evaluation.

Predicts P(driver does not finish the race) -- a pre-race binary market in its
own right ("driver to retire"), in three timing modes:

- pre_weekend: nothing from the race weekend (Thursday / Friday morning).
- post_fp2:    pre_weekend + FP2 long-run pace (Friday night, normal weekends).
- race:        the full pre-race feature set incl. real grid + qualifying
               (Saturday evening, post-quali -- mirrors the podium model).

Reuses the Phase 1.5 stack (LogReg + XGB + LGBM, default params) through the
feature-set-parametrised factories in mvp.py -- each model trains on the mode's
feature set. Same eval harness as prequali: walk-forward OOF -> isotonic
calibration -> Brier on the sealed holdout test set (2024-07-01 on). No Optuna
(Phase 3.7 lesson: tuning overfits at this N, and DNF is rarer still).

Two baselines anchor the Go/No-Go question:
- ConstantRate    -- the trivial Brier floor (predict the base rate).
- TeamReliability -- a 1-feature logistic regression on the constructor's recent
  DNF rate (team_form_dnf_rate_l10). The DNF analogue of the RecentQualiForm
  baseline: "this car retires about as often as it has lately". If the full
  model can't beat this, the extra features add nothing.

Run: `python -m src.models.dnf eval`
     `python -m src.models.dnf eval --mode race`
     `python -m src.models.dnf select`
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.eval.calibration import IsotonicCalibrator, ece
from src.eval.metrics import brier, paired_bootstrap_brier_ci
from src.eval.walk_forward import FitPredictFn, oof_predictions
from src.features.build import (
    FEATURE_COLUMNS,
    PRE_QUALI_FEATURE_SETS,
    TARGET_DNF,
)
from src.models.mvp import (
    make_constant_fit_predict,
    make_default_lgbm_fit_predict,
    make_default_xgb_fit_predict,
    make_logreg_fit_predict,
    prepare_dev_test,
)

# Three timing modes, widest signal last. pre_weekend / post_fp2 reuse the
# pre-quali feature sets (no grid/quali); race is the full post-quali set.
DNF_FEATURE_SETS: dict[str, list[str]] = {
    "pre_weekend": PRE_QUALI_FEATURE_SETS["pre_weekend"],
    "post_fp2": PRE_QUALI_FEATURE_SETS["post_fp2"],
    "race": FEATURE_COLUMNS,
}
ALL_MODES = tuple(DNF_FEATURE_SETS)
DEFAULT_MODE = "all"

# Order shown in the report. ConstantRate + TeamReliability are baselines, the
# rest are the real stack; Ensemble is XGB+LGBM averaged after calibration.
_MODEL_ORDER = (
    "ConstantRate",
    "TeamReliability",
    "LogisticRegression",
    "XGBoost",
    "LightGBM",
    "Ensemble",
)
_BASELINES = ("ConstantRate", "TeamReliability")

_TEAM_RELIABILITY_FEATURE = "team_form_dnf_rate_l10"


def make_team_reliability_fit_predict(target_col: str = TARGET_DNF) -> FitPredictFn:
    """Domain baseline: 1-feature logistic regression on the constructor's recent
    DNF rate. Median-imputes early-season / new-team rows (no history)."""

    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        med = train[_TEAM_RELIABILITY_FEATURE].median()
        x_train = train[[_TEAM_RELIABILITY_FEATURE]].fillna(med).to_numpy()
        x_val = val[[_TEAM_RELIABILITY_FEATURE]].fillna(med).to_numpy()
        scaler = StandardScaler().fit(x_train)
        model = LogisticRegression(max_iter=1000).fit(
            scaler.transform(x_train), train[target_col].astype(int)
        )
        return model.predict_proba(scaler.transform(x_val))[:, 1]

    return fit_predict


def _specs(numeric_features: list[str]) -> list[tuple[str, FitPredictFn]]:
    t = TARGET_DNF
    return [
        ("ConstantRate", make_constant_fit_predict(t)),
        ("TeamReliability", make_team_reliability_fit_predict(t)),
        ("LogisticRegression", make_logreg_fit_predict(t, numeric_features=numeric_features)),
        ("XGBoost", make_default_xgb_fit_predict(t, numeric_features=numeric_features)),
        ("LightGBM", make_default_lgbm_fit_predict(t, numeric_features=numeric_features)),
    ]


@dataclass
class DNFEval:
    mode: str
    name: str
    brier_raw: float
    brier_cal: float
    ece_cal: float
    raw_prob: np.ndarray


def evaluate_mode(mode: str) -> list[DNFEval]:
    """Train + calibrate every model for one mode and score the holdout."""
    numeric_features = DNF_FEATURE_SETS[mode]
    dev, test = prepare_dev_test(TARGET_DNF)
    if dev.empty or test.empty:
        raise RuntimeError("empty dev/test split -- run `just build` first")

    y_true = test[TARGET_DNF].astype(int).to_numpy()

    raw_by: dict[str, np.ndarray] = {}
    cal_by: dict[str, np.ndarray] = {}
    for name, fit_predict in _specs(numeric_features):
        raw = np.asarray(fit_predict(dev, test), dtype=float)
        if name == "ConstantRate":
            cal = raw  # flat prob -> calibrator degenerate; cal == raw
        else:
            oof = oof_predictions(dev, fit_predict, target_col=TARGET_DNF)
            calibrator = IsotonicCalibrator.fit(oof.y_prob, oof.y_true)
            cal = calibrator.transform(raw)
        raw_by[name] = raw
        cal_by[name] = cal

    # Ensemble = equal-weight XGB+LGBM, mirroring final_eval._build_ensemble.
    raw_by["Ensemble"] = (raw_by["XGBoost"] + raw_by["LightGBM"]) / 2.0
    cal_by["Ensemble"] = (cal_by["XGBoost"] + cal_by["LightGBM"]) / 2.0

    return [
        DNFEval(
            mode=mode,
            name=name,
            brier_raw=brier(y_true, raw_by[name]),
            brier_cal=brier(y_true, cal_by[name]),
            ece_cal=ece(y_true, cal_by[name]),
            raw_prob=raw_by[name],
        )
        for name in _MODEL_ORDER
    ]


# Models eligible for selection. TeamReliability is included on purpose: if the
# richer features add nothing, the 1-feature baseline is the honest deployable
# pick rather than a more complex model that predicts no better.
_SELECTABLE = ("TeamReliability", "LogisticRegression", "XGBoost", "LightGBM", "Ensemble")


def dev_briers(mode: str) -> dict[str, float]:
    """Walk-forward Brier on the DEV set for every selectable model.

    Selection must never touch the holdout (same rule as prequali) -- the pick is
    made here on dev OOF; the holdout in evaluate_mode only confirms it.
    """
    numeric_features = DNF_FEATURE_SETS[mode]
    dev, _ = prepare_dev_test(TARGET_DNF)

    oof_by = {
        name: oof_predictions(dev, fit_predict, target_col=TARGET_DNF)
        for name, fit_predict in _specs(numeric_features)
        if name != "ConstantRate"
    }
    y_true = oof_by["LogisticRegression"].y_true
    briers = {name: brier(o.y_true, o.y_prob) for name, o in oof_by.items()}
    ens_prob = (oof_by["XGBoost"].y_prob + oof_by["LightGBM"].y_prob) / 2.0
    briers["Ensemble"] = brier(y_true, ens_prob)
    return briers


def select_best(mode: str) -> tuple[str, float]:
    """Best model for one mode by dev walk-forward Brier."""
    briers = dev_briers(mode)
    name = min(_SELECTABLE, key=lambda n: briers[n])
    return name, briers[name]


def run_select(mode_arg: str) -> int:
    modes = list(ALL_MODES) if mode_arg == "all" else [mode_arg]
    print("=" * 100)
    print("DNF model selection -- DEV walk-forward Brier (holdout untouched)")
    print("=" * 100)
    header = f"{'mode':<12s}  " + "  ".join(f"{n:>17s}" for n in _SELECTABLE)
    print(header)
    print("-" * 100)
    for m in modes:
        briers = dev_briers(m)
        best = min(_SELECTABLE, key=lambda n: briers[n])
        cells = "  ".join(
            (f"*{briers[n]:.4f}*" if n == best else f" {briers[n]:.4f} ").rjust(17)
            for n in _SELECTABLE
        )
        print(f"{m:<12s}  {cells}")
    print("=" * 100)
    print("* = lowest dev Brier (selected). Confirm on holdout with `dnf eval`.")
    return 0


def _print_block(modes: list[str], test_meta: pd.DataFrame) -> None:
    y_true = test_meta[TARGET_DNF].astype(int).to_numpy()
    race_ids = test_meta["race_id"]

    print()
    print("=" * 88)
    print(
        f"DNF holdout eval  (test: {len(test_meta)} rows, {race_ids.nunique()} races, "
        f"base rate {y_true.mean():.3f})"
    )
    print("=" * 88)
    print(f"{'mode':<12s}  {'model':<18s}  {'brier_raw':>9s}  {'brier_cal':>9s}  {'ece_cal':>8s}")
    print("-" * 88)

    best_per_mode: dict[str, DNFEval] = {}
    baseline_per_mode: dict[str, DNFEval] = {}
    for mode in modes:
        evals = evaluate_mode(mode)
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
        baseline_per_mode[mode] = by_name["TeamReliability"]

    # Go/No-Go: best real model vs the TeamReliability domain baseline, per mode.
    for mode in modes:
        best = best_per_mode[mode]
        base = baseline_per_mode[mode]
        lo, hi = paired_bootstrap_brier_ci(race_ids, y_true, best.raw_prob, base.raw_prob)
        diff = best.brier_raw - base.brier_raw
        if hi < 0:
            verdict = f"{best.name} beats TeamReliability at 95%"
        elif diff < 0:
            verdict = f"{best.name} ahead but NOT significant (CI straddles 0)"
        else:
            verdict = f"{best.name} does NOT beat TeamReliability"
        print(
            f"[{mode}] best={best.name} brier_raw={best.brier_raw:.4f} vs "
            f"TeamReliability={base.brier_raw:.4f}  diff={diff:+.4f}  "
            f"95% CI [{lo:+.4f}, {hi:+.4f}]  ->  {verdict}"
        )

    # Feature lift: does more weekend signal help the best model?
    order = [m for m in ("pre_weekend", "post_fp2", "race") if m in best_per_mode]
    for prev, cur in zip(order, order[1:], strict=False):
        delta = best_per_mode[cur].brier_raw - best_per_mode[prev].brier_raw
        arrow = "helps" if delta < 0 else "hurts/no change"
        print(f"Lift (best model): {cur} - {prev} brier = {delta:+.4f}  ({arrow})")


def run(mode_arg: str) -> int:
    modes = list(ALL_MODES) if mode_arg == "all" else [mode_arg]
    _, test_meta = prepare_dev_test(TARGET_DNF)
    _print_block(modes, test_meta)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd, helptext in (
        ("eval", "Walk-forward + holdout Brier for the DNF models vs baselines."),
        ("select", "Best model per mode by DEV Brier (holdout untouched)."),
    ):
        sp = sub.add_parser(cmd, help=helptext)
        sp.add_argument(
            "--mode",
            choices=(*ALL_MODES, "all"),
            default=DEFAULT_MODE,
            help=f"Timing mode (default: {DEFAULT_MODE}).",
        )
    args = p.parse_args(argv)

    if args.cmd == "eval":
        return run(args.mode)
    if args.cmd == "select":
        return run_select(args.mode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
