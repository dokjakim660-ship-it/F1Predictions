"""Final MVP evaluation on the sealed holdout test set (podium + teammate H2H).

Trains the MVP model set on the full dev set, calibrates each with isotonic
regression fitted on leak-free out-of-fold predictions, and scores raw vs.
calibrated probabilities on the holdout test set (races 2024-07-01 onward).
ConstantRate and the logistic baseline are scored on the same split, so every
number in the final table is directly comparable.

Outputs (per target):
- one MLflow run per model, under experiment "mvp_{target}"
- predictions/mvp_test_{target}.parquet  (per-row test probs, every model)
- models/reliability_mvp_{target}.png    (reliability diagram)
- a models/CHANGELOG.md row per real model

Run: `python -m src.models.final_eval run --target podium`
     `python -m src.models.final_eval run --target teammate`
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.eval.calibration import IsotonicCalibrator, calibration_plot, ece
from src.eval.metrics import brier, logloss, paired_bootstrap_brier_ci, top3_accuracy_per_race
from src.eval.walk_forward import FitPredictFn, oof_predictions
from src.models.mvp import (
    DEFAULT_TARGET,
    TARGETS,
    make_constant_fit_predict,
    make_logreg_fit_predict,
    make_top3_quali_fit_predict,
    prepare_dev_test,
)
from src.models.tune import best_params, make_lgbm_fit_predict, make_xgb_fit_predict
from src.utils.paths import MLRUNS_DIR, MODELS_DIR, PREDICTIONS_DIR

CHANGELOG_PATH = MODELS_DIR / "CHANGELOG.md"

# Real models, ranked for the best-model pick (excludes the ConstantRate floor).
# The Ensemble is XGB+LGBM averaged after calibration -- added in run() once
# the base evals exist; carried here so it shows in the CHANGELOG and plot.
_REAL_MODELS = ("LogisticRegression", "XGBoost", "LightGBM", "Ensemble")


def _reliability_plot_path(target_short: str) -> Path:
    return MODELS_DIR / f"reliability_mvp_{target_short}.png"


def _predictions_path(target_short: str) -> Path:
    return PREDICTIONS_DIR / f"mvp_test_{target_short}.parquet"


def _experiment_name(target_short: str) -> str:
    return f"mvp_{target_short}"


@dataclass
class ModelEval:
    name: str
    raw_prob: np.ndarray
    cal_prob: np.ndarray
    metrics: dict[str, float]
    params: dict = field(default_factory=dict)


def _model_specs(target_col: str, target_short: str) -> list[tuple[str, FitPredictFn, dict]]:
    """Name, fit_predict closure, and hyperparameters for each MVP model."""
    xgb_params = best_params("xgboost", target_short)
    lgbm_params = best_params("lightgbm", target_short)
    specs: list[tuple[str, FitPredictFn, dict]] = [
        ("ConstantRate", make_constant_fit_predict(target_col), {}),
        ("LogisticRegression", make_logreg_fit_predict(target_col), {}),
        ("XGBoost", make_xgb_fit_predict(xgb_params, target_col), xgb_params),
        ("LightGBM", make_lgbm_fit_predict(lgbm_params, target_col), lgbm_params),
    ]
    # The Top-3-Quali baseline (PLANNING.md §10 success criterion) only makes
    # sense for the podium target -- "top-3 grid" has no analogue for the
    # teammate H2H, where both team-mates start adjacent grid slots anyway.
    if target_short == "podium":
        specs.insert(1, ("Top3Quali", make_top3_quali_fit_predict(target_col), {}))
    return specs


def _evaluate_model(
    name: str,
    fit_predict: FitPredictFn,
    params: dict,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
) -> ModelEval:
    # Final model trains on the full dev set and scores the holdout test set.
    raw_prob = np.asarray(fit_predict(dev, test), dtype=float)

    # Isotonic calibrator is fitted on leak-free out-of-fold dev predictions.
    oof = oof_predictions(dev, fit_predict, target_col=target_col)
    calibrator = IsotonicCalibrator.fit(oof.y_prob, oof.y_true)
    cal_prob = calibrator.transform(raw_prob)

    y_true = test[target_col].astype(int).to_numpy()
    race_ids = test["race_id"]
    metrics = {
        "brier_raw": brier(y_true, raw_prob),
        "brier_cal": brier(y_true, cal_prob),
        "logloss_raw": logloss(y_true, raw_prob),
        "logloss_cal": logloss(y_true, cal_prob),
        "ece_raw": ece(y_true, raw_prob),
        "ece_cal": ece(y_true, cal_prob),
        "top3_raw": top3_accuracy_per_race(race_ids, y_true, raw_prob),
        "top3_cal": top3_accuracy_per_race(race_ids, y_true, cal_prob),
    }
    return ModelEval(
        name=name, raw_prob=raw_prob, cal_prob=cal_prob, metrics=metrics, params=params
    )


def _build_ensemble(
    xgb_ev: ModelEval,
    lgbm_ev: ModelEval,
    y_true: np.ndarray,
    race_ids: pd.Series,
) -> ModelEval:
    """Equal-weight blend of XGB + LGBM. Trees were tied within bootstrap noise
    on the dev set, so the textbook follow-up is to average them and check if
    decorrelated errors buy a bit of headroom on the holdout.
    """
    raw_prob = (xgb_ev.raw_prob + lgbm_ev.raw_prob) / 2.0
    cal_prob = (xgb_ev.cal_prob + lgbm_ev.cal_prob) / 2.0
    metrics = {
        "brier_raw": brier(y_true, raw_prob),
        "brier_cal": brier(y_true, cal_prob),
        "logloss_raw": logloss(y_true, raw_prob),
        "logloss_cal": logloss(y_true, cal_prob),
        "ece_raw": ece(y_true, raw_prob),
        "ece_cal": ece(y_true, cal_prob),
        "top3_raw": top3_accuracy_per_race(race_ids, y_true, raw_prob),
        "top3_cal": top3_accuracy_per_race(race_ids, y_true, cal_prob),
    }
    return ModelEval(
        name="Ensemble", raw_prob=raw_prob, cal_prob=cal_prob, metrics=metrics, params={}
    )


def _save_predictions(
    test: pd.DataFrame, evals: list[ModelEval], target_col: str, target_short: str
) -> Path:
    out = test[["race_id", "year", "round", "driver_id", target_col]].copy()
    for ev in evals:
        key = ev.name.lower()
        out[f"prob_{key}_raw"] = ev.raw_prob
        out[f"prob_{key}_cal"] = ev.cal_prob
    path = _predictions_path(target_short)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def _log_mlflow(evals: list[ModelEval], plot_path: Path, target_short: str) -> None:
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())
    mlflow.set_experiment(_experiment_name(target_short))
    for ev in evals:
        with mlflow.start_run(run_name=ev.name):
            mlflow.log_param("model", ev.name)
            mlflow.log_param("feature_set", "rich_v1")
            mlflow.log_param("target", target_short)
            if ev.params:
                mlflow.log_params(ev.params)
            mlflow.log_metrics({f"test_{k}": v for k, v in ev.metrics.items()})
            if plot_path.exists():
                mlflow.log_artifact(str(plot_path))


def _git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _append_changelog(evals: list[ModelEval], target_short: str) -> None:
    if not CHANGELOG_PATH.exists():
        return
    sha = _git_sha()
    today = date.today().isoformat()
    note = f"Phase 1.4 — rich features ({target_short}), walk-forward+Optuna (raw probs)"
    existing = CHANGELOG_PATH.read_text(encoding="utf-8")
    target_marker = f"({target_short})"

    rows: list[str] = []
    for ev in evals:
        if ev.name not in _REAL_MODELS:
            continue
        # Dedupe: if a row dated today already names this model AND target, this
        # is a same-day re-run -- skip to keep the changelog readable.
        row_prefix = f"| {today} | {ev.name} |"
        if any(row_prefix in ln and target_marker in ln for ln in existing.splitlines()):
            continue
        rows.append(
            f"| {today} | {ev.name} | {sha} | {ev.metrics['brier_raw']:.4f} "
            f"| {ev.metrics['ece_raw']:.4f} | {note} |"
        )
    if not rows:
        return

    lines = [ln for ln in existing.splitlines() if "_no models yet_" not in ln]
    table_rows = [i for i, ln in enumerate(lines) if ln.lstrip().startswith("|")]
    insert_at = table_rows[-1] + 1
    lines[insert_at:insert_at] = rows
    CHANGELOG_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _print_report(
    test: pd.DataFrame,
    evals: list[ModelEval],
    target_col: str,
    target_short: str,
    predictions_path: Path,
) -> None:
    by_name = {ev.name: ev for ev in evals}
    print()
    print("=" * 95)
    print(
        f"Final MVP evaluation ({target_short}) -- holdout test set (2024-07-01 onward): "
        f"{len(test)} rows, {test['race_id'].nunique()} races"
    )
    print("=" * 95)
    print(
        f"{'model':<20s}  {'brier_raw':>9s}  {'brier_cal':>9s}  {'ece_raw':>7s}  "
        f"{'ece_cal':>7s}  {'logloss_cal':>11s}  {'top3_cal':>8s}"
    )
    print("-" * 95)
    for ev in evals:
        m = ev.metrics
        print(
            f"{ev.name:<20s}  {m['brier_raw']:>9.4f}  {m['brier_cal']:>9.4f}  "
            f"{m['ece_raw']:>7.4f}  {m['ece_cal']:>7.4f}  {m['logloss_cal']:>11.4f}  "
            f"{m['top3_cal']:>8.3f}"
        )
    print("=" * 95)

    ranked = sorted(_REAL_MODELS, key=lambda n: by_name[n].metrics["brier_raw"])
    best, second = ranked[0], ranked[1]
    y_true = test[target_col].astype(int).to_numpy()
    lo, hi = paired_bootstrap_brier_ci(
        test["race_id"], y_true, by_name[best].raw_prob, by_name[second].raw_prob
    )
    verdict = "significant at 95%" if hi < 0 else "NOT significant (CI straddles 0)"
    print(f"Paired bootstrap (1000x, races resampled): Brier({best}) - Brier({second}), raw probs")
    print(f"  95% CI [{lo:+.4f}, {hi:+.4f}]  ->  {verdict}")

    # Phase 1 success criterion (PLANNING.md §10): the best real model must
    # beat the "Top-3-Quali = Podium" F1-domain baseline on the holdout test.
    if "Top3Quali" in by_name:
        bl_lo, bl_hi = paired_bootstrap_brier_ci(
            test["race_id"], y_true, by_name[best].raw_prob, by_name["Top3Quali"].raw_prob
        )
        bl_diff = by_name[best].metrics["brier_raw"] - by_name["Top3Quali"].metrics["brier_raw"]
        if bl_hi < 0:
            bl_verdict = "Phase 1 success criterion MET (best model beats baseline at 95%)"
        elif bl_diff < 0:
            bl_verdict = "best model is ahead but NOT significant at 95% (CI straddles 0)"
        else:
            bl_verdict = "Phase 1 success criterion FAILED (best model does NOT beat baseline)"
        print(f"Paired bootstrap: Brier({best}) - Brier(Top3Quali), raw probs")
        print(f"  95% CI [{bl_lo:+.4f}, {bl_hi:+.4f}]  (mean diff {bl_diff:+.4f})  ->  {bl_verdict}")

    helped = [
        n for n in _REAL_MODELS if by_name[n].metrics["brier_cal"] < by_name[n].metrics["brier_raw"]
    ]
    if helped:
        print(f"Isotonic calibration lowered Brier for: {', '.join(helped)}")
    else:
        print("Isotonic calibration lowered Brier for no model -- raw probs already calibrated.")
    print(f"Best model (raw Brier): {best} = {by_name[best].metrics['brier_raw']:.4f}")
    print("Lower brier/ece/logloss = better; higher top3 = better.")
    print(f"MLflow -> mlruns/   |   predictions -> {predictions_path}")


def run(target_short: str = DEFAULT_TARGET) -> int:
    if target_short not in TARGETS:
        print(
            f"[final_eval] unknown target {target_short!r}; choose from {tuple(TARGETS)}",
            file=sys.stderr,
        )
        return 1
    target_col = TARGETS[target_short]
    dev, test = prepare_dev_test(target_col)
    if dev.empty or test.empty:
        print("[final_eval] empty dev/test split -- run `just build` first.", file=sys.stderr)
        return 1

    evals = [
        _evaluate_model(name, fit_predict, params, dev, test, target_col)
        for name, fit_predict, params in _model_specs(target_col, target_short)
    ]

    y_true = test[target_col].astype(int).to_numpy()
    by_name = {ev.name: ev for ev in evals}
    evals.append(_build_ensemble(by_name["XGBoost"], by_name["LightGBM"], y_true, test["race_id"]))
    curves = {f"{ev.name} (cal)": (y_true, ev.cal_prob) for ev in evals if ev.name in _REAL_MODELS}
    plot_path = calibration_plot(curves, _reliability_plot_path(target_short))

    predictions_path = _save_predictions(test, evals, target_col, target_short)
    _log_mlflow(evals, plot_path, target_short)
    _append_changelog(evals, target_short)
    _print_report(test, evals, target_col, target_short, predictions_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Train on dev, calibrate, evaluate on the holdout test set.")
    r.add_argument(
        "--target",
        choices=tuple(TARGETS),
        default=DEFAULT_TARGET,
        help=f"Which target to evaluate (default: {DEFAULT_TARGET}).",
    )
    args = p.parse_args(argv)

    if args.cmd == "run":
        return run(args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
