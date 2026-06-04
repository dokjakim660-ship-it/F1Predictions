"""A/B of calibration methods on the sealed holdout: raw vs isotonic vs beta
vs Venn-Abers, scored overall AND per era slice (pre-2026 vs the 2026 reg
reset).

Why this is the highest-value lever before going live: betting EV depends on
the *absolute* probability being trustworthy, not on the Brier rank. The drift
report showed isotonic does not lower Brier for the tree models and leaves ECE
around 0.03, so there is concrete calibration headroom to chase.

For each real model (LogReg, XGB, LGBM, Ensemble) we fit every calibrator on
the leak-free out-of-fold dev predictions and transform the holdout, exactly as
final_eval does, so the comparison is apples-to-apples with what gets deployed.

Run: python -m src.eval.calib_ab run            # both targets
     python -m src.eval.calib_ab run --target podium
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.eval.calibration import (
    BetaCalibrator,
    IsotonicCalibrator,
    VennAbersCalibrator,
    ece,
    pair_normalize_teammate,
)
from src.eval.metrics import brier
from src.eval.walk_forward import oof_predictions
from src.models.final_eval import _model_specs
from src.models.mvp import DEFAULT_TARGET, TARGETS, prepare_dev_test

# Real models only -- the ConstantRate floor and Top3Quali baseline are not
# calibrated, they ARE the reference. Ensemble is built below from XGB+LGBM.
_BASE_MODELS = ("LogisticRegression", "XGBoost", "LightGBM")

# name -> calibrator class (None == identity / raw probabilities).
_CALIBRATORS: dict[str, type | None] = {
    "raw": None,
    "isotonic": IsotonicCalibrator,
    "beta": BetaCalibrator,
    "venn-abers": VennAbersCalibrator,
}


def _base_probs(
    dev: pd.DataFrame, test: pd.DataFrame, target_col: str, target_short: str
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """For each model: (raw test probs, OOF dev probs, OOF dev truth).

    The ensemble is the equal-weight mean of XGB+LGBM on both the test probs and
    the OOF probs. OOF rows are concatenated fold-by-fold in a deterministic
    order shared by every model, so the two trees' OOF arrays are row-aligned
    and can be averaged directly.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, fit_predict, _params in _model_specs(target_col, target_short):
        if name not in _BASE_MODELS:
            continue
        raw = np.asarray(fit_predict(dev, test), dtype=float)
        oof = oof_predictions(dev, fit_predict, target_col=target_col)
        out[name] = (raw, oof.y_prob, oof.y_true)
    xr, xo, xt = out["XGBoost"]
    lr, lo, _lt = out["LightGBM"]
    out["Ensemble"] = ((xr + lr) / 2.0, (xo + lo) / 2.0, xt)
    return out


def _calibrate(
    method: str,
    raw_test: np.ndarray,
    oof_prob: np.ndarray,
    oof_true: np.ndarray,
    test: pd.DataFrame,
    target_short: str,
) -> np.ndarray:
    cls = _CALIBRATORS[method]
    cal = raw_test if cls is None else cls.fit(oof_prob, oof_true).transform(raw_test)
    # Same post-hoc coupling final_eval applies, so teammate numbers are
    # comparable to what ships. Independent calibration of two team-mates can
    # break the P_A + P_B = 1 logic; this restores it.
    if target_short == "teammate":
        cal = pair_normalize_teammate(cal, test["constructor_id"].to_numpy())
    return cal


def _slices(test: pd.DataFrame) -> dict[str, np.ndarray]:
    year = test["year"].to_numpy()
    mask_26 = year >= 2026
    masks = {"all": np.ones(len(test), dtype=bool), "<2026": ~mask_26}
    if mask_26.any():
        masks["2026"] = mask_26
    return masks


def run(target_short: str) -> int:
    if target_short not in TARGETS:
        print(f"[calib_ab] unknown target {target_short!r}", file=sys.stderr)
        return 1
    target_col = TARGETS[target_short]
    dev, test = prepare_dev_test(target_col)
    if dev.empty or test.empty:
        print("[calib_ab] empty dev/test -- run `just build` first.", file=sys.stderr)
        return 1

    y_true = test[target_col].astype(int).to_numpy()
    masks = _slices(test)
    base = _base_probs(dev, test, target_col, target_short)

    print()
    print("=" * 100)
    print(
        f"Calibration A/B ({target_short}) -- holdout {len(test)} rows, "
        f"{test['race_id'].nunique()} races  |  metric: ECE (lower=better) / Brier (lower=better)"
    )
    print("=" * 100)
    header = f"{'model':<18s}  {'method':<11s}"
    for s in masks:
        header += f"  {'ece_' + s:>10s}  {'brier_' + s:>11s}"
    print(header)
    print("-" * 100)

    # Track the lowest-ECE method per model on the full holdout for a summary.
    best_by_model: dict[str, tuple[str, float]] = {}
    for name in (*_BASE_MODELS, "Ensemble"):
        raw_test, oof_prob, oof_true = base[name]
        for method in _CALIBRATORS:
            cal = _calibrate(method, raw_test, oof_prob, oof_true, test, target_short)
            row = f"{name:<18s}  {method:<11s}"
            for s, m in masks.items():
                row += f"  {ece(y_true[m], cal[m]):>10.4f}  {brier(y_true[m], cal[m]):>11.4f}"
            print(row)
            e_all = ece(y_true, cal)
            if name not in best_by_model or e_all < best_by_model[name][1]:
                best_by_model[name] = (method, e_all)
        print("-" * 100)

    print("Best calibration per model (lowest ece_all):")
    for name in (*_BASE_MODELS, "Ensemble"):
        method, e = best_by_model[name]
        print(f"  {name:<18s} -> {method:<11s} (ece_all {e:.4f})")
    print("=" * 100)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="A/B calibration methods on the holdout.")
    r.add_argument(
        "--target",
        choices=(*TARGETS, "both"),
        default="both",
        help="Which target to evaluate (default: both).",
    )
    args = p.parse_args(argv)
    if args.cmd != "run":
        return 0
    targets = tuple(TARGETS) if args.target == "both" else (args.target,)
    rc = 0
    for t in targets:
        rc |= run(t)
    return rc


if __name__ == "__main__":
    sys.exit(main())
