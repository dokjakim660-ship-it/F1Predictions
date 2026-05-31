"""Phase 5.2 C1 ablation: does an explicit P(DNF) signal help the podium model?

The podium target already excludes DNFs (a retirement is target_podium=0) and
team_form_dnf_rate_l10 is already a feature -- so the podium model arguably
already "knows" about reliability. This ablation tests whether adding a learned,
calibrated-scale P(DNF) as one extra feature lowers holdout Brier at all.

P(DNF) is generated leak-free:
- dev (training) rows: GroupKFold(race) cross-fit -- every dev row is scored by a
  DNF model that never trained on its race.
- test (holdout) rows: a single DNF model trained on all of dev.

Each podium model is then trained twice -- on FEATURE_COLUMNS (baseline) and on
FEATURE_COLUMNS + ["pred_dnf"] (augmented) -- and scored raw on the sealed
holdout. Verdict via paired race-bootstrap CI on Brier. Eval-only.

Expected (given the Phase 5.2 verdict that DNF is near-unpredictable): no
material improvement. Documented as such.

Run: `python -m src.models.compose_dnf eval`
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.eval.metrics import brier, paired_bootstrap_brier_ci
from src.features.build import FEATURE_COLUMNS, TARGET_DNF, TARGET_PODIUM
from src.models.mvp import (
    make_default_lgbm_fit_predict,
    make_default_xgb_fit_predict,
    make_logreg_fit_predict,
    prepare_dev_test,
)

_PRED_DNF = "pred_dnf"
_N_SPLITS = 5


# The DNF model that generates the candidate feature. LightGBM on the full
# feature set -- a fair "best effort" DNF signal (the holdout verdict says even
# this ties the team-reliability baseline, but we give the ablation the strong
# version so a null result is conclusive).
def _dnf_signal_fit_predict():
    return make_default_lgbm_fit_predict(TARGET_DNF, numeric_features=FEATURE_COLUMNS)


def _crossfit_dev_pred_dnf(dev: pd.DataFrame) -> np.ndarray:
    """Leak-free P(DNF) for every dev row via GroupKFold over races."""
    races = dev["race_id"].to_numpy()
    pred = np.full(len(dev), np.nan)
    fit_predict = _dnf_signal_fit_predict()
    gkf = GroupKFold(n_splits=_N_SPLITS)
    for tr_idx, va_idx in gkf.split(dev, groups=races):
        tr = dev.iloc[tr_idx]
        va = dev.iloc[va_idx]
        pred[va_idx] = np.asarray(fit_predict(tr, va), dtype=float)
    return pred


def _attach_pred_dnf(dev: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dev = dev.copy()
    test = test.copy()
    dev[_PRED_DNF] = _crossfit_dev_pred_dnf(dev)
    # Test rows: a single DNF model trained on all of dev (test never trains it).
    test[_PRED_DNF] = np.asarray(_dnf_signal_fit_predict()(dev, test), dtype=float)
    return dev, test


_AUG_FEATURES = [*FEATURE_COLUMNS, _PRED_DNF]


def _podium_specs(numeric_features: list[str]):
    t = TARGET_PODIUM
    return {
        "LogisticRegression": make_logreg_fit_predict(t, numeric_features=numeric_features),
        "XGBoost": make_default_xgb_fit_predict(t, numeric_features=numeric_features),
        "LightGBM": make_default_lgbm_fit_predict(t, numeric_features=numeric_features),
    }


def _predict_stack(specs, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, np.ndarray]:
    probs = {name: np.asarray(fp(train, test), dtype=float) for name, fp in specs.items()}
    probs["Ensemble"] = (probs["XGBoost"] + probs["LightGBM"]) / 2.0
    return probs


@dataclass
class ComposeEval:
    name: str
    brier_base: float
    brier_aug: float
    prob_base: np.ndarray
    prob_aug: np.ndarray


def evaluate() -> tuple[list[ComposeEval], pd.Series, np.ndarray]:
    dev, test = prepare_dev_test(TARGET_PODIUM)
    if dev.empty or test.empty:
        raise RuntimeError("empty dev/test split -- run `just build` first")
    dev, test = _attach_pred_dnf(dev, test)

    y_true = test[TARGET_PODIUM].astype(int).to_numpy()
    race_ids = test["race_id"]

    base = _predict_stack(_podium_specs(FEATURE_COLUMNS), dev, test)
    aug = _predict_stack(_podium_specs(_AUG_FEATURES), dev, test)

    order = ("LogisticRegression", "XGBoost", "LightGBM", "Ensemble")
    evals = [
        ComposeEval(
            name=name,
            brier_base=brier(y_true, base[name]),
            brier_aug=brier(y_true, aug[name]),
            prob_base=base[name],
            prob_aug=aug[name],
        )
        for name in order
    ]
    return evals, race_ids, y_true


def run() -> int:
    evals, race_ids, y_true = evaluate()
    print()
    print("=" * 84)
    print(
        f"Compose-DNF ablation -- podium holdout Brier  "
        f"(test: {len(y_true)} rows, {race_ids.nunique()} races)"
    )
    print("=" * 84)
    print(f"{'model':<20s}  {'brier_base':>10s}  {'brier_aug':>10s}  {'diff(aug-base)':>14s}")
    print("-" * 84)
    for e in evals:
        diff = e.brier_aug - e.brier_base
        print(f"{e.name:<20s}  {e.brier_base:>10.4f}  {e.brier_aug:>10.4f}  {diff:>+14.4f}")
    print("-" * 84)

    # Verdict on the strongest base model: is the augmented version significantly
    # better (negative CI), worse, or indistinguishable?
    best = min(evals, key=lambda e: e.brier_base)
    lo, hi = paired_bootstrap_brier_ci(race_ids, y_true, best.prob_aug, best.prob_base)
    diff = best.brier_aug - best.brier_base
    if hi < 0:
        verdict = "pred_dnf HELPS (significant)"
    elif lo > 0:
        verdict = "pred_dnf HURTS (significant)"
    else:
        verdict = "no significant effect (CI straddles 0)"
    print(
        f"[{best.name}] aug-base brier diff={diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
        f"->  {verdict}"
    )
    print("Negative/!significant = adding an explicit P(DNF) feature does not help podium.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("eval", help="Holdout ablation: podium with vs without a P(DNF) feature.")
    args = p.parse_args(argv)
    if args.cmd == "eval":
        return run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
