"""One-shot diagnostic: compare brier_cal under three regimes for both targets.

Legacy   : eps=0.0, no pair-norm  (= pre-Phase-3.8 behaviour)
+cap     : eps=0.02, no pair-norm
+pair    : eps=0.02 + pair_normalize_teammate  (teammate target only)

Run: `python -m scripts.compare_calibration`
"""

from __future__ import annotations

from src.eval.calibration import IsotonicCalibrator, pair_normalize_teammate
from src.eval.metrics import brier
from src.eval.walk_forward import oof_predictions
from src.models.mvp import TARGETS, make_logreg_fit_predict, prepare_dev_test
from src.models.tune import best_params, make_lgbm_fit_predict, make_xgb_fit_predict


def _specs(target_col, target_short):
    return [
        ("LogisticRegression", make_logreg_fit_predict(target_col)),
        ("XGBoost", make_xgb_fit_predict(best_params("xgboost", target_short), target_col)),
        ("LightGBM", make_lgbm_fit_predict(best_params("lightgbm", target_short), target_col)),
    ]


def _cal_table(target_short):
    target_col = TARGETS[target_short]
    dev, test = prepare_dev_test(target_col)
    y_test = test[target_col].astype(int).to_numpy()
    constructor_ids = test["constructor_id"].to_numpy()
    apply_pair = target_short == "teammate"

    cals_by_scheme = {"legacy": {}, "cap": {}, "full": {}}
    for name, fp in _specs(target_col, target_short):
        raw = fp(dev, test)
        oof = oof_predictions(dev, fp, target_col=target_col)
        cal_legacy = IsotonicCalibrator.fit(oof.y_prob, oof.y_true, eps=0.0).transform(raw)
        cal_cap = IsotonicCalibrator.fit(oof.y_prob, oof.y_true, eps=0.02).transform(raw)
        cal_full = pair_normalize_teammate(cal_cap, constructor_ids) if apply_pair else cal_cap
        cals_by_scheme["legacy"][name] = cal_legacy
        cals_by_scheme["cap"][name] = cal_cap
        cals_by_scheme["full"][name] = cal_full

    # Ensemble = mean(XGB, LGBM) under each scheme
    for scheme, d in cals_by_scheme.items():
        d["Ensemble"] = (d["XGBoost"] + d["LightGBM"]) / 2.0

    print()
    print("=" * 72)
    print(f"Target = {target_short}  (test set: {len(y_test)} rows)")
    print("=" * 72)
    print(f"{'Model':<22s}  {'legacy':>10s}  {'+cap':>10s}  {'+pair-norm':>12s}")
    print("-" * 72)
    for name in ("LogisticRegression", "XGBoost", "LightGBM", "Ensemble"):
        b_l = brier(y_test, cals_by_scheme["legacy"][name])
        b_c = brier(y_test, cals_by_scheme["cap"][name])
        b_f = brier(y_test, cals_by_scheme["full"][name])
        d_c = b_c - b_l
        d_f = b_f - b_l
        print(
            f"{name:<22s}  {b_l:>10.4f}  {b_c:>10.4f} ({d_c:+.4f})  "
            f"{b_f:>10.4f} ({d_f:+.4f})"
        )


if __name__ == "__main__":
    for target in ("teammate", "podium"):
        _cal_table(target)
