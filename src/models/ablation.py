"""Phase 3.6.6 ablation -- how much does each new feature group buy us?

Trains XGB + LGBM + LogReg + Ensemble on:
  - the full Phase-3.6 feature set (baseline)
  - the full set minus weather features
  - the full set minus sprint features
  - the full set minus WCC standings features
  - the full set minus era_2026plus
  - the full set with time-decay weights DISABLED (decay_per_month=None)

Each variant is fitted on dev and scored on the locked holdout test set,
calibrated with isotonic regression on OOF dev predictions. Reports
brier_cal per model + the ensemble blend. Lower = better; the delta against
the baseline row tells you what that feature group is worth.

Run: `python -m src.models.ablation run --target podium`
     `python -m src.models.ablation run --target teammate`
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.eval.calibration import IsotonicCalibrator, pair_normalize_teammate
from src.eval.metrics import brier
from src.features.build import CATEGORICAL_COLUMNS
from src.features.build import FEATURE_COLUMNS as ALL_NUMERIC_FEATURES
from src.models.mvp import (
    DEFAULT_DECAY_PER_MONTH,
    DEFAULT_TARGET,
    TARGETS,
    compute_time_decay_weights,
    prepare_dev_test,
)
from src.models.tune import best_params

_RANDOM_STATE = 42

# Feature groups added in Phase 3.6 -- each variant drops exactly one of these.
WEATHER_FEATS = [
    "weather_temp_c_race_hour",
    "weather_is_wet_race_hour",
    "weather_precip_mm_day_total",
    "weather_wind_kph_race_hour",
    "weather_temp_c_day_max",
]
SPRINT_FEATS = [
    "sprint_position",
    "sprint_gap_to_winner_ms",
    "sprint_minus_quali_pos",
    "has_sprint",
]
WCC_FEATS = ["team_season_points_pre_race", "team_season_pos_pre_race"]
ERA_2026_FEATS = ["era_2026plus"]
OVERTAKING_FEATS = ["track_overtakes_prior_mean"]
TEAM_EXEC_FEATS = ["team_exec_residual_l5", "team_exec_residual_l10"]
TEAMMATE_QUALI_FEATS = ["driver_teammate_quali_gap_l5"]
PIT_CREW_FEATS = ["team_pit_speed_resid_l5"]
START_FEATS = ["driver_start_pos_gain_l5"]


@dataclass
class VariantSpec:
    name: str
    drop_features: list[str]
    decay_per_month: float | None


def _make_variants() -> list[VariantSpec]:
    return [
        VariantSpec("Full (baseline)", drop_features=[], decay_per_month=DEFAULT_DECAY_PER_MONTH),
        VariantSpec(
            "- weather", drop_features=WEATHER_FEATS, decay_per_month=DEFAULT_DECAY_PER_MONTH
        ),
        VariantSpec(
            "- sprint", drop_features=SPRINT_FEATS, decay_per_month=DEFAULT_DECAY_PER_MONTH
        ),
        VariantSpec(
            "- WCC standings", drop_features=WCC_FEATS, decay_per_month=DEFAULT_DECAY_PER_MONTH
        ),
        VariantSpec(
            "- era_2026plus", drop_features=ERA_2026_FEATS, decay_per_month=DEFAULT_DECAY_PER_MONTH
        ),
        VariantSpec(
            "- overtaking", drop_features=OVERTAKING_FEATS, decay_per_month=DEFAULT_DECAY_PER_MONTH
        ),
        VariantSpec(
            "- team execution", drop_features=TEAM_EXEC_FEATS, decay_per_month=DEFAULT_DECAY_PER_MONTH
        ),
        VariantSpec(
            "- teammate quali", drop_features=TEAMMATE_QUALI_FEATS,
            decay_per_month=DEFAULT_DECAY_PER_MONTH,
        ),
        VariantSpec(
            "- pit crew", drop_features=PIT_CREW_FEATS, decay_per_month=DEFAULT_DECAY_PER_MONTH
        ),
        VariantSpec(
            "- start", drop_features=START_FEATS, decay_per_month=DEFAULT_DECAY_PER_MONTH
        ),
        VariantSpec("- time-decay weights", drop_features=[], decay_per_month=None),
    ]


def _numeric_feats_for_variant(drop: list[str]) -> list[str]:
    return [f for f in ALL_NUMERIC_FEATURES if f not in drop]


def _xgb_train_predict(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    numeric_feats: list[str],
    target_col: str,
    decay_per_month: float | None,
    target_short: str,
) -> np.ndarray:
    feats = numeric_feats + CATEGORICAL_COLUMNS
    params = best_params("xgboost", target_short)
    model = xgb.XGBClassifier(
        **params,
        eval_metric="logloss",
        enable_categorical=True,
        random_state=_RANDOM_STATE,
        n_jobs=-1,
    )
    w = (
        compute_time_decay_weights(dev["race_date"], decay_per_month)
        if decay_per_month is not None
        else None
    )
    model.fit(dev[feats], dev[target_col].astype(int), sample_weight=w)
    return model.predict_proba(test[feats])[:, 1]


def _lgbm_train_predict(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    numeric_feats: list[str],
    target_col: str,
    decay_per_month: float | None,
    target_short: str,
) -> np.ndarray:
    feats = numeric_feats + CATEGORICAL_COLUMNS
    params = best_params("lightgbm", target_short)
    model = lgb.LGBMClassifier(
        **params,
        subsample_freq=1,
        random_state=_RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    w = (
        compute_time_decay_weights(dev["race_date"], decay_per_month)
        if decay_per_month is not None
        else None
    )
    model.fit(dev[feats], dev[target_col].astype(int), sample_weight=w)
    return model.predict_proba(test[feats])[:, 1]


def _logreg_matrix(df: pd.DataFrame, numeric_feats: list[str], medians: pd.Series) -> np.ndarray:
    numeric = df[numeric_feats].fillna(medians)
    dummies = pd.get_dummies(df["track_id"], prefix="trk").astype(float)
    return pd.concat([numeric, dummies], axis=1).to_numpy()


def _logreg_train_predict(
    dev: pd.DataFrame,
    test: pd.DataFrame,
    numeric_feats: list[str],
    target_col: str,
    decay_per_month: float | None,
) -> np.ndarray:
    medians = dev[numeric_feats].median(numeric_only=True)
    x_dev = _logreg_matrix(dev, numeric_feats, medians)
    x_test = _logreg_matrix(test, numeric_feats, medians)
    scaler = StandardScaler().fit(x_dev)
    w = (
        compute_time_decay_weights(dev["race_date"], decay_per_month)
        if decay_per_month is not None
        else None
    )
    model = LogisticRegression(max_iter=1000).fit(
        scaler.transform(x_dev), dev[target_col].astype(int), sample_weight=w
    )
    return model.predict_proba(scaler.transform(x_test))[:, 1]


def _oof_predict_xgb(
    dev: pd.DataFrame,
    numeric_feats: list[str],
    target_col: str,
    decay_per_month: float | None,
    target_short: str,
    n_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Race-grouped OOF predictions on dev -- the calibration fit set."""
    feats = numeric_feats + CATEGORICAL_COLUMNS
    params = best_params("xgboost", target_short)
    races = dev["race_id"].unique()
    fold_assign = pd.Series({r: i % n_folds for i, r in enumerate(sorted(races))})
    oof = np.full(len(dev), np.nan)
    for k in range(n_folds):
        val_races = fold_assign[fold_assign == k].index
        tr = dev[~dev["race_id"].isin(val_races)]
        va = dev[dev["race_id"].isin(val_races)]
        model = xgb.XGBClassifier(
            **params,
            eval_metric="logloss",
            enable_categorical=True,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
        )
        w = (
            compute_time_decay_weights(tr["race_date"], decay_per_month)
            if decay_per_month is not None
            else None
        )
        model.fit(tr[feats], tr[target_col].astype(int), sample_weight=w)
        oof[dev.index.isin(va.index)] = model.predict_proba(va[feats])[:, 1]
    y_true = dev[target_col].astype(int).to_numpy()
    return oof, y_true


def _oof_predict_lgbm(
    dev: pd.DataFrame,
    numeric_feats: list[str],
    target_col: str,
    decay_per_month: float | None,
    target_short: str,
    n_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    feats = numeric_feats + CATEGORICAL_COLUMNS
    params = best_params("lightgbm", target_short)
    races = dev["race_id"].unique()
    fold_assign = pd.Series({r: i % n_folds for i, r in enumerate(sorted(races))})
    oof = np.full(len(dev), np.nan)
    for k in range(n_folds):
        val_races = fold_assign[fold_assign == k].index
        tr = dev[~dev["race_id"].isin(val_races)]
        va = dev[dev["race_id"].isin(val_races)]
        model = lgb.LGBMClassifier(
            **params,
            subsample_freq=1,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        w = (
            compute_time_decay_weights(tr["race_date"], decay_per_month)
            if decay_per_month is not None
            else None
        )
        model.fit(tr[feats], tr[target_col].astype(int), sample_weight=w)
        oof[dev.index.isin(va.index)] = model.predict_proba(va[feats])[:, 1]
    y_true = dev[target_col].astype(int).to_numpy()
    return oof, y_true


def _oof_predict_logreg(
    dev: pd.DataFrame,
    numeric_feats: list[str],
    target_col: str,
    decay_per_month: float | None,
    n_folds: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    races = dev["race_id"].unique()
    fold_assign = pd.Series({r: i % n_folds for i, r in enumerate(sorted(races))})
    oof = np.full(len(dev), np.nan)
    for k in range(n_folds):
        val_races = fold_assign[fold_assign == k].index
        tr = dev[~dev["race_id"].isin(val_races)]
        va = dev[dev["race_id"].isin(val_races)]
        medians = tr[numeric_feats].median(numeric_only=True)
        x_tr = _logreg_matrix(tr, numeric_feats, medians)
        x_va = _logreg_matrix(va, numeric_feats, medians)
        scaler = StandardScaler().fit(x_tr)
        w = (
            compute_time_decay_weights(tr["race_date"], decay_per_month)
            if decay_per_month is not None
            else None
        )
        model = LogisticRegression(max_iter=1000).fit(
            scaler.transform(x_tr), tr[target_col].astype(int), sample_weight=w
        )
        oof[dev.index.isin(va.index)] = model.predict_proba(scaler.transform(x_va))[:, 1]
    y_true = dev[target_col].astype(int).to_numpy()
    return oof, y_true


def _calibrate_and_score(
    raw: np.ndarray,
    oof_prob: np.ndarray,
    oof_true: np.ndarray,
    y_test: np.ndarray,
    *,
    constructor_ids: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    calibrator = IsotonicCalibrator.fit(oof_prob, oof_true)
    cal = calibrator.transform(raw)
    if constructor_ids is not None:
        cal = pair_normalize_teammate(cal, constructor_ids)
    return brier(y_test, cal), cal


def run_variant(
    variant: VariantSpec, dev: pd.DataFrame, test: pd.DataFrame, target_col: str, target_short: str
) -> dict[str, float]:
    numeric = _numeric_feats_for_variant(variant.drop_features)
    y_test = test[target_col].astype(int).to_numpy()

    # Train + raw predict on test
    raw_xgb = _xgb_train_predict(
        dev, test, numeric, target_col, variant.decay_per_month, target_short
    )
    raw_lgbm = _lgbm_train_predict(
        dev, test, numeric, target_col, variant.decay_per_month, target_short
    )
    raw_logreg = _logreg_train_predict(dev, test, numeric, target_col, variant.decay_per_month)

    # OOF on dev for calibration
    oof_xgb, y_oof = _oof_predict_xgb(
        dev, numeric, target_col, variant.decay_per_month, target_short
    )
    oof_lgbm, _ = _oof_predict_lgbm(dev, numeric, target_col, variant.decay_per_month, target_short)
    oof_logreg, _ = _oof_predict_logreg(dev, numeric, target_col, variant.decay_per_month)

    # For teammate, couple constructor pairs (P_A + P_B = 1) on cal probs before
    # scoring; ensemble averages two pair-normed legs so it stays pair-coupled.
    pair_ids = test["constructor_id"].to_numpy() if target_short == "teammate" else None
    brier_xgb, cal_xgb = _calibrate_and_score(
        raw_xgb, oof_xgb, y_oof, y_test, constructor_ids=pair_ids
    )
    brier_lgbm, cal_lgbm = _calibrate_and_score(
        raw_lgbm, oof_lgbm, y_oof, y_test, constructor_ids=pair_ids
    )
    brier_logreg, _ = _calibrate_and_score(
        raw_logreg, oof_logreg, y_oof, y_test, constructor_ids=pair_ids
    )
    brier_ens = brier(y_test, (cal_xgb + cal_lgbm) / 2.0)

    return {
        "xgb": brier_xgb,
        "lgbm": brier_lgbm,
        "logreg": brier_logreg,
        "ensemble": brier_ens,
    }


def _print_table(target_short: str, results: list[tuple[str, dict[str, float]]]) -> None:
    baseline = results[0][1]
    print()
    print("=" * 88)
    print(f"Phase 3.6 ablation -- target = {target_short}")
    print("Lower brier_cal = better. Delta vs baseline (positive = feature helped).")
    print("=" * 88)
    print(f"{'Variant':<28s}  {'XGB':>9s}  {'LGBM':>9s}  {'LogReg':>9s}  {'Ensemble':>9s}")
    print("-" * 88)
    for name, m in results:
        delta = (
            ""
            if name == results[0][0]
            else (f"  d_ens={m['ensemble'] - baseline['ensemble']:+.4f}")
        )
        print(
            f"{name:<28s}  {m['xgb']:>9.4f}  {m['lgbm']:>9.4f}  {m['logreg']:>9.4f}  "
            f"{m['ensemble']:>9.4f}{delta}"
        )
    print("=" * 88)


def run(target_short: str = DEFAULT_TARGET) -> int:
    if target_short not in TARGETS:
        print(f"unknown target {target_short!r}", file=sys.stderr)
        return 1
    target_col = TARGETS[target_short]

    dev, test = prepare_dev_test(target_col)
    dev = dev.reset_index(drop=True)
    test = test.reset_index(drop=True)
    print(f"[ablation] dev: {len(dev)} rows / {dev['race_id'].nunique()} races")
    print(f"[ablation] test: {len(test)} rows / {test['race_id'].nunique()} races")

    results: list[tuple[str, dict[str, float]]] = []
    for v in _make_variants():
        print(f"[ablation] running variant: {v.name}")
        results.append((v.name, run_variant(v, dev, test, target_col, target_short)))

    _print_table(target_short, results)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    p_run = sub.add_parser("run", help="Run the ablation table")
    p_run.add_argument("--target", default=DEFAULT_TARGET, choices=tuple(TARGETS))
    args = p.parse_args(argv)
    if args.cmd == "run":
        return run(args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
