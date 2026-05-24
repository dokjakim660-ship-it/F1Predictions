"""Phase 3.3 next-race inference.

Re-runs the Phase 1.5 model stack against a single future race:
- trains XGB + LGBM (Optuna-tuned) + LogisticRegression on the full historical
  feature table (mvp.parquet),
- fits an isotonic calibrator per model on leak-free OOF predictions over the
  dev fold-set (same logic as final_eval.py),
- predict_probas onto data/features/next_race.parquet,
- blends XGB + LGBM calibrated probs into an Ensemble — same combination as
  Phase 1.5's published best.

Train-on-demand instead of loading a pickled model: full training takes a few
seconds and removes a stale-pickle failure mode. Monthly re-tuning + saved
artefacts are deferred to a later phase ([[project_mlops]] re-training plan).

Run: `python -m src.models.predict_next run --year Y --round N`
     `python -m src.models.predict_next run --year Y --round N --target teammate`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.calibration import IsotonicCalibrator, pair_normalize_teammate
from src.eval.walk_forward import oof_predictions, split_dev_test
from src.features.next_race import load_next_race
from src.models.mvp import (
    DEFAULT_DECAY_PER_MONTH,
    DEFAULT_TARGET,
    TARGETS,
    load_model_frame,
    make_constant_fit_predict,
    make_logreg_fit_predict,
    make_top3_quali_fit_predict,
)
from src.models.tune import best_params, make_lgbm_fit_predict, make_xgb_fit_predict
from src.utils.paths import PREDICTIONS_DIR

# Per-target time-decay defaults from the Phase 3.6.6 ablation on the
# locked holdout test set:
#   podium   : decay HURT (-0.0008 ensemble Brier) -> keep weights off
#   teammate : decay HELPED (+0.0009 ensemble Brier) -> keep weights on
# Callers can still override via predict_next_race(... decay_per_month=...).
_DEFAULT_DECAY_PER_TARGET: dict[str, float | None] = {
    "podium": None,
    "teammate": DEFAULT_DECAY_PER_MONTH,
}


# One-file-per-target output. The Streamlit "Next Race" page in Phase 3.4
# will read this same path.
def predictions_path(target_short: str) -> Path:
    return PREDICTIONS_DIR / f"next_race_{target_short}.parquet"


# Models we predict + ensemble. Mirrors final_eval._REAL_MODELS minus the
# Top3Quali baseline (kept only for the podium target — see _model_specs).
_REAL_MODELS = ("LogisticRegression", "XGBoost", "LightGBM", "Ensemble")


def _model_specs(target_col: str, target_short: str, *, decay_per_month: float | None = None):
    """Same factories final_eval uses, in the same order. Trees use the Optuna
    best params from `models/optuna.db`; LogReg uses sklearn defaults.

    decay_per_month: opt-in time-decay sample weights (Phase 3.6.5). Default
    None preserves the historical unweighted behaviour.
    """
    xgb_params = best_params("xgboost", target_short)
    lgbm_params = best_params("lightgbm", target_short)
    specs = [
        ("ConstantRate", make_constant_fit_predict(target_col), {}),
        (
            "LogisticRegression",
            make_logreg_fit_predict(target_col, decay_per_month=decay_per_month),
            {},
        ),
        (
            "XGBoost",
            make_xgb_fit_predict(xgb_params, target_col, decay_per_month=decay_per_month),
            xgb_params,
        ),
        (
            "LightGBM",
            make_lgbm_fit_predict(lgbm_params, target_col, decay_per_month=decay_per_month),
            lgbm_params,
        ),
    ]
    if target_short == "podium":
        specs.insert(1, ("Top3Quali", make_top3_quali_fit_predict(target_col), {}))
    return specs


def _align_track_categories(next_df: pd.DataFrame, mvp_df: pd.DataFrame) -> pd.DataFrame:
    """next_race's track_id must share mvp.parquet's category set, or the LogReg
    one-hot encoder emits a different column ordering for train vs val.
    """
    categories = mvp_df["track_id"].cat.categories
    next_df = next_df.copy()
    next_df["track_id"] = pd.Categorical(next_df["track_id"], categories=categories)
    return next_df


def _predict_one_model(
    name: str,
    fit_predict,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    next_df: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (raw_prob, cal_prob) aligned to next_df row order.

    Trained on `train_df` (= dev + test, the full historical record). Calibrator
    is fitted on `dev_df` OOF predictions — same construction as Phase 1.5, so
    the calibration curve matches the published reliability plot.
    """
    raw_prob = np.asarray(fit_predict(train_df, next_df), dtype=float)
    if name == "ConstantRate":
        # ConstantRate produces a flat probability; a calibrator fit on its OOF
        # is degenerate (single x value). Skip — calibrated == raw.
        return raw_prob, raw_prob
    oof = oof_predictions(dev_df, fit_predict, target_col=target_col)
    calibrator = IsotonicCalibrator.fit(oof.y_prob, oof.y_true)
    cal_prob = calibrator.transform(raw_prob)
    return raw_prob, cal_prob


def predict_next_race(
    year: int,
    round_no: int,
    target_short: str = DEFAULT_TARGET,
    *,
    decay_per_month: float | None | str = "auto",
) -> pd.DataFrame:
    if decay_per_month == "auto":
        decay_per_month = _DEFAULT_DECAY_PER_TARGET.get(target_short)
    if target_short not in TARGETS:
        raise ValueError(f"Unknown target {target_short!r}; choose from {tuple(TARGETS)}")
    target_col = TARGETS[target_short]
    race_id = f"{year}_{round_no:02d}"

    mvp = load_model_frame()
    mvp = mvp[mvp[target_col].notna()].copy()

    nxt = load_next_race()
    if (nxt["race_id"] != race_id).any():
        present = nxt["race_id"].unique().tolist()
        raise ValueError(
            f"next_race.parquet holds {present}, not {race_id}. "
            f"Run: just build-next-features {year} {round_no}"
        )
    nxt = _align_track_categories(nxt, mvp)

    dev, _ = split_dev_test(mvp)  # OOF for calibrators uses dev only (same as Phase 1.5).

    out = nxt[["race_id", "year", "round", "driver_id", "constructor_id", "grid"]].copy()
    out["q_position"] = nxt["q_position"].values
    out["has_fp2"] = nxt["has_fp2"].values

    raw_by_name: dict[str, np.ndarray] = {}
    cal_by_name: dict[str, np.ndarray] = {}
    for name, fit_predict, _ in _model_specs(
        target_col, target_short, decay_per_month=decay_per_month
    ):
        raw, cal = _predict_one_model(name, fit_predict, mvp, dev, nxt, target_col)
        raw_by_name[name] = raw
        cal_by_name[name] = cal
        out[f"prob_{name.lower()}_raw"] = raw
        out[f"prob_{name.lower()}_cal"] = cal

    # Per-driver calibration leaves teammate pairs un-coupled (Mercedes ended up
    # at 1.00 + 0.42 = 1.42 in Phase 3.6.7 on 2026 R5). Pair-norm forces each
    # constructor pair to sum to 1.0, which is what the target actually models
    # ("one of the two beats the other"). Applied per individual model BEFORE
    # the ensemble blend so the ensemble averages self-consistent pair views;
    # raw probs are left alone — they are the pre-calibration model output.
    if target_short == "teammate":
        constructor_ids = nxt["constructor_id"].to_numpy()
        for name in list(cal_by_name):
            cal_by_name[name] = pair_normalize_teammate(cal_by_name[name], constructor_ids)
            out[f"prob_{name.lower()}_cal"] = cal_by_name[name]

    # Ensemble = equal-weight mean of XGB + LGBM, mirroring final_eval._build_ensemble.
    raw_by_name["Ensemble"] = (raw_by_name["XGBoost"] + raw_by_name["LightGBM"]) / 2.0
    cal_by_name["Ensemble"] = (cal_by_name["XGBoost"] + cal_by_name["LightGBM"]) / 2.0
    out["prob_ensemble_raw"] = raw_by_name["Ensemble"]
    out["prob_ensemble_cal"] = cal_by_name["Ensemble"]

    return out.sort_values("grid").reset_index(drop=True)


def _print_table(df: pd.DataFrame, target_short: str, next_meta: pd.DataFrame) -> None:
    """Sa-evening CLI table — Ensemble cal prob front and center, XGB/LGBM next to it."""
    meta = next_meta[["driver_id", "driver_family_name", "constructor_name"]].drop_duplicates(
        subset=["driver_id"]
    )
    df = df.merge(meta, on="driver_id", how="left")
    label = "P(Podium)" if target_short == "podium" else "P(Beat Teammate)"
    print()
    print("=" * 90)
    print(f"Next-race predictions ({target_short}) — {df['race_id'].iloc[0]}")
    print("=" * 90)
    cols = f"{'grid':>4s}  {'driver':<22s}  {'team':<16s}  {label:>11s}  {'XGB':>6s}  {'LGBM':>6s}"
    print(cols)
    print("-" * 90)
    for _, r in df.iterrows():
        name = (r.get("driver_family_name") or r["driver_id"])[:22]
        team = (r.get("constructor_name") or r["constructor_id"])[:16]
        grid = int(r["grid"]) if pd.notna(r["grid"]) else 0
        print(
            f"{grid:>4d}  {name:<22s}  {team:<16s}  "
            f"{r['prob_ensemble_cal']:>10.1%}  {r['prob_xgboost_cal']:>5.1%}  "
            f"{r['prob_lightgbm_cal']:>5.1%}"
        )
    print("=" * 90)


def run(year: int, round_no: int, target_short: str) -> int:
    df = predict_next_race(year, round_no, target_short)
    path = predictions_path(target_short)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"[predict_next] saved -> {path}")
    _print_table(df, target_short, load_next_race())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Train + predict for one future race")
    r.add_argument("--year", type=int, required=True)
    r.add_argument("--round", type=int, required=True)
    r.add_argument(
        "--target",
        choices=tuple(TARGETS),
        default=DEFAULT_TARGET,
        help=f"Target to predict (default: {DEFAULT_TARGET}).",
    )
    args = p.parse_args(argv)

    if args.cmd == "run":
        return run(args.year, args.round, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
