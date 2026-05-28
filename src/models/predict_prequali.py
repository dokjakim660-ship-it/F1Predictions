"""Phase 4.2.4c pre-quali next-race inference.

For one upcoming race, trains the pre-quali stack on the full historical table
and predicts the four qualifying outcomes BEFORE qualifying, in both timing
modes (pre_weekend / post_fp2). Mirrors predict_next.py, but:

- features come from the reduced pre-quali sets (no grid/quali), so the same
  next_race_prequali.parquet row drives both modes (pre_weekend ignores FP2,
  post_fp2 uses it),
- every model is written out per (mode, target) so the app can switch models and
  draw the pre_weekend -> post_fp2 delta. LogisticRegression is the documented
  default (most robust at this N -- see `prequali select`).

Train-on-demand (no pickle): a few seconds per model, no stale-artefact risk.

Run: `python -m src.models.predict_prequali run --year Y --round N`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.calibration import IsotonicCalibrator, pair_normalize_teammate
from src.eval.walk_forward import oof_predictions, split_dev_test
from src.features.build import PRE_QUALI_FEATURE_SETS
from src.features.next_race_prequali import load_next_race_prequali
from src.models.mvp import load_model_frame
from src.models.prequali import PRE_QUALI_TARGETS, _specs
from src.utils.paths import PREDICTIONS_DIR

# Models written to the output parquet, in display order. ConstantRate is dropped
# (a flat base rate is useless in the app); RecentQualiForm stays as the visible
# baseline. Ensemble is XGB+LGBM averaged.
_OUTPUT_MODELS = ("RecentQualiForm", "LogisticRegression", "XGBoost", "LightGBM", "Ensemble")
_MODES = ("pre_weekend", "post_fp2")
DEFAULT_MODEL = "LogisticRegression"


def predictions_path(target_short: str) -> Path:
    return PREDICTIONS_DIR / f"next_race_prequali_{target_short}.parquet"


def _align_track_categories(next_df: pd.DataFrame, mvp_df: pd.DataFrame) -> pd.DataFrame:
    categories = mvp_df["track_id"].cat.categories
    next_df = next_df.copy()
    next_df["track_id"] = pd.Categorical(next_df["track_id"], categories=categories)
    return next_df


def _predict_one(
    name: str,
    fit_predict,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    next_df: pd.DataFrame,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """(raw, cal) for one model on the next-race rows. Calibrator fitted on dev OOF."""
    raw = np.asarray(fit_predict(train_df, next_df), dtype=float)
    oof = oof_predictions(dev_df, fit_predict, target_col=target_col)
    calibrator = IsotonicCalibrator.fit(oof.y_prob, oof.y_true)
    return raw, calibrator.transform(raw)


def _predict_mode_target(
    mode: str,
    target_short: str,
    mvp: pd.DataFrame,
    nxt: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """All model probs (raw + cal) for one (mode, target), keyed prob_{model}_{raw,cal}."""
    target_col = PRE_QUALI_TARGETS[target_short]
    numeric_features = PRE_QUALI_FEATURE_SETS[mode]
    is_teammate = target_short == "teammate_quali"
    constructor_ids = nxt["constructor_id"].to_numpy()

    train = mvp[mvp[target_col].notna()].copy()
    dev, _ = split_dev_test(train)

    raw_by: dict[str, np.ndarray] = {}
    cal_by: dict[str, np.ndarray] = {}
    for name, fit_predict in _specs(target_col, numeric_features):
        if name == "ConstantRate":
            continue
        raw, cal = _predict_one(name, fit_predict, train, dev, nxt, target_col)
        raw_by[name] = raw
        cal_by[name] = cal

    raw_by["Ensemble"] = (raw_by["XGBoost"] + raw_by["LightGBM"]) / 2.0
    cal_by["Ensemble"] = (cal_by["XGBoost"] + cal_by["LightGBM"]) / 2.0

    if is_teammate:
        for name in list(cal_by):
            cal_by[name] = pair_normalize_teammate(cal_by[name], constructor_ids)

    out: dict[str, np.ndarray] = {}
    for name in _OUTPUT_MODELS:
        key = name.lower()
        out[f"prob_{mode}_{key}_raw"] = raw_by[name]
        out[f"prob_{mode}_{key}_cal"] = cal_by[name]
    return out


def predict_prequali_target(year: int, round_no: int, target_short: str) -> pd.DataFrame:
    race_id = f"{year}_{round_no:02d}"
    mvp = load_model_frame()
    nxt = load_next_race_prequali()
    if (nxt["race_id"] != race_id).any():
        present = nxt["race_id"].unique().tolist()
        raise ValueError(
            f"next_race_prequali.parquet holds {present}, not {race_id}. "
            f"Run: just build-prequali-features {year} {round_no}"
        )
    nxt = _align_track_categories(nxt, mvp)

    out = nxt[["race_id", "driver_id", "constructor_id"]].copy()
    out["driver_family_name"] = nxt["driver_family_name"].to_numpy()
    out["constructor_name"] = nxt["constructor_name"].to_numpy()
    out["has_fp2"] = nxt["has_fp2"].to_numpy()
    out["recent_quali_pos"] = nxt["driver_form_quali_pos_l5"].to_numpy()

    for mode in _MODES:
        for col, vals in _predict_mode_target(mode, target_short, mvp, nxt).items():
            out[col] = vals
    return out.reset_index(drop=True)


def _print_table(df: pd.DataFrame, target_short: str) -> None:
    """CLI view: default model (LogReg) cal probs, both modes + the FP2 delta."""
    key = DEFAULT_MODEL.lower()
    pw = f"prob_pre_weekend_{key}_cal"
    pf = f"prob_post_fp2_{key}_cal"
    show = df[["driver_family_name", "constructor_name", "recent_quali_pos", pw, pf]].copy()
    show["delta"] = show[pf] - show[pw]
    show = show.sort_values(pf, ascending=False)
    rid = df["race_id"].iloc[0]
    print()
    print("=" * 84)
    print(f"Pre-quali predictions ({target_short}) -- {rid}  [model: {DEFAULT_MODEL}]")
    print("=" * 84)
    print(
        f"{'driver':<20s}  {'team':<15s}  {'recentQ':>7s}  "
        f"{'preWE':>7s}  {'postFP2':>7s}  {'delta':>7s}"
    )
    print("-" * 84)
    for _, r in show.iterrows():
        name = (r["driver_family_name"] or "")[:20]
        team = (r["constructor_name"] or "")[:15]
        rq = f"{r['recent_quali_pos']:.1f}" if pd.notna(r["recent_quali_pos"]) else "--"
        print(
            f"{name:<20s}  {team:<15s}  {rq:>7s}  {r[pw]:>6.1%}  {r[pf]:>6.1%}  {r['delta']:>+6.1%}"
        )
    print("=" * 84)


def run(year: int, round_no: int, target_short: str) -> int:
    targets = list(PRE_QUALI_TARGETS) if target_short == "all" else [target_short]
    for t in targets:
        df = predict_prequali_target(year, round_no, t)
        path = predictions_path(t)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"[predict_prequali] saved -> {path}")
        _print_table(df, t)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Train + predict pre-quali for one future race.")
    r.add_argument("--year", type=int, required=True)
    r.add_argument("--round", type=int, required=True)
    r.add_argument(
        "--target",
        choices=(*PRE_QUALI_TARGETS, "all"),
        default="all",
        help="Quali target (default: all).",
    )
    args = p.parse_args(argv)

    if args.cmd == "run":
        return run(args.year, args.round, args.target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
