"""Phase 5.2 DNF next-race inference.

Trains the DNF stack on the full historical table and predicts P(driver does
not finish) for one upcoming race in every timing mode:

- race:        full post-quali feature set, predicted on next_race.parquet
               (Saturday evening, real grid known).
- post_fp2 /   reduced pre-quali feature sets, predicted on
  pre_weekend:  next_race_prequali.parquet (no grid/quali).

Mirrors predict_next.py: train-on-demand (no pickle), isotonic calibrator fitted
on leak-free dev OOF predictions, one parquet out.

IMPORTANT (Phase 5.2 holdout verdict): no DNF model beats the TeamReliability
baseline -- retirements are near-unpredictable beyond a team's recent reliability
rate at this N. The deployed default is therefore TeamReliability (it at least
varies per team, unlike the flat ConstantRate, and has the best calibration); the
full stack is written out so the app can show it, but treat any single-model edge
as noise.

Run: `python -m src.models.predict_dnf run --year Y --round N`
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from src.eval.calibration import IsotonicCalibrator
from src.eval.walk_forward import oof_predictions, split_dev_test
from src.features.build import TARGET_DNF
from src.features.next_race import load_next_race
from src.features.next_race_prequali import load_next_race_prequali
from src.models.dnf import ALL_MODES, DNF_FEATURE_SETS, _specs
from src.models.mvp import load_model_frame
from src.utils.paths import PREDICTIONS_DIR

# Dev-selected honest default (holdout verdict: nothing beats it). Shown first in
# the app; the rest of the stack is written out for comparison only.
DEFAULT_MODEL = "teamreliability"
DEFAULT_MODE = "race"

PREDICTIONS_PATH = PREDICTIONS_DIR / "next_race_dnf.parquet"

# Reads next_race_prequali.parquet for the pre-quali modes, next_race.parquet for
# the post-quali race mode -- same split the ranking inference uses.
_PREQUALI_MODES = ("pre_weekend", "post_fp2")


def _align_track_categories(next_df: pd.DataFrame, mvp_df: pd.DataFrame) -> pd.DataFrame:
    categories = mvp_df["track_id"].cat.categories
    next_df = next_df.copy()
    next_df["track_id"] = pd.Categorical(next_df["track_id"], categories=categories)
    return next_df


def _predict_one(
    name: str,
    fit_predict,
    train: pd.DataFrame,
    dev: pd.DataFrame,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """(raw, cal) for one model, aligned to `frame` rows. Calibrator fitted on
    dev OOF -- same construction as predict_next."""
    raw = np.asarray(fit_predict(train, frame), dtype=float)
    if name == "ConstantRate":
        return raw, raw
    oof = oof_predictions(dev, fit_predict, target_col=TARGET_DNF)
    calibrator = IsotonicCalibrator.fit(oof.y_prob, oof.y_true)
    return raw, calibrator.transform(raw)


def predict_dnf(year: int, round_no: int) -> pd.DataFrame:
    race_id = f"{year}_{round_no:02d}"
    mvp = load_model_frame()
    mvp = mvp[mvp[TARGET_DNF].notna()].copy()
    dev, _ = split_dev_test(mvp)

    race_frame = _align_track_categories(load_next_race(), mvp)
    if (race_frame["race_id"] != race_id).any():
        present = race_frame["race_id"].unique().tolist()
        raise ValueError(
            f"next_race.parquet holds {present}, not {race_id}. "
            f"Run: just build-next-features {year} {round_no}"
        )
    prequali_frame = _align_track_categories(load_next_race_prequali(), mvp)

    # Base identity table, anchored on the post-quali race frame (it carries grid).
    out = race_frame[["race_id", "driver_id", "constructor_id", "grid"]].copy()
    out["driver_family_name"] = race_frame["driver_family_name"].to_numpy()
    out["constructor_name"] = race_frame["constructor_name"].to_numpy()

    for mode in ALL_MODES:
        feats = DNF_FEATURE_SETS[mode]
        frame = prequali_frame if mode in _PREQUALI_MODES else race_frame
        raw_by: dict[str, np.ndarray] = {}
        cal_by: dict[str, np.ndarray] = {}
        for name, fit_predict in _specs(feats):
            raw, cal = _predict_one(name, fit_predict, mvp, dev, frame)
            raw_by[name] = raw
            cal_by[name] = cal
        raw_by["Ensemble"] = (raw_by["XGBoost"] + raw_by["LightGBM"]) / 2.0
        cal_by["Ensemble"] = (cal_by["XGBoost"] + cal_by["LightGBM"]) / 2.0

        mode_df = pd.DataFrame({"driver_id": frame["driver_id"].to_numpy()})
        for name in raw_by:
            key = name.lower()
            mode_df[f"prob_{mode}_{key}_raw"] = raw_by[name]
            mode_df[f"prob_{mode}_{key}_cal"] = cal_by[name]
        # Merge by driver_id so the pre-quali entry list (possibly a different row
        # order / a stray driver) lines up with the race frame's drivers.
        out = out.merge(mode_df, on="driver_id", how="left")

    return out.sort_values("grid").reset_index(drop=True)


def _print_table(df: pd.DataFrame) -> None:
    col = f"prob_{DEFAULT_MODE}_{DEFAULT_MODEL}_cal"
    show = df.sort_values(col, ascending=False).reset_index(drop=True)
    print()
    print("=" * 78)
    print(
        f"Next-race P(DNF) -- {df['race_id'].iloc[0]}  "
        f"[model: {DEFAULT_MODEL}, mode: {DEFAULT_MODE}]"
    )
    print("=" * 78)
    print(f"{'grid':>4s}  {'driver':<22s}  {'team':<18s}  {'P(DNF)':>8s}")
    print("-" * 78)
    for _, r in show.iterrows():
        name = (r["driver_family_name"] or r["driver_id"])[:22]
        team = (r["constructor_name"] or r["constructor_id"])[:18]
        grid = int(r["grid"]) if pd.notna(r["grid"]) else 0
        print(f"{grid:>4d}  {name:<22s}  {team:<18s}  {r[col]:>7.1%}")
    print("=" * 78)
    print("Reminder: DNF does NOT beat the team-reliability baseline -- read as a")
    print("calibrated reliability estimate, not a sharp edge.")


def run(year: int, round_no: int) -> int:
    df = predict_dnf(year, round_no)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"[predict_dnf] saved -> {PREDICTIONS_PATH}")
    _print_table(df)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Train + predict P(DNF) for one future race.")
    r.add_argument("--year", type=int, required=True)
    r.add_argument("--round", type=int, required=True)
    args = p.parse_args(argv)

    if args.cmd == "run":
        return run(args.year, args.round)
    return 0


if __name__ == "__main__":
    sys.exit(main())
