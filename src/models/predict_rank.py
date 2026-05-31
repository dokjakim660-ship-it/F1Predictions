"""Phase 5 ranking next-race inference.

For one upcoming race, trains the ranking stack on the full historical table and
predicts the complete order for two tasks:

- quali: each driver's qualifying position BEFORE qualifying, in both timing
  modes (pre_weekend / post_fp2), from next_race_prequali.parquet.
- race:  each driver's race finishing position on Saturday evening (post-quali),
  from next_race.parquet.

Mirrors predict_prequali.py, minus calibration (the output is an order, not a
probability). Every model is written out per (mode) so the app can switch model
and compare position-regression against the LambdaMART ranker. Each model emits
a raw score (`score_*`, the regressors' "expected place") and the argsorted
integer placement (`rank_*`, 1..N within the race).

Default model = Ridge (dev-selected for race + pre_weekend, most interpretable;
see `rank select`). Train-on-demand (no pickle), a few seconds per model.

Run: `python -m src.models.predict_rank run --year Y --round N`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.rank_metrics import to_race_ranks
from src.features.next_race import load_next_race
from src.features.next_race_prequali import load_next_race_prequali
from src.models.mvp import load_model_frame
from src.models.rank import RANK_TASKS, _display_order, _ensemble_score, _specs
from src.utils.paths import PREDICTIONS_DIR

DEFAULT_MODEL = "Ridge"


def predictions_path(task: str) -> Path:
    return PREDICTIONS_DIR / f"next_race_rank_{task}.parquet"


def _align_track_categories(next_df: pd.DataFrame, mvp_df: pd.DataFrame) -> pd.DataFrame:
    categories = mvp_df["track_id"].cat.categories
    next_df = next_df.copy()
    next_df["track_id"] = pd.Categorical(next_df["track_id"], categories=categories)
    return next_df


def _load_next_frame(task: str, year: int, round_no: int) -> pd.DataFrame:
    """The task-specific next-race feature frame, with its race_id validated."""
    race_id = f"{year}_{round_no:02d}"
    if task == "quali":
        nxt = load_next_race_prequali()
        rebuild = f"just build-prequali-features {year} {round_no}"
    else:
        nxt = load_next_race()
        rebuild = f"just build-next-features {year} {round_no}"
    if (nxt["race_id"] != race_id).any():
        present = nxt["race_id"].unique().tolist()
        raise ValueError(f"next-race frame holds {present}, not {race_id}. Run: {rebuild}")
    return nxt


def _base_columns(task: str, nxt: pd.DataFrame) -> pd.DataFrame:
    """Identity + reference columns the app joins onto the predictions."""
    out = nxt[["race_id", "driver_id", "constructor_id"]].copy()
    out["driver_family_name"] = nxt["driver_family_name"].to_numpy()
    out["constructor_name"] = nxt["constructor_name"].to_numpy()
    if task == "quali":
        out["has_fp2"] = nxt["has_fp2"].to_numpy()
        out["recent_quali_pos"] = nxt["driver_form_quali_pos_l5"].to_numpy()
    else:
        out["grid"] = nxt["grid"].to_numpy()
    return out


def predict_rank_task(year: int, round_no: int, task: str) -> pd.DataFrame:
    cfg = RANK_TASKS[task]
    mvp = load_model_frame()
    nxt = _align_track_categories(_load_next_frame(task, year, round_no), mvp)
    train = mvp[mvp[cfg.target].notna()].copy()
    race_ids = nxt["race_id"].to_numpy()

    out = _base_columns(task, nxt)
    for mode, feats in cfg.modes.items():
        score_by: dict[str, np.ndarray] = {}
        asc_by: dict[str, bool] = {}
        for name, fit_predict, asc in _specs(task, feats):
            score_by[name] = np.asarray(fit_predict(train, nxt), dtype=float)
            asc_by[name] = asc
        score_by["RegEnsemble"] = _ensemble_score(score_by)
        asc_by["RegEnsemble"] = True

        for name in _display_order(task):
            key = name.lower()
            score = score_by[name]
            out[f"score_{mode}_{key}"] = score
            out[f"rank_{mode}_{key}"] = to_race_ranks(race_ids, score, asc_by[name]).astype(int)
    return out.reset_index(drop=True)


def _default_mode(task: str) -> str:
    """The mode the CLI table + app land on by default."""
    return "post_fp2" if task == "quali" else "race"


def _print_table(df: pd.DataFrame, task: str) -> None:
    mode = _default_mode(task)
    key = DEFAULT_MODEL.lower()
    rank_col = f"rank_{mode}_{key}"
    score_col = f"score_{mode}_{key}"
    ref_col = "grid" if task == "race" else "recent_quali_pos"
    show = df.sort_values(rank_col).reset_index(drop=True)
    rid = df["race_id"].iloc[0]

    print()
    print("=" * 76)
    print(f"Predicted {task} order -- {rid}  [model: {DEFAULT_MODEL}, mode: {mode}]")
    print("=" * 76)
    print(f"{'#':>2s}  {'driver':<20s}  {'team':<16s}  {ref_col:>10s}  {'exp_pos':>7s}")
    print("-" * 76)
    for _, r in show.iterrows():
        name = (r["driver_family_name"] or r["driver_id"])[:20]
        team = (r["constructor_name"] or r["constructor_id"])[:16]
        ref = r[ref_col]
        ref_s = f"{ref:.1f}" if pd.notna(ref) else "--"
        print(
            f"{int(r[rank_col]):>2d}  {name:<20s}  {team:<16s}  {ref_s:>10s}  {r[score_col]:>7.2f}"
        )
    print("=" * 76)


def run(year: int, round_no: int, task_arg: str) -> int:
    tasks = list(RANK_TASKS) if task_arg == "all" else [task_arg]
    for task in tasks:
        df = predict_rank_task(year, round_no, task)
        path = predictions_path(task)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        print(f"[predict_rank] saved -> {path}")
        _print_table(df, task)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Train + predict the full order for one future race.")
    r.add_argument("--year", type=int, required=True)
    r.add_argument("--round", type=int, required=True)
    r.add_argument(
        "--task",
        choices=(*RANK_TASKS, "all"),
        default="all",
        help="Ranking task (default: all).",
    )
    args = p.parse_args(argv)

    if args.cmd == "run":
        return run(args.year, args.round, args.task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
