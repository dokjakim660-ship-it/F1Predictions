"""Phase 3.2 next-race feature builder.

Builds a model-ready feature row per driver for a SINGLE future race
(`data/features/next_race.parquet`). Reuses the existing L3 pipeline by
synthesizing a pseudo-results row per driver from the FastF1 qualifying
snapshot and running `compute_features` over the combined history.

Contract:
- Race-outcome columns (finish_position, dnf, laps_completed, points) and
  both targets (target_podium, target_beat_teammate) are NaN on the next-race
  rows. Inference must never look at them.
- Every other feature is the same value `compute_features` would emit if the
  race were already complete — the `.shift(1)` guard inside the rolling
  features means race N never feeds into its own features, so adding race N
  as a pseudo-row at the end of history does not leak its own data.
- Quali features come from FastF1 Q for the target race. FP2 features fold to
  NaN/0 on sprint weekends (`has_fp2=0`) without breaking the pipeline.

Run: `python -m src.features.next_race build --year Y --round N`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build import (
    TARGET_PODIUM,
    TARGET_TEAMMATE,
    compute_features,
)
from src.process.fastf1 import load_sessions
from src.process.jolpica import load_results
from src.process.openmeteo import load_weather
from src.utils.paths import FEATURES_DIR
from src.utils.race_inventory import load_inventory
from src.utils.tracks import load_tracks

NEXT_RACE_PARQUET = FEATURES_DIR / "next_race.parquet"

# Columns the historical results spine carries that the pseudo-row must also
# provide (even if NaN) so compute_features sees a uniform schema.
_RACE_OUTCOME_NAN_COLS = [
    "finish_position",
    "position_text",
    "points",
    "laps_completed",
    "status",
    "finish_time_ms",
    "fastest_lap_ms",
    "fastest_lap_rank",
    "fastest_lap_avg_speed_kph",
]


def _latest_driver_meta(results: pd.DataFrame) -> pd.DataFrame:
    """driver_id -> last-known name/code/nationality, used to fill the pseudo-row."""
    last = results.sort_values("race_date").groupby("driver_id", as_index=False).tail(1)
    return last[
        [
            "driver_id",
            "driver_code",
            "driver_given_name",
            "driver_family_name",
            "driver_dob",
            "driver_nationality",
        ]
    ].drop_duplicates(subset=["driver_id"])


def _latest_constructor_name(results: pd.DataFrame) -> pd.Series:
    """constructor_id -> last-known constructor_name from historical results."""
    last = results.sort_values("race_date").groupby("constructor_id", as_index=True).tail(1)
    return last.set_index("constructor_id")["constructor_name"]


def synthesize_pseudo_results(
    year: int,
    round_no: int,
    *,
    sessions: pd.DataFrame,
    inv: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per driver for the target race, schema-compatible with results.parquet.

    Race-outcome columns are NaN (race hasn't happened). driver_dob / driver_*
    metadata is sourced from the most recent historical row for each driver;
    rookies with no history end up with NaN dob (handled downstream by
    `_add_driver_age` returning NaN — XGB/LGBM tolerate it).
    """
    race_id = f"{year}_{round_no:02d}"
    inv_row = inv[(inv["year"] == year) & (inv["round"] == round_no)]
    if inv_row.empty:
        raise ValueError(
            f"{race_id} not in race_inventory.parquet. "
            "Run: just ingest-schedule (or just ingest-next YEAR ROUND)."
        )
    race_date = inv_row.iloc[0]["race_date"]

    q = sessions[(sessions["year"] == year) & (sessions["round"] == round_no)]
    if q.empty or not q["has_qualifying"].any() or q["q_position"].isna().all():
        raise ValueError(
            f"No qualifying data for {race_id} in sessions.parquet. "
            "Run `just ingest-next YEAR ROUND` after qualifying is over, "
            "then `uv run python -m src.process.fastf1 build`."
        )

    # Sort by quali position so the grid mapping is deterministic.
    q = q.sort_values("q_position").reset_index(drop=True)

    pseudo = pd.DataFrame(
        {
            "race_id": race_id,
            "year": year,
            "round": round_no,
            "race_date": race_date,
            "driver_id": q["driver_id"].astype(str),
            "constructor_id": q["team_id"].astype(str),
            # Grid == quali position until the predict-time post-processor wires
            # in penalty awareness. Pit-lane starts (grid=0) are handled by
            # _add_grid_features downstream, so leaving q_position here is safe.
            "grid": q["q_position"].astype("Int64"),
            "dnf": False,  # required dtype: build pipeline treats NaN as missing
        }
    )

    # Fill metadata from most-recent historical row per driver.
    meta = _latest_driver_meta(results)
    pseudo = pseudo.merge(meta, on="driver_id", how="left")

    # Fill constructor_name from most-recent historical row per constructor.
    cname = _latest_constructor_name(results)
    pseudo["constructor_name"] = pseudo["constructor_id"].map(cname).fillna("")

    # Outcome columns: explicit NaN so compute_features sees the right dtype.
    for col in _RACE_OUTCOME_NAN_COLS:
        pseudo[col] = np.nan

    # Schema alignment: column order matches results.parquet so the concat below
    # stays clean. Extra columns on results.parquet (none expected) survive via
    # outer concat; reindex makes the intent explicit.
    return pseudo.reindex(columns=results.columns, fill_value=np.nan)


def build_next_race_features(year: int, round_no: int) -> pd.DataFrame:
    """End-to-end: feed the next-race pseudo-row through compute_features.

    Drops any pre-existing results row for the target race (defensive — if the
    race ran since the last L2 build, we still want our pre-race prediction
    based on quali, not the actual result).
    """
    results = load_results()
    sessions = load_sessions()
    inv = load_inventory()
    tracks = load_tracks()
    weather = load_weather()

    race_id = f"{year}_{round_no:02d}"
    results = results[results["race_id"] != race_id].copy()

    pseudo = synthesize_pseudo_results(year, round_no, sessions=sessions, inv=inv, results=results)
    combined = pd.concat([results, pseudo], ignore_index=True)

    df = compute_features(combined, sessions, inv, tracks, weather)
    next_rows = df[df["race_id"] == race_id].copy()

    # Force targets + race-outcome meta to NaN. compute_features computes them
    # from finish_position which is NaN here, but _add_targets' fallbacks
    # (`.between` on NaN, the teammate score arithmetic) produce 0 / spurious
    # values for those rows. Hardwire NaN so inference code can rely on isna().
    next_rows[TARGET_PODIUM] = np.nan
    next_rows[TARGET_TEAMMATE] = np.nan
    next_rows["finish_position"] = np.nan
    next_rows["dnf"] = np.nan

    return next_rows.sort_values(["race_id", "grid", "driver_id"]).reset_index(drop=True)


def save_next_race(df: pd.DataFrame, path: Path = NEXT_RACE_PARQUET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_next_race(path: Path = NEXT_RACE_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: just build-next-features YEAR ROUND")
    return pd.read_parquet(path)


def _print_summary(df: pd.DataFrame) -> None:
    print(f"[features.next_race] {len(df)} drivers for {df['race_id'].iloc[0]}")
    print(
        f"[features.next_race] has_fp2: {int(df['has_fp2'].iloc[0])}  "
        f"q-coverage: {df['q_position'].notna().mean():.0%}  "
        f"FP2 long-run gap defined: {df['fp2_long_run_gap_ms'].notna().mean():.0%}"
    )
    cols = ["grid", "driver_id", "constructor_id", "q_position", "q_gap_to_pole_ms"]
    print(df[cols].head(10).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build", help="Build next-race feature parquet")
    p_build.add_argument("--year", type=int, required=True)
    p_build.add_argument("--round", type=int, required=True)
    p_show = sub.add_parser("show", help="Print summary of the current next_race.parquet")
    _ = p_show

    args = p.parse_args(argv)

    if args.cmd == "build":
        df = build_next_race_features(args.year, args.round)
        out = save_next_race(df)
        print(f"[features.next_race] saved -> {out}")
        _print_summary(df)
        return 0
    if args.cmd == "show":
        _print_summary(load_next_race())
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
