"""Phase 4.2.4b pre-quali next-race feature builder.

Sibling of next_race.py, but for predictions made BEFORE qualifying. The
post-quali builder keys the driver list off the qualifying classification;
pre-quali there is no quali yet, so the lineup comes from whatever FastF1
sessions already exist for the weekend (FP2 on a Friday night), falling back to
the most recent completed race's lineup for a true pre-FP1 run.

Contract (stricter than next_race.py): in addition to race-outcome columns,
every QUALI-derived feature is forced to NaN -- grid, q_position, quali gaps,
quali-beat-teammate, and the sprint columns. Those are exactly the columns the
post_fp2 feature set drops from the full set, so the nulled set is derived from
the feature-set definitions instead of being re-listed here. FP2 long-run pace
is KEPT (legal post-FP2), so a single parquet feeds both the pre_weekend model
(ignores FP2) and the post_fp2 model (uses it).

Run: `python -m src.features.next_race_prequali build --year Y --round N`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_POST_FP2,
    QUALI_TARGETS,
    RANK_TARGETS,
    TARGET_PODIUM,
    TARGET_TEAMMATE,
    compute_features,
)
from src.features.next_race import (
    _RACE_OUTCOME_NAN_COLS,
    _latest_constructor_name,
    _latest_driver_meta,
)
from src.process.fastf1 import load_sessions
from src.process.jolpica import load_results
from src.process.openmeteo import load_weather
from src.utils.paths import FEATURES_DIR
from src.utils.race_inventory import load_inventory
from src.utils.tracks import load_tracks

NEXT_RACE_PREQUALI_PARQUET = FEATURES_DIR / "next_race_prequali.parquet"

# Quali/grid/sprint feature columns to null out for the pre-quali contract --
# exactly the columns the post_fp2 feature set drops from the full set. Derived,
# not re-listed, so it can never drift from the feature-set definitions.
_QUALI_FEATURE_COLS = [c for c in FEATURE_COLUMNS if c not in set(FEATURE_COLUMNS_POST_FP2)]


def _target_race_lineup(
    year: int, round_no: int, sessions: pd.DataFrame, results: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """driver_id + constructor_id for the target race.

    Prefer the FastF1 session entry list (FP2/Q/S already ingested for this
    weekend); fall back to the most recent completed race's lineup for a true
    pre-FP1 run where no session exists yet. Returns (lineup, source_tag).
    """
    sess = sessions[(sessions["year"] == year) & (sessions["round"] == round_no)]
    sess = sess[sess["driver_id"].notna() & sess["team_id"].notna()]
    if not sess.empty:
        lineup = (
            sess[["driver_id", "team_id"]]
            .drop_duplicates(subset=["driver_id"])
            .rename(columns={"team_id": "constructor_id"})
        )
        return lineup.reset_index(drop=True), "session"

    last_rid = results.sort_values("race_date")["race_id"].iloc[-1]
    last = results[results["race_id"] == last_rid]
    lineup = last[["driver_id", "constructor_id"]].drop_duplicates(subset=["driver_id"])
    return lineup.reset_index(drop=True), "last_race"


def synthesize_prequali_results(
    year: int,
    round_no: int,
    *,
    sessions: pd.DataFrame,
    inv: pd.DataFrame,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """One results-schema row per driver for the target race, with NO quali.

    grid is NaN (no qualifying yet); the pipeline tolerates it and the
    quali/grid features are nulled downstream anyway. Driver/constructor
    metadata is sourced from the most recent historical row per id.
    """
    race_id = f"{year}_{round_no:02d}"
    inv_row = inv[(inv["year"] == year) & (inv["round"] == round_no)]
    if inv_row.empty:
        raise ValueError(
            f"{race_id} not in race_inventory.parquet. Run: just ingest-next YEAR ROUND."
        )
    race_date = inv_row.iloc[0]["race_date"]

    lineup, source = _target_race_lineup(year, round_no, sessions, results)
    if lineup.empty:
        raise ValueError(f"No driver lineup found for {race_id} (no sessions, no prior races).")
    print(f"[next_race_prequali] lineup source: {source} ({len(lineup)} drivers)")

    pseudo = pd.DataFrame(
        {
            "race_id": race_id,
            "year": year,
            "round": round_no,
            "race_date": race_date,
            "driver_id": lineup["driver_id"].astype(str),
            "constructor_id": lineup["constructor_id"].astype(str),
            # float NaN (not Int64 NA): _add_grid_features does (grid == 1).astype(int),
            # which can't cast a nullable NA but folds float NaN to False cleanly.
            # grid features are nulled downstream anyway.
            "grid": np.nan,  # no quali yet
            "dnf": False,  # build pipeline treats NaN as missing; race hasn't run
        }
    )

    meta = _latest_driver_meta(results)
    pseudo = pseudo.merge(meta, on="driver_id", how="left")
    cname = _latest_constructor_name(results)
    pseudo["constructor_name"] = pseudo["constructor_id"].map(cname).fillna("")

    for col in _RACE_OUTCOME_NAN_COLS:
        pseudo[col] = np.nan

    return pseudo.reindex(columns=results.columns, fill_value=np.nan)


def build_prequali_features(year: int, round_no: int) -> pd.DataFrame:
    """End-to-end pre-quali feature row(s): synthesize -> compute -> null quali.

    Drops any pre-existing results row for the target race (defensive: a
    pre-quali prediction must not see the race result even if it has since run).
    """
    results = load_results()
    sessions = load_sessions()
    inv = load_inventory()
    tracks = load_tracks()
    weather = load_weather()

    race_id = f"{year}_{round_no:02d}"
    results = results[results["race_id"] != race_id].copy()

    pseudo = synthesize_prequali_results(
        year, round_no, sessions=sessions, inv=inv, results=results
    )
    combined = pd.concat([results, pseudo], ignore_index=True)

    df = compute_features(combined, sessions, inv, tracks, weather)
    rows = df[df["race_id"] == race_id].copy()

    # Pre-quali contract: null every quali/grid/sprint feature + all targets +
    # race-outcome meta. FP2 + historical/track/weather features survive.
    for col in _QUALI_FEATURE_COLS:
        rows[col] = np.nan
    for col in (*QUALI_TARGETS, *RANK_TARGETS, TARGET_PODIUM, TARGET_TEAMMATE):
        rows[col] = np.nan
    rows["finish_position"] = np.nan
    rows["dnf"] = np.nan
    rows["grid"] = np.nan

    return rows.sort_values("driver_id").reset_index(drop=True)


def save_next_race_prequali(df: pd.DataFrame, path: Path = NEXT_RACE_PREQUALI_PARQUET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_next_race_prequali(path: Path = NEXT_RACE_PREQUALI_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: just build-prequali-features YEAR ROUND")
    return pd.read_parquet(path)


def _print_summary(df: pd.DataFrame) -> None:
    has_fp2 = int(df["has_fp2"].fillna(0).max())
    fp2_cov = df["fp2_long_run_gap_ms"].notna().mean()
    form_cov = df["driver_form_quali_pos_l5"].notna().mean()
    print(f"[next_race_prequali] {len(df)} drivers for {df['race_id'].iloc[0]}")
    print(
        f"[next_race_prequali] has_fp2: {has_fp2}  FP2 long-run gap defined: {fp2_cov:.0%}  "
        f"recent quali form defined: {form_cov:.0%}"
    )
    cols = ["driver_id", "constructor_id", "driver_form_quali_pos_l5", "fp2_long_run_gap_ms"]
    print(df[cols].head(10).to_string(index=False))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Build the pre-quali next-race feature parquet")
    b.add_argument("--year", type=int, required=True)
    b.add_argument("--round", type=int, required=True)
    sub.add_parser("show", help="Print summary of the current next_race_prequali.parquet")

    args = p.parse_args(argv)
    if args.cmd == "build":
        df = build_prequali_features(args.year, args.round)
        out = save_next_race_prequali(df)
        print(f"[next_race_prequali] saved -> {out}")
        _print_summary(df)
        return 0
    if args.cmd == "show":
        _print_summary(load_next_race_prequali())
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
