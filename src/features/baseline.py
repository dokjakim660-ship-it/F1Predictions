"""Baseline feature builder: results.parquet -> data/features/baseline.parquet.

One row per (race_id, driver_id). Columns are kept deliberately minimal -- this
is the "naive comparison" feature set the MVP model will need to beat:

- grid (raw + log + flags) — already strongly predictive on its own
- driver rolling form over the last 5 classified races (avg finish, DNF rate)
- constructor rolling points over the last 5 races (both cars combined)
- driver age at race date

All rolling features are computed with .shift(1) within the per-driver /
per-constructor group, so race N's feature value reflects only races strictly
before N. This is the no-lookahead invariant tested by tests/test_no_leakage.py.

Run: `python -m src.features.baseline build`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.process.jolpica import RESULTS_PARQUET
from src.utils.paths import FEATURES_DIR

BASELINE_PARQUET = FEATURES_DIR / "baseline.parquet"

_ROLLING_WINDOW = 5
# Replace DNFs with a worst-case position for the rolling-avg-finish signal so
# DNF-heavy drivers don't look "fast" just because their finishes are filtered.
_DNF_POSITION_PROXY = 20


def _add_driver_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["driver_id", "race_date"]).reset_index(drop=True)
    pos_proxy = df["finish_position"].where(~df["dnf"], _DNF_POSITION_PROXY).astype(float)
    df["_pos_proxy"] = pos_proxy

    grp = df.groupby("driver_id", sort=False)
    df["driver_rolling_avg_pos_l5"] = grp["_pos_proxy"].transform(
        lambda x: x.rolling(window=_ROLLING_WINDOW, min_periods=1).mean().shift(1)
    )
    df["driver_rolling_dnf_rate_l5"] = grp["dnf"].transform(
        lambda x: x.astype(float).rolling(window=_ROLLING_WINDOW, min_periods=1).mean().shift(1)
    )
    df["driver_career_race_count"] = grp.cumcount()  # 0 for first race, monotone.
    df = df.drop(columns=["_pos_proxy"])
    return df


def _add_constructor_rolling(df: pd.DataFrame) -> pd.DataFrame:
    # Aggregate points per (constructor, race) first -- otherwise rolling double-counts
    # the two cars per team within the same race.
    con_per_race = (
        df.groupby(["constructor_id", "race_id", "race_date"], as_index=False)["points"]
        .sum()
        .sort_values(["constructor_id", "race_date"])
        .reset_index(drop=True)
    )
    con_per_race["con_rolling_pts_l5"] = con_per_race.groupby("constructor_id")["points"].transform(
        lambda x: x.rolling(window=_ROLLING_WINDOW, min_periods=1).mean().shift(1)
    )
    return df.merge(
        con_per_race[["constructor_id", "race_id", "con_rolling_pts_l5"]],
        on=["constructor_id", "race_id"],
        how="left",
    )


def _add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    # grid==0 means pit-lane start; treat as worst grid (back of pack proxy = 21).
    grid_eff = df["grid"].where(df["grid"] > 0, 21).astype(float)
    df["grid_effective"] = grid_eff
    df["grid_log"] = np.log(grid_eff)
    df["is_pole"] = (df["grid"] == 1).astype(int)
    df["is_top3_grid"] = ((df["grid"] >= 1) & (df["grid"] <= 3)).astype(int)
    return df


def _add_driver_age(df: pd.DataFrame) -> pd.DataFrame:
    race_d = pd.to_datetime(df["race_date"])
    dob = pd.to_datetime(df["driver_dob"], errors="coerce")
    df["driver_age_years"] = (race_d - dob).dt.days / 365.25
    return df


def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    df["target_podium"] = (
        df["finish_position"].between(1, 3, inclusive="both") & (~df["dnf"])
    ).astype(int)
    return df


def build_baseline_features(
    results_path: Path = RESULTS_PARQUET,
) -> pd.DataFrame:
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} missing. Run: python -m src.process.jolpica build first."
        )
    res = pd.read_parquet(results_path)
    df = _add_target(res)
    df = _add_grid_features(df)
    df = _add_driver_age(df)
    df = _add_driver_rolling(df)
    df = _add_constructor_rolling(df)
    return df.sort_values(["year", "round", "finish_position"]).reset_index(drop=True)


def save_features(df: pd.DataFrame, path: Path = BASELINE_PARQUET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_features(path: Path = BASELINE_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m src.features.baseline build")
    return pd.read_parquet(path)


def _print_summary(df: pd.DataFrame) -> None:
    print(f"[features.baseline] {len(df)} rows, {df.shape[1]} columns")
    print(f"[features.baseline] podium rate (target): {df['target_podium'].mean():.3f}")
    print(
        f"[features.baseline] NaN counts in feature cols: "
        f"driver_rolling_avg_pos_l5={df['driver_rolling_avg_pos_l5'].isna().sum()}, "
        f"con_rolling_pts_l5={df['con_rolling_pts_l5'].isna().sum()}"
    )
    by_year = df.groupby("year").size()
    for year, n in by_year.items():
        print(f"  {year}: {n} rows")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="results.parquet -> data/features/baseline.parquet")
    sub.add_parser("show", help="Print summary of the current baseline features parquet")
    args = p.parse_args(argv)

    if args.cmd == "build":
        df = build_baseline_features()
        out = save_features(df)
        print(f"[features.baseline] saved -> {out}")
        _print_summary(df)
        return 0
    if args.cmd == "show":
        df = load_features()
        _print_summary(df)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
