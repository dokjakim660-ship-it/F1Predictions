"""Lookahead-leakage guards.

Two layers:

- L2 (processed/) -- this file, active. The processed tables must not contain
  rows for races that have not yet happened. This is a cheap perimeter check
  before any feature engineering touches the data; if a "future" race already
  has a results row, something upstream (ingest, schedule) is wrong.

- L3 (features/) -- mvp.parquet guards for the three known traps from
  project_feature_decisions.md: a rolling feature built without .shift(1),
  actual race weather used in place of a forecast, and team history keyed on
  the driver instead of the constructor.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.features.build import _DNF_POSITION_PROXY, FEATURE_COLUMNS, FEATURES_PARQUET
from src.process.fastf1 import SESSIONS_PARQUET
from src.process.jolpica import RESULTS_PARQUET
from src.process.openmeteo import WEATHER_PARQUET
from src.utils.race_inventory import INVENTORY_PATH

# ---------------------------------------------------------------------------
# L2 (processed/) -- active perimeter checks
# ---------------------------------------------------------------------------


def _load_or_skip(path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.name} not built yet -- run `just build-l2`")
    return pd.read_parquet(path)


def test_results_has_no_future_races() -> None:
    df = _load_or_skip(RESULTS_PARQUET)
    today = date.today()
    dates = pd.to_datetime(df["race_date"]).dt.date
    future = df[dates > today]
    assert future.empty, (
        f"{len(future)} results rows for races after today ({today}): "
        f"{future['race_id'].unique().tolist()[:5]}"
    )


def test_weather_only_references_past_or_today_races() -> None:
    weather = _load_or_skip(WEATHER_PARQUET)
    inv = _load_or_skip(INVENTORY_PATH)[["race_id", "race_date"]]
    df = weather.merge(inv, on="race_id", how="left", validate="one_to_one")
    today = date.today()
    dates = pd.to_datetime(df["race_date"]).dt.date
    future = df[dates > today]
    assert future.empty, f"{len(future)} weather rows for future races"


def test_sessions_only_references_past_or_today_races() -> None:
    sessions = _load_or_skip(SESSIONS_PARQUET)
    inv = _load_or_skip(INVENTORY_PATH)[["race_id", "race_date"]]
    df = sessions[["race_id"]].drop_duplicates().merge(inv, on="race_id", how="left")
    today = date.today()
    dates = pd.to_datetime(df["race_date"]).dt.date
    future = df[dates > today]
    assert future.empty, f"{len(future)} FastF1 race_ids for future races"


def test_processed_race_ids_are_subset_of_inventory() -> None:
    results = _load_or_skip(RESULTS_PARQUET)
    inv = _load_or_skip(INVENTORY_PATH)
    extra = set(results["race_id"]) - set(inv["race_id"])
    assert not extra, f"processed results have race_ids not in inventory: {sorted(extra)[:5]}"


# ---------------------------------------------------------------------------
# L3 (features/) -- mvp.parquet leakage guards
# ---------------------------------------------------------------------------


def _load_features_or_skip() -> pd.DataFrame:
    if not FEATURES_PARQUET.exists():
        pytest.skip(f"{FEATURES_PARQUET.name} not built yet -- run `just build`")
    return pd.read_parquet(FEATURES_PARQUET)


def test_rolling_features_have_no_lookahead() -> None:
    """A .shift(1) rolling feature at race N must use only races strictly < N.

    driver_form_finish_l5 is re-derived from the table's own outcome columns
    and must match. A missing .shift(1) -- the classic lookahead bug -- folds
    race N's own finish into its feature and breaks this equality.
    """
    df = _load_features_or_skip().sort_values(["driver_id", "year", "round"])
    proxy = df["finish_position"].where(~df["dnf"], _DNF_POSITION_PROXY).astype(float)
    expected = proxy.groupby(df["driver_id"], sort=False).transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )
    pd.testing.assert_series_equal(
        df["driver_form_finish_l5"].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )
    # Structural cross-check: each driver's first-ever race has no prior history.
    first_race = df.groupby("driver_id", sort=False).head(1)
    assert first_race["driver_form_finish_l5"].isna().all()
    assert first_race["driver_form_finish_l10"].isna().all()
    assert (first_race["driver_career_races"] == 0).all()


def test_pre_race_uses_forecast_not_actual_weather() -> None:
    """The MVP feature set carries no actual-weather columns.

    Only race-hour *actuals* exist in weather.parquet; using them in a pre-race
    model would leak. Weather is deferred until a forecast ingest exists -- this
    guard fails loudly if a weather_* feature is added to the set before then.
    """
    leaked = [c for c in FEATURE_COLUMNS if c.startswith("weather_")]
    assert not leaked, f"actual-weather features leaked into the MVP set: {leaked}"


def test_team_features_follow_team_not_driver_after_switch() -> None:
    """team_form_* is keyed on the constructor, not the driver.

    Both cars of one constructor in one race must carry identical team_form
    values -- the feature travels with the car. That is exactly what lets a
    driver who switches teams inherit the new team's history, not their own.
    """
    df = _load_features_or_skip()
    team_cols = [c for c in FEATURE_COLUMNS if c.startswith("team_form_")]
    assert team_cols, "expected team_form_* features in FEATURE_COLUMNS"
    per_team_race = df.groupby(["race_id", "constructor_id"])[team_cols].nunique(dropna=False)
    inconsistent = per_team_race[per_team_race.gt(1).any(axis=1)]
    assert inconsistent.empty, (
        f"team_form_* varies between team-mates in {len(inconsistent)} constructor-races "
        f"(feature is keyed on the driver, not the team):\n{inconsistent.head()}"
    )
