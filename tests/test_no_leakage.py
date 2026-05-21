"""Lookahead-leakage guards.

Two layers:

- L2 (processed/) -- this file, active. The processed tables must not contain
  rows for races that have not yet happened. This is a cheap perimeter check
  before any feature engineering touches the data; if a "future" race already
  has a results row, something upstream (ingest, schedule) is wrong.

- L3 (features/) -- the rolling-feature, weather, and driver/team-switch
  leakage tests are stubbed below and will be implemented once
  src/features/build.py exists. See project_feature_decisions.md for the
  three known traps (roll without .shift(1), race weather vs forecast,
  driver/team historic mixup).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

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
# L3 (features/) -- placeholders, fill once src/features/build.py exists
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Phase 1.2: implement once src/features/build.py exists")
def test_rolling_features_have_no_lookahead() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 1.2: implement once weather features exist")
def test_pre_race_uses_forecast_not_actual_weather() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 1.2: implement once team features exist")
def test_team_features_follow_team_not_driver_after_switch() -> None:
    raise NotImplementedError
