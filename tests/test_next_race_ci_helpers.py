"""Tests for the CI helpers behind the predict-next-race GitHub Action:
resolve_next_round (which race to predict) and qualifying_ready (the gate that
stops a grid-less prediction from being deployed)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.features.next_race import qualifying_ready
from src.utils.race_inventory import resolve_next_round

_INV = pd.DataFrame(
    {
        "race_id": ["2026_05", "2026_06", "2026_07"],
        "year": [2026, 2026, 2026],
        "round": [5, 6, 7],
        "gp_name": ["Canada", "Monaco", "Spain"],
        "race_date": [date(2026, 5, 24), date(2026, 6, 7), date(2026, 6, 14)],
    }
)


def test_resolve_next_round_picks_earliest_future_race() -> None:
    info = resolve_next_round(_INV, date(2026, 6, 4))
    assert info is not None
    assert (info["year"], info["round"]) == (2026, 6)
    assert info["race_id"] == "2026_06"
    assert info["days_until"] == 3


def test_resolve_next_round_zero_days_on_race_day() -> None:
    # A Sunday-morning run on race day still resolves to that day's race.
    info = resolve_next_round(_INV, date(2026, 6, 7))
    assert info is not None and info["round"] == 6
    assert info["days_until"] == 0


def test_resolve_next_round_rolls_to_following_race_after_one_passes() -> None:
    info = resolve_next_round(_INV, date(2026, 6, 8))
    assert info is not None and info["round"] == 7


def test_resolve_next_round_none_when_all_past() -> None:
    assert resolve_next_round(_INV, date(2027, 1, 1)) is None


def _sessions(has_qualifying: bool, q_positions: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2026] * len(q_positions),
            "round": [6] * len(q_positions),
            "has_qualifying": [has_qualifying] * len(q_positions),
            "q_position": q_positions,
        }
    )


def test_qualifying_ready_true_when_positions_present() -> None:
    sessions = _sessions(True, [1.0, 2.0, 3.0])
    assert qualifying_ready(2026, 6, sessions=sessions) is True


def test_qualifying_ready_false_when_all_positions_nan() -> None:
    # Pre-qualifying: the Q session exists in the schedule but has no results.
    sessions = _sessions(True, [np.nan, np.nan])
    assert qualifying_ready(2026, 6, sessions=sessions) is False


def test_qualifying_ready_false_when_race_absent() -> None:
    sessions = _sessions(True, [1.0, 2.0])
    assert qualifying_ready(2026, 99, sessions=sessions) is False
