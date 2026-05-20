"""Lookahead-leakage guard. Will be filled in Phase 1 once feature builder exists.

The intent: for every rolling/aggregate feature, assert that at race N the value
depends only on data strictly before N. See project_feature_decisions.md memory
for the three known leakage traps (roll without .shift(1), race weather vs
forecast, driver/team historic mixup).
"""

import pytest


@pytest.mark.skip(reason="Phase 1: implement once src/features/build.py exists")
def test_rolling_features_have_no_lookahead() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 1: implement once weather features exist")
def test_pre_race_uses_forecast_not_actual_weather() -> None:
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 1: implement once team features exist")
def test_team_features_follow_team_not_driver_after_switch() -> None:
    raise NotImplementedError


def test_placeholder_runs() -> None:
    # Smoke: confirms pytest is wired up. Remove when real tests land.
    assert 1 + 1 == 2
