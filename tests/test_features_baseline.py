"""Tests for src/features/baseline.py.

Two flavours:

- Lookahead unit tests on a tiny synthetic frame: the rolling features must
  reflect ONLY races strictly before the current row. Race-N's feature value
  is built from races 1..N-1, never N..end. This is the no-leakage invariant
  Phase-1 needs because the Optuna objective (Brier) is highly sensitive to
  any leak.

- A handful of integration smoke tests that load the built baseline parquet
  and check structural invariants. Skip cleanly if the parquet isn't built.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.baseline import (
    BASELINE_PARQUET,
    _add_constructor_rolling,
    _add_driver_rolling,
    _add_grid_features,
    _add_target,
    build_baseline_features,
)


def _toy_frame() -> pd.DataFrame:
    """3 drivers x 4 races. Hand-built so rolling values are easy to check."""
    rows = [
        # race_id, race_date, driver_id, constructor_id, grid, finish_position, points, dnf
        ("R01", "2024-03-01", "alice", "alpha", 1, 1, 25, False),
        ("R01", "2024-03-01", "bob", "alpha", 2, 2, 18, False),
        ("R01", "2024-03-01", "carol", "beta", 3, 3, 15, False),
        ("R02", "2024-03-08", "alice", "alpha", 1, 2, 18, False),
        ("R02", "2024-03-08", "bob", "alpha", 3, 1, 25, False),
        ("R02", "2024-03-08", "carol", "beta", 2, 99, 0, True),  # DNF
        ("R03", "2024-03-15", "alice", "alpha", 2, 1, 25, False),
        ("R03", "2024-03-15", "bob", "alpha", 1, 3, 15, False),
        ("R03", "2024-03-15", "carol", "beta", 3, 2, 18, False),
        ("R04", "2024-03-22", "alice", "alpha", 1, 1, 25, False),
        ("R04", "2024-03-22", "bob", "alpha", 2, 2, 18, False),
        ("R04", "2024-03-22", "carol", "beta", 3, 3, 15, False),
    ]
    cols = [
        "race_id",
        "race_date",
        "driver_id",
        "constructor_id",
        "grid",
        "finish_position",
        "points",
        "dnf",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df["race_date"] = pd.to_datetime(df["race_date"]).dt.date
    df["year"] = 2024
    df["round"] = df["race_id"].str[1:].astype(int)
    df["status"] = "Finished"
    df["driver_dob"] = pd.to_datetime("1995-01-01").date()
    return df


# ---------------------------------------------------------------------------
# Lookahead / shift correctness
# ---------------------------------------------------------------------------


class TestDriverRolling:
    def test_first_race_features_are_nan(self) -> None:
        df = _add_driver_rolling(_toy_frame())
        # Every driver's first race -> no history -> NaN.
        first_per_driver = df.sort_values(["driver_id", "race_date"]).groupby("driver_id").head(1)
        assert first_per_driver["driver_rolling_avg_pos_l5"].isna().all()
        assert first_per_driver["driver_rolling_dnf_rate_l5"].isna().all()

    def test_career_count_starts_at_zero(self) -> None:
        df = _add_driver_rolling(_toy_frame())
        first_per_driver = df.sort_values(["driver_id", "race_date"]).groupby("driver_id").head(1)
        assert (first_per_driver["driver_career_race_count"] == 0).all()

    def test_alice_rolling_avg_pos_uses_only_past(self) -> None:
        # Alice positions: R01=1, R02=2, R03=1, R04=1
        # Rolling avg at R03 should = mean([1, 2]) = 1.5 (uses R01 + R02 only).
        # Rolling avg at R04 should = mean([1, 2, 1]) = 1.3333.
        df = _add_driver_rolling(_toy_frame())
        alice = df[df["driver_id"] == "alice"].sort_values("race_date").reset_index(drop=True)
        assert alice.loc[2, "driver_rolling_avg_pos_l5"] == pytest.approx(1.5)
        assert alice.loc[3, "driver_rolling_avg_pos_l5"] == pytest.approx(4.0 / 3)

    def test_carol_dnf_uses_proxy_position(self) -> None:
        # Carol R01=3, R02=DNF(20), R03=2.
        # Rolling avg at R03 = mean([3, 20]) = 11.5.
        df = _add_driver_rolling(_toy_frame())
        carol = df[df["driver_id"] == "carol"].sort_values("race_date").reset_index(drop=True)
        assert carol.loc[2, "driver_rolling_avg_pos_l5"] == pytest.approx(11.5)

    def test_carol_dnf_rate_correct(self) -> None:
        # Carol DNFs: R01=0, R02=1, R03=0, R04=0.
        # Rolling DNF rate at R03 = mean([0, 1]) = 0.5.
        df = _add_driver_rolling(_toy_frame())
        carol = df[df["driver_id"] == "carol"].sort_values("race_date").reset_index(drop=True)
        assert carol.loc[2, "driver_rolling_dnf_rate_l5"] == pytest.approx(0.5)

    def test_no_leakage_into_current_race(self) -> None:
        # Property check: for every row, the rolling avg position computed from
        # the row's own driver's PAST rows must equal the feature value.
        # If the rolling didn't shift correctly, the feature would include the
        # row's own finish position.
        df = (
            _add_driver_rolling(_toy_frame())
            .sort_values(["driver_id", "race_date"])
            .reset_index(drop=True)
        )
        for did in df["driver_id"].unique():
            sub = df[df["driver_id"] == did].sort_values("race_date").reset_index(drop=True)
            for i in range(len(sub)):
                past = sub.iloc[:i]
                if past.empty:
                    assert pd.isna(sub.loc[i, "driver_rolling_avg_pos_l5"])
                else:
                    proxy = past["finish_position"].where(~past["dnf"], 20).astype(float)
                    expected = proxy.tail(5).mean()
                    assert sub.loc[i, "driver_rolling_avg_pos_l5"] == pytest.approx(expected)


class TestConstructorRolling:
    def test_first_race_per_constructor_is_nan(self) -> None:
        df = _toy_frame()
        df = _add_constructor_rolling(df)
        # R01 is the first race for both constructors -> NaN.
        r01 = df[df["race_id"] == "R01"]
        assert r01["con_rolling_pts_l5"].isna().all()

    def test_alpha_team_combined_points_used(self) -> None:
        # Alpha R01 total = 25+18 = 43; R02 = 18+25 = 43; R03 = 25+15 = 40.
        # Rolling at R03 for any alpha row = mean([43, 43]) = 43.0 (uses R01+R02).
        # Rolling at R04 = mean([43, 43, 40]) = 42.0.
        df = _add_constructor_rolling(_toy_frame())
        alpha_r03 = df[(df["constructor_id"] == "alpha") & (df["race_id"] == "R03")]
        assert alpha_r03["con_rolling_pts_l5"].iloc[0] == pytest.approx(43.0)
        alpha_r04 = df[(df["constructor_id"] == "alpha") & (df["race_id"] == "R04")]
        assert alpha_r04["con_rolling_pts_l5"].iloc[0] == pytest.approx(42.0)


class TestTargetAndGrid:
    def test_target_excludes_dnf_even_if_position_in_top_3(self) -> None:
        # Carol R02: position=99 status but dnf=True -- not a podium.
        df = _add_target(_toy_frame())
        carol_r02 = df[(df["driver_id"] == "carol") & (df["race_id"] == "R02")]
        assert carol_r02["target_podium"].iloc[0] == 0

    def test_target_includes_dnf_false_top_3(self) -> None:
        df = _add_target(_toy_frame())
        alice_r01 = df[(df["driver_id"] == "alice") & (df["race_id"] == "R01")]
        assert alice_r01["target_podium"].iloc[0] == 1

    def test_grid_log_uses_pit_lane_proxy(self) -> None:
        df = _toy_frame().copy()
        df.loc[0, "grid"] = 0  # Force a pit-lane start.
        df = _add_grid_features(df)
        # grid==0 -> grid_effective=21 -> grid_log=log(21).
        assert df.loc[0, "grid_effective"] == 21
        assert df.loc[0, "grid_log"] == pytest.approx(np.log(21))


# ---------------------------------------------------------------------------
# Integration: full builder end-to-end on the synthetic frame
# ---------------------------------------------------------------------------


def test_build_baseline_features_full_pipeline_smoke(monkeypatch) -> None:
    # build_baseline_features() reads a parquet -- swap it for the toy frame.
    toy = _toy_frame()
    monkeypatch.setattr("src.features.baseline.pd.read_parquet", lambda _p: toy)
    monkeypatch.setattr("src.features.baseline.Path.exists", lambda _self: True)
    df = build_baseline_features()
    assert {
        "target_podium",
        "grid_log",
        "driver_rolling_avg_pos_l5",
        "con_rolling_pts_l5",
    }.issubset(df.columns)
    assert len(df) == len(toy)


# ---------------------------------------------------------------------------
# Real-data smoke: only runs after `just build-l2` + features built
# ---------------------------------------------------------------------------


def test_real_features_first_row_per_driver_has_nan_rolling() -> None:
    if not BASELINE_PARQUET.exists():
        pytest.skip("baseline.parquet not built yet")
    df = pd.read_parquet(BASELINE_PARQUET)
    df = df.sort_values(["driver_id", "race_date"])
    first_per_driver = df.groupby("driver_id").head(1)
    # 100% of driver-debuts should have NaN rolling avg (no history).
    nan_share = first_per_driver["driver_rolling_avg_pos_l5"].isna().mean()
    assert nan_share == 1.0, f"only {nan_share:.0%} of driver-debuts have NaN rolling"
