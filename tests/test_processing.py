"""Smoke + unit tests for the L2 processing layer (src/process/*).

Two layers of tests:
- Pure-function unit tests on parsing / lookup helpers (no I/O).
- Integration smoke tests that load the built parquets and check structural
  invariants. These skip cleanly when the parquets are missing so a fresh
  clone can still run `pytest` without a full data build.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.process import fastf1 as fastf1_proc
from src.process import jolpica as jolpica_proc
from src.process import openmeteo as openmeteo_proc

# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


class TestJolpicaParse:
    def test_lap_time_minutes(self) -> None:
        assert jolpica_proc._lap_time_to_ms("1:34.364") == 94_364.0

    def test_lap_time_seconds_only(self) -> None:
        assert jolpica_proc._lap_time_to_ms("59.123") == 59_123.0

    def test_lap_time_short_millis_pads(self) -> None:
        # "1:34.36" should be parsed as 1:34.360, not 1:34.036.
        assert jolpica_proc._lap_time_to_ms("1:34.36") == 94_360.0

    def test_lap_time_empty_returns_none(self) -> None:
        assert jolpica_proc._lap_time_to_ms(None) is None
        assert jolpica_proc._lap_time_to_ms("") is None
        assert jolpica_proc._lap_time_to_ms("garbage") is None

    @pytest.mark.parametrize("status", ["Finished", "+1 Lap", "+2 Laps"])
    def test_classified_statuses(self, status: str) -> None:
        assert jolpica_proc._is_classified(status) is True

    @pytest.mark.parametrize(
        "status", ["Engine", "Collision", "Retired", "Accident", "Disqualified"]
    )
    def test_unclassified_statuses(self, status: str) -> None:
        assert jolpica_proc._is_classified(status) is False

    def test_row_extraction_minimal(self) -> None:
        payload = {
            "Driver": {
                "driverId": "max_verstappen",
                "code": "VER",
                "givenName": "Max",
                "familyName": "Verstappen",
                "dateOfBirth": "1997-09-30",
                "nationality": "Dutch",
            },
            "Constructor": {"constructorId": "red_bull", "name": "Red Bull"},
            "grid": "1",
            "position": "1",
            "positionText": "1",
            "points": "26",
            "laps": "57",
            "status": "Finished",
            "Time": {"millis": "5504742", "time": "1:31:44.742"},
            "FastestLap": {
                "rank": "1",
                "Time": {"time": "1:32.608"},
                "AverageSpeed": {"speed": "210.383"},
            },
        }
        row = jolpica_proc._row_from_result(2024, 1, "2024-03-02", payload)
        assert row["race_id"] == "2024_01"
        assert row["driver_id"] == "max_verstappen"
        assert row["grid"] == 1
        assert row["finish_position"] == 1
        assert row["dnf"] is False
        assert row["finish_time_ms"] == 5_504_742
        assert row["fastest_lap_ms"] == 92_608.0
        assert row["fastest_lap_rank"] == 1

    def test_row_extraction_dnf(self) -> None:
        payload = {
            "Driver": {"driverId": "perez"},
            "Constructor": {"constructorId": "red_bull"},
            "grid": "5",
            "position": "20",
            "positionText": "R",
            "points": "0",
            "laps": "12",
            "status": "Engine",
        }
        row = jolpica_proc._row_from_result(2024, 1, "2024-03-02", payload)
        assert row["dnf"] is True
        assert row["finish_time_ms"] is None
        assert row["fastest_lap_ms"] is None


class TestOpenMeteoLocalTime:
    def test_normal_positive_offset_same_day(self) -> None:
        # Bahrain 2024: 15:00 UTC + 3h = 18:00 local, same date.
        local = openmeteo_proc._race_local_dt("2024-03-02", "15:00:00Z", 3 * 3600)
        assert local == datetime(2024, 3, 2, 18, 0)

    def test_vegas_negative_offset_crosses_midnight_utc(self) -> None:
        # Vegas 2024: Jolpica stores date=2024-11-23, time=06:00 UTC (Sunday morning).
        # Real local race start = Saturday 2024-11-23 22:00 (Pacific Standard, -8h).
        # Naive shift gives Friday 22:00; our 24h-bump fixes it to Saturday 22:00.
        local = openmeteo_proc._race_local_dt("2024-11-23", "06:00:00Z", -8 * 3600)
        assert local == datetime(2024, 11, 23, 22, 0)

    def test_fallback_time_used_when_missing(self) -> None:
        # No time given -> fallback 14:00 UTC. UTC+1 -> 15:00 local same day.
        local = openmeteo_proc._race_local_dt("2024-06-09", "", 3600)
        assert local == datetime(2024, 6, 9, 15, 0)

    def test_find_hour_idx_hits(self) -> None:
        times = [f"2024-03-02T{h:02d}:00" for h in range(24)]
        idx = openmeteo_proc._find_hour_idx(times, datetime(2024, 3, 2, 18, 0))
        assert idx == 18

    def test_find_hour_idx_miss_returns_none(self) -> None:
        times = [f"2024-03-02T{h:02d}:00" for h in range(24)]
        idx = openmeteo_proc._find_hour_idx(times, datetime(2024, 3, 3, 18, 0))
        assert idx is None


# ---------------------------------------------------------------------------
# Integration smoke tests (skip if parquets not yet built)
# ---------------------------------------------------------------------------


def _skip_if_missing(path) -> pd.DataFrame:
    if not path.exists():
        pytest.skip(f"{path.name} not built yet -- run `just build-l2`")
    return pd.read_parquet(path)


class TestResultsParquet:
    def test_loads_with_expected_columns(self) -> None:
        df = _skip_if_missing(jolpica_proc.RESULTS_PARQUET)
        required = {
            "race_id",
            "year",
            "round",
            "race_date",
            "driver_id",
            "constructor_id",
            "grid",
            "finish_position",
            "points",
            "status",
            "dnf",
            "finish_time_ms",
        }
        assert required.issubset(set(df.columns))

    def test_driver_race_combinations_unique(self) -> None:
        df = _skip_if_missing(jolpica_proc.RESULTS_PARQUET)
        dup = df.duplicated(subset=["race_id", "driver_id"]).sum()
        assert dup == 0, f"{dup} duplicate (race_id, driver_id) rows"

    def test_dnf_implies_no_finish_time(self) -> None:
        df = _skip_if_missing(jolpica_proc.RESULTS_PARQUET)
        bad = df[df["dnf"] & df["finish_time_ms"].notna()]
        assert len(bad) == 0, f"{len(bad)} DNF rows still have a finish_time_ms"

    def test_dnf_rate_in_realistic_range(self) -> None:
        df = _skip_if_missing(jolpica_proc.RESULTS_PARQUET)
        rate = df["dnf"].mean()
        # Modern era: ~10-30% per season. Anything outside [5%, 40%] -> processor bug.
        assert 0.05 < rate < 0.40, f"DNF rate {rate:.1%} outside plausible band"

    def test_grid_in_valid_range(self) -> None:
        df = _skip_if_missing(jolpica_proc.RESULTS_PARQUET)
        grids = df["grid"].dropna()
        assert grids.min() >= 0  # 0 = pit-lane start
        assert grids.max() <= 24  # safety margin above current 20-car grid


class TestWeatherParquet:
    def test_loads_with_expected_columns(self) -> None:
        df = _skip_if_missing(openmeteo_proc.WEATHER_PARQUET)
        required = {
            "race_id",
            "year",
            "round",
            "race_local_dt",
            "race_hour_aligned",
            "weather_temp_c_race_hour",
            "weather_precip_mm_race_hour",
            "weather_wind_kph_race_hour",
            "weather_is_wet_race_hour",
            "weather_temp_c_day_max",
            "weather_precip_mm_day_total",
        }
        assert required.issubset(set(df.columns))

    def test_one_row_per_race(self) -> None:
        df = _skip_if_missing(openmeteo_proc.WEATHER_PARQUET)
        dup = df.duplicated(subset=["race_id"]).sum()
        assert dup == 0

    def test_all_race_hours_aligned(self) -> None:
        # The Vegas off-by-one fix should leave 0 misaligned.
        df = _skip_if_missing(openmeteo_proc.WEATHER_PARQUET)
        misaligned = (~df["race_hour_aligned"]).sum()
        assert misaligned == 0, f"{misaligned} races still misaligned"

    def test_temperatures_plausible(self) -> None:
        df = _skip_if_missing(openmeteo_proc.WEATHER_PARQUET)
        t = df["weather_temp_c_race_hour"].dropna()
        # F1 races run in -5C..50C realistic band.
        assert t.min() >= -5
        assert t.max() <= 50


class TestSessionsParquet:
    def test_loads_with_expected_columns(self) -> None:
        df = _skip_if_missing(fastf1_proc.SESSIONS_PARQUET)
        required = {
            "race_id",
            "year",
            "round",
            "driver_abbr",
            "driver_id",
            "team_id",
            "q_position",
            "q_best_ms",
            "q_gap_to_pole_ms",
            "race_clean_median_lap_ms",
            "race_pace_gap_to_leader_ms",
            "fp2_long_run_median_ms",
            "fp2_short_run_best_ms",
            "has_qualifying",
            "has_race",
            "has_fp2",
        }
        assert required.issubset(set(df.columns))

    def test_driver_race_combinations_unique(self) -> None:
        df = _skip_if_missing(fastf1_proc.SESSIONS_PARQUET)
        dup = df.duplicated(subset=["race_id", "driver_abbr"]).sum()
        assert dup == 0

    def test_pole_has_zero_gap(self) -> None:
        df = _skip_if_missing(fastf1_proc.SESSIONS_PARQUET)
        races_with_quali = df[df["has_qualifying"]]
        if races_with_quali.empty:
            pytest.skip("no qualifying sessions processed yet")
        per_race_min_gap = races_with_quali.groupby("race_id")["q_gap_to_pole_ms"].min()
        # Pole-sitter should always be at gap 0 (mod NaN).
        assert (per_race_min_gap.dropna() == 0).all()

    def test_quali_times_plausible(self) -> None:
        df = _skip_if_missing(fastf1_proc.SESSIONS_PARQUET)
        q = df["q_best_ms"].dropna()
        if q.empty:
            pytest.skip("no qualifying sessions processed yet")
        # F1 quali laps: roughly 60s (Monza-like) .. 110s (Spa/Baku-like).
        assert q.min() > 55_000
        assert q.max() < 120_000


class TestCrossTableAlignment:
    def test_jolpica_and_fastf1_driver_ids_consistent_where_both_present(self) -> None:
        if not jolpica_proc.RESULTS_PARQUET.exists() or not fastf1_proc.SESSIONS_PARQUET.exists():
            pytest.skip("processed parquets not built yet")
        j = pd.read_parquet(jolpica_proc.RESULTS_PARQUET)[["race_id", "driver_id"]]
        f = pd.read_parquet(fastf1_proc.SESSIONS_PARQUET)[["race_id", "driver_id"]].dropna()
        shared = set(j["race_id"]) & set(f["race_id"])
        if not shared:
            pytest.skip("no overlapping races between Jolpica + FastF1 yet")
        # For each shared race, every FastF1 driver_id should appear in Jolpica
        # for that race (modulo edge cases — assert >=90% overlap as a smoke band).
        overlap_rates: list[float] = []
        for rid in shared:
            jids = set(j[j["race_id"] == rid]["driver_id"])
            fids = set(f[f["race_id"] == rid]["driver_id"])
            if not fids:
                continue
            overlap_rates.append(len(jids & fids) / len(fids))
        assert sum(overlap_rates) / len(overlap_rates) >= 0.90, (
            f"low jolpica/fastf1 driver_id overlap: {sum(overlap_rates) / len(overlap_rates):.2%}"
        )
