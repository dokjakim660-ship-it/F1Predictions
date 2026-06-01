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


def test_future_race_weather_is_forecast_not_archive() -> None:
    """Weather features are pre-race legal only because a not-yet-run race pulls
    a forecast, never the actual race-hour archive -- the archive would leak the
    very conditions the race is run in. The L2 weather table tags each row
    forecast/archive; every future race must be forecast-sourced.

    (Phase 3.6 replaced the old "no weather features at all" guard: weather is
    now in the feature set, sourced from a forecast for the upcoming race and
    from the archive for historical training rows.)
    """
    weather = _load_or_skip(WEATHER_PARQUET)
    assert "weather_source" in weather.columns, "weather.parquet missing forecast/archive tag"
    unexpected = set(weather["weather_source"].unique()) - {"forecast", "archive"}
    assert not unexpected, f"unexpected weather_source values: {unexpected}"

    inv = _load_or_skip(INVENTORY_PATH)[["race_id", "race_date"]]
    df = weather.merge(inv, on="race_id", how="left")
    today = date.today()
    future = df[pd.to_datetime(df["race_date"]).dt.date > today]
    leaked = future[future["weather_source"] != "forecast"]
    assert leaked.empty, (
        f"{len(leaked)} future races carry actual-archive weather (leak): "
        f"{leaked['race_id'].tolist()[:5]}"
    )


def test_track_overtaking_index_is_lagged_circuit_level() -> None:
    """track_overtakes_prior_mean is a circuit property from STRICTLY-PRIOR races.

    Two guarantees, both broken by the obvious lookahead bug (folding race N's own
    overtake count into its feature):
      - every driver in one race shares the value (it is per-circuit, not per-car);
      - a circuit's first race in the data has no history -> NaN;
      - the value equals the expanding mean of that circuit's earlier races,
        re-derived independently from overtakes.parquet.
    """
    df = _load_features_or_skip()
    col = "track_overtakes_prior_mean"
    assert col in df.columns, f"{col} missing from feature table"

    # Per-circuit, not per-car: constant across drivers in a (track, race).
    per_race = df.groupby(["track_id", "race_id"])[col].nunique(dropna=False)
    assert (per_race <= 1).all(), f"{col} varies between team-mates -- not circuit-level"

    # Independent re-derivation from the raw overtake counts.
    from src.process.overtakes import OVERTAKES_PARQUET
    from src.utils.tracks import race_to_track_id

    ov = _load_or_skip(OVERTAKES_PARQUET)
    inv = _load_or_skip(INVENTORY_PATH)[["race_id", "circuit_id"]]
    ov = ov[ov["is_usable"].astype(bool)].merge(inv, on="race_id", how="left")
    ov["track_id"] = [race_to_track_id(r, c) for r, c in zip(ov["race_id"], ov["circuit_id"])]
    ov = ov.sort_values(["track_id", "year", "round"])
    ov["expected"] = (
        ov.groupby("track_id")["n_overtakes"]
        .transform(lambda x: x.expanding(min_periods=1).mean().shift(1))
    )

    got = df[["race_id", "track_id", col]].drop_duplicates(["race_id", "track_id"])
    merged = got.merge(ov[["race_id", "expected"]], on="race_id", how="left")
    pd.testing.assert_series_equal(
        merged[col].reset_index(drop=True),
        merged["expected"].reset_index(drop=True),
        check_names=False,
    )


def test_team_execution_residual_is_lagged_and_team_level() -> None:
    """team_exec_residual_l5/l10 = pace-rank minus finish, averaged over the
    constructor's STRICTLY-PRIOR races.

    Guards the three traps:
      - per-constructor, not per-car: both team-mates share the value;
      - independently re-derived from race pace (sessions) + finish (features)
        equals the stored value -- which is only true if the rolling mean is
        .shift(1)-lagged (folding race N's own result in would break it);
      - DNF rows never contribute their own residual.
    """
    df = _load_features_or_skip()
    cols = ["team_exec_residual_l5", "team_exec_residual_l10"]
    for col in cols:
        assert col in df.columns, f"{col} missing from feature table"

    # Per-constructor, not per-car: constant across both cars in a (team, race).
    per_team_race = df.groupby(["race_id", "constructor_id"])[cols].nunique(dropna=False)
    assert not per_team_race[per_team_race.gt(1).any(axis=1)].shape[0], (
        f"{cols} vary between team-mates -- residual keyed on the driver, not the team"
    )

    # Independent re-derivation from race pace + finishing position.
    sessions = _load_or_skip(SESSIONS_PARQUET)[["race_id", "driver_id", "race_clean_median_lap_ms"]]
    base = df[["race_id", "driver_id", "constructor_id", "race_date", "finish_position", "dnf"]].merge(
        sessions, on=["race_id", "driver_id"], how="left"
    )
    fin = base[(~base["dnf"].astype(bool)) & base["race_clean_median_lap_ms"].notna()].copy()
    fin = fin.sort_values(["race_date", "driver_id"])
    fin["_rank"] = fin.groupby("race_id", sort=False)["race_clean_median_lap_ms"].rank(method="first")
    fin["_resid_raw"] = fin["_rank"] - fin["finish_position"].astype(float)
    # Match the production floor/ceiling debias: per-pace-rank mean over STRICTLY
    # prior races (expanding().shift(1)), not a global mean.
    fin["_resid"] = fin["_resid_raw"] - fin.groupby("_rank", sort=False)["_resid_raw"].transform(
        lambda x: x.expanding().mean().shift(1)
    )
    resid = fin.groupby(["constructor_id", "race_id"], as_index=False)["_resid"].mean()

    per_race = (
        df[["constructor_id", "race_id", "race_date"]]
        .drop_duplicates()
        .merge(resid, on=["constructor_id", "race_id"], how="left")
        .sort_values(["constructor_id", "race_date"])
        .reset_index(drop=True)
    )
    grp = per_race.groupby("constructor_id", sort=False)["_resid"]
    for col, window in (("team_exec_residual_l5", 5), ("team_exec_residual_l10", 10)):
        per_race[f"exp_{col}"] = grp.transform(
            lambda x, w=window: x.rolling(window=w, min_periods=1).mean().shift(1)
        )
        got = df[["race_id", "constructor_id", col]].drop_duplicates(["race_id", "constructor_id"])
        merged = got.merge(
            per_race[["race_id", "constructor_id", f"exp_{col}"]],
            on=["race_id", "constructor_id"],
            how="left",
        )
        pd.testing.assert_series_equal(
            merged[col].reset_index(drop=True),
            merged[f"exp_{col}"].reset_index(drop=True),
            check_names=False,
        )


def test_teammate_quali_gap_is_lagged_and_symmetric() -> None:
    """driver_teammate_quali_gap_l5 = lagged rolling mean of the per-race gap to
    one's own team-mate in qualifying.

    Guards:
      - within a two-car constructor in one race, the two raw gaps are exact
        negatives (driver A is +x vs B iff B is -x vs A);
      - the stored feature equals the .shift(1) rolling mean re-derived
        independently from q_gap_to_pole_ms (folding the current race in would
        break the equality -> catches a missing lag).
    """
    df = _load_features_or_skip()
    col = "driver_teammate_quali_gap_l5"
    assert col in df.columns, f"{col} missing from feature table"

    # Re-derive the per-race raw gap from q_gap_to_pole_ms.
    work = df[["race_id", "constructor_id", "driver_id", "race_date", "q_gap_to_pole_ms"]].copy()
    grp = work.groupby(["race_id", "constructor_id"], sort=False)["q_gap_to_pole_ms"]
    n_valid = grp.transform(lambda s: s.notna().sum())
    teammate = grp.transform("sum") - work["q_gap_to_pole_ms"]
    work["_raw"] = (work["q_gap_to_pole_ms"] - teammate).where(
        (n_valid == 2) & work["q_gap_to_pole_ms"].notna()
    )

    # Symmetry: the two valid gaps in a constructor-race sum to ~0.
    two_car = work[n_valid == 2].groupby(["race_id", "constructor_id"])["_raw"].sum()
    assert two_car.abs().max() < 1e-6, "team-mate gaps are not antisymmetric within a constructor"

    # Independent lagged re-derivation.
    work = work.sort_values(["driver_id", "race_date"]).reset_index(drop=True)
    work["expected"] = work.groupby("driver_id", sort=False)["_raw"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean().shift(1)
    )
    merged = df[["race_id", "driver_id", col]].merge(
        work[["race_id", "driver_id", "expected"]], on=["race_id", "driver_id"], how="left"
    )
    pd.testing.assert_series_equal(
        merged[col].reset_index(drop=True),
        merged["expected"].reset_index(drop=True),
        check_names=False,
    )


def _sessions_col_or_skip(col: str) -> pd.DataFrame:
    s = _load_or_skip(SESSIONS_PARQUET)
    if col not in s.columns:
        pytest.skip(f"{col} not in sessions.parquet -- rerun `python -m src.process.fastf1 build`")
    return s[["race_id", "driver_id", col]]


def test_pit_crew_speed_is_lagged_track_debiased_and_team_level() -> None:
    """team_pit_speed_resid_l5 = lagged constructor pit-lane time vs track norm."""
    df = _load_features_or_skip()
    col = "team_pit_speed_resid_l5"
    assert col in df.columns, f"{col} missing"
    # team-level: both cars share the value.
    per = df.groupby(["race_id", "constructor_id"])[col].nunique(dropna=False)
    assert (per <= 1).all(), f"{col} varies between team-mates"

    sess = _sessions_col_or_skip("race_pit_lane_median_ms")
    base = df[["race_id", "driver_id", "constructor_id", "race_date", "track_id"]].merge(
        sess, on=["race_id", "driver_id"], how="left"
    )
    base = base[base["race_pit_lane_median_ms"].notna()].copy()
    base["_r"] = base["race_pit_lane_median_ms"] - base.groupby("race_id")[
        "race_pit_lane_median_ms"
    ].transform("median")
    resid = base.groupby(["constructor_id", "race_id"], as_index=False)["_r"].mean()
    per_race = (
        df[["constructor_id", "race_id", "race_date"]].drop_duplicates()
        .merge(resid, on=["constructor_id", "race_id"], how="left")
        .sort_values(["constructor_id", "race_date"]).reset_index(drop=True)
    )
    per_race["exp"] = per_race.groupby("constructor_id", sort=False)["_r"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )
    m = df[["race_id", "constructor_id", col]].drop_duplicates(["race_id", "constructor_id"]).merge(
        per_race[["race_id", "constructor_id", "exp"]], on=["race_id", "constructor_id"], how="left"
    )
    pd.testing.assert_series_equal(
        m[col].reset_index(drop=True), m["exp"].reset_index(drop=True), check_names=False
    )


def test_start_performance_is_lagged_per_driver() -> None:
    """driver_start_pos_gain_l5 = lagged rolling (grid_effective - lap1_position)."""
    df = _load_features_or_skip()
    col = "driver_start_pos_gain_l5"
    assert col in df.columns, f"{col} missing"
    sess = _sessions_col_or_skip("race_lap1_position")
    base = df[["race_id", "driver_id", "race_date", "grid_effective"]].merge(
        sess, on=["race_id", "driver_id"], how="left"
    )
    base["_g"] = base["grid_effective"] - base["race_lap1_position"]
    base = base.sort_values(["driver_id", "race_date"]).reset_index(drop=True)
    base["exp"] = base.groupby("driver_id", sort=False)["_g"].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
    )
    m = df[["race_id", "driver_id", col]].merge(
        base[["race_id", "driver_id", "exp"]], on=["race_id", "driver_id"], how="left"
    )
    pd.testing.assert_series_equal(
        m[col].reset_index(drop=True), m["exp"].reset_index(drop=True), check_names=False
    )


def test_wet_skill_delta_is_lagged_per_driver() -> None:
    """driver_wet_skill_delta = lagged (dry_mean - wet_mean) of the finish-position
    proxy, split by weather_is_wet_race_hour, re-derived independently.

    Guards:
      - equals the .shift(1)-lagged per-driver expanding means of the wet and dry
        position proxies (folding race N's own result in -- a missing lag -- would
        break the equality);
      - structurally NaN until a driver has both a prior wet and a prior dry race.
    """
    df = _load_features_or_skip()
    col = "driver_wet_skill_delta"
    assert col in df.columns, f"{col} missing from feature table"

    work = df.sort_values(["driver_id", "year", "round"]).reset_index(drop=True)
    pp = work["finish_position"].where(~work["dnf"], _DNF_POSITION_PROXY).astype(float)
    wet = work["weather_is_wet_race_hour"]
    wet_mean = pp.where(wet == 1.0).groupby(work["driver_id"], sort=False).transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )
    dry_mean = pp.where(wet == 0.0).groupby(work["driver_id"], sort=False).transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )
    expected = dry_mean - wet_mean
    pd.testing.assert_series_equal(
        work[col].reset_index(drop=True),
        expected.reset_index(drop=True),
        check_names=False,
    )

    # A driver's first-ever race has no prior wet or dry history -> NaN.
    first_race = work.groupby("driver_id", sort=False).head(1)
    assert first_race[col].isna().all()


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
