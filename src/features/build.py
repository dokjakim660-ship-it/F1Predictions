"""L3 feature builder: L2 processed tables -> data/features/mvp.parquet.

One row per (race_id, driver_id) -- the model-ready table the Phase 1.4 podium
model consumes. Joins onto the Jolpica results spine:

- results.parquet  (Jolpica)  -- targets, grid, driver/constructor identity
- sessions.parquet (FastF1)   -- qualifying gaps + FP2 long-run pace
- tracks.csv       (manual)   -- circuit attributes

Pre-race contract: every feature is knowable on Saturday evening -- after
qualifying, before the race. Current-race RESULT columns (finish_position,
dnf, points, race-pace) feed only the targets and the .shift(1)-lagged rolling
history; they are never a feature for their own race. Weather joins the feature
set as a forecast for the upcoming race (pre-race legal) and as the race-hour
archive for historical training rows -- see process/openmeteo and the
test_no_leakage forecast guard.

Cross-track comparability: raw lap times (q_best_ms, FP2 medians) are NOT
comparable between circuits, so only GAP features are emitted -- a 0.3s gap to
pole or to the fastest FP2 long run means the same thing at Monaco and at Spa.

All rolling features use .shift(1) inside the per-driver / per-constructor /
per-(driver,track) group, so race N sees only races strictly before N. This is
the no-lookahead invariant guarded by tests/test_no_leakage.py.

Run: `python -m src.features.build build`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.process.fastf1 import load_sessions
from src.process.jolpica import load_results
from src.process.openmeteo import load_weather
from src.process.overtakes import load_overtakes
from src.utils.paths import FEATURES_DIR
from src.utils.race_inventory import load_inventory
from src.utils.tracks import load_tracks, race_to_track_id

FEATURES_PARQUET = FEATURES_DIR / "mvp.parquet"

# DNFs get a worst-case finishing position in rolling-form averages, so a
# DNF-heavy driver does not look "fast" just because retirements are excluded.
_DNF_POSITION_PROXY = 20

# Columns pulled off the FastF1 sessions table. Everything else there is either
# a key, redundant with Jolpica, or current-race RACE pace (leakage).
_SESSION_COLS = [
    "race_id",
    "driver_id",
    "q_position",
    "q_gap_to_pole_ms",
    "fp2_long_run_median_ms",
    "fp2_short_run_best_ms",
    "fp2_long_run_lap_count",
    "has_fp2",
    "sprint_position",
    "sprint_gap_to_winner_ms",
    "has_sprint",
]

# The model-ready numeric feature set (~31 columns). Exported so the Phase 1.4
# model code imports one list instead of re-declaring column names.
FEATURE_COLUMNS = [
    # Grid / starting position
    "grid_effective",
    "grid_log",
    "is_pole",
    "is_top3_grid",
    # Qualifying (this race, post-quali -> pre-race legal)
    "q_position",
    "q_gap_to_pole_ms",
    "quali_beat_teammate",
    # FP2 pace (this race)
    "fp2_long_run_gap_ms",
    "fp2_short_run_gap_ms",
    "fp2_long_run_lap_count",
    "has_fp2",
    # Sprint result (this race; NaN on non-sprint weekends, has_sprint flags it)
    "sprint_position",
    "sprint_gap_to_winner_ms",
    "sprint_minus_quali_pos",
    "has_sprint",
    # Driver rolling form (lagged)
    "driver_form_finish_l5",
    "driver_form_finish_l10",
    "driver_form_podium_rate_l10",
    "driver_form_points_l5",
    "driver_form_dnf_rate_l10",
    "driver_form_quali_pos_l5",
    "driver_career_races",
    "driver_age_years",
    # Constructor rolling form (lagged)
    "team_form_points_l5",
    "team_form_finish_l5",
    "team_form_podium_rate_l10",
    "team_form_dnf_rate_l10",
    "team_form_quali_gap_pole_l5",
    # Constructor standings entering this race (lagged within-season cumsum)
    "team_season_points_pre_race",
    "team_season_pos_pre_race",
    # Track attributes
    "track_length_km",
    "track_n_corners",
    "track_n_drs_zones",
    "track_is_street",
    # Circuit overtaking difficulty: mean on-track passes in this track's prior
    # races (lagged, no-lookahead -- see _add_track_overtaking).
    "track_overtakes_prior_mean",
    # Driver x track history (lagged)
    "driver_track_finish_l3",
    # Weather at race time (forecast pre-race, archive post-race -- see process/openmeteo)
    "weather_temp_c_race_hour",
    "weather_is_wet_race_hour",
    "weather_precip_mm_day_total",
    "weather_wind_kph_race_hour",
    "weather_temp_c_day_max",
    # Season / era
    "season_progress",
    "era_2022plus",
    "era_2026plus",
]

_WEATHER_COLS = [
    "race_id",
    "weather_temp_c_race_hour",
    "weather_is_wet_race_hour",
    "weather_precip_mm_day_total",
    "weather_wind_kph_race_hour",
    "weather_temp_c_day_max",
]

# String feature kept for native categorical handling (LightGBM) / one-hot
# (XGBoost). Out of FEATURE_COLUMNS because it cannot go through a scaler.
CATEGORICAL_COLUMNS = ["track_id"]

# --- Phase 4.2 pre-quali feature sets ------------------------------------
# Two reduced views of FEATURE_COLUMNS for models that predict BEFORE qualifying
# has happened. Derived (not re-declared) so they stay in sync as the base set
# evolves. The pre-race FEATURE_COLUMNS above is unchanged -- pre-race models
# keep seeing grid + quali.

# Produced by the qualifying session itself (grid slot + quali timing). Illegal
# for any pre-quali model -- this is exactly what we are trying to predict.
_QUALI_DERIVED_FEATURES = [
    "grid_effective",
    "grid_log",
    "is_pole",
    "is_top3_grid",
    "q_position",
    "q_gap_to_pole_ms",
    "quali_beat_teammate",
    "sprint_minus_quali_pos",  # derived from q_position
]

# FP2 long-run pace: knowable Friday night (post-FP2) but not before the
# weekend starts. The dividing line between the two pre-quali views.
_FP2_FEATURES = [
    "fp2_long_run_gap_ms",
    "fp2_short_run_gap_ms",
    "fp2_long_run_lap_count",
    "has_fp2",
]

# Sprint result: the sprint runs Saturday morning -- after the pre-weekend cut
# and, on a normal weekend, after the post-FP2 cut too. Excluded from both
# pre-quali sets for a consistent timing contract. (Sprint pace as a pre-quali
# signal on sprint weekends is a possible later refinement.)
_SPRINT_FEATURES = [
    "sprint_position",
    "sprint_gap_to_winner_ms",
    "has_sprint",
]

# Post-FP2: everything except quali-derived and sprint features (keeps FP2).
FEATURE_COLUMNS_POST_FP2 = [
    c for c in FEATURE_COLUMNS if c not in set(_QUALI_DERIVED_FEATURES + _SPRINT_FEATURES)
]

# Pre-weekend: also drop FP2 -- nothing from the race weekend at all.
FEATURE_COLUMNS_PRE_WEEKEND = [c for c in FEATURE_COLUMNS_POST_FP2 if c not in set(_FP2_FEATURES)]

# Pre-quali feature sets keyed by mode name -- the predict/eval drivers iterate
# over this so a new mode is added in one place.
PRE_QUALI_FEATURE_SETS = {
    "pre_weekend": FEATURE_COLUMNS_PRE_WEEKEND,
    "post_fp2": FEATURE_COLUMNS_POST_FP2,
}

TARGET_PODIUM = "target_podium"
TARGET_TEAMMATE = "target_beat_teammate"

# Phase 5.2 DNF target: P(driver does not finish the race). A pre-race binary
# market in its own right ("driver to retire"); also the finish/no-finish leg of
# an expected-position decomposition. Stored as float so the next-race synth rows
# (dnf NaN, race not run) propagate to a NaN target the way the other targets do.
TARGET_DNF = "target_dnf"

# Phase 4.2 pre-quali targets, all derived from final qualifying position. Kept
# in dedicated target columns (separate from quali_beat_teammate the feature)
# so pre-quali models can train on the target without the feature accidentally
# leaking the answer; pre-race models keep using the feature as before.
TARGET_POLE = "target_pole"
TARGET_TOP3_QUALI = "target_top3_quali"
TARGET_TOP10_QUALI = "target_top10_quali"
TARGET_QUALI_BEAT_TEAMMATE = "target_quali_beat_teammate"

QUALI_TARGETS = [
    TARGET_POLE,
    TARGET_TOP3_QUALI,
    TARGET_TOP10_QUALI,
    TARGET_QUALI_BEAT_TEAMMATE,
]

# Phase 5 ranking targets: the exact within-race finishing/qualifying order,
# 1..N gap-free. Unlike every other target these are NOT binary -- they feed the
# learning-to-rank / position-regression models in src/models/rank.py, scored on
# rank metrics (position MAE, Spearman) instead of Brier. Both are derived from
# the same race/quali ordering the binary targets already use:
#   target_quali_rank -- the gap-free q_rank built in _add_quali_targets.
#   target_race_rank  -- the DNF-aware _race_score order built in _add_targets
#                        (finishers by position, DNFs behind them by laps).
TARGET_QUALI_RANK = "target_quali_rank"
TARGET_RACE_RANK = "target_race_rank"

RANK_TARGETS = [TARGET_QUALI_RANK, TARGET_RACE_RANK]

_META_COLS = [
    "race_id",
    "year",
    "round",
    "race_date",
    "driver_id",
    "driver_code",
    "driver_family_name",
    "constructor_id",
    "constructor_name",
    "grid",
    "finish_position",
    "dnf",
]


def _pos_proxy(df: pd.DataFrame) -> pd.Series:
    """Finishing position with DNFs replaced by a worst-case proxy."""
    return df["finish_position"].where(~df["dnf"], _DNF_POSITION_PROXY).astype(float)


def _roll_shift(s: pd.Series, window: int) -> pd.Series:
    """Rolling mean of the strictly-preceding races (.shift(1) = no lookahead)."""
    return s.rolling(window=window, min_periods=1).mean().shift(1)


def _add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df[TARGET_PODIUM] = (
        df["finish_position"].between(1, 3, inclusive="both") & (~df["dnf"])
    ).astype(int)

    # DNF target: 1 if the driver retired. astype(float) keeps the historical
    # bool as 0.0/1.0 while leaving the next-race synth rows (dnf NaN) NaN, so
    # prepare_dev_test drops them exactly like the teammate target.
    df[TARGET_DNF] = df["dnf"].astype(float)

    # Teammate H2H: rank the two cars of one constructor by race classification.
    # Finishers (lower position better) always beat DNFs; among DNFs, more laps
    # completed wins. A single numeric score keeps that ordering: finishers land
    # in 1..~25, DNFs land near 1e6 so they always rank behind a finisher.
    classified = np.where(df["dnf"], 1.0, 0.0)
    order = np.where(
        df["dnf"],
        -df["laps_completed"].fillna(0).astype(float),
        df["finish_position"].fillna(99).astype(float),
    )
    df["_race_score"] = classified * 1_000_000.0 + order

    grp = df.groupby(["race_id", "constructor_id"], sort=False)["_race_score"]
    best = grp.transform("min")
    n_cars = grp.transform("size")
    has_tie = grp.transform(lambda s: s.duplicated(keep=False).any())

    beat = (df["_race_score"] == best).astype(float)
    # Only defined for the normal two-cars-per-constructor case; a tie (e.g. both
    # cars retired on the same lap) has no winner.
    beat[(n_cars != 2) | has_tie.astype(bool)] = np.nan
    df[TARGET_TEAMMATE] = beat

    # Exact race finishing rank 1..N, gap-free, from the same _race_score order:
    # finishers rank by position, DNFs fall behind by laps completed. method="first"
    # breaks the (impossible-for-finishers, possible-for-same-lap-DNFs) ties
    # deterministically so every race resolves to exactly N distinct ranks.
    df[TARGET_RACE_RANK] = df.groupby("race_id", sort=False)["_race_score"].rank(method="first")
    return df.drop(columns=["_race_score"])


def _add_grid_features(df: pd.DataFrame) -> pd.DataFrame:
    # grid == 0 means a pit-lane start; treat it as the worst grid slot (21).
    grid_eff = df["grid"].where(df["grid"] > 0, 21).astype(float)
    df["grid_effective"] = grid_eff
    df["grid_log"] = np.log(grid_eff)
    df["is_pole"] = (df["grid"] == 1).astype(int)
    df["is_top3_grid"] = df["grid"].between(1, 3, inclusive="both").astype(int)
    return df


def _add_driver_age(df: pd.DataFrame) -> pd.DataFrame:
    race_d = pd.to_datetime(df["race_date"])
    dob = pd.to_datetime(df["driver_dob"], errors="coerce")
    df["driver_age_years"] = (race_d - dob).dt.days / 365.25
    return df


def _add_quali_features(df: pd.DataFrame) -> pd.DataFrame:
    df["q_position"] = pd.to_numeric(df["q_position"], errors="coerce")

    # quali_beat_teammate: 1 if this driver out-qualified their team-mate. NaN
    # unless the constructor fielded exactly two cars that both set a time.
    grp = df.groupby(["race_id", "constructor_id"], sort=False)["q_position"]
    best = grp.transform("min")
    worst = grp.transform("max")
    n_cars = grp.transform("size")

    beat = pd.Series(np.nan, index=df.index, dtype=float)
    beat[df["q_position"] == best] = 1.0
    beat[df["q_position"] == worst] = 0.0
    beat[(n_cars != 2) | df["q_position"].isna() | (best == worst)] = np.nan
    df["quali_beat_teammate"] = beat
    return df


def _add_quali_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 4.2 pre-quali targets, derived from final qualifying position.

    All four are NaN where q_position itself is NaN (driver didn't set a quali
    time -- DNQ, withdrawal). Pole/Top3/Top10 are binary 0/1; teammate mirrors
    the quali_beat_teammate feature (same logic, kept as a separate target so
    the feature/target boundary stays explicit).

    FastF1's q_position occasionally ties two drivers on the same slot and skips
    the next one (~4 of 178 races, e.g. 2024 R15 puts both Albon and Sainz on
    P10). A naive `q_position <= N` then yields N+1 "top-N" drivers. So we
    re-rank within each race -- official position dominates, gap-to-pole breaks
    the tie -- so "top N in qualifying" always resolves to exactly N drivers.

    Must run AFTER _add_quali_features, since both q_position and the
    quali_beat_teammate feature are produced there.
    """
    # Combined sort key: q_position * 1e7 dominates, gap-to-pole (max ~3.6s in
    # the data, << 1e7) breaks ties deterministically. NaN gap -> 1e6 (sorts to
    # the back of its position slot, harmless). rank(method="first") yields a
    # gap-free 1..N ranking; NaN q_position propagates to a NaN rank -> NaN target.
    sort_key = df["q_position"] * 1e7 + df["q_gap_to_pole_ms"].fillna(1e6)
    q_rank = sort_key.groupby(df["race_id"], sort=False).rank(method="first")
    defined = q_rank.notna()

    df[TARGET_POLE] = q_rank.eq(1).astype(float).where(defined)
    df[TARGET_TOP3_QUALI] = q_rank.le(3).astype(float).where(defined)
    df[TARGET_TOP10_QUALI] = q_rank.le(10).astype(float).where(defined)
    df[TARGET_QUALI_BEAT_TEAMMATE] = df["quali_beat_teammate"]
    # The full qualifying order (Phase 5 ranking target): the same gap-free 1..N
    # rank, kept as a column instead of thresholded. NaN where q_position is NaN.
    df[TARGET_QUALI_RANK] = q_rank.where(defined)
    return df


def _add_fp2_features(df: pd.DataFrame) -> pd.DataFrame:
    # Gap to the fastest car in the same FP2 session -- cross-track comparable.
    lr_best = df.groupby("race_id", sort=False)["fp2_long_run_median_ms"].transform("min")
    df["fp2_long_run_gap_ms"] = df["fp2_long_run_median_ms"] - lr_best

    sr_best = df.groupby("race_id", sort=False)["fp2_short_run_best_ms"].transform("min")
    df["fp2_short_run_gap_ms"] = df["fp2_short_run_best_ms"] - sr_best

    # Left-join leaves NaN where FastF1 had no row; .eq(True) folds that to 0
    # without the object-dtype downcasting warning that .fillna(False) raises.
    df["has_fp2"] = df["has_fp2"].eq(True).astype(int)
    return df


def _add_sprint_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sprint-derived features. NaN where the race has no sprint (~85% of races)."""
    df["sprint_position"] = pd.to_numeric(df["sprint_position"], errors="coerce")
    df["sprint_minus_quali_pos"] = df["sprint_position"] - df["q_position"]
    # has_sprint may arrive as bool, object-with-NaN, or numpy bool depending on
    # the left-merge path; .eq(True) folds all three to a clean 0/1.
    df["has_sprint"] = df["has_sprint"].eq(True).astype(int)
    return df


def _add_driver_form(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["driver_id", "race_date"]).reset_index(drop=True)
    df["_pp"] = _pos_proxy(df)
    g = df.groupby("driver_id", sort=False)

    df["driver_form_finish_l5"] = g["_pp"].transform(lambda x: _roll_shift(x, 5))
    df["driver_form_finish_l10"] = g["_pp"].transform(lambda x: _roll_shift(x, 10))
    df["driver_form_podium_rate_l10"] = g[TARGET_PODIUM].transform(
        lambda x: _roll_shift(x.astype(float), 10)
    )
    df["driver_form_points_l5"] = g["points"].transform(lambda x: _roll_shift(x, 5))
    df["driver_form_dnf_rate_l10"] = g["dnf"].transform(lambda x: _roll_shift(x.astype(float), 10))
    df["driver_form_quali_pos_l5"] = g["q_position"].transform(lambda x: _roll_shift(x, 5))
    df["driver_career_races"] = g.cumcount()
    return df.drop(columns=["_pp"])


def _add_team_form(df: pd.DataFrame) -> pd.DataFrame:
    # Collapse the two cars to one row per (constructor, race) before rolling,
    # otherwise a five-race window only spans 2.5 actual race weekends.
    per_race = (
        df.assign(_pp=_pos_proxy(df))
        .groupby(["constructor_id", "race_id", "race_date"], as_index=False)
        .agg(
            team_points=("points", "sum"),
            team_finish=("_pp", "mean"),
            team_podiums=(TARGET_PODIUM, "sum"),
            team_dnf=("dnf", "mean"),
            team_q_gap=("q_gap_to_pole_ms", "min"),
        )
        .sort_values(["constructor_id", "race_date"])
        .reset_index(drop=True)
    )
    g = per_race.groupby("constructor_id", sort=False)
    per_race["team_form_points_l5"] = g["team_points"].transform(lambda x: _roll_shift(x, 5))
    per_race["team_form_finish_l5"] = g["team_finish"].transform(lambda x: _roll_shift(x, 5))
    # team_podiums is 0..2 per race; halve it to a per-car podium rate.
    per_race["team_form_podium_rate_l10"] = g["team_podiums"].transform(
        lambda x: _roll_shift(x / 2.0, 10)
    )
    per_race["team_form_dnf_rate_l10"] = g["team_dnf"].transform(lambda x: _roll_shift(x, 10))
    per_race["team_form_quali_gap_pole_l5"] = g["team_q_gap"].transform(lambda x: _roll_shift(x, 5))

    out_cols = [
        "constructor_id",
        "race_id",
        "team_form_points_l5",
        "team_form_finish_l5",
        "team_form_podium_rate_l10",
        "team_form_dnf_rate_l10",
        "team_form_quali_gap_pole_l5",
    ]
    return df.merge(per_race[out_cols], on=["constructor_id", "race_id"], how="left")


def _add_team_standings(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative WCC points and rank ENTERING the current race (no leakage).

    Sums both cars' points per race, accumulates within (year, constructor) and
    shifts by one so the current race is excluded. Rank is across constructors
    within the same race (lower = better). Round 1 of each season returns NaN
    for both -- no within-season history yet, downstream median-impute handles
    it.
    """
    per_race = (
        df.groupby(["year", "constructor_id", "race_id", "race_date"], as_index=False)["points"]
        .sum()
        .rename(columns={"points": "_team_race_points"})
        .sort_values(["year", "constructor_id", "race_date"])
        .reset_index(drop=True)
    )
    per_race["team_season_points_pre_race"] = per_race.groupby(
        ["year", "constructor_id"], sort=False
    )["_team_race_points"].transform(lambda x: x.cumsum().shift(1))
    per_race["team_season_pos_pre_race"] = per_race.groupby(["year", "race_id"], sort=False)[
        "team_season_points_pre_race"
    ].rank(ascending=False, method="min")
    out_cols = [
        "year",
        "constructor_id",
        "race_id",
        "team_season_points_pre_race",
        "team_season_pos_pre_race",
    ]
    return df.merge(per_race[out_cols], on=["year", "constructor_id", "race_id"], how="left")


def _add_track_features(df: pd.DataFrame, inv: pd.DataFrame, tracks: pd.DataFrame) -> pd.DataFrame:
    circuits = inv[["race_id", "circuit_id"]].drop_duplicates()
    df = df.merge(circuits, on="race_id", how="left")
    df["track_id"] = [
        race_to_track_id(rid, cid) for rid, cid in zip(df["race_id"], df["circuit_id"], strict=True)
    ]

    attrs = tracks[["track_id", "length_km", "n_corners", "n_drs_zones", "track_type"]]
    df = df.merge(attrs, on="track_id", how="left")
    df = df.rename(
        columns={
            "length_km": "track_length_km",
            "n_corners": "track_n_corners",
            "n_drs_zones": "track_n_drs_zones",
        }
    )
    df["track_is_street"] = (df["track_type"] == "street").astype(int)
    return df.drop(columns=["circuit_id", "track_type"])


def _add_driver_track_history(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["driver_id", "track_id", "race_date"]).reset_index(drop=True)
    df["_pp"] = _pos_proxy(df)
    g = df.groupby(["driver_id", "track_id"], sort=False)
    df["driver_track_finish_l3"] = g["_pp"].transform(lambda x: _roll_shift(x, 3))
    return df.drop(columns=["_pp"])


def _add_track_overtaking(df: pd.DataFrame, overtakes: pd.DataFrame) -> pd.DataFrame:
    """Lagged circuit-overtaking index: mean on-track passes in this track's
    PRIOR races (expanding mean, .shift(1) = no lookahead).

    The current race's own overtake count is itself a race outcome (it is derived
    from the finishing-order churn), so it can never feed its own row. A circuit's
    first appearance in the data starts NaN and is median-imputed downstream, like
    every other cold-start lagged feature. Collapsing to one row per (track, race)
    first keeps a 20-car race from counting as 20 observations in the window --
    the same guard _add_team_form uses.
    """
    ov = overtakes.loc[overtakes["is_usable"].astype(bool), ["race_id", "n_overtakes"]]
    per_race = (
        df[["track_id", "race_id", "race_date"]]
        .drop_duplicates(subset=["track_id", "race_id"])
        .merge(ov, on="race_id", how="left")
        .sort_values(["track_id", "race_date"])
    )
    g = per_race.groupby("track_id", sort=False)
    per_race["track_overtakes_prior_mean"] = g["n_overtakes"].transform(
        lambda x: x.expanding(min_periods=1).mean().shift(1)
    )
    return df.merge(
        per_race[["track_id", "race_id", "track_overtakes_prior_mean"]],
        on=["track_id", "race_id"],
        how="left",
    )


def _add_weather_features(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(weather[_WEATHER_COLS], on="race_id", how="left")
    # is_wet stored as nullable bool/object -> cast to 0/1 float, NaN preserved
    # so median-imputation can handle it downstream (mvp.py:fillna(medians)).
    df["weather_is_wet_race_hour"] = df["weather_is_wet_race_hour"].map({True: 1.0, False: 0.0})
    return df


def _add_season_era(df: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    # Race number normalised by the season length, so "round 10" means the same
    # part of the year whether the calendar has 21 races or 24.
    season_rounds = inv.groupby("year")["round"].max().rename("season_rounds")
    df = df.merge(season_rounds, left_on="year", right_index=True, how="left")
    df["season_progress"] = df["round"] / df["season_rounds"]
    df = df.drop(columns=["season_rounds"])
    df["era_2022plus"] = (df["year"] >= 2022).astype(int)
    df["era_2026plus"] = (df["year"] >= 2026).astype(int)
    return df


def compute_features(
    results: pd.DataFrame,
    sessions: pd.DataFrame,
    inv: pd.DataFrame,
    tracks: pd.DataFrame,
    weather: pd.DataFrame,
    overtakes: pd.DataFrame,
) -> pd.DataFrame:
    """Run the L2 -> L3 feature pipeline on pre-loaded tables.

    Split out of build_features() so the Phase 3.2 next-race builder can drive
    the same pipeline with a synthesized pseudo-results row appended onto the
    historical results table.
    """
    missing = [c for c in _SESSION_COLS if c not in sessions.columns]
    if missing:
        raise KeyError(f"sessions.parquet missing expected columns: {missing}")

    df = results.merge(sessions[_SESSION_COLS], on=["race_id", "driver_id"], how="left")

    df = _add_targets(df)
    df = _add_grid_features(df)
    df = _add_driver_age(df)
    df = _add_quali_features(df)
    df = _add_quali_targets(df)
    df = _add_fp2_features(df)
    df = _add_sprint_features(df)
    df = _add_driver_form(df)
    df = _add_team_form(df)
    df = _add_team_standings(df)
    df = _add_track_features(df, inv, tracks)
    df = _add_driver_track_history(df)
    df = _add_track_overtaking(df, overtakes)
    df = _add_weather_features(df, weather)
    df = _add_season_era(df, inv)

    out_cols = (
        _META_COLS
        + [TARGET_PODIUM, TARGET_TEAMMATE, TARGET_DNF]
        + QUALI_TARGETS
        + RANK_TARGETS
        + FEATURE_COLUMNS
        + CATEGORICAL_COLUMNS
    )
    df = df[out_cols].sort_values(["year", "round", "driver_id"]).reset_index(drop=True)
    return df


def build_features() -> pd.DataFrame:
    return compute_features(
        load_results(),
        load_sessions(),
        load_inventory(),
        load_tracks(),
        load_weather(),
        load_overtakes(),
    )


def save_features(df: pd.DataFrame, path: Path = FEATURES_PARQUET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_features(path: Path = FEATURES_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m src.features.build build")
    return pd.read_parquet(path)


def _print_summary(df: pd.DataFrame) -> None:
    n_races = df[["year", "round"]].drop_duplicates().shape[0]
    matched = df["q_gap_to_pole_ms"].notna().mean()
    print(f"[features.build] {len(df)} rows, {df.shape[1]} cols, {n_races} races")
    print(f"[features.build] {len(FEATURE_COLUMNS)} numeric features + {CATEGORICAL_COLUMNS}")
    print(f"[features.build] podium rate: {df[TARGET_PODIUM].mean():.3f}")
    print(
        f"[features.build] dnf rate: {df[TARGET_DNF].mean():.3f} "
        f"(defined {df[TARGET_DNF].notna().mean():.1%} of rows)"
    )
    print(
        f"[features.build] teammate target defined: "
        f"{df[TARGET_TEAMMATE].notna().mean():.1%} of rows"
    )
    for t in QUALI_TARGETS:
        defined = df[t].notna().mean()
        rate = df[t].mean()
        print(f"[features.build] {t}: rate {rate:.3f} (defined {defined:.1%} of rows)")
    for t in RANK_TARGETS:
        defined = df[t].notna().mean()
        max_rank = df[t].max()
        print(f"[features.build] {t}: max rank {max_rank:.0f} (defined {defined:.1%} of rows)")
    print(f"[features.build] FastF1 quali join matched: {matched:.1%} of rows")
    print(f"[features.build] has_fp2: {df['has_fp2'].mean():.1%} of rows")
    high_nan = {
        c: f"{df[c].isna().mean():.1%}" for c in FEATURE_COLUMNS if df[c].isna().mean() > 0.05
    }
    print(f"[features.build] features >5% NaN (rookies / no-FP2 expected): {high_nan}")
    by_year = df.groupby("year").size()
    for year, n in by_year.items():
        print(f"  {year}: {n} rows")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="L2 processed tables -> data/features/mvp.parquet")
    sub.add_parser("show", help="Print summary of the current feature table")
    args = p.parse_args(argv)

    if args.cmd == "build":
        df = build_features()
        out = save_features(df)
        print(f"[features.build] saved -> {out}")
        _print_summary(df)
        return 0
    if args.cmd == "show":
        _print_summary(load_features())
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
