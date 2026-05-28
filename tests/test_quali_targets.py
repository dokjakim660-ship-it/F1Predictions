"""Phase 4.2.0 pre-quali target guards.

Four new targets (pole, top3-quali, top10-quali, quali-beat-teammate) live in
mvp.parquet alongside the existing podium / teammate targets. They are all
derived from `q_position`, so the guards here are structural (exactly N drivers
per race after the tie-repair re-rank), the plain `q_position <= N` formula on
tie-free races, NaN propagation (driver didn't qualify -> target is NaN), and
distribution plausibility on the full historical table.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build import (
    FEATURE_COLUMNS_POST_FP2,
    FEATURE_COLUMNS_PRE_WEEKEND,
    FEATURES_PARQUET,
    PRE_QUALI_FEATURE_SETS,
    QUALI_TARGETS,
    TARGET_POLE,
    TARGET_QUALI_BEAT_TEAMMATE,
    TARGET_TOP3_QUALI,
    TARGET_TOP10_QUALI,
)


def _load_features_or_skip() -> pd.DataFrame:
    if not FEATURES_PARQUET.exists():
        pytest.skip(f"{FEATURES_PARQUET.name} not built yet -- run `just build`")
    return pd.read_parquet(FEATURES_PARQUET)


def _quali_count_per_race(df: pd.DataFrame) -> pd.Series:
    """Number of drivers with a quali time per race."""
    return df[df["q_position"].notna()].groupby("race_id").size()


def test_pole_is_exactly_one_per_qualified_race() -> None:
    """After the tie-repair re-rank, every race with at least one quali time has
    exactly one pole-sitter."""
    df = _load_features_or_skip()
    n_quali = _quali_count_per_race(df)
    poles = df.groupby("race_id")[TARGET_POLE].sum(min_count=1)
    races = n_quali[n_quali >= 1].index
    bad = poles.loc[races][poles.loc[races] != 1.0]
    assert bad.empty, f"races without exactly one pole: {bad.head().to_dict()}"


def test_top3_quali_is_exactly_three_per_race() -> None:
    df = _load_features_or_skip()
    n_quali = _quali_count_per_race(df)
    top3 = df.groupby("race_id")[TARGET_TOP3_QUALI].sum(min_count=1)
    races = n_quali[n_quali >= 3].index
    bad = top3.loc[races][top3.loc[races] != 3.0]
    assert bad.empty, f"races without exactly three top-3-quali: {bad.head().to_dict()}"


def test_top10_quali_is_exactly_ten_per_race() -> None:
    df = _load_features_or_skip()
    n_quali = _quali_count_per_race(df)
    top10 = df.groupby("race_id")[TARGET_TOP10_QUALI].sum(min_count=1)
    races = n_quali[n_quali >= 10].index
    bad = top10.loc[races][top10.loc[races] != 10.0]
    assert bad.empty, f"races without exactly ten top-10-quali: {bad.head().to_dict()}"


def test_targets_match_q_position_in_tie_free_races() -> None:
    """Where FastF1's q_position has no ties (the vast majority of races), the
    targets reduce to the plain `q_position <= N` formula. The tie-repair only
    diverges from this on the handful of glitched races, which we exclude here.
    """
    df = _load_features_or_skip()
    has_q = df[df["q_position"].notna()].copy()
    tie_races = (
        has_q.groupby("race_id")["q_position"]
        .apply(lambda x: x.duplicated().any())
        .pipe(lambda s: s[s].index)
    )
    clean = has_q[~has_q["race_id"].isin(tie_races)]
    assert (clean[TARGET_POLE] == (clean["q_position"] == 1).astype(float)).all()
    assert (clean[TARGET_TOP3_QUALI] == clean["q_position"].between(1, 3).astype(float)).all()
    assert (clean[TARGET_TOP10_QUALI] == clean["q_position"].between(1, 10).astype(float)).all()


def test_quali_beat_teammate_target_mirrors_feature() -> None:
    """The target column is a deliberate copy of the quali_beat_teammate
    feature: same logic, kept separate so pre-quali models can train on it
    without the feature accidentally leaking the answer."""
    df = _load_features_or_skip()
    pd.testing.assert_series_equal(
        df[TARGET_QUALI_BEAT_TEAMMATE].astype(float).reset_index(drop=True),
        df["quali_beat_teammate"].astype(float).reset_index(drop=True),
        check_names=False,
    )


def test_quali_targets_nan_where_q_position_is_nan() -> None:
    df = _load_features_or_skip()
    no_q = df[df["q_position"].isna()]
    if no_q.empty:
        pytest.skip("no rows with NaN q_position in mvp.parquet")
    for t in (TARGET_POLE, TARGET_TOP3_QUALI, TARGET_TOP10_QUALI):
        leaks = (~no_q[t].isna()).sum()
        assert leaks == 0, f"{t} not NaN for {leaks} rows where q_position is NaN"


def test_quali_target_distributions_plausible() -> None:
    """Sanity ranges across all races where q_position is set. Pole ~1/20,
    top-3 ~3/20, top-10 ~10/20, teammate ~50%. Ranges are loose -- they exist
    to catch a swapped target column, not to assert a precise rate.
    """
    df = _load_features_or_skip()
    has_q = df["q_position"].notna()

    pole_rate = df.loc[has_q, TARGET_POLE].mean()
    assert 0.03 <= pole_rate <= 0.08, f"pole rate {pole_rate:.3f} outside [0.03, 0.08]"

    top3_rate = df.loc[has_q, TARGET_TOP3_QUALI].mean()
    assert 0.10 <= top3_rate <= 0.20, f"top3-quali rate {top3_rate:.3f} outside [0.10, 0.20]"

    top10_rate = df.loc[has_q, TARGET_TOP10_QUALI].mean()
    assert 0.40 <= top10_rate <= 0.60, f"top10-quali rate {top10_rate:.3f} outside [0.40, 0.60]"

    teammate_rate = df[TARGET_QUALI_BEAT_TEAMMATE].mean()
    assert 0.40 <= teammate_rate <= 0.60, (
        f"quali teammate rate {teammate_rate:.3f} outside [0.40, 0.60]"
    )


def test_all_quali_targets_present_in_table() -> None:
    """QUALI_TARGETS must all materialise as columns -- guards against a future
    refactor dropping one of them from compute_features's out_cols list."""
    df = _load_features_or_skip()
    missing = [t for t in QUALI_TARGETS if t not in df.columns]
    assert not missing, f"quali targets missing from mvp.parquet: {missing}"


# --- Phase 4.2.1 pre-quali feature-set guards --------------------------------

# Anything produced by qualifying, grid, or the sprint -- none of it may appear
# in a pre-quali feature set, since the race weekend hasn't happened yet.
_BANNED_PRE_QUALI = (
    "grid_effective",
    "grid_log",
    "is_pole",
    "is_top3_grid",
    "q_position",
    "q_gap_to_pole_ms",
    "quali_beat_teammate",
    "sprint_position",
    "sprint_gap_to_winner_ms",
    "sprint_minus_quali_pos",
    "has_sprint",
)


def test_pre_quali_feature_sets_exclude_weekend_data() -> None:
    """No grid / quali / sprint feature may leak into either pre-quali set."""
    for mode, fset in PRE_QUALI_FEATURE_SETS.items():
        leaked = [c for c in _BANNED_PRE_QUALI if c in fset]
        assert not leaked, f"{mode} feature set leaks weekend data: {leaked}"


def test_pre_weekend_excludes_fp2_post_fp2_includes_it() -> None:
    """The only difference between the two sets is FP2 pace."""
    fp2 = ("fp2_long_run_gap_ms", "fp2_short_run_gap_ms", "fp2_long_run_lap_count", "has_fp2")
    for col in fp2:
        assert col not in FEATURE_COLUMNS_PRE_WEEKEND, f"{col} should not be in pre_weekend"
        assert col in FEATURE_COLUMNS_POST_FP2, f"{col} should be in post_fp2"
    # post_fp2 == pre_weekend + the four FP2 features, nothing else.
    assert set(FEATURE_COLUMNS_POST_FP2) - set(FEATURE_COLUMNS_PRE_WEEKEND) == set(fp2)


def test_pre_quali_feature_sets_are_nonempty_and_in_table() -> None:
    """Both sets must be non-empty and every column must exist in mvp.parquet."""
    df = _load_features_or_skip()
    for mode, fset in PRE_QUALI_FEATURE_SETS.items():
        assert fset, f"{mode} feature set is empty"
        missing = [c for c in fset if c not in df.columns]
        assert not missing, f"{mode} references columns not in mvp.parquet: {missing}"
