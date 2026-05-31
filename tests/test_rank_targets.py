"""Phase 5 ranking-target guards (target_quali_rank, target_race_rank).

Both targets live in mvp.parquet next to the binary targets. Unlike those they
are full 1..N orders, so the guards are structural: gap-free within each race,
DNFs ordered behind finishers, NaN where the underlying position is undefined,
and forced NaN on the next-race inference rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import (
    FEATURES_PARQUET,
    RANK_TARGETS,
    TARGET_QUALI_RANK,
    TARGET_RACE_RANK,
)
from src.features.next_race import build_next_race_features
from src.features.next_race_prequali import build_prequali_features
from src.process.fastf1 import SESSIONS_PARQUET
from src.process.jolpica import RESULTS_PARQUET


def _load_features_or_skip() -> pd.DataFrame:
    if not FEATURES_PARQUET.exists():
        pytest.skip(f"{FEATURES_PARQUET.name} not built yet -- run `just build`")
    return pd.read_parquet(FEATURES_PARQUET)


def _is_gap_free_1_to_n(values: pd.Series) -> bool:
    ranks = sorted(values.tolist())
    return ranks == list(range(1, len(ranks) + 1))


def test_rank_targets_present_in_table() -> None:
    df = _load_features_or_skip()
    missing = [t for t in RANK_TARGETS if t not in df.columns]
    assert not missing, f"rank targets missing from mvp.parquet: {missing}"


def test_race_rank_is_gap_free_per_race() -> None:
    """Every race resolves to exactly the ranks 1..N with no gaps or duplicates."""
    df = _load_features_or_skip()
    defined = df[df[TARGET_RACE_RANK].notna()]
    bad = [
        rid
        for rid, sub in defined.groupby("race_id", sort=False)
        if not _is_gap_free_1_to_n(sub[TARGET_RACE_RANK].astype(int))
    ]
    assert not bad, f"race_rank not a 1..N permutation in races: {bad[:5]}"


def test_quali_rank_is_gap_free_among_qualified() -> None:
    df = _load_features_or_skip()
    defined = df[df[TARGET_QUALI_RANK].notna()]
    bad = [
        rid
        for rid, sub in defined.groupby("race_id", sort=False)
        if not _is_gap_free_1_to_n(sub[TARGET_QUALI_RANK].astype(int))
    ]
    assert not bad, f"quali_rank not a 1..N permutation in races: {bad[:5]}"


def test_race_rank_places_dnfs_behind_finishers() -> None:
    """In every race, the worst finisher ranks ahead of the best DNF -- the
    DNF-aware ordering the target is built on."""
    df = _load_features_or_skip()
    offenders: list[str] = []
    for rid, sub in df.groupby("race_id", sort=False):
        fin = sub[sub["dnf"] == False]  # noqa: E712 -- pandas boolean mask
        dnf = sub[sub["dnf"] == True]  # noqa: E712
        if fin.empty or dnf.empty:
            continue
        if fin[TARGET_RACE_RANK].max() > dnf[TARGET_RACE_RANK].min():
            offenders.append(rid)
    assert not offenders, f"DNF ranked ahead of a finisher in races: {offenders[:5]}"


def test_quali_rank_nan_where_q_position_nan() -> None:
    df = _load_features_or_skip()
    no_q = df[df["q_position"].isna()]
    if no_q.empty:
        pytest.skip("no rows with NaN q_position in mvp.parquet")
    leaks = int((~no_q[TARGET_QUALI_RANK].isna()).sum())
    assert leaks == 0, f"quali_rank not NaN for {leaks} rows where q_position is NaN"


def test_race_rank_matches_finish_order_in_clean_race() -> None:
    """On a race with no DNFs, race_rank must reproduce the finishing order
    exactly (rank == finish_position)."""
    df = _load_features_or_skip()
    clean = [
        rid
        for rid, sub in df.groupby("race_id", sort=False)
        if (sub["dnf"] == False).all() and sub["finish_position"].notna().all()  # noqa: E712
    ]
    if not clean:
        pytest.skip("no fully-classified race in mvp.parquet")
    sub = df[df["race_id"] == clean[0]]
    assert (
        sub[TARGET_RACE_RANK].astype(int).to_numpy()
        == sub["finish_position"].astype(int).to_numpy()
    ).all()


# --- next-race nulling (inference must never see a target) -------------------


def _require_build_artifacts() -> pd.DataFrame:
    for p in (FEATURES_PARQUET, SESSIONS_PARQUET, RESULTS_PARQUET):
        if not p.exists():
            pytest.skip(f"{p.name} not built -- run `just build && just build-l2` first")
    return pd.read_parquet(FEATURES_PARQUET)


def _pick_recent_complete_race(mvp: pd.DataFrame) -> tuple[int, int]:
    candidates = (
        mvp[mvp["has_fp2"].astype(bool) & mvp["q_position"].notna()][["year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
    )
    if candidates.empty:
        pytest.skip("no historical race with Q + FP2 coverage in mvp.parquet")
    last = candidates.iloc[-1]
    return int(last["year"]), int(last["round"])


def test_next_race_builders_null_rank_targets() -> None:
    """Both next-race builders must force the rank targets to NaN so inference
    cannot accidentally read a leaked order."""
    mvp = _require_build_artifacts()
    year, round_no = _pick_recent_complete_race(mvp)

    nxt = build_next_race_features(year, round_no)
    pq = build_prequali_features(year, round_no)
    for frame, label in ((nxt, "next_race"), (pq, "next_race_prequali")):
        for t in RANK_TARGETS:
            assert frame[t].isna().all(), f"{t} must be NaN on {label} rows"
            # np.nan guard: a leaked 0/1 would pass isna()==False above, this is belt-and-braces.
            assert not np.isfinite(frame[t].to_numpy(dtype=float)).any()
