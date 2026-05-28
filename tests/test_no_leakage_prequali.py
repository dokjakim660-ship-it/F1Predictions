"""Phase 4.2.4b pre-quali next-race builder leakage guards.

The pre-quali builder synthesizes a driver row with NO qualifying and re-runs
compute_features. Two invariants:

1. Every quali/grid/sprint feature + all targets are NaN on the pre-quali rows.
   The pre-quali contract forbids any weekend-derived signal.

2. The pre-quali-LEGAL features (FEATURE_COLUMNS_POST_FP2: historical form, FP2
   pace, track, weather) reproduce mvp.parquet exactly for a past race. Those
   features come from the .shift(1) history + same-race FP2, so adding the
   target race as a pseudo-row at the end of history must not change them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build import (
    FEATURE_COLUMNS,
    FEATURE_COLUMNS_POST_FP2,
    FEATURES_PARQUET,
    QUALI_TARGETS,
    TARGET_PODIUM,
    TARGET_TEAMMATE,
)
from src.features.next_race_prequali import build_prequali_features
from src.process.fastf1 import SESSIONS_PARQUET
from src.process.jolpica import RESULTS_PARQUET

_QUALI_FEATURE_COLS = [c for c in FEATURE_COLUMNS if c not in set(FEATURE_COLUMNS_POST_FP2)]


def _require_artifacts() -> pd.DataFrame:
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


def test_prequali_quali_features_and_targets_are_null() -> None:
    mvp = _require_artifacts()
    year, round_no = _pick_recent_complete_race(mvp)

    pq = build_prequali_features(year, round_no)

    for col in _QUALI_FEATURE_COLS:
        assert pq[col].isna().all(), f"{col} must be NaN on pre-quali rows (quali leak)"
    for col in (*QUALI_TARGETS, TARGET_PODIUM, TARGET_TEAMMATE):
        assert pq[col].isna().all(), f"{col} target must be NaN on pre-quali rows"
    assert pq["finish_position"].isna().all()
    assert pq["grid"].isna().all()


def test_prequali_legal_features_match_historical_row() -> None:
    """Round-trip: pre-quali-legal features reproduce mvp.parquet for a past race.

    Compares only the FEATURE_COLUMNS_POST_FP2 set (historical form + FP2 + track
    + weather) on the drivers present in both tables. If a future change leaks the
    target race into a .shift(1) feature, the value diverges here.
    """
    mvp = _require_artifacts()
    year, round_no = _pick_recent_complete_race(mvp)
    race_id = f"{year}_{round_no:02d}"

    pq = build_prequali_features(year, round_no).set_index("driver_id")
    hist = mvp[mvp["race_id"] == race_id].set_index("driver_id")
    common = pq.index.intersection(hist.index)
    assert len(common) >= 15, f"too few shared drivers ({len(common)}) to trust the round-trip"

    diffs: list[str] = []
    for col in FEATURE_COLUMNS_POST_FP2:
        a = hist.loc[common, col]
        b = pq.loc[common, col]
        if a.dtype.kind in "fi" and b.dtype.kind in "fi":
            equal = (a.fillna(-1e18) == b.fillna(-1e18)).all()
        else:
            equal = (a.astype(str) == b.astype(str)).all()
        if not equal:
            diffs.append(col)
    assert not diffs, (
        f"{len(diffs)} pre-quali-legal features diverged from mvp.parquet: {diffs[:10]}"
    )


def test_prequali_rowcount_matches_session_lineup() -> None:
    mvp = _require_artifacts()
    year, round_no = _pick_recent_complete_race(mvp)

    pq = build_prequali_features(year, round_no)
    sessions = pd.read_parquet(SESSIONS_PARQUET)
    sess = sessions[(sessions["year"] == year) & (sessions["round"] == round_no)]
    sess = sess[sess["driver_id"].notna() & sess["team_id"].notna()]
    expected = sess["driver_id"].nunique()
    assert len(pq) == expected, f"row count {len(pq)} != session drivers {expected}"
