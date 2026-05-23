"""Phase 3.2 next-race feature builder leakage guards.

The next-race builder synthesizes a pseudo-results row (race-outcome columns
all NaN) and re-runs the same compute_features pipeline used for training.
Two invariants must hold for the result to be safe for inference:

1. Targets and finish-state columns are NaN — never accidentally 0 or False.
   Inference code keys off `.isna()` to skip them.

2. ZERO leakage of the target race's own outcome into its features.
   This is verified against the historical mvp.parquet: every model feature
   on a past race that we re-run through `build_next_race_features` must
   match the value in mvp.parquet exactly. The `.shift(1)` rolling guard is
   what makes this hold; if a future refactor drops it, this test breaks.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.build import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    FEATURES_PARQUET,
    TARGET_PODIUM,
    TARGET_TEAMMATE,
)
from src.features.next_race import build_next_race_features
from src.process.fastf1 import SESSIONS_PARQUET
from src.process.jolpica import RESULTS_PARQUET


def _require_artifacts() -> pd.DataFrame:
    for p in (FEATURES_PARQUET, SESSIONS_PARQUET, RESULTS_PARQUET):
        if not p.exists():
            pytest.skip(f"{p.name} not built — run `just build && just build-l2` first")
    return pd.read_parquet(FEATURES_PARQUET)


def _pick_recent_complete_race(mvp: pd.DataFrame) -> tuple[int, int]:
    """Latest race in mvp.parquet with full FP2 + Q coverage — best round-trip target."""
    candidates = (
        mvp[mvp["has_fp2"].astype(bool) & mvp["q_position"].notna()][["year", "round"]]
        .drop_duplicates()
        .sort_values(["year", "round"])
    )
    if candidates.empty:
        pytest.skip("no historical race with Q + FP2 coverage in mvp.parquet")
    last = candidates.iloc[-1]
    return int(last["year"]), int(last["round"])


def test_next_race_targets_are_nan() -> None:
    mvp = _require_artifacts()
    year, round_no = _pick_recent_complete_race(mvp)

    nxt = build_next_race_features(year, round_no)

    assert nxt[TARGET_PODIUM].isna().all(), "target_podium must be NaN on next-race rows"
    assert nxt[TARGET_TEAMMATE].isna().all(), "target_beat_teammate must be NaN on next-race rows"
    assert nxt["finish_position"].isna().all(), "finish_position must be NaN on next-race rows"
    assert nxt["dnf"].isna().all(), "dnf must be NaN on next-race rows"


def test_next_race_features_match_historical_row() -> None:
    """Round-trip: build_next_race_features on a past race must reproduce mvp.parquet exactly.

    This is the no-leakage invariant. If a future change drops .shift(1) in a
    rolling feature, race N's own finish leaks into its feature and the value
    diverges from the historical row.
    """
    mvp = _require_artifacts()
    year, round_no = _pick_recent_complete_race(mvp)
    race_id = f"{year}_{round_no:02d}"

    nxt = build_next_race_features(year, round_no)
    mvp_row = mvp[mvp["race_id"] == race_id].sort_values("driver_id").reset_index(drop=True)
    nxt_row = nxt.sort_values("driver_id").reset_index(drop=True)

    assert len(mvp_row) == len(nxt_row), (
        f"row count mismatch: mvp={len(mvp_row)} nxt={len(nxt_row)}"
    )
    assert (mvp_row["driver_id"].values == nxt_row["driver_id"].values).all(), (
        "driver ordering diverged"
    )

    diffs: list[str] = []
    for col in FEATURE_COLUMNS + CATEGORICAL_COLUMNS:
        a = mvp_row[col]
        b = nxt_row[col]
        if a.dtype.kind in "fi" and b.dtype.kind in "fi":
            equal = (a.fillna(-1e18) == b.fillna(-1e18)).all()
        else:
            equal = (a.fillna("__NA__") == b.fillna("__NA__")).all()
        if not equal:
            diffs.append(col)

    assert not diffs, (
        f"{len(diffs)} features leaked target-race data into themselves "
        f"(.shift(1) likely missing): {diffs[:10]}"
    )


def test_next_race_rowcount_matches_quali_drivers() -> None:
    """One row per driver who set a quali time — no phantom rows."""
    mvp = _require_artifacts()
    year, round_no = _pick_recent_complete_race(mvp)

    nxt = build_next_race_features(year, round_no)
    sessions = pd.read_parquet(SESSIONS_PARQUET)
    q = sessions[
        (sessions["year"] == year)
        & (sessions["round"] == round_no)
        & sessions["q_position"].notna()
    ]
    assert len(nxt) == len(q), f"row count {len(nxt)} != quali drivers {len(q)}"
