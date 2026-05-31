"""Phase 5.2 DNF-target guards (target_dnf).

target_dnf is the binary "driver did not finish" outcome, stored in mvp.parquet
next to the other binary targets. The guards are: it is exactly the dnf column
as 0/1, defined for every historical (classified) row, and forced NaN on the
next-race inference rows so inference can never read a leaked outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURES_PARQUET, TARGET_DNF
from src.features.next_race import build_next_race_features
from src.features.next_race_prequali import build_prequali_features
from src.process.fastf1 import SESSIONS_PARQUET
from src.process.jolpica import RESULTS_PARQUET


def _load_features_or_skip() -> pd.DataFrame:
    if not FEATURES_PARQUET.exists():
        pytest.skip(f"{FEATURES_PARQUET.name} not built yet -- run `just build`")
    return pd.read_parquet(FEATURES_PARQUET)


def test_dnf_target_present_in_table() -> None:
    df = _load_features_or_skip()
    assert TARGET_DNF in df.columns, "target_dnf missing from mvp.parquet"


def test_dnf_target_is_binary() -> None:
    df = _load_features_or_skip()
    vals = set(df[TARGET_DNF].dropna().unique().tolist())
    assert vals <= {0.0, 1.0}, f"target_dnf has non-binary values: {vals}"


def test_dnf_target_matches_dnf_column() -> None:
    """target_dnf is just the dnf flag as a float -- it must agree row for row."""
    df = _load_features_or_skip()
    assert (df[TARGET_DNF].to_numpy() == df["dnf"].astype(float).to_numpy()).all()


def test_dnf_target_defined_for_all_historical_rows() -> None:
    df = _load_features_or_skip()
    n_nan = int(df[TARGET_DNF].isna().sum())
    assert n_nan == 0, f"target_dnf NaN for {n_nan} historical rows (dnf should always be known)"


def test_dnf_base_rate_is_plausible() -> None:
    """Sanity floor: F1 retirement rate sits roughly in the 5-20% band."""
    df = _load_features_or_skip()
    rate = float(df[TARGET_DNF].mean())
    assert 0.03 < rate < 0.25, f"implausible DNF base rate {rate:.3f}"


# --- next-race nulling (inference must never see the target) -----------------


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


def test_next_race_builders_null_dnf_target() -> None:
    """Both next-race builders must force target_dnf to NaN so inference cannot
    accidentally read a leaked 0/1 outcome."""
    mvp = _require_build_artifacts()
    year, round_no = _pick_recent_complete_race(mvp)

    nxt = build_next_race_features(year, round_no)
    pq = build_prequali_features(year, round_no)
    for frame, label in ((nxt, "next_race"), (pq, "next_race_prequali")):
        assert frame[TARGET_DNF].isna().all(), f"target_dnf must be NaN on {label} rows"
        assert not np.isfinite(frame[TARGET_DNF].to_numpy(dtype=float)).any()
