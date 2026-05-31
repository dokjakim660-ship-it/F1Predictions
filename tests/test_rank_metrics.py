"""Unit tests for the Phase 5 ranking metrics (src/eval/rank_metrics.py).

Hand-built two-race examples: every metric has a known answer so a regression in
the rank-conversion or the per-race aggregation surfaces immediately.
"""

from __future__ import annotations

import math

import numpy as np

from src.eval.rank_metrics import (
    exact_rate,
    paired_bootstrap_mae_ci,
    position_mae,
    spearman_per_race,
    to_race_ranks,
    topk_hit_rate,
)

# Two races of three drivers each. Row order is the driver order.
RACE_IDS = np.array(["r1", "r1", "r1", "r2", "r2", "r2"])
TRUE_RANK = np.array([1, 2, 3, 1, 2, 3])


def test_to_race_ranks_ascending() -> None:
    # Lower score = better place. r1 scores [3,1,2] -> ranks [3,1,2];
    # r2 scores [5,9,7] -> ranks [1,3,2].
    scores = np.array([3.0, 1.0, 2.0, 5.0, 9.0, 7.0])
    out = to_race_ranks(RACE_IDS, scores, ascending=True)
    assert out.tolist() == [3, 1, 2, 1, 3, 2]


def test_to_race_ranks_descending() -> None:
    # Higher score = better place (ranker output sense).
    scores = np.array([3.0, 1.0, 2.0, 5.0, 9.0, 7.0])
    out = to_race_ranks(RACE_IDS, scores, ascending=False)
    assert out.tolist() == [1, 3, 2, 3, 1, 2]


def test_to_race_ranks_are_gap_free_permutation() -> None:
    scores = np.array([0.7, 0.7, 0.1, 9.0, 2.0, 2.0])  # ties included
    out = to_race_ranks(RACE_IDS, scores, ascending=True)
    for race in ("r1", "r2"):
        ranks = sorted(out[RACE_IDS == race].tolist())
        assert ranks == [1, 2, 3], f"{race} ranks not a 1..N permutation: {ranks}"


def test_position_mae() -> None:
    pred = np.array([1, 2, 3, 2, 1, 3])  # two swapped in r2
    # |0|+|0|+|0|+|1|+|1|+|0| = 2 over 6 rows
    assert math.isclose(position_mae(TRUE_RANK, pred), 2 / 6, abs_tol=1e-12)
    assert position_mae(TRUE_RANK, TRUE_RANK) == 0.0


def test_spearman_perfect_and_reversed() -> None:
    assert math.isclose(spearman_per_race(RACE_IDS, TRUE_RANK, TRUE_RANK), 1.0, abs_tol=1e-12)
    reversed_pred = np.array([3, 2, 1, 3, 2, 1])
    assert math.isclose(spearman_per_race(RACE_IDS, TRUE_RANK, reversed_pred), -1.0, abs_tol=1e-12)


def test_topk_hit_rate_k1() -> None:
    # r1 predicted leader correct, r2 predicted leader wrong -> 0.5 mean.
    pred = np.array([1, 2, 3, 2, 1, 3])
    assert math.isclose(topk_hit_rate(RACE_IDS, TRUE_RANK, pred, 1), 0.5, abs_tol=1e-12)


def test_topk_hit_rate_k3_full_overlap() -> None:
    # With exactly three drivers, any permutation has the same top-3 set -> 1.0.
    pred = np.array([3, 2, 1, 2, 3, 1])
    assert math.isclose(topk_hit_rate(RACE_IDS, TRUE_RANK, pred, 3), 1.0, abs_tol=1e-12)


def test_exact_rate() -> None:
    pred = np.array([1, 2, 3, 2, 1, 3])  # 4 of 6 slots exact
    assert math.isclose(exact_rate(TRUE_RANK, pred), 4 / 6, abs_tol=1e-12)


def test_paired_bootstrap_mae_ci_detects_better_model() -> None:
    """A = perfect order, B = reversed. A's MAE is always lower, so the CI for
    MAE(A) - MAE(B) is entirely below zero."""
    pred_a = TRUE_RANK.copy()
    pred_b = np.array([3, 2, 1, 3, 2, 1])
    lo, hi = paired_bootstrap_mae_ci(RACE_IDS, TRUE_RANK, pred_a, pred_b)
    assert hi < 0.0, f"expected fully-negative CI, got [{lo:+.3f}, {hi:+.3f}]"
