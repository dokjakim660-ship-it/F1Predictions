"""Tests for the isotonic calibrator, the ECE metric, and teammate pair-norm."""

from __future__ import annotations

import numpy as np

from src.eval.calibration import (
    DEFAULT_CALIB_EPS,
    IsotonicCalibrator,
    ece,
    pair_normalize_teammate,
)


def test_ece_low_for_a_calibrated_generator() -> None:
    # Outcomes drawn with probability == y_prob => well calibrated by construction.
    rng = np.random.default_rng(0)
    y_prob = rng.uniform(0.0, 1.0, 5000)
    y_true = (rng.uniform(0.0, 1.0, 5000) < y_prob).astype(int)
    assert ece(y_true, y_prob) < 0.05


def test_ece_high_for_a_miscalibrated_generator() -> None:
    # Predict 0.9 everywhere but the event almost never happens.
    rng = np.random.default_rng(1)
    y_prob = np.full(2000, 0.9)
    y_true = (rng.uniform(0.0, 1.0, 2000) < 0.1).astype(int)
    assert ece(y_true, y_prob) > 0.5


def test_isotonic_calibrator_is_monotonic_and_bounded() -> None:
    rng = np.random.default_rng(2)
    raw = rng.uniform(0.0, 1.0, 2000)
    y_true = (rng.uniform(0.0, 1.0, 2000) < raw).astype(int)
    calibrator = IsotonicCalibrator.fit(raw, y_true)

    out = calibrator.transform(np.linspace(0.0, 1.0, 50))
    assert np.all(np.diff(out) >= -1e-9)  # non-decreasing
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_isotonic_calibrator_default_eps_clips_extremes() -> None:
    # All raws in [0.9, 1.0] with all y_true=1 would push the top bin to 1.0
    # under the old y_max=1.0. With DEFAULT_CALIB_EPS=0.02 the ceiling is 0.98.
    raw = np.linspace(0.9, 1.0, 100)
    y_true = np.ones(100, dtype=int)
    calibrator = IsotonicCalibrator.fit(raw, y_true)

    out = calibrator.transform(np.linspace(0.0, 1.0, 200))
    assert out.max() <= 1.0 - DEFAULT_CALIB_EPS + 1e-9
    assert out.min() >= DEFAULT_CALIB_EPS - 1e-9


def test_isotonic_calibrator_eps_zero_recovers_legacy_bounds() -> None:
    # Explicit opt-out: callers that want the pre-Phase 3.8 unclipped behaviour
    # can pass eps=0.0. We expect a 1.0 to be reachable again.
    raw = np.concatenate([np.zeros(50), np.ones(50)])
    y_true = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
    calibrator = IsotonicCalibrator.fit(raw, y_true, eps=0.0)
    assert calibrator.transform(np.array([1.0]))[0] == 1.0
    assert calibrator.transform(np.array([0.0]))[0] == 0.0


def test_pair_normalize_teammate_sums_to_one() -> None:
    # Mercedes-style pair: independent calibrated probs sum to >1; after
    # pair-norm both legs should add to exactly 1.
    probs = np.array([1.00, 0.42, 0.95, 0.30])
    constructors = np.array(["mercedes", "mercedes", "red_bull", "red_bull"])
    out = pair_normalize_teammate(probs, constructors)
    assert abs(out[0] + out[1] - 1.0) < 1e-12
    assert abs(out[2] + out[3] - 1.0) < 1e-12
    # Russell stays the favorite; the ratio is preserved.
    assert out[0] > out[1]
    assert abs(out[0] / out[1] - probs[0] / probs[1]) < 1e-12


def test_pair_normalize_teammate_passes_through_singletons() -> None:
    # A constructor with only one row (mid-season substitution leaves a single
    # leg) has no pair to normalize against -- leave it untouched.
    probs = np.array([0.80, 0.20, 0.55])
    constructors = np.array(["mercedes", "mercedes", "alpine"])
    out = pair_normalize_teammate(probs, constructors)
    assert abs(out[0] + out[1] - 1.0) < 1e-12
    assert out[2] == 0.55


def test_pair_normalize_teammate_passes_through_triples() -> None:
    # Defensive: if a constructor somehow has 3 rows (data bug), we don't try
    # to invent a normalization rule -- pass through unchanged.
    probs = np.array([0.6, 0.5, 0.4])
    constructors = np.array(["x", "x", "x"])
    out = pair_normalize_teammate(probs, constructors)
    np.testing.assert_array_equal(out, probs)


def test_pair_normalize_teammate_passes_through_zero_sum() -> None:
    # Both calibrated probs at 0 -> can't divide; leave them untouched rather
    # than producing NaN.
    probs = np.array([0.0, 0.0, 0.7, 0.3])
    constructors = np.array(["a", "a", "b", "b"])
    out = pair_normalize_teammate(probs, constructors)
    assert out[0] == 0.0 and out[1] == 0.0
    assert abs(out[2] + out[3] - 1.0) < 1e-12
