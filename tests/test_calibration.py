"""Tests for the isotonic calibrator and the ECE metric."""

from __future__ import annotations

import numpy as np

from src.eval.calibration import IsotonicCalibrator, ece


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
