"""Tests for the isotonic calibrator, the ECE metric, and teammate pair-norm."""

from __future__ import annotations

import numpy as np

from src.eval.calibration import (
    DEFAULT_CALIB_EPS,
    DEPLOYED_CALIBRATOR,
    BetaCalibrator,
    IsotonicCalibrator,
    VennAbersCalibrator,
    deployed_calibrate,
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


def _miscalibrated_overconfident(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw scores that are systematically over-confident: the true event rate is
    a shrunk version of the score, so a good calibrator must pull probs toward
    the centre and lower ECE."""
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.0, 1.0, n)
    true_p = 0.5 + 0.4 * (raw - 0.5)  # squashed toward 0.5
    y_true = (rng.uniform(0.0, 1.0, n) < true_p).astype(int)
    return raw, y_true


def test_beta_calibrator_bounded_and_improves_ece() -> None:
    raw, y_true = _miscalibrated_overconfident(4000, seed=10)
    cal = BetaCalibrator.fit(raw, y_true).transform(raw)
    assert cal.min() >= DEFAULT_CALIB_EPS - 1e-9
    assert cal.max() <= 1.0 - DEFAULT_CALIB_EPS + 1e-9
    assert ece(y_true, cal) < ece(y_true, raw)


def test_beta_calibrator_is_monotonic() -> None:
    raw, y_true = _miscalibrated_overconfident(4000, seed=11)
    fitted = BetaCalibrator.fit(raw, y_true)
    out = fitted.transform(np.linspace(0.01, 0.99, 50))
    assert np.all(np.diff(out) >= -1e-9)  # logistic in log-odds features => monotone


def test_venn_abers_calibrator_bounded_and_improves_ece() -> None:
    raw, y_true = _miscalibrated_overconfident(3000, seed=12)
    cal = VennAbersCalibrator.fit(raw, y_true).transform(raw)
    assert cal.min() >= DEFAULT_CALIB_EPS - 1e-9
    assert cal.max() <= 1.0 - DEFAULT_CALIB_EPS + 1e-9
    assert ece(y_true, cal) < ece(y_true, raw)


def test_venn_abers_calibrator_is_monotonic() -> None:
    raw, y_true = _miscalibrated_overconfident(3000, seed=13)
    fitted = VennAbersCalibrator.fit(raw, y_true)
    out = fitted.transform(np.linspace(0.05, 0.95, 30))
    assert np.all(np.diff(out) >= -1e-9)  # IVAP is monotone in the score


def test_deployed_calibrate_podium_is_raw_passthrough() -> None:
    # Podium policy is None (raw) -- the helper must return raw unchanged and
    # must not require meaningful OOF inputs.
    assert DEPLOYED_CALIBRATOR["podium"] is None
    raw = np.array([0.1, 0.5, 0.9])
    out = deployed_calibrate("podium", raw, np.array([]), np.array([]))
    np.testing.assert_array_equal(out, raw)


def test_deployed_calibrate_teammate_uses_venn_abers() -> None:
    assert DEPLOYED_CALIBRATOR["teammate"] is VennAbersCalibrator
    raw, y_true = _miscalibrated_overconfident(2000, seed=14)
    out = deployed_calibrate("teammate", raw, raw, y_true)
    assert out.min() >= DEFAULT_CALIB_EPS - 1e-9
    assert out.max() <= 1.0 - DEFAULT_CALIB_EPS + 1e-9
    assert ece(y_true, out) < ece(y_true, raw)


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
