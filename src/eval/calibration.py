"""Probability calibration: isotonic regression + calibration diagnostics.

Tree models and class reweighting can leave predict_proba miscalibrated -- the
ranking stays fine but the absolute probabilities drift. For EV betting the
absolute number must be trustworthy, so every MVP model gets an isotonic
calibration layer fitted on leak-free out-of-fold predictions.

- IsotonicCalibrator -- monotonic raw-prob -> calibrated-prob mapping.
- ece()              -- Expected Calibration Error, the headline calib metric.
- calibration_plot() -- reliability diagram, saved as an MLflow artefact.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

matplotlib.use("Agg")  # headless backend -- write files, never open a window


# Default floor/ceiling for the isotonic output. With ~2700 dev rows, high-prob
# bins where every OOF example happened to be correct collapse to exactly 1.0
# (mirror at the low end). That overstates confidence — a 100% F1 prediction
# ignores DNF/safety-car risk, and a 0% prediction ignores chaos upside.
# eps=0.02 clips both tails to [0.02, 0.98], which is ~50:1 odds — beyond what
# bookmakers price anyway.
DEFAULT_CALIB_EPS = 0.02


@dataclass
class IsotonicCalibrator:
    """Monotonic mapping from raw model probabilities to calibrated ones."""

    iso: IsotonicRegression

    @classmethod
    def fit(
        cls,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        *,
        eps: float = DEFAULT_CALIB_EPS,
    ) -> IsotonicCalibrator:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=eps, y_max=1.0 - eps)
        iso.fit(np.asarray(y_prob, dtype=float), np.asarray(y_true, dtype=float))
        return cls(iso=iso)

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        return self.iso.transform(np.asarray(y_prob, dtype=float))


def pair_normalize_teammate(
    probs: np.ndarray,
    constructor_ids: np.ndarray,
) -> np.ndarray:
    """Normalize per-driver teammate probabilities so each constructor pair sums to 1.

    For the teammate target ("driver beats their team-mate"), the per-driver
    isotonic calibrator treats each row independently — so a Mercedes pair can
    end up at P(Russell)=1.00 + P(Antonelli)=0.42 = 1.42, which is logically
    impossible (modulo double-DNF, ignored here). This rescales each pair so
    that P_A + P_B = 1.

    Constructors with !=2 rows (e.g., mid-season substitution leaves a single
    row) are passed through unchanged — there is no pair to normalize against.
    Pairs whose sum is 0 are also passed through (cannot divide by zero); that
    only happens if both legs were calibrated to the floor.
    """
    probs = np.asarray(probs, dtype=float).copy()
    constructor_ids = np.asarray(constructor_ids)
    df = pd.DataFrame({"c": constructor_ids, "p": probs, "i": np.arange(len(probs))})
    for _, grp in df.groupby("c", sort=False):
        if len(grp) != 2:
            continue
        s = float(grp["p"].sum())
        if s <= 0.0:
            continue
        probs[grp["i"].to_numpy()] = grp["p"].to_numpy() / s
    return probs


def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: mean |confidence - accuracy| over prob bins.

    Lower is better; a perfectly calibrated model scores 0. The roadmap target
    is ECE < 0.05.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    total = len(y_prob)
    err = 0.0
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        err += (n / total) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(err)


def calibration_plot(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
    n_bins: int = 10,
    title: str = "Reliability diagram (test set)",
) -> Path:
    """Reliability diagram. `curves` maps a label -> (y_true, y_prob)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfectly calibrated")
    for label, (y_true, y_prob) in curves.items():
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
        xs: list[float] = []
        ys: list[float] = []
        for b in range(n_bins):
            mask = idx == b
            if not mask.any():
                continue
            xs.append(float(y_prob[mask].mean()))
            ys.append(float(y_true[mask].mean()))
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed podium frequency")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
