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
from sklearn.linear_model import LogisticRegression

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


@dataclass
class BetaCalibrator:
    """Beta calibration (Kull et al. 2017): a smooth 3-parameter alternative to
    isotonic. Fits a logistic regression on [ln(p), -ln(1-p)], which spans the
    beta-distribution family. Unlike isotonic it never produces flat steps or
    collapses high-prob bins to a single value, so it tends to give better-
    behaved tails when calibration data is thin -- exactly our small-N regime.
    """

    lr: LogisticRegression
    eps: float = DEFAULT_CALIB_EPS

    @staticmethod
    def _features(y_prob: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1.0 - 1e-6)
        return np.column_stack([np.log(p), -np.log1p(-p)])

    @classmethod
    def fit(
        cls, y_prob: np.ndarray, y_true: np.ndarray, *, eps: float = DEFAULT_CALIB_EPS
    ) -> BetaCalibrator:
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(cls._features(y_prob), np.asarray(y_true, dtype=int))
        return cls(lr=lr, eps=eps)

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        out = self.lr.predict_proba(self._features(y_prob))[:, 1]
        return np.clip(out, self.eps, 1.0 - self.eps)


@dataclass
class VennAbersCalibrator:
    """Inductive Venn-Abers predictor (Vovk & Petej 2014). For a test score s it
    refits isotonic twice on the calibration set augmented with (s, 0) and
    (s, 1), reads p0 and p1 at s, and returns p = p1 / (1 - p0 + p1). This is
    automatically perfectly calibrated in the Venn sense and robust at small N,
    at the cost of carrying the calibration set and refitting per query (cheap:
    O(n_unique) isotonic fits; trivial for the ~22-row inference path).
    """

    cal_scores: np.ndarray
    cal_labels: np.ndarray
    eps: float = DEFAULT_CALIB_EPS

    @classmethod
    def fit(
        cls, y_prob: np.ndarray, y_true: np.ndarray, *, eps: float = DEFAULT_CALIB_EPS
    ) -> VennAbersCalibrator:
        return cls(
            cal_scores=np.asarray(y_prob, dtype=float),
            cal_labels=np.asarray(y_true, dtype=float),
            eps=eps,
        )

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        x = np.asarray(y_prob, dtype=float)
        uniq, inv = np.unique(x, return_inverse=True)
        out = np.empty(len(uniq), dtype=float)
        labels0 = np.append(self.cal_labels, 0.0)
        labels1 = np.append(self.cal_labels, 1.0)
        for i, v in enumerate(uniq):
            scores = np.append(self.cal_scores, v)
            iso0 = IsotonicRegression(out_of_bounds="clip").fit(scores, labels0)
            iso1 = IsotonicRegression(out_of_bounds="clip").fit(scores, labels1)
            p0 = float(iso0.predict([v])[0])
            p1 = float(iso1.predict([v])[0])
            denom = 1.0 - p0 + p1
            out[i] = p1 / denom if denom > 0 else p1
        return np.clip(out[inv], self.eps, 1.0 - self.eps)


# Per-target deployed calibration policy, chosen by the calibration A/B
# (src/eval/calib_ab.py) on the sealed holdout, overall and per era slice.
#   podium   : None (raw) -- LGBM/Ensemble are already well-calibrated raw
#              (ECE ~0.024-0.026, best Brier 0.0624). Isotonic/beta/Venn-Abers
#              all RAISED ECE and Brier, worst on the 2026 reg-reset slice.
#   teammate : Venn-Abers on the deployed LogReg -- ties the best Brier
#              (0.1953), nearly halves ECE (0.0305 -> 0.0201), and is the most
#              robust on the 2026 slice (0.0864 vs isotonic 0.1152).
# A value of None means raw probabilities ship as the calibrated column.
DEPLOYED_CALIBRATOR: dict[str, type | None] = {
    "podium": None,
    "teammate": VennAbersCalibrator,
}


def deployed_calibrate(
    target_short: str,
    raw_prob: np.ndarray,
    oof_prob: np.ndarray,
    oof_true: np.ndarray,
) -> np.ndarray:
    """Apply the deployed per-target calibrator (see DEPLOYED_CALIBRATOR).

    Fits the chosen calibrator on leak-free OOF dev predictions and transforms
    the raw probabilities. If the policy is None (raw), returns raw_prob
    unchanged -- callers should still skip the OOF computation in that case.
    """
    cls = DEPLOYED_CALIBRATOR.get(target_short)
    if cls is None:
        return np.asarray(raw_prob, dtype=float)
    return cls.fit(oof_prob, oof_true).transform(raw_prob)


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
