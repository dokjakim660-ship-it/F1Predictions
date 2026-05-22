"""Walk-forward validation harness for the podium models.

Expanding-window cross-validation per PLANNING.md section 7. Four folds: each
trains on every race before a cutoff and validates on the following ~6-month
window. The holdout test set (races on/after 2024-07-01) is split off by
`split_dev_test` and never appears in any fold -- Optuna and model selection
only ever see the dev folds.

Race-group integrity is automatic: a race happens on a single date, so a
date-boundary split cannot place one race_id in both train and val.

Objective: mean(Brier) + 0.2 * std(Brier) across folds, LOWER is better. This
is `+`, not the `-` written in PLANNING.md section 7 -- the doc's stated goal
is to penalise unstable models, and for a minimised loss that means adding the
std term, not subtracting it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.eval.metrics import brier

# Races on/after this date are the holdout test set. Optuna NEVER sees them.
TEST_CUTOFF = pd.Timestamp("2024-07-01")

# Each fold is (train_end, val_end): train = race_date < train_end,
# val = train_end <= race_date < val_end. The train window expands every fold.
_FOLD_BOUNDARIES: tuple[tuple[str, str], ...] = (
    ("2022-07-01", "2023-01-01"),
    ("2023-01-01", "2023-07-01"),
    ("2023-07-01", "2024-01-01"),
    ("2024-01-01", "2024-07-01"),
)

_STD_PENALTY = 0.2

# fit_predict(train_df, val_df) -> 1-D array of P(podium) for val_df rows.
FitPredictFn = Callable[[pd.DataFrame, pd.DataFrame], np.ndarray]


def _race_dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["race_date"])


def split_dev_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into dev (where folds live) and the untouchable holdout test set."""
    dates = _race_dates(df)
    dev = df[dates < TEST_CUTOFF].copy()
    test = df[dates >= TEST_CUTOFF].copy()
    return dev, test


def make_folds(df: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return [(train_df, val_df), ...] for the four expanding-window folds."""
    dates = _race_dates(df)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for train_end, val_end in _FOLD_BOUNDARIES:
        t_end = pd.Timestamp(train_end)
        v_end = pd.Timestamp(val_end)
        train = df[dates < t_end].copy()
        val = df[(dates >= t_end) & (dates < v_end)].copy()
        folds.append((train, val))
    return folds


@dataclass
class WalkForwardResult:
    fold_briers: list[float]
    fold_val_sizes: list[int]

    @property
    def mean_brier(self) -> float:
        return float(np.mean(self.fold_briers))

    @property
    def std_brier(self) -> float:
        return float(np.std(self.fold_briers))

    @property
    def objective(self) -> float:
        """mean + 0.2*std across folds. LOWER is better; Optuna minimises this."""
        return self.mean_brier + _STD_PENALTY * self.std_brier

    def __str__(self) -> str:
        per_fold = "  ".join(f"f{i + 1}={b:.4f}" for i, b in enumerate(self.fold_briers))
        return (
            f"brier per fold: {per_fold}  |  mean={self.mean_brier:.4f} "
            f"std={self.std_brier:.4f}  objective={self.objective:.4f}"
        )


@dataclass
class OOFPredictions:
    """Out-of-fold predictions concatenated across every fold's val window."""

    race_ids: np.ndarray
    y_true: np.ndarray
    y_prob: np.ndarray


def oof_predictions(
    df: pd.DataFrame,
    fit_predict: FitPredictFn,
    target_col: str = "target_podium",
) -> OOFPredictions:
    """Collect val predictions from every fold.

    Each row is scored by a model that never trained on it, so the result is a
    leak-free dataset for fitting a calibration mapping.
    """
    race_ids: list[np.ndarray] = []
    y_true: list[np.ndarray] = []
    y_prob: list[np.ndarray] = []
    for train, val in make_folds(df):
        if train.empty or val.empty:
            raise RuntimeError("Empty walk-forward fold -- check the feature table coverage.")
        prob = np.asarray(fit_predict(train, val), dtype=float)
        if len(prob) != len(val):
            raise ValueError(f"fit_predict returned {len(prob)} probs for {len(val)} val rows")
        race_ids.append(val["race_id"].to_numpy())
        y_true.append(val[target_col].astype(int).to_numpy())
        y_prob.append(prob)
    return OOFPredictions(
        race_ids=np.concatenate(race_ids),
        y_true=np.concatenate(y_true),
        y_prob=np.concatenate(y_prob),
    )


def evaluate(
    df: pd.DataFrame,
    fit_predict: FitPredictFn,
    target_col: str = "target_podium",
) -> WalkForwardResult:
    """Run `fit_predict` across all folds and aggregate the per-fold Brier score.

    `fit_predict(train_df, val_df)` must fit on train_df and return a 1-D array
    of P(podium) for val_df, aligned to val_df's row order.
    """
    briers: list[float] = []
    sizes: list[int] = []
    for train, val in make_folds(df):
        if train.empty or val.empty:
            raise RuntimeError(
                "Empty walk-forward fold -- does the feature table cover 2018..2024-06?"
            )
        y_prob = np.asarray(fit_predict(train, val), dtype=float)
        if len(y_prob) != len(val):
            raise ValueError(
                f"fit_predict returned {len(y_prob)} probabilities for {len(val)} val rows"
            )
        y_true = val[target_col].astype(int).to_numpy()
        briers.append(brier(y_true, y_prob))
        sizes.append(len(val))
    return WalkForwardResult(fold_briers=briers, fold_val_sizes=sizes)
