"""Evaluation metrics for binary podium prediction.

Three metrics matter for this project, in priority order:

1. Brier score -- proper scoring rule, the Optuna objective for the MVP model.
2. Log-loss -- secondary, sensitive to confidently-wrong predictions.
3. Top-3 accuracy per race -- intuitive, what the user "feels" when watching:
   how often does the model's top-3 by probability match the actual podium?

All functions take 1-D arrays/Series of equal length. Top-3 accuracy needs
race_ids alongside so it can group predictions per race.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

_LOGLOSS_CLIP = 1e-9


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(y_prob, _LOGLOSS_CLIP, 1 - _LOGLOSS_CLIP)
    return float(log_loss(y_true, p, labels=[0, 1]))


def top3_accuracy_per_race(
    race_ids: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """Fraction of (race, actual-podium-driver) pairs where the driver was in the
    model's top-3 by probability for that race.

    Each race contributes 3 binary checks (one per actual podium driver). Races
    with fewer than 3 classified podium drivers (rare edge case) contribute
    proportionally fewer checks.
    """
    df = pd.DataFrame({"race_id": race_ids.values, "y_true": y_true, "y_prob": y_prob})
    correct = 0
    total = 0
    for _, sub in df.groupby("race_id", sort=False):
        top3_idx = sub["y_prob"].nlargest(3).index
        predicted_top3 = sub.loc[top3_idx]
        actual_podium_mask = sub["y_true"] == 1
        n_actual_podium = int(actual_podium_mask.sum())
        if n_actual_podium == 0:
            continue
        n_hits = int((predicted_top3["y_true"] == 1).sum())
        correct += n_hits
        total += min(n_actual_podium, 3)
    return correct / total if total > 0 else float("nan")


def summary(
    name: str,
    race_ids: pd.Series,
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    return {
        "model": name,
        "brier": brier(y_true, y_prob),
        "log_loss": logloss(y_true, y_prob),
        "top3_acc": top3_accuracy_per_race(race_ids, y_true, y_prob),
        "n_obs": int(len(y_true)),
        "n_races": int(pd.Series(race_ids).nunique()),
    }


def format_summary_row(s: dict[str, float]) -> str:
    return (
        f"{s['model']:<28s}  brier={s['brier']:.4f}  "
        f"log_loss={s['log_loss']:.4f}  top3_acc={s['top3_acc']:.3f}  "
        f"n_obs={s['n_obs']}  n_races={s['n_races']}"
    )
