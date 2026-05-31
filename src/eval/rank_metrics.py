"""Evaluation metrics for full-grid RANKING (Phase 5).

The rest of the project predicts binary outcomes scored by Brier; the ranking
models in src/models/rank.py instead predict a complete within-race order, so
they need order-aware metrics. Every model (position-regression, LambdaMART
ranker, the trivial baseline) emits one continuous score per driver; this module
turns that into a per-race rank and scores it against the true order.

Metrics, in interpretability order:

1. position MAE  -- mean |predicted place - actual place|. "On average we are
   ~2.4 places off." The headline number.
2. Spearman      -- rank correlation per race, averaged. 1.0 = perfect order,
   0 = random, -1.0 = reversed. Since both vectors are already gap-free ranks,
   Spearman is just Pearson on them (no scipy needed).
3. top-k hit rate -- overlap of the predicted vs actual top-k set per race.
   k=1 is "did we name the pole-sitter / winner", k=3 the podium overlap.
4. exact rate    -- fraction of drivers placed in exactly the right slot.

All functions take 1-D arrays/Series of equal length plus race_ids to group by.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_race_ranks(
    race_ids: pd.Series | np.ndarray,
    scores: np.ndarray,
    ascending: bool = True,
) -> np.ndarray:
    """Turn per-driver model scores into a gap-free 1..N rank within each race.

    ascending=True  -- lower score is better (position-regression: a predicted
                       position of 1.2 should rank ahead of 3.8).
    ascending=False -- higher score is better (LambdaMART ranker scores, or any
                       probability where bigger = more likely to win).

    method="first" breaks ties by row order so every race resolves to exactly N
    distinct ranks 1..N, matching the true_rank targets.
    """
    df = pd.DataFrame({"race_id": np.asarray(race_ids), "score": np.asarray(scores, dtype=float)})
    ranks = df.groupby("race_id", sort=False)["score"].rank(method="first", ascending=ascending)
    return ranks.to_numpy()


def position_mae(true_rank: np.ndarray, pred_rank: np.ndarray) -> float:
    """Mean absolute error in finishing position, averaged over all drivers."""
    t = np.asarray(true_rank, dtype=float)
    p = np.asarray(pred_rank, dtype=float)
    return float(np.mean(np.abs(t - p)))


def spearman_per_race(
    race_ids: pd.Series | np.ndarray,
    true_rank: np.ndarray,
    pred_rank: np.ndarray,
) -> float:
    """Mean per-race rank correlation. Both inputs are already 1..N ranks, so
    Spearman reduces to the Pearson correlation of the two rank vectors.

    Races with fewer than two drivers (never happens in practice) are skipped.
    """
    df = pd.DataFrame(
        {"race_id": np.asarray(race_ids), "t": np.asarray(true_rank), "p": np.asarray(pred_rank)}
    )
    cors: list[float] = []
    for _, sub in df.groupby("race_id", sort=False):
        if len(sub) < 2:
            continue
        t = sub["t"].to_numpy(dtype=float)
        p = sub["p"].to_numpy(dtype=float)
        if t.std() == 0 or p.std() == 0:
            continue
        cors.append(float(np.corrcoef(t, p)[0, 1]))
    return float(np.mean(cors)) if cors else float("nan")


def topk_hit_rate(
    race_ids: pd.Series | np.ndarray,
    true_rank: np.ndarray,
    pred_rank: np.ndarray,
    k: int,
) -> float:
    """Average per-race overlap between the predicted and actual top-k sets.

    Per race: of the k drivers actually in the top k, what fraction did the
    model also place in its top k? k=1 -> exact pole/winner hit (0 or 1); k=3 ->
    podium overlap in thirds. Order within the top k is ignored -- this is a set
    metric.
    """
    df = pd.DataFrame(
        {"race_id": np.asarray(race_ids), "t": np.asarray(true_rank), "p": np.asarray(pred_rank)}
    )
    rates: list[float] = []
    for _, sub in df.groupby("race_id", sort=False):
        true_top = sub["t"] <= k
        n_true = int(true_top.sum())
        if n_true == 0:
            continue
        hits = int((true_top & (sub["p"] <= k)).sum())
        rates.append(hits / min(k, n_true))
    return float(np.mean(rates)) if rates else float("nan")


def exact_rate(true_rank: np.ndarray, pred_rank: np.ndarray) -> float:
    """Fraction of drivers placed in exactly the right slot."""
    t = np.asarray(true_rank, dtype=float)
    p = np.asarray(pred_rank, dtype=float)
    return float(np.mean(t == p))


def summary(
    name: str,
    race_ids: pd.Series | np.ndarray,
    true_rank: np.ndarray,
    pred_rank: np.ndarray,
) -> dict[str, float]:
    return {
        "model": name,
        "pos_mae": position_mae(true_rank, pred_rank),
        "spearman": spearman_per_race(race_ids, true_rank, pred_rank),
        "top1": topk_hit_rate(race_ids, true_rank, pred_rank, 1),
        "top3": topk_hit_rate(race_ids, true_rank, pred_rank, 3),
        "exact": exact_rate(true_rank, pred_rank),
        "n_obs": int(len(true_rank)),
        "n_races": int(pd.Series(np.asarray(race_ids)).nunique()),
    }


def format_summary_row(s: dict[str, float]) -> str:
    return (
        f"{s['model']:<20s}  pos_mae={s['pos_mae']:.3f}  spearman={s['spearman']:.3f}  "
        f"top1={s['top1']:.3f}  top3={s['top3']:.3f}  exact={s['exact']:.3f}"
    )


def paired_bootstrap_mae_ci(
    race_ids: pd.Series | np.ndarray,
    true_rank: np.ndarray,
    pred_a_rank: np.ndarray,
    pred_b_rank: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% CI for position_MAE(a) - position_MAE(b), resampling whole races.

    The ranking analogue of metrics.paired_bootstrap_brier_ci: races are the
    resampling unit so the CI respects race-level correlation. A fully-negative
    interval means model A's position MAE is significantly lower (A ranks
    better); an interval straddling zero means the gap is not significant at 95%.
    """
    race_ids = np.asarray(race_ids)
    t = np.asarray(true_rank, dtype=float)
    ae_a = np.abs(np.asarray(pred_a_rank, dtype=float) - t)
    ae_b = np.abs(np.asarray(pred_b_rank, dtype=float) - t)
    races = np.unique(race_ids)
    race_to_rows = {r: np.flatnonzero(race_ids == r) for r in races}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampled = rng.choice(races, size=len(races), replace=True)
        rows = np.concatenate([race_to_rows[r] for r in sampled])
        diffs[i] = ae_a[rows].mean() - ae_b[rows].mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
