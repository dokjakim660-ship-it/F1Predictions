"""Phase 5 ranking models -- predict the full within-race order.

Two tasks, symmetric to the project's two existing timing contexts:

- quali: predict each driver's qualifying position (target_quali_rank) BEFORE
  qualifying, from the reduced pre-quali feature sets (pre_weekend / post_fp2).
  The full-order generalisation of the four binary quali markets.
- race:  predict each driver's race finishing position (target_race_rank,
  DNF-aware) on Saturday evening, from the full pre-race feature set (real grid
  + quali included).

Two ML methods are built and compared head-to-head (user decision):

- (A) position regression -- XGB/LGBM/Ridge regress the numeric position, then
  we argsort within each race -> rank 1..N. The predicted value is an
  interpretable "expected place".
- (B) learning-to-rank (LambdaMART) -- XGBRanker (rank:ndcg) / LGBMRanker
  (lambdarank) with group=race. Optimises the ordering directly; emits an
  abstract score, higher = better.

Both are scored with rank metrics (position MAE, Spearman, top-k) instead of
Brier -- no calibration, since the output is an order, not a probability. A
trivial domain baseline anchors each task: grid order for race, recent average
qualifying position for quali. The A/B verdict uses a paired race-bootstrap CI
on position MAE, mirroring the prequali Go/No-Go.

Run: `python -m src.models.rank eval --task race`
     `python -m src.models.rank select --task quali`   (dev-only model pick)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.eval.rank_metrics import (
    paired_bootstrap_mae_ci,
    position_mae,
    summary,
    to_race_ranks,
)
from src.eval.walk_forward import FitPredictFn, oof_predictions
from src.features.build import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    PRE_QUALI_FEATURE_SETS,
    TARGET_QUALI_RANK,
    TARGET_RACE_RANK,
)
from src.models.mvp import (
    _LGBM_DEFAULT_PARAMS,
    _RANDOM_STATE,
    _XGB_DEFAULT_PARAMS,
    _logreg_matrix,
    prepare_dev_test,
)

# A model emits one score per driver; `ascending` says how to turn it into a
# rank (True = lower score is a better place, the natural sense for a regressed
# position and for the baseline feature; False = higher score is better, the
# sense of a LambdaMART ranker output).
ASCENDING_POSITION = True
ASCENDING_RANKER = False


@dataclass(frozen=True)
class RankTask:
    name: str
    target: str
    # mode name -> numeric feature list. quali has two timing modes; race has one.
    modes: dict[str, list[str]]
    baseline_name: str
    baseline_feature: str  # order drivers by this column (ascending) as the anchor


RANK_TASKS: dict[str, RankTask] = {
    "quali": RankTask(
        name="quali",
        target=TARGET_QUALI_RANK,
        modes=PRE_QUALI_FEATURE_SETS,
        baseline_name="RecentQualiForm",
        baseline_feature="driver_form_quali_pos_l5",
    ),
    "race": RankTask(
        name="race",
        target=TARGET_RACE_RANK,
        modes={"race": FEATURE_COLUMNS},
        baseline_name="GridOrder",
        baseline_feature="grid_effective",
    ),
}

# Derived regression ensemble (mean of the two tree regressors' predicted
# positions) + display/selection groupings.
_REG_MODELS = ("Ridge", "XGBReg", "LGBMReg", "RegEnsemble")
_RANK_MODELS = ("XGBRanker", "LGBMRanker")
_SELECTABLE = (*_REG_MODELS, *_RANK_MODELS)


# --- model factories -----------------------------------------------------


def make_feature_order_fit_predict(feature: str) -> FitPredictFn:
    """Baseline: order drivers by a single column (lower = better place).

    Rookies with a NaN history value are median-imputed from the train slice so
    they sort into the middle of the grid instead of the front or back.
    """

    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        med = train[feature].median()
        return val[feature].fillna(med).to_numpy(dtype=float)

    return fit_predict


def make_ridge_fit_predict(target_col: str, numeric_features: list[str]) -> FitPredictFn:
    """Linear position regression -- the Ridge analogue of the LogReg reference.
    Reuses mvp._logreg_matrix (median-impute + one-hot track_id) + a scaler."""

    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        medians = train[numeric_features].median(numeric_only=True)
        x_train = _logreg_matrix(train, medians, numeric_features)
        x_val = _logreg_matrix(val, medians, numeric_features)
        scaler = StandardScaler().fit(x_train)
        model = Ridge().fit(scaler.transform(x_train), train[target_col].astype(float))
        return model.predict(scaler.transform(x_val))

    return fit_predict


def make_xgb_regressor_fit_predict(target_col: str, numeric_features: list[str]) -> FitPredictFn:
    tree_features = list(numeric_features) + CATEGORICAL_COLUMNS

    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        model = xgb.XGBRegressor(
            **_XGB_DEFAULT_PARAMS,
            objective="reg:squarederror",
            enable_categorical=True,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(train[tree_features], train[target_col].astype(float))
        return model.predict(val[tree_features])

    return fit_predict


def make_lgbm_regressor_fit_predict(target_col: str, numeric_features: list[str]) -> FitPredictFn:
    tree_features = list(numeric_features) + CATEGORICAL_COLUMNS

    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        model = lgb.LGBMRegressor(
            **_LGBM_DEFAULT_PARAMS,
            objective="regression",
            random_state=_RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(train[tree_features], train[target_col].astype(float))
        return model.predict(val[tree_features])

    return fit_predict


def _relevance(df: pd.DataFrame, target_col: str) -> np.ndarray:
    """LambdaMART relevance from the rank target: best driver gets the highest
    gain. rel = (race size - rank), so rank 1 -> N-1 and the last driver -> 0.
    Non-negative ints within each race, <= ~21, well inside LightGBM's default
    label_gain table."""
    grp_max = df.groupby("race_id", sort=False)[target_col].transform("max")
    return (grp_max - df[target_col]).round().astype(int).to_numpy()


def make_xgb_ranker_fit_predict(target_col: str, numeric_features: list[str]) -> FitPredictFn:
    tree_features = list(numeric_features) + CATEGORICAL_COLUMNS

    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        # XGBRanker wants qid sorted (non-decreasing); sorting by race_id makes the
        # category codes monotonic, so qid is valid and groups stay contiguous.
        tr = train.sort_values("race_id")
        qid = tr["race_id"].astype("category").cat.codes.to_numpy()
        model = xgb.XGBRanker(
            **_XGB_DEFAULT_PARAMS,
            objective="rank:ndcg",
            enable_categorical=True,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(tr[tree_features], _relevance(tr, target_col), qid=qid)
        return model.predict(val[tree_features])

    return fit_predict


def make_lgbm_ranker_fit_predict(target_col: str, numeric_features: list[str]) -> FitPredictFn:
    tree_features = list(numeric_features) + CATEGORICAL_COLUMNS

    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        tr = train.sort_values("race_id")
        group = tr.groupby("race_id", sort=False).size().to_numpy()
        model = lgb.LGBMRanker(
            **_LGBM_DEFAULT_PARAMS,
            objective="lambdarank",
            random_state=_RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(tr[tree_features], _relevance(tr, target_col), group=group)
        return model.predict(val[tree_features])

    return fit_predict


# (name, fit_predict, ascending) for the base models -- RegEnsemble is derived
# from XGBReg + LGBMReg afterwards, so it is not listed here.
def _specs(task: str, numeric_features: list[str]) -> list[tuple[str, FitPredictFn, bool]]:
    cfg = RANK_TASKS[task]
    t = cfg.target
    baseline = make_feature_order_fit_predict(cfg.baseline_feature)
    return [
        (cfg.baseline_name, baseline, ASCENDING_POSITION),
        ("Ridge", make_ridge_fit_predict(t, numeric_features), ASCENDING_POSITION),
        ("XGBReg", make_xgb_regressor_fit_predict(t, numeric_features), ASCENDING_POSITION),
        ("LGBMReg", make_lgbm_regressor_fit_predict(t, numeric_features), ASCENDING_POSITION),
        ("XGBRanker", make_xgb_ranker_fit_predict(t, numeric_features), ASCENDING_RANKER),
        ("LGBMRanker", make_lgbm_ranker_fit_predict(t, numeric_features), ASCENDING_RANKER),
    ]


def _display_order(task: str) -> tuple[str, ...]:
    base = RANK_TASKS[task].baseline_name
    return (base, "Ridge", "XGBReg", "LGBMReg", "RegEnsemble", *_RANK_MODELS)


# --- evaluation ----------------------------------------------------------


@dataclass
class RankEval:
    mode: str
    task: str
    name: str
    pos_mae: float
    spearman: float
    top1: float
    top3: float
    exact: float
    pred_rank: np.ndarray


def _ensemble_score(score_by: dict[str, np.ndarray]) -> np.ndarray:
    """Mean of the two tree regressors' predicted positions (ascending sense)."""
    return (score_by["XGBReg"] + score_by["LGBMReg"]) / 2.0


def evaluate_task_mode(task: str, mode: str) -> tuple[list[RankEval], np.ndarray, np.ndarray]:
    """Train every model on dev, predict the holdout, return per-model rank evals
    plus the shared (race_ids, true_rank) for the bootstrap verdict."""
    cfg = RANK_TASKS[task]
    feats = cfg.modes[mode]
    dev, test = prepare_dev_test(cfg.target)
    if dev.empty or test.empty:
        raise RuntimeError("empty dev/test split -- run `just build` first")

    race_ids = test["race_id"].to_numpy()
    true_rank = test[cfg.target].to_numpy()

    score_by: dict[str, np.ndarray] = {}
    asc_by: dict[str, bool] = {}
    for name, fit_predict, asc in _specs(task, feats):
        score_by[name] = np.asarray(fit_predict(dev, test), dtype=float)
        asc_by[name] = asc
    score_by["RegEnsemble"] = _ensemble_score(score_by)
    asc_by["RegEnsemble"] = ASCENDING_POSITION

    evals: list[RankEval] = []
    for name in _display_order(task):
        pred = to_race_ranks(race_ids, score_by[name], asc_by[name])
        s = summary(name, race_ids, true_rank, pred)
        evals.append(
            RankEval(
                mode=mode,
                task=task,
                name=name,
                pos_mae=s["pos_mae"],
                spearman=s["spearman"],
                top1=s["top1"],
                top3=s["top3"],
                exact=s["exact"],
                pred_rank=pred,
            )
        )
    return evals, race_ids, true_rank


def dev_pos_mae(task: str, mode: str) -> dict[str, float]:
    """Walk-forward OOF position MAE on the DEV set for every selectable model.

    Model selection must never touch the holdout (same rule as prequali): the
    pick is made here on dev OOF; the holdout in evaluate_task_mode only confirms.
    """
    cfg = RANK_TASKS[task]
    feats = cfg.modes[mode]
    dev, _ = prepare_dev_test(cfg.target)

    specs = _specs(task, feats)
    oof_by = {name: oof_predictions(dev, fp, target_col=cfg.target) for name, fp, _ in specs}
    asc_by = {name: asc for name, _, asc in specs}

    out: dict[str, float] = {}
    for name, oof in oof_by.items():
        pred = to_race_ranks(oof.race_ids, oof.y_prob, asc_by[name])
        out[name] = position_mae(oof.y_true, pred)
    ens_score = _ensemble_score({n: oof_by[n].y_prob for n in ("XGBReg", "LGBMReg")})
    ens_pred = to_race_ranks(oof_by["XGBReg"].race_ids, ens_score, ASCENDING_POSITION)
    out["RegEnsemble"] = position_mae(oof_by["XGBReg"].y_true, ens_pred)
    return out


def select_best(task: str, mode: str) -> tuple[str, float]:
    maes = dev_pos_mae(task, mode)
    name = min(_SELECTABLE, key=lambda n: maes[n])
    return name, maes[name]


# --- reporting -----------------------------------------------------------


def _print_task_block(task: str) -> None:
    cfg = RANK_TASKS[task]
    _, test = prepare_dev_test(cfg.target)
    n_rows = len(test)
    n_races = test["race_id"].nunique()

    print()
    print("=" * 96)
    print(
        f"Rank holdout eval -- task = {task}  (test: {n_rows} rows, {n_races} races; "
        f"lower pos_mae = better, higher spearman/top-k = better)"
    )
    print("=" * 96)
    print(
        f"{'mode':<12s}  {'model':<16s}  {'pos_mae':>8s}  {'spearman':>8s}  "
        f"{'top1':>6s}  {'top3':>6s}  {'exact':>6s}"
    )
    print("-" * 96)

    for mode in cfg.modes:
        evals, race_ids, true_rank = evaluate_task_mode(task, mode)
        by_name = {e.name: e for e in evals}
        for e in evals:
            tag = "  <- baseline" if e.name == cfg.baseline_name else ""
            print(
                f"{mode:<12s}  {e.name:<16s}  {e.pos_mae:>8.3f}  {e.spearman:>8.3f}  "
                f"{e.top1:>6.3f}  {e.top3:>6.3f}  {e.exact:>6.3f}{tag}"
            )
        print("-" * 96)

        # Verdict 1: best overall model vs the trivial baseline.
        best = min(_SELECTABLE, key=lambda n: by_name[n].pos_mae)
        base = by_name[cfg.baseline_name]
        lo, hi = paired_bootstrap_mae_ci(
            race_ids, true_rank, by_name[best].pred_rank, base.pred_rank
        )
        diff = by_name[best].pos_mae - base.pos_mae
        if hi < 0:
            verdict = f"{best} beats {cfg.baseline_name} at 95%"
        elif diff < 0:
            verdict = f"{best} ahead but NOT significant (CI straddles 0)"
        else:
            verdict = f"{best} does NOT beat {cfg.baseline_name}"
        print(
            f"[{mode}] best={best} pos_mae={by_name[best].pos_mae:.3f} vs "
            f"{cfg.baseline_name}={base.pos_mae:.3f}  diff={diff:+.3f}  "
            f"95% CI [{lo:+.3f}, {hi:+.3f}]  ->  {verdict}"
        )

        # Verdict 2: position-regression vs learning-to-rank (the A/B the user asked for).
        best_reg = min(_REG_MODELS, key=lambda n: by_name[n].pos_mae)
        best_rank = min(_RANK_MODELS, key=lambda n: by_name[n].pos_mae)
        lo2, hi2 = paired_bootstrap_mae_ci(
            race_ids, true_rank, by_name[best_reg].pred_rank, by_name[best_rank].pred_rank
        )
        d2 = by_name[best_reg].pos_mae - by_name[best_rank].pos_mae
        if hi2 < 0:
            ab = f"regression ({best_reg}) beats ranker ({best_rank}) at 95%"
        elif lo2 > 0:
            ab = f"ranker ({best_rank}) beats regression ({best_reg}) at 95%"
        else:
            ab = f"regression vs ranker not significant ({best_reg} vs {best_rank})"
        print(
            f"[{mode}] A/B regression vs ranker: {best_reg}={by_name[best_reg].pos_mae:.3f} "
            f"vs {best_rank}={by_name[best_rank].pos_mae:.3f}  diff={d2:+.3f}  "
            f"95% CI [{lo2:+.3f}, {hi2:+.3f}]  ->  {ab}"
        )
        print("-" * 96)


def run_select(task_arg: str) -> int:
    tasks = list(RANK_TASKS) if task_arg == "both" else [task_arg]
    print("=" * 84)
    print("Rank model selection -- DEV walk-forward position MAE (holdout untouched)")
    print("=" * 84)
    header = f"{'task':<8s}  {'mode':<12s}  " + "  ".join(f"{n:>11s}" for n in _SELECTABLE)
    print(header)
    print("-" * 84)
    for task in tasks:
        for mode in RANK_TASKS[task].modes:
            maes = dev_pos_mae(task, mode)
            best = min(_SELECTABLE, key=lambda n: maes[n])
            cells = "  ".join(
                (f"*{maes[n]:.3f}*" if n == best else f" {maes[n]:.3f} ").rjust(11)
                for n in _SELECTABLE
            )
            print(f"{task:<8s}  {mode:<12s}  {cells}")
    print("=" * 84)
    print("* = lowest dev position MAE (selected). Confirm on holdout with `rank eval`.")
    return 0


def run(task_arg: str) -> int:
    tasks = list(RANK_TASKS) if task_arg == "both" else [task_arg]
    for task in tasks:
        _print_task_block(task)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd, helptext in (
        ("eval", "Holdout rank metrics + A/B verdict (regression vs ranker vs baseline)."),
        ("select", "Best model per (task, mode) by DEV position MAE (holdout untouched)."),
    ):
        sp = sub.add_parser(cmd, help=helptext)
        sp.add_argument(
            "--task",
            choices=(*RANK_TASKS, "both"),
            default="both",
            help="Ranking task (default: both).",
        )
    args = p.parse_args(argv)

    if args.cmd == "eval":
        return run(args.task)
    if args.cmd == "select":
        return run_select(args.task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
