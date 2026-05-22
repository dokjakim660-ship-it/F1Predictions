"""Optuna hyperparameter tuning for the MVP tree models (podium + teammate H2H).

Increment C of the Phase 1.4 pipeline. Searches XGBoost and LightGBM
hyperparameters with a TPE sampler, minimising the walk-forward objective
(mean + 0.2*std of per-fold Brier) on the dev set. The holdout test set is
never touched here.

Each (target, model) pair gets its own SQLite-persisted study (models/optuna.db,
`load_if_exists=True`) so re-running adds trials instead of starting over, and
`tune show --target ...` inspects the best result later.

Run: `python -m src.models.tune xgboost --trials 50 --target podium`
     `python -m src.models.tune lightgbm --trials 50 --target teammate`
     `python -m src.models.tune show --target teammate`
"""

from __future__ import annotations

import argparse
import sys

import lightgbm as lgb
import optuna
import xgboost as xgb
from optuna.samplers import TPESampler

from src.eval.walk_forward import FitPredictFn, evaluate
from src.features.build import TARGET_PODIUM
from src.models.mvp import (
    DEFAULT_TARGET,
    TARGETS,
    TREE_FEATURES,
    prepare_dev_test,
)
from src.utils.paths import OPTUNA_DB

_RANDOM_STATE = 42
_MODELS = ("xgboost", "lightgbm")


def _storage() -> str:
    return f"sqlite:///{OPTUNA_DB.as_posix()}"


def _study_name(model: str, target_short: str) -> str:
    # _v2 keeps the Phase 1.2 feature-set bump from polluting the v1 trials;
    # the target prefix keeps podium and teammate searches strictly separated.
    return f"{target_short}_{model}_v2"


def _existing_studies() -> set[str]:
    if not OPTUNA_DB.exists():
        return set()
    return set(optuna.study.get_all_study_names(storage=_storage()))


def _suggest_xgb(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def _suggest_lgbm(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    }


def make_xgb_fit_predict(params: dict, target_col: str = TARGET_PODIUM) -> FitPredictFn:
    def fit_predict(train, val):
        model = xgb.XGBClassifier(
            **params,
            eval_metric="logloss",
            enable_categorical=True,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(train[TREE_FEATURES], train[target_col].astype(int))
        return model.predict_proba(val[TREE_FEATURES])[:, 1]

    return fit_predict


def make_lgbm_fit_predict(params: dict, target_col: str = TARGET_PODIUM) -> FitPredictFn:
    def fit_predict(train, val):
        model = lgb.LGBMClassifier(
            **params,
            subsample_freq=1,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        # LightGBM auto-detects the pandas category column as a categorical feature.
        model.fit(train[TREE_FEATURES], train[target_col].astype(int))
        return model.predict_proba(val[TREE_FEATURES])[:, 1]

    return fit_predict


def _make_fit_predict(model: str, params: dict, target_col: str = TARGET_PODIUM) -> FitPredictFn:
    factory = make_xgb_fit_predict if model == "xgboost" else make_lgbm_fit_predict
    return factory(params, target_col)


def _objective(trial: optuna.Trial, dev, model: str, target_col: str) -> float:
    params = _suggest_xgb(trial) if model == "xgboost" else _suggest_lgbm(trial)
    fit_predict = _make_fit_predict(model, params, target_col)
    result = evaluate(dev, fit_predict, target_col=target_col)
    trial.set_user_attr("mean_brier", result.mean_brier)
    trial.set_user_attr("std_brier", result.std_brier)
    return result.objective


def tune(model: str, n_trials: int, target_short: str = DEFAULT_TARGET) -> optuna.Study:
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
    if target_short not in TARGETS:
        raise ValueError(f"target must be one of {tuple(TARGETS)}, got {target_short!r}")
    OPTUNA_DB.parent.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    target_col = TARGETS[target_short]
    dev, _ = prepare_dev_test(target_col)

    study = optuna.create_study(
        study_name=_study_name(model, target_short),
        direction="minimize",
        storage=_storage(),
        sampler=TPESampler(seed=_RANDOM_STATE),
        load_if_exists=True,
    )
    study.optimize(lambda t: _objective(t, dev, model, target_col), n_trials=n_trials)
    return study


def best_params(model: str, target_short: str = DEFAULT_TARGET) -> dict:
    """Best hyperparameters for a tuned (target, model), loaded from its study."""
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
    name = _study_name(model, target_short)
    if name not in _existing_studies():
        raise RuntimeError(
            f"No Optuna study for ({target_short}, {model}). "
            f"Run: python -m src.models.tune {model} --trials 50 --target {target_short}"
        )
    study = optuna.load_study(study_name=name, storage=_storage())
    return dict(study.best_params)


def _print_best(model: str, study: optuna.Study) -> None:
    best = study.best_trial
    mean_b = best.user_attrs.get("mean_brier", float("nan"))
    std_b = best.user_attrs.get("std_brier", float("nan"))
    print(
        f"{model:<10s}  best obj={study.best_value:.4f}  "
        f"(mean={mean_b:.4f} std={std_b:.4f})  over {len(study.trials)} trials"
    )
    for key, value in study.best_params.items():
        shown = f"{value:.4g}" if isinstance(value, float) else value
        print(f"    {key} = {shown}")


def _show(target_short: str) -> None:
    existing = _existing_studies()
    print("=" * 78)
    print(
        f"Optuna studies ({target_short}) -- best walk-forward objective (mean + 0.2*std of Brier)"
    )
    print("=" * 78)
    for model in _MODELS:
        name = _study_name(model, target_short)
        if name not in existing:
            print(
                f"{model:<10s}  no study yet -- "
                f"run `tune {model} --trials 50 --target {target_short}`"
            )
            continue
        study = optuna.load_study(study_name=name, storage=_storage())
        _print_best(model, study)
    print("=" * 78)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for model in _MODELS:
        sp = sub.add_parser(model, help=f"Run Optuna trials for {model}.")
        sp.add_argument("--trials", type=int, default=50, help="Number of trials (default 50).")
        sp.add_argument(
            "--target",
            choices=tuple(TARGETS),
            default=DEFAULT_TARGET,
            help=f"Which target to tune (default: {DEFAULT_TARGET}).",
        )
    sh = sub.add_parser("show", help="Print the best trial of each persisted study.")
    sh.add_argument(
        "--target",
        choices=tuple(TARGETS),
        default=DEFAULT_TARGET,
        help=f"Which target's studies to show (default: {DEFAULT_TARGET}).",
    )
    args = p.parse_args(argv)

    if args.cmd in _MODELS:
        study = tune(args.cmd, args.trials, args.target)
        _print_best(args.cmd, study)
        return 0
    if args.cmd == "show":
        _show(args.target)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
