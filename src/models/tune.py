"""Optuna hyperparameter tuning for the MVP tree models.

Increment C of the Phase 1.4 pipeline skeleton. Searches XGBoost and LightGBM
hyperparameters with a TPE sampler, minimising the walk-forward objective
(mean + 0.2*std of per-fold Brier) on the dev set. The holdout test set is
never touched here.

Each model gets its own study, persisted to SQLite (models/optuna.db) with
`load_if_exists=True` -- re-running adds trials instead of starting over, and
`tune show` inspects the best result later.

Run: `python -m src.models.tune xgboost --trials 50`
     `python -m src.models.tune lightgbm --trials 50`
     `python -m src.models.tune show`
"""

from __future__ import annotations

import argparse
import sys

import lightgbm as lgb
import optuna
import xgboost as xgb
from optuna.samplers import TPESampler

from src.eval.walk_forward import FitPredictFn, evaluate, split_dev_test
from src.models.mvp import TARGET, TREE_FEATURES, load_model_frame
from src.utils.paths import OPTUNA_DB

_RANDOM_STATE = 42
_MODELS = ("xgboost", "lightgbm")


def _storage() -> str:
    return f"sqlite:///{OPTUNA_DB.as_posix()}"


def _study_name(model: str) -> str:
    # _v2 marks the Phase 1.2 feature-set bump (baseline-9 -> rich table): a
    # fresh study name keeps the old skeleton trials from polluting the search.
    return f"podium_{model}_v2"


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


def make_xgb_fit_predict(params: dict) -> FitPredictFn:
    def fit_predict(train, val):
        model = xgb.XGBClassifier(
            **params,
            eval_metric="logloss",
            enable_categorical=True,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(train[TREE_FEATURES], train[TARGET].astype(int))
        return model.predict_proba(val[TREE_FEATURES])[:, 1]

    return fit_predict


def make_lgbm_fit_predict(params: dict) -> FitPredictFn:
    def fit_predict(train, val):
        model = lgb.LGBMClassifier(
            **params,
            subsample_freq=1,
            random_state=_RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        # LightGBM auto-detects the pandas category column as a categorical feature.
        model.fit(train[TREE_FEATURES], train[TARGET].astype(int))
        return model.predict_proba(val[TREE_FEATURES])[:, 1]

    return fit_predict


def _make_fit_predict(model: str, params: dict) -> FitPredictFn:
    return make_xgb_fit_predict(params) if model == "xgboost" else make_lgbm_fit_predict(params)


def _objective(trial: optuna.Trial, dev, model: str) -> float:
    params = _suggest_xgb(trial) if model == "xgboost" else _suggest_lgbm(trial)
    result = evaluate(dev, _make_fit_predict(model, params))
    trial.set_user_attr("mean_brier", result.mean_brier)
    trial.set_user_attr("std_brier", result.std_brier)
    return result.objective


def tune(model: str, n_trials: int) -> optuna.Study:
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
    OPTUNA_DB.parent.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = load_model_frame()
    dev, _ = split_dev_test(df)

    study = optuna.create_study(
        study_name=_study_name(model),
        direction="minimize",
        storage=_storage(),
        sampler=TPESampler(seed=_RANDOM_STATE),
        load_if_exists=True,
    )
    study.optimize(lambda t: _objective(t, dev, model), n_trials=n_trials)
    return study


def best_params(model: str) -> dict:
    """Best hyperparameters for a tuned model, loaded from its Optuna study."""
    if model not in _MODELS:
        raise ValueError(f"model must be one of {_MODELS}, got {model!r}")
    name = _study_name(model)
    if name not in _existing_studies():
        raise RuntimeError(
            f"No Optuna study for {model!r}. Run: python -m src.models.tune {model} --trials 50"
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


def _show() -> None:
    existing = _existing_studies()
    print("=" * 78)
    print("Optuna studies -- best walk-forward objective (mean + 0.2*std of Brier)")
    print("=" * 78)
    for model in _MODELS:
        name = _study_name(model)
        if name not in existing:
            print(f"{model:<10s}  no study yet -- run `tune {model} --trials 50`")
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
    sub.add_parser("show", help="Print the best trial of each persisted study.")
    args = p.parse_args(argv)

    if args.cmd in _MODELS:
        study = tune(args.cmd, args.trials)
        _print_best(args.cmd, study)
        return 0
    if args.cmd == "show":
        _show()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
