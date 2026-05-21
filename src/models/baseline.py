"""Baseline podium-probability models.

Trains and compares four reference models on the same train/test split so the
MVP model in Phase 1.4 has a concrete Brier target to beat:

1. ConstantRate           -- P(podium) = 0.15 for every driver. The "knowing
                              nothing" floor. Brier ~ 0.1275 by construction.
2. GridOnly               -- Logistic regression on grid features only. The
                              "Top-3-Quali" baseline the roadmap calls out as
                              the latticed risk #1 (project_roadmap memory).
3. ConstructorOnly        -- Logistic regression on rolling constructor points.
                              Measures how much pure team-form gets you.
4. BaselineLogistic       -- All non-FP2 features combined. The number Phase
                              1.4 has to beat to justify FastF1 long-run-pace.

Train: 2018-2023 race seasons.   Test: 2024-2025.   Future races (2026 partial)
are dropped entirely. Rookie rows (no rolling history) are imputed with global
medians so the model can score them at predict time.

Outputs:
- predictions/baseline.parquet   (one row per test-set driver-race with all four
                                  model probabilities side-by-side)
- models/baseline_logistic.pkl   (the full BaselineLogistic estimator + columns)

Run: `python -m src.models.baseline train`
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import format_summary_row, summary
from src.features.baseline import BASELINE_PARQUET, load_features
from src.utils.paths import MODELS_DIR, PREDICTIONS_DIR

PREDICTIONS_PARQUET = PREDICTIONS_DIR / "baseline.parquet"
MODEL_PATH = MODELS_DIR / "baseline_logistic.pkl"

_TRAIN_YEARS = (2018, 2019, 2020, 2021, 2022, 2023)
_TEST_YEARS = (2024, 2025)

_GRID_FEATURES = ["grid_effective", "grid_log", "is_pole", "is_top3_grid"]
_CONSTRUCTOR_FEATURES = ["con_rolling_pts_l5"]
_DRIVER_FEATURES = [
    "driver_rolling_avg_pos_l5",
    "driver_rolling_dnf_rate_l5",
    "driver_career_race_count",
    "driver_age_years",
]
_FULL_FEATURES = _GRID_FEATURES + _CONSTRUCTOR_FEATURES + _DRIVER_FEATURES


@dataclass
class FittedModel:
    name: str
    scaler: StandardScaler | None
    estimator: LogisticRegression | None
    features: list[str]
    constant_prob: float | None = None
    impute_medians: dict[str, float] | None = None

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.constant_prob is not None:
            return np.full(len(df), self.constant_prob)
        x = df[self.features].copy()
        if self.impute_medians:
            x = x.fillna(self.impute_medians)
        x_scaled = self.scaler.transform(x.values)
        return self.estimator.predict_proba(x_scaled)[:, 1]


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["year"].isin(_TRAIN_YEARS)].copy()
    test = df[df["year"].isin(_TEST_YEARS)].copy()
    return train, test


def _fit_constant(train: pd.DataFrame) -> FittedModel:
    return FittedModel(
        name="ConstantRate(P=0.15)",
        scaler=None,
        estimator=None,
        features=[],
        constant_prob=float(train["target_podium"].mean()),
    )


def _fit_logistic(name: str, train: pd.DataFrame, features: list[str]) -> FittedModel:
    medians = train[features].median(numeric_only=True).to_dict()
    x = train[features].fillna(medians)
    y = train["target_podium"].astype(int).values
    scaler = StandardScaler().fit(x.values)
    est = LogisticRegression(max_iter=1000, C=1.0).fit(scaler.transform(x.values), y)
    return FittedModel(
        name=name,
        scaler=scaler,
        estimator=est,
        features=features,
        impute_medians=medians,
    )


def fit_all(train: pd.DataFrame) -> list[FittedModel]:
    return [
        _fit_constant(train),
        _fit_logistic("GridOnly", train, _GRID_FEATURES),
        _fit_logistic("ConstructorOnly", train, _CONSTRUCTOR_FEATURES),
        _fit_logistic("BaselineLogistic", train, _FULL_FEATURES),
    ]


def evaluate(models: list[FittedModel], test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-row predictions df, per-model summary df)."""
    preds = test[["race_id", "year", "round", "driver_id", "target_podium"]].copy()
    summaries: list[dict] = []
    y_true = test["target_podium"].astype(int).to_numpy()
    for m in models:
        col = f"prob_{m.name.split('(')[0].lower()}"
        prob = m.predict_proba(test)
        preds[col] = prob
        summaries.append(summary(m.name, test["race_id"], y_true, prob))
    return preds, pd.DataFrame(summaries)


def save_predictions(preds: pd.DataFrame, path: Path = PREDICTIONS_PARQUET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(path, index=False)
    return path


def save_baseline_model(model: FittedModel, path: Path = MODEL_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    return path


def train_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_features()
    train, test = _split(df)
    if train.empty or test.empty:
        raise RuntimeError(
            f"Empty split: train={len(train)} test={len(test)}. "
            "Did you build features for the expected years?"
        )
    models = fit_all(train)
    preds, summaries = evaluate(models, test)
    save_predictions(preds)
    full_model = next(m for m in models if m.name == "BaselineLogistic")
    save_baseline_model(full_model)
    return preds, summaries


def _print_results(train: pd.DataFrame, test: pd.DataFrame, summaries: pd.DataFrame) -> None:
    print()
    print("=" * 90)
    print(
        f"Baseline podium models  |  train: {len(train)} rows ({train['year'].min()}-"
        f"{train['year'].max()})  |  test: {len(test)} rows ({test['year'].min()}-"
        f"{test['year'].max()})"
    )
    print("=" * 90)
    for _, row in summaries.iterrows():
        print(format_summary_row(row.to_dict()))
    print("=" * 90)
    print("Lower brier/log_loss = better. Higher top3_acc = better.")
    print(f"Predictions -> {PREDICTIONS_PARQUET}")
    print(f"BaselineLogistic model -> {MODEL_PATH}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("train", help="Fit baseline models, evaluate on 2024-2025, save artefacts.")
    args = p.parse_args(argv)

    if args.cmd == "train":
        if not BASELINE_PARQUET.exists():
            print(
                f"[models.baseline] {BASELINE_PARQUET} missing. "
                "Run `python -m src.features.baseline build` first.",
                file=sys.stderr,
            )
            return 1
        df = load_features()
        train, test = _split(df)
        models = fit_all(train)
        preds, summaries = evaluate(models, test)
        save_predictions(preds)
        save_baseline_model(next(m for m in models if m.name == "BaselineLogistic"))
        _print_results(train, test, summaries)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
