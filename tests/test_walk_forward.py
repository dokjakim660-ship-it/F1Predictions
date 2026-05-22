"""Tests for the walk-forward harness: fold structure + race-group integrity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eval.walk_forward import (
    TEST_CUTOFF,
    evaluate,
    make_folds,
    oof_predictions,
    split_dev_test,
)
from src.features.baseline import BASELINE_PARQUET, load_features


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    if not BASELINE_PARQUET.exists():
        pytest.skip("baseline.parquet not built -- run `just build-baseline-features`")
    return load_features()


def test_dev_test_split_is_clean(features: pd.DataFrame) -> None:
    dev, test = split_dev_test(features)
    assert (pd.to_datetime(dev["race_date"]) < TEST_CUTOFF).all()
    assert (pd.to_datetime(test["race_date"]) >= TEST_CUTOFF).all()
    assert len(dev) + len(test) == len(features)


def test_folds_are_expanding(features: pd.DataFrame) -> None:
    folds = make_folds(features)
    assert len(folds) == 4
    prev_train_size = -1
    for train, val in folds:
        assert not train.empty and not val.empty
        assert len(train) > prev_train_size  # expanding train window
        prev_train_size = len(train)


def test_no_race_in_both_train_and_val(features: pd.DataFrame) -> None:
    for train, val in make_folds(features):
        overlap = set(train["race_id"]) & set(val["race_id"])
        assert not overlap, f"race_id leaks across train/val: {sorted(overlap)}"


def test_folds_never_touch_test_set(features: pd.DataFrame) -> None:
    for train, val in make_folds(features):
        assert (pd.to_datetime(train["race_date"]) < TEST_CUTOFF).all()
        assert (pd.to_datetime(val["race_date"]) < TEST_CUTOFF).all()


def test_evaluate_runs_with_constant_predictor(features: pd.DataFrame) -> None:
    dev, _ = split_dev_test(features)

    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        rate = float(train["target_podium"].mean())
        return np.full(len(val), rate)

    result = evaluate(dev, fit_predict)
    assert len(result.fold_briers) == 4
    assert 0.0 < result.mean_brier < 0.25
    assert result.objective >= result.mean_brier  # +std penalty, never subtracts


def test_oof_predictions_cover_every_fold_row(features: pd.DataFrame) -> None:
    dev, _ = split_dev_test(features)

    def fit_predict(train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
        return np.full(len(val), float(train["target_podium"].mean()))

    oof = oof_predictions(dev, fit_predict)
    expected = sum(len(val) for _, val in make_folds(dev))
    assert len(oof.y_prob) == expected
    assert len(oof.y_true) == len(oof.race_ids) == expected
    assert set(np.unique(oof.y_true)).issubset({0, 1})
