"""Phase 4.2.8 composition: leakage-safe predicted-quali feature helpers.

Covers the parts that are easy to get wrong and don't need the full feature
table: the out-of-fold/in-sample dev feature (every dev row gets a value, the
walk-forward windows are honoured, and NaN-target train rows are dropped), and
that the holdout feature is produced by a fit that never sees the holdout.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eval.walk_forward import TEST_CUTOFF
from src.models import compose_prequali as cp


def _dev_frame() -> pd.DataFrame:
    """Dev rows straddling the first walk-forward boundary (2022-07-01).

    Pre-2022-07 rows fall outside every val window -> in-sample fallback path.
    """
    dates = pd.to_datetime(
        ["2021-05-01", "2021-05-01", "2022-09-01", "2022-09-01", "2023-09-01", "2023-09-01"]
    )
    return pd.DataFrame(
        {
            "race_id": ["a", "a", "b", "b", "c", "c"],
            "race_date": dates,
            "driver_id": ["d1", "d2", "d1", "d2", "d1", "d2"],
            "target_x": [1.0, 0.0, 1.0, 0.0, np.nan, 0.0],
        }
    )


def test_oof_feature_fills_every_row():
    dev = _dev_frame()

    # Trivial fit_predict: train base rate for every val row. Records which train
    # sizes it saw so we can assert NaN-target rows were dropped.
    seen_train_sizes: list[int] = []

    def fit_predict(train, val):
        seen_train_sizes.append(len(train))
        return np.full(len(val), float(train["target_x"].mean()))

    # NaN-dropping wrapper around our trivial predictor (mirrors _pq_fit_predict).
    def dropping(train, val):
        return fit_predict(train[train["target_x"].notna()], val)

    out = cp._oof_feature(dev, dropping, "target_x")
    assert out.notna().all()
    assert list(out.index) == list(dev.index)
    # The lone NaN-target row never contributed to any train slice.
    assert all(s <= dev["target_x"].notna().sum() for s in seen_train_sizes)


def test_augment_adds_pq_columns_without_leak(monkeypatch):
    dev = _dev_frame()
    test = pd.DataFrame(
        {
            "race_id": ["z", "z"],
            "race_date": pd.to_datetime([TEST_CUTOFF, TEST_CUTOFF]),
            "driver_id": ["d1", "d2"],
            "target_x": [1.0, 0.0],
        }
    )

    # Stub the four pre-quali targets down to the one synthetic column, and the
    # pre-quali feature set to something trivial, so we exercise _augment's
    # plumbing without the real 30-feature table.
    monkeypatch.setattr(cp, "_PQ_TARGETS", {"pq_x": "target_x"})
    monkeypatch.setattr(cp, "PRE_QUALI_FEATURE_SETS", {cp._PQ_MODE: []})

    captured: dict[str, int] = {}

    def fake_fit_predict(target_col, numeric_features):
        def fp(train, val):
            captured["last_train_ids"] = set(train["race_id"])
            return np.full(len(val), 0.5)

        return fp

    monkeypatch.setattr(cp, "_pq_fit_predict", fake_fit_predict)

    dev_aug, test_aug = cp._augment(dev, test)
    assert "pq_x" in dev_aug.columns and "pq_x" in test_aug.columns
    assert dev_aug["pq_x"].notna().all()
    assert test_aug["pq_x"].notna().all()
    # The holdout feature was produced by a fit trained only on dev race_ids.
    assert "z" not in captured["last_train_ids"]
