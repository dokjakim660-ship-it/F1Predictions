"""Phase 5.2 C1 compose-DNF ablation smoke + leakage guard."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.build import FEATURES_PARQUET
from src.models import compose_dnf as cd


def _skip_if_no_features() -> None:
    if not FEATURES_PARQUET.exists():
        pytest.skip(f"{FEATURES_PARQUET.name} not built -- run `just build`")


def test_compose_dnf_smoke() -> None:
    _skip_if_no_features()
    evals, race_ids, y_true = cd.evaluate()
    assert len(evals) == 4
    for e in evals:
        assert np.isfinite(e.brier_base) and 0.0 <= e.brier_base <= 1.0
        assert np.isfinite(e.brier_aug) and 0.0 <= e.brier_aug <= 1.0
        # pred_dnf is a probability -> both stacks stay in [0, 1].
        assert ((e.prob_aug >= 0.0) & (e.prob_aug <= 1.0)).all()
    assert len(y_true) == len(race_ids)
