"""Phase 4.2.2 pre-quali driver smoke tests.

The heavy lifting (8 model x fold trainings) is verified by running
`python -m src.models.prequali eval`; these guards just keep the wiring honest:
the target map points at real columns, and the RecentQualiForm baseline returns
well-formed probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.build import FEATURES_PARQUET
from src.models.prequali import (
    _RECENT_QUALI_FEATURE,
    PRE_QUALI_TARGETS,
    make_recent_quali_form_fit_predict,
)


def test_pre_quali_targets_point_at_real_columns() -> None:
    if not FEATURES_PARQUET.exists():
        pytest.skip("mvp.parquet not built -- run `just build`")
    df = pd.read_parquet(FEATURES_PARQUET)
    missing = [col for col in PRE_QUALI_TARGETS.values() if col not in df.columns]
    assert not missing, f"pre-quali targets reference missing columns: {missing}"
    assert _RECENT_QUALI_FEATURE in df.columns


def test_recent_quali_form_returns_valid_probabilities() -> None:
    train = pd.DataFrame(
        {
            _RECENT_QUALI_FEATURE: [1.0, 3.0, 18.0, np.nan, 10.0, 2.0],
            "target_top10_quali": [1, 1, 0, 0, 1, 1],
        }
    )
    val = pd.DataFrame({_RECENT_QUALI_FEATURE: [1.0, 20.0, np.nan]})
    fit_predict = make_recent_quali_form_fit_predict("target_top10_quali")
    probs = fit_predict(train, val)
    assert probs.shape == (3,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    # Front-runner (avg quali P1) should be likelier top-10 than a backmarker (P20).
    assert probs[0] > probs[1]
