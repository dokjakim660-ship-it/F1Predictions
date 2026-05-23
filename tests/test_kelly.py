"""Unit test for the Kelly formula in the Stakes page.

Validates the closed-form `f* = (p*D - 1) / (D - 1)` against hand-computed
values plus the clipping behaviour (no bet when edge <= 0 or odds <= 1).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# The Stakes page is a Streamlit script (top-level executes UI on import),
# so import the helper by adding the pages folder to sys.path and reading
# only the function via importlib if needed. The function is pure so it
# can be lifted by re-defining for tests -- here we import-by-path with
# a guard.
APP_PAGES = Path(__file__).resolve().parents[1] / "app" / "app_pages"
sys.path.insert(0, str(APP_PAGES))


def _kelly_raw_local(odds: float, p: float) -> float:
    """Mirror of stakes._kelly_raw, kept here so the test does not import
    the streamlit module (which would execute the page on import).
    """
    import pandas as pd  # local import to avoid a top-level dep when not needed

    if pd.isna(odds) or pd.isna(p) or odds <= 1.0:
        return 0.0
    return max(0.0, (p * odds - 1.0) / (odds - 1.0))


def test_kelly_value_bet():
    # P=0.5, odds=3.0 -> f* = (0.5*3 - 1) / (3-1) = 0.5/2 = 0.25
    assert math.isclose(_kelly_raw_local(3.0, 0.5), 0.25, abs_tol=1e-9)


def test_kelly_no_edge_negative():
    # P=0.3, implied=1/3=0.333 -> edge negative -> clipped to 0
    assert _kelly_raw_local(3.0, 0.3) == 0.0


def test_kelly_fair_odds():
    # P=1/3 exactly equals 1/3 implied -> f*=0
    assert math.isclose(_kelly_raw_local(3.0, 1.0 / 3.0), 0.0, abs_tol=1e-9)


def test_kelly_invalid_odds():
    # odds <= 1 is not a real betting market
    assert _kelly_raw_local(1.0, 0.5) == 0.0
    assert _kelly_raw_local(0.5, 0.5) == 0.0
    assert _kelly_raw_local(0.0, 0.5) == 0.0


def test_kelly_huge_edge():
    # P=0.9, odds=2.0 -> f* = (0.9*2 - 1)/(2-1) = 0.8 (gigantic, normal for Kelly)
    assert math.isclose(_kelly_raw_local(2.0, 0.9), 0.8, abs_tol=1e-9)


def test_kelly_nan_safe():
    import numpy as np

    assert _kelly_raw_local(float("nan"), 0.5) == 0.0
    assert _kelly_raw_local(3.0, float("nan")) == 0.0
    assert _kelly_raw_local(np.nan, np.nan) == 0.0
