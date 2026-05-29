"""Shared Kelly-staking helper.

Single-bet Kelly fraction for a binary bet at decimal odds D and win prob p:

    f* = (p*D - 1) / (D - 1)

clipped to 0 when there is no edge. Lives here so the Stakes page, the
Pre-Quali Stakes page and the ROI evaluator all agree on one definition
(previously copy-pasted into each).
"""

from __future__ import annotations

import math


def kelly_raw(odds: float, p: float) -> float:
    """Single-bet Kelly fraction, clipped to 0. NaN-safe.

    Returns 0 for invalid odds (<= 1) or non-positive edge -- "no bet" sits
    naturally at f*=0.
    """
    if odds is None or p is None:
        return 0.0
    try:
        odds = float(odds)
        p = float(p)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(odds) or math.isnan(p) or odds <= 1.0:
        return 0.0
    return max(0.0, (p * odds - 1.0) / (odds - 1.0))
