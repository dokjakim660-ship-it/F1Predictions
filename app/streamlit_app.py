"""F1 Predictions — Streamlit multi-page app entry (Phase 2.1).

Reads the holdout-test predictions committed by `just final-eval TARGET=...`
and renders methodology + backtest pages. Wired via `st.navigation` so the
Phase-2.2+ pages (Feature Importance, eventually Next Race) drop in as new
files in `app/app_pages/` without touching the page modules themselves.

Run locally:  `just app`
              `uv run --group app streamlit run app/streamlit_app.py`
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="F1 Predictions", page_icon="🏎️", layout="wide")

next_race_page = st.Page(
    "app_pages/next_race.py",
    title="Next Race",
    icon=":material/flag:",
    default=True,
)
stakes_page = st.Page(
    "app_pages/stakes.py",
    title="Stakes",
    icon=":material/payments:",
)
methodology_page = st.Page(
    "app_pages/methodology.py",
    title="Methodology",
    icon=":material/description:",
)
backtest_page = st.Page(
    "app_pages/backtest.py",
    title="Backtest History",
    icon=":material/show_chart:",
)
importance_page = st.Page(
    "app_pages/importance.py",
    title="Feature Importance",
    icon=":material/bar_chart:",
)

page = st.navigation(
    [next_race_page, stakes_page, methodology_page, backtest_page, importance_page],
    position="top",
)

st.title("🏎️ F1 Predictions")
st.caption(
    "Pre-race podium + teammate-H2H probabilities from FastF1 telemetry. "
    "Built from scratch as a learning project — also trying to beat Tipico."
)

page.run()
