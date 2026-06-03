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

import ui

st.set_page_config(page_title="F1 Predictions", page_icon="🏎️", layout="wide")

# Global design system + the anti-reflow CSS block that stops the HF Spaces
# iframe resize loop. ui.inject() carries both (full rationale lives there).
# Must run once, early, before any page renders.
ui.inject()
ui.enable_altair_theme()

next_race_page = st.Page(
    "app_pages/next_race.py",
    title="Next Race",
    icon=":material/flag:",
    default=True,
)
pre_quali_page = st.Page(
    "app_pages/pre_quali.py",
    title="Pre-Quali",
    icon=":material/timer:",
)
rank_page = st.Page(
    "app_pages/rank.py",
    title="Grid & Finish Order",
    icon=":material/format_list_numbered:",
)
dnf_page = st.Page(
    "app_pages/dnf.py",
    title="DNF Risk",
    icon=":material/car_crash:",
)
stakes_page = st.Page(
    "app_pages/stakes.py",
    title="Stakes",
    icon=":material/payments:",
)
stakes_quali_page = st.Page(
    "app_pages/stakes_quali.py",
    title="Pre-Quali Stakes",
    icon=":material/casino:",
)
roi_page = st.Page(
    "app_pages/roi.py",
    title="ROI Tracker",
    icon=":material/trending_up:",
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
feature_dist_page = st.Page(
    "app_pages/feature_dist.py",
    title="Feature Distribution",
    icon=":material/analytics:",
)

page = st.navigation(
    {
        "Race weekend": [next_race_page, pre_quali_page, rank_page, dnf_page],
        "Betting": [stakes_page, stakes_quali_page, roi_page],
        "Analysis": [backtest_page, importance_page, feature_dist_page],
        "About": [methodology_page],
    },
    position="sidebar",
)

ui.app_header(
    "F1 Predictions",
    "2026 · calibrated ensemble",
    status="podium · quali · grid · DNF · ROI",
)

# Theme toggle. Rendered as sidebar user-content, which CSS reorders ABOVE the
# navigation (see ui.py sidebar order rules). Changing it flips a session-state
# flag and reruns, so the top-of-script ui.inject() repaints the whole chrome.
with st.sidebar:
    ui.theme_toggle()
    st.divider()

page.run()
