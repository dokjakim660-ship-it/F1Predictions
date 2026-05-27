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

# Stop the Altair-hover wackeln on HF Spaces. Root cause: hovering an Altair
# data point shows a tooltip; if the tooltip extends past the viewport, the
# iframe gains a scrollbar; the scrollbar shrinks the container by ~15px;
# `use_container_width=True` resizes the chart; the mouse is no longer on the
# point; tooltip + scrollbar disappear; container widens; chart re-renders;
# mouse is back on the point -- infinite reflow loop. Forcing the vertical
# scrollbar to always be present (even when content fits) makes the width
# constant and breaks the loop.
st.markdown(
    """
    <style>
    /* (1) Always-on vertical scrollbar -- prevents the iframe width from
       jumping by ~15 px when content briefly overflows on hover. */
    html { overflow-y: scroll !important; }

    /* (2) Hide vega-embed's "..." action menu -- its mouseenter/leave
       changes chart padding by a few px, re-triggering the reflow loop. */
    .vega-embed details, .vega-embed summary { display: none !important; }

    /* (3) Pull the Vega tooltip out of document flow. position:fixed +
       pointer-events:none means hovering can never enlarge the document
       height, so Streamlit never reports a new iframe height to HF Spaces. */
    #vg-tooltip-element {
        position: fixed !important;
        pointer-events: none !important;
        z-index: 9999 !important;
    }

    /* (4) CSS containment on every chart container -- layout changes inside
       the chart (hover state, tooltip, axis tick recompute) stay inside the
       chart and don't propagate to the parent's geometry. */
    [data-testid="stVegaLiteChart"],
    [data-testid="stAltairChart"] {
        contain: layout style !important;
        transition: none !important;
    }
    [data-testid="stVegaLiteChart"] *,
    [data-testid="stAltairChart"] * {
        transition: none !important;
        animation: none !important;
    }

    /* (5) Continuous mouse-INDEPENDENT wackeln = Streamlit's iframe-resize
       postMessage loop with the HF host. Streamlit reports a slightly
       different height every tick; HF resizes the iframe; the height changes
       trigger a sub-pixel content reflow that flips the next reported height
       again. Containment on the top-level containers stops that propagation
       so Streamlit reports a stable height. */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"] {
        contain: layout style !important;
    }

    /* (6) Subpixel rendering can make Altair charts report fractional heights
       on each tick; round them. */
    [data-testid="stVegaLiteChart"] {
        height: auto !important;
        min-height: 1px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    [next_race_page, stakes_page, roi_page, methodology_page, backtest_page, importance_page, feature_dist_page],
    position="top",
)

st.title("🏎️ F1 Predictions")
st.caption(
    "Pre-race podium + teammate-H2H probabilities from FastF1 telemetry. "
    "Built from scratch as a learning project — also trying to beat Tipico."
)

page.run()
