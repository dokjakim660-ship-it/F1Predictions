"""Feature Importance page -- what the best model leans on per target.

Reads the precomputed `predictions/importance_{target}.parquet` written by
`just importance <target>`. The app never imports SHAP itself: the parquet
is the contract, so HF Spaces stays inference-only.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[2]

target_short = st.radio(
    "Target", ["podium", "teammate"], horizontal=True, key="importance_target_radio"
)

imp_path = REPO / "predictions" / f"importance_{target_short}.parquet"
if not imp_path.exists():
    st.error(
        f"Importance file missing: `{imp_path.relative_to(REPO)}`. "
        f"Run `just importance {target_short}` first."
    )
    st.stop()

imp = pd.read_parquet(imp_path)

if target_short == "podium":
    model_name = "XGBoost (best model on holdout test)"
    score_label = "mean(|SHAP|)"
    blurb = (
        "Bar length = **mean of |SHAP|** per feature across all holdout-test rows. "
        "Arrow = mean signed SHAP -- ⬆ means a higher value of this feature pushes "
        "P(podium) **up** on average, ⬇ pushes it **down**. "
        "Grid + qualifying-gap features dominate, matching the Phase-1 finding "
        "that pre-race grid order carries most of the podium signal."
    )
else:
    model_name = "LogisticRegression (best model on holdout test)"
    score_label = "|standardised coefficient|"
    blurb = (
        "Bar length = **|standardised LogReg coefficient|**. Arrow = sign of the "
        "coefficient -- ⬆ means a higher value of this feature increases "
        "P(beat team-mate), ⬇ decreases it. `track_id` is the L2 norm across all "
        "track one-hot levels (direction is mixed, so no arrow). "
        "Team-level and driver-level form pull in partly opposite directions -- "
        "what predicts beating the team-mate is the **gap** between them, not "
        "the absolute level."
    )

c1, c2, c3 = st.columns(3)
c1.metric("Model", model_name.split(" (")[0])
c2.metric("Features ranked", f"{len(imp):,}")
c3.metric("Score", score_label)

st.markdown(blurb)

top_n = st.slider("Top features to show", min_value=5, max_value=min(40, len(imp)), value=20)
top = imp.head(top_n).copy()


def _direction_label(d: float) -> str:
    if pd.isna(d):
        return "mixed"
    return "increases" if d > 0 else "decreases"


top["direction_label"] = top["direction"].map(_direction_label)
top["display_feature"] = top.apply(
    lambda r: f"⬆ {r['feature']}"
    if r["direction_label"] == "increases"
    else (f"⬇ {r['feature']}" if r["direction_label"] == "decreases" else f"• {r['feature']}"),
    axis=1,
)

chart = (
    alt.Chart(top)
    .mark_bar()
    .encode(
        x=alt.X("importance:Q", title=score_label),
        y=alt.Y("display_feature:N", sort="-x", title=None),
        color=alt.Color(
            "direction_label:N",
            scale=alt.Scale(
                domain=["increases", "decreases", "mixed"],
                range=["#2ca02c", "#d62728", "#888888"],
            ),
            legend=alt.Legend(title=f"Effect on P({target_short})"),
        ),
        tooltip=[
            alt.Tooltip("feature:N"),
            alt.Tooltip("importance:Q", format=".4f"),
            alt.Tooltip("direction:Q", format="+.4f"),
        ],
    )
    .properties(height=28 * top_n + 60)
)
st.altair_chart(chart, width="stretch")

st.subheader("Full ranked table")
st.dataframe(imp, hide_index=True, width="stretch")
