"""Pre-Quali page -- Phase 4.2.5. Thursday/Friday forecast, before qualifying.

Reads `predictions/next_race_prequali_{target}.parquet` from
`just predict-pre-quali YEAR ROUND`. Each driver carries probabilities for two
timing modes (pre_weekend / post_fp2) x every model x raw/cal, so this page can:

- show a single mode (Post-FP2 is the richest on a normal weekend),
- show the FP2 effect as a pre_weekend -> post_fp2 delta (table + optional chart).

raw probabilities are the display default: at this sample size the isotonic
calibrator collapses to a few steps, so raw differentiates drivers better. A
"Calibrated" toggle switches to cal. LogisticRegression is the default model
(most robust per `prequali select`); the others are selectable for comparison.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO / "data" / "reference" / "race_inventory.parquet"

_TARGETS = {
    "pole": "Pole",
    "top3_quali": "Top-3 Q",
    "top10_quali": "Top-10 Q",
    "teammate_quali": "Teammate",
}

# Short model labels for narrow HF iframes (see feedback-hf-iframe-layout).
_DISPLAY_NAMES = {
    "recentqualiform": "recentQ",
    "logisticregression": "logreg",
    "xgboost": "xgb",
    "lightgbm": "lgbm",
    "ensemble": "ensemble",
}
# logreg first = the default model the selector lands on.
_MODEL_ORDER = ["logisticregression", "xgboost", "lightgbm", "ensemble", "recentqualiform"]

# How many top rows get a medal, by target. teammate/top10 just sort.
_MEDAL_COUNT = {"pole": 1, "top3_quali": 3, "top10_quali": 0, "teammate_quali": 0}


def _short(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name)


@st.cache_data(show_spinner=False)
def _load_predictions(target_short: str) -> pd.DataFrame:
    path = REPO / "predictions" / f"next_race_prequali_{target_short}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_inventory() -> pd.DataFrame:
    inv = pd.read_parquet(INVENTORY_PATH)[["race_id", "gp_name", "circuit_name", "race_date"]]
    inv["race_date"] = pd.to_datetime(inv["race_date"])
    return inv


def _prob_col(mode: str, model: str, rawcal: str) -> str:
    return f"prob_{mode}_{model}_{rawcal}"


def _models_present(preds: pd.DataFrame, rawcal: str) -> list[str]:
    present = {
        c.removeprefix("prob_post_fp2_").removesuffix(f"_{rawcal}")
        for c in preds.columns
        if c.startswith("prob_post_fp2_") and c.endswith(f"_{rawcal}")
    }
    return [m for m in _MODEL_ORDER if m in present]


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# --- Page body ----------------------------------------------------------

target_short = st.radio(
    "Target",
    list(_TARGETS),
    format_func=lambda t: _TARGETS[t],
    horizontal=True,
    key="prequali_target",
)

preds = _load_predictions(target_short)
if preds.empty:
    st.error(
        f"No pre-quali predictions yet for `{target_short}`. "
        f"Run `just build-prequali-features YEAR ROUND` then "
        f"`just predict-pre-quali YEAR ROUND` locally and re-deploy."
    )
    st.stop()

preds["driver_family_name"] = preds["driver_family_name"].fillna(preds["driver_id"])
preds["constructor_name"] = preds["constructor_name"].fillna(preds["constructor_id"])

race_id = preds["race_id"].iloc[0]
has_fp2 = int(preds["has_fp2"].fillna(0).max())

inv = _load_inventory()
race_meta = inv[inv["race_id"] == race_id]
if not race_meta.empty:
    gp_name = race_meta["gp_name"].iloc[0]
    circuit = race_meta["circuit_name"].iloc[0]
    race_date = race_meta["race_date"].iloc[0].date()
    st.markdown(f"### {gp_name} — pre-qualifying")
    st.caption(f"{circuit} · {race_date} · forecast before qualifying")
else:
    st.markdown(f"### {race_id} — pre-qualifying")

if has_fp2 == 0:
    st.warning(
        "**Sprint weekend** — no FP2 long-run data. Pre-Weekend and Post-FP2 "
        "predictions coincide (there is no FP2 to move them)."
    )

# --- Controls -----------------------------------------------------------

view = st.radio(
    "View",
    ["post_fp2", "pre_weekend", "delta"],
    format_func={
        "post_fp2": "Post-FP2",
        "pre_weekend": "Pre-Weekend",
        "delta": "FP2 effect (Δ)",
    }.get,
    horizontal=True,
    key="prequali_view",
)

use_cal = st.toggle(
    "Calibrated",
    value=False,
    key="prequali_cal",
    help="Off = raw probs (smoother, better driver separation at this N). "
    "On = isotonic-calibrated (matches the Stakes/Kelly probs).",
)
rawcal = "cal" if use_cal else "raw"
model_options = _models_present(preds, rawcal)
model = st.radio(
    "Model",
    model_options,
    format_func=_short,
    horizontal=True,
    key="prequali_model",
    help="logreg = default (most robust at this N). recentQ = 1-feature baseline "
    "(recent average quali position).",
)

label = _TARGETS[target_short]

# --- Single-mode views --------------------------------------------------

if view in ("post_fp2", "pre_weekend"):
    col = _prob_col(view, model, rawcal)
    disp = preds.sort_values(col, ascending=False).reset_index(drop=True)

    fav = disp.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Drivers", len(disp))
    c2.metric("FP2 data", "yes" if has_fp2 else "no")
    c3.metric(f"Top {label}", fav["driver_family_name"])

    st.subheader(f"P({label}) — {'Post-FP2' if view == 'post_fp2' else 'Pre-Weekend'}")
    medals = ["🥇", "🥈", "🥉"]
    n_medal = _MEDAL_COUNT[target_short]
    rank = [medals[i] if i < n_medal else "" for i in range(len(disp))]
    table = pd.DataFrame(
        {
            "": rank,
            "driver": disp["driver_family_name"],
            "team": disp["constructor_name"],
            "recentQ": disp["recent_quali_pos"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—"),
            f"P({label})": disp[col].map(_fmt_pct),
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")

# --- Delta view (FP2 effect) -------------------------------------------

else:
    pw = _prob_col("pre_weekend", model, rawcal)
    pf = _prob_col("post_fp2", model, rawcal)
    d = preds.copy()
    d["delta"] = d[pf] - d[pw]
    d = d.sort_values(pf, ascending=False).reset_index(drop=True)

    movers = d.reindex(d["delta"].abs().sort_values(ascending=False).index)
    up = movers[movers["delta"] > 0].head(1)
    down = movers[movers["delta"] < 0].head(1)
    c1, c2, c3 = st.columns(3)
    c1.metric("Drivers", len(d))
    c2.metric(
        "FP2 lifted",
        up["driver_family_name"].iloc[0] if not up.empty else "—",
        f"{up['delta'].iloc[0] * 100:+.1f} pp" if not up.empty else None,
    )
    c3.metric(
        "FP2 dropped",
        down["driver_family_name"].iloc[0] if not down.empty else "—",
        f"{down['delta'].iloc[0] * 100:+.1f} pp" if not down.empty else None,
    )

    st.subheader(f"P({label}) — FP2 effect")
    st.caption("How FP2 long-run pace moved each driver: Pre-Weekend → Post-FP2.")

    tbl = pd.DataFrame(
        {
            "driver": d["driver_family_name"],
            "team": d["constructor_name"],
            "preWE": d[pw],
            "postFP2": d[pf],
            "Δ": d["delta"],
        }
    )

    def _color_delta(v: float) -> str:
        if v > 0.005:
            return "color: #2e7d32"
        if v < -0.005:
            return "color: #c62828"
        return "color: gray"

    styled = tbl.style.map(_color_delta, subset=["Δ"]).format(
        {"preWE": _fmt_pct, "postFP2": _fmt_pct, "Δ": lambda v: f"{v * 100:+.1f} pp"}
    )
    st.dataframe(styled, hide_index=True, width="stretch")

    with st.expander("FP2 effect chart"):
        if has_fp2 == 0:
            st.info("Sprint weekend — no FP2, so there is no movement to chart.")
        else:
            chart_df = tbl[["driver", "Δ"]].copy()
            chart_df["pp"] = chart_df["Δ"] * 100
            bar = (
                alt.Chart(chart_df)
                .mark_bar()
                .encode(
                    x=alt.X("pp:Q", title="Δ probability (pp)"),
                    y=alt.Y("driver:N", sort="-x", title=None),
                    color=alt.condition(
                        alt.datum.pp > 0, alt.value("#2e7d32"), alt.value("#c62828")
                    ),
                    tooltip=[
                        alt.Tooltip("driver:N"),
                        alt.Tooltip("pp:Q", title="Δ pp", format="+.1f"),
                    ],
                )
                .properties(height=max(220, 18 * len(chart_df)))
            )
            st.altair_chart(bar, use_container_width=True)

# --- Cross-model comparison --------------------------------------------

with st.expander("Compare across all models"):
    st.caption(
        f"P({label}), {('calibrated' if use_cal else 'raw')}, "
        f"{'Post-FP2' if view != 'pre_weekend' else 'Pre-Weekend'}. "
        "Big logreg↔tree gaps flag drivers the stack is least sure about."
    )
    cmp_mode = "pre_weekend" if view == "pre_weekend" else "post_fp2"
    cmp = preds.copy()
    sort_col = _prob_col(cmp_mode, model, rawcal)
    cmp = cmp.sort_values(sort_col, ascending=False).reset_index(drop=True)
    cmp_cols = {"driver": cmp["driver_family_name"], "team": cmp["constructor_name"]}
    for m in model_options:
        cmp_cols[_short(m)] = cmp[_prob_col(cmp_mode, m, rawcal)].map(_fmt_pct)
    st.dataframe(pd.DataFrame(cmp_cols), hide_index=True, width="stretch")
