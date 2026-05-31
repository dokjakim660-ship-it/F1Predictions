"""DNF page -- Phase 5.2. P(driver does not finish) for the upcoming race.

Reads `predictions/next_race_dnf.parquet` produced by `just predict-dnf YEAR
ROUND` and renders calibrated per-driver retirement probabilities in three
timing modes (race / post_fp2 / pre_weekend).

Honest framing (holdout verdict): no DNF model beats the team-reliability
baseline -- retirements are near-unpredictable beyond a team's recent reliability
rate. The numbers are a calibrated reliability estimate, NOT a sharp betting
edge. The page says so up front and defaults to the TeamReliability model.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO / "data" / "reference" / "race_inventory.parquet"
PREDICTIONS_PATH = REPO / "predictions" / "next_race_dnf.parquet"

_MODE_LABELS = {
    "race": "race (post-quali)",
    "post_fp2": "post-FP2",
    "pre_weekend": "pre-weekend",
}
_MODE_ORDER = ["race", "post_fp2", "pre_weekend"]

_DISPLAY_NAMES = {
    "constantrate": "const",
    "teamreliability": "team-rel",
    "logisticregression": "logreg",
    "xgboost": "xgb",
    "lightgbm": "lgbm",
    "ensemble": "ensemble",
}
# Default first; ConstantRate dropped (flat per-driver, useless in a table).
_MODEL_ORDER = ["teamreliability", "logisticregression", "lightgbm", "xgboost", "ensemble"]


def _short(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name)


@st.cache_data(show_spinner=False)
def _load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(PREDICTIONS_PATH)


@st.cache_data(show_spinner=False)
def _load_inventory() -> pd.DataFrame:
    inv = pd.read_parquet(INVENTORY_PATH)[["race_id", "gp_name", "circuit_name", "race_date"]]
    inv["race_date"] = pd.to_datetime(inv["race_date"])
    return inv


def _models_available(preds: pd.DataFrame, mode: str) -> list[str]:
    cols = [c for c in preds.columns if c.startswith(f"prob_{mode}_") and c.endswith("_cal")]
    present = {c.removeprefix(f"prob_{mode}_").removesuffix("_cal") for c in cols}
    return [m for m in _MODEL_ORDER if m in present]


# --- Page body ----------------------------------------------------------

preds = _load_predictions()
if preds.empty:
    st.error("No DNF predictions yet. Run `just predict-dnf YEAR ROUND` locally and re-deploy.")
    st.stop()

race_id = preds["race_id"].iloc[0]
inv = _load_inventory()
race_meta = inv[inv["race_id"] == race_id]
if not race_meta.empty:
    st.markdown(f"### {race_meta['gp_name'].iloc[0]} — DNF risk")
    st.caption(f"{race_meta['circuit_name'].iloc[0]} · {race_meta['race_date'].iloc[0].date()}")
else:
    st.markdown(f"### {race_id} — DNF risk")

st.warning(
    "**Negative result (Phase 5.2):** no DNF model beats the team-reliability "
    "baseline on the holdout — retirements are near-unpredictable beyond a team's "
    "recent reliability rate. Read these as a calibrated reliability estimate, "
    "not a betting edge. Default model = TeamReliability."
)

mode = st.radio(
    "Timing",
    _MODE_ORDER,
    format_func=lambda m: _MODE_LABELS[m],
    horizontal=True,
    key="dnf_mode",
    help="race = post-quali (real grid); post_fp2 / pre_weekend drop quali/grid.",
)

model_options = _models_available(preds, mode)
if not model_options:
    st.error(f"No calibrated probability columns for mode `{mode}`.")
    st.stop()

model_name = st.radio(
    "Model",
    model_options,
    format_func=_short,
    horizontal=True,
    key="dnf_model",
    help="TeamReliability = 1-feature logit on the team's recent DNF rate (the baseline nothing beats).",
)
prob_col = f"prob_{mode}_{model_name}_cal"

mean_p = float(preds[prob_col].mean())
c1, c2, c3 = st.columns(3)
c1.metric("Drivers", len(preds))
c2.metric("Mean P(DNF)", f"{mean_p * 100:.1f}%")
c3.metric("Highest risk", preds.sort_values(prob_col).iloc[-1]["driver_family_name"])

# --- Per-driver table ---------------------------------------------------

st.subheader("Per-driver P(DNF)")
st.caption("P(driver does not finish the race). Higher = more likely to retire.")

display = preds.sort_values(prob_col, ascending=False).reset_index(drop=True)
display["P_str"] = (display[prob_col] * 100).map(lambda x: f"{x:.1f}%")
display["grid"] = display["grid"].astype("Int64")
table = pd.DataFrame(
    {
        "grid": display["grid"],
        "driver": display["driver_family_name"],
        "team": display["constructor_name"],
        "P(DNF)": display["P_str"],
    }
)
st.dataframe(table, hide_index=True, width="stretch")

# --- Cross-model comparison expander ------------------------------------

with st.expander("Compare across all models"):
    st.caption(
        "Calibrated P(DNF) side by side. The models barely separate from the "
        "TeamReliability baseline — that's the whole finding."
    )
    cmp = preds.sort_values(prob_col, ascending=False).reset_index(drop=True)
    cmp_cols = {
        "grid": cmp["grid"].astype("Int64"),
        "driver": cmp["driver_family_name"],
        "team": cmp["constructor_name"],
    }
    for m in model_options:
        col = f"prob_{mode}_{m}_cal"
        cmp_cols[_short(m)] = (cmp[col] * 100).map(lambda x: f"{x:.1f}%")
    st.dataframe(pd.DataFrame(cmp_cols), hide_index=True, width="stretch")
