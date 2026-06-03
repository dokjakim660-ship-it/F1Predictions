"""Next Race page -- Phase 3.4. Saturday-evening forecast view.

Reads `predictions/next_race_{podium,teammate}.parquet` produced by
`just predict-next YEAR ROUND` and renders calibrated per-driver
probabilities for the upcoming race. Joins `data/features/next_race.parquet`
for human-readable driver + team names and `data/reference/race_inventory.parquet`
for the GP name, circuit, and date.

Falls back to a clear "missing artefact, run X" message if the predictions
parquet has not been produced for the current race yet -- the page never
silently shows stale data because the train-on-demand predict step is cheap.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import metrics
import ui

REPO = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO / "data" / "reference" / "race_inventory.parquet"
FEATURES_PATH = REPO / "data" / "features" / "next_race.parquet"

# Shared with backtest.py -- keep in sync if new models join the stack.
_DISPLAY_NAMES = {
    "constantrate": "const",
    "top3quali": "top3-quali",
    "logisticregression": "logreg",
    "xgboost": "xgb",
    "lightgbm": "lgbm",
    "ensemble": "ensemble",
}

# Order shown in the model selector. Ensemble is the published best for podium
# and ties XGB; LogReg is best for teammate (Phase 1.5).
_MODEL_ORDER = ["ensemble", "xgboost", "lightgbm", "logisticregression", "top3quali"]


def _short(name: str) -> str:
    return _DISPLAY_NAMES.get(name, name)


@st.cache_data(show_spinner=False)
def _load_predictions(target_short: str) -> pd.DataFrame:
    path = REPO / "predictions" / f"next_race_{target_short}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_inventory() -> pd.DataFrame:
    inv = pd.read_parquet(INVENTORY_PATH)[["race_id", "gp_name", "circuit_name", "race_date"]]
    inv["race_date"] = pd.to_datetime(inv["race_date"])
    return inv


@st.cache_data(show_spinner=False)
def _load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(FEATURES_PATH)


def _cal_prob_cols(preds: pd.DataFrame) -> list[str]:
    return [c for c in preds.columns if c.startswith("prob_") and c.endswith("_cal")]


def _model_name(col: str) -> str:
    return col.removeprefix("prob_").removesuffix("_cal")


# --- Page body ----------------------------------------------------------

ui.page_header(
    "Next Race",
    eyebrow="Race weekend · Saturday forecast",
    desc="Calibrated per-driver podium and team-mate probabilities for the "
    "upcoming Grand Prix, blended across the active model ensemble.",
)

target_short = st.radio("Target", ["podium", "teammate"], horizontal=True, key="next_race_target")

preds = _load_predictions(target_short)
if preds.empty:
    st.error(
        f"No predictions yet for `next_race_{target_short}`. "
        f"Run `just predict-next YEAR ROUND` locally and re-deploy."
    )
    st.stop()

# Enrich with display names from the features parquet (driver_family_name,
# constructor_name). Falls back to the raw ids if features.parquet has not
# been built yet -- the page still renders, just with less-pretty labels.
features = _load_features()
if not features.empty:
    name_cols = features[["driver_id", "driver_family_name", "constructor_name"]].drop_duplicates(
        subset=["driver_id"]
    )
    preds = preds.merge(name_cols, on="driver_id", how="left")
preds["driver_family_name"] = preds.get("driver_family_name", preds["driver_id"]).fillna(
    preds["driver_id"]
)
preds["constructor_name"] = preds.get("constructor_name", preds["constructor_id"]).fillna(
    preds["constructor_id"]
)

race_id = preds["race_id"].iloc[0]
round_no = int(preds["round"].iloc[0])
has_fp2 = int(preds["has_fp2"].iloc[0])

inv = _load_inventory()
race_meta = inv[inv["race_id"] == race_id]
if not race_meta.empty:
    gp_name = race_meta["gp_name"].iloc[0]
    circuit = race_meta["circuit_name"].iloc[0]
    race_date = race_meta["race_date"].iloc[0].date()
    ui.section(gp_name, sub=f"Round {round_no} · {circuit} · {race_date}")
else:
    ui.section(str(race_id), sub=f"Round {round_no}")

if has_fp2 == 0:
    st.warning(
        "**Sprint weekend** — no FP2 long-run data available. Predictions lean on "
        "form + qualifying only; the usual FP2 long-run pace edge is missing."
    )

# Headline metrics. Three columns -- the iframe-layout note caps us at ~4.
pole_row = preds.sort_values("grid").iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("Drivers", len(preds))
c2.metric("FP2 data", "yes" if has_fp2 else "no")
c3.metric("Pole", pole_row["driver_family_name"])

# Model selector. Skip ConstantRate -- it is just the base rate, useless here.
prob_cols = _cal_prob_cols(preds)
model_names = [_model_name(c) for c in prob_cols if _model_name(c) != "constantrate"]

# Order best-first by holdout-test Brier so the strongest model sits leftmost
# (and is the default selection). Fall back to the documented preference order
# for any model without a holdout score.
briers = metrics.holdout_brier(target_short)
model_options = [m for m in _MODEL_ORDER if m in model_names]
model_options.sort(key=lambda m: (briers.get(m, float("inf")), _MODEL_ORDER.index(m)))
if not model_options:  # defensive: should never happen given the predict script
    st.error("No usable calibrated probability columns in predictions parquet.")
    st.stop()

model_name = st.radio(
    "Model",
    model_options,
    format_func=_short,
    horizontal=True,
    key=f"next_race_model_{target_short}",
    help="Ordered best-first by calibrated holdout Brier. Ensemble = mean(XGB, LGBM).",
)
prob_col = f"prob_{model_name}_cal"

# Model comparison — which model is active + how they rank on the holdout test.
if briers:
    ui.section("Model comparison", sub="calibrated holdout Brier · lower is better")
    ui.model_comparison(
        [{"key": m, "name": _short(m), "score": briers.get(m)} for m in model_options],
        active_key=model_name,
    )
    st.caption(
        "Bar length = relative performance (best model = full bar). The highlighted "
        "row is the model driving the probabilities below."
    )

# --- Per-driver table ---------------------------------------------------

ui.section(
    "Per-driver probabilities",
    sub="per-driver, sorted by probability",
)
if target_short == "podium":
    st.caption(
        "P(driver finishes P1–P3). Independent per driver — the top-3 rows sum "
        "to ~3.0 by construction (three podium spots)."
    )
else:
    st.caption(
        "P(driver finishes ahead of their constructor teammate). Per-driver "
        "isotonic calibration + pair-norm so each team's two legs sum to 100%."
    )

display = preds.sort_values(prob_col, ascending=False).reset_index(drop=True)
display["grid"] = display["grid"].astype(int)

prob_label = "P(podium)" if target_short == "podium" else "P(beat tm)"
if target_short == "podium":
    rank_emoji = ["🥇", "🥈", "🥉"] + ["—"] * max(0, len(display) - 3)
    display["rank"] = rank_emoji[: len(display)]
    table = pd.DataFrame(
        {
            "": display["rank"],
            "grid": display["grid"],
            "driver": display["driver_family_name"],
            "team": display["constructor_name"],
            prob_label: ui.pct(display[prob_col]),
        }
    )
else:
    table = pd.DataFrame(
        {
            "grid": display["grid"],
            "driver": display["driver_family_name"],
            "team": display["constructor_name"],
            prob_label: ui.pct(display[prob_col]),
        }
    )
st.dataframe(
    ui.team_styler(table),
    hide_index=True,
    width="stretch",
    column_config={prob_label: ui.prob_column(prob_label)},
)

# --- Team-pair view for teammate target --------------------------------

if target_short == "teammate":
    ui.section("Team pairs", sub="favored driver per constructor")
    st.caption("Per team: the favored driver and their calibrated edge over the teammate.")

    pair_rows = []
    for _, grp in preds.groupby("constructor_id"):
        if len(grp) != 2:
            continue
        grp_sorted = grp.sort_values(prob_col, ascending=False)
        a, b = grp_sorted.iloc[0], grp_sorted.iloc[1]
        pair_rows.append(
            {
                "team": a["constructor_name"],
                "favored": f"{a['driver_family_name']} (P{int(a['grid'])})",
                "P_fav": float(a[prob_col]),
                "vs.": f"{b['driver_family_name']} (P{int(b['grid'])})",
            }
        )
    pairs_df = pd.DataFrame(pair_rows).sort_values("P_fav", ascending=False)
    pairs_display = pd.DataFrame(
        {
            "team": pairs_df["team"],
            "favored": pairs_df["favored"],
            "P": ui.pct(pairs_df["P_fav"]),
            "vs.": pairs_df["vs."],
        }
    )
    st.dataframe(
        ui.team_styler(pairs_display),
        hide_index=True,
        width="stretch",
        column_config={"P": ui.prob_column("P")},
    )

# --- Cross-model comparison expander -----------------------------------

with st.expander("Compare across all models"):
    st.caption(
        "Calibrated probabilities side by side. Big disagreements between "
        "tree models (xgb/lgbm) and linear (logreg) flag drivers where the "
        "model stack is least sure."
    )
    cmp = preds.sort_values(prob_col, ascending=False).reset_index(drop=True)
    cmp_cols = {
        "grid": cmp["grid"].astype(int),
        "driver": cmp["driver_family_name"],
        "team": cmp["constructor_name"],
    }
    for m in model_options:
        cmp_cols[_short(m)] = (cmp[f"prob_{m}_cal"] * 100).map(lambda x: f"{x:.1f}%")
    st.dataframe(pd.DataFrame(cmp_cols), hide_index=True, width="stretch")
