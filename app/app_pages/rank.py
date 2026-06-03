"""Grid & Finish Order page -- Phase 5 full-grid ranking.

Reads predictions/next_race_rank_{quali,race}.parquet from
`just predict-rank YEAR ROUND` and shows the predicted COMPLETE order for the
upcoming race:

- Race order: each driver's predicted finishing place (Saturday evening, real
  grid known), next to the grid slot and the gain/loss vs grid.
- Quali order: each driver's predicted qualifying place BEFORE qualifying, in
  two timing modes (Pre-Weekend / Post-FP2).

Each driver carries a predicted placement + an "expected position" score for
every model, so the page can switch model and contrast position-regression with
the LambdaMART ranker. Ridge is the default (dev-selected, most interpretable);
the rankers tend to nail the top-k set but place mid-grid less precisely.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

import ui

REPO = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO / "data" / "reference" / "race_inventory.parquet"

# Short model labels for narrow HF iframes (see feedback-hf-iframe-layout).
_MODEL_LABELS = {
    "ridge": "ridge",
    "xgbreg": "xgb-reg",
    "lgbmreg": "lgbm-reg",
    "regensemble": "reg-ens",
    "xgbranker": "xgb-rank",
    "lgbmranker": "lgbm-rank",
    "gridorder": "grid",
    "recentqualiform": "recentQ",
}
# ridge first = the model the selector lands on. Baseline last.
_RACE_MODELS = ["ridge", "xgbreg", "lgbmreg", "regensemble", "xgbranker", "lgbmranker", "gridorder"]
_QUALI_MODELS = [
    "ridge", "xgbreg", "lgbmreg", "regensemble", "xgbranker", "lgbmranker", "recentqualiform"
]
_RANKER_KEYS = {"xgbranker", "lgbmranker"}
_MEDALS = ["🥇", "🥈", "🥉"]


@st.cache_data(show_spinner=False)
def _load(task: str) -> pd.DataFrame:
    path = REPO / "predictions" / f"next_race_rank_{task}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_inventory() -> pd.DataFrame:
    inv = pd.read_parquet(INVENTORY_PATH)[["race_id", "gp_name", "circuit_name", "race_date"]]
    inv["race_date"] = pd.to_datetime(inv["race_date"])
    return inv


def _label(key: str) -> str:
    return _MODEL_LABELS.get(key, key)


def _models_present(preds: pd.DataFrame, candidates: list[str], prefix: str) -> list[str]:
    return [m for m in candidates if f"{prefix}{m}" in preds.columns]


def _race_header(race_id: str, subtitle: str) -> None:
    inv = _load_inventory()
    meta = inv[inv["race_id"] == race_id]
    if not meta.empty:
        ui.section(
            f"{meta['gp_name'].iloc[0]} — {subtitle}",
            sub=f"{meta['circuit_name'].iloc[0]} · {meta['race_date'].iloc[0].date()}",
        )
    else:
        ui.section(f"{race_id} — {subtitle}")


def _medal_column(n: int, count: int = 3) -> list[str]:
    return [_MEDALS[i] if i < count else "" for i in range(n)]


# --- Page body ----------------------------------------------------------

ui.page_header(
    "Grid & Finish Order",
    eyebrow="Race weekend · Full-grid ranking",
    desc="Predicted complete order — every driver gets an exact place, not just a "
    "probability. Ridge is the default; switch to compare regression vs the "
    "learning-to-rank models.",
)

task_label = st.radio(
    "Order",
    ["Race order", "Quali order"],
    horizontal=True,
    key="rank_task",
)
task = "race" if task_label == "Race order" else "quali"

preds = _load(task)
if preds.empty:
    st.error(
        f"No ranking predictions yet for `{task}`. Run `just predict-rank YEAR ROUND` "
        "locally and re-deploy."
    )
    st.stop()

preds = preds.copy()
preds["driver_family_name"] = preds["driver_family_name"].fillna(preds["driver_id"])
preds["constructor_name"] = preds["constructor_name"].fillna(preds["constructor_id"])
race_id = preds["race_id"].iloc[0]


# --- Quali order --------------------------------------------------------

if task == "quali":
    _race_header(race_id, "predicted qualifying order")
    has_fp2 = int(pd.to_numeric(preds["has_fp2"], errors="coerce").fillna(0).max())

    mode = st.radio(
        "Timing",
        ["post_fp2", "pre_weekend"],
        format_func={"post_fp2": "Post-FP2", "pre_weekend": "Pre-Weekend"}.get,
        horizontal=True,
        key="rank_quali_mode",
    )
    if has_fp2 == 0:
        st.warning(
            "**Sprint weekend** — no FP2 long-run data, so Post-FP2 and Pre-Weekend "
            "orders coincide."
        )

    models = _models_present(preds, _QUALI_MODELS, f"rank_{mode}_")
    model = st.radio(
        "Model",
        models,
        format_func=_label,
        horizontal=True,
        key="rank_quali_model",
        help="ridge = default (most interpretable). recentQ = baseline (recent average "
        "quali position). *-rank = LambdaMART learning-to-rank.",
    )

    rank_col = f"rank_{mode}_{model}"
    score_col = f"score_{mode}_{model}"
    disp = preds.sort_values(rank_col).reset_index(drop=True)
    is_ranker = model in _RANKER_KEYS

    c1, c2, c3 = st.columns(3)
    c1.metric("Drivers", len(disp))
    c2.metric("FP2 data", "yes" if has_fp2 else "no")
    c3.metric("Predicted pole", disp["driver_family_name"].iloc[0])

    ui.section("Predicted grid", sub="Post-FP2" if mode == "post_fp2" else "Pre-Weekend")
    score_label = "score" if is_ranker else "exp pos"
    table = pd.DataFrame(
        {
            "": _medal_column(len(disp)),
            "#": disp[rank_col].astype(int),
            "driver": disp["driver_family_name"],
            "team": disp["constructor_name"],
            "recentQ": disp["recent_quali_pos"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "—"),
            score_label: disp[score_col].map(lambda x: f"{x:.2f}" if is_ranker else f"{x:.1f}"),
        }
    )
    st.dataframe(table, hide_index=True, width="stretch")


# --- Race order ---------------------------------------------------------

else:
    _race_header(race_id, "predicted race result")
    models = _models_present(preds, _RACE_MODELS, "rank_race_")
    model = st.radio(
        "Model",
        models,
        format_func=_label,
        horizontal=True,
        key="rank_race_model",
        help="ridge = default (most interpretable). grid = baseline (start order). "
        "*-rank = LambdaMART learning-to-rank.",
    )

    rank_col = f"rank_race_{model}"
    score_col = f"score_race_{model}"
    disp = preds.sort_values(rank_col).reset_index(drop=True)
    is_ranker = model in _RANKER_KEYS
    grid = pd.to_numeric(disp["grid"], errors="coerce")
    delta = grid - disp[rank_col].astype(int)  # + = predicted to gain places vs grid

    movers = disp.assign(_d=delta).reindex(delta.abs().sort_values(ascending=False).index)
    top_mover = movers.iloc[0] if not movers.empty else None

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted winner", disp["driver_family_name"].iloc[0])
    c2.metric(
        "Biggest mover",
        top_mover["driver_family_name"] if top_mover is not None else "—",
        f"{int(top_mover['_d']):+d} vs grid" if top_mover is not None else None,
    )
    c3.metric("Drivers", len(disp))

    ui.section("Predicted finishing order", sub="Δgrid vs start")
    st.caption("Δgrid = places gained (+) or lost (−) versus the starting grid.")
    score_label = "score" if is_ranker else "exp pos"
    table = pd.DataFrame(
        {
            "": _medal_column(len(disp)),
            "#": disp[rank_col].astype(int),
            "driver": disp["driver_family_name"],
            "team": disp["constructor_name"],
            "grid": grid.map(lambda x: f"{int(x)}" if pd.notna(x) else "—"),
            "Δgrid": delta,
            score_label: disp[score_col].map(lambda x: f"{x:.2f}" if is_ranker else f"{x:.1f}"),
        }
    )

    def _color_delta(v: float) -> str:
        if v > 0:
            return f"color: {ui.GOOD}"
        if v < 0:
            return f"color: {ui.BAD}"
        return f"color: {ui.TEXT_FAINT}"

    styled = table.style.map(_color_delta, subset=["Δgrid"]).format({"Δgrid": "{:+d}"})
    st.dataframe(styled, hide_index=True, width="stretch")


# --- Cross-model comparison --------------------------------------------

with st.expander("Compare models (predicted place)"):
    st.caption(
        "Each model's predicted placement side by side. Big gaps between the "
        "regression and *-rank columns flag drivers the methods disagree on."
    )
    if task == "quali":
        prefix = f"rank_{st.session_state.get('rank_quali_mode', 'post_fp2')}_"
        cmp_models = _models_present(preds, _QUALI_MODELS, prefix)
    else:
        prefix = "rank_race_"
        cmp_models = _models_present(preds, _RACE_MODELS, prefix)

    sort_col = f"{prefix}{cmp_models[0]}"
    cmp = preds.sort_values(sort_col).reset_index(drop=True)
    cmp_cols: dict[str, object] = {
        "driver": cmp["driver_family_name"],
        "team": cmp["constructor_name"],
    }
    for m in cmp_models:
        cmp_cols[_label(m)] = cmp[f"{prefix}{m}"].astype(int)
    st.dataframe(pd.DataFrame(cmp_cols), hide_index=True, width="stretch")
