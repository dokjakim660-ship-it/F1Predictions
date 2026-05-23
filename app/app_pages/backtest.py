"""Backtest History page -- calibrated Brier + CIs + trend + per-race drilldown.

Reads the holdout-test predictions committed by `just final-eval TARGET=...`
and joins them against the race inventory (data/reference/race_inventory.parquet)
for human-readable Grand-Prix names and dates. All Brier confidence intervals
are recomputed in-app via race-level bootstrap (1000 resamples), cached so
flipping the target radio stays snappy.
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[2]
INVENTORY_PATH = REPO / "data" / "reference" / "race_inventory.parquet"
_BOOT_N = 1000
_BOOT_SEED = 42


@st.cache_data(show_spinner=False)
def _load_predictions(target_short: str) -> pd.DataFrame:
    preds_path = REPO / "predictions" / f"mvp_test_{target_short}.parquet"
    return pd.read_parquet(preds_path)


@st.cache_data(show_spinner=False)
def _load_inventory() -> pd.DataFrame:
    inv = pd.read_parquet(INVENTORY_PATH)[["race_id", "gp_name", "race_date"]]
    inv["race_date"] = pd.to_datetime(inv["race_date"])
    return inv


def _cal_prob_cols(preds: pd.DataFrame) -> list[str]:
    return sorted(c for c in preds.columns if c.startswith("prob_") and c.endswith("_cal"))


def _model_name(prob_col: str) -> str:
    return prob_col.removeprefix("prob_").removesuffix("_cal")


@st.cache_data(show_spinner="Bootstrapping Brier CIs (1000x races resampled)…")
def _brier_table_with_ci(target_short: str, target_col: str) -> pd.DataFrame:
    """Per-model calibrated Brier + 95% bootstrap CI (races resampled with replacement)."""
    preds = _load_predictions(target_short)
    y_true = preds[target_col].astype(int).to_numpy()
    race_ids = preds["race_id"].to_numpy()
    races = np.unique(race_ids)
    race_to_rows = {r: np.flatnonzero(race_ids == r) for r in races}

    rng = np.random.default_rng(_BOOT_SEED)
    # Pre-sample the resample indices once -- every model uses the same row sets,
    # which makes the CI ranges across models directly comparable (paired-style).
    sampled = [
        np.concatenate([race_to_rows[r] for r in rng.choice(races, size=len(races), replace=True)])
        for _ in range(_BOOT_N)
    ]

    rows = []
    for col in _cal_prob_cols(preds):
        y_prob = preds[col].to_numpy()
        se = (y_prob - y_true) ** 2
        boot = np.array([se[rows_].mean() for rows_ in sampled])
        rows.append(
            {
                "model": _model_name(col),
                "brier_cal": float(se.mean()),
                "brier_lo": float(np.percentile(boot, 2.5)),
                "brier_hi": float(np.percentile(boot, 97.5)),
            }
        )
    return pd.DataFrame(rows).sort_values("brier_cal").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _per_race_brier(target_short: str, target_col: str) -> pd.DataFrame:
    """Long-format per-race calibrated Brier (one row per race x model)."""
    preds = _load_predictions(target_short)
    inv = _load_inventory()

    y_true = preds[target_col].astype(int)
    per_race_parts = []
    for col in _cal_prob_cols(preds):
        se = (preds[col] - y_true) ** 2
        per_race = se.groupby(preds["race_id"]).mean().rename("brier")
        per_race = per_race.reset_index()
        per_race["model"] = _model_name(col)
        per_race_parts.append(per_race)
    long = pd.concat(per_race_parts, ignore_index=True)
    return long.merge(inv, on="race_id", how="left").sort_values("race_date")


# --- Page body -----------------------------------------------------------

target_short = st.radio("Target", ["podium", "teammate"], horizontal=True, key="target_radio")
target_col = "target_podium" if target_short == "podium" else "target_beat_teammate"

preds_path = REPO / "predictions" / f"mvp_test_{target_short}.parquet"
if not preds_path.exists():
    st.error(
        f"Predictions file missing: `{preds_path.relative_to(REPO)}`. "
        f"Run `just final-eval TARGET={target_short}` first."
    )
    st.stop()
if not INVENTORY_PATH.exists():
    st.error(
        f"Race inventory missing: `{INVENTORY_PATH.relative_to(REPO)}`. "
        "Run `just ingest-schedule` first."
    )
    st.stop()

preds = _load_predictions(target_short)

c1, c2, c3 = st.columns(3)
c1.metric("Holdout rows", f"{len(preds):,}")
c2.metric("Holdout races", int(preds["race_id"].nunique()))
c3.metric("Base rate", f"{preds[target_col].mean():.1%}")

# --- Brier table with 95% CI --------------------------------------------

st.subheader("Calibrated Brier per model")
st.caption(
    "Point estimate + 95% bootstrap CI (1000× race-level resamples). "
    "Lower = better. Overlapping CIs ≈ tie within sampling noise."
)
brier_df = _brier_table_with_ci(target_short, target_col)

brier_display = brier_df.assign(
    ci_low=brier_df["brier_lo"].round(4),
    ci_high=brier_df["brier_hi"].round(4),
).rename(columns={"brier_cal": "brier"})[["model", "brier", "ci_low", "ci_high"]]
st.dataframe(brier_display, hide_index=True, width="stretch")

ci_chart = (
    alt.Chart(brier_df)
    .mark_errorbar(thickness=2, color="#1f77b4")
    .encode(
        x=alt.X("brier_lo:Q", scale=alt.Scale(zero=False), title="Brier (calibrated)"),
        x2="brier_hi:Q",
        y=alt.Y("model:N", sort=brier_df["model"].tolist(), title=None),
    )
)
ci_points = (
    alt.Chart(brier_df)
    .mark_point(filled=True, size=120, color="#1f77b4")
    .encode(
        x="brier_cal:Q",
        y=alt.Y("model:N", sort=brier_df["model"].tolist()),
        tooltip=[
            "model",
            alt.Tooltip("brier_cal:Q", format=".4f", title="point"),
            alt.Tooltip("brier_lo:Q", format=".4f", title="CI low"),
            alt.Tooltip("brier_hi:Q", format=".4f", title="CI high"),
        ],
    )
)
st.altair_chart((ci_chart + ci_points).properties(height=30 * len(brier_df) + 60), width="stretch")

# --- Reliability plot (existing artefact) -------------------------------

plot_path = REPO / "models" / f"reliability_mvp_{target_short}.png"
if plot_path.exists():
    st.subheader("Reliability diagram (holdout test, calibrated)")
    st.image(str(plot_path), width="stretch")

# --- Brier trend across the holdout -------------------------------------

st.subheader("Brier trend across the holdout test")
st.caption(
    "Per-race calibrated Brier, one point per Grand Prix. "
    "Spikes mark messy weekends (wet races, safety cars, multi-DNF chaos); "
    "flat = the model is consistently right or consistently wrong."
)
trend_df = _per_race_brier(target_short, target_col)
trend_chart = (
    alt.Chart(trend_df)
    .mark_line(point=True, interpolate="monotone")
    .encode(
        x=alt.X("race_date:T", title="Race date"),
        y=alt.Y("brier:Q", title="Per-race Brier"),
        color=alt.Color("model:N", title="Model"),
        tooltip=[
            alt.Tooltip("gp_name:N", title="Grand Prix"),
            alt.Tooltip("race_date:T", title="Date"),
            alt.Tooltip("model:N"),
            alt.Tooltip("brier:Q", format=".4f"),
        ],
    )
    .properties(height=380)
    .interactive()
)
st.altair_chart(trend_chart, width="stretch")

# --- Per-race drilldown -------------------------------------------------

st.subheader("Per-race predictions")
inv = _load_inventory()
race_labels = (
    preds[["race_id"]]
    .drop_duplicates()
    .merge(inv, on="race_id", how="left")
    .sort_values("race_date", ascending=False)
)
race_labels["label"] = race_labels.apply(
    lambda r: f"{r['race_id']} · {r['gp_name']} · {r['race_date'].date()}", axis=1
)
label_to_id = dict(zip(race_labels["label"], race_labels["race_id"]))
chosen_label = st.selectbox("Pick a race", race_labels["label"].tolist())
race_id = label_to_id[chosen_label]

race_df = preds[preds["race_id"] == race_id].copy()
prob_cols_cal = _cal_prob_cols(preds)

# Per-race Brier for context above the drilldown table.
y_true_race = race_df[target_col].astype(int).to_numpy()
per_race_briers = {
    _model_name(col): float(np.mean((race_df[col].to_numpy() - y_true_race) ** 2))
    for col in prob_cols_cal
}
best_in_race = min(per_race_briers, key=per_race_briers.get)
b_cols = st.columns(len(per_race_briers))
for (name, b), col in zip(per_race_briers.items(), b_cols):
    col.metric(
        f"{name}{' (race best)' if name == best_in_race else ''}",
        f"Brier {b:.4f}",
    )

sort_col = f"prob_{best_in_race}_cal"
show_cols = ["driver_id", target_col] + prob_cols_cal
table = race_df[show_cols].sort_values(sort_col, ascending=False)
st.dataframe(table, hide_index=True, width="stretch")
