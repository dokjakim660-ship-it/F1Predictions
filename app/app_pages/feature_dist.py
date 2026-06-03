"""Feature Distribution page -- visualise how each feature is spread across the dataset.

Feature type is auto-detected so each gets a sensible chart:
- BINARY  (0/1)                    -> two-bar count chart
- DISCRETE (small int range)       -> bar chart per integer value
- CONTINUOUS                       -> histogram

A second view shows the *conditional* podium rate per value/bin -- which is the
question that actually matters for modelling: "at this feature value, how often
does a driver podium?".

Reads data/features/mvp.parquet (build via `just features`).
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import ui

REPO = Path(__file__).resolve().parents[2]
FEAT_PATH = REPO / "data" / "features" / "mvp.parquet"

# ── Features grouped by category (cleaner dropdown UX) ─────────────────────
FEATURE_GROUPS: dict[str, list[str]] = {
    "Grid / Starting position": [
        "grid_effective", "grid_log", "is_pole", "is_top3_grid",
    ],
    "Qualifying": [
        "q_position", "q_gap_to_pole_ms", "quali_beat_teammate",
    ],
    "FP2 pace": [
        "fp2_long_run_gap_ms", "fp2_short_run_gap_ms",
        "fp2_long_run_lap_count", "has_fp2",
    ],
    "Sprint": [
        "sprint_position", "sprint_gap_to_winner_ms",
        "sprint_minus_quali_pos", "has_sprint",
    ],
    "Driver form (rolling, lagged)": [
        "driver_form_finish_l5", "driver_form_finish_l10",
        "driver_form_podium_rate_l10", "driver_form_points_l5",
        "driver_form_dnf_rate_l10", "driver_form_quali_pos_l5",
        "driver_career_races", "driver_age_years",
    ],
    "Constructor form (rolling, lagged)": [
        "team_form_points_l5", "team_form_finish_l5",
        "team_form_podium_rate_l10", "team_form_dnf_rate_l10",
        "team_form_quali_gap_pole_l5",
    ],
    "Constructor standings": [
        "team_season_points_pre_race", "team_season_pos_pre_race",
    ],
    "Track attributes": [
        "track_length_km", "track_n_corners",
        "track_n_drs_zones", "track_is_street",
    ],
    "Driver × Track history": [
        "driver_track_finish_l3",
    ],
    "Weather": [
        "weather_temp_c_race_hour", "weather_is_wet_race_hour",
        "weather_precip_mm_day_total", "weather_wind_kph_race_hour",
        "weather_temp_c_day_max",
    ],
    "Season / Era": [
        "season_progress", "era_2022plus", "era_2026plus",
    ],
}

GROUP_OPTIONS = {
    "Podium (target)": ("target_podium", {1: "Podium", 0: "No podium"}),
    "Beat teammate (target)": ("target_beat_teammate", {1.0: "Beat", 0.0: "Lost"}),
    "Finish bucket": ("_finish_bucket", None),
    "Grid bucket": ("_grid_bucket", None),
    "Era": ("_era_label", None),
}

# Constant 2-tone palette for binary targets so green/red has a stable meaning.
TARGET_COLOR_SCALE = alt.Scale(
    domain=["Podium", "No podium", "Beat", "Lost"],
    range=[ui.GOOD, ui.BAD, ui.GOOD, ui.BAD],
)


# ── Load + cache ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading feature table…")
def _load() -> pd.DataFrame:
    if not FEAT_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(FEAT_PATH)
    df["_finish_bucket"] = pd.cut(
        df["finish_position"], bins=[0, 3, 10, 99],
        labels=["P1-3 (podium)", "P4-10", "P11+"],
    ).astype(str)
    df["_grid_bucket"] = pd.cut(
        df["grid_effective"], bins=[0, 3, 6, 10, 99],
        labels=["P1-3", "P4-6", "P7-10", "P11+"],
    ).astype(str)
    df["_era_label"] = np.where(
        df["year"] >= 2026, "2026+",
        np.where(df["year"] >= 2022, "2022-25", "Pre-2022"),
    )
    return df


def _detect_kind(s: pd.Series) -> str:
    """Return 'binary', 'discrete', or 'continuous'."""
    nn = s.dropna()
    if nn.empty:
        return "continuous"
    uniq = nn.unique()
    n_uniq = len(uniq)
    if n_uniq <= 2 and set(np.round(uniq, 6)).issubset({0.0, 1.0}):
        return "binary"
    # Integer-ish (within rounding) and few distinct values -> discrete
    if n_uniq <= 25 and np.allclose(nn, np.round(nn), equal_nan=False):
        return "discrete"
    return "continuous"


df = _load()
if df.empty:
    st.error(
        f"Feature file missing: `{FEAT_PATH.relative_to(REPO)}`. "
        "Run `just features` to build it."
    )
    st.stop()


# ── Header + controls ──────────────────────────────────────────────────────
ui.page_header(
    "Feature Distribution",
    eyebrow="Analysis · Data",
    desc="How each feature is spread across the dataset, and the conditional "
    "podium rate per value/bin — the question that actually matters for modelling.",
)

c1, c2 = st.columns([2, 3])
with c1:
    category = st.selectbox("Category", options=list(FEATURE_GROUPS.keys()), index=0)
with c2:
    feature = st.selectbox("Feature", options=FEATURE_GROUPS[category])

c3, c4, c5 = st.columns([2, 1, 1])
with c3:
    group_label = st.selectbox(
        "Colour by", options=list(GROUP_OPTIONS.keys()), index=0,
        help="Splits the chart by this variable. Pick a target to see how the "
             "feature relates to podium / teammate-H2H outcomes.",
    )
group_col, group_map = GROUP_OPTIONS[group_label]
kind = _detect_kind(df[feature])

with c4:
    if kind == "continuous":
        n_bins = st.number_input("Bins", min_value=5, max_value=80, value=25, step=5)
    else:
        n_bins = None
with c5:
    normalize = st.toggle("Normalize", value=False, help="Show share within each colour group instead of raw count.")


# ── Build plot-ready frame ─────────────────────────────────────────────────
# dict.fromkeys preserves order while deduplicating -- avoids a duplicate-column
# DataFrame when group_col == feature (e.g. "target_podium" picked as both).
cols = list(dict.fromkeys([feature, group_col, "target_podium"]))
plot_df = df[cols].copy()
plot_df = plot_df[plot_df[feature].notna()]

if group_map is not None:
    plot_df["_group"] = plot_df[group_col].map(group_map)
else:
    plot_df["_group"] = plot_df[group_col].astype(str).replace("nan", None)

plot_df = plot_df[plot_df["_group"].notna()]


# ── Top metrics ────────────────────────────────────────────────────────────
raw = df[feature]
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Kind", kind)
m2.metric("Rows", f"{len(plot_df):,}")
m3.metric("Mean", f"{raw.mean():.3g}" if kind != "binary" else f"{raw.mean():.2%}")
m4.metric("Median", f"{raw.median():.3g}")
m5.metric("% NaN", f"{raw.isna().mean() * 100:.1f}%")


# ── Chart helpers ──────────────────────────────────────────────────────────
def _color_encoding() -> alt.Color:
    if group_col in ("target_podium", "target_beat_teammate"):
        return alt.Color("_group:N", title=group_label, scale=TARGET_COLOR_SCALE)
    return alt.Color("_group:N", title=group_label, scale=alt.Scale(scheme="tableau10"))


count_field = "share:Q" if normalize else "count():Q"
count_title = "Share within group" if normalize else "Number of samples"


def _binary_chart() -> alt.Chart:
    # Aggregate counts ourselves so we can compute per-group share cleanly.
    agg = (
        plot_df.assign(_v=plot_df[feature].astype(int))
        .groupby(["_group", "_v"]).size().reset_index(name="count")
    )
    agg["share"] = agg["count"] / agg.groupby("_group")["count"].transform("sum")
    return (
        alt.Chart(agg).mark_bar()
        .encode(
            x=alt.X("_v:O", title=feature, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("_group:N"),
            y=alt.Y("share:Q" if normalize else "count:Q", title=count_title,
                    axis=alt.Axis(format="%") if normalize else alt.Axis()),
            color=_color_encoding(),
            tooltip=[
                alt.Tooltip("_v:O", title=feature),
                alt.Tooltip("_group:N", title=group_label),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        ).properties(height=380)
    )


def _discrete_chart() -> alt.Chart:
    agg = (
        plot_df.assign(_v=plot_df[feature].astype(float))
        .groupby(["_group", "_v"]).size().reset_index(name="count")
    )
    agg["share"] = agg["count"] / agg.groupby("_group")["count"].transform("sum")
    return (
        alt.Chart(agg).mark_bar()
        .encode(
            x=alt.X("_v:O", title=feature, axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("_group:N"),
            y=alt.Y("share:Q" if normalize else "count:Q", title=count_title,
                    axis=alt.Axis(format="%") if normalize else alt.Axis()),
            color=_color_encoding(),
            tooltip=[
                alt.Tooltip("_v:O", title=feature),
                alt.Tooltip("_group:N", title=group_label),
                alt.Tooltip("count:Q", title="Count"),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        ).properties(height=380)
    )


def _continuous_chart() -> alt.Chart:
    if normalize:
        # transform_joinaggregate to compute group totals, then divide.
        return (
            alt.Chart(plot_df).mark_bar(opacity=0.6, binSpacing=0)
            .transform_bin(["bin_start", "bin_end"], field=feature, bin=alt.Bin(maxbins=n_bins))
            .transform_aggregate(count="count()", groupby=["bin_start", "bin_end", "_group"])
            .transform_joinaggregate(group_total="sum(count)", groupby=["_group"])
            .transform_calculate(share="datum.count / datum.group_total")
            .encode(
                x=alt.X("bin_start:Q", title=feature),
                x2="bin_end:Q",
                y=alt.Y("share:Q", title="Share within group", axis=alt.Axis(format="%")),
                color=_color_encoding(),
                tooltip=[
                    alt.Tooltip("bin_start:Q", title="Bin start", format=".3g"),
                    alt.Tooltip("bin_end:Q", title="Bin end", format=".3g"),
                    alt.Tooltip("_group:N", title=group_label),
                    alt.Tooltip("count:Q"),
                    alt.Tooltip("share:Q", format=".1%"),
                ],
            ).properties(height=380)
        )
    return (
        alt.Chart(plot_df).mark_bar(opacity=0.6, binSpacing=0)
        .encode(
            x=alt.X(f"{feature}:Q", bin=alt.Bin(maxbins=n_bins), title=feature),
            y=alt.Y("count():Q", title="Number of samples", stack=None),
            color=_color_encoding(),
            tooltip=[
                alt.Tooltip(f"{feature}:Q", bin=alt.Bin(maxbins=n_bins), title="Bin"),
                alt.Tooltip("_group:N", title=group_label),
                alt.Tooltip("count():Q", title="Count"),
            ],
        ).properties(height=380)
    )


# ── Main chart ─────────────────────────────────────────────────────────────
ui.section(f"Distribution of {feature}", sub=kind)
if kind == "binary":
    ui.altair_chart(_binary_chart())
elif kind == "discrete":
    ui.altair_chart(_discrete_chart())
else:
    ui.altair_chart(_continuous_chart())


# ── Conditional podium rate -- the modelling-relevant view ────────────────
ui.section("Conditional podium rate", sub="P(podium) per value/bin")
st.caption(
    "Y-axis = P(podium) given the feature value/bin. Error bars are ±1 SE. "
    "Bars with very few samples are noisy — check the sample-count tooltip."
)

if kind == "binary":
    bucket_df = df[[feature, "target_podium"]].dropna().copy()
    bucket_df["_x"] = bucket_df[feature].astype(int).astype(str)
elif kind == "discrete":
    bucket_df = df[[feature, "target_podium"]].dropna().copy()
    bucket_df["_x"] = bucket_df[feature].astype(int).astype(str)
else:
    bucket_df = df[[feature, "target_podium"]].dropna().copy()
    bucket_df["_x"] = pd.cut(bucket_df[feature], bins=n_bins or 20).astype(str)

rate = (
    bucket_df.groupby("_x", sort=False)["target_podium"]
    .agg(["mean", "count", "std"]).reset_index()
    .rename(columns={"mean": "podium_rate", "count": "n", "std": "sd"})
)
rate["se"] = (rate["podium_rate"] * (1 - rate["podium_rate"]) / rate["n"]).pow(0.5)
rate["lo"] = (rate["podium_rate"] - rate["se"]).clip(lower=0)
rate["hi"] = (rate["podium_rate"] + rate["se"]).clip(upper=1)

# Sort numerically when possible
try:
    rate = rate.assign(_sort=rate["_x"].astype(float)).sort_values("_sort")
    sort_field = list(rate["_x"])
except (ValueError, TypeError):
    sort_field = list(rate["_x"])

base = alt.Chart(rate).encode(
    x=alt.X("_x:N", title=feature, sort=sort_field, axis=alt.Axis(labelAngle=-30)),
)
bars = base.mark_bar(color=ui.ACCENT).encode(
    y=alt.Y("podium_rate:Q", title="P(podium)", scale=alt.Scale(domain=[0, 1])),
    tooltip=[
        alt.Tooltip("_x:N", title=feature),
        alt.Tooltip("n:Q", title="Samples"),
        alt.Tooltip("podium_rate:Q", title="P(podium)", format=".1%"),
        alt.Tooltip("se:Q", title="SE", format=".3f"),
    ],
)
errs = base.mark_errorbar(color=ui.TEXT_MUTED).encode(y="lo:Q", y2="hi:Q")
baseline = alt.Chart(pd.DataFrame({"y": [df["target_podium"].mean()]})).mark_rule(
    color=ui.TEXT_FAINT, strokeDash=[4, 4],
).encode(y="y:Q")

ui.altair_chart((bars + errs + baseline).properties(height=320))
st.caption(f"Dashed line = base rate ({df['target_podium'].mean():.1%}).")
