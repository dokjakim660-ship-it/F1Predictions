"""ROI Tracker page -- Phase 4.

Shows the running P&L of Kelly-sized bets placed since odds were first saved.
Data comes from data/roi/log.parquet, written by `just post-race YEAR ROUND`.

Workflow:
  Sa-Abend  ->  Stakes-Page: Quoten eintragen + "Save odds" klicken
  So-Abend  ->  Terminal: `just post-race 2026 6`
  Dann hier ->  P&L aktualisiert
"""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[2]
ROI_LOG = REPO / "data" / "roi" / "log.parquet"
INVENTORY_PATH = REPO / "data" / "reference" / "race_inventory.parquet"


@st.cache_data(show_spinner=False)
def _load_log() -> pd.DataFrame:
    if not ROI_LOG.exists():
        return pd.DataFrame()
    return pd.read_parquet(ROI_LOG)


@st.cache_data(show_spinner=False)
def _load_inventory() -> pd.DataFrame:
    if not INVENTORY_PATH.exists():
        return pd.DataFrame()
    inv = pd.read_parquet(INVENTORY_PATH)[["race_id", "gp_name", "race_date"]]
    inv["race_date"] = pd.to_datetime(inv["race_date"])
    return inv


# --- Page body ------------------------------------------------------------

log = _load_log()

if log.empty:
    st.info(
        "No ROI data yet. Workflow:\n\n"
        "1. Sa-Abend nach Quali: **Stakes-Page** öffnen, Tipico-Quoten eintragen, "
        "**Save odds** klicken.\n"
        "2. So-Abend nach dem Rennen: `just post-race YEAR ROUND` im Terminal.\n\n"
        "Dann taucht hier das erste Ergebnis auf."
    )
    st.stop()

inv = _load_inventory()
if not inv.empty:
    log = log.merge(inv, on="race_id", how="left")
    log["race_date"] = pd.to_datetime(log.get("race_date", pd.NaT))
    log = log.sort_values("race_date")
else:
    log = log.sort_values(["year", "round"])
    log["gp_name"] = log["race_id"]

# --- Top metrics ----------------------------------------------------------

total_stake = log["stake_eur"].sum()
total_pnl = log["pnl_eur"].sum()
roi_pct = total_pnl / total_stake * 100 if total_stake > 0 else 0.0
win_rate = log["won"].mean() * 100
n_races = log["race_id"].nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Races tracked", n_races)
c2.metric("Total staked", f"€{total_stake:.2f}")
c3.metric("P&L", f"€{total_pnl:+.2f}")
c4.metric("ROI", f"{roi_pct:+.1f}%", delta=f"{win_rate:.0f}% win rate")

st.divider()

# --- Cumulative P&L chart -------------------------------------------------

st.subheader("Cumulative P&L")

log["cumulative_pnl"] = log["pnl_eur"].cumsum()
log["cumulative_stake"] = log["stake_eur"].cumsum()
log["cumulative_roi"] = log["cumulative_pnl"] / log["cumulative_stake"] * 100

# One point per bet, x = bet index (chronological)
log["bet_index"] = range(1, len(log) + 1)
log["label"] = log.apply(
    lambda r: f"{r.get('gp_name', r['race_id'])} · {r['driver_id']} · {r['target']}",
    axis=1,
)

pnl_chart = (
    alt.Chart(log)
    .mark_line(point=True, interpolate="monotone")
    .encode(
        x=alt.X("bet_index:Q", title="Bet #"),
        y=alt.Y("cumulative_pnl:Q", title="Cumulative P&L (€)"),
        color=alt.value("#1f77b4"),
        tooltip=[
            alt.Tooltip("label:N", title="Bet"),
            alt.Tooltip("odds:Q", format=".2f", title="Odds"),
            alt.Tooltip("edge:Q", format="+.1%", title="Edge"),
            alt.Tooltip("stake_eur:Q", format=".2f", title="Stake €"),
            alt.Tooltip("won:N", title="Won"),
            alt.Tooltip("pnl_eur:Q", format="+.2f", title="P&L €"),
            alt.Tooltip("cumulative_pnl:Q", format="+.2f", title="Cumul. P&L €"),
        ],
    )
    .properties(height=300)
    .interactive()
)

zero_line = (
    alt.Chart(pd.DataFrame({"y": [0]}))
    .mark_rule(strokeDash=[4, 4], color="gray", opacity=0.5)
    .encode(y="y:Q")
)

st.altair_chart((pnl_chart + zero_line), use_container_width=True)

# --- Per-race breakdown ---------------------------------------------------

st.subheader("Per-race breakdown")

race_summary = (
    log.groupby(["race_id", "gp_name"] if "gp_name" in log.columns else ["race_id"])
    .agg(
        bets=("pnl_eur", "count"),
        staked=("stake_eur", "sum"),
        pnl=("pnl_eur", "sum"),
        wins=("won", "sum"),
    )
    .reset_index()
)
race_summary["ROI%"] = (race_summary["pnl"] / race_summary["staked"] * 100).map(
    lambda x: f"{x:+.1f}%"
)
race_summary["staked"] = race_summary["staked"].map(lambda x: f"€{x:.2f}")
race_summary["pnl"] = race_summary["pnl"].map(lambda x: f"€{x:+.2f}")
race_summary["win rate"] = (race_summary["wins"] / race_summary["bets"] * 100).map(
    lambda x: f"{x:.0f}%"
)
race_summary = race_summary.drop(columns=["wins"])
st.dataframe(race_summary, hide_index=True, use_container_width=True)

# --- Individual bets ------------------------------------------------------

with st.expander("All individual bets"):
    display = log[
        ["gp_name" if "gp_name" in log.columns else "race_id",
         "target", "driver_id", "odds", "p_model", "edge",
         "stake_eur", "won", "pnl_eur"]
    ].copy()
    display.columns = [
        "GP", "target", "driver", "odds", "P(model)", "edge", "stake €", "won", "P&L €"
    ]
    display["P(model)"] = (display["P(model)"] * 100).map(lambda x: f"{x:.1f}%")
    display["edge"] = display["edge"].map(lambda x: f"{x:+.1%}")
    display["odds"] = display["odds"].map(lambda x: f"{x:.2f}")
    display["stake €"] = display["stake €"].map(lambda x: f"€{x:.2f}")
    display["P&L €"] = display["P&L €"].map(lambda x: f"€{x:+.2f}")
    display["won"] = display["won"].map({True: "✓", False: "✗"})
    st.dataframe(display, hide_index=True, use_container_width=True)
