"""Stakes page -- Phase 3.5. Kelly-staking calculator on top of Next Race predictions.

Inputs:
- Bankroll (EUR) -- total amount the user is willing to deploy.
- Kelly fraction (0.0-1.0, default 0.25 = Quarter-Kelly) -- shrink factor against
  raw Kelly to absorb model-miscalibration risk.
- Decimal odds per driver, entered in an `st.data_editor` table.

Output per target (podium + teammate H2H):
- Per value-bet row: odds, edge (P_model - implied), Kelly %, EUR stake.
- Total stake summary + warning if total > bankroll.

Model choice mirrors the Phase-1.5 holdout-test winner: ensemble for podium
(ties XGB), LogReg for teammate. No user-visible model selector here -- the
"Next Race" page already serves that; this page is the betting calculator.

Kelly math (single binary bet, decimal odds D, win prob p):
    f* = (p*D - 1) / (D - 1)
clipped to 0 when there is no edge. We then multiply by the user's chosen
fraction (default 0.25) and by the bankroll to get the EUR stake.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[2]
FEATURES_PATH = REPO / "data" / "features" / "next_race.parquet"

# Drop computed Kelly stakes below this EUR threshold from the output table.
# Bookmaker minimums (Tipico etc.) are typically 1.00, so Kelly fractions
# that would size a bet below this are not actionable -- skip rather than
# floor-up (flooring would over-stake and break the Kelly variance bound).
MIN_STAKE_EUR = 1.0

# Best calibrated model per target on the Phase-1.5 holdout test
# (Ensemble ties XGB for podium @ Brier 0.0616; LogReg wins teammate @ 0.1958).
_DEFAULT_MODEL = {"podium": "ensemble", "teammate": "logisticregression"}
_TARGET_DESCRIPTIONS = {
    "podium": "Bet pays if the driver finishes P1–P3. Model = ensemble.",
    "teammate": (
        "Bet pays if the driver finishes ahead of their constructor teammate. Model = logreg."
    ),
}


@st.cache_data(show_spinner=False)
def _load_predictions(target_short: str) -> pd.DataFrame:
    path = REPO / "predictions" / f"next_race_{target_short}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        return pd.DataFrame()
    return pd.read_parquet(FEATURES_PATH)


def _kelly_raw(odds: float, p: float) -> float:
    """Single-bet Kelly fraction, clipped to 0. NaN-safe.

    f* = (p*D - 1) / (D - 1). Returns 0 for invalid odds (<= 1) or non-positive
    edge -- "no bet" sits naturally at f*=0.
    """
    if pd.isna(odds) or pd.isna(p) or odds <= 1.0:
        return 0.0
    return max(0.0, (p * odds - 1.0) / (odds - 1.0))


def _enrich_with_names(preds: pd.DataFrame) -> pd.DataFrame:
    """Merge driver_family_name + constructor_name from features.parquet. Falls
    back to raw ids if the features file is missing so the page still renders.
    """
    features = _load_features()
    if not features.empty:
        names = features[["driver_id", "driver_family_name", "constructor_name"]].drop_duplicates(
            subset=["driver_id"]
        )
        preds = preds.merge(names, on="driver_id", how="left")
    preds["driver_family_name"] = preds.get("driver_family_name", preds["driver_id"]).fillna(
        preds["driver_id"]
    )
    preds["constructor_name"] = preds.get("constructor_name", preds["constructor_id"]).fillna(
        preds["constructor_id"]
    )
    return preds


def _render_target_section(target_short: str, bankroll: float, kelly_frac: float) -> None:
    preds = _load_predictions(target_short)
    if preds.empty:
        st.warning(
            f"No predictions yet for `next_race_{target_short}`. "
            f"Run `just predict-next YEAR ROUND` locally and re-deploy."
        )
        return

    preds = _enrich_with_names(preds)
    model_col = f"prob_{_DEFAULT_MODEL[target_short]}_cal"

    # Editor input: sorted by model probability so the most-likely (= usually
    # the most interesting to bet on) drivers are at the top of the table.
    editor_df = (
        pd.DataFrame(
            {
                "driver": preds["driver_family_name"],
                "team": preds["constructor_name"],
                "grid": preds["grid"].astype(int),
                "P": (preds[model_col] * 100).round(1),  # 0-100 scale for %.1f%% format
                "odds": 0.0,
            }
        )
        .sort_values("P", ascending=False)
        .reset_index(drop=True)
    )

    edited = st.data_editor(
        editor_df,
        column_config={
            "driver": st.column_config.TextColumn("driver", disabled=True),
            "team": st.column_config.TextColumn("team", disabled=True),
            "grid": st.column_config.NumberColumn("grid", disabled=True, format="%d"),
            "P": st.column_config.NumberColumn(
                "P(model)",
                disabled=True,
                format="%.1f%%",
                help="Calibrated probability from the Phase-1.5 best model.",
            ),
            "odds": st.column_config.NumberColumn(
                "Odds",
                min_value=0.0,
                max_value=999.0,
                step=0.05,
                format="%.2f",
                help="Decimal odds (e.g. 3.50). Leave 0 to skip this driver.",
            ),
        },
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        key=f"stakes_editor_{target_short}",
    )

    # Compute per-row Kelly + stake. Convert P back to 0-1 scale for the formula.
    edited = edited.copy()
    edited["p_dec"] = edited["P"] / 100.0
    edited["implied"] = [
        1.0 / o if (pd.notna(o) and o > 1.0) else float("nan") for o in edited["odds"]
    ]
    edited["edge"] = edited["p_dec"] - edited["implied"]
    edited["kelly_raw"] = [
        _kelly_raw(o, p) for o, p in zip(edited["odds"], edited["p_dec"], strict=True)
    ]
    edited["kelly_pct"] = edited["kelly_raw"] * kelly_frac
    edited["stake"] = edited["kelly_pct"] * bankroll

    value_bets = edited[edited["kelly_raw"] > 0].copy()
    bets = value_bets[value_bets["stake"] >= MIN_STAKE_EUR].copy()
    any_odds = (edited["odds"] > 1.0).any()
    if bets.empty:
        if not value_bets.empty:
            st.info(
                f"Value bets found but all sized below the €{MIN_STAKE_EUR:.0f} minimum — "
                "raise the bankroll or the Kelly fraction to take them, or wait for "
                "a race with a bigger edge."
            )
        elif any_odds:
            st.info(
                "No value bets at the given odds — for every entered row the implied "
                "probability already exceeds the model's P. Either the odds are too "
                "short or the model disagrees with the market."
            )
        else:
            st.caption("Enter decimal odds in the table above to compute stakes.")
        return

    display = pd.DataFrame(
        {
            "driver": bets["driver"],
            "odds": bets["odds"].map(lambda x: f"{x:.2f}"),
            "edge": (bets["edge"] * 100).map(lambda x: f"+{x:.1f}%"),
            "Kelly%": (bets["kelly_pct"] * 100).map(lambda x: f"{x:.2f}%"),
            "stake": bets["stake"].map(lambda x: f"€{x:.2f}"),
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")

    total = float(bets["stake"].sum())
    pct = (total / bankroll * 100) if bankroll > 0 else 0.0
    c1, c2 = st.columns(2)
    c1.metric("Total stake", f"€{total:.2f}")
    c2.metric("Bankroll used", f"{pct:.1f}%")
    if total > bankroll:
        st.warning(
            f"Total stake (€{total:.2f}) exceeds bankroll (€{bankroll:.2f}). "
            "Lower the Kelly fraction or skip some bets — Kelly is a single-bet "
            "rule, the sum across independent bets can blow past 100%."
        )


# --- Page body --------------------------------------------------------

st.markdown("### Kelly Sizing")
st.caption(
    "Enter the decimal odds your bookmaker offers per driver and the calculator "
    "returns the Kelly-optimal stake against the model's calibrated probability. "
    "Default is Quarter-Kelly (0.25×) — robust against imperfect calibration."
)

c1, c2 = st.columns(2)
bankroll = c1.number_input(
    "Bankroll (€)",
    min_value=0.0,
    value=100.0,
    step=10.0,
    format="%.2f",
    key="stakes_bankroll",
    help="Total amount you would be willing to deploy across all bets this race.",
)
kelly_frac = c2.slider(
    "Kelly fraction",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    key="stakes_kelly_frac",
    help="Multiplier on raw Kelly. 0.25 = Quarter-Kelly (recommended), 1.0 = full Kelly.",
)

st.divider()
st.subheader("Podium bets")
st.caption(_TARGET_DESCRIPTIONS["podium"])
_render_target_section("podium", bankroll, kelly_frac)

st.divider()
st.subheader("Teammate H2H bets")
st.caption(_TARGET_DESCRIPTIONS["teammate"])
_render_target_section("teammate", bankroll, kelly_frac)

st.divider()
with st.expander("How Kelly sizing works"):
    st.markdown(
        """
- **Decimal odds D** — the bookmaker's offer. €1 staked returns €D if you win.
- **Implied probability** — `1/D`. What the bookmaker effectively prices,
  inclusive of their margin (so the implied probs across all outcomes of a
  market sum to >100%, the excess is the overround).
- **Edge** — `P(model) − 1/D`. Positive edge means the model thinks the bet
  is mispriced in your favour.
- **Raw Kelly fraction** — `f* = (P·D − 1) / (D − 1)`. The bankroll fraction
  that maximises expected log-growth for a single bet. Clipped to 0 when
  edge ≤ 0 (i.e. don't bet).
- **Your fraction** (slider) — multiplies `f*`. Half-Kelly halves variance
  for ~25% less expected growth; Quarter-Kelly halves variance again.
  Use full Kelly only if you trust the calibration completely — in practice
  even calibrated models are slightly off, and Kelly is brutally punishing
  when `P` is overestimated.
- **Stake** — `kelly_pct × bankroll`.

The calculator treats each driver as an independent single bet — so the
**sum** across multiple bets can exceed 100% of bankroll. That is a known
limitation of single-bet Kelly: it does not coordinate across simultaneous
bets. The page surfaces a warning when that happens; lower the fraction
or skip the weakest-edge bets.
"""
    )
