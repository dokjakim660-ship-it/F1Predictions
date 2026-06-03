"""Pre-Quali Stakes page -- Phase 4.2.6. Kelly staking on the four qualifying
markets, placed BEFORE qualifying.

Mirrors the pre-race Stakes page but reads the pre-quali predictions
(`predictions/next_race_prequali_{target}.parquet`, written by
`just predict-pre-quali YEAR ROUND`). Each driver carries probabilities for two
timing modes (pre_weekend / post_fp2); we bet at the richest available mode --
post_fp2 on a normal weekend, pre_weekend on a sprint weekend where no FP2
long-run data exists.

Markets:
- pole           -- driver takes pole position.
- top3_quali     -- driver qualifies P1-P3.
- top10_quali    -- driver reaches Q3 (qualifies P1-P10).
- teammate_quali -- driver out-qualifies their constructor teammate.

Model = LogisticRegression calibrated probs (the documented pre-quali default,
most robust at this N). Calibrated -- not raw -- because Kelly needs probabilities
that mean what they say, and raw probs are deliberately sharpened for ranking.

Save odds persists `data/odds/{race_id}_{target}.json` and archives the
prediction parquet to `predictions/archive/{race_id}_prequali_{target}.parquet`
so `just post-quali` can compute realised ROI after qualifying.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import streamlit as st

import ui

REPO = Path(__file__).resolve().parents[2]
ODDS_DIR = REPO / "data" / "odds"
ARCHIVE_DIR = REPO / "predictions" / "archive"

# Drop computed Kelly stakes below this EUR threshold (bookmaker minimums are
# typically 1.00, so smaller sizes are not actionable -- skip rather than
# floor-up, which would over-stake and break the Kelly variance bound).
MIN_STAKE_EUR = 1.0

# LogReg is the documented pre-quali default (most robust at this N).
_MODEL_KEY = "logisticregression"

_TARGETS = {
    "pole": "Pole",
    "top3_quali": "Top-3 Q",
    "top10_quali": "Top-10 Q (Q3)",
    "teammate_quali": "Teammate Q",
}
_TARGET_DESCRIPTIONS = {
    "pole": "Bet pays if the driver takes pole position.",
    "top3_quali": "Bet pays if the driver qualifies P1–P3.",
    "top10_quali": "Bet pays if the driver reaches Q3 (qualifies P1–P10).",
    "teammate_quali": "Bet pays if the driver out-qualifies their constructor teammate.",
}


def _kelly_raw(odds: float, p: float) -> float:
    """Single-bet Kelly fraction, clipped to 0. NaN-safe.

    f* = (p*D - 1) / (D - 1). Mirrors src.utils.kelly.kelly_raw, kept local so
    the page stays self-contained for the HF Spaces deploy (which ships app/ but
    not src/).
    """
    if pd.isna(odds) or pd.isna(p) or odds <= 1.0:
        return 0.0
    return max(0.0, (p * odds - 1.0) / (odds - 1.0))


@st.cache_data(show_spinner=False)
def _load_predictions(target_short: str) -> pd.DataFrame:
    path = REPO / "predictions" / f"next_race_prequali_{target_short}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _bet_mode(preds: pd.DataFrame) -> str:
    """Richest deployable mode: post_fp2 unless this is a sprint weekend."""
    has_fp2 = int(preds["has_fp2"].fillna(0).max())
    return "post_fp2" if has_fp2 else "pre_weekend"


def _odds_path(race_id: str, target_short: str) -> Path:
    return ODDS_DIR / f"{race_id}_{target_short}.json"


def _load_saved_odds(race_id: str, target_short: str) -> dict[str, float]:
    p = _odds_path(race_id, target_short)
    if p.exists():
        return json.loads(p.read_text())
    return {}


def _save_odds(race_id: str, target_short: str, odds_map: dict[str, float]) -> None:
    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    _odds_path(race_id, target_short).write_text(json.dumps(odds_map, indent=2))


def _archive_predictions(race_id: str, target_short: str) -> None:
    src = REPO / "predictions" / f"next_race_prequali_{target_short}.parquet"
    if src.exists():
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, ARCHIVE_DIR / f"{race_id}_prequali_{target_short}.parquet")


def _render_target_section(target_short: str, bankroll: float, kelly_frac: float) -> None:
    preds = _load_predictions(target_short)
    if preds.empty:
        st.warning(
            f"No pre-quali predictions yet for `{target_short}`. "
            f"Run `just predict-pre-quali YEAR ROUND` locally and re-deploy."
        )
        return

    preds = preds.copy()
    preds["driver_family_name"] = preds["driver_family_name"].fillna(preds["driver_id"])
    preds["constructor_name"] = preds["constructor_name"].fillna(preds["constructor_id"])

    mode = _bet_mode(preds)
    model_col = f"prob_{mode}_{_MODEL_KEY}_cal"
    if model_col not in preds.columns:
        st.error(f"Missing column `{model_col}` in predictions — re-run predict-pre-quali.")
        return

    race_id = str(preds.iloc[0]["race_id"])
    saved_odds = _load_saved_odds(race_id, target_short)

    mode_label = "Post-FP2" if mode == "post_fp2" else "Pre-Weekend (sprint — no FP2)"
    st.caption(f"Betting mode: **{mode_label}** · model: logreg (calibrated)")

    editor_df = (
        pd.DataFrame(
            {
                "driver_id": preds["driver_id"],
                "driver": preds["driver_family_name"],
                "team": preds["constructor_name"],
                "recentQ": preds["recent_quali_pos"].round(1),
                "P": (preds[model_col] * 100).round(1),
                "odds": preds["driver_id"].map(lambda d: saved_odds.get(d, 0.0)),
            }
        )
        .sort_values("P", ascending=False)
        .reset_index(drop=True)
    )

    edited = st.data_editor(
        editor_df,
        column_config={
            "driver_id": None,  # hidden — used for save key only
            "driver": st.column_config.TextColumn("driver", disabled=True),
            "team": st.column_config.TextColumn("team", disabled=True),
            "recentQ": st.column_config.NumberColumn(
                "recentQ",
                disabled=True,
                format="%.1f",
                help="Driver's recent average qualifying position (last 5).",
            ),
            "P": st.column_config.NumberColumn(
                "P(model)",
                disabled=True,
                format="%.1f%%",
                help="Calibrated pre-quali probability (logreg).",
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
        key=f"qstakes_editor_{target_short}",
    )

    if st.button(f"Save odds for {race_id}", key=f"qsave_odds_{target_short}"):
        try:
            odds_map = {
                str(row["driver_id"]): float(row["odds"])
                for _, row in edited.iterrows()
                if float(row["odds"]) > 1.0
            }
            _save_odds(race_id, target_short, odds_map)
            _archive_predictions(race_id, target_short)
            st.success(
                f"Saved {len(odds_map)} odds for {race_id} ({target_short}). "
                "Predictions archived for post-quali ROI eval."
            )
        except Exception as e:
            st.error(f"Could not save (running on HF Spaces read-only fs?): {e}")

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
                "raise the bankroll or the Kelly fraction to take them."
            )
        elif any_odds:
            st.info(
                "No value bets at the given odds — the implied probability already "
                "exceeds the model's P for every entered row."
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
            "Lower the Kelly fraction or skip some bets — Kelly is a single-bet rule."
        )


# --- Page body --------------------------------------------------------

ui.page_header(
    "Pre-Quali Stakes",
    eyebrow="Betting · Kelly sizing",
    desc="Kelly-optimal stakes for the four qualifying markets, placed before "
    "qualifying. Enter the decimal odds your bookmaker offers and the calculator "
    "sizes each bet against the model's calibrated probability. Default is "
    "Quarter-Kelly (0.25×).",
)

# Sprint-weekend banner: pull has_fp2 from whichever target is available.
for _t in _TARGETS:
    _probe = _load_predictions(_t)
    if not _probe.empty:
        if int(_probe["has_fp2"].fillna(0).max()) == 0:
            st.warning(
                "**Sprint weekend** — no FP2 long-run data, so bets use the Pre-Weekend mode."
            )
        break

c1, c2 = st.columns(2)
bankroll = c1.number_input(
    "Bankroll (€)",
    min_value=0.0,
    value=100.0,
    step=10.0,
    format="%.2f",
    key="qstakes_bankroll",
    help="Total amount you would deploy across all bets this weekend.",
)
kelly_frac = c2.slider(
    "Kelly fraction",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    key="qstakes_kelly_frac",
    help="Multiplier on raw Kelly. 0.25 = Quarter-Kelly (recommended), 1.0 = full Kelly.",
)

target_short = st.radio(
    "Market",
    list(_TARGETS),
    format_func=lambda t: _TARGETS[t],
    horizontal=True,
    key="qstakes_target",
)

ui.section(f"{_TARGETS[target_short]} bets")
st.caption(_TARGET_DESCRIPTIONS[target_short])
_render_target_section(target_short, bankroll, kelly_frac)
