"""Methodology page — what the app predicts, how it's trained, what we learned."""

from __future__ import annotations

import streamlit as st

import metrics
import ui

ui.page_header(
    "Methodology",
    eyebrow="About · How it works",
    desc="What the models predict, where the signal comes from, and how it is "
    "evaluated.",
)

# Pretty names for the prose. The winning model + Brier are read live from the
# same holdout file the Next Race comparison card and Backtest page use, so this
# page can never drift out of sync with what the app actually shows.
_NICE = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "logisticregression": "LogisticRegression",
    "ensemble": "Ensemble",
    "top3quali": "Top-3-quali baseline",
}


def _best_line(target_short: str) -> str:
    bm = metrics.best_model(target_short)
    if bm is None:
        return "Best model: see **Backtest History** for the live holdout ranking."
    name, brier = bm
    return f"Best model: **{_NICE.get(name, name)} (calibrated)**, Brier **{brier:.4f}** on the holdout test."


# Holdout extent (rows/races) — read live so the figures match the data shipped.
_rows, _races = metrics.holdout_shape("podium")
_holdout_extent = (
    f"{_races} races, ~{round(_rows / 10) * 10} rows per target"
    if _rows
    else "the sealed post-2024-06 window"
)

st.markdown(
    f"""
### What the app predicts

Several F1 betting markets, grouped by **when** you would place the bet.

**After qualifying** — the grid is known, the race isn't:

- **Podium** — P(driver finishes P1–P3). Imbalanced base rate ~15 %.
  {_best_line("podium")}
- **Teammate H2H** — P(driver finishes ahead of their constructor team-mate).
  Balanced ~50 %. {_best_line("teammate")}
- **Grid & Finish Order** — a full predicted classification (every driver gets
  an exact place), via position-regression (Ridge default) vs. learning-to-rank.
- **DNF Risk** — P(driver retires), as a calibrated reliability estimate.

**Before qualifying** — Thursday/Friday, no grid yet:

- Four **qualifying markets** — pole, top-3, top-10 (Q3), and out-qualify the
  team-mate — each in two timing modes (**pre-weekend** vs. **post-FP2**) so you
  can see how much the FP2 long-run pace moves the forecast.

**Betting layer:**

- **Kelly stakes** (pre-race and pre-quali) size each bet against the model's
  calibrated probability — Quarter-Kelly by default, robust to miscalibration.
- The **ROI tracker** settles placed bets race-by-race: the actual
  "do we beat Tipico's margin?" scoreboard. The live loop runs from 2026.

### Where the signal comes from

One row = one driver × one race, joined from three sources. It started as ~31
MVP features and grew through a Phase-5 wave of additions:

- **FastF1 telemetry** — quali gap to pole, **FP2 long-run pace gap** (the
  secret weapon), short-run pace, pit-lane speed, race-start position gain.
- **Jolpica results** — grid, rolling driver & constructor form (finishes, DNF
  rate, points), team standings — all `.shift(1)`-lagged so race N sees only
  races strictly before it.
- **Track attributes** — length, corners, DRS zones, street vs. permanent, and
  a measured **overtaking index** rebuilt from lap-by-lap position changes.
- **Derived skill/strategy** — teammate quali gap, wet-weather skill delta,
  form momentum, and a de-biased team-execution residual (result vs. pace).
- Plus driver × track history, season progress, and an era flag for the 2022
  ground-effect rules.

### How it is evaluated

- **Walk-forward** validation, four expanding-window folds across 2018 →
  mid-2024. Optuna only ever sees these.
- **Holdout test** = 2024-07 onward ({_holdout_extent}). Sealed; the tuner
  never touches it. Headline metric is **Brier** for the probability markets,
  **position-MAE** for the ranking models.
- **Isotonic calibration** before any probability is reported (mandatory after
  `scale_pos_weight`), and paired bootstrap (1000×, races resampled) for every
  model comparison — so "model A beats B" means it beats sampling noise too.

### What the data keeps telling us

The honest part, and the most consistent finding across every extension:
**the starting grid carries most of the race-day signal.**

- A podium model fed *predicted* qualifying instead of the real grid loses
  significantly — and doesn't even beat the trivial "top-3 on the grid = podium"
  rule (holdout Brier 0.0752).
- Full-race ranking barely improves on simply using the starting order.
- Retirements are unpredictable beyond a team's recent reliability rate — no DNF
  model beats that baseline.

These negative results are kept on purpose: knowing what *doesn't* add edge is
as valuable as what does. It's also why the markets with the most independent
value are the **qualifying** ones — priced before the grid that dominates
everything downstream even exists. The open question is the only one that
matters: over a full season, do the calibrated edges clear the bookmaker's
margin?
    """
)
