"""ROI evaluation — Phase 4.

After a race, loads:
  - archived pre-race predictions (predictions/archive/{race_id}_{target}.parquet)
  - saved odds              (data/odds/{race_id}_{target}.json)
  - actual race results     (data/raw/jolpica/results/{race_id}.json)

Computes per-bet P&L and appends to data/roi/log.parquet.

CLI:
    python -m src.eval.roi run --year 2026 --round 6
    python -m src.eval.roi show
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from src.utils.paths import ODDS_DIR, PREDICTIONS_DIR, RAW_DIR, ROI_DIR

ROI_LOG = ROI_DIR / "log.parquet"
ARCHIVE_DIR = PREDICTIONS_DIR / "archive"
RESULTS_DIR = RAW_DIR / "jolpica" / "results"

MIN_STAKE_EUR = 1.0
_DEFAULT_MODEL = {"podium": "prob_ensemble_cal", "teammate": "prob_logisticregression_cal"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _actual_podium(race_id: str) -> set[str]:
    data = json.loads((RESULTS_DIR / f"{race_id}.json").read_text())
    results = data["MRData"]["RaceTable"]["Races"][0]["Results"]
    return {r["Driver"]["driverId"] for r in results if r["position"] in ("1", "2", "3")}


def _actual_positions(race_id: str) -> dict[str, int]:
    data = json.loads((RESULTS_DIR / f"{race_id}.json").read_text())
    results = data["MRData"]["RaceTable"]["Races"][0]["Results"]
    return {r["Driver"]["driverId"]: int(r["position"]) for r in results}


def _kelly_raw(odds: float, p: float) -> float:
    if odds <= 1.0:
        return 0.0
    return max(0.0, (p * odds - 1.0) / (odds - 1.0))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_race(
    year: int,
    round_: int,
    kelly_frac: float = 0.25,
    bankroll: float = 100.0,
) -> pd.DataFrame:
    """Return DataFrame of all placed bets with P&L for one race."""
    race_id = f"{year}_{round_:02d}"
    rows: list[dict] = []

    for target in ("podium", "teammate"):
        odds_path = ODDS_DIR / f"{race_id}_{target}.json"
        preds_path = ARCHIVE_DIR / f"{race_id}_{target}.parquet"

        if not odds_path.exists():
            print(f"[roi] no saved odds for {race_id} {target} — skipping")
            continue
        if not preds_path.exists():
            print(f"[roi] no archived predictions for {race_id} {target} — skipping")
            continue

        odds_map: dict[str, float] = json.loads(odds_path.read_text())
        preds = pd.read_parquet(preds_path)
        model_col = _DEFAULT_MODEL[target]

        if model_col not in preds.columns:
            print(f"[roi] column {model_col} missing in {preds_path.name} — skipping")
            continue

        # Build actual outcome lookup
        if target == "podium":
            podium = _actual_podium(race_id)
        else:
            pos = _actual_positions(race_id)
            # teammate pairs from constructor_id in the predictions
            from collections import defaultdict
            by_team: dict[str, list[str]] = defaultdict(list)
            for _, r in preds.iterrows():
                by_team[r["constructor_id"]].append(r["driver_id"])
            beat_teammate: dict[str, bool] = {}
            for drivers in by_team.values():
                if len(drivers) == 2:
                    a, b = drivers
                    if a in pos and b in pos:
                        beat_teammate[a] = pos[a] < pos[b]
                        beat_teammate[b] = pos[b] < pos[a]

        for _, row in preds.iterrows():
            driver_id = str(row["driver_id"])
            odds = float(odds_map.get(driver_id, 0.0))
            if odds <= 1.0:
                continue

            p = float(row[model_col])
            kelly_raw = _kelly_raw(odds, p)
            if kelly_raw <= 0:
                continue

            stake = kelly_raw * kelly_frac * bankroll
            if stake < MIN_STAKE_EUR:
                continue

            if target == "podium":
                won = driver_id in podium
            else:
                won = beat_teammate.get(driver_id, False)

            pnl = (odds - 1.0) * stake if won else -stake
            rows.append(
                {
                    "race_id": race_id,
                    "year": year,
                    "round": round_,
                    "target": target,
                    "driver_id": driver_id,
                    "p_model": round(p, 4),
                    "odds": odds,
                    "implied_prob": round(1.0 / odds, 4),
                    "edge": round(p - 1.0 / odds, 4),
                    "kelly_frac": kelly_frac,
                    "stake_eur": round(stake, 2),
                    "bankroll_eur": bankroll,
                    "won": won,
                    "pnl_eur": round(pnl, 2),
                }
            )

    return pd.DataFrame(rows)


def append_to_log(df: pd.DataFrame) -> None:
    ROI_DIR.mkdir(parents=True, exist_ok=True)
    if ROI_LOG.exists():
        existing = pd.read_parquet(ROI_LOG)
        if not df.empty:
            existing = existing[~existing["race_id"].isin(df["race_id"].unique())]
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df
    combined.to_parquet(ROI_LOG, index=False)
    print(f"[roi] log updated -> {ROI_LOG}  ({len(combined)} total bets)")


def load_log() -> pd.DataFrame:
    if not ROI_LOG.exists():
        return pd.DataFrame()
    return pd.read_parquet(ROI_LOG)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Evaluate one race and append to log")
    run_p.add_argument("--year", type=int, required=True)
    run_p.add_argument("--round", type=int, dest="round_", required=True)
    run_p.add_argument("--kelly-frac", type=float, default=0.25)
    run_p.add_argument("--bankroll", type=float, default=100.0)

    sub.add_parser("show", help="Print ROI log summary")

    args = p.parse_args(argv)

    if args.cmd == "run":
        df = evaluate_race(args.year, args.round_, args.kelly_frac, args.bankroll)
        if df.empty:
            print("[roi] no bets to record — odds saved for this race?")
            return 0
        print(df[["driver_id", "target", "odds", "edge", "stake_eur", "won", "pnl_eur"]].to_string())
        append_to_log(df)

    elif args.cmd == "show":
        df = load_log()
        if df.empty:
            print("[roi] log is empty — run post-race after entering odds")
            return 0
        total_stake = df["stake_eur"].sum()
        total_pnl = df["pnl_eur"].sum()
        roi = total_pnl / total_stake * 100 if total_stake > 0 else 0.0
        win_rate = df["won"].mean() * 100
        print(f"[roi] {len(df)} bets across {df['race_id'].nunique()} races")
        print(f"      staked: €{total_stake:.2f}  P&L: €{total_pnl:+.2f}  ROI: {roi:+.1f}%  win rate: {win_rate:.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
