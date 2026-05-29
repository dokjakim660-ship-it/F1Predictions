"""ROI evaluation — Phase 4.

After a race, loads:
  - archived pre-race predictions (predictions/archive/{race_id}_{target}.parquet)
  - saved odds              (data/odds/{race_id}_{target}.json)
  - actual race results     (data/raw/jolpica/results/{race_id}.json)

Computes per-bet P&L and appends to data/roi/log.parquet.

Phase 4.2.7 adds the four pre-quali markets (pole / top3_quali / top10_quali /
teammate_quali). Their archived predictions live under the `_prequali_` prefix,
and the realised outcome comes from the target columns of the feature table
(data/features/mvp.parquet) — exactly the re-ranked, tie-handled definition the
models trained against — so `just build` must have run for the completed race.

CLI:
    python -m src.eval.roi run --year 2026 --round 6        # race markets
    python -m src.eval.roi run-quali --year 2026 --round 6  # quali markets
    python -m src.eval.roi show
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

import pandas as pd

from src.utils.kelly import kelly_raw
from src.utils.paths import FEATURES_DIR, ODDS_DIR, PREDICTIONS_DIR, RAW_DIR, ROI_DIR

ROI_LOG = ROI_DIR / "log.parquet"
ARCHIVE_DIR = PREDICTIONS_DIR / "archive"
RESULTS_DIR = RAW_DIR / "jolpica" / "results"
MVP_FEATURES = FEATURES_DIR / "mvp.parquet"

MIN_STAKE_EUR = 1.0
_DEFAULT_MODEL = {"podium": "prob_ensemble_cal", "teammate": "prob_logisticregression_cal"}

# Pre-quali markets: bet target -> the feature-table target column holding the
# realised 0/1 outcome (NaN where the driver set no quali time).
_QUALI_TARGET_COL = {
    "pole": "target_pole",
    "top3_quali": "target_top3_quali",
    "top10_quali": "target_top10_quali",
    "teammate_quali": "target_quali_beat_teammate",
}
# LogReg is the documented pre-quali default (most robust at this N).
_QUALI_MODEL_KEY = "logisticregression"


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


def _place_bets(
    preds: pd.DataFrame,
    model_col: str,
    odds_map: dict[str, float],
    won_fn: Callable[[str], bool],
    *,
    race_id: str,
    year: int,
    round_: int,
    target: str,
    kelly_frac: float,
    bankroll: float,
) -> list[dict]:
    """Size + settle one (market) across its drivers, returning bet-log rows.

    A bet is placed only when odds > 1 and the Kelly fraction sizes it at or
    above the minimum stake. `won_fn(driver_id)` returns the realised outcome.
    """
    rows: list[dict] = []
    for _, row in preds.iterrows():
        driver_id = str(row["driver_id"])
        odds = float(odds_map.get(driver_id, 0.0))
        if odds <= 1.0:
            continue

        p = float(row[model_col])
        k_raw = kelly_raw(odds, p)
        if k_raw <= 0:
            continue

        stake = k_raw * kelly_frac * bankroll
        if stake < MIN_STAKE_EUR:
            continue

        won = won_fn(driver_id)
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
    return rows


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def _teammate_beat_lookup(preds: pd.DataFrame, pos: dict[str, int]) -> dict[str, bool]:
    """driver_id -> beat their teammate, from constructor pairs in `preds`."""
    from collections import defaultdict

    by_team: dict[str, list[str]] = defaultdict(list)
    for _, r in preds.iterrows():
        by_team[r["constructor_id"]].append(r["driver_id"])
    beat: dict[str, bool] = {}
    for drivers in by_team.values():
        if len(drivers) == 2:
            a, b = drivers
            if a in pos and b in pos:
                beat[a] = pos[a] < pos[b]
                beat[b] = pos[b] < pos[a]
    return beat


def evaluate_race(
    year: int,
    round_: int,
    kelly_frac: float = 0.25,
    bankroll: float = 100.0,
) -> pd.DataFrame:
    """Return DataFrame of all placed bets with P&L for one race (race markets)."""
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

        if target == "podium":
            podium = _actual_podium(race_id)

            def won_fn(driver_id: str, _podium: set[str] = podium) -> bool:
                return driver_id in _podium
        else:
            beat = _teammate_beat_lookup(preds, _actual_positions(race_id))

            def won_fn(driver_id: str, _beat: dict[str, bool] = beat) -> bool:
                return _beat.get(driver_id, False)

        rows.extend(
            _place_bets(
                preds,
                model_col,
                odds_map,
                won_fn,
                race_id=race_id,
                year=year,
                round_=round_,
                target=target,
                kelly_frac=kelly_frac,
                bankroll=bankroll,
            )
        )

    return pd.DataFrame(rows)


def _actual_quali_outcomes(race_id: str, target_col: str) -> dict[str, bool]:
    """driver_id -> realised 0/1 quali outcome from the feature table.

    Reads target_col from mvp.parquet for the given race. Drivers with a NaN
    target (no quali time set) are simply absent — a bet on them settles as a
    loss via the default in the caller. Raises if the race has no rows yet
    (feature table not rebuilt for the completed weekend).
    """
    df = pd.read_parquet(MVP_FEATURES, columns=["race_id", "driver_id", target_col])
    race = df[df["race_id"] == race_id]
    if race.empty:
        raise RuntimeError(
            f"no rows for {race_id} in {MVP_FEATURES.name} — run `just build` after "
            "the weekend so the realised quali targets are present."
        )
    return {
        str(r["driver_id"]): bool(r[target_col] == 1.0)
        for _, r in race.iterrows()
        if pd.notna(r[target_col])
    }


def evaluate_quali_race(
    year: int,
    round_: int,
    kelly_frac: float = 0.25,
    bankroll: float = 100.0,
) -> pd.DataFrame:
    """Return DataFrame of all placed bets with P&L for one race (quali markets)."""
    race_id = f"{year}_{round_:02d}"
    rows: list[dict] = []

    for target, target_col in _QUALI_TARGET_COL.items():
        odds_path = ODDS_DIR / f"{race_id}_{target}.json"
        preds_path = ARCHIVE_DIR / f"{race_id}_prequali_{target}.parquet"

        if not odds_path.exists():
            print(f"[roi] no saved odds for {race_id} {target} — skipping")
            continue
        if not preds_path.exists():
            print(f"[roi] no archived predictions for {race_id} {target} — skipping")
            continue

        odds_map: dict[str, float] = json.loads(odds_path.read_text())
        preds = pd.read_parquet(preds_path)

        # Bet at the richest deployable mode: post_fp2 unless this is a sprint
        # weekend (no FP2), matching the Pre-Quali Stakes page.
        has_fp2 = int(preds["has_fp2"].fillna(0).max())
        mode = "post_fp2" if has_fp2 else "pre_weekend"
        model_col = f"prob_{mode}_{_QUALI_MODEL_KEY}_cal"
        if model_col not in preds.columns:
            print(f"[roi] column {model_col} missing in {preds_path.name} — skipping")
            continue

        outcomes = _actual_quali_outcomes(race_id, target_col)

        def won_fn(driver_id: str, _out: dict[str, bool] = outcomes) -> bool:
            return _out.get(driver_id, False)

        rows.extend(
            _place_bets(
                preds,
                model_col,
                odds_map,
                won_fn,
                race_id=race_id,
                year=year,
                round_=round_,
                target=target,
                kelly_frac=kelly_frac,
                bankroll=bankroll,
            )
        )

    return pd.DataFrame(rows)


def append_to_log(df: pd.DataFrame) -> None:
    ROI_DIR.mkdir(parents=True, exist_ok=True)
    if ROI_LOG.exists():
        existing = pd.read_parquet(ROI_LOG)
        if not df.empty:
            # Replace only the (race_id, target) combinations being rewritten, so
            # re-running the quali markets does not clobber the race markets for
            # the same race_id (both share race_id but differ on target).
            written = set(zip(df["race_id"], df["target"], strict=True))
            existing_pairs = zip(existing["race_id"], existing["target"], strict=True)
            mask = [pair not in written for pair in existing_pairs]
            existing = existing[mask]
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

    for cmd, helptext in (
        ("run", "Evaluate the race markets (podium/teammate) and append to log"),
        ("run-quali", "Evaluate the quali markets (pole/top3/top10/teammate-Q) and append to log"),
    ):
        sp = sub.add_parser(cmd, help=helptext)
        sp.add_argument("--year", type=int, required=True)
        sp.add_argument("--round", type=int, dest="round_", required=True)
        sp.add_argument("--kelly-frac", type=float, default=0.25)
        sp.add_argument("--bankroll", type=float, default=100.0)

    sub.add_parser("show", help="Print ROI log summary")

    args = p.parse_args(argv)

    if args.cmd in ("run", "run-quali"):
        evaluator = evaluate_race if args.cmd == "run" else evaluate_quali_race
        df = evaluator(args.year, args.round_, args.kelly_frac, args.bankroll)
        if df.empty:
            print("[roi] no bets to record — odds saved for this race?")
            return 0
        print(
            df[["driver_id", "target", "odds", "edge", "stake_eur", "won", "pnl_eur"]].to_string()
        )
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
        print(
            f"      staked: €{total_stake:.2f}  P&L: €{total_pnl:+.2f}  "
            f"ROI: {roi:+.1f}%  win rate: {win_rate:.0f}%"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
