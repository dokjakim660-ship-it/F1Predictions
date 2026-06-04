"""Build the canonical race inventory from cached Jolpica schedules.

Output: data/reference/race_inventory.parquet - one row per scheduled race
with stable (race_id, year, round) keys plus circuit coords used downstream
for Open-Meteo weather fetches.

Run *after* `python -m src.ingest.jolpica_ingest schedule --years ...`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from src.ingest.jolpica_ingest import SCHEDULE_DIR
from src.utils.gha import emit_output
from src.utils.paths import REFERENCE_DIR

INVENTORY_PATH = REFERENCE_DIR / "race_inventory.parquet"


def _row_from_jolpica_race(year: int, race: dict) -> dict:
    circuit = race["Circuit"]
    loc = circuit["Location"]
    round_no = int(race["round"])
    return {
        "race_id": f"{year}_{round_no:02d}",
        "year": year,
        "round": round_no,
        "gp_name": race["raceName"],
        "circuit_id": circuit["circuitId"],
        "circuit_name": circuit["circuitName"],
        "country": loc.get("country", ""),
        "locality": loc.get("locality", ""),
        "lat": float(loc["lat"]),
        "lon": float(loc["long"]),
        "race_date": race["date"],
        "race_time_utc": race.get("time", ""),
    }


def build_inventory(schedule_dir: Path = SCHEDULE_DIR) -> pd.DataFrame:
    files = sorted(schedule_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No schedule JSONs in {schedule_dir}. "
            "Run: python -m src.ingest.jolpica_ingest schedule --years 2018 ... first."
        )

    rows: list[dict] = []
    for f in files:
        year = int(f.stem)
        data = json.loads(f.read_text())
        for race in data["MRData"]["RaceTable"]["Races"]:
            rows.append(_row_from_jolpica_race(year, race))

    df = pd.DataFrame(rows)
    df["race_date"] = pd.to_datetime(df["race_date"]).dt.date
    df = df.sort_values(["year", "round"]).reset_index(drop=True)
    return df


def save_inventory(df: pd.DataFrame, path: Path = INVENTORY_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_inventory(path: Path = INVENTORY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m src.utils.race_inventory build")
    return pd.read_parquet(path)


def resolve_next_round(df: pd.DataFrame, reference_date: date) -> dict | None:
    """The earliest race whose date is on/after `reference_date`.

    Returns None if every race in the inventory is in the past. `days_until` is
    >= 0; it is 0 on race day itself (so a Sunday-morning run still resolves to
    that day's race).
    """
    dates = pd.to_datetime(df["race_date"]).dt.date
    future = df[dates >= reference_date].copy()
    if future.empty:
        return None
    future["_d"] = pd.to_datetime(future["race_date"]).dt.date
    nxt = future.sort_values("_d").iloc[0]
    race_date = nxt["_d"]
    return {
        "year": int(nxt["year"]),
        "round": int(nxt["round"]),
        "race_id": str(nxt["race_id"]),
        "gp_name": str(nxt["gp_name"]),
        "race_date": race_date.isoformat(),
        "days_until": (race_date - reference_date).days,
    }


def _print_summary(df: pd.DataFrame) -> None:
    print(f"[inventory] {len(df)} races across {df['year'].nunique()} seasons")
    by_year = df.groupby("year").size()
    for year, n in by_year.items():
        print(f"  {year}: {n} races")
    print(f"[inventory] first: {df.iloc[0]['race_id']}  {df.iloc[0]['gp_name']}")
    print(f"[inventory] last:  {df.iloc[-1]['race_id']}  {df.iloc[-1]['gp_name']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="Read cached schedules -> race_inventory.parquet")
    sub.add_parser("show", help="Print summary of the current inventory")
    p_next = sub.add_parser(
        "next-round", help="Resolve the upcoming race (for the predict-next CI job)"
    )
    p_next.add_argument(
        "--reference-date",
        default=None,
        help="YYYY-MM-DD to resolve against (default: today UTC).",
    )
    p_next.add_argument(
        "--max-days",
        type=int,
        default=3,
        help="has_race=true only if the next race is within this many days (default 3).",
    )

    args = p.parse_args(argv)

    if args.cmd == "build":
        df = build_inventory()
        out = save_inventory(df)
        print(f"[inventory] saved -> {out}")
        _print_summary(df)
        return 0
    if args.cmd == "show":
        df = load_inventory()
        _print_summary(df)
        return 0
    if args.cmd == "next-round":
        ref = (
            date.fromisoformat(args.reference_date)
            if args.reference_date
            else datetime.now(timezone.utc).date()
        )
        info = resolve_next_round(load_inventory(), ref)
        if info is None:
            print(f"[inventory] no race on/after {ref} -- inventory ends in the past.")
            emit_output(has_race="false")
            return 0
        within = info["days_until"] <= args.max_days
        print(
            f"[inventory] next race (ref {ref}): {info['race_id']} {info['gp_name']} "
            f"on {info['race_date']} ({info['days_until']}d away)  ->  "
            f"has_race={'true' if within else 'false'} (max-days {args.max_days})"
        )
        emit_output(
            has_race="true" if within else "false",
            year=info["year"],
            round=info["round"],
            race_id=info["race_id"],
            days_until=info["days_until"],
        )
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
