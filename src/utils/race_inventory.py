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
from pathlib import Path

import pandas as pd

from src.ingest.jolpica_ingest import SCHEDULE_DIR
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
