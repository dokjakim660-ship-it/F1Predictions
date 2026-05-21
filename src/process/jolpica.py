"""L2 processor: Jolpica race-result JSONs -> data/processed/results.parquet.

One row per (race_id, driver_id). Columns are a stable subset of the Jolpica
schema, normalised to the types we actually need downstream (ints, floats,
booleans, ISO date). DNFs are detected via the `status` field, not the
`position` field, because Jolpica still assigns a classification position to
drivers that retired after enough laps.

Run: `python -m src.process.jolpica build`
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

from src.ingest.jolpica_ingest import RESULTS_DIR
from src.utils.paths import PROCESSED_DIR

RESULTS_PARQUET = PROCESSED_DIR / "results.parquet"

# Jolpica `status` values that mean the driver crossed the finish line. Anything
# else (Retired, Engine, Collision, Accident, Disqualified, ...) is treated as DNF
# for modelling purposes -- the driver did not produce a classified race result
# even if the timing system kept logging a cumulative time.
_CLASSIFIED_PATTERNS = (
    re.compile(r"^Finished$"),
    re.compile(r"^\+\d+ Laps?$"),
    re.compile(r"^Lapped$"),
)

_LAP_TIME_RE = re.compile(r"^(?:(\d+):)?(\d+)\.(\d+)$")


def _lap_time_to_ms(s: str | None) -> float | None:
    """Parse "M:SS.mmm" or "SS.mmm" into milliseconds. Returns NaN-safe None."""
    if not s:
        return None
    m = _LAP_TIME_RE.match(s.strip())
    if not m:
        return None
    minutes = int(m.group(1)) if m.group(1) else 0
    seconds = int(m.group(2))
    millis = int(m.group(3).ljust(3, "0")[:3])
    return float(minutes * 60_000 + seconds * 1000 + millis)


def _is_classified(status: str) -> bool:
    return any(p.match(status) for p in _CLASSIFIED_PATTERNS)


def _safe_int(x: str | None) -> int | None:
    if x is None or x == "":
        return None
    try:
        return int(x)
    except ValueError:
        return None


def _safe_float(x: str | None) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _row_from_result(year: int, round_no: int, race_date: str, r: dict) -> dict:
    drv = r["Driver"]
    con = r["Constructor"]
    time_block = r.get("Time") or {}
    fl = r.get("FastestLap") or {}
    fl_time = (fl.get("Time") or {}).get("time")
    fl_speed = (fl.get("AverageSpeed") or {}).get("speed")
    status = r["status"]
    dnf = not _is_classified(status)
    # Jolpica logs a cumulative time even for Retired/Disqualified drivers (time
    # at which they exited / would have been at). Drop it for DNFs so downstream
    # code can treat `finish_time_ms IS NULL` as a clean DNF signal.
    finish_time_ms = None if dnf else _safe_int(time_block.get("millis"))

    return {
        "race_id": f"{year}_{round_no:02d}",
        "year": year,
        "round": round_no,
        "race_date": race_date,
        "driver_id": drv["driverId"],
        "driver_code": drv.get("code", ""),
        "driver_given_name": drv.get("givenName", ""),
        "driver_family_name": drv.get("familyName", ""),
        "driver_dob": drv.get("dateOfBirth", ""),
        "driver_nationality": drv.get("nationality", ""),
        "constructor_id": con["constructorId"],
        "constructor_name": con.get("name", ""),
        "grid": _safe_int(r.get("grid")),
        "finish_position": _safe_int(r.get("position")),
        "position_text": r.get("positionText", ""),
        "points": _safe_float(r.get("points")),
        "laps_completed": _safe_int(r.get("laps")),
        "status": status,
        "dnf": dnf,
        "finish_time_ms": finish_time_ms,
        "fastest_lap_ms": _lap_time_to_ms(fl_time),
        "fastest_lap_rank": _safe_int(fl.get("rank")),
        "fastest_lap_avg_speed_kph": _safe_float(fl_speed),
    }


def _rows_from_race_json(payload: dict) -> list[dict]:
    races = payload["MRData"]["RaceTable"]["Races"]
    if not races:
        return []
    race = races[0]
    year = int(race["season"])
    round_no = int(race["round"])
    race_date = race["date"]
    return [_row_from_result(year, round_no, race_date, r) for r in race["Results"]]


def build_results(results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    files = sorted(results_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No result JSONs in {results_dir}. "
            "Run: python -m src.ingest.jolpica_ingest results --all"
        )

    rows: list[dict] = []
    for f in files:
        payload = json.loads(f.read_text())
        rows.extend(_rows_from_race_json(payload))

    df = pd.DataFrame(rows)
    df["race_date"] = pd.to_datetime(df["race_date"]).dt.date
    df["driver_dob"] = pd.to_datetime(df["driver_dob"], errors="coerce").dt.date

    # Stable, deterministic ordering.
    df = df.sort_values(["year", "round", "finish_position", "driver_id"]).reset_index(drop=True)
    return df


def save_results(df: pd.DataFrame, path: Path = RESULTS_PARQUET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_results(path: Path = RESULTS_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m src.process.jolpica build")
    return pd.read_parquet(path)


def _print_summary(df: pd.DataFrame) -> None:
    n_races = df[["year", "round"]].drop_duplicates().shape[0]
    n_drivers = df["driver_id"].nunique()
    n_dnf = int(df["dnf"].sum())
    print(
        f"[process.jolpica] {len(df)} driver-race rows across {n_races} races, {n_drivers} drivers"
    )
    print(f"[process.jolpica] DNFs: {n_dnf} ({n_dnf / len(df):.1%})")
    by_year = df.groupby("year").size()
    for year, n in by_year.items():
        print(f"  {year}: {n} rows")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="RAW result JSONs -> data/processed/results.parquet")
    sub.add_parser("show", help="Print summary of the current results parquet")

    args = p.parse_args(argv)

    if args.cmd == "build":
        df = build_results()
        out = save_results(df)
        print(f"[process.jolpica] saved -> {out}")
        _print_summary(df)
        return 0
    if args.cmd == "show":
        df = load_results()
        _print_summary(df)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
