"""FastF1 ingest: per-race Q + R + FP2 weekend dumps to data/raw/fastf1/.

Each session writes four artifacts to data/raw/fastf1/{year}_{round:02d}_{code}/:
    session_info.json  - small metadata blob, also doubles as "done" marker
    results.parquet    - driver-level result table (one row per driver)
    laps.parquet       - lap-level table (Compound, LapTime, Stint, sectors, ...)
    weather.parquet    - track weather time-series

If a session is not available (sprint weekend, future race, FastF1 hiccup), the
session_info.json file is still written with an "error" key so re-runs skip it
without re-trying the network.

Subcommands:
    smoke                            - Phase 0 smoke (kept)
    race    --year Y --round N       - pull one race weekend (Q + R + FP2)
    season  --year Y                 - pull every completed race in a season
    all                              - pull every completed race in the inventory
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import fastf1
import pandas as pd

from src.utils.paths import FASTF1_CACHE, RAW_DIR
from src.utils.race_inventory import load_inventory

FASTF1_DIR = RAW_DIR / "fastf1"
SESSIONS: tuple[str, ...] = ("Q", "R", "FP2")


def _ensure_cache() -> None:
    FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE))


def _session_dir(year: int, round_no: int, code: str) -> Path:
    return FASTF1_DIR / f"{year}_{round_no:02d}_{code}"


def _is_done(out_dir: Path) -> bool:
    return (out_dir / "session_info.json").exists()


def _write_parquet_safe(df: pd.DataFrame | None, path: Path) -> int:
    if df is None or len(df) == 0:
        return 0
    df.reset_index(drop=True).to_parquet(path, index=False)
    return len(df)


def _save_session(year: int, round_no: int, code: str, *, refresh: bool = False) -> bool:
    out_dir = _session_dir(year, round_no, code)
    if not refresh and _is_done(out_dir):
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        s = fastf1.get_session(year, round_no, code)
        s.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as e:
        err = {"year": year, "round": round_no, "code": code, "error": str(e)[:300]}
        (out_dir / "session_info.json").write_text(json.dumps(err, indent=2))
        print(f"[fastf1] {year} R{round_no:02d} {code} - UNAVAILABLE ({str(e)[:80]})")
        return False

    event_name = ""
    event_date = ""
    if hasattr(s, "event") and s.event is not None:
        event_name = str(s.event.get("EventName", "") or "")
        event_date = str(s.event.get("EventDate", "") or "")

    info = {
        "year": year,
        "round": round_no,
        "code": code,
        "session_name": str(getattr(s, "name", "") or ""),
        "event_name": event_name,
        "event_date": event_date,
        "session_date": str(s.date) if getattr(s, "date", None) is not None else "",
        "total_laps": int(getattr(s, "total_laps", 0) or 0),
    }

    n_results = _write_parquet_safe(s.results, out_dir / "results.parquet")
    n_laps = _write_parquet_safe(s.laps, out_dir / "laps.parquet")
    n_weather = _write_parquet_safe(s.weather_data, out_dir / "weather.parquet")

    info["n_results"] = n_results
    info["n_laps"] = n_laps
    info["n_weather"] = n_weather
    (out_dir / "session_info.json").write_text(json.dumps(info, indent=2))

    print(
        f"[fastf1] {year} R{round_no:02d} {code} - OK "
        f"({n_results} drivers, {n_laps} laps, {n_weather} wx)"
    )
    return True


def pull_race_weekend(year: int, round_no: int, *, refresh: bool = False) -> int:
    _ensure_cache()
    ok = 0
    fresh_any = False
    for code in SESSIONS:
        was_done = _is_done(_session_dir(year, round_no, code))
        if _save_session(year, round_no, code, refresh=refresh):
            ok += 1
        if refresh or not was_done:
            fresh_any = True
    if not fresh_any:
        print(f"[fastf1] {year} R{round_no:02d} all {len(SESSIONS)} sessions cached")
    return ok


def pull_season(year: int, *, refresh: bool = False) -> None:
    inv = load_inventory()
    season = inv.query("year == @year").sort_values("round")
    if season.empty:
        raise ValueError(f"No races for {year} in race_inventory. Build inventory first.")
    today_iso = date.today().isoformat()
    for _, row in season.iterrows():
        race_date = str(row["race_date"])
        if race_date > today_iso:
            print(f"[fastf1] {year} R{int(row['round']):02d} {row['gp_name']:<30s} - future, skip")
            continue
        pull_race_weekend(year, int(row["round"]), refresh=refresh)


def pull_all(*, refresh: bool = False) -> None:
    inv = load_inventory()
    today_iso = date.today().isoformat()
    for _, row in inv.iterrows():
        if str(row["race_date"]) > today_iso:
            continue
        pull_race_weekend(int(row["year"]), int(row["round"]), refresh=refresh)


def smoke_test(year: int = 2024, gp: str = "Monaco", session: str = "R") -> int:
    _ensure_cache()
    print(f"[fastf1] loading {year} {gp} {session} ...")
    s = fastf1.get_session(year, gp, session)
    s.load(laps=True, telemetry=False, weather=False, messages=False)
    laps = s.laps
    print(f"[fastf1] OK - {len(laps)} laps, {laps['Driver'].nunique()} drivers")
    print(f"[fastf1] fastest lap: {laps['LapTime'].min()}")
    print(f"[fastf1] cache dir:   {FASTF1_CACHE}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("smoke", help="Phase 0 smoke")

    p_race = sub.add_parser("race", help="Pull one race weekend (Q + R + FP2)")
    p_race.add_argument("--year", type=int, required=True)
    p_race.add_argument("--round", type=int, required=True)
    p_race.add_argument("--refresh", action="store_true")

    p_season = sub.add_parser("season", help="Pull all completed races in a season")
    p_season.add_argument("--year", type=int, required=True)
    p_season.add_argument("--refresh", action="store_true")

    p_all = sub.add_parser("all", help="Pull every completed race in the inventory")
    p_all.add_argument("--refresh", action="store_true")

    args = p.parse_args(argv)

    if args.cmd in (None, "smoke"):
        return smoke_test()
    if args.cmd == "race":
        pull_race_weekend(args.year, args.round, refresh=args.refresh)
        return 0
    if args.cmd == "season":
        pull_season(args.year, refresh=args.refresh)
        return 0
    if args.cmd == "all":
        pull_all(refresh=args.refresh)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
