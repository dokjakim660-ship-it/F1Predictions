"""Jolpica-F1 (Ergast fork) ingest: per-season schedule + per-race results.

Subcommands:
    smoke                            - one-season schedule fetch (Phase 0 carry-over)
    schedule  --years Y [Y ...]      - pull and cache per-season schedule JSONs
    results   --year Y [--round N]   - pull race-results for a season (or one race)

All cached JSONs land under data/raw/jolpica/. Cached files are reused on rerun;
pass --refresh to force a re-download. The current calendar year is always
refreshed (races are still being added).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import requests

from src.utils.paths import RAW_DIR

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
JOLPICA_DIR = RAW_DIR / "jolpica"
SCHEDULE_DIR = JOLPICA_DIR / "schedules"
RESULTS_DIR = JOLPICA_DIR / "results"

_REQUEST_DELAY_S = 0.25
_RETRIES = 3
_TIMEOUT_S = 20


def _get(url: str) -> dict:
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            r = requests.get(url, timeout=_TIMEOUT_S)
            r.raise_for_status()
            time.sleep(_REQUEST_DELAY_S)
            return r.json()
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            wait = 2**attempt
            print(f"[jolpica] {url} failed ({e!s}) - retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"[jolpica] giving up on {url}") from last_exc


def fetch_schedule(year: int) -> dict:
    return _get(f"{JOLPICA_BASE}/{year}.json")


def save_schedule(year: int, *, refresh: bool = False) -> Path:
    SCHEDULE_DIR.mkdir(parents=True, exist_ok=True)
    target = SCHEDULE_DIR / f"{year}.json"
    if target.exists() and not refresh:
        print(f"[jolpica] schedule {year} cached -> {target.name}")
        return target
    data = fetch_schedule(year)
    target.write_text(json.dumps(data, indent=2))
    n_races = len(data["MRData"]["RaceTable"]["Races"])
    print(f"[jolpica] schedule {year} saved - {n_races} races -> {target.name}")
    return target


def fetch_results(year: int, round_no: int) -> dict:
    return _get(f"{JOLPICA_BASE}/{year}/{round_no}/results.json?limit=100")


def save_results(year: int, round_no: int, *, refresh: bool = False) -> tuple[Path, bool]:
    """Returns (path, was_fresh). was_fresh=True means a new fetch happened."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    target = RESULTS_DIR / f"{year}_{round_no:02d}.json"
    if target.exists() and not refresh:
        return target, False
    data = fetch_results(year, round_no)
    target.write_text(json.dumps(data, indent=2))
    return target, True


def bulk_pull_schedules(years: list[int], *, refresh: bool = False) -> list[Path]:
    current_year = date.today().year
    paths = []
    for y in years:
        # Always refresh the current season - calendar can shift mid-year.
        force = refresh or y >= current_year
        paths.append(save_schedule(y, refresh=force))
    return paths


def bulk_pull_results_all_seasons(*, refresh: bool = False) -> None:
    """Pull results for every cached schedule (every season we know about)."""
    for sched in sorted(SCHEDULE_DIR.glob("*.json")):
        year = int(sched.stem)
        bulk_pull_results(year, refresh=refresh)


def bulk_pull_results(year: int, *, refresh: bool = False) -> list[Path]:
    sched_path = save_schedule(year, refresh=(year >= date.today().year))
    sched = json.loads(sched_path.read_text())
    races = sched["MRData"]["RaceTable"]["Races"]
    today_iso = date.today().isoformat()

    paths = []
    for race in races:
        round_no = int(race["round"])
        if race["date"] > today_iso:
            print(f"[jolpica] {year} R{round_no:02d} {race['raceName']:<30s} - future, skip")
            continue
        out, fresh = save_results(year, round_no, refresh=refresh)
        marker = "fresh" if fresh else "cached"
        print(f"[jolpica] {year} R{round_no:02d} {race['raceName']:<30s} -> {out.name} ({marker})")
        paths.append(out)
    return paths


def smoke_test(year: int = 2024) -> int:
    data = fetch_schedule(year)
    races = data["MRData"]["RaceTable"]["Races"]
    print(f"[jolpica] OK - {len(races)} races in {year}")
    for race in races[:3]:
        print(f"  - R{race['round']:>2} {race['date']}  {race['raceName']}")
    if len(races) > 3:
        print(f"  ... and {len(races) - 3} more")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("smoke", help="Phase-0 smoke: pull 2024 schedule and print head")

    p_sched = sub.add_parser("schedule", help="Pull per-season schedule JSONs")
    p_sched.add_argument("--years", type=int, nargs="+", required=True)
    p_sched.add_argument("--refresh", action="store_true")

    p_res = sub.add_parser("results", help="Pull race-result JSONs for a season")
    p_res.add_argument(
        "--year", type=int, default=None, help="Pull for this season. Required unless --all is set."
    )
    p_res.add_argument("--round", type=int, default=None)
    p_res.add_argument("--all", action="store_true", help="Pull results for every cached season.")
    p_res.add_argument("--refresh", action="store_true")

    args = p.parse_args(argv)

    if args.cmd in (None, "smoke"):
        return smoke_test()
    if args.cmd == "schedule":
        bulk_pull_schedules(args.years, refresh=args.refresh)
        return 0
    if args.cmd == "results":
        if args.all:
            bulk_pull_results_all_seasons(refresh=args.refresh)
        elif args.year is None:
            p.error("results: pass --year YEAR or --all")
        elif args.round is not None:
            out, fresh = save_results(args.year, args.round, refresh=args.refresh)
            marker = "fresh" if fresh else "cached"
            print(f"[jolpica] {args.year} R{args.round:02d} results -> {out.name} ({marker})")
        else:
            bulk_pull_results(args.year, refresh=args.refresh)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
