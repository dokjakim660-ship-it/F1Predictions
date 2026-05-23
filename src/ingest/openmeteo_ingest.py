"""Open-Meteo historical-weather ingest: per-race hourly snapshot.

For each race in race_inventory.parquet we pull a 24-hour hourly window
on race_date at the circuit's lat/lon. The Open-Meteo Archive API is free,
stable, no key required.

For future races (Phase 3.1 next-race-predict) we instead hit the Forecast
endpoint, which goes ~16 days out. Forecast snapshots land in a separate
`{year}_{round:02d}.forecast.json` so they don't clobber the archive (which
becomes available a day or two after the race).

Subcommands:
    smoke                            - Phase 0 smoke (kept)
    race    --year Y --round N       - pull one race (archive)
    next    --year Y --round N       - pull forecast for a future race
    season  --year Y                 - pull all completed races in a season
    all                              - pull every completed race in the inventory
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
from src.utils.race_inventory import load_inventory

OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_DIR = RAW_DIR / "openmeteo"

_HOURLY_VARS = [
    "temperature_2m",
    "precipitation",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "relative_humidity_2m",
]

_REQUEST_DELAY_S = 0.2
_RETRIES = 3
_TIMEOUT_S = 20


def _get(url: str, params: dict) -> dict:
    last_exc: Exception | None = None
    for attempt in range(_RETRIES):
        try:
            r = requests.get(url, params=params, timeout=_TIMEOUT_S)
            r.raise_for_status()
            time.sleep(_REQUEST_DELAY_S)
            return r.json()
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            wait = 2**attempt
            print(f"[open-meteo] {url} failed ({e!s}) - retry in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"[open-meteo] giving up on {url}") from last_exc


def fetch_race_weather(lat: float, lon: float, race_date_iso: str) -> dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": race_date_iso,
        "end_date": race_date_iso,
        "hourly": ",".join(_HOURLY_VARS),
        "timezone": "auto",
    }
    return _get(OPENMETEO_ARCHIVE, params)


def fetch_race_forecast(lat: float, lon: float, race_date_iso: str) -> dict:
    """Pull the hourly forecast for a single race day from the /v1/forecast endpoint.

    Open-Meteo's forecast horizon is ~16 days; calling this for a race further out
    will return an error. Payload shape matches the archive endpoint (same hourly
    keys + utc_offset_seconds), so the downstream processor can consume either.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": race_date_iso,
        "end_date": race_date_iso,
        "hourly": ",".join(_HOURLY_VARS),
        "timezone": "auto",
    }
    return _get(OPENMETEO_FORECAST, params)


def save_race_weather(
    year: int,
    round_no: int,
    lat: float,
    lon: float,
    race_date_iso: str,
    *,
    refresh: bool = False,
) -> tuple[Path, bool]:
    OPENMETEO_DIR.mkdir(parents=True, exist_ok=True)
    target = OPENMETEO_DIR / f"{year}_{round_no:02d}.json"
    if target.exists() and not refresh:
        return target, False
    data = fetch_race_weather(lat, lon, race_date_iso)
    target.write_text(json.dumps(data, indent=2))
    return target, True


def save_race_forecast(
    year: int,
    round_no: int,
    lat: float,
    lon: float,
    race_date_iso: str,
) -> tuple[Path, bool]:
    """Always (over)writes — forecasts move, and we want the latest pull."""
    OPENMETEO_DIR.mkdir(parents=True, exist_ok=True)
    target = OPENMETEO_DIR / f"{year}_{round_no:02d}.forecast.json"
    data = fetch_race_forecast(lat, lon, race_date_iso)
    target.write_text(json.dumps(data, indent=2))
    return target, True


def pull_race(year: int, round_no: int, *, refresh: bool = False) -> Path:
    inv = load_inventory()
    row = inv.query("year == @year and round == @round_no")
    if row.empty:
        raise ValueError(f"{year} R{round_no:02d} not in race_inventory")
    r = row.iloc[0]
    race_date_iso = str(r["race_date"])
    out, fresh = save_race_weather(
        year, round_no, float(r["lat"]), float(r["lon"]), race_date_iso, refresh=refresh
    )
    marker = "fresh" if fresh else "cached"
    print(f"[open-meteo] {year} R{round_no:02d} {r['gp_name']:<30s} -> {out.name} ({marker})")
    return out


def pull_next_race(year: int, round_no: int) -> Path:
    """Forecast pull for a single future race. Always re-fetches (forecasts drift)."""
    inv = load_inventory()
    row = inv.query("year == @year and round == @round_no")
    if row.empty:
        raise ValueError(f"{year} R{round_no:02d} not in race_inventory")
    r = row.iloc[0]
    race_date_iso = str(r["race_date"])
    out, _ = save_race_forecast(year, round_no, float(r["lat"]), float(r["lon"]), race_date_iso)
    print(f"[open-meteo] {year} R{round_no:02d} {r['gp_name']:<30s} -> {out.name} (forecast)")
    return out


def pull_season(year: int, *, refresh: bool = False) -> None:
    inv = load_inventory()
    season = inv.query("year == @year").sort_values("round")
    if season.empty:
        raise ValueError(f"No races for {year} in race_inventory.")
    today_iso = date.today().isoformat()
    for _, r in season.iterrows():
        race_date_iso = str(r["race_date"])
        if race_date_iso > today_iso:
            print(f"[open-meteo] {year} R{int(r['round']):02d} {r['gp_name']:<30s} - future, skip")
            continue
        out, fresh = save_race_weather(
            year,
            int(r["round"]),
            float(r["lat"]),
            float(r["lon"]),
            race_date_iso,
            refresh=refresh,
        )
        marker = "fresh" if fresh else "cached"
        print(
            f"[open-meteo] {year} R{int(r['round']):02d} {r['gp_name']:<30s} "
            f"-> {out.name} ({marker})"
        )


def pull_all(*, refresh: bool = False) -> None:
    inv = load_inventory()
    today_iso = date.today().isoformat()
    for _, r in inv.iterrows():
        race_date_iso = str(r["race_date"])
        if race_date_iso > today_iso:
            continue
        out, fresh = save_race_weather(
            int(r["year"]),
            int(r["round"]),
            float(r["lat"]),
            float(r["lon"]),
            race_date_iso,
            refresh=refresh,
        )
        marker = "fresh" if fresh else "cached"
        print(
            f"[open-meteo] {int(r['year'])} R{int(r['round']):02d} {r['gp_name']:<30s} "
            f"-> {out.name} ({marker})"
        )


def smoke_test(
    lat: float = 43.7347,
    lon: float = 7.4206,
    date_iso: str = "2024-05-26",
) -> int:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_iso,
        "end_date": date_iso,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "auto",
    }
    print(f"[open-meteo] GET {OPENMETEO_ARCHIVE} for ({lat}, {lon}) on {date_iso}")
    r = requests.get(OPENMETEO_ARCHIVE, params=params, timeout=15)
    r.raise_for_status()

    data = r.json()
    hours = data["hourly"]["time"]
    temps = data["hourly"]["temperature_2m"]
    rain = data["hourly"]["precipitation"]
    print(f"[open-meteo] OK - {len(hours)} hourly samples")
    print(f"[open-meteo] temp range: {min(temps):.1f}C - {max(temps):.1f}C")
    print(f"[open-meteo] total precip: {sum(rain):.1f} mm")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("smoke", help="Phase 0 smoke")

    p_race = sub.add_parser("race", help="Pull weather for one race")
    p_race.add_argument("--year", type=int, required=True)
    p_race.add_argument("--round", type=int, required=True)
    p_race.add_argument("--refresh", action="store_true")

    p_next = sub.add_parser("next", help="Pull forecast for one future race")
    p_next.add_argument("--year", type=int, required=True)
    p_next.add_argument("--round", type=int, required=True)

    p_season = sub.add_parser("season", help="Pull weather for all completed races in a season")
    p_season.add_argument("--year", type=int, required=True)
    p_season.add_argument("--refresh", action="store_true")

    p_all = sub.add_parser("all", help="Pull weather for every completed race in the inventory")
    p_all.add_argument("--refresh", action="store_true")

    args = p.parse_args(argv)

    if args.cmd in (None, "smoke"):
        return smoke_test()
    if args.cmd == "race":
        pull_race(args.year, args.round, refresh=args.refresh)
        return 0
    if args.cmd == "next":
        pull_next_race(args.year, args.round)
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
