"""L2 processor: Open-Meteo race-day weather JSONs -> data/processed/weather.parquet.

One row per race_id. Two flavours of feature on the same row:

- Race-hour snapshot (`weather_*_race_hour`): conditions at the actual race
  start hour, in local time at the circuit. This is the "what the drivers
  experience" view.
- Race-day aggregates (`weather_*_day_*`): max/total over the full 24h, used
  as a robust fallback when race-time alignment goes wonky (Vegas, etc.).

The race start in UTC comes from race_inventory.race_time_utc; we shift it
into circuit-local time using utc_offset_seconds from the JSON, then look up
the matching hour in hourly.time (also local).

Run: `python -m src.process.openmeteo build`
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.ingest.openmeteo_ingest import OPENMETEO_DIR
from src.utils.paths import PROCESSED_DIR
from src.utils.race_inventory import load_inventory

WEATHER_PARQUET = PROCESSED_DIR / "weather.parquet"

_DEFAULT_UTC_FALLBACK = "14:00:00Z"  # Used only if inventory has empty race_time_utc.
_WET_PRECIP_MM_THRESHOLD = 0.5  # Hourly precip above this = "wet at race time".


def _parse_utc_hhmmss(s: str) -> tuple[int, int]:
    """'15:00:00Z' -> (15, 0). Returns hour, minute."""
    s = s.rstrip("Z").strip()
    parts = s.split(":")
    return int(parts[0]), int(parts[1])


def _race_local_dt(race_date_iso: str, race_time_utc: str, utc_offset_seconds: int) -> datetime:
    hh, mm = _parse_utc_hhmmss(race_time_utc or _DEFAULT_UTC_FALLBACK)
    race_utc = datetime.fromisoformat(f"{race_date_iso}T{hh:02d}:{mm:02d}:00")
    race_local = race_utc + timedelta(seconds=utc_offset_seconds)
    # Jolpica stores `date` as the *local* race date but `time` as the *UTC*
    # hour of the race start. For late Saturday-night races that cross midnight
    # UTC (Vegas, Qatar before 2024), naive combination puts race_local on the
    # day before race_date. Shift forward by 24h so the lookup hits the right
    # row of OpenMeteo's hourly grid.
    race_date = datetime.fromisoformat(race_date_iso).date()
    if race_local.date() < race_date:
        race_local += timedelta(days=1)
    return race_local


def _find_hour_idx(times_local_iso: list[str], race_local_dt: datetime) -> int | None:
    target = race_local_dt.replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:%M")
    for i, t in enumerate(times_local_iso):
        if t == target:
            return i
    return None


def _row_from_payload(
    year: int, round_no: int, payload: dict, race_date_iso: str, race_time_utc: str
) -> dict:
    hourly = payload["hourly"]
    times = hourly["time"]
    offset_s = int(payload.get("utc_offset_seconds", 0))
    race_local = _race_local_dt(race_date_iso, race_time_utc, offset_s)
    idx = _find_hour_idx(times, race_local)

    def at(var: str) -> float | None:
        if idx is None:
            return None
        v = hourly.get(var)
        if v is None or idx >= len(v):
            return None
        return None if v[idx] is None else float(v[idx])

    def series(var: str) -> list[float]:
        return [x for x in (hourly.get(var) or []) if x is not None]

    temps = series("temperature_2m")
    precip = series("precipitation")
    rain = series("rain")
    wind = series("wind_speed_10m")

    return {
        "race_id": f"{year}_{round_no:02d}",
        "year": year,
        "round": round_no,
        "circuit_lat": payload.get("latitude"),
        "circuit_lon": payload.get("longitude"),
        "circuit_elevation_m": payload.get("elevation"),
        "race_local_dt": race_local,
        "race_hour_aligned": idx is not None,
        # --- Race-hour snapshot ---
        "weather_temp_c_race_hour": at("temperature_2m"),
        "weather_precip_mm_race_hour": at("precipitation"),
        "weather_rain_mm_race_hour": at("rain"),
        "weather_wind_kph_race_hour": at("wind_speed_10m"),
        "weather_wind_dir_deg_race_hour": at("wind_direction_10m"),
        "weather_cloud_pct_race_hour": at("cloud_cover"),
        "weather_humidity_pct_race_hour": at("relative_humidity_2m"),
        "weather_is_wet_race_hour": (
            None
            if at("precipitation") is None
            else bool(at("precipitation") >= _WET_PRECIP_MM_THRESHOLD)
        ),
        # --- Race-day aggregates (robust fallback) ---
        "weather_temp_c_day_max": max(temps) if temps else None,
        "weather_temp_c_day_min": min(temps) if temps else None,
        "weather_precip_mm_day_total": sum(precip) if precip else 0.0,
        "weather_rain_mm_day_total": sum(rain) if rain else 0.0,
        "weather_wind_kph_day_max": max(wind) if wind else None,
    }


def build_weather(weather_dir: Path = OPENMETEO_DIR) -> pd.DataFrame:
    files = sorted(weather_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(
            f"No weather JSONs in {weather_dir}. Run: python -m src.ingest.openmeteo_ingest all"
        )

    inv = load_inventory().set_index(["year", "round"])
    rows: list[dict] = []
    for f in files:
        # Archive files are `<year>_<round>.json`; forecast files (Phase 3.1) are
        # `<year>_<round>.forecast.json`. Both coexist for races between forecast
        # pull and archive availability; archive wins below via drop_duplicates.
        stem = f.stem
        is_forecast = stem.endswith(".forecast")
        base = stem[: -len(".forecast")] if is_forecast else stem
        try:
            year_str, round_str = base.split("_")
            year, round_no = int(year_str), int(round_str)
        except ValueError:
            print(f"[process.openmeteo] {f.name}: cannot parse year/round, skipping")
            continue
        key = (year, round_no)
        if key not in inv.index:
            print(f"[process.openmeteo] {f.name}: not in race_inventory, skipping")
            continue
        inv_row = inv.loc[key]
        payload = json.loads(f.read_text())
        row = _row_from_payload(
            year,
            round_no,
            payload,
            str(inv_row["race_date"]),
            str(inv_row["race_time_utc"] or ""),
        )
        row["weather_source"] = "forecast" if is_forecast else "archive"
        rows.append(row)

    df = pd.DataFrame(rows)
    # Same race may have both archive and forecast rows after the race is run.
    # Archive sorts before "forecast" alphabetically -> keep="first" wins archive.
    df = df.sort_values(["year", "round", "weather_source"]).drop_duplicates(
        ["year", "round"], keep="first"
    )
    df = df.sort_values(["year", "round"]).reset_index(drop=True)
    return df


def save_weather(df: pd.DataFrame, path: Path = WEATHER_PARQUET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_weather(path: Path = WEATHER_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m src.process.openmeteo build")
    return pd.read_parquet(path)


def _print_summary(df: pd.DataFrame) -> None:
    n_unaligned = int((~df["race_hour_aligned"]).sum())
    wet_count = int(df["weather_is_wet_race_hour"].fillna(False).astype(bool).sum())
    print(f"[process.openmeteo] {len(df)} races, {n_unaligned} race-hour misaligned")
    print(f"[process.openmeteo] wet at race hour (>={_WET_PRECIP_MM_THRESHOLD}mm/h): {wet_count}")
    if n_unaligned:
        bad = df[~df["race_hour_aligned"]][["race_id", "race_local_dt"]]
        print("[process.openmeteo] misaligned race-ids (using day aggregates only):")
        for _, r in bad.iterrows():
            print(f"  - {r['race_id']}  local_dt={r['race_local_dt']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="RAW weather JSONs -> data/processed/weather.parquet")
    sub.add_parser("show", help="Print summary of the current weather parquet")

    args = p.parse_args(argv)

    if args.cmd == "build":
        df = build_weather()
        out = save_weather(df)
        print(f"[process.openmeteo] saved -> {out}")
        _print_summary(df)
        return 0
    if args.cmd == "show":
        df = load_weather()
        _print_summary(df)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
