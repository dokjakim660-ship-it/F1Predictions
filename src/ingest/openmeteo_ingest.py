"""Phase 0 smoke test: pull historical weather for one race day from Open-Meteo."""

from __future__ import annotations

import sys

import requests

OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def smoke_test(
    lat: float = 43.7347,  # Monaco
    lon: float = 7.4206,
    date: str = "2024-05-26",  # Monaco GP 2024 race day
) -> int:
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "auto",
    }
    print(f"[open-meteo] GET {OPENMETEO_ARCHIVE} for ({lat}, {lon}) on {date}")
    r = requests.get(OPENMETEO_ARCHIVE, params=params, timeout=15)
    r.raise_for_status()

    data = r.json()
    hours = data["hourly"]["time"]
    temps = data["hourly"]["temperature_2m"]
    rain = data["hourly"]["precipitation"]
    print(f"[open-meteo] OK — {len(hours)} hourly samples")
    print(f"[open-meteo] temp range: {min(temps):.1f}°C – {max(temps):.1f}°C")
    print(f"[open-meteo] total precip: {sum(rain):.1f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(smoke_test())
