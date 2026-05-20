"""Phase 0 smoke test: hit Jolpica-F1 (Ergast fork) and pull one season's schedule."""

from __future__ import annotations

import sys

import requests

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"


def smoke_test(year: int = 2024) -> int:
    url = f"{JOLPICA_BASE}/{year}.json"
    print(f"[jolpica] GET {url}")
    r = requests.get(url, timeout=15)
    r.raise_for_status()

    races = r.json()["MRData"]["RaceTable"]["Races"]
    print(f"[jolpica] OK — {len(races)} races in {year}")
    for race in races[:3]:
        print(f"  - R{race['round']:>2} {race['date']}  {race['raceName']}")
    if len(races) > 3:
        print(f"  ... and {len(races) - 3} more")
    return 0


if __name__ == "__main__":
    sys.exit(smoke_test())
