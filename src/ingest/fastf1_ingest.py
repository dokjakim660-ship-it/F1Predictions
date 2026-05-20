"""Phase 0 smoke test: load one race via FastF1 and prove the cache works."""

from __future__ import annotations

import sys

import fastf1

from src.utils.paths import FASTF1_CACHE


def smoke_test(year: int = 2024, gp: str = "Monaco", session: str = "R") -> int:
    FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE))

    print(f"[fastf1] loading {year} {gp} {session} ...")
    s = fastf1.get_session(year, gp, session)
    s.load(laps=True, telemetry=False, weather=False, messages=False)

    laps = s.laps
    print(f"[fastf1] OK — {len(laps)} laps, {laps['Driver'].nunique()} drivers")
    print(f"[fastf1] fastest lap: {laps['LapTime'].min()}")
    print(f"[fastf1] cache dir:   {FASTF1_CACHE}")
    return 0


if __name__ == "__main__":
    sys.exit(smoke_test())
