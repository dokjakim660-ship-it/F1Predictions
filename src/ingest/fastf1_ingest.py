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
    next    --year Y --round N       - pull only Q + FP2 for a future race (no R)
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
from fastf1.exceptions import RateLimitExceededError

from src.utils.paths import FASTF1_CACHE, RAW_DIR
from src.utils.race_inventory import load_inventory

FASTF1_DIR = RAW_DIR / "fastf1"
# "S" = Sprint race (2021+). FastF1 raises on non-sprint weekends -> we write an
# error marker (same pattern as FP2 on sprint weekends), consumed downstream as
# has_sprint=0. Sprint Qualifying ("SQ", 2023+ grid-setter) is NOT included --
# the race-craft + pace signal lives in the Sprint result itself, not its grid.
SESSIONS: tuple[str, ...] = ("Q", "R", "FP2", "S")
NEXT_SESSIONS: tuple[str, ...] = ("Q", "FP2", "S")


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


def _safe_df(s, attr: str):
    """Return getattr(s, attr) or None if FastF1 raises DataNotLoadedError.

    Some sessions load partially (e.g. driver list ok, lap data missing). FastF1
    only raises on property access, so we have to guard each one individually
    instead of trusting Session.load() to either succeed or raise.
    """
    try:
        return getattr(s, attr)
    except Exception:
        return None


def _save_session(year: int, round_no: int, code: str, *, refresh: bool = False) -> bool:
    out_dir = _session_dir(year, round_no, code)
    if not refresh and _is_done(out_dir):
        return True

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        s = fastf1.get_session(year, round_no, code)
        s.load(laps=True, telemetry=False, weather=True, messages=False)

        event_name = ""
        event_date = ""
        try:
            if getattr(s, "event", None) is not None:
                event_name = str(s.event.get("EventName", "") or "")
                event_date = str(s.event.get("EventDate", "") or "")
        except Exception:
            pass

        n_results = _write_parquet_safe(_safe_df(s, "results"), out_dir / "results.parquet")
        n_laps = _write_parquet_safe(_safe_df(s, "laps"), out_dir / "laps.parquet")
        n_weather = _write_parquet_safe(_safe_df(s, "weather_data"), out_dir / "weather.parquet")

        info = {
            "year": year,
            "round": round_no,
            "code": code,
            "session_name": str(getattr(s, "name", "") or ""),
            "event_name": event_name,
            "event_date": event_date,
            "session_date": str(s.date) if getattr(s, "date", None) is not None else "",
            "total_laps": int(getattr(s, "total_laps", 0) or 0),
            "n_results": n_results,
            "n_laps": n_laps,
            "n_weather": n_weather,
        }
        (out_dir / "session_info.json").write_text(json.dumps(info, indent=2))
    except RateLimitExceededError:
        # Temporary - DO NOT write a marker, so the next run retries this session.
        # Re-raise so the whole bulk pull stops cleanly instead of burning through
        # every remaining race with the same error.
        raise
    except Exception as e:
        err = {"year": year, "round": round_no, "code": code, "error": str(e)[:300]}
        (out_dir / "session_info.json").write_text(json.dumps(err, indent=2))
        print(f"[fastf1] {year} R{round_no:02d} {code} - UNAVAILABLE ({str(e)[:80]})")
        return False

    tag = "OK" if n_laps > 0 or n_results > 0 else "EMPTY"
    print(
        f"[fastf1] {year} R{round_no:02d} {code} - {tag} "
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


def pull_next_race_weekend(year: int, round_no: int) -> int:
    """Lightweight ingest for a future race: only Q + FP2, no R.

    Always refreshes (no skip-on-marker): the typical call pattern is "run before
    qualifying" (empty Q marker) then "run again after qualifying" (real Q data),
    and a cached empty/error marker would block the second pull. Skipping R
    avoids writing an "error" marker for a race that hasn't happened. Sprint
    weekends keep producing an FP2 error marker — consumed downstream as
    `has_fp2=0`.
    """
    _ensure_cache()
    ok = 0
    for code in NEXT_SESSIONS:
        if _save_session(year, round_no, code, refresh=True):
            ok += 1
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


def prune_rate_limit_markers() -> int:
    """Drop session_info.json markers that were written for a transient rate-limit hit.

    Keeps permanent "session does not exist" markers (sprint weekends without FP2 etc.).
    Run this after a bulk pull aborted on the FastF1 500-calls/h limit, then rerun
    `ingest-all` to retry those sessions.
    """
    if not FASTF1_DIR.exists():
        print("[prune] nothing to do - data/raw/fastf1 missing")
        return 0
    pruned = 0
    kept = 0
    for session_info in FASTF1_DIR.glob("*/session_info.json"):
        try:
            info = json.loads(session_info.read_text())
        except json.JSONDecodeError:
            continue
        err = info.get("error", "")
        if not err:
            continue
        if "calls/h" in err or "Too Many Requests" in err or "429" in err:
            session_dir = session_info.parent
            for f in session_dir.iterdir():
                f.unlink()
            session_dir.rmdir()
            pruned += 1
        else:
            kept += 1
    print(f"[prune] removed {pruned} rate-limit markers, kept {kept} permanent markers")
    return 0


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

    p_next = sub.add_parser("next", help="Pull Q + FP2 for a future race (no R)")
    p_next.add_argument("--year", type=int, required=True)
    p_next.add_argument("--round", type=int, required=True)

    p_season = sub.add_parser("season", help="Pull all completed races in a season")
    p_season.add_argument("--year", type=int, required=True)
    p_season.add_argument("--refresh", action="store_true")

    p_all = sub.add_parser("all", help="Pull every completed race in the inventory")
    p_all.add_argument("--refresh", action="store_true")

    sub.add_parser(
        "prune-rate-limit",
        help="Remove session markers caused by transient rate-limit hits (so they get retried)",
    )

    args = p.parse_args(argv)

    try:
        if args.cmd in (None, "smoke"):
            return smoke_test()
        if args.cmd == "race":
            pull_race_weekend(args.year, args.round, refresh=args.refresh)
            return 0
        if args.cmd == "next":
            pull_next_race_weekend(args.year, args.round)
            return 0
        if args.cmd == "season":
            pull_season(args.year, refresh=args.refresh)
            return 0
        if args.cmd == "all":
            pull_all(refresh=args.refresh)
            return 0
        if args.cmd == "prune-rate-limit":
            return prune_rate_limit_markers()
    except RateLimitExceededError as e:
        print(
            f"\n[fastf1] STOP - FastF1 rate limit hit ({e}).\n"
            "        Wait ~1 hour, then rerun the same command.\n"
            "        Already-pulled sessions skip via session_info.json markers."
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
