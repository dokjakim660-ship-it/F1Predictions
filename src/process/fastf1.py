"""L2 processor: FastF1 per-session parquets -> data/processed/sessions.parquet.

One row per (race_id, driver_id). Columns are NULL where the corresponding
session (Q / R / FP2) was not cached for that race -- the FastF1 ingest has
known partial coverage in 2024+ until the backfill catches up, so this
processor must stay tolerant of holes.

Lap-time milliseconds are derived from pandas Timedelta columns. Clean-lap
definition (race): green flag (TrackStatus == "1"), accurate, not deleted,
not in/out of pit, not the start lap. Long-run definition (FP2): stints with
>= 5 consecutive clean laps on the same compound and TyreLife >= 4.

Run: `python -m src.process.fastf1 build`
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from src.utils.paths import PROCESSED_DIR, RAW_DIR

FASTF1_DIR = RAW_DIR / "fastf1"
SESSIONS_PARQUET = PROCESSED_DIR / "sessions.parquet"

_SESSION_DIR_RE = re.compile(r"^(\d{4})_(\d{2})_(FP2|Q|R|S)$")
_MIN_LONG_RUN_STINT_LAPS = 5
_MIN_TYRE_LIFE_FOR_LONG_RUN = 4


def _td_ms(s: pd.Series) -> pd.Series:
    """Timedelta -> float ms with NaN where missing."""
    return s.dt.total_seconds() * 1000.0


def _scan_session_dirs() -> dict[tuple[int, int], dict[str, Path]]:
    """Group cached FastF1 session dirs by (year, round). Returns {(y,r): {code: path}}."""
    out: dict[tuple[int, int], dict[str, Path]] = {}
    if not FASTF1_DIR.exists():
        return out
    for d in sorted(FASTF1_DIR.iterdir()):
        if not d.is_dir():
            continue
        m = _SESSION_DIR_RE.match(d.name)
        if not m:
            continue
        year, round_no, code = int(m.group(1)), int(m.group(2)), m.group(3)
        out.setdefault((year, round_no), {})[code] = d
    return out


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001 — defensive: corrupt cache file shouldn't crash run
        print(f"[process.fastf1] WARN: failed to read {path}: {e}")
        return None


def _quali_rows(session_dir: Path, race_id: str, year: int, round_no: int) -> pd.DataFrame:
    results = _safe_read(session_dir / "results.parquet")
    if results is None or results.empty:
        return pd.DataFrame()

    df = results.copy()
    for col in ("Q1", "Q2", "Q3"):
        df[f"{col.lower()}_ms"] = _td_ms(df[col])
    df["q_best_ms"] = df[["q1_ms", "q2_ms", "q3_ms"]].min(axis=1, skipna=True)
    pole_ms = df["q_best_ms"].min(skipna=True)
    df["q_gap_to_pole_ms"] = df["q_best_ms"] - pole_ms

    keep = pd.DataFrame(
        {
            "race_id": race_id,
            "year": year,
            "round": round_no,
            "driver_abbr": df["Abbreviation"].astype(str),
            "driver_id": df["DriverId"].astype(str),
            "team_id": df["TeamId"].astype(str),
            "team_name": df["TeamName"].astype(str),
            "q_position": df["Position"],
            "q1_ms": df["q1_ms"],
            "q2_ms": df["q2_ms"],
            "q3_ms": df["q3_ms"],
            "q_best_ms": df["q_best_ms"],
            "q_gap_to_pole_ms": df["q_gap_to_pole_ms"],
        }
    )
    return keep


def _race_pace_rows(session_dir: Path, race_id: str, year: int, round_no: int) -> pd.DataFrame:
    laps = _safe_read(session_dir / "laps.parquet")
    if laps is None or laps.empty:
        return pd.DataFrame()

    laps = laps.copy()
    laps["lap_ms"] = _td_ms(laps["LapTime"])

    is_clean = (
        laps["lap_ms"].notna()
        & (laps["TrackStatus"].astype(str) == "1")
        & (laps["IsAccurate"].fillna(False).astype(bool))
        & (laps["Deleted"] != True)  # noqa: E712 — object dtype: None/False both pass
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & (laps["LapNumber"] > 1)
    )
    clean = laps.loc[is_clean]

    # Aggregate per driver across all laps (counts) and across clean laps (pace).
    per_driver_all = (
        laps.groupby("Driver", sort=False)
        .agg(
            race_lap_count=("LapNumber", "max"),
            race_n_pits=("PitInTime", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    n_compounds = (
        laps.groupby("Driver", sort=False)["Compound"]
        .apply(lambda s: s.dropna().loc[lambda x: x != "nan"].nunique())
        .reset_index(name="race_n_compounds")
    )
    per_driver_clean = (
        clean.groupby("Driver", sort=False)
        .agg(
            race_clean_lap_count=("lap_ms", "size"),
            race_clean_median_lap_ms=("lap_ms", "median"),
        )
        .reset_index()
    )

    # Pit-lane time per stop = the out-lap's PitOutTime minus the in-lap's
    # PitInTime (entry-to-exit). Median over a driver's stops; raw, so it still
    # carries the track's pit-lane transit -- the constructor feature debiases
    # that by track downstream. Filter to positive, sane (<60s) durations.
    lap_sorted = laps.sort_values(["Driver", "LapNumber"])
    prev_in = lap_sorted.groupby("Driver", sort=False)["PitInTime"].shift(1)
    lap_sorted = lap_sorted.assign(_pit_ms=_td_ms(lap_sorted["PitOutTime"] - prev_in))
    pit = (
        lap_sorted[(lap_sorted["_pit_ms"] > 0) & (lap_sorted["_pit_ms"] < 60_000)]
        .groupby("Driver", sort=False)["_pit_ms"]
        .median()
        .reset_index(name="race_pit_lane_median_ms")
    )

    # Lap-1 classified position -> start performance (grid - lap1) is built later.
    lap1 = (
        laps.loc[laps["LapNumber"] == 1, ["Driver", "Position"]]
        .dropna()
        .drop_duplicates("Driver")
        .rename(columns={"Position": "race_lap1_position"})
    )

    df = (
        per_driver_all.merge(n_compounds, on="Driver", how="left")
        .merge(per_driver_clean, on="Driver", how="left")
        .merge(pit, on="Driver", how="left")
        .merge(lap1, on="Driver", how="left")
    )

    leader_ms = df["race_clean_median_lap_ms"].min(skipna=True)
    df["race_pace_gap_to_leader_ms"] = df["race_clean_median_lap_ms"] - leader_ms

    df.insert(0, "race_id", race_id)
    df.insert(1, "year", year)
    df.insert(2, "round", round_no)
    df = df.rename(columns={"Driver": "driver_abbr"})
    return df


def _fp2_rows(session_dir: Path, race_id: str, year: int, round_no: int) -> pd.DataFrame:
    laps = _safe_read(session_dir / "laps.parquet")
    if laps is None or laps.empty:
        return pd.DataFrame()

    laps = laps.copy()
    laps["lap_ms"] = _td_ms(laps["LapTime"])

    is_clean = (
        laps["lap_ms"].notna()
        & (laps["TrackStatus"].astype(str) == "1")
        & (laps["IsAccurate"].fillna(False).astype(bool))
        & (laps["Deleted"] != True)  # noqa: E712 — object dtype: None/False both pass
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
    )
    clean = laps.loc[is_clean].copy()

    # Short run: best individual clean lap = quali-sim proxy.
    short_run = (
        clean.groupby("Driver", sort=False)["lap_ms"]
        .min()
        .reset_index(name="fp2_short_run_best_ms")
    )

    # Long run: longest stint (>=5 laps, TyreLife >= 4) per driver, median pace.
    long_run_candidates = clean[clean["TyreLife"] >= _MIN_TYRE_LIFE_FOR_LONG_RUN]
    long_run_groups = long_run_candidates.groupby(["Driver", "Stint", "Compound"], sort=False).agg(
        n_laps=("lap_ms", "size"),
        median_ms=("lap_ms", "median"),
    )
    long_run_groups = long_run_groups[long_run_groups["n_laps"] >= _MIN_LONG_RUN_STINT_LAPS]
    long_run_groups = long_run_groups.reset_index()

    if long_run_groups.empty:
        long_run = pd.DataFrame(
            columns=[
                "Driver",
                "fp2_long_run_lap_count",
                "fp2_long_run_median_ms",
                "fp2_long_run_compound",
            ]
        )
    else:
        # Pick the longest stint per driver (break ties by faster median).
        long_run_groups = long_run_groups.sort_values(
            ["Driver", "n_laps", "median_ms"], ascending=[True, False, True]
        )
        long_run = (
            long_run_groups.groupby("Driver", sort=False)
            .first()
            .reset_index()
            .rename(
                columns={
                    "n_laps": "fp2_long_run_lap_count",
                    "median_ms": "fp2_long_run_median_ms",
                    "Compound": "fp2_long_run_compound",
                }
            )[
                [
                    "Driver",
                    "fp2_long_run_lap_count",
                    "fp2_long_run_median_ms",
                    "fp2_long_run_compound",
                ]
            ]
        )

    df = short_run.merge(long_run, on="Driver", how="outer")
    df.insert(0, "race_id", race_id)
    df.insert(1, "year", year)
    df.insert(2, "round", round_no)
    df = df.rename(columns={"Driver": "driver_abbr"})
    return df


def _sprint_rows(session_dir: Path, race_id: str, year: int, round_no: int) -> pd.DataFrame:
    """Sprint finishing position + gap to winner per driver.

    FastF1 stores Time as Timedelta: full race duration for P1, gap to leader
    for P2..Pn, NaN for DNF/DNS. We expose gap_to_winner_ms = 0 for the winner,
    otherwise the raw gap. DNF rows carry NaN (median-imputed downstream).
    """
    results = _safe_read(session_dir / "results.parquet")
    if results is None or results.empty:
        return pd.DataFrame()

    df = results.copy()
    df["_time_ms"] = _td_ms(df["Time"])
    leader_pos = df["Position"].min(skipna=True)
    gap = df["_time_ms"].where(df["Position"] != leader_pos, 0.0)

    return pd.DataFrame(
        {
            "race_id": race_id,
            "year": year,
            "round": round_no,
            "driver_abbr": df["Abbreviation"].astype(str),
            "sprint_position": df["Position"],
            "sprint_gap_to_winner_ms": gap,
        }
    )


# Driver-id/team strings that FastF1 leaves blank when a session predates the
# official entry-list mapping (notably 2026 FP2-only results files): treat them
# as missing so the historical fallback can fill them.
_BLANK_IDS = frozenset({"", "nan", "none", "<na>", "na"})


def _blank_to_na(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace blank/`nan`/`<NA>` id strings with real NaN so fillna can work."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        col = df[c].astype(object)
        mask = col.isna() | col.astype(str).str.strip().str.lower().isin(_BLANK_IDS)
        df[c] = col.where(~mask)
    return df


def _driver_dim_from_quali_or_race(quali_dir: Path | None, race_dir: Path | None) -> pd.DataFrame:
    """Driver-abbr -> driver_id/team mapping from the most-trustworthy results file."""
    for d in (race_dir, quali_dir):
        if d is None:
            continue
        results = _safe_read(d / "results.parquet")
        if results is not None and not results.empty:
            dim = pd.DataFrame(
                {
                    "driver_abbr": results["Abbreviation"].astype(str),
                    "driver_id": results["DriverId"].astype(str),
                    "team_id": results["TeamId"].astype(str),
                    "team_name": results["TeamName"].astype(str),
                }
            ).drop_duplicates(subset=["driver_abbr"])
            return _blank_to_na(dim, ["driver_id", "team_id", "team_name"]).dropna(
                subset=["driver_id"]
            )
    return pd.DataFrame(columns=["driver_abbr", "driver_id", "team_id", "team_name"])


def _global_driver_dim(
    sessions: dict[tuple[int, int], dict[str, Path]],
) -> pd.DataFrame:
    """driver_abbr -> driver_id/team from every weekend that has Q or R results.

    An FP2-only weekend (a pre-quali Friday) has no Q/R file, and FastF1's 2026
    FP2 results.parquet carries an empty DriverId, so its FP2 rows would get a
    null driver_id and silently drop out of the `["race_id", "driver_id"]`
    feature join -- losing all FP2 long-run pace. This collects the abbr->id
    mapping from every completed weekend (latest occurrence wins, so a reused
    three-letter code maps to the current driver) to fill those gaps.
    """
    frames: list[pd.DataFrame] = []
    for _key, code_to_dir in sorted(sessions.items()):
        dim = _driver_dim_from_quali_or_race(code_to_dir.get("Q"), code_to_dir.get("R"))
        if not dim.empty:
            frames.append(dim)
    if not frames:
        return pd.DataFrame(columns=["driver_abbr", "driver_id", "team_id", "team_name"])
    alld = pd.concat(frames, ignore_index=True)
    return alld.drop_duplicates(subset=["driver_abbr"], keep="last").reset_index(drop=True)


def build_sessions(fastf1_dir: Path = FASTF1_DIR) -> pd.DataFrame:
    sessions = _scan_session_dirs()
    if not sessions:
        raise FileNotFoundError(
            f"No FastF1 session dirs in {fastf1_dir}. Run: python -m src.ingest.fastf1_ingest all"
        )

    # abbr->driver_id fallback for FP2-only weekends (built from all completed
    # weekends), so a pre-quali Friday's FP2 pace still joins on driver_id.
    global_dim = _global_driver_dim(sessions)

    all_rows: list[pd.DataFrame] = []
    for (year, round_no), code_to_dir in sessions.items():
        race_id = f"{year}_{round_no:02d}"
        q_dir = code_to_dir.get("Q")
        r_dir = code_to_dir.get("R")
        fp2_dir = code_to_dir.get("FP2")
        s_dir = code_to_dir.get("S")

        q_df = _quali_rows(q_dir, race_id, year, round_no) if q_dir else pd.DataFrame()
        r_df = _race_pace_rows(r_dir, race_id, year, round_no) if r_dir else pd.DataFrame()
        fp2_df = _fp2_rows(fp2_dir, race_id, year, round_no) if fp2_dir else pd.DataFrame()
        s_df = _sprint_rows(s_dir, race_id, year, round_no) if s_dir else pd.DataFrame()

        keys = ["race_id", "year", "round", "driver_abbr"]
        frames = [
            df[keys + [c for c in df.columns if c not in keys]]
            for df in (q_df, r_df, fp2_df, s_df)
            if not df.empty
        ]
        if not frames:
            continue
        merged = frames[0]
        for nxt in frames[1:]:
            # Avoid column collision on driver_id/team_id/team_name (only quali has them).
            overlap = set(merged.columns) & set(nxt.columns) - set(keys)
            merged = merged.merge(nxt.drop(columns=list(overlap)), on=keys, how="outer")

        # Map driver_abbr -> driver_id/team. Prefer this weekend's own Q/R results;
        # for an FP2-only (pre-quali) weekend there are none, so fall back to the
        # global history -- otherwise driver_id stays null and the FP2 pace is lost.
        merged = _blank_to_na(merged, ["driver_id", "team_id", "team_name"])
        dim_df = _driver_dim_from_quali_or_race(q_dir, r_dir)
        if dim_df.empty and not global_dim.empty:
            dim_df = global_dim[global_dim["driver_abbr"].isin(merged["driver_abbr"].unique())]
        if not dim_df.empty:
            merged = merged.merge(dim_df, on="driver_abbr", how="left", suffixes=("", "_dim"))
            for col in ("driver_id", "team_id", "team_name"):
                dimcol = f"{col}_dim"
                if dimcol not in merged.columns:
                    continue
                merged[col] = (
                    merged[col].fillna(merged[dimcol]) if col in merged.columns else merged[dimcol]
                )
                merged = merged.drop(columns=[dimcol])

        merged["has_qualifying"] = not q_df.empty
        merged["has_race"] = not r_df.empty
        merged["has_fp2"] = not fp2_df.empty
        merged["has_sprint"] = not s_df.empty
        all_rows.append(merged)

    df = pd.concat(all_rows, ignore_index=True)
    df = df.sort_values(["year", "round", "driver_abbr"]).reset_index(drop=True)
    return df


def save_sessions(df: pd.DataFrame, path: Path = SESSIONS_PARQUET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_sessions(path: Path = SESSIONS_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m src.process.fastf1 build")
    return pd.read_parquet(path)


def _print_summary(df: pd.DataFrame) -> None:
    n_races = df[["year", "round"]].drop_duplicates().shape[0]
    n_with_quali = df[df["has_qualifying"]][["year", "round"]].drop_duplicates().shape[0]
    n_with_race = df[df["has_race"]][["year", "round"]].drop_duplicates().shape[0]
    n_with_fp2 = df[df["has_fp2"]][["year", "round"]].drop_duplicates().shape[0]
    n_with_sprint = df[df.get("has_sprint", False)][["year", "round"]].drop_duplicates().shape[0]
    print(f"[process.fastf1] {len(df)} driver-race rows across {n_races} races")
    print(
        f"[process.fastf1] coverage of {n_races}: "
        f"Q={n_with_quali}  R={n_with_race}  FP2={n_with_fp2}  S={n_with_sprint}"
    )
    by_year = df[["year", "round"]].drop_duplicates().groupby("year").size()
    for year, n in by_year.items():
        print(f"  {year}: {n} races")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="RAW FastF1 session parquets -> data/processed/sessions.parquet")
    sub.add_parser("show", help="Print summary of the current sessions parquet")

    args = p.parse_args(argv)

    if args.cmd == "build":
        df = build_sessions()
        out = save_sessions(df)
        print(f"[process.fastf1] saved -> {out}")
        _print_summary(df)
        return 0
    if args.cmd == "show":
        df = load_sessions()
        _print_summary(df)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
