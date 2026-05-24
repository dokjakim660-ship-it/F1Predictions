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

    df = per_driver_all.merge(n_compounds, on="Driver", how="left").merge(
        per_driver_clean, on="Driver", how="left"
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


def _driver_dim_from_quali_or_race(quali_dir: Path | None, race_dir: Path | None) -> pd.DataFrame:
    """Driver-abbr -> driver_id/team mapping from the most-trustworthy results file."""
    for d in (race_dir, quali_dir):
        if d is None:
            continue
        results = _safe_read(d / "results.parquet")
        if results is not None and not results.empty:
            return pd.DataFrame(
                {
                    "driver_abbr": results["Abbreviation"].astype(str),
                    "driver_id": results["DriverId"].astype(str),
                    "team_id": results["TeamId"].astype(str),
                    "team_name": results["TeamName"].astype(str),
                }
            ).drop_duplicates(subset=["driver_abbr"])
    return pd.DataFrame(columns=["driver_abbr", "driver_id", "team_id", "team_name"])


def build_sessions(fastf1_dir: Path = FASTF1_DIR) -> pd.DataFrame:
    sessions = _scan_session_dirs()
    if not sessions:
        raise FileNotFoundError(
            f"No FastF1 session dirs in {fastf1_dir}. Run: python -m src.ingest.fastf1_ingest all"
        )

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
        dim_df = _driver_dim_from_quali_or_race(q_dir, r_dir)

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

        # Fill driver_id/team_id/team_name from dim if missing (e.g. R-only races).
        if not dim_df.empty:
            merged = merged.merge(dim_df, on="driver_abbr", how="left", suffixes=("", "_dim"))
            for col in ("driver_id", "team_id", "team_name"):
                if f"{col}_dim" in merged.columns:
                    merged[col] = (
                        merged[col].fillna(merged[f"{col}_dim"])
                        if col in merged.columns
                        else merged[f"{col}_dim"]
                    )
                    merged = merged.drop(columns=[f"{col}_dim"])

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
