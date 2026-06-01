"""L2 processor: FastF1 race laps -> data/processed/overtakes.parquet.

One row per race with a count of genuine on-track overtakes, reconstructed from
the per-lap Position column. This is the raw signal behind the track overtaking
index (a circuit-difficulty feature in src/features/build.py).

Pairwise pass definition: between consecutive green laps N-1 -> N, driver A is
credited with one pass over driver B when A was behind B on lap N-1 and ahead on
lap N. Counting ordered pairs (not summed position deltas) is what keeps the
numbers honest:

  - Pit cycles don't inflate. When B pits and drops behind A..F, every flipped
    pair involves B, who pitted -> all excluded. The cars B fell behind never
    changed order among themselves, so they generate no phantom passes. (The
    summed-delta proxy double-counts exactly these and overstates ~3-5x.)
  - Lapping doesn't inflate. Position is the classified race position, so a
    leader lapping a backmarker leaves both their positions unchanged -> no swap.

Exclusions: lap 1 (start shuffle), any lap where either car was in/out of the
pit (this lap or the previous one), and laps run under a non-green TrackStatus
(SC / VSC / red). A pass is only counted when BOTH cars are clean on lap N.

Coverage: races with no usable Position data (early-season FastF1 holes) are
emitted with n_overtakes = NaN and is_usable = False so downstream code can skip
them rather than treat a hole as "zero overtakes".

Run: `python -m src.process.overtakes build`
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import PROCESSED_DIR, RAW_DIR

FASTF1_DIR = RAW_DIR / "fastf1"
OVERTAKES_PARQUET = PROCESSED_DIR / "overtakes.parquet"

_RACE_DIR_RE = re.compile(r"^(\d{4})_(\d{2})_R$")
# TrackStatus digits that mean the track is not fully green: 4=SC, 5=red,
# 6/7=VSC deployed/ending. A status string may concatenate several (e.g. "14").
_NON_GREEN = set("4567")


def _is_green(track_status: object) -> bool:
    s = "" if track_status is None or (isinstance(track_status, float) and np.isnan(track_status)) else str(track_status)
    return not (_NON_GREEN & set(s))


def count_overtakes(laps: pd.DataFrame) -> dict | None:
    """Count genuine on-track passes in one race. None if Position data unusable."""
    need = {"Driver", "LapNumber", "Position"}
    if not need.issubset(laps.columns):
        return None

    df = laps[["Driver", "LapNumber", "Position", "PitInTime", "PitOutTime", "TrackStatus"]].copy()
    df = df.dropna(subset=["LapNumber", "Position"])
    if df.empty or df["Position"].nunique() < 2:
        return None

    df["LapNumber"] = df["LapNumber"].astype(int)
    df["pit"] = df["PitInTime"].notna() | df["PitOutTime"].notna()
    df["green"] = df["TrackStatus"].map(_is_green)

    # Wide matrices indexed by lap (rows) x driver (cols). NaN where a driver has
    # no lap (retired / not yet started); NaN never satisfies the swap test. We
    # carry NaN into numpy and resolve holes there: a missing lap counts as
    # "pitted" and "not green" (contaminated), but the isnan(pos) guard already
    # drops it, so the choice only hardens the mask.
    pos = df.pivot_table(index="LapNumber", columns="Driver", values="Position", aggfunc="first")
    pit = df.pivot_table(index="LapNumber", columns="Driver", values="pit", aggfunc="max")
    grn = df.pivot_table(index="LapNumber", columns="Driver", values="green", aggfunc="min")
    pos = pos.sort_index()

    laps_idx = pos.index.to_numpy()
    P = pos.to_numpy(dtype=float)
    PIT = np.nan_to_num(pit.reindex_like(pos).to_numpy(dtype=float), nan=1.0) > 0.5
    GRN = np.nan_to_num(grn.reindex_like(pos).to_numpy(dtype=float), nan=0.0) > 0.5

    total = 0
    for i in range(1, len(laps_idx)):
        if laps_idx[i] - laps_idx[i - 1] != 1 or laps_idx[i] <= 1:
            continue  # only consecutive transitions; skip the start lap
        prev, cur = P[i - 1], P[i]
        # A clean on lap i: ran this lap and the previous one, green now, no pit
        # in/out on either lap.
        clean = (
            ~np.isnan(cur) & ~np.isnan(prev)
            & GRN[i] & ~PIT[i] & ~PIT[i - 1]
        )
        if clean.sum() < 2:
            continue
        a_behind_prev = prev[:, None] > prev[None, :]  # A was behind B
        a_ahead_now = cur[:, None] < cur[None, :]  # A is now ahead of B
        pair_ok = clean[:, None] & clean[None, :]
        passes = a_behind_prev & a_ahead_now & pair_ok
        total += int(passes.sum())

    n_drivers = pos.shape[1]
    n_finishers = int((~np.isnan(P[-1])).sum())
    return {
        "n_overtakes": total,
        "total_laps": int(laps_idx.max()),
        "n_drivers": n_drivers,
        "n_dnf": n_drivers - n_finishers,
    }


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception as e:  # noqa: BLE001 — corrupt cache shouldn't crash the run
        print(f"[process.overtakes] WARN: failed to read {path}: {e}")
        return None


def build_overtakes(fastf1_dir: Path = FASTF1_DIR) -> pd.DataFrame:
    if not fastf1_dir.exists():
        raise FileNotFoundError(
            f"No FastF1 dir {fastf1_dir}. Run: python -m src.ingest.fastf1_ingest all"
        )

    rows: list[dict] = []
    for d in sorted(fastf1_dir.glob("*_R")):
        m = _RACE_DIR_RE.match(d.name)
        if not m:
            continue
        year, round_no = int(m.group(1)), int(m.group(2))
        race_id = f"{year}_{round_no:02d}"
        laps = _safe_read(d / "laps.parquet")
        result = count_overtakes(laps) if laps is not None else None
        if result is None:
            rows.append(
                {"race_id": race_id, "year": year, "round": round_no,
                 "n_overtakes": np.nan, "total_laps": np.nan, "n_drivers": np.nan,
                 "n_dnf": np.nan, "is_usable": False}
            )
        else:
            rows.append({"race_id": race_id, "year": year, "round": round_no,
                         **result, "is_usable": True})

    df = pd.DataFrame(rows).sort_values(["year", "round"]).reset_index(drop=True)
    return df


def save_overtakes(df: pd.DataFrame, path: Path = OVERTAKES_PARQUET) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_overtakes(path: Path = OVERTAKES_PARQUET) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m src.process.overtakes build")
    return pd.read_parquet(path)


def _print_summary(df: pd.DataFrame) -> None:
    usable = df[df["is_usable"]]
    print(f"[process.overtakes] {len(df)} races, {len(usable)} usable")
    if usable.empty:
        return
    o = usable["n_overtakes"]
    print(
        f"[process.overtakes] overtakes/race: mean={o.mean():.1f} "
        f"median={o.median():.0f} min={o.min():.0f} max={o.max():.0f}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build", help="RAW race laps -> data/processed/overtakes.parquet")
    sub.add_parser("show", help="Print summary of the current overtakes parquet")

    args = p.parse_args(argv)

    if args.cmd == "build":
        df = build_overtakes()
        out = save_overtakes(df)
        print(f"[process.overtakes] saved -> {out}")
        _print_summary(df)
        return 0
    if args.cmd == "show":
        _print_summary(load_overtakes())
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
