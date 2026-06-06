"""Local odds entry for the quali betting markets -- the CLI counterpart to the
HF Stakes page's Save button. HF Spaces' filesystem is ephemeral/read-only, so
odds typed into the deployed app are lost and never reach the git repo where
`roi run-quali` reads them. This saves them locally instead, in two steps:

  1. template -- write an editable CSV per market (driver, team, the deployed
     model's P, a blank `odds` column), pre-filled from any odds already saved
     for the race so re-editing is non-destructive.
  2. save     -- read the filled CSV(s), write data/odds/{race_id}_{target}.json
     (driver_id -> decimal odds, only > 1.0) AND archive the prediction parquet
     to predictions/archive/{race_id}_prequali_{target}.parquet, so
     `roi run-quali` can settle every model against these odds after the session.

Decimal odds may use a comma (German bookmaker format, "1,70") -- parsed here.

Run:
  python -m src.eval.odds_entry template --year 2026 --round 6
  # edit data/odds/templates/2026_06_pole.csv -> fill the `odds` column, save
  python -m src.eval.odds_entry save --year 2026 --round 6
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

from src.utils.paths import ODDS_DIR, PREDICTIONS_DIR

ARCHIVE_DIR = PREDICTIONS_DIR / "archive"
TEMPLATE_DIR = ODDS_DIR / "templates"

# Markets this CLI handles + their prediction parquet stem. Same four pre-quali
# markets roi.evaluate_quali_race settles.
_TARGETS = ("pole", "top3_quali", "top10_quali", "teammate_quali")
# Deployed model whose probability is shown for orientation in the template.
_DEPLOYED_MODEL = "logisticregression"


def _prediction_path(target: str) -> Path:
    return PREDICTIONS_DIR / f"next_race_prequali_{target}.parquet"


def _template_path(race_id: str, target: str) -> Path:
    return TEMPLATE_DIR / f"{race_id}_{target}.csv"


def _odds_path(race_id: str, target: str) -> Path:
    return ODDS_DIR / f"{race_id}_{target}.json"


def _parse_odds(raw: object) -> float:
    """Decimal odds string/number -> float; comma-decimal and blanks tolerated."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return 0.0
    s = str(raw).strip().replace(",", ".")
    if not s:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _load_predictions(target: str, race_id: str) -> pd.DataFrame:
    path = _prediction_path(target)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} missing -- run "
            f"`just predict-pre-quali {race_id[:4]} {int(race_id[5:])}` first."
        )
    df = pd.read_parquet(path)
    if (df["race_id"] != race_id).any():
        present = sorted(df["race_id"].unique())
        raise ValueError(f"{path.name} holds {present}, not {race_id} -- re-run predict-pre-quali.")
    return df


def write_template(year: int, round_no: int, target: str) -> Path:
    race_id = f"{year}_{round_no:02d}"
    df = _load_predictions(target, race_id)

    mode = "post_fp2" if int(df["has_fp2"].fillna(0).max()) else "pre_weekend"
    pcol = f"prob_{mode}_{_DEPLOYED_MODEL}_cal"

    saved = {}
    if _odds_path(race_id, target).exists():
        saved = json.loads(_odds_path(race_id, target).read_text())

    out = pd.DataFrame(
        {
            "driver_id": df["driver_id"],
            "driver": df["driver_family_name"].fillna(df["driver_id"]),
            "team": df["constructor_name"].fillna(df["constructor_id"]),
            "P_model_pct": (df[pcol] * 100).round(1),
            "odds": [saved.get(str(d), "") for d in df["driver_id"]],
        }
    ).sort_values("P_model_pct", ascending=False)

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _template_path(race_id, target)
    out.to_csv(path, index=False)
    return path


def save_odds(year: int, round_no: int, target: str) -> tuple[int, Path, Path] | None:
    """Read the filled template, write the odds JSON + archive the prediction.

    Returns (n_odds, odds_json_path, archive_path) or None if no template/odds.
    """
    race_id = f"{year}_{round_no:02d}"
    tmpl = _template_path(race_id, target)
    if not tmpl.exists():
        print(f"[odds] no template for {race_id} {target} -- run `template` first. Skipping.")
        return None

    df = pd.read_csv(tmpl)
    odds_map = {
        str(r["driver_id"]): o for _, r in df.iterrows() if (o := _parse_odds(r.get("odds"))) > 1.0
    }
    if not odds_map:
        print(f"[odds] {race_id} {target}: no odds > 1.0 in the template -- nothing saved.")
        return None

    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    odds_json = _odds_path(race_id, target)
    odds_json.write_text(json.dumps(odds_map, indent=2))

    # Archive the prediction parquet AS BET so roi settles the exact pre-quali
    # snapshot the odds were entered against (mirrors the Stakes page).
    src = _prediction_path(target)
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive = ARCHIVE_DIR / f"{race_id}_prequali_{target}.parquet"
    shutil.copy2(src, archive)

    print(
        f"[odds] {race_id} {target}: saved {len(odds_map)} odds "
        f"-> {odds_json.name}, archived prediction"
    )
    return len(odds_map), odds_json, archive


def _targets_arg(target: str) -> list[str]:
    return list(_TARGETS) if target == "all" else [target]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for cmd, helptext in (
        ("template", "Write editable odds CSV(s) for a race's quali markets."),
        ("save", "Read filled CSV(s) -> odds JSON + archived prediction for ROI."),
    ):
        sp = sub.add_parser(cmd, help=helptext)
        sp.add_argument("--year", type=int, required=True)
        sp.add_argument("--round", type=int, required=True)
        sp.add_argument("--target", choices=(*_TARGETS, "all"), default="all")
    args = p.parse_args(argv)

    targets = _targets_arg(args.target)
    if args.cmd == "template":
        for t in targets:
            try:
                path = write_template(args.year, args.round, t)
                print(f"[odds] template -> {path}")
            except (FileNotFoundError, ValueError) as e:
                print(f"[odds] {t}: {e}")
        print(
            "\nEdit the `odds` column (decimal, comma or dot), then: "
            f"just save-quali-odds {args.year} {args.round}"
        )
        return 0
    if args.cmd == "save":
        for t in targets:
            save_odds(args.year, args.round, t)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
