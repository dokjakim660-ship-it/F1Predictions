"""Local odds entry for the betting markets -- the CLI counterpart to the HF
Stakes pages' Save buttons. HF Spaces' filesystem is ephemeral/read-only, so
odds typed into the deployed app are lost and never reach the git repo where
`roi` reads them. This saves them locally instead, in two steps:

  1. template -- write an editable CSV per market (driver, team, the deployed
     model's P, a blank `odds` column), pre-filled from any odds already saved
     for the race so re-editing is non-destructive.
  2. save     -- read the filled CSV(s), write data/odds/{race_id}_{target}.json
     (driver_id -> decimal odds, only > 1.0) AND archive the prediction parquet
     to predictions/archive/ under the name roi expects, so `roi run`/`run-quali`
     can settle every model against these odds after the session.

Covers both market families:
  - quali (pre-quali, predicted BEFORE qualifying): pole / top3_quali /
    top10_quali / teammate_quali -- enter odds after FP2, before qualifying.
  - race  (pre-race, predicted AFTER qualifying): podium / teammate -- enter
    odds after qualifying, before the race (needs `just predict-next` first).

Decimal odds may use a comma (German bookmaker format, "1,70") -- parsed here.

Run:
  python -m src.eval.odds_entry template --year 2026 --round 6 --target quali
  # edit data/odds/templates/2026_06_pole.csv -> fill the `odds` column, save
  python -m src.eval.odds_entry save --year 2026 --round 6 --target quali
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.paths import ODDS_DIR, PREDICTIONS_DIR

ARCHIVE_DIR = PREDICTIONS_DIR / "archive"
TEMPLATE_DIR = ODDS_DIR / "templates"


@dataclass(frozen=True)
class _Market:
    """One betting market: where its predictions live and how roi names them."""

    name: str  # CLI/target key + odds-JSON stem (e.g. "pole", "podium")
    pred_stem: str  # prediction parquet stem under predictions/
    archive_prefix: str  # "" (race) or "prequali_" (quali) in the archive filename
    deployed_model: str  # model whose P is shown in the template for orientation
    mode_based: bool  # quali probs carry a post_fp2/pre_weekend mode; race do not
    rebuild_hint: str  # `just <hint> Y N` to (re)generate the prediction parquet


_QUALI_TARGETS = ("pole", "top3_quali", "top10_quali", "teammate_quali")
_RACE_TARGETS = ("podium", "teammate")

_MARKETS: dict[str, _Market] = {
    t: _Market(
        t, f"next_race_prequali_{t}", "prequali_", "logisticregression", True, "predict-pre-quali"
    )
    for t in _QUALI_TARGETS
}
# Deployed race models mirror roi._RACE_MODEL (podium=ensemble, teammate=logreg).
_MARKETS["podium"] = _Market("podium", "next_race_podium", "", "ensemble", False, "predict-next")
_MARKETS["teammate"] = _Market(
    "teammate", "next_race_teammate", "", "logisticregression", False, "predict-next"
)


def _prediction_path(m: _Market) -> Path:
    return PREDICTIONS_DIR / f"{m.pred_stem}.parquet"


def _archive_path(m: _Market, race_id: str) -> Path:
    return ARCHIVE_DIR / f"{race_id}_{m.archive_prefix}{m.name}.parquet"


def _template_path(race_id: str, name: str) -> Path:
    return TEMPLATE_DIR / f"{race_id}_{name}.csv"


def _odds_path(race_id: str, name: str) -> Path:
    return ODDS_DIR / f"{race_id}_{name}.json"


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


def _load_predictions(m: _Market, year: int, round_no: int) -> pd.DataFrame:
    race_id = f"{year}_{round_no:02d}"
    path = _prediction_path(m)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} missing -- run `just {m.rebuild_hint} {year} {round_no}` first."
        )
    df = pd.read_parquet(path)
    if (df["race_id"] != race_id).any():
        present = sorted(df["race_id"].unique())
        raise ValueError(
            f"{path.name} holds {present}, not {race_id} -- "
            f"run `just {m.rebuild_hint} {year} {round_no}`."
        )
    return df


def _p_column(m: _Market, df: pd.DataFrame) -> str:
    if not m.mode_based:
        return f"prob_{m.deployed_model}_cal"
    mode = "post_fp2" if int(df["has_fp2"].fillna(0).max()) else "pre_weekend"
    return f"prob_{mode}_{m.deployed_model}_cal"


def write_template(year: int, round_no: int, name: str) -> Path:
    m = _MARKETS[name]
    race_id = f"{year}_{round_no:02d}"
    df = _load_predictions(m, year, round_no)

    pcol = _p_column(m, df)
    # Race parquets carry only ids; quali parquets also carry display names.
    driver = df["driver_family_name"] if "driver_family_name" in df.columns else df["driver_id"]
    team = df["constructor_name"] if "constructor_name" in df.columns else df["constructor_id"]

    saved = {}
    if _odds_path(race_id, name).exists():
        saved = json.loads(_odds_path(race_id, name).read_text())

    out = pd.DataFrame(
        {
            "driver_id": df["driver_id"],
            "driver": driver,
            "team": team,
            "P_model_pct": (df[pcol] * 100).round(1),
            "odds": [saved.get(str(d), "") for d in df["driver_id"]],
        }
    ).sort_values("P_model_pct", ascending=False)

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    path = _template_path(race_id, name)
    out.to_csv(path, index=False)
    return path


def save_odds(year: int, round_no: int, name: str) -> tuple[int, Path, Path] | None:
    """Read the filled template, write the odds JSON + archive the prediction.

    Returns (n_odds, odds_json_path, archive_path) or None if no template/odds.
    """
    m = _MARKETS[name]
    race_id = f"{year}_{round_no:02d}"
    tmpl = _template_path(race_id, name)
    if not tmpl.exists():
        print(f"[odds] no template for {race_id} {name} -- run `template` first. Skipping.")
        return None

    df = pd.read_csv(tmpl)
    odds_map = {
        str(r["driver_id"]): o for _, r in df.iterrows() if (o := _parse_odds(r.get("odds"))) > 1.0
    }
    if not odds_map:
        print(f"[odds] {race_id} {name}: no odds > 1.0 in the template -- nothing saved.")
        return None

    ODDS_DIR.mkdir(parents=True, exist_ok=True)
    odds_json = _odds_path(race_id, name)
    odds_json.write_text(json.dumps(odds_map, indent=2))

    # Archive the prediction parquet AS BET so roi settles the exact snapshot the
    # odds were entered against (mirrors the Stakes pages).
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    archive = _archive_path(m, race_id)
    shutil.copy2(_prediction_path(m), archive)

    print(
        f"[odds] {race_id} {name}: saved {len(odds_map)} odds "
        f"-> {odds_json.name}, archived prediction"
    )
    return len(odds_map), odds_json, archive


def _targets_arg(target: str) -> list[str]:
    if target == "all":
        return list(_MARKETS)
    if target == "quali":
        return list(_QUALI_TARGETS)
    if target == "race":
        return list(_RACE_TARGETS)
    return [target]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    choices = (*_MARKETS, "all", "quali", "race")
    for cmd, helptext in (
        ("template", "Write editable odds CSV(s) for a race's markets."),
        ("save", "Read filled CSV(s) -> odds JSON + archived prediction for ROI."),
    ):
        sp = sub.add_parser(cmd, help=helptext)
        sp.add_argument("--year", type=int, required=True)
        sp.add_argument("--round", type=int, required=True)
        sp.add_argument("--target", choices=choices, default="all")
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
            "\nEdit the `odds` column (decimal, comma or dot), then run save, e.g.:\n"
            f"  python -m src.eval.odds_entry save "
            f"--year {args.year} --round {args.round} --target {args.target}"
        )
        return 0
    if args.cmd == "save":
        for t in targets:
            save_odds(args.year, args.round, t)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
