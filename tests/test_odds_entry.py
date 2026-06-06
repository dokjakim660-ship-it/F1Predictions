"""Local odds-entry CLI: template -> fill -> save round-trip.

Covers the bits that matter for the downstream ROI settle: comma-decimal odds
parsing, the >1.0 filter, and that save writes the odds JSON AND archives the
prediction snapshot under the name roi.evaluate_quali_race expects.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src.eval import odds_entry


@pytest.fixture
def wired(tmp_path, monkeypatch):
    odds = tmp_path / "odds"
    tmpl = odds / "templates"
    archive = tmp_path / "archive"
    preds = tmp_path / "predictions"
    for d in (odds, tmpl, archive, preds):
        d.mkdir(parents=True)
    monkeypatch.setattr(odds_entry, "ODDS_DIR", odds)
    monkeypatch.setattr(odds_entry, "TEMPLATE_DIR", tmpl)
    monkeypatch.setattr(odds_entry, "ARCHIVE_DIR", archive)
    monkeypatch.setattr(odds_entry, "PREDICTIONS_DIR", preds)
    return tmp_path


def _write_pred(wired) -> None:
    pd.DataFrame(
        {
            "race_id": "2099_01",
            "driver_id": ["ver", "ham", "nor"],
            "driver_family_name": ["Verstappen", "Hamilton", "Norris"],
            "constructor_id": ["rb", "ferrari", "mclaren"],
            "constructor_name": ["Red Bull", "Ferrari", "McLaren"],
            "has_fp2": 1,
            "prob_post_fp2_logisticregression_cal": [0.4, 0.3, 0.2],
            "prob_pre_weekend_logisticregression_cal": [0.3, 0.3, 0.3],
        }
    ).to_parquet(odds_entry._prediction_path(odds_entry._MARKETS["pole"]), index=False)


def test_template_then_save_roundtrip(wired):
    _write_pred(wired)

    tmpl_path = odds_entry.write_template(2099, 1, "pole")
    assert tmpl_path.exists()

    # Simulate the user filling the odds column: comma-decimal, one blank (skip),
    # one <= 1.0 (skip -- not a real bet).
    df = pd.read_csv(tmpl_path)
    df["odds"] = df["odds"].astype("object")  # all-blank column reads as float; allow strings
    df.loc[df["driver_id"] == "ver", "odds"] = "2,50"  # German comma decimal
    df.loc[df["driver_id"] == "ham", "odds"] = ""  # left blank
    df.loc[df["driver_id"] == "nor", "odds"] = "1.0"  # not > 1.0
    df.to_csv(tmpl_path, index=False)

    result = odds_entry.save_odds(2099, 1, "pole")
    assert result is not None
    n, odds_json, archive = result

    saved = json.loads(odds_json.read_text())
    assert saved == {"ver": 2.5}  # comma parsed, blank + <=1.0 dropped
    assert n == 1
    # Archived under the exact name roi.evaluate_quali_race reads.
    assert archive.name == "2099_01_prequali_pole.parquet"
    assert archive.exists()


def test_save_without_template_is_noop(wired):
    _write_pred(wired)
    assert odds_entry.save_odds(2099, 1, "pole") is None


def test_race_market_roundtrip_uses_id_fallback(wired):
    """Race parquets carry only ids (no family/team names) and no timing mode."""
    pd.DataFrame(
        {
            "race_id": "2099_01",
            "driver_id": ["ver", "ham"],
            "constructor_id": ["rb", "ferrari"],
            "prob_ensemble_cal": [0.5, 0.3],  # deployed podium model, no mode prefix
        }
    ).to_parquet(odds_entry._prediction_path(odds_entry._MARKETS["podium"]), index=False)

    tmpl = odds_entry.write_template(2099, 1, "podium")
    df = pd.read_csv(tmpl)
    # Falls back to driver_id / constructor_id for the display columns.
    assert df.loc[df["driver_id"] == "ver", "driver"].iloc[0] == "ver"
    assert df.loc[df["driver_id"] == "ver", "team"].iloc[0] == "rb"

    df["odds"] = df["odds"].astype("object")
    df.loc[df["driver_id"] == "ver", "odds"] = "2.5"
    df.to_csv(tmpl, index=False)

    result = odds_entry.save_odds(2099, 1, "podium")
    assert result is not None
    _, odds_json, archive = result
    assert json.loads(odds_json.read_text()) == {"ver": 2.5}
    # Race archive has NO `prequali_` infix (roi.evaluate_race reads this name).
    assert archive.name == "2099_01_podium.parquet"
