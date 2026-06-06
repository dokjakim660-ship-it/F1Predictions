"""Phase 4.2.7 quali-market ROI eval.

Covers the parts that differ from the race markets: mode selection from has_fp2,
the realised-outcome lookup from the feature table's target columns, and that
appending the quali markets does not clobber the race markets for the same race.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from src.eval import roi


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Point all roi.* path globals at a tmp dir and return it."""
    odds = tmp_path / "odds"
    archive = tmp_path / "archive"
    feats = tmp_path / "mvp.parquet"
    odds.mkdir()
    archive.mkdir()
    monkeypatch.setattr(roi, "ODDS_DIR", odds)
    monkeypatch.setattr(roi, "ARCHIVE_DIR", archive)
    monkeypatch.setattr(roi, "MVP_FEATURES", feats)
    monkeypatch.setattr(roi, "ROI_DIR", tmp_path)
    monkeypatch.setattr(roi, "ROI_LOG", tmp_path / "log.parquet")
    return tmp_path


def _write_pole_fixture(wired, *, has_fp2: int) -> None:
    race_id = "2099_01"
    preds = pd.DataFrame(
        {
            "race_id": race_id,
            "driver_id": ["ver", "ham", "nor"],
            "constructor_id": ["rb", "merc", "mcl"],
            "has_fp2": has_fp2,
            # pre_weekend probs deliberately different so we can tell modes apart.
            "prob_pre_weekend_logisticregression_cal": [0.10, 0.10, 0.10],
            "prob_post_fp2_logisticregression_cal": [0.50, 0.30, 0.60],
        }
    )
    preds.to_parquet(roi.ARCHIVE_DIR / f"{race_id}_prequali_pole.parquet", index=False)
    (roi.ODDS_DIR / f"{race_id}_pole.json").write_text(
        json.dumps({"ver": 3.0, "ham": 3.0, "nor": 2.0})
    )
    # Realised outcome: ver took pole, nobody else.
    mvp = pd.DataFrame(
        {
            "race_id": race_id,
            "driver_id": ["ver", "ham", "nor"],
            "target_pole": [1.0, 0.0, 0.0],
            "target_top3_quali": [1.0, 1.0, 1.0],
            "target_top10_quali": [1.0, 1.0, 1.0],
            "target_quali_beat_teammate": [1.0, 0.0, 1.0],
        }
    )
    mvp.to_parquet(roi.MVP_FEATURES, index=False)


def test_quali_pole_pnl_post_fp2(wired):
    _write_pole_fixture(wired, has_fp2=1)
    df = roi.evaluate_quali_race(2099, 1, kelly_frac=0.25, bankroll=100.0)

    # ham has negative edge (p=0.30 < implied 0.333) -> no bet. ver + nor placed.
    assert set(df["driver_id"]) == {"ver", "nor"}

    ver = df[df["driver_id"] == "ver"].iloc[0]
    # kelly_raw = (0.5*3 - 1)/(3-1) = 0.25 -> stake 0.25*0.25*100 = 6.25
    assert math.isclose(ver["stake_eur"], 6.25, abs_tol=1e-6)
    assert bool(ver["won"]) is True
    assert math.isclose(ver["pnl_eur"], (3.0 - 1.0) * 6.25, abs_tol=1e-6)

    nor = df[df["driver_id"] == "nor"].iloc[0]
    # kelly_raw = (0.6*2 - 1)/(2-1) = 0.2 -> stake 0.2*0.25*100 = 5.0, lost
    assert math.isclose(nor["stake_eur"], 5.0, abs_tol=1e-6)
    assert bool(nor["won"]) is False
    assert math.isclose(nor["pnl_eur"], -5.0, abs_tol=1e-6)


def test_quali_mode_falls_back_on_sprint(wired):
    # has_fp2=0 -> pre_weekend probs (all 0.10) -> implied 1/3 or 1/2 always higher
    # -> zero value bets.
    _write_pole_fixture(wired, has_fp2=0)
    df = roi.evaluate_quali_race(2099, 1)
    assert df.empty


def test_quali_logs_every_model(wired):
    """All five pre-quali models present in the parquet are paper-traded + tagged."""
    race_id = "2099_01"
    base = {
        "race_id": race_id,
        "driver_id": ["ver", "ham", "nor"],
        "constructor_id": ["rb", "merc", "mcl"],
        "has_fp2": 1,
    }
    # Give every model a column where ver is a clear value bet (p well above 1/3).
    models = ["recentqualiform", "logisticregression", "xgboost", "lightgbm", "ensemble"]
    for m in models:
        base[f"prob_post_fp2_{m}_cal"] = [0.50, 0.05, 0.05]
        base[f"prob_pre_weekend_{m}_cal"] = [0.10, 0.10, 0.10]
    pd.DataFrame(base).to_parquet(roi.ARCHIVE_DIR / f"{race_id}_prequali_pole.parquet", index=False)
    (roi.ODDS_DIR / f"{race_id}_pole.json").write_text(json.dumps({"ver": 3.0}))
    pd.DataFrame(
        {
            "race_id": race_id,
            "driver_id": ["ver", "ham", "nor"],
            "target_pole": [1.0, 0.0, 0.0],
            "target_top3_quali": [1.0, 1.0, 1.0],
            "target_top10_quali": [1.0, 1.0, 1.0],
            "target_quali_beat_teammate": [1.0, 0.0, 1.0],
        }
    ).to_parquet(roi.MVP_FEATURES, index=False)

    df = roi.evaluate_quali_race(2099, 1)
    # Every model placed ver's pole bet, each tagged distinctly.
    assert set(df["model"]) == set(models)
    assert (df["driver_id"] == "ver").all()
    assert (df["target"] == "pole").all()


def test_append_keeps_other_models_and_markets(wired):
    base_cols = dict(
        race_id="2099_01",
        year=2099,
        round=1,
        driver_id="ver",
        p_model=0.5,
        odds=3.0,
        implied_prob=0.3333,
        edge=0.1667,
        kelly_frac=0.25,
        stake_eur=6.25,
        bankroll_eur=100.0,
        won=True,
        pnl_eur=12.5,
    )
    race_df = pd.DataFrame([{**base_cols, "target": "podium", "model": "ensemble"}])
    quali_lr = pd.DataFrame([{**base_cols, "target": "pole", "model": "logisticregression"}])
    quali_ens = pd.DataFrame([{**base_cols, "target": "pole", "model": "ensemble"}])

    roi.append_to_log(race_df)
    roi.append_to_log(pd.concat([quali_lr, quali_ens], ignore_index=True))

    log = roi.load_log()
    # Race market + both quali models coexist (dedup is per race_id/target/model).
    assert len(log) == 3
    assert set(zip(log["target"], log["model"], strict=True)) == {
        ("podium", "ensemble"),
        ("pole", "logisticregression"),
        ("pole", "ensemble"),
    }

    # Re-running only the LogReg pole bet replaces its own row, leaving the
    # ensemble pole bet and the podium row intact.
    roi.append_to_log(quali_lr)
    log = roi.load_log()
    assert len(log) == 3
