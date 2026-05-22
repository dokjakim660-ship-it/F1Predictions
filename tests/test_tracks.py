"""Tests for the hand-curated track reference table."""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils.race_inventory import INVENTORY_PATH
from src.utils.tracks import load_tracks, race_to_track_id


def test_tracks_table_loads_without_nulls() -> None:
    df = load_tracks()
    assert len(df) >= 32
    for col in ("track_id", "circuit_id", "length_km", "n_corners", "n_drs_zones", "track_type"):
        assert df[col].notna().all(), f"null values in column {col}"


def test_track_ids_are_unique() -> None:
    assert load_tracks()["track_id"].is_unique


def test_track_attributes_in_plausible_range() -> None:
    df = load_tracks()
    assert df["length_km"].between(3.0, 8.0).all()
    assert df["n_corners"].between(7, 30).all()
    assert df["n_drs_zones"].between(0, 5).all()
    assert df["track_type"].isin({"permanent", "street"}).all()


def test_every_inventoried_circuit_has_a_track_row() -> None:
    if not INVENTORY_PATH.exists():
        pytest.skip("race_inventory.parquet not built -- run `just ingest-schedule`")
    inv = pd.read_parquet(INVENTORY_PATH)
    track_ids = set(load_tracks()["track_id"])
    missing = [
        (row.race_id, row.circuit_id)
        for row in inv[["race_id", "circuit_id"]].itertuples()
        if race_to_track_id(row.race_id, row.circuit_id) not in track_ids
    ]
    assert not missing, f"races with no matching track row: {missing[:5]}"


def test_sakhir_outer_circuit_override() -> None:
    assert race_to_track_id("2020_16", "bahrain") == "bahrain_outer"
    assert race_to_track_id("2024_01", "bahrain") == "bahrain"
