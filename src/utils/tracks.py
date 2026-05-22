"""Track reference attributes (data/reference/tracks.csv) + race->track mapping.

Hand-curated circuit attributes (length, corners, DRS zones, type). They feed
the track features in src/features/build.py and are deliberately a *cheap
secondary* signal -- values are approximate public circuit specs, and small
errors do not matter for a gradient-boosted model.

Most races resolve to a track by their Jolpica circuit_id. The exception is the
2020 Sakhir GP (race_id 2020_16): Jolpica labels it circuit_id "bahrain", but it
ran the 3.5 km Bahrain Outer Circuit -- a drastically shorter layout -- so it
gets its own track_id "bahrain_outer".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.paths import REFERENCE_DIR

TRACKS_CSV = REFERENCE_DIR / "tracks.csv"

# race_id -> track_id, where the raced layout differs from the Jolpica
# circuit_id. The Sakhir GP 2020 ran the Bahrain Outer Circuit.
_RACE_TRACK_OVERRIDES = {"2020_16": "bahrain_outer"}


def load_tracks(path: Path = TRACKS_CSV) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- the tracks reference table is hand-curated.")
    return pd.read_csv(path)


def race_to_track_id(race_id: str, circuit_id: str) -> str:
    """Resolve the track_id for a race: its circuit_id, or a layout override."""
    return _RACE_TRACK_OVERRIDES.get(race_id, circuit_id)
