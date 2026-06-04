"""Tiny helper for emitting GitHub Actions step outputs from Python CLIs.

Steps consume these via `steps.<id>.outputs.<key>`. Outside CI (no
GITHUB_OUTPUT env var) this is a no-op, so the same commands run locally.
"""

from __future__ import annotations

import os
from pathlib import Path


def emit_output(**kv: object) -> None:
    """Append `key=value` lines to $GITHUB_OUTPUT, if set. No-op locally."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with Path(path).open("a", encoding="utf-8") as fh:
        for key, value in kv.items():
            fh.write(f"{key}={value}\n")
