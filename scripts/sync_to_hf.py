"""Copy the deployable subset of the repo into a local HuggingFace Space clone.

Phase 2.4 deployment helper. Mirrors only what the runtime needs:

- deploy/hf/Dockerfile          -> {hf}/Dockerfile
- deploy/hf/requirements.txt    -> {hf}/requirements.txt
- deploy/hf/README.md           -> {hf}/README.md   (overwrites HFs auto-generated one)
- app/                          -> {hf}/app/
- predictions/mvp_test_*.parquet,
  predictions/importance_*.parquet
                                -> {hf}/predictions/   (only the parquets the app reads;
                                                        baseline.parquet from Phase 1.3
                                                        stays behind, on purpose)
- models/reliability_mvp_*.png  -> {hf}/models/
- data/reference/race_inventory.parquet
                                -> {hf}/data/reference/race_inventory.parquet

Each mirror dir is wiped before copy so deletions on the source side surface
in the HF clone (and stale files dont silently linger between deploys).

The script does NOT touch the HF clones .git directory and does NOT commit
or push -- it just stages the files. The user runs `git status`, sanity-checks
the diff, then commits + pushes from the HF clone themselves.

Run: `python scripts/sync_to_hf.py <path-to-hf-clone>`
or:  `just deploy-hf HF_PATH=<path>`
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (src_spec, dest_relative_to_hf, kind)
# - kind="file": src_spec is one repo-relative path -> dest is the file path
# - kind="dir":  src_spec is a directory           -> dest is a directory (wiped + copied)
# - kind="globs": src_spec is a tuple of glob patterns -> dest is a directory; each
#                 matching file lands in dest/<basename>, dest wiped once before copy
SYNC_PLAN: list[tuple[str | tuple[str, ...], str, str]] = [
    ("deploy/hf/Dockerfile", "Dockerfile", "file"),
    ("deploy/hf/requirements.txt", "requirements.txt", "file"),
    ("deploy/hf/README.md", "README.md", "file"),
    ("app", "app", "dir"),
    (
        ("predictions/mvp_test_*.parquet", "predictions/importance_*.parquet"),
        "predictions",
        "globs",
    ),
    (("models/reliability_mvp_*.png",), "models", "globs"),
    ("data/reference/race_inventory.parquet", "data/reference/race_inventory.parquet", "file"),
]


def _assert_hf_clone(hf_path: Path) -> None:
    if not hf_path.exists():
        raise SystemExit(f"HF clone path does not exist: {hf_path}")
    if not (hf_path / ".git").exists():
        raise SystemExit(
            f"{hf_path} is not a git repository. "
            "Clone the empty HF Space repo there first."
        )


def _copy_file(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _copy_dir(src: Path, dest: Path) -> None:
    # Wipe before copy so removed source files surface as deletions in the HF clone.
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _copy_globs(patterns: tuple[str, ...], dest_dir: Path) -> int:
    """Copy each file matched by any of `patterns` (relative to REPO_ROOT) into dest_dir.

    dest_dir is wiped once before the first copy so stale files vanish. Multiple
    patterns are unioned and deduplicated. Returns the total number of files copied.
    """
    src_files: list[Path] = []
    for pattern in patterns:
        matched = sorted(REPO_ROOT.glob(pattern))
        if not matched:
            raise SystemExit(
                f"No files matched glob {pattern!r} under {REPO_ROOT}. "
                "Run `just final-eval` / `just importance` first to produce them."
            )
        src_files.extend(matched)
    # Dedupe in case two patterns overlap.
    src_files = sorted(set(src_files))

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in src_files:
        shutil.copy2(src, dest_dir / src.name)
    return len(src_files)


def sync(hf_path: Path) -> None:
    _assert_hf_clone(hf_path)
    print(f"[sync] target: {hf_path}")

    for src_spec, dest_rel, kind in SYNC_PLAN:
        dest = hf_path / dest_rel
        if kind == "file":
            assert isinstance(src_spec, str)
            src = REPO_ROOT / src_spec
            if not src.exists():
                raise SystemExit(f"Source file missing: {src}")
            _copy_file(src, dest)
            print(f"  file   {src_spec:<40s} -> {dest_rel}")
        elif kind == "dir":
            assert isinstance(src_spec, str)
            src = REPO_ROOT / src_spec
            if not src.exists():
                raise SystemExit(f"Source dir missing: {src}")
            _copy_dir(src, dest)
            n = sum(1 for _ in dest.rglob("*") if _.is_file())
            print(f"  dir    {src_spec:<40s} -> {dest_rel}/  ({n} files)")
        elif kind == "globs":
            assert isinstance(src_spec, tuple)
            n = _copy_globs(src_spec, dest)
            label = ", ".join(src_spec)
            print(f"  globs  {label:<40s} -> {dest_rel}/  ({n} files)")

    print()
    print(f"[sync] done. Next: cd {hf_path} && git status && git add -A && git commit && git push")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("hf_path", type=Path, help="Path to the local HF Space clone (a git repo).")
    args = p.parse_args(argv)
    sync(args.hf_path.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
