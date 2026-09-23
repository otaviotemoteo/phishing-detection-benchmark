#!/usr/bin/env python3
"""
Rewrite absolute artifact paths in experiment manifests to repository-relative ones.

The manifests written before the repository audit recorded artifact paths as they
existed on the author's machine, for example
``/home/<user>/.../phishing_detection/results/models/RandomForest_uci_*.joblib``.
That leaks a local directory layout and is useless to anyone else, so every such
path is cut down to the part that is stable across clones: ``results/models/...``.

**Only the ``artifacts`` block is touched.** Metrics, cost, hyperparameters,
dataset hash, git commit, seed and timestamp are recorded measurements of runs
that are not going to be repeated, and rewriting any of them would invalidate
the numbers the dissertation cites. Key order is preserved and files are written
back with the same ``indent=2`` formatting.

Usage:
    python scripts/migrate_manifests.py               # rewrite results/manifests/
    python scripts/migrate_manifests.py --dry-run     # report what would change
    python scripts/migrate_manifests.py --dir PATH    # a different manifest directory
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFESTS_DIR = REPO_ROOT / "results" / "manifests"

# Top-level repository directories an artifact path can legitimately point into.
# The first one found in an absolute path marks where the repository-relative
# portion begins.
_REPO_DIRS = ("results", "plots", "data", "notebooks", "src", "scripts")


def to_repo_relative(path_value: str) -> str:
    """Cut an absolute artifact path down to its repository-relative part.

    Args:
        path_value: A recorded artifact path, absolute or already relative.

    Returns:
        The path starting at a known top-level repository directory, with
        forward slashes. Returned unchanged when it is already relative or when
        no repository directory can be located in it (reported by the caller).
    """
    normalized = path_value.replace("\\", "/")
    parts = normalized.split("/")
    for i, part in enumerate(parts):
        if part in _REPO_DIRS:
            return "/".join(parts[i:])
    return path_value


def migrate_file(
    path: Path, *, dry_run: bool = False
) -> tuple[list[tuple[str, str, str]], dict]:
    """Rewrite one manifest's artifact paths in place.

    Args:
        path: The manifest JSON file.
        dry_run: If True, compute the changes without writing.

    Returns:
        ``(changes, manifest)`` where ``changes`` lists
        ``(artifact_key, old_value, new_value)`` per path changed and
        ``manifest`` is the migrated dict (written to disk unless ``dry_run``).
    """
    manifest = json.loads(path.read_text())
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        return [], manifest

    changes: list[tuple[str, str, str]] = []
    for key, value in artifacts.items():
        if not isinstance(value, str):
            continue
        new_value = to_repo_relative(value)
        if new_value != value:
            artifacts[key] = new_value
            changes.append((key, value, new_value))

    if changes and not dry_run:
        # json.dumps preserves insertion order, so the manifest keeps its shape.
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return changes, manifest


def unresolved_absolute_paths(manifest: dict) -> list[str]:
    """Return artifact values that still look absolute after migration."""
    artifacts = manifest.get("artifacts") or {}
    return [
        value
        for value in artifacts.values()
        if isinstance(value, str) and (value.startswith("/") or ":" in value[:3])
    ]


def main(argv: list[str] | None = None) -> int:
    """Migrate every manifest in the chosen directory and report what changed."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--dir", default=str(DEFAULT_MANIFESTS_DIR), help="manifest directory to migrate"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report changes without writing files"
    )
    args = parser.parse_args(argv)

    directory = Path(args.dir)
    if not directory.is_dir():
        print(f"No such manifest directory: {directory}", file=sys.stderr)
        return 1

    manifests = sorted(directory.glob("*.json"))
    n_changed = 0
    n_paths = 0
    leftovers: list[str] = []

    for manifest_path in manifests:
        changes, migrated = migrate_file(manifest_path, dry_run=args.dry_run)
        if changes:
            n_changed += 1
            n_paths += len(changes)
            print(f"{manifest_path.name}: {len(changes)} path(s) rewritten")
            for key, old, new in changes:
                print(f"    {key}: {old}  ->  {new}")
        remaining = unresolved_absolute_paths(migrated)
        leftovers.extend(f"{manifest_path.name}: {value}" for value in remaining)

    verb = "would rewrite" if args.dry_run else "rewrote"
    print(f"\n{verb} {n_paths} path(s) across {n_changed}/{len(manifests)} manifest(s).")

    if leftovers:
        print("\nStill absolute after migration (no known repo directory found):")
        for item in leftovers:
            print(f"  {item}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
