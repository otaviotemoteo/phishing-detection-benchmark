#!/usr/bin/env python3
"""
Compare the same experiments across platforms, metric by metric.

The reproducibility claim in this project has a stated boundary: metric values
reproduce bitwise when the platform is the same and do not when it changes,
because a seed pins the random choices and not the arithmetic. This script turns
that claim into evidence. It reads experiment manifests, groups them by the run
they describe (model plus dataset, timestamp removed), and for every run present
on more than one platform reports each platform's metrics and the largest
absolute gap between them.

Platform comes from each manifest's ``environment`` block. Manifests written
before that block existed are read through `src.utils.manifests.manifest_environment`,
which labels them with the inferred reference profile and marks them as
inferred, so an assumption is never mistaken for a record. The ``inferred``
column in the output says which is which.

Output: ``results/analysis/platform_comparison.csv``, one row per run and
platform, with the per-run maximum gap repeated on each of that run's rows.

Usage:
    python scripts/compare_platforms.py                          # results/manifests/
    python scripts/compare_platforms.py DIR_A DIR_B [DIR_C ...]  # several platforms
    python scripts/compare_platforms.py --out PATH.csv

Exit code:
    0 - comparison written, or only one platform available (not an error)
    1 - no manifests found in the given directories
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.manifests import (  # noqa: E402  (needs REPO_ROOT on sys.path)
    load_manifest,
    manifest_environment,
)

DEFAULT_MANIFESTS_DIR = REPO_ROOT / "results" / "manifests"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "analysis" / "platform_comparison.csv"

METRIC_COLUMNS = ("accuracy", "precision", "recall", "f1", "auc_roc")

# An experiment id ends with the run timestamp; stripping it leaves the run name,
# which is the same across platforms.
_TIMESTAMP = re.compile(r"_\d{8}_\d{6}$")


def run_name(manifest: dict) -> str:
    """The model+dataset name of a run, with its timestamp removed."""
    return _TIMESTAMP.sub("", manifest.get("experiment_id", ""))


def collect(directories: list[Path]) -> list[dict]:
    """Read every manifest under the given directories."""
    manifests = []
    for directory in directories:
        if not directory.is_dir():
            print(f"  skipped (not a directory): {directory}")
            continue
        found = sorted(directory.glob("*.json"))
        print(f"  {len(found):3d} manifest(s) in {directory}")
        manifests.extend(load_manifest(path) for path in found)
    return manifests


def index_by_run_and_platform(manifests: list[dict]) -> dict[str, dict[str, dict]]:
    """Group manifests by run name, then by platform profile.

    When a run was executed more than once on the same platform, the most recent
    manifest wins and the earlier ones are counted, since repeated runs on one
    platform are the same-platform reproduction and not a cross-platform gap.
    """
    index: dict[str, dict[str, dict]] = {}
    for manifest in manifests:
        name = run_name(manifest)
        if not name or "metrics" not in manifest:
            continue
        env = manifest_environment(manifest)
        profile = env.get("profile", "unknown")
        cell = index.setdefault(name, {}).get(profile)
        entry = {"manifest": manifest, "environment": env, "n_runs": 1}
        if cell is None:
            index[name][profile] = entry
        else:
            cell["n_runs"] += 1
            if manifest.get("timestamp", "") > cell["manifest"].get("timestamp", ""):
                cell["manifest"] = manifest
                cell["environment"] = env
    return index


def max_gaps(platform_cells: dict[str, dict]) -> dict[str, float | None]:
    """Largest absolute difference per metric across the platforms of one run."""
    gaps: dict[str, float | None] = {}
    for metric in METRIC_COLUMNS:
        values = [
            cell["manifest"]["metrics"][metric]
            for cell in platform_cells.values()
            if isinstance(cell["manifest"].get("metrics", {}).get(metric), (int, float))
        ]
        gaps[metric] = round(max(values) - min(values), 6) if len(values) > 1 else None
    return gaps


def build_rows(index: dict[str, dict[str, dict]]) -> list[dict]:
    """Flatten the index into output rows for runs present on several platforms."""
    rows = []
    for name in sorted(index):
        platform_cells = index[name]
        if len(platform_cells) < 2:
            continue
        gaps = max_gaps(platform_cells)
        for profile in sorted(platform_cells):
            cell = platform_cells[profile]
            manifest = cell["manifest"]
            env = cell["environment"]
            metrics = manifest.get("metrics", {})
            row = {
                "run": name,
                "model": manifest.get("model"),
                "dataset": manifest.get("dataset"),
                "profile": profile,
                "blas": env.get("blas", "unknown"),
                "torch_backend": env.get("torch_backend", "unknown"),
                "gpu_name": env.get("gpu_name") or "",
                "inferred_profile": env.get("inferred", False),
                "timestamp": manifest.get("timestamp"),
                "runs_on_this_platform": cell["n_runs"],
            }
            for metric in METRIC_COLUMNS:
                value = metrics.get(metric)
                row[metric] = round(value, 6) if isinstance(value, (int, float)) else None
            for metric in METRIC_COLUMNS:
                row[f"{metric}_max_gap"] = gaps[metric]
            rows.append(row)
    return rows


def write_csv(rows: list[dict], out_path: Path) -> None:
    """Write the comparison rows to ``out_path``, creating its directory."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    """Print each compared run and the overall worst gap per metric."""
    print("\nRuns present on more than one platform:")
    for row in rows:
        metrics = "  ".join(f"{m}={row[m]:.4f}" for m in METRIC_COLUMNS if row[m] is not None)
        print(
            f"  {row['run']:<48} {row['profile']:<20} "
            f"blas={row['blas']:<11} torch={row['torch_backend']:<5} {metrics}"
        )

    print("\nLargest absolute gap between platforms, per metric:")
    for metric in METRIC_COLUMNS:
        gaps = [row[f"{metric}_max_gap"] for row in rows if row[f"{metric}_max_gap"] is not None]
        if gaps:
            worst = max(gaps)
            print(f"  {metric:<10} {worst:.6f}")
        else:
            print(f"  {metric:<10} not comparable")


def _display_path(path: Path) -> str:
    """Show a path relative to the repository when it lives inside it."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    """Read the manifest directories, compare platforms, and write the CSV."""
    parser = argparse.ArgumentParser(description="Compare experiments across platforms.")
    parser.add_argument(
        "dirs",
        nargs="*",
        default=[str(DEFAULT_MANIFESTS_DIR)],
        help="manifest directories to read (default: results/manifests/)",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUTPUT), help="where to write the comparison CSV"
    )
    args = parser.parse_args(argv)

    directories = [Path(d) for d in (args.dirs or [str(DEFAULT_MANIFESTS_DIR)])]
    print("Reading manifests:")
    manifests = collect(directories)
    if not manifests:
        print("\nNo manifests found. Nothing to compare.", file=sys.stderr)
        return 1

    index = index_by_run_and_platform(manifests)
    profiles = sorted({profile for cells in index.values() for profile in cells})
    print(f"\n{len(manifests)} manifest(s), {len(index)} run(s), {len(profiles)} platform(s): "
          f"{', '.join(profiles)}")

    rows = build_rows(index)
    if not rows:
        print(
            "\nOnly one platform is represented in these manifests, so there is\n"
            "nothing to compare. Cross-platform comparison needs manifests from a\n"
            "second platform: run the pipeline there and point this script at both\n"
            "directories, for example:\n"
            "  python scripts/compare_platforms.py results/manifests results/manifests_macos"
        )
        return 0

    out_path = Path(args.out)
    write_csv(rows, out_path)
    print_summary(rows)
    print(f"\nWrote {len(rows)} row(s) to {_display_path(out_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
