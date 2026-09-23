"""
Experiment manifest generation and reading (DEVELOPMENT.md §6.4).

Every trained model emits a JSON manifest capturing exactly what produced it:
code version, dataset hash, seed, hyperparameters, library versions, platform,
metrics, cost, and artifact paths. If results ever fail to reproduce, the
manifest is the first place to look.

Two notes on reading manifests written by earlier versions of this module:

- The ``environment`` block was added during the repository audit. The 66
  manifests already in ``results/manifests/`` predate it and will never have
  one, because they record real runs that are not going to be repeated. Use
  `manifest_environment` to read it, which marks an absent block as inferred
  rather than pretending it was recorded.
- Artifact paths in those same manifests were absolute paths on the author's
  machine, rewritten to repository-relative paths by
  ``scripts/migrate_manifests.py``. Nothing else in them was touched.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.config import MANIFESTS_DIR
from src.utils.environment import LEGACY_PROFILE, detect_environment
from src.utils.io import DATASET_FILENAMES, load_dataset_hashes

# Packages whose versions are worth pinning into each manifest for reproducibility.
_TRACKED_PACKAGES = (
    "scikit-learn",
    "numpy",
    "pandas",
    "imbalanced-learn",
    "xgboost",
    "torch",
)


def _git_commit() -> str | None:
    """Return the short git commit hash, or ``None`` outside a repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


def _library_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return versions


def _dataset_hash(dataset: str) -> str | None:
    """Look up the recorded SHA256 for a dataset short name."""
    filename = DATASET_FILENAMES.get(dataset)
    if filename is None:
        return None
    return load_dataset_hashes().get(filename, {}).get("sha256")


def save_manifest(
    experiment_id: str,
    model_name: str,
    dataset: str,
    hyperparameters: dict,
    metrics: dict,
    cost: dict,
    artifacts: dict,
    seed: int,
) -> Path:
    """Write ``results/manifests/<experiment_id>.json`` per the §6.4 schema.

    Args:
        experiment_id: Unique id, e.g. ``"LogisticRegression_uci_20260623_140000"``.
        model_name: Estimator/model name.
        dataset: Dataset short name (``'uci'`` | ``'mendeley'`` | ``'iscx'``).
        hyperparameters: JSON-serializable hyperparameter dict.
        metrics: Output of `src.evaluation.metrics.compute_metrics`.
        cost: Cost dict (training time, inference time, RAM, GPU, params).
        artifacts: Mapping of artifact name -> saved path.
        seed: The random seed used.

    Returns:
        The manifest path written.
    """
    manifest = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().astimezone().isoformat(),
        "model": model_name,
        "dataset": dataset,
        "dataset_hash": _dataset_hash(dataset),
        "git_commit": _git_commit(),
        "seed": seed,
        "hyperparameters": hyperparameters,
        "library_versions": _library_versions(),
        "environment": detect_environment(),
        "metrics": metrics,
        "cost": cost,
        "artifacts": artifacts,
    }
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS_DIR / f"{experiment_id}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def load_manifest(path: Path) -> dict:
    """Read one manifest JSON file, exactly as written.

    Args:
        path: Path to the manifest file.

    Returns:
        The parsed manifest dict.
    """
    return json.loads(Path(path).read_text())


def load_manifests(directory: Path = MANIFESTS_DIR) -> list[dict]:
    """Read every manifest in ``directory``, sorted by filename.

    Args:
        directory: Directory holding ``*.json`` manifests. Defaults to
            ``results/manifests/``.

    Returns:
        A list of manifest dicts; empty if the directory does not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return [load_manifest(p) for p in sorted(directory.glob("*.json"))]


def manifest_environment(manifest: dict) -> dict:
    """Return a manifest's platform block, flagging it when it had to be inferred.

    Manifests written before the audit carry no ``environment`` block. Rather
    than fail or silently invent one, this returns the legacy profile with
    ``inferred=True`` and a note saying where that profile came from, so a
    reader never mistakes an inference for a recorded measurement.

    Args:
        manifest: A manifest dict from `load_manifest`.

    Returns:
        The recorded environment block with ``inferred: False``, or a minimal
        inferred block with ``inferred: True`` and a ``note``.
    """
    recorded = manifest.get("environment")
    if isinstance(recorded, dict) and recorded:
        return {**recorded, "inferred": False}
    return {
        "profile": LEGACY_PROFILE,
        "inferred": True,
        "note": (
            "No environment block recorded. Profile inferred from the repository "
            "history (all pre-audit runs were executed on the reference Linux "
            "machine), not read from the manifest."
        ),
    }
