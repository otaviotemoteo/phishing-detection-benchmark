"""
Experiment manifest generation (DEVELOPMENT.md §6.4).

Every trained model emits a JSON manifest capturing exactly what produced it:
code version, dataset hash, seed, hyperparameters, library versions, metrics, cost,
and artifact paths. If results ever fail to reproduce, the manifest is the first
place to look.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from src.config import MANIFESTS_DIR
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
        "metrics": metrics,
        "cost": cost,
        "artifacts": artifacts,
    }
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFESTS_DIR / f"{experiment_id}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
