"""
Standardized I/O utilities.

Currently provides dataset hashing for reproducibility (DEVELOPMENT.md §6.3):
every experiment manifest records the SHA256 of its input data so results can
always be tied to the exact dataset version used. If results ever differ between
runs, the first check is whether the dataset hash still matches.

Run as a module to (re)generate the hash registry after downloading datasets:

    python -m src.utils.io

This writes ``data/dataset_hashes.json`` — the entry point already documented in
``data/README.md``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATA_DIR

# Datasets expected in DATA_DIR. Keys are the short names used throughout the
# project; values are the canonical on-disk filenames.
DATASET_FILENAMES: dict[str, str] = {
    "uci": "uci_phishing.csv",
    "mendeley": "mendeley_phishing.csv",
    "iscx": "iscx_url2016.csv",
}

HASHES_PATH: Path = DATA_DIR / "dataset_hashes.json"

_CHUNK_SIZE = 8192  # bytes read per iteration when hashing


def hash_dataset(filepath: Path) -> str:
    """Compute the SHA256 hex digest of a file.

    Reads the file in fixed-size chunks so that arbitrarily large datasets
    (e.g. the Mendeley URL+HTML dataset) hash with constant memory.

    Args:
        filepath: Path to the file to hash.

    Returns:
        The SHA256 digest as a lowercase hex string.

    Raises:
        FileNotFoundError: If ``filepath`` does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {filepath}")
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def write_dataset_hashes(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """Hash every known dataset present in ``data_dir`` and persist the registry.

    Datasets that have not been downloaded yet are skipped with a warning rather
    than raising, so this can be run incrementally as files arrive.

    Args:
        data_dir: Directory containing the dataset CSV files. Defaults to the
            project ``DATA_DIR``.

    Returns:
        The registry dict that was written to ``dataset_hashes.json``.
    """
    registry: dict[str, dict] = {}
    for name, filename in DATASET_FILENAMES.items():
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  SKIP  {filename} not found — download it first")
            continue
        stat = filepath.stat()
        digest = hash_dataset(filepath)
        registry[filename] = {
            "name": name,
            "sha256": digest,
            "bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        }
        print(f"  OK    {filename}  {digest[:12]}...  ({stat.st_size:,} bytes)")

    hashes_path = data_dir / "dataset_hashes.json"
    hashes_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote {len(registry)} hash(es) to {hashes_path}")
    return registry


def load_dataset_hashes(data_dir: Path = DATA_DIR) -> dict[str, dict]:
    """Load the dataset hash registry.

    Args:
        data_dir: Directory containing ``dataset_hashes.json``. Defaults to the
            project ``DATA_DIR``.

    Returns:
        The registry dict, or an empty dict if the file does not exist yet.
    """
    hashes_path = data_dir / "dataset_hashes.json"
    if not hashes_path.exists():
        return {}
    return json.loads(hashes_path.read_text())


if __name__ == "__main__":
    write_dataset_hashes()
