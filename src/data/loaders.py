"""
Dataset loaders.

Thin readers that return the raw, as-published dataframe for a dataset. No
cleaning, splitting, or label normalization happens here — that is a Phase 2
preprocessing concern. EDA (Phase 1) deliberately inspects the data exactly as
it was downloaded.

The dataset short names and on-disk filenames are defined once in
``src.utils.io`` and reused here to avoid drift.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_DIR
from src.utils.io import DATASET_FILENAMES

# Short name -> absolute CSV path.
DATASET_FILES: dict[str, Path] = {
    name: DATA_DIR / filename for name, filename in DATASET_FILENAMES.items()
}


def dataset_path(name: str) -> Path:
    """Return the on-disk CSV path for a dataset short name.

    Args:
        name: One of ``'uci'``, ``'mendeley'``, ``'iscx'``.

    Returns:
        The absolute path where the dataset CSV is expected.

    Raises:
        KeyError: If ``name`` is not a known dataset.
    """
    if name not in DATASET_FILES:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(DATASET_FILES)}")
    return DATASET_FILES[name]


def is_available(name: str) -> bool:
    """Return True if the dataset CSV exists on disk."""
    return dataset_path(name).exists()


def available_datasets() -> list[str]:
    """Return the short names of datasets currently present in ``data/``."""
    return [name for name in DATASET_FILES if is_available(name)]


def load_raw(name: str) -> pd.DataFrame:
    """Load a dataset's raw CSV exactly as published.

    Args:
        name: Short dataset name — ``'uci'``, ``'mendeley'``, or ``'iscx'``.

    Returns:
        The dataset as a DataFrame, unmodified.

    Raises:
        KeyError: If ``name`` is not a known dataset.
        FileNotFoundError: If the CSV has not been downloaded yet.
    """
    path = dataset_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {path}. "
            f"Run `bash scripts/download_datasets.sh` first."
        )
    df = pd.read_csv(path)
    assert len(df) > 0, f"Loaded an empty dataframe for '{name}' from {path}"
    return df
