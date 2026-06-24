"""
Preprocessing: label normalization, stratified splitting, SMOTE, scaling.

**Order matters for validity** (DEVELOPMENT.md §6.2). The pipeline must always:
split first → fit the scaler on the training set only → SMOTE the training set
only. Validation and test sets are never resampled and are transformed with the
train-fitted scaler. This module provides the building blocks; the orchestration
order lives in `src.experiments.runner`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    RANDOM_SEED,
    SMOTE_K_NEIGHBORS,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

# The configured split ratios must form a valid partition.
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "split ratios must sum to 1"

# Canonical label convention (D-004): 1 = phishing (positive), 0 = legitimate.
PHISHING: int = 1
LEGITIMATE: int = 0


def to_xy(df: pd.DataFrame, dataset: str) -> tuple[pd.DataFrame, pd.Series]:
    """Extract features ``X`` and a binary label ``y`` (1=phishing, 0=legit).

    Args:
        df: Raw dataframe from `src.data.loaders.load_raw`.
        dataset: ``'uci'`` | ``'mendeley'`` | ``'iscx'``.

    Returns:
        ``(X, y)`` with ``y`` in the D-004 convention (1 = phishing, 0 = legit):
        - ``uci``: ``X`` is the 30 structured features; ``Result`` remapped.
        - ``mendeley``: ``X`` is lexical URL features (Planejamento §4.4); ``y`` is
          ``result`` (already 0/1).
        - ``iscx``: binarized **phishing vs benign** (D-006) — other attack classes
          dropped; ``X`` is the 79 lexical features (NaNs left for the pipeline imputer).

    Raises:
        KeyError: For an unknown dataset name.
    """
    if dataset == "uci":
        assert "Result" in df.columns, "UCI must have a 'Result' column"
        y = df["Result"].map({-1: PHISHING, 1: LEGITIMATE})
        assert y.notna().all(), "Unexpected UCI 'Result' values (expected only -1/1)"
        X = df.drop(columns=["Result"])
        return X, y.astype(int)

    if dataset == "mendeley":
        from src.data.feature_engineering import extract_url_features

        assert {"url", "result"}.issubset(df.columns), "Mendeley needs 'url' and 'result'"
        X = extract_url_features(df["url"])
        y = df["result"].astype(int)
        assert set(y.unique()).issubset({LEGITIMATE, PHISHING}), "Mendeley result must be 0/1"
        return X, y

    if dataset == "iscx":
        label = "URL_Type_obf_Type"
        assert label in df.columns, f"ISCX needs the '{label}' column"
        # D-006: binary phishing-vs-benign; drop Defacement/malware/spam.
        sub = df[df[label].isin(["phishing", "benign"])].copy()
        y = (sub[label] == "phishing").astype(int)
        X = sub.drop(columns=[label]).apply(pd.to_numeric, errors="coerce")
        # ISCX has inf in some ratio columns (÷0); the imputer only fills NaN.
        X = X.replace([np.inf, -np.inf], np.nan)
        return X.reset_index(drop=True), y.reset_index(drop=True)

    raise KeyError(f"Unknown dataset '{dataset}'")


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train/val/test split at ``TRAIN/VAL/TEST_RATIO`` (Planejamento §4.3).

    Performed as two successive stratified splits. Fails loud (§1.5) if the splits
    overlap, do not sum to the input size, or do not preserve class proportions.

    Returns:
        ``X_train, X_val, X_test, y_train, y_val, y_test``.
    """
    # 1) Carve off the training set; `temp` holds val + test.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=y,
        random_state=RANDOM_SEED,
    )
    # 2) Split `temp` into val/test, keeping the configured 15/15 proportion.
    rel_test = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=rel_test,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    # Fail loud: sizes sum, splits are disjoint, class balance preserved.
    assert len(X_train) + len(X_val) + len(X_test) == len(X)
    idx_train, idx_val, idx_test = set(X_train.index), set(X_val.index), set(X_test.index)
    assert idx_train.isdisjoint(idx_val), "train/val overlap"
    assert idx_train.isdisjoint(idx_test), "train/test overlap"
    assert idx_val.isdisjoint(idx_test), "val/test overlap"
    overall = y.mean()
    for name, part in (("train", y_train), ("val", y_val), ("test", y_test)):
        assert abs(part.mean() - overall) < 0.02, f"{name} class proportion drifted"

    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train):
    """Oversample the minority (phishing) class on the **training set only**.

    Reference: Chawla et al. (2002). Never call this on validation or test data
    (DEVELOPMENT.md §6.2).

    Args:
        X_train: Training features (array-like).
        y_train: Training labels.

    Returns:
        ``(X_resampled, y_resampled)`` with the classes balanced.
    """
    smote = SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_SEED)
    return smote.fit_resample(X_train, y_train)


def fit_scaler(X_train) -> StandardScaler:
    """Fit a ``StandardScaler`` on the **training features only** (§6.2)."""
    return StandardScaler().fit(X_train)


def transform_features(scaler: StandardScaler, X):
    """Apply an already-fitted scaler. Returns a NumPy array."""
    return scaler.transform(X)


def stratified_subsample(X, y, n: int, seed: int = RANDOM_SEED):
    """Return a stratified subsample of about ``n`` rows (no-op if already ≤ n).

    Used to keep RBF-SVM training feasible on the large datasets (D-006); the class
    balance is preserved so the subsample stays representative.
    """
    if len(X) <= n:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=n, stratify=y, random_state=seed
    )
    return X_sub, y_sub
