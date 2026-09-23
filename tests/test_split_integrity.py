"""
Invariants of the train/validation/test split (DEVELOPMENT.md §6.2, Planejamento §4.3).

The split is the foundation every reported number stands on: if the three sets
overlap, or if the class balance drifts between them, every metric downstream is
measuring something other than what it claims to. These tests run on small
synthetic frames, never on the real datasets, and never re-execute the pipeline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import TEST_RATIO, TRAIN_RATIO, VAL_RATIO
from src.data.preprocessing import split_data

N_SAMPLES = 1000
PHISHING_RATE = 0.35


def make_dataset(n: int = N_SAMPLES, rate: float = PHISHING_RATE, seed: int = 7):
    """Build a small synthetic (X, y) pair with a known class balance."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    n_positive = int(round(n * rate))
    y = pd.Series([1] * n_positive + [0] * (n - n_positive), name="label")
    # Shuffle so the label is not correlated with row position.
    order = rng.permutation(n)
    return X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)


def test_splits_are_disjoint():
    X, y = make_dataset()
    X_train, X_val, X_test, *_ = split_data(X, y)

    train, val, test = set(X_train.index), set(X_val.index), set(X_test.index)
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == len(X)


def test_split_proportions_match_the_configured_ratios():
    X, y = make_dataset()
    X_train, X_val, X_test, *_ = split_data(X, y)

    for part, ratio in ((X_train, TRAIN_RATIO), (X_val, VAL_RATIO), (X_test, TEST_RATIO)):
        expected = ratio * len(X)
        assert abs(len(part) - expected) <= 1, f"{len(part)} rows, expected about {expected}"


def test_stratification_preserves_the_class_balance():
    X, y = make_dataset()
    _, _, _, y_train, y_val, y_test = split_data(X, y)

    overall = y.mean()
    for name, part in (("train", y_train), ("val", y_val), ("test", y_test)):
        assert abs(part.mean() - overall) <= 0.01, f"{name} drifted to {part.mean():.4f}"


def test_split_is_identical_across_calls_with_the_same_seed():
    X, y = make_dataset()
    first = split_data(X, y)
    second = split_data(X, y)

    for a, b in zip(first, second):
        assert list(a.index) == list(b.index)


def test_labels_follow_their_rows_through_the_split():
    X, y = make_dataset()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    for X_part, y_part in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
        assert list(X_part.index) == list(y_part.index)
        assert (y_part == y.loc[y_part.index]).all()
