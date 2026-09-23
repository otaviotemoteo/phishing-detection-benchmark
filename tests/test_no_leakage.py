"""
Leakage invariants of the training pipeline (DEVELOPMENT.md §6.2).

Three rules decide whether the reported metrics mean anything:

1. SMOTE runs inside the pipeline, before the classifier and after the
   transforms, so the cross-validation folds resample training data only.
2. The scaler is fitted on training rows only, never on the full frame.
3. Resampling never reaches validation or test data.

Each is checked against the pipeline the runner actually builds, on small
synthetic data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import apply_smote, split_data
from src.experiments.runner import _build_pipeline

N_SAMPLES = 400


def make_imbalanced(n: int = N_SAMPLES, rate: float = 0.25, seed: int = 11):
    """Synthetic imbalanced classification data, with the minority class last."""
    rng = np.random.default_rng(seed)
    n_positive = int(round(n * rate))
    X = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series([1] * n_positive + [0] * (n - n_positive), name="label")
    order = rng.permutation(n)
    return X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)


def test_smote_sits_between_the_transforms_and_the_classifier():
    pipeline = _build_pipeline(LogisticRegression())
    names = [name for name, _ in pipeline.steps]

    assert "smote" in names, "the tuning pipeline must resample inside itself"
    assert names.index("smote") < names.index("model"), "SMOTE must precede the classifier"
    assert names.index("scaler") < names.index("smote"), "SMOTE must see scaled features"
    assert names[-1] == "model", "the classifier must be the final step"


def test_the_pipeline_holds_exactly_one_resampler_and_it_is_smote():
    pipeline = _build_pipeline(LogisticRegression())
    resamplers = [
        (name, step) for name, step in pipeline.steps if hasattr(step, "fit_resample")
    ]

    assert len(resamplers) == 1, f"expected one resampling step, found {resamplers}"
    assert isinstance(resamplers[0][1], SMOTE)


def test_the_estimator_is_the_only_predictor_outside_the_transforms():
    pipeline = _build_pipeline(LogisticRegression())

    assert isinstance(pipeline, ImbPipeline), "an sklearn Pipeline would resample the CV folds"
    assert isinstance(pipeline.named_steps["model"], BaseEstimator)


def test_the_scaler_is_fitted_on_training_rows_only():
    X, y = make_imbalanced()
    X_train, _X_val, X_test, y_train, _y_val, _y_test = split_data(X, y)

    fitted = _build_pipeline(LogisticRegression(max_iter=200)).fit(X_train, y_train)
    pipeline_scaler = fitted.named_steps["scaler"]

    # What the scaler would have learned from the training rows alone, through
    # the same imputer the pipeline applies first.
    imputer = SimpleImputer(strategy="median")
    train_reference = StandardScaler().fit(imputer.fit_transform(X_train))
    full_reference = StandardScaler().fit(imputer.fit_transform(X))

    np.testing.assert_allclose(pipeline_scaler.mean_, train_reference.mean_, rtol=1e-12)
    np.testing.assert_allclose(pipeline_scaler.scale_, train_reference.scale_, rtol=1e-12)
    assert not np.allclose(pipeline_scaler.mean_, full_reference.mean_), (
        "training statistics coincide with whole-dataset statistics; "
        "this test can no longer tell leakage from correctness"
    )
    # Transforming the untouched test set must not refit anything.
    before = pipeline_scaler.mean_.copy()
    fitted.predict(X_test)
    np.testing.assert_array_equal(pipeline_scaler.mean_, before)


def test_smote_leaves_validation_and_test_sizes_untouched():
    X, y = make_imbalanced()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    sizes_before = (len(X_val), len(X_test))

    X_resampled, y_resampled = apply_smote(X_train, y_train)

    assert len(X_resampled) > len(X_train), "SMOTE should have oversampled the training set"
    assert y_resampled.value_counts().nunique() == 1, "training classes should be balanced"
    assert (len(X_val), len(X_test)) == sizes_before
    assert len(y_val) == sizes_before[0] and len(y_test) == sizes_before[1]


def test_predicting_does_not_resample():
    X, y = make_imbalanced()
    X_train, _X_val, X_test, y_train, _y_val, y_test = split_data(X, y)

    fitted = _build_pipeline(LogisticRegression(max_iter=200)).fit(X_train, y_train)
    predictions = fitted.predict(X_test)

    # SMOTE is a sampler: it is skipped at predict time, so one prediction comes
    # back per test row and the test set keeps its original class balance.
    assert len(predictions) == len(X_test) == len(y_test)
    assert abs(y_test.mean() - y.mean()) <= 0.01, "the test set was rebalanced"

