"""
Standardized metric computation (DEVELOPMENT.md §8).

Every model on every dataset is scored through `compute_metrics`, guaranteeing the
same definitions and the same positive class (phishing = 1, per D-004) everywhere.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Positive class is phishing (D-004). Kept local to avoid importing the heavier
# preprocessing module just for a constant.
_POSITIVE_LABEL = 1


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict:
    """Compute the standard classification metrics.

    Precision, recall, and F1 are reported for the phishing class (label 1), so
    recall is the fraction of phishing correctly detected (DEVELOPMENT.md §8.3).

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_proba: Predicted probability of the phishing class.

    Returns:
        A dict with ``accuracy``, ``precision``, ``recall``, ``f1``, ``auc_roc``.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=_POSITIVE_LABEL, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=_POSITIVE_LABEL, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=_POSITIVE_LABEL, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
    }
