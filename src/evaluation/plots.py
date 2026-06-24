"""
Per-model evaluation figures: confusion matrix and ROC curve.

Saved at publication DPI (`PLOT_DPI`) to the standard results directories. These
are the per-experiment diagnostics; the cross-model comparison figures come later
in `06_comparisons.ipynb` (Phase 7).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

from src.config import PLOT_DPI

# Display order matches the D-004 convention: index 0 = legitimate, 1 = phishing.
_CLASS_LABELS = ("legitimate", "phishing")


def save_confusion_matrix(y_true, y_pred, path) -> Path:
    """Plot and save a confusion matrix (rows = true, cols = predicted).

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        path: Destination PNG path.

    Returns:
        The path written.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=_CLASS_LABELS).plot(
        ax=ax, cmap="Blues", colorbar=False
    )
    ax.set_title("Confusion matrix")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def save_feature_importance(estimator, feature_names, path, top_n: int = 15):
    """Save a top-N feature-importance bar chart for tree-based models.

    Args:
        estimator: A fitted estimator; must expose ``feature_importances_`` or
            this is a no-op.
        feature_names: Column names aligned with the importance vector.
        path: Destination PNG path.
        top_n: Number of top features to show.

    Returns:
        The path written, or ``None`` if the estimator has no importances.
    """
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return None

    top = (
        pd.Series(importances, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top.index, top.values, color="#2a9d8f")
    ax.set_title(f"Top {top_n} feature importances")
    ax.set_xlabel("importance")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def save_roc_curve(y_true, y_proba, path, name: str = "model") -> Path:
    """Plot and save a ROC curve with the random-classifier diagonal.

    Args:
        y_true: Ground-truth binary labels.
        y_proba: Predicted probability of the phishing class.
        path: Destination PNG path.
        name: Legend label for the curve.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="random")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path
