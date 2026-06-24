"""
Classical ML model factory (DEVELOPMENT.md §5, Planejamento §5.1).

`get_model(name)` returns a fresh estimator, a hyperparameter search space, and a
flag indicating whether the model needs a training subsample on large datasets
(SVM only — D-006). Param-grid keys are **bare**; the runner prefixes them with
``model__`` for the imblearn pipeline.
"""
from __future__ import annotations

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.config import RANDOM_SEED

# Short key -> human-readable name used in metrics_ml.csv and artifact filenames.
MODEL_DISPLAY: dict[str, str] = {
    "dt": "DecisionTree",
    "rf": "RandomForest",
    "xgb": "XGBoost",
    "cat": "CatBoost",
    "lr": "LogisticRegression",
    "svm": "SVM",
}
MODEL_NAMES: list[str] = list(MODEL_DISPLAY)


def get_model(name: str) -> tuple[object, dict, bool]:
    """Return ``(estimator, param_grid, needs_subsample)`` for a model key.

    Args:
        name: One of ``dt, rf, xgb, cat, lr, svm``.

    Raises:
        KeyError: For an unknown model key.
    """
    key = name.lower()

    if key == "dt":
        est = DecisionTreeClassifier(random_state=RANDOM_SEED)
        grid = {
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"],
        }
        return est, grid, False

    if key == "rf":
        est = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
        grid = {
            "n_estimators": [100, 200, 400],
            "max_depth": [None, 10, 20, 30],
            "max_features": ["sqrt", "log2", None],
        }
        return est, grid, False

    if key == "xgb":
        est = XGBClassifier(
            random_state=RANDOM_SEED, tree_method="hist", eval_metric="logloss", n_jobs=-1
        )
        grid = {
            "n_estimators": [200, 400, 600],
            "learning_rate": [0.03, 0.1, 0.3],
            "max_depth": [3, 6, 9],
            "subsample": [0.7, 1.0],
        }
        return est, grid, False

    if key == "cat":
        # allow_writing_files=False stops CatBoost from littering a catboost_info/ dir.
        est = CatBoostClassifier(random_state=RANDOM_SEED, verbose=0, allow_writing_files=False)
        grid = {
            "depth": [4, 6, 8],
            "iterations": [200, 400, 600],
            "learning_rate": [0.03, 0.1, 0.3],
        }
        return est, grid, False

    if key == "lr":
        est = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        grid = {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        }
        return est, grid, False

    if key == "svm":
        est = SVC(probability=True, random_state=RANDOM_SEED)
        grid = {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf"],
        }
        return est, grid, True  # needs_subsample on large datasets (D-006)

    raise KeyError(f"Unknown model '{name}'. Known: {MODEL_NAMES}")
