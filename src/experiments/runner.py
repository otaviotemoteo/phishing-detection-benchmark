"""
End-to-end experiment runner (Phase 2+).

`train_and_evaluate` ties the whole pipeline together for one (model, dataset)
pair and persists every standard artifact. Per DEVELOPMENT.md §1.2, swapping the
model is a one-line change — the same function drives Logistic Regression today
and the full classical/DL line-up in later phases.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib

from src.config import (
    CONFUSION_MATRICES_DIR,
    METRICS_ML_CSV,
    MODELS_DIR,
    RANDOM_SEED,
    REPO_ROOT,
    ROC_CURVES_DIR,
)
from src.data.loaders import load_raw
from src.data.preprocessing import (
    apply_smote,
    fit_scaler,
    split_data,
    to_xy,
    transform_features,
)
from src.evaluation.cost import measure_inference_time, CostTracker
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import save_confusion_matrix, save_roc_curve
from src.utils.io import append_metrics_row
from src.utils.manifests import save_manifest
from src.utils.seeds import set_all_seeds

_MLFLOW_EXPERIMENT = "phishing-detection"


def _json_safe(params: dict) -> dict:
    """Coerce a hyperparameter dict to JSON-serializable values."""
    safe = {}
    for key, value in params.items():
        try:
            json.dumps(value)
            safe[key] = value
        except TypeError:
            safe[key] = str(value)
    return safe


def _log_mlflow(experiment_id, model_name, dataset, hyperparameters, metrics, cost,
                artifact_paths) -> None:
    """Best-effort MLflow logging to the local store (D-002). Never fatal."""
    try:
        import mlflow

        mlflow.set_tracking_uri(f"file:{REPO_ROOT / 'mlruns'}")
        mlflow.set_experiment(_MLFLOW_EXPERIMENT)
        with mlflow.start_run(run_name=experiment_id):
            mlflow.log_param("model", model_name)
            mlflow.log_param("dataset", dataset)
            for key, value in hyperparameters.items():
                mlflow.log_param(f"hp_{key}", value)
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            for key, value in cost.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"cost_{key}", value)
            for artifact in artifact_paths:
                if Path(artifact).exists():
                    mlflow.log_artifact(artifact)
    except Exception as exc:  # pragma: no cover - logging must never break a run
        print(f"  (mlflow logging skipped: {exc})")


def train_and_evaluate(
    model,
    dataset: str,
    model_name: str | None = None,
    *,
    log_mlflow: bool = True,
) -> dict:
    """Train ``model`` on ``dataset``, evaluate on a held-out test set, persist all artifacts.

    Pipeline order avoids data leakage (DEVELOPMENT.md §6.2): load -> to_xy ->
    stratified split -> fit scaler on **train** -> scale all splits -> SMOTE the
    **train** set only -> fit -> evaluate on the untouched **test** set. Scaling
    precedes SMOTE so the synthetic-sample k-NN runs on comparably-scaled features.

    Artifacts written: model+scaler (joblib), confusion matrix, ROC curve, JSON
    manifest, and a row appended to ``metrics_ml.csv``.

    Args:
        model: Any sklearn-compatible estimator (fit/predict/predict_proba).
        dataset: ``'uci'`` (mendeley/iscx pending Phase 3/4 feature engineering).
        model_name: Label for artifact names; defaults to the estimator class name.
        log_mlflow: If True, also log to the local MLflow store (best-effort).

    Returns:
        The metrics dict from `compute_metrics`.
    """
    set_all_seeds(RANDOM_SEED)
    model_name = model_name or type(model).__name__
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{model_name}_{dataset}_{timestamp}"

    # Load and prepare.
    X, y = to_xy(load_raw(dataset), dataset)
    X_train, _X_val, X_test, y_train, _y_val, y_test = split_data(X, y)

    # Fit scaler on train only, scale all splits, then SMOTE the scaled train set.
    scaler = fit_scaler(X_train)
    X_train_s = transform_features(scaler, X_train)
    X_test_s = transform_features(scaler, X_test)
    X_train_res, y_train_res = apply_smote(X_train_s, y_train)

    # Train with cost tracking.
    with CostTracker() as tracker:
        model.fit(X_train_res, y_train_res)
        tracker.update_ram()

    # Evaluate on the untouched test set.
    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    cost = {
        "training_time_s": round(tracker.elapsed_s, 4),
        "inference_time_ms_per_sample": round(measure_inference_time(model, X_test_s), 6),
        "peak_ram_mb": round(tracker.peak_ram_mb, 1),
        "gpu_used": False,
        "n_parameters": None,
    }

    # Persist artifacts.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{experiment_id}.joblib"
    joblib.dump({"model": model, "scaler": scaler}, model_path)
    cm_path = save_confusion_matrix(
        y_test, y_pred, CONFUSION_MATRICES_DIR / f"{model_name}_{dataset}.png"
    )
    roc_path = save_roc_curve(
        y_test, y_proba, ROC_CURVES_DIR / f"{model_name}_{dataset}.png", name=model_name
    )

    artifacts = {
        "model_path": str(model_path),
        "confusion_matrix": str(cm_path),
        "roc_curve": str(roc_path),
    }
    hyperparameters = model.get_params() if hasattr(model, "get_params") else {}
    manifest_path = save_manifest(
        experiment_id,
        model_name,
        dataset,
        _json_safe(hyperparameters),
        metrics,
        cost,
        artifacts,
        RANDOM_SEED,
    )

    # Standardized comparison row (§8.2).
    append_metrics_row(
        METRICS_ML_CSV,
        {
            "model": model_name,
            "dataset": dataset,
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "auc_roc": round(metrics["auc_roc"], 4),
            "train_time_s": cost["training_time_s"],
            "inference_time_ms_per_sample": cost["inference_time_ms_per_sample"],
        },
    )

    if log_mlflow:
        _log_mlflow(
            experiment_id,
            model_name,
            dataset,
            _json_safe(hyperparameters),
            metrics,
            cost,
            [str(manifest_path), str(cm_path), str(roc_path)],
        )

    print(
        f"[{experiment_id}] acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
        f"recall={metrics['recall']:.4f} auc={metrics['auc_roc']:.4f}"
    )
    print(f"  manifest -> {manifest_path.relative_to(REPO_ROOT)}")
    return metrics
