"""
End-to-end experiment runner (Phase 2+).

`train_and_evaluate` ties the whole pipeline together for one (model, dataset)
pair and persists every standard artifact. Per DEVELOPMENT.md §1.2, swapping the
model is a one-line change.

All transforms (impute → scale → SMOTE) live **inside** an `imblearn` Pipeline, so
when a hyperparameter search is requested they are applied per-CV-fold on training
data only — never leaking across folds or into the test set (§6.2). SMOTE is a
sampler, so it is automatically skipped at predict/transform time.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.config import (
    CONFUSION_MATRICES_DIR,
    CV_FOLDS,
    METRICS_ML_CSV,
    MODELS_DIR,
    PLOTS_FEATURE_IMPORTANCE_DIR,
    RANDOM_SEED,
    RANDOMIZED_SEARCH_N_ITER,
    REPO_ROOT,
    ROC_CURVES_DIR,
    SEARCH_SCORING,
    SMOTE_K_NEIGHBORS,
)
from src.data.loaders import load_raw
from src.data.preprocessing import split_data, stratified_subsample, to_xy
from src.evaluation.cost import CostTracker, measure_inference_time
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import (
    save_confusion_matrix,
    save_feature_importance,
    save_roc_curve,
)
from src.utils.io import upsert_metrics_row
from src.utils.manifests import save_manifest
from src.utils.seeds import set_all_seeds

_MLFLOW_EXPERIMENT = "phishing-detection"


def _json_safe(params: dict) -> dict:
    """Coerce a dict to JSON-serializable values."""
    safe = {}
    for key, value in params.items():
        try:
            json.dumps(value)
            safe[key] = value
        except TypeError:
            safe[key] = str(value)
    return safe


def _grid_size(grid: dict) -> int:
    size = 1
    for values in grid.values():
        size *= len(values)
    return size


def _build_pipeline(estimator) -> ImbPipeline:
    """Impute (median) → standardize → SMOTE (train only) → estimator."""
    return ImbPipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(k_neighbors=SMOTE_K_NEIGHBORS, random_state=RANDOM_SEED)),
            ("model", estimator),
        ]
    )


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
    param_grid: dict | None = None,
    max_train_samples: int | None = None,
    log_mlflow: bool = True,
) -> dict:
    """Train ``model`` on ``dataset``, evaluate on a held-out test set, persist artifacts.

    Pipeline (no leakage, §6.2): load → to_xy → stratified split → fit an
    impute→scale→SMOTE→model pipeline on **train** → evaluate on the untouched
    **test** set. With ``param_grid``, a `RandomizedSearchCV` (5-fold) tunes the
    pipeline on train; the validation split is reserved for Phase 4.

    Artifacts: fitted pipeline (joblib), confusion matrix, ROC curve, feature
    importance (tree models), JSON manifest, and a row upserted into ``metrics_ml.csv``.

    Args:
        model: Any sklearn-compatible estimator (fit/predict/predict_proba).
        dataset: ``'uci'`` | ``'mendeley'`` | ``'iscx'``.
        model_name: Label for artifacts; defaults to the estimator class name.
        param_grid: Bare-named hyperparameter grid; if given, runs RandomizedSearchCV.
        max_train_samples: If set, stratified-subsample the training set to this size
            (SVM on large datasets — D-006).
        log_mlflow: If True, also log to the local MLflow store (best-effort).

    Returns:
        The test-set metrics dict from `compute_metrics`.
    """
    set_all_seeds(RANDOM_SEED)
    model_name = model_name or type(model).__name__
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"{model_name}_{dataset}_{timestamp}"

    # Load and split.
    X, y = to_xy(load_raw(dataset), dataset)
    feature_names = list(X.columns)
    X_train, _X_val, X_test, y_train, _y_val, y_test = split_data(X, y)

    # Optional training subsample (SVM on large datasets).
    subsample_n = None
    if max_train_samples is not None and len(X_train) > max_train_samples:
        X_train, y_train = stratified_subsample(X_train, y_train, max_train_samples)
        subsample_n = len(X_train)

    pipeline = _build_pipeline(model)

    # Train (with optional hyperparameter search), tracking cost.
    best_params: dict = {}
    cv_score: float | None = None
    with CostTracker() as tracker:
        if param_grid:
            n_iter = min(RANDOMIZED_SEARCH_N_ITER, _grid_size(param_grid))
            search = RandomizedSearchCV(
                pipeline,
                {f"model__{k}": v for k, v in param_grid.items()},
                n_iter=n_iter,
                cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
                scoring=SEARCH_SCORING,
                n_jobs=-1,
                random_state=RANDOM_SEED,
                refit=True,
            )
            search.fit(X_train, y_train)
            fitted = search.best_estimator_
            best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
            cv_score = float(search.best_score_)
        else:
            fitted = pipeline.fit(X_train, y_train)
        tracker.update_ram()

    final_estimator = fitted.named_steps["model"]

    # Evaluate on the untouched test set.
    y_pred = fitted.predict(X_test)
    y_proba = fitted.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    cost = {
        "training_time_s": round(tracker.elapsed_s, 4),
        "inference_time_ms_per_sample": round(measure_inference_time(fitted, X_test), 6),
        "peak_ram_mb": round(tracker.peak_ram_mb, 1),
        "gpu_used": False,
        "n_parameters": None,
    }
    if subsample_n is not None:
        cost["train_subsample_n"] = subsample_n

    # Persist artifacts.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{experiment_id}.joblib"
    joblib.dump(fitted, model_path)
    cm_path = save_confusion_matrix(
        y_test, y_pred, CONFUSION_MATRICES_DIR / f"{model_name}_{dataset}.png"
    )
    roc_path = save_roc_curve(
        y_test, y_proba, ROC_CURVES_DIR / f"{model_name}_{dataset}.png", name=model_name
    )
    fi_path = save_feature_importance(
        final_estimator, feature_names, PLOTS_FEATURE_IMPORTANCE_DIR / f"{model_name}_{dataset}.png"
    )

    artifacts = {
        "model_path": str(model_path),
        "confusion_matrix": str(cm_path),
        "roc_curve": str(roc_path),
    }
    if fi_path is not None:
        artifacts["feature_importance"] = str(fi_path)

    manifest_metrics = dict(metrics)
    if cv_score is not None:
        manifest_metrics["cv_f1"] = round(cv_score, 4)
    hyperparameters = _json_safe({**final_estimator.get_params(), "_best_params": best_params})

    manifest_path = save_manifest(
        experiment_id, model_name, dataset, hyperparameters,
        manifest_metrics, cost, artifacts, RANDOM_SEED,
    )

    # Standardized comparison row (§8.2), upserted (one row per model+dataset).
    upsert_metrics_row(
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
        artifact_paths = [str(manifest_path), str(cm_path), str(roc_path)]
        if fi_path is not None:
            artifact_paths.append(str(fi_path))
        _log_mlflow(experiment_id, model_name, dataset, hyperparameters, metrics, cost, artifact_paths)

    cv_note = f" cv_f1={cv_score:.4f}" if cv_score is not None else ""
    print(
        f"[{experiment_id}] acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
        f"recall={metrics['recall']:.4f} auc={metrics['auc_roc']:.4f}{cv_note}"
    )
    return metrics
