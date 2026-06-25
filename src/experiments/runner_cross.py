"""
Cross-dataset generalization runner (Phase 6).

Trains on one raw-URL dataset and evaluates on a *different* one, to test whether
models learn general phishing patterns or memorize a dataset (Planejamento §10).
Both datasets are reduced to a shared representation derived from the raw URL:
- **classical:** 13 lexical URL features (`extract_url_features`)
- **deep:** char sequences fed to CNN-LSTM

**URL normalization (D-010):** the scheme (`http(s)://`) is stripped before feature
extraction / tokenization. Without this, cross-dataset transfer mostly measures a
formatting artifact (Mendeley URLs carry the scheme ~100% of the time; the Kaggle
set ~11%), not phishing patterns.

Each run records two rows in `metrics_crossdataset.csv`: a **within** baseline
(train A → A's held-out test) and the **cross** result (train A → all of B), both
on normalized URLs and the same trained model, so the F1 drop is apples-to-apples.
"""
from __future__ import annotations

from datetime import datetime

import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config import (
    CONFUSION_MATRICES_DIR,
    CROSS_MAX_TRAIN_SAMPLES,
    CV_FOLDS,
    DL_LEARNING_RATE,
    MAX_URL_LENGTH,
    METRICS_CROSSDATASET_CSV,
    MODELS_DIR,
    RANDOM_SEED,
    RANDOMIZED_SEARCH_N_ITER,
    ROC_CURVES_DIR,
    SEARCH_SCORING,
)
from src.data.feature_engineering import (
    build_char_vocab,
    encode_urls,
    extract_url_features,
    vocab_size,
)
from src.data.loaders import load_raw
from src.data.preprocessing import split_data, stratified_subsample
from src.evaluation.cost import (
    CostTracker,
    count_parameters,
    gpu_used_mb,
    measure_inference_time,
)
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import save_confusion_matrix, save_roc_curve
from src.experiments.runner import _build_pipeline, _grid_size, _json_safe, _log_mlflow
from src.experiments.runner_deep import (
    _DEVICE,
    _USE_AMP,
    _loader,
    _measure_inference,
    _predict_proba,
    _train_loop,
)
from src.models.classical import MODEL_DISPLAY, get_model
from src.models.deep import DEEP_MODEL_DISPLAY, get_deep_model
from src.utils.io import upsert_metrics_row
from src.utils.manifests import save_manifest
from src.utils.seeds import set_all_seeds


def _normalize_urls(urls):
    """Strip the URL scheme so lexical features align across datasets (D-010)."""
    return urls.astype(str).str.replace(r"^https?://", "", regex=True)


def _save_and_record(experiment_id, display, train_ds, test_ds, y_true, y_pred, y_proba,
                     cost, hyperparameters, model_path=None, log_mlflow=False):
    """Save confusion matrix + ROC + manifest and upsert one metrics row."""
    metrics = compute_metrics(y_true, y_pred, y_proba)
    tag = f"{display}_{train_ds}_to_{test_ds}"
    cm = save_confusion_matrix(y_true, y_pred, CONFUSION_MATRICES_DIR / f"{tag}.png")
    roc = save_roc_curve(y_true, y_proba, ROC_CURVES_DIR / f"{tag}.png", name=display)
    artifacts = {"confusion_matrix": str(cm), "roc_curve": str(roc)}
    if model_path is not None:
        artifacts["model_path"] = str(model_path)
    manifest = save_manifest(
        experiment_id, display, f"{train_ds}->{test_ds}", hyperparameters, metrics, cost,
        artifacts, RANDOM_SEED,
    )
    upsert_metrics_row(
        METRICS_CROSSDATASET_CSV,
        {
            "model": display,
            "train_dataset": train_ds,
            "test_dataset": test_ds,
            "accuracy": round(metrics["accuracy"], 4),
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "auc_roc": round(metrics["auc_roc"], 4),
            "train_time_s": cost["training_time_s"],
            "inference_time_ms_per_sample": cost["inference_time_ms_per_sample"],
        },
        keys=("model", "train_dataset", "test_dataset"),
    )
    kind = "within" if train_ds == test_ds else "cross "
    print(f"  [{kind}] {train_ds}->{test_ds}: f1={metrics['f1']:.4f} "
          f"auc={metrics['auc_roc']:.4f} recall={metrics['recall']:.4f}")
    if log_mlflow:
        _log_mlflow(experiment_id, display, f"{train_ds}->{test_ds}", hyperparameters, metrics,
                    cost, [str(manifest), str(cm), str(roc)])
    return metrics


def cross_classical(model_key: str, train_ds: str, test_ds: str, *, log_mlflow: bool = True) -> dict:
    """Train a classical model on A's URL features (tuned), eval within-A and cross-B."""
    set_all_seeds(RANDOM_SEED)
    display = MODEL_DISPLAY[model_key]
    estimator, grid, _ = get_model(model_key)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    dfa = load_raw(train_ds)
    Xa = extract_url_features(_normalize_urls(dfa["url"]))
    ya = dfa["result"].astype(int)
    Xa_tr, _Xv, Xa_te, ya_tr, _yv, ya_te = split_data(Xa, ya)
    Xa_tr, ya_tr = stratified_subsample(Xa_tr, ya_tr, CROSS_MAX_TRAIN_SAMPLES)

    dfb = load_raw(test_ds)
    Xb = extract_url_features(_normalize_urls(dfb["url"]))
    yb = dfb["result"].astype(int)

    search = RandomizedSearchCV(
        _build_pipeline(estimator),
        {f"model__{k}": v for k, v in grid.items()},
        n_iter=min(RANDOMIZED_SEARCH_N_ITER, _grid_size(grid)),
        cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        scoring=SEARCH_SCORING, n_jobs=-1, random_state=RANDOM_SEED, refit=True,
    )
    with CostTracker() as tracker:
        search.fit(Xa_tr, ya_tr)
        tracker.update_ram()
    fitted = search.best_estimator_
    hyper = _json_safe(fitted.named_steps["model"].get_params())

    def cost_for(X):
        return {
            "training_time_s": round(tracker.elapsed_s, 4),
            "inference_time_ms_per_sample": round(measure_inference_time(fitted, X), 6),
            "peak_ram_mb": round(tracker.peak_ram_mb, 1),
            "gpu_used": False, "n_parameters": None, "train_n": int(len(Xa_tr)),
        }

    # within baseline (A held-out)
    _save_and_record(f"{display}_{train_ds}_to_{train_ds}_{ts}", display, train_ds, train_ds,
                     ya_te, fitted.predict(Xa_te), fitted.predict_proba(Xa_te)[:, 1],
                     cost_for(Xa_te), hyper)
    # cross (all of B)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{display}_{train_ds}_to_{test_ds}_{ts}.joblib"
    joblib.dump(fitted, model_path)
    return _save_and_record(f"{display}_{train_ds}_to_{test_ds}_{ts}", display, train_ds, test_ds,
                            yb, fitted.predict(Xb), fitted.predict_proba(Xb)[:, 1],
                            cost_for(Xb), hyper, model_path, log_mlflow)


def cross_deep(model_key: str, train_ds: str, test_ds: str, *, log_mlflow: bool = True) -> dict:
    """Train a char-level model on A's URLs, eval within-A and cross-B."""
    set_all_seeds(RANDOM_SEED)
    display = DEEP_MODEL_DISPLAY[model_key]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    dfa = load_raw(train_ds)
    ua_tr, ua_val, ua_te, ya_tr, ya_val, ya_te = split_data(
        _normalize_urls(dfa["url"]), dfa["result"].astype(int)
    )
    ua_tr, ya_tr = stratified_subsample(ua_tr, ya_tr, CROSS_MAX_TRAIN_SAMPLES)

    dfb = load_raw(test_ds)
    urls_b, yb = _normalize_urls(dfb["url"]), dfb["result"].astype(int)

    vocab = build_char_vocab(ua_tr)
    n_vocab = vocab_size(vocab)
    X_tr = encode_urls(ua_tr, vocab, MAX_URL_LENGTH)
    X_val = encode_urls(ua_val, vocab, MAX_URL_LENGTH)
    X_ate = encode_urls(ua_te, vocab, MAX_URL_LENGTH)
    X_b = encode_urls(urls_b, vocab, MAX_URL_LENGTH)

    model = get_deep_model(model_key, n_vocab).to(_DEVICE)
    pos_weight = torch.tensor(
        [(ya_tr == 0).sum() / max(int((ya_tr == 1).sum()), 1)], dtype=torch.float32, device=_DEVICE
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda") if _USE_AMP else None
    with CostTracker() as tracker:
        _hist, epochs = _train_loop(
            model, _loader(X_tr, ya_tr, True), _loader(X_val, ya_val, False),
            criterion, optimizer, scaler, tracker,
        )
    gpu_mb = gpu_used_mb()
    hyper = {"architecture": display, "vocab_size": n_vocab, "max_url_length": MAX_URL_LENGTH}
    base = {
        "training_time_s": round(tracker.elapsed_s, 4),
        "peak_ram_mb": round(tracker.peak_ram_mb, 1),
        "gpu_used": _USE_AMP, "gpu_mem_mb": round(gpu_mb, 1) if gpu_mb else None,
        "n_parameters": count_parameters(model), "epochs_trained": epochs, "train_n": int(len(ua_tr)),
    }

    def cost_for(X):
        return {**base, "inference_time_ms_per_sample": round(_measure_inference(model, X), 6)}

    # within baseline (A held-out)
    pa = _predict_proba(model, X_ate)
    _save_and_record(f"{display}_{train_ds}_to_{train_ds}_{ts}", display, train_ds, train_ds,
                     ya_te, (pa >= 0.5).astype(int), pa, cost_for(X_ate), hyper)
    # cross (all of B)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{display}_{train_ds}_to_{test_ds}_{ts}.pt"
    torch.save(model.state_dict(), model_path)
    pb = _predict_proba(model, X_b)
    return _save_and_record(f"{display}_{train_ds}_to_{test_ds}_{ts}", display, train_ds, test_ds,
                            yb, (pb >= 0.5).astype(int), pb, cost_for(X_b), hyper, model_path, log_mlflow)
