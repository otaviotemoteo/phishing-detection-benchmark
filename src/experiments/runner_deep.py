"""
Deep Learning experiment runner (Phase 4).

Mirrors `src.experiments.runner` but for PyTorch char-level models: tokenize raw
URLs, train with early stopping + mixed precision on the GPU, and persist the same
standard artifacts (metrics_dl.csv row, manifest, confusion matrix, ROC, training
curve). Reuses the entire evaluation/saving half — only the training loop is new.
"""
from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    CONFUSION_MATRICES_DIR,
    DL_BATCH_SIZE,
    DL_DROPOUT,
    DL_EMBED_DIM,
    DL_LEARNING_RATE,
    DL_MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    MAX_URL_LENGTH,
    METRICS_DL_CSV,
    MODELS_DIR,
    RANDOM_SEED,
    REPO_ROOT,
    ROC_CURVES_DIR,
    TRAINING_CURVES_DIR,
)
from src.data.feature_engineering import build_char_vocab, encode_urls, vocab_size
from src.data.loaders import load_raw
from src.data.preprocessing import split_data
from src.evaluation.cost import CostTracker, count_parameters, gpu_used_mb
from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import save_confusion_matrix, save_roc_curve, save_training_curve
from src.experiments.runner import _log_mlflow
from src.models.deep import DEEP_MODEL_DISPLAY, get_deep_model
from src.utils.io import upsert_metrics_row
from src.utils.manifests import save_manifest
from src.utils.seeds import set_all_seeds

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_USE_AMP = _DEVICE.type == "cuda"


def _loader(X: np.ndarray, y, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.tensor(y.values, dtype=torch.float32))
    return DataLoader(ds, batch_size=DL_BATCH_SIZE, shuffle=shuffle)


def _run_epoch(model, loader, criterion, optimizer=None, scaler=None) -> float:
    """Run one epoch; trains if `optimizer` is given, else evaluates. Returns mean loss."""
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(_DEVICE), yb.to(_DEVICE)
            if training:
                optimizer.zero_grad()
            with torch.autocast(device_type=_DEVICE.type, enabled=_USE_AMP):
                loss = criterion(model(xb), yb)
            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total += loss.item() * len(yb)
            n += len(yb)
    return total / n


@torch.no_grad()
def _predict_proba(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=DL_BATCH_SIZE)
    probs = []
    for (xb,) in loader:
        with torch.autocast(device_type=_DEVICE.type, enabled=_USE_AMP):
            logits = model(xb.to(_DEVICE))
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs)


@torch.no_grad()
def _measure_inference(model, X: np.ndarray, n_warmup: int = 3, n_max: int = 1000) -> float:
    """Average inference time (ms/sample), batched, with warmup."""
    model.eval()
    n = min(n_max, len(X))
    loader = DataLoader(TensorDataset(torch.from_numpy(X[:n])), batch_size=DL_BATCH_SIZE)
    for _ in range(n_warmup):
        for (xb,) in loader:
            with torch.autocast(device_type=_DEVICE.type, enabled=_USE_AMP):
                model(xb.to(_DEVICE))
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for (xb,) in loader:
        with torch.autocast(device_type=_DEVICE.type, enabled=_USE_AMP):
            model(xb.to(_DEVICE))
    if _DEVICE.type == "cuda":
        torch.cuda.synchronize()
    return ((time.perf_counter() - start) / n) * 1000.0


def _train_loop(model, train_loader, val_loader, criterion, optimizer, scaler, tracker=None):
    """Train with early stopping on val loss; restore the best weights.

    Returns:
        ``(history, epochs_trained)`` where history has ``train_loss``/``val_loss`` lists.
    """
    history = {"train_loss": [], "val_loss": []}
    best_val, best_state, patience, epochs_trained = float("inf"), None, 0, 0
    for epoch in range(1, DL_MAX_EPOCHS + 1):
        tr_loss = _run_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss = _run_epoch(model, val_loader, criterion)
        history["train_loss"].append(round(tr_loss, 5))
        history["val_loss"].append(round(val_loss, 5))
        if tracker is not None:
            tracker.update_ram()
        epochs_trained = epoch
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history, epochs_trained


def train_and_evaluate_deep(
    model_key: str,
    dataset: str = "mendeley",
    *,
    subset_n: int | None = None,
    log_mlflow: bool = True,
) -> dict:
    """Train a char-level DL model on raw URLs and persist all standard artifacts.

    Args:
        model_key: ``'cnn'`` | ``'lstm'`` | ``'cnnlstm'``.
        dataset: Must expose a raw ``url`` column (only Mendeley today — D-008).
        subset_n: If set, use only the first N rows (pipeline smoke test).
        log_mlflow: Best-effort MLflow logging.

    Returns:
        The test-set metrics dict.
    """
    set_all_seeds(RANDOM_SEED)
    display = DEEP_MODEL_DISPLAY[model_key]
    experiment_id = f"{display}_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    df = load_raw(dataset)
    assert "url" in df.columns, f"DL needs raw URLs; dataset '{dataset}' has none (D-008)"
    urls = df["url"].astype(str)
    y = df["result"].astype(int)
    if subset_n:
        urls, y = urls.iloc[:subset_n], y.iloc[:subset_n]

    # Same split routine as classical → comparable test set.
    urls_tr, urls_val, urls_te, y_tr, y_val, y_te = split_data(urls, y)

    # Tokenize: vocab fit on TRAIN only (D-007).
    vocab = build_char_vocab(urls_tr)
    n_vocab = vocab_size(vocab)
    X_tr = encode_urls(urls_tr, vocab, MAX_URL_LENGTH)
    X_val = encode_urls(urls_val, vocab, MAX_URL_LENGTH)
    X_te = encode_urls(urls_te, vocab, MAX_URL_LENGTH)

    train_loader = _loader(X_tr, y_tr, shuffle=True)
    val_loader = _loader(X_val, y_val, shuffle=False)

    model = get_deep_model(model_key, n_vocab).to(_DEVICE)
    # Handle imbalance via pos_weight, not SMOTE (D-008).
    pos_weight = torch.tensor(
        [(y_tr == 0).sum() / max(int((y_tr == 1).sum()), 1)], dtype=torch.float32, device=_DEVICE
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=DL_LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda") if _USE_AMP else None

    with CostTracker() as tracker:
        history, epochs_trained = _train_loop(
            model, train_loader, val_loader, criterion, optimizer, scaler, tracker
        )

    gpu_mb = gpu_used_mb()

    # Evaluate on the untouched test set.
    y_proba = _predict_proba(model, X_te)
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = compute_metrics(y_te, y_pred, y_proba)

    cost = {
        "training_time_s": round(tracker.elapsed_s, 4),
        "inference_time_ms_per_sample": round(_measure_inference(model, X_te), 6),
        "peak_ram_mb": round(tracker.peak_ram_mb, 1),
        "gpu_used": _USE_AMP,
        "gpu_mem_mb": round(gpu_mb, 1) if gpu_mb else None,
        "n_parameters": count_parameters(model),
        "epochs_trained": epochs_trained,
    }

    # Persist artifacts.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{experiment_id}.pt"
    torch.save(model.state_dict(), model_path)
    cm_path = save_confusion_matrix(y_te, y_pred, CONFUSION_MATRICES_DIR / f"{display}_{dataset}.png")
    roc_path = save_roc_curve(y_te, y_proba, ROC_CURVES_DIR / f"{display}_{dataset}.png", name=display)
    tc_path = save_training_curve(history, TRAINING_CURVES_DIR / f"{display}_{dataset}.png")

    artifacts = {
        "model_path": str(model_path),
        "confusion_matrix": str(cm_path),
        "roc_curve": str(roc_path),
        "training_curve": str(tc_path),
    }
    hyperparameters = {
        "architecture": display,
        "vocab_size": n_vocab,
        "max_url_length": MAX_URL_LENGTH,
        "embed_dim": DL_EMBED_DIM,
        "dropout": DL_DROPOUT,
        "batch_size": DL_BATCH_SIZE,
        "learning_rate": DL_LEARNING_RATE,
        "max_epochs": DL_MAX_EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "pos_weight": round(float(pos_weight.item()), 4),
    }
    manifest_path = save_manifest(
        experiment_id, display, dataset, hyperparameters, metrics, cost, artifacts, RANDOM_SEED
    )

    upsert_metrics_row(
        METRICS_DL_CSV,
        {
            "model": display,
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
        artifact_paths = [str(manifest_path), str(cm_path), str(roc_path), str(tc_path)]
        _log_mlflow(experiment_id, display, dataset, hyperparameters, metrics, cost, artifact_paths)

    print(
        f"[{experiment_id}] acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} "
        f"recall={metrics['recall']:.4f} auc={metrics['auc_roc']:.4f} "
        f"epochs={epochs_trained} time={cost['training_time_s']:.0f}s"
    )
    print(f"  manifest -> {manifest_path.relative_to(REPO_ROOT)}")
    return metrics
