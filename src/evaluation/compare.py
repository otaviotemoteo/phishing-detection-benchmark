"""
Final comparison figures (Phase 7, Planejamento §9.1).

Reads the metrics CSVs and reloads saved models to produce publication-ready
charts in ``plots/final/`` at 300 DPI. No retraining: predictions for the
ROC/confusion/feature-importance charts are recomputed by reloading the saved
model and reproducing the deterministic (seeded) test split.

Figures are designed to stay legible in grayscale (the dissertation print test):
distinct linestyles/markers on ROC, numeric annotations on the heatmap.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

from src.config import (
    METRICS_CROSSDATASET_CSV,
    METRICS_DL_CSV,
    METRICS_ML_CSV,
    MODELS_DIR,
    PLOT_DPI,
    PLOTS_FINAL_DIR,
)
from src.data.loaders import load_raw
from src.data.preprocessing import split_data, to_xy

DATASETS = ["uci", "mendeley", "iscx"]
_METRICS = ["accuracy", "precision", "recall", "f1"]
_LINESTYLES = ["-", "--", "-.", ":"]

_XY_CACHE: dict = {}
_PRED_CACHE: dict = {}


# ----------------------------------------------------------------------------- metrics
def load_all_metrics() -> pd.DataFrame:
    """Combine classical + deep metrics with a ``family`` column."""
    ml = pd.read_csv(METRICS_ML_CSV)
    ml["family"] = "classical"
    dl = pd.read_csv(METRICS_DL_CSV)
    dl["family"] = "deep"
    return pd.concat([ml, dl], ignore_index=True)


# ----------------------------------------------------------------------------- predictions
def _model_file(model_display: str, dataset: str) -> Path | None:
    """Newest saved within-dataset model for (model, dataset), or None."""
    candidates = [
        f for pat in (f"{model_display}_{dataset}_*.joblib", f"{model_display}_{dataset}_*.pt")
        for f in MODELS_DIR.glob(pat)
        if "_to_" not in f.name
    ]
    return max(candidates, key=lambda f: f.stat().st_mtime) if candidates else None


def _classical_test(dataset):
    if dataset not in _XY_CACHE:
        X, y = to_xy(load_raw(dataset), dataset)
        _, _, X_test, _, _, y_test = split_data(X, y)
        _XY_CACHE[dataset] = (X_test, y_test)
    return _XY_CACHE[dataset]


def get_predictions(model_display: str, dataset: str):
    """Reload a saved model and return (y_true, y_proba) on the held-out test set."""
    cache_key = (model_display, dataset)
    if cache_key in _PRED_CACHE:
        return _PRED_CACHE[cache_key]
    path = _model_file(model_display, dataset)
    if path is None:
        return None

    if path.suffix == ".joblib":
        pipeline = joblib.load(path)
        X_test, y_test = _classical_test(dataset)
        proba = pipeline.predict_proba(X_test)[:, 1]
        y_true = np.asarray(y_test)
    else:  # PyTorch deep model
        import torch

        from src.config import MAX_URL_LENGTH
        from src.data.feature_engineering import build_char_vocab, encode_urls, vocab_size
        from src.experiments.runner_deep import _DEVICE, _predict_proba
        from src.models.deep import DEEP_MODEL_DISPLAY, get_deep_model

        key = {v: k for k, v in DEEP_MODEL_DISPLAY.items()}[model_display]
        df = load_raw(dataset)
        u_tr, _uv, u_te, _ytr, _yv, y_te = split_data(df["url"].astype(str), df["result"].astype(int))
        vocab = build_char_vocab(u_tr)
        model = get_deep_model(key, vocab_size(vocab)).to(_DEVICE)
        model.load_state_dict(torch.load(path, map_location=_DEVICE))
        proba = _predict_proba(model, encode_urls(u_te, vocab, MAX_URL_LENGTH))
        y_true = np.asarray(y_te)

    _PRED_CACHE[cache_key] = (y_true, proba)
    return y_true, proba


def _models_for(metrics: pd.DataFrame, dataset: str) -> list[str]:
    return metrics[metrics["dataset"] == dataset].sort_values("f1", ascending=False)["model"].tolist()


def _save(fig, name: str) -> Path:
    PLOTS_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_FINAL_DIR / name
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- charts
def fig_metric_bars(metrics: pd.DataFrame) -> Path:
    """Chart 1 — grouped accuracy/precision/recall/F1 bars, one panel per dataset."""
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(10, 11))
    shades = ["#264653", "#2a9d8f", "#e9c46a", "#e76f51"]
    hatches = ["", "//", "..", "xx"]
    for ax, ds in zip(axes, DATASETS):
        sub = metrics[metrics["dataset"] == ds].sort_values("f1", ascending=False)
        x = np.arange(len(sub))
        w = 0.2
        for i, (m, c, h) in enumerate(zip(_METRICS, shades, hatches)):
            ax.bar(x + (i - 1.5) * w, sub[m], w, label=m, color=c, hatch=h, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["model"], rotation=25, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("score")
        ax.set_title(f"{ds.upper()} — model metrics")
        ax.legend(ncol=4, fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, "metric_bars.png")


def fig_roc_overlay(metrics: pd.DataFrame) -> Path:
    """Chart 2 — overlaid ROC curves, one panel per dataset (legend with AUC)."""
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 5))
    for ax, ds in zip(axes, DATASETS):
        for i, model in enumerate(_models_for(metrics, ds)):
            pred = get_predictions(model, ds)
            if pred is None:
                continue
            fpr, tpr, _ = roc_curve(pred[0], pred[1])
            ax.plot(fpr, tpr, _LINESTYLES[i % len(_LINESTYLES)], lw=1.6,
                    label=f"{model} (AUC {auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8, linestyle=":")
        ax.set_title(f"{ds.upper()} — ROC")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    return _save(fig, "roc_overlay.png")


def fig_f1_heatmap(metrics: pd.DataFrame) -> Path:
    """Chart 3 — F1 heatmap (models × datasets), annotated for grayscale legibility."""
    pivot = metrics.pivot_table(index="model", columns="dataset", values="f1")
    pivot = pivot.reindex(columns=[d for d in DATASETS if d in pivot.columns])
    pivot = pivot.sort_values(by=[c for c in DATASETS if c in pivot.columns], ascending=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", vmin=0.7, vmax=1.0,
                linewidths=0.5, cbar_kws={"label": "F1"}, ax=ax)
    ax.set_title("F1 by model and dataset")
    fig.tight_layout()
    return _save(fig, "f1_heatmap.png")


def fig_confusion_grid(metrics: pd.DataFrame, dataset: str = "mendeley") -> Path:
    """Chart 4 — confusion-matrix grid for all models on one dataset."""
    models = _models_for(metrics, dataset)
    ncol = 3
    nrow = int(np.ceil(len(models) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow))
    for ax, model in zip(axes.ravel(), models):
        pred = get_predictions(model, dataset)
        if pred is None:
            ax.axis("off")
            continue
        cm = confusion_matrix(pred[0], (pred[1] >= 0.5).astype(int), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["legit", "phish"], yticklabels=["legit", "phish"], ax=ax)
        ax.set_title(model, fontsize=10)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    for ax in axes.ravel()[len(models):]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices — {dataset.upper()}", y=1.01)
    fig.tight_layout()
    return _save(fig, f"confusion_grid_{dataset}.png")


def fig_time_vs_f1(metrics: pd.DataFrame) -> Path:
    """Chart 5 — training time vs F1 (fast + accurate = upper-left)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    markers = {"uci": "o", "mendeley": "s", "iscx": "^"}
    colors = {"classical": "#264653", "deep": "#e76f51"}
    for _, r in metrics.iterrows():
        ax.scatter(r["train_time_s"], r["f1"], marker=markers.get(r["dataset"], "o"),
                   color=colors[r["family"]], s=70, edgecolor="white", zorder=3)
        ax.annotate(f"{r['model'][:7]}", (r["train_time_s"], r["f1"]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("training time (s, log scale)")
    ax.set_ylabel("F1")
    ax.set_title("Training cost vs F1  (markers: o=UCI s=Mendeley ^=ISCX)")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f) for f, c in colors.items()]
    ax.legend(handles=handles, title="family", loc="lower right")
    fig.tight_layout()
    return _save(fig, "time_vs_f1.png")


def fig_feature_importance(dataset: str = "uci", top_n: int = 15) -> Path:
    """Chart 6 — top-N feature importances for RandomForest and XGBoost."""
    X, _ = to_xy(load_raw(dataset), dataset)
    names = list(X.columns)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, model in zip(axes, ["RandomForest", "XGBoost"]):
        path = _model_file(model, dataset)
        est = joblib.load(path).named_steps["model"]
        top = pd.Series(est.feature_importances_, index=names).sort_values(ascending=False).head(top_n)
        top.sort_values().plot(kind="barh", color="#2a9d8f", ax=ax)
        ax.set_title(f"{model} — top {top_n} features ({dataset.upper()})")
        ax.set_xlabel("importance")
    fig.tight_layout()
    return _save(fig, "feature_importance.png")


def fig_crossdataset_drop() -> Path:
    """Bonus — within-dataset vs cross-dataset F1 (the generalization finding)."""
    df = pd.read_csv(METRICS_CROSSDATASET_CSV)
    rows = []
    for (model, train), g in df.groupby(["model", "train_dataset"]):
        w = g[g["test_dataset"] == train]
        c = g[g["test_dataset"] != train]
        if len(w) and len(c):
            rows.append({"label": f"{model}\n(train {train[:4]})",
                         "within": w["f1"].iloc[0], "cross": c["f1"].iloc[0]})
    comp = pd.DataFrame(rows).sort_values("within", ascending=False)
    x = np.arange(len(comp))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 0.2, comp["within"], 0.4, label="within-dataset", color="#2a9d8f")
    ax.bar(x + 0.2, comp["cross"], 0.4, label="cross-dataset", color="#e76f51", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["label"], fontsize=8)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.set_title("Generalization gap: within-dataset vs cross-dataset F1")
    ax.legend()
    fig.tight_layout()
    return _save(fig, "crossdataset_drop.png")


def generate_all() -> list[Path]:
    """Generate every final figure; return the saved paths."""
    metrics = load_all_metrics()
    return [
        fig_metric_bars(metrics),
        fig_roc_overlay(metrics),
        fig_f1_heatmap(metrics),
        fig_confusion_grid(metrics, "mendeley"),
        fig_time_vs_f1(metrics),
        fig_feature_importance("uci"),
        fig_crossdataset_drop(),
    ]
