"""
Global configuration constants.

All magic numbers, paths, and shared settings live here.
If a value is used in more than one place, it belongs in this file.
"""
from __future__ import annotations

from pathlib import Path

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_SEED: int = 42

# =============================================================================
# Data splits (stratified)
# =============================================================================
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# =============================================================================
# SMOTE
# =============================================================================
SMOTE_K_NEIGHBORS: int = 5

# =============================================================================
# Cross-validation
# =============================================================================
CV_FOLDS: int = 5

# =============================================================================
# Hyperparameter search (classical ML — Phase 3)
# =============================================================================
RANDOMIZED_SEARCH_N_ITER: int = 20  # candidates sampled per RandomizedSearchCV
SEARCH_SCORING: str = "f1"  # phishing is the positive class (D-004)
SVM_MAX_TRAIN_SAMPLES: int = 15_000  # stratified cap for SVM on large datasets (D-006)
CROSS_MAX_TRAIN_SAMPLES: int = 80_000  # stratified train cap for cross-dataset runs (D-010)

# =============================================================================
# Feature engineering (URLs)
# =============================================================================
MAX_URL_LENGTH: int = 200  # Truncation length for char-level DL models (D-007)

# =============================================================================
# Deep Learning training
# =============================================================================
DL_BATCH_SIZE: int = 64
DL_LEARNING_RATE: float = 1e-3
DL_MAX_EPOCHS: int = 50
EARLY_STOPPING_PATIENCE: int = 3

# Char-level architecture dims (Planejamento §6.1–§6.2)
DL_EMBED_DIM: int = 32
CNN_FILTERS: int = 128
CNN_KERNEL: int = 3
LSTM_HIDDEN: int = 128
DL_DROPOUT: float = 0.3

# =============================================================================
# Transformer fine-tuning (memory-constrained for GTX 1060 3GB)
# =============================================================================
TRANSFORMER_MODEL_NAME: str = "distilbert-base-uncased"
TRANSFORMER_BATCH_SIZE: int = 8
TRANSFORMER_MAX_SEQ_LENGTH: int = 128
TRANSFORMER_MAX_EPOCHS: int = 3
TRANSFORMER_LEARNING_RATE: float = 2e-5
TRANSFORMER_DATASET_CAP: int = 20_000  # Cap dataset size for fine-tuning

# =============================================================================
# Paths
# =============================================================================
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
RESULTS_DIR: Path = REPO_ROOT / "results"
PLOTS_DIR: Path = REPO_ROOT / "plots"
NOTEBOOKS_DIR: Path = REPO_ROOT / "notebooks"

MANIFESTS_DIR: Path = RESULTS_DIR / "manifests"
MODELS_DIR: Path = RESULTS_DIR / "models"
CONFUSION_MATRICES_DIR: Path = RESULTS_DIR / "confusion_matrices"
ROC_CURVES_DIR: Path = RESULTS_DIR / "roc_curves"
TRAINING_CURVES_DIR: Path = RESULTS_DIR / "training_curves"

PLOTS_EDA_DIR: Path = PLOTS_DIR / "eda"
PLOTS_FEATURE_IMPORTANCE_DIR: Path = PLOTS_DIR / "feature_importance"
PLOTS_FINAL_DIR: Path = PLOTS_DIR / "final"

# =============================================================================
# Metrics CSVs
# =============================================================================
METRICS_ML_CSV: Path = RESULTS_DIR / "metrics_ml.csv"
METRICS_DL_CSV: Path = RESULTS_DIR / "metrics_dl.csv"
METRICS_TRANSFORMERS_CSV: Path = RESULTS_DIR / "metrics_transformers.csv"
METRICS_CROSSDATASET_CSV: Path = RESULTS_DIR / "metrics_crossdataset.csv"

# =============================================================================
# Plot defaults
# =============================================================================
PLOT_DPI: int = 300
PLOT_FIGSIZE_DEFAULT: tuple[float, float] = (8.0, 6.0)
