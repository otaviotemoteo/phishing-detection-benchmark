# Phishing Detection Benchmark — Development Guide

**Project:** Comparative evaluation of AI-based phishing detection (Classical ML, Deep Learning, Transformers)
**Author:** Otávio Fernandes Temoteo
**Institution:** SENAI Antonio Adolpho Lobbe — Scientific Initiation Program
**Document status:** Living document. Update as decisions evolve.

---

## Purpose of this document

This is the operational guide for the experimental phase of the Scientific Initiation. It defines the development methodology, technical stack, roadmap, conventions, and reproducibility procedures that must be followed throughout the project.

It is intentionally written to be **indexed by AI coding assistants** (Claude Code, Cursor, Copilot Chat). When working on any task related to this project, the assistant should read this document first as authoritative context. All decisions documented here override conflicting suggestions from generic best-practice knowledge.

This document complements but does not replace:
- `Planejamento - Experimento Prático.docx` — the high-level experimental design
- The Scientific Initiation dissertation (literature review and methodology)

---

## Table of contents

1. [Guiding principles](#1-guiding-principles)
2. [Technical stack](#2-technical-stack)
3. [Repository structure](#3-repository-structure)
4. [Roadmap — 8 phases](#4-roadmap--8-phases)
5. [Coding conventions](#5-coding-conventions)
6. [Reproducibility requirements](#6-reproducibility-requirements)
7. [Computational cost measurement](#7-computational-cost-measurement)
8. [Documentation artifacts](#8-documentation-artifacts)
9. [Hardware constraints and contingencies](#9-hardware-constraints-and-contingencies)
10. [How to use this guide with an AI assistant](#10-how-to-use-this-guide-with-an-ai-assistant)
11. [Templates](#11-templates)

---

## 1. Guiding principles

These principles take precedence over local optimization. When a decision must be made and no specific guideline applies, default to the principle.

### 1.1 Reproducibility wins over elegance

The worst case in this project is producing a result reported in the dissertation that cannot be reproduced three months later. Every decision should be made with reproducibility in mind:

- Pin all dependency versions exactly (`==`, not `>=`)
- Set random seeds for every stochastic operation
- Save raw experiment outputs to disk before any post-processing
- Version all code in Git with descriptive commits
- Hash datasets at preprocessing time and record the hash with each experiment

### 1.2 Pipeline before model

Do not start with the most interesting model. Start with the most boring possible model running end-to-end through the entire pipeline.

The correct order is:
1. Build the full pipeline (preprocessing → split → SMOTE → train → evaluate → save metrics → save artifacts) with a Logistic Regression on a 1000-sample subset of the UCI dataset.
2. Confirm everything works, every artifact is saved correctly, every metric is computed correctly.
3. **Then** swap in the real models. Swapping should be a single-line change.

If swapping models is not a one-line change, the pipeline is not properly modularized and must be refactored before proceeding.

### 1.3 Decisions matter more than results

The dissertation will be evaluated on the rigor of methodological decisions, not on whether the final accuracy is 97.4% or 98.1%. Every non-trivial decision must be documented in `DECISIONS.md` with: context, decision, alternatives considered, consequences.

When in doubt about whether something is worth documenting: document it.

### 1.4 No premature optimization

Do not optimize for speed, memory, or elegance until a baseline result is obtained. Premature optimization is the leading cause of bugs that invalidate experiments.

### 1.5 Fail loud, fail early

Use assertions liberally. Validate data shapes, value ranges, and class distributions at every pipeline stage. A pipeline that silently produces wrong results is worse than a pipeline that crashes.

```python
assert X_train.shape[0] == y_train.shape[0], f"Mismatch: {X_train.shape[0]} vs {y_train.shape[0]}"
assert set(y_train.unique()) == {0, 1}, f"Unexpected labels: {set(y_train.unique())}"
assert not X_train.isnull().any().any(), "NaN values found in training data"
```

---

## 2. Technical stack

### 2.1 Core libraries

| Library | Pinned version | Purpose |
|---|---|---|
| Python | 3.11.x | Runtime (3.12+ has compatibility issues with some ML libs) |
| numpy | 1.26.4 | Numerical operations. **Do not upgrade to 2.x** — breaks compat with many libs |
| pandas | 2.2.3 | Data manipulation |
| scikit-learn | 1.5.2 | Classical ML, metrics, splits, CV |
| imbalanced-learn | 0.12.4 | SMOTE |
| xgboost | 2.1.1 | XGBoost classifier |
| catboost | 1.2.7 | CatBoost classifier |
| lightgbm | 4.5.0 | LightGBM classifier (optional, for comparison) |
| torch | 2.4.1+cu121 | Deep learning (CNN, LSTM, Transformers) |
| transformers | 4.45.2 | Hugging Face — DistilBERT |
| datasets | 3.0.1 | Hugging Face — dataset loading |
| matplotlib | 3.9.2 | Plotting |
| seaborn | 0.13.2 | Statistical plots |
| psutil | 6.0.0 | RAM monitoring |
| py3nvml | 0.2.7 | GPU memory monitoring |
| mlflow | 2.16.2 | Experiment tracking |
| joblib | 1.4.2 | Model serialization |
| tqdm | 4.66.5 | Progress bars |

### 2.2 Stack decisions

**Why PyTorch instead of TensorFlow:**
- Hugging Face is PyTorch-first; reproducing modern phishing-DL literature is easier in PyTorch.
- More explicit VRAM control, critical for the 3 GB GTX 1060 constraint.
- Recent literature (2024–2026) on phishing with DL/Transformers is predominantly PyTorch.

**Why MLflow instead of Weights & Biases:**
- 100% local execution aligned with the privacy/reproducibility requirements of the project.
- Open source, no external accounts required.
- Direct integration with sklearn, PyTorch, and Hugging Face.

**Why imbalanced-learn instead of manual SMOTE:**
- Reference implementation aligned with Chawla et al. (2002).
- Integrated with sklearn pipelines via `Pipeline` from `imblearn.pipeline` (critical to avoid data leakage — see §6).

### 2.3 CUDA and driver requirements

- NVIDIA driver: 535+ (for CUDA 12.1)
- CUDA toolkit: 12.1
- cuDNN: 8.9.x

Verify with:
```bash
nvidia-smi  # driver and CUDA version
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## 3. Repository structure

```
phishing-detection-benchmark/
│
├── README.md                          # User-facing readme (how to reproduce)
├── DEVELOPMENT.md                     # This file
├── DECISIONS.md                       # Architecture Decision Records
├── EXPERIMENT_LOG.md                  # Daily experiment log
├── requirements.txt                   # Pinned dependencies
├── .gitignore
├── .python-version                    # 3.11.x
│
├── data/                              # Raw datasets (gitignored)
│   ├── uci_phishing.csv
│   ├── mendeley_phishing.csv
│   ├── iscx_url2016.csv
│   └── dataset_hashes.json            # SHA256 of each dataset for reproducibility
│
├── notebooks/                         # Exploratory/narrative notebooks
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_ml_classical.ipynb
│   ├── 04_deep_learning.ipynb
│   ├── 05_transformers.ipynb
│   └── 06_comparisons.ipynb
│
├── src/                               # Reusable Python modules
│   ├── __init__.py
│   ├── data/
│   │   ├── loaders.py                 # Dataset loading functions
│   │   ├── preprocessing.py           # Cleaning, splits, SMOTE
│   │   └── feature_engineering.py     # URL feature extraction
│   ├── models/
│   │   ├── classical.py               # sklearn-based models
│   │   ├── deep.py                    # PyTorch CNN, LSTM, hybrid
│   │   └── transformers.py            # DistilBERT fine-tuning
│   ├── evaluation/
│   │   ├── metrics.py                 # Standardized metric computation
│   │   ├── cost.py                    # Computational cost tracking
│   │   └── plots.py                   # Confusion matrices, ROC curves
│   └── utils/
│       ├── seeds.py                   # Centralized seed management
│       ├── manifests.py               # Experiment manifest generation
│       └── io.py                      # Standardized save/load
│
├── results/                           # All experiment outputs (gitignored except CSVs)
│   ├── metrics_ml.csv                 # Classical ML results
│   ├── metrics_dl.csv                 # Deep Learning results
│   ├── metrics_transformers.csv       # Transformer results
│   ├── metrics_crossdataset.csv       # Cross-dataset generalization
│   ├── manifests/                     # JSON manifest per experiment
│   ├── models/                        # Saved model artifacts (joblib, .pt)
│   ├── confusion_matrices/
│   └── roc_curves/
│
├── plots/                             # Final publication-ready plots
│   ├── eda/
│   ├── feature_importance/
│   └── final/                         # Plots that go into the dissertation
│
├── mlruns/                            # MLflow tracking directory (gitignored)
│
└── scripts/                           # Standalone executable scripts
    ├── download_datasets.sh
    ├── run_all.sh                     # Full pipeline reproduction
    └── verify_environment.py          # Sanity check for setup
```

### 3.1 Why src/ exists in addition to notebooks/

Notebooks are for **narrative and exploration**. Reusable logic lives in `src/`. A notebook should import from `src` and orchestrate, not implement.

Bad: a 600-line notebook that defines functions inline.
Good: a 200-line notebook that imports `from src.models.classical import train_random_forest` and calls it three times with different datasets.

This separation makes the code testable, reusable, and survivable when notebooks become messy.

### 3.2 .gitignore essentials

```
# Datasets — never commit
data/*.csv
data/*.zip

# Models — too large for Git
results/models/
*.pt
*.joblib
*.h5

# MLflow tracking
mlruns/

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
venv/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

---

## 4. Roadmap — 8 phases

Each phase has a clear exit criterion. Do not proceed to phase N+1 until phase N's exit criterion is satisfied. Resist the temptation to work on multiple phases in parallel.

### Phase 0 — Setup and versioning

**Tasks:**
- Create GitHub repo with the structure from §3
- Initialize `EXPERIMENT_LOG.md` and `DECISIONS.md` with header sections
- Set up Python 3.11 virtual environment
- Install pinned dependencies from `requirements.txt`
- Configure CUDA (driver, toolkit, cuDNN) for PyTorch
- Write `scripts/verify_environment.py` that checks Python version, library versions, GPU availability
- Create a minimal `run_all.sh` (even if it only echoes "TODO")

**Exit criterion:**
1. Clone the repo into a fresh directory.
2. Create venv, install deps.
3. Run `python scripts/verify_environment.py` — all checks pass.
4. Run `bash scripts/run_all.sh` — executes without error (even if it does nothing yet).

### Phase 1 — Exploratory Data Analysis

**Tasks:**
- Download the three datasets via documented sources
- Compute SHA256 hashes; save to `data/dataset_hashes.json`
- In `01_eda.ipynb`, for each dataset:
  - Sample count, feature count, feature types
  - Class distribution (% phishing vs legitimate)
  - Descriptive statistics per feature
  - Duplicates, missing values, anomalies
  - Sample inspection of 10 representative URLs per class
- Save all plots to `plots/eda/` at 300 DPI

**Exit criterion:**
You can describe each dataset in three sentences from memory. All EDA plots saved.

### Phase 2 — End-to-end minimal pipeline

**Tasks:**
- Implement `src/data/preprocessing.py`: split (stratified 70/15/15), SMOTE on train only, normalization
- Implement `src/evaluation/metrics.py`: standardized metric computation returning a dict
- Implement `src/evaluation/cost.py`: training time, inference time, peak RAM, GPU usage, parameter count
- Implement `src/utils/manifests.py`: generates JSON manifest per experiment
- Implement `src/utils/seeds.py`: single function `set_all_seeds(seed=42)` that seeds Python, NumPy, PyTorch, sklearn
- In `02_preprocessing.ipynb` + `03_ml_classical.ipynb`, run Logistic Regression on UCI end-to-end
- Verify: `metrics_ml.csv` has all columns, manifest JSON saved, confusion matrix saved, ROC saved

**Exit criterion:**
Run a single notebook cell that says `train_and_evaluate(model=LogisticRegression(), dataset='uci')` and get a complete CSV row, manifest, confusion matrix, and ROC curve, all saved to standard locations.

### Phase 3 — Classical ML complete

**Tasks:**
- Implement remaining models in `src/models/classical.py`: Decision Tree, Random Forest, XGBoost, CatBoost, Logistic Regression, SVM
- Run all 6 models on all 3 datasets (18 experiments)
- Apply GridSearchCV or RandomizedSearchCV with stratified 5-fold CV
- Generate feature importance plots for tree-based models
- Populate `metrics_ml.csv` completely

**Exit criterion:**
`metrics_ml.csv` has 18 rows. All confusion matrices and ROC curves saved. Feature importance plots saved for RF, XGBoost, CatBoost.

### Phase 4 — Deep Learning pipeline

**Tasks:**
- Implement character-level tokenization in `src/data/feature_engineering.py`
- Implement CNN, LSTM, CNN-LSTM in `src/models/deep.py` (PyTorch)
- Start with CNN on a 1000-sample subset of Mendeley to verify the DL pipeline end-to-end
- Once verified, scale to full datasets
- Use `EarlyStopping` (patience=3), `ModelCheckpoint` for best val_loss
- Mixed precision training (`torch.amp`) if VRAM is tight
- Populate `metrics_dl.csv`

**Exit criterion:**
`metrics_dl.csv` complete for CNN, LSTM, CNN-LSTM on Mendeley and ISCX.

### Phase 5 — Transformers (optional)

**Risk:** GTX 1060 with 3 GB VRAM is the bottleneck. Mitigations:
- Dataset capped at 20,000 samples (Planejamento §7.1)
- `batch_size=8` or `16`, never higher
- `max_seq_length=128` (not 512)
- `fp16` mixed precision
- Gradient checkpointing if needed

**Tasks:**
- Fine-tune `distilbert-base-uncased` on Mendeley
- If memory issues persist, document attempts and migrate to VPS (see §9)
- Record metrics and cost identically to other layers

**Exit criterion:**
Either: DistilBERT results in `metrics_transformers.csv` with cost metrics, OR a documented justification in `DECISIONS.md` of why fine-tuning was not feasible and which alternative was adopted.

### Phase 6 — Cross-dataset generalization

**Tasks:**
- Train on dataset A, test on dataset B for selected model pairs
- RF on UCI → test on ISCX
- XGBoost on UCI → test on ISCX
- CNN-LSTM on Mendeley → test on ISCX
- Record in `metrics_crossdataset.csv`

**Exit criterion:**
Cross-dataset CSV populated. Performance drop (or absence) is the key finding to discuss.

### Phase 7 — Final visualizations

**Tasks:**
- Implement `06_comparisons.ipynb` reading all CSVs
- Generate 6 plots from Planejamento §9.1:
  1. Bar charts per metric
  2. Overlaid ROC curves
  3. Heatmap (models × datasets, F1)
  4. Confusion matrix grid
  5. Training time vs F1 scatter
  6. Feature importance bars
- Export at 300 DPI to `plots/final/`
- Verify legibility in grayscale (print test)

**Exit criterion:**
All 6 final plots in `plots/final/` ready for dissertation insertion.

---

## 5. Coding conventions

### 5.1 Python style

- Follow PEP 8. Use `ruff` for linting and `black` for formatting (line length 100).
- Type hints on public functions:
  ```python
  def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
      ...
  ```
- Docstrings on all public functions (Google style):
  ```python
  def train_model(model, X_train, y_train):
      """Train a classifier and return training time.

      Args:
          model: A sklearn-compatible classifier with a fit() method.
          X_train: Training features, shape (n_samples, n_features).
          y_train: Training labels, shape (n_samples,).

      Returns:
          A tuple (trained_model, training_time_seconds).
      """
  ```

### 5.2 Imports

Order: standard library → third-party → local. Separate with blank lines.

```python
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.evaluation.metrics import compute_metrics
from src.utils.seeds import set_all_seeds
```

### 5.3 Magic numbers

No magic numbers in code. Define constants at the top of the module or in `src/config.py`:

```python
# src/config.py
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SMOTE_K_NEIGHBORS = 5
MAX_URL_LENGTH = 200
EARLY_STOPPING_PATIENCE = 3
```

### 5.4 Notebooks must run top to bottom

A notebook that requires running cells out of order is broken. Test reproducibility by restarting the kernel and running all cells. If anything breaks, fix the notebook.

### 5.5 Commit messages

Format:
```
<type>: <concise description>

<optional longer explanation>
```

Types: `feat`, `fix`, `docs`, `refactor`, `experiment`, `data`.

Bad: `fix bug`
Good: `fix: prevent SMOTE from being applied before train/test split`

Bad: `update notebook`
Good: `experiment: add Random Forest on Mendeley with GridSearchCV`

---

## 6. Reproducibility requirements

### 6.1 Seed everything

Every script and notebook must call `set_all_seeds(42)` before any stochastic operation:

```python
# src/utils/seeds.py
import random
import numpy as np
import torch

def set_all_seeds(seed: int = 42) -> None:
    """Seed all sources of randomness for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Determinism flags (slight perf cost, worth it)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 6.2 Data leakage prevention

The single most common source of invalid results in this kind of project. Hard rules:

- **SMOTE only on training data.** Never on validation, never on test.
- **Fit scalers/normalizers only on training data.** Apply (transform) the same fitted scaler to val/test.
- **Use imblearn.pipeline.Pipeline, not sklearn.pipeline.Pipeline**, when SMOTE is involved — it correctly applies SMOTE only during fit, not during cross-validation folds.
- **Hyperparameter tuning never touches the test set.** Tune with CV on training data only.

Wrong:
```python
X_resampled, y_resampled = smote.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled)
```

Right:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
# X_test stays untouched
```

### 6.3 Dataset versioning via hashing

```python
# src/utils/io.py
import hashlib

def hash_dataset(filepath: Path) -> str:
    """Compute SHA256 of a dataset file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()
```

Every experiment manifest records the hash of the input data. If results differ between runs, the first check is whether the dataset hash matches.

### 6.4 Experiment manifests

Every model trained must produce a JSON manifest:

```json
{
  "experiment_id": "rf_uci_20260712_143022",
  "timestamp": "2026-07-12T14:30:22-03:00",
  "model": "RandomForestClassifier",
  "dataset": "uci_phishing",
  "dataset_hash": "a3f1c9...",
  "git_commit": "8b2c4f9",
  "seed": 42,
  "hyperparameters": {
    "n_estimators": 200,
    "max_depth": null,
    "max_features": "sqrt"
  },
  "library_versions": {
    "python": "3.11.9",
    "sklearn": "1.5.2",
    "numpy": "1.26.4"
  },
  "metrics": {
    "accuracy": 0.974,
    "precision": 0.971,
    "recall": 0.975,
    "f1": 0.972,
    "auc_roc": 0.977
  },
  "cost": {
    "training_time_s": 14.2,
    "inference_time_ms_per_sample": 0.31,
    "peak_ram_mb": 412,
    "gpu_used": false,
    "n_parameters": null
  },
  "artifacts": {
    "model_path": "results/models/rf_uci_20260712_143022.joblib",
    "confusion_matrix": "results/confusion_matrices/rf_uci.png",
    "roc_curve": "results/roc_curves/rf_uci.png"
  }
}
```

---

## 7. Computational cost measurement

Cost metrics must be measured identically across all model families to be comparable. The standard interface:

```python
# src/evaluation/cost.py
import time
import psutil
import os

class CostTracker:
    """Context manager to track training cost metrics."""

    def __init__(self):
        self.start_time = None
        self.peak_ram_mb = 0
        self.process = psutil.Process(os.getpid())

    def __enter__(self):
        self.start_time = time.perf_counter()
        self.peak_ram_mb = self.process.memory_info().rss / 1024 / 1024
        return self

    def update_ram(self):
        """Call periodically during training to track peak."""
        current = self.process.memory_info().rss / 1024 / 1024
        if current > self.peak_ram_mb:
            self.peak_ram_mb = current

    def __exit__(self, *args):
        self.elapsed_s = time.perf_counter() - self.start_time

# Usage
with CostTracker() as tracker:
    model.fit(X_train, y_train)
    tracker.update_ram()

cost = {
    "training_time_s": tracker.elapsed_s,
    "peak_ram_mb": tracker.peak_ram_mb,
}
```

**Inference time** must be measured on a batch from the test set, averaged over at least 100 samples, with warmup:

```python
# Warmup
for _ in range(10):
    _ = model.predict(X_test[:32])

# Measure
start = time.perf_counter()
n_samples = min(1000, len(X_test))
_ = model.predict(X_test[:n_samples])
elapsed = time.perf_counter() - start
inference_time_ms_per_sample = (elapsed / n_samples) * 1000
```

**Parameter count** for PyTorch models:
```python
def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
```

**GPU memory** (when applicable):
```python
import py3nvml.py3nvml as nvml
nvml.nvmlInit()
handle = nvml.nvmlDeviceGetHandleByIndex(0)
mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
gpu_used_mb = mem_info.used / 1024 / 1024
```

---

## 8. Documentation artifacts

Three living documents must be maintained throughout the project.

### 8.1 EXPERIMENT_LOG.md

Reverse-chronological daily log. Template entry:

```markdown
## 2026-07-12

### What I did
Ran Random Forest on UCI with default hyperparameters as a sanity check.

### Results
- Accuracy: 97.4%, F1: 97.2%, AUC: 97.7%
- Training time: 14.2s
- Result consistent with Emmanuel (2025) baseline of 97.3%

### What worked
- Pipeline from Phase 2 swapped cleanly to RF with one-line change
- Manifest JSON generated correctly

### What didn't
- Initially forgot to set seed; reran with set_all_seeds(42) and results were stable

### Next
- Tune n_estimators and max_depth via GridSearchCV
- Then move to Mendeley dataset
```

### 8.2 DECISIONS.md

Architecture Decision Records. Template:

```markdown
## D-007: URL truncation at 200 characters for DL models

**Date:** 2026-07-15
**Status:** Accepted
**Phase:** 4

### Context
URLs in the Mendeley dataset range from 12 to 2843 characters in length.
The 95th percentile is 187 characters. Training with full-length URLs
would require padding to 2843, with most samples being mostly padding.

### Decision
Truncate URLs at 200 characters. Pad shorter URLs with a dedicated PAD token.

### Alternatives considered
- 100 characters: would truncate 18% of URLs, losing distinguishing features.
- 500 characters: 4× memory cost with marginal information gain.
- Variable length with bucketing: rejected for simplicity in this iteration.

### Consequences
- May reduce detection accuracy for very long phishing URLs (uncommon but real).
- Mitigation: add `url_length` as an auxiliary feature for hybrid models.
- Saves ~3× memory during DL training compared to 500.

### References
- Statistic computed in `notebooks/01_eda.ipynb`, cell 18.
```

### 8.3 README.md (user-facing)

This is different from `DEVELOPMENT.md`. The user-facing README is the entry point for someone visiting the repo. It should contain:

- One-paragraph description of the project
- Quick start (clone, install, run)
- Link to `DEVELOPMENT.md` for contributors
- Citation information
- License

Keep it short. The detail lives in `DEVELOPMENT.md`.

---

## 9. Hardware constraints and contingencies

### 9.1 Primary environment

```
CPU:        Intel Core i5-7400 (4C/4T, up to 3.5 GHz)
GPU:        NVIDIA GeForce GTX 1060 3 GB
RAM:        16 GB DDR4
Storage:    223 GB SSD + 466 GB HDD
OS:         Linux Mint 22.3 (Kernel 6.17)
```

### 9.2 Memory budget per model family

| Family | Expected VRAM | Strategy |
|---|---|---|
| Classical ML | 0 (CPU only) | No constraint |
| CNN/LSTM/CNN-LSTM | 1.5–2.5 GB | batch_size=64, embedding_dim=32 |
| DistilBERT fine-tune | 2.5–3.0 GB (tight) | batch_size=8, max_seq=128, fp16 |

### 9.3 Contingency: VPS migration

Trigger conditions:
- OOM (out of memory) errors persist after applying all memory mitigations
- A single training run takes longer than 12 hours

Recommended VPS specs:
- GPU: NVIDIA T4 (16 GB) or better
- Providers: Vast.ai (cheapest), Lambda Labs, Paperspace
- Expected cost: under $5 for a complete Transformer training run

When using VPS, record in the manifest:
- VPS provider and instance type
- GPU model
- Total wall-clock time

The methodology section of the dissertation must mention this contingency if invoked.

### 9.4 Avoid catastrophic data loss

- All experiment results (CSVs, manifests) are committed to Git daily
- Trained model files are too large for Git — back up `results/models/` to external storage weekly
- The HDD (466 GB) is the dedicated location for model checkpoints

---

## 10. How to use this guide with an AI assistant

When working with Claude Code, Cursor, Copilot Chat, or any AI coding assistant:

### 10.1 Provide this document as context

At the start of every session, ensure the assistant has read `DEVELOPMENT.md`. In Claude Code or Cursor, this is automatic if the file is in the project. In a chat-based assistant, paste a link or relevant sections.

### 10.2 Reference specific sections in prompts

Bad: "help me train a Random Forest"
Good: "implement training for Random Forest following §3 (repo structure), §5 (conventions), §6 (reproducibility), and §7 (cost metrics) of DEVELOPMENT.md"

### 10.3 Require justification against this guide

When the assistant suggests something:
- If it conflicts with a guideline here, ask it to justify the deviation
- If the deviation is sound, document the new decision in `DECISIONS.md`
- If the guideline is the right call, override the assistant

### 10.4 Push back on premature optimization

AI assistants often suggest optimizations (caching, multiprocessing, fancy architectures). Reject these until a baseline is established. Reference §1.4.

### 10.5 Use the assistant to maintain documentation

After each experiment session, ask the assistant to:
- Draft the entry for `EXPERIMENT_LOG.md` based on what was done
- Identify any decisions worth recording in `DECISIONS.md`
- Update version numbers in `requirements.txt` if anything changed

---

## 11. Templates

### 11.1 New experiment script template

```python
"""
<Brief description of what this experiment does.>

Phase: <0-7>
Related decisions: <D-XXX references>
"""
from pathlib import Path

from src.data.loaders import load_dataset
from src.data.preprocessing import preprocess, split_data, apply_smote
from src.models.classical import train_model
from src.evaluation.metrics import compute_metrics
from src.evaluation.cost import CostTracker
from src.utils.seeds import set_all_seeds
from src.utils.manifests import save_manifest

EXPERIMENT_ID = "<model>_<dataset>_<timestamp>"
DATASET = "<uci|mendeley|iscx>"
MODEL_CONFIG = {...}

def main():
    set_all_seeds(42)

    # Load
    X, y = load_dataset(DATASET)

    # Split (stratified)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # SMOTE on train only
    X_train, y_train = apply_smote(X_train, y_train)

    # Train with cost tracking
    with CostTracker() as tracker:
        model = train_model(MODEL_CONFIG, X_train, y_train)
        tracker.update_ram()

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_proba)

    # Save artifacts
    save_manifest(EXPERIMENT_ID, model, metrics, tracker, DATASET, MODEL_CONFIG)

if __name__ == "__main__":
    main()
```

### 11.2 DECISIONS.md entry template

```markdown
## D-XXX: <Concise decision title>

**Date:** YYYY-MM-DD
**Status:** Proposed | Accepted | Superseded by D-YYY
**Phase:** <0-7>

### Context
<What problem motivated this decision? What constraints applied?>

### Decision
<What was decided? Be specific.>

### Alternatives considered
<List 2-3 alternatives and why each was rejected.>

### Consequences
<Positive and negative implications. What does this make easier or harder?>

### References
<Links to code, papers, notebooks, prior decisions.>
```

### 11.3 EXPERIMENT_LOG.md entry template

```markdown
## YYYY-MM-DD

### What I did
<1-3 sentences describing the session's work.>

### Results
<Bullet list of concrete outcomes: metrics, errors, observations.>

### What worked
<What went smoothly. Worth replicating.>

### What didn't
<Bugs, dead ends, surprises. Worth avoiding.>

### Next
<What to tackle next session. Keep it concrete.>
```

---

## Appendix A — Quick reference commands

```bash
# Environment setup
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_environment.py

# Run an experiment
python -m src.experiments.run_classical --model rf --dataset uci

# Launch MLflow UI
mlflow ui --backend-store-uri ./mlruns

# Generate a manifest hash for the current dataset
python -c "from src.utils.io import hash_dataset; print(hash_dataset('data/uci_phishing.csv'))"

# Regenerate all final plots from CSVs (no retraining)
jupyter nbconvert --execute notebooks/06_comparisons.ipynb

# Reproduce the full pipeline
bash scripts/run_all.sh
```

## Appendix B — When to update this document

Update `DEVELOPMENT.md` when:
- A new tool is added to the stack
- A guideline is revised based on lessons learned
- A phase definition changes
- A new template is introduced

Update via PR with a clear commit message: `docs: update DEVELOPMENT.md §X to reflect <change>`.

Do not update for one-off experiment decisions — those belong in `DECISIONS.md`.

---

**End of DEVELOPMENT.md**

