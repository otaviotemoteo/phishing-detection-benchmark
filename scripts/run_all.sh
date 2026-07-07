#!/bin/bash
# =============================================================================
# Phishing Detection Benchmark — Full Pipeline Execution
# =============================================================================
# Reproduces the entire experimental pipeline from scratch (Phases 0-7):
# environment check -> dataset download -> EDA -> classical ML benchmark ->
# deep learning benchmark -> cross-dataset generalization -> final figures.
#
# Usage:
#   bash scripts/run_all.sh
#
# Prerequisites:
#   - Python 3.11 venv at .venv/ with requirements.txt installed (the script
#     prefers .venv/bin automatically; manual activation is not required)
#   - ISCX-URL2016 downloaded manually (registration-gated; the download step
#     prints instructions). The other three datasets are fetched automatically.
#
# Expected runtime on the reference machine (i5-7400, GTX 1060 3 GB):
#   Phase 3 ~40 min (CPU) | Phase 4 ~7 min (GPU) | Phase 6 ~50 min | ~2 h total.
# Without a GPU, Phases 4/6 fall back to CPU and take substantially longer.
# All results are seeded (42) — metric values reproduce exactly; timing
# columns naturally vary run to run.
# =============================================================================

set -e  # Exit on first error
set -u  # Treat unset variables as errors

# Run from the repo root and prefer the project venv's tools, so the script
# works from any CWD and without manual `source .venv/bin/activate`.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

echo "============================================================"
echo "Phishing Detection Benchmark — full pipeline"
echo "Started: $(date -Iseconds)"
echo "============================================================"

# -----------------------------------------------------------------------------
# Phase 0 — Verify environment
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 0] Verifying environment..."
python scripts/verify_environment.py

# -----------------------------------------------------------------------------
# Phase 1 — EDA
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 1] Downloading datasets..."
bash scripts/download_datasets.sh

# Preflight: every phase below assumes all four dataset CSVs exist. Fail fast
# with a clear message instead of dying (or silently skipping) mid-pipeline.
missing=0
for csv in uci_phishing.csv mendeley_phishing.csv iscx_url2016.csv malicious_urls.csv; do
  if [[ ! -f "data/$csv" ]]; then
    echo "MISSING: data/$csv"
    missing=1
  fi
done
if [[ "$missing" -eq 1 ]]; then
  echo ""
  echo "Aborting: dataset(s) missing (instructions printed by the download step"
  echo "above — ISCX-URL2016 requires a manual, registration-gated download)."
  echo "Re-run scripts/run_all.sh after completing them."
  exit 1
fi

echo ""
echo "[Phase 1] Running EDA notebook..."
python -m nbconvert --execute notebooks/01_eda.ipynb --to notebook --inplace

# -----------------------------------------------------------------------------
# Phase 2 — Preprocessing + minimal pipeline
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 2] Preprocessing + minimal pipeline..."
python -m nbconvert --execute notebooks/02_preprocessing.ipynb --to notebook --inplace
python -m nbconvert --execute notebooks/03_ml_classical.ipynb --to notebook --inplace

# -----------------------------------------------------------------------------
# Phase 3 — Classical ML
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 3] Classical ML benchmark (6 models x 3 datasets, ~40 min)..."
python -m src.experiments.run_classical --all
python -m nbconvert --execute notebooks/03_ml_classical.ipynb --to notebook --inplace

# -----------------------------------------------------------------------------
# Phase 4 — Deep Learning
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 4] Deep Learning benchmark (CNN/LSTM/CNN-LSTM on Mendeley, ~7 min on GPU)..."
python -m src.experiments.run_deep --all
python -m nbconvert --execute notebooks/04_deep_learning.ipynb --to notebook --inplace

# -----------------------------------------------------------------------------
# Phase 5 — Transformers (SKIPPED — see D-009)
# -----------------------------------------------------------------------------
# DistilBERT fine-tuning was deliberately skipped (optional/advanced, URLs-only,
# 3 GB VRAM). Justification documented in docs/DECISIONS.md (D-009).

# -----------------------------------------------------------------------------
# Phase 6 — Cross-dataset generalization
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 6] Cross-dataset generalization (Mendeley <-> malicious_urls, ~50 min)..."
python -m src.experiments.run_cross --all
python -m nbconvert --execute notebooks/05_crossdataset.ipynb --to notebook --inplace

# -----------------------------------------------------------------------------
# Phase 7 — Final visualizations
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 7] Generating final comparison figures..."
python -m nbconvert --execute notebooks/06_comparisons.ipynb --to notebook --inplace

echo ""
echo "============================================================"
echo "Pipeline complete: $(date -Iseconds)"
echo "Results: ./results/"
echo "Plots:   ./plots/final/"
echo "============================================================"
