#!/bin/bash
# =============================================================================
# Phishing Detection Benchmark — Full Pipeline Execution
# =============================================================================
# This script reproduces the entire experimental pipeline from scratch.
# Each phase will be uncommented and implemented as the project progresses.
#
# Usage:
#   bash scripts/run_all.sh
#
# Prerequisites:
#   - venv activated
#   - requirements.txt installed
#   - `python scripts/verify_environment.py` passes
#   - Datasets downloaded into ./data/
# =============================================================================

set -e  # Exit on first error
set -u  # Treat unset variables as errors

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

echo ""
echo "[Phase 1] Running EDA notebook..."
jupyter nbconvert --execute notebooks/01_eda.ipynb --to notebook --inplace

# -----------------------------------------------------------------------------
# Phase 2 — Preprocessing + minimal pipeline
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 2] Preprocessing + minimal pipeline..."
jupyter nbconvert --execute notebooks/02_preprocessing.ipynb --to notebook --inplace
jupyter nbconvert --execute notebooks/03_ml_classical.ipynb --to notebook --inplace

# -----------------------------------------------------------------------------
# Phase 3 — Classical ML
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 3] Classical ML benchmark (6 models x 3 datasets)..."
python -m src.experiments.run_classical --all
jupyter nbconvert --execute notebooks/03_ml_classical.ipynb --to notebook --inplace

# -----------------------------------------------------------------------------
# Phase 4 — Deep Learning (TODO)
# -----------------------------------------------------------------------------
# echo ""
# echo "[Phase 4] Deep Learning benchmark..."
# jupyter nbconvert --execute notebooks/04_deep_learning.ipynb --to notebook --inplace

# -----------------------------------------------------------------------------
# Phase 5 — Transformers (TODO)
# -----------------------------------------------------------------------------
# echo ""
# echo "[Phase 5] Transformer fine-tuning..."
# jupyter nbconvert --execute notebooks/05_transformers.ipynb --to notebook --inplace

# -----------------------------------------------------------------------------
# Phase 6 — Cross-dataset (TODO)
# -----------------------------------------------------------------------------
# echo ""
# echo "[Phase 6] Cross-dataset generalization..."
# (TODO)

# -----------------------------------------------------------------------------
# Phase 7 — Final visualizations (TODO)
# -----------------------------------------------------------------------------
# echo ""
# echo "[Phase 7] Generating final plots..."
# jupyter nbconvert --execute notebooks/06_comparisons.ipynb --to notebook --inplace

echo ""
echo "============================================================"
echo "Pipeline complete: $(date -Iseconds)"
echo "Results: ./results/"
echo "Plots:   ./plots/final/"
echo "============================================================"
