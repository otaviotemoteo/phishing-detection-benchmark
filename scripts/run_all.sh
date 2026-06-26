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
# Phase 4 — Deep Learning
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 4] Deep Learning benchmark (CNN/LSTM/CNN-LSTM on Mendeley)..."
python -m src.experiments.run_deep --all
jupyter nbconvert --execute notebooks/04_deep_learning.ipynb --to notebook --inplace

# -----------------------------------------------------------------------------
# Phase 5 — Transformers (SKIPPED — see D-009)
# -----------------------------------------------------------------------------
# DistilBERT fine-tuning was deliberately skipped (optional/advanced, URLs-only,
# 3 GB VRAM). Justification documented in docs/DECISIONS.md (D-009).

# -----------------------------------------------------------------------------
# Phase 6 — Cross-dataset generalization
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 6] Cross-dataset generalization (Mendeley <-> malicious_urls)..."
python -m src.experiments.run_cross --all
jupyter nbconvert --execute notebooks/05_crossdataset.ipynb --to notebook --inplace

# -----------------------------------------------------------------------------
# Phase 7 — Final visualizations
# -----------------------------------------------------------------------------
echo ""
echo "[Phase 7] Generating final comparison figures..."
jupyter nbconvert --execute notebooks/06_comparisons.ipynb --to notebook --inplace

echo ""
echo "============================================================"
echo "Pipeline complete: $(date -Iseconds)"
echo "Results: ./results/"
echo "Plots:   ./plots/final/"
echo "============================================================"
