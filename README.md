# Phishing Detection Benchmark

A comparative evaluation of AI-based phishing detection across three model families: Classical Machine Learning, Deep Learning, and Transformers.

This project is part of a Scientific Initiation (Iniciação Científica) at SENAI Antonio Adolpho Lobbe, conducted by Otávio Fernandes Temoteo.

## Overview

The goal is to benchmark phishing detection models under a unified experimental protocol, evaluating both predictive performance and computational cost. Three datasets are used: UCI Phishing Websites, Mendeley Phishing Dataset, and ISCX-URL2016.

Model families evaluated:

- **Classical ML:** Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost, SVM
- **Deep Learning:** CNN, LSTM, CNN-LSTM (hybrid)
- **Transformers:** DistilBERT (fine-tuned)

## Quick start

```bash
# Clone
git clone https://github.com/<user>/phishing-detection-benchmark.git
cd phishing-detection-benchmark

# Environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Verify setup
python scripts/verify_environment.py

# Download datasets (see scripts/download_datasets.sh for sources)
bash scripts/download_datasets.sh

# Run the full pipeline
bash scripts/run_all.sh
```

## Repository structure

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §3 for the full structure rationale.

Key directories:

- `src/` — reusable Python modules (data, models, evaluation, utils)
- `notebooks/` — narrative experiment notebooks (01–06)
- `results/` — metrics CSVs, manifests, saved models, plots
- `plots/final/` — publication-ready figures for the dissertation

## Reproducibility

Every experiment is reproducible from:

- Pinned dependency versions in `requirements.txt`
- Fixed random seed (42) across all stochastic operations
- Dataset hashes recorded in `data/dataset_hashes.json`
- Per-experiment JSON manifests in `results/manifests/`

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) §6 for the full reproducibility protocol.

## Documentation

- [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) — comprehensive development guide (roadmap, conventions, reproducibility, cost measurement)
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — Architecture Decision Records (ADRs) for non-trivial methodological choices
- [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md) — chronological lab journal

## License

Academic project. Code released under MIT License. Datasets retain their original licenses (see `data/README.md` once datasets are downloaded).

## Citation

If you use this work, please cite the Scientific Initiation dissertation:

> TEMOTEO, O. F. Uso de Inteligência Artificial na Detecção de Ataques de Phishing. 2026. Trabalho de Iniciação Científica — SENAI Antonio Adolpho Lobbe, São Carlos, 2026.
