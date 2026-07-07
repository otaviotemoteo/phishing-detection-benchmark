# Phishing Detection Benchmark

A reproducible, cost-aware benchmark of AI-based phishing detection comparing two model families — Classical Machine Learning and Deep Learning — under a unified experimental protocol, including a cross-dataset generalization test that turned out to be the central finding.

This project is part of a Scientific Initiation (Iniciação Científica) at SENAI Antonio Adolpho Lobbe, conducted by Otávio Fernandes Temoteo.

## Key findings

1. **Within a dataset, everything works.** Tree ensembles and character-level neural networks reach 0.94–0.999 AUC on their own held-out test sets. On the raw-URL dataset (Mendeley), the char-level models beat every classical model by ~8 F1 points — learned representations outperform hand-crafted lexical features.

| Dataset | Best model | F1 | AUC |
|---|---|---|---|
| UCI (30 expert features) | Random Forest | 0.967 | 0.997 |
| Mendeley (raw URLs) — classical | XGBoost / CatBoost | 0.857 | 0.960 |
| Mendeley (raw URLs) — deep | LSTM | 0.937 | 0.991 |
| ISCX-URL2016 (79 lexical features) | XGBoost | 0.987 | 0.999 |

2. **Across datasets, every model collapses.** Trained on one raw-URL corpus and tested on another (Mendeley ↔ Kaggle "Malicious URLs"), F1 drops from 0.76–0.94 to 0.30–0.52 and AUC falls to 0.45–0.64 — near random. The char-level CNN-LSTM, best within-dataset, generalizes no better than the lexical models. High benchmark scores are largely **dataset memorization**, not transferable phishing knowledge.

![Generalization gap: within-dataset vs cross-dataset F1](plots/final/crossdataset_drop.png)

3. **The drop is genuine, not an artifact.** A URL-formatting confound (the `http://` scheme present in ~100% of Mendeley URLs vs ~11.5% of the Kaggle set) was detected and neutralized before measuring — see [D-010](docs/DECISIONS.md) for the full story.

Full numbers, figures, and interpretation: **[docs/RESULTS.md](docs/RESULTS.md)**.

## What was evaluated

- **Classical ML** (6 models × 3 datasets): Logistic Regression, Decision Tree, Random Forest, XGBoost, CatBoost, SVM — leakage-safe `imblearn` pipelines (impute → scale → SMOTE → model) tuned with RandomizedSearchCV.
- **Deep Learning** (3 models on Mendeley): character-level CNN, LSTM, CNN-LSTM in PyTorch — mixed precision, early stopping, class weighting.
- **Cross-dataset generalization** (3 models × 2 directions): Random Forest, XGBoost, CNN-LSTM trained on one raw-URL corpus and tested on the other, with a matched within-dataset baseline.
- **Transformers (DistilBERT) were deliberately skipped** — the phase was optional, the corpora are URL-only, and the 3 GB GPU made it the highest-risk/lowest-value phase. Justification recorded in [D-009](docs/DECISIONS.md).

Every experiment records predictive metrics (accuracy, precision, recall, F1, AUC-ROC — phishing = positive class) **and** computational cost (training time, inference latency, peak RAM, GPU usage, parameter count), plus a JSON manifest tying the result to the exact code, data, and seed that produced it.

## Datasets

| Dataset | Size | Representation | Fetch |
|---|---|---|---|
| [UCI Phishing Websites](https://archive.ics.uci.edu/dataset/327/phishing+websites) | 11,055 | 30 expert-extracted features | automatic |
| [Mendeley n96ncsr5g4/1](https://data.mendeley.com/datasets/n96ncsr5g4/1) | 80,000 | raw URL + label | automatic |
| [Malicious URLs (Kaggle)](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset) | 522,214 (phishing+benign subset) | raw URL + label | automatic |
| [ISCX-URL2016](https://www.unb.ca/cic/datasets/url-2016.html) | 36,707 | 79 lexical features | **manual** (registration form) |

See [data/README.md](data/README.md) for schemas, caveats, and the SHA-256 hash registry.

## Quick start

```bash
git clone https://github.com/otaviotemoteo/phishing-detection-benchmark.git
cd phishing-detection-benchmark

# Environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_environment.py

# Datasets — 3 of 4 download automatically; ISCX-URL2016 requires a one-time
# manual download (UNB CIC registration form; the script prints instructions)
bash scripts/download_datasets.sh

# Full pipeline: EDA -> classical ML -> deep learning -> cross-dataset -> figures
bash scripts/run_all.sh
```

Expected runtime on the reference machine (i5-7400, GTX 1060 3 GB, 16 GB RAM): **~2 h total** — classical benchmark ~40 min (CPU), deep learning ~7 min (GPU), cross-dataset ~50 min, plus notebook execution. A GPU is optional: PyTorch falls back to CPU (slower). Results land in `results/` (metrics CSVs + manifests) and `plots/final/` (figures). Re-runs reproduce the metric values exactly (seed 42); timing columns naturally vary.

### Running pieces individually

```bash
python -m src.experiments.run_classical --all            # 18 classical experiments
python -m src.experiments.run_classical --model rf --dataset uci
python -m src.experiments.run_deep --all                 # CNN/LSTM/CNN-LSTM on Mendeley
python -m src.experiments.run_deep --model cnn --subset 1000   # fast smoke test
python -m src.experiments.run_cross --all                # 6 cross-dataset runs (both directions)

python -m nbconvert --execute notebooks/06_comparisons.ipynb --to notebook --inplace  # regenerate final figures
mlflow ui --backend-store-uri ./mlruns                   # browse tracked runs
```

The notebooks (`notebooks/01_eda.ipynb` … `06_comparisons.ipynb`) are the narrative layer — they import from `src/` and orchestrate; all reusable logic lives in `src/`.

## Repository structure

- `src/` — reusable modules: `data/` (loaders, preprocessing, URL feature engineering), `models/` (classical factory, PyTorch nets), `experiments/` (runners + CLIs), `evaluation/` (metrics, cost tracking, plots, final figures), `utils/` (seeds, hashing, manifests)
- `notebooks/` — 01 EDA, 02 preprocessing, 03 classical ML, 04 deep learning, 05 cross-dataset, 06 final comparisons
- `results/` — metrics CSVs (tracked), JSON manifests (tracked), saved models / confusion matrices / ROC curves (gitignored)
- `plots/final/` — publication-ready figures at 300 DPI (tracked)
- `scripts/` — dataset download/conversion, environment check, full-pipeline runner

Full rationale: [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) §3.

## Reproducibility

Every number in `results/` is auditable and reproducible from:

- Pinned dependency versions in `requirements.txt` (installed Python is checked by `scripts/verify_environment.py`)
- A single fixed seed (42) applied to Python, NumPy, PyTorch, and CUDA via `src/utils/seeds.py`, with cuDNN determinism flags
- SHA-256 dataset hashes in `data/dataset_hashes.json`, recorded into every experiment manifest
- Per-experiment JSON manifests in `results/manifests/` (hyperparameters, library versions, git commit, metrics, cost, artifact paths)
- Local MLflow tracking in `mlruns/`

Protocol details (leakage prevention, split policy, cost measurement): [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) §6–§7.

### Verified end-to-end

This is not aspirational — the full pipeline was re-run from scratch on a later date (2026-07-07) with exactly the commands above:

```bash
bash scripts/download_datasets.sh
bash scripts/run_all.sh
```

and reproduced **all 33 experiments with identical metric values** (accuracy, precision, recall, F1, AUC — every cell), with 7 of the 8 final figures regenerated byte-identical. You can inspect the evidence before running anything yourself: `results/manifests/` keeps the manifests from both runs — compare any experiment across dates, e.g. `RandomForest_uci_20260624_170037.json` vs `RandomForest_uci_20260707_133708.json`: same dataset hash, same hyperparameters, identical metrics; only the wall-clock timings (and the one figure derived from them, `time_vs_f1.png`) differ, as timing is the one thing a seed cannot pin.

## Documentation

- [docs/RESULTS.md](docs/RESULTS.md) — **all results, figures, and interpretation** (start here to see what came out)
- [docs/DECISIONS.md](docs/DECISIONS.md) — 10 Architecture Decision Records (D-001…D-010) covering every non-trivial methodological choice
- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) — development guide: stack, conventions, roadmap, reproducibility protocol
- [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md) — chronological lab journal of every experiment session

## License

Academic project. Code released under the MIT License. Datasets retain their original licenses (see [data/README.md](data/README.md)).

## Citation

If you use this work, please cite the Scientific Initiation dissertation:

> TEMOTEO, O. F. Uso de Inteligência Artificial na Detecção de Ataques de Phishing. 2026. Trabalho de Iniciação Científica — SENAI Antonio Adolpho Lobbe, São Carlos, 2026.
