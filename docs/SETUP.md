# Setup and running

Conventions, roadmap and the full reproducibility protocol are in
[`DEVELOPMENT.md`](DEVELOPMENT.md). This file is how to get it running.

---

## Environment

Python 3.11.

### Platforms

Developed and originally run on **Linux**, and run end to end on **macOS**
(Apple Silicon): all 33 experiments complete there, with the caveat about
numerical agreement in "The verification run" below. Every script is written to
work on both: where a command differs between GNU and BSD userland, the work is
done in Python instead, which is already a hard dependency.

**Windows is untested.** The shell scripts assume a POSIX shell, so WSL is the
path most likely to work, and nobody has confirmed it. If you try it and it
breaks, that is a gap in this project rather than in your setup.

macOS needs one system library that pip cannot provide:

```bash
brew install libomp
```

XGBoost's wheel links against the OpenMP runtime, which ships with GCC on Linux
and does not exist on macOS by default. Without it, importing xgboost fails with
`Library not loaded: @rpath/libomp.dylib` and takes the environment check down
with it.

```bash
git clone https://github.com/otaviotemoteo/phishing-detection-benchmark.git
cd phishing-detection-benchmark

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/verify_environment.py
```

`verify_environment.py` checks the interpreter and the installed versions
against the pins in `requirements.txt`. Run it before anything else: a pipeline
that half-works on a mismatched library still writes numbers to `results/`, and
those numbers are worse than a crash because they look fine.

A GPU is optional. PyTorch falls back to CPU for the deep learning phase, which
is slower but produces the same values.

## Datasets

### The one manual step, done first

ISCX-URL2016 is distributed by UNB CIC behind a registration form and cannot be
fetched non-interactively. Do this before anything else and the rest of the
pipeline runs untouched:

1. Open <https://www.unb.ca/cic/datasets/url-2016.html>
2. Complete the "Download this dataset" form (name, email, organization). It is
   free.
3. Download the archive and extract it.
4. Place the classified-URL CSV at `data/iscx_url2016.csv`. Alternatively, drop
   the extracted files into `data/iscx_url2016/` and they are consolidated
   during EDA and preprocessing.

There is no way around this and there should not be. A URL that appears to skip
the form is a session cookie doing the work, and it answers `403 Registration
required` to anyone else, so putting one in the download script would hand every
future user a confusing failure while quietly routing around the provider's
terms.

### Everything else

```bash
bash scripts/download_datasets.sh
```

Fetches the other three with no account and no token: UCI and Mendeley straight
from their providers, and Malicious URLs through Kaggle's anonymous public API.
If the anonymous Kaggle route ever stops working, the script prints a manual
fallback. Run it once before the form to pull those three, and again after
placing the ISCX file so it is picked up and hashed.

Three of these are the main datasets, evaluated in the within-dataset stage. The
fourth, Malicious URLs, is used only in the cross-dataset transfer test, for the
reasons set out in [`../data/README.md`](../data/README.md).

`run_all.sh` refuses to start until all four CSVs are present, deliberately: a
pipeline that silently skips a dataset writes a results directory that looks
complete and is not. Every corpus is hash-pinned in `data/dataset_hashes.json`,
and the hash is copied into each experiment's manifest.

## Running everything

```bash
bash scripts/run_all.sh
```

EDA, then the classical benchmark, then deep learning, then the cross-dataset
transfer test, then the final figures. On the reference machine (Linux, i5-7400,
GTX 1060 3 GB, 16 GB RAM) the whole thing takes about two hours: roughly 40
minutes for the classical benchmark on CPU, about 7 minutes for deep learning on
GPU, about 50 minutes for the cross-dataset phase, plus notebook execution.
Without a CUDA GPU, as on macOS, PyTorch falls back to CPU and the deep learning
phase takes considerably longer while producing the same values.

Metric CSVs and manifests land in `results/`, figures in `plots/final/` at 300
DPI. Metric values reproduce exactly under seed 42; timing columns vary run to
run, because wall-clock time is the one thing a seed cannot pin.

## Running pieces

```bash
python -m src.experiments.run_classical --all                  # 18 classical experiments
python -m src.experiments.run_classical --model rf --dataset uci
python -m src.experiments.run_deep --all                       # CNN, LSTM, CNN-LSTM on Mendeley
python -m src.experiments.run_deep --model cnn --subset 1000   # fast smoke test
python -m src.experiments.run_cross --all                      # 6 cross-dataset runs, both directions
```

Regenerating the final figures without re-running the experiments:

```bash
python -m nbconvert --execute notebooks/06_comparisons.ipynb --to notebook --inplace
```

Browsing tracked runs:

```bash
mlflow ui --backend-store-uri ./mlruns
```

## What is in each directory

- `src/` holds every reusable module: `data/` (loaders, preprocessing, URL
  feature engineering), `models/` (the classical model factory and the PyTorch
  networks), `experiments/` (runners and their command-line interfaces),
  `evaluation/` (metrics, cost tracking, plots, final figures), `utils/` (seeds,
  hashing, manifests).
- `notebooks/` is the narrative layer, 01 through 06. The notebooks import from
  `src/` and orchestrate; they hold no reusable logic themselves. The rationale
  for that split is in [`DEVELOPMENT.md`](DEVELOPMENT.md) §3.1.
- `results/` keeps the metric CSVs and the JSON manifests, both tracked in git.
  Saved models, confusion matrices and ROC curves are generated and gitignored.
- `plots/final/` holds the publication figures, tracked.
- `scripts/` has dataset download and conversion, the environment check, and the
  full-pipeline runner.

## What makes a run reproducible

- Dependency versions pinned in `requirements.txt`, verified by
  `scripts/verify_environment.py`.
- One seed, 42, applied to Python, NumPy, PyTorch and CUDA through
  `src/utils/seeds.py`, with the cuDNN determinism flags set.
- SHA-256 dataset hashes in `data/dataset_hashes.json`, recorded into every
  manifest.
- A JSON manifest per experiment in `results/manifests/`: hyperparameters,
  library versions, git commit, metrics, cost, artifact paths.
- Local MLflow tracking in `mlruns/`.

Leakage prevention, the split policy and how cost is measured are in
[`DEVELOPMENT.md`](DEVELOPMENT.md) §6 and §7.

## The verification run

The pipeline was re-run from scratch on 2026-07-07 with the two commands above,
on the same Linux machine, and reproduced all 33 experiments with identical
metric values in every cell: accuracy, precision, recall, F1 and AUC. Seven of
the eight final figures came out byte-identical.

**Across platforms it is not bitwise, and the gap is uneven.** The full pipeline
was re-run on macOS on Apple Silicon, with every pinned library resolving to the
same version (numpy 1.26.4, scikit-learn 1.5.2, xgboost 2.1.1, torch 2.4.1) and
the same four dataset hashes. All 33 experiments completed and were compared
against the manifests in this repository, metric by metric:

| Family | Metrics | Identical | Largest gap | Median gap |
|---|---|---|---|---|
| Classical | 90 | 18 (20%) | 0.043 | 0.0009 |
| Deep, within-dataset | 15 | 0 | 0.034 | 0.0036 |
| Cross-dataset | 60 | 0 | 0.074 | 0.0013 |

The divergence is concentrated in the neural models rather than spread evenly.
In the cross-dataset test, the four tree-based runs come back at 0.407, 0.302,
0.397 and 0.318 F1 against 0.407, 0.301, 0.396 and 0.317 published here, which is
agreement to the third decimal. The CNN-LSTM runs are the ones that move, and
they are also the only ones that trained on a GPU originally and on CPU here.

**No conclusion moves.** The headline finding, that transfer collapses, holds
with the same shape: cross-dataset F1 landed in 0.30 to 0.49 against the 0.30 to
0.52 reported, and AUC in 0.42 to 0.63 against 0.45 to 0.64. The CNN-LSTM still
lands below 0.5 AUC in one direction, so the observation that its ranking
partially inverts survives too.

**The cause is not fully isolated, and saying so is part of the result.** For the
neural models it is clearly the execution path: CUDA on the reference machine
against CPU here, which is a different set of kernels and a different reduction
order, not a subtler version of the same computation. For the classical models
the likely contributor is the linear algebra underneath, since NumPy and
scikit-learn bind to OpenBLAS on x86 Linux and to Apple's Accelerate framework on
Apple Silicon. That has not been isolated by controlled experiment here, and a
0.043 swing in one recall value is larger than a pure accumulation-order argument
comfortably explains.

What this establishes is narrower and more useful than a single number: a seed
pins the random choices, not the arithmetic, and bitwise reproducibility is a
property of a platform rather than of a pipeline. Compare your run against the
tables here expecting agreement in the conclusions and in the third decimal for
the tree models, not in every digit.

The evidence is in the repository, so it can be checked without running
anything. `results/manifests/` keeps both runs side by side. Compare any
experiment across the two dates, for example:

```bash
diff results/manifests/RandomForest_uci_20260624_170037.json \
     results/manifests/RandomForest_uci_20260707_133708.json
```

Same dataset hash, same hyperparameters, identical metrics. Only the wall-clock
timings differ, along with the one figure derived from them, `time_vs_f1.png`.
