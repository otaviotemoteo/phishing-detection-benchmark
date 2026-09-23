# Setup and running

Conventions, roadmap and the full reproducibility protocol are in
[`DEVELOPMENT.md`](DEVELOPMENT.md). This file is how to get it running.

---

## Environment

Python 3.11.

### Platforms

Developed and originally run on **Linux**, and run end to end on **macOS**
(Apple Silicon) and on **Windows 11**: all 33 experiments complete on all three,
with the caveat about numerical agreement in "The verification run" below. Every
script is written to work on all of them: where a command differs between GNU and
BSD userland, the work is done in Python instead, which is already a hard
dependency.

**Windows runs natively and does not need WSL.** Run the two shell scripts from
Git Bash, which ships with Git for Windows; nothing else changes. Two
Windows-specific details are handled in the code rather than left to you. The
scripts look for the interpreter in `.venv/Scripts/` as well as `.venv/bin/`,
because on Windows neither `python` nor `python3` on PATH is the venv's. And
`src.evaluation` pins a non-interactive matplotlib backend on import: without it
matplotlib selects TkAgg, whose Tk objects are finalized on a joblib worker
thread during the classical benchmark and abort the interpreter outright with
`Tcl_AsyncDelete: async handler deleted by the wrong thread`.

macOS needs one system library that pip cannot provide:

```bash
brew install libomp
```

XGBoost's wheel links against the OpenMP runtime, which ships with GCC on Linux
and does not exist on macOS by default. Without it, importing xgboost fails with
`Library not loaded: @rpath/libomp.dylib` and takes the environment check down
with it.

`requirements.txt` declares only what the code imports, each pinned with `==`.
The Hugging Face stack (`transformers`, `datasets`, `tokenizers`, `accelerate`),
`torchvision`, `lightgbm` and `tqdm` were declared but never imported, and were
removed during the repository audit: the transformer phase was dropped (D-009)
and the other three were never used. Nothing in the pipeline changed, and the
install is several gigabytes smaller.

```bash
git clone https://github.com/otaviotemoteo/phishing-detection-benchmark.git
cd phishing-detection-benchmark

python3.11 -m venv .venv
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
pip install -r requirements.txt
python scripts/verify_environment.py
```

`verify_environment.py` runs before anything else, and does two things. It
identifies the platform (operating system, architecture, BLAS backend, PyTorch
backend) and prints what that platform means for your run. Then it checks the
interpreter, the pinned versions, the directory structure, the dataset hashes
and the integrity of the three result tables, printing the exact fix under
anything that fails. A pipeline that half-works on a mismatched library still
writes numbers to `results/`, and those numbers are worse than a crash because
they look fine.

It exits 0 when everything passes and 1 when something fails. A platform outside
the three below is a warning, not a failure.

### The three platform profiles

The script reduces the platform to one label, and that label is what decides
what to expect. The table repeats what the script prints, because this is the
document you read before you have an environment to run it in.

| Profile | BLAS behind NumPy | PyTorch backend | What to expect from the numbers |
|---|---|---|---|
| `linux-x86_64-cuda` | OpenBLAS, x86-64 build | CUDA | The reference platform. Metric values reproduce exactly |
| `macos-arm64-mps` | OpenBLAS, arm64 build | MPS | Classical models agree to about the third decimal; neural models diverge in the last decimals |
| `windows-x86_64-cpu` | OpenBLAS, x86-64 build | CPU | All 33 experiments complete (2026-09-01, i5-14500). Classical metrics agree most closely of the three; the neural phases run on CPU and are slower |

Verified end to end on all three. The manifests committed here come from the
reference platform only, so `scripts/compare_platforms.py` cannot yet put the
three side by side: see "The verification run" below for the comparison that was
done by hand, and commit the macOS and Windows manifests to automate it.

Anything else is labelled `unsupported`: the pipeline should still run, but no
baseline exists for it, so a numerical difference cannot be attributed.

A GPU is optional. PyTorch falls back to CPU for the deep learning phase, which
is slower; on the same platform it produces the same values, across platforms it
does not (see "The verification run" below).

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
Without a CUDA GPU, as on macOS and on the Windows machine described below,
PyTorch falls back to CPU. On a modern processor that costs less than it sounds:
on an i5-14500 the deep learning phase took 42 minutes against the 7-minute GPU
figure, and the classical benchmark 28 minutes against 40. It does move the
neural metrics, though, which is the subject of "The verification run".

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
- `scripts/` has dataset download and conversion, the environment check, the
  full-pipeline runner, the manifest path migration, and the cross-platform
  comparison tool.
- `tests/` holds the invariant tests: they use synthetic data, need no dataset
  and no trained model, and run in seconds. `pytest tests/ -v`.
- `results/analysis/` holds artifacts derived after the experiment finished,
  kept separate from the original results in the root of `results/`.

## What makes a run reproducible

- Dependency versions pinned in `requirements.txt`, verified by
  `scripts/verify_environment.py`.
- One seed, 42, applied to Python, NumPy, PyTorch and CUDA through
  `src/utils/seeds.py`, with the cuDNN determinism flags set.
- SHA-256 dataset hashes in `data/dataset_hashes.json`, recorded into every
  manifest.
- A JSON manifest per experiment in `results/manifests/`: hyperparameters,
  library versions, platform, git commit, metrics, cost, artifact paths.
  The `environment` block (operating system, architecture, BLAS backend,
  PyTorch backend, profile) is recorded from the audit onwards; the 66
  manifests committed before it predate the block and are read as the inferred
  reference profile, marked as inferred rather than recorded.
- SHA-256 of the three result tables in `results/CHECKSUMS.txt`, so a clone can
  prove its copy is the one the dissertation cites: `sha256sum -c results/CHECKSUMS.txt`.
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

**A third platform turned the suspected cause into a test.** The pipeline was run
again on 2026-09-01 on Windows 11 (i5-14500, no GPU), natively under Git Bash,
with every pinned version resolving the same and the same four dataset hashes.
Windows shares x86-64 and OpenBLAS with the Linux reference but shares neither
its operating system, its toolchain, nor its execution path for the neural
models, which makes it close to a controlled test of the explanation below:

| Family | Metrics | Identical | Largest gap | Median gap |
|---|---|---|---|---|
| Classical | 90 | 49 (54%) | 0.0120 | 0.0000 |
| Deep, within-dataset | 15 | 0 | 0.0247 | 0.0047 |
| Cross-dataset | 60 | 1 (2%) | 0.0505 | 0.0011 |

The two families separate exactly where that explanation says they should. The
classical family tightened sharply against macOS: the median gap is zero, more
than half of all 90 metrics come back bit-identical, and the worst case fell from
0.043 to 0.0120. The neural family, which ran on CPU here just as it did on
macOS, did not improve at all. Sharing an architecture with the reference helps
precisely where the linear algebra is implicated, and nowhere else.

Nothing moves on Windows either. Cross-dataset F1 landed in 0.301 to 0.507
against the 0.301 to 0.520 reported and AUC in 0.403 to 0.624 against 0.453 to
0.642, while the within-dataset baseline reproduced identically at 0.756 to
0.938. The same single transfer, CNN-LSTM from malicious_urls to Mendeley, still
lands below 0.5 AUC. One cross-dataset run, RandomForest from Mendeley to
malicious_urls, came back bit-identical.

**The cause is now partly isolated, and the remainder is not.** For the neural
models it was always the execution path: CUDA on the reference machine against
CPU elsewhere, which is a different set of kernels and a different reduction
order, not a subtler version of the same computation. Both non-reference
platforms ran them on CPU and both diverged by a similar amount, which is what
that account predicts.

For the classical models the suspected contributor was the linear algebra
underneath, and the shape of that suspicion had to be corrected. All three
platforms run the same BLAS library: under the pinned `numpy==1.26.4`, the PyPI
wheel bundles OpenBLAS on Linux, on Windows and on Apple Silicon alike. NumPy
only began shipping Accelerate-linked wheels in 2.0, and only for macOS 14 and
newer, which the pin excludes. What separates macOS from the other two is
therefore not the library but the build: an arm64 OpenBLAS with ARM kernels and
its own threading against the x86-64 build Linux and Windows share.

Windows is the closest thing here to a test of that, and the prediction held:
same architecture as the reference, gaps collapsing to a median of zero, while
macOS, on a different architecture, stayed an order of magnitude further away.
The corrected account is the tighter one. Two different libraries can differ in
any number of ways; one library built for two instruction sets differs in the
kernels selected and in the order operations are reduced, which is exactly the
kind of difference that moves a final decimal and leaves a conclusion standing.

It is still not a controlled experiment, because the OS, the compiler and libm
all changed alongside the BLAS build, so the honest claim is that the evidence
now favours the explanation rather than settling it. A 0.043 swing in one macOS
recall value remains larger than a pure accumulation-order argument comfortably
explains.

The BLAS backend is not taken on trust anywhere: `scripts/verify_environment.py`
reads it from `numpy.__config__` and prints it in the platform header, so each of
the three runs reports its own.

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
