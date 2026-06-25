# Experiment Log

Chronological lab journal. New entries append to the **top** (reverse-chronological order).

Each entry follows the template in `DEVELOPMENT.md` §11.3:

- **What I did** — brief description of the session
- **Results** — concrete outcomes, metrics, observations
- **What worked** — replicable wins
- **What didn't** — bugs, dead ends, surprises
- **Next** — concrete next step

---

## 2026-06-25 — Phase 4 (Deep Learning)

### What I did

Built the character-level Deep Learning layer: tokenization, three PyTorch
architectures (CNN, LSTM, CNN-LSTM), a GPU training runner (early stopping + mixed
precision), and ran all three on Mendeley.

### Results

- New code: char tokenization in `feature_engineering.py` (`build_char_vocab`,
  `encode_urls`), `src/models/deep.py` (CharCNN/CharLSTM/CharCNNLSTM),
  `src/experiments/runner_deep.py` (torch training loop) + `run_deep.py`,
  `save_training_curve` in `plots.py`.
- **3/3 trained on full 80k Mendeley in 6.7 min** on the GTX 1060 (fp16, batch 64,
  vocab 81, max_len 200); early stopping fired for all (13–15 epochs); GPU peaked
  ~650 MB of 2.9 GB.
- Test results: **LSTM F1 0.937 / AUC 0.991** (227 s), CNN-LSTM 0.932 / 0.988 (88 s),
  CNN 0.923 / 0.987 (78 s). Params 23k–94k.
- **Headline: all three DL models beat all six classical models on Mendeley** —
  DL ~0.92–0.94 F1 vs classical best (XGBoost) 0.857. Char-level learning is ≈ +8 F1
  points over hand-crafted lexical URL features.
- **D-007** (truncate@200, vocab from train) + **D-008** (Mendeley-only DL,
  `pos_weight` not SMOTE) recorded. `metrics_dl.csv` + 3 manifests/CM/ROC/training
  curves; notebook 04 reads + visualizes (0 errors).

### What worked

- The 1k-subset smoke test validated the whole pipeline in ~7 s before the full run.
- The evaluation/saving half (metrics, plots, manifests, cost) was reused verbatim —
  only the training loop was new, exactly as planned.
- Mixed precision + a small embedding kept the 3 GB GPU comfortable.

### What didn't

- LSTM is ~3× slower than CNN (227 s vs 78 s) for a ~1.5-point F1 gain — a clear
  cost-benefit point for the dissertation.
- The CNN-LSTM hybrid did **not** beat plain LSTM here (contrary to Alshingiti 2023) —
  a finding worth discussing.

### Next

- Phase 5 (optional): DistilBERT fine-tuning on Mendeley URLs — memory-constrained on
  3 GB (batch 8, max_seq 128, fp16, capped dataset), or document why it's skipped.
- ISCX DL + cross-dataset still pending raw ISCX URLs (D-005/D-008).

---

## 2026-06-24 — Phase 3 (classical ML complete)

### What I did

Built and ran the full classical ML benchmark — **6 models × 3 datasets = 18 tuned
experiments**. Added URL feature engineering, the model factory, leakage-safe tuning,
and the `run_classical` orchestrator.

### Results

- New code: `src/data/feature_engineering.py` (13 lexical URL features),
  `src/models/classical.py` (6-model factory + grids), `src/experiments/run_classical.py`
  (CLI). `runner.py` refactored to an `imblearn` Pipeline (impute→scale→SMOTE→model)
  wrapped in `RandomizedSearchCV` (5-fold, F1); added `upsert_metrics_row` +
  `save_feature_importance`.
- **18/18 succeeded in 39.5 min** (CPU). `metrics_ml.csv` complete; 18 confusion
  matrices + ROC + manifests; 12 feature-importance plots (DT/RF/XGB/CatBoost × 3).
- Best F1 per dataset: **UCI** RandomForest 0.967 (AUC 0.997); **Mendeley**
  XGBoost/CatBoost 0.857 (AUC 0.96); **ISCX** XGBoost 0.987 (AUC 0.999).
- Tree ensembles (RF/XGB/CatBoost) consistently beat DT and LR; XGBoost the most
  consistent top performer.
- **Mendeley is clearly hardest** (~0.86 F1) — lexical URL features are weaker than
  UCI's structured features and ISCX's 79-feature set. A real, discussable finding.
- Protocol (D-006): ISCX = phishing-vs-benign; SVM training capped at 15k on Mendeley
  (recorded in manifest); val split reserved for Phase 4.

### What worked

- Smoke-testing DecisionTree on all 3 datasets first caught two bugs before the long
  run: ISCX `inf` values (→ NaN→median impute) and a `--all` CLI flag mismatch.
- The `imblearn` Pipeline unifies impute/scale/SMOTE inside CV — no leakage — and the
  saved artifact is a self-contained pipeline.
- The one-line model swap (§1.2) held: the same runner drove all six models.

### What didn't

- First full-run attempt no-op'd on an unrecognized `--all` flag (added the flag, re-ran).
- SVM with `probability=True` is slow; the 15k Mendeley cap kept it feasible (~5 min).

### Next

- **Phase 4 — Deep Learning** (CNN / LSTM / CNN-LSTM, PyTorch) on the raw URL character
  sequence; char-level tokenization; GTX 1060 3 GB VRAM constraints apply. Uses the
  reserved val split for early stopping.
- ISCX raw-URL acquisition remains open for the Phase 6 cross-dataset test (D-005).

---

## 2026-06-24 — ISCX-URL2016 integrated

### What I did

Integrated ISCX-URL2016 (downloaded manually via the UNB CIC form). Relocated it into
`data/`, refreshed hashes, profiled it in `01_eda.ipynb`, and recorded the schema
finding (**D-005**).

### Results

- `data/iscx_url2016.csv`: 36,707 × 80 — **79 lexical URL features** + a 5-class
  `URL_Type_obf_Type` label (benign / phishing / Defacement / malware / spam;
  phishing ~20.7%). ~26% duplicate rows; NaNs in 9 columns.
- All three datasets now hashed; `available_datasets()` == `[uci, mendeley, iscx]`.
- `01_eda.ipynb` §3 now profiles ISCX (multi-class distribution plot saved); 0 errors.
- **D-005** recorded: this is the *feature export*, not raw URLs; its 79-feature space
  ≠ UCI's 30, so the §10 cross-dataset plan needs a shared URL-feature representation
  (raw URLs + our own feature engineering), to be finalized at Phase 6.

### What worked

- The file arrived as a plain CSV → `load_raw("iscx")` works with no converter.

### What didn't

- It is the lexical-FEATURE version, not the raw URLs the Planejamento assumed — so
  UCI(30-feat)→ISCX(79-feat) transfer is not directly possible (surfaced as D-005).
- The file initially landed at the repo root (would have been committed); moved into
  `data/` (gitignored).

### Next

- Unchanged: Phase 3 (classical ML line-up + URL feature engineering). That URL-feature
  work also unblocks the preferred D-005 cross-dataset approach.
- Before Phase 6: obtain the ISCX **raw URL lists** (a second download) if pursuing the
  preferred D-005 option.

---

## 2026-06-23 — Phase 2 (minimal pipeline)

### What I did

Built the end-to-end minimal pipeline (Phase 2): preprocessing, metrics, cost
tracking, plots, manifests, and a single `train_and_evaluate` orchestrator, then ran
Logistic Regression on UCI through it via notebooks `02_preprocessing` + `03_ml_classical`.

### Results

- New modules: `src/data/preprocessing.py` (`to_xy` label-normalization per **D-004**,
  stratified 70/15/15 `split_data`, `apply_smote` on train, `StandardScaler`),
  `src/evaluation/{metrics,cost,plots}.py`, `src/utils/manifests.py`,
  `src/experiments/runner.py` (`train_and_evaluate`), plus `io.append_metrics_row`.
- **LogisticRegression on UCI:** accuracy 0.936, precision 0.931, recall 0.924,
  F1 0.928, AUC 0.981 — consistent with literature LR baselines (~0.92–0.93).
- One call produced: a `metrics_ml.csv` row, a JSON manifest (dataset_hash matches the
  registry, git commit, seed, library versions, full cost block), confusion matrix + ROC
  at 300 DPI, and a model+scaler `.joblib`. An MLflow run was logged to `./mlruns`.
- Both notebooks execute top-to-bottom with 0 errors; re-runs give identical seeded metrics.
- **D-004** recorded (positive class = phishing = 1; UCI `Result` remapped).

### What worked

- Smoke-testing `train_and_evaluate` as a plain function before wrapping it in notebooks —
  the one-line model swap (§1.2) holds.
- Scaling-before-SMOTE keeps the synthetic-sample k-NN on comparable scales; the
  leakage assertions (disjoint splits, train-only scaler/SMOTE) all pass.

### What didn't

- `metrics_ml.csv` uses append semantics, so re-running `03` adds duplicate UCI rows; a
  one-row-per-(model, dataset) dedupe is deferred to Phase 3 when the table fills to 18 rows.
- `LogisticRegression`'s `multi_class` shows a sklearn deprecation marker in `get_params` —
  harmless, recorded verbatim in the manifest.

### Next

- **Phase 3:** implement the 6-model factory in `src/models/classical.py` (DT, RF, XGBoost,
  CatBoost, LogReg, SVM), add GridSearch/RandomizedSearch tuning on the reserved validation
  split, generate feature-importance plots, and fill `metrics_ml.csv` to 18 rows.
- Build `src/data/feature_engineering.py` (URL features) so Mendeley/ISCX work for classical
  ML — `to_xy` currently raises `NotImplementedError` for them.
- ISCX still pending manual download (parallel errand).

---

## 2026-06-23 — Phase 1 (EDA)

### What I did

Executed Phase 1 (Exploratory Data Analysis). Built dataset download/ingestion
tooling, auto-downloaded the UCI and Mendeley datasets, profiled both in
`notebooks/01_eda.ipynb`, and resolved the Mendeley dataset discrepancy (D-003).

### Results

- New reusable code: `src/utils/io.py` (chunked SHA256 + `python -m src.utils.io`
  → `data/dataset_hashes.json`), `src/data/loaders.py` (`load_raw`), and ingestion
  scripts `scripts/convert_datasets.py` + `scripts/download_datasets.sh`, wired into
  `scripts/run_all.sh`.
- **UCI:** 11,055 × 30 features (values in {-1,0,1}), 0 missing; 6,157 legitimate
  (`Result=1`) vs 4,898 phishing (`Result=-1`, ~44%); 5,206 exact-duplicate rows
  (expected — coarse discretization collapses distinct sites).
- **Mendeley n96ncsr5g4/1:** 80,000 URLs, 50,000 legitimate / 30,000 phishing
  (~37.5%); URL length 13–1,641 chars (median 51, p95 ~136); 152 duplicate URLs.
  Confirmed the public release ships URL + label only (no HTML content) → D-003 Accepted.
- 6 EDA figures saved to `plots/eda/` at 300 DPI; notebook executes top-to-bottom
  with 0 errors (verified via `nbconvert --execute`).
- **ISCX-URL2016:** pending (registration-gated; manual step documented).

### What worked

- Probing source APIs *before* scripting caught the Mendeley identity/schema issue
  at the source (single `index.sql`, no HTML), avoiding a wrong assumption.
- Building the notebook via `nbformat` → clean, reproducible, headless-executable.
- UCI (11,055) and Mendeley (80,000) row counts matched published specs exactly.

### What didn't

- ISCX-URL2016 can't be automated — UNB CIC gates it behind a form and the old
  direct hosts now redirect to a landing page. Fell back to a guided manual download.
- The Planejamento's "URL + full HTML" for Mendeley was only half-true: the public
  release is URL + HTML *filename*, no page content. Recorded in D-003.

### Next

- Manually fetch ISCX-URL2016 → `data/iscx_url2016.csv`, re-run the downloader + EDA
  (the notebook's ISCX section auto-activates when the file is present).
- Begin Phase 2: `src/data/preprocessing.py` (stratified 70/15/15 split, SMOTE on
  train only) and `src/evaluation/metrics.py`, then the minimal
  LogisticRegression-on-UCI end-to-end pipeline.

---

## 2026-06-23

### What I did

Initial project setup. Created repository structure following `DEVELOPMENT.md` §3. Initialized this experiment log, `DECISIONS.md`, `requirements.txt` with pinned versions, `.gitignore`, and `scripts/verify_environment.py`.

Recorded the first two architectural decisions:

- **D-001:** Adopt PyTorch instead of TensorFlow for DL/Transformers.
- **D-002:** Use MLflow for experiment tracking.

### Results

- Repository skeleton in place
- Two ADRs documented
- Verification script ready to run after `pip install -r requirements.txt`

### What worked

- Structuring documentation up front before writing any model code
- Pinning all dependencies at creation time (no `>=`, no floating versions)

### What didn't

- N/A — first session

### Next

- Run `pip install -r requirements.txt` in a fresh venv
- Run `python scripts/verify_environment.py` and confirm all checks pass
- Begin Phase 1: download datasets, compute hashes, start EDA notebook

---
