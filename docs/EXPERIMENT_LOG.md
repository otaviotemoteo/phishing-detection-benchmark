# Experiment Log

Chronological lab journal. New entries append to the **top** (reverse-chronological order).

Each entry follows the template in `DEVELOPMENT.md` §11.3:

- **What I did** — brief description of the session
- **Results** — concrete outcomes, metrics, observations
- **What worked** — replicable wins
- **What didn't** — bugs, dead ends, surprises
- **Next** — concrete next step

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
