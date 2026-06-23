# Experiment Log

Chronological lab journal. New entries append to the **top** (reverse-chronological order).

Each entry follows the template in `DEVELOPMENT.md` §11.3:

- **What I did** — brief description of the session
- **Results** — concrete outcomes, metrics, observations
- **What worked** — replicable wins
- **What didn't** — bugs, dead ends, surprises
- **Next** — concrete next step

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
