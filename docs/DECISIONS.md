# Architecture Decision Records

This document records non-trivial methodological and technical decisions made during the project. Each decision is recorded with context, alternatives considered, and consequences, in the spirit of [ADRs](https://adr.github.io/).

**Conventions:**

- New decisions append to the **bottom** of this file (chronological order).
- IDs are sequential: D-001, D-002, ...
- A decision can be **Proposed**, **Accepted**, or **Superseded by D-XXX**.
- Once Accepted, the decision should not be edited — supersede with a new entry instead.

**Template:** see `DEVELOPMENT.md` §11.2.

---

## D-001: Adopt PyTorch instead of TensorFlow for Deep Learning and Transformers

**Date:** 2026-06-23
**Status:** Accepted
**Phase:** 0

### Context

The original Planejamento document listed TensorFlow/Keras as the Deep Learning framework. However, the Hugging Face `transformers` library (which will be used for DistilBERT fine-tuning) is PyTorch-first, with TensorFlow as a second-class citizen in terms of documentation, examples, and community support. The recent literature on phishing detection with DL/Transformers (2024–2026) is also predominantly written in PyTorch.

Additionally, the hardware constraint (GTX 1060 with 3 GB VRAM) requires explicit and granular control over memory allocation — gradient checkpointing, mixed precision, manual `.to(device)` calls — which is more straightforward in PyTorch than in TF/Keras.

### Decision

Use PyTorch (2.4.1+cu121) as the unified deep learning framework for both CNN/LSTM/CNN-LSTM (Layer 2) and DistilBERT fine-tuning (Layer 3). Classical ML (Layer 1) remains in scikit-learn.

### Alternatives considered

- **TensorFlow/Keras:** original choice. Rejected due to weaker Hugging Face integration and less granular memory control.
- **JAX:** rejected as it has a steeper learning curve and a smaller ecosystem for transformer fine-tuning.
- **Mixed (TF for CNN/LSTM, PyTorch for Transformers):** rejected to avoid maintaining two parallel codebases.

### Consequences

**Positive:**
- Direct alignment with Hugging Face documentation and modern phishing-DL literature.
- More explicit VRAM management, critical for the 3 GB GPU.
- A single deep learning stack to maintain.

**Negative:**
- The Planejamento document mentions TF/Keras and must be updated, or the discrepancy explained in the dissertation methodology.
- Slightly steeper learning curve for someone primarily familiar with Keras.

### References

- `DEVELOPMENT.md` §2.2 (stack rationale)
- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers

---

## D-002: Use MLflow for experiment tracking instead of Weights & Biases

**Date:** 2026-06-23
**Status:** Accepted
**Phase:** 0

### Context

Experiment tracking is required to ensure reproducibility and to compare results across the multiple model–dataset combinations (18+ experiments expected). Two mainstream tools were evaluated: MLflow and Weights & Biases (W&B).

### Decision

Use MLflow 2.16.2 as the experiment tracking tool, running the local backend (`./mlruns` directory).

### Alternatives considered

- **Weights & Biases (W&B):** more polished UI, but requires a cloud account and sends experiment data to external servers. Rejected to keep the project fully local and avoid institutional privacy concerns.
- **No tracking, manual CSV logging:** rejected — does not scale beyond ~5 experiments and provides no run-level artifacts or diff capabilities.

### Consequences

**Positive:**
- 100% local execution, aligned with the project's privacy posture.
- Open source, no external dependencies.
- Direct integration with sklearn, PyTorch, and Hugging Face.
- The `mlruns/` directory can be archived as part of the dissertation supplementary materials.

**Negative:**
- UI is less polished than W&B.
- Requires running `mlflow ui` locally to inspect results, vs. W&B's always-available web dashboard.

### References

- `DEVELOPMENT.md` §2.2
- MLflow documentation: https://mlflow.org/

---

## D-003: Mendeley dataset selection — n96ncsr5g4/1 (URL + label)

**Date:** 2026-06-23
**Status:** Accepted
**Phase:** 1

### Context

The Planejamento (source of truth) specifies the Mendeley dataset `n96ncsr5g4/1`,
required because the experimental design (§7.1) fine-tunes DistilBERT on text
content. The committed `data/README.md` previously referenced `c2gw7fy2j4`
(58,645 samples, ~100 pre-extracted numeric features), which is incorrect: it
cannot supply raw text for the Transformer layer and mismatches the Planejamento.

Phase 1 EDA (`notebooks/01_eda.ipynb`) downloaded `n96ncsr5g4/1` and confirmed its
actual schema directly from the source, resolving the discrepancy.

### Decision

Adopt Mendeley **`n96ncsr5g4/1`** ("Phishing Websites Dataset", DOI
10.17632/n96ncsr5g4.1). Confirmed schema:

- **80,000 instances**, labelled `result` = 0 (legitimate, 50,000) / 1 (phishing,
  30,000) — ~37.5% phishing.
- The published download is a single `index.sql` (~10 MB) with columns
  `rec_id, url, website, result, created_date`, ingested to
  `data/mendeley_phishing.csv` by `scripts/convert_datasets.py`.
- `website` is the *filename* of the captured HTML page (e.g.
  `1635698138155948.html`). **The HTML page content is NOT part of the public
  download** — only the URL index ships.

### Alternatives considered

- `c2gw7fy2j4` (58,645 samples, ~100 extracted features): rejected — lacks raw
  text for the Transformer layer; mismatches the Planejamento.

### Consequences

**Positive:**
- Provides 80k raw URLs + labels, directly usable for URL feature engineering
  (Layer 1), character-level CNN/LSTM (Layer 2), and DistilBERT on URL text
  (Layer 3). Sample count and label semantics match the Planejamento.

**Negative / caveats:**
- HTML *content* is unavailable, so HTML-body modelling is out of scope unless the
  pages are separately crawled (not planned). DistilBERT therefore consumes the
  **URL string as text**, consistent with DEVELOPMENT.md §4.4 (raw URL via embedding).
- `data/README.md` was corrected to reflect the real dataset (was 58,645 / `c2gw7fy2j4`).

### References

- Planejamento §3, §7.1; DEVELOPMENT.md §4.4
- Source: https://data.mendeley.com/datasets/n96ncsr5g4/1 (DOI 10.17632/n96ncsr5g4.1)
- Schema confirmed in `notebooks/01_eda.ipynb` §2; ingestion in
  `scripts/convert_datasets.py`; hashes in `data/dataset_hashes.json`.

---

## D-004: Positive class = phishing (label 1); normalize all dataset labels to {0,1}

**Date:** 2026-06-23
**Status:** Accepted
**Phase:** 2

### Context

Metrics must be comparable across datasets, but the raw label encodings differ:
UCI uses `Result` ∈ {1 = legitimate, -1 = phishing}, while Mendeley uses
{0 = legitimate, 1 = phishing}. Precision, recall, and F1 are asymmetric — they
depend on which class is "positive" — so a single, explicit convention is required
before any model is trained. DEVELOPMENT.md §8.3 also argues that false negatives
(missed phishing) are the costlier error, so **recall of the phishing class** is a
headline metric and must be unambiguous.

### Decision

Normalize every dataset's label to **{0 = legitimate, 1 = phishing}** and treat
**phishing (1) as the positive class** in all metric computations
(`pos_label=1`). UCI is remapped `-1 → 1` and `1 → 0` in
`src.data.preprocessing.to_xy`; Mendeley already matches. With this convention,
*recall* = fraction of phishing correctly caught and *precision* = fraction of
phishing alarms that were real.

### Alternatives considered

- Keep each dataset's native encoding: rejected — metrics would silently mean
  different things per dataset, breaking comparability.
- Positive class = legitimate: rejected — inverts recall away from the
  attack-detection framing the dissertation emphasizes (§8.3).

### Consequences

- All `metrics_ml.csv` / manifest precision/recall/F1 values describe the phishing
  class consistently across datasets and model families.
- `to_xy` is the single point where label semantics are fixed; any new dataset
  must map into this convention there.

### References

- DEVELOPMENT.md §8.1–§8.3 (metric definitions, FN cost)
- `src/data/preprocessing.py` (`to_xy`), `src/evaluation/metrics.py`

---

## D-005: ISCX-URL2016 is the lexical-feature export (not raw URLs) — cross-dataset impact

**Date:** 2026-06-24
**Status:** Proposed
**Phase:** 1 (finding) / 6 (decision to finalize)

### Context

ISCX-URL2016 was obtained via the UNB CIC registration form. The export in hand is
the **lexical-feature** version: 36,707 rows × **79 numeric URL features** plus a
5-class label `URL_Type_obf_Type` (benign 7,781 / phishing 7,586 / Defacement 7,930
/ malware 6,712 / spam 6,698). It also carries ~26% duplicate rows and NaNs in 9
columns.

The Planejamento (§3, §10) described ISCX as "classified raw URLs" and the
cross-dataset generalization test trains RF/XGBoost on **UCI (30 features)** and
tests on **ISCX**. But ISCX's 79 lexical features are a *different feature space*
from UCI's 30 structured features, so a model fit on UCI cannot be applied to ISCX
directly. Confirmed in `notebooks/01_eda.ipynb` §3.

### Decision

(Proposed — finalize at Phase 6.) Do **not** attempt UCI→ISCX in raw feature space.
Preferred approach: obtain the ISCX **raw URL lists** and apply the project's own
shared URL feature engineering (`src/data/feature_engineering.py`, Phase 3/4) to
**both** Mendeley and ISCX, producing a common representation in which cross-dataset
transfer is meaningful. For any within-ISCX use, binarize as **phishing vs benign**
(dropping Defacement/malware/spam) to match the phishing-vs-legitimate framing
(D-004); NaNs handled by median impute at preprocessing.

### Alternatives considered

- Train/test across mismatched feature spaces (UCI 30 ↔ ISCX 79): rejected —
  not well-defined; would produce meaningless transfer results.
- Use ISCX's 79 lexical features as the shared schema and re-extract them for UCI/
  Mendeley: rejected for now — re-implementing ISCX's exact feature extractor is
  brittle; our own URL feature set (§4.4) is simpler and already planned.
- Keep this feature export and use ISCX within-dataset only (no cross-dataset):
  fallback if raw URL lists prove hard to obtain.

### Consequences

- The §10 cross-dataset experiment depends on a shared URL-feature representation
  built in Phase 3/4, not on this feature CSV.
- This export remains usable for a within-ISCX phishing-vs-benign benchmark.
- May require a second ISCX download (raw URL lists) before Phase 6.

### References

- Planejamento §3, §10; DEVELOPMENT.md §4.4 (URL feature engineering)
- Schema confirmed in `notebooks/01_eda.ipynb` §3; hash in `data/dataset_hashes.json`

---
