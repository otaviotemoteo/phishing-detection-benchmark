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

## D-006: Classical ML protocol — leakage-safe tuning, per-dataset features, SVM cap

**Date:** 2026-06-24
**Status:** Accepted
**Phase:** 3

### Context

Phase 3 runs 6 models × 3 datasets with hyperparameter tuning. Several protocol
choices affect validity and feasibility and must be fixed once for all 18 runs.

### Decision

1. **Leakage-safe tuning.** Each experiment is a single `imblearn.pipeline.Pipeline`
   — `SimpleImputer(median) → StandardScaler → SMOTE → estimator` — wrapped in
   `RandomizedSearchCV` (`n_iter=20`, `StratifiedKFold(5)`, `scoring="f1"`). Because
   impute/scale/SMOTE live inside the pipeline, they are refit per CV fold on
   training data only — never leaking across folds or into test (§6.2). SMOTE, a
   sampler, is auto-skipped at predict time.
2. **Splits.** CV-tune on the **70% train**; evaluate once on the **15% test**. The
   **15% val is reserved for Phase 4** DL early-stopping (kept identical across phases).
3. **Per-dataset features** (within-dataset benchmark): UCI uses its 30 structured
   features; Mendeley uses lexical URL features (`src/data/feature_engineering.py`);
   ISCX uses its 79 lexical features. Cross-dataset alignment stays a Phase 6 problem (D-005).
4. **ISCX = phishing vs benign.** Binarize by keeping only `phishing` (1) and
   `benign` (0) rows (~15k), dropping Defacement/malware/spam — so all datasets mean
   "phishing vs legitimate" (finalizes the open part of D-005). `inf` ratio values
   are mapped to NaN and median-imputed.
5. **SVM subsample cap.** RBF-SVM trains on a stratified ≤15,000-row sample on large
   datasets (full data for UCI) to keep O(n²) training feasible; the cap is recorded
   in each manifest (`cost.train_subsample_n`).

### Alternatives considered

- `sklearn.pipeline.Pipeline` for tuning: rejected — applies SMOTE during transform
  on CV-validation folds, leaking synthetic samples (§6.2 names this explicitly).
- ISCX phishing-vs-all-others: rejected — mixes benign with other malware in the
  negative class, breaking comparability with UCI/Mendeley (see D-005).
- SVM on UCI only: rejected — would leave the comparison table incomplete (16/18).

### Consequences

- All 18 experiments are tuned and scored identically and reproducibly.
- The saved artifact per run is the fitted `imblearn` Pipeline (self-contained).
- SVM results on large datasets reflect a capped training set — noted in the dissertation.

### References

- DEVELOPMENT.md §6.2 (leakage), §5 (models); Planejamento §5.1
- `src/experiments/runner.py`, `src/models/classical.py`, `src/data/preprocessing.py`

---

## D-007: Character-level DL input — truncate/pad URLs to 200, vocab from train

**Date:** 2026-06-25
**Status:** Accepted
**Phase:** 4

### Context

The char-level DL models need fixed-length integer sequences. Mendeley URLs range
13–1,641 characters (Phase 1 EDA: median 51, p95 ~136). Padding every sequence to
the max (1,641) would make most of each input padding and waste memory on the
3 GB GPU.

### Decision

Truncate/pad URLs to **`MAX_URL_LENGTH = 200`** characters (covers >95% of Mendeley
URLs with little waste). Build the character vocabulary from **training URLs only**
(`build_char_vocab`), reserving id 0 = PAD and 1 = UNK; unseen characters at
inference map to UNK. Encoding is post-padded (`encode_urls`). Finalizes the
`MAX_URL_LENGTH` "see D-XXX" note in `config.py`.

### Alternatives considered

- Pad to max length (1,641): ~8× the memory for marginal information gain — rejected.
- 100 chars: would truncate ~25% of URLs past their distinguishing tail — rejected.
- Vocab from the full dataset: minor, but fitting on train only keeps the protocol
  leak-free and consistent with the rest of the pipeline.

### Consequences

- Very long phishing URLs are truncated at 200 (uncommon; acceptable trade-off).
- Fixed 200-length keeps VRAM predictable (embed 32, batch 64 → fits in <1 GB).

### References

- Phase 1 EDA URL-length stats (`notebooks/01_eda.ipynb` §2); `src/config.py`
- `src/data/feature_engineering.py` (`build_char_vocab`, `encode_urls`)

---

## D-008: Deep Learning scope = Mendeley only; imbalance via pos_weight (not SMOTE)

**Date:** 2026-06-25
**Status:** Accepted
**Phase:** 4

### Context

Char-level DL consumes raw URL strings. Of the three datasets, only **Mendeley**
has raw URLs — UCI is 30 structured features and our ISCX export is 79 lexical
features (D-005), neither containing URL text. Separately, SMOTE (used for the
classical layer) cannot synthesize meaningful character sequences.

### Decision

1. Run the DL layer (CNN, LSTM, CNN-LSTM) on **Mendeley only**. ISCX DL is deferred
   until raw ISCX URLs are sourced (ties to D-005); UCI is out of scope for DL.
2. Handle Mendeley's class imbalance (50k/30k) with **`pos_weight` in
   `BCEWithLogitsLoss`** (= n_neg/n_pos on train), **not SMOTE**.

### Alternatives considered

- DL on UCI/ISCX numeric features via an MLP: rejected — that is not the
  "learn from the raw URL" story the dissertation contrasts against classical ML.
- SMOTE on token sequences: rejected — interpolating integer char-ids is meaningless.
- Random oversampling of minority URLs: viable, but `pos_weight` is simpler and
  avoids duplicating sequences.

### Consequences

- The DL exit criterion is **3 experiments on Mendeley** (not Mendeley+ISCX).
- DL-vs-classical comparison is made on Mendeley, where both layers have results.
- Sourcing ISCX raw URLs later would unblock both ISCX DL and the Phase 6
  cross-dataset test (D-005).

### References

- D-005 (ISCX schema); Planejamento §6; DEVELOPMENT.md §4.4
- `src/experiments/runner_deep.py`, `src/models/deep.py`

---

## D-009: Skip Phase 5 (DistilBERT fine-tuning) — documented justification

**Date:** 2026-06-25
**Status:** Accepted
**Phase:** 5

### Context

Phase 5 (DistilBERT fine-tuning) is marked **optional/advanced** in both the
Planejamento (§7) and DEVELOPMENT.md §4. Its exit criterion explicitly permits
*"a documented justification of why fine-tuning was not feasible and which
alternative was adopted"* in lieu of running it. It is also **not part of the
required Scientific Initiation written texts** — only the implementation roadmap.

### Decision

Skip DistilBERT fine-tuning. Rationale:

1. **Optional and not in the required texts** — pure enrichment, not needed for the SI.
2. **URLs only, no HTML** (D-003). DistilBERT's strength is natural-language context,
   and its English word-piece tokenizer fragments URL strings awkwardly. On short
   URLs the Phase 4 char-level models already do very well (LSTM F1 **0.937** on
   Mendeley), so the marginal gain from a transformer is uncertain — it might not beat
   the LSTM.
3. **Hardware.** The GTX 1060 (2.9 GB) makes transformer fine-tuning the most
   resource-intensive and OOM-prone phase (the plan even carries a VPS contingency).
4. **Higher-value work remains.** Phase 6 (cross-dataset generalization) and Phase 7
   (final comparison figures) contribute more to the dissertation.

### Alternatives considered

- **Partial / pre-trained DistilBERT** (capped fine-tune): still URLs-only; modest
  expected value for the setup effort.
- **Full fine-tuning:** highest cost and risk for an uncertain win — rejected.

### Consequences

- The model-family comparison is **Classical vs Deep Learning** (two families, not
  three); `metrics_transformers.csv` stays empty.
- Revisitable (supersede this ADR) if full HTML or raw ISCX URLs are obtained and a
  transformer becomes worthwhile.

### References

- D-003 (Mendeley is URL-only); Phase 4 results (`metrics_dl.csv`)
- Planejamento §7; DEVELOPMENT.md §4 (Phase 5 exit criterion)

---

## D-010: Cross-dataset protocol — second dataset, URL normalization, within/cross design

**Date:** 2026-06-25
**Status:** Accepted
**Phase:** 6

### Context

Cross-dataset generalization (train A, test B) requires A and B to share a feature
representation. Only raw-URL datasets can share *our* representation, so UCI (30
discretized features) and the ISCX feature export (D-005) are excluded. We need a
**second raw-URL dataset** alongside Mendeley.

The literal ISCX raw URLs are not in the CIC distribution (feature-only, D-005), so
we use the Kaggle **"Malicious URLs" dataset** (sid321axn; 651k URLs; ISCX-*derived*
among other sources) as `malicious_urls` — the phishing+benign subset (522,214 rows).
It is not literally ISCX but serves the purpose: an independent raw-URL phishing corpus.

**URL-formatting artifact discovered:** Mendeley URLs include the scheme
(`http(s)://`) ~100% of the time; the Kaggle set ~11.5%. Lexical features
(`has_https`, `url_length`, `path_depth`) therefore mean different things across the
two datasets, so *naive* cross-dataset transfer collapses to near-random (DecisionTree
AUC 0.545) for **formatting** reasons, not phishing.

### Decision

1. **Datasets:** Mendeley ↔ `malicious_urls` (both raw URLs). UCI and the ISCX feature
   export are excluded from cross-dataset.
2. **Normalize URLs** — strip the `http(s)://` scheme before feature extraction /
   tokenization (`_normalize_urls`), so transfer measures phishing patterns, not formatting.
3. **Protocol:** train on A's training split (stratified-capped at
   `CROSS_MAX_TRAIN_SAMPLES = 80k`), tuned with the same RandomizedSearchCV as Phase 3.
   Record a **within** baseline (A → A's held-out test) *and* the **cross** result
   (A → all of B) — both normalized, same trained model — so the F1 drop is fair.
   Run both directions for **RF, XGBoost, CNN-LSTM**.
4. **phishing vs benign** binary (D-006 consistency).

### Alternatives considered

- Literal ISCX raw URLs: unavailable from CIC (feature-only, D-005); the Kaggle set is
  the accessible, larger, ISCX-derived substitute.
- No normalization: rejected — measures a formatting artifact, not generalization.
- Comparing cross to the Phase 3/4 raw-feature within numbers: rejected as unfair — the
  in-runner normalized within baseline is the controlled comparison.

### Consequences

- Even after normalization a substantial drop is expected (lexical URL features are
  dataset-specific). The drop magnitude, and the **char-level vs lexical** contrast, are
  the findings (Planejamento §10).
- The within baseline recorded here (normalized, capped) differs slightly from the
  Phase 3/4 within results (raw, full) — it exists specifically for the cross comparison.

### References

- D-005 (ISCX schema), D-006 (phishing-vs-benign); Planejamento §10
- `src/experiments/runner_cross.py`; scheme diagnostic in `notebooks/05_crossdataset.ipynb`

---
