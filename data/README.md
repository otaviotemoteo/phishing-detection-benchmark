# Datasets

This directory holds the four datasets used in the experiments. **Raw CSV files are not committed to Git** (see `.gitignore`); only the hash registry and this README are tracked.

## Datasets used

| Dataset | Samples | Features | Source |
|---|---|---|---|
| UCI Phishing Websites | 11,055 | 30 structured (±1) | https://archive.ics.uci.edu/dataset/327/phishing+websites |
| Mendeley Phishing Websites Dataset | 80,000 | raw URL (+ HTML filename) | https://data.mendeley.com/datasets/n96ncsr5g4/1 |
| ISCX-URL2016 | 36,707 | 79 lexical URL features (5-class label) | https://www.unb.ca/cic/datasets/url-2016.html |
| Malicious URLs (Kaggle) | 522,214 after filtering | raw URL + binary label | https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset |

> **Mendeley note (see D-003):** the published download is a single `index.sql`
> with columns `rec_id, url, website, result, created_date` (80,000 rows; `result`
> 0 = legitimate / 1 = phishing). `website` is the captured page's *filename* —
> the HTML *content* is not distributed. `scripts/convert_datasets.py` converts the
> SQL dump to `mendeley_phishing.csv`.
>
> **ISCX note (see D-005):** UNB CIC distributes ISCX-URL2016 behind a registration
> form, so it is **not** auto-downloaded — `scripts/download_datasets.sh` prints the
> manual steps. The export in use is the *lexical-feature* version (79 numeric features
> + a 5-class `URL_Type_obf_Type` label), **not** raw URLs; this affects the
> cross-dataset plan (D-005). It loads directly via `load_raw("iscx")` (already a CSV,
> no converter needed).
>
> **Malicious URLs note (see D-010):** the second raw-URL corpus for the Phase 6
> cross-dataset test (ISCX-derived, among other sources). Auto-downloaded anonymously
> via Kaggle's public API by `scripts/download_datasets.sh`; the converter keeps only
> the `phishing` (1) and `benign` (0) classes from the original 651k rows (dropping
> defacement/malware) and writes the Mendeley-compatible schema `url, result`.

## Expected filenames after download

```
data/
├── uci_phishing.csv
├── mendeley_phishing.csv
├── iscx_url2016.csv
├── malicious_urls.csv
└── dataset_hashes.json    (auto-generated, committed)
```

## Generating dataset hashes

After downloading, compute and save SHA-256 hashes:

```bash
python -m src.utils.io  # generates data/dataset_hashes.json
```

This runs automatically at the end of `scripts/download_datasets.sh`.

These hashes are recorded in every experiment manifest to guarantee that results are tied to the exact dataset version used.

## PhishTank exclusion

Although the literature often mentions PhishTank, it is **deliberately excluded** from this work's experimental design. PhishTank is a continuously-updated live feed rather than a versioned snapshot: two downloads on different days yield different data, making cross-study comparison and the bitwise reproducibility this project requires (hash-pinned datasets recorded in every manifest) infeasible.
