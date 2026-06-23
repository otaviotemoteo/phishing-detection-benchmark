# Datasets

This directory holds the three datasets used in the experiments. **Raw CSV files are not committed to Git** (see `.gitignore`); only the hash registry and this README are tracked.

## Datasets used

| Dataset | Samples | Features | Source |
|---|---|---|---|
| UCI Phishing Websites | 11,055 | 30 | https://archive.ics.uci.edu/ml/datasets/phishing+websites |
| Mendeley Phishing Dataset | 58,645 | ~100 | https://data.mendeley.com/datasets/c2gw7fy2j4/3 |
| ISCX-URL2016 | ~36,000 | URL string | https://www.unb.ca/cic/datasets/url-2016.html |

## Expected filenames after download

```
data/
├── uci_phishing.csv
├── mendeley_phishing.csv
├── iscx_url2016.csv
└── dataset_hashes.json    (auto-generated, committed)
```

## Generating dataset hashes

After downloading, compute and save SHA-256 hashes:

```bash
python -m src.utils.io  # (to be implemented — generates data/dataset_hashes.json)
```

These hashes are recorded in every experiment manifest to guarantee that results are tied to the exact dataset version used.

## PhishTank exclusion

Although the literature often mentions PhishTank, it is **deliberately excluded** from this work's experimental design (see §5.1 of the dissertation and D-XXX in `DECISIONS.md`). PhishTank is a continuously-updated repository, making cross-study comparison and bitwise reproducibility infeasible.
