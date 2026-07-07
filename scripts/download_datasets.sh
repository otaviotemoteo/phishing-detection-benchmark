#!/usr/bin/env bash
# =============================================================================
# Phishing Detection Benchmark — Dataset Downloader
# =============================================================================
# Fetches the four datasets into ./data/ and converts each to a flat CSV.
# Idempotent: a dataset whose CSV already exists is skipped.
#
#   UCI Phishing Websites  — auto (zip -> ARFF -> uci_phishing.csv)
#   Mendeley n96ncsr5g4/1  — auto (index.sql -> mendeley_phishing.csv)
#   Malicious URLs (Kaggle)— auto (anonymous Kaggle API zip -> malicious_urls.csv)
#   ISCX-URL2016           — MANUAL: UNB CIC distributes it behind a
#                            registration form, so it cannot be fetched
#                            non-interactively. The script prints instructions.
#
# After downloading it (re)generates data/dataset_hashes.json.
#
# Usage:
#   bash scripts/download_datasets.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$REPO_ROOT/data"
mkdir -p "$DATA_DIR"

# Prefer the project venv's Python so the script works without manual activation.
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

UCI_CSV="$DATA_DIR/uci_phishing.csv"
MENDELEY_CSV="$DATA_DIR/mendeley_phishing.csv"
ISCX_CSV="$DATA_DIR/iscx_url2016.csv"
MALICIOUS_CSV="$DATA_DIR/malicious_urls.csv"

UCI_URL="https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"
MENDELEY_URL="https://data.mendeley.com/public-files/datasets/n96ncsr5g4/files/dac80106-cc68-43c3-8810-96408c09fbbc/file_downloaded"
# Anonymous download endpoint for public Kaggle datasets (no account needed).
MALICIOUS_URL="https://www.kaggle.com/api/v1/datasets/download/sid321axn/malicious-urls-dataset"

echo "============================================================"
echo "Dataset download  (data dir: $DATA_DIR)"
echo "============================================================"

# -----------------------------------------------------------------------------
# UCI Phishing Websites
# -----------------------------------------------------------------------------
if [[ -f "$UCI_CSV" ]]; then
  echo "[UCI] present, skipping: $UCI_CSV"
else
  echo "[UCI] downloading..."
  tmpzip="$DATA_DIR/_uci_phishing.zip"
  tmpdir="$DATA_DIR/_uci_extract"
  curl -fSL --retry 3 -o "$tmpzip" "$UCI_URL"
  rm -rf "$tmpdir" && mkdir -p "$tmpdir"
  "$PYTHON" -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$tmpzip" "$tmpdir"
  arff="$(find "$tmpdir" -iname '*.arff' | head -1)"
  [[ -n "$arff" ]] || { echo "[UCI] ERROR: no .arff found in archive" >&2; exit 1; }
  "$PYTHON" "$SCRIPT_DIR/convert_datasets.py" uci "$arff" "$UCI_CSV"
  rm -rf "$tmpzip" "$tmpdir"
fi

# -----------------------------------------------------------------------------
# Mendeley Phishing Websites Dataset (n96ncsr5g4, version 1)
# -----------------------------------------------------------------------------
if [[ -f "$MENDELEY_CSV" ]]; then
  echo "[Mendeley] present, skipping: $MENDELEY_CSV"
else
  echo "[Mendeley] downloading index.sql (~10 MB)..."
  tmpsql="$DATA_DIR/_mendeley_index.sql"
  curl -fSL --retry 3 -o "$tmpsql" "$MENDELEY_URL"
  "$PYTHON" "$SCRIPT_DIR/convert_datasets.py" mendeley "$tmpsql" "$MENDELEY_CSV"
  rm -f "$tmpsql"
fi

# -----------------------------------------------------------------------------
# Malicious URLs (Kaggle sid321axn/malicious-urls-dataset) — Phase 6 cross-dataset
# corpus (D-010). Filtered to phishing+benign with schema (url, result) by the
# converter, matching Mendeley.
# -----------------------------------------------------------------------------
if [[ -f "$MALICIOUS_CSV" ]]; then
  echo "[malicious_urls] present, skipping: $MALICIOUS_CSV"
else
  raw="$DATA_DIR/malicious_phish.csv"
  if [[ ! -f "$raw" ]]; then
    echo "[malicious_urls] downloading from Kaggle (anonymous public API)..."
    tmpzip="$DATA_DIR/_malicious_urls.zip"
    tmpdir="$DATA_DIR/_malicious_extract"
    if curl -fSL --retry 3 -o "$tmpzip" "$MALICIOUS_URL"; then
      rm -rf "$tmpdir" && mkdir -p "$tmpdir"
      "$PYTHON" -c "import sys,zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" "$tmpzip" "$tmpdir"
      found="$(find "$tmpdir" -iname 'malicious_phish.csv' | head -1)"
      if [[ -n "$found" ]]; then
        mv "$found" "$raw"
      else
        echo "[malicious_urls] ERROR: malicious_phish.csv not found in the archive" >&2
      fi
      rm -rf "$tmpzip" "$tmpdir"
    else
      rm -f "$tmpzip"
      echo "[malicious_urls] WARN: anonymous Kaggle download failed."
    fi
  fi
  if [[ -f "$raw" ]]; then
    "$PYTHON" "$SCRIPT_DIR/convert_datasets.py" malicious_urls "$raw" "$MALICIOUS_CSV"
    rm -f "$raw"
  else
    cat <<EOF

[malicious_urls] MANUAL FALLBACK
  The anonymous Kaggle download did not succeed. To fetch it manually:
    1. Open:  https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
    2. Download and extract the archive (it contains malicious_phish.csv).
    3. Place the raw file at:
         $DATA_DIR/malicious_phish.csv
  Then re-run:  bash scripts/download_datasets.sh
EOF
  fi
fi

# -----------------------------------------------------------------------------
# ISCX-URL2016 (manual — registration-gated)
# -----------------------------------------------------------------------------
if [[ -f "$ISCX_CSV" ]]; then
  echo "[ISCX] present, skipping: $ISCX_CSV"
else
  cat <<EOF

[ISCX] MANUAL STEP REQUIRED
  ISCX-URL2016 is distributed by UNB CIC behind a registration form and cannot
  be downloaded non-interactively.
    1. Open:  https://www.unb.ca/cic/datasets/url-2016.html
    2. Complete the "Download this dataset" form (name, email, organization).
    3. Download the ISCX-URL2016 archive and extract it.
    4. Place the classified-URL CSV at:
         $ISCX_CSV
       (Alternatively, drop the extracted files in data/iscx_url2016/ and they
        will be consolidated during EDA/preprocessing.)
  Then re-run:  bash scripts/download_datasets.sh
EOF
fi

# -----------------------------------------------------------------------------
# Hash registry (reproducibility — DEVELOPMENT.md §6.3)
# -----------------------------------------------------------------------------
echo ""
echo "[hashes] regenerating data/dataset_hashes.json..."
( cd "$REPO_ROOT" && "$PYTHON" -m src.utils.io )

echo ""
echo "============================================================"
echo "Download step complete."
[[ -f "$ISCX_CSV" ]] || echo "NOTE: ISCX-URL2016 still pending (manual step above)."
[[ -f "$MALICIOUS_CSV" ]] || echo "NOTE: malicious_urls still pending (see fallback above)."
echo "============================================================"
