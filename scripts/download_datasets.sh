#!/usr/bin/env bash
# =============================================================================
# Phishing Detection Benchmark — Dataset Downloader
# =============================================================================
# Fetches the three datasets into ./data/ and converts each to a flat CSV.
# Idempotent: a dataset whose CSV already exists is skipped.
#
#   UCI Phishing Websites  — auto (zip -> ARFF -> uci_phishing.csv)
#   Mendeley n96ncsr5g4/1  — auto (index.sql -> mendeley_phishing.csv)
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

UCI_URL="https://archive.ics.uci.edu/static/public/327/phishing+websites.zip"
MENDELEY_URL="https://data.mendeley.com/public-files/datasets/n96ncsr5g4/files/dac80106-cc68-43c3-8810-96408c09fbbc/file_downloaded"

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
echo "============================================================"
