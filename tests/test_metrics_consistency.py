"""
Agreement between the published metric tables and the manifests behind them.

`results/metrics_ml.csv`, `results/metrics_dl.csv` and
`results/metrics_crossdataset.csv` hold the 33 experiments the dissertation
cites. Each row is a rounded view of one manifest in `results/manifests/`, which
keeps full precision. These tests read both and check that the tables were never
edited away from the runs that produced them. Nothing here re-runs an
experiment, and nothing here writes.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = REPO_ROOT / "results" / "manifests"

METRIC_COLUMNS = ("accuracy", "precision", "recall", "f1", "auc_roc")

# The CSVs store four decimals; the manifests store full precision. Half a unit
# in the last written place is the most rounding can account for.
ROUNDING_TOLERANCE = 5e-5

EXPECTED_ROWS = {
    "results/metrics_ml.csv": 18,
    "results/metrics_dl.csv": 3,
    "results/metrics_crossdataset.csv": 12,
}

# An experiment id is the run's name followed by a run timestamp.
_TIMESTAMP = re.compile(r"_\d{8}_\d{6}$")


def read_csv(rel_path: str) -> list[dict]:
    """Read one metrics table as a list of row dicts."""
    with open(REPO_ROOT / rel_path, newline="") as f:
        return list(csv.DictReader(f))


def manifests_for(run_name: str) -> list[dict]:
    """Every manifest written for a run, across the dates it was executed."""
    found = []
    for path in sorted(MANIFESTS_DIR.glob("*.json")):
        stem = path.stem
        if not _TIMESTAMP.search(stem):
            continue
        if _TIMESTAMP.sub("", stem) == run_name:
            found.append(json.loads(path.read_text()))
    return found


def run_name_for(row: dict) -> str:
    """The manifest name a metrics row corresponds to."""
    if "train_dataset" in row:
        return f"{row['model']}_{row['train_dataset']}_to_{row['test_dataset']}"
    return f"{row['model']}_{row['dataset']}"


def all_rows():
    """Yield ``(table_path, row)`` for every row of all three tables."""
    for rel_path in EXPECTED_ROWS:
        for row in read_csv(rel_path):
            yield rel_path, row


@pytest.mark.parametrize("rel_path,expected", sorted(EXPECTED_ROWS.items()))
def test_each_table_has_the_expected_number_of_rows(rel_path, expected):
    assert len(read_csv(rel_path)) == expected


def test_the_three_tables_hold_thirty_three_experiments():
    assert sum(len(read_csv(p)) for p in EXPECTED_ROWS) == 33


def test_every_row_has_at_least_one_manifest():
    missing = [
        f"{rel_path}: {run_name_for(row)}"
        for rel_path, row in all_rows()
        if not manifests_for(run_name_for(row))
    ]
    assert not missing, f"metric rows with no manifest: {missing}"


def test_every_manifest_agrees_with_its_row_within_rounding():
    disagreements = []
    for rel_path, row in all_rows():
        run_name = run_name_for(row)
        for manifest in manifests_for(run_name):
            recorded = manifest["metrics"]
            for column in METRIC_COLUMNS:
                published = float(row[column])
                if abs(published - recorded[column]) > ROUNDING_TOLERANCE:
                    disagreements.append(
                        f"{rel_path} {run_name} {column}: "
                        f"table {published} vs manifest {recorded[column]:.6f} "
                        f"({manifest['experiment_id']})"
                    )
    assert not disagreements, "\n".join(disagreements)


def test_every_published_metric_is_a_probability():
    out_of_range = [
        f"{rel_path} {run_name_for(row)} {column}={row[column]}"
        for rel_path, row in all_rows()
        for column in METRIC_COLUMNS
        if not 0.0 <= float(row[column]) <= 1.0
    ]
    assert not out_of_range, f"metric values outside [0, 1]: {out_of_range}"


def test_every_manifest_metric_is_a_probability():
    out_of_range = []
    for path in sorted(MANIFESTS_DIR.glob("*.json")):
        metrics = json.loads(path.read_text()).get("metrics", {})
        for name, value in metrics.items():
            if isinstance(value, (int, float)) and not 0.0 <= float(value) <= 1.0:
                out_of_range.append(f"{path.name} {name}={value}")
    assert not out_of_range, f"manifest metrics outside [0, 1]: {out_of_range}"


def test_every_row_is_unique():
    for rel_path in EXPECTED_ROWS:
        names = [run_name_for(row) for row in read_csv(rel_path)]
        assert len(names) == len(set(names)), f"duplicate rows in {rel_path}"


def test_manifests_record_the_project_seed():
    seeds = {json.loads(p.read_text()).get("seed") for p in MANIFESTS_DIR.glob("*.json")}
    assert seeds == {42}, f"manifests report seeds {seeds}, expected only 42"


def test_no_manifest_leaks_an_absolute_artifact_path():
    leaked = []
    for path in sorted(MANIFESTS_DIR.glob("*.json")):
        artifacts = json.loads(path.read_text()).get("artifacts") or {}
        for key, value in artifacts.items():
            if isinstance(value, str) and (value.startswith("/") or value[1:3] == ":\\"):
                leaked.append(f"{path.name}:{key} = {value}")
    assert not leaked, f"absolute paths in manifests: {leaked}"
