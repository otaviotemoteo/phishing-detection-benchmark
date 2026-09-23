"""
Cross-platform comparison tool (scripts/compare_platforms.py).

The repository currently holds manifests from one platform only, so the tool
cannot be exercised on real data until the macOS and Windows runs are committed.
These tests drive it with synthetic manifests instead: two platforms, known
metric gaps, and the single-platform case that must not be treated as a failure.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_script():
    """Import scripts/compare_platforms.py, which is a script rather than a module."""
    path = REPO_ROOT / "scripts" / "compare_platforms.py"
    spec = importlib.util.spec_from_file_location("compare_platforms", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_platforms"] = module
    spec.loader.exec_module(module)
    return module


compare_platforms = load_script()


def _timestamp_from(experiment_id: str) -> str:
    """Turn the ``_YYYYmmdd_HHMMSS`` suffix of an id into an ISO timestamp."""
    date, time = experiment_id.split("_")[-2:]
    return f"{date[:4]}-{date[4:6]}-{date[6:]}T{time[:2]}:{time[2:4]}:{time[4:]}-03:00"


def write_manifest(directory: Path, experiment_id: str, metrics: dict, environment=None):
    """Write one synthetic manifest and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": experiment_id,
        "timestamp": _timestamp_from(experiment_id),
        "model": experiment_id.split("_")[0],
        "dataset": experiment_id.split("_")[1],
        "seed": 42,
        "metrics": metrics,
        "cost": {"training_time_s": 1.0},
        "artifacts": {"model_path": f"results/models/{experiment_id}.joblib"},
    }
    if environment is not None:
        manifest["environment"] = environment
    path = directory / f"{experiment_id}.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


LINUX = {
    "os": {"system": "Linux", "release": "6.17.0"},
    "arch": "x86_64",
    "blas": "openblas",
    "gpu_name": "GeForce GTX 1060 3GB",
    "torch_backend": "cuda",
    "profile": "linux-x86_64-cuda",
}
MACOS = {
    "os": {"system": "Darwin", "release": "25.6.0"},
    "arch": "arm64",
    "blas": "accelerate",
    "gpu_name": "Apple Silicon GPU (MPS)",
    "torch_backend": "mps",
    "profile": "macos-arm64-mps",
}

BASE_METRICS = {
    "accuracy": 0.9710669,
    "precision": 0.9699042,
    "recall": 0.9646258,
    "f1": 0.9672578,
    "auc_roc": 0.9968253,
}


@pytest.fixture
def two_platforms(tmp_path):
    """One run recorded on two platforms, with a known 0.002 gap in recall."""
    linux_dir = tmp_path / "linux"
    macos_dir = tmp_path / "macos"
    write_manifest(linux_dir, "RandomForest_uci_20260707_133708", BASE_METRICS, LINUX)
    shifted = {**BASE_METRICS, "recall": BASE_METRICS["recall"] + 0.002}
    write_manifest(macos_dir, "RandomForest_uci_20260820_090000", shifted, MACOS)
    return linux_dir, macos_dir


def test_run_name_drops_the_timestamp():
    manifest = {"experiment_id": "RandomForest_uci_20260707_133708"}

    assert compare_platforms.run_name(manifest) == "RandomForest_uci"


def test_cross_dataset_run_names_survive_the_timestamp_strip():
    manifest = {"experiment_id": "CNN-LSTM_mendeley_to_malicious_urls_20260707_145508"}

    assert compare_platforms.run_name(manifest) == "CNN-LSTM_mendeley_to_malicious_urls"


def test_a_single_platform_is_reported_and_is_not_an_error(tmp_path, capsys):
    directory = tmp_path / "linux"
    write_manifest(directory, "RandomForest_uci_20260707_133708", BASE_METRICS, LINUX)
    out = tmp_path / "comparison.csv"

    exit_code = compare_platforms.main([str(directory), "--out", str(out)])

    assert exit_code == 0
    assert not out.exists(), "nothing to compare, so nothing should be written"
    assert "Only one platform" in capsys.readouterr().out


def test_no_manifests_at_all_is_an_error(tmp_path):
    assert compare_platforms.main([str(tmp_path / "empty"), "--out", str(tmp_path / "o.csv")]) == 1


def test_two_platforms_produce_one_row_each(two_platforms, tmp_path):
    linux_dir, macos_dir = two_platforms
    out = tmp_path / "comparison.csv"

    exit_code = compare_platforms.main([str(linux_dir), str(macos_dir), "--out", str(out)])

    assert exit_code == 0
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2
    assert {row["profile"] for row in rows} == {"linux-x86_64-cuda", "macos-arm64-mps"}
    assert {row["run"] for row in rows} == {"RandomForest_uci"}


def test_the_reported_gap_is_the_absolute_difference(two_platforms, tmp_path):
    linux_dir, macos_dir = two_platforms
    out = tmp_path / "comparison.csv"

    compare_platforms.main([str(linux_dir), str(macos_dir), "--out", str(out)])

    rows = list(csv.DictReader(open(out)))
    for row in rows:
        assert float(row["recall_max_gap"]) == pytest.approx(0.002, abs=1e-6)
        assert float(row["f1_max_gap"]) == pytest.approx(0.0, abs=1e-9)


def test_backend_columns_come_from_the_environment_block(two_platforms, tmp_path):
    linux_dir, macos_dir = two_platforms
    out = tmp_path / "comparison.csv"

    compare_platforms.main([str(linux_dir), str(macos_dir), "--out", str(out)])

    rows = {row["profile"]: row for row in csv.DictReader(open(out))}
    assert rows["linux-x86_64-cuda"]["blas"] == "openblas"
    assert rows["linux-x86_64-cuda"]["torch_backend"] == "cuda"
    assert rows["macos-arm64-mps"]["blas"] == "accelerate"
    assert rows["macos-arm64-mps"]["torch_backend"] == "mps"
    assert rows["linux-x86_64-cuda"]["inferred_profile"] == "False"


def test_a_manifest_without_an_environment_block_is_marked_inferred(tmp_path):
    legacy_dir = tmp_path / "legacy"
    macos_dir = tmp_path / "macos"
    write_manifest(legacy_dir, "RandomForest_uci_20260707_133708", BASE_METRICS, None)
    write_manifest(macos_dir, "RandomForest_uci_20260820_090000", BASE_METRICS, MACOS)
    out = tmp_path / "comparison.csv"

    compare_platforms.main([str(legacy_dir), str(macos_dir), "--out", str(out)])

    rows = {row["profile"]: row for row in csv.DictReader(open(out))}
    assert rows["linux-x86_64-cuda"]["inferred_profile"] == "True"
    assert rows["macos-arm64-mps"]["inferred_profile"] == "False"


def test_repeated_runs_on_one_platform_are_counted_not_compared(tmp_path):
    linux_dir = tmp_path / "linux"
    write_manifest(linux_dir, "RandomForest_uci_20260624_170037", BASE_METRICS, LINUX)
    write_manifest(linux_dir, "RandomForest_uci_20260707_133708", BASE_METRICS, LINUX)
    macos_dir = tmp_path / "macos"
    write_manifest(macos_dir, "RandomForest_uci_20260820_090000", BASE_METRICS, MACOS)
    out = tmp_path / "comparison.csv"

    compare_platforms.main([str(linux_dir), str(macos_dir), "--out", str(out)])

    rows = {row["profile"]: row for row in csv.DictReader(open(out))}
    assert len(rows) == 2
    assert rows["linux-x86_64-cuda"]["runs_on_this_platform"] == "2"


def test_runs_present_on_only_one_platform_are_left_out(two_platforms, tmp_path):
    linux_dir, macos_dir = two_platforms
    write_manifest(linux_dir, "SVM_iscx_20260707_141313", BASE_METRICS, LINUX)
    out = tmp_path / "comparison.csv"

    compare_platforms.main([str(linux_dir), str(macos_dir), "--out", str(out)])

    runs = {row["run"] for row in csv.DictReader(open(out))}
    assert runs == {"RandomForest_uci"}
