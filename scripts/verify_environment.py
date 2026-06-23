#!/usr/bin/env python3
"""
Environment verification script.

Run this after installing requirements.txt to confirm that the development
environment is correctly set up. The script checks:

  1. Python version (3.11.x required)
  2. All required libraries are installed with the expected versions
  3. PyTorch can detect the GPU and CUDA is available
  4. The repository directory structure exists
  5. Basic write permissions on `results/` and `plots/`

Usage:
    python scripts/verify_environment.py

Exit code:
    0 — all checks passed
    1 — one or more checks failed (details printed to stderr)
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Expected library versions (must match requirements.txt)
EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "sklearn": "1.5.2",  # scikit-learn imports as `sklearn`
    "imblearn": "0.12.4",  # imbalanced-learn imports as `imblearn`
    "xgboost": "2.1.1",
    "catboost": "1.2.7",
    "lightgbm": "4.5.0",
    "torch": "2.4.1",
    "transformers": "4.45.2",
    "matplotlib": "3.9.2",
    "seaborn": "0.13.2",
    "psutil": "6.0.0",
    "mlflow": "2.16.2",
    "joblib": "1.4.2",
}

EXPECTED_DIRS = [
    "data",
    "notebooks",
    "src/data",
    "src/models",
    "src/evaluation",
    "src/utils",
    "results/manifests",
    "results/models",
    "results/confusion_matrices",
    "results/roc_curves",
    "plots/eda",
    "plots/feature_importance",
    "plots/final",
    "scripts",
]


# ---------- ANSI colors (degrade gracefully if not a TTY) ----------
def _supports_color() -> bool:
    return sys.stdout.isatty()


GREEN = "\033[92m" if _supports_color() else ""
RED = "\033[91m" if _supports_color() else ""
YELLOW = "\033[93m" if _supports_color() else ""
RESET = "\033[0m" if _supports_color() else ""


def ok(msg: str) -> None:
    print(f"  {GREEN}OK{RESET}    {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}WARN{RESET}  {msg}")


# ---------- Checks ----------
def check_python_version() -> bool:
    print("\n[1/5] Python version")
    major, minor = sys.version_info[:2]
    if major == 3 and minor == 11:
        ok(f"Python {sys.version.split()[0]}")
        return True
    fail(f"Python 3.11.x required, found {major}.{minor}")
    return False


def check_libraries() -> bool:
    print("\n[2/5] Library versions")
    all_ok = True
    for lib, expected in EXPECTED_VERSIONS.items():
        try:
            mod = importlib.import_module(lib)
            actual = getattr(mod, "__version__", "unknown")
            if actual == expected:
                ok(f"{lib} {actual}")
            else:
                warn(f"{lib} {actual} (expected {expected})")
                # Version mismatch is a warning, not a failure — but record it
        except ImportError:
            fail(f"{lib} not installed")
            all_ok = False
    return all_ok


def check_gpu() -> bool:
    print("\n[3/5] GPU and CUDA")
    try:
        import torch
    except ImportError:
        fail("PyTorch not installed")
        return False

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        n_devices = torch.cuda.device_count()
        ok(f"CUDA available — {n_devices} device(s), CUDA {cuda_version}")
        ok(f"Device 0: {device_name}")

        # Memory check
        props = torch.cuda.get_device_properties(0)
        total_mem_gb = props.total_memory / 1024 ** 3
        if total_mem_gb < 4:
            warn(
                f"GPU has only {total_mem_gb:.1f} GB VRAM — Transformer fine-tuning "
                f"will require aggressive memory mitigations (see DEVELOPMENT.md §9.2)"
            )
        else:
            ok(f"GPU VRAM: {total_mem_gb:.1f} GB")
        return True
    else:
        warn("CUDA not available — classical ML will work, but DL/Transformer training will fall back to CPU (slow)")
        return True  # Not a hard failure


def check_directory_structure() -> bool:
    print("\n[4/5] Directory structure")
    repo_root = Path(__file__).resolve().parent.parent
    all_ok = True
    for d in EXPECTED_DIRS:
        path = repo_root / d
        if path.exists() and path.is_dir():
            ok(d)
        else:
            fail(f"Missing directory: {d}")
            all_ok = False
    return all_ok


def check_write_permissions() -> bool:
    print("\n[5/5] Write permissions")
    repo_root = Path(__file__).resolve().parent.parent
    test_dirs = ["results", "plots", "data"]
    all_ok = True
    for d in test_dirs:
        path = repo_root / d
        test_file = path / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            ok(f"{d}/ is writable")
        except Exception as e:
            fail(f"{d}/ not writable: {e}")
            all_ok = False
    return all_ok


# ---------- Main ----------
def main() -> int:
    print("=" * 70)
    print("Phishing Detection Benchmark - Environment Verification")
    print("=" * 70)

    results = [
        check_python_version(),
        check_libraries(),
        check_gpu(),
        check_directory_structure(),
        check_write_permissions(),
    ]

    print("\n" + "=" * 70)
    if all(results):
        print(f"{GREEN}All checks passed.{RESET} You are ready to start Phase 1.")
        print("=" * 70)
        return 0
    else:
        n_failed = sum(1 for r in results if not r)
        print(f"{RED}{n_failed} check(s) failed.{RESET} Review the output above.")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
