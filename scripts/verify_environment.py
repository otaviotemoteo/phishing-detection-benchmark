#!/usr/bin/env python3
"""
Environment verification, platform identification, and what to do next.

This is the first command anyone should run in this repository. It does two jobs.

First, it identifies the platform: operating system, architecture, the BLAS
backend NumPy is linked against, and the PyTorch execution backend. Those four
facts decide how the repository behaves, because the project runs three
different ways (CUDA on Linux, MPS on Apple Silicon, CPU on Windows) and which
one applies changes both the commands to run and what to expect from the
numbers. The script prints the guidance for the detected profile so nobody has
to discover it by trial and error.

Second, it checks that the environment can actually produce the published
results: interpreter version, pinned library versions, compute backend,
directory structure, write permissions, dataset integrity, and the integrity of
the three result CSVs the dissertation cites.

Every failure prints the command that fixes it, for the platform detected, not a
list of alternatives for platforms you are not on.

Usage:
    python scripts/verify_environment.py

    # Print another platform's guidance block (display only, no detection change):
    PHISHBENCH_PROFILE=macos-arm64-mps python scripts/verify_environment.py

Exit code:
    0 - all checks passed (an untested platform is a warning, not a failure)
    1 - one or more checks failed
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.environment import (  # noqa: E402  (needs REPO_ROOT on sys.path)
    KNOWN_PROFILES,
    PROFILE_LINUX_CUDA,
    PROFILE_MACOS_MPS,
    PROFILE_UNSUPPORTED,
    PROFILE_WINDOWS_CPU,
    detect_environment,
)

# Expected library versions (must match requirements.txt).
EXPECTED_VERSIONS = {
    "numpy": "1.26.4",
    "pandas": "2.2.3",
    "scipy": "1.14.1",
    "sklearn": "1.5.2",  # scikit-learn imports as `sklearn`
    "imblearn": "0.12.4",  # imbalanced-learn imports as `imblearn`
    "xgboost": "2.1.1",
    "catboost": "1.2.7",
    "torch": "2.4.1",
    "matplotlib": "3.9.2",
    "seaborn": "0.13.2",
    "psutil": "6.0.0",
    "mlflow": "2.16.2",
    "joblib": "1.4.2",
}

# Import name -> the name pip installs it under.
PIP_NAMES = {
    "sklearn": "scikit-learn",
    "imblearn": "imbalanced-learn",
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
    "results/analysis",
    "plots/eda",
    "plots/feature_importance",
    "plots/final",
    "scripts",
    "tests",
]

# The three result tables the dissertation cites, with their expected number of
# data rows (header excluded).
RESULT_CSVS = {
    "results/metrics_ml.csv": 18,
    "results/metrics_dl.csv": 3,
    "results/metrics_crossdataset.csv": 12,
}

CHECKSUMS_PATH = "results/CHECKSUMS.txt"

DATASET_FILES = (
    "uci_phishing.csv",
    "mendeley_phishing.csv",
    "iscx_url2016.csv",
    "malicious_urls.csv",
)


# -----------------------------------------------------------------------------
# Platform guidance
# -----------------------------------------------------------------------------
# One block per profile. This script is the single source for them: the README
# points here rather than repeating the text, so the two cannot drift apart.
PROFILE_GUIDANCE: dict[str, list[str]] = {
    PROFILE_LINUX_CUDA: [
        "This is the reference platform. The results committed to results/ were",
        "produced here, and a run on this profile reproduces them value for value.",
        "",
        "Requirements for the pinned torch 2.4.1 build:",
        "  - NVIDIA driver 535 or newer (CUDA 12.1 runtime)",
        "  - check with: nvidia-smi",
        "  - check torch sees it: python -c \"import torch; print(torch.cuda.is_available(), torch.version.cuda)\"",
        "",
        "If torch reports CUDA unavailable, the pipeline still runs on CPU, but the",
        "deep learning phases get much slower and the neural metrics will not match",
        "the published ones bitwise.",
    ],
    PROFILE_MACOS_MPS: [
        "Apple Silicon. Everything runs, and the full pipeline has been executed",
        "end to end on this platform, but the arithmetic underneath is not the same",
        "as on the reference machine:",
        "",
        "  - The neural models train on MPS, not CUDA. Different kernels and a",
        "    different reduction order, not a subtler version of the same computation.",
        "  - The BLAS backend behind NumPy and scikit-learn is whatever the pinned",
        "    wheel links against on this architecture (printed above), and it is not",
        "    the x86-64 OpenBLAS build the reference results came from.",
        "",
        "Practical consequence: the classical models reproduce to about the third",
        "decimal, the neural ones diverge in the last decimals (up to 0.074 in the",
        "cross-dataset runs). Every conclusion holds; individual digits do not.",
        "That is expected here and is not a sign of a broken installation.",
        "",
        "One system library pip cannot provide, for XGBoost's OpenMP runtime:",
        "  brew install libomp",
        "",
        "If a dependency has no arm64 wheel, do not switch to an unpinned version:",
        "install it under Rosetta in an x86-64 environment, or run the pipeline on",
        "the reference platform. Changing a pin changes the numbers.",
    ],
    PROFILE_WINDOWS_CPU: [
        "Windows without a CUDA GPU. All 33 experiments have been run end to end on",
        "this platform (2026-09-01, i5-14500), natively, with no WSL involved.",
        "",
        "Run the two shell scripts from Git Bash rather than PowerShell: they assume",
        "a POSIX shell. Both look for the interpreter in .venv/Scripts/ as well as",
        ".venv/bin/, so the venv is found without activating it. You can also skip",
        "them and call the stages directly:",
        "  python -m src.experiments.run_classical --all",
        "  python -m src.experiments.run_deep --all",
        "  python -m src.experiments.run_cross --all",
        "",
        "Expect the order of magnitude, not the reference timings: the deep learning",
        "phase took 42 minutes on the recorded run against 7 on the reference GPU,",
        "and the classical benchmark 28 against 40. It has not hung; it is training.",
        "",
        "On the numbers, this is the closest of the three platforms to the reference:",
        "it shares x86-64 and the same BLAS family, and 54% of the classical metrics",
        "came back bit-identical, with a median gap of zero. The neural models ran on",
        "CPU rather than CUDA and moved as much as they did on macOS. Every",
        "conclusion holds; the last decimals of the neural runs do not.",
        "",
        "The manifests from that run are not committed here, so compare_platforms.py",
        "has nothing to line up against the reference yet. docs/SETUP.md carries the",
        "comparison that was made by hand.",
    ],
    PROFILE_UNSUPPORTED: [
        "This combination of operating system, architecture and compute backend was",
        "not tested for this project. The pipeline will most likely run: nothing in",
        "it is platform specific beyond the shell scripts and the compute backend.",
        "",
        "What is not available here is a baseline. Numerical divergence from the",
        "published results cannot be attributed, because there is no run on this",
        "platform to compare against. Treat conclusions as reproducible and exact",
        "metric values as unverified.",
        "",
        "The three profiles with a known baseline are:",
        f"  {', '.join(KNOWN_PROFILES)}",
    ],
}


# ---------- ANSI colors (degrade gracefully if not a TTY) ----------
def _supports_color() -> bool:
    return sys.stdout.isatty()


GREEN = "\033[92m" if _supports_color() else ""
RED = "\033[91m" if _supports_color() else ""
YELLOW = "\033[93m" if _supports_color() else ""
BOLD = "\033[1m" if _supports_color() else ""
RESET = "\033[0m" if _supports_color() else ""


def ok(msg: str) -> None:
    """Print a passing check."""
    print(f"  {GREEN}OK{RESET}    {msg}")


def fail(msg: str) -> None:
    """Print a failing check; the caller follows it with `fix` lines."""
    print(f"  {RED}FAIL{RESET}  {msg}")


def warn(msg: str) -> None:
    """Print a check that did not pass but does not block the pipeline."""
    print(f"  {YELLOW}WARN{RESET}  {msg}")


def fix(msg: str) -> None:
    """Print an actionable remediation line under a failure or warning."""
    print(f"        -> {msg}")


# ---------- Platform header and guidance ----------
def print_platform(env: dict) -> None:
    """Print the detected platform facts that decide how this repository behaves."""
    os_info = env["os"]
    print(f"\n{BOLD}Platform{RESET}")
    print(f"  os             {os_info['system']} {os_info['release']}")
    print(f"  arch           {env['arch']}")
    print(f"  python         {sys.version.split()[0]} ({env['python_impl']})")
    print(f"  cpu            {env['cpu'] or 'not reported by the OS'}")
    print(f"  blas           {env['blas']}")
    print(f"  torch backend  {env['torch_backend']}")
    print(f"  gpu            {env['gpu_name'] or 'none'}")
    print(f"  {BOLD}profile        {env['profile']}{RESET}")


def print_guidance(profile: str) -> None:
    """Print the guidance block for a profile."""
    print(f"\n{BOLD}What this means for you ({profile}){RESET}")
    for line in PROFILE_GUIDANCE.get(profile, PROFILE_GUIDANCE[PROFILE_UNSUPPORTED]):
        print(f"  {line}" if line else "")


def _python_install_hint(profile: str) -> str:
    """Return the Python 3.11 installation command for the detected profile."""
    if profile == PROFILE_MACOS_MPS:
        return "brew install python@3.11 && python3.11 -m venv .venv && source .venv/bin/activate"
    if profile == PROFILE_WINDOWS_CPU:
        return (
            "install Python 3.11 from python.org (or `winget install Python.Python.3.11`), "
            "then: py -3.11 -m venv .venv"
        )
    return (
        "sudo apt install python3.11 python3.11-venv  # or use pyenv, "
        "then: python3.11 -m venv .venv && source .venv/bin/activate"
    )


def _pip_prefix(profile: str) -> str:
    """Return the venv pip invocation that matches the detected profile."""
    return ".venv\\Scripts\\pip" if profile == PROFILE_WINDOWS_CPU else ".venv/bin/pip"


# ---------- Checks ----------
def check_python_version(env: dict) -> bool:
    """Check the interpreter is 3.11.x, as declared in .python-version."""
    print(f"\n{BOLD}[1/7] Python version{RESET}")
    major, minor = sys.version_info[:2]
    if major == 3 and minor == 11:
        ok(f"Python {sys.version.split()[0]}")
        return True
    fail(f"Python 3.11.x required, found {major}.{minor} ({sys.executable})")
    fix("the required version is declared in .python-version")
    fix(_python_install_hint(env["profile"]))
    return False


def check_libraries(env: dict) -> bool:
    """Check every pinned library imports; a version mismatch warns, absence fails."""
    print(f"\n{BOLD}[2/7] Library versions{RESET}")
    pip = _pip_prefix(env["profile"])
    all_ok = True
    for lib, expected in EXPECTED_VERSIONS.items():
        pip_name = PIP_NAMES.get(lib, lib)
        try:
            mod = importlib.import_module(lib)
        except ImportError as exc:
            fail(f"{lib} not installed ({exc.msg})")
            fix(f"{pip} install {pip_name}=={expected}")
            if lib == "xgboost" and env["profile"] == PROFILE_MACOS_MPS:
                fix("on macOS XGBoost also needs the OpenMP runtime: brew install libomp")
            all_ok = False
            continue
        actual = getattr(mod, "__version__", "unknown")
        if actual == expected:
            ok(f"{lib} {actual}")
        else:
            warn(f"{lib} {actual} (pinned at {expected})")
            fix(f"{pip} install {pip_name}=={expected}")
            fix("an unpinned version changes the numbers; only the pins reproduce results/")
    return all_ok


def check_compute_backend(env: dict) -> bool:
    """Report the compute backend. Never a failure: CPU is slow, not broken."""
    print(f"\n{BOLD}[3/7] Compute backend{RESET}")
    backend = env["torch_backend"]
    try:
        import torch
    except ImportError:
        fail("PyTorch not installed")
        fix(f"{_pip_prefix(env['profile'])} install torch==2.4.1")
        return False

    if backend == "cuda":
        ok(f"CUDA {torch.version.cuda}, {torch.cuda.device_count()} device(s)")
        ok(f"device 0: {env['gpu_name']}")
        total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_mem_gb < 4:
            warn(f"GPU has {total_mem_gb:.1f} GB VRAM")
            fix("this is the reference machine's budget; the pinned batch sizes fit it")
        else:
            ok(f"GPU VRAM: {total_mem_gb:.1f} GB")
    elif backend == "mps":
        ok("Apple MPS backend available")
        fix("the neural models train on MPS; last-decimal divergence is expected")
    else:
        warn("no GPU backend: PyTorch will train on CPU")
        fix("classical models are unaffected; the deep learning phases get much slower")
    return True


def check_directory_structure() -> bool:
    """Check the directories the pipeline writes into all exist."""
    print(f"\n{BOLD}[4/7] Directory structure{RESET}")
    all_ok = True
    missing = []
    for d in EXPECTED_DIRS:
        path = REPO_ROOT / d
        if path.exists() and path.is_dir():
            ok(d)
        else:
            fail(f"missing directory: {d}")
            missing.append(d)
            all_ok = False
    if missing:
        fix("run from a full clone of the repository, not a partial copy")
        fix(f"recreate with: mkdir -p {' '.join(missing)}")
    return all_ok


def check_write_permissions() -> bool:
    """Check results/, plots/ and data/ can actually be written to."""
    print(f"\n{BOLD}[5/7] Write permissions{RESET}")
    all_ok = True
    for d in ("results", "plots", "data"):
        path = REPO_ROOT / d
        test_file = path / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            ok(f"{d}/ is writable")
        except Exception as exc:
            fail(f"{d}/ not writable: {exc}")
            fix(f"the pipeline writes its output here: chmod u+w {path}")
            all_ok = False
    return all_ok


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def check_datasets() -> bool:
    """Datasets are not committed. Absent is a warning; wrong bytes is a failure."""
    print(f"\n{BOLD}[6/7] Datasets{RESET}")
    registry_path = REPO_ROOT / "data" / "dataset_hashes.json"
    if not registry_path.exists():
        warn("data/dataset_hashes.json not found")
        fix("bash scripts/download_datasets.sh   # fetches 3 of 4 and writes the registry")
        return True

    registry = json.loads(registry_path.read_text())
    all_ok = True
    n_present = 0
    for filename in DATASET_FILES:
        path = REPO_ROOT / "data" / filename
        recorded = registry.get(filename, {}).get("sha256")
        if not path.exists():
            warn(f"{filename} not downloaded")
            if filename == "iscx_url2016.csv":
                fix("ISCX-URL2016 is behind UNB CIC's registration form; the download")
                fix("script prints the steps, and data/README.md documents the schema")
            else:
                fix("bash scripts/download_datasets.sh")
            continue
        n_present += 1
        if recorded is None:
            warn(f"{filename} present but not in the hash registry")
            fix("python -m src.utils.io   # regenerates data/dataset_hashes.json")
            continue
        digest = _sha256(path)
        if digest == recorded:
            ok(f"{filename} matches the recorded hash")
        else:
            fail(f"{filename} does not match the recorded hash")
            fix(f"recorded {recorded[:16]}..., found {digest[:16]}...")
            fix("this is NOT the file the published results were computed from;")
            fix("a run against it will not reproduce them. Re-download it, or")
            fix("regenerate the registry with `python -m src.utils.io` and accept")
            fix("that the numbers become a different experiment.")
            all_ok = False
    if n_present == 0:
        fix("no dataset present yet: this is expected on a fresh clone")
    return all_ok


def check_results_integrity() -> bool:
    """The three CSVs the dissertation cites must be present, intact and complete."""
    print(f"\n{BOLD}[7/7] Published results{RESET}")
    all_ok = True

    recorded: dict[str, str] = {}
    checksums = REPO_ROOT / CHECKSUMS_PATH
    if checksums.exists():
        for line in checksums.read_text().splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                recorded[parts[1].strip()] = parts[0].strip()
    else:
        warn(f"{CHECKSUMS_PATH} not found; checking row counts only")
        fix(f"regenerate with: sha256sum {' '.join(RESULT_CSVS)} > {CHECKSUMS_PATH}")

    for rel_path, expected_rows in RESULT_CSVS.items():
        path = REPO_ROOT / rel_path
        if not path.exists():
            fail(f"missing: {rel_path}")
            fix("these tables are tracked in git; restore with: git checkout -- results/")
            all_ok = False
            continue

        n_rows = sum(1 for line in path.read_text().splitlines() if line.strip()) - 1
        if n_rows != expected_rows:
            fail(f"{rel_path}: {n_rows} data rows, expected {expected_rows}")
            fix("an incomplete table means a partial pipeline run overwrote it;")
            fix("restore with: git checkout -- results/")
            all_ok = False
            continue

        expected_hash = recorded.get(rel_path)
        if expected_hash is None:
            ok(f"{rel_path}: {n_rows} rows (no checksum recorded)")
            continue
        digest = _sha256(path)
        if digest == expected_hash:
            ok(f"{rel_path}: {n_rows} rows, checksum matches")
        else:
            fail(f"{rel_path}: checksum does not match {CHECKSUMS_PATH}")
            fix(f"recorded {expected_hash[:16]}..., found {digest[:16]}...")
            fix("these are the values cited in the dissertation. Restore them with:")
            fix("git checkout -- results/   (and re-check what overwrote them)")
            all_ok = False
    return all_ok


# ---------- Main ----------
def main() -> int:
    """Identify the platform, print its guidance, run every check, return the exit code."""
    print("=" * 74)
    print("Phishing Detection Benchmark - Environment Verification")
    print("=" * 74)

    env = detect_environment()
    print_platform(env)

    # Display-only override, so all three guidance blocks can be reviewed from
    # one machine. It never changes what is detected or checked.
    forced = os.environ.get("PHISHBENCH_PROFILE")
    if forced:
        print(f"\n  (PHISHBENCH_PROFILE={forced}: showing that profile's guidance instead)")
    print_guidance(forced or env["profile"])

    results = [
        check_python_version(env),
        check_libraries(env),
        check_compute_backend(env),
        check_directory_structure(),
        check_write_permissions(),
        check_datasets(),
        check_results_integrity(),
    ]

    print("\n" + "=" * 74)
    if all(results):
        print(f"{GREEN}All checks passed.{RESET}")
        if env["profile"] == PROFILE_UNSUPPORTED:
            print("Platform not among the three tested; see the guidance above before")
            print("comparing your numbers against the published ones.")
        print("Next: bash scripts/download_datasets.sh, then bash scripts/run_all.sh")
        print("=" * 74)
        return 0

    n_failed = sum(1 for r in results if not r)
    print(f"{RED}{n_failed} check(s) failed.{RESET} Each failure above prints its fix.")
    print("=" * 74)
    return 1


if __name__ == "__main__":
    sys.exit(main())
