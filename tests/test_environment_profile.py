"""
Platform detection and the manifest reader's tolerance for older manifests.

`src.utils.environment` is the single source of truth about which platform a run
happened on, and the profile label it produces is what `verify_environment.py`
and `compare_platforms.py` branch on. The three real combinations are checked by
simulating the platform values rather than by running on three machines, so this
file behaves the same wherever it is executed.
"""
from __future__ import annotations

from src.utils.environment import (
    KNOWN_PROFILES,
    LEGACY_PROFILE,
    PROFILE_LINUX_CUDA,
    PROFILE_MACOS_MPS,
    PROFILE_UNSUPPORTED,
    PROFILE_WINDOWS_CPU,
    derive_profile,
    describe,
    detect_environment,
)
from src.utils.manifests import manifest_environment

EXPECTED_KEYS = {
    "os",
    "arch",
    "python_impl",
    "cpu",
    "blas",
    "gpu_name",
    "torch_backend",
    "profile",
}


def test_detect_environment_returns_every_expected_key():
    env = detect_environment()

    assert set(env) == EXPECTED_KEYS
    assert set(env["os"]) == {"system", "release"}


def test_detect_environment_reports_a_usable_profile_and_backend():
    env = detect_environment()

    assert env["profile"] in (*KNOWN_PROFILES, PROFILE_UNSUPPORTED)
    assert env["torch_backend"] in {"cuda", "mps", "cpu"}
    assert env["blas"] != "", "the BLAS backend must be reported, even as 'unknown'"


def test_detect_environment_is_json_serializable():
    import json

    json.loads(json.dumps(detect_environment()))


def test_the_three_real_platforms_map_to_their_profiles():
    assert derive_profile("Linux", "x86_64", "cuda") == PROFILE_LINUX_CUDA
    assert derive_profile("Darwin", "arm64", "mps") == PROFILE_MACOS_MPS
    assert derive_profile("Windows", "AMD64", "cpu") == PROFILE_WINDOWS_CPU


def test_profile_detection_ignores_case_and_architecture_spelling():
    assert derive_profile("linux", "X86_64", "CUDA") == PROFILE_LINUX_CUDA
    assert derive_profile("Darwin", "aarch64", "mps") == PROFILE_MACOS_MPS
    assert derive_profile("Windows", "x86_64", "cpu") == PROFILE_WINDOWS_CPU


def test_unknown_combinations_are_unsupported_rather_than_an_error():
    for system, machine, backend in (
        ("Linux", "x86_64", "cpu"),  # the reference OS without its GPU
        ("Darwin", "x86_64", "cpu"),  # Intel Mac
        ("Windows", "x86_64", "cuda"),  # Windows with a GPU: plausible, never tested
        ("FreeBSD", "riscv64", "cpu"),
        ("", "", ""),
    ):
        assert derive_profile(system, machine, backend) == PROFILE_UNSUPPORTED


def test_profile_detection_survives_missing_values():
    assert derive_profile(None, None, None) == PROFILE_UNSUPPORTED


def test_detection_reads_the_platform_through_replaceable_helpers(monkeypatch):
    """The whole detection, not just `derive_profile`, follows a simulated platform."""
    import platform

    from src.utils import environment

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(environment, "_detect_torch", lambda: ("cuda", "GeForce GTX 1060 3GB"))
    monkeypatch.setattr(environment, "_detect_blas", lambda: "openblas")

    env = environment.detect_environment()

    assert env["profile"] == PROFILE_LINUX_CUDA
    assert env["gpu_name"] == "GeForce GTX 1060 3GB"
    assert env["blas"] == "openblas"


def test_describe_renders_the_platform_on_one_line():
    line = describe(detect_environment())

    assert "profile=" in line
    assert "\n" not in line


def test_the_manifest_reader_accepts_a_manifest_without_an_environment_block():
    legacy = {"experiment_id": "RandomForest_uci_20260707_133708", "metrics": {}}

    env = manifest_environment(legacy)

    assert env["profile"] == LEGACY_PROFILE
    assert env["inferred"] is True
    assert "inferred" in env["note"].lower()


def test_the_manifest_reader_prefers_a_recorded_environment_block():
    recorded = {
        "experiment_id": "RandomForest_uci_20260910_120000",
        "environment": {"profile": PROFILE_MACOS_MPS, "blas": "accelerate"},
    }

    env = manifest_environment(recorded)

    assert env["profile"] == PROFILE_MACOS_MPS
    assert env["blas"] == "accelerate"
    assert env["inferred"] is False


def test_an_empty_environment_block_counts_as_absent():
    env = manifest_environment({"environment": {}})

    assert env["inferred"] is True
    assert env["profile"] == LEGACY_PROFILE


def test_the_real_manifests_are_all_read_as_inferred():
    """The 66 committed manifests predate the block and must never claim otherwise."""
    from pathlib import Path

    from src.utils.manifests import load_manifests

    manifests_dir = Path(__file__).resolve().parent.parent / "results" / "manifests"
    manifests = load_manifests(manifests_dir)

    assert len(manifests) == 66
    for manifest in manifests:
        env = manifest_environment(manifest)
        assert env["profile"] in (*KNOWN_PROFILES, PROFILE_UNSUPPORTED)
        if "environment" not in manifest:
            assert env["inferred"] is True
