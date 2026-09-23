"""
Platform detection: the single source of truth about where a run happened.

The reproducibility argument in this project has a boundary: metric values
reproduce bitwise on the same platform and do not across platforms, because the
seed pins the random choices and not the arithmetic. Naming the platform is
therefore part of the result, so `detect_environment` records the operating
system, the architecture, the BLAS backend behind NumPy/scikit-learn, and the
PyTorch execution backend, and reduces them to a short `profile` label.

Three profiles are real in this project:

- ``linux-x86_64-cuda``  reference platform, where the versioned results were produced
- ``macos-arm64-mps``    Accelerate instead of OpenBLAS, MPS instead of CUDA
- ``windows-x86_64-cpu`` no GPU in the tested path

Anything else is labelled ``unsupported``, which is a signal and not an error.

Used in three places, and defined only here so the three cannot drift apart:
`src.utils.manifests` (records the block into new manifests),
`scripts/verify_environment.py` and `scripts/compare_platforms.py`.
"""
from __future__ import annotations

import contextlib
import io
import platform
import re

# Profile labels. The first three are the platforms this project actually ran on.
PROFILE_LINUX_CUDA = "linux-x86_64-cuda"
PROFILE_MACOS_MPS = "macos-arm64-mps"
PROFILE_WINDOWS_CPU = "windows-x86_64-cpu"
PROFILE_UNSUPPORTED = "unsupported"

KNOWN_PROFILES: tuple[str, ...] = (
    PROFILE_LINUX_CUDA,
    PROFILE_MACOS_MPS,
    PROFILE_WINDOWS_CPU,
)

# Profile assumed for the 66 manifests written before the `environment` block
# existed. It is an inference from the repository history, not a recorded value,
# and every reader that applies it must say so (see `manifest_environment`).
LEGACY_PROFILE = PROFILE_LINUX_CUDA

# BLAS backend names worth distinguishing. Ordered: the first match wins, so
# more specific names come before generic ones.
_BLAS_NAMES = ("openblas", "accelerate", "mkl", "blis", "atlas", "netlib")

# x86-64 goes by two names depending on who is asking: uname says x86_64,
# Windows says AMD64.
_X86_64 = ("x86_64", "amd64")
_ARM64 = ("arm64", "aarch64")


def _detect_blas() -> str:
    """Return the BLAS backend NumPy is linked against, lowercase.

    Tries the structured NumPy build configuration first and falls back to
    parsing the text `numpy.show_config()` prints, because the structured form
    only exists on Meson-built NumPy (1.25+).

    Returns:
        One of ``openblas``, ``accelerate``, ``mkl``, ``blis``, ``atlas``,
        ``netlib``, or ``unknown`` when nothing identifiable is reported.
    """
    try:
        import numpy as np
    except ImportError:
        return "unknown"

    # Structured configuration (NumPy >= 1.25).
    with contextlib.suppress(Exception):
        config = np.__config__.show(mode="dicts")  # type: ignore[call-arg]
        blas = config.get("Build Dependencies", {}).get("blas", {})
        name = str(blas.get("name", "")).lower()
        for known in _BLAS_NAMES:
            if known in name:
                return known

    # Fallback: whatever show_config() prints.
    with contextlib.suppress(Exception):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            np.show_config()
        text = buffer.getvalue().lower()
        for known in _BLAS_NAMES:
            if re.search(known, text):
                return known

    return "unknown"


def _detect_torch() -> tuple[str, str | None]:
    """Return ``(torch_backend, gpu_name)`` for the installed PyTorch.

    Returns:
        ``backend`` is ``cuda``, ``mps`` or ``cpu``; ``cpu`` is also what is
        reported when PyTorch is not installed at all, since no accelerated
        backend is reachable in that case. ``gpu_name`` is the device name when
        one is available and ``None`` otherwise.
    """
    try:
        import torch
    except ImportError:
        return "cpu", None

    with contextlib.suppress(Exception):
        if torch.cuda.is_available():
            return "cuda", torch.cuda.get_device_name(0)

    with contextlib.suppress(Exception):
        if torch.backends.mps.is_available():
            # PyTorch exposes no device name for MPS; the integrated Apple GPU
            # is the only device it can mean.
            return "mps", "Apple Silicon GPU (MPS)"

    return "cpu", None


def derive_profile(system: str, machine: str, torch_backend: str) -> str:
    """Reduce OS, architecture and torch backend to a profile label.

    Kept as a pure function so the three real combinations can be checked
    without running on three machines.

    Args:
        system: `platform.system()` output, e.g. ``Linux``, ``Darwin``, ``Windows``.
        machine: `platform.machine()` output, e.g. ``x86_64``, ``arm64``, ``AMD64``.
        torch_backend: ``cuda``, ``mps`` or ``cpu``.

    Returns:
        One of `KNOWN_PROFILES`, or ``unsupported`` for any other combination.
    """
    system = (system or "").lower()
    machine = (machine or "").lower()
    backend = (torch_backend or "").lower()

    if system == "linux" and machine in _X86_64 and backend == "cuda":
        return PROFILE_LINUX_CUDA
    if system == "darwin" and machine in _ARM64 and backend == "mps":
        return PROFILE_MACOS_MPS
    if system == "windows" and machine in _X86_64 and backend == "cpu":
        return PROFILE_WINDOWS_CPU
    return PROFILE_UNSUPPORTED


def detect_environment() -> dict:
    """Describe the platform this process is running on.

    Returns:
        A JSON-serializable dict with:

        - ``os``: ``{"system": ..., "release": ...}``
        - ``arch``: machine architecture (``x86_64``, ``arm64``, ...)
        - ``python_impl``: ``CPython``, ``PyPy``, ...
        - ``cpu``: processor string as the OS reports it (often empty on Linux)
        - ``blas``: NumPy's BLAS backend (``openblas``, ``accelerate``, ``mkl``, ...)
        - ``gpu_name``: accelerator device name, or ``None``
        - ``torch_backend``: ``cuda``, ``mps`` or ``cpu``
        - ``profile``: one of `KNOWN_PROFILES` or ``unsupported``
    """
    torch_backend, gpu_name = _detect_torch()
    system = platform.system()
    machine = platform.machine()
    return {
        "os": {"system": system, "release": platform.release()},
        "arch": machine,
        "python_impl": platform.python_implementation(),
        "cpu": platform.processor(),
        "blas": _detect_blas(),
        "gpu_name": gpu_name,
        "torch_backend": torch_backend,
        "profile": derive_profile(system, machine, torch_backend),
    }


def describe(env: dict) -> str:
    """Render an environment dict as one compact human-readable line."""
    os_info = env.get("os") or {}
    system = os_info.get("system", "?")
    release = os_info.get("release", "?")
    gpu = env.get("gpu_name") or "none"
    return (
        f"{system} {release} / {env.get('arch', '?')} / "
        f"blas={env.get('blas', '?')} / torch={env.get('torch_backend', '?')} / "
        f"gpu={gpu} / profile={env.get('profile', '?')}"
    )


if __name__ == "__main__":
    import json

    print(json.dumps(detect_environment(), indent=2))
