"""
Phase 6 orchestrator — cross-dataset generalization.

Trains on one raw-URL dataset and tests on the other, both directions, for
classical (RF, XGBoost) and deep (CNN-LSTM) models. Writes one row per experiment
to ``metrics_crossdataset.csv``. Datasets: Mendeley and ``malicious_urls``.

Usage:
    python -m src.experiments.run_cross --all
"""
from __future__ import annotations

import argparse
import time

from src.data.loaders import is_available
from src.experiments.runner_cross import cross_classical, cross_deep

_A, _B = "mendeley", "malicious_urls"

# (model_key, kind, train_dataset, test_dataset)
PAIRS = [
    ("rf", "classical", _A, _B),
    ("rf", "classical", _B, _A),
    ("xgb", "classical", _A, _B),
    ("xgb", "classical", _B, _A),
    ("cnnlstm", "deep", _A, _B),
    ("cnnlstm", "deep", _B, _A),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the cross-dataset benchmark.")
    parser.add_argument("--all", action="store_true", help="run all pairs (the default)")
    parser.parse_args()

    for ds in (_A, _B):
        if not is_available(ds):
            print(f"Required dataset '{ds}' is missing — cannot run cross-dataset.")
            return

    print(f"Cross-dataset: {len(PAIRS)} experiments ({_A} <-> {_B})")
    ok, start = 0, time.perf_counter()
    for model_key, kind, train_ds, test_ds in PAIRS:
        print(f"\n=== {model_key}: {train_ds} -> {test_ds} ===", flush=True)
        t0 = time.perf_counter()
        try:
            (cross_classical if kind == "classical" else cross_deep)(model_key, train_ds, test_ds)
            print(f"    done in {time.perf_counter() - t0:.0f}s", flush=True)
            ok += 1
        except Exception as exc:
            print(f"    FAILED {model_key} {train_ds}->{test_ds}: {exc!r}", flush=True)

    print(f"\nDone: {ok}/{len(PAIRS)} in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
