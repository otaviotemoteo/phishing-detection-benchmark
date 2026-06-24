"""
Phase 3 orchestrator — run the classical ML benchmark.

Runs each (model, dataset) pair through `train_and_evaluate` with hyperparameter
search, writing one upserted row per experiment to ``metrics_ml.csv``. The heavy
compute lives here (not in a notebook) so it can run in the background and resume.

Usage:
    python -m src.experiments.run_classical --all
    python -m src.experiments.run_classical --model rf --dataset uci
    python -m src.experiments.run_classical --model svm        # all datasets
"""
from __future__ import annotations

import argparse
import time

from src.config import SVM_MAX_TRAIN_SAMPLES
from src.data.loaders import available_datasets
from src.experiments.runner import train_and_evaluate
from src.models.classical import MODEL_DISPLAY, MODEL_NAMES, get_model

DATASETS = ["uci", "mendeley", "iscx"]


def run_one(model_key: str, dataset: str) -> bool:
    """Run a single experiment; return True on success."""
    estimator, grid, needs_subsample = get_model(model_key)
    display = MODEL_DISPLAY[model_key]
    max_train = SVM_MAX_TRAIN_SAMPLES if needs_subsample else None
    print(f"\n=== {display} on {dataset} ===", flush=True)
    t0 = time.perf_counter()
    try:
        train_and_evaluate(
            estimator, dataset, model_name=display,
            param_grid=grid, max_train_samples=max_train,
        )
        print(f"    done in {time.perf_counter() - t0:.0f}s", flush=True)
        return True
    except Exception as exc:  # keep the long run alive if one experiment fails
        print(f"    FAILED {display} on {dataset}: {exc!r}", flush=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the classical ML benchmark.")
    parser.add_argument("--model", default="all", choices=["all", *MODEL_NAMES])
    parser.add_argument("--dataset", default="all", choices=["all", *DATASETS])
    parser.add_argument(
        "--all", action="store_true", help="run all models on all datasets (the default)"
    )
    args = parser.parse_args()

    models = MODEL_NAMES if args.model == "all" else [args.model]
    present = set(available_datasets())
    datasets = [d for d in (DATASETS if args.dataset == "all" else [args.dataset]) if d in present]

    total = len(models) * len(datasets)
    print(f"Classical benchmark: {len(models)} model(s) x {len(datasets)} dataset(s) = {total} runs")

    ok = 0
    start = time.perf_counter()
    for dataset in datasets:
        for model_key in models:
            ok += run_one(model_key, dataset)

    elapsed = time.perf_counter() - start
    print(f"\nDone: {ok}/{total} succeeded in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
