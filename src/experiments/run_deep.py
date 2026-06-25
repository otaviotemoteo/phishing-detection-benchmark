"""
Phase 4 orchestrator — run the deep learning benchmark (Mendeley raw URLs).

Runs each char-level model through `train_and_evaluate_deep`, writing one upserted
row per experiment to ``metrics_dl.csv``. Prints the total wall-clock for the
EXPERIMENT_LOG entry.

Usage:
    python -m src.experiments.run_deep --all
    python -m src.experiments.run_deep --model cnn --subset 1000   # pipeline check
"""
from __future__ import annotations

import argparse
import time

from src.data.loaders import is_available
from src.experiments.runner_deep import train_and_evaluate_deep
from src.models.deep import DEEP_MODEL_DISPLAY, DEEP_MODEL_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deep learning benchmark.")
    parser.add_argument("--model", default="all", choices=["all", *DEEP_MODEL_NAMES])
    parser.add_argument("--dataset", default="mendeley")
    parser.add_argument("--all", action="store_true", help="run all DL models (the default)")
    parser.add_argument("--subset", type=int, default=None, help="use first N rows (smoke test)")
    args = parser.parse_args()

    if not is_available(args.dataset):
        print(f"Dataset '{args.dataset}' not available.")
        return

    models = DEEP_MODEL_NAMES if args.model == "all" else [args.model]
    tag = f" (subset {args.subset})" if args.subset else ""
    print(f"Deep learning: {len(models)} model(s) on {args.dataset}{tag} | device-aware (GPU if available)")

    ok, start = 0, time.perf_counter()
    for key in models:
        print(f"\n=== {DEEP_MODEL_DISPLAY[key]} on {args.dataset} ===", flush=True)
        t0 = time.perf_counter()
        try:
            train_and_evaluate_deep(key, args.dataset, subset_n=args.subset)
            print(f"    done in {time.perf_counter() - t0:.0f}s", flush=True)
            ok += 1
        except Exception as exc:
            print(f"    FAILED {DEEP_MODEL_DISPLAY[key]}: {exc!r}", flush=True)

    print(f"\nDone: {ok}/{len(models)} in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
