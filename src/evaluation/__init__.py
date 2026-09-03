"""src.evaluation — standardized metrics, cost tracking, per-experiment and final plots.

Importing this package pins a non-interactive matplotlib backend outside of
notebooks. Every figure here is written with ``savefig`` and closed, never shown,
so no interactive backend is needed. On Windows the default choice is TkAgg, and
its Tk objects get finalized on a joblib worker thread during the classical
benchmark, aborting the interpreter outright with "Tcl_AsyncDelete: async handler
deleted by the wrong thread" (exit 3).

This lives in ``__init__`` rather than in ``plots``/``compare`` so that it runs
before either module imports ``pyplot``, which is when matplotlib resolves its
backend.
"""
from __future__ import annotations

import os
import sys

# A notebook kernel has already chosen its own (inline) backend — leave it alone.
# setdefault so an explicitly exported MPLBACKEND still wins.
if "ipykernel" not in sys.modules:
    os.environ.setdefault("MPLBACKEND", "Agg")
