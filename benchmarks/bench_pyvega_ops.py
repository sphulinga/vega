#!/usr/bin/env python3
"""Micro-benchmarks for common PyVega RDD operations.

Shortcut for ``bench_python_ops.py --engine pyvega`` (same options):

    VEGA_SERVER=build-release/vega_server python benchmarks/bench_pyvega_ops.py [--n 200000]

Prints one line per operation with the best-of-N wall time in milliseconds,
and optionally writes a JSON report (``--json out.json``) so two builds can
be compared side by side.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_python_ops import main  # noqa: E402

if __name__ == "__main__":
    main(["--engine", "pyvega", *sys.argv[1:]])
