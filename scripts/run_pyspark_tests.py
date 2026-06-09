#!/usr/bin/env python3
"""Run upstream Apache PySpark tests against the PyVega compatibility layer."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPAT_ROOT = REPO_ROOT / "scripts" / "pyspark_compat"
SPARK_TESTS = REPO_ROOT / "third_party" / "spark" / "python" / "pyspark" / "tests"


@dataclass
class ModuleReport:
    module: str
    collected: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    import_error: str | None = None
    failures: list[str] = field(default_factory=list)


def discover_modules(filter_prefix: str | None = None) -> list[str]:
    modules: list[str] = []
    for path in sorted(SPARK_TESTS.rglob("test_*.py")):
        rel = path.relative_to(REPO_ROOT / "third_party" / "spark" / "python")
        mod = rel.with_suffix("").as_posix().replace("/", ".")
        if filter_prefix:
            if mod != filter_prefix and not mod.startswith(filter_prefix + "."):
                continue
        modules.append(mod)
    return modules


def run_module(module_name: str) -> ModuleReport:
    report = ModuleReport(module=module_name)
    try:
        mod = importlib.import_module(module_name)
    except Exception as exc:
        report.import_error = f"{type(exc).__name__}: {exc}"
        report.errors = 1
        return report

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(mod)
    report.collected = suite.countTestCases()

    result = unittest.TextTestRunner(stream=open(os.devnull, "w"), verbosity=0).run(suite)
    report.passed = report.collected - len(result.failures) - len(result.errors) - len(result.skipped)
    report.failed = len(result.failures)
    report.errors = len(result.errors)
    report.skipped = len(result.skipped)
    for test, trace in result.failures + result.errors:
        report.failures.append(f"{test}\n{trace}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filter",
        default="pyspark.tests.test_rdd",
        help="Module prefix filter (default: core RDD tests only)",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "test-results" / "pyspark_vs_pyvega.json"),
        help="JSON report path",
    )
    args = parser.parse_args()

    os.environ.setdefault("SPARK_HOME", str(REPO_ROOT / "third_party" / "spark"))
    sys.path[:0] = [str(REPO_ROOT / "python"), str(COMPAT_ROOT)]

    modules = discover_modules(args.filter)
    reports = [run_module(name) for name in modules]

    summary = {
        "modules": len(reports),
        "collected": sum(r.collected for r in reports),
        "passed": sum(r.passed for r in reports),
        "failed": sum(r.failed for r in reports),
        "errors": sum(r.errors for r in reports),
        "skipped": sum(r.skipped for r in reports),
        "import_failures": sum(1 for r in reports if r.import_error),
    }

    payload = {"summary": summary, "reports": [r.__dict__ for r in reports]}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Report written to {out_path}")
    return 0 if summary["failed"] == 0 and summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
