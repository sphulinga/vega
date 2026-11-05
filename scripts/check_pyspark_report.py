#!/usr/bin/env python3
"""Gate CI on the PySpark compatibility report.

Fails when a test outside ``scripts/pyspark_known_failures.txt`` fails or
errors, when a module fails to import, or when a known failure starts passing
(so the allowlist is kept accurate).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_TEST_ID = re.compile(r"^(\w+) \(([\w.]+)\)")


def load_known(path: Path) -> set[str]:
    known: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            known.add(line)
    return known


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(ROOT / "test-results" / "pyspark_vs_pyvega.json"))
    parser.add_argument("--known", default=str(ROOT / "scripts" / "pyspark_known_failures.txt"))
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    known = load_known(Path(args.known))
    summary = report["summary"]
    print(
        "collected={collected} passed={passed} failed={failed} errors={errors} "
        "import_failures={import_failures}".format(**summary)
    )

    failing: dict[str, str] = {}
    problems: list[str] = []
    for module in report["reports"]:
        if module.get("import_error"):
            problems.append(f"import failure in {module['module']}: {module['import_error']}")
        for entry in module["failures"]:
            first, _, rest = entry.partition("\n")
            match = _TEST_ID.match(first)
            test_id = f"{match.group(2)}" if match else first.strip()
            last = rest.strip().splitlines()[-1] if rest.strip() else ""
            failing[test_id] = last

    unexpected = sorted(set(failing) - known)
    fixed = sorted(known - set(failing))
    for test_id in unexpected:
        problems.append(f"unexpected failure: {test_id}\n    {failing[test_id]}")
    for test_id in fixed:
        problems.append(f"known failure now passes (remove it from the list): {test_id}")

    for test_id in sorted(set(failing) & known):
        print(f"known JVM-only failure: {test_id}")
    if problems:
        print("\nPySpark compatibility regression:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("PySpark compatibility matches expectations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
