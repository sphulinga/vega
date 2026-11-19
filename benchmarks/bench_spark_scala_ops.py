#!/usr/bin/env python3
"""Run benchmarks/spark_scala_ops.scala on Apache Spark (Scala RDD API).

The Scala counterpart of bench_pyvega_ops.py: same datasets, operations and
timing method (best of --repeat per operation), and the same JSON report
format, so the two can be compared side by side.

Spark is located via --spark-home, $SPARK_HOME, an installed ``pyspark``
package (which bundles the full Spark distribution, including spark-shell),
or ``spark-shell`` on PATH. Needs Java 17+ for Spark 4.

    pip install pyspark
    python benchmarks/bench_spark_scala_ops.py --workers 4 --json spark.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().with_name("spark_scala_ops.scala")


def find_spark_shell(spark_home: str | None) -> tuple[str, dict[str, str]]:
    env = os.environ.copy()
    candidates = []
    if spark_home:
        candidates.append(Path(spark_home))
    if env.get("SPARK_HOME"):
        candidates.append(Path(env["SPARK_HOME"]))
    try:
        import pyspark  # noqa: PLC0415 - optional dependency

        candidates.append(Path(pyspark.__file__).parent)
    except ImportError:
        pass
    for home in candidates:
        shell = home / "bin" / "spark-shell"
        if shell.is_file():
            env["SPARK_HOME"] = str(home)
            return str(shell), env
    on_path = shutil.which("spark-shell")
    if on_path:
        return on_path, env
    sys.exit("spark-shell not found: set SPARK_HOME, pass --spark-home, or pip install pyspark")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=4, help="local[N] executor threads")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1,
                        help="untimed passes over all operations first (JIT warm-up)")
    parser.add_argument("--driver-memory", default="4g")
    parser.add_argument("--spark-home", default=None)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    shell, env = find_spark_shell(args.spark_home)
    env["BENCH_N"] = str(args.n)
    env["BENCH_REPEAT"] = str(args.repeat)
    env["BENCH_WARMUP"] = str(args.warmup)
    cmd = [
        shell,
        "--master", f"local[{args.workers}]",
        "--driver-memory", args.driver_memory,
        "--conf", "spark.ui.enabled=false",
        "-i", str(SCRIPT),
    ]
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    report: dict[str, float] = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT\t"):
            _, name, ms = line.split("\t")
            report[name] = round(float(ms), 2)
            print(f"{name:34s} {float(ms):10.2f} ms")
    if not report:
        sys.exit(f"no results from spark-shell (exit {proc.returncode}):\n{proc.stderr[-3000:]}")

    version = subprocess.run(
        [shell, "--version"], env=env, text=True, capture_output=True, check=False
    )
    spark_version = next(
        (w for line in (version.stderr + version.stdout).splitlines() if "version" in line
         for w in line.split() if w[:1].isdigit()), "unknown")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"engine": f"spark-{spark_version}-scala", "n": args.n,
                       "workers": args.workers, "warmup": args.warmup,
                       "results": report}, handle, indent=2)


if __name__ == "__main__":
    main()
