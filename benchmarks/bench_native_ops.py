#!/usr/bin/env python3
"""Compare the engines' native APIs: Vega's C++ RDD API vs Spark's Scala RDD API.

Runs the same operations, datasets, element types and partitioning on either
engine (benchmarks/vega_cpp_ops.cpp and benchmarks/spark_scala_ops.scala):

    cmake --build build-release --target vega_cpp_ops
    python benchmarks/bench_native_ops.py --engine vega-cpp    --json vega_cpp.json
    python benchmarks/bench_native_ops.py --engine spark-scala --json spark.json

Spark is located via --spark-home, $SPARK_HOME, an installed ``pyspark``
package (which bundles the full Spark distribution, including spark-shell), or
``spark-shell`` on PATH; Spark 4 needs Java 17+.

Each operation's time is its best of --repeat runs after --warmup untimed
passes (the JVM's JIT needs warming; Vega gets the same pass). The JSON report
holds the times (ms) and each operation's result (a count or sum), so the two
engines can be checked for doing the same work.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCALA_SCRIPT = HERE / "spark_scala_ops.scala"


def find_spark_shell(spark_home: str | None) -> tuple[str, dict[str, str]]:
    env = os.environ.copy()
    candidates = [Path(p) for p in (spark_home, env.get("SPARK_HOME")) if p]
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


def spark_version(shell: str, env: dict[str, str]) -> str:
    out = subprocess.run([shell, "--version"], env=env, text=True,
                         capture_output=True, check=False)
    for line in (out.stderr + out.stdout).splitlines():
        if "version" in line:
            for word in line.split():
                if word[:1].isdigit():
                    return word
    return "unknown"


def run_vega_cpp(args) -> tuple[str, str]:
    binary = Path(args.binary) if args.binary else HERE.parent / "build-release" / "examples" / "vega_cpp_ops"
    if not binary.is_file():
        sys.exit(f"{binary} not found: cmake --build build-release --target vega_cpp_ops")
    cmd = [str(binary), "--n", str(args.n), "--workers", str(args.workers),
           "--partitions", str(args.partitions), "--repeat", str(args.repeat),
           "--warmup", str(args.warmup)]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        sys.exit(f"vega_cpp_ops failed (exit {proc.returncode}):\n{proc.stderr[-3000:]}")
    return proc.stdout, "vega-cpp"


def run_spark_scala(args) -> tuple[str, str]:
    shell, env = find_spark_shell(args.spark_home)
    env.update(BENCH_N=str(args.n), BENCH_REPEAT=str(args.repeat),
               BENCH_WARMUP=str(args.warmup), BENCH_PARTITIONS=str(args.partitions))
    cmd = [shell, "--master", f"local[{args.workers}]", "--driver-memory", args.driver_memory,
           "--conf", "spark.ui.enabled=false", "-i", str(SCALA_SCRIPT)]
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    if "RESULT\t" not in proc.stdout:
        sys.exit(f"spark-shell produced no results (exit {proc.returncode}):\n{proc.stderr[-3000:]}")
    return proc.stdout, f"spark-{spark_version(shell, env)}-scala"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", choices=["vega-cpp", "spark-scala"], required=True)
    parser.add_argument("--n", type=int, default=1_000_000)
    parser.add_argument("--workers", type=int, default=4,
                        help="Vega worker threads / Spark local[N]")
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1,
                        help="untimed passes over all operations first")
    parser.add_argument("--binary", default=None, help="vega_cpp_ops path (vega-cpp)")
    parser.add_argument("--spark-home", default=None, help="Spark install (spark-scala)")
    parser.add_argument("--driver-memory", default="6g", help="Spark driver memory")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    stdout, label = run_vega_cpp(args) if args.engine == "vega-cpp" else run_spark_scala(args)
    times: dict[str, float] = {}
    values: dict[str, int] = {}
    for line in stdout.splitlines():
        if line.startswith("RESULT\t"):
            _, name, ms, value = line.split("\t")
            times[name] = round(float(ms), 3)
            values[name] = int(value)
            print(f"{name:30s} {float(ms):10.2f} ms   result={value}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"engine": label, "n": args.n, "workers": args.workers,
                       "partitions": args.partitions, "warmup": args.warmup,
                       "results": times, "values": values}, handle, indent=2)


if __name__ == "__main__":
    main()
