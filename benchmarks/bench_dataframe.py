#!/usr/bin/env python3
"""DataFrame benchmark: Vega C++ DataFrames vs Spark SQL vs Sail.

Runs the same queries over the same generated tables on each engine:

    vega-cpp     benchmarks/vega_cpp_dataframe.cpp  (Vega's C++ DataFrame API)
    spark-scala  benchmarks/spark_scala_dataframe.scala  (Spark SQL, Scala DataFrame API,
                 table cached in memory)
    sail         benchmarks/sail_dataframe.py  (Sail via the PySpark Spark Connect
                 client; reads the table from uncompressed Parquet, since Sail
                 has no in-memory cache yet)

    cmake --build build-release --target vega_cpp_dataframe
    python benchmarks/bench_dataframe.py --n 10000000 \\
        --sail-python /path/to/venv-with-pysail/bin/python --json df.json

Every engine uses the same number of cores (Vega workers / Spark local[N] /
Sail's default of all cores) and partitions (also used as the shuffle
partition count). Each query's time is its best of --repeat runs after a
warm-up pass; engines run --rounds times in rotating order and the best round
is kept. The runner checks that all engines return the same answers (floating
point sums to a relative 1e-9).

Spark is found via --spark-home, $SPARK_HOME, an installed pyspark, or PATH.
Sail needs a Python with ``pysail``, ``pyspark-client`` (<= 4.0 for pysail
0.7) and ``pyarrow``; pass it with --sail-python. Skip engines with --engines.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_native_ops import find_spark_shell, spark_version  # noqa: E402

ENGINES = ["vega-cpp", "spark-scala", "sail"]


def parse(stdout: str) -> dict[str, tuple[float, str]]:
    out = {}
    for line in stdout.splitlines():
        if line.startswith("RESULT\t"):
            _, name, ms, value = line.split("\t")
            out[name] = (float(ms), value)
    return out


def run(engine: str, args) -> tuple[str, dict[str, tuple[float, str]]]:
    env = os.environ.copy()
    env.update(BENCH_N=str(args.n), BENCH_REPEAT=str(args.repeat), BENCH_WARMUP="1",
               BENCH_PARTITIONS=str(args.partitions))
    if engine == "vega-cpp":
        binary = Path(args.binary or HERE.parent / "build-release" / "examples" / "vega_cpp_dataframe")
        if not binary.is_file():
            sys.exit(f"{binary} not found: cmake --build build-release --target vega_cpp_dataframe")
        cmd = [str(binary), "--n", str(args.n), "--workers", str(args.workers),
               "--partitions", str(args.partitions), "--repeat", str(args.repeat)]
        label = "vega-cpp"
    elif engine == "spark-scala":
        shell, senv = find_spark_shell(args.spark_home)
        env.update({k: v for k, v in senv.items() if k == "SPARK_HOME"})
        cmd = [shell, "--master", f"local[{args.workers}]", "--driver-memory", args.driver_memory,
               "--conf", "spark.ui.enabled=false",
               "--conf", f"spark.sql.shuffle.partitions={args.partitions}",
               "-i", str(HERE / "spark_scala_dataframe.scala")]
        label = f"spark-{spark_version(shell, env)}-sql"
    else:
        cmd = [args.sail_python, str(HERE / "sail_dataframe.py")]
        ver = subprocess.run([args.sail_python, "-c",
                              "import importlib.metadata as m; print(m.version('pysail'))"],
                             text=True, capture_output=True, check=False).stdout.strip()
        label = f"sail-{ver or 'unknown'}"
    proc = subprocess.run(cmd, env=env, text=True, capture_output=True, check=False)
    results = parse(proc.stdout)
    if not results:
        sys.exit(f"{engine} produced no results (exit {proc.returncode}):\n{proc.stderr[-3000:]}")
    return label, results


def same(a: str, b: str) -> bool:
    xs, ys = a.split("|"), b.split("|")
    if len(xs) != len(ys):
        return False
    for x, y in zip(xs, ys):
        fx, fy = float(x), float(y)
        if not math.isclose(fx, fy, rel_tol=1e-9, abs_tol=1e-9):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", default=",".join(ENGINES))
    parser.add_argument("--n", type=int, default=10_000_000)
    parser.add_argument("--workers", type=int, default=4, help="cores for every engine")
    parser.add_argument("--partitions", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--binary", default=None, help="vega_cpp_dataframe path")
    parser.add_argument("--spark-home", default=None)
    parser.add_argument("--driver-memory", default="6g")
    parser.add_argument("--sail-python", default=sys.executable)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    engines = [e for e in args.engines.split(",") if e]
    for e in engines:
        if e not in ENGINES:
            sys.exit(f"unknown engine {e}; choose from {ENGINES}")

    labels: dict[str, str] = {}
    best: dict[str, dict[str, float]] = {e: {} for e in engines}
    values: dict[str, dict[str, str]] = {e: {} for e in engines}
    for rnd in range(args.rounds):
        order = engines[rnd % len(engines):] + engines[:rnd % len(engines)]
        for e in order:
            print(f"round {rnd + 1}/{args.rounds}: {e} ...", file=sys.stderr, flush=True)
            labels[e], res = run(e, args)
            for q, (ms, v) in res.items():
                best[e][q] = min(best[e].get(q, math.inf), ms)
                values[e][q] = v

    queries = list(best[engines[0]])
    ok = True
    for q in queries:
        ref = values[engines[0]][q]
        for e in engines[1:]:
            if not same(ref, values[e].get(q, "nan")):
                ok = False
                print(f"MISMATCH {q}: {engines[0]}={ref} {e}={values[e].get(q)}", file=sys.stderr)

    head = f"{'query':36s}" + "".join(f"{labels[e]:>18s}" for e in engines)
    print(head)
    for q in queries:
        row = f"{q:36s}" + "".join(f"{best[e].get(q, math.nan):16.1f}ms" for e in engines)
        print(row)
    print("results agree across engines" if ok else "RESULTS DIFFER (see above)")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"n": args.n, "workers": args.workers, "partitions": args.partitions,
                       "repeat": args.repeat, "rounds": args.rounds,
                       "engines": {labels[e]: {"results_ms": best[e], "values": values[e]}
                                   for e in engines},
                       "agree": ok}, fh, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
