#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass


RESULT_RE = re.compile(
    r"RESULT\s+count=(?P<count>\d+)\s+median_s=(?P<median>[0-9.]+)\s+mean_s=(?P<mean>[0-9.]+)\s+times_s=(?P<times>[0-9.,]+)"
)


@dataclass
class BenchResult:
    name: str
    count: int
    median_s: float
    mean_s: float
    times: list[float]


def parse_result(text: str, engine: str) -> BenchResult:
    m = RESULT_RE.search(text)
    if not m:
        raise RuntimeError(f"Could not parse RESULT line for {engine}.\nOutput:\n{text}")
    return BenchResult(
        name=engine,
        count=int(m.group("count")),
        median_s=float(m.group("median")),
        mean_s=float(m.group("mean")),
        times=[float(x) for x in m.group("times").split(",") if x],
    )


def run_cmd(cmd: list[str], env: dict[str, str], cwd: str) -> str:
    p = subprocess.run(cmd, env=env, cwd=cwd, text=True, capture_output=True, check=False)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Vega C++ core vs Spark Scala.")
    parser.add_argument("--n", type=int, default=5_000_000)
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--build-dir", default="build-release")
    args = parser.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_dir = os.path.join(repo, args.build_dir)

    if not os.path.exists(os.path.join(build_dir, "CMakeCache.txt")):
        run_cmd(
            ["cmake", "-S", repo, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"],
            os.environ.copy(),
            repo,
        )

    # Ensure benchmark binary exists and is up to date.
    run_cmd(["cmake", "--build", build_dir, "--target", "core_benchmark", "-j8"], os.environ.copy(), repo)

    cpp_out = run_cmd(
        [
            os.path.join(build_dir, "examples", "core_benchmark"),
            "--n",
            str(args.n),
            "--partitions",
            str(args.partitions),
            "--workers",
            str(args.workers),
            "--repeats",
            str(args.repeats),
        ],
        os.environ.copy(),
        repo,
    )
    cpp_res = parse_result(cpp_out, "vega_cpp")

    env = os.environ.copy()
    env["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@21"
    env["PATH"] = env["JAVA_HOME"] + "/bin:" + env.get("PATH", "")
    env["BENCH_N"] = str(args.n)
    env["BENCH_PARTITIONS"] = str(args.partitions)
    env["BENCH_REPEATS"] = str(args.repeats)

    spark_out = run_cmd(
        [
            "spark-shell",
            "--master",
            f"local[{args.workers}]",
            "--conf",
            "spark.ui.enabled=false",
            "--conf",
            "spark.driver.memory=2g",
            "-i",
            os.path.join(repo, "benchmarks", "spark_scala_benchmark.scala"),
        ],
        env,
        repo,
    )
    spark_res = parse_result(spark_out, "spark_scala")

    if cpp_res.count != spark_res.count:
        raise RuntimeError(f"Count mismatch: {cpp_res.count} != {spark_res.count}")

    speedup = spark_res.median_s / cpp_res.median_s if cpp_res.median_s > 0 else float("inf")

    print(f"Config: n={args.n}, partitions={args.partitions}, workers={args.workers}, repeats={args.repeats}")
    print("")
    print("| engine | median_s | mean_s | count |")
    print("|---|---:|---:|---:|")
    print(f"| {cpp_res.name} | {cpp_res.median_s:.4f} | {cpp_res.mean_s:.4f} | {cpp_res.count} |")
    print(f"| {spark_res.name} | {spark_res.median_s:.4f} | {spark_res.mean_s:.4f} | {spark_res.count} |")
    print("")
    print(f"Median speedup (spark_scala / vega_cpp): {speedup:.2f}x")
    print(f"Vega C++ times: {', '.join(f'{t:.4f}' for t in cpp_res.times)}")
    print(f"Spark Scala times: {', '.join(f'{t:.4f}' for t in spark_res.times)}")


if __name__ == "__main__":
    main()
