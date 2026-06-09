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


def run_cmd(cmd: list[str], env: dict[str, str], cwd: str) -> str:
    p = subprocess.run(cmd, env=env, cwd=cwd, text=True, capture_output=True, check=False)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{out}")
    return out


def parse_result(out: str, name: str) -> BenchResult:
    m = RESULT_RE.search(out)
    if not m:
        raise RuntimeError(f"Could not parse RESULT for {name}.\nOutput:\n{out}")
    return BenchResult(
        name=name,
        count=int(m.group("count")),
        median_s=float(m.group("median")),
        mean_s=float(m.group("mean")),
        times=[float(x) for x in m.group("times").split(",") if x],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare shuffle-heavy performance across Vega, Spark Scala, and Sail.")
    parser.add_argument("--n", type=int, default=20_000_000)
    parser.add_argument("--partitions", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-keys", type=int, default=1_000_000)
    parser.add_argument("--build-dir", default="build-release")
    args = parser.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_dir = os.path.join(repo, args.build_dir)
    env = os.environ.copy()

    if not os.path.exists(os.path.join(build_dir, "CMakeCache.txt")):
        run_cmd(["cmake", "-S", repo, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"], env, repo)

    run_cmd(["cmake", "--build", build_dir, "--target", "shuffle_benchmark", "-j8"], env, repo)

    cpp_out = run_cmd(
        [
            os.path.join(build_dir, "examples", "shuffle_benchmark"),
            "--n", str(args.n),
            "--partitions", str(args.partitions),
            "--workers", str(args.workers),
            "--repeats", str(args.repeats),
            "--num-keys", str(args.num_keys),
        ],
        env,
        repo,
    )
    cpp = parse_result(cpp_out, "vega_cpp")

    spark_env = env.copy()
    spark_env["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@21"
    spark_env["PATH"] = spark_env["JAVA_HOME"] + "/bin:" + spark_env.get("PATH", "")
    spark_env["SPARK_LOCAL_IP"] = spark_env.get("SPARK_LOCAL_IP", "127.0.0.1")
    spark_env["BENCH_N"] = str(args.n)
    spark_env["BENCH_PARTITIONS"] = str(args.partitions)
    spark_env["BENCH_REPEATS"] = str(args.repeats)
    spark_env["BENCH_NUM_KEYS"] = str(args.num_keys)

    scala_out = run_cmd(
        [
            "spark-shell",
            "--master", f"local[{args.workers}]",
            "--conf", "spark.ui.enabled=false",
            "--conf", "spark.driver.memory=2g",
            "-i", os.path.join(repo, "benchmarks", "spark_scala_shuffle_benchmark.scala"),
        ],
        spark_env,
        repo,
    )
    scala = parse_result(scala_out, "spark_scala")

    sail_env = env.copy()
    sail_env["BENCH_N"] = str(args.n)
    sail_env["BENCH_PARTITIONS"] = str(args.partitions)
    sail_env["BENCH_REPEATS"] = str(args.repeats)
    sail_env["BENCH_NUM_KEYS"] = str(args.num_keys)
    sail_env["SPARK_LOCAL_IP"] = "127.0.0.1"

    sail_out = run_cmd(
        [os.path.join(repo, ".venv", "bin", "python"), os.path.join(repo, "benchmarks", "sail_shuffle_benchmark.py")],
        sail_env,
        repo,
    )
    sail = parse_result(sail_out, "sail")

    if not (cpp.count == scala.count == sail.count):
        raise RuntimeError(
            f"Count mismatch: vega={cpp.count}, spark_scala={scala.count}, sail={sail.count}"
        )

    print(
        f"Config: n={args.n}, partitions={args.partitions}, workers={args.workers}, "
        f"repeats={args.repeats}, num_keys={args.num_keys}"
    )
    print("")
    print("| engine | median_s | mean_s | count | vs_vega |")
    print("|---|---:|---:|---:|---:|")
    print(f"| {cpp.name} | {cpp.median_s:.4f} | {cpp.mean_s:.4f} | {cpp.count} | 1.00x |")
    print(f"| {scala.name} | {scala.median_s:.4f} | {scala.mean_s:.4f} | {scala.count} | {scala.median_s / cpp.median_s:.2f}x |")
    print(f"| {sail.name} | {sail.median_s:.4f} | {sail.mean_s:.4f} | {sail.count} | {sail.median_s / cpp.median_s:.2f}x |")
    print("")
    print(f"Vega times: {', '.join(f'{t:.4f}' for t in cpp.times)}")
    print(f"Spark Scala times: {', '.join(f'{t:.4f}' for t in scala.times)}")
    print(f"Sail times: {', '.join(f'{t:.4f}' for t in sail.times)}")


if __name__ == "__main__":
    main()
