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


def parse_result(output: str, name: str) -> BenchResult:
    m = RESULT_RE.search(output)
    if not m:
        raise RuntimeError(f"Could not parse RESULT line for {name}.\nOutput:\n{output}")
    return BenchResult(
        name=name,
        count=int(m.group("count")),
        median_s=float(m.group("median")),
        mean_s=float(m.group("mean")),
        times=[float(x) for x in m.group("times").split(",") if x],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Vega C++ core vs Sail.")
    parser.add_argument("--n", type=int, default=5_000_000)
    parser.add_argument("--partitions", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--build-dir", default="build-release")
    args = parser.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    build_dir = os.path.join(repo, args.build_dir)
    if not os.path.exists(os.path.join(build_dir, "CMakeCache.txt")):
        run_cmd(["cmake", "-S", repo, "-B", build_dir, "-DCMAKE_BUILD_TYPE=Release"], os.environ.copy(), repo)

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
    env["BENCH_N"] = str(args.n)
    env["BENCH_PARTITIONS"] = str(args.partitions)
    env["BENCH_REPEATS"] = str(args.repeats)
    env["SPARK_LOCAL_IP"] = env.get("SPARK_LOCAL_IP", "127.0.0.1")

    sail_out = run_cmd(
        [
            os.path.join(repo, ".venv", "bin", "python"),
            os.path.join(repo, "benchmarks", "sail_benchmark.py"),
        ],
        env,
        repo,
    )
    sail_res = parse_result(sail_out, "sail")

    if cpp_res.count != sail_res.count:
        raise RuntimeError(f"Result mismatch: vega={cpp_res.count}, sail={sail_res.count}")

    speedup = sail_res.median_s / cpp_res.median_s if cpp_res.median_s > 0 else float("inf")

    print(f"Config: n={args.n}, partitions={args.partitions}, workers={args.workers}, repeats={args.repeats}")
    print("")
    print("| engine | median_s | mean_s | count |")
    print("|---|---:|---:|---:|")
    print(f"| {cpp_res.name} | {cpp_res.median_s:.4f} | {cpp_res.mean_s:.4f} | {cpp_res.count} |")
    print(f"| {sail_res.name} | {sail_res.median_s:.4f} | {sail_res.mean_s:.4f} | {sail_res.count} |")
    print("")
    print(f"Median speedup (sail / vega_cpp): {speedup:.2f}x")
    print(f"Vega C++ times: {', '.join(f'{t:.4f}' for t in cpp_res.times)}")
    print(f"Sail times: {', '.join(f'{t:.4f}' for t in sail_res.times)}")


if __name__ == "__main__":
    main()
