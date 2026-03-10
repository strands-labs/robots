#!/usr/bin/env python3
"""
Newton GPU Solver Benchmark

Benchmarks all 7 Newton solvers across multiple environment counts.
Generates performance table: solver × num_envs → env-steps/s.

Usage:
    python scripts/newton_benchmark.py
    python scripts/newton_benchmark.py --solvers mujoco,featherstone --envs 16,256
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

ALL_SOLVERS = ["mujoco", "featherstone", "semi_implicit", "xpbd", "vbd", "style3d", "implicit_mpm"]
DEFAULT_ENV_COUNTS = [16, 256, 1024, 4096]
BENCHMARK_STEPS = 500


def benchmark_solver(
    solver: str,
    num_envs: int,
    robot_name: str = "so100",
    num_steps: int = BENCHMARK_STEPS,
) -> Dict[str, Any]:
    """Benchmark a single solver at a given env count.

    Returns dict with fps, memory, timing info.
    """
    result = {
        "solver": solver,
        "num_envs": num_envs,
        "robot": robot_name,
        "num_steps": num_steps,
        "status": "pending",
    }

    try:
        from strands_robots.newton.newton_backend import SOLVER_MAP, NewtonBackend, NewtonConfig

        if solver not in SOLVER_MAP:
            result["status"] = "error"
            result["error"] = f"Unknown solver: {solver}"
            return result

        config = NewtonConfig(
            num_envs=num_envs,
            solver=solver,
            device="cuda:0",
            dt=0.005,
        )

        backend = NewtonBackend(config)
        backend.create_world()
        backend.add_robot(robot_name)

        if num_envs > 1:
            backend.replicate(num_envs=num_envs)

        # Warmup
        for _ in range(10):
            backend.step()

        # Benchmark
        start = time.perf_counter()
        for _ in range(num_steps):
            backend.step()
        elapsed = time.perf_counter() - start

        total_env_steps = num_steps * num_envs
        fps = total_env_steps / elapsed if elapsed > 0 else 0

        result.update({
            "status": "success",
            "elapsed_seconds": elapsed,
            "total_env_steps": total_env_steps,
            "fps": fps,
            "steps_per_second": num_steps / elapsed if elapsed > 0 else 0,
        })

        backend.destroy()

    except ImportError as e:
        result["status"] = "skip"
        result["error"] = f"Newton not available: {e}"
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def run_benchmark(
    solvers: List[str] = None,
    env_counts: List[int] = None,
    robot_name: str = "so100",
    num_steps: int = BENCHMARK_STEPS,
    output_dir: str = "./benchmark_results",
) -> Dict[str, Any]:
    """Run full Newton benchmark matrix.

    Args:
        solvers: List of solver names to benchmark
        env_counts: List of environment counts
        robot_name: Robot to use
        num_steps: Steps per benchmark run
        output_dir: Output directory

    Returns:
        Dict with all results and generated reports
    """
    solvers = solvers or ALL_SOLVERS
    env_counts = env_counts or DEFAULT_ENV_COUNTS
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    matrix = {}  # solver -> {num_envs -> fps}

    total_runs = len(solvers) * len(env_counts)
    completed = 0

    print(f"Newton Benchmark: {len(solvers)} solvers × {len(env_counts)} env counts = {total_runs} runs")
    print(f"Steps per run: {num_steps}, Robot: {robot_name}")
    print("=" * 70)

    for solver in solvers:
        matrix[solver] = {}
        for num_envs in env_counts:
            completed += 1
            print(f"[{completed}/{total_runs}] {solver:20s} | {num_envs:5d} envs | ", end="", flush=True)

            result = benchmark_solver(solver, num_envs, robot_name, num_steps)
            all_results.append(result)

            if result["status"] == "success":
                fps = result["fps"]
                matrix[solver][num_envs] = fps
                print(f"{fps:>12,.0f} env-steps/s")
            elif result["status"] == "skip":
                matrix[solver][num_envs] = 0
                print(f"SKIPPED ({result.get('error', '')[:40]})")
            else:
                matrix[solver][num_envs] = 0
                print(f"FAILED ({result.get('error', '')[:40]})")

    # Generate markdown table
    md_lines = [
        "# Newton GPU Solver Benchmark",
        f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Robot: {robot_name}, Steps: {num_steps}",
        "",
        "| Solver | " + " | ".join(f"{n} envs" for n in env_counts) + " |",
        "|--------|" + "|".join(["--------:"] * len(env_counts)) + "|",
    ]

    for solver in solvers:
        row = []
        for n in env_counts:
            fps = matrix.get(solver, {}).get(n, 0)
            if fps > 0:
                row.append(f"{fps:,.0f}")
            else:
                row.append("—")
        md_lines.append(f"| {solver} | " + " | ".join(row) + " |")

    md_lines.append("\n*env-steps/s (higher is better)*")
    md_content = "\n".join(md_lines)

    # Save outputs
    md_path = os.path.join(output_dir, "newton_benchmark.md")
    with open(md_path, "w") as f:
        f.write(md_content)

    csv_path = os.path.join(output_dir, "newton_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["solver", "num_envs", "fps", "elapsed_s", "status", "error"])
        for r in all_results:
            writer.writerow([
                r.get("solver"), r.get("num_envs"), r.get("fps", 0),
                r.get("elapsed_seconds", 0), r.get("status"), r.get("error", ""),
            ])

    json_path = os.path.join(output_dir, "newton_benchmark.json")
    with open(json_path, "w") as f:
        json.dump({"results": all_results, "matrix": matrix}, f, indent=2)

    print(f"\n{'=' * 70}")
    print("✅ Benchmark complete!")
    print(f"   Markdown: {md_path}")
    print(f"   CSV: {csv_path}")
    print(f"\n{md_content}")

    return {"results": all_results, "matrix": matrix, "markdown": md_content}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Newton GPU Solver Benchmark")
    parser.add_argument("--solvers", type=str, default=",".join(ALL_SOLVERS),
                       help="Comma-separated solver names")
    parser.add_argument("--envs", type=str, default="16,256,1024,4096",
                       help="Comma-separated env counts")
    parser.add_argument("--robot", type=str, default="so100")
    parser.add_argument("--steps", type=int, default=BENCHMARK_STEPS)
    parser.add_argument("--output", type=str, default="./benchmark_results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    solvers = [s.strip() for s in args.solvers.split(",")]
    env_counts = [int(n.strip()) for n in args.envs.split(",")]

    run_benchmark(
        solvers=solvers,
        env_counts=env_counts,
        robot_name=args.robot,
        num_steps=args.steps,
        output_dir=args.output,
    )
