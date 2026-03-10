#!/usr/bin/env python3
"""
Sample 06 — Benchmark: Compare MuJoCo (CPU) vs Newton (GPU) vs Isaac Sim (RTX).

Runs the same task on all three simulation backends and produces a comparison
table with throughput, latency, and quality metrics.

Level: 3 (Advanced)
Hardware: GPU recommended (CPU-only runs MuJoCo path only)

Usage:
    # Full comparison (requires GPU + Isaac Sim):
    python samples/06_raytraced_training/benchmark.py

    # CPU-only (MuJoCo only):
    python samples/06_raytraced_training/benchmark.py --cpu-only

    # Newton solvers comparison:
    python samples/06_raytraced_training/benchmark.py --newton-only --envs 4096
"""

from __future__ import annotations

import argparse
import time


def benchmark_mujoco(num_steps: int = 1000) -> dict:
    """Benchmark MuJoCo CPU simulation.

    MuJoCo is strands-robots' default backend — fast single-environment
    simulation for prototyping and testing.

    Args:
        num_steps: Steps to benchmark.

    Returns:
        Benchmark results.
    """
    print("\n┌─────────────────────────────────────────┐")
    print("│  MuJoCo (CPU, 1 env)                     │")
    print("└─────────────────────────────────────────┘")

    try:
        import mujoco

        # Load a simple model
        xml = """
        <mujoco>
          <option timestep="0.002"/>
          <worldbody>
            <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
            <geom type="plane" size="5 5 0.1" rgba=".9 .9 .9 1"/>
            <body pos="0 0 0.5">
              <joint type="free"/>
              <geom type="box" size="0.1 0.1 0.1" mass="1" rgba="1 0 0 1"/>
            </body>
          </worldbody>
        </mujoco>
        """
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        # Warmup
        for _ in range(10):
            mujoco.mj_step(model, data)

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(num_steps):
            mujoco.mj_step(model, data)
        elapsed = time.perf_counter() - t0

        throughput = int(num_steps / elapsed)
        step_time = elapsed / num_steps * 1_000_000  # microseconds

        print(f"  Steps:      {num_steps:,}")
        print(f"  Elapsed:    {elapsed:.4f}s")
        print(f"  Throughput: {throughput:,} steps/s")
        print(f"  Per step:   {step_time:.1f}μs")
        print(f"  Final z:    {data.qpos[2]:.4f}")

        return {
            "backend": "MuJoCo (CPU)",
            "num_envs": 1,
            "throughput": throughput,
            "elapsed": elapsed,
            "step_time_us": step_time,
            "success": True,
        }

    except ImportError:
        print("  ⚠️ mujoco not installed — skip")
        return {"backend": "MuJoCo (CPU)", "success": False, "error": "not installed"}


def benchmark_newton(
    solver: str = "featherstone",
    num_envs: int = 4096,
    num_steps: int = 1000,
    device: str = "cuda:0",
) -> dict:
    """Benchmark Newton GPU simulation.

    Args:
        solver: Newton solver backend.
        num_envs: Parallel environments.
        num_steps: Steps to benchmark.
        device: CUDA device.

    Returns:
        Benchmark results.
    """
    print("\n┌─────────────────────────────────────────┐")
    print(f"│  Newton ({solver}, {num_envs} envs){'':>{32 - len(solver) - len(str(num_envs))}}│")
    print("└─────────────────────────────────────────┘")

    try:
        from strands_robots.newton import NewtonBackend, NewtonConfig

        config = NewtonConfig(num_envs=num_envs, solver=solver, device=device)
        backend = NewtonBackend(config)
        backend.create_world(gravity=(0, 0, -9.81))
        backend.add_robot("so100")
        backend.replicate(num_envs=num_envs)

        # Warmup
        for _ in range(10):
            backend.step()

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(num_steps):
            backend.step()
        elapsed = time.perf_counter() - t0

        throughput = int(num_steps * num_envs / elapsed)
        step_time = elapsed / num_steps * 1000  # ms

        print(f"  Envs:       {num_envs:,}")
        print(f"  Steps:      {num_steps:,}")
        print(f"  Elapsed:    {elapsed:.3f}s")
        print(f"  Throughput: {throughput:,} env-steps/s")
        print(f"  Per step:   {step_time:.2f}ms")

        backend.destroy()

        return {
            "backend": f"Newton ({solver})",
            "num_envs": num_envs,
            "throughput": throughput,
            "elapsed": elapsed,
            "step_time_ms": step_time,
            "success": True,
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {"backend": f"Newton ({solver})", "success": False, "error": str(e)}


def benchmark_isaac_sim(
    num_envs: int = 64,
    num_steps: int = 500,
) -> dict:
    """Benchmark Isaac Sim GPU simulation.

    Args:
        num_envs: Parallel environments (Isaac Sim + cloner).
        num_steps: Steps to benchmark.

    Returns:
        Benchmark results.
    """
    print("\n┌─────────────────────────────────────────┐")
    print(f"│  Isaac Sim (RTX, {num_envs} envs){'':>{26 - len(str(num_envs))}}│")
    print("└─────────────────────────────────────────┘")

    try:
        from strands_robots.isaac import is_isaac_sim_available

        if not is_isaac_sim_available():
            print("  ⚠️ Isaac Sim not available — skip")
            return {"backend": "Isaac Sim (RTX)", "success": False, "error": "not installed"}

        from strands_robots.isaac import IsaacSimBackend

        backend = IsaacSimBackend()
        backend.create_world()
        backend.add_robot("so100")

        # Warmup
        for _ in range(10):
            backend.step()

        # Benchmark
        t0 = time.perf_counter()
        for _ in range(num_steps):
            backend.step()
        elapsed = time.perf_counter() - t0

        throughput = int(num_steps * num_envs / elapsed)
        step_time = elapsed / num_steps * 1000

        print(f"  Envs:       {num_envs:,}")
        print(f"  Steps:      {num_steps:,}")
        print(f"  Elapsed:    {elapsed:.3f}s")
        print(f"  Throughput: {throughput:,} env-steps/s")
        print(f"  Per step:   {step_time:.2f}ms")

        backend.destroy()

        return {
            "backend": "Isaac Sim (RTX)",
            "num_envs": num_envs,
            "throughput": throughput,
            "elapsed": elapsed,
            "step_time_ms": step_time,
            "success": True,
        }

    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {"backend": "Isaac Sim (RTX)", "success": False, "error": str(e)}


def print_results_table(results: list[dict]) -> None:
    """Print a formatted comparison table.

    Args:
        results: List of benchmark result dicts.
    """
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print("\n\n" + "=" * 72)
    print("                    BENCHMARK RESULTS")
    print("=" * 72)
    print(f"  {'Backend':<25s} {'Envs':>8s} {'Throughput':>16s} {'Step Time':>12s}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 16} {'─' * 12}")

    for r in sorted(successful, key=lambda x: x["throughput"], reverse=True):
        step_time = r.get("step_time_ms", r.get("step_time_us", 0) / 1000)
        unit = "ms"
        if step_time < 0.1:
            step_time = r.get("step_time_us", step_time * 1000)
            unit = "μs"
        print(
            f"  {r['backend']:<25s} {r['num_envs']:>8,} {r['throughput']:>12,} /s {step_time:>8.2f}{unit}"
        )

    if failed:
        print()
        for r in failed:
            print(f"  {r['backend']:<25s} {'N/A':>8s} {'─ ' + r.get('error', 'failed')[:30]:>30s}")

    if len(successful) >= 2:
        fastest = max(successful, key=lambda x: x["throughput"])
        slowest = min(successful, key=lambda x: x["throughput"])
        speedup = fastest["throughput"] / max(slowest["throughput"], 1)
        print(f"\n  🏆 Fastest: {fastest['backend']} ({speedup:.0f}× faster than {slowest['backend']})")

    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Sample 06: 3-Backend Simulation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cpu-only", action="store_true", help="MuJoCo CPU only")
    parser.add_argument("--newton-only", action="store_true", help="Newton GPU only (all solvers)")
    parser.add_argument("--envs", type=int, default=4096, help="Parallel envs for GPU backends")
    parser.add_argument("--steps", type=int, default=500, help="Steps per benchmark")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")

    args = parser.parse_args()

    print("🏁 strands-robots Simulation Backend Benchmark")
    print("=" * 72)

    results = []

    if args.cpu_only:
        results.append(benchmark_mujoco(num_steps=args.steps))
    elif args.newton_only:
        from strands_robots.newton import SOLVER_MAP

        for solver in sorted(SOLVER_MAP.keys()):
            results.append(benchmark_newton(
                solver=solver,
                num_envs=args.envs,
                num_steps=args.steps,
                device=args.device,
            ))
    else:
        # Full comparison: MuJoCo → Newton → Isaac Sim
        results.append(benchmark_mujoco(num_steps=args.steps))

        for solver in ["mujoco", "featherstone", "semi_implicit"]:
            results.append(benchmark_newton(
                solver=solver,
                num_envs=args.envs,
                num_steps=args.steps,
                device=args.device,
            ))

        results.append(benchmark_isaac_sim(
            num_envs=min(args.envs, 64),  # Isaac Sim typically fewer envs
            num_steps=args.steps,
        ))

    print_results_table(results)


if __name__ == "__main__":
    main()
