#!/usr/bin/env python3
"""
Sample 06 — Newton Parallel: 4096-env GPU physics with solver comparison.

Demonstrates Newton's massive parallelism by running thousands of simulation
environments simultaneously on a single GPU. Compares throughput across all
7 physics solvers supported by Newton.

Level: 3 (Advanced)
Hardware: NVIDIA GPU with 24GB+ VRAM
Runners: thor.yml (AGX Thor) | isaac-sim.yml (EC2 L40S)

Usage:
    python samples/06_raytraced_training/newton_parallel.py

    # With specific solver and env count:
    python samples/06_raytraced_training/newton_parallel.py --solver featherstone --envs 8192

    # Compare all solvers:
    python samples/06_raytraced_training/newton_parallel.py --compare-solvers
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def run_newton_parallel(
    solver: str = "mujoco",
    num_envs: int = 4096,
    num_steps: int = 500,
    device: str = "cuda:0",
) -> dict:
    """Run Newton with parallel environments and benchmark throughput.

    Args:
        solver: Newton solver backend (mujoco, featherstone, semi_implicit, xpbd, ...).
        num_envs: Number of parallel environments on GPU.
        num_steps: Number of simulation steps to benchmark.
        device: CUDA device string.

    Returns:
        Dictionary with benchmark results.
    """
    from strands_robots.newton import NewtonBackend, NewtonConfig

    print(f"\n🚀 Newton Parallel: {num_envs} envs × {solver} solver on {device}")
    print("=" * 60)

    # ── 1. Create backend with config ──
    config = NewtonConfig(
        num_envs=num_envs,
        solver=solver,
        device=device,
    )
    backend = NewtonBackend(config)

    # ── 2. Create world with ground plane ──
    backend.create_world(gravity=(0, 0, -9.81))
    print(f"✅ World created (solver={solver}, device={device})")

    # ── 3. Add a robot ──
    # Newton supports procedural construction for common robots (so100, koch, etc.)
    # and URDF/MJCF loading for any robot with a model file.
    backend.add_robot("so100")
    print("✅ Robot added: so100 (6-DOF desktop arm)")

    # ── 4. Replicate across environments ──
    rep_info = backend.replicate(num_envs=num_envs)
    bodies = rep_info.get("env_info", {}).get("bodies_total", "?")
    joints = rep_info.get("env_info", {}).get("joints_total", "?")
    print(f"✅ Replicated: {num_envs} envs ({bodies} bodies, {joints} joints)")

    # ── 5. Warmup (JIT compile CUDA kernels) ──
    print("⏳ Warming up (JIT compilation)...")
    for _ in range(10):
        backend.step()

    # ── 6. Benchmark ──
    print(f"📊 Benchmarking {num_steps} steps...")
    t0 = time.perf_counter()
    for _ in range(num_steps):
        backend.step()
    elapsed = time.perf_counter() - t0

    throughput = int(num_steps * num_envs / elapsed)
    step_time_ms = elapsed / num_steps * 1000

    # ── 7. Read observations ──
    obs = backend.get_observation("so100")
    robot_obs = obs.get("observations", {}).get("so100", {})
    joint_pos = robot_obs.get("joint_positions")

    print(f"\n{'─' * 60}")
    print(f"  Solver:        {solver}")
    print(f"  Environments:  {num_envs:,}")
    print(f"  Steps:         {num_steps:,}")
    print(f"  Elapsed:       {elapsed:.3f}s")
    print(f"  Per step:      {step_time_ms:.2f}ms")
    print(f"  Throughput:    {throughput:,} env-steps/s")
    if joint_pos is not None:
        print(f"  Joint pos[0]:  {np.array(joint_pos[:6]).round(4)}")
    print(f"{'─' * 60}")

    # ── 8. Cleanup ──
    backend.destroy()

    return {
        "solver": solver,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "elapsed": elapsed,
        "throughput": throughput,
        "step_time_ms": step_time_ms,
    }


def compare_solvers(
    num_envs: int = 1024,
    num_steps: int = 200,
    device: str = "cuda:0",
) -> list[dict]:
    """Compare throughput across Newton's 7 solver backends.

    Not all solvers support articulated robots equally, so we catch errors
    gracefully and report which solvers succeeded.

    Args:
        num_envs: Number of parallel environments.
        num_steps: Steps per solver benchmark.
        device: CUDA device.

    Returns:
        List of benchmark result dicts.
    """
    from strands_robots.newton import SOLVER_MAP

    print("\n⚡ Newton Solver Comparison")
    print(f"   {num_envs} parallel envs × {num_steps} steps each")
    print("=" * 60)

    results = []
    for solver_name in sorted(SOLVER_MAP.keys()):
        try:
            result = run_newton_parallel(
                solver=solver_name,
                num_envs=num_envs,
                num_steps=num_steps,
                device=device,
            )
            results.append(result)
        except Exception as e:
            print(f"\n  ❌ {solver_name}: {e}")
            results.append({"solver": solver_name, "error": str(e)})

    # ── Summary table ──
    print("\n\n🏆 SOLVER COMPARISON RESULTS")
    print("=" * 60)
    print(f"  {'Solver':<18s} {'Throughput':>15s} {'Step time':>12s} {'Status'}")
    print(f"  {'─' * 18} {'─' * 15} {'─' * 12} {'─' * 8}")

    successful = [r for r in results if "throughput" in r]
    for r in sorted(successful, key=lambda x: x["throughput"], reverse=True):
        print(
            f"  {r['solver']:<18s} {r['throughput']:>12,} /s {r['step_time_ms']:>9.2f}ms   ✅"
        )

    failed = [r for r in results if "error" in r]
    for r in failed:
        err_short = r["error"][:40]
        print(f"  {r['solver']:<18s} {'N/A':>15s} {'N/A':>12s}   ❌ {err_short}")

    if successful:
        fastest = max(successful, key=lambda x: x["throughput"])
        print(f"\n  🏆 Fastest: {fastest['solver']} ({fastest['throughput']:,} env-steps/s)")

    return results


def run_diffsim_demo(device: str = "cuda:0") -> dict:
    """Demonstrate Newton's differentiable simulation.

    Optimizes initial velocity of a ball to hit a target position using
    gradient-based optimization through the physics engine.

    Args:
        device: CUDA device.

    Returns:
        Optimization results.
    """
    from strands_robots.newton import NewtonBackend, NewtonConfig

    print("\n🧮 Newton Differentiable Simulation Demo")
    print("=" * 60)
    print("  Optimizing ball trajectory to reach target via gradient descent")

    config = NewtonConfig(
        solver="semi_implicit",
        device=device,
        enable_differentiable=True,
    )
    backend = NewtonBackend(config)
    backend.create_world()

    # Use run_diffsim for trajectory optimization
    # This builds a simple particle scene internally and runs gradient descent
    result = backend.run_diffsim(
        num_steps=36,
        loss_fn=None,  # Uses built-in target-reaching loss
        optimize_params="initial_velocity",
        lr=0.02,
        iterations=50,
    )

    backend.destroy()

    if result.get("success"):
        print("\n  ✅ Optimization converged!")
        print(f"     Initial loss: {result.get('initial_loss', '?'):.4f}")
        print(f"     Final loss:   {result.get('final_loss', '?'):.4f}")
        print(f"     Iterations:   {result.get('iterations', '?')}")
    else:
        print(f"\n  ⚠️ Optimization result: {result}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Sample 06: Newton GPU-Parallel Physics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--solver",
        default="mujoco",
        choices=["mujoco", "featherstone", "semi_implicit", "xpbd", "vbd", "style3d", "implicit_mpm"],
        help="Newton solver backend (default: mujoco)",
    )
    parser.add_argument("--envs", type=int, default=4096, help="Number of parallel envs (default: 4096)")
    parser.add_argument("--steps", type=int, default=500, help="Benchmark steps (default: 500)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device (default: cuda:0)")
    parser.add_argument("--compare-solvers", action="store_true", help="Compare all 7 solvers")
    parser.add_argument("--diffsim", action="store_true", help="Run differentiable sim demo")

    args = parser.parse_args()

    if args.compare_solvers:
        compare_solvers(num_envs=args.envs, num_steps=args.steps, device=args.device)
    elif args.diffsim:
        run_diffsim_demo(device=args.device)
    else:
        run_newton_parallel(
            solver=args.solver,
            num_envs=args.envs,
            num_steps=args.steps,
            device=args.device,
        )


if __name__ == "__main__":
    main()
