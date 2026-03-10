#!/usr/bin/env python3
"""Compare MuJoCo vs SemiImplicit solvers on same scene."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import newton.examples as ne  # noqa: E402
import numpy as np  # noqa: E402

from strands_robots.newton import NewtonBackend, NewtonConfig  # noqa: E402

print("⚡ Newton Example: Solver Comparison")
print("=" * 50)

results = {}
for solver_name in ["mujoco", "semi_implicit"]:
    try:
        b = NewtonBackend(NewtonConfig(solver=solver_name, device="cuda:0"))
        b.create_world()
        b.add_robot("quad", urdf_path=ne.get_asset("quadruped.urdf"))
        b.replicate(num_envs=1024)

        # Warmup
        for _ in range(3):
            b.step()

        t0 = time.time()
        for _ in range(200):
            b.step()
        elapsed = time.time() - t0

        throughput = int(200 * 1024 / elapsed)
        obs = b.get_observation("quad")
        jp = obs["observations"]["quad"]["joint_positions"]

        results[solver_name] = {
            "throughput": throughput,
            "elapsed": elapsed,
            "joint_sum": float(np.sum(np.abs(jp))) if jp is not None else 0,
        }
        b.destroy()
        print(f"  {solver_name:15s}: {throughput:>10,} env-steps/s  ({elapsed:.3f}s)")
    except Exception as e:
        print(f"  {solver_name:15s}: FAILED — {e}")
        results[solver_name] = None

if len(results) >= 2 and all(v is not None for v in results.values()):
    fastest = max(results, key=lambda k: results[k]["throughput"])
    print(f"\n🏆 Fastest: {fastest}")

print("\n✅ Done!")
