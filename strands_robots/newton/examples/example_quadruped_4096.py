#!/usr/bin/env python3
"""4096 parallel quadrupeds on GPU — Newton's massive parallelism showcase."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import newton.examples as ne  # noqa: E402

from strands_robots.newton import NewtonBackend, NewtonConfig  # noqa: E402

print("🦿 Newton Example: 4096 Parallel Quadrupeds")
print("=" * 50)

b = NewtonBackend(NewtonConfig(solver="mujoco", device="cuda:0"))
b.create_world()
b.add_robot("quad", urdf_path=ne.get_asset("quadruped.urdf"))
r = b.replicate(num_envs=4096)
print(f"Replicated: {r['env_info']['bodies_total']} bodies, {r['env_info']['joints_total']} joints")

# Warmup
for _ in range(5):
    b.step()

# Benchmark
for n_steps in [100, 500, 1000]:
    t0 = time.time()
    for _ in range(n_steps):
        b.step()
    elapsed = time.time() - t0
    throughput = int(n_steps * 4096 / elapsed)
    print(f"  {n_steps} steps: {elapsed:.3f}s → {throughput:,} env-steps/s")

obs = b.get_observation("quad")
jp = obs["observations"]["quad"]["joint_positions"]
print(f"\nJoint positions (first env): {jp[:6].round(4)}")
print(f"Sim time: {b._sim_time:.4f}s, Steps: {b._step_count}")
b.destroy()
print("\n✅ Done!")
