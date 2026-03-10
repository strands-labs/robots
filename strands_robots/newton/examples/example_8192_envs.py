#!/usr/bin/env python3
"""Push Thor to the limit: 8192 parallel environments."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import newton  # noqa: E402
import newton.examples as ne  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True
wp.init()

print("🚀 Newton: 8192 Parallel Environments on NVIDIA Thor")
print("=" * 55)

for num_envs in [256, 1024, 2048, 4096, 8192]:
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.add_mjcf(ne.get_asset("nv_ant.xml"))

    main = newton.ModelBuilder()
    main.replicate(builder, num_envs, spacing=(3.0, 3.0, 0.0))
    main.add_ground_plane()
    newton.solvers.SolverMuJoCo.register_custom_attributes(main)

    model = main.finalize()
    solver = newton.solvers.SolverMuJoCo(model)
    s0, s1 = model.state(), model.state()
    ctrl = model.control()

    # Warmup
    for _ in range(3):
        s0.clear_forces()
        solver.step(s0, s1, ctrl, None, 1.0 / 200.0)
        s0, s1 = s1, s0

    # Benchmark
    t0 = time.time()
    for _ in range(100):
        s0.clear_forces()
        solver.step(s0, s1, ctrl, None, 1.0 / 200.0)
        s0, s1 = s1, s0
    elapsed = time.time() - t0

    throughput = int(100 * num_envs / elapsed)
    bodies = model.body_count
    vram_gb = bodies * 7 * 4 / (1024**3)  # rough estimate

    print(f"  {num_envs:>5} envs | {bodies:>7} bodies | {elapsed:>6.3f}s | {throughput:>10,} env-steps/s")

    del model, solver, s0, s1, ctrl, main
    wp.synchronize()

print("\n✅ Done!")
