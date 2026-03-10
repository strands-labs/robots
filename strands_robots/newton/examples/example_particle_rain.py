#!/usr/bin/env python3
"""100 particles falling with collision — SemiImplicit + CollisionPipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import time  # noqa: E402

import newton  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True
print("🌧️ Newton Example: Particle Rain (100 particles)")
print("=" * 50)

builder = newton.ModelBuilder()
np.random.seed(42)
for i in range(100):
    pos = wp.vec3(np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(1, 5))
    vel = wp.vec3(0, 0, np.random.uniform(-2, 0))
    builder.add_particle(pos=pos, vel=vel, mass=1.0)

builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=1e4, kd=10, mu=0.5))
model = builder.finalize()
model.soft_contact_ke = 1e4
model.soft_contact_kd = 10
model.soft_contact_mu = 0.5
model.soft_contact_restitution = 0.3
solver = newton.solvers.SolverSemiImplicit(model)
s0, s1 = model.state(), model.state()
ctrl = model.control()
pipeline = newton.CollisionPipeline(model, broad_phase="explicit", soft_contact_margin=5.0)
contacts = pipeline.contacts()
pipeline.collide(s0, contacts)

print(f"Particles: {model.particle_count}")
print("Simulating 300 steps...")

t0 = time.time()
for step in range(300):
    s0.clear_forces()
    pipeline.collide(s0, contacts)
    solver.step(s0, s1, ctrl, contacts, 1.0 / 60.0)
    s0, s1 = s1, s0
    if step % 100 == 0:
        positions = s0.particle_q.numpy()
        avg_z = np.mean(positions[:, 2])
        print(f"  Step {step}: avg_z={avg_z:.3f}")

elapsed = time.time() - t0
positions = s0.particle_q.numpy()
print("\nFinal state:")
print(f"  Time: {elapsed:.3f}s for 300 steps")
print(f"  Avg Z: {np.mean(positions[:, 2]):.4f}")
print(f"  Min Z: {np.min(positions[:, 2]):.4f}")
print(f"  All above ground: {np.all(positions[:, 2] >= -0.1)}")
print(f"  Spread X: [{np.min(positions[:, 0]):.2f}, {np.max(positions[:, 0]):.2f}]")
print(f"  Spread Y: [{np.min(positions[:, 1]):.2f}, {np.max(positions[:, 1]):.2f}]")
print("\n✅ Done!")
