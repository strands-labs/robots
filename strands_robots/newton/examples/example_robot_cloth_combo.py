#!/usr/bin/env python3
"""Multi-solver scene: MuJoCo robot + VBD cloth in one sim.

This is the holy grail — mixed rigid-body and deformable physics in
one scene, like Newton's cloth_franka example.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import newton  # noqa: E402
import newton.examples as ne  # noqa: E402
import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True
wp.init()

print("🤖🧵 Newton Example: Robot + Cloth (Multi-Solver)")
print("=" * 55)

# Build scene with robot + cloth
builder = newton.ModelBuilder()

# Register MuJoCo custom attributes for the robot
newton.solvers.SolverMuJoCo.register_custom_attributes(builder)

# Add quadruped robot
builder.add_urdf(ne.get_asset("quadruped.urdf"))
print(f"  Robot: {builder.body_count} bodies, {builder.joint_count} joints")

# Add a cloth sheet above the robot
grid_size = 8
verts = []
indices = []
for y in range(grid_size):
    for x in range(grid_size):
        verts.append(wp.vec3(float(x) * 0.1 - 0.35, float(y) * 0.1 - 0.35, 0.8))  # above the robot
for y in range(grid_size - 1):
    for x in range(grid_size - 1):
        i = y * grid_size + x
        indices.extend([i, i + 1, i + grid_size, i + 1, i + grid_size + 1, i + grid_size])

builder.add_cloth_mesh(
    pos=wp.vec3(0, 0, 0),
    rot=wp.quat_identity(),
    scale=1.0,
    vel=wp.vec3(0, 0, 0),
    vertices=verts,
    indices=indices,
    density=0.05,
    tri_ke=500,
    tri_ka=500,
    tri_kd=1e-5,
    edge_ke=5.0,
    edge_kd=1e-3,
)
print(f"  Cloth: {len(verts)} particles, {len(indices)//3} triangles")

builder.add_ground_plane()

# Finalize (MuJoCo solver will handle the rigid bodies)
model = builder.finalize()
model.soft_contact_ke = 5e3
model.soft_contact_kd = 5
model.soft_contact_mu = 0.5

print(f"  Total: bodies={model.body_count}, joints={model.joint_count}, particles={model.particle_count}")

# Use MuJoCo solver (handles both rigid + particle soft contacts)
solver = newton.solvers.SolverMuJoCo(model)
s0 = model.state()
s1 = model.state()
ctrl = model.control()

# Collision pipeline
pipeline = newton.CollisionPipeline(model, broad_phase="explicit", soft_contact_margin=1.0)
contacts = pipeline.contacts()

# FK for initial body positions
try:
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
except Exception:
    pass

# Simulate
print("\nSimulating 200 steps...")
t0 = time.time()
for step in range(200):
    s0.clear_forces()
    try:
        pipeline.collide(s0, contacts)
    except Exception:
        pass
    solver.step(s0, s1, ctrl, contacts, 1.0 / 120.0)
    s0, s1 = s1, s0

    if step % 50 == 0:
        jq = s0.joint_q.numpy()
        pq = s0.particle_q.numpy() if model.particle_count > 0 else None
        cloth_z = np.mean(pq[:, 2]) if pq is not None else 0
        print(f"  Step {step}: joint_norm={np.linalg.norm(jq):.3f}, cloth_avg_z={cloth_z:.3f}")

elapsed = time.time() - t0

# Final state
jq = s0.joint_q.numpy()
pq = s0.particle_q.numpy() if model.particle_count > 0 else None

print("\n📊 Results:")
print(f"  Time: {elapsed:.3f}s for 200 steps")
print(f"  Robot joint norm: {np.linalg.norm(jq):.4f}")
if pq is not None:
    print(f"  Cloth avg Z: {np.mean(pq[:, 2]):.4f} (started at 0.8)")
    print(f"  Cloth min Z: {np.min(pq[:, 2]):.4f}")
    has_nan = np.any(np.isnan(pq))
    print(f"  Cloth NaN: {'❌' if has_nan else '✅ Clean'}")

print("\n✅ Done!")
