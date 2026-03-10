#!/usr/bin/env python3
"""Differentiable simulation: optimize ball trajectory to hit target."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import newton  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True
print("🧮 Newton Example: Differentiable Ball Optimization")
print("=" * 50)

builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
builder.add_particle(pos=wp.vec3(0, -0.5, 1), vel=wp.vec3(0, 5, -5), mass=1.0)
ke, kd, mu = 1e4, 1e1, 0.2
builder.add_shape_box(
    body=-1,
    xform=wp.transform(wp.vec3(0, 2, 1), wp.quat_identity()),
    hx=1.0,
    hy=0.25,
    hz=1.0,
    cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=0, kd=kd, mu=mu),
)
builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=ke, kf=0, kd=kd, mu=mu))
model = builder.finalize(requires_grad=True)
model.soft_contact_ke = ke
model.soft_contact_kd = kd
model.soft_contact_mu = mu
model.soft_contact_restitution = 1.0
solver = newton.solvers.SolverSemiImplicit(model)

target = wp.vec3(0, -2, 1.5)
loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)


@wp.kernel
def loss_kernel(pos: wp.array(dtype=wp.vec3), target: wp.vec3, loss: wp.array(dtype=float)):
    delta = pos[0] - target
    loss[0] = wp.dot(delta, delta)


@wp.kernel
def step_kernel(x: wp.array(dtype=wp.vec3), grad: wp.array(dtype=wp.vec3), alpha: float):
    tid = wp.tid()
    x[tid] = x[tid] - grad[tid] * alpha


N_steps, substeps = 36, 8
dt = (1.0 / 60.0) / substeps
states = [model.state(requires_grad=True) for _ in range(N_steps * substeps + 1)]
control = model.control()
pipeline = newton.CollisionPipeline(model, broad_phase="explicit", soft_contact_margin=10.0, requires_grad=True)
contacts = pipeline.contacts()
pipeline.collide(states[0], contacts)

print(f"Target: {target}")
print("Optimizing initial velocity over 50 iterations...")
print()

losses = []
for it in range(50):
    tape = wp.Tape()
    with tape:
        for s in range(N_steps):
            for i in range(substeps):
                t = s * substeps + i
                states[t].clear_forces()
                solver.step(states[t], states[t + 1], control, contacts, dt)
        wp.launch(loss_kernel, dim=1, inputs=[states[-1].particle_q, target, loss])
    tape.backward(loss)
    loss_val = float(loss.numpy()[0])
    losses.append(loss_val)
    if it % 10 == 0:
        vel = states[0].particle_qd.numpy()[0]
        print(f"  iter {it:3d}: loss={loss_val:.4f}  vel=[{vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}]")
    x = states[0].particle_qd
    wp.launch(step_kernel, dim=len(x), inputs=[x, x.grad, 0.02])
    tape.zero()

print(f"\n  Initial loss: {losses[0]:.4f}")
print(f"  Final loss:   {losses[-1]:.4f}")
print(f"  Reduction:    {(1-losses[-1]/losses[0])*100:.1f}%")
final_vel = states[0].particle_qd.numpy()[0]
print(f"  Optimized vel: [{final_vel[0]:.3f}, {final_vel[1]:.3f}, {final_vel[2]:.3f}]")
print("\n✅ Done!")
