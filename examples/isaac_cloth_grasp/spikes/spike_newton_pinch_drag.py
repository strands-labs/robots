"""Spike 2 (issue #5): Newton (VBD FEM cloth) frictional PINCH + DRAG on the GB10.

The question: does a *frictional enclosure grasp* HOLD on Newton's FEM cloth through a
sustained drag — the thing PhysX particle cloth cannot do (rigid grippers penetrate +
slip; NVIDIA forum 332704)?

Test (cm scale, the regime Newton's own cloth+Franka example uses for VBD numerics):

1. A cloth sheet drapes over a box ("mattress") with an overhang hanging down the +x
   face — exactly the bed-cover geometry.
2. Two rigid finger PLATES (kinematic, like a parallel pinch) close on the hanging
   overhang — the cloth is *enclosed*, not point-pinched: the plates squeeze a strip
   of cloth between high-friction faces.
3. The plates then drag up and across the box top ("headward") through a full
   ~45 cm stroke, hauling the whole sheet against gravity + mattress friction.

PASS = the pinched cloth strip stays between the plates through the full stroke
(slip < a few cm), eye-verified in rendered frames + a printed slip metric.

Kinematic plates are legitimate HERE: this is a unit test of the CLOTH CONTACT MODEL
(does friction hold?), not of the robot (no robot in scene). The #2-demo integration
keeps the no-kinematic-cheats rule — the hand will be the force-limited Inspire hand.

Run::

    .spikes/newton-venv/bin/python examples/isaac_cloth_grasp/spikes/spike_newton_pinch_drag.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverVBD

p = argparse.ArgumentParser()
p.add_argument("--out", default=str(Path(__file__).parent / "out" / "newton_pinch_drag"))
p.add_argument("--fps", type=int, default=10)
p.add_argument("--substeps", type=int, default=10)
p.add_argument("--iterations", type=int, default=10)
ARGS = p.parse_args()

# ── geometry (cm) ────────────────────────────────────────────────────────────
BOX_H = (40.0, 60.0, 25.0)        # half-extents: box top at z=50, +x face at x=40
BOX_TOP = 2.0 * BOX_H[2]
BOX_XFACE = BOX_H[0]
CELL = 2.5
DIMX, DIMY = 40, 32               # 100 x 80 cm sheet
SHEET_Z0 = BOX_TOP + 2.0
# Sheet spawns centred so ~30 cm overhangs the +x edge
SHEET_X0 = BOX_XFACE + 30.0 - DIMX * CELL   # grid spans [SHEET_X0, SHEET_X0 + DIMX*CELL]
SHEET_Y0 = -DIMY * CELL / 2.0

PLATE_H = (1.0, 6.0, 4.0)         # finger plates: thin in x, 12 cm wide, 8 cm tall
GRIP_Z = BOX_TOP - 8.0            # pinch the hanging overhang 8 cm below the top edge
PLATE_GAP_OPEN = 12.0
PLATE_GAP_CLOSED = 2.1            # squeeze: cloth strip ~2*particle_radius = 1.6 cm

T_SETTLE, T_CLOSE, T_LIFT, T_DRAG, T_HOLD = 1.5, 0.6, 0.8, 2.5, 0.6
DRAG_DX = -45.0                   # full stroke across the box top ("headward")
LIFT_DZ = 14.0                    # up over the box edge first


def main() -> None:
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)

    builder = newton.ModelBuilder(gravity=-981.0)   # cm/s^2
    builder.add_ground_plane()

    # mattress box (static, world body -1)
    box_cfg = builder.default_shape_cfg.copy()
    box_cfg.mu = 0.5
    builder.add_shape_box(-1, wp.transform(wp.vec3(0.0, 0.0, BOX_H[2]), wp.quat_identity()),
                          hx=BOX_H[0], hy=BOX_H[1], hz=BOX_H[2], cfg=box_cfg)

    # finger plates: kinematic bodies, high-friction faces (rubberized fingertips)
    plate_cfg = builder.default_shape_cfg.copy()
    plate_cfg.mu = 1.5
    plate_cfg.density = 1.0
    plates = []
    for k in (0, 1):
        b = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, -50.0 - 10.0 * k), wp.quat_identity()),
            is_kinematic=True, label=f"plate{k}")
        builder.add_shape_box(b, hx=PLATE_H[0], hy=PLATE_H[1], hz=PLATE_H[2], cfg=plate_cfg)
        plates.append(b)

    # cloth: FEM triangle grid (VBD), bed-cover-ish stiffness (cm-scale recipe from
    # Newton's own cloth_franka example)
    builder.add_cloth_grid(
        pos=wp.vec3(SHEET_X0, SHEET_Y0, SHEET_Z0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=DIMX, dim_y=DIMY, cell_x=CELL, cell_y=CELL,
        mass=0.4,                       # per particle; ~0.5 kg sheet total
        tri_ke=1.0e4, tri_ka=1.0e4, tri_kd=1.5e-6,
        edge_ke=5.0, edge_kd=1.0e-2,
        particle_radius=0.8,
    )
    builder.color(include_bending=True)

    model = builder.finalize()
    model.soft_contact_ke = 1.0e4
    model.soft_contact_kd = 1.0e-2
    model.soft_contact_mu = 0.25        # cloth-cloth

    solver = SolverVBD(
        model, iterations=ARGS.iterations,
        particle_enable_self_contact=True,
        particle_self_contact_radius=0.2, particle_self_contact_margin=0.2,
        particle_collision_detection_interval=-1,
    )

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()

    n_part = model.particle_count
    frame_dt = 1.0 / ARGS.fps
    sim_dt = (1.0 / 60.0) / ARGS.substeps
    steps_per_frame = max(1, round(frame_dt / (1.0 / 60.0)))

    # plate motion script ------------------------------------------------------
    grip_x = BOX_XFACE + 1.2            # the hanging overhang sits just off the +x face
    t_total = T_SETTLE + T_CLOSE + T_LIFT + T_DRAG + T_HOLD

    def plate_pose(t: float):
        """Return ((x0,z0),(x1,z1), vx, vz) for outer/inner plates at time t."""
        if t < T_SETTLE:
            return (grip_x + PLATE_GAP_OPEN, GRIP_Z), (grip_x - PLATE_GAP_OPEN, GRIP_Z), 0.0, 0.0
        t1 = t - T_SETTLE
        if t1 < T_CLOSE:                # close the pinch
            a = t1 / T_CLOSE
            gap = PLATE_GAP_OPEN + (PLATE_GAP_CLOSED - PLATE_GAP_OPEN) * a
            v = (PLATE_GAP_CLOSED - PLATE_GAP_OPEN) / T_CLOSE
            return (grip_x + gap / 2.0, GRIP_Z), (grip_x - gap / 2.0, GRIP_Z), v, 0.0
        t2 = t1 - T_CLOSE
        if t2 < T_LIFT:                 # lift up over the edge
            a = t2 / T_LIFT
            z = GRIP_Z + LIFT_DZ * a
            return (grip_x + PLATE_GAP_CLOSED / 2.0, z), (grip_x - PLATE_GAP_CLOSED / 2.0, z), \
                0.0, LIFT_DZ / T_LIFT
        t3 = t2 - T_LIFT
        if t3 < T_DRAG:                 # the headward stroke
            a = t3 / T_DRAG
            x = grip_x + DRAG_DX * a
            z = GRIP_Z + LIFT_DZ
            return (x + PLATE_GAP_CLOSED / 2.0, z), (x - PLATE_GAP_CLOSED / 2.0, z), \
                DRAG_DX / T_DRAG, 0.0
        x = grip_x + DRAG_DX
        z = GRIP_Z + LIFT_DZ
        return (x + PLATE_GAP_CLOSED / 2.0, z), (x - PLATE_GAP_CLOSED / 2.0, z), 0.0, 0.0

    def set_plates(state, t):
        (xo, zo), (xi, zi), vx, vz = plate_pose(t)
        q = state.body_q.numpy()
        qd = state.body_qd.numpy()
        q[plates[0]] = [xo, 0.0, zo, 0.0, 0.0, 0.0, 1.0]
        q[plates[1]] = [xi, 0.0, zi, 0.0, 0.0, 0.0, 1.0]
        qd[plates[0]] = [vx, 0.0, vz, 0.0, 0.0, 0.0]   # linear first, world frame
        qd[plates[1]] = [vx, 0.0, vz, 0.0, 0.0, 0.0]
        state.body_q.assign(q)
        state.body_qd.assign(qd)

    # bookkeeping for the slip metric -----------------------------------------
    gripped_ids: np.ndarray | None = None
    grip_offset0 = None

    tri_indices = model.tri_indices.numpy().reshape(-1, 3)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    def render(frame_i, t, pts):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.computed_zorder = False
        # box
        bx, by, bz = BOX_H
        for (s, axis) in (((1, 1), 2),):
            pass
        corners = np.array([[sx * bx, sy * by, bz + sz * bz]
                            for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        faces_idx = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]
        ax.add_collection3d(Poly3DCollection([corners[list(f)] for f in faces_idx],
                                             facecolor="#5a7aa6", alpha=0.35, zorder=1))
        # cloth
        ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], triangles=tri_indices,
                        color="#ddddec", edgecolor="none", linewidth=0, alpha=0.95, zorder=2)
        # plates
        (xo, zo), (xi, zi), _, _ = plate_pose(t)
        for (px, pz), col in (((xo, zo), "#cc3333"), ((xi, zi), "#cc8833")):
            hx, hy, hz = PLATE_H
            c = np.array([[px + sx * hx, sy * hy, pz + sz * hz]
                          for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            ax.add_collection3d(Poly3DCollection([c[list(f)] for f in faces_idx],
                                                 facecolor=col, alpha=0.9, zorder=3))
        ax.set_xlim(-70, 80); ax.set_ylim(-75, 75); ax.set_zlim(0, 90)
        ax.set_box_aspect((150, 150, 90))
        ax.view_init(elev=18, azim=-55)
        ax.set_title(f"t={t:.2f}s")
        fig.tight_layout()
        fig.savefig(out / f"frame_{frame_i:04d}.png", dpi=80)
        plt.close(fig)

    t = 0.0
    frame_i = 0
    n_frames = int(t_total * ARGS.fps) + 1
    print(f"[spike] {n_part} particles, {len(tri_indices)} tris, {n_frames} frames", flush=True)
    while t < t_total:
        for _ in range(steps_per_frame):
            for _ in range(ARGS.substeps):
                state_0.clear_forces()
                set_plates(state_0, t)
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0
                t += sim_dt

        pts = state_0.particle_q.numpy()

        # latch the gripped set the moment the pinch closes
        if gripped_ids is None and t >= T_SETTLE + T_CLOSE + 0.05:
            (xo, zo), (xi, zi), _, _ = plate_pose(t)
            cx = (xo + xi) / 2.0
            m = ((np.abs(pts[:, 0] - cx) < 2.5) & (np.abs(pts[:, 2] - zo) < PLATE_H[2])
                 & (np.abs(pts[:, 1]) < PLATE_H[1]))
            gripped_ids = np.where(m)[0]
            if len(gripped_ids):
                grip_offset0 = pts[gripped_ids].mean(axis=0) - np.array([cx, 0.0, zo])
            print(f"[spike] t={t:.2f}s pinch closed on {len(gripped_ids)} particles", flush=True)

        if gripped_ids is not None and len(gripped_ids):
            (xo, zo), (xi, zi), _, _ = plate_pose(t)
            cx = (xo + xi) / 2.0
            off = pts[gripped_ids].mean(axis=0) - np.array([cx, 0.0, zo])
            slip = np.linalg.norm(off - grip_offset0)
            print(f"[spike] t={t:.2f}s slip={slip:.2f} cm  cloth-min-x={pts[:, 0].min():.1f}",
                  flush=True)

        render(frame_i, t, pts)
        frame_i += 1

    # verdict ------------------------------------------------------------------
    pts = state_0.particle_q.numpy()
    if gripped_ids is not None and len(gripped_ids):
        (xo, zo), (xi, zi), _, _ = plate_pose(t_total)
        cx = (xo + xi) / 2.0
        off = pts[gripped_ids].mean(axis=0) - np.array([cx, 0.0, zo])
        slip = np.linalg.norm(off - grip_offset0)
        print(f"[spike] FINAL slip = {slip:.2f} cm over a {abs(DRAG_DX):.0f} cm stroke "
              f"({'HOLDS' if slip < 5.0 else 'SLIPS'})", flush=True)
    else:
        print("[spike] FINAL: pinch never latched particles — grasp FAILED", flush=True)
    print(f"[spike] DONE — {frame_i} frames in {out}", flush=True)


if __name__ == "__main__":
    main()
