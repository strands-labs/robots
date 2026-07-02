"""Issue #5 acceptance harness: sensor-targeted GRIP → full DRAG stroke → PEEL under snag.

The full grasp lifecycle on the Newton FEM cloth, with every force bounded and no
privileged-state targeting:

  SETTLE     the bed-cover drapes over the MATTRESS-ON-FRAME with a ~30 cm overhang
             hanging down the +x mattress side. The frame is inset (narrower than
             the mattress), as on a real bed — the air gap under the mattress edge
             is what makes the hanging hem graspable from the side.
  SEEK_DOWN  the wrist descends OUTSIDE the bed to hem height — pure bed-geometry
             prior (the same knowledge the #2 robots use to walk to the bedside).
  SEEK_IN    the open jaw slides in horizontally, straddling the hanging hem —
             the side-approach + slide-in primitive the cloth-grasping literature
             converges on (Tirumala et al. IROS'22; Qian et al. IROS'20). The
             short-range proximity sensor (bundled #2 sensor model) fires only
             within centimetres of actual cloth; the slide stops on the reading.
  CENTER     slide the remaining sensed distance so the hem is between the pads.
  GRIP       the force-limited pinch closes (Inspire-ratio budget, ≈4.1× cover wt).
  LIFT       up over the mattress edge (the flap drapes over the lower pad).
  DRAG A     the wrist hauls headward through a FULL 45 cm stroke. PASS = the
             cloth stays in the pinch (slip ≪ stroke), eye-verified in frames.
  DRAG B     the cover's far edge is pinned ("snag") and the wrist keeps pulling →
             the load exceeds what friction at the pinch budget can transmit.
             PASS = the cloth PEELS physically; the sensor notices, the #2
             GraspDecision calls the slip, the hand opens (requirement E).

Run::

    .spikes/newton-venv/bin/python examples/isaac_cloth_grasp/harness_grip_drag_peel.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))

import newton  # noqa: E402
from newton.solvers import SolverVBD  # noqa: E402

from newton_cloth import (  # noqa: E402
    BedsheetParams, ForceLimitedPinch, GraspDecision, HandClothSensor, PinchParams,
    add_bedsheet,
)
from newton_cloth.bedsheet import cloth_points, finalize_cloth_materials  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--out", default=str(Path(__file__).parent / "spikes" / "out" / "grip_drag_peel"))
p.add_argument("--fps", type=int, default=10)
# 12 substeps × 16 VBD iterations is the validated config that holds the drag (v21):
# the stiff fingertip contact (sized to the pinch force) needs the extra implicit
# iterations to converge without the pad punching through the wad.
p.add_argument("--substeps", type=int, default=12)
p.add_argument("--iterations", type=int, default=16)
ARGS = p.parse_args()

# ── scene (cm): mattress on an inset frame, like the #2 bed ─────────────────
MAT_H = (40.0, 60.0, 12.5)         # mattress half-extents; top at 50, bottom at 25
FRAME_H = (32.0, 52.0, 12.5)       # inset frame under it; the 8 cm side air gap
MAT_TOP = 25.0 + 2 * MAT_H[2]
MAT_BOT = 25.0
MAT_XFACE = MAT_H[0]
OVERHANG = 30.0                    # hem bottom z ≈ 20: the flap ENDS just below the
                                   # mattress, so the open jaw can STRADDLE the free hem
                                   # edge (a 45 cm floor-length curtain cannot be straddled
                                   # at all — the slide-in just plows through it and the
                                   # cloth wraps the leading pad; eye-verified v16)

X_CLEAR = MAT_XFACE + 8.0          # descend line: jaw fully outside the bed
HEM_Z = MAT_BOT - 3.5              # grab height: below the mattress, beside the frame
PARK = (X_CLEAR, 0.0, MAT_TOP + 25.0)

T_SETTLE = 1.5
SEEK_SPEED = 18.0                  # cm/s (descent and slide-in)
T_CLOSE = 1.4   # close + the ~0.7 s pinch-force ramp
T_RETRACT = 1.4                    # slide back out from under the mattress edge FIRST
LIFT_DZ, T_LIFT = 38.0, 3.2        # then up over the mattress edge — SLOWLY: a fast
                                   # haul peels a thin hem grip around the pad edge
DRAG_DX, T_DRAG_A = -45.0, 2.5     # the full headward stroke
DRAG_B_DX, T_DRAG_B = -35.0, 2.0   # keep pulling into the snag
T_TAIL = 0.6


def main() -> int:
    out = Path(ARGS.out)
    out.mkdir(parents=True, exist_ok=True)

    sheet_params = BedsheetParams()
    builder = newton.ModelBuilder(gravity=-981.0)
    builder.add_ground_plane()
    box_cfg = builder.default_shape_cfg.copy()
    box_cfg.mu = 0.5
    builder.add_shape_box(-1, wp.transform(wp.vec3(0.0, 0.0, 25.0 + MAT_H[2]), wp.quat_identity()),
                          hx=MAT_H[0], hy=MAT_H[1], hz=MAT_H[2], cfg=box_cfg)
    builder.add_shape_box(-1, wp.transform(wp.vec3(0.0, 0.0, FRAME_H[2]), wp.quat_identity()),
                          hx=FRAME_H[0], hy=FRAME_H[1], hz=FRAME_H[2], cfg=box_cfg)

    sp = sheet_params
    sheet_mass = (sp.dim_x + 1) * (sp.dim_y + 1) * sp.particle_mass
    pinch = ForceLimitedPinch(builder, sheet_mass, PinchParams(plate_hz=3.0, plate_hy=8.0),
                              spawn=(PARK[0], 0.0, PARK[2] + 20.0))

    # sheet on the mattress top, OVERHANG past the +x edge
    sheet = add_bedsheet(
        builder,
        pos=(MAT_XFACE + OVERHANG - sp.dim_x * sp.cell, -sp.dim_y * sp.cell / 2.0, MAT_TOP + 2.0),
        params=sp)
    builder.color(include_bending=True)

    model = builder.finalize()
    finalize_cloth_materials(model, sp)

    solver = SolverVBD(
        model, iterations=ARGS.iterations,
        particle_enable_self_contact=True,
        particle_self_contact_radius=0.2, particle_self_contact_margin=0.2,
        particle_collision_detection_interval=-1,
        friction_epsilon=1.0e-3,   # tighter stick region (IPC smoothed Coulomb)
    )

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    contacts = model.contacts()

    sensor = HandClothSensor(sensing_range=15.0)
    decision = GraspDecision(contact_gap=4.0, slip_gap=12.0, slip_patience=8)

    sim_dt = (1.0 / 60.0) / ARGS.substeps
    frame_dt = 1.0 / ARGS.fps
    steps_per_frame = max(1, round(frame_dt * 60.0))

    tri_indices = model.tri_indices.numpy().reshape(-1, 3)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    FACES = [(0, 1, 3, 2), (4, 5, 7, 6), (0, 1, 5, 4), (2, 3, 7, 6), (0, 2, 6, 4), (1, 3, 7, 5)]

    def boxpoly(center, half):
        c = np.array([[center[0] + sx * half[0], center[1] + sy * half[1], center[2] + sz * half[2]]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return [c[list(f)] for f in FACES]

    def render(frame_i, t, pts, plate_pos, label):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.computed_zorder = False
        ax.add_collection3d(Poly3DCollection(boxpoly((0, 0, FRAME_H[2]), FRAME_H),
                                             facecolor="#46608c", alpha=0.35, zorder=1))
        ax.add_collection3d(Poly3DCollection(boxpoly((0, 0, 25.0 + MAT_H[2]), MAT_H),
                                             facecolor="#5a7aa6", alpha=0.35, zorder=1))
        ax.plot_trisurf(pts[:, 0], pts[:, 1], pts[:, 2], triangles=tri_indices,
                        color="#ddddec", edgecolor="none", alpha=0.95, zorder=2)
        php = pinch.p
        for pp, col in zip(plate_pos, ("#cc3333", "#cc8833")):
            ax.add_collection3d(Poly3DCollection(
                boxpoly(pp, (php.plate_hx, php.plate_hy, php.plate_hz)),
                facecolor=col, alpha=0.9, zorder=3))
        ax.set_xlim(-70, 80); ax.set_ylim(-75, 75); ax.set_zlim(0, 95)
        ax.set_box_aspect((150, 150, 95))
        ax.view_init(elev=18, azim=-55)
        ax.set_title(f"{label}  t={t:.2f}s")
        fig.tight_layout()
        fig.savefig(out / f"frame_{frame_i:04d}.png", dpi=80)
        plt.close(fig)

    # ── behaviour state machine ───────────────────────────────────────────────
    phase = "SETTLE"
    phase_t0 = 0.0
    grab_pose = None
    slide_x = None
    gripped_ids = None
    grip_offset0 = None
    released_at = None
    verdicts = {}

    def gripped_slip(pts):
        c = pinch.center(state_0)
        off = pts[gripped_ids].mean(axis=0) - c
        return float(np.linalg.norm(off - grip_offset0))

    t = 0.0
    frame_i = 0
    t_total = T_SETTLE + 4.5 + T_CLOSE + T_RETRACT + T_LIFT + T_DRAG_A + T_DRAG_B + T_TAIL
    print(f"[harness] particles={model.particle_count} tris={len(tri_indices)}", flush=True)

    while t < t_total:
        for _ in range(steps_per_frame):
            for _ in range(ARGS.substeps):
                state_0.clear_forces()
                pinch.apply_forces(state_0, sim_dt)
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0
                t += sim_dt

            pts_now = cloth_points(state_0, sheet)
            # proximity sensor at the pinch midpoint between the pads (where a real
            # palm/fingertip sensor sits when the jaw straddles fabric)
            hand = pinch.center(state_0)
            sensed = sensor.read(hand, pts_now)
            true_gap = sensor.true_gap(hand, pts_now)

            if phase == "SETTLE":
                pinch.command(PARK)
                if t - phase_t0 >= T_SETTLE:
                    phase, phase_t0 = "SEEK_DOWN", t
                    print(f"[harness] t={t:.2f}s SEEK_DOWN (outside the bed)", flush=True)
            elif phase == "SEEK_DOWN":
                z = PARK[2] - SEEK_SPEED * (t - phase_t0)
                pinch.command((X_CLEAR, 0.0, max(z, HEM_Z)), (0.0, 0.0, -SEEK_SPEED))
                if z <= HEM_Z:
                    phase, phase_t0 = "SEEK_IN", t
                    print(f"[harness] t={t:.2f}s SEEK_IN (slide the open jaw under the "
                          "mattress edge)", flush=True)
            elif phase == "SEEK_IN":
                x = X_CLEAR - SEEK_SPEED * 0.5 * (t - phase_t0)
                pinch.command((x, 0.0, HEM_Z), (-SEEK_SPEED * 0.5, 0.0, 0.0))
                if decision.on_reach(sensed, true_gap):
                    slide_x = hand[0] - sensed       # carry on by the sensed distance
                    phase, phase_t0 = "CENTER", t
                    print(f"[harness] t={t:.2f}s sensor contact (sensed={sensed:.1f}cm) "
                          "→ CENTER", flush=True)
                elif x < MAT_XFACE - 6.0:
                    print(f"[harness] t={t:.2f}s SEEK FAILED — sensor never fired", flush=True)
                    verdicts["seek"] = "FAIL"
                    phase = "DONE"
            elif phase == "CENTER":
                xc = pinch.cmd_center[0]
                xn = max(slide_x, xc - SEEK_SPEED * 0.5 * sim_dt * ARGS.substeps)
                pinch.command((xn, 0.0, HEM_Z), (-SEEK_SPEED * 0.5, 0.0, 0.0))
                if xn <= slide_x + 0.05:
                    grab_pose = np.array([slide_x, 0.0, HEM_Z])
                    pinch.command(grab_pose)
                    pinch.close()
                    phase, phase_t0 = "CLOSE", t
                    print(f"[harness] t={t:.2f}s centred → CLOSE", flush=True)
            elif phase == "CLOSE":
                pinch.command(grab_pose)
                if t - phase_t0 >= T_CLOSE:
                    c = pinch.center(state_0)
                    m = (np.abs(pts_now[:, 0] - c[0]) < 3.0) \
                        & (np.abs(pts_now[:, 2] - c[2]) < pinch.p.plate_hz) \
                        & (np.abs(pts_now[:, 1] - c[1]) < pinch.p.plate_hy)
                    gripped_ids = np.where(m)[0]
                    pp = pinch.plate_positions(state_0)
                    print(f"[harness] diag: pads x={pp[0][0]:.2f}/{pp[1][0]:.2f} "
                          f"z={pp[0][2]:.1f}/{pp[1][2]:.1f} gap={pinch.gap(state_0):.2f} "
                          f"(diagnostic only)", flush=True)
                    if len(gripped_ids) == 0:
                        print(f"[harness] t={t:.2f}s GRIP FAILED — nothing pinched", flush=True)
                        verdicts["grip"] = "FAIL"
                        phase = "DONE"
                    else:
                        grip_offset0 = pts_now[gripped_ids].mean(axis=0) - c
                        verdicts["grip"] = f"PASS ({len(gripped_ids)} particles pinched)"
                        phase, phase_t0 = "RETRACT", t
                        print(f"[harness] t={t:.2f}s GRIPPED {len(gripped_ids)} particles "
                              "→ RETRACT (clear the mattress edge — the inner finger "
                              "cannot lift through the mattress)", flush=True)
            elif phase == "RETRACT":
                a = min(1.0, (t - phase_t0) / T_RETRACT)
                rx = grab_pose[0] + (X_CLEAR + 2.0 - grab_pose[0]) * a
                pinch.command((rx, 0.0, HEM_Z),
                              ((X_CLEAR + 2.0 - grab_pose[0]) / T_RETRACT, 0.0, 0.0))
                if a >= 1.0:
                    grab_pose = np.array([X_CLEAR + 2.0, 0.0, HEM_Z])
                    phase, phase_t0 = "LIFT", t
                    print(f"[harness] t={t:.2f}s clear of the edge → LIFT", flush=True)
            elif phase == "LIFT":
                a = min(1.0, (t - phase_t0) / T_LIFT)
                pinch.command(grab_pose + np.array([2.0 * a, 0.0, LIFT_DZ * a]),
                              (2.0 / T_LIFT, 0.0, LIFT_DZ / T_LIFT))
                if a >= 1.0:
                    phase, phase_t0 = "DRAG_A", t
                    print(f"[harness] t={t:.2f}s DRAG A — the full headward stroke", flush=True)
            elif phase == "DRAG_A":
                a = min(1.0, (t - phase_t0) / T_DRAG_A)
                pinch.command(grab_pose + np.array([2.0 + DRAG_DX * a, 0.0, LIFT_DZ]),
                              (DRAG_DX / T_DRAG_A, 0.0, 0.0))
                decision.on_pull(sensed, true_gap)
                if a >= 1.0:
                    slip_a = gripped_slip(pts_now)
                    held = decision.holding and slip_a < 6.0
                    verdicts["drag_full_stroke"] = (
                        f"{'PASS' if held else 'FAIL'} (slip {slip_a:.2f} cm over "
                        f"{abs(DRAG_DX):.0f} cm, decision.holding={decision.holding})")
                    print(f"[harness] t={t:.2f}s DRAG A done: {verdicts['drag_full_stroke']}",
                          flush=True)
                    inv = model.particle_inv_mass.numpy()
                    nx, ny = sheet["dim"]
                    far = [sheet["vid"](0, j) for j in range(0, ny + 1)]   # head-side edge
                    inv[far] = 0.0
                    model.particle_inv_mass.assign(inv)
                    phase, phase_t0 = "DRAG_B", t
                    print(f"[harness] t={t:.2f}s SNAG pinned {len(far)} far-edge particles "
                          "→ DRAG B (overload)", flush=True)
            elif phase == "DRAG_B":
                a = min(1.0, (t - phase_t0) / T_DRAG_B)
                pinch.command(grab_pose + np.array([2.0 + DRAG_DX + DRAG_B_DX * a, 0.0, LIFT_DZ]),
                              (DRAG_B_DX / T_DRAG_B, 0.0, 0.0))
                if decision.on_pull(sensed, true_gap) and released_at is None:
                    released_at = t
                    pinch.open()
                    print(f"[harness] t={t:.2f}s decision: SLIPPED → OPEN (release, req E)",
                          flush=True)
                if a >= 1.0:
                    slip_b = gripped_slip(pts_now)
                    peeled = slip_b > 8.0 or released_at is not None
                    verdicts["peel_under_snag"] = (
                        f"{'PASS' if peeled else 'FAIL'} (slip {slip_b:.2f} cm, "
                        f"released={'yes' if released_at else 'no'})")
                    print(f"[harness] t={t:.2f}s DRAG B done: {verdicts['peel_under_snag']}",
                          flush=True)
                    phase, phase_t0 = "TAIL", t
            elif phase == "TAIL":
                if t - phase_t0 >= T_TAIL:
                    phase = "DONE"
            if phase == "DONE":
                break

        pts = cloth_points(state_0, sheet)
        if not np.isfinite(pts).all():
            verdicts["stability"] = "FAIL (NaN cloth state)"
            print("[harness] NaN cloth state — ABORT", flush=True)
            break
        render(frame_i, t, pts, pinch.plate_positions(state_0), phase)
        frame_i += 1
        if phase == "DONE":
            break

    verdicts.setdefault("stability", "PASS (no NaN, ran to completion)")
    print("\n[harness] ── VERDICTS ──", flush=True)
    for k, v in verdicts.items():
        print(f"[harness]   {k:20s} {v}", flush=True)
    print(f"[harness] {frame_i} frames in {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
