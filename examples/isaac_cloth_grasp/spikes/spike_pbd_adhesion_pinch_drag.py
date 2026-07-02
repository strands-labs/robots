"""Spike 4 (issue #5): PBD particle cloth + ADHESION — the DexGarmentLab recipe, in-engine.

DexGarmentLab (arXiv:2505.11032, Isaac Sim 4.5) reports dexterous hands grasping PBD
particle-cloth garments **by physical force, without attachments**, by tuning the PBD
material: particle–rigid ADHESION (contact persists without continuous pressure),
high particle–rigid FRICTION, plus adhesion/friction *scales* for internal cohesion.
NVIDIA's own advice on the particle-cloth grip failure (forum 332704) adds: convex
colliders and 240–360 Hz physics.

This spike applies that recipe to the ISSUE-#2 BEDSHEET (the exact `cloth.py` builder
from the bed-making demo) and runs the same pinch-and-drag protocol as the Newton and
FEM spikes. A PASS here is the cheapest possible #2 integration: keep the particle
cloth, tune the material, grip by friction+adhesion — no tabs, no springs.

Run::

    cd ~/workspaces/git/IsaacLab
    export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
      ~/workspaces/git/robots-issue5/examples/isaac_cloth_grasp/spikes/spike_pbd_adhesion_pinch_drag.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--frames-dir", default=str(Path(__file__).parent / "out" / "pbd_adhesion_pinch_drag"))
p.add_argument("--adhesion", type=float, default=0.5,
               help="PBD material adhesion (0 = baseline / the documented failure)")
p.add_argument("--friction", type=float, default=1.2)
ARGS = p.parse_args()

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({"headless": True, "enable_cameras": True})
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

from examples.isaac_bed_making import cloth as clothmod  # noqa: E402

SIM_DT = 1.0 / 240.0      # NVIDIA particle-grip advice: 240–360 Hz

# geometry (m) — mirrors the other spikes
BOX_H = (0.40, 0.60, 0.25)
BOX_TOP = 2 * BOX_H[2]
BOX_XFACE = BOX_H[0]
PLATE_H = (0.01, 0.06, 0.04)
GRIP_Z = BOX_TOP - 0.10
GAP_OPEN, GAP_CLOSED = 0.24, 0.033  # centres: > 2cm plate thickness + ~50% squeeze
                                    # of the 2.5cm particle slab (gap 0.018 had the
                                    # plates interpenetrating + ejecting the particles)
T_SETTLE, T_CLOSE, T_LIFT, T_DRAG, T_HOLD = 2.0, 0.6, 0.8, 2.5, 0.6
DRAG_DX, LIFT_DZ = -0.45, 0.14
GRIP_X = BOX_XFACE + 0.012


def plate_pose(t: float):
    if t < T_SETTLE:
        return (GRIP_X + GAP_OPEN / 2, GRIP_Z), (GRIP_X - GAP_OPEN / 2, GRIP_Z)
    t1 = t - T_SETTLE
    if t1 < T_CLOSE:
        a = t1 / T_CLOSE
        gap = GAP_OPEN + (GAP_CLOSED - GAP_OPEN) * a
        return (GRIP_X + gap / 2, GRIP_Z), (GRIP_X - gap / 2, GRIP_Z)
    t2 = t1 - T_CLOSE
    if t2 < T_LIFT:
        z = GRIP_Z + LIFT_DZ * (t2 / T_LIFT)
        return (GRIP_X + GAP_CLOSED / 2, z), (GRIP_X - GAP_CLOSED / 2, z)
    t3 = t2 - T_LIFT
    x = GRIP_X + DRAG_DX * min(1.0, t3 / T_DRAG)
    z = GRIP_Z + LIFT_DZ
    return (x + GAP_CLOSED / 2, z), (x - GAP_CLOSED / 2, z)


def main() -> int:
    frames_dir = Path(ARGS.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0", use_fabric=False))
    stage = omni.usd.get_context().get_stage()

    g = UsdGeom.Cube.Define(stage, "/World/Ground")
    g.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(g)
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
    xf.AddScaleOp().Set(Gf.Vec3f(10, 10, 0.1))
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    b = UsdGeom.Cube.Define(stage, "/World/Mattress")
    b.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(b)
    xf.AddTranslateOp().Set(Gf.Vec3d(0, 0, BOX_H[2]))
    xf.AddScaleOp().Set(Gf.Vec3f(2 * BOX_H[0], 2 * BOX_H[1], 2 * BOX_H[2]))
    b.CreateDisplayColorAttr().Set([Gf.Vec3f(0.35, 0.45, 0.65)])
    UsdPhysics.CollisionAPI.Apply(b.GetPrim())
    UsdLux.DistantLight.Define(stage, "/World/Light").CreateIntensityAttr(3000.0)

    scene_path = clothmod.find_physics_scene_path(stage)
    clothmod.enable_gpu_dynamics(stage, scene_path)

    # ── the ISSUE-#2 bedsheet, 1.0 x 0.8 m, 30 cm overhang past the +x edge ──
    sheet = clothmod.build_bedsheet(
        stage, scene_path, "/World/Sheet",
        size=(1.0, 0.8), resolution=(40, 32),
        origin=(BOX_XFACE + 0.30 - 0.50, 0.0, BOX_TOP + 0.02),
        particle_mass=0.0004)

    # ── the DexGarmentLab material recipe on TOP of the #2 cloth material ────
    pmat = stage.GetPrimAtPath("/World/Sheet_material")
    pbd = PhysxSchema.PhysxPBDMaterialAPI(pmat)
    pbd.CreateFrictionAttr(float(ARGS.friction))               # particle–rigid friction
    pbd.CreateParticleFrictionScaleAttr(1.0)                   # cloth internal friction
    if ARGS.adhesion > 0.0:
        pbd.CreateAdhesionAttr(float(ARGS.adhesion))           # particle–rigid adhesion
        pbd.CreateAdhesionOffsetScaleAttr(1.2)
        pbd.CreateParticleAdhesionScaleAttr(1.0)
    print(f"[spike] PBD material: friction={ARGS.friction} adhesion={ARGS.adhesion}", flush=True)

    # ── kinematic high-friction plates ───────────────────────────────────────
    from omni.physx.scripts import physicsUtils
    mat = UsdShade.Material.Define(stage, "/World/FingerMat")
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(1.5)
    mapi.CreateDynamicFrictionAttr(1.5)
    mapi.CreateRestitutionAttr(0.0)
    plates = []
    for k, color in ((0, (0.8, 0.2, 0.2)), (1, (0.8, 0.53, 0.2))):
        c = UsdGeom.Cube.Define(stage, f"/World/Plate{k}")
        c.CreateSizeAttr(1.0)
        xf = UsdGeom.Xformable(c)
        (x0, z0), (x1, z1) = plate_pose(0.0)
        x, z = (x0, z0) if k == 0 else (x1, z1)
        tr = xf.AddTranslateOp()
        tr.Set(Gf.Vec3d(x, 0.0, z))
        xf.AddScaleOp().Set(Gf.Vec3f(*[2 * h for h in PLATE_H]))
        c.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
        UsdPhysics.RigidBodyAPI.Apply(c.GetPrim()).CreateKinematicEnabledAttr(True)
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        physicsUtils.add_physics_material_to_prim(stage, c.GetPrim(), Sdf.Path("/World/FingerMat"))
        plates.append(tr)

    from isaaclab.sensors.camera import Camera, CameraCfg
    cam = Camera(cfg=CameraCfg(
        prim_path="/World/CameraSensor", update_period=0, height=480, width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=22.0, focus_distance=400.0,
                                         horizontal_aperture=20.955,
                                         clipping_range=(0.05, 1.0e5))))

    print("[spike] sim.reset()", flush=True)
    sim.reset()
    cam.set_world_poses_from_view(
        torch.tensor([[1.9, -2.0, 1.5]], device=sim.device),
        torch.tensor([[0.0, 0.0, 0.45]], device=sim.device))
    sim.step(render=False)

    view = clothmod.make_cloth_view("/World/Sheet")

    t_total = T_SETTLE + T_CLOSE + T_LIFT + T_DRAG + T_HOLD
    n_steps = int(t_total / SIM_DT)
    gripped = None
    off0 = None
    frame = 0
    from PIL import Image
    for i in range(n_steps):
        t = i * SIM_DT
        (x0, z0), (x1, z1) = plate_pose(t)
        plates[0].Set(Gf.Vec3d(x0, 0.0, z0))
        plates[1].Set(Gf.Vec3d(x1, 0.0, z1))
        sim.step(render=False)

        if gripped is None and t >= T_SETTLE + T_CLOSE + 0.05:
            pn = clothmod.view_positions(view)
            over = pn[pn[:, 0] > BOX_XFACE - 0.005]
            if len(over):
                print(f"[spike] overhang span x[{over[:,0].min():.3f},{over[:,0].max():.3f}] "
                      f"z[{over[:,2].min():.3f},{over[:,2].max():.3f}]", flush=True)
            cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
            m = ((np.abs(pn[:, 0] - cx) < 0.03) & (np.abs(pn[:, 2] - cz) < PLATE_H[2])
                 & (np.abs(pn[:, 1]) < PLATE_H[1]))
            gripped = np.where(m)[0]
            if len(gripped):
                off0 = pn[gripped].mean(axis=0) - np.array([cx, 0.0, cz])
            print(f"[spike] t={t:.2f}s pinch closed on {len(gripped)} particles", flush=True)

        if i % 24 == 0:
            pn = clothmod.sync_mesh_from_view(view, sheet.mesh)
            sim.render()
            cam.update(dt=SIM_DT)
            outp = cam.data.output["rgb"]
            if outp is not None and outp.shape[0] > 0:
                rgb = outp[0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                Image.fromarray(rgb).save(frames_dir / f"frame_{frame:04d}.png")
                frame += 1
            if gripped is not None and len(gripped) and i % 120 == 0:
                cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
                off = pn[gripped].mean(axis=0) - np.array([cx, 0.0, cz])
                slip = float(np.linalg.norm(off - off0))
                print(f"[spike] t={t:.2f}s slip={slip*100:.1f} cm", flush=True)
            if not np.isfinite(pn).all():
                print(f"[spike] t={t:.2f}s NaN cloth — ABORT", flush=True)
                break

    pn = clothmod.view_positions(view)
    if gripped is not None and len(gripped) and np.isfinite(pn).all():
        (x0, z0), (x1, z1) = plate_pose(t_total)
        cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
        off = pn[gripped].mean(axis=0) - np.array([cx, 0.0, cz])
        slip = float(np.linalg.norm(off - off0))
        print(f"[spike] FINAL slip = {slip*100:.1f} cm over {abs(DRAG_DX)*100:.0f} cm stroke "
              f"({'HOLDS' if slip < 0.05 else 'SLIPS'})", flush=True)
    else:
        print("[spike] FINAL: no grip latched or NaN — FAILED", flush=True)
    print(f"[spike] DONE — {frame} frames in {frames_dir}", flush=True)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
