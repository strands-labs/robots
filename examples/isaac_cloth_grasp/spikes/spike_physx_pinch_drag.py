"""Spike 3 (issue #5): PhysX NEW surface-deformable (FEM cloth) frictional PINCH + DRAG.

Protocol matches the Newton spike (spike_newton_pinch_drag.py) so the two cloth
representations are compared apples-to-apples:

1. the FEM sheet drapes over a box with an overhang down the +x face;
2. two high-friction kinematic plates pinch the hanging overhang;
3. the plates drag up and across the box top through a full ~45 cm stroke.

PASS = the cloth stays pinched through the stroke (slip ≪ stroke), eye-verified
from rendered frames; slip metric printed from the FEM nodal tensor view.

The NVIDIA-documented grasp failure (forum 332704: grippers penetrate + slip) is
about PBD *particle* cloth. The 5.1 deformable-beta FEM surface cloth has real
``surface_thickness`` and FEM contact, so friction may simply work — that is what
this spike measures, in-engine, where a PASS would drop straight into issue #2.

Run::

    cd ~/workspaces/git/IsaacLab
    export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
      ~/workspaces/git/robots-issue5/examples/isaac_cloth_grasp/spikes/spike_physx_pinch_drag.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--frames-dir", default=str(Path(__file__).parent / "out" / "physx_pinch_drag"))
ARGS = p.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

app_launcher = AppLauncher({
    "headless": True,
    "enable_cameras": True,
    "kit_args": "--/persistent/physics/enableDeformableBeta=true",
})
simulation_app = app_launcher.app

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt  # noqa: E402

carb.settings.get_settings().set_bool("/persistent/physics/enableDeformableBeta", True)

SIM_DT = 1.0 / 120.0

# ── geometry (m) — mirrors the Newton spike (which is in cm) ─────────────────
BOX_H = (0.40, 0.60, 0.25)
BOX_TOP = 2 * BOX_H[2]
BOX_XFACE = BOX_H[0]
SHEET_W, SHEET_D = 1.00, 0.80      # 40x32-cell grid equivalent
OVERHANG = 0.30
PLATE_H = (0.01, 0.06, 0.04)
GRIP_Z = BOX_TOP - 0.10   # mid-flap (z 0.20-0.50)
GAP_OPEN, GAP_CLOSED = 0.30, 0.028  # surface gap 8mm ≈ the FEM slab thickness   # centre sep > 2*plate_hx=0.02 (no interpenetration)
T_SETTLE, T_CLOSE, T_LIFT, T_DRAG, T_HOLD = 2.0, 0.6, 0.8, 2.5, 0.6
DRAG_DX, LIFT_DZ = -0.45, 0.14
GRIP_X = BOX_XFACE + 0.012  # flap hangs flush at x=0.405 (run-3 telemetry)


AIM = {"x": GRIP_X, "z": GRIP_Z}


def plate_pose(t: float):
    gx, gz = AIM["x"], AIM["z"]
    if t < T_SETTLE:
        return (gx + GAP_OPEN / 2, gz), (gx - GAP_OPEN / 2, gz)
    t1 = t - T_SETTLE
    if t1 < T_CLOSE:
        a = t1 / T_CLOSE
        gap = GAP_OPEN + (GAP_CLOSED - GAP_OPEN) * a
        return (gx + gap / 2, gz), (gx - gap / 2, gz)
    t2 = t1 - T_CLOSE
    if t2 < T_LIFT:
        z = gz + LIFT_DZ * (t2 / T_LIFT)
        return (gx + GAP_CLOSED / 2, z), (gx - GAP_CLOSED / 2, z)
    t3 = t2 - T_LIFT
    x = gx + DRAG_DX * min(1.0, t3 / T_DRAG)
    z = gz + LIFT_DZ
    return (x + GAP_CLOSED / 2, z), (x - GAP_CLOSED / 2, z)


def main() -> int:
    frames_dir = Path(ARGS.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0", use_fabric=False))
    stage = omni.usd.get_context().get_stage()

    # ground + box + light
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
    xf.AddScaleOp().Set(Gf.Vec3f(*[2 * h for h in BOX_H][:2], 2 * BOX_H[2]))
    b.CreateDisplayColorAttr().Set([Gf.Vec3f(0.35, 0.45, 0.65)])
    UsdPhysics.CollisionAPI.Apply(b.GetPrim())
    UsdLux.DistantLight.Define(stage, "/World/Light").CreateIntensityAttr(3000.0)

    # physics scene flags
    scene_path = None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            scene_path = prim.GetPath()
            break
    psa = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(scene_path))
    psa.CreateEnableGPUDynamicsAttr(True)
    psa.CreateBroadphaseTypeAttr("GPU")
    try:
        psa.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1048576)
    except Exception:
        psa.CreateGpuMaxDeformableSurfaceContactsAttr(4 * 1048576)

    # ── FEM sheet (same recipe as spike v2, sized like the Newton sheet) ─────
    from omni.physx.scripts import deformableUtils, physicsUtils

    nx, nyc = 40, 32
    tri_points, tri_indices = deformableUtils.create_triangle_mesh_square(dimx=nx, dimy=nyc, scale=1.0)
    ox = BOX_XFACE + OVERHANG - SHEET_W / 2.0   # centre so OVERHANG hangs past +x edge
    pts = [Gf.Vec3f(float(q[0] * SHEET_W), float(q[1] * SHEET_D), 0.0) for q in tri_points]
    root = UsdGeom.Xform.Define(stage, "/World/Sheet")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Sheet/mesh")
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexIndicesAttr(tri_indices)
    mesh.CreateFaceVertexCountsAttr([3] * (len(tri_indices) // 3))
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.86, 0.86, 0.92)])
    mesh.CreateDoubleSidedAttr(True)
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(Gf.Vec3d(ox, 0.0, BOX_TOP + 0.02))

    deformableUtils.create_auto_surface_deformable_hierarchy(
        stage, root_prim_path=root.GetPath(),
        simulation_mesh_path="/World/Sheet/simMesh",
        cooking_src_mesh_path=mesh.GetPath(),
        cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True)
    prim = root.GetPrim()
    prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    prim.GetAttribute("physxDeformableBody:selfCollision").Set(True)
    prim.GetAttribute("physxDeformableBody:enableSpeculativeCCD").Set(True)
    prim.GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(16)
    prim.GetAttribute("physxDeformableBody:collisionPairUpdateFrequency").Set(4)
    prim.GetAttribute("physxDeformableBody:collisionIterationMultiplier").Set(4)
    rest_offset, contact_offset = 0.004, 0.012
    smp = stage.GetPrimAtPath("/World/Sheet/simMesh")
    smp.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
    PhysxSchema.PhysxCollisionAPI(smp).CreateRestOffsetAttr(rest_offset)
    PhysxSchema.PhysxCollisionAPI(smp).CreateContactOffsetAttr(contact_offset)
    prim.GetAttribute("physxDeformableBody:maxLinearVelocity").Set(contact_offset / SIM_DT)

    deformableUtils.add_surface_deformable_material(
        stage, "/World/SheetMaterial",
        dynamic_friction=0.6, surface_thickness=0.004,
        surface_stretch_stiffness=1000.0, surface_shear_stiffness=0.5,
        surface_bend_stiffness=0.05)
    physicsUtils.add_physics_material_to_prim(stage, prim, "/World/SheetMaterial")
    mp = stage.GetPrimAtPath("/World/SheetMaterial")
    mp.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    mp.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.02)
    mp.GetAttribute("physxDeformableMaterial:bendDamping").Set(0.01)

    # ── kinematic high-friction finger plates ────────────────────────────────
    mat = UsdShade.Material.Define(stage, "/World/FingerMat")
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(1.5)
    mapi.CreateDynamicFrictionAttr(1.5)
    mapi.CreateRestitutionAttr(0.0)
    plates = []
    for k, color in ((0, (0.8, 0.2, 0.2)), (1, (0.8, 0.53, 0.2))):
        pp = f"/World/Plate{k}"
        c = UsdGeom.Cube.Define(stage, pp)
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
        physicsUtils.add_physics_material_to_prim(stage, c.GetPrim(), "/World/FingerMat")
        plates.append(tr)

    # camera
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

    import omni.physics.tensors as tensors
    sv = tensors.create_simulation_view("torch")
    sv.set_subspace_roots("/")
    dview = sv.create_surface_deformable_body_view("/World/Sheet")
    print(f"[spike] deformable view count={dview.count}", flush=True)

    sim_mesh = UsdGeom.Mesh.Get(stage, "/World/Sheet/simMesh")
    UsdGeom.Imageable(sim_mesh).MakeVisible()
    sim_mesh.GetPrim().GetAttribute("purpose").Set("default")
    sim_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.86, 0.86, 0.92)])
    sim_mesh.CreateDoubleSidedAttr(True)
    UsdGeom.Imageable(mesh).MakeInvisible()

    def nodal():
        pos = dview.get_simulation_nodal_positions()
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        return np.asarray(pos, dtype=float).reshape(dview.count, -1, 3)[0]

    def blit(pn):
        sim_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(pn.astype(np.float32)))

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

        if t >= T_SETTLE - 0.1 and t < T_SETTLE - 0.1 + SIM_DT:
            pn = nodal()
            band = pn[(pn[:, 0] > BOX_XFACE - 0.005) & (pn[:, 2] < BOX_TOP - 0.03)
                      & (np.abs(pn[:, 1]) < PLATE_H[1])]
            if len(band):
                AIM["x"] = float(band[:, 0].mean())
                AIM["z"] = float(np.clip(band[:, 2].mean(), 0.25, BOX_TOP - 0.06))
                print(f"[spike] ADAPTIVE AIM x={AIM['x']:.3f} z={AIM['z']:.3f} "
                      f"({len(band)} flap nodes)", flush=True)

        if gripped is None and t >= T_SETTLE + T_CLOSE + 0.05:
            pn = nodal()
            over = pn[pn[:, 0] > BOX_XFACE - 0.005]
            if len(over):
                print(f"[spike] overhang span x[{over[:,0].min():.3f},{over[:,0].max():.3f}] "
                      f"z[{over[:,2].min():.3f},{over[:,2].max():.3f}] "
                      f"y[{over[:,1].min():.3f},{over[:,1].max():.3f}]", flush=True)
            cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
            m = ((np.abs(pn[:, 0] - cx) < 0.03) & (np.abs(pn[:, 2] - cz) < PLATE_H[2])
                 & (np.abs(pn[:, 1]) < PLATE_H[1]))
            gripped = np.where(m)[0]
            if len(gripped):
                off0 = pn[gripped].mean(axis=0) - np.array([cx, 0.0, cz])
            print(f"[spike] t={t:.2f}s pinch closed on {len(gripped)} nodes", flush=True)

        if i % 12 == 0:
            pn = nodal()
            blit(pn)
            sim.render()
            cam.update(dt=SIM_DT)
            outp = cam.data.output["rgb"]
            if outp is not None and outp.shape[0] > 0:
                rgb = outp[0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                Image.fromarray(rgb).save(frames_dir / f"frame_{frame:04d}.png")
                frame += 1
            if gripped is not None and len(gripped) and i % 60 == 0:
                cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
                off = pn[gripped].mean(axis=0) - np.array([cx, 0.0, cz])
                slip = float(np.linalg.norm(off - off0))
                print(f"[spike] t={t:.2f}s slip={slip*100:.1f} cm", flush=True)
            if not np.isfinite(pn).all():
                print(f"[spike] t={t:.2f}s NaN cloth — ABORT", flush=True)
                break

    pn = nodal()
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
