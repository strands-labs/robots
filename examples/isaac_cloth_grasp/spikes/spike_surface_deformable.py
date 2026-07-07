"""Spike 1 (issue #5): NEW PhysX deformable-beta SURFACE deformable (FEM cloth) on the GB10.

Isaac Sim 5.1 ships a new deformable model behind the carb setting
``/persistent/physics/enableDeformableBeta`` (PhysX 107.3): *surface deformables* are
FEM cloth with a real ``surface_thickness`` (not a zero-thickness PBD membrane) and
*volume deformables* are FEM/hex soft bodies. The bundled FrankaDeformableDemo grips
volume deformables by **pure finger friction** — no attachment — which is exactly the
enclosure grasp issue #5 needs.

This spike answers the cheapest question first: does a surface deformable COOK,
SIMULATE and RENDER headless on the DGX Spark (aarch64, GB10)?

Scene: a static box (mattress proxy) + a surface-deformable sheet spawned flat above
it with an overhang. If the model works, the sheet drapes over the box edge. Frames
are rendered to PNG for eye-verification (never trust telemetry alone — but we also
print max vertex displacement as a cross-check that the FEM solver is alive).

Run (Isaac Lab python on the Spark)::

    cd ~/workspaces/git/IsaacLab
    export LD_PRELOAD="$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1"
    PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
      ~/workspaces/git/robots-issue5/examples/isaac_cloth_grasp/spikes/spike_surface_deformable.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--frames-dir", default=str(Path(__file__).parent / "out" / "surface_deformable"))
p.add_argument("--seconds", type=float, default=4.0)
p.add_argument("--no-fabric", action="store_true",
               help="Run with use_fabric=False (check whether deformable skin sync needs USD path)")
ARGS = p.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

# The deformable beta is gated by a persistent carb setting; pass it as a kit arg so
# it is set before omni.physx parses anything, and set it again post-launch for safety.
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
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics  # noqa: E402

carb.settings.get_settings().set_bool("/persistent/physics/enableDeformableBeta", True)
print("[spike] deformableBeta =",
      carb.settings.get_settings().get("/persistent/physics/enableDeformableBeta"), flush=True)

SIM_DT = 1.0 / 120.0


def make_quad_grid(stage, prim_path: str, size, resolution, origin):
    """Author a flat triangle-grid mesh (FEM cloth wants triangles; the PBD quad-grid
    spring gotcha does not apply to FEM)."""
    from omni.physx.scripts import deformableUtils

    nx, ny = resolution
    tri_points, tri_indices = deformableUtils.create_triangle_mesh_square(dimx=nx, dimy=ny, scale=1.0)
    Wx, Wy = size
    ox, oy, oz = origin
    pts = [Gf.Vec3f(float(pt[0] * Wx), float(pt[1] * Wy), 0.0) for pt in tri_points]
    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexIndicesAttr(tri_indices)
    mesh.CreateFaceVertexCountsAttr([3] * (len(tri_indices) // 3))
    mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.86, 0.86, 0.92)])
    mesh.CreateDoubleSidedAttr(True)
    xf = UsdGeom.Xformable(mesh)
    xf.AddTranslateOp().Set(Gf.Vec3d(ox, oy, oz))
    return mesh


def main() -> int:
    frames_dir = Path(ARGS.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=SIM_DT, device="cuda:0", use_fabric=not ARGS.no_fabric))
    stage = omni.usd.get_context().get_stage()

    # Ground
    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(ground)
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.05))
    xf.AddScaleOp().Set(Gf.Vec3f(10.0, 10.0, 0.1))
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())

    # Mattress proxy: static box, top at z=0.5
    box = UsdGeom.Cube.Define(stage, "/World/Mattress")
    box.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(box)
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.25))
    xf.AddScaleOp().Set(Gf.Vec3f(0.8, 1.2, 0.5))
    box.CreateDisplayColorAttr().Set([Gf.Vec3f(0.35, 0.45, 0.65)])
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())

    # Light
    from pxr import UsdLux
    light = UsdLux.DistantLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(3000.0)

    # ── Surface deformable sheet ──────────────────────────────────────────────
    from omni.physx.scripts import deformableUtils, physicsUtils

    scene_path = None
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            scene_path = prim.GetPath()
            break
    assert scene_path is not None, "no physics scene found"
    scene_prim = stage.GetPrimAtPath(scene_path)
    psa = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
    psa.CreateEnableGPUDynamicsAttr(True)
    psa.CreateBroadphaseTypeAttr("GPU")
    # Surface-deformable contact buffer (the bundled demo sizes it explicitly)
    try:
        psa.GetGpuMaxDeformableSurfaceContactsAttr().Set(4 * 1048576)
    except Exception:
        psa.CreateGpuMaxDeformableSurfaceContactsAttr(4 * 1048576)

    root = UsdGeom.Xform.Define(stage, "/World/Sheet")
    skin = make_quad_grid(stage, "/World/Sheet/mesh", size=(1.4, 1.4), resolution=(40, 40),
                          origin=(0.25, 0.0, 0.6))

    ok = deformableUtils.create_auto_surface_deformable_hierarchy(
        stage,
        root_prim_path=root.GetPath(),
        simulation_mesh_path="/World/Sheet/simMesh",
        cooking_src_mesh_path=skin.GetPath(),
        cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True,
    )
    print("[spike] create_auto_surface_deformable_hierarchy ->", ok, flush=True)

    prim = root.GetPrim()
    prim.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    prim.GetAttribute("physxDeformableBody:selfCollision").Set(True)
    prim.GetAttribute("physxDeformableBody:enableSpeculativeCCD").Set(True)
    prim.GetAttribute("physxDeformableBody:solverPositionIterationCount").Set(16)
    prim.GetAttribute("physxDeformableBody:collisionPairUpdateFrequency").Set(4)
    prim.GetAttribute("physxDeformableBody:collisionIterationMultiplier").Set(4)

    rest_offset, contact_offset = 0.004, 0.012
    sim_mesh_prim = stage.GetPrimAtPath("/World/Sheet/simMesh")
    sim_mesh_prim.ApplyAPI(PhysxSchema.PhysxCollisionAPI)
    PhysxSchema.PhysxCollisionAPI(sim_mesh_prim).CreateRestOffsetAttr(rest_offset)
    PhysxSchema.PhysxCollisionAPI(sim_mesh_prim).CreateContactOffsetAttr(contact_offset)
    prim.GetAttribute("physxDeformableBody:maxLinearVelocity").Set(contact_offset / SIM_DT)

    mat_path = "/World/SheetMaterial"
    deformableUtils.add_surface_deformable_material(
        stage, mat_path,
        dynamic_friction=0.6,
        surface_thickness=0.004,          # 4 mm — a real (thick) bedsheet/duvet skin
        surface_stretch_stiffness=1000.0,
        surface_shear_stiffness=0.5,
        surface_bend_stiffness=0.05,
    )
    physicsUtils.add_physics_material_to_prim(stage, prim, mat_path)
    mat_prim = stage.GetPrimAtPath(mat_path)
    mat_prim.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    mat_prim.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(0.02)
    mat_prim.GetAttribute("physxDeformableMaterial:bendDamping").Set(0.01)

    # Camera
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
        torch.tensor([[2.2, -2.2, 1.6]], device=sim.device),
        torch.tensor([[0.0, 0.0, 0.45]], device=sim.device))

    # Live FEM state via the tensor API (the new-deformable analogue of the
    # particle-cloth tensor view — USD/Fabric do NOT auto-sync GPU deformation).
    sim.step(render=False)
    import omni.physics.tensors as tensors
    sv = tensors.create_simulation_view("torch")
    sv.set_subspace_roots("/")
    dview = None
    try:
        dview = sv.create_surface_deformable_body_view("/World/Sheet")
        print(f"[spike] surface deformable view: count={dview.count}", flush=True)
    except Exception as e:
        print(f"[spike] surface deformable view FAILED: {e}", flush=True)

    sim_mesh = UsdGeom.Mesh.Get(stage, "/World/Sheet/simMesh")
    skin_mesh = UsdGeom.Mesh.Get(stage, "/World/Sheet/mesh")
    # Render the SIM mesh (same topology as the nodal positions we read) — unhide it
    # and hide the stale skin mesh.
    if sim_mesh:
        UsdGeom.Imageable(sim_mesh).MakeVisible()
        sim_mesh.GetPrim().GetAttribute("purpose").Set("default")
        sim_mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.86, 0.86, 0.92)])
    if skin_mesh:
        UsdGeom.Imageable(skin_mesh).MakeInvisible()

    def nodal_positions():
        if dview is None:
            return None
        pos = dview.get_simulation_nodal_positions()
        if hasattr(pos, "detach"):
            pos = pos.detach().cpu().numpy()
        return np.asarray(pos, dtype=float).reshape(dview.count, -1, 3)[0]

    p0 = nodal_positions()
    if p0 is not None:
        print(f"[spike] sim mesh nodes: {p0.shape[0]}", flush=True)

    from pxr import Vt

    def blit():
        p = nodal_positions()
        if p is None or sim_mesh is None:
            return None
        # world -> local: simMesh inherits the /World/Sheet root xform (identity here)
        sim_mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(p.astype(np.float32)))
        return p

    n_steps = int(ARGS.seconds / SIM_DT)
    frame = 0
    from PIL import Image
    for i in range(n_steps):
        sim.step(render=False)
        if i % 12 == 0:  # ~10 fps
            blit()
            sim.render()
            cam.update(dt=SIM_DT)
            out = cam.data.output["rgb"]
            if out is not None and out.shape[0] > 0:
                rgb = out[0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                Image.fromarray(rgb).save(frames_dir / f"frame_{frame:04d}.png")
                frame += 1
        if i % 120 == 0 and p0 is not None:
            p = nodal_positions()
            if p is not None:
                d = np.linalg.norm(p - p0, axis=1)
                print(f"[spike] t={i*SIM_DT:.2f}s FEM nodal max displacement {d.max():.4f} m",
                      flush=True)

    print(f"[spike] DONE — {frame} frames in {frames_dir}", flush=True)
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
