#!/usr/bin/env python3
"""
Load Cagatay's room 3D scan into Isaac Sim with Newton/PhysX physics.
Objects dropped from ceiling bounce off room surfaces.
"""
import os
import time

import numpy as np
from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True, "width": 1920, "height": 1080})

import omni.usd  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

print("=" * 60)
print("🏠 Loading Cagatay's Room into Isaac Sim + GPU Physics")
print("=" * 60)

ROOM_USD = "/home/ubuntu/room_sim/extracted/3_7_2026.usda"
OUTPUT_DIR = "/home/ubuntu/room_sim/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Create world ──
print("\n🌍 Creating simulation world...")
world = World(stage_units_in_meters=1.0, physics_dt=1.0/120.0, rendering_dt=1.0/60.0)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

# ── Ground plane ──
world.scene.add_default_ground_plane()

# ── Lighting ──
light = stage.DefinePrim("/World/DomeLight", "DomeLight")
light.GetAttribute("inputs:intensity").Set(2000.0)

# ── Load room by sublayering (avoids defaultPrim reference issue) ──
print(f"\n🏠 Loading room: {ROOM_USD}")

# Open the room stage to read geometry, then copy into our stage
room_stage = Usd.Stage.Open(ROOM_USD)

# Create a container Xform for the room
room_path = "/World/Room"
room_xf = UsdGeom.Xform.Define(stage, room_path)

# Rotate Y-up → Z-up and translate
room_xf.AddRotateXOp().Set(-90.0)
room_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.0))

# Copy each mesh from the room scan into our stage
mesh_count = 0
total_verts = 0
for prim in room_stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        src_mesh = UsdGeom.Mesh(prim)
        pts = src_mesh.GetPointsAttr().Get()
        fvc = src_mesh.GetFaceVertexCountsAttr().Get()
        fvi = src_mesh.GetFaceVertexIndicesAttr().Get()
        norms = src_mesh.GetNormalsAttr().Get()
        pvapi = UsdGeom.PrimvarsAPI(prim)
        uvs = pvapi.GetPrimvar("st")

        if not pts or not fvc or not fvi:
            continue

        # Create mesh in our stage
        dst_path = f"{room_path}/mesh_{mesh_count}"
        dst_mesh = UsdGeom.Mesh.Define(stage, dst_path)
        dst_mesh.CreatePointsAttr(pts)
        dst_mesh.CreateFaceVertexCountsAttr(fvc)
        dst_mesh.CreateFaceVertexIndicesAttr(fvi)
        dst_mesh.CreateSubdivisionSchemeAttr("none")

        if norms:
            dst_mesh.CreateNormalsAttr(norms)
            interp = src_mesh.GetNormalsInterpolation()
            dst_mesh.SetNormalsInterpolation(interp)

        if uvs:
            st_data = uvs.Get()
            if st_data:
                dst_pvapi = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(dst_path))
                dst_pv = dst_pvapi.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray)
                dst_pv.Set(st_data)
                dst_pv.SetInterpolation(uvs.GetInterpolation())

        # Copy transform from source
        src_xf = UsdGeom.Xformable(prim)
        ops = src_xf.GetOrderedXformOps()
        dst_xfable = UsdGeom.Xformable(stage.GetPrimAtPath(dst_path))
        for op in ops:
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                dst_xfable.AddTranslateOp().Set(op.Get())
            elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                dst_xfable.AddScaleOp().Set(op.Get())
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                dst_xfable.AddOrientOp().Set(op.Get())

        # Set display color
        colors = src_mesh.GetDisplayColorAttr().Get()
        if colors:
            dst_mesh.CreateDisplayColorAttr(colors)
        else:
            dst_mesh.CreateDisplayColorAttr([Gf.Vec3f(0.7, 0.7, 0.7)])

        # ── Add collision physics ──
        dst_prim = stage.GetPrimAtPath(dst_path)
        UsdPhysics.CollisionAPI.Apply(dst_prim)
        mesh_coll = UsdPhysics.MeshCollisionAPI.Apply(dst_prim)
        mesh_coll.CreateApproximationAttr("meshSimplification")
        physx_coll = PhysxSchema.PhysxCollisionAPI.Apply(dst_prim)
        physx_coll.CreateContactOffsetAttr(0.02)
        physx_coll.CreateRestOffsetAttr(0.0)

        mesh_count += 1
        total_verts += len(pts)

print(f"  ✅ Copied {mesh_count} meshes ({total_verts:,} vertices) with collision")

# ── Physics scene ──
print("\n⚙️ Configuring GPU physics...")
phys_path = "/World/PhysicsScene"
phys = UsdPhysics.Scene.Define(stage, phys_path)
phys.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
phys.CreateGravityMagnitudeAttr(9.81)

physx = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(phys_path))
physx.CreateEnableCCDAttr(True)
physx.CreateEnableStabilizationAttr(True)
physx.CreateEnableGPUDynamicsAttr(True)
physx.CreateGpuMaxNumPartitionsAttr(8)
physx.CreateBroadphaseTypeAttr("GPU")
print("  ✅ GPU physics enabled")

# ── Spawn objects to drop ──
print("\n📦 Spawning physics objects...")
obj_count = 0

def make_cube(name, pos, size=0.1, color=(0.8, 0.2, 0.2), mass=1.0):
    path = f"/World/Objects/{name}"
    p = stage.DefinePrim(path, "Cube")
    UsdGeom.Cube(p).CreateSizeAttr(size * 2)
    UsdGeom.Xformable(p).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdGeom.Cube(p).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(mass)
    UsdPhysics.CollisionAPI.Apply(p)
    # Add bounce
    mat_path = f"/World/Objects/{name}_mat"
    mat_prim = stage.DefinePrim(mat_path, "Material")
    PhysxSchema.PhysxMaterialAPI.Apply(mat_prim)
    phys_mat = UsdPhysics.MaterialAPI.Apply(mat_prim)
    phys_mat.CreateStaticFrictionAttr(0.5)
    phys_mat.CreateDynamicFrictionAttr(0.3)
    phys_mat.CreateRestitutionAttr(0.4)

def make_sphere(name, pos, radius=0.05, color=(0.2, 0.5, 0.9), mass=0.5):
    path = f"/World/Objects/{name}"
    p = stage.DefinePrim(path, "Sphere")
    UsdGeom.Sphere(p).CreateRadiusAttr(radius)
    UsdGeom.Xformable(p).AddTranslateOp().Set(Gf.Vec3d(*pos))
    UsdGeom.Sphere(p).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdPhysics.RigidBodyAPI.Apply(p)
    UsdPhysics.MassAPI.Apply(p).CreateMassAttr(mass)
    UsdPhysics.CollisionAPI.Apply(p)

# Cubes
for i in range(20):
    x = np.random.uniform(-2.0, 2.0)
    y = np.random.uniform(-3.0, 3.0)
    z = np.random.uniform(4.0, 7.0)
    s = np.random.uniform(0.05, 0.12)
    c = tuple(np.random.uniform(0.2, 1.0, 3))
    make_cube(f"cube_{i}", (x, y, z), size=s, color=c, mass=s*10)
    obj_count += 1

# Spheres (bouncy balls)
for i in range(20):
    x = np.random.uniform(-2.0, 2.0)
    y = np.random.uniform(-3.0, 3.0)
    z = np.random.uniform(4.0, 8.0)
    r = np.random.uniform(0.03, 0.08)
    c = tuple(np.random.uniform(0.2, 1.0, 3))
    make_sphere(f"ball_{i}", (x, y, z), radius=r, color=c, mass=r*5)
    obj_count += 1

print(f"  ✅ {obj_count} objects spawned above room")

# ── Initialize ──
print("\n🚀 Initializing simulation...")
world.reset()

# ── Run physics simulation ──
SIM_SECONDS = 5.0
PHYSICS_HZ = 120
total_steps = int(SIM_SECONDS * PHYSICS_HZ)

print(f"\n🎬 Running {SIM_SECONDS}s physics ({total_steps} steps)...")
t0 = time.time()

for step in range(total_steps):
    world.step(render=False)
    if step % (PHYSICS_HZ * 1) == 0:
        elapsed = time.time() - t0
        sim_t = step / PHYSICS_HZ
        print(f"  Step {step}/{total_steps} | Sim: {sim_t:.1f}s | Wall: {elapsed:.1f}s")

elapsed = time.time() - t0
print("\n✅ Simulation complete!")
print(f"  Steps: {total_steps}")
print(f"  Wall time: {elapsed:.1f}s")
print(f"  Realtime factor: {SIM_SECONDS/elapsed:.2f}x")

# ── Save output USD ──
output_usd = os.path.join(OUTPUT_DIR, "cagatay_room_physics.usd")
stage.GetRootLayer().Export(output_usd)
size_mb = os.path.getsize(output_usd) / (1024*1024)
print(f"\n💾 Scene saved: {output_usd} ({size_mb:.1f} MB)")

# ── Also save as USDA (text) for inspection ──
output_usda = os.path.join(OUTPUT_DIR, "cagatay_room_physics.usda")
stage.GetRootLayer().Export(output_usda)

# ── Summary ──
print("\n" + "=" * 60)
print("🏠 Room Physics Simulation Summary")
print("=" * 60)
print("  Room: ~7.6m × 3.5m × 13.3m (170K vertices)")
print(f"  Meshes with collision: {mesh_count}")
print(f"  Physics objects: {obj_count}")
print("  GPU: NVIDIA L40S (PhysX GPU)")
print(f"  Output: {output_usd}")
print("=" * 60)

sim_app.close()
