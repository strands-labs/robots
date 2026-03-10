#!/usr/bin/env python3
"""
🔥 FLOOR IS LAVA v4 — Unitree G1 in Cagatay's Room
Fixed: contact callback signature + collision shape detection
"""
import json
import os
import time

import numpy as np
from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True, "width": 1920, "height": 1080})

import omni.kit.commands  # noqa: E402
import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

print("=" * 70)
print("🔥 FLOOR IS LAVA v4 — Unitree G1 in Cagatay's Room")
print("=" * 70)

ROOM_USD = "/home/ubuntu/room_sim/extracted/3_7_2026.usda"
G1_MJCF = "/home/ubuntu/strands-gtc-nvidia/strands_robots/assets/unitree_g1/scene.xml"
OUTPUT_DIR = "/home/ubuntu/room_sim/output"

# ═══ World ═══
print("\n🌍 Creating world...")
world = World(stage_units_in_meters=1.0, physics_dt=1.0/240.0, rendering_dt=1.0/60.0)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

phys_path = "/World/PhysicsScene"
phys = UsdPhysics.Scene.Define(stage, phys_path)
phys.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
phys.CreateGravityMagnitudeAttr(9.81)
physx = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(phys_path))
physx.CreateEnableCCDAttr(True)
physx.CreateEnableStabilizationAttr(True)
physx.CreateEnableGPUDynamicsAttr(True)
physx.CreateBroadphaseTypeAttr("GPU")
print("  ✅ GPU PhysX 240Hz")

light = stage.DefinePrim("/World/DomeLight", "DomeLight")
light.GetAttribute("inputs:intensity").Set(3000.0)

# ═══ Room (LAVA) ═══
print("\n🏠 Loading room as LAVA...")
room_stage = Usd.Stage.Open(ROOM_USD)
room_path = "/World/Room"
room_xf = UsdGeom.Xform.Define(stage, room_path)
room_xf.AddRotateXOp().Set(-90.0)

mesh_count = 0
total_verts = 0
for prim in room_stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        src_mesh = UsdGeom.Mesh(prim)
        pts = src_mesh.GetPointsAttr().Get()
        fvc = src_mesh.GetFaceVertexCountsAttr().Get()
        fvi = src_mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not fvc or not fvi:
            continue

        dst_path = f"{room_path}/mesh_{mesh_count}"
        dst_mesh = UsdGeom.Mesh.Define(stage, dst_path)
        dst_mesh.CreatePointsAttr(pts)
        dst_mesh.CreateFaceVertexCountsAttr(fvc)
        dst_mesh.CreateFaceVertexIndicesAttr(fvi)
        dst_mesh.CreateSubdivisionSchemeAttr("none")
        norms = src_mesh.GetNormalsAttr().Get()
        if norms:
            dst_mesh.CreateNormalsAttr(norms)
            dst_mesh.SetNormalsInterpolation(src_mesh.GetNormalsInterpolation())
        src_xf = UsdGeom.Xformable(prim)
        dst_xfable = UsdGeom.Xformable(stage.GetPrimAtPath(dst_path))
        for op in src_xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                dst_xfable.AddTranslateOp().Set(op.Get())
            elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                dst_xfable.AddScaleOp().Set(op.Get())
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                dst_xfable.AddOrientOp().Set(op.Get())
        dst_mesh.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.15, 0.05)])
        dp = stage.GetPrimAtPath(dst_path)
        UsdPhysics.CollisionAPI.Apply(dp)
        UsdPhysics.MeshCollisionAPI.Apply(dp).CreateApproximationAttr("meshSimplification")
        PhysxSchema.PhysxContactReportAPI.Apply(dp).CreateThresholdAttr(0)
        mesh_count += 1
        total_verts += len(pts)

print(f"  ✅ {mesh_count} meshes ({total_verts:,} verts) — ALL LAVA 🔥")

fp = stage.DefinePrim("/World/LavaFloor", "Cube")
UsdGeom.Cube(fp).CreateSizeAttr(200.0)
UsdGeom.Xformable(fp).AddTranslateOp().Set(Gf.Vec3d(0, 0, -100.05))
UsdGeom.Cube(fp).CreateDisplayColorAttr([Gf.Vec3f(1.0, 0.3, 0.0)])
UsdPhysics.CollisionAPI.Apply(fp)
PhysxSchema.PhysxContactReportAPI.Apply(fp).CreateThresholdAttr(0)

# ═══ Import G1 ═══
print("\n🤖 Importing G1...")
_, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
g1_path = "/World/G1"
_, g1_prim_path = omni.kit.commands.execute(
    "MJCFCreateAsset", mjcf_path=G1_MJCF, import_config=import_config, prim_path=g1_path
)

g1_prim = stage.GetPrimAtPath(g1_path)
g1_xf = UsdGeom.Xformable(g1_prim)
g1_xf.ClearXformOpOrder()
g1_xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 1.05))

# ─── Enable contact reporting on ALL G1 prims with CollisionAPI ───
coll_count = 0
for child in Usd.PrimRange(g1_prim):
    if child.HasAPI(UsdPhysics.CollisionAPI):
        if not child.HasAPI(PhysxSchema.PhysxContactReportAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(child).CreateThresholdAttr(0)
        coll_count += 1

# Also add collision+reporting to rigid body prims that lack it
for child in Usd.PrimRange(g1_prim):
    if child.HasAPI(UsdPhysics.RigidBodyAPI):
        if not child.HasAPI(UsdPhysics.CollisionAPI):
            # Add a sphere collision approximation
            UsdPhysics.CollisionAPI.Apply(child)
            PhysxSchema.PhysxContactReportAPI.Apply(child).CreateThresholdAttr(0)
            coll_count += 1

joint_count = sum(1 for c in Usd.PrimRange(g1_prim) if c.IsA(UsdPhysics.Joint))
body_count = sum(1 for c in Usd.PrimRange(g1_prim) if c.HasAPI(UsdPhysics.RigidBodyAPI))
print(f"  ✅ G1: {joint_count} joints, {body_count} bodies, {coll_count} contact-enabled")

# ═══ Safe Platforms ═══
platforms = [
    {"name": "start",   "pos": (0.0, 0.0, 0.025),  "size": (1.0, 1.0, 0.05)},
    {"name": "step_1",  "pos": (1.5, 0.5, 0.025),   "size": (0.6, 0.6, 0.05)},
    {"name": "step_2",  "pos": (3.0, -0.3, 0.025),  "size": (0.5, 0.5, 0.05)},
    {"name": "step_3",  "pos": (4.5, 0.8, 0.025),   "size": (0.7, 0.4, 0.05)},
    {"name": "goal",    "pos": (6.0, 0.0, 0.025),   "size": (1.0, 1.0, 0.05)},
]
for plat in platforms:
    path = f"/World/Platforms/{plat['name']}"
    p = stage.DefinePrim(path, "Cube")
    UsdGeom.Cube(p).CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(p)
    xf.AddTranslateOp().Set(Gf.Vec3d(*plat["pos"]))
    sx, sy, sz = plat["size"]
    xf.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))
    UsdGeom.Cube(p).CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.2)])
    UsdPhysics.CollisionAPI.Apply(p)
    PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0)
print(f"  🟢 {len(platforms)} safe platforms")

# ═══ Gait ═══
drives = []
drive_names = []
for child in Usd.PrimRange(stage.GetPrimAtPath(g1_path)):
    if child.IsA(UsdPhysics.Joint):
        drive = UsdPhysics.DriveAPI.Get(child, "angular")
        if drive and drive.GetTargetPositionAttr():
            drives.append(drive)
            drive_names.append(child.GetName())

print(f"  🦿 {len(drives)} driven joints")

GAIT_FREQ = 1.5
GAIT = {
    "hip_pitch": {"amp": 0.35, "off": -0.1},
    "hip_roll": {"amp": 0.08, "off": 0.0},
    "knee": {"amp": 0.5, "off": -0.3},
    "ankle_pitch": {"amp": 0.25, "off": 0.1},
    "ankle_roll": {"amp": 0.04, "off": 0.0},
    "shoulder_pitch": {"amp": 0.2, "off": 0.0},
    "shoulder_roll": {"amp": 0.05, "off": 0.2},
    "elbow": {"amp": 0.15, "off": -0.4},
    "waist": {"amp": 0.02, "off": 0.0},
}

def gait_target(name, t):
    n = name.lower()
    for key, cfg in GAIT.items():
        if key in n:
            phase = np.pi if "right" in n else 0.0
            return cfg["off"] + cfg["amp"] * np.sin(2*np.pi*GAIT_FREQ*t + phase)
    return 0.0

# ═══ Contact tracking via callback (CORRECT SIGNATURE: contact_headers, sdk_handle) ═══
contact_events_buffer = []
alive_flag = [True]
death_info_buf = [None]

def on_contact(contact_headers, sdk_handle):
    """PhysX contact report callback — note the 2-arg signature!"""
    if not alive_flag[0]:
        return
    for header in contact_headers:
        a0 = str(header.actor0)
        a1 = str(header.actor1)
        robot_hit = "/G1/" in a0 or "/G1/" in a1
        if not robot_hit:
            continue
        other = a1 if "/G1/" in a0 else a0
        robot_part = a0 if "/G1/" in a0 else a1

        if "/Platforms/" in other:
            contact_events_buffer.append(("SAFE", other.split("/")[-1], robot_part.split("/")[-1]))
        elif "/Room/" in other or "/LavaFloor" in other:
            alive_flag[0] = False
            part = other.split("/")[-1]
            limb = robot_part.split("/")[-1]
            reason = "FELL INTO LAVA 🔥" if "/LavaFloor" in other else f"TOUCHED {part} 💀🔥"
            death_info_buf[0] = (reason, limb, part)

sim_iface = get_physx_simulation_interface()
contact_sub = sim_iface.subscribe_contact_report_events(on_contact)
print("  📡 Contact callback registered")

# ═══ Simulation ═══
print("\n🚀 Initializing...")
world.reset()

SIM_SECONDS = 10.0
PHYSICS_HZ = 240
POLICY_HZ = 50
total_steps = int(SIM_SECONDS * PHYSICS_HZ)
policy_interval = PHYSICS_HZ // POLICY_HZ

print("\n🎬 FLOOR IS LAVA — GO!")
print(f"   {SIM_SECONDS}s | {PHYSICS_HZ}Hz | G1: {joint_count}J/{coll_count}C | Room: {mesh_count} lava")
print("─" * 70)

episode = {
    "robot": "Unitree G1 29-DoF EDU Plus (NVIDIA Orin)",
    "room": "Cagatay's Room (iPhone LiDAR scan, 170K verts)",
    "rules": "Floor is Lava! Room contact = death.",
    "physics_hz": PHYSICS_HZ, "policy_hz": POLICY_HZ,
    "platforms": len(platforms), "room_meshes": mesh_count,
    "joints": joint_count, "colliders": coll_count,
    "events": [], "timeline": [],
}

platforms_reached = set()
death_step = None
t0 = time.time()

for step in range(total_steps):
    sim_time = step / PHYSICS_HZ

    if step % policy_interval == 0 and alive_flag[0]:
        for drive, dname in zip(drives, drive_names):
            target = gait_target(dname, sim_time)
            try:
                drive.GetTargetPositionAttr().Set(float(np.degrees(target)))
            except Exception:
                pass

    world.step(render=False)

    # Process contacts
    if contact_events_buffer:
        for etype, name, limb in contact_events_buffer:
            if etype == "SAFE" and name not in platforms_reached:
                platforms_reached.add(name)
                episode["events"].append({"t": round(sim_time,3), "event": "PLATFORM", "platform": name})
                print(f"  🟢 t={sim_time:.2f}s — {name} (via {limb})")
        contact_events_buffer.clear()

    if not alive_flag[0] and death_step is None:
        death_step = step
        reason, limb, obj = death_info_buf[0]
        episode["events"].append({
            "t": round(sim_time,3), "event": "DEATH", "reason": reason,
            "robot_part": limb, "lava_object": obj
        })
        print(f"\n  💀 DEATH at t={sim_time:.2f}s — {reason}")
        print(f"     G1 [{limb}] → [{obj}]")
        # Don't break — let physics settle for visual

    if step % (PHYSICS_HZ * 2) == 0 and step > 0:
        elapsed = time.time() - t0
        st = "✅" if alive_flag[0] else "💀"
        print(f"  ⏱️  t={sim_time:.1f}s | {elapsed:.1f}s wall | {st} | plats: {len(platforms_reached)}/{len(platforms)}")
        episode["timeline"].append({"t": round(sim_time,1), "alive": alive_flag[0], "plats": len(platforms_reached)})

elapsed = time.time() - t0
survival = (death_step / PHYSICS_HZ) if death_step else SIM_SECONDS
score = len(platforms_reached) * 100 + int(survival * 10)

episode.update({
    "survival_s": round(survival, 3), "wall_s": round(elapsed, 1),
    "alive": alive_flag[0], "death_reason": death_info_buf[0][0] if death_info_buf[0] else None,
    "platforms_reached": list(platforms_reached), "score": score,
    "realtime_factor": round(SIM_SECONDS / elapsed, 2),
})

print("\n" + "=" * 70)
print("🔥 FLOOR IS LAVA — RESULTS")
print("=" * 70)
print("  🤖 Unitree G1 (29-DoF, EDU Plus)")
print(f"  🏠 Cagatay's Room ({mesh_count} meshes, {total_verts:,} verts)")
print(f"  ⏱️  Survival: {survival:.2f}s / {SIM_SECONDS:.0f}s")
alive_str = "✅ SURVIVED!" if alive_flag[0] else f"💀 {death_info_buf[0][0]}" if death_info_buf[0] else "💀"
print(f"  {alive_str}")
print(f"  🟢 Platforms: {len(platforms_reached)}/{len(platforms)}")
print(f"  🏆 Score: {score}")
print(f"  ⚡ {elapsed:.1f}s wall ({episode['realtime_factor']}x RT) — NVIDIA L40S GPU")

results_path = os.path.join(OUTPUT_DIR, "floor_is_lava_results.json")
with open(results_path, "w") as f:
    json.dump(episode, f, indent=2)
scene_usd = os.path.join(OUTPUT_DIR, "g1_floor_is_lava.usd")
stage.GetRootLayer().Export(scene_usd)
size_mb = os.path.getsize(scene_usd) / (1024*1024)
print(f"\n💾 {results_path}")
print(f"💾 {scene_usd} ({size_mb:.1f} MB)")
print("=" * 70)

contact_sub = None
sim_app.close()
