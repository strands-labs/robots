#!/usr/bin/env python3
"""🔥 FLOOR IS LAVA — FINAL VERSION
- Delete FixedJoint for free-floating robot
- XformCache readback (only method that works with MJCF)
- Detect explosion and stop scoring at that point
- 120Hz (faster), render every 4 steps"""
import json
import math
import time

import numpy as np
from isaacsim import SimulationApp

sim_app = SimulationApp({"headless": True, "width": 1920, "height": 1080})
import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
from omni.isaac.core import World  # noqa: E402
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics  # noqa: E402

OUTPUT = "/home/ubuntu/room_sim/output"
world = World(stage_units_in_meters=1.0, physics_dt=1.0/120.0, rendering_dt=1.0/30.0)
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
phys = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
phys.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
phys.CreateGravityMagnitudeAttr(9.81)
px = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
px.CreateEnableCCDAttr(True)
px.CreateEnableGPUDynamicsAttr(True)
px.CreateBroadphaseTypeAttr("GPU")
stage.DefinePrim("/World/DomeLight", "DomeLight").GetAttribute("inputs:intensity").Set(3000.0)
# Room
rs = Usd.Stage.Open("/home/ubuntu/room_sim/extracted/3_7_2026.usda")
UsdGeom.Xform.Define(stage, "/World/Room").AddRotateXOp().Set(-90.0)
mc = 0
verts = 0
for p in rs.Traverse():
    if p.IsA(UsdGeom.Mesh):
        sm = UsdGeom.Mesh(p)
        pts = sm.GetPointsAttr().Get()
        fvc = sm.GetFaceVertexCountsAttr().Get()
        fvi = sm.GetFaceVertexIndicesAttr().Get()
        if not pts or not fvc or not fvi:
            continue
        dp = "/World/Room/m%d" % mc
        dm = UsdGeom.Mesh.Define(stage, dp)
        dm.CreatePointsAttr(pts)
        dm.CreateFaceVertexCountsAttr(fvc)
        dm.CreateFaceVertexIndicesAttr(fvi)
        dm.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.15, 0.05)])
        UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(dp))
        UsdPhysics.MeshCollisionAPI.Apply(stage.GetPrimAtPath(dp)).CreateApproximationAttr("meshSimplification")
        sxf = UsdGeom.Xformable(p)
        dxf = UsdGeom.Xformable(stage.GetPrimAtPath(dp))
        for op in sxf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                dxf.AddTranslateOp().Set(op.Get())
            elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
                dxf.AddScaleOp().Set(op.Get())
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                dxf.AddOrientOp().Set(op.Get())
        mc += 1
        verts += len(pts)
# Catch floor
fp = stage.DefinePrim("/World/CatchFloor", "Cube")
UsdGeom.Cube(fp).CreateSizeAttr(200.0)
UsdGeom.Xformable(fp).AddTranslateOp().Set(Gf.Vec3d(0, 0, -100.05))
UsdPhysics.CollisionAPI.Apply(fp)
# G1
_, cfg = omni.kit.commands.execute("MJCFCreateImportConfig")
_, _ = omni.kit.commands.execute("MJCFCreateAsset", mjcf_path="/home/ubuntu/strands-gtc-nvidia/strands_robots/assets/unitree_g1/scene.xml", import_config=cfg, prim_path="/World/G1")
g1 = stage.GetPrimAtPath("/World/G1")
# DELETE FixedJoint — free the robot
stage.RemovePrim("/World/G1/joints/rootJoint_pelvis")
inner_pelvis = stage.GetPrimAtPath("/World/G1/pelvis/pelvis")
inner_pelvis.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, 0.8))
for ch in Usd.PrimRange(g1):
    if ch.HasAPI(UsdPhysics.RigidBodyAPI) and not ch.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(ch)
# Green safe pad
sp = stage.DefinePrim("/World/SafePad", "Cube")
UsdGeom.Cube(sp).CreateSizeAttr(1.0)
UsdGeom.Xformable(sp).AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.025))
UsdGeom.Xformable(sp).AddScaleOp().Set(Gf.Vec3d(3, 3, 0.05))
UsdGeom.Cube(sp).CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.9, 0.2)])
UsdPhysics.CollisionAPI.Apply(sp)
# Drives
drives = []
dnames = []
for ch in Usd.PrimRange(g1):
    if ch.IsA(UsdPhysics.Joint):
        d = UsdPhysics.DriveAPI.Get(ch, "angular")
        if d and d.GetTargetPositionAttr():
            drives.append(d)
            dnames.append(ch.GetName())
jc = sum(1 for c in Usd.PrimRange(g1) if c.IsA(UsdPhysics.Joint))
bc = sum(1 for c in Usd.PrimRange(g1) if c.HasAPI(UsdPhysics.RigidBodyAPI))
GAIT_FREQ = 1.2
GAIT = {"hip_pitch":(.25,-.1),"hip_roll":(.05,0),"knee":(.3,-.15),"ankle_pitch":(.15,.05),"ankle_roll":(.02,0),"shoulder_pitch":(.1,0),"elbow":(.08,-.2),"waist":(.01,0)}
def gait_fn(name, t):
    n = name.lower()
    for k,(a,o) in GAIT.items():
        if k in n:
            return o + a*np.sin(2*np.pi*GAIT_FREQ*t + (np.pi if "right" in n else 0))
    return 0.0
# ═══ RUN ═══
world.reset()
SIM_S = 10.0
HZ = 120
total = int(SIM_S*HZ)
LAVA_Z = 0.05
result = {"events": [], "traces": []}
alive = True
death_step = None
death_reason = None
t0 = time.time()
min_z = 999
exploded = False
explode_t = None
for step in range(total):
    st = step / HZ
    if step % max(1, HZ//50) == 0 and not exploded:
        for d, dn in zip(drives, dnames):
            try:
                d.GetTargetPositionAttr().Set(float(np.degrees(gait_fn(dn, st))))
            except Exception:
                pass
    world.step(render=(step % 4 == 0))
    if step % 4 == 0 and step % 12 == 0:
        xfc = UsdGeom.XformCache()
        try:
            xfm = xfc.GetLocalToWorldTransform(inner_pelvis)
            tv = xfm.ExtractTranslation()
            pz = float(tv[2])
            if math.isnan(pz):
                continue
            if abs(pz) > 50:
                if not exploded:
                    exploded = True
                    explode_t = st
                    result["events"].append({"t": round(st,3), "event": "EXPLOSION", "z": round(pz,1)})
                continue
            if pz < min_z:
                min_z = pz
        except Exception:
            continue
        if step % (HZ // 2) == 0:
            result["traces"].append({"t": round(st, 2), "z": round(pz, 4)})
        if alive and st > 0.3 and pz < LAVA_Z:
            alive = False
            death_step = step
            death_reason = "LAVA! pelvis z=%.4fm" % pz
            result["events"].append({"t": round(st,3), "event": "DEATH", "z": round(pz,4)})
elapsed = time.time() - t0
surv_s = (explode_t if explode_t else ((death_step/HZ) if death_step else SIM_S))
score = int(surv_s * 10)
if alive and not exploded:
    score += 500
result.update({
    "robot": "Unitree G1 29-DoF EDU Plus (NVIDIA Orin)",
    "room": "Cagatay's Room (%d meshes, %s verts)" % (mc, "{:,}".format(verts)),
    "lava_threshold_z": LAVA_Z,
    "survival_s": round(surv_s, 3),
    "wall_s": round(elapsed, 1),
    "alive": alive and not exploded,
    "death_reason": death_reason or ("Articulation explosion at %.1fs" % explode_t if explode_t else None),
    "exploded": exploded,
    "explode_time": round(explode_t, 3) if explode_t else None,
    "min_pelvis_z": round(min_z, 4) if min_z < 900 else None,
    "joints": jc, "bodies": bc, "drives": len(drives),
    "score": score,
    "realtime_factor": round(SIM_S/max(0.1,elapsed), 2),
    "physics_hz": HZ, "gpu": "NVIDIA L40S",
    "notes": "MJCF FixedJoint deleted for free-float. XformCache readback. Explodes ~1.8s due to articulation instability without root constraint.",
})
with open(OUTPUT + "/floor_is_lava_results.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
stage.GetRootLayer().Export(OUTPUT + "/g1_floor_is_lava.usd")
sim_app.close()
