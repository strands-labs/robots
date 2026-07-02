"""Author a MOBILE-base Inspire-hand G1 USD (one-time asset fix).

The stock `g1_29dof_inspire_hand.usd` puts the `ArticulationRootAPI` on a baked
`/Robot/root_joint` PhysicsFixedJoint that pins the pelvis to the world. Setting
`fix_root_link=False` only *disables* that joint, after which Isaac can no longer resolve
the articulation root (it still looks for it on `root_joint`) -> "Failed to create
articulation". This is the NVIDIA-forum "Inspire mobile-base USD compatibility" issue.

This writes a tiny LOCAL override layer that references the stock USD and:
  * deactivates `/Robot/root_joint` (removes the world pin), and
  * applies the `ArticulationRootAPI` (+ PhysxArticulationAPI) to `/Robot/pelvis`,
so the manager-based spawn path builds a true floating articulation. The output references
the stock USD (stays tiny; no inlined meshes), so it needs the same asset server the stock
config already uses.

Run once:
  cd ~/workspaces/git/IsaacLab && export LD_PRELOAD=$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1 \
    && ./isaaclab.sh -p ~/workspaces/git/robots/examples/isaac_bed_making/rl/make_inspire_mobile_usd.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import os  # noqa: E402

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

SRC = f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1_29dof_inspire_hand.usd"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "g1_inspire_mobile.usd")


def _roots_with_articulation_api(stage: Usd.Stage) -> list[str]:
    found = []
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            found.append(prim.GetPath().pathString)
    return found


def main() -> int:
    # 1) Inspect the stock asset so we know exactly where the root API lives.
    src_stage = Usd.Stage.Open(SRC)
    default_prim = src_stage.GetDefaultPrim()
    print(f"[usd] source default prim: {default_prim.GetPath()}")
    print(f"[usd] source prims carrying ArticulationRootAPI: {_roots_with_articulation_api(src_stage)}")

    # 2) Author the local override layer.
    if os.path.exists(OUT):
        os.remove(OUT)
    stage = Usd.Stage.CreateNew(OUT)
    robot = stage.DefinePrim("/Robot", "Xform")
    robot.GetReferences().AddReference(SRC)
    stage.SetDefaultPrim(robot)

    # remove the world pin
    rj = stage.OverridePrim("/Robot/root_joint")
    rj.SetActive(False)

    # put the articulation root on the pelvis link
    pelvis = stage.OverridePrim("/Robot/pelvis")
    UsdPhysics.ArticulationRootAPI.Apply(pelvis)
    PhysxSchema.PhysxArticulationAPI.Apply(pelvis)

    stage.GetRootLayer().Save()

    # 3) Verify the composed result.
    check = Usd.Stage.Open(OUT)
    rj_active = check.GetPrimAtPath("/Robot/root_joint").IsActive()
    pelvis_is_root = check.GetPrimAtPath("/Robot/pelvis").HasAPI(UsdPhysics.ArticulationRootAPI)
    print(f"[usd] wrote {OUT}")
    print(f"[usd] composed root_joint active = {rj_active} (want False)")
    print(f"[usd] composed pelvis has ArticulationRootAPI = {pelvis_is_root} (want True)")
    print(f"[usd] composed prims carrying ArticulationRootAPI: {_roots_with_articulation_api(check)}")
    ok = (not rj_active) and pelvis_is_root
    print(f"[usd] OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


rc = main()
simulation_app.close()
import sys  # noqa: E402

sys.exit(rc)
