"""Spawn-validation probe for the free-base Inspire-hand G1 (run BEFORE building the env).

The Inspire USD bakes a fixed root joint + gravity-off (it ships for tabletop manipulation).
`locomotion.make_floating_base` had to deactivate that joint by hand in the standalone demo,
so before relying on the manager-based spawn path we must CONFIRM that `fix_root_link=False`
alone yields a true floating base here.

Checks (printed as PASS/FAIL):
  1. robot.is_fixed_base is False              -> the baked root joint was disabled (free base)
  2. all 29 body joints + the EE/foot/pelvis bodies resolve by name
  3. with gravity ON and zero action, the robot STANDS for 3 s (pelvis z stays ~0.7-0.85,
     doesn't fall to the floor or explode) -> stable spawn, deploy gains hold it up

Run:
  cd ~/workspaces/git/IsaacLab && export LD_PRELOAD=$LD_PRELOAD:/lib/aarch64-linux-gnu/libgomp.so.1 \
    && ./isaaclab.sh -p ~/workspaces/git/robots/examples/isaac_bed_making/rl/check_spawn.py
"""

from isaaclab.app import AppLauncher

app_launcher = AppLauncher({"headless": True})
simulation_app = app_launcher.app

import os  # noqa: E402
import sys  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rl.robot_cfg import (  # noqa: E402
    BODY_JOINTS,
    FOOT_BODIES,
    LEFT_EE_BODY,
    PELVIS_BODY,
    RIGHT_EE_BODY,
    make_bed_g1_cfg,
)


def main() -> int:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
    sim.set_camera_view([2.5, 2.5, 2.0], [0.0, 0.0, 0.8])

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1500.0).func("/World/Light", sim_utils.DomeLightCfg(intensity=1500.0))

    robot = Articulation(make_bed_g1_cfg(prim_path="/World/Robot"))

    sim.reset()

    ok = True

    # 1) Free base?
    free = not bool(robot.is_fixed_base)
    print(f"[check] is_fixed_base={robot.is_fixed_base}  ->  free base: {'PASS' if free else 'FAIL'}")
    ok = ok and free

    # 2) Names resolve?
    body_ids, body_names = robot.find_joints(BODY_JOINTS, preserve_order=False)
    n_body = len(body_ids)
    print(f"[check] resolved {n_body}/29 body joints  ->  {'PASS' if n_body == 29 else 'FAIL'}")
    ok = ok and (n_body == 29)
    for label, expr in [
        ("right EE", RIGHT_EE_BODY),
        ("left EE", LEFT_EE_BODY),
        ("pelvis", PELVIS_BODY),
        ("feet", FOOT_BODIES),
    ]:
        ids, names = robot.find_bodies(expr, preserve_order=False)
        print(f"[check]   body '{expr}' ({label}) -> {names}")
        ok = ok and (len(ids) >= 1)
    print(f"[check] total joints={robot.num_joints}, total bodies={robot.num_bodies}")

    # 3) Stands under gravity with zero action (hold default joint targets) for 3 s.
    default_targets = robot.data.default_joint_pos.clone()
    z0 = robot.data.root_pos_w[0, 2].item()
    zmin, zmax = z0, z0
    steps = 600  # 3 s at dt=0.005
    for i in range(steps):
        robot.set_joint_position_target(default_targets)
        robot.write_data_to_sim()
        sim.step()
        robot.update(0.005)
        z = robot.data.root_pos_w[0, 2].item()
        zmin, zmax = min(zmin, z), max(zmax, z)
    zf = robot.data.root_pos_w[0, 2].item()
    stood = (0.55 < zf < 0.95) and (zmin > 0.40)
    print(
        f"[check] pelvis z: spawn={z0:.3f} final={zf:.3f} min={zmin:.3f} max={zmax:.3f}  "
        f"->  stands: {'PASS' if stood else 'FAIL'}"
    )
    ok = ok and stood

    print(f"[check] OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


rc = main()
simulation_app.close()
sys.exit(rc)
