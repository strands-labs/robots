#!/usr/bin/env python3
"""Multiple robots in one scene — MuJoCo solver."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import newton.examples as ne  # noqa: E402

from strands_robots.newton import NewtonBackend, NewtonConfig  # noqa: E402

print("🤖🤖 Newton Example: Multi-Robot Scene")
print("=" * 50)

b = NewtonBackend(NewtonConfig(solver="mujoco", device="cuda:0"))
b.create_world()

# Add 3 quadrupeds at different positions
for i, pos in enumerate([(0, 0, 0), (2, 0, 0), (-2, 0, 0)]):
    r = b.add_robot(f"quad_{i}", urdf_path=ne.get_asset("quadruped.urdf"), position=pos)
    print(f"  Added quad_{i}: {r['robot_info']['num_joints']} joints at {pos}")

# Add ant
r = b.add_robot("ant", urdf_path=ne.get_asset("nv_ant.xml"), position=(0, 2, 0), data_config={"format": "mjcf"})
print(f"  Added ant: {r['robot_info']['num_joints']} joints")

state = b.get_state()
print(f"\nScene: {len(state['robots'])} robots")

t0 = time.time()
for _ in range(200):
    b.step()
elapsed = time.time() - t0

obs = b.get_observation()
print(f"\n200 steps in {elapsed:.3f}s")
for rname, data in obs["observations"].items():
    jp = data.get("joint_positions")
    if jp is not None:
        print(f"  {rname}: {len(jp)} joints, pos[0]={jp[0]:.4f}")

b.destroy()
print("\n✅ Done!")
