#!/usr/bin/env python3
"""Roll out the Cosmos 3 VLA policy in MuJoCo and (optionally) record an episode.

This is the end-to-end smoke test for the ``cosmos3`` policy provider: a
Franka/DROID arm in MuJoCo is driven by the real
``nvidia/Cosmos3-Nano-Policy-DROID`` policy served by the Cosmos Framework
RoboLab WebSocket server.

Prerequisites
-------------
1. Client deps (this package + the Cosmos 3 service extra + sim):

     pip install -e '.[sim-mujoco]'
     pip install 'strands-robots[cosmos3-service]'   # msgpack + websockets only
     # robot_descriptions provides the Franka MJCF asset (part of sim-mujoco)

   The ``cosmos3-service`` extra is intentionally **numpy-version agnostic**
   (no ``openpi-client``): it ships only ``msgpack`` + ``websockets`` plus a
   vendored numpy packer. That means it composes cleanly with ``lerobot``
   (``numpy>=2``) for LeRobotDataset recording in the same env.

2. The policy server (holds the GPU) — from a Cosmos Framework checkout:

     uv sync --all-extras --group=cu130-train --group=policy-server
     python -m cosmos_framework.scripts.action_policy_server_robolab \
         --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port 8000
     # wait for:  curl http://localhost:8000/healthz   -> 200

Run
---
    # headless servers need an EGL/OSMesa GL backend for offscreen rendering
    MUJOCO_GL=egl python examples/cosmos3_sim_rollout.py
    MUJOCO_GL=egl python examples/cosmos3_sim_rollout.py --record /tmp/c3_rollout.mp4

Verified on a single L40S: 32-step DROID action chunks (~3 s warm/chunk), the
Franka arm physically moves under the policy, and an MP4 is recorded.
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost", help="Policy-server host.")
    parser.add_argument("--port", type=int, default=8000, help="Policy-server WebSocket port.")
    parser.add_argument("--instruction", default="pick up the red cube", help="Task instruction.")
    parser.add_argument("--n-steps", type=int, default=24, help="Control steps to roll out.")
    parser.add_argument("--control-frequency", type=float, default=15.0, help="Control Hz (match policy fps).")
    parser.add_argument("--action-horizon", type=int, default=8, help="Steps consumed per policy chunk.")
    parser.add_argument("--record", metavar="MP4_PATH", default=None, help="Record the rollout to this MP4.")
    parser.add_argument("--robot", default="franka", help="Sim arm data_config (franka/panda — DROID = Franka).")
    args = parser.parse_args()

    # Headless GL default so the example runs on servers out of the box.
    os.environ.setdefault("MUJOCO_GL", "egl")

    try:
        from strands_robots import Simulation
        from strands_robots.policies.cosmos3 import Cosmos3Policy
    except ImportError as e:
        print(f"Missing deps: {e}\nInstall: pip install -e '.[sim-mujoco]' && pip install 'strands-robots[cosmos3-service]'")
        return 2

    print(f"Building MuJoCo world with a '{args.robot}' arm + cube + 3 cameras ...")
    sim = Simulation(tool_name="sim", mesh=False)
    sim.create_world()
    # DROID == Franka Emika Panda. The 'droid' Cosmos embodiment drives a
    # Franka/DROID-class arm; use the 'franka' (or 'panda') sim asset here.
    sim.add_robot(name="arm", data_config=args.robot)
    sim.add_object(name="cube", shape="box", position=[0.4, 0.0, 0.05],
                   size=[0.025, 0.025, 0.025], color=[1, 0, 0, 1], mass=0.1)
    sim.add_camera(name="wrist", position=[0.3, 0.0, 0.5], target=[0.4, 0, 0.05])
    sim.add_camera(name="front", position=[0.9, 0.0, 0.4], target=[0.4, 0, 0.05])
    sim.add_camera(name="side",  position=[0.4, 0.6, 0.4], target=[0.4, 0, 0.05])

    # The DROID policy conditions on 3 cameras (wrist + 2 exterior) + 7-DOF
    # joints + gripper. Map our sim camera names onto the server's OpenPI keys.
    obs_mapping = {
        "wrist": "observation/wrist_image_left",
        "front": "observation/exterior_image_1_left",
        "side":  "observation/exterior_image_2_left",
    }
    # The Cosmos3 client uses a self-contained msgpack+websockets transport
    # (numpy-version agnostic — composes with lerobot).
    policy = Cosmos3Policy(
        embodiment="droid", host=args.host, port=args.port,
        robot=args.robot,                # map joint_0..6/gripper -> sim actuator names
        observation_mapping=obs_mapping,
    )

    video = None
    if args.record:
        video = {"path": args.record, "camera": "front", "fps": int(args.control_frequency)}

    print(f"Rolling out Cosmos 3 (ws://{args.host}:{args.port}) — {args.n_steps} steps ...")
    try:
        result = sim.run_policy(
            robot_name="arm",
            policy_object=policy,
            instruction=args.instruction,
            n_steps=args.n_steps,
            control_frequency=args.control_frequency,
            action_horizon=args.action_horizon,
            video=video,
        )
    except ConnectionError as e:
        # Cosmos3WebsocketClient raises an actionable hint when the server is down.
        print(str(e))
        return 1

    status = result.get("status") if isinstance(result, dict) else "unknown"
    detail = result.get("content") if isinstance(result, dict) else result
    print(f"status: {status}")
    print(detail)
    return 0 if status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
