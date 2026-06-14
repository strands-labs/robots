#!/usr/bin/env python3
"""SO101 pick-and-place in MuJoCo simulation — same policy, sim robot.

Identical to ``molmoact2_so101_pickplace.py`` but swaps the robot to sim mode.
This demonstrates that the abstraction works: one line changes from real to sim,
everything else (policy, inference loop, observation keys) stays the same.

Usage:
  export STRANDS_TRUST_REMOTE_CODE=1
  export MUJOCO_GL=egl   # headless rendering (CI / SSH sessions)
  python molmoact2_sim_pickplace.py --task "Pick up the red cube"
  python molmoact2_sim_pickplace.py --dry-run  # policy inference only, no actuation
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from strands_robots import Robot
from strands_robots.policies import create_policy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("molmoact2_sim")

REPO = "allenai/MolmoAct2-SO100_101"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Pick up the red cube and place it on the plate")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # Only difference from the real example: mode="sim" (the default).
    # No serial port, no camera hardware — MuJoCo provides observations.
    sim = Robot("so101")  # mode="sim" is the default
    log.info("Simulation world created: %s", sim.tool_name)

    # Add a camera to the sim world for visual observations.
    sim._dispatch_action(
        "add_camera",
        {
            "camera_name": "front",
            "position": [0.5, 0.0, 0.5],
            "target": [0.0, 0.0, 0.1],
            "width": 640,
            "height": 480,
        },
    )

    # Same create_policy call — the abstraction is identical.
    policy = create_policy(REPO, embodiment="so_real", device=args.device)
    policy.reset()

    # Retrieve initial sim state to verify observation keys.
    state = sim._dispatch_action("get_state", {})
    log.info("Sim state keys: %s", list(state.keys()) if isinstance(state, dict) else "N/A")

    async def run():
        period = 1.0 / args.hz
        for step in range(args.steps):
            # In sim mode, observations come from MuJoCo rendering.
            obs_result = sim._dispatch_action("get_observation", {"camera_name": "front"})
            t = time.time()
            actions = await policy.get_actions(obs_result, args.task)
            dt = time.time() - t
            a = actions[0]
            log.info("step %d infer=%.2fs action=%s", step, dt, {k: round(v, 1) for k, v in a.items()})
            if not args.dry_run:
                sim._dispatch_action("set_joint_positions", {"positions": a})
            await asyncio.sleep(max(0, period - dt))

    try:
        asyncio.run(run())
    finally:
        sim.destroy()
        log.info("Done.")


if __name__ == "__main__":
    main()
