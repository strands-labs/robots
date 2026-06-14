#!/usr/bin/env python3
"""SO101 pick-and-place driven by MolmoAct2 — strands_robots simplified API.

Demonstrates how strands_robots wraps the entire robot + policy lifecycle.
The user never imports lerobot directly; all complexity is behind two calls:

  1. ``Robot("so101", mode="real", ...)`` — creates a connected hardware robot
  2. ``create_policy(REPO, embodiment="so_real", ...)`` — loads MolmoAct2 with
     correct motor-key mapping, camera renames, normalization, and processors.

Hardware requirements:
  - SO101 follower arm on a serial port
  - Front camera (OpenCV-compatible, index 0)
  - CUDA GPU for inference (or cpu with --device cpu)

Usage:
  export STRANDS_TRUST_REMOTE_CODE=1
  python molmoact2_so101_pickplace.py --task "Pick up the pen"
  python molmoact2_so101_pickplace.py --dry-run  # no motor commands
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from strands_robots import Robot
from strands_robots.policies import create_policy

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("molmoact2_so101")

REPO = "allenai/MolmoAct2-SO100_101"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--calibration-id", default="orange_follower")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--task", default="Pick up the pen and place it on the paper")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # strands_robots.Robot wraps lerobot hardware setup.  The cameras dict
    # maps logical names (matching embodiment obs_rename) to OpenCV config.
    robot = Robot(
        "so101",
        mode="real",
        port=args.port,
        id=args.calibration_id,
        cameras={
            "front": {
                "type": "opencv",
                "index_or_path": args.camera,
                "width": 640,
                "height": 480,
                "fps": 30,
            }
        },
    )
    log.info("Connecting SO101 @ %s (id=%s)...", args.port, args.calibration_id)
    robot.robot.connect(calibrate=False)
    log.info("Connected. obs keys: %s", list(robot.robot.get_observation().keys()))

    # create_policy auto-detects MolmoAct2, loads the 'so_real' embodiment
    # (motor keys + camera renames), builds processors, and returns a ready
    # Policy instance.
    policy = create_policy(REPO, embodiment="so_real", device=args.device)
    policy.reset()

    async def run():
        period = 1.0 / args.hz
        for step in range(args.steps):
            obs = robot.robot.get_observation()
            t = time.time()
            actions = await policy.get_actions(obs, args.task)
            dt = time.time() - t
            a = actions[0]
            log.info("step %d infer=%.2fs action=%s", step, dt, {k: round(v, 1) for k, v in a.items()})
            if not args.dry_run:
                robot.robot.send_action(a)
            await asyncio.sleep(max(0, period - dt))

    try:
        asyncio.run(run())
    finally:
        try:
            robot.robot.disconnect()
        except Exception as e:
            log.warning("disconnect: %s", str(e)[:80])
        log.info("Done.")


if __name__ == "__main__":
    main()
