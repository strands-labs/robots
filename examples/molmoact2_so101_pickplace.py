#!/usr/bin/env python3
"""SO101 pick-and-place driven by MolmoAct2 — strands_robots simplified API.

Demonstrates how strands_robots.policies.create_policy() eliminates manual
configuration that the raw lerobot path requires. The embodiment="so_real"
parameter handles:
  - Motor key mapping (shoulder_pan.pos, shoulder_lift.pos, ...)
  - Camera observation rename (front -> observation.images.image)
  - State/action dimensionality reconciliation (dim_policy="pad")
  - MolmoAct2 transformers-native checkpoint detection
  - Normalization tag auto-discovery from norm_stats.json
  - Processor bridge construction (pre/post processing pipelines)

Hardware requirements:
  - SO101 follower arm on a serial port
  - Front camera (OpenCV-compatible, index 0)
  - CUDA GPU for inference (or cpu for testing)

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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("molmoact2_so101")

REPO = "allenai/MolmoAct2-SO100_101"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM1")
    ap.add_argument("--cal", default="orange_follower")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--task", default="Pick up the pen and place it on the paper")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from lerobot.cameras.opencv import OpenCVCameraConfig
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    from strands_robots.policies import create_policy

    # Camera config: 'front' matches the so_real embodiment's obs_rename
    cam_cfg = {"front": OpenCVCameraConfig(index_or_path=args.camera, width=640, height=480, fps=30)}
    robot = SO101Follower(SO101FollowerConfig(port=args.port, id=args.cal, cameras=cam_cfg))
    log.info("Connecting SO101 @ %s (cal=%s)...", args.port, args.cal)
    robot.connect(calibrate=False)
    log.info("Connected. obs keys: %s", list(robot.get_observation().keys()))

    # ONE call creates and configures the entire policy:
    #   - Detects MolmoAct2 transformers-native checkpoint
    #   - Loads 'so_real' embodiment (motor keys + camera renames)
    #   - Builds MolmoAct2Config, norm_tag, processor bridge
    #   - robot_state_keys auto-set from embodiment.action_keys
    policy = create_policy(REPO, embodiment="so_real", device="cuda")
    policy.reset()

    async def run():
        period = 1.0 / args.hz
        for step in range(args.steps):
            obs = robot.get_observation()
            t = time.time()
            actions = await policy.get_actions(obs, args.task)
            dt = time.time() - t
            a = actions[0]
            log.info("step %d infer=%.2fs action=%s", step, dt, {k: round(v, 1) for k, v in a.items()})
            if not args.dry_run:
                robot.send_action(a)
            await asyncio.sleep(max(0, period - dt))

    try:
        asyncio.run(run())
    finally:
        try:
            robot.disconnect()
        except Exception as e:
            log.warning("disconnect: %s", str(e)[:80])
        log.info("Done.")


if __name__ == "__main__":
    main()
