"""Kimodo + MuJoCo example: text prompt -> Unitree G1 motion in sim -> MP4.

The pipeline this script runs:
  text prompt -> Kimodo diffusion -> 29-DOF qpos frames -> SLERP upsample to the
  control rate -> applied as G1 joint targets -> rendered MP4

Kimodo is a *kinematic* whole-body generator: it emits joint targets for all 29
leg + waist + arm DOFs, not torques. Applied directly, as here, the result is the
faithful visualisation of the generated motion.

Making the robot FOLLOW that motion under physics is a separate stage - a
controller that tracks the reference, with Kimodo's 29 targets as its input:

    prompt -> Kimodo -> 29 joint targets -> reference tracker -> torques -> robot

Generator and tracker are in series over the same joints, which is a cascade, not
a composition. ``CompositePolicy`` merges two policies over DISJOINT joint groups
(locomotion legs+waist plus manipulation arms) and cannot express it; handing it a
whole-body generator plus a whole-body controller gives both children the same
joints and discards one child's output entirely. ``WBCPolicy`` in particular is
not a reference tracker at all - its only command input is a target base velocity
and it has no reference-pose input. See ``docs/policies/kimodo.md``.

Run:
  STRANDS_TRUST_REMOTE_CODE=1 python examples/kimodo/kimodo_g1_walking.py \
      --prompt "a person walking forward with confident strides"
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Kimodo -> G1 MuJoCo demo")
    parser.add_argument(
        "--prompt",
        default="a person walking forward with confident strides",
        help="Natural-language motion description for Kimodo",
    )
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-frames", type=int, default=120)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--control-hz", type=int, default=50)
    parser.add_argument(
        "--out",
        default="kimodo_g1_walking.mp4",
        help="Output MP4 path (rendered from the sim's front camera)",
    )
    args = parser.parse_args()

    # Headless GL is required on Jetson/Docker.
    os.environ.setdefault("MUJOCO_GL", "egl")

    from strands_robots import Robot

    sim = Robot("g1", mesh=False)
    sim.add_camera(name="front", position=[3.0, 0.0, 1.2], target=[0.0, 0.0, 0.8])

    result = sim.run_policy(
        robot_name="g1",
        policy_provider="kimodo",
        policy_config={
            "diffusion_steps": args.diffusion_steps,
            "guidance_scale": args.guidance_scale,
            "num_frames": args.num_frames,
            "device": "cuda",
            "dtype": "fp16",
        },
        instruction=args.prompt,
        n_steps=args.n_steps,
        control_frequency=args.control_hz,
        video={"path": args.out, "camera": "front", "fps": 25},
    )
    print(f"Done. Video: {args.out}")
    print(f"Sim result: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
