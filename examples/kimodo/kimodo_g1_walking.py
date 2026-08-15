"""Kimodo + MuJoCo end-to-end example: text prompt -> G1 walking in sim.

The full text-to-motion-to-physics pipeline:
  text prompt -> Kimodo diffusion -> qpos frames -> SLERP upsample ->
  ProtoMotions GTP ONNX tracker -> G1 physics @ 1kHz -> rendered MP4

Run:
  STRANDS_TRUST_REMOTE_CODE=1 python examples/kimodo/kimodo_g1_walking.py \
      --prompt "a person walking forward with confident strides"

The tracker layer is composed via CompositePolicy: Kimodo emits motion targets,
WBC/PD tracks them. If you don't have the tracker installed, run WITHOUT
--tracker to visualise the kinematic reference directly (sets qpos each tick).
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
        "--tracker",
        action="store_true",
        help="Compose with WBC tracker (requires [wbc] extra + weights)",
    )
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

    policy_config = {
        "diffusion_steps": args.diffusion_steps,
        "guidance_scale": args.guidance_scale,
        "num_frames": args.num_frames,
        "device": "cuda",
        "dtype": "fp16",
    }

    provider = "kimodo"
    if args.tracker:
        # Composite: Kimodo emits targets, WBC tracks them.
        provider = "composite"
        policy_config = {
            "layers": [
                {
                    "provider": "kimodo",
                    "config": policy_config,
                },
                {"provider": "wbc"},
            ]
        }

    result = sim.run_policy(
        robot_name="g1",
        policy_provider=provider,
        policy_config=policy_config,
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
