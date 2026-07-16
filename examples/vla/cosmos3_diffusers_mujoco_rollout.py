#!/usr/bin/env python3
"""End-to-end: drive a MuJoCo Franka/Panda arm with Cosmos 3 (diffusers backend).

Runs the full in-process pipeline the ``cosmos3-diffusers`` + ``cosmos3-sim``
extras enable:

    Cosmos3OmniPipeline  ->  raw [-1,1] unified action chunk
                         ->  denormalize_quantile (bundled q01/q99 stats)
                         ->  decode_pose_trajectory (relative EE pose -> absolute SE3)
                         ->  MinkIKBridge (differential IK on the MuJoCo model)
                         ->  MuJoCo joint targets the arm tracks

and reports the Cartesian tracking error. With ``--render`` it writes a
side-by-side video (MuJoCo arm | Cosmos predicted world).

This needs a CUDA GPU, the Cosmos 3 weights, native diffusers-from-source
(ships ``Cosmos3OmniPipeline``), and the sim extra. It is the runnable form of
the headless Thor rollout attached to PR #458.

    uv pip install "strands-robots[cosmos3-diffusers,cosmos3-sim]" \
        "diffusers @ git+https://github.com/huggingface/diffusers"
    python examples/vla/cosmos3_diffusers_mujoco_rollout.py --instruction "pick up the red cube" --render out.mp4
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="nvidia/Cosmos3-Nano", help="HF repo id / local path")
    ap.add_argument("--instruction", default="pick up the red cube", help="task prompt")
    ap.add_argument("--embodiment", default="droid", help="Cosmos 3 embodiment key")
    ap.add_argument("--steps", type=int, default=16, help="diffusion sampling steps")
    ap.add_argument("--render", default=None, metavar="MP4", help="write a side-by-side rollout video here")
    args = ap.parse_args()

    from strands_robots.policies.cosmos3 import (
        Cosmos3Policy,
        MinkIKBridge,
        decode_cosmos_chunk_to_targets,
    )
    from strands_robots.policies.cosmos3.embodiments import get_embodiment

    # 1) Cosmos 3 in-process forward pass -> raw [-1, 1] action chunk + world video.
    policy = Cosmos3Policy(
        embodiment=args.embodiment,
        backend="diffusers",
        model=args.model,
        mode="policy",
    )
    policy.set_robot_state_keys([f"joint_{i}" for i in range(7)] + ["gripper"])

    img = np.full((480, 640, 3), 127, dtype=np.uint8)
    obs = {
        "observation/wrist_image_left": img,
        "observation/exterior_image_1_left": img,
        "observation/exterior_image_2_left": img,
        **{f"joint_{i}": 0.0 for i in range(7)},
        "gripper": 0.0,
    }
    policy.get_actions_sync(obs, args.instruction)
    raw_chunk = policy.last_rollout["action"]
    world = policy.last_rollout["video"]
    print(f"Cosmos action chunk: {np.asarray(raw_chunk).shape}  world video: {np.asarray(world).shape}")

    # 2) De-normalize -> decode EE poses -> IK to MuJoCo joint targets.
    import mujoco
    from robot_descriptions import panda_mj_description

    model = mujoco.MjModel.from_xml_path(panda_mj_description.MJCF_PATH)
    bridge = MinkIKBridge(model, ee_frame_name="hand", ee_frame_type="body")
    q_init = np.zeros(model.nq)
    q_init[:7] = [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79]

    out = decode_cosmos_chunk_to_targets(
        np.asarray(raw_chunk, dtype=np.float32), get_embodiment(args.embodiment), bridge, q_init
    )
    print(f"joint targets: {out['qpos'].shape}  tracking error: {out['tracking_error']}")

    # 3) Optional: render MuJoCo arm tracking the trajectory, side-by-side with Cosmos world.
    if args.render:
        import os

        os.environ.setdefault("MUJOCO_GL", "egl")
        import imageio.v3 as iio
        from PIL import Image

        scene = os.path.join(os.path.dirname(panda_mj_description.MJCF_PATH), "scene.xml")
        sm = mujoco.MjModel.from_xml_path(scene) if os.path.exists(scene) else model
        d = mujoco.MjData(sm)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0.4, 0.0, 0.4]
        cam.distance, cam.azimuth, cam.elevation = 1.8, 130, -20
        r = mujoco.Renderer(sm, 480, 640)
        traj = np.vstack([q_init, out["qpos"]])
        worldv = np.asarray(world)
        if worldv.dtype != np.uint8:
            worldv = np.clip(worldv * 255 if worldv.max() <= 1.0 else worldv, 0, 255).astype(np.uint8)
        frames = []
        for i, qi in enumerate(traj):
            d.qpos[: len(qi)] = qi
            mujoco.mj_forward(sm, d)
            r.update_scene(d, camera=cam)
            left = r.render()
            j = min(i, len(worldv) - 1)
            right = np.asarray(Image.fromarray(worldv[j]).resize((640, 480)))
            frames.append(np.concatenate([left, right], axis=1))
        iio.imwrite(args.render, np.stack(frames), fps=8, codec="libx264")
        print(f"wrote {args.render}")
        # Tegra/EGL: the GL destructor can crash at exit; hard-exit to skip it.
        os._exit(0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
