"""Collect, train, and run a MolmoAct2 policy with Strands Robots and LeRobot.

The companion runnable example for the blog post of the same name. It covers the
full data flywheel for a vision-language-action (VLA) policy, all on one machine:

  1. COLLECT  - drive a pretrained MolmoAct2 checkpoint in MuJoCo simulation to
                bootstrap a LeRobotDataset of pick-and-place demonstrations, with
                domain randomization for variety, and optionally push it to the
                Hugging Face Hub.
  2. TRAIN    - fine-tune MolmoAct2 on that dataset. Training runs UPSTREAM in
                LeRobot (strands-robots ships data collection + inference, not
                training). This script prints the exact, copy-pasteable command;
                you run it on a GPU box (a g6.4xlarge / L4 24GB is enough).
  3. RUN      - load your fine-tuned checkpoint back through the same Robot()
                abstraction and run it in sim (or on real hardware with one
                keyword change).

Why MolmoAct2 generates the demos: a pretrained foundation model gives you
task-meaningful trajectories to bootstrap from, so the fine-tuned policy learns a
sharpened, task-specific behavior rather than learning from random motion.

------------------------------------------------------------------------------
Requirements
------------------------------------------------------------------------------
MolmoAct2's policy class shipped in LeRobot AFTER the 0.5.1 PyPI release, so the
plain extra is not enough yet. Install LeRobot from source alongside the
molmoact2 extra (this is the documented path in pyproject.toml):

    uv pip install "strands-robots[sim-mujoco,molmoact2,mesh]" \
        "lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git"

    export STRANDS_TRUST_REMOTE_CODE=1   # MolmoAct2 loads with trust_remote_code
    export MUJOCO_GL=egl                 # headless rendering (SSH / CI / cloud GPU)

A CUDA GPU is required for every phase here (MolmoAct2 is a 16B-class VLA).

------------------------------------------------------------------------------
Quick start (one machine, end to end)
------------------------------------------------------------------------------
    # 1. Collect 50 demonstration episodes into a local dataset
    python collect_train_run_molmoact2.py collect --episodes 50

    # 2. Print the training command (run it yourself on the GPU box)
    python collect_train_run_molmoact2.py train

    # 3. Run your fine-tuned checkpoint in sim
    python collect_train_run_molmoact2.py run --checkpoint outputs/molmoact2_ft/checkpoints/last

Push the collected dataset to the Hub instead of keeping it local:

    export HF_TOKEN=hf_...
    python collect_train_run_molmoact2.py collect --episodes 50 --hf-user my_user

Repository: https://github.com/strands-labs/robots
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("collect_train_run_molmoact2")

# The pretrained checkpoint that bootstraps the demonstrations. It is a
# transformers-native MolmoAct2 checkpoint for the SO-100/101 embodiment; the
# LerobotLocal path auto-detects model_type=="molmoact2" and routes it.
TEACHER_REPO = "allenai/MolmoAct2-SO100_101"

# The robot/embodiment. Both are the sim SO-101: ROBOT spawns the MuJoCo arm,
# and the "so101" embodiment maps the sim's numeric joint keys ("1".."6") into
# observation.state and converts the sim's RADIANS to the DEGREES the MolmoAct2
# checkpoint expects. (The "so_real" embodiment is for a *physical* SO-101 whose
# state keys are "<motor>.pos"; on this sim it would leave observation.state
# empty and MolmoAct2 would fail with "requires observation.state". Switch ROBOT
# to mode="real" and EMBODIMENT to "so_real" to drive the physical arm.)
ROBOT = "so101"
EMBODIMENT = "so101"

DEFAULT_TASK = "Pick up the red cube and place it on the plate"


# ---------------------------------------------------------------------------
# Scene setup (shared by collect + run)
# ---------------------------------------------------------------------------


def _must(sim: Any, action: str, params: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a sim action and fail loud on error.

    The dispatch router returns ``{"status": "error", ...}`` for bad params
    (e.g. an unknown kwarg) rather than raising. Swallowing that result would
    let scene setup silently no-op - a camera that never gets added means the
    recorded dataset has no frames and inference sees no images. Per AGENTS.md
    ("no silent defaults on error"), surface it immediately.
    """
    result = sim._dispatch_action(action, params)
    if isinstance(result, dict) and result.get("status") == "error":
        raise RuntimeError(f"sim action {action!r} failed: {result.get('content', result)}")
    return result


def _build_scene(sim: Any) -> None:
    """Compose the pick-and-place scene: two cameras, a red cube, a plate.

    MolmoAct2-SO100_101 was trained with TWO RGB views (a top and a side
    camera). The "so101" embodiment's ``obs_rename`` maps camera source keys
    ``front`` -> ``observation.images.image`` and ``wrist`` ->
    ``observation.images.wrist_image``, so the cameras MUST be named ``front``
    and ``wrist`` here; the rename then produces the image keys the model
    expects. (Naming them ``image``/``wrist_image`` directly skips the rename
    and MolmoAct2 fails with "image_keys missing from observation".)

    Note: ``add_camera`` takes ``name`` (not ``camera_name``); the dispatch
    router rejects unknown kwargs. ``render`` is the action that takes
    ``camera_name``. Every call is routed through ``_must`` so a schema
    mismatch fails loud instead of silently producing a camera-less scene.

    Kept deterministic so the recorded dataset has a stable feature schema;
    per-episode variety comes from ``randomize`` in the collection loop.
    """
    # Top-down-ish view. Named "front" so obs_rename -> observation.images.image.
    _must(
        sim,
        "add_camera",
        {
            "name": "front",
            "position": [0.3, 0.0, 0.7],
            "target": [0.1, 0.0, 0.05],
            "width": 640,
            "height": 480,
        },
    )
    # Side view. Named "wrist" so obs_rename -> observation.images.wrist_image.
    _must(
        sim,
        "add_camera",
        {
            "name": "wrist",
            "position": [0.5, -0.4, 0.4],
            "target": [0.1, 0.0, 0.1],
            "width": 640,
            "height": 480,
        },
    )
    _must(
        sim,
        "add_object",
        {
            "name": "red_cube",
            "shape": "box",
            "size": [0.02, 0.02, 0.02],
            "position": [0.2, 0.0, 0.02],
            "color": [0.85, 0.12, 0.12, 1.0],
            "mass": 0.05,
        },
    )
    _must(
        sim,
        "add_object",
        {
            "name": "plate",
            "shape": "cylinder",
            "size": [0.06, 0.005],
            "position": [0.0, 0.22, 0.005],
            "color": [0.9, 0.9, 0.9, 1.0],
            "mass": 0.2,
            "is_static": True,
        },
    )


# ---------------------------------------------------------------------------
# Phase 1: COLLECT
# ---------------------------------------------------------------------------


def collect(args: argparse.Namespace) -> int:
    """Drive the pretrained teacher policy in sim and record a LeRobotDataset."""
    from strands_robots import Robot
    from strands_robots.policies import create_policy

    push_to_hub = bool(args.hf_user)
    repo_id = f"{args.hf_user}/{args.dataset_name}" if push_to_hub else f"local/{args.dataset_name}"

    sim = Robot(ROBOT)  # mode="sim" (default)
    log.info("Sim world created: %s", sim.tool_name)
    _build_scene(sim)

    # The pretrained teacher that generates the demonstrations.
    log.info("Loading teacher policy %s (this downloads weights on first run)...", TEACHER_REPO)
    teacher = create_policy(TEACHER_REPO, embodiment=EMBODIMENT, device=args.device)

    # One recording session; each episode appends to the same dataset.
    start = sim._dispatch_action(
        "start_recording",
        {"repo_id": repo_id, "task": args.task, "fps": args.fps, "push_to_hub": push_to_hub},
    )
    if start.get("status") == "error":
        log.error("start_recording failed: %s", start)
        return 1

    for ep in range(args.episodes):
        sim._dispatch_action("reset", {})
        # Domain randomization gives the dataset visual variety so the
        # fine-tuned policy generalizes instead of overfitting one scene.
        sim._dispatch_action("randomize", {"randomize_colors": True, "randomize_lighting": True})
        # Drive the teacher; run_policy records each frame into the open dataset.
        sim._dispatch_action(
            "run_policy",
            {
                "robot_name": ROBOT,
                "instruction": args.task,
                "policy_object": teacher,
                "n_steps": args.steps,
            },
        )
        log.info("episode %d/%d recorded", ep + 1, args.episodes)

    stop = sim._dispatch_action("stop_recording", {})
    log.info("stop_recording: %s", stop.get("content", stop))

    cache = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
    log.info("Dataset written to %s", cache)
    if push_to_hub:
        log.info("Pushed to https://huggingface.co/datasets/%s", repo_id)
    sim.destroy()
    return 0


# ---------------------------------------------------------------------------
# Phase 2: TRAIN (runs upstream in LeRobot - we print the exact command)
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> int:
    """Print the upstream LeRobot fine-tuning command.

    strands-robots does not train; training runs in LeRobot. We do not shell out
    to it automatically because it is a long, GPU-heavy job you will want to run
    and monitor yourself (a g6.4xlarge / L4 24GB is sufficient for MolmoAct2).
    """
    repo_id = f"{args.hf_user}/{args.dataset_name}" if args.hf_user else f"local/{args.dataset_name}"
    dataset_root = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id

    cmd = (
        "lerobot-train \\\n"
        "    --policy.type=molmoact2 \\\n"
        f"    --policy.pretrained_path={TEACHER_REPO} \\\n"
        "    --policy.push_to_hub=false \\\n"
        "    --policy.repo_id=local/molmoact2_ft \\\n"
        f"    --dataset.repo_id={repo_id} \\\n"
        f"    --dataset.root={dataset_root} \\\n"
        f"    --output_dir={args.output_dir} \\\n"
        "    --batch_size=1 \\\n"
        "    --steps=20000 \\\n"
        "    --save_freq=5000"
    )
    print("\n" + "=" * 72)
    print("TRAIN - run this on your GPU box (g6.4xlarge / L4 24GB is enough):")
    print("=" * 72)
    print("\n# Install LeRobot from source (MolmoAct2 ships on the molmoact2 branch,")
    print("# post-0.5.1 PyPI). MolmoAct2 is integrated INTO LeRobot as a policy, so")
    print("# fine-tuning uses LeRobot's standard trainer on a LeRobot v3 dataset:")
    print('uv pip install "strands-robots[sim-mujoco,molmoact2,mesh]" \\')
    print('    "lerobot[feetech] @ git+https://github.com/huggingface/lerobot.git"\n')
    print("# Fine-tune MolmoAct2 on your collected dataset:")
    print(cmd)
    print("\n# NOTE: the exact flag names are defined by LeRobot's MolmoAct2 policy")
    print("# integration. Confirm against the authoritative doc in the LeRobot")
    print("# source tree: docs/source/molmoact2.mdx (and `lerobot-train --help`).")
    print("# --policy.repo_id + --policy.push_to_hub=false are required by the")
    print("# trainer's config validation even for a purely local run.")
    print("# Fits a 24GB L4 (g6.4xlarge) in bfloat16; float32 needs ~24-26GB.")
    print("=" * 72 + "\n")
    return 0


# ---------------------------------------------------------------------------
# Phase 3: RUN (load the fine-tuned checkpoint back through Robot())
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Run a fine-tuned checkpoint in sim through the same Robot() abstraction."""
    from strands_robots import Robot
    from strands_robots.policies import create_policy

    if not args.checkpoint:
        raise SystemExit("run requires --checkpoint <path-or-hf-repo> (your fine-tuned policy)")

    sim = Robot(ROBOT)  # swap to mode="real" + cameras={...} to deploy to hardware
    log.info("Sim world created: %s", sim.tool_name)
    _build_scene(sim)

    log.info("Loading fine-tuned policy from %s", args.checkpoint)
    policy = create_policy(args.checkpoint, embodiment=EMBODIMENT, device=args.device)
    policy.reset()

    async def rollout() -> None:
        period = 1.0 / args.hz
        for step in range(args.steps):
            obs = sim._dispatch_action("get_observation", {"robot_name": ROBOT})
            t = asyncio.get_event_loop().time()
            actions = await policy.get_actions(obs, args.task)
            dt = asyncio.get_event_loop().time() - t
            a = actions[0]
            log.info("step %d infer=%.2fs", step, dt)
            sim._dispatch_action("set_joint_positions", {"positions": a})
            await asyncio.sleep(max(0.0, period - dt))

    try:
        asyncio.run(rollout())
        # "front" is one of the cameras _build_scene adds (see the obs_rename note there).
        sim._dispatch_action("render", {"camera_name": "front"})
    finally:
        sim.destroy()
        log.info("Done.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="collect_train_run_molmoact2",
        description="Collect, train, and run a MolmoAct2 policy with Strands Robots + LeRobot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="phase", required=True)

    common_data = argparse.ArgumentParser(add_help=False)
    common_data.add_argument("--dataset-name", default="molmoact2-pickplace", help="Dataset slug.")
    common_data.add_argument("--hf-user", default=None, help="HF username; if unset, dataset stays local.")
    common_data.add_argument("--task", default=DEFAULT_TASK, help="Natural-language task.")
    common_data.add_argument("--device", default="cuda", help="Torch device (cuda/cpu).")

    c = sub.add_parser("collect", parents=[common_data], help="Record demonstrations with the teacher policy.")
    c.add_argument("--episodes", type=int, default=50, help="Number of episodes to record.")
    c.add_argument("--steps", type=int, default=200, help="Steps per episode.")
    c.add_argument("--fps", type=int, default=30, help="Recording FPS.")

    t = sub.add_parser("train", parents=[common_data], help="Print the upstream LeRobot training command.")
    t.add_argument("--output-dir", default="outputs/molmoact2_ft", help="Training output dir.")

    r = sub.add_parser("run", parents=[common_data], help="Run a fine-tuned checkpoint in sim.")
    r.add_argument("--checkpoint", required=True, help="Path or HF repo of your fine-tuned policy.")
    r.add_argument("--steps", type=int, default=200, help="Rollout steps.")
    r.add_argument("--hz", type=float, default=5.0, help="Control loop frequency.")

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.phase == "collect":
        return collect(args)
    if args.phase == "train":
        return train(args)
    if args.phase == "run":
        return run(args)
    raise SystemExit(f"Unknown phase: {args.phase!r}")


if __name__ == "__main__":
    raise SystemExit(main())
