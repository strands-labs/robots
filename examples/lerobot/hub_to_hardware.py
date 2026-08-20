"""From Hugging Face Hub to robot hardware with Strands Agents and LeRobot.

A runnable end-to-end example that mirrors the blog post of the same name.
The script:

  1. Builds a Strands agent (Claude Opus 4.8 on Bedrock by default) with
     the LeRobot AgentTools and a Robot('so100') in simulation or hardware.
  2. Records a demonstration as a LeRobotDataset (locally by default; pushes
     to the Hub when --hf-user is set and HF_TOKEN is exported with write scope).
  3. Runs a policy on the same robot. The Mock policy is the default so the
     script runs end-to-end without Docker, a GPU, or a Hub checkpoint.
  4. Optionally deploys the same agent code to a physical SO-101 with --mode real.
  5. Broadcasts a command to every peer on the local Zenoh mesh.
  6. Cleans up any long-running resources the workflow started.

Quick start (no hardware, no GPU, no Hub credentials needed):

    # Dev/lab mesh posture
    export STRANDS_MESH_LOCAL_DEV=1

    python hub_to_hardware.py

(For sim-only runs you can disable the mesh entirely with STRANDS_MESH=false.)

Note on Step 5: by default the robot_mesh tool routes every physically-actuating
action -- emergency_stop, broadcast, tell, send, stop, rpc -- through a
human-in-the-loop approval interrupt (set STRANDS_MESH_HITL_ACTIONS to "all",
"none", or a comma-separated subset to tune this). The first time the agent
invokes a broadcast, you'll see a "robot_mesh-broadcast-approval" prompt in the
terminal. Type "y" (or "yes" / "approve") to authorize. To skip Step 5 entirely,
pass --skip-mesh.

Push the recorded dataset to the Hub (requires HF_TOKEN with write scope):

    export HF_TOKEN=hf_...
    python hub_to_hardware.py --hf-user my_user

Override the LLM (verify exact Bedrock ID in your AWS console):

    python hub_to_hardware.py --model-id global.anthropic.claude-sonnet-4-6

The AWS region resolves from your AWS environment (AWS_REGION /
AWS_DEFAULT_REGION env vars, ~/.aws/config, or instance metadata). To
override per-run, pass --aws-region <region>.

Run with the GR00T container as the policy (requires Docker + NVIDIA GPU):

    python hub_to_hardware.py \\
        --policy groot \\
        --checkpoint nvidia/GR00T-N1.7-LIBERO

Run on physical hardware (assumes SO-101 already calibrated via lerobot calibrate):

    python hub_to_hardware.py \\
        --mode real \\
        --port /dev/ttyACM0 \\
        --leader-port /dev/ttyACM1

A note on recording shape: this example records ONE demonstration per run
(a single LeRobotDataset episode of N steps). The dataset format also
supports multi-episode shapes, but a single longer episode keeps the
agent-driven story honest - you tell the agent in English to record a
demonstration once, and the tool sequence comes out in one shot.
Production multi-episode collection wraps the loop in Python; that
pattern lives in this folder's README under "Production patterns."

Repository: https://github.com/strands-labs/robots
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("hub_to_hardware")


# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
# Claude Opus 4.8 on Bedrock orchestrates the LeRobot tool surface cleanly
# in one shot - lower-tier models work but issue more defensive state-
# querying calls and are more likely to drift on multi-step loops.
#
# IMPORTANT: verify the exact model ID in your AWS Bedrock console
# (Model catalog → Anthropic → Claude Opus 4.8). Cross-region inference
# profile IDs are prefixed by ``us.``, ``eu.``, etc. - pick the one for
# your region. Override at runtime via --model-id or STRANDS_BEDROCK_MODEL_ID
# without editing this file.
DEFAULT_MODEL_ID = "global.anthropic.claude-opus-4-8"  # ← verify in AWS console
# Region is intentionally not defaulted in code. It resolves from the
# --aws-region CLI flag, then AWS_REGION / AWS_DEFAULT_REGION env vars,
# then boto3's standard chain (~/.aws/config, instance metadata).


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


def _build_bedrock_model(model_id: str, region: str | None) -> Any | None:
    """Construct a Strands BedrockModel client.

    Returns the model on success, None on any failure (import error, auth
    error, model-not-enabled). The caller falls back to Strands' default
    model on None - the workflow still runs, just on whatever Strands picks.

    ``region`` may be None, in which case boto3's standard resolution chain
    (env vars, ~/.aws/config, instance metadata) decides the region.
    """
    try:
        from strands.models import BedrockModel
    except ImportError:
        logger.warning(
            "strands.models.BedrockModel not importable. Falling back to "
            "Strands' default model. Upgrade strands-agents to pin Bedrock models."
        )
        return None

    try:
        kwargs: dict[str, Any] = {"model_id": model_id}
        if region:
            kwargs["region_name"] = region
        model = BedrockModel(**kwargs)
        logger.info(
            "Using Bedrock model: %s (region %s)",
            model_id,
            region or "<resolved from AWS environment>",
        )
        return model
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "BedrockModel(%s, region=%s) init failed: %s. Falling back to "
            "Strands' default. Common causes: model not enabled in this AWS "
            "account, wrong region, or stale model ID - check the Bedrock console.",
            model_id,
            region or "<unset>",
            exc,
        )
        return None


def build_agent(
    *,
    mode: str = "sim",
    policy: str = "mock",
    port: str | None = None,
    leader_port: str | None = None,
    model_id: str | None = None,
    aws_region: str | None = None,
) -> Any:
    """Build the Strands agent with the right tool set for the run."""
    from strands import Agent

    from strands_robots import (
        Robot,
        robot_mesh,
    )

    robot_kwargs: dict[str, Any] = {"data_config": "so100_dualcam"}
    if mode == "real":
        if not port:
            raise SystemExit("--mode real requires --port (e.g. /dev/ttyACM0). Hardware paths can't be guessed safely.")
        robot_kwargs.update(
            port=port,
            cameras={
                "front": {
                    "type": "opencv",
                    "index_or_path": "/dev/video0",
                    "fps": 30,
                },
                "wrist": {
                    "type": "opencv",
                    "index_or_path": "/dev/video2",
                    "fps": 30,
                },
            },
        )

    robot = Robot("so100", mode=mode, **robot_kwargs)
    tools: list[Any] = [robot, robot_mesh]

    if policy == "groot":
        from strands_robots import gr00t_inference

        tools.append(gr00t_inference)

    resolved_model_id = model_id or os.environ.get("STRANDS_BEDROCK_MODEL_ID") or DEFAULT_MODEL_ID
    resolved_region = aws_region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    model = _build_bedrock_model(resolved_model_id, resolved_region)

    agent = Agent(model=model, tools=tools) if model else Agent(tools=tools)
    agent._leader_port = leader_port  # type: ignore[attr-defined]
    # Stash the robot so cleanup() can tear down its sim world + mesh session
    # on exit. Both Simulation and HardwareRobot expose cleanup(); without it
    # the Zenoh mesh listener and the MuJoCo executor keep non-daemon threads
    # alive and the process never exits after the workflow finishes.
    agent._robot = robot  # type: ignore[attr-defined]
    return agent


# ---------------------------------------------------------------------------
# Step 2: Record a demonstration
# ---------------------------------------------------------------------------


def record_demonstration(
    agent: Any,
    *,
    mode: str,
    repo_id: str,
    num_steps: int,
    task: str,
    push_to_hub: bool,
) -> Any:
    """Record one demonstration as a LeRobotDataset.

    Acknowledges that the Robot() factory already initialised the world,
    so the agent doesn't waste calls destroying and recreating it. One
    prompt → one tool sequence → one episode.
    """
    push_clause = (
        f"Push the result to {repo_id} when done."
        if push_to_hub
        else "Keep the dataset local - do not push to the Hub."
    )

    if mode == "sim":
        prompt = (
            f"A simulation world is already set up with the so100 robot "
            f"(initialised by the Robot() factory). Compose the scene and "
            f"record one demonstration:\n"
            f"\n"
            f"Scene:\n"
            f"  - Add a small red cube near the robot (about 3cm on a side).\n"
            f"  - Add a front-facing camera named 'front' looking at the cube.\n"
            f"\n"
            f"Recording:\n"
            f"  - Start recording to repo_id={repo_id!r} at FPS 30 with "
            f"task={task!r}.\n"
            f"  - Run the Mock policy for {num_steps} steps with "
            f"instruction={task!r}.\n"
            f"  - Call stop_recording to finalize the episode. {push_clause}"
        )
    else:
        leader_port = getattr(agent, "_leader_port", None)
        if not leader_port:
            raise SystemExit("Hardware recording requires --leader-port (e.g. /dev/ttyACM1)")
        prompt = (
            f"Record one demonstration of '{task}' on the SO-101, "
            f"with the leader on {leader_port}. Write the dataset "
            f"to {repo_id} at FPS 30. {push_clause}"
        )

    return agent(prompt)


# ---------------------------------------------------------------------------
# Step 3: Run a policy on the robot
# ---------------------------------------------------------------------------


def run_policy(
    agent: Any,
    *,
    policy: str,
    checkpoint: str | None,
    instruction: str,
) -> Any:
    """Run a policy on the robot the agent already has bound."""
    if policy == "mock":
        prompt = (
            f"Run the Mock policy on the robot for 200 steps with the "
            f"instruction '{instruction}'. Render the final frame."
        )

    elif policy == "groot":
        if not checkpoint:
            raise SystemExit("--policy groot requires --checkpoint <hf_repo>, e.g. nvidia/GR00T-N1.7-LIBERO")
        prompt = (
            f"Use gr00t_inference lifecycle='full' to bring up the GR00T "
            f"container on port 5555 with checkpoint {checkpoint}. Then "
            f"run the policy on the robot with the instruction "
            f"'{instruction}' for 200 steps. Render the final frame."
        )

    elif policy == "lerobot_local":
        if not checkpoint:
            raise SystemExit(
                "--policy lerobot_local requires --checkpoint <hf_repo>, e.g. lerobot/act_aloha_sim_transfer_cube_human"
            )
        if os.environ.get("STRANDS_TRUST_REMOTE_CODE") != "1":
            raise SystemExit(
                "lerobot_local requires STRANDS_TRUST_REMOTE_CODE=1 to opt "
                "in to trust_remote_code loading. Re-run with that env var."
            )
        # MolmoAct2 checkpoints (e.g. allenai/MolmoAct2-SO100_101) also run
        # through this path: LerobotLocalPolicy auto-detects model_type ==
        # "molmoact2" from the checkpoint's config.json and routes to the
        # dedicated transformers-native loader. No separate provider needed.
        prompt = (
            f"Run the LerobotLocal policy '{checkpoint}' on the robot with "
            f"the instruction '{instruction}' for 200 steps. Render the "
            f"final frame."
        )

    else:
        raise SystemExit(f"Unknown policy: {policy!r}")

    return agent(prompt)


# ---------------------------------------------------------------------------
# Step 5: Mesh broadcast
# ---------------------------------------------------------------------------


def broadcast_to_mesh(agent: Any, instruction: str = "go to home pose") -> Any:
    """Discover mesh peers and broadcast a command to all of them."""
    prompt = (
        f"Use robot_mesh to list every peer and local robot, then "
        f"broadcast the instruction '{instruction}' to each one with a "
        f"5-second timeout."
    )
    return agent(prompt)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def cleanup(agent: Any, *, policy: str) -> None:
    """Tear down any long-running resources the workflow started."""
    if policy == "groot":
        try:
            agent("Stop the GR00T inference service on port 5555.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("GR00T stop reported: %s", exc)

    # Release the robot's sim world / hardware connection and its Zenoh mesh
    # session. Both Simulation and HardwareRobot expose cleanup(); these own
    # non-daemon threads (the mesh listener, the MuJoCo executor) that would
    # otherwise keep the interpreter alive and make the script appear to hang
    # after "Workflow finished."
    robot = getattr(agent, "_robot", None)
    if robot is not None and hasattr(robot, "cleanup"):
        try:
            robot.cleanup()
        except Exception as exc:  # noqa: BLE001 - best-effort teardown
            logger.warning("Robot cleanup reported: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="hub_to_hardware",
        description="From Hugging Face Hub to robot hardware - the runnable example.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--mode",
        choices=("sim", "real"),
        default="sim",
        help="Robot execution mode (default: sim, no hardware required).",
    )
    p.add_argument(
        "--policy",
        choices=("mock", "groot", "lerobot_local"),
        default="mock",
        help="Policy provider (default: mock, no GPU required).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="HF repo for the policy checkpoint (required for --policy groot or lerobot_local).",
    )

    # LLM knobs
    p.add_argument(
        "--model-id",
        default=None,
        help=f"Bedrock model ID to drive the agent. Default: {DEFAULT_MODEL_ID}. "
        f"Override here or via STRANDS_BEDROCK_MODEL_ID. Verify the exact "
        f"ID against your AWS Bedrock console.",
    )
    p.add_argument(
        "--aws-region",
        default=None,
        help="AWS region for Bedrock. If unset, resolves from AWS_REGION / "
        "AWS_DEFAULT_REGION env vars or ~/.aws/config (boto3's standard chain).",
    )

    # Hardware knobs
    p.add_argument(
        "--port",
        default=None,
        help="USB device of the SO-101 follower (--mode real only).",
    )
    p.add_argument(
        "--leader-port",
        default=None,
        help="USB device of the SO-101 leader arm (--mode real only).",
    )

    # Recording knobs
    p.add_argument(
        "--hf-user",
        default=None,
        help="HF username for the dataset repo. If unset, the dataset stays local.",
    )
    p.add_argument(
        "--dataset-name",
        default="strands-cube-pick",
        help="Dataset name under <hf-user>. Default: strands-cube-pick.",
    )
    p.add_argument(
        "--num-steps",
        type=int,
        default=1000,
        help="Number of policy steps to record in the demonstration (default: 1000, ≈ 33s of data at 30fps).",
    )
    p.add_argument(
        "--instruction",
        default="pick up the red cube",
        help="Natural-language task instruction.",
    )
    p.add_argument(
        "--clean-cache",
        action="store_true",
        help="Delete the local LeRobotDataset cache for this repo before recording.",
    )

    # Skip flags
    p.add_argument(
        "--skip-record",
        action="store_true",
        help="Skip the recording step (Step 2).",
    )
    p.add_argument(
        "--skip-mesh",
        action="store_true",
        help="Skip the mesh broadcast step (Step 5).",
    )

    return p.parse_args(argv)


def banner(title: str) -> None:
    bar = "=" * 60
    logger.info(bar)
    logger.info(title)
    logger.info(bar)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    push_to_hub = bool(args.hf_user)
    repo_id = f"{args.hf_user}/{args.dataset_name}" if push_to_hub else f"local/{args.dataset_name}"

    if args.clean_cache:
        import shutil

        cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
        if cache_dir.exists():
            logger.info("Removing cache at %s", cache_dir)
            shutil.rmtree(cache_dir)

    logger.info(
        "Starting workflow (mode=%s, policy=%s, push_to_hub=%s, repo=%s, num_steps=%d)",
        args.mode,
        args.policy,
        push_to_hub,
        repo_id,
        args.num_steps,
    )

    banner("Step 1: Build the agent")
    agent = build_agent(
        mode=args.mode,
        policy=args.policy,
        port=args.port,
        leader_port=args.leader_port,
        model_id=args.model_id,
        aws_region=args.aws_region,
    )

    try:
        if not args.skip_record:
            banner("Step 2: Record a demonstration")
            record_demonstration(
                agent,
                mode=args.mode,
                repo_id=repo_id,
                num_steps=args.num_steps,
                task=args.instruction,
                push_to_hub=push_to_hub,
            )
        else:
            logger.info("Skipping Step 2 (--skip-record)")

        banner("Step 3/4: Run a policy on the robot")
        run_policy(
            agent,
            policy=args.policy,
            checkpoint=args.checkpoint,
            instruction=args.instruction,
        )

        if not args.skip_mesh:
            banner("Step 5: Mesh broadcast")
            broadcast_to_mesh(agent)
        else:
            logger.info("Skipping Step 5 (--skip-mesh)")

    finally:
        cleanup(agent, policy=args.policy)

    logger.info("Workflow finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
