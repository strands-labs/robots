#!/usr/bin/env python3
"""
Sample 07 — GR00T N1.6 Inference Demo

Load GR00T N1.6 and run inference in MuJoCo simulation.
Demonstrates all 3 modes: service (ZMQ), local N1.5, local N1.6.

Requirements:
    pip install strands-robots[vla] isaac-gr00t
    GPU with 24GB+ VRAM

Usage:
    # Local N1.6 inference (recommended)
    python samples/07_groot_finetuning/inference_demo.py

    # Service mode (connect to running GR00T Docker)
    python samples/07_groot_finetuning/inference_demo.py --mode service --host localhost --port 5555

    # Custom checkpoint
    python samples/07_groot_finetuning/inference_demo.py --model /data/checkpoints/my_groot/best

    # Different robot
    python samples/07_groot_finetuning/inference_demo.py --data-config unitree_g1 --robot unitree_g1
"""

import argparse
import logging
import sys
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_groot_policy(args):
    """Create a GR00T policy in the requested mode.

    Three modes supported by strands_robots.policies.groot.Gr00tPolicy:
      1. service: Connect to running ZMQ inference server (no GPU needed locally)
      2. local_n15: Load N1.5 checkpoint directly on GPU
      3. local_n16: Load N1.6 model from HuggingFace (default)
    """
    from strands_robots.policies.groot import Gr00tPolicy

    if args.mode == "service":
        # Mode 1: Service mode — connects to running GR00T Docker/server
        print(f"🔌 Connecting to GR00T service at {args.host}:{args.port}")
        policy = Gr00tPolicy(
            data_config=args.data_config,
            host=args.host,
            port=args.port,
        )

    elif args.mode == "local_n15":
        # Mode 2: Local N1.5 — load from checkpoint directory
        print(f"📦 Loading GR00T N1.5 from {args.model}")
        policy = Gr00tPolicy(
            data_config=args.data_config,
            model_path=args.model,
            embodiment_tag=args.embodiment_tag,
            groot_version="n1.5",
        )

    else:
        # Mode 3: Local N1.6 — load from HuggingFace (default)
        print(f"🧠 Loading GR00T N1.6 from {args.model}")
        policy = Gr00tPolicy(
            data_config=args.data_config,
            model_path=args.model,
            embodiment_tag=args.embodiment_tag,
            groot_version="n1.6",
            denoising_steps=args.denoising_steps,
            device=args.device,
        )

    return policy


def create_mock_observation(data_config_name: str) -> dict:
    """Create a mock observation matching the data config's expected format.

    In a real pipeline, these come from camera images and robot joint encoders.
    Here we create synthetic data to demonstrate the observation format.
    """
    from strands_robots.policies.groot.data_config import load_data_config

    config = load_data_config(data_config_name)

    obs = {}

    # Camera observations: 224×224 RGB images (GR00T's expected resolution)
    for video_key in config.video_keys:
        camera_name = video_key.replace("video.", "")
        obs[camera_name] = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        print(f"  📷 {camera_name}: shape={obs[camera_name].shape}")

    # State observations: joint angles (radians)
    for state_key in config.state_keys:
        state_name = state_key.replace("state.", "")
        # Determine state dimension from the key name
        if "gripper" in state_name:
            dim = 1
        elif "single_arm" in state_name:
            dim = 5
        elif "arm" in state_name:
            dim = 7
        elif "waist" in state_name:
            dim = 3
        elif "head" in state_name:
            dim = 2
        elif "leg" in state_name:
            dim = 6
        else:
            dim = 6  # Default
        obs[state_key] = np.random.uniform(-np.pi, np.pi, dim).astype(np.float64)
        print(f"  🦾 {state_key}: dim={dim}")

    return obs


def run_inference_demo(policy, data_config_name: str, instruction: str, num_steps: int = 5):
    """Run GR00T inference and display action outputs.

    Shows how GR00T takes (image, state, language) → action chunk.
    """
    print(f"\n🎯 Task: \"{instruction}\"")
    print(f"📊 Running {num_steps} inference steps...\n")

    # Tell the policy which robot state keys to map actions back to
    from strands_robots.policies.groot.data_config import load_data_config
    config = load_data_config(data_config_name)

    # Build robot state key list (flattened from state config keys)
    robot_state_keys = []
    for state_key in config.state_keys:
        state_name = state_key.replace("state.", "")
        if "gripper" in state_name:
            robot_state_keys.append(f"{state_name}_0")
        elif "single_arm" in state_name:
            robot_state_keys.extend([f"{state_name}_{i}" for i in range(5)])
        elif "arm" in state_name:
            robot_state_keys.extend([f"{state_name}_{i}" for i in range(7)])
        else:
            robot_state_keys.extend([f"{state_name}_{i}" for i in range(6)])

    policy.set_robot_state_keys(robot_state_keys)

    import asyncio
    loop = asyncio.new_event_loop()

    latencies = []
    for step in range(num_steps):
        # Create observation
        obs = create_mock_observation(data_config_name)

        # Run inference
        t0 = time.perf_counter()
        actions = loop.run_until_complete(
            policy.get_actions(obs, instruction)
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        latencies.append(latency_ms)

        # Display results
        print(f"  Step {step + 1}: {len(actions)} action(s), latency={latency_ms:.1f}ms")
        if actions:
            first_action = actions[0]
            action_values = list(first_action.values())[:6]  # Show first 6 values
            formatted = [f"{v:+.4f}" for v in action_values]
            print(f"         First action: [{', '.join(formatted)}, ...]")

    loop.close()

    # Summary
    print("\n📈 Inference Summary:")
    print(f"   Actions per call:  {len(actions)} (action horizon)")
    print(f"   Mean latency:      {np.mean(latencies):.1f} ms")
    print(f"   Median latency:    {np.median(latencies):.1f} ms")
    print(f"   Min/Max:           {np.min(latencies):.1f} / {np.max(latencies):.1f} ms")
    freq = 1000.0 / np.mean(latencies) if np.mean(latencies) > 0 else 0
    print(f"   Inference freq:    {freq:.1f} Hz")


def main():
    parser = argparse.ArgumentParser(
        description="GR00T N1.6 inference demo in simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: local N1.6 with SO-100
  python inference_demo.py

  # Service mode
  python inference_demo.py --mode service --host localhost --port 5555

  # Different robot
  python inference_demo.py --data-config unitree_g1 --robot unitree_g1

  # Custom checkpoint
  python inference_demo.py --model /data/checkpoints/my_groot/best
""",
    )
    parser.add_argument(
        "--mode", choices=["local_n16", "local_n15", "service"],
        default="local_n16",
        help="Inference mode (default: local_n16)",
    )
    parser.add_argument(
        "--model", default="nvidia/GR00T-N1-2B",
        help="Model path or HuggingFace ID",
    )
    parser.add_argument(
        "--data-config", default="so100_dualcam",
        help="Embodiment data config name (default: so100_dualcam)",
    )
    parser.add_argument(
        "--embodiment-tag", default="new_embodiment",
        help="Embodiment tag for the model",
    )
    parser.add_argument(
        "--instruction", default="pick up the red cube and place it on the plate",
        help="Language instruction for the task",
    )
    parser.add_argument("--host", default="localhost", help="ZMQ host (service mode)")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port (service mode)")
    parser.add_argument("--denoising-steps", type=int, default=4, help="Diffusion denoising steps")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--steps", type=int, default=5, help="Number of inference steps to run")
    args = parser.parse_args()

    print("🧠 GR00T N1.6 — Inference Demo")
    print("=" * 60)
    print(f"  Mode:        {args.mode}")
    print(f"  Model:       {args.model}")
    print(f"  Data config: {args.data_config}")
    print(f"  Device:      {args.device}")
    print()

    # Show observation format before loading model
    print("📋 Observation Format:")
    create_mock_observation(args.data_config)  # Display format (side effect)
    print()

    # Create policy
    try:
        policy = create_groot_policy(args)
    except ImportError as e:
        print(f"\n❌ Cannot create GR00T policy: {e}")
        print("\nTo install Isaac-GR00T:")
        print("  pip install isaac-gr00t")
        print("\nOr use service mode (no local GPU needed):")
        print("  python inference_demo.py --mode service --host <groot_server>")
        sys.exit(1)

    # Run inference
    run_inference_demo(
        policy,
        data_config_name=args.data_config,
        instruction=args.instruction,
        num_steps=args.steps,
    )

    print("\n✅ Demo complete!")
    print("   Next: python samples/07_groot_finetuning/finetune_groot.py")


if __name__ == "__main__":
    main()
