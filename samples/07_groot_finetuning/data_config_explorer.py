#!/usr/bin/env python3
"""
Sample 07 — GR00T Data Config Explorer

Explore all 21 embodiment data configurations that GR00T N1.6 supports.
Each config maps a robot's sensors (cameras, joints) to GR00T's input format.

This script requires NO GPU — it only inspects the configuration registry.

Usage:
    python samples/07_groot_finetuning/data_config_explorer.py
    python samples/07_groot_finetuning/data_config_explorer.py --config so100_dualcam
    python samples/07_groot_finetuning/data_config_explorer.py --filter unitree
"""

import argparse
import sys


def list_all_configs() -> None:
    """List all registered data configurations with their properties."""
    from strands_robots.policies.groot.data_config import DATA_CONFIG_MAP

    print(f"📋 GR00T Data Configurations ({len(DATA_CONFIG_MAP)} total)")
    print("=" * 80)

    # Group by robot family
    families = {}
    for name, config in DATA_CONFIG_MAP.items():
        # Determine family from name prefix
        if name.startswith("so100") or name.startswith("so101"):
            family = "SO-100/101 (Low-cost arms)"
        elif name.startswith("fourier"):
            family = "Fourier GR-1 (Humanoid)"
        elif name.startswith("unitree"):
            family = "Unitree G1 (Humanoid)"
        elif "panda" in name:
            family = "Franka Panda (Research arm)"
        elif name.startswith("oxe"):
            family = "Open X-Embodiment"
        elif name.startswith("libero"):
            family = "LIBERO (Simulation)"
        elif name.startswith("agibot"):
            family = "AgiBOT"
        elif name.startswith("galaxea"):
            family = "Galaxea"
        else:
            family = "Other"

        if family not in families:
            families[family] = []
        families[family].append((name, config))

    for family, configs in families.items():
        print(f"\n  🤖 {family}")
        print(f"  {'─' * 76}")

        for name, config in configs:
            cameras = ", ".join(config.video_keys)
            states = ", ".join(config.state_keys)
            actions = ", ".join(config.action_keys)
            n_obs = len(config.observation_indices)
            n_act = len(config.action_indices)

            print(f"\n    📌 {name}")
            print(f"       Cameras ({len(config.video_keys)}):  {cameras}")
            print(f"       State ({len(config.state_keys)}):    {states}")
            print(f"       Actions ({len(config.action_keys)}): {actions}")
            print(f"       Obs indices:  {config.observation_indices[:5]}{'...' if n_obs > 5 else ''} ({n_obs} total)")
            print(f"       Act indices:  {config.action_indices[:5]}{'...' if n_act > 5 else ''} ({n_act} total)")


def inspect_single_config(config_name: str) -> None:
    """Deep-dive into a single data configuration."""
    from strands_robots.policies.groot.data_config import load_data_config

    try:
        config = load_data_config(config_name)
    except ValueError as e:
        print(f"❌ {e}")
        from strands_robots.policies.groot.data_config import DATA_CONFIG_MAP
        print(f"\nAvailable configs: {', '.join(sorted(DATA_CONFIG_MAP.keys()))}")
        sys.exit(1)

    print(f"🔍 Deep Dive: {config_name}")
    print("=" * 60)
    print(f"  Type: {type(config).__name__}")

    # Video modality
    print("\n  📷 Video (cameras):")
    for key in config.video_keys:
        camera_name = key.replace("video.", "")
        print(f"     {key:<30} → GR00T expects 224×224 RGB")
        print(f"     {'':30}   Camera name: '{camera_name}'")

    # State modality
    print("\n  🦾 State (proprioception):")
    for key in config.state_keys:
        state_name = key.replace("state.", "")
        # Estimate dimension from name
        if "gripper" in state_name:
            dim, desc = 1, "gripper opening (0=closed, 1=open)"
        elif "single_arm" in state_name:
            dim, desc = 5, "5-DOF arm joint angles (radians)"
        elif "arm" in state_name:
            dim, desc = 7, "7-DOF arm joint angles (radians)"
        elif "waist" in state_name:
            dim, desc = 3, "waist yaw/pitch/roll (radians)"
        elif "head" in state_name:
            dim, desc = 2, "head pan/tilt (radians)"
        elif "leg" in state_name:
            dim, desc = 6, "6-DOF leg joints (radians)"
        else:
            dim, desc = "?", "joint positions"
        print(f"     {key:<30} → dim={dim}, {desc}")

    # Action modality
    print("\n  🎯 Action (output):")
    for key in config.action_keys:
        action_name = key.replace("action.", "")
        if "gripper" in action_name:
            dim, desc = 1, "gripper target"
        elif "single_arm" in action_name:
            dim, desc = 5, "5-DOF arm target positions"
        elif "arm" in action_name:
            dim, desc = 7, "7-DOF arm target positions"
        elif "waist" in action_name:
            dim, desc = 3, "waist target"
        elif "head" in action_name:
            dim, desc = 2, "head target"
        elif "leg" in action_name:
            dim, desc = 6, "leg target"
        else:
            dim, desc = "?", "joint targets"
        print(f"     {key:<30} → dim={dim}, {desc}")

    # Language modality
    print("\n  💬 Language:")
    for key in config.language_keys:
        print(f"     {key}")

    # Temporal indices
    print("\n  ⏱️  Temporal Configuration:")
    print(f"     Observation indices: {config.observation_indices}")
    print(f"     Action indices:      {config.action_indices}")
    print(f"     Action horizon:      {len(config.action_indices)} steps")

    # Modality config (for GR00T internal use)
    print("\n  📦 Modality Config (GR00T format):")
    mc = config.modality_config()
    for modality_name, modality_cfg in mc.items():
        print(f"     {modality_name}:")
        print(f"       keys:    {modality_cfg.modality_keys}")
        print(f"       indices: {modality_cfg.delta_indices}")


def show_custom_config_example() -> None:
    """Show how to create a custom data config for a new robot."""
    print("\n🛠️  Creating a Custom Data Config")
    print("=" * 60)
    print("""
  from strands_robots.policies.groot.data_config import (
      create_custom_data_config,
      DATA_CONFIG_MAP,
  )

  # Register a config for your custom robot
  my_config = create_custom_data_config(
      name="my_robot_3cam",
      video_keys=["video.front", "video.wrist_left", "video.wrist_right"],
      state_keys=["state.left_arm", "state.right_arm", "state.torso"],
      action_keys=["action.left_arm", "action.right_arm", "action.torso"],
      language_keys=["annotation.human.task_description"],
      observation_indices=[0],       # Current frame only
      action_indices=list(range(16)),  # 16-step action horizon
  )

  # Now usable everywhere:
  assert "my_robot_3cam" in DATA_CONFIG_MAP

  # Use with GR00T policy:
  from strands_robots.policies.groot import Gr00tPolicy
  policy = Gr00tPolicy(
      data_config="my_robot_3cam",
      model_path="nvidia/GR00T-N1-2B",
  )
""")


def main():
    parser = argparse.ArgumentParser(
        description="Explore GR00T embodiment data configurations",
    )
    parser.add_argument("--config", default=None, help="Inspect a specific config by name")
    parser.add_argument("--filter", default=None, help="Filter configs by substring")
    parser.add_argument("--custom-example", action="store_true", help="Show custom config creation example")
    args = parser.parse_args()

    print("🧠 GR00T N1.6 — Data Config Explorer")
    print("=" * 60)

    if args.custom_example:
        show_custom_config_example()
        return

    if args.config:
        inspect_single_config(args.config)
    elif args.filter:
        from strands_robots.policies.groot.data_config import DATA_CONFIG_MAP

        matching = {k: v for k, v in DATA_CONFIG_MAP.items() if args.filter.lower() in k.lower()}
        if not matching:
            print(f"❌ No configs matching '{args.filter}'")
            print(f"   Available: {', '.join(sorted(DATA_CONFIG_MAP.keys()))}")
            sys.exit(1)

        print(f"Found {len(matching)} configs matching '{args.filter}':\n")
        for name in sorted(matching):
            inspect_single_config(name)
            print()
    else:
        list_all_configs()
        print("\n" + "─" * 60)
        print("💡 Tip: Use --config <name> for a deep-dive on any config")
        print("        Use --custom-example to learn how to add your own robot")


if __name__ == "__main__":
    main()
