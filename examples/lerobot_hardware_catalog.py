#!/usr/bin/env python3
"""End-to-end: every LeRobot robot is reachable through ``strands_robots.Robot``.

``strands_robots`` is a thin natural-language + policy layer over LeRobot's
hardware drivers. This example walks the strands registry and shows that every
robot with hardware support resolves a canonical name and a LeRobot
``robot_type`` - the mapping ``Robot(name, mode="real")`` uses to construct the
underlying LeRobot driver. It then builds a real Unitree G1 in simulation (no
hardware, no GPU) to prove the same factory drives the catalog's most complex
robot.

Everything here goes through the ``strands_robots`` public API - the registry
read helpers and the ``Robot()`` factory - never ``import lerobot`` directly.
That is the whole point: users program against ``Robot("g1")``, not the driver.

Run:
    python examples/lerobot_hardware_catalog.py        # hardware catalog
    python examples/lerobot_hardware_catalog.py --g1   # focus: Unitree G1 in sim
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")


def show_catalog() -> int:
    """List every strands robot with hardware support + its LeRobot type."""
    from strands_robots.registry import format_robot_table, get_hardware_type, list_robots, resolve_name

    hw_robots = list_robots(mode="real")
    print(f"strands_robots exposes {len(hw_robots)} robot(s) with LeRobot hardware support.")
    print("Each maps a friendly name -> canonical name -> LeRobot robot_type:\n")

    header = f"{'name':<16} {'canonical':<16} {'lerobot_type':<24} category"
    print(header)
    print("-" * len(header))
    for entry in hw_robots:
        name = entry["name"]
        canonical = resolve_name(name)
        lerobot_type = get_hardware_type(canonical) or "?"
        print(f"{name:<16} {canonical:<16} {lerobot_type:<24} {entry.get('category', '')}")

    print("\nDrive any of them for real with, e.g.:")
    print("    from strands_robots import Robot")
    print("    arm = Robot('so100', mode='real', port='/dev/ttyACM0')")
    print("    arm('pick up the red cube', policy_port=8080)")
    print("\nFull registry (sim + real):\n")
    print(format_robot_table())
    return 0


def show_g1() -> int:
    """Build a Unitree G1 in simulation through the same ``Robot()`` factory.

    The G1 is the catalog's most complex robot - a 29-DOF humanoid. In
    ``mode='real'`` a background ONNX locomotion controller owns the legs+waist
    while the agent commands the arms; here we use the default ``mode='sim'`` so
    it runs in MuJoCo with no hardware and no GPU.
    """
    from strands_robots import Robot
    from strands_robots.registry import get_hardware_type, get_robot, resolve_name

    print("=== Unitree G1 (29-DOF humanoid) ===\n")
    canonical = resolve_name("g1")  # alias -> canonical
    info = get_robot(canonical)
    print(f"  alias 'g1' resolves to:  {canonical}")
    print(f"  description:             {info['description']}")
    print(f"  category:                {info['category']}")
    print(f"  lerobot hardware type:   {get_hardware_type(canonical)}")
    print(f"  sim asset (MuJoCo):      {info.get('asset', {}).get('model_xml')}")

    print("\n  Building it in simulation via Robot('g1') ...")
    sim = Robot("g1", mesh=False)
    try:
        print(f"  -> {type(sim).__name__} ready (MuJoCo backend, no hardware).")
    finally:
        sim.destroy()

    print("\n  Drive it for real (locomotion + agent-controlled arms):")
    print("    g1 = Robot('g1', mode='real',")
    print("               robot_ip='192.168.123.164',")
    print("               controller='GrootLocomotionController')")
    print("    # Background thread owns legs+waist; send_action() publishes arm targets.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--g1", action="store_true", help="Focus on the Unitree G1 humanoid (builds it in sim).")
    args = p.parse_args()

    try:
        return show_g1() if args.g1 else show_catalog()
    except ImportError as e:
        print(
            f"This example needs the simulation extra: pip install 'strands-robots[sim-mujoco]'\n  {e}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
