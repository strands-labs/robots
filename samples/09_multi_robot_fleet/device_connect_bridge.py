#!/usr/bin/env python3
"""
Sample 09 — ARM device_connect Bridge

Demonstrate ARM's device_connect integration with strands-robots:
  - Show the drop-in replacement pattern
  - Cross-platform discovery (Reachy Mini uses same Zenoh)
  - Subscribe to external Zenoh topics via mesh.subscribe()

ARM's device_connect (https://github.com/ARM-software/device_connect)
is a Zenoh-native device discovery and communication layer. When installed,
strands-robots auto-detects it and uses ARM's implementation for the
robot_mesh tool.

Requirements:
    pip install strands-robots[zenoh]
    # Optional: pip install git+https://github.com/ARM-software/device_connect.git

Run:
    python device_connect_bridge.py
"""

import sys
import time

from strands_robots import Robot


def check_device_connect():
    """Check if ARM device_connect is installed."""
    try:
        import device_connect  # noqa: F401
        return True
    except ImportError:
        return False


def demo_robot_mesh_tool():
    """Show the robot_mesh tool API (works with or without device_connect)."""
    from strands_robots.tools.robot_mesh import robot_mesh

    print("\n📋 robot_mesh Tool API")
    print("-" * 40)

    # The tool works whether device_connect is installed or not.
    # If device_connect IS installed, it auto-routes to ARM's implementation.
    result = robot_mesh(action="status")
    print(f"  Status: {result}")

    result = robot_mesh(action="peers")
    print(f"  Peers: {result.get('content', [{}])[0].get('text', '')[:200]}")


def demo_subscribe_external():
    """Subscribe to Reachy Mini and other external Zenoh topics."""
    print("\n📡 External Topic Subscriptions")
    print("-" * 40)

    robot = Robot("so100", peer_id="bridge_demo")

    if robot.mesh is None or not robot.mesh.alive:
        print("  ⚠️  Mesh not available")
        return

    # Subscribe to Reachy Mini topics (if a Reachy is on the network)
    # Reachy Mini publishes joint data via native Zenoh
    print("\n  Subscribing to Reachy Mini topics...")
    robot.mesh.subscribe("reachy_mini/joint_positions", name="reachy_joints")
    robot.mesh.subscribe("reachy_mini/head_pose", name="reachy_head")
    robot.mesh.subscribe("reachy_mini/*", name="reachy_all")
    print("  ✅ Subscribed to: reachy_mini/joint_positions")
    print("  ✅ Subscribed to: reachy_mini/head_pose")
    print("  ✅ Subscribed to: reachy_mini/* (wildcard)")

    # Subscribe to device_connect topics (if devices are on the network)
    robot.mesh.subscribe("device_connect/*/presence", name="dc_presence")
    print("  ✅ Subscribed to: device_connect/*/presence")

    # Subscribe to any joint data on the entire network
    robot.mesh.subscribe("*/joint_positions", name="all_joints")
    print("  ✅ Subscribed to: */joint_positions (wildcard)")

    # Wait a moment for any messages
    print("\n  ⏳ Listening for 3 seconds...")
    time.sleep(3.0)

    # Check inbox for received messages
    print("\n  📬 Inbox:")
    if hasattr(robot.mesh, "inbox"):
        for name, msgs in robot.mesh.inbox.items():
            count = len(msgs)
            if count > 0:
                _, last = msgs[-1]
                print(f"    • {name}: {count} messages (last: {str(last)[:100]})")
            else:
                print(f"    • {name}: 0 messages (no device found)")
    else:
        print("    (no inbox)")

    # Cleanup
    robot.mesh.stop()
    return robot


def demo_drop_in_pattern():
    """Show how device_connect is a drop-in replacement."""
    print("\n🔄 Drop-in Replacement Pattern")
    print("-" * 40)

    has_dc = check_device_connect()

    print(f"""
    # robot_mesh works identically whether device_connect is installed or not.
    # strands-robots auto-detects the best available backend.

    from strands_robots.tools.robot_mesh import robot_mesh

    # These calls work with BOTH backends:
    robot_mesh(action="peers")                    # List all robots
    robot_mesh(action="tell", target="...",       # Tell robot to do something
               instruction="pick up the cube")
    robot_mesh(action="emergency_stop")           # Fleet-wide stop
    robot_mesh(action="subscribe",                # Subscribe to any topic
               target="reachy_mini/*")

    Current backend: {"ARM device_connect ✅" if has_dc else "strands-robots zenoh_mesh (native)"}
    """)


def main():
    print("🔌 ARM device_connect Bridge Demo")
    print("=" * 60)

    # Check device_connect availability
    has_dc = check_device_connect()
    print(f"\n  ARM device_connect installed: {'✅ Yes' if has_dc else '❌ No (using native zenoh_mesh)'}")
    print("  eclipse-zenoh: ", end="")
    try:
        import zenoh  # noqa: F401
        print("✅ Yes")
    except ImportError:
        print("❌ No — install with: pip install eclipse-zenoh")
        sys.exit(1)

    # Demo 1: Drop-in pattern explanation
    demo_drop_in_pattern()

    # Demo 2: robot_mesh tool
    demo_robot_mesh_tool()

    # Demo 3: External topic subscriptions
    demo_subscribe_external()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
    Key points:
    • robot_mesh tool is the same API regardless of backend
    • mesh.subscribe() bridges to ANY Zenoh topic on the network
    • Reachy Mini publishes joint data via native Zenoh — no adapter needed
    • ARM device_connect adds enterprise-grade discovery features
    • Same ToolSpec = agents work identically with either backend
    """)

    print("✅ Bridge demo complete!")


if __name__ == "__main__":
    main()
