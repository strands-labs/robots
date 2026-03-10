#!/usr/bin/env python3
"""
Sample 09 — Zenoh Mesh Demo

Create 3 simulated robots, demonstrate auto-discovery, send commands
between them, and visualize the mesh topology.

Requirements:
    pip install strands-robots[zenoh]

Run:
    python zenoh_mesh_demo.py
"""

import time

from strands_robots import Robot


def main():
    # ── Step 1: Create 3 robots — they auto-join the mesh ──────

    print("🤖 Creating 3 simulated robots...\n")

    # Each Robot() automatically starts a Zenoh mesh peer.
    # peer_id defaults to "{tool_name}-{random_hex}" but we set explicit IDs
    # so the demo is reproducible.
    arm_left = Robot("so100", peer_id="left_arm")
    arm_right = Robot("so100", peer_id="right_arm")
    humanoid = Robot("unitree_g1", peer_id="humanoid")

    # Give Zenoh a moment for multicast discovery (same-process peers
    # discover each other within ~1 heartbeat = 0.5s)
    time.sleep(1.0)

    # ── Step 2: Peer discovery ─────────────────────────────────

    print("=" * 60)
    print("🔗 MESH TOPOLOGY")
    print("=" * 60)

    for robot, name in [
        (arm_left, "Left Arm"),
        (arm_right, "Right Arm"),
        (humanoid, "Humanoid"),
    ]:
        mesh = robot.mesh
        if mesh is None:
            print(f"  ⚠️  {name}: mesh disabled (eclipse-zenoh not installed)")
            continue

        print(f"\n  {name} (peer_id={mesh.peer_id})")
        print(f"    Type: {mesh.peer_type}")
        print(f"    Alive: {mesh.alive}")

        peers = mesh.peers
        if peers:
            print(f"    Peers ({len(peers)}):")
            for p in peers:
                icon = {"robot": "🤖", "sim": "🎮"}.get(p.get("type", ""), "🔧")
                print(f"      {icon} {p['peer_id']} ({p.get('type','?')}) — {p.get('age',0):.1f}s ago")
        else:
            print("    Peers: (none discovered yet)")

    # ── Step 3: Send commands between robots ───────────────────

    print("\n" + "=" * 60)
    print("📡 CROSS-ROBOT COMMANDS")
    print("=" * 60)

    if arm_left.mesh and arm_left.mesh.alive:
        # Send a status query to the humanoid
        print("\n  📨 left_arm → humanoid: status query")
        response = arm_left.mesh.send("humanoid", {"action": "status"}, timeout=5.0)
        print(f"    Response: {response}")

        # Tell the right arm to do something
        print("\n  📨 left_arm → right_arm: 'wave hello'")
        response = arm_left.mesh.tell(
            "right_arm",
            "wave hello",
            policy_provider="mock",
            duration=3.0,
        )
        print(f"    Response: {type(response).__name__} — {str(response)[:200]}")

        # Broadcast to ALL peers
        print("\n  📢 left_arm → broadcast: status")
        responses = arm_left.mesh.broadcast({"action": "status"}, timeout=3.0)
        print(f"    Got {len(responses)} responses")
        for r in responses:
            print(f"      • {r.get('responder_id', '?')}: {r.get('type', '?')}")

    else:
        print("\n  ⚠️  Mesh not available — install eclipse-zenoh to enable")
        print("     pip install eclipse-zenoh")

    # ── Step 4: Feature introspection via mesh ─────────────────

    print("\n" + "=" * 60)
    print("📋 REMOTE FEATURE QUERY")
    print("=" * 60)

    if arm_left.mesh and arm_left.mesh.alive:
        print("\n  📨 left_arm → humanoid: features query")
        response = arm_left.mesh.send("humanoid", {"action": "features"}, timeout=5.0)
        if isinstance(response, dict) and "error" not in response:
            print(f"    Humanoid features: {str(response)[:300]}")
        else:
            print(f"    Response: {response}")

    # ── Cleanup ────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("🧹 CLEANUP")
    print("=" * 60)

    for robot, name in [
        (arm_left, "Left Arm"),
        (arm_right, "Right Arm"),
        (humanoid, "Humanoid"),
    ]:
        if robot.mesh:
            robot.mesh.stop()
            print(f"  🔌 {name} ({robot.mesh.peer_id}) off mesh")

    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
