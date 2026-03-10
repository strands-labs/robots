#!/usr/bin/env python3
"""
Sample 09 — Fleet Orchestration

Coordinate multiple robots for a collaborative task:
  - Robot A picks up an object
  - Robot A hands it to Robot B
  - Robot B places it at a target location

Includes status monitoring and error handling.

Requirements:
    pip install strands-robots[zenoh]

Run:
    python fleet_orchestration.py
"""

import sys
import time

from strands_robots import Robot


def wait_for_peers(robot, expected_peers, timeout=5.0):
    """Wait until the expected number of peers are discovered."""
    start = time.time()
    while time.time() - start < timeout:
        if len(robot.mesh.peers) >= expected_peers:
            return True
        time.sleep(0.2)
    return len(robot.mesh.peers) >= expected_peers


def get_fleet_status(orchestrator):
    """Broadcast a status query and print results."""
    responses = orchestrator.mesh.broadcast({"action": "status"}, timeout=3.0)
    print(f"\n  📊 Fleet Status ({len(responses)} robots responding):")
    for r in responses:
        peer = r.get("responder_id", "unknown")
        status = r.get("result", r)
        if isinstance(status, dict):
            task = status.get("status", status.get("task_status", "idle"))
            instr = status.get("instruction", "")
            print(f"    • {peer}: {task}" + (f" — {instr}" if instr else ""))
        else:
            print(f"    • {peer}: {status}")
    return responses


def main():
    print("🤖 Fleet Orchestration Demo")
    print("=" * 60)

    # ── Create the fleet ──────────────────────────────────────

    print("\n📦 Creating fleet...")

    # Three robots with explicit peer IDs for coordination
    picker = Robot("so100", peer_id="picker")
    placer = Robot("so100", peer_id="placer")
    mobile = Robot("unitree_g1", peer_id="mobile_base")

    fleet = [
        (picker, "picker", "Picks up objects"),
        (placer, "placer", "Places objects at target"),
        (mobile, "mobile_base", "Mobile base / transporter"),
    ]

    # Verify mesh is active
    for robot, name, role in fleet:
        if robot.mesh is None or not robot.mesh.alive:
            print(f"\n  ⚠️  {name}: mesh not available")
            print("     Install eclipse-zenoh: pip install eclipse-zenoh")
            print("     Or run with: pip install strands-robots[zenoh]")
            sys.exit(1)
        print(f"  ✅ {name} ({role}) — peer_id={robot.mesh.peer_id}")

    # Wait for peer discovery
    print("\n🔍 Waiting for peer discovery...")
    if not wait_for_peers(picker, expected_peers=2, timeout=3.0):
        print(f"  ⚠️  Only found {len(picker.mesh.peers)} peers (expected 2)")
    else:
        print(f"  ✅ All {len(picker.mesh.peers)} peers discovered")

    # ── Phase 1: Status check ─────────────────────────────────

    print("\n" + "=" * 60)
    print("Phase 1: Fleet Status Check")
    print("=" * 60)

    get_fleet_status(picker)

    # ── Phase 2: Collaborative task ───────────────────────────

    print("\n" + "=" * 60)
    print("Phase 2: Collaborative Pick-and-Place")
    print("=" * 60)

    # Step 1: Mobile base navigates to pickup location
    print("\n  Step 1: Mobile base → navigate to pickup zone")
    result = picker.mesh.tell(
        "mobile_base",
        "walk to the pickup zone",
        policy_provider="mock",
        duration=3.0,
    )
    print(f"    Result: {str(result)[:200]}")

    # Step 2: Picker picks up the object
    print("\n  Step 2: Picker → pick up the red cube")
    result = picker.mesh.tell(
        "picker",  # Can send to self via mesh
        "pick up the red cube",
        policy_provider="mock",
        duration=3.0,
    )
    print(f"    Result: {str(result)[:200]}")

    # Status check mid-task
    print("\n  --- Mid-task status ---")
    get_fleet_status(picker)

    # Step 3: Mobile base moves to handoff position
    print("\n  Step 3: Mobile base → navigate to handoff zone")
    result = picker.mesh.tell(
        "mobile_base",
        "walk to the handoff zone",
        policy_provider="mock",
        duration=3.0,
    )
    print(f"    Result: {str(result)[:200]}")

    # Step 4: Picker hands off to placer
    print("\n  Step 4: Placer → receive object from picker")
    result = picker.mesh.tell(
        "placer",
        "receive object from the picker arm",
        policy_provider="mock",
        duration=3.0,
    )
    print(f"    Result: {str(result)[:200]}")

    # Step 5: Placer places at target
    print("\n  Step 5: Placer → place object on the shelf")
    result = picker.mesh.tell(
        "placer",
        "place the object on the shelf",
        policy_provider="mock",
        duration=3.0,
    )
    print(f"    Result: {str(result)[:200]}")

    # ── Phase 3: Final status ─────────────────────────────────

    print("\n" + "=" * 60)
    print("Phase 3: Final Fleet Status")
    print("=" * 60)

    get_fleet_status(picker)

    # ── Cleanup ────────────────────────────────────────────────

    print("\n🧹 Shutting down fleet...")
    for robot, name, _ in fleet:
        if robot.mesh:
            robot.mesh.stop()
            print(f"  🔌 {name} off mesh")

    print("\n✅ Fleet orchestration complete!")


if __name__ == "__main__":
    main()
