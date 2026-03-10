#!/usr/bin/env python3
"""
Sample 09 — Emergency Stop

Safety demo: fleet-wide emergency stop and graceful shutdown.

The emergency_stop() method broadcasts {"action": "stop"} to ALL peers
on the Zenoh mesh. Every robot that receives it immediately halts
its current task and zeros torques.

Requirements:
    pip install strands-robots[zenoh]

Run:
    python emergency_stop.py
"""

import sys
import time

from strands_robots import Robot


def main():
    print("🚨 Emergency Stop Demo")
    print("=" * 60)

    # ── Create a fleet ─────────────────────────────────────────

    print("\n🤖 Creating fleet of 3 robots...")

    robots = [
        Robot("so100", peer_id="arm_alpha"),
        Robot("so100", peer_id="arm_beta"),
        Robot("unitree_g1", peer_id="walker"),
    ]

    # Verify mesh
    for robot in robots:
        if robot.mesh is None or not robot.mesh.alive:
            print(f"\n  ⚠️  {robot.mesh.peer_id if robot.mesh else 'robot'}: mesh not available")
            print("     Install: pip install eclipse-zenoh")
            sys.exit(1)
        print(f"  ✅ {robot.mesh.peer_id} on mesh")

    # Wait for peer discovery
    time.sleep(1.0)

    # ── Start tasks on all robots ──────────────────────────────

    print("\n" + "=" * 60)
    print("Phase 1: Start Tasks")
    print("=" * 60)

    orchestrator = robots[0]  # arm_alpha orchestrates

    # Give each robot a task
    tasks = [
        ("arm_alpha", "continuously wave left and right"),
        ("arm_beta", "reach for the blue cube"),
        ("walker", "walk forward slowly"),
    ]

    for target, instruction in tasks:
        print(f"\n  📨 → {target}: '{instruction}'")
        result = orchestrator.mesh.tell(
            target,
            instruction,
            policy_provider="mock",
            duration=30.0,  # Long duration — will be interrupted
        )
        print(f"    Started: {str(result)[:150]}")

    # Let them run briefly
    print("\n  ⏳ Running for 2 seconds...")
    time.sleep(2.0)

    # Check status
    print("\n  📊 Fleet status before E-STOP:")
    responses = orchestrator.mesh.broadcast({"action": "status"}, timeout=3.0)
    for r in responses:
        peer = r.get("responder_id", "?")
        status = r.get("result", {})
        if isinstance(status, dict):
            print(f"    • {peer}: {status.get('status', status.get('task_status', '?'))}")
        else:
            print(f"    • {peer}: {status}")

    # ── EMERGENCY STOP ─────────────────────────────────────────

    print("\n" + "=" * 60)
    print("🚨 EMERGENCY STOP")
    print("=" * 60)

    print("\n  Sending emergency_stop() to ALL peers...")
    start = time.time()
    responses = orchestrator.mesh.emergency_stop()
    elapsed = time.time() - start

    print(f"\n  ⏱️  E-STOP completed in {elapsed*1000:.1f} ms")
    print(f"  📥 {len(responses)} robots responded:")
    for r in responses:
        peer = r.get("responder_id", "?")
        print(f"    🛑 {peer}: stopped")

    # ── Verify all stopped ─────────────────────────────────────

    print("\n" + "=" * 60)
    print("Phase 2: Verify All Stopped")
    print("=" * 60)

    time.sleep(0.5)
    print("\n  📊 Fleet status after E-STOP:")
    responses = orchestrator.mesh.broadcast({"action": "status"}, timeout=3.0)
    for r in responses:
        peer = r.get("responder_id", "?")
        status = r.get("result", {})
        if isinstance(status, dict):
            task_status = status.get("status", status.get("task_status", "?"))
            print(f"    ✅ {peer}: {task_status}")
        else:
            print(f"    ✅ {peer}: {status}")

    # ── Graceful shutdown ──────────────────────────────────────

    print("\n" + "=" * 60)
    print("Phase 3: Graceful Shutdown")
    print("=" * 60)

    print("\n  🧹 Stopping mesh for each robot...")
    for robot in robots:
        if robot.mesh and robot.mesh.alive:
            peer_id = robot.mesh.peer_id
            robot.mesh.stop()
            print(f"    🔌 {peer_id} disconnected from mesh")

    print("""
    ✅ Emergency stop demo complete!

    Key takeaways:
    • emergency_stop() broadcasts to ALL peers simultaneously
    • Response time is typically < 10ms on LAN (Zenoh multicast)
    • Every robot handles the stop command independently
    • Always call mesh.stop() during graceful shutdown
    • For real hardware: stop zeros torques via the robot driver
    """)


if __name__ == "__main__":
    main()
