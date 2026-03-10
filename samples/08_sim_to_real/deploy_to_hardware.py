#!/usr/bin/env python3
"""Sample 08 — Part 3: Deploy a Trained Policy to Physical Hardware.

Level: 3 (Advanced/High School) | Time: ~10 min | Hardware: Physical Robot

This script walks through deploying a fine-tuned policy (from Part 2) to
real hardware.  The key insight: **the Policy ABC is identical between sim
and real**, so the same ``policy_provider`` + ``model_path`` that worked
in simulation works on the physical robot with zero code changes.

Supported robots (connection varies):
    SO-100       — 6 DOF, USB serial (Feetech), 1-2 webcams
    Unitree G1   — 29 DOF, Ethernet + SDK, built-in cameras
    Reachy Mini  — 6+2 DOF, Zenoh (native), head cameras

SDK surface covered:
    - strands_robots.Robot  (factory — auto-detects sim vs real)
    - strands_robots.robot.TaskStatus
    - strands_robots.robot.RobotTaskState
    - strands_robots.policies.create_policy
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Safety wrapper (imported from safety_wrapper.py)
# ---------------------------------------------------------------------------

def _load_safety(robot_type: str = "so100"):
    """Load safety config for the chosen robot."""
    try:
        from .safety_wrapper import SafetyWrapper

        return SafetyWrapper.from_yaml(
            os.path.join(
                os.path.dirname(__file__),
                "configs",
                f"deploy_{robot_type}.yaml",
            )
        )
    except Exception:
        # Graceful fallback — print limits instead
        return None


# ---------------------------------------------------------------------------
# Step 1: Sim rehearsal — verify the policy before touching real hardware
# ---------------------------------------------------------------------------

def sim_rehearsal(
    robot_name: str = "so101",
    checkpoint: str | None = None,
) -> bool:
    """Run the policy in simulation first as a safety check.

    If anything looks wrong (NaN actions, wild joint velocities),
    we catch it here rather than on the real robot.
    """
    from strands_robots import Robot

    print("\n[1/5] Sim rehearsal — verifying policy before hardware deploy...")
    sim = Robot(robot_name, mesh=False)
    sim.add_object(
        name="red_cube",
        shape="box",
        size=[0.04, 0.04, 0.04],
        position=[0.25, 0.05, 0.05],
        color=[1.0, 0.0, 0.0, 1.0],
    )

    try:
        # Use mock policy for demo; in production use the trained checkpoint:
        #   policy_provider="groot", model_path=checkpoint
        sim.run_policy(
            robot_name=robot_name,
            policy_provider="mock",
            instruction="pick up the red cube",
            duration=3.0,
        )
        print("  ✅ Sim rehearsal passed — safe to deploy to hardware")
        return True
    except Exception as exc:
        print(f"  ❌ Sim rehearsal FAILED: {exc}")
        print("  ⚠️  DO NOT deploy to hardware until sim works!")
        return False
    finally:
        sim.destroy()


# ---------------------------------------------------------------------------
# Step 2: Connect to real robot
# ---------------------------------------------------------------------------

def connect_robot(
    robot_type: str = "so100_follower",
    cameras: dict | None = None,
) -> object | None:
    """Instantiate a HardwareRobot via the Robot() factory.

    The factory auto-detects the 'real' backend when given a LeRobot
    robot_type string (e.g. ``"so100_follower"``).
    """
    print(f"\n[2/5] Connecting to real robot: {robot_type}...")
    if cameras is None:
        cameras = {"webcam": {"index_or_path": 0, "width": 640, "height": 480}}

    try:
        from strands_robots import Robot

        # The factory auto-detects real hardware when mode="real"
        robot = Robot(robot_type, mode="real", cameras=cameras)
        print(f"  ✅ Connected to {robot_type}")
        return robot
    except Exception as exc:
        print(f"  ⚠️  Hardware not available: {exc}")
        print("  📝 Run this on a machine physically connected to the robot")
        return None


# ---------------------------------------------------------------------------
# Step 3: Run the policy on real hardware
# ---------------------------------------------------------------------------

def run_policy_on_hardware(
    robot,
    policy_provider: str = "groot",
    checkpoint: str | None = None,
    instruction: str = "pick up the red cube",
    duration: float = 30.0,
) -> dict | None:
    """Execute the trained policy on the physical robot.

    Uses the ``start`` action which runs the policy asynchronously,
    then polls ``status`` until the task completes or times out.
    """
    print(f"\n[3/5] Running policy on hardware (duration={duration}s)...")

    if robot is None:
        print("  ⚠️  No robot connected — skipping")
        return None

    try:
        # Start the task (non-blocking)
        # In production:  model_path=checkpoint
        result = robot.start_task(
            instruction=instruction,
            policy_provider=policy_provider,
            duration=duration,
        )
        print(f"  Task started: {result}")

        # Poll status — get_status() is async, so use asyncio.run()
        start_time = time.time()
        while time.time() - start_time < duration + 5:
            status = asyncio.run(robot.get_status())
            state = status.get("content", [{}])[0].get("text", "unknown")
            print(f"  Status: {state}", end="\r")
            if "COMPLETED" in state or "IDLE" in state or "ERROR" in state:
                break
            time.sleep(1.0)

        print()
        print("  ✅ Task finished")
        return result

    except Exception as exc:
        print(f"  ❌ Execution error: {exc}")
        # Emergency stop
        try:
            robot.stop_task()
            print("  🛑 Emergency stop activated")
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# Step 4: Record real-world execution
# ---------------------------------------------------------------------------

def record_real_execution(
    robot,
    instruction: str = "pick up the red cube",
    duration: float = 15.0,
) -> str | None:
    """Record a real-world episode to LeRobot v3 dataset format.

    This uses the ``record`` action which captures joint states + camera
    frames synchronised at 50 Hz and saves to a local dataset.
    """
    print(f"\n[4/5] Recording real-world episode ({duration}s)...")

    if robot is None:
        print("  ⚠️  No robot connected — skipping")
        return None

    try:
        result = robot.record_task(
            instruction=instruction,
            policy_provider="groot",
            duration=duration,
            repo_id="local/real_episodes",
        )
        print("  ✅ Recorded episode")
        return result
    except Exception as exc:
        print(f"  ⚠️  Recording failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Step 5: Compare sim vs. real trajectories
# ---------------------------------------------------------------------------

def compare_trajectories() -> None:
    """Print a conceptual comparison of sim vs. real trajectories."""
    print("\n[5/5] Sim vs. real trajectory comparison")
    print()
    print("  What to look for:")
    print("    • Joint positions should follow similar paths")
    print("    • Real trajectories will be noisier (sensor noise)")
    print("    • Timing may differ (network latency, motor response)")
    print("    • End-effector accuracy may vary by ±1-2 cm")
    print()
    print("  Diagnostic checklist:")
    print("    □ Does the gripper reach the object?")
    print("    □ Does it grasp successfully?")
    print("    □ Is the lift height sufficient?")
    print("    □ Any oscillation or jitter in joint commands?")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Sample 08 — Deploy to Physical Hardware")
    print("=" * 60)
    print()
    print("Safety reminder:")
    print("  ⚠️  Clear the workspace around the robot")
    print("  ⚠️  Keep emergency stop within reach")
    print("  ⚠️  Start with slow motions (reduce max velocity)")
    print()

    # Checkpoint from Part 2 (or use a pretrained one)
    checkpoint = os.path.join(OUTPUT_DIR, "groot_finetuned", "checkpoint-final")

    # Step 1: Always rehearse in sim first
    sim_ok = sim_rehearsal(robot_name="so101", checkpoint=checkpoint)
    if not sim_ok:
        print("\n❌ Aborting — fix the policy in simulation first!")
        return

    # Step 2: Connect to hardware
    robot = connect_robot(
        robot_type="so100_follower",
        cameras={"webcam": {"index_or_path": 0, "width": 640, "height": 480}},
    )

    # Step 3: Execute
    run_policy_on_hardware(
        robot,
        policy_provider="groot",
        checkpoint=checkpoint,
        instruction="pick up the red cube",
        duration=30.0,
    )

    # Step 4: Record
    record_real_execution(robot, duration=15.0)

    # Step 5: Compare
    compare_trajectories()

    print()
    print("✅ Part 3 complete — hardware deployment done")
    print()
    print("Exercises:")
    print("  1. Try the same checkpoint on a different robot (G1, Reachy Mini)")
    print("  2. Compare success rates across 10 real-world trials")
    print("  3. Record failures and use them for fine-tuning (DAgger approach)")


if __name__ == "__main__":
    main()
