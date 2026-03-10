#!/usr/bin/env python3
"""
Sample 09 — Live VLA Execution Stream

Subscribe to VLA execution streams from multiple robots and
plot joint trajectories in real-time.

Each robot publishes observation + action data at 50 Hz to:
    strands/{peer_id}/stream

This script subscribes to those streams and collects data for
visualization. Without matplotlib, it prints to the console.

Requirements:
    pip install strands-robots[zenoh]
    pip install matplotlib  (optional, for plotting)

Run:
    python live_stream.py
"""

import sys
import time
from collections import defaultdict

from strands_robots import Robot

# ── Data collector ─────────────────────────────────────────────

class StreamCollector:
    """Collect VLA execution stream data from multiple robots."""

    def __init__(self, max_steps=500):
        self.max_steps = max_steps
        # {peer_id: {"steps": [], "obs": {key: [values]}, "act": {key: [values]}}}
        self.data = defaultdict(lambda: {
            "steps": [],
            "obs": defaultdict(list),
            "act": defaultdict(list),
            "instructions": [],
            "policies": [],
        })

    def on_step(self, topic, data):
        """Callback for each VLA step received from any robot."""
        peer = data.get("peer_id", "unknown")
        step = data.get("step", 0)
        obs = data.get("observation", {})
        act = data.get("action", {})

        entry = self.data[peer]
        entry["steps"].append(step)
        entry["instructions"].append(data.get("instruction", ""))
        entry["policies"].append(data.get("policy", ""))

        # Collect joint-level observation data
        for key, val in obs.items():
            if isinstance(val, (int, float)):
                entry["obs"][key].append(val)
            elif isinstance(val, list) and len(val) < 50:
                entry["obs"][key].append(val)

        # Collect joint-level action data
        for key, val in act.items():
            if isinstance(val, (int, float)):
                entry["act"][key].append(val)
            elif isinstance(val, list) and len(val) < 50:
                entry["act"][key].append(val)

        # Trim to max_steps
        if len(entry["steps"]) > self.max_steps:
            entry["steps"] = entry["steps"][-self.max_steps:]
            entry["instructions"] = entry["instructions"][-self.max_steps:]
            entry["policies"] = entry["policies"][-self.max_steps:]
            for k in list(entry["obs"]):
                entry["obs"][k] = entry["obs"][k][-self.max_steps:]
            for k in list(entry["act"]):
                entry["act"][k] = entry["act"][k][-self.max_steps:]

    def summary(self):
        """Print a summary of collected data."""
        print(f"\n📊 Stream Data Summary ({len(self.data)} robots):")
        for peer, entry in self.data.items():
            n = len(entry["steps"])
            obs_keys = list(entry["obs"].keys())[:5]
            act_keys = list(entry["act"].keys())[:5]
            instr = entry["instructions"][-1] if entry["instructions"] else ""
            policy = entry["policies"][-1] if entry["policies"] else ""
            print(f"\n  🤖 {peer}: {n} steps")
            print(f"     Instruction: {instr}")
            print(f"     Policy: {policy}")
            print(f"     Obs keys: {obs_keys}")
            print(f"     Act keys: {act_keys}")


def try_plot(collector):
    """Try to plot with matplotlib. Falls back to console output."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  ⚠️  matplotlib not installed — skipping plot")
        print("     Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(
        len(collector.data), 2,
        figsize=(14, 4 * len(collector.data)),
        squeeze=False,
    )
    fig.suptitle("Live VLA Execution Streams", fontsize=14)

    for idx, (peer, entry) in enumerate(collector.data.items()):
        steps = entry["steps"]
        if not steps:
            continue

        # Plot observations
        ax_obs = axes[idx][0]
        for key, vals in list(entry["obs"].items())[:6]:
            if vals and isinstance(vals[0], (int, float)):
                ax_obs.plot(steps[: len(vals)], vals, label=key, linewidth=0.8)
        ax_obs.set_title(f"{peer} — Observations")
        ax_obs.set_xlabel("Step")
        ax_obs.set_ylabel("Value")
        ax_obs.legend(fontsize=7, ncol=2)
        ax_obs.grid(True, alpha=0.3)

        # Plot actions
        ax_act = axes[idx][1]
        for key, vals in list(entry["act"].items())[:6]:
            if vals and isinstance(vals[0], (int, float)):
                ax_act.plot(steps[: len(vals)], vals, label=key, linewidth=0.8)
        ax_act.set_title(f"{peer} — Actions")
        ax_act.set_xlabel("Step")
        ax_act.set_ylabel("Value")
        ax_act.legend(fontsize=7, ncol=2)
        ax_act.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("stream_plot.png", dpi=150)
    print("\n  📈 Plot saved to stream_plot.png")
    plt.close()


def main():
    print("📡 Live VLA Stream Demo")
    print("=" * 60)

    # ── Create robots ──────────────────────────────────────────

    print("\n🤖 Creating robots...")
    arm1 = Robot("so100", peer_id="arm1")
    arm2 = Robot("so100", peer_id="arm2")

    if arm1.mesh is None or not arm1.mesh.alive:
        print("\n  ⚠️  Mesh not available — install eclipse-zenoh")
        sys.exit(1)

    # Wait for peer discovery
    time.sleep(1.0)
    print(f"  ✅ arm1 on mesh, {len(arm1.mesh.peers)} peer(s) discovered")

    # ── Subscribe to streams ───────────────────────────────────

    print("\n📡 Subscribing to VLA execution streams...")

    collector = StreamCollector(max_steps=200)

    # Watch arm2's execution stream from arm1's perspective
    arm1.mesh.on_stream("arm2", callback=collector.on_step)
    print("  ✅ Watching arm2's stream via arm1")

    # Also subscribe to arm1's own stream (for self-monitoring)
    arm1.mesh.on_stream("arm1", callback=collector.on_step)
    print("  ✅ Watching arm1's own stream")

    # ── Run policies to generate stream data ───────────────────

    print("\n🏃 Running mock policies to generate stream data...")

    # Tell arm2 to run a mock policy (this generates stream data)
    arm1.mesh.tell(
        "arm2",
        "wave hello with sinusoidal motion",
        policy_provider="mock",
        duration=3.0,
    )

    # Run a mock policy on arm1 directly too
    arm1.mesh.tell(
        "arm1",
        "reach forward and back",
        policy_provider="mock",
        duration=3.0,
    )

    # Wait for execution to complete
    print("  ⏳ Waiting for policy execution...")
    time.sleep(5.0)

    # ── Display results ────────────────────────────────────────

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    collector.summary()

    # Try to plot if matplotlib is available
    try_plot(collector)

    # Also show buffered messages if using buffer mode
    for robot_name, robot in [("arm1", arm1)]:
        if hasattr(robot.mesh, "inbox"):
            for sub_name, msgs in robot.mesh.inbox.items():
                if msgs:
                    print(f"\n  📬 {robot_name} inbox['{sub_name}']: {len(msgs)} messages")
                    # Show last message
                    _, last = msgs[-1]
                    print(f"     Last: step={last.get('step')}, "
                          f"policy={last.get('policy')}, "
                          f"obs_keys={list(last.get('observation', {}).keys())[:5]}")

    # ── Cleanup ────────────────────────────────────────────────

    print("\n🧹 Cleanup...")
    for robot in [arm1, arm2]:
        if robot.mesh:
            robot.mesh.stop()

    print("\n✅ Live stream demo complete!")


if __name__ == "__main__":
    main()
