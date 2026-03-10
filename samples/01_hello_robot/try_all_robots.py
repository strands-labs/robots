#!/usr/bin/env python3
"""
Sample 01 (bonus): Try All Robots — render every category of robot.

Loops through representative robots from each category:
  Arms, Bimanual, Humanoids, Quadrupeds, Hands

Each one is created, run with a mock policy, and rendered to PNG.

Requirements:
    pip install strands-robots[sim]
"""

import os

from strands_robots import Robot

# Representative robots from each category
ROBOTS = [
    ("so100", "6-DOF tabletop arm"),
    ("panda", "7-DOF industrial arm (Franka)"),
    ("aloha", "Bimanual (2× ViperX 300s)"),
    ("unitree_g1", "29-DOF humanoid"),
    ("unitree_go2", "Quadruped"),
    ("reachy_mini", "6-DOF Stewart head"),
]

os.makedirs("robot_gallery", exist_ok=True)

print("=" * 60)
print("🤖 Robot Gallery — Rendering all categories")
print("=" * 60)

for robot_name, description in ROBOTS:
    print(f"\n{'─' * 40}")
    print(f"🤖 {robot_name}: {description}")

    try:
        # Create robot
        sim = Robot(robot_name)

        # Run mock policy briefly to get an interesting pose
        sim.run_policy(robot_name=robot_name, policy_provider="mock", duration=0.5)

        # Render
        result = sim.render()
        if result["status"] == "success":
            png_bytes = result["content"][1]["image"]["source"]["bytes"]
            path = f"robot_gallery/{robot_name}.png"
            with open(path, "wb") as f:
                f.write(png_bytes)
            print(f"   📸 Saved to {path} ({len(png_bytes):,} bytes)")
        else:
            print(f"   ❌ Render failed: {result['content'][0]['text']}")

        # Clean up
        sim.destroy()

    except Exception as e:
        print(f"   ❌ Error: {e}")

print(f"\n{'=' * 60}")
print("🎉 Gallery complete! Check robot_gallery/ directory")
print(f"{'=' * 60}")
