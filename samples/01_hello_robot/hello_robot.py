#!/usr/bin/env python3
"""
Sample 01: Hello Robot — Your first simulated robot in 1 line.

This is the simplest possible strands-robots script. It:
1. Creates a simulated SO-100 arm (one line!)
2. Runs a mock policy (sine wave) to make it move
3. Renders the scene as a PNG image
4. Records a video of the robot in action

Requirements:
    pip install strands-robots[sim]

No GPU, no API keys, no accounts needed.
"""

from strands_robots import Robot

# ─── 1. Create a robot ───────────────────────────────────────────────
# Robot() auto-detects: no hardware connected → simulation mode
sim = Robot("so100")
print("✅ Created SO-100 robot in simulation")

# ─── 2. See what's in the world ──────────────────────────────────────
info = sim.list_robots()
print(f"\n🤖 Robots in world: {info}")

# ─── 3. Check the state before moving ────────────────────────────────
state_before = sim.get_state()
print(f"\n📊 State before: {state_before['content'][0]['text']}")

# ─── 4. Make it move! ────────────────────────────────────────────────
# Mock policy = smooth sine wave on all joints. Great for testing.
print("\n🏃 Running mock policy for 2 seconds...")
result = sim.run_policy(robot_name="so100", policy_provider="mock", duration=2.0)
print(f"   {result['content'][0]['text']}")

# ─── 5. Check state after ────────────────────────────────────────────
state_after = sim.get_state()
print(f"\n📊 State after: {state_after['content'][0]['text']}")

# ─── 6. Take a photo ─────────────────────────────────────────────────
render_result = sim.render()
if render_result["status"] == "success":
    png_bytes = render_result["content"][1]["image"]["source"]["bytes"]
    with open("my_robot.png", "wb") as f:
        f.write(png_bytes)
    print(f"\n📸 Saved render to my_robot.png ({len(png_bytes)} bytes)")

# ─── 7. Record a video ───────────────────────────────────────────────
try:
    video_result = sim.record_video(
        robot_name="so100",
        policy_provider="mock",
        duration=3.0,
        fps=30,
        output_path="hello_robot.mp4",
    )
    print(f"\n🎬 {video_result['content'][0]['text']}")
except Exception as e:
    print("\n⚠️  Video recording requires imageio: pip install imageio[ffmpeg]")
    print(f"   Error: {e}")

print("\n🎉 Done! You just created and controlled your first robot.")
