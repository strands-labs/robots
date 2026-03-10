#!/usr/bin/env python3
"""
Sample 01 — Exercise Solutions

Solutions for the three exercises in the README.
"""

# ─── Exercise 1: Create a Robot('panda') and render it ───────────────
def exercise_1():
    """Create a Franka Panda arm and save a render."""
    from strands_robots import Robot

    sim = Robot("panda")
    result = sim.render()
    if result["status"] == "success":
        with open("panda_render.png", "wb") as f:
            f.write(result["content"][1]["image"]["source"]["bytes"])
        print("✅ Saved panda_render.png")
    sim.destroy()


# ─── Exercise 2: Record videos of 3 different robots ─────────────────
def exercise_2():
    """Record a short video for three different robots."""
    from strands_robots import Robot

    for name in ["so100", "panda", "aloha"]:
        sim = Robot(name)
        try:
            sim.record_video(
                robot_name=name,
                policy_provider="mock",
                duration=2.0,
                fps=30,
                output_path=f"{name}_demo.mp4",
            )
            print(f"✅ Saved {name}_demo.mp4")
        except ImportError:
            print("⚠️  Video requires: pip install imageio[ffmpeg]")
            break
        finally:
            sim.destroy()


# ─── Exercise 3: Print joint positions before and after ───────────────
def exercise_3():
    """Compare joint positions before and after running a policy."""
    from strands_robots import Robot

    sim = Robot("so100")

    state_before = sim.get_state()
    print(f"Before: {state_before['content'][0]['text']}")

    sim.run_policy(robot_name="so100", policy_provider="mock", duration=1.0)

    state_after = sim.get_state()
    print(f"After:  {state_after['content'][0]['text']}")
    sim.destroy()


if __name__ == "__main__":
    print("=" * 50)
    print("Exercise 1: Render a Panda")
    exercise_1()

    print("\nExercise 2: Record 3 robots")
    exercise_2()

    print("\nExercise 3: Compare states")
    exercise_3()
