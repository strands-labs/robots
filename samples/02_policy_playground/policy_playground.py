#!/usr/bin/env python3
"""
Sample 02: Policy Playground
K12 Level 1 (Elementary)

Learn what policies are and how they control robots!

This script demonstrates:
1. Creating a robot simulation
2. Running a policy (the robot's "brain")
3. Recording video of robot behavior
4. Exploring available policy providers
5. Understanding the Policy abstract base class
"""

from strands_robots.policy import create_policy, list_providers

from strands_robots import Robot


def main():
    """Main policy playground demonstration."""
    print("=" * 60)
    print("🎮 Welcome to the Policy Playground!")
    print("=" * 60)
    print()

    # Step 1: Create a robot simulation
    print("📦 Step 1: Creating a robot simulation...")
    print("   Robot: so100 (Stanford Humanoid)")
    print()

    sim = Robot("so100")

    print("✅ Robot created successfully!")
    print()

    # Step 2: Run the mock policy
    print("=" * 60)
    print("🎪 Step 2: Running the Mock Policy")
    print("=" * 60)
    print()
    print("The 'mock' policy creates sinusoidal (wave-like) motion.")
    print("This is perfect for testing that everything works!")
    print()
    print("Running for 3 seconds...")
    print()

    sim.run_policy(
        robot_name="so100",
        policy_provider="mock",
        duration=3.0,
    )

    print("✅ Mock policy completed!")
    print()

    # Step 3: Record a video
    print("=" * 60)
    print("🎥 Step 3: Recording Video")
    print("=" * 60)
    print()
    print("Recording a 3-second video of the mock policy...")
    print("Output: mock_policy_demo.mp4")
    print()

    sim.record_video(
        robot_name="so100",
        policy_provider="mock",
        duration=3.0,
        fps=30,
        output_path="mock_policy_demo.mp4",
    )

    print("✅ Video recorded successfully!")
    print("   Watch: mock_policy_demo.mp4")
    print()

    # Step 4: Explore available policy providers
    print("=" * 60)
    print("🗺️  Step 4: Exploring Policy Providers")
    print("=" * 60)
    print()
    print("The strands-robots SDK supports many policy providers!")
    print()

    # Get all providers
    providers = list_providers()

    print(f"📊 Total providers available: {len(providers)}")
    print()

    # Categorize providers
    local_providers = ["mock", "stopped", "random"]
    gpu_providers = [
        "pi0",
        "act",
        "smolvla",
        "openvla",
        "groot",
        "diffusion_policy",
        "tdmpc",
        "vqbet",
    ]

    print("✅ Local Providers (no GPU needed):")
    for provider in local_providers:
        if provider in providers:
            print(f"   • {provider}")
    print()

    print("🎮 GPU Providers (need special hardware):")
    for provider in gpu_providers:
        if provider in providers:
            print(f"   • {provider}")
    print()

    print("🔍 All available providers:")
    for i, provider in enumerate(providers, 1):
        print(f"   {i:2d}. {provider}")
    print()

    # Step 5: Create a policy directly
    print("=" * 60)
    print("🧠 Step 5: Understanding the Policy ABC")
    print("=" * 60)
    print()
    print("Let's create a policy instance directly to see the structure.")
    print()

    # Create a mock policy instance
    policy = create_policy("mock")

    print(f"✅ Policy created: {type(policy).__name__}")
    print()
    print("📚 Every policy inherits from the Policy ABC (Abstract Base Class)")
    print("   and must implement this method:")
    print()
    print("   async def get_actions(self, observations, instruction=None):")
    print("       # observations: Current robot state")
    print("       # instruction: Optional text command")
    print("       # Returns: List of action dictionaries")
    print()
    print("This interface allows any policy to control any robot!")
    print()

    # Display policy info
    print("🔍 Mock Policy Details:")
    print(f"   • Type: {type(policy).__name__}")
    print(f"   • Module: {type(policy).__module__}")
    print(f"   • Has get_actions: {hasattr(policy, 'get_actions')}")
    print()

    # Summary
    print("=" * 60)
    print("🎓 Summary")
    print("=" * 60)
    print()
    print("You learned:")
    print("  ✅ How to create a robot simulation")
    print("  ✅ How to run a policy on a robot")
    print("  ✅ How to record videos of robot behavior")
    print(f"  ✅ That there are {len(providers)} policy providers available")
    print("  ✅ The difference between local and GPU providers")
    print("  ✅ The Policy ABC interface (get_actions)")
    print()
    print("🚀 Next Steps:")
    print("  1. Try Exercise 1: Test different robots")
    print("  2. Try Exercise 2: Compare mock vs stopped policies")
    print("  3. Try Exercise 3: Create your own custom policy")
    print()
    print("Happy exploring! 🤖")
    print()


if __name__ == "__main__":
    main()
