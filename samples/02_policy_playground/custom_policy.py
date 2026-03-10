#!/usr/bin/env python3
"""
Sample 02: Custom Policy Exercise (Advanced)
K12 Level 1 (Elementary)

Create your own robot policy from scratch!

This advanced exercise demonstrates:
1. Creating a custom policy class
2. Implementing the get_actions() method
3. Registering a custom policy with the system
4. Using your policy like any built-in policy
5. Understanding policy parameters and control
"""

import numpy as np
from strands_robots.policy import Policy, register_policy

from strands_robots import Robot


class WavePolicy(Policy):
    """
    A custom policy that makes robots wave!

    This policy creates a smooth waving motion by using sinusoidal
    functions with different frequencies for different joints.

    Think of it as teaching the robot to wave hello! 👋
    """

    def __init__(
        self,
        frequency: float = 1.0,
        amplitude: float = 0.2,
        wave_joints: list[int] | None = None,
    ):
        """
        Initialize the wave policy.

        Args:
            frequency: How fast to wave (Hz). Higher = faster waving.
            amplitude: How far to wave (radians). Higher = bigger waves.
            wave_joints: Which joints should wave. If None, all joints wave.
        """
        super().__init__()
        self.frequency = frequency
        self.amplitude = amplitude
        self.wave_joints = wave_joints
        self.time = 0.0

        print("🌊 Wave Policy Created!")
        print(f"   • Frequency: {frequency} Hz")
        print(f"   • Amplitude: {amplitude} radians")
        print(f"   • Wave Joints: {wave_joints or 'all'}")

    async def get_actions(
        self, observations: list[dict], instruction: str | None = None
    ):
        """
        Generate waving actions for the robot.

        This is the key method that every policy must implement!

        Args:
            observations: List of observation dicts (one per robot)
            instruction: Optional text instruction (not used here)

        Returns:
            List of action dicts (one per robot)
        """
        actions = []

        for obs in observations:
            # Get the current joint positions from observations
            joint_positions = obs.get("joint_positions", np.array([]))
            num_joints = len(joint_positions)

            # Create wave motion
            # Use sine and cosine for smooth, rhythmic motion
            wave1 = self.amplitude * np.sin(2 * np.pi * self.frequency * self.time)
            wave2 = self.amplitude * np.cos(2 * np.pi * self.frequency * self.time)

            # Create action array
            action = np.zeros(num_joints)

            if self.wave_joints is None:
                # Wave all joints with alternating patterns
                for i in range(num_joints):
                    if i % 2 == 0:
                        action[i] = wave1
                    else:
                        action[i] = wave2
            else:
                # Only wave specified joints
                for joint_idx in self.wave_joints:
                    if joint_idx < num_joints:
                        action[joint_idx] = wave1

            # Create action dictionary
            # Format matches what the robot expects
            action_dict = {
                "joint_positions": action.tolist(),
            }

            # Add gripper action if the robot has grippers
            if "gripper_position" in obs:
                # Open and close gripper in rhythm with waves
                gripper_action = 0.5 + 0.5 * np.sin(
                    2 * np.pi * self.frequency * self.time
                )
                action_dict["gripper"] = [gripper_action]

            actions.append(action_dict)

        # Increment time for next step
        self.time += 0.01  # Assume 100 Hz control

        return actions


class HelloWavePolicy(Policy):
    """
    A friendly policy that makes a robot wave hello!

    This targets specific arm joints to create a natural waving motion,
    like a person waving hello to someone.
    """

    def __init__(self, arm: str = "right"):
        """
        Initialize the hello wave policy.

        Args:
            arm: Which arm to wave ("right" or "left")
        """
        super().__init__()
        self.arm = arm
        self.time = 0.0

        print("👋 Hello Wave Policy Created!")
        print(f"   • Waving arm: {arm}")

    async def get_actions(
        self, observations: list[dict], instruction: str | None = None
    ):
        """Generate a friendly waving motion."""
        actions = []

        for obs in observations:
            joint_positions = obs.get("joint_positions", np.array([]))
            num_joints = len(joint_positions)

            # Start with current positions (no movement)
            action = np.zeros(num_joints)

            # For humanoid robots, right arm is usually joints 3-6
            # Left arm is usually joints 7-10
            # (This varies by robot!)
            if self.arm == "right" and num_joints > 6:
                # Shoulder: lift arm up and down
                action[3] = 0.3 * np.sin(2 * np.pi * 0.5 * self.time)
                # Elbow: bend while waving
                action[4] = 0.2 * np.sin(2 * np.pi * 1.0 * self.time)
                # Wrist: rotate for wave
                action[5] = 0.4 * np.sin(2 * np.pi * 2.0 * self.time)

            elif self.arm == "left" and num_joints > 10:
                # Mirror the motion for left arm
                action[7] = 0.3 * np.sin(2 * np.pi * 0.5 * self.time)
                action[8] = 0.2 * np.sin(2 * np.pi * 1.0 * self.time)
                action[9] = 0.4 * np.sin(2 * np.pi * 2.0 * self.time)

            actions.append({"joint_positions": action.tolist()})

            self.time += 0.01

        return actions


def main():
    """Demonstrate custom policies."""
    print("=" * 60)
    print("🎨 Custom Policy Lab")
    print("=" * 60)
    print()
    print("Learn to create your own robot policies!")
    print()

    # Step 1: Register custom policies
    print("📝 Step 1: Registering Custom Policies")
    print("=" * 60)
    print()

    # Register the wave policy
    register_policy("wave", WavePolicy)
    print("✅ Registered: 'wave' policy")

    # Register the hello wave policy
    register_policy("hello_wave", HelloWavePolicy)
    print("✅ Registered: 'hello_wave' policy")

    print()
    print("Now you can use them just like built-in policies!")
    print()

    # Step 2: Test the wave policy
    print("📝 Step 2: Testing Wave Policy")
    print("=" * 60)
    print()

    sim = Robot("so100")

    print("Running wave policy (3 seconds)...")
    sim.run_policy(
        robot_name="so100",
        policy_provider="wave",
        duration=3.0,
    )
    print("✅ Wave policy completed!")
    print()

    # Record video
    print("🎥 Recording wave policy video...")
    sim.record_video(
        robot_name="so100",
        policy_provider="wave",
        duration=3.0,
        fps=30,
        output_path="custom_wave_policy.mp4",
    )
    print("✅ Video saved: custom_wave_policy.mp4")
    print()

    # Step 3: Test the hello wave policy
    print("📝 Step 3: Testing Hello Wave Policy")
    print("=" * 60)
    print()

    print("Running hello_wave policy (3 seconds)...")
    sim.run_policy(
        robot_name="so100",
        policy_provider="hello_wave",
        duration=3.0,
    )
    print("✅ Hello wave policy completed!")
    print()

    # Record video
    print("🎥 Recording hello wave policy video...")
    sim.record_video(
        robot_name="so100",
        policy_provider="hello_wave",
        duration=3.0,
        fps=30,
        output_path="custom_hello_wave_policy.mp4",
    )
    print("✅ Video saved: custom_hello_wave_policy.mp4")
    print()

    # Summary
    print("=" * 60)
    print("🎓 What You Built!")
    print("=" * 60)
    print()
    print("You created TWO custom policies:")
    print()
    print("1. WavePolicy")
    print("   • Makes all joints wave in a pattern")
    print("   • Customizable frequency and amplitude")
    print("   • Can target specific joints")
    print()
    print("2. HelloWavePolicy")
    print("   • Makes the robot wave hello with one arm")
    print("   • Targets specific arm joints")
    print("   • Creates natural, friendly motion")
    print()
    print("=" * 60)
    print("🚀 Challenges")
    print("=" * 60)
    print()
    print("1. Easy: Change the wave frequency to make it faster/slower")
    print()
    print("2. Medium: Create a 'dance' policy with complex motions")
    print()
    print("3. Hard: Create a policy that responds to the instruction")
    print("   parameter (e.g., 'wave fast' vs 'wave slow')")
    print()
    print("4. Expert: Create a policy that uses observations to")
    print("   avoid obstacles or balance")
    print()
    print("=" * 60)
    print("💡 Key Concepts")
    print("=" * 60)
    print()
    print("Creating a custom policy requires:")
    print()
    print("  ✅ Inherit from Policy base class")
    print("  ✅ Implement async get_actions(observations, instruction)")
    print("  ✅ Return list of action dictionaries")
    print("  ✅ Register with register_policy(name, PolicyClass)")
    print()
    print("Then use it just like any built-in policy!")
    print()
    print("Happy policy creating! 🎨🤖")
    print()


if __name__ == "__main__":
    main()
