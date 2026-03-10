#!/usr/bin/env python3
"""
Sample 02 (bonus): Anatomy of a Policy — create your own custom policy.

Demonstrates:
1. The Policy ABC (abstract base class)
2. How to subclass it
3. Registering a custom policy with the registry
4. Using it in simulation via the standard interface

Requirements:
    pip install strands-robots[sim]
"""

from typing import Any, Dict, List

from strands_robots import Robot, register_policy
from strands_robots.policies import Policy


# ─── Define a custom policy ──────────────────────────────────────────
class GoToCenterPolicy(Policy):
    """A simple policy that moves all joints toward zero (center position).

    This demonstrates the Policy ABC:
    - provider_name: identifies the policy
    - set_robot_state_keys: configures which joints to control
    - get_actions: the brain — takes observations, returns actions
    """

    def __init__(self, speed: float = 0.1, **kwargs):
        self.speed = speed
        self.robot_state_keys: List[str] = []

    @property
    def provider_name(self) -> str:
        return "go_to_center"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self.robot_state_keys = robot_state_keys
        print(f"   🔧 Configured for {len(robot_state_keys)} joints: {robot_state_keys}")

    async def get_actions(
        self, observation_dict: Dict[str, Any], instruction: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Move each joint toward zero at a fixed speed."""
        actions = []
        for i in range(4):  # 4-step action horizon
            action = {}
            for key in self.robot_state_keys:
                # Get current position from observation (default to 0)
                state = observation_dict.get("observation.state", {})
                if hasattr(state, "__len__"):
                    idx = self.robot_state_keys.index(key) if key in self.robot_state_keys else 0
                    current = float(state[idx]) if idx < len(state) else 0.0
                else:
                    current = 0.0
                # Move toward zero
                action[key] = current - self.speed * (1.0 if current > 0 else -1.0)
            actions.append(action)
        return actions


# ─── Register it ─────────────────────────────────────────────────────
register_policy("go_to_center", lambda: GoToCenterPolicy)
print("✅ Registered 'go_to_center' policy")

# ─── Use it via the standard interface ────────────────────────────────
print("\n🤖 Creating SO-100 robot...")
sim = Robot("so100")

# First, move the robot to an interesting pose with mock
print("\n🎭 Running mock policy to get an interesting pose...")
sim.run_policy(robot_name="so100", policy_provider="mock", duration=1.0)

# Now use our custom policy to center it
print("\n🎯 Running go_to_center policy...")
result = sim.run_policy(
    robot_name="so100",
    policy_provider="go_to_center",
    duration=2.0,
    speed=0.05,
)
print(f"   {result['content'][0]['text']}")

sim.destroy()
print("\n🎉 You just created and ran a custom policy!")
print("   The same register_policy() mechanism powers all 17+ built-in providers.")
