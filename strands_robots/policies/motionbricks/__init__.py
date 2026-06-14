"""MotionBricks generative motion policy provider.

Real-time generative motion model (15,000 FPS) for the G1 humanoid.
Three-component architecture: VQVAE motion tokenizer + pose model + root model.
Outputs full-body qpos (29-DOF G1) at MuJoCo timestep.

Supports 12+ locomotion styles: walk, zombie walk, happy dance, stealth walk,
injured gait, hand/elbow crawling, boxing walk, gun walk, scared walk, etc.

Reference: https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks
License: Apache 2.0 (code), NVIDIA Open Model License (weights)

Usage::

    from strands_robots.policies import create_policy

    policy = create_policy(
        "motionbricks",
        checkpoint="nvidia/MotionBricks-G1",
        style="walk_zombie",
    )
    policy.set_robot_state_keys(MOTIONBRICKS_JOINT_NAMES)
    actions = policy.get_actions_sync(obs, "", target_velocity=[0.5, 0, 0])
"""

from strands_robots.policies.motionbricks.motionbricks_policy import (
    MOTIONBRICKS_JOINT_NAMES,
    MOTIONBRICKS_STYLES,
    MotionBricksPolicy,
)

__all__ = ["MotionBricksPolicy", "MOTIONBRICKS_JOINT_NAMES", "MOTIONBRICKS_STYLES"]
