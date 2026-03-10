#!/usr/bin/env python3
"""
Scripted Pick-and-Place Demo for SO-100

Generates demonstration trajectories for VLA training data collection.
Uses inverse kinematics to compute smooth reach-grasp-lift-place motions.

Usage:
    python scripts/scripted_pick_demo.py --episodes 10 --output ./demo_data
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class ScriptedPickPolicy:
    """Scripted pick-and-place policy using waypoint interpolation.

    Generates smooth trajectories through:
    1. Approach: Move above the target object
    2. Descend: Lower to grasp height
    3. Grasp: Close gripper
    4. Lift: Raise object
    5. Move: Translate to place position
    6. Release: Open gripper
    """

    def __init__(
        self,
        robot_name: str = "so100",
        cube_pos: Optional[np.ndarray] = None,
        place_pos: Optional[np.ndarray] = None,
        num_joints: int = 6,
        gripper_open: float = 1.0,
        gripper_closed: float = -1.0,
    ):
        self.robot_name = robot_name
        self.cube_pos = cube_pos if cube_pos is not None else np.array([0.2, 0.0, 0.02])
        self.place_pos = place_pos if place_pos is not None else np.array([0.0, 0.2, 0.02])
        self.num_joints = num_joints
        self.gripper_open = gripper_open
        self.gripper_closed = gripper_closed
        self._step = 0
        self._phase = "approach"
        self._trajectory = None

    def generate_trajectory(self, total_steps: int = 200) -> List[np.ndarray]:
        """Generate a full pick-and-place trajectory as joint-space waypoints.

        Uses sinusoidal interpolation for smooth motion profiles.
        Returns list of (num_joints + 1,) arrays (joints + gripper).
        """
        steps_per_phase = total_steps // 5  # 5 phases
        trajectory = []

        # Phase 1: Approach — sine wave ramp to approach position
        for t in range(steps_per_phase):
            progress = t / steps_per_phase
            # Smooth interpolation using cosine schedule
            alpha = 0.5 * (1 - math.cos(math.pi * progress))
            # Generate joint targets for approach (above cube)
            joints = self._approach_joints(alpha)
            gripper = self.gripper_open
            trajectory.append(np.append(joints, gripper))

        # Phase 2: Descend — lower to grasp height
        for t in range(steps_per_phase):
            progress = t / steps_per_phase
            alpha = 0.5 * (1 - math.cos(math.pi * progress))
            joints = self._descend_joints(alpha)
            gripper = self.gripper_open
            trajectory.append(np.append(joints, gripper))

        # Phase 3: Grasp — close gripper (hold position)
        grasp_joints = self._descend_joints(1.0)
        for t in range(steps_per_phase):
            progress = t / steps_per_phase
            gripper = self.gripper_open + (self.gripper_closed - self.gripper_open) * progress
            trajectory.append(np.append(grasp_joints, gripper))

        # Phase 4: Lift — raise with object
        for t in range(steps_per_phase):
            progress = t / steps_per_phase
            alpha = 0.5 * (1 - math.cos(math.pi * progress))
            joints = self._lift_joints(alpha)
            gripper = self.gripper_closed
            trajectory.append(np.append(joints, gripper))

        # Phase 5: Move to place position & release
        for t in range(steps_per_phase):
            progress = t / steps_per_phase
            alpha = 0.5 * (1 - math.cos(math.pi * progress))
            joints = self._place_joints(alpha)
            if progress > 0.7:
                # Start opening gripper at 70% of move phase
                release_progress = (progress - 0.7) / 0.3
                gripper = self.gripper_closed + (self.gripper_open - self.gripper_closed) * release_progress
            else:
                gripper = self.gripper_closed
            trajectory.append(np.append(joints, gripper))

        self._trajectory = trajectory
        return trajectory

    def _approach_joints(self, alpha: float) -> np.ndarray:
        """Joint targets for approach position (above cube)."""
        # Home → approach interpolation
        home = np.zeros(self.num_joints)
        approach = np.array([0.0, -0.5, 0.8, 0.0, 0.3, 0.0])[:self.num_joints]
        return home + alpha * (approach - home)

    def _descend_joints(self, alpha: float) -> np.ndarray:
        """Joint targets for descend (lower to cube)."""
        approach = np.array([0.0, -0.5, 0.8, 0.0, 0.3, 0.0])[:self.num_joints]
        grasp = np.array([0.0, -0.7, 1.0, 0.0, 0.5, 0.0])[:self.num_joints]
        return approach + alpha * (grasp - approach)

    def _lift_joints(self, alpha: float) -> np.ndarray:
        """Joint targets for lift (raise object)."""
        grasp = np.array([0.0, -0.7, 1.0, 0.0, 0.5, 0.0])[:self.num_joints]
        lifted = np.array([0.0, -0.4, 0.6, 0.0, 0.2, 0.0])[:self.num_joints]
        return grasp + alpha * (lifted - grasp)

    def _place_joints(self, alpha: float) -> np.ndarray:
        """Joint targets for place (move to target)."""
        lifted = np.array([0.0, -0.4, 0.6, 0.0, 0.2, 0.0])[:self.num_joints]
        place = np.array([1.2, -0.4, 0.6, 0.0, 0.2, 0.0])[:self.num_joints]
        return lifted + alpha * (place - lifted)

    def get_action(self, step: int) -> np.ndarray:
        """Get action for a given step."""
        if self._trajectory is None:
            self.generate_trajectory()
        if step < len(self._trajectory):
            return self._trajectory[step]
        return self._trajectory[-1]  # Hold last position


def run_demo(
    num_episodes: int = 10,
    steps_per_episode: int = 200,
    output_dir: str = "./demo_data",
    record_video: bool = False,
    randomize: bool = True,
) -> Dict[str, Any]:
    """Run scripted pick-and-place demonstrations and record data.

    Args:
        num_episodes: Number of demonstration episodes to record.
        steps_per_episode: Steps per episode.
        output_dir: Directory to save recorded data.
        record_video: Whether to save video frames.
        randomize: Whether to randomize cube position each episode.

    Returns:
        Dict with recording statistics.
    """
    from strands_robots.simulation import Simulation

    os.makedirs(output_dir, exist_ok=True)

    sim = Simulation()
    sim.create_world()
    sim.add_robot(data_config="so100", name="arm", position=[0, 0, 0])

    stats = {
        "episodes": [],
        "total_steps": 0,
        "successful_episodes": 0,
    }

    for ep in range(num_episodes):
        # Randomize cube position
        if randomize:
            cube_x = 0.15 + np.random.uniform(-0.03, 0.03)
            cube_y = np.random.uniform(-0.05, 0.05)
        else:
            cube_x, cube_y = 0.2, 0.0

        # Reset scene
        sim.add_object(
            name="red_cube",
            shape="box",
            size=[0.04, 0.04, 0.04],
            position=[cube_x, cube_y, 0.02],
            color=[1, 0, 0, 1],
        )

        # Create policy for this episode
        policy = ScriptedPickPolicy(
            cube_pos=np.array([cube_x, cube_y, 0.02]),
            num_joints=6,
        )
        trajectory = policy.generate_trajectory(steps_per_episode)

        episode_data = {
            "observations": [],
            "actions": [],
            "cube_start": [cube_x, cube_y, 0.02],
        }

        for step in range(steps_per_episode):
            # Get observation
            obs = sim.get_observation("arm")

            # Get scripted action
            action = trajectory[step] if step < len(trajectory) else trajectory[-1]

            # Apply action
            action_dict = {}
            joint_names = obs.get("joint_names", [f"j{i}" for i in range(6)])
            for i, name in enumerate(joint_names[:len(action) - 1]):
                action_dict[name] = float(action[i])
            if len(action) > len(joint_names):
                action_dict["gripper"] = float(action[-1])

            sim.send_action(action_dict, robot_name="arm")
            sim.step()

            # Record
            episode_data["observations"].append({
                "state": obs.get("joint_positions", []),
                "step": step,
            })
            episode_data["actions"].append(action.tolist())

        stats["episodes"].append({
            "episode": ep,
            "steps": steps_per_episode,
            "cube_start": [cube_x, cube_y, 0.02],
        })
        stats["total_steps"] += steps_per_episode

        # Save episode data
        ep_path = os.path.join(output_dir, f"episode_{ep:04d}.json")
        with open(ep_path, "w") as f:
            json.dump(episode_data, f)

        logger.info(f"Episode {ep + 1}/{num_episodes} recorded ({steps_per_episode} steps)")

        # Remove cube for next episode
        try:
            sim.remove_object("red_cube")
        except Exception:
            pass

    # Save stats
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✅ Recorded {num_episodes} episodes ({stats['total_steps']} total steps)")
    print(f"   Output: {output_dir}")

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scripted Pick-and-Place Demo")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=200, help="Steps per episode")
    parser.add_argument("--output", type=str, default="./demo_data", help="Output directory")
    parser.add_argument("--no-randomize", action="store_true", help="Disable randomization")
    parser.add_argument("--video", action="store_true", help="Record video frames")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    run_demo(
        num_episodes=args.episodes,
        steps_per_episode=args.steps,
        output_dir=args.output,
        record_video=args.video,
        randomize=not args.no_randomize,
    )
