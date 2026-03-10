#!/usr/bin/env python3
"""Sample 08 — Safety Wrapper for Hardware Deployment.

Provides safety utilities for deploying trained policies to physical robots:
    - Joint position limits (per-robot)
    - Joint velocity caps
    - Workspace boundary enforcement
    - Emergency stop integration
    - Configurable via YAML

These checks wrap the policy output BEFORE it reaches the actuators.
If any limit is violated, the command is clamped (soft limit) or the
robot is stopped (hard limit).

Usage:
    safety = SafetyWrapper.from_yaml("configs/deploy_so100.yaml")
    safe_actions = safety.check(raw_actions, current_state)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["SafetyWrapper", "SafetyConfig", "SafetyViolation"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class JointLimits:
    """Per-joint position and velocity limits (radians, rad/s)."""

    name: str
    pos_min: float = -math.pi
    pos_max: float = math.pi
    vel_max: float = 2.0  # rad/s


@dataclass
class WorkspaceBounds:
    """Cartesian bounding box for the end-effector (metres)."""

    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    z_min: float = 0.0   # Don't go below the table
    z_max: float = 1.0


@dataclass
class SafetyConfig:
    """Full safety configuration for a robot."""

    robot_type: str = "so100"
    joint_limits: List[JointLimits] = field(default_factory=list)
    workspace: WorkspaceBounds = field(default_factory=WorkspaceBounds)
    max_velocity_scale: float = 1.0  # Global velocity multiplier (0–1)
    emergency_stop_on_violation: bool = False  # Hard stop vs. soft clamp


class SafetyViolation(Exception):
    """Raised when a hard safety limit is violated."""


# ---------------------------------------------------------------------------
# Safety wrapper
# ---------------------------------------------------------------------------

class SafetyWrapper:
    """Wraps raw policy actions with safety checks.

    Designed to sit between the policy output and the robot actuator
    commands.  All checks happen at ~50 Hz (the control loop rate).
    """

    def __init__(self, config: SafetyConfig) -> None:
        self.config = config
        self._violation_count = 0

    # -- Factory ----------------------------------------------------------

    @classmethod
    def from_yaml(cls, yaml_path: str) -> SafetyWrapper:
        """Load safety config from a YAML file.

        Expected format::

            robot_type: so100
            max_velocity_scale: 0.5
            emergency_stop_on_violation: false
            workspace:
              x_min: -0.5
              ...
            joints:
              - name: joint_1
                pos_min: -2.6
                pos_max: 2.6
                vel_max: 1.5
              ...
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")

        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        joints = [
            JointLimits(
                name=j["name"],
                pos_min=j.get("pos_min", -math.pi),
                pos_max=j.get("pos_max", math.pi),
                vel_max=j.get("vel_max", 2.0),
            )
            for j in raw.get("joints", [])
        ]

        ws_raw = raw.get("workspace", {})
        workspace = WorkspaceBounds(**{k: ws_raw[k] for k in ws_raw if hasattr(WorkspaceBounds, k)})

        config = SafetyConfig(
            robot_type=raw.get("robot_type", "so100"),
            joint_limits=joints,
            workspace=workspace,
            max_velocity_scale=raw.get("max_velocity_scale", 1.0),
            emergency_stop_on_violation=raw.get("emergency_stop_on_violation", False),
        )
        return cls(config)

    @classmethod
    def for_robot(cls, robot_type: str) -> SafetyWrapper:
        """Create a default safety wrapper for common robot types."""
        defaults: Dict[str, SafetyConfig] = {
            "so100": SafetyConfig(
                robot_type="so100",
                joint_limits=[
                    JointLimits("joint_1", -2.6, 2.6, 1.5),
                    JointLimits("joint_2", -1.8, 1.8, 1.5),
                    JointLimits("joint_3", -2.6, 2.6, 1.5),
                    JointLimits("joint_4", -1.8, 1.8, 1.5),
                    JointLimits("joint_5", -2.6, 2.6, 1.5),
                    JointLimits("gripper", 0.0, 1.0, 2.0),
                ],
                max_velocity_scale=0.5,
            ),
            "g1": SafetyConfig(
                robot_type="g1",
                joint_limits=[
                    JointLimits(f"joint_{i}", -3.14, 3.14, 2.0)
                    for i in range(29)
                ],
                max_velocity_scale=0.3,  # Start slow with humanoid
                emergency_stop_on_violation=True,
            ),
            "reachy_mini": SafetyConfig(
                robot_type="reachy_mini",
                joint_limits=[
                    JointLimits(f"joint_{i}", -2.0, 2.0, 1.0)
                    for i in range(8)
                ],
                max_velocity_scale=0.5,
            ),
        }
        config = defaults.get(robot_type, SafetyConfig(robot_type=robot_type))
        return cls(config)

    # -- Core check -------------------------------------------------------

    def check(
        self,
        actions: np.ndarray,
        current_positions: Optional[np.ndarray] = None,
        dt: float = 0.02,  # 50 Hz
    ) -> np.ndarray:
        """Validate and clamp raw policy actions.

        Args:
            actions: Raw joint-position targets from the policy, shape (N,).
            current_positions: Current joint positions, shape (N,).  Used
                to compute velocity and enforce velocity limits.
            dt: Control loop timestep in seconds.

        Returns:
            Safe actions (clamped to limits).

        Raises:
            SafetyViolation: If ``emergency_stop_on_violation`` is True
                and a hard limit is hit.
        """
        safe = np.array(actions, dtype=np.float64)

        # 1. Joint position limits
        for i, jl in enumerate(self.config.joint_limits):
            if i >= len(safe):
                break
            if safe[i] < jl.pos_min or safe[i] > jl.pos_max:
                self._violation_count += 1
                if self.config.emergency_stop_on_violation:
                    raise SafetyViolation(
                        f"Joint {jl.name} position {safe[i]:.3f} "
                        f"outside [{jl.pos_min:.3f}, {jl.pos_max:.3f}]"
                    )
                safe[i] = np.clip(safe[i], jl.pos_min, jl.pos_max)
                logger.debug("Clamped %s to [%.3f, %.3f]", jl.name, jl.pos_min, jl.pos_max)

        # 2. Velocity limits
        if current_positions is not None:
            velocity = (safe - current_positions) / dt
            for i, jl in enumerate(self.config.joint_limits):
                if i >= len(velocity):
                    break
                max_vel = jl.vel_max * self.config.max_velocity_scale
                if abs(velocity[i]) > max_vel:
                    self._violation_count += 1
                    # Clamp to max allowed delta
                    max_delta = max_vel * dt
                    delta = safe[i] - current_positions[i]
                    safe[i] = current_positions[i] + np.clip(delta, -max_delta, max_delta)
                    logger.debug(
                        "Velocity-clamped %s: %.3f rad/s → %.3f rad/s",
                        jl.name,
                        velocity[i],
                        max_vel,
                    )

        return safe

    def check_workspace(
        self,
        ee_position: Sequence[float],
    ) -> bool:
        """Check if the end-effector is within the workspace bounds.

        Args:
            ee_position: (x, y, z) in metres.

        Returns:
            True if within bounds; False (or raises) if outside.
        """
        ws = self.config.workspace
        x, y, z = ee_position[:3]
        in_bounds = (
            ws.x_min <= x <= ws.x_max
            and ws.y_min <= y <= ws.y_max
            and ws.z_min <= z <= ws.z_max
        )
        if not in_bounds:
            self._violation_count += 1
            if self.config.emergency_stop_on_violation:
                raise SafetyViolation(
                    f"End-effector at ({x:.3f}, {y:.3f}, {z:.3f}) "
                    f"outside workspace bounds"
                )
            logger.warning(
                "End-effector out of workspace: (%.3f, %.3f, %.3f)", x, y, z
            )
        return in_bounds

    @property
    def violation_count(self) -> int:
        """Total number of safety violations detected so far."""
        return self._violation_count

    def reset(self) -> None:
        """Reset the violation counter (e.g. between episodes)."""
        self._violation_count = 0

    def __repr__(self) -> str:
        return (
            f"SafetyWrapper(robot={self.config.robot_type}, "
            f"joints={len(self.config.joint_limits)}, "
            f"vel_scale={self.config.max_velocity_scale}, "
            f"violations={self._violation_count})"
        )


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

def main() -> None:
    """Quick demo of the safety wrapper."""
    print("SafetyWrapper demo")
    print("=" * 40)

    safety = SafetyWrapper.for_robot("so100")
    print(f"Created: {safety}")

    # Simulate a safe action
    current = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    action = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.6])
    safe = safety.check(action, current)
    print(f"\nSafe action:    {action} → {safe}  (violations: {safety.violation_count})")

    # Simulate a dangerous action (exceeds limits)
    dangerous = np.array([5.0, -3.0, 0.1, 0.1, 0.1, 2.0])
    safe2 = safety.check(dangerous, current)
    print(f"Clamped action: {dangerous} → {safe2}  (violations: {safety.violation_count})")

    # Workspace check
    in_bounds = safety.check_workspace([0.2, 0.1, 0.15])
    print(f"\nWorkspace check (0.2, 0.1, 0.15): {'✅ OK' if in_bounds else '❌ Out'}")

    out_bounds = safety.check_workspace([0.2, 0.1, -0.5])
    print(f"Workspace check (0.2, 0.1, -0.5): {'✅ OK' if out_bounds else '❌ Out'}")

    print(f"\nTotal violations: {safety.violation_count}")


if __name__ == "__main__":
    main()
