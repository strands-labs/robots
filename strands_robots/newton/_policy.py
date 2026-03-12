"""Policy execution — run_policy and policy creation."""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ._core import NewtonBackend


def run_policy(
    backend: NewtonBackend,
    robot_name: str,
    policy_provider: str = "mock",
    instruction: str = "",
    duration: float = 10.0,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run a policy loop for the given robot."""
    if robot_name not in backend._robots:
        return {"success": False, "message": f"Robot '{robot_name}' not found."}

    if backend._model is None:
        from ._scene import finalize_model

        finalize_model(backend)

    policy = _create_policy(backend, robot_name, policy_provider, instruction, **kwargs)
    dt = backend._config.physics_dt * backend._config.substeps
    num_steps = int(math.ceil(duration / dt))

    trajectory = []
    errors = []
    t0 = time.perf_counter()

    for i in range(num_steps):
        obs = backend.get_observation(robot_name)
        try:
            actions = policy.get_actions(
                obs.get("observations", {}).get(robot_name, {}), instruction
            )
            backend.step(actions)
            trajectory.append(
                {"step": i, "sim_time": backend._sim_time, "observation": obs}
            )
        except Exception as exc:
            errors.append(str(exc))

    wall_time = time.perf_counter() - t0
    return {
        "success": True,
        "steps_executed": backend._step_count,
        "sim_time": backend._sim_time,
        "wall_time": wall_time,
        "realtime_factor": (
            float(backend._sim_time / wall_time) if wall_time > 0 else 0.0
        ),
        "trajectory": trajectory,
        "errors": errors,
    }


def _create_policy(
    backend: NewtonBackend,
    robot_name: str,
    policy_provider: str,
    instruction: str = "",
    **kwargs: Any,
):
    """Create a policy instance for the given robot."""
    n_joints = backend._robots.get(robot_name, {}).get("num_joints", 0)

    if policy_provider == "mock":
        return _MockPolicy(n_joints)

    try:
        from strands_robots.policies import create_policy

        return create_policy(
            provider=policy_provider,
            robot_name=robot_name,
            instruction=instruction,
            **kwargs,
        )
    except Exception:
        return _MockPolicy(n_joints)


class _MockPolicy:
    """Zero-action mock policy for testing."""

    def __init__(self, n_joints: int):
        self._n_joints = n_joints

    def get_actions(self, obs, instr):
        return np.zeros(self._n_joints, dtype=np.float32)
