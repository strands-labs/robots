"""Policy execution — run_policy (blocking) and start_policy (async)."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from ._rendering import apply_sim_action, get_sim_observation
from ._types import TrajectoryStep

if TYPE_CHECKING:
    from ._core import MujocoBackend

logger = logging.getLogger(__name__)


def run_policy(
    sim: MujocoBackend,
    robot_name: str,
    policy_provider: str = "mock",
    instruction: str = "",
    duration: float = 10.0,
    action_horizon: int = 8,
    control_frequency: float = 50.0,
    fast_mode: bool = False,
    **policy_kwargs,
) -> Dict[str, Any]:
    """Run a policy on a simulated robot (blocking). Same Policy ABC as robot.py."""
    if sim._world is None or sim._world._data is None:
        return {"status": "error", "content": [{"text": "❌ No simulation."}]}
    if robot_name not in sim._world.robots:
        return {
            "status": "error",
            "content": [{"text": f"❌ Robot '{robot_name}' not found."}],
        }

    robot = sim._world.robots[robot_name]

    try:
        from strands_robots._async_utils import _resolve_coroutine
        from strands_robots.policies import create_policy as _create_policy

        policy = _create_policy(policy_provider, **policy_kwargs)
        policy.set_robot_state_keys(robot.joint_names)

        robot.policy_running = True
        robot.policy_instruction = instruction
        robot.policy_steps = 0

        start_time = time.time()
        action_sleep = 1.0 / control_frequency

        while time.time() - start_time < duration and robot.policy_running:
            observation = get_sim_observation(sim, robot_name)

            coro_or_result = policy.get_actions(observation, instruction)
            actions = _resolve_coroutine(coro_or_result)

            for action_dict in actions[:action_horizon]:
                if not robot.policy_running:
                    break

                # Record trajectory if recording
                if sim._world._recording:
                    sim._world._trajectory.append(
                        TrajectoryStep(
                            timestamp=time.time(),
                            sim_time=sim._world.sim_time,
                            robot_name=robot_name,
                            observation={
                                k: v
                                for k, v in observation.items()
                                if not isinstance(v, np.ndarray)
                            },
                            action=action_dict,
                            instruction=instruction,
                        )
                    )
                    if sim._world._dataset_recorder is not None:
                        sim._world._dataset_recorder.add_frame(
                            observation=observation,
                            action=action_dict,
                            task=instruction,
                        )

                apply_sim_action(sim, robot_name, action_dict)
                robot.policy_steps += 1
                if not fast_mode:
                    time.sleep(action_sleep)

        elapsed = time.time() - start_time
        robot.policy_running = False

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"✅ Policy complete on '{robot_name}'\n"
                        f"🧠 {policy_provider} | 🎯 {instruction}\n"
                        f"⏱️ {elapsed:.1f}s | 📊 {robot.policy_steps} steps | "
                        f"🕐 sim_t={sim._world.sim_time:.3f}s"
                    )
                }
            ],
        }

    except Exception as e:
        robot.policy_running = False
        return {"status": "error", "content": [{"text": f"❌ Policy failed: {e}"}]}


def start_policy(
    sim: MujocoBackend,
    robot_name: str,
    policy_provider: str = "mock",
    instruction: str = "",
    duration: float = 10.0,
    fast_mode: bool = False,
    **policy_kwargs,
) -> Dict[str, Any]:
    """Start policy execution in background (non-blocking)."""
    if sim._world is None or sim._world._data is None:
        return {"status": "error", "content": [{"text": "❌ No simulation."}]}
    if robot_name not in sim._world.robots:
        return {
            "status": "error",
            "content": [{"text": f"❌ Robot '{robot_name}' not found."}],
        }

    future = sim._executor.submit(
        run_policy,
        sim,
        robot_name,
        policy_provider,
        instruction,
        duration,
        fast_mode=fast_mode,
        **policy_kwargs,
    )
    sim._policy_threads[robot_name] = future

    return {
        "status": "success",
        "content": [{"text": f"🚀 Policy started on '{robot_name}' (async)"}],
    }
