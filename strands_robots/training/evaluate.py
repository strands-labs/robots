"""Standalone evaluation harness for any policy on any task."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _error_result(provider_name: str, error: str) -> Dict[str, Any]:
    return {
        "success_rate": 0.0,
        "mean_reward": 0.0,
        "num_episodes": 0,
        "episodes": [],
        "policy_provider": provider_name,
        "error": error,
    }


def _create_env(backend, robot_name, task, max_steps_per_episode, render, kwargs):
    """Create a simulation environment for the given backend.

    Returns the env, or an error dict if creation fails.
    """
    provider_name = "unknown"

    if backend == "newton":
        try:
            from strands_robots.newton import NewtonConfig
            from strands_robots.newton.newton_gym_env import NewtonGymEnv

            newton_kwargs = {}
            if "newton_config" in kwargs:
                newton_kwargs["config"] = kwargs.pop("newton_config")
            elif "num_envs" in kwargs or "solver" in kwargs:
                newton_kwargs["config"] = NewtonConfig(
                    num_envs=kwargs.pop("num_envs", 1),
                    solver=kwargs.pop("solver", "mujoco"),
                    device=kwargs.pop("device", "cuda:0"),
                )

            env = NewtonGymEnv(
                robot_name=robot_name,
                task=task,
                render_mode="rgb_array" if render else None,
                max_episode_steps=max_steps_per_episode,
                **newton_kwargs,
            )
            logger.info(
                "🚀 Using Newton GPU backend for evaluation (solver=%s)",
                env._config.solver,
            )
            return env
        except ImportError as e:
            logger.warning("Newton backend not available: %s", e)
            return _error_result(
                provider_name,
                f"Newton backend requires newton and warp-lang: {e}",
            )
        except Exception as e:
            logger.warning("Newton env creation failed: %s", e)
            return _error_result(provider_name, f"Newton environment creation failed: {e}")

    elif backend == "isaac":
        try:
            from strands_robots.isaac.isaac_gym_env import IsaacGymEnv

            isaac_kwargs = {}
            if "num_envs" in kwargs:
                isaac_kwargs["num_envs"] = kwargs.pop("num_envs")
            if "device" in kwargs:
                isaac_kwargs["device"] = kwargs.pop("device")

            env = IsaacGymEnv(
                robot_name=robot_name,
                task=task,
                render_mode="rgb_array" if render else None,
                max_episode_steps=max_steps_per_episode,
                **isaac_kwargs,
            )
            logger.info("🚀 Using Isaac Sim GPU backend for evaluation")
            return env
        except ImportError as e:
            logger.warning("Isaac backend not available: %s", e)
            return _error_result(provider_name, f"Isaac backend requires Isaac Sim: {e}")
        except Exception as e:
            logger.warning("Isaac env creation failed: %s", e)
            return _error_result(provider_name, f"Isaac environment creation failed: {e}")

    else:
        try:
            from strands_robots.envs import StrandsSimEnv

            env_kwargs = {
                "robot_name": robot_name,
                "task": task,
                "max_episode_steps": max_steps_per_episode,
                "render_mode": "rgb_array" if render else None,
            }
            env_kwargs.update(kwargs)
            return StrandsSimEnv(**env_kwargs)
        except ImportError:
            return _error_result(provider_name, "gymnasium and mujoco required for evaluation")


def evaluate(
    policy,
    task: str,
    robot_name: str = "",
    num_episodes: int = 50,
    max_steps_per_episode: int = 1000,
    backend: str = "mujoco",
    render: bool = False,
    seed: int = 42,
    env=None,
    **kwargs,
) -> Dict[str, Any]:
    """Standalone evaluation harness for any policy on any task.

    Runs N episodes of a policy in simulation and reports success rate,
    mean reward, and per-episode statistics.

    Args:
        policy: A Policy instance (from create_policy) or any object with
                get_actions(observation_dict, instruction) method.
        task: Natural language task description.
        robot_name: Robot model name (e.g. "so100", "unitree_g1").
        num_episodes: Number of evaluation episodes.
        max_steps_per_episode: Maximum steps per episode before truncation.
        backend: Simulation backend — "mujoco" (default), "newton", or "isaac".
        render: Whether to render frames during evaluation.
        seed: Random seed for reproducibility.
        env: Pre-created gymnasium environment. If provided, ``backend``,
            ``robot_name``, and env-related kwargs are ignored.
        **kwargs: Additional kwargs passed to the environment.

    Returns:
        Dict with:
            - success_rate: float (0-100)
            - mean_reward: float
            - num_episodes: int
            - episodes: List[Dict] with per-episode stats
            - policy_provider: str
    """
    import asyncio
    import inspect

    import numpy as np

    episodes = []
    successes = 0
    total_reward = 0.0

    owns_env = env is None
    if owns_env:
        env = _create_env(backend, robot_name, task, max_steps_per_episode, render, kwargs)
        if isinstance(env, dict):
            return env

    _async_loop = None
    if inspect.iscoroutinefunction(getattr(policy, "get_actions", None)):
        _async_loop = asyncio.new_event_loop()

    try:
        for ep in range(num_episodes):
            obs, info = env.reset(seed=seed + ep)
            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                if isinstance(obs, dict):
                    obs_dict = obs
                else:
                    obs_dict = {"observation.state": obs}

                try:
                    if _async_loop is not None:
                        actions = _async_loop.run_until_complete(policy.get_actions(obs_dict, task))
                    else:
                        actions = policy.get_actions(obs_dict, task)
                except Exception as e:
                    logger.debug("Policy error at step %s: %s", episode_steps, e)
                    actions = [{}]

                if isinstance(actions, list) and len(actions) > 0:
                    action_dict = actions[0]
                    if isinstance(action_dict, dict):
                        action_vec = list(action_dict.values())
                        action_vec = [v for v in action_vec if isinstance(v, (int, float))]
                        action = np.array(action_vec, dtype=np.float32)
                    else:
                        action = np.array(action_dict, dtype=np.float32)
                else:
                    action = env.action_space.sample() * 0

                action = (
                    np.clip(
                        action[: env.action_space.shape[0]],
                        env.action_space.low,
                        env.action_space.high,
                    )
                    if len(action) >= env.action_space.shape[0]
                    else env.action_space.sample() * 0
                )

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1
                done = terminated or truncated

            is_success = info.get("is_success", False)
            if is_success:
                successes += 1
            total_reward += episode_reward

            episodes.append(
                {
                    "episode": ep,
                    "steps": episode_steps,
                    "reward": episode_reward,
                    "success": is_success,
                }
            )
    finally:
        if _async_loop is not None:
            _async_loop.close()
        if owns_env:
            env.close()

    success_rate = (successes / num_episodes * 100) if num_episodes > 0 else 0.0
    mean_reward = total_reward / num_episodes if num_episodes > 0 else 0.0

    logger.info(
        "📊 Evaluation: %s episodes, success=%.1f%%, reward=%.2f",
        num_episodes,
        success_rate,
        mean_reward,
    )

    return {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "num_episodes": num_episodes,
        "episodes": episodes,
        "policy_provider": getattr(policy, "provider_name", "unknown"),
        "task": task,
        "robot_name": robot_name,
    }
