"""Gymnasium wrapper for Newton GPU-accelerated simulation.

Bridges NewtonBackend's batched GPU API into the standard gym.Env interface
so that Stable Baselines 3, CleanRL, and other RL libraries work out-of-the-box.

For single-env cases, implements gym.Env.
For multi-env (num_envs > 1), observations/actions are batched along the first
axis (shape: [num_envs, ...]).

Usage:
    from strands_robots.newton import NewtonConfig
    from strands_robots.newton.newton_gym_env import NewtonGymEnv

    # Single env
    env = NewtonGymEnv(
        robot_name="so100",
        task="pick up the red cube",
        config=NewtonConfig(num_envs=1, solver="mujoco"),
    )
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

    # Vectorized (4096 parallel envs)
    vec_env = NewtonGymEnv(
        robot_name="unitree_g1",
        task="walk forward at 1 m/s",
        config=NewtonConfig(num_envs=4096, solver="featherstone"),
    )
    obs, info = vec_env.reset()
    obs, reward, terminated, truncated, info = vec_env.step(actions)  # batched
"""

import logging
from typing import Any, Callable, List, Optional

import numpy as np

from strands_robots.gym_env import HAS_GYM

logger = logging.getLogger(__name__)


if HAS_GYM:
    from strands_robots.gym_env import BaseGymEnv

    class NewtonGymEnv(BaseGymEnv):
        """Gymnasium wrapper around NewtonBackend for RL training.

        Implements the standard gym.Env interface. When num_envs > 1,
        observations and actions are batched along the first axis
        (shape: [num_envs, ...]) compatible with SB3's VecEnv expectations.
        """

        metadata = {"render_modes": ["rgb_array", "null"], "render_fps": 30}

        def __init__(
            self,
            robot_name: str = "so100",
            task: str = "manipulation task",
            config: Optional[Any] = None,
            urdf_path: Optional[str] = None,
            render_mode: Optional[str] = None,
            max_episode_steps: int = 1000,
            reward_fn: Optional[Callable] = None,
            success_fn: Optional[Callable] = None,
            obs_keys: Optional[List[str]] = None,
        ):
            from strands_robots.newton import NewtonConfig

            self._config = config or NewtonConfig(num_envs=1)
            self._urdf_path = urdf_path
            self._obs_keys = obs_keys or ["joint_q", "joint_qd"]
            self._backend = None

            super().__init__(
                robot_name=robot_name,
                task=task,
                num_envs=self._config.num_envs,
                render_mode=render_mode,
                max_episode_steps=max_episode_steps,
                reward_fn=reward_fn,
                success_fn=success_fn,
            )

            self._init_backend()

        def _init_backend(self) -> None:
            from strands_robots.newton import NewtonBackend

            self._backend = NewtonBackend(self._config)
            self._backend.create_world()

            add_kwargs = {"name": self._robot_name}
            if self._urdf_path:
                add_kwargs["urdf_path"] = self._urdf_path

            result = self._backend.add_robot(**add_kwargs)
            if not result.get("success"):
                raise RuntimeError(f"Failed to add robot '{self._robot_name}': {result.get('message')}")

            robot_info = result.get("robot_info", {})
            self._n_joints = robot_info.get("num_joints", 0)

            if self._num_envs > 1:
                rep_result = self._backend.replicate(self._num_envs)
                if not rep_result.get("success"):
                    raise RuntimeError(f"Replicate failed: {rep_result.get('message')}")
            else:
                self._backend._finalize_model()

            # Use real state vector length for accurate space dimensions
            if self._backend._state_0 is not None:
                total_q_len = len(self._backend._state_0.joint_q.numpy())
                dof_per_env = total_q_len // max(self._num_envs, 1)
                if dof_per_env > 0:
                    self._n_joints = dof_per_env

            self._set_spaces(self._n_joints)

            logger.info(
                "NewtonGymEnv initialized: robot=%s, num_envs=%d, obs_dim=%d, act_dim=%d, solver=%s",
                self._robot_name, self._num_envs, self._n_joints * 2, self._n_joints, self._config.solver,
            )

        def _reset_backend(self, options=None) -> None:
            env_ids = None
            if options and "env_ids" in options:
                env_ids = options["env_ids"]
            self._backend.reset(env_ids=env_ids)

        def _get_obs(self) -> np.ndarray:
            result = self._backend.get_observation(self._robot_name)
            obs_data = result.get("observations", {}).get(self._robot_name, {})

            jpos = obs_data.get("joint_q")
            jvel = obs_data.get("joint_qd")

            if jpos is None:
                logger.warning(
                    "joint_q not found in observation (available keys: %s). "
                    "Falling back to zeros — RL training will be ineffective.",
                    list(obs_data.keys()),
                )
                jpos = np.zeros(self._n_joints * self._num_envs, dtype=np.float32)
            if jvel is None:
                logger.warning(
                    "joint_qd not found in observation (available keys: %s). "
                    "Falling back to zeros — RL training will be ineffective.",
                    list(obs_data.keys()),
                )
                jvel = np.zeros(self._n_joints * self._num_envs, dtype=np.float32)

            jpos = np.asarray(jpos, dtype=np.float32).flatten()
            jvel = np.asarray(jvel, dtype=np.float32).flatten()

            if self._num_envs > 1:
                n = self._n_joints
                try:
                    jpos_batched = jpos.reshape(self._num_envs, n)
                    jvel_batched = jvel.reshape(self._num_envs, n)
                    obs = np.concatenate([jpos_batched, jvel_batched], axis=1)
                except ValueError:
                    obs = np.zeros((self._num_envs, n * 2), dtype=np.float32)
            else:
                obs = np.concatenate([jpos[: self._n_joints], jvel[: self._n_joints]])

            return obs.astype(np.float32)

        def _step_physics(self, action: np.ndarray) -> dict:
            result = self._backend.step(action)
            return {"sim_time": result.get("sim_time", 0.0)}

        def _render_frame(self) -> Optional[np.ndarray]:
            result = self._backend.render()
            return result.get("image")

        def _destroy_backend(self) -> None:
            if self._backend is not None:
                self._backend.destroy()
                self._backend = None

        @property
        def unwrapped_backend(self):
            """Access the underlying NewtonBackend for advanced usage."""
            return self._backend

        def __repr__(self) -> str:
            return (
                f"NewtonGymEnv(robot={self._robot_name!r}, num_envs={self._num_envs}, solver={self._config.solver!r})"
            )

else:

    class NewtonGymEnv:
        """Stub when gymnasium is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium required: pip install gymnasium")


__all__ = ["NewtonGymEnv"]
