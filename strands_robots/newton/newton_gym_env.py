"""Gymnasium wrapper for Newton GPU-accelerated simulation.

Bridges NewtonBackend's batched GPU API into the standard gym.Env interface
so that Stable Baselines 3, CleanRL, and other RL libraries work out-of-the-box.

For single-env cases, implements gym.Env.
For multi-env (num_envs > 1), implements gym.vector.VectorEnv for proper
batched obs/actions — which is what SB3 expects for GPU-parallel training.

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
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces

    HAS_GYM = True
except ImportError:
    HAS_GYM = False


if HAS_GYM:

    class NewtonGymEnv(gym.Env):
        """Gymnasium wrapper around NewtonBackend for RL training.

        Implements the standard gym.Env interface. When num_envs > 1,
        observations and actions are batched along the first axis
        (shape: [num_envs, ...]) compatible with SB3's VecEnv expectations.

        Note: For true SB3 VectorEnv compatibility with auto-reset,
        use stable_baselines3.common.vec_env.DummyVecEnv or wrap this
        with gymnasium.vector.SyncVectorEnv. For massive parallelism,
        the batched interface here is more efficient than spawning
        separate processes.
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
            """
            Args:
                robot_name: Robot model name (resolved via strands_robots.assets).
                task: Task description string.
                config: NewtonConfig instance. If None, uses default (1 env, mujoco solver).
                urdf_path: Explicit URDF/MJCF path override.
                render_mode: "rgb_array" or None.
                max_episode_steps: Maximum steps before truncation.
                reward_fn: Custom reward function(obs_dict, action) -> float or np.ndarray.
                success_fn: Custom success function(obs_dict) -> bool or np.ndarray.
                obs_keys: Which observation keys to include. Default: ["joint_positions", "joint_velocities"].
            """
            super().__init__()

            # Lazy import to avoid hard dep on Newton at module level
            from strands_robots.newton import NewtonBackend, NewtonConfig

            self._config = config or NewtonConfig(num_envs=1)
            self._robot_name = robot_name
            self._task = task
            self._urdf_path = urdf_path
            self.render_mode = render_mode
            self.max_episode_steps = max_episode_steps
            self.reward_fn = reward_fn
            self.success_fn = success_fn
            self._obs_keys = obs_keys or ["joint_positions", "joint_velocities"]

            self._backend: Optional[NewtonBackend] = None
            self._step_count = 0
            self._num_envs = self._config.num_envs

            # Initialize backend to determine spaces
            self._init_backend()

        def _init_backend(self):
            """Initialize Newton backend and determine observation/action spaces."""
            from strands_robots.newton import NewtonBackend

            self._backend = NewtonBackend(self._config)
            self._backend.create_world()

            add_kwargs = {"name": self._robot_name}
            if self._urdf_path:
                add_kwargs["urdf_path"] = self._urdf_path

            result = self._backend.add_robot(**add_kwargs)
            if not result.get("success"):
                raise RuntimeError(
                    f"Failed to add robot '{self._robot_name}': {result.get('message')}"
                )

            robot_info = result.get("robot_info", {})
            self._n_joints = robot_info.get("num_joints", 0)

            # Replicate if multi-env
            if self._num_envs > 1:
                rep_result = self._backend.replicate(self._num_envs)
                if not rep_result.get("success"):
                    raise RuntimeError(f"Replicate failed: {rep_result.get('message')}")
            else:
                # Force model finalization for single env
                self._backend._finalize_model()

            # Determine observation dimension
            # For each env: joint_positions (n_joints) + joint_velocities (n_joints)
            obs_dim = self._n_joints * 2  # qpos + qvel

            # Action space: joint torques/positions
            act_dim = self._n_joints

            if self._num_envs > 1:
                # Batched spaces [num_envs, dim]
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._num_envs, obs_dim),
                    dtype=np.float32,
                )
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self._num_envs, act_dim),
                    dtype=np.float32,
                )
            else:
                # Single env spaces
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_dim,),
                    dtype=np.float32,
                )
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(act_dim,),
                    dtype=np.float32,
                )

            logger.info(
                "NewtonGymEnv initialized: robot=%s, num_envs=%d, obs_dim=%d, act_dim=%d, solver=%s",
                self._robot_name,
                self._num_envs,
                obs_dim,
                act_dim,
                self._config.solver,
            )

        def _get_obs(self) -> np.ndarray:
            """Get current observation from Newton backend.

            Returns:
                Observation array, shape [obs_dim] (single) or [num_envs, obs_dim] (batched).
            """
            result = self._backend.get_observation(self._robot_name)
            obs_data = result.get("observations", {}).get(self._robot_name, {})

            jpos = obs_data.get("joint_positions")
            jvel = obs_data.get("joint_velocities")

            if jpos is None:
                jpos = np.zeros(self._n_joints * self._num_envs, dtype=np.float32)
            if jvel is None:
                jvel = np.zeros(self._n_joints * self._num_envs, dtype=np.float32)

            jpos = np.asarray(jpos, dtype=np.float32).flatten()
            jvel = np.asarray(jvel, dtype=np.float32).flatten()

            if self._num_envs > 1:
                # Reshape to [num_envs, n_joints] then concat
                n = self._n_joints
                try:
                    jpos_batched = jpos.reshape(self._num_envs, n)
                    jvel_batched = jvel.reshape(self._num_envs, n)
                    obs = np.concatenate([jpos_batched, jvel_batched], axis=1)
                except ValueError:
                    # Fallback: pad or truncate
                    obs = np.zeros((self._num_envs, n * 2), dtype=np.float32)
            else:
                obs = np.concatenate([jpos[: self._n_joints], jvel[: self._n_joints]])

            return obs.astype(np.float32)

        def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None,
        ) -> Tuple[np.ndarray, Dict]:
            """Reset the environment.

            For multi-env, resets all environments simultaneously.
            """
            super().reset(seed=seed)

            env_ids = None
            if options and "env_ids" in options:
                env_ids = options["env_ids"]

            self._backend.reset(env_ids=env_ids)
            self._step_count = 0

            obs = self._get_obs()
            info = {"task": self._task, "num_envs": self._num_envs}

            return obs, info

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, Any, Any, Any, Dict]:
            """Take a step in all environments simultaneously.

            Args:
                action: Shape [act_dim] (single) or [num_envs, act_dim] (batched).

            Returns:
                Tuple of (obs, reward, terminated, truncated, info).
                For multi-env, reward/terminated/truncated are arrays of shape [num_envs].
            """
            action = np.asarray(action, dtype=np.float32)

            # Step the physics
            result = self._backend.step(action)
            self._step_count += 1

            # Get observation
            obs = self._get_obs()

            # Compute reward
            if self._num_envs > 1:
                if self.reward_fn is not None:
                    reward = self.reward_fn(obs, action)
                    if not isinstance(reward, np.ndarray):
                        reward = np.full(
                            self._num_envs, float(reward), dtype=np.float32
                        )
                else:
                    reward = np.zeros(self._num_envs, dtype=np.float32)

                # Truncation check
                truncated = np.full(
                    self._num_envs, self._step_count >= self.max_episode_steps
                )

                # Success/termination check
                if self.success_fn is not None:
                    terminated = self.success_fn(obs)
                    if not isinstance(terminated, np.ndarray):
                        terminated = np.full(self._num_envs, bool(terminated))
                else:
                    terminated = np.full(self._num_envs, False)
            else:
                if self.reward_fn is not None:
                    reward = float(self.reward_fn(obs, action))
                else:
                    reward = 0.0

                truncated = self._step_count >= self.max_episode_steps

                if self.success_fn is not None:
                    terminated = bool(self.success_fn(obs))
                else:
                    terminated = False

            info = {
                "step": self._step_count,
                "task": self._task,
                "sim_time": result.get("sim_time", 0.0),
                "is_success": (
                    terminated if isinstance(terminated, bool) else terminated.any()
                ),
            }

            return obs, reward, terminated, truncated, info

        def render(self) -> Optional[np.ndarray]:
            """Render the scene."""
            if self.render_mode == "rgb_array":
                result = self._backend.render()
                return result.get("image")
            return None

        def close(self):
            """Clean up Newton backend."""
            if self._backend is not None:
                self._backend.destroy()
                self._backend = None

        @property
        def num_envs(self) -> int:
            """Number of parallel environments."""
            return self._num_envs

        @property
        def unwrapped_backend(self):
            """Access the underlying NewtonBackend for advanced usage."""
            return self._backend

        def __repr__(self) -> str:
            return (
                f"NewtonGymEnv(robot={self._robot_name!r}, "
                f"num_envs={self._num_envs}, "
                f"solver={self._config.solver!r})"
            )

else:

    class NewtonGymEnv:
        """Stub when gymnasium is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium required: pip install gymnasium")


__all__ = ["NewtonGymEnv"]
