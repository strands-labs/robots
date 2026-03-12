"""Gymnasium wrapper for Isaac Sim GPU-accelerated simulation.

Bridges IsaacSimBackend's GPU-parallel API into the standard gym.Env interface
so that Stable Baselines 3, CleanRL, LeRobot, and other RL libraries work
directly with the Isaac Sim backend.

Follows the same pattern as NewtonGymEnv (strands_robots/newton/_gym_env.py)
but wraps IsaacSimBackend instead of NewtonBackend.

For single-env cases, implements gym.Env.
For multi-env (num_envs > 1), observations/actions are batched along the first
axis (shape: [num_envs, ...]).

Usage:
    from strands_robots.isaac.isaac_gym_env import IsaacGymEnv

    # Single env
    env = IsaacGymEnv(
        robot_name="so100",
        task="pick up the red cube",
        num_envs=1,
    )
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)

    # Vectorized (16 parallel envs on GPU)
    env = IsaacGymEnv(
        robot_name="so100",
        task="manipulation task",
        num_envs=16,
        device="cuda:0",
    )
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(actions)  # batched

Note:
    Isaac Sim must be available on the system. If Isaac Sim is not installed,
    this module provides a stub that raises ImportError on instantiation.
    The IsaacSimBackend itself handles Isaac Sim/Lab lazy imports internally.
"""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces

    HAS_GYM = True
except ImportError:
    HAS_GYM = False


if HAS_GYM:

    class IsaacGymEnv(gym.Env):
        """Gymnasium wrapper around IsaacSimBackend for RL training.

        Implements the standard gym.Env interface. When ``num_envs > 1``,
        observations and actions are batched along the first axis
        (shape ``[num_envs, ...]``).

        The environment lazily initializes the Isaac Sim backend on first
        ``reset()`` call (or during ``__init__`` if possible) so that
        construction cost is paid only once.

        Attributes:
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
        """

        metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

        # Default joint count used for placeholder spaces when the backend
        # is not yet initialised.  Overwritten by _init_backend() once the
        # actual robot is loaded.
        _DEFAULT_N_JOINTS = 6

        def __init__(
            self,
            robot_name: str = "so100",
            task: str = "manipulation task",
            num_envs: int = 1,
            device: str = "cuda:0",
            render_mode: Optional[str] = None,
            max_episode_steps: int = 1000,
            physics_dt: float = 1.0 / 200.0,
            rendering_dt: float = 1.0 / 60.0,
            headless: bool = True,
            reward_fn: Optional[Callable] = None,
            success_fn: Optional[Callable] = None,
            usd_path: Optional[str] = None,
        ):
            """Initialise the Isaac Sim Gymnasium environment.

            Args:
                robot_name: Robot model name (resolved via strands_robots assets
                    or Isaac Lab built-in assets).
                task: Task description string (passed to policies).
                num_envs: Number of parallel GPU environments.
                device: CUDA device (e.g. ``"cuda:0"``).
                render_mode: ``"rgb_array"`` or ``None``.
                max_episode_steps: Maximum steps before truncation.
                physics_dt: Physics timestep.
                rendering_dt: Rendering timestep.
                headless: Run without GUI display.
                reward_fn: Custom reward ``(obs, action) -> float | np.ndarray``.
                success_fn: Custom success ``(obs) -> bool | np.ndarray``.
                usd_path: Explicit USD asset path (skips auto-resolution).
            """
            super().__init__()

            from .isaac_sim_backend import IsaacSimBackend, IsaacSimConfig

            self._robot_name = robot_name
            self._task = task
            self._num_envs = num_envs
            self._device = device
            self.render_mode = render_mode
            self.max_episode_steps = max_episode_steps
            self._usd_path = usd_path
            self.reward_fn = reward_fn
            self.success_fn = success_fn

            self._config = IsaacSimConfig(
                num_envs=num_envs,
                device=device,
                physics_dt=physics_dt,
                rendering_dt=rendering_dt,
                headless=headless,
                enable_cameras=render_mode == "rgb_array",
                render_mode=render_mode or "rgb_array",
            )

            self._backend: Optional[IsaacSimBackend] = None
            self._step_count = 0
            self._n_joints = 0

            # Set placeholder spaces so that callers (e.g. StrandsSimEnv)
            # can always read observation_space / action_space, even when
            # the backend initialisation is deferred.  _init_backend()
            # overwrites these with accurate values once the robot is loaded.
            self._set_default_spaces(self._DEFAULT_N_JOINTS)

            # Try to initialise the backend eagerly.  This may fail if
            # Isaac Sim is not available — in that case we defer to reset().
            try:
                self._init_backend()
            except Exception as e:
                logger.warning(
                    f"Deferred Isaac backend init (will retry on reset): {e}"
                )

        # ── Space helpers ───────────────────────────────────────────

        def _set_default_spaces(self, n_joints: int) -> None:
            """Set observation/action spaces for *n_joints* joints.

            Used both during deferred init (placeholder) and after the
            backend provides the real joint count.
            """
            obs_dim = n_joints * 2  # joint_pos + joint_vel

            if self._num_envs > 1:
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._num_envs, obs_dim),
                    dtype=np.float32,
                )
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self._num_envs, n_joints),
                    dtype=np.float32,
                )
            else:
                self.observation_space = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(obs_dim,),
                    dtype=np.float32,
                )
                self.action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(n_joints,),
                    dtype=np.float32,
                )

        def _init_backend(self):
            """Create the IsaacSimBackend world and add the robot."""
            from .isaac_sim_backend import IsaacSimBackend

            self._backend = IsaacSimBackend(config=self._config)

            # Create world
            result = self._backend.create_world()
            if result.get("status") != "success":
                raise RuntimeError(f"create_world failed: {result}")

            # Add robot
            add_kwargs: Dict[str, Any] = {"name": self._robot_name}
            if self._usd_path:
                add_kwargs["usd_path"] = self._usd_path
            else:
                add_kwargs["data_config"] = self._robot_name

            result = self._backend.add_robot(**add_kwargs)
            if result.get("status") != "success":
                raise RuntimeError(f"add_robot('{self._robot_name}') failed: {result}")

            # Determine joint count from the backend's robot
            robot = self._backend._robot
            if robot is not None and hasattr(robot, "num_joints"):
                self._n_joints = robot.num_joints
            else:
                # Conservative default
                self._n_joints = self._DEFAULT_N_JOINTS

            # Rebuild spaces with the real joint count
            self._set_default_spaces(self._n_joints)

            logger.info(
                "IsaacGymEnv initialised: robot=%s, num_envs=%d, "
                "obs_dim=%d, act_dim=%d, device=%s",
                self._robot_name,
                self._num_envs,
                self._n_joints * 2,
                self._n_joints,
                self._device,
            )

        # ── Observation helper ──────────────────────────────────────

        def _get_obs(self) -> np.ndarray:
            """Get current observation from the Isaac backend.

            Returns:
                Observation array of shape ``[obs_dim]`` (single env) or
                ``[num_envs, obs_dim]`` (batched).
            """
            if self._backend is None or self._backend._robot is None:
                n = self._n_joints or self._DEFAULT_N_JOINTS
                if self._num_envs > 1:
                    return np.zeros((self._num_envs, n * 2), dtype=np.float32)
                return np.zeros(n * 2, dtype=np.float32)

            robot = self._backend._robot

            # GPU tensors → numpy
            jpos = robot.data.joint_pos
            jvel = robot.data.joint_vel

            if hasattr(jpos, "cpu"):
                jpos = jpos.cpu().numpy()
                jvel = jvel.cpu().numpy()
            else:
                jpos = np.asarray(jpos, dtype=np.float32)
                jvel = np.asarray(jvel, dtype=np.float32)

            if self._num_envs > 1:
                # jpos, jvel: (num_envs, n_joints)
                obs = np.concatenate([jpos, jvel], axis=1).astype(np.float32)
            else:
                obs = np.concatenate(
                    [jpos.flatten()[: self._n_joints], jvel.flatten()[: self._n_joints]]
                ).astype(np.float32)

            return obs

        # ── Gymnasium API ───────────────────────────────────────────

        def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[Dict] = None,
        ) -> Tuple[np.ndarray, Dict]:
            """Reset the environment (all envs if batched)."""
            super().reset(seed=seed)

            if self._backend is None:
                self._init_backend()

            self._backend.reset()
            self._step_count = 0

            obs = self._get_obs()
            info = {"task": self._task, "num_envs": self._num_envs}
            return obs, info

        def step(self, action: np.ndarray) -> Tuple[np.ndarray, Any, Any, Any, Dict]:
            """Step all environments simultaneously.

            Args:
                action: Shape ``[act_dim]`` (single) or
                    ``[num_envs, act_dim]`` (batched).

            Returns:
                ``(obs, reward, terminated, truncated, info)``
            """
            if self._backend is None:
                raise RuntimeError("Call reset() before step()")

            action = np.asarray(action, dtype=np.float32)

            # Convert to torch tensor for Isaac backend
            try:
                import torch

                action_tensor = torch.tensor(
                    action if action.ndim == 2 else action[np.newaxis],
                    device=self._device,
                    dtype=torch.float32,
                )
                if self._num_envs > 1 and action_tensor.shape[0] == 1:
                    action_tensor = action_tensor.expand(self._num_envs, -1)
            except ImportError:
                action_tensor = action

            self._backend.step(action_tensor)
            self._step_count += 1

            obs = self._get_obs()

            # -- Reward --
            if self._num_envs > 1:
                if self.reward_fn is not None:
                    reward = self.reward_fn(obs, action)
                    if not isinstance(reward, np.ndarray):
                        reward = np.full(
                            self._num_envs, float(reward), dtype=np.float32
                        )
                else:
                    reward = np.zeros(self._num_envs, dtype=np.float32)

                truncated = np.full(
                    self._num_envs,
                    self._step_count >= self.max_episode_steps,
                )

                if self.success_fn is not None:
                    terminated = self.success_fn(obs)
                    if not isinstance(terminated, np.ndarray):
                        terminated = np.full(self._num_envs, bool(terminated))
                else:
                    terminated = np.full(self._num_envs, False)
            else:
                reward = (
                    float(self.reward_fn(obs, action))
                    if self.reward_fn is not None
                    else 0.0
                )
                truncated = self._step_count >= self.max_episode_steps
                terminated = (
                    bool(self.success_fn(obs)) if self.success_fn is not None else False
                )

            info = {
                "step": self._step_count,
                "task": self._task,
                "is_success": (
                    terminated if isinstance(terminated, bool) else terminated.any()
                ),
            }

            return obs, reward, terminated, truncated, info

        def render(self) -> Optional[np.ndarray]:
            """Render the scene via Isaac Sim RTX."""
            if self.render_mode == "rgb_array" and self._backend is not None:
                result = self._backend.render()
                content = result.get("content", [])
                for item in content:
                    if "image" in item:
                        import io

                        from PIL import Image

                        img = Image.open(io.BytesIO(item["image"]["source"]["bytes"]))
                        return np.array(img)
            return None

        def close(self):
            """Destroy the Isaac Sim backend and release GPU resources."""
            if self._backend is not None:
                self._backend.destroy()
                self._backend = None

        # ── Properties ──────────────────────────────────────────────

        @property
        def num_envs(self) -> int:
            """Number of parallel GPU environments."""
            return self._num_envs

        @property
        def unwrapped_backend(self):
            """Access the underlying IsaacSimBackend."""
            return self._backend

        def __repr__(self) -> str:
            return (
                f"IsaacGymEnv(robot={self._robot_name!r}, "
                f"num_envs={self._num_envs}, device={self._device!r})"
            )

else:

    class IsaacGymEnv:
        """Stub when gymnasium is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium required: pip install gymnasium")


__all__ = ["IsaacGymEnv"]
