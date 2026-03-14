"""Base Gymnasium environment for all strands-robots simulation backends.

Provides the shared logic that MuJoCo, Newton, and Isaac gym wrappers all need:
- Observation/action space construction (single vs batched)
- step() reward/terminated/truncated branching
- reset()/render()/close() skeletons

Subclasses implement only the backend-specific methods:
- _init_backend()
- _get_obs()
- _step_physics(action)
- _render_frame() -> Optional[np.ndarray]
- _destroy_backend()
"""

import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces

    HAS_GYM = True
except (ImportError, AttributeError, OSError):
    HAS_GYM = False


def _ensure_gym():
    if not HAS_GYM:
        raise ImportError("gymnasium required: pip install gymnasium")


if HAS_GYM:

    class BaseGymEnv(gym.Env):
        """Abstract base for all strands-robots Gymnasium environments.

        Handles the shared reward/terminated/truncated logic for both
        single-env and batched (num_envs > 1) modes.

        Subclasses must implement:
            _init_backend() — create the physics backend and set self._n_joints, self._num_envs
            _get_obs() -> np.ndarray — return current observation
            _step_physics(action) -> dict — advance physics, return info dict
            _render_frame() -> Optional[np.ndarray] — render one frame
            _destroy_backend() — clean up backend resources
        """

        metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

        def __init__(
            self,
            robot_name: str = "so100",
            task: str = "manipulation task",
            num_envs: int = 1,
            render_mode: Optional[str] = None,
            max_episode_steps: int = 1000,
            reward_fn: Optional[Callable] = None,
            success_fn: Optional[Callable] = None,
            **kwargs,
        ):
            super().__init__()
            self._robot_name = robot_name
            self._task = task
            self._num_envs = num_envs
            self.render_mode = render_mode
            self.max_episode_steps = max_episode_steps
            self.reward_fn = reward_fn
            self.success_fn = success_fn
            self._step_count = 0
            self._n_joints = 0

        # ── Space helpers ───────────────────────────────────────────

        def _set_spaces(self, n_joints: int, obs_dim: Optional[int] = None, act_dim: Optional[int] = None) -> None:
            """Set observation/action spaces for the given dimensions.

            Args:
                n_joints: Number of joints (used as default for act_dim).
                obs_dim: Observation dimension. Defaults to n_joints * 2 (pos + vel).
                act_dim: Action dimension. Defaults to n_joints.
            """
            obs_dim = obs_dim or n_joints * 2
            act_dim = act_dim or n_joints

            if self._num_envs > 1:
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self._num_envs, obs_dim), dtype=np.float32,
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(self._num_envs, act_dim), dtype=np.float32,
                )
            else:
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32,
                )

        # ── Reward / termination (shared across all backends) ──────

        def _compute_reward_terminated_truncated(self, obs, action):
            """Compute reward, terminated, truncated for single or batched envs.

            Returns:
                (reward, terminated, truncated)
            """
            if self._num_envs > 1:
                if self.reward_fn is not None:
                    reward = self.reward_fn(obs, action)
                    if not isinstance(reward, np.ndarray):
                        reward = np.full(self._num_envs, float(reward), dtype=np.float32)
                else:
                    reward = np.zeros(self._num_envs, dtype=np.float32)

                truncated = np.full(self._num_envs, self._step_count >= self.max_episode_steps)

                if self.success_fn is not None:
                    terminated = self.success_fn(obs)
                    if not isinstance(terminated, np.ndarray):
                        terminated = np.full(self._num_envs, bool(terminated))
                else:
                    terminated = np.full(self._num_envs, False)
            else:
                reward = float(self.reward_fn(obs, action)) if self.reward_fn is not None else 0.0
                truncated = self._step_count >= self.max_episode_steps
                terminated = bool(self.success_fn(obs)) if self.success_fn is not None else False

            return reward, terminated, truncated

        # ── Gymnasium API ───────────────────────────────────────────

        def reset(self, seed=None, options=None) -> Tuple[Any, Dict]:
            super().reset(seed=seed)
            self._reset_backend(options=options)
            self._step_count = 0
            obs = self._get_obs()
            return obs, {"task": self._task, "num_envs": self._num_envs}

        def step(self, action: np.ndarray) -> Tuple[Any, Any, Any, Any, Dict]:
            action = np.asarray(action, dtype=np.float32)
            step_info = self._step_physics(action)
            self._step_count += 1
            obs = self._get_obs()
            reward, terminated, truncated = self._compute_reward_terminated_truncated(obs, action)

            info = {
                "step": self._step_count,
                "task": self._task,
                "is_success": (terminated if isinstance(terminated, bool) else terminated.any()),
            }
            info.update(step_info)
            return obs, reward, terminated, truncated, info

        def render(self) -> Optional[np.ndarray]:
            if self.render_mode == "rgb_array":
                return self._render_frame()
            return None

        def close(self):
            self._destroy_backend()

        # ── Properties ──────────────────────────────────────────────

        @property
        def num_envs(self) -> int:
            return self._num_envs

        # ── Abstract methods (backend-specific) ────────────────────

        @abstractmethod
        def _init_backend(self) -> None:
            """Initialize the physics backend. Must set self._n_joints and call self._set_spaces()."""
            ...

        @abstractmethod
        def _reset_backend(self, options=None) -> None:
            """Reset the physics backend to initial state."""
            ...

        @abstractmethod
        def _get_obs(self) -> np.ndarray:
            """Return current observation array."""
            ...

        @abstractmethod
        def _step_physics(self, action: np.ndarray) -> dict:
            """Advance physics by one control step. Return extra info dict."""
            ...

        @abstractmethod
        def _render_frame(self) -> Optional[np.ndarray]:
            """Render and return an RGB frame, or None."""
            ...

        @abstractmethod
        def _destroy_backend(self) -> None:
            """Clean up backend resources."""
            ...

else:

    class BaseGymEnv:
        """Stub when gymnasium is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium required: pip install gymnasium")


__all__ = ["BaseGymEnv", "HAS_GYM", "_ensure_gym"]
