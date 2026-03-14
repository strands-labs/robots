"""Gymnasium wrapper for Isaac Sim GPU-accelerated simulation.

Bridges IsaacSimBackend's GPU-parallel API into the standard gym.Env interface
so that Stable Baselines 3, CleanRL, LeRobot, and other RL libraries work
directly with the Isaac Sim backend.

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
from typing import Any, Callable, Dict, Optional

import numpy as np

from strands_robots.gym_env import HAS_GYM

logger = logging.getLogger(__name__)

_DEFAULT_N_JOINTS = 6


if HAS_GYM:
    from strands_robots.gym_env import BaseGymEnv

    class IsaacGymEnv(BaseGymEnv):
        """Gymnasium wrapper around IsaacSimBackend for RL training.

        Implements the standard gym.Env interface. When ``num_envs > 1``,
        observations and actions are batched along the first axis
        (shape ``[num_envs, ...]``).
        """

        metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

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
            from .isaac_sim_backend import IsaacSimConfig

            self._device = device
            self._usd_path = usd_path
            self._config = IsaacSimConfig(
                num_envs=num_envs,
                device=device,
                physics_dt=physics_dt,
                rendering_dt=rendering_dt,
                headless=headless,
                enable_cameras=render_mode == "rgb_array",
                render_mode=render_mode or "rgb_array",
            )
            self._backend = None

            super().__init__(
                robot_name=robot_name,
                task=task,
                num_envs=num_envs,
                render_mode=render_mode,
                max_episode_steps=max_episode_steps,
                reward_fn=reward_fn,
                success_fn=success_fn,
            )

            # Set placeholder spaces, then try eager init
            self._set_spaces(_DEFAULT_N_JOINTS)
            try:
                self._init_backend()
            except Exception as e:
                logger.warning(f"Deferred Isaac backend init (will retry on reset): {e}")

        def _init_backend(self) -> None:
            from .isaac_sim_backend import IsaacSimBackend

            self._backend = IsaacSimBackend(config=self._config)

            result = self._backend.create_world()
            if result.get("status") != "success":
                raise RuntimeError(f"create_world failed: {result}")

            add_kwargs: Dict[str, Any] = {"name": self._robot_name}
            if self._usd_path:
                add_kwargs["usd_path"] = self._usd_path
            else:
                add_kwargs["data_config"] = self._robot_name

            result = self._backend.add_robot(**add_kwargs)
            if result.get("status") != "success":
                raise RuntimeError(f"add_robot('{self._robot_name}') failed: {result}")

            robot = self._backend._robot
            if robot is not None and hasattr(robot, "num_joints"):
                self._n_joints = robot.num_joints
            else:
                self._n_joints = _DEFAULT_N_JOINTS

            self._set_spaces(self._n_joints)

            logger.info(
                "IsaacGymEnv initialised: robot=%s, num_envs=%d, obs_dim=%d, act_dim=%d, device=%s",
                self._robot_name, self._num_envs, self._n_joints * 2, self._n_joints, self._device,
            )

        def _reset_backend(self, options=None) -> None:
            if self._backend is None:
                self._init_backend()
            self._backend.reset()

        def _get_obs(self) -> np.ndarray:
            if self._backend is None or self._backend._robot is None:
                n = self._n_joints or _DEFAULT_N_JOINTS
                if self._num_envs > 1:
                    return np.zeros((self._num_envs, n * 2), dtype=np.float32)
                return np.zeros(n * 2, dtype=np.float32)

            robot = self._backend._robot
            jpos = robot.data.joint_pos
            jvel = robot.data.joint_vel

            if hasattr(jpos, "cpu"):
                jpos = jpos.cpu().numpy()
                jvel = jvel.cpu().numpy()
            else:
                jpos = np.asarray(jpos, dtype=np.float32)
                jvel = np.asarray(jvel, dtype=np.float32)

            if self._num_envs > 1:
                obs = np.concatenate([jpos, jvel], axis=1).astype(np.float32)
            else:
                obs = np.concatenate([jpos.flatten()[: self._n_joints], jvel.flatten()[: self._n_joints]]).astype(
                    np.float32
                )
            return obs

        def _step_physics(self, action: np.ndarray) -> dict:
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
            return {}

        def _render_frame(self) -> Optional[np.ndarray]:
            if self._backend is None:
                return None
            result = self._backend.render()
            content = result.get("content", [])
            for item in content:
                if "image" in item:
                    import io

                    from PIL import Image

                    img = Image.open(io.BytesIO(item["image"]["source"]["bytes"]))
                    return np.array(img)
            return None

        def _destroy_backend(self) -> None:
            if self._backend is not None:
                self._backend.destroy()
                self._backend = None

        @property
        def unwrapped_backend(self):
            """Access the underlying IsaacSimBackend."""
            return self._backend

        def __repr__(self) -> str:
            return f"IsaacGymEnv(robot={self._robot_name!r}, num_envs={self._num_envs}, device={self._device!r})"

else:

    class IsaacGymEnv:
        """Stub when gymnasium is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium required: pip install gymnasium")


__all__ = ["IsaacGymEnv"]
