"""Gymnasium environment wrapper for strands-robots Simulation.

Wraps the MuJoCo Simulation as a standard gymnasium.Env so LeRobot's
train/eval scripts work directly with our sim.

Provides standard Gymnasium step/reset/render interface for RL training.
"""

import logging
from typing import Any, Optional

import numpy as np

from strands_robots.gym_env import HAS_GYM

logger = logging.getLogger(__name__)

try:
    import mujoco

    HAS_MUJOCO = True
except (ImportError, AttributeError, OSError):
    HAS_MUJOCO = False


if HAS_GYM:
    from strands_robots.gym_env import BaseGymEnv

    class StrandsSimEnv(BaseGymEnv):
        """Gymnasium wrapper for strands_robots.Simulation.

        Converts the action-based Simulation API into the standard
        Gym step()/reset()/render() interface.

        Usage:
            env = StrandsSimEnv(
                robot_name="so100",
                task="pick up the red cube",
                render_mode="rgb_array",
            )
            obs, info = env.reset()
            for _ in range(1000):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    obs, info = env.reset()
            env.close()

        Integration with LeRobot:
            # Register as a Gymnasium environment
            gymnasium.register(
                id="StrandsSim-v0",
                entry_point="strands_robots.envs:StrandsSimEnv",
                kwargs={"robot_name": "so100"},
            )

            # Use with lerobot-eval
            lerobot-eval --env.type=StrandsSim-v0 --policy.path=...
        """

        metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

        def __init__(
            self,
            robot_name: str = "so100",
            data_config: Optional[str] = None,
            task: str = "manipulation task",
            render_mode: str = "rgb_array",
            render_width: int = 640,
            render_height: int = 480,
            max_episode_steps: int = 1000,
            physics_dt: float = 0.002,
            control_dt: float = 0.02,
            objects: Optional[list] = None,
            cameras: Optional[list] = None,
            reward_fn: Optional[callable] = None,
            success_fn: Optional[callable] = None,
        ):
            """Create a MuJoCo-backed Gymnasium environment.

            For GPU-accelerated backends, use :class:`NewtonGymEnv` or
            :class:`IsaacGymEnv` directly.

            Args:
                robot_name: Robot model name (str) or a Simulation instance.
                data_config: Optional data config name for joint mapping.
                task: Task description (for VLA policies).
                render_mode: "rgb_array" or "human".
                render_width: Render width in pixels.
                render_height: Render height in pixels.
                max_episode_steps: Max steps per episode.
                physics_dt: Physics timestep.
                control_dt: Control timestep (actions applied at this rate).
                objects: List of objects to add.
                cameras: List of cameras to add.
                reward_fn: Custom reward function(obs, action) -> float.
                success_fn: Custom success function(obs) -> bool.
            """
            assert HAS_MUJOCO, "mujoco required: pip install mujoco"

            # Support passing a Simulation object directly
            from strands_robots.simulation import Simulation

            self._passed_sim = None
            if not isinstance(robot_name, str):
                if isinstance(robot_name, Simulation):
                    self._passed_sim = robot_name
                else:
                    raise TypeError(
                        f"robot_name must be a string or Simulation instance, got {type(robot_name).__name__}"
                    )
                _robots = self._passed_sim._world.robots if hasattr(self._passed_sim, "_world") else {}
                robot_name = next(iter(_robots), "robot")

            self.data_config = data_config or robot_name
            self.render_width = render_width
            self.render_height = render_height
            self.physics_dt = physics_dt
            self.control_dt = control_dt
            self.n_substeps = max(1, int(control_dt / physics_dt))
            self.objects_config = objects or []
            self.cameras_config = cameras or []
            self._sim = None
            self._include_pixels = False
            self._viewer = None

            super().__init__(
                robot_name=robot_name,
                task=task,
                num_envs=1,
                render_mode=render_mode,
                max_episode_steps=max_episode_steps,
                reward_fn=reward_fn,
                success_fn=success_fn,
            )

            self._init_backend()

        def _init_backend(self) -> None:
            from strands_robots.simulation import Simulation

            if self._passed_sim is not None:
                self._sim = self._passed_sim
                self._passed_sim = None
            else:
                self._sim = Simulation(tool_name="gym_sim", default_timestep=self.physics_dt)
                self._sim._dispatch_action("create_world", {})
                self._sim._dispatch_action("add_robot", {"data_config": self._robot_name, "name": "robot"})

            for obj in self.objects_config:
                self._sim._dispatch_action("add_object", obj)
            for cam in self.cameras_config:
                self._sim._dispatch_action("add_camera", cam)

            model = self._sim._world._model
            n_qpos = model.nq
            n_qvel = model.nv
            n_ctrl = model.nu

            # Action space with actuator limits
            act_low = np.full(n_ctrl, -1.0, dtype=np.float32)
            act_high = np.full(n_ctrl, 1.0, dtype=np.float32)
            for i in range(n_ctrl):
                if model.actuator_ctrllimited[i]:
                    act_low[i] = float(model.actuator_ctrlrange[i, 0])
                    act_high[i] = float(model.actuator_ctrlrange[i, 1])

            from gymnasium import spaces as sp

            self.action_space = sp.Box(low=act_low, high=act_high, dtype=np.float32)

            # Observation space: qpos + qvel + (optional) image
            obs_dim = n_qpos + n_qvel
            obs_low = np.full(obs_dim, -np.inf, dtype=np.float32)
            obs_high = np.full(obs_dim, np.inf, dtype=np.float32)

            if self.render_mode == "rgb_array" or self.cameras_config:
                self.observation_space = sp.Dict({
                    "state": sp.Box(low=obs_low, high=obs_high, dtype=np.float32),
                    "pixels": sp.Box(low=0, high=255, shape=(self.render_height, self.render_width, 3), dtype=np.uint8),
                })
                self._include_pixels = True
            else:
                self.observation_space = sp.Box(low=obs_low, high=obs_high, dtype=np.float32)
                self._include_pixels = False

            self._n_joints = n_ctrl

        def _reset_backend(self, options=None) -> None:
            if self._sim is not None:
                self._sim._dispatch_action("reset", {})
            else:
                self._init_backend()

        def _get_obs(self) -> Any:
            data = self._sim._world._data
            state = np.concatenate([data.qpos.astype(np.float32), data.qvel.astype(np.float32)])

            if self._include_pixels:
                renderer = mujoco.Renderer(self._sim._world._model, self.render_height, self.render_width)
                renderer.update_scene(data)
                pixels = renderer.render().copy()
                del renderer
                return {"state": state, "pixels": pixels}
            return state

        def _step_physics(self, action: np.ndarray) -> dict:
            data = self._sim._world._data
            np.copyto(data.ctrl[: len(action)], action)
            for _ in range(self.n_substeps):
                mujoco.mj_step(self._sim._world._model, data)
            return {}

        def _render_frame(self) -> Optional[np.ndarray]:
            renderer = mujoco.Renderer(self._sim._world._model, self.render_height, self.render_width)
            renderer.update_scene(self._sim._world._data)
            frame = renderer.render().copy()
            del renderer
            return frame

        def render(self) -> Optional[np.ndarray]:
            if self.render_mode == "rgb_array":
                return self._render_frame()
            elif self.render_mode == "human":
                if not hasattr(self, "_viewer") or self._viewer is None:
                    self._viewer = mujoco.viewer.launch_passive(self._sim._world._model, self._sim._world._data)
                self._viewer.sync()
            return None

        def _destroy_backend(self) -> None:
            if hasattr(self, "_viewer") and self._viewer is not None:
                self._viewer.close()
                self._viewer = None
            if self._sim is not None:
                self._sim._dispatch_action("destroy", {})
                self._sim = None

        @property
        def unwrapped_sim(self):
            """Access the underlying Simulation object for advanced usage."""
            return self._sim

    # Register the environment with Gymnasium
    try:
        import gymnasium as gym

        gym.register(id="StrandsSim-v0", entry_point="strands_robots.envs:StrandsSimEnv")
    except Exception:
        pass

else:

    class StrandsSimEnv:
        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium required: pip install gymnasium")
