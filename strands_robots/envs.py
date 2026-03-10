#!/usr/bin/env python3
"""
Gymnasium environment wrapper for strands-robots Simulation.

Wraps the MuJoCo Simulation as a standard gymnasium.Env so LeRobot's
train/eval scripts work directly with our sim.

Provides standard Gymnasium step/reset/render interface for RL training.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from gymnasium import spaces

    HAS_GYM = True
except ImportError:
    HAS_GYM = False

try:
    import mujoco

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False


if HAS_GYM:

    class StrandsSimEnv(gym.Env):
        """
        Gymnasium wrapper for strands_robots.Simulation.

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
            backend: str = "mujoco",
            num_envs: int = 1,
            device: str = "cuda:0",
        ):
            """
            Args:
                robot_name: Robot model name (from asset registry or URDF path)
                data_config: Optional data config name for joint mapping
                task: Task description (for VLA policies)
                render_mode: "rgb_array" or "human"
                render_width: Render width in pixels
                render_height: Render height in pixels
                max_episode_steps: Max steps per episode
                physics_dt: Physics timestep
                control_dt: Control timestep (actions applied at this rate)
                objects: List of objects to add: [{"name": "cube", "shape": "box", ...}]
                cameras: List of cameras to add: [{"name": "front", "pos": [...], ...}]
                reward_fn: Custom reward function(obs, action, next_obs) -> float
                success_fn: Custom success function(obs) -> bool
                backend: Simulation backend — "mujoco" (default), "isaac", or "newton"
                num_envs: Number of parallel GPU envs (for isaac/newton backends)
                device: CUDA device for GPU backends (e.g. "cuda:0")
            """
            # For non-MuJoCo backends, delegate to the appropriate Gym env
            if backend == "isaac":
                super().__init__()
                from strands_robots.isaac.isaac_gym_env import IsaacGymEnv

                self._delegate = IsaacGymEnv(
                    robot_name=robot_name,
                    task=task,
                    num_envs=num_envs,
                    device=device,
                    render_mode=render_mode,
                    max_episode_steps=max_episode_steps,
                    physics_dt=physics_dt,
                    reward_fn=reward_fn,
                    success_fn=success_fn,
                )
                # Copy spaces from delegate
                self.observation_space = self._delegate.observation_space
                self.action_space = self._delegate.action_space
                self._backend = backend
                return
            elif backend == "newton":
                super().__init__()
                from strands_robots.newton.newton_backend import NewtonConfig
                from strands_robots.newton.newton_gym_env import NewtonGymEnv

                self._delegate = NewtonGymEnv(
                    robot_name=robot_name,
                    task=task,
                    config=NewtonConfig(num_envs=num_envs, device=device),
                    render_mode=render_mode,
                    max_episode_steps=max_episode_steps,
                    reward_fn=reward_fn,
                    success_fn=success_fn,
                )
                self.observation_space = self._delegate.observation_space
                self.action_space = self._delegate.action_space
                self._backend = backend
                return

            super().__init__()
            self._delegate = None
            self._backend = "mujoco"

            assert HAS_MUJOCO, "mujoco required: pip install mujoco"

            self.robot_name = robot_name
            self.data_config = data_config or robot_name
            self.task = task
            self.render_mode = render_mode
            self.render_width = render_width
            self.render_height = render_height
            self.max_episode_steps = max_episode_steps
            self.physics_dt = physics_dt
            self.control_dt = control_dt
            self.n_substeps = max(1, int(control_dt / physics_dt))
            self.objects_config = objects or []
            self.cameras_config = cameras or []
            self.reward_fn = reward_fn
            self.success_fn = success_fn

            # Will be initialized in reset()
            self._sim = None
            self._step_count = 0
            self._initialized = False

            # Initialize sim to get spaces
            self._init_sim()

        def _init_sim(self):
            """Initialize the simulation to determine observation/action spaces."""
            from strands_robots.simulation import Simulation

            self._sim = Simulation(
                tool_name="gym_sim",
                default_timestep=self.physics_dt,
            )

            # Create world
            self._sim._dispatch_action("create_world", {})

            # Add robot
            self._sim._dispatch_action(
                "add_robot",
                {
                    "data_config": self.robot_name,
                    "name": "robot",
                },
            )

            # Add objects
            for obj in self.objects_config:
                self._sim._dispatch_action("add_object", obj)

            # Add cameras
            for cam in self.cameras_config:
                self._sim._dispatch_action("add_camera", cam)

            # Determine spaces from sim state
            model = self._sim._world._model

            n_qpos = model.nq
            n_qvel = model.nv
            n_ctrl = model.nu

            # Action space: joint actuator controls
            act_low = np.full(n_ctrl, -1.0, dtype=np.float32)
            act_high = np.full(n_ctrl, 1.0, dtype=np.float32)
            # Use actuator limits if available
            for i in range(n_ctrl):
                if model.actuator_ctrllimited[i]:
                    act_low[i] = float(model.actuator_ctrlrange[i, 0])
                    act_high[i] = float(model.actuator_ctrlrange[i, 1])

            self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

            # Observation space: qpos + qvel + (optional) image
            obs_dim = n_qpos + n_qvel
            obs_low = np.full(obs_dim, -np.inf, dtype=np.float32)
            obs_high = np.full(obs_dim, np.inf, dtype=np.float32)

            if self.render_mode == "rgb_array" or self.cameras_config:
                # Include image in observation
                self.observation_space = spaces.Dict(
                    {
                        "state": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                        "pixels": spaces.Box(
                            low=0,
                            high=255,
                            shape=(self.render_height, self.render_width, 3),
                            dtype=np.uint8,
                        ),
                    }
                )
                self._include_pixels = True
            else:
                self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
                self._include_pixels = False

            self._initialized = True

        def _get_obs(self) -> Any:
            """Get current observation from sim."""
            data = self._sim._world._data

            state = np.concatenate(
                [
                    data.qpos.astype(np.float32),
                    data.qvel.astype(np.float32),
                ]
            )

            if self._include_pixels:
                # Render image
                renderer = mujoco.Renderer(self._sim._world._model, self.render_height, self.render_width)
                renderer.update_scene(data)
                pixels = renderer.render().copy()
                # mujoco 3.x Renderer has no close() — it's garbage-collected
                del renderer

                return {
                    "state": state,
                    "pixels": pixels,
                }
            else:
                return state

        def reset(self, seed=None, options=None) -> Tuple[Any, Dict]:
            """Reset environment to initial state."""
            if hasattr(self, "_delegate") and self._delegate is not None:
                return self._delegate.reset(seed=seed, options=options)

            super().reset(seed=seed)

            if self._sim is not None:
                # Reset simulation
                self._sim._dispatch_action("reset", {})
            else:
                self._init_sim()

            self._step_count = 0
            obs = self._get_obs()

            return obs, {"task": self.task}

        def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
            """Take a step in the environment."""
            if hasattr(self, "_delegate") and self._delegate is not None:
                return self._delegate.step(action)

            data = self._sim._world._data

            # Apply action to actuators
            np.copyto(data.ctrl[: len(action)], action)

            # Step physics
            for _ in range(self.n_substeps):
                mujoco.mj_step(self._sim._world._model, data)

            self._step_count += 1

            # Get observation
            obs = self._get_obs()

            # Compute reward
            if self.reward_fn is not None:
                reward = float(self.reward_fn(obs, action))
            else:
                reward = 0.0

            # Check termination
            terminated = False
            if self.success_fn is not None:
                terminated = bool(self.success_fn(obs))

            # Check truncation (max steps)
            truncated = self._step_count >= self.max_episode_steps

            info = {
                "step": self._step_count,
                "task": self.task,
                "is_success": terminated,
            }

            return obs, reward, terminated, truncated, info

        def render(self) -> Optional[np.ndarray]:
            """Render the environment."""
            if hasattr(self, "_delegate") and self._delegate is not None:
                return self._delegate.render()

            if self.render_mode == "rgb_array":
                renderer = mujoco.Renderer(self._sim._world._model, self.render_height, self.render_width)
                renderer.update_scene(self._sim._world._data)
                frame = renderer.render().copy()
                # mujoco 3.x Renderer has no close() — it's garbage-collected
                del renderer
                return frame
            elif self.render_mode == "human":
                # Use MuJoCo viewer
                if not hasattr(self, "_viewer") or self._viewer is None:
                    self._viewer = mujoco.viewer.launch_passive(self._sim._world._model, self._sim._world._data)
                self._viewer.sync()
                return None

        def close(self):
            """Clean up resources."""
            if hasattr(self, "_delegate") and self._delegate is not None:
                return self._delegate.close()

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
        gym.spec("StrandsSim-v0")
    except Exception:
        # Not registered yet (spec() raises in some gym versions when not found)
        gym.register(
            id="StrandsSim-v0",
            entry_point="strands_robots.envs:StrandsSimEnv",
        )

else:
    # Stub when gymnasium not installed
    class StrandsSimEnv:
        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium required: pip install gymnasium")
