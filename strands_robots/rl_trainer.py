#!/usr/bin/env python3
"""
RL Training Module for Newton GPU-Accelerated Environments

Provides PPO and SAC training wrappers using stable-baselines3,
targeting Newton's GPU-parallel environments (4096+ envs).

This fills the gap in the training matrix — create_trainer() covers
imitation learning (GR00T, LeRobot, DreamGen, Cosmos) but not RL.

Usage:
    from strands_robots.rl_trainer import create_rl_trainer

    # PPO for locomotion (4096 parallel envs)
    trainer = create_rl_trainer(
        algorithm="ppo",
        env_config={
            "robot_name": "unitree_g1",
            "task": "walk forward at 1 m/s",
            "backend": "newton",
            "num_envs": 4096,
        },
        total_timesteps=10_000_000,
        output_dir="./ppo_g1_locomotion",
    )
    trainer.train()

    # SAC for manipulation (1024 envs)
    trainer = create_rl_trainer(
        algorithm="sac",
        env_config={
            "robot_name": "so100",
            "task": "pick up the red cube",
            "backend": "mujoco",
        },
        total_timesteps=2_000_000,
        output_dir="./sac_pick_cube",
    )
    trainer.train()
"""

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for RL training."""

    # Algorithm
    algorithm: str = "ppo"  # "ppo" or "sac"

    # Environment
    robot_name: str = "so100"
    task: str = "manipulation task"
    backend: str = "mujoco"  # "mujoco" or "newton"
    num_envs: int = 1

    # Newton-specific
    newton_solver: str = "featherstone"
    newton_dt: float = 0.005
    enable_cuda_graphs: bool = True

    # Training
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    n_steps: int = 128  # PPO rollout length
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2  # PPO
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    buffer_size: int = 1_000_000  # SAC replay buffer
    tau: float = 0.005  # SAC soft update

    # Output
    output_dir: str = "./rl_outputs"
    save_freq: int = 50000
    eval_freq: int = 10000
    eval_episodes: int = 10
    log_interval: int = 10
    seed: int = 42

    # Reward
    reward_type: str = "default"  # "default", "sparse", "dense"

    # Misc
    device: str = "auto"
    verbose: int = 1
    use_wandb: bool = False


class RewardFunction:
    """Collection of reward functions for common tasks."""

    @staticmethod
    def locomotion_reward(
        obs: Any,
        action: Any,
        target_velocity: float = 1.0,
        energy_penalty: float = 0.01,
        alive_bonus: float = 1.0,
    ) -> float:
        """Reward for locomotion tasks (walk, run).

        Components:
        - Forward velocity tracking
        - Energy penalty (action magnitude)
        - Alive bonus (upright)
        """
        if isinstance(obs, dict):
            state = obs.get("state", obs.get("observation.state", np.zeros(1)))
        else:
            state = obs

        state = np.asarray(state).flatten()

        # Simple forward velocity reward (use qvel[0] as x-velocity)
        # This is a generic version — specific robots need custom obs parsing
        forward_vel = state[len(state) // 2] if len(state) > 1 else 0.0
        vel_reward = -abs(forward_vel - target_velocity)

        # Energy penalty
        action_arr = np.asarray(action).flatten() if action is not None else np.zeros(1)
        energy = energy_penalty * np.sum(action_arr**2)

        # Alive bonus (assume robot is alive if we got here)
        reward = alive_bonus + vel_reward - energy

        return float(reward)

    @staticmethod
    def manipulation_reward(
        obs: Any,
        action: Any,
        target_pos: Optional[np.ndarray] = None,
        grasp_bonus: float = 10.0,
        reach_scale: float = 1.0,
    ) -> float:
        """Reward for manipulation tasks (pick, place, stack).

        Components:
        - Distance to target object
        - Grasp success bonus
        - Action smoothness
        """
        if isinstance(obs, dict):
            state = obs.get("state", obs.get("observation.state", np.zeros(1)))
        else:
            state = obs

        state = np.asarray(state).flatten()

        # Generic distance-based reward
        if target_pos is not None:
            # Use end-effector position from state (last 3 of qpos usually)
            n_qpos = len(state) // 2
            ee_pos = state[max(0, n_qpos - 3) : n_qpos]
            if len(ee_pos) >= 3 and len(target_pos) >= 3:
                dist = np.linalg.norm(ee_pos - target_pos[:3])
                reach_reward = -reach_scale * dist
            else:
                reach_reward = 0.0
        else:
            reach_reward = 0.0

        # Action smoothness penalty
        action_arr = np.asarray(action).flatten() if action is not None else np.zeros(1)
        smoothness = -0.001 * np.sum(action_arr**2)

        return float(reach_reward + smoothness)

    @staticmethod
    def sparse_success_reward(obs: Any, action: Any, success: bool = False) -> float:
        """Binary sparse reward — 1.0 on success, 0.0 otherwise."""
        return 1.0 if success else 0.0


class PickAndPlaceReward:
    """4-phase structured reward function for pick-and-place tasks.

    Designed for Isaac Sim / MuJoCo manipulation environments with
    single-arm robots (SO-101, Panda, UR5e). Implements curriculum-style
    phase progression:

        Phase 1 — Reach:     Move end-effector toward the target object.
        Phase 2 — Grasp:     Close gripper on object; detect contact + lift.
        Phase 3 — Transport: Carry object toward the placement target.
        Phase 4 — Place:     Release object at the target and verify stability.

    The reward is dense within each phase and includes shaping terms for
    energy efficiency, action smoothness, and grasp stability.

    Usage with ``create_rl_trainer``::

        reward = PickAndPlaceReward(
            object_pos_indices=(7, 10),    # qpos[7:10] = object xyz
            ee_pos_indices=(0, 3),         # qpos[0:3] = EE xyz (or derived)
            gripper_index=6,               # qpos[6] = gripper aperture
            target_place_pos=np.array([0.3, 0.0, 0.75]),
        )
        trainer = create_rl_trainer(
            algorithm="ppo",
            env_config={
                "robot_name": "so101",
                "task": "pick and place cube",
                "backend": "isaac",
                "num_envs": 50,
            },
            reward_fn=reward,
        )

    Usage with ``StrandsSimEnv``::

        reward = PickAndPlaceReward(...)
        env = StrandsSimEnv(
            robot_name="so101",
            task="pick and place",
            reward_fn=reward,
        )

    The class is callable: ``reward(obs, action) -> float``.

    Refs:
        - Issue #124: Marble 3D → Isaac Sim Pick-and-Place Pipeline
        - Issue #65: GR00T N1.6 Full Lifecycle
    """

    # Phase constants
    PHASE_REACH = 0
    PHASE_GRASP = 1
    PHASE_TRANSPORT = 2
    PHASE_PLACE = 3

    def __init__(
        self,
        # Observation layout — indices into the flat state vector
        object_pos_indices: tuple = (7, 10),
        ee_pos_indices: tuple = (0, 3),
        gripper_index: int = 6,
        contact_force_index: Optional[int] = None,
        # Target positions
        target_place_pos: Optional[np.ndarray] = None,
        # Phase thresholds
        reach_threshold: float = 0.05,
        lift_height: float = 0.10,
        place_threshold: float = 0.05,
        gripper_closed_threshold: float = 0.02,
        # Reward weights
        reach_scale: float = 2.0,
        grasp_bonus: float = 5.0,
        lift_bonus: float = 3.0,
        transport_scale: float = 2.0,
        place_bonus: float = 10.0,
        stability_bonus: float = 5.0,
        energy_penalty: float = 0.001,
        action_smoothness_penalty: float = 0.0005,
        # Contact detection
        contact_force_threshold: float = 0.1,
        # Phase auto-advance
        auto_advance: bool = True,
    ):
        """Initialise the pick-and-place reward function.

        Args:
            object_pos_indices: ``(start, end)`` slice into ``qpos`` for the
                object's XYZ position.  For Isaac Sim ``DirectRLEnv`` the
                observation layout depends on the env; for MuJoCo this is
                the freejoint's first 3 elements.
            ee_pos_indices: ``(start, end)`` slice into ``qpos`` for the
                end-effector position (or the last 3 qpos of a serial arm).
            gripper_index: Index of the gripper aperture in ``qpos`` (0 =
                closed for SO-101/LeRobot convention).
            contact_force_index: Optional index into observation for the
                net contact force magnitude at the gripper.  When *None*,
                gripper closure is used as a proxy for grasp.
            target_place_pos: XYZ target where the object should be placed.
                Defaults to ``[0.3, 0.0, 0.75]`` (slightly forward on table).
            reach_threshold: Distance (m) at which Phase 1 completes.
            lift_height: Height (m) above initial Z the object must reach
                for Phase 2 to complete.
            place_threshold: Distance (m) from ``target_place_pos`` at which
                Phase 4 succeeds.
            gripper_closed_threshold: Gripper aperture below which the
                gripper is considered "closed".
            reach_scale: Scaling factor for the reach distance reward.
            grasp_bonus: One-time bonus when grasp is detected.
            lift_bonus: Bonus for lifting the object above ``lift_height``.
            transport_scale: Scaling factor for transport distance reward.
            place_bonus: Bonus for successfully placing the object.
            stability_bonus: Extra reward if the object remains stable after
                placement for a short period (checked via velocity).
            energy_penalty: Per-step penalty on sum of action² (energy).
            action_smoothness_penalty: Per-step penalty on action changes.
            contact_force_threshold: Minimum contact force to count as grasp.
            auto_advance: Automatically advance phases based on observations.
        """
        self.obj_idx = object_pos_indices
        self.ee_idx = ee_pos_indices
        self.gripper_idx = gripper_index
        self.contact_idx = contact_force_index
        self.target_place = (
            np.array(target_place_pos, dtype=np.float64)
            if target_place_pos is not None
            else np.array([0.3, 0.0, 0.75], dtype=np.float64)
        )

        # Thresholds
        self.reach_thresh = reach_threshold
        self.lift_height = lift_height
        self.place_thresh = place_threshold
        self.grip_closed_thresh = gripper_closed_threshold
        self.contact_thresh = contact_force_threshold

        # Reward weights
        self.reach_scale = reach_scale
        self.grasp_bonus = grasp_bonus
        self.lift_bonus = lift_bonus
        self.transport_scale = transport_scale
        self.place_bonus = place_bonus
        self.stability_bonus = stability_bonus
        self.energy_penalty = energy_penalty
        self.smooth_penalty = action_smoothness_penalty

        self.auto_advance = auto_advance

        # Internal state
        self._phase = self.PHASE_REACH
        self._prev_action: Optional[np.ndarray] = None
        self._initial_obj_z: Optional[float] = None
        self._grasp_awarded = False
        self._lift_awarded = False
        self._place_awarded = False
        self._steps_at_target = 0

    # ── Core interface ───────────────────────────────────────────

    def __call__(self, obs: Any, action: Any) -> float:
        """Compute reward for the current observation and action.

        This is the callable interface expected by ``StrandsSimEnv``,
        ``IsaacGymEnv``, and ``rl_trainer._get_reward_fn()``.

        Args:
            obs: Observation — either a flat numpy array (``qpos + qvel``)
                or a dict with a ``"state"`` key.
            action: Action array applied this step.

        Returns:
            Scalar reward (float).
        """
        state = self._extract_state(obs)
        action_arr = np.asarray(action, dtype=np.float64).flatten() if action is not None else np.zeros(1)

        # Extract positions
        ee_pos = state[self.ee_idx[0] : self.ee_idx[1]]
        obj_pos = state[self.obj_idx[0] : self.obj_idx[1]]
        gripper = state[self.gripper_idx] if self.gripper_idx < len(state) else 0.0

        # Contact force (if available)
        contact_force = (
            state[self.contact_idx] if self.contact_idx is not None and self.contact_idx < len(state) else None
        )

        # Cache initial object Z for lift detection
        if self._initial_obj_z is None and len(obj_pos) >= 3:
            self._initial_obj_z = float(obj_pos[2])

        reward = 0.0

        # ── Phase 1: Reach ─────────────────────────────────────
        if self._phase == self.PHASE_REACH:
            dist_to_obj = np.linalg.norm(ee_pos - obj_pos) if len(ee_pos) >= 3 and len(obj_pos) >= 3 else 1.0
            # Dense shaping: negative distance
            reward += self.reach_scale * (1.0 - np.tanh(5.0 * dist_to_obj))

            if self.auto_advance and dist_to_obj < self.reach_thresh:
                self._phase = self.PHASE_GRASP
                logger.debug("PickAndPlace: Phase 1→2 (Reach→Grasp)")

        # ── Phase 2: Grasp ─────────────────────────────────────
        elif self._phase == self.PHASE_GRASP:
            # Keep rewarding proximity
            dist_to_obj = np.linalg.norm(ee_pos - obj_pos) if len(ee_pos) >= 3 and len(obj_pos) >= 3 else 1.0
            reward += self.reach_scale * (1.0 - np.tanh(5.0 * dist_to_obj))

            # Detect grasp
            is_grasping = self._detect_grasp(gripper, contact_force)
            if is_grasping and not self._grasp_awarded:
                reward += self.grasp_bonus
                self._grasp_awarded = True

            # Detect lift
            if is_grasping and len(obj_pos) >= 3 and self._initial_obj_z is not None:
                height_gained = obj_pos[2] - self._initial_obj_z
                # Continuous lift reward
                reward += self.lift_bonus * np.clip(height_gained / self.lift_height, 0.0, 1.0)

                if height_gained > self.lift_height and not self._lift_awarded:
                    reward += self.lift_bonus
                    self._lift_awarded = True
                    if self.auto_advance:
                        self._phase = self.PHASE_TRANSPORT
                        logger.debug("PickAndPlace: Phase 2→3 (Grasp→Transport)")

        # ── Phase 3: Transport ─────────────────────────────────
        elif self._phase == self.PHASE_TRANSPORT:
            # Distance from object to target placement
            dist_to_target = np.linalg.norm(obj_pos - self.target_place) if len(obj_pos) >= 3 else 1.0
            reward += self.transport_scale * (1.0 - np.tanh(3.0 * dist_to_target))

            # Penalise dropping (object Z falls back to table)
            if len(obj_pos) >= 3 and self._initial_obj_z is not None:
                if obj_pos[2] < self._initial_obj_z + 0.02:
                    # Object was dropped — go back to reach
                    reward -= 2.0
                    if self.auto_advance:
                        self._phase = self.PHASE_REACH
                        self._grasp_awarded = False
                        self._lift_awarded = False
                        logger.debug("PickAndPlace: Phase 3→1 (dropped, back to Reach)")

            if self.auto_advance and dist_to_target < self.place_thresh:
                self._phase = self.PHASE_PLACE
                self._steps_at_target = 0
                logger.debug("PickAndPlace: Phase 3→4 (Transport→Place)")

        # ── Phase 4: Place ─────────────────────────────────────
        elif self._phase == self.PHASE_PLACE:
            dist_to_target = np.linalg.norm(obj_pos - self.target_place) if len(obj_pos) >= 3 else 1.0
            reward += self.transport_scale * (1.0 - np.tanh(3.0 * dist_to_target))

            # Reward opening gripper at target
            gripper_open = gripper > self.grip_closed_thresh
            if gripper_open and dist_to_target < self.place_thresh:
                if not self._place_awarded:
                    reward += self.place_bonus
                    self._place_awarded = True
                    logger.debug("PickAndPlace: Place bonus awarded!")

                # Stability check — object should stay at target
                self._steps_at_target += 1
                if self._steps_at_target >= 10:
                    reward += self.stability_bonus

        # ── Global penalties ───────────────────────────────────
        # Energy penalty
        reward -= self.energy_penalty * float(np.sum(action_arr**2))

        # Action smoothness penalty
        if self._prev_action is not None and len(action_arr) == len(self._prev_action):
            delta = action_arr - self._prev_action
            reward -= self.smooth_penalty * float(np.sum(delta**2))

        self._prev_action = action_arr.copy()

        return float(reward)

    # ── Helper methods ───────────────────────────────────────────

    def _extract_state(self, obs: Any) -> np.ndarray:
        """Extract flat state vector from observation."""
        if isinstance(obs, dict):
            state = obs.get(
                "state", obs.get("observation.state", obs.get("observation", obs.get("policy", np.zeros(1))))
            )
            if isinstance(state, dict):
                # ManagerBasedRLEnv: concatenate all values
                vals = []
                for v in state.values():
                    vals.append(np.asarray(v).flatten())
                return np.concatenate(vals) if vals else np.zeros(1)
            return np.asarray(state, dtype=np.float64).flatten()
        return np.asarray(obs, dtype=np.float64).flatten()

    def _detect_grasp(self, gripper_val: float, contact_force: Optional[float]) -> bool:
        """Detect whether the robot has grasped the object.

        Uses contact force if available, otherwise falls back to gripper
        aperture as a proxy.
        """
        gripper_closed = gripper_val < self.grip_closed_thresh

        if contact_force is not None:
            return gripper_closed and contact_force > self.contact_thresh

        # Proxy: gripper is closed
        return gripper_closed

    def reset(self) -> None:
        """Reset internal state for a new episode.

        Call this at the start of each episode (``env.reset()``). If using
        ``StrandsSimEnv`` the env does NOT call this automatically — wrap
        in a ``gymnasium.Wrapper`` or call explicitly.
        """
        self._phase = self.PHASE_REACH
        self._prev_action = None
        self._initial_obj_z = None
        self._grasp_awarded = False
        self._lift_awarded = False
        self._place_awarded = False
        self._steps_at_target = 0

    @property
    def current_phase(self) -> int:
        """Current reward phase (0=Reach, 1=Grasp, 2=Transport, 3=Place)."""
        return self._phase

    @property
    def phase_name(self) -> str:
        """Human-readable name of the current phase."""
        names = {0: "Reach", 1: "Grasp", 2: "Transport", 3: "Place"}
        return names.get(self._phase, "Unknown")

    @property
    def is_success(self) -> bool:
        """Whether the task has been completed (place bonus awarded)."""
        return self._place_awarded

    def get_info(self) -> Dict[str, Any]:
        """Return diagnostic info for logging."""
        return {
            "phase": self._phase,
            "phase_name": self.phase_name,
            "is_success": self.is_success,
            "grasp_awarded": self._grasp_awarded,
            "lift_awarded": self._lift_awarded,
            "place_awarded": self._place_awarded,
            "steps_at_target": self._steps_at_target,
        }

    def __repr__(self) -> str:
        return (
            f"PickAndPlaceReward("
            f"phase={self.phase_name}, "
            f"target={self.target_place.tolist()}, "
            f"success={self.is_success})"
        )


class RLTrainer(ABC):
    """Abstract base class for RL trainers."""

    @abstractmethod
    def train(self) -> Dict[str, Any]:
        """Run RL training loop."""
        pass

    @abstractmethod
    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate trained policy."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save trained model."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load trained model."""
        pass


class SB3Trainer(RLTrainer):
    """RL Trainer using stable-baselines3 (PPO/SAC).

    Supports both single-env MuJoCo and vectorized Newton environments.
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self._model = None
        self._env = None

    def _create_env(self):
        """Create the training environment."""
        if self.config.backend == "newton":
            return self._create_newton_env()
        else:
            return self._create_mujoco_env()

    def _create_mujoco_env(self):
        """Create MuJoCo-backed Gymnasium environment."""
        from strands_robots.envs import StrandsSimEnv

        # Select reward function
        reward_fn = self._get_reward_fn()

        env = StrandsSimEnv(
            robot_name=self.config.robot_name,
            task=self.config.task,
            render_mode=None,  # No rendering during training
            reward_fn=reward_fn,
        )

        return env

    def _create_newton_env(self):
        """Create Newton GPU-backed environment.

        Newton environments provide batched obs/actions for massive parallelism.
        Uses NewtonGymEnv which wraps NewtonBackend in a gym.Env interface.
        """
        try:
            from strands_robots.newton import NewtonConfig
            from strands_robots.newton.newton_gym_env import NewtonGymEnv

            newton_config = NewtonConfig(
                num_envs=self.config.num_envs,
                solver=self.config.newton_solver,
                physics_dt=self.config.newton_dt,
                enable_cuda_graph=self.config.enable_cuda_graphs,
                device="cuda:0",
            )

            reward_fn = self._get_reward_fn()

            env = NewtonGymEnv(
                robot_name=self.config.robot_name,
                task=self.config.task,
                config=newton_config,
                render_mode=None,
                reward_fn=reward_fn,
            )

            logger.info(
                f"🚀 Newton GPU env created: {self.config.num_envs} envs, " f"solver={self.config.newton_solver}"
            )
            return env

        except ImportError as e:
            logger.warning(
                f"Newton backend not available ({e}), falling back to MuJoCo. "
                "Install: pip install newton-sim warp-lang"
            )
            return self._create_mujoco_env()
        except Exception as e:
            logger.warning(f"Newton env creation failed ({e}), falling back to MuJoCo")
            return self._create_mujoco_env()

    def _get_reward_fn(self) -> Optional[Callable]:
        """Get reward function based on config.

        Returns a callable ``(obs, action) -> float``.  For pick-and-place
        tasks detected via the task string, returns a
        :class:`PickAndPlaceReward` instance (callable with per-episode
        ``reset()``).  The 4-phase reward (Reach→Grasp→Transport→Place) is
        crucial for the Marble 3D → Isaac Sim training pipeline (#124).
        """
        task_lower = self.config.task.lower()

        if "walk" in task_lower or "locomotion" in task_lower or "run" in task_lower:
            return lambda obs, action: RewardFunction.locomotion_reward(obs, action)
        elif "pick" in task_lower and "place" in task_lower:
            # 4-phase structured reward for pick-and-place (#124)
            return PickAndPlaceReward()
        elif "pick" in task_lower or "grasp" in task_lower or "cube" in task_lower:
            return lambda obs, action: RewardFunction.manipulation_reward(obs, action)
        else:
            return None

    def train(self) -> Dict[str, Any]:
        """Run SB3 training."""
        try:
            from stable_baselines3 import PPO, SAC
            from stable_baselines3.common.callbacks import EvalCallback  # noqa: F401
            from stable_baselines3.common.callbacks import (
                CheckpointCallback,
            )
        except ImportError:
            logger.error("stable-baselines3 required: pip install stable-baselines3")
            return {
                "status": "error",
                "error": "stable-baselines3 not installed",
                "install": "pip install stable-baselines3",
            }

        os.makedirs(self.config.output_dir, exist_ok=True)

        # Create environment
        self._env = self._create_env()

        # Create model
        algo_cls = PPO if self.config.algorithm == "ppo" else SAC

        model_kwargs = {
            "policy": "MlpPolicy",
            "env": self._env,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "gamma": self.config.gamma,
            "verbose": self.config.verbose,
            "seed": self.config.seed,
            "device": self.config.device,
        }

        if self.config.algorithm == "ppo":
            model_kwargs.update(
                {
                    "n_steps": self.config.n_steps,
                    "gae_lambda": self.config.gae_lambda,
                    "clip_range": self.config.clip_range,
                    "ent_coef": self.config.ent_coef,
                    "vf_coef": self.config.vf_coef,
                }
            )
        elif self.config.algorithm == "sac":
            model_kwargs.update(
                {
                    "buffer_size": self.config.buffer_size,
                    "tau": self.config.tau,
                }
            )

        self._model = algo_cls(**model_kwargs)

        # Callbacks
        callbacks = []
        checkpoint_cb = CheckpointCallback(
            save_freq=self.config.save_freq,
            save_path=os.path.join(self.config.output_dir, "checkpoints"),
            name_prefix="rl_model",
        )
        callbacks.append(checkpoint_cb)

        # Training
        logger.info(
            f"🚀 Starting {self.config.algorithm.upper()} training: "
            f"{self.config.total_timesteps} timesteps, "
            f"robot={self.config.robot_name}, backend={self.config.backend}"
        )
        start_time = time.time()

        self._model.learn(
            total_timesteps=self.config.total_timesteps,
            callback=callbacks,
            log_interval=self.config.log_interval,
        )

        elapsed = time.time() - start_time

        # Save final model
        final_path = os.path.join(self.config.output_dir, "final_model")
        self._model.save(final_path)

        # Save config
        config_path = os.path.join(self.config.output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(vars(self.config), f, indent=2)

        result = {
            "status": "success",
            "algorithm": self.config.algorithm,
            "total_timesteps": self.config.total_timesteps,
            "training_time_seconds": elapsed,
            "model_path": final_path,
            "config_path": config_path,
            "fps": self.config.total_timesteps / elapsed if elapsed > 0 else 0,
        }

        logger.info(f"✅ Training complete: {elapsed:.1f}s, " f"{result['fps']:.0f} steps/s, saved to {final_path}")
        return result

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate the trained model."""
        if self._model is None:
            return {"status": "error", "error": "No model trained or loaded"}

        if self._env is None:
            self._env = self._create_env()

        episodes = []
        total_reward = 0.0
        successes = 0

        for ep in range(num_episodes):
            obs, info = self._env.reset()
            episode_reward = 0.0
            done = False
            steps = 0

            while not done:
                action, _ = self._model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self._env.step(action)
                episode_reward += reward
                steps += 1
                done = terminated or truncated

            if info.get("is_success", False):
                successes += 1

            total_reward += episode_reward
            episodes.append(
                {
                    "episode": ep,
                    "reward": episode_reward,
                    "steps": steps,
                    "success": info.get("is_success", False),
                }
            )

        return {
            "success_rate": successes / num_episodes * 100,
            "mean_reward": total_reward / num_episodes,
            "num_episodes": num_episodes,
            "episodes": episodes,
        }

    def save(self, path: str) -> None:
        """Save the trained model."""
        if self._model:
            self._model.save(path)

    def load(self, path: str) -> None:
        """Load a trained model."""
        try:
            from stable_baselines3 import PPO, SAC

            algo_cls = PPO if self.config.algorithm == "ppo" else SAC
            self._model = algo_cls.load(path)
        except ImportError:
            raise ImportError("stable-baselines3 required")


def create_rl_trainer(
    algorithm: str = "ppo",
    env_config: Optional[Dict[str, Any]] = None,
    total_timesteps: int = 1_000_000,
    output_dir: str = "./rl_outputs",
    **kwargs,
) -> SB3Trainer:
    """Create an RL trainer instance.

    Args:
        algorithm: "ppo" or "sac"
        env_config: Environment configuration dict
        total_timesteps: Total training timesteps
        output_dir: Output directory for checkpoints
        **kwargs: Additional RLConfig parameters

    Returns:
        SB3Trainer instance

    Examples:
        # PPO for G1 locomotion
        trainer = create_rl_trainer(
            algorithm="ppo",
            env_config={"robot_name": "unitree_g1", "task": "walk forward"},
            total_timesteps=10_000_000,
        )

        # SAC for manipulation
        trainer = create_rl_trainer(
            algorithm="sac",
            env_config={"robot_name": "so100", "task": "pick up the red cube"},
            total_timesteps=2_000_000,
        )
    """
    env_config = env_config or {}

    config = RLConfig(
        algorithm=algorithm,
        robot_name=env_config.get("robot_name", "so100"),
        task=env_config.get("task", "manipulation task"),
        backend=env_config.get("backend", "mujoco"),
        num_envs=env_config.get("num_envs", 1),
        newton_solver=env_config.get("newton_solver", "featherstone"),
        total_timesteps=total_timesteps,
        output_dir=output_dir,
        **kwargs,
    )

    return SB3Trainer(config)


__all__ = [
    "RLConfig",
    "RLTrainer",
    "SB3Trainer",
    "RewardFunction",
    "PickAndPlaceReward",
    "create_rl_trainer",
]
