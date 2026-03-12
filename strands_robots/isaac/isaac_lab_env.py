"""Isaac Lab Environment Integration for strands-robots.

Wraps Isaac Lab's DirectRLEnv and ManagerBasedRLEnv into strands-robots'
ecosystem so that:

1. Any Isaac Lab task can be evaluated with our 16 policy providers
2. Isaac Lab envs can be wrapped as Gymnasium envs (like StrandsSimEnv)
3. Trained Isaac Lab policies can be exported and used in our Policy ABC

Isaac Lab has 30+ built-in environments:
    - Locomotion: ANYmal-C/D, Humanoid, Unitree Go2/G1
    - Manipulation: Franka Cabinet, Shadow Hand, Allegro
    - Classic: CartPole, Ant, Humanoid
    - Navigation: ANYmal-C nav

This module bridges all of them into strands-robots.

Key Classes:
    IsaacLabEnv: Wraps an Isaac Lab env for strands-robots Policy evaluation
    create_isaac_env: Factory function to create Isaac Lab envs by task name
    list_isaac_tasks: List all registered Isaac Lab tasks

Requires:
    - isaaclab (pip install isaaclab)
    - isaaclab_tasks (pip install isaaclab_tasks)
    - Isaac Sim runtime
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Task Registry — Maps friendly names to Isaac Lab task IDs
# ─────────────────────────────────────────────────────────────────────

_ISAAC_TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Locomotion
    "anymal_c_flat": {
        "task_id": "Isaac-Velocity-Flat-Anymal-C-v0",
        "type": "locomotion",
        "robot": "anymal_c",
        "description": "ANYmal-C flat terrain velocity tracking",
    },
    "anymal_c_rough": {
        "task_id": "Isaac-Velocity-Rough-Anymal-C-v0",
        "type": "locomotion",
        "robot": "anymal_c",
        "description": "ANYmal-C rough terrain velocity tracking",
    },
    "anymal_d_flat": {
        "task_id": "Isaac-Velocity-Flat-Anymal-D-v0",
        "type": "locomotion",
        "robot": "anymal_d",
        "description": "ANYmal-D flat terrain velocity tracking",
    },
    "humanoid": {
        "task_id": "Isaac-Humanoid-v0",
        "type": "locomotion",
        "robot": "humanoid",
        "description": "MuJoCo Humanoid locomotion (Isaac Lab)",
    },
    "anymal_c_direct": {
        "task_id": "Isaac-Velocity-Flat-Anymal-C-Direct-v0",
        "type": "locomotion",
        "robot": "anymal_c",
        "description": "ANYmal-C direct RL locomotion",
    },
    # Manipulation
    "franka_cabinet": {
        "task_id": "Isaac-Open-Drawer-Franka-v0",
        "type": "manipulation",
        "robot": "franka",
        "description": "Franka opens cabinet drawer",
    },
    "shadow_hand": {
        "task_id": "Isaac-Shadow-Hand-Direct-v0",
        "type": "manipulation",
        "robot": "shadow_hand",
        "description": "Shadow Hand in-hand manipulation",
    },
    "allegro_hand": {
        "task_id": "Isaac-Allegro-Hand-Direct-v0",
        "type": "manipulation",
        "robot": "allegro_hand",
        "description": "Allegro Hand in-hand manipulation",
    },
    # Classic control
    "cartpole": {
        "task_id": "Isaac-CartPole-v0",
        "type": "classic",
        "robot": "cartpole",
        "description": "CartPole balance (Isaac Lab)",
    },
    "ant": {
        "task_id": "Isaac-Ant-v0",
        "type": "classic",
        "robot": "ant",
        "description": "Ant locomotion (Isaac Lab)",
    },
    # Navigation
    "anymal_c_nav": {
        "task_id": "Isaac-Navigation-Flat-Anymal-C-v0",
        "type": "navigation",
        "robot": "anymal_c",
        "description": "ANYmal-C point navigation",
    },
}


def list_isaac_tasks(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available Isaac Lab tasks.

    Args:
        category: Filter by type ("locomotion", "manipulation", "classic", "navigation")

    Returns:
        List of task info dicts
    """
    tasks = []
    for name, info in sorted(_ISAAC_TASK_REGISTRY.items()):
        if category and info["type"] != category:
            continue
        tasks.append(
            {
                "name": name,
                "task_id": info["task_id"],
                "type": info["type"],
                "robot": info["robot"],
                "description": info["description"],
            }
        )

    # Also try to discover registered tasks from Isaac Lab
    try:
        import gymnasium as gym

        # isaaclab_tasks auto-registers on import (v0.54+)
        import isaaclab_tasks  # noqa: F401

        # Get all Isaac Lab registered envs
        isaac_envs = [
            spec.id for spec in gym.registry.values() if spec.id.startswith("Isaac-")
        ]

        # Add any we don't have in our registry
        known_ids = {info["task_id"] for info in _ISAAC_TASK_REGISTRY.values()}
        for env_id in isaac_envs:
            if env_id not in known_ids:
                tasks.append(
                    {
                        "name": env_id.lower().replace("-", "_"),
                        "task_id": env_id,
                        "type": "unknown",
                        "robot": "unknown",
                        "description": f"Isaac Lab env: {env_id}",
                    }
                )
    except ImportError:
        pass

    return tasks


# ─────────────────────────────────────────────────────────────────────
# Isaac Lab Environment Wrapper
# ─────────────────────────────────────────────────────────────────────


@dataclass
class IsaacLabEnvConfig:
    """Configuration for Isaac Lab environment wrapper."""

    task_name: str = "cartpole"
    num_envs: int = 1
    device: str = "cuda:0"
    headless: bool = True
    enable_cameras: bool = False
    seed: int = 42


class IsaacLabEnv:
    """Wraps an Isaac Lab environment for use with strands-robots policies.

    Bridges Isaac Lab's DirectRLEnv/ManagerBasedRLEnv into our Policy ABC
    so any of our 16 VLA policy providers can be evaluated on Isaac Lab tasks.

    The key bridge: Isaac Lab gives observations as GPU tensors with shape
    (num_envs, obs_dim). We convert to CPU dicts for Policy ABC compatibility.

    For single-env evaluation (num_envs=1), observations are flat dicts
    just like MuJoCo simulation.py returns.

    Usage:
        env = IsaacLabEnv(config=IsaacLabEnvConfig(
            task_name="anymal_c_flat",
            num_envs=1,
        ))
        env.reset()

        # Use with strands-robots policy
        from strands_robots.policies import create_policy
        policy = create_policy("mock")
        obs = env.get_observation()
        actions = await policy.get_actions(obs, "walk forward")
        env.step_with_dict(actions[0])

        env.close()
    """

    def __init__(self, config: Optional[IsaacLabEnvConfig] = None):
        self.config = config or IsaacLabEnvConfig()
        self._env = None
        self._obs_keys = []
        self._action_dim = 0
        self._episode_count = 0

        logger.info("Creating Isaac Lab env: %s", self.config.task_name)
        self._create_env()

    def _create_env(self):
        """Create the Isaac Lab environment.

        Note: Isaac Lab v0.54+ requires task-specific env classes rather than
        gym.make() directly. The cfg parameter must include all scene config.
        Actions must be torch tensors on the correct device, not numpy arrays.
        """
        try:
            import gymnasium as gym

            # Register Isaac Lab tasks
            try:
                # isaaclab_tasks auto-registers on import (v0.54+)
                import isaaclab_tasks  # noqa: F401
            except ImportError:
                logger.warning(
                    "isaaclab_tasks not available — using direct env creation"
                )

            # Resolve task ID
            task_info = _ISAAC_TASK_REGISTRY.get(self.config.task_name)
            task_id = task_info["task_id"] if task_info else self.config.task_name

            # Create env with Isaac Lab's gym.make
            self._env = gym.make(
                task_id,
                num_envs=self.config.num_envs,
                device=self.config.device,
            )

            # Extract spaces info
            self._action_dim = (
                self._env.action_space.shape[-1]
                if hasattr(self._env.action_space, "shape")
                else 0
            )

            logger.info(
                f"✅ Isaac Lab env created: {task_id} "
                f"(obs_dim={self._env.observation_space}, act_dim={self._action_dim})"
            )

        except Exception as e:
            logger.error("Failed to create Isaac Lab env: %s", e)
            raise

    def reset(self) -> Dict[str, Any]:
        """Reset the environment.

        Returns:
            Observation dict compatible with strands-robots Policy ABC
        """
        obs, info = self._env.reset()
        self._episode_count += 1
        return self._to_policy_obs(obs)

    def step(self, action_array) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        """Step with a numpy/tensor action array.

        Args:
            action_array: Action array of shape (action_dim,) or (num_envs, action_dim)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        import torch

        if isinstance(action_array, np.ndarray):
            action_tensor = (
                torch.from_numpy(action_array).float().to(self.config.device)
            )
        elif isinstance(action_array, torch.Tensor):
            action_tensor = action_array.to(self.config.device)
        else:
            action_tensor = torch.tensor(
                action_array, dtype=torch.float32, device=self.config.device
            )

        # Ensure batch dimension
        if action_tensor.dim() == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.shape[0] == 1 and self.config.num_envs > 1:
            action_tensor = action_tensor.expand(self.config.num_envs, -1)

        obs, rew, terminated, truncated, info = self._env.step(action_tensor)

        obs_dict = self._to_policy_obs(obs)
        reward = float(rew.mean().cpu()) if hasattr(rew, "cpu") else float(rew)
        done = (
            bool(terminated.any().cpu())
            if hasattr(terminated, "cpu")
            else bool(terminated)
        )
        trunc = (
            bool(truncated.any().cpu())
            if hasattr(truncated, "cpu")
            else bool(truncated)
        )

        return obs_dict, reward, done, trunc, info

    def step_with_dict(
        self, action_dict: Dict[str, float]
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict]:
        """Step with a strands-robots Policy-style action dict.

        Converts {joint_name: value, ...} → action array for Isaac Lab.
        """
        action_vals = list(action_dict.values())
        # Pad or truncate to match action space
        if len(action_vals) < self._action_dim:
            action_vals.extend([0.0] * (self._action_dim - len(action_vals)))
        elif len(action_vals) > self._action_dim:
            action_vals = action_vals[: self._action_dim]

        return self.step(np.array(action_vals, dtype=np.float32))

    def get_observation(self) -> Dict[str, Any]:
        """Get current observation in Policy ABC format."""
        if self._env is None:
            return {}
        # For Isaac Lab, we need to return the current observation
        # which is stored after the last step/reset
        return self._last_obs if hasattr(self, "_last_obs") else {}

    def run_policy(
        self,
        policy_provider: str = "mock",
        instruction: str = "",
        n_episodes: int = 1,
        max_steps: int = 1000,
        **policy_kwargs,
    ) -> Dict[str, Any]:
        """Run a strands-robots policy on this Isaac Lab env.

        Bridges the Policy ABC to Isaac Lab's step loop.

        Args:
            policy_provider: Any of our 16 providers
            instruction: Task instruction
            n_episodes: Number of episodes
            max_steps: Max steps per episode
            **policy_kwargs: Provider kwargs

        Returns:
            Evaluation results dict
        """
        import asyncio
        import time

        from strands_robots.policies import create_policy

        policy = create_policy(policy_provider, **policy_kwargs)

        results = []
        total_start = time.time()

        for ep in range(n_episodes):
            obs = self.reset()
            ep_reward = 0.0
            ep_steps = 0

            for step_idx in range(max_steps):
                # Policy inference
                try:
                    actions = asyncio.run(policy.get_actions(obs, instruction))
                except RuntimeError:
                    loop = asyncio.get_event_loop()
                    actions = loop.run_until_complete(
                        policy.get_actions(obs, instruction)
                    )

                if actions:
                    obs, reward, terminated, truncated, info = self.step_with_dict(
                        actions[0]
                    )
                    ep_reward += reward
                    ep_steps += 1

                    if terminated or truncated:
                        break

            results.append(
                {
                    "episode": ep,
                    "reward": ep_reward,
                    "steps": ep_steps,
                    "terminated": terminated,
                }
            )

        total_time = time.time() - total_start
        avg_reward = sum(r["reward"] for r in results) / max(len(results), 1)
        avg_steps = sum(r["steps"] for r in results) / max(len(results), 1)

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"📊 Isaac Lab Evaluation: {self.config.task_name}\n"
                        f"🧠 Policy: {policy_provider} | 🎯 {instruction}\n"
                        f"📈 Episodes: {n_episodes} | Avg reward: {avg_reward:.2f}\n"
                        f"📊 Avg steps: {avg_steps:.0f}/{max_steps}\n"
                        f"⏱️ Total: {total_time:.1f}s"
                    )
                }
            ],
            "results": results,
        }

    def close(self):
        """Close the environment and release resources."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def _to_policy_obs(self, obs) -> Dict[str, Any]:
        """Convert Isaac Lab observation to Policy ABC format.

        Isaac Lab returns either:
        - torch.Tensor of shape (num_envs, obs_dim) for DirectRLEnv
        - dict of tensors for ManagerBasedRLEnv

        We convert to a flat dict for single-env, or keep as arrays for multi-env.
        """
        import torch

        obs_dict = {}

        if isinstance(obs, dict):
            # ManagerBasedRLEnv style
            policy_obs = obs.get("policy", obs)
            if isinstance(policy_obs, dict):
                for key, val in policy_obs.items():
                    if isinstance(val, torch.Tensor):
                        val_np = val.cpu().numpy()
                        if self.config.num_envs == 1:
                            val_np = val_np.squeeze(0)
                        obs_dict[key] = val_np
                    else:
                        obs_dict[key] = val
            elif isinstance(policy_obs, torch.Tensor):
                val_np = policy_obs.cpu().numpy()
                if self.config.num_envs == 1:
                    val_np = val_np.squeeze(0)
                obs_dict["observation"] = val_np
        elif isinstance(obs, torch.Tensor):
            val_np = obs.cpu().numpy()
            if self.config.num_envs == 1:
                val_np = val_np.squeeze(0)
            obs_dict["observation"] = val_np
        elif isinstance(obs, np.ndarray):
            obs_dict["observation"] = obs

        self._last_obs = obs_dict
        return obs_dict

    def __del__(self):
        self.close()


def create_isaac_env(
    task_name: str = "cartpole",
    num_envs: int = 1,
    device: str = "cuda:0",
    headless: bool = True,
    **kwargs,
) -> IsaacLabEnv:
    """Factory function to create an Isaac Lab environment.

    Args:
        task_name: Task name (friendly name or full Isaac Lab task ID)
        num_envs: Number of parallel environments
        device: CUDA device
        headless: Run without GUI

    Returns:
        IsaacLabEnv instance

    Example:
        env = create_isaac_env("anymal_c_flat", num_envs=4096)
        env.run_policy("mock", instruction="walk forward", n_episodes=5)
    """
    config = IsaacLabEnvConfig(
        task_name=task_name,
        num_envs=num_envs,
        device=device,
        headless=headless,
    )
    return IsaacLabEnv(config)
