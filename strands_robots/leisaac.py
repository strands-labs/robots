"""LeIsaac × LeRobot EnvHub Integration for strands-robots.

Bridges Lightwheel AI's LeIsaac (IsaacLab-based environments) into
strands-robots' Policy ABC and Simulation toolchain.

LeIsaac provides photorealistic IsaacSim environments with everyday
manipulation tasks. This module enables:

1. Loading LeIsaac envs from HuggingFace EnvHub (one-liner)
2. Running strands-robots policies (any of 16 VLA providers) against them
3. Recording video of policy rollouts
4. Evaluating policies with success metrics
5. Bridging teleoperation data collection

Architecture:
    ┌───────────────────────────────────────────────────────────────┐
    │  strands-robots                                               │
    │                                                               │
    │  ┌──────────────┐     ┌───────────────────────────────────┐  │
    │  │ Policy ABC    │────▶│  LeIsaacEnv (this module)          │  │
    │  │ 16 providers  │     │  • load from HuggingFace EnvHub   │  │
    │  │ groot, act,   │     │  • gymnasium.Env interface         │  │
    │  │ diffusion,    │     │  • video recording                 │  │
    │  │ smolvla, π₀.. │     │  • policy evaluation               │  │
    │  └──────────────┘     └───────────┬───────────────────────┘  │
    │                                   │                           │
    │                       ┌───────────▼───────────────────────┐  │
    │                       │  LeRobot envs.factory.make_env    │  │
    │                       │  ↓                                │  │
    │                       │  LightwheelAI/leisaac_env (Hub)   │  │
    │                       │  ↓                                │  │
    │                       │  NVIDIA IsaacLab + IsaacSim       │  │
    │                       │  (GPU ray-traced physics & render) │  │
    │                       └───────────────────────────────────┘  │
    └───────────────────────────────────────────────────────────────┘

Available Environments:
    ┌────────────────────────────────────┬──────────────────────────┐
    │ Environment ID                     │ Task                     │
    ├────────────────────────────────────┼──────────────────────────┤
    │ so101_pick_orange                  │ Pick 3 oranges → plate   │
    │ so101_lift_cube                    │ Lift the red cube        │
    │ so101_clean_toytable               │ Clean toy objects → box  │
    │ bi_so101_fold_cloth                │ Fold cloth (bimanual)    │
    │ lekiwi_cleanup_trash               │ Pick tissue → trash bin  │
    └────────────────────────────────────┴──────────────────────────┘

Usage:
    # Load environment from Hub
    env = LeIsaacEnv("so101_pick_orange")

    # Run with any strands-robots policy
    from strands_robots.policies import create_policy
    policy = create_policy("groot", host="localhost", port=5555)
    results = env.rollout(policy, instruction="Pick the orange", n_episodes=5)

    # Record video
    env.record_video(policy, "pick_orange.mp4", instruction="Pick the orange")

    # As a Strands AgentTool
    agent = Agent(tools=[leisaac_env])
    agent("Load the pick orange environment and evaluate with mock policy")

Requirements:
    - NVIDIA GPU with CUDA 12+ and IsaacSim
    - pip install leisaac[isaaclab]
    - pip install lerobot

Reference:
    - https://huggingface.co/docs/lerobot/envhub_leisaac
    - https://github.com/LightwheelAI/leisaac
    - https://huggingface.co/LightwheelAI/leisaac_env
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from strands_robots.policies import Policy

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

ENVHUB_REPO = "LightwheelAI/leisaac_env"

# Task registry — maps short names to EnvHub script paths and metadata
LEISAAC_TASKS: Dict[str, Dict[str, Any]] = {
    "so101_pick_orange": {
        "env_script": f"{ENVHUB_REPO}:envs/so101_pick_orange.py",
        "env_id": "LeIsaac-SO101-PickOrange-v0",
        "description": "Pick three oranges and put them into the plate",
        "robot": "SO101 Follower (single-arm)",
        "category": "manipulation",
        "default_instruction": "Pick up the orange and place it on the plate",
    },
    "so101_lift_cube": {
        "env_script": f"{ENVHUB_REPO}:envs/so101_lift_cube.py",
        "env_id": "LeIsaac-SO101-LiftCube-v0",
        "description": "Lift the red cube up",
        "robot": "SO101 Follower (single-arm)",
        "category": "manipulation",
        "default_instruction": "Lift the red cube",
    },
    "so101_clean_toytable": {
        "env_script": f"{ENVHUB_REPO}:envs/so101_clean_toytable.py",
        "env_id": "LeIsaac-SO101-CleanToyTable-v0",
        "description": "Pick two letter-E objects into the box",
        "robot": "SO101 Follower (single-arm, bi-arm)",
        "category": "manipulation",
        "default_instruction": "Pick up the toys and put them in the box",
    },
    "bi_so101_fold_cloth": {
        "env_script": f"{ENVHUB_REPO}:envs/bi_so101_fold_cloth.py",
        "env_id": "LeIsaac-SO101-FoldCloth-BiArm-v0",
        "description": "Fold the cloth with bimanual arms",
        "robot": "SO101 Follower (bi-arm)",
        "category": "manipulation",
        "default_instruction": "Fold the cloth",
        "needs_initialize": True,
    },
}

# Supported policy types for LeIsaac evaluation
LEISAAC_POLICY_TYPES = {
    "gr00tn1.5": {"flag": "--policy_type=gr00tn1.5", "provider": "groot"},
    "gr00tn1.6": {"flag": "--policy_type=gr00tn1.6", "provider": "groot"},
    "lerobot-smolvla": {
        "flag": "--policy_type=lerobot-smolvla",
        "provider": "lerobot_async",
    },
    "openpi": {"flag": "--policy_type=openpi", "provider": "lerobot_local"},
}


def list_tasks() -> List[Dict[str, str]]:
    """List all available LeIsaac tasks.

    Returns:
        List of task info dicts
    """
    tasks = []
    for name, info in LEISAAC_TASKS.items():
        tasks.append(
            {
                "name": name,
                "env_id": info["env_id"],
                "description": info["description"],
                "robot": info["robot"],
                "category": info["category"],
                "instruction": info["default_instruction"],
            }
        )
    return tasks


def format_task_table() -> str:
    """Format tasks as a human-readable table."""
    lines = [
        "LeIsaac × LeRobot EnvHub Tasks:",
        f"{'Task':<25s} {'Robot':<30s} {'Description'}",
        "─" * 85,
    ]
    for name, info in LEISAAC_TASKS.items():
        lines.append(f"{name:<25s} {info['robot']:<30s} {info['description']}")
    lines.append(f"\nSource: {ENVHUB_REPO}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# LeIsaac Environment Wrapper
# ─────────────────────────────────────────────────────────────────────


@dataclass
class RolloutResult:
    """Result from a policy rollout in a LeIsaac environment."""

    n_episodes: int = 0
    n_successes: int = 0
    success_rate: float = 0.0
    avg_steps: float = 0.0
    avg_reward: float = 0.0
    total_time: float = 0.0
    episodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_episodes": self.n_episodes,
            "n_successes": self.n_successes,
            "success_rate": round(self.success_rate, 4),
            "avg_steps": round(self.avg_steps, 1),
            "avg_reward": round(self.avg_reward, 4),
            "total_time": round(self.total_time, 1),
        }


class LeIsaacEnv:
    """Wrapper for LeIsaac environments loaded from HuggingFace EnvHub.

    Provides a unified interface to:
    - Load IsaacLab environments from the Hub
    - Run strands-robots policies against them
    - Record videos and evaluate metrics

    Usage:
        env = LeIsaacEnv("so101_pick_orange")
        env.reset()
        obs, reward, done, truncated, info = env.step(action)

        # Or with policy rollout:
        from strands_robots.policies import create_policy
        policy = create_policy("mock")
        result = env.rollout(policy, instruction="Pick the orange")
    """

    def __init__(
        self,
        task_name: str = "so101_pick_orange",
        n_envs: int = 1,
        render_mode: str = "rgb_array",
        trust_remote_code: bool = True,
    ):
        """Initialize LeIsaac environment.

        Args:
            task_name: Task name from LEISAAC_TASKS or custom EnvHub reference
            n_envs: Number of parallel environments
            render_mode: Rendering mode
            trust_remote_code: Trust remote code from HuggingFace Hub
        """
        self.task_name = task_name
        self.n_envs = n_envs
        self.render_mode = render_mode

        # Resolve task info
        if task_name in LEISAAC_TASKS:
            self.task_info = LEISAAC_TASKS[task_name]
            self.env_script = self.task_info["env_script"]
        else:
            # Custom EnvHub reference
            self.task_info = {
                "description": f"Custom task: {task_name}",
                "default_instruction": "",
                "robot": "unknown",
            }
            self.env_script = task_name

        self._env = None
        self._raw_env = None
        self._loaded = False

        logger.info(f"🏭 LeIsaacEnv initialized: {task_name} ({self.task_info.get('description', '')})")

    def load(self) -> bool:
        """Load the environment from HuggingFace EnvHub.

        Returns:
            True if loaded successfully
        """
        try:
            from lerobot.envs.factory import make_env

            from strands_robots.policies._utils import check_trust_remote_code

            check_trust_remote_code(self.env_script)
            logger.info("Loading from EnvHub: %s", self.env_script)
            envs_dict = make_env(
                self.env_script,
                n_envs=self.n_envs,
                trust_remote_code=True,
            )

            # Extract the environment
            suite_name = next(iter(envs_dict))
            sync_vector_env = envs_dict[suite_name][0]
            self._env = sync_vector_env
            self._raw_env = sync_vector_env.envs[0].unwrapped

            # Initialize if needed (e.g., fold cloth task)
            if self.task_info.get("needs_initialize") and hasattr(self._raw_env, "initialize"):
                self._raw_env.initialize()

            self._loaded = True
            logger.info("✅ LeIsaac environment loaded: %s", self.task_name)
            return True

        except ImportError as e:
            logger.error(f"❌ Failed to load LeIsaac env: {e}\nInstall: pip install leisaac[isaaclab] lerobot")
            return False
        except Exception as e:
            logger.error("❌ Failed to load LeIsaac env: %s", e)
            return False

    def reset(self) -> Tuple[Any, Dict]:
        """Reset the environment."""
        if not self._loaded:
            if not self.load():
                raise RuntimeError("Failed to load LeIsaac environment")
        return self._raw_env.reset()

    def step(self, action) -> Tuple[Any, float, bool, bool, Dict]:
        """Take a step in the environment."""
        return self._raw_env.step(action)

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if hasattr(self._raw_env, "render"):
            return self._raw_env.render()
        return None

    def close(self):
        """Close the environment."""
        if self._raw_env is not None:
            self._raw_env.close()

    def get_joint_names(self) -> List[str]:
        """Get robot joint names from the environment."""
        if self._raw_env is None:
            return []
        # Try common attribute names
        for attr in ["joint_names", "robot_joint_names", "action_names"]:
            if hasattr(self._raw_env, attr):
                return list(getattr(self._raw_env, attr))
        # Fallback: generate from action space
        if hasattr(self._raw_env, "action_space"):
            n = self._raw_env.action_space.shape[0]
            return [f"joint_{i}" for i in range(n)]
        return []

    def rollout(
        self,
        policy: "Policy",
        instruction: str = "",
        n_episodes: int = 5,
        max_steps: int = 500,
        render: bool = False,
    ) -> RolloutResult:
        """Run a policy for multiple episodes and collect metrics.

        Delegates to :func:`strands_robots.training.evaluate.evaluate` for the
        actual episode loop, then converts the result to a :class:`RolloutResult`.

        Args:
            policy: A strands-robots Policy instance
            instruction: Natural language instruction for the task
            n_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode
            render: Whether to render during rollout

        Returns:
            RolloutResult with success rate, rewards, etc.
        """
        if not instruction:
            instruction = self.task_info.get("default_instruction", "")

        joint_names = self.get_joint_names()
        policy.set_robot_state_keys(joint_names)

        if not self._loaded:
            if not self.load():
                raise RuntimeError("Failed to load LeIsaac environment")

        from strands_robots.training.evaluate import evaluate

        eval_result = evaluate(
            policy=policy,
            task=instruction,
            num_episodes=n_episodes,
            max_steps_per_episode=max_steps,
            render=render,
            env=self._raw_env,
        )

        result = RolloutResult(
            n_episodes=eval_result["num_episodes"],
            n_successes=sum(1 for e in eval_result["episodes"] if e["success"]),
            avg_steps=sum(e["steps"] for e in eval_result["episodes"]) / max(n_episodes, 1),
            avg_reward=eval_result["mean_reward"],
            total_time=0.0,
            episodes=eval_result["episodes"],
        )
        result.success_rate = result.n_successes / max(n_episodes, 1)

        logger.info(
            f"📊 Rollout: {result.n_successes}/{n_episodes} success "
            f"({result.success_rate:.1%}), avg_steps={result.avg_steps:.0f}"
        )
        return result

    def record_video(
        self,
        policy: "Policy",
        output_path: str,
        instruction: str = "",
        max_steps: int = 300,
        fps: int = 30,
    ) -> Dict[str, Any]:
        """Record a video of a policy rollout.

        Args:
            policy: Policy to run
            output_path: Output MP4 file path
            instruction: Task instruction
            max_steps: Maximum steps
            fps: Video frames per second

        Returns:
            Dict with recording info
        """
        try:
            from strands_robots.video import VideoEncoder
        except ImportError:
            # Fallback to imageio
            import imageio

            VideoEncoder = None

        if not instruction:
            instruction = self.task_info.get("default_instruction", "")

        joint_names = self.get_joint_names()
        policy.set_robot_state_keys(joint_names)

        obs, info = self.reset()
        frames = []
        total_reward = 0.0

        for step_idx in range(max_steps):
            # Render frame
            frame = self.render()
            if frame is not None:
                frames.append(frame)

            # Policy inference
            obs_dict = self._obs_to_dict(obs)
            actions = asyncio.run(policy.get_actions(obs_dict, instruction))

            if actions:
                action = self._dict_to_action(actions[0])
            else:
                action = np.zeros(self._raw_env.action_space.shape)

            obs, reward, terminated, truncated, info = self.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        # Encode video
        if frames:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

            if VideoEncoder:
                with VideoEncoder(output_path, fps=fps) as enc:
                    for frame in frames:
                        enc.add_frame(frame)
            else:
                writer = imageio.get_writer(output_path, fps=fps)
                for frame in frames:
                    writer.append_data(frame)
                writer.close()

        return {
            "output_path": output_path,
            "frames": len(frames),
            "steps": step_idx + 1,
            "reward": float(total_reward),
            "success": info.get("is_success", False),
        }

    def _obs_to_dict(self, obs: Any) -> Dict[str, Any]:
        """Convert gym observation to strands-robots policy dict format."""
        if isinstance(obs, dict):
            result = {}
            for key, value in obs.items():
                if hasattr(value, "numpy"):
                    value = value.numpy()
                if isinstance(value, np.ndarray):
                    if value.ndim >= 2:  # Image
                        result[key] = value
                    else:  # State vector
                        joint_names = self.get_joint_names()
                        for i, jn in enumerate(joint_names):
                            if i < len(value):
                                result[jn] = float(value[i])
                else:
                    result[key] = value
            return result
        elif isinstance(obs, np.ndarray):
            result = {}
            joint_names = self.get_joint_names()
            for i, jn in enumerate(joint_names):
                if i < len(obs):
                    result[jn] = float(obs[i])
            return result
        return {"observation": obs}

    def _dict_to_action(self, action_dict: Dict[str, Any]) -> np.ndarray:
        """Convert policy action dict to gym action array."""
        joint_names = self.get_joint_names()
        action = np.zeros(len(joint_names), dtype=np.float32)
        for i, jn in enumerate(joint_names):
            if jn in action_dict:
                action[i] = float(action_dict[jn])
        return action

    def __repr__(self) -> str:
        return f"LeIsaacEnv(task={self.task_name!r}, loaded={self._loaded}, robot={self.task_info.get('robot', '?')})"


# ─────────────────────────────────────────────────────────────────────
# Strands AgentTool Interface
# ─────────────────────────────────────────────────────────────────────

try:
    from strands.tools.decorator import tool as strands_tool

    @strands_tool
    def leisaac_env(
        action: str = "list",
        task: str = "so101_pick_orange",
        policy_provider: str = "mock",
        instruction: str = "",
        n_episodes: int = 5,
        max_steps: int = 300,
        output_path: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        """LeIsaac × LeRobot EnvHub — photorealistic IsaacSim environments.

        Load GPU-accelerated simulation environments from HuggingFace Hub,
        run any VLA policy against them, record videos, and evaluate.

        Powered by Lightwheel AI's LeIsaac + NVIDIA IsaacLab.

        Args:
            action: Action to perform:
                - "list": List available tasks
                - "load": Load an environment
                - "rollout": Run policy rollout with metrics
                - "record": Record video of policy execution
                - "info": Get environment details
            task: Task name (so101_pick_orange, so101_lift_cube, etc.)
            policy_provider: Policy provider (mock, groot, lerobot_local, etc.)
            instruction: Natural language instruction
            n_episodes: Number of evaluation episodes
            max_steps: Max steps per episode
            output_path: Video output path (for record action)

        Returns:
            Dict with status and content

        Examples:
            leisaac_env(action="list")
            leisaac_env(action="rollout", task="so101_pick_orange",
                       policy_provider="groot", instruction="Pick the orange")
            leisaac_env(action="record", task="so101_lift_cube",
                       output_path="lift_cube.mp4")
        """
        try:
            if action == "list":
                table = format_task_table()
                return {
                    "status": "success",
                    "content": [{"text": f"🏭 {table}"}],
                }

            elif action == "info":
                if task in LEISAAC_TASKS:
                    info = LEISAAC_TASKS[task]
                    text = (
                        f"🏭 LeIsaac Task: {task}\n"
                        f"  Environment: {info['env_id']}\n"
                        f"  Description: {info['description']}\n"
                        f"  Robot: {info['robot']}\n"
                        f"  Category: {info['category']}\n"
                        f"  Instruction: {info['default_instruction']}\n"
                        f"  EnvHub: {info['env_script']}\n"
                        f"\n  Supported policies: {', '.join(LEISAAC_POLICY_TYPES.keys())}\n"
                        f"  + any strands-robots policy (mock, lerobot_local, groot, ...)"
                    )
                else:
                    text = f"❌ Unknown task: {task}. Use action='list' to see available tasks."
                return {"status": "success", "content": [{"text": text}]}

            elif action == "load":
                env = LeIsaacEnv(task)
                success = env.load()
                if success:
                    return {
                        "status": "success",
                        "content": [{"text": f"✅ LeIsaac environment loaded: {task}"}],
                    }
                else:
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"❌ Failed to load LeIsaac environment: {task}\n"
                                    f"Requirements:\n"
                                    f"  - NVIDIA GPU with CUDA 12+\n"
                                    f"  - pip install leisaac[isaaclab]\n"
                                    f"  - pip install lerobot\n"
                                    f"See: https://huggingface.co/docs/lerobot/envhub_leisaac"
                                )
                            }
                        ],
                    }

            elif action == "rollout":
                from strands_robots.policies import create_policy

                env = LeIsaacEnv(task)
                if not env.load():
                    return {
                        "status": "error",
                        "content": [{"text": "❌ Failed to load env"}],
                    }

                policy = create_policy(policy_provider, **kwargs)
                result = env.rollout(
                    policy,
                    instruction=instruction or env.task_info.get("default_instruction", ""),
                    n_episodes=n_episodes,
                    max_steps=max_steps,
                )
                env.close()

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"📊 LeIsaac Rollout: {task}\n"
                                f"  Policy: {policy_provider}\n"
                                f"  Episodes: {result.n_episodes}\n"
                                f"  Success: {result.n_successes}/{result.n_episodes} ({result.success_rate:.1%})\n"
                                f"  Avg steps: {result.avg_steps:.0f}\n"
                                f"  Avg reward: {result.avg_reward:.4f}\n"
                                f"  Time: {result.total_time:.1f}s"
                            )
                        },
                        {"json": result.to_dict()},
                    ],
                }

            elif action == "record":
                from strands_robots.policies import create_policy

                env = LeIsaacEnv(task)
                if not env.load():
                    return {
                        "status": "error",
                        "content": [{"text": "❌ Failed to load env"}],
                    }

                if not output_path:
                    output_path = f"leisaac_{task}.mp4"

                policy = create_policy(policy_provider, **kwargs)
                info = env.record_video(
                    policy,
                    output_path=output_path,
                    instruction=instruction or env.task_info.get("default_instruction", ""),
                    max_steps=max_steps,
                )
                env.close()

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"🎬 Video recorded: {output_path}\n"
                                f"  Frames: {info['frames']}, Steps: {info['steps']}\n"
                                f"  Reward: {info['reward']:.4f}, Success: {info['success']}"
                            )
                        }
                    ],
                }

            else:
                return {
                    "status": "error",
                    "content": [{"text": f"Unknown action: {action}. Valid: list, info, load, rollout, record"}],
                }

        except Exception as e:
            logger.error("LeIsaac tool error: %s", e)
            return {"status": "error", "content": [{"text": f"❌ Error: {str(e)}"}]}

except ImportError:
    # Strands not available — LeIsaacEnv still works standalone
    leisaac_env = None


# ─────────────────────────────────────────────────────────────────────
# Convenience functions
# ─────────────────────────────────────────────────────────────────────


def create_leisaac_env(
    task: str = "so101_pick_orange",
    n_envs: int = 1,
    auto_load: bool = True,
) -> LeIsaacEnv:
    """Create a LeIsaac environment (convenience function).

    Args:
        task: Task name or custom EnvHub reference
        n_envs: Number of parallel environments
        auto_load: Automatically load the environment

    Returns:
        LeIsaacEnv instance

    Example:
        env = create_leisaac_env("so101_pick_orange")
        obs, info = env.reset()
    """
    env = LeIsaacEnv(task, n_envs=n_envs)
    if auto_load:
        env.load()
    return env


__all__ = [
    "LeIsaacEnv",
    "create_leisaac_env",
    "leisaac_env",
    "list_tasks",
    "format_task_table",
    "LEISAAC_TASKS",
    "RolloutResult",
]
