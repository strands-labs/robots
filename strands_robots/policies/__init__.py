"""Policy Abstraction for Universal VLA Support.

Plugin-based registry — all provider definitions live in ``registry/policies.json``.
No hardcoded if/elif chains. New providers are auto-discovered or registered at runtime.

Built-in providers (see policies.json for full list):
    - mock: Sinusoidal test actions
    - groot: NVIDIA GR00T via ZMQ
    - lerobot_async: LeRobot gRPC (ACT, Pi0, SmolVLA, Diffusion, ...)
    - lerobot_local: Direct HuggingFace inference (no server)
    - dreamgen: GR00T-Dreams IDM + VLA
    - dreamzero: DreamZero world action model (WebSocket)
    - cosmos_predict: NVIDIA Cosmos world model
    - gear_sonic: GEAR-SONIC humanoid controller

Usage::

    from strands_robots.policies import create_policy, Policy

    # By provider name
    policy = create_policy("lerobot_local",
        pretrained_name_or_path="lerobot/act_aloha_sim")

    # By smart string (auto-resolves provider)
    policy = create_policy("lerobot/act_aloha_sim")
    policy = create_policy("mock")
    policy = create_policy("localhost:8080")

    # Custom provider
    register_policy("my_provider", lambda: MyPolicy, aliases=["my"])
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

from strands_robots.registry import (
    import_policy_class,
    resolve_policy_string,
)
from strands_robots.registry import (
    list_policy_providers as _registry_list_providers,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Policy ABC
# ─────────────────────────────────────────────────────────────────────


class Policy(ABC):
    """Abstract base class for VLA policies.

    All policies implement ``async get_actions()``.  For convenience, a
    synchronous wrapper ``get_actions_sync()`` is provided.
    """

    @abstractmethod
    async def get_actions(
        self, observation_dict: Dict[str, Any], instruction: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Get actions from policy given observation and instruction.

        Args:
            observation_dict: Robot observation (cameras + state).
            instruction: Natural language instruction.

        Returns:
            List of action dicts for robot execution.
        """
        pass

    def get_actions_sync(
        self, observation_dict: Dict[str, Any], instruction: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Synchronous convenience wrapper around ``get_actions()``.

        Safe to call from sync code, event loops, or notebooks.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.get_actions(observation_dict, instruction, **kwargs),
                ).result()
        else:
            return asyncio.run(
                self.get_actions(observation_dict, instruction, **kwargs)
            )

    @abstractmethod
    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        """Configure the policy with robot state keys."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name for identification."""
        pass


class MockPolicy(Policy):
    """Mock policy for testing — generates smooth sinusoidal trajectories."""

    def __init__(self, **kwargs):
        self.robot_state_keys: List[str] = []
        self._step = 0
        logger.info("🎭 Mock Policy initialized")

    @property
    def provider_name(self) -> str:
        return "mock"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self.robot_state_keys = robot_state_keys

    async def get_actions(
        self, observation_dict: Dict[str, Any], instruction: str, **kwargs
    ) -> List[Dict[str, Any]]:
        """Return smooth sinusoidal actions."""
        if not self.robot_state_keys:
            if "observation.state" in observation_dict:
                state = observation_dict["observation.state"]
                dim = len(state) if hasattr(state, "__len__") else 6
            else:
                dim = 6
            self.robot_state_keys = [f"joint_{i}" for i in range(dim)]

        mock_actions = []
        for i in range(8):
            action_dict = {}
            t = (self._step + i) * 0.02
            for j, key in enumerate(self.robot_state_keys):
                freq = 0.3 + j * 0.15
                phase = j * math.pi / 3
                action_dict[key] = 0.5 * math.sin(2 * math.pi * freq * t + phase)
            mock_actions.append(action_dict)

        self._step += len(mock_actions)
        return mock_actions


# ─────────────────────────────────────────────────────────────────────
# Runtime registration (for user-defined providers not in JSON)
# ─────────────────────────────────────────────────────────────────────

_runtime_registry: Dict[str, Callable[[], Type[Policy]]] = {}
_runtime_aliases: Dict[str, str] = {}


def register_policy(
    name: str,
    loader: Callable[[], Type[Policy]],
    aliases: Optional[List[str]] = None,
):
    """Register a custom policy provider at runtime.

    Use this to add providers without editing policies.json.

    Example::

        from strands_robots.policies import register_policy

        register_policy("my_provider", lambda: MyPolicy, aliases=["my"])
        policy = create_policy("my_provider", ...)
    """
    _runtime_registry[name] = loader
    if aliases:
        for alias in aliases:
            _runtime_aliases[alias] = name


def list_providers() -> List[str]:
    """List all available policy provider names (JSON + runtime)."""
    names = _registry_list_providers()
    names.extend(_runtime_registry.keys())
    names.extend(_runtime_aliases.keys())
    return sorted(set(names))


# ─────────────────────────────────────────────────────────────────────
# create_policy — the main entry point
# ─────────────────────────────────────────────────────────────────────


def create_policy(provider: str, **kwargs) -> Policy:
    """Create a policy instance.

    Accepts either a provider name or a smart string:

    - Provider name: ``create_policy("lerobot_local", pretrained_name_or_path="...")``
    - HuggingFace ID: ``create_policy("lerobot/act_aloha_sim")``
    - Server URL: ``create_policy("localhost:8080")``
    - Shorthand: ``create_policy("mock")``

    All provider definitions live in ``registry/policies.json``.

    Args:
        provider: Provider name, HF model ID, or server URL.
        **kwargs: Provider-specific parameters.

    Returns:
        Policy instance ready for ``get_actions()``.

    .. warning:: Security — trust_remote_code

        Most HF VLA providers load models with ``trust_remote_code=True``.
        Only load models from organisations you trust. Set
        ``STRANDS_TRUST_REMOTE_CODE=1`` to acknowledge and silence the
        runtime warning.
    """
    # 1. Check runtime registry first (user-registered providers)
    resolved_name = _runtime_aliases.get(provider, provider)
    if resolved_name in _runtime_registry:
        PolicyClass = _runtime_registry[resolved_name]()
        return PolicyClass(**kwargs)

    # 2. Check if this looks like a smart string (HF ID, URL, etc.)
    _needs_resolution = (
        "/" in provider
        or (":" in provider and not provider.replace("_", "").isalpha())
        or provider.startswith("ws://")
        or provider.startswith("grpc://")
        or provider.startswith("zmq://")
    )

    if _needs_resolution:
        try:
            resolved_provider, resolved_kwargs = resolve_policy_string(provider, **kwargs)
        except Exception:
            resolved_provider = None
            resolved_kwargs = {}

        if resolved_provider:
            PolicyClass = import_policy_class(resolved_provider)
            return PolicyClass(**resolved_kwargs)

    # 3. Standard lookup from policies.json
    PolicyClass = import_policy_class(provider)
    return PolicyClass(**kwargs)


__all__ = [
    "Policy",
    "MockPolicy",
    "create_policy",
    "register_policy",
    "list_providers",
]
