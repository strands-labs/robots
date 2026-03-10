#!/usr/bin/env python3
"""
Policy Abstraction for Universal VLA Support

Plugin-based registry — no hardcoded if/elif chains.
New providers are auto-discovered or registered at runtime.

Built-in providers:
- groot: NVIDIA Isaac GR00T (N1.5/N1.6) via ZMQ inference container
- lerobot_async: LeRobot gRPC async inference (ACT, Pi0, SmolVLA, Diffusion, etc.)
- lerobot_local: LeRobot direct HuggingFace inference (no server needed)
- dreamgen: GR00T-Dreams IDM and VLA (DreamGen neural trajectories)
- mock: Sinusoidal actions for testing
"""

import importlib
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# Regex for valid provider names: lowercase letter followed by lowercase
# alphanumeric or underscores. Prevents path traversal and unexpected
# module names from reaching importlib.import_module().
_VALID_PROVIDER_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class Policy(ABC):
    """Abstract base class for VLA policies.

    All policies implement async get_actions(). For convenience, a synchronous
    wrapper ``get_actions_sync()`` is provided so callers don't need asyncio::

        actions = policy.get_actions_sync(obs, "pick up the cube")
    """

    @abstractmethod
    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Get actions from policy given observation and instruction.

        Args:
            observation_dict: Robot observation (cameras + state)
            instruction: Natural language instruction
            **kwargs: Provider-specific parameters

        Returns:
            List of action dictionaries for robot execution
        """
        pass

    def get_actions_sync(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Synchronous convenience wrapper around get_actions().

        Handles the common case where callers don't use asyncio.
        Safe to call from sync code, event loops, or notebooks.

        Example:
            actions = policy.get_actions_sync(obs, "pick up cube")
            for action in actions:
                sim.send_action(action)
                sim.step(5)
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already in an async context — create a new thread to avoid deadlock
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.get_actions(observation_dict, instruction, **kwargs)).result()
        else:
            return asyncio.run(self.get_actions(observation_dict, instruction, **kwargs))

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
        self.robot_state_keys = []
        self._step = 0
        logger.info("🎭 Mock Policy initialized")

    @property
    def provider_name(self) -> str:
        return "mock"

    def set_robot_state_keys(self, robot_state_keys: List[str]) -> None:
        self.robot_state_keys = robot_state_keys
        logger.info(f"🔧 Mock robot state keys: {self.robot_state_keys}")

    async def get_actions(self, observation_dict: Dict[str, Any], instruction: str, **kwargs) -> List[Dict[str, Any]]:
        """Return smooth sinusoidal actions (much better than random noise)."""
        import math

        # Auto-generate keys if none set (infer from observation)
        if not self.robot_state_keys:
            if "observation.state" in observation_dict:
                state = observation_dict["observation.state"]
                dim = len(state) if hasattr(state, "__len__") else 6
            else:
                dim = 6  # Default to 6-DOF
            self.robot_state_keys = [f"joint_{i}" for i in range(dim)]
            logger.info(f"🎭 Mock auto-generated {dim} joint keys")

        mock_actions = []
        for i in range(8):
            action_dict = {}
            t = (self._step + i) * 0.02  # 50Hz
            for j, key in enumerate(self.robot_state_keys):
                freq = 0.3 + j * 0.15
                phase = j * math.pi / 3
                action_dict[key] = 0.5 * math.sin(2 * math.pi * freq * t + phase)
            mock_actions.append(action_dict)

        self._step += len(mock_actions)
        return mock_actions


# =============================================================================
# Policy Registry — the plugin system
# =============================================================================


class PolicyRegistry:
    """Plugin registry for policy providers.

    Supports three ways to add providers:
    1. Built-in: Pre-registered at module load time
    2. register(): Explicit registration at runtime
    3. Auto-discovery: Dynamic import from strands_robots.policies.<name>

    The registry stores lazy loaders (callables that return the class)
    so we never import a provider until it's actually used.
    """

    def __init__(self):
        self._registry: Dict[str, Callable[[], Type[Policy]]] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self,
        name: str,
        loader: Callable[[], Type[Policy]],
        aliases: Optional[List[str]] = None,
    ):
        """Register a policy provider.

        Args:
            name: Canonical provider name (e.g. "groot")
            loader: Callable that returns the Policy subclass (lazy import)
            aliases: Alternative names that map to this provider
        """
        self._registry[name] = loader
        if aliases:
            for alias in aliases:
                self._aliases[alias] = name

    def get(self, name: str) -> Type[Policy]:
        """Get a policy class by name.

        Resolution order:
        1. Direct registry lookup
        2. Alias resolution
        3. Auto-discovery via dynamic import

        Args:
            name: Provider name or alias

        Returns:
            Policy subclass

        Raises:
            ValueError: If provider not found anywhere
        """
        # 1. Direct lookup
        if name in self._registry:
            return self._registry[name]()

        # 2. Alias
        canonical = self._aliases.get(name)
        if canonical and canonical in self._registry:
            return self._registry[canonical]()

        # 3. Auto-discovery: try importing strands_robots.policies.<name>
        policy_cls = self._auto_discover(name)
        if policy_cls:
            return policy_cls

        available = self.list_providers()
        raise ValueError(
            f"Unknown policy provider: '{name}'. "
            f"Available: {available}. "
            f"Or create strands_robots/policies/{name}/__init__.py with a "
            f"{name.capitalize()}Policy(Policy) class."
        )

    def _auto_discover(self, name: str) -> Optional[Type[Policy]]:
        """Try to dynamically import a policy provider by convention.

        Convention: strands_robots.policies.<name>.<Name>Policy
        Example: strands_robots.policies.foo -> FooPolicy

        Security: Provider names are validated against ^[a-z][a-z0-9_]*$
        before being passed to importlib.import_module() to prevent
        path traversal or unexpected module resolution.
        """
        # Validate provider name before passing to importlib.import_module()
        if not _VALID_PROVIDER_NAME_RE.match(name):
            logger.debug(f"Auto-discovery skipped for '{name}': invalid provider name format")
            return None

        try:
            module = importlib.import_module(f"strands_robots.policies.{name}")
            # Try <Name>Policy (e.g. FooPolicy)
            class_name = f"{name.capitalize()}Policy"
            if hasattr(module, class_name):
                cls = getattr(module, class_name)
                logger.info(f"Auto-discovered policy provider: {name} -> {class_name}")
                # Cache it for future lookups
                self._registry[name] = lambda c=cls: c
                return cls

            # Try checking __all__ for any Policy subclass
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, Policy) and attr is not Policy:
                    logger.info(f"Auto-discovered policy provider: {name} -> {attr_name}")
                    self._registry[name] = lambda c=attr: c
                    return attr

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Auto-discovery failed for '{name}': {e}")

        return None

    def list_providers(self) -> List[str]:
        """List all registered provider names (canonical + aliases)."""
        names = list(self._registry.keys())
        names.extend(self._aliases.keys())
        return sorted(set(names))

    def __contains__(self, name: str) -> bool:
        return name in self._registry or name in self._aliases


# Global registry instance
_registry = PolicyRegistry()


# =============================================================================
# Register built-in providers (lazy imports — nothing loaded until used)
# =============================================================================

_registry.register(
    "mock",
    loader=lambda: MockPolicy,
)

_registry.register(
    "groot",
    loader=lambda: importlib.import_module("strands_robots.policies.groot").Gr00tPolicy,
)

_registry.register(
    "lerobot_async",
    loader=lambda: importlib.import_module("strands_robots.policies.lerobot_async").LerobotAsyncPolicy,
)

_registry.register(
    "lerobot_local",
    loader=lambda: importlib.import_module("strands_robots.policies.lerobot_local").LerobotLocalPolicy,
    aliases=["lerobot"],
)

_registry.register(
    "dreamgen",
    loader=lambda: importlib.import_module("strands_robots.policies.dreamgen").DreamgenPolicy,
)

_registry.register(
    "dreamzero",
    loader=lambda: importlib.import_module("strands_robots.policies.dreamzero").DreamzeroPolicy,
    aliases=["dream_zero", "world_action_model"],
)

_registry.register(
    "omnivla",
    loader=lambda: importlib.import_module("strands_robots.policies.omnivla").OmnivlaPolicy,
    aliases=["omni_vla", "omnivla_nav", "navigation_vla"],
)

_registry.register(
    "openvla",
    loader=lambda: importlib.import_module("strands_robots.policies.openvla").OpenvlaPolicy,
    aliases=["open_vla"],
)

_registry.register(
    "internvla",
    loader=lambda: importlib.import_module("strands_robots.policies.internvla").InternvlaPolicy,
    aliases=["intern_vla", "internvla_a1", "internvla_m1"],
)

_registry.register(
    "rdt",
    loader=lambda: importlib.import_module("strands_robots.policies.rdt").RdtPolicy,
    aliases=["rdt_1b", "robotics_diffusion_transformer"],
)

_registry.register(
    "magma",
    loader=lambda: importlib.import_module("strands_robots.policies.magma").MagmaPolicy,
    aliases=["magma_8b", "microsoft_magma"],
)

_registry.register(
    "unifolm",
    loader=lambda: importlib.import_module("strands_robots.policies.unifolm").UnifolmPolicy,
    aliases=["unitree", "unitree_vla", "unifolm_vla"],
)

_registry.register(
    "alpamayo",
    loader=lambda: importlib.import_module("strands_robots.policies.alpamayo").AlpamayoPolicy,
    aliases=["alpamayo_r1", "alpamayo_1", "nvidia_alpamayo"],
)

_registry.register(
    "robobrain",
    loader=lambda: importlib.import_module("strands_robots.policies.robobrain").RobobrainPolicy,
    aliases=["robobrain2", "robobrain_2", "baai_robobrain"],
)

_registry.register(
    "cogact",
    loader=lambda: importlib.import_module("strands_robots.policies.cogact").CogactPolicy,
    aliases=["cog_act", "cogact_base"],
)

_registry.register(
    "cosmos_predict",
    loader=lambda: importlib.import_module("strands_robots.policies.cosmos_predict").CosmosPredictPolicy,
    aliases=[
        "cosmos",
        "cosmos_predict2",
        "cosmos_predict_2b",
        "cosmos_predict_14b",
        "cosmos_policy",
        "cosmos_world_model",
        "nvidia_cosmos",
    ],
)

_registry.register(
    "gear_sonic",
    loader=lambda: importlib.import_module("strands_robots.policies.gear_sonic").GearSonicPolicy,
    aliases=["sonic", "gear", "humanoid_sonic"],
)

_registry.register(
    "go1",
    loader=lambda: importlib.import_module("strands_robots.policies.go1").Go1Policy,
    aliases=["go_1", "agibot_go1", "agibot_world", "go1_air"],
)


# =============================================================================
# Public API
# =============================================================================


def register_policy(
    name: str,
    loader: Callable[[], Type[Policy]],
    aliases: Optional[List[str]] = None,
):
    """Register a custom policy provider.

    Use this to add your own policy providers without modifying strands-robots code.

    Example:
        from strands_robots.policies import register_policy

        def load_my_policy():
            from my_package import MyPolicy
            return MyPolicy

        register_policy("my_provider", load_my_policy, aliases=["my", "custom"])

        # Now works:
        policy = create_policy("my_provider", ...)
    """
    _registry.register(name, loader, aliases)


def list_providers() -> List[str]:
    """List all available policy provider names."""
    return _registry.list_providers()


def create_policy(provider: str, **kwargs) -> Policy:
    """Create a policy instance.

    Accepts either:
    1. Provider name: create_policy("lerobot_local", pretrained_name_or_path="...")
    2. Smart string: create_policy("lerobot/act_aloha_sim") — auto-resolves provider

    Smart resolution (when provider contains "/" or looks like a URL):
        "lerobot/act_aloha_sim"         → lerobot_local
        "openvla/openvla-7b"            → openvla
        "microsoft/Magma-8B"            → magma
        "localhost:8080"                → lerobot_async (gRPC)
        "ws://gpu:9000"                 → dreamzero (WebSocket)
        "zmq://host:5555"              → groot (ZMQ)
        "mock"                          → mock

    .. warning::
        Many VLA providers require ``trust_remote_code=True`` for HuggingFace model loading,
        which allows the model repository to execute arbitrary Python during ``from_pretrained()``.
        When using smart string resolution (e.g. ``create_policy("some-org/some-model")``),
        unknown organizations silently fall back to ``lerobot_local``, which will load and
        potentially execute code from the HuggingFace repo. Only load models from trusted sources.

    Args:
        provider: Provider name, HuggingFace model ID, or server URL
        **kwargs: Provider-specific parameters

    Returns:
        Policy instance

    Examples:
        # Explicit provider (traditional)
        policy = create_policy("lerobot_local",
            pretrained_name_or_path="lerobot/act_aloha_sim_transfer_cube_human")

        # Smart string (new — auto-resolves)
        policy = create_policy("lerobot/act_aloha_sim_transfer_cube_human")
        policy = create_policy("openvla/openvla-7b")
        policy = create_policy("mock")
        policy = create_policy("localhost:8080")
    """
    # Check if this looks like a smart string (HF model ID, URL, shorthand)
    # vs a registered provider name
    _needs_resolution = (
        "/" in provider
        or (":" in provider and (not provider.replace("_", "").isalpha()))
        or provider.startswith("ws://")
        or provider.startswith("grpc://")
        or provider.startswith("zmq://")
    )

    if _needs_resolution:
        try:
            from strands_robots.policy_resolver import resolve_policy

            resolved_provider, resolved_kwargs = resolve_policy(provider, **kwargs)
        except ImportError:
            # Resolver module not available — fall through to direct lookup
            resolved_provider = None
            resolved_kwargs = {}
        except Exception as e:
            # Log unexpected resolver errors instead of silently swallowing them
            logger.debug(f"Policy resolver failed for '{provider}': {e}")
            resolved_provider = None
            resolved_kwargs = {}

        if resolved_provider:
            PolicyClass = _registry.get(resolved_provider)
            return PolicyClass(**resolved_kwargs)

    PolicyClass = _registry.get(provider)
    return PolicyClass(**kwargs)


__all__ = [
    "Policy",
    "MockPolicy",
    "PolicyRegistry",
    "create_policy",
    "register_policy",
    "list_providers",
]
