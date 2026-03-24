"""Policy factory — create_policy() and runtime registration."""

import logging
from collections.abc import Callable

from strands_robots.policies.base import Policy
from strands_robots.registry import import_policy_class, list_policy_providers, resolve_policy

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Runtime registration (for user-defined providers not in JSON)
# ─────────────────────────────────────────────────────────────────────

_runtime_registry: dict[str, Callable[[], type[Policy]]] = {}
_runtime_aliases: dict[str, str] = {}


def register_policy(
    name: str,
    loader: Callable[[], type[Policy]],
    aliases: list[str] | None = None,
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


def list_providers() -> list[str]:
    """List all available policy provider names (JSON + runtime)."""
    names = list_policy_providers()
    names.extend(_runtime_registry.keys())
    names.extend(_runtime_aliases.keys())
    return sorted(set(names))


def create_policy(provider: str, **kwargs) -> Policy:
    """Create a policy instance.

    Accepts either a provider name or a smart string:

    - Provider name: ``create_policy("groot", port=5555)``
    - ZMQ URL: ``create_policy("zmq://localhost:5555")``
    - Shorthand: ``create_policy("mock")``

    All provider definitions live in ``registry/policies.json``.

    Args:
        provider: Provider name, HF model ID, or server URL.
        **kwargs: Provider-specific parameters.

    Returns:
        Policy instance ready for get_actions().

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
            resolved_provider, resolved_kwargs = resolve_policy(provider, **kwargs)
        except ImportError:
            resolved_provider = None
            resolved_kwargs = {}
        except Exception as e:
            logger.warning("Policy resolution failed for '%s': %s", provider, e)
            resolved_provider = None
            resolved_kwargs = {}

        if resolved_provider:
            PolicyClass = import_policy_class(resolved_provider)
            return PolicyClass(**resolved_kwargs)

    # 3. Standard lookup from policies.json
    PolicyClass = import_policy_class(provider)
    return PolicyClass(**kwargs)
