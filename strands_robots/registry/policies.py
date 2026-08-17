"""Policy registry - resolve, import, and configure policy providers.

All provider definitions live in policies.json.  This module provides
the public read API for resolving smart policy strings, importing provider
classes, and building provider-specific kwargs.
"""

import importlib
import logging
import re
from typing import Any

from .loader import _load

logger = logging.getLogger(__name__)


def _build_alias_map() -> dict[str, str]:
    """Build alias/shorthand → canonical provider mapping from provider entries."""
    reg = _load("policies")
    alias_map: dict[str, str] = {}
    for name, info in reg.get("providers", {}).items():
        for alias in info.get("aliases", []):
            alias_map[alias] = name
        for shorthand in info.get("shorthands", []):
            alias_map[shorthand] = name
    return alias_map


def get_policy_provider(name: str) -> dict[str, Any] | None:
    """Get policy provider config by name or alias.

    Args:
        name: Provider name or alias (e.g. "groot", "lerobot", "cosmos").

    Returns:
        Provider dict with module, class, config_keys, defaults, etc.
        None if not found.
    """
    reg = _load("policies")
    alias_map = _build_alias_map()
    canonical = alias_map.get(name, name)
    return reg.get("providers", {}).get(canonical)


def list_policy_providers() -> list[str]:
    """List all registered policy provider names (canonical only)."""
    reg = _load("policies")
    return sorted(reg.get("providers", {}).keys())


def resolve_policy(policy: str, **extra_kwargs) -> tuple[str, dict[str, Any]]:
    """Resolve a smart policy string to (provider_name, kwargs).

    Accepts HuggingFace model IDs, server URLs, or shorthand names
    and returns the canonical provider + ready-to-use kwargs.

    Resolution order:
        1. URL patterns (ws://, zmq://, grpc://, host:port)
        2. Shorthand names (mock, groot, lerobot_local, ...)
        3. HuggingFace model IDs (org/model)
        4. Registered provider name
        5. Fallback to lerobot_local

    Args:
        policy: Smart string - HF model ID, URL, or provider name.
        **extra_kwargs: Additional kwargs merged into result.

    Returns:
        (provider_name, kwargs_dict) tuple.

    Examples::
        resolve_policy("lerobot/act_aloha_sim")
        # → ("lerobot_local", {"pretrained_name_or_path": "lerobot/act_aloha_sim"})

        resolve_policy("zmq://localhost:5555")
        # → ("groot", {"host": "localhost", "port": 5555})

        resolve_policy("grpc://gpu-box:8080")
        # → ("lerobot_async", {"server_address": "gpu-box:8080"})

        resolve_policy("mock")
        # → ("mock", {})
    """
    reg = _load("policies")
    providers = reg.get("providers", {})
    policy = policy.strip()
    kwargs: dict[str, Any] = {}

    # 1. URL pattern matching - check each provider's url_patterns
    for prov_name, prov_info in providers.items():
        for pattern in prov_info.get("url_patterns", []):
            if re.match(pattern, policy):
                if pattern.startswith("^wss?://"):
                    # Pass the full URL through as ``endpoint`` so the scheme
                    # (ws:// vs wss://) and any path survive; also split out
                    # host/port for providers that consume them directly.
                    kwargs["endpoint"] = policy
                    match = re.match(r"wss?://([^:/]+):?(\d+)?", policy)
                    if match:
                        kwargs["host"] = match.group(1)
                        kwargs["port"] = int(match.group(2) or 8000)
                elif pattern.startswith("^cosmos3://"):
                    # Cosmos 3 service-mode URL: cosmos3://[host[:port]] -> kwargs.
                    # Without this branch the pattern matches but no parser
                    # populates host/port, so create_policy("cosmos3://prod:9000")
                    # silently falls back to the default localhost:8000 (#317).
                    match = re.match(r"cosmos3://([^:/]+):?(\d+)?", policy)
                    if match:
                        kwargs["host"] = match.group(1)
                        kwargs["port"] = int(match.group(2) or 8000)
                elif pattern.startswith("^vera://"):
                    # VERA service-mode URL: vera://host[:port] -> kwargs.
                    # Mirrors the cosmos3 branch (#317): the pattern matches but
                    # without this parser nothing populates host/server_port, so
                    # create_policy("vera://gpu-box:9000") silently connects to
                    # the default 127.0.0.1. VERA's port kwarg is server_port;
                    # leave it unset when the URL omits a port so the
                    # per-embodiment default still applies.
                    match = re.match(r"vera://([^:/]+):?(\d+)?", policy)
                    if match:
                        kwargs["host"] = match.group(1)
                        if match.group(2):
                            kwargs["server_port"] = int(match.group(2))
                elif pattern.startswith("^zmq://"):
                    match = re.match(r"zmq://([^:]+):(\d+)", policy)
                    if match:
                        kwargs["host"] = match.group(1)
                        kwargs["port"] = int(match.group(2))
                elif pattern.startswith("^grpc://"):
                    kwargs["server_address"] = policy.replace("grpc://", "")
                elif ":" in policy and "/" not in policy:
                    kwargs["server_address"] = policy
                kwargs.update(extra_kwargs)
                return prov_name, kwargs

    # 2. Shorthand names - built from each provider's shorthands list
    alias_map = _build_alias_map()
    if policy.lower() in alias_map:
        kwargs.update(extra_kwargs)
        return alias_map[policy.lower()], kwargs

    # 3. HuggingFace model IDs (org/model)
    if "/" in policy:
        # Check model_id_overrides across all providers
        for prov_name, prov_info in providers.items():
            for prefix in prov_info.get("model_id_overrides", []):
                if policy.lower().startswith(prefix):
                    kwargs["pretrained_name_or_path"] = policy
                    kwargs.update(extra_kwargs)
                    return prov_name, kwargs

        # Check hf_orgs
        org = policy.split("/")[0].lower()
        for prov_name, prov_info in providers.items():
            if org in prov_info.get("hf_orgs", []):
                kwargs["pretrained_name_or_path"] = policy
                kwargs.update(extra_kwargs)
                return prov_name, kwargs

        # Unknown org → find default HF provider
        for prov_name, prov_info in providers.items():
            if prov_info.get("is_hf_default"):
                kwargs["pretrained_name_or_path"] = policy
                kwargs.update(extra_kwargs)
                return prov_name, kwargs

        # Absolute fallback
        kwargs["pretrained_name_or_path"] = policy
        kwargs.update(extra_kwargs)
        return "lerobot_local", kwargs

    # 4. Check if it's a registered provider name
    if get_policy_provider(policy.lower()):
        kwargs.update(extra_kwargs)
        return policy.lower(), kwargs

    # 5. Fallback
    logger.warning("Unrecognised policy '%s', falling back to lerobot_local", policy)
    kwargs["pretrained_name_or_path"] = policy
    kwargs.update(extra_kwargs)
    return "lerobot_local", kwargs


def _provider_import_error(provider: str, exc: ImportError, extra: str | None) -> ImportError:
    """Translate a failed provider-module import into an actionable error.

    A policy provider's module may import an optional dependency at import time
    (e.g. ``lerobot_local`` imports ``torch``). When that dependency is absent
    the import machinery raises a bare ``ModuleNotFoundError: No module named
    'torch'`` which names neither the provider the caller asked for nor the way
    to fix it -- so a caller who asked for one provider is left holding an error
    about a package they never mentioned.

    Every other provider defers its heavy import and reports the remedy through
    :func:`~strands_robots.utils.require_optional` /
    :func:`~strands_robots.utils.require_optionals`, which name the extra that
    ships the dependency. This is the same report for the providers whose
    dependency is needed to import the module at all, so the remedy does not
    depend on WHERE a provider happens to import its dependency.

    Args:
        provider: Canonical provider name the caller asked for.
        exc: The ``ImportError`` raised while importing the provider's module.
        extra: ``pyproject.toml`` extras group that ships the dependency, as
            declared by the provider's ``extra`` field in ``policies.json``.
            ``None`` when the provider declares none, in which case the missing
            module is named without an install command for a specific extra.

    Returns:
        An ``ImportError`` naming the provider, the missing module and the
        remedy. The caller should ``raise ... from exc`` to keep the original
        traceback.
    """
    missing = getattr(exc, "name", None) or "an optional dependency"
    if extra:
        remedy = f"Install the extra that ships it:\n  uv pip install 'strands-robots[{extra}]'"
    else:
        remedy = f"Install {missing!r} (or the strands-robots extra that ships it) and retry."
    return ImportError(
        f"Policy provider {provider!r} needs an optional dependency that is not installed:\n  {exc}\n\n{remedy}"
    )


def import_policy_class(provider: str) -> type:
    """Dynamically import and return the Policy class for a provider.

    Uses the module + class paths from policies.json.  Falls back to
    auto-discovery (strands_robots.policies.<name>) if not in JSON.

    Args:
        provider: Canonical provider name.

    Returns:
        The Policy subclass.

    Raises:
        ValueError: If the provider does not exist.
        ImportError: If the provider exists but its module cannot be imported,
            naming the provider, the missing module and the remedy (see
            :func:`_provider_import_error`). A provider whose module is present
            but whose optional dependency is missing reports that rather than
            being misreported as an unknown provider.
    """
    config = get_policy_provider(provider)
    if config:
        # Resolve alias to canonical for module lookup
        reg = _load("policies")
        alias_map = _build_alias_map()
        canonical = alias_map.get(provider, provider)
        config = reg.get("providers", {}).get(canonical, config)

        try:
            mod = importlib.import_module(config["module"])
        except ImportError as exc:
            # A provider whose module needs an optional dependency at import
            # time (lerobot_local imports torch) otherwise raises a bare
            # "No module named 'torch'" naming neither this provider nor the
            # remedy - the dead end _provider_import_error exists to close.
            raise _provider_import_error(canonical, exc, config.get("extra")) from exc
        return getattr(mod, config["class"])

    # Auto-discovery fallback
    try:
        mod = importlib.import_module(f"strands_robots.policies.{provider}")
        class_name = f"{provider.capitalize()}Policy"
        if hasattr(mod, class_name):
            return getattr(mod, class_name)
        from strands_robots.policies import Policy

        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and issubclass(attr, Policy) and attr is not Policy:
                return attr
    except ImportError as exc:
        # Distinguish "this provider does not exist" from "it exists but its
        # optional dependency is missing". Only the former is an unknown
        # provider; reporting the latter that way sends the caller to check a
        # name that was correct.
        if getattr(exc, "name", None) != f"strands_robots.policies.{provider}":
            raise _provider_import_error(provider, exc, None) from exc

    raise ValueError(f"Unknown policy provider: '{provider}'. Available: {list_policy_providers()}")


def build_policy_kwargs(
    provider: str,
    policy_port: int | None = None,
    policy_host: str | None = None,
    model_path: str | None = None,
    server_address: str | None = None,
    policy_type: str | None = None,
    data_config: Any = None,
    **extra,
) -> dict[str, Any]:
    """Build provider-specific kwargs from generic parameters.

    Maps generic parameter names (policy_port, model_path, ...) to
    the provider-specific keys declared in policies.json.

    Args:
        provider: Policy provider name.
        policy_port: Port number (groot, cosmos3, moveit2, remote).
        policy_host: Hostname.  ``None`` leaves the key unset so the
            provider's registry default -- or, failing that, its own
            constructor default -- applies.
        model_path: Local model path or HF ID.
        server_address: Full server address host:port (grpc:// URLs, remote providers).
        policy_type: Sub-type (pi0, act, smolvla, ...).
        data_config: Data configuration for groot.
        **extra: Any additional provider-specific kwargs.  A key declared in
            the provider's ``config_keys`` is forwarded; any other key is
            dropped, which is what ``config_keys`` exists to decide.

    Returns:
        Dict of kwargs ready for create_policy(provider, **kwargs).

        Where two sources name the same key, the more explicit one wins:
        a provider-specific value in ``extra`` beats the generic parameter
        that maps onto it (``host`` beats ``policy_host``), and both beat the
        provider's registry default.  A default only ever fills a key the
        caller left unset.
    """
    config = get_policy_provider(provider) or {}
    allowed_keys = set(config.get("config_keys", []))
    defaults = dict(config.get("defaults", {}))
    kwargs: dict[str, Any] = {}

    param_map = {
        "port": policy_port,
        "host": policy_host,
        "data_config": data_config,
        "server_address": server_address
        or (
            f"{policy_host}:{policy_port}" if policy_host and policy_port and "server_address" in allowed_keys else None
        ),
        "model_path": model_path,
        "pretrained_name_or_path": (
            model_path
            if model_path and "pretrained_name_or_path" in allowed_keys
            else extra.get("pretrained_name_or_path")
        ),
        "policy_type": policy_type,
    }

    # Precedence: a value the caller supplied under the provider's own key in
    # ``extra`` wins over the generic parameter that maps onto the same key,
    # which in turn wins over the registry default.  ``resolve_policy`` applies
    # the same rule with its trailing ``kwargs.update(extra_kwargs)``.
    #
    # The order of these three loops is the contract.  Applying ``extra`` last
    # while skipping keys already in ``kwargs`` inverts it: the defaults loop
    # inserts the default first, so the ``key not in kwargs`` guard then sees
    # that default and discards a value the caller did supply -- the same
    # silent fallback to a default that #317 fixed for URL-parsed host/port.
    for key, value in extra.items():
        if key in allowed_keys:
            kwargs[key] = value

    for key, value in param_map.items():
        if value is not None and key in allowed_keys and key not in kwargs:
            kwargs[key] = value

    for key, default_val in defaults.items():
        if key not in kwargs:
            kwargs[key] = default_val

    return kwargs
