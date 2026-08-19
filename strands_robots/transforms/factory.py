"""Dataset-transform factory - create_transform() and runtime registration.

Mirrors :mod:`strands_robots.training.factory` for the third provider shape:
where ``create_policy`` resolves a provider's inference class and
``create_trainer`` its training class, ``create_transform`` resolves its
dataset-transform class. A provider that ships one may declare it in
``registry/policies.json`` under a ``"transform"`` block alongside its policy,
exactly as trainers use a ``"trainer"`` block; the built-in transforms
(``mock``, ``cosmos_transfer``) are registered at package import through the
runtime registry instead, because ``cosmos_transfer`` is a generation model
with no policy identity to hang a JSON block on.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

from strands_robots.registry.policies import get_policy_provider, list_policy_providers
from strands_robots.transforms.base import DatasetTransform

logger = logging.getLogger(__name__)

# Runtime registration (built-ins and user-defined transforms not in JSON).
_runtime_registry: dict[str, Callable[[], type[DatasetTransform]]] = {}
_runtime_aliases: dict[str, str] = {}


def register_transform(
    name: str,
    loader: Callable[[], type[DatasetTransform]],
    aliases: list[str] | None = None,
) -> None:
    """Register a custom dataset transform at runtime.

    Use this to add augmentation backends without editing policies.json.

    Example::

        from strands_robots.transforms import register_transform

        register_transform("my_v2v", lambda: MyTransform, aliases=["mv"])
        transform = create_transform("my_v2v")

    Args:
        name: Provider name.
        loader: Zero-arg callable returning the :class:`DatasetTransform`
            subclass (deferred import so heavy deps load only on use).
        aliases: Optional alternate names.
    """
    _runtime_registry[name] = loader
    if aliases:
        for alias in aliases:
            _runtime_aliases[alias] = name


def list_transforms() -> list[str]:
    """List provider names that have a dataset transform (runtime + JSON ``transform`` blocks)."""
    names: list[str] = list(_runtime_registry.keys())
    names.extend(_runtime_aliases.keys())
    for provider in list_policy_providers():
        cfg = get_policy_provider(provider)
        if cfg and "transform" in cfg:
            names.append(provider)
    return sorted(set(names))


def import_transform_class(provider: str) -> type[DatasetTransform]:
    """Import and return the :class:`DatasetTransform` subclass for a provider.

    Resolution order:
      1. The provider's ``"transform"`` block in policies.json.
      2. Auto-discovery fallback: ``strands_robots.transforms.<provider>`` with
         a class named ``<Provider>Transform`` or the first
         :class:`DatasetTransform` subclass.

    Raises:
        ValueError: If no transform can be resolved for the provider.
        ImportError: If the declared module can't be imported.
    """
    cfg = get_policy_provider(provider)
    if cfg and "transform" in cfg:
        tcfg = cfg["transform"]
        mod = importlib.import_module(tcfg["module"])
        cls: type[DatasetTransform] = getattr(mod, tcfg["class"])
        return cls

    # Auto-discovery fallback: strands_robots.transforms.<provider>
    try:
        mod = importlib.import_module(f"strands_robots.transforms.{provider}")
        class_name = f"{provider.capitalize()}Transform"
        if hasattr(mod, class_name):
            cls = getattr(mod, class_name)
            return cls
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, type) and issubclass(attr, DatasetTransform) and attr is not DatasetTransform:
                return attr
    except ImportError:
        # No strands_robots.transforms.<provider> module; fall through to the
        # ValueError below so the caller gets the full "available" list.
        pass

    raise ValueError(f"No transform registered for provider '{provider}'. Available transforms: {list_transforms()}")


def create_transform(provider: str, **kwargs: Any) -> DatasetTransform:
    """Create a :class:`DatasetTransform` for a provider.

    The data-side peer of ``create_policy`` / ``create_trainer``.

    Args:
        provider: Provider name or alias (``"mock"``, ``"cosmos_transfer"``,
            or a runtime-registered name).
        **kwargs: Forwarded to the transform constructor.

    Returns:
        A ready :class:`DatasetTransform` instance.

    Raises:
        ValueError: If no transform is registered for the provider.
    """
    # 1. Runtime registry first (built-ins and user-registered transforms).
    resolved = _runtime_aliases.get(provider, provider)
    if resolved in _runtime_registry:
        return _runtime_registry[resolved]()(**kwargs)

    # 2. Registry / auto-discovery.
    TransformClass = import_transform_class(provider)
    return TransformClass(**kwargs)
