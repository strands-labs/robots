"""Shared utilities for strands-robots."""

import importlib
import logging

logger = logging.getLogger(__name__)

# Cache of lazy-loaded modules
_lazy_modules: dict[str, object] = {}


def require_optional(
    module_name: str,
    *,
    pip_install: str | None = None,
    extra: str | None = None,
    purpose: str = "",
) -> object:
    """Import an optional dependency, raising a clear error if missing.

    Once imported, the module is cached so subsequent calls are free.

    Args:
        module_name: Dotted module name to import (e.g. ``"zmq"``).
        pip_install: Explicit pip package name if it differs from *module_name*.
        extra: ``pyproject.toml`` extras group (e.g. ``"groot-service"``).
        purpose: Human-readable description shown in the error message.

    Returns:
        The imported module object.

    Raises:
        ImportError: With a helpful install instruction.
    """
    if module_name in _lazy_modules:
        return _lazy_modules[module_name]

    try:
        module = importlib.import_module(module_name)
        _lazy_modules[module_name] = module
        return module
    except ImportError:
        install_hint = pip_install or module_name
        parts = [f"'{module_name}' is required"]
        if purpose:
            parts[0] += f" for {purpose}"
        parts.append("Install with:")
        if extra:
            parts.append(f"  pip install 'strands-robots[{extra}]'")
        parts.append(f"  pip install {install_hint}")
        raise ImportError("\n".join(parts)) from None
