"""LeRobot policy class resolution.

Resolves the correct LeRobot policy class from:
- HuggingFace Hub config.json (auto-detect)
- Explicit type string (user-specified)

Resolution strategies (in order):
1. PreTrainedConfig draccus resolution (LeRobot 0.5+)
2. Manual config.json reading (fallback for custom/third-party)
3. Direct submodule import: lerobot.policies.{type}.modeling_{type}
4. Package-level import: lerobot.policies.{type}
5. Legacy factory: lerobot.policies.factory.get_policy_class
6. PreTrainedPolicy fallback (only if concrete, not abstract)
"""

import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module-level flag: ensures we only attempt draccus registry bootstrap once.
_CONFIGS_REGISTERED = False


def _ensure_policy_configs_registered() -> None:
    """Ensure LeRobot policy config classes are registered in the draccus choice registry.

    LeRobot 0.5+ uses lazy registration: config classes like ACTConfig are only
    added to PreTrainedConfig's choice registry when their module is first imported.
    Importing ANY one of them triggers registration of ALL policies because
    each config module has module-level side effects that populate the registry.

    This function imports a single known config to bootstrap the entire registry.
    It's safe to call multiple times — the import is idempotent.
    """
    global _CONFIGS_REGISTERED
    if _CONFIGS_REGISTERED:
        return

    try:
        # Importing any policy config triggers registration of ALL policies.
        # ACTConfig is the most common; if it doesn't exist, lerobot is too old
        # for draccus-based config and the caller should fall through to manual resolution.
        importlib.import_module("lerobot.policies.act.configuration_act")
        _CONFIGS_REGISTERED = True
        logger.debug("LeRobot policy configs registered in draccus choice registry")
    except (ImportError, ModuleNotFoundError):
        # Pre-0.5 lerobot or missing policy subpackage — that's OK,
        # the caller will fall through to manual resolution.
        logger.debug("Could not import lerobot policy configs for draccus registration")
    except Exception as exc:
        logger.debug("Unexpected error during policy config registration: %s", exc)


def resolve_policy_class_from_hub(pretrained_name_or_path: str) -> tuple[type[Any], str]:
    """Resolve the LeRobot policy class from a pretrained path or HF repo.

    Uses PreTrainedConfig.from_pretrained() which handles config resolution,
    class lookup, and weight loading via the draccus config registry.

    Falls back to reading config.json manually + class name matching if
    the draccus path fails (e.g. third-party policies not in registry).

    Args:
        pretrained_name_or_path: HF model ID or local directory path.

    Returns:
        Tuple of (PolicyClass, policy_type_string).

    Raises:
        ValueError: If policy type cannot be determined from config.
        ImportError: If the resolved policy class cannot be imported.
    """
    # Strategy 1: PreTrainedConfig draccus resolution → concrete class.
    try:
        from lerobot.configs.policies import PreTrainedConfig

        # LeRobot 0.5+ uses a lazy draccus choice registry.  Policy config
        # classes are only registered when their module is first imported.
        # Importing any one config (e.g. ACTConfig) triggers registration of
        # ALL policies via their module-level @ChoiceRegistry decorators.
        _ensure_policy_configs_registered()

        config = PreTrainedConfig.from_pretrained(pretrained_name_or_path)
        policy_type = getattr(config, "type", type(config).__name__.replace("Config", "").lower())
        logger.info("Auto-resolved via PreTrainedConfig: '%s' -> type='%s'", pretrained_name_or_path, policy_type)

        PolicyClass = resolve_policy_class_by_name(policy_type)
        return PolicyClass, policy_type
    except ImportError:
        raise  # Missing lerobot is a real error, don't swallow
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("PreTrainedConfig resolution failed, trying manual: %s", exc)
    except Exception as exc:
        # draccus raises DecodingError/ParsingError which are NOT subclasses
        # of RuntimeError/ValueError — they inherit from DraccusException → Exception.
        # Catch broadly here but only for draccus-related errors.
        if "draccus" in type(exc).__module__ or "DecodingError" in type(exc).__name__:
            logger.debug("PreTrainedConfig draccus error, trying manual: %s", exc)
        else:
            raise

    # Strategy 2: Manual config.json reading (fallback for custom/third-party)
    policy_type = _read_policy_type_from_config(pretrained_name_or_path)

    if not policy_type:
        raise ValueError(
            f"Could not determine policy type from '{pretrained_name_or_path}'. "
            f"No 'type' field found in config.json. "
            f"Pass policy_type= explicitly."
        )

    PolicyClass = resolve_policy_class_by_name(policy_type)
    logger.info("Auto-resolved: '%s' -> type='%s' -> %s", pretrained_name_or_path, policy_type, PolicyClass.__name__)
    return PolicyClass, policy_type


def _ensure_lerobot_policies_importable() -> None:
    """Ensure ``lerobot.policies`` is registered in ``sys.modules`` without executing
    its ``__init__.py``.

    LeRobot 0.5+ has a ``lerobot/policies/__init__.py`` that eagerly imports
    **all** policy packages (groot, act, diffusion, …).  The groot import chain
    pulls in ``transformers`` → ``flash_attn`` which can crash at module load
    time on environments with ABI mismatches (e.g. wrong torch / flash-attn
    version combo).

    By inserting a lightweight stub package for ``lerobot.policies`` we allow
    ``importlib.import_module("lerobot.policies.<type>.modeling_<type>")`` to
    resolve the parent without triggering the heavy ``__init__``.

    This is safe because:
    - The stub only provides ``__path__`` (required by the import machinery).
    - Individual policy subpackages (``act/``, ``diffusion/``) have their own
      ``__init__.py`` and ``modeling_*`` modules that are self-contained.
    - If ``lerobot.policies`` was already imported successfully (e.g. on a
      well-configured machine), this function is a no-op.
    """
    import sys
    import types

    key = "lerobot.policies"
    if key in sys.modules:
        # Already imported (successfully or via a previous stub) — nothing to do.
        return

    try:
        import lerobot

        policies_dir = Path(lerobot.__path__[0]) / "policies"
        if not policies_dir.is_dir():
            return  # no policies directory → nothing we can stub

        stub = types.ModuleType(key)
        stub.__path__ = [str(policies_dir)]
        stub.__package__ = key
        stub.__file__ = str(policies_dir / "__init__.py")
        sys.modules[key] = stub
        logger.debug("Installed lightweight stub for lerobot.policies (%s)", policies_dir)
    except Exception as exc:
        logger.debug("Could not install lerobot.policies stub: %s", exc)


def resolve_policy_class_by_name(policy_type: str) -> type[Any]:
    """Resolve policy class from an explicit type string.

    Resolution strategies (in order):
        1. Direct submodule import: lerobot.policies.{type}.modeling_{type}
        2. Package-level import: lerobot.policies.{type}
        3. Legacy factory: lerobot.policies.factory.get_policy_class
        4. PreTrainedPolicy fallback (only if concrete, not abstract)

    LeRobot 0.5+ puts concrete classes in ``modeling_*`` submodules
    (e.g. ``lerobot.policies.act.modeling_act.ACTPolicy``) while the
    package ``__init__`` may re-export only the config.

    Args:
        policy_type: LeRobot policy type string (e.g. "act", "diffusion", "smolvla").

    Returns:
        The resolved policy class.

    Raises:
        ImportError: If no matching class can be found.
    """
    # Ensure lerobot.policies parent is importable without triggering its
    # __init__.py, which in LeRobot 0.5+ eagerly imports groot → transformers
    # → flash-attention and can crash if the env has ABI mismatches or missing
    # optional deps.  We inject a lightweight stub module so that
    # ``importlib.import_module("lerobot.policies.act.modeling_act")``
    # can resolve the parent package without executing the real __init__.
    _ensure_lerobot_policies_importable()

    # Strategy 1: modeling_* submodule (LeRobot 0.5+ convention)
    for submodule_name in [f"modeling_{policy_type}", "modeling"]:
        try:
            module = importlib.import_module(f"lerobot.policies.{policy_type}.{submodule_name}")
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and attr_name.endswith("Policy")
                    and attr_name != "PreTrainedPolicy"
                    and hasattr(obj, "from_pretrained")
                ):
                    return obj
        except ImportError:
            pass

    # Strategy 2: Direct package-level import
    try:
        module = importlib.import_module(f"lerobot.policies.{policy_type}")
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            if (
                isinstance(obj, type)
                and attr_name.endswith("Policy")
                and attr_name != "PreTrainedPolicy"
                and hasattr(obj, "from_pretrained")
            ):
                return obj
    except ImportError:
        pass

    # Strategy 3: Legacy get_policy_class (LeRobot <0.4)
    try:
        from lerobot.policies.factory import get_policy_class

        return get_policy_class(policy_type)
    except (ImportError, AttributeError, RuntimeError):
        pass

    # Strategy 4: PreTrainedPolicy — only if it's NOT abstract
    try:
        from lerobot.policies.pretrained import PreTrainedPolicy

        if not inspect.isabstract(PreTrainedPolicy):
            return PreTrainedPolicy
    except ImportError:
        pass

    raise ImportError(
        f"Could not resolve LeRobot policy class for type '{policy_type}'. "
        f"Tried: lerobot.policies.{policy_type}.modeling_{policy_type}, "
        f"lerobot.policies.{policy_type}, factory, PreTrainedPolicy. "
        f"Ensure lerobot is installed (pip install lerobot)."
    )


def _read_policy_type_from_config(pretrained_name_or_path: str) -> str | None:
    """Read policy type from config.json (local or HF Hub).

    Args:
        pretrained_name_or_path: Local path or HF model ID.

    Returns:
        Policy type string or None if not found.
    """
    # Try local path first
    local_path = Path(pretrained_name_or_path)
    if local_path.is_dir() and (local_path / "config.json").exists():
        with open(local_path / "config.json") as config_file:
            config = json.load(config_file)
        return config.get("type")

    # Try downloading from HuggingFace Hub
    try:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(pretrained_name_or_path, "config.json")
        with open(config_path) as config_file:
            config = json.load(config_file)
        return config.get("type")
    except (ImportError, OSError, ValueError, KeyError) as exc:
        logger.warning("Could not download config.json: %s", exc)

    return None


__all__ = ["resolve_policy_class_from_hub", "resolve_policy_class_by_name"]
