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

import functools
import importlib
import inspect
import json
import keyword
import logging
import pkgutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@functools.cache
def _ensure_policy_configs_registered() -> None:
    """Ensure LeRobot policy config classes are registered in the draccus choice registry.

    LeRobot 0.5+ uses lazy registration: each policy config class
    (``ACTConfig``, ``MolmoAct2Config``, ...) calls
    ``@PreTrainedConfig.register_subclass(...)`` at module import time.
    The previous strategy here was to import ONE known config (``act``)
    on the assumption that lerobot's eager ``policies/__init__.py`` would
    pull in every other policy as a side effect.

    That assumption is fragile:

    1. ``_ensure_lerobot_policies_importable()`` (below) installs a
       lightweight stub for ``lerobot.policies`` so we can resolve
       individual subpackages without executing the heavy
       ``__init__.py`` (which pulls in groot/transformers and can crash
       on flash-attn ABI mismatches). With the stub in place, importing
       a single ``configuration_*`` does NOT cascade -- only the one
       policy gets registered.

    2. Even WITHOUT the stub, the precedent is set: lerobot has
       progressively made subsystems lazy (``lerobot.robots.__init__``
       no longer imports its drivers eagerly). When the same lazy-init
       hits ``lerobot.policies``, every brand-new policy lerobot ships
       (e.g. ``molmoact2``, merged in lerobot PR #3604) becomes
       invisible to ``PreTrainedConfig.from_pretrained`` until something
       imports the matching subpackage by hand.

    The fix is the same pattern as ``hardware_robot._ensure_lerobot_robots_registered``:
    walk every subpackage of ``lerobot.policies`` with ``pkgutil`` and
    import each one once. That triggers every
    ``@PreTrainedConfig.register_subclass`` decorator unconditionally,
    so the registry is complete regardless of how lazy
    ``lerobot.policies.__init__`` becomes. A single ``act`` import shim
    can never be "just one more new policy" away from breaking.

    ``@functools.cache`` makes the second call a dict lookup -- the
    full walk only happens once per process.

    Caching contract for callers
    ----------------------------
    The cache is keyed on the empty argument tuple, so it is decoupled
    from ``sys.modules`` state. If something later in the process
    invalidates the lerobot import graph (a test reloads modules, a
    multiprocess spawn that mutates module state, an explicit
    ``importlib.reload(lerobot.policies)`` to pick up a freshly stubbed
    layout), callers MUST invoke
    ``_ensure_policy_configs_registered.cache_clear()`` before calling
    this helper again -- otherwise the cache returns immediately and
    the new ``sys.modules`` state is never re-walked. The
    ``test_molmoact2_registered_after_stubbed_lerobot_policies``
    regression test exercises exactly this contract.

    ImportError early-return contract
    ---------------------------------
    The ``ImportError`` early-return below (lerobot not installed, or a
    partial install / namespace conflict that survives
    ``_ensure_lerobot_policies_importable``) is **also** memoised by
    ``@functools.cache``. Once it returns, the no-op state is frozen for
    the lifetime of the process: subsequent calls hit the cache *before*
    the import is retried, so a lerobot that becomes importable later in
    the same process (e.g. a Jupyter ``pip install``) is never re-walked.
    Callers that recover from a missing/partial lerobot install MUST call
    ``_ensure_policy_configs_registered.cache_clear()`` before retrying.
    This is acceptable as a default because the realistic failure mode
    (no lerobot at all) is terminal for the resolution path.
    """
    # Make sure lerobot.policies is at least registered in sys.modules
    # without executing its (potentially heavy) __init__.
    _ensure_lerobot_policies_importable()

    try:
        import lerobot.policies as _lr_policies
    except ImportError:
        # lerobot not installed at all -- caller will fall through to
        # manual config.json resolution and produce a clean error.
        logger.debug("lerobot not installed; skipping policy config registration")
        return

    # Enumerate every immediate subpackage of ``lerobot.policies`` and
    # import its ``configuration_*`` module (or the package itself,
    # which most subpackages re-export the configuration from). Each
    # subpackage's config import runs
    # ``@PreTrainedConfig.register_subclass(...)`` as a side effect.
    #
    # Two enumeration sources, unioned:
    #
    # 1. ``pkgutil.iter_modules`` -- yields regular packages (those with
    #    ``__init__.py``). We filter with ``is_pkg=True`` so non-package
    #    siblings (``factory.py``, ``utils.py``, ``pretrained.py``,
    #    ``pi_gemma.py``) are excluded. Importing those as a package-level
    #    fallback would pull in transformers/diffusers -- exactly the heavy
    #    import graph the stub mechanism exists to avoid.
    #
    # 2. On-disk directory listing of every ``__path__`` entry. In lerobot
    #    0.5.x several subpackages are PEP 420 *namespace packages* (no
    #    ``__init__.py``) -- e.g. ``act/``, ``diffusion/``, ``smolvla/``,
    #    ``tdmpc/``, ``vqbet/``. ``iter_modules`` skips these (or yields
    #    them with ``is_pkg=False``), so the directory scan is the ground
    #    truth for namespace-package coverage on the stub-active codepath.
    #
    # The union of both sources covers all subpackages regardless of layout.
    # See issue #278 for the upstream lerobot layout that motivated this.
    sub_names: set[str] = set()
    for _, sub_name, _is_pkg in pkgutil.iter_modules(_lr_policies.__path__):
        if _is_pkg:
            sub_names.add(sub_name)
    for path_entry in _lr_policies.__path__:
        try:
            for child in Path(path_entry).iterdir():
                if not child.is_dir():
                    continue
                name = child.name
                # Skip dunder dirs (``__pycache__``) and dot dirs.
                if name.startswith("_") or name.startswith("."):
                    continue
                # Identifier check: only valid Python module names.
                # ``str.isidentifier()`` accepts Python *keywords* (``class``,
                # ``for``, ``is``, ...); a directory named after a keyword would
                # pass this filter and then fail at ``import`` with a
                # ``SyntaxError`` -- which is NOT a subclass of ``ImportError``,
                # so it would escape the per-candidate catch and abort the walk.
                # Mirror ``pkgutil``'s own filtering by also rejecting keywords.
                if not name.isidentifier() or keyword.iskeyword(name):
                    continue
                sub_names.add(name)
        except OSError as exc:
            # ``__path__`` entry not enumerable (rare; e.g. zip-imported
            # lerobot or a stale path). Fall through to whatever
            # iter_modules already produced.
            logger.debug("[policy resolution] cannot scan %s: %s", path_entry, exc)

    for sub_name in sorted(sub_names):
        # Try the canonical configuration_<name> module first because
        # it skips importing modeling_* (which is the heavy-deps file
        # that pulls in transformers/flash-attn). Fall back to the
        # package itself if that fails.
        for candidate in (
            f"{_lr_policies.__name__}.{sub_name}.configuration_{sub_name}",
            f"{_lr_policies.__name__}.{sub_name}",
        ):
            try:
                importlib.import_module(candidate)
                break  # one success per subpackage is enough
            except ImportError as exc:
                # Module simply not present (no ``configuration_<name>``
                # in this subpackage): expected, fall through to the
                # next candidate. Debug-level only -- this happens for
                # most subpackages that re-export from the package init.
                logger.debug("[policy resolution] skip %s: %s", candidate, exc)
                continue
            except (AttributeError, RuntimeError, TypeError) as exc:
                # Decorator-time failures are real and observable but
                # MUST NOT abort the walk -- otherwise one buggy
                # subpackage poisons every later registration AND the
                # @functools.cache freezes the half-populated state for
                # the rest of the process. Concrete cases this catches:
                #   * ``@PreTrainedConfig.register_subclass(...)``
                #     re-registering an already-known key
                #     (RuntimeError / TypeError depending on draccus
                #     version)
                #   * a ``configuration_*`` that imports cleanly but
                #     references a renamed attribute on a sibling
                #     module (AttributeError)
                #   * a draccus version-check that raises RuntimeError
                #     during module import
                # We log at WARNING (not DEBUG) because, unlike a missing
                # ``configuration_<name>`` shim, this signals a genuine
                # bug in either lerobot or strands_robots and an operator
                # would want to see it in normal log output.
                logger.warning(
                    "[policy resolution] %s raised %s during registration; "
                    "skipping this subpackage and continuing the walk: %s",
                    candidate,
                    type(exc).__name__,
                    exc,
                )
                # Try the next candidate for THIS subpackage; if both
                # raise, we move on to the next subpackage.
                continue


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
        # of RuntimeError/ValueError - they inherit from DraccusException → Exception.
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
    **all** policy packages (groot, act, diffusion, ...).  The groot import chain
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
        # Already imported (successfully or via a previous stub) - nothing to do.
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

    # Strategy 4: PreTrainedPolicy - only if it's NOT abstract
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
