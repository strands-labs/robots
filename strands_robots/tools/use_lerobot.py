#!/usr/bin/env python3
"""
Universal LeRobot integration - like use_aws wraps boto3.client[*], this wraps lerobot[*].

Instead of hardcoding actions, dynamically access ANY lerobot module, class,
or function. The agent discovers what's available (including draccus-registered
robot/teleop/camera/policy configs) and calls it directly.

Highlights
* **Zero hardcoding** - config choices come from lerobot's own draccus
  ``ChoiceRegistry.get_known_choices()`` registries, not a static dict.
* **Full fidelity** - results are serialized without lossy per-item truncation;
  numpy arrays are summarized structurally and image frames are returned as
  real Strands ``image`` content blocks (viewable by the model).
* **Self-describing** - ``__describe__`` introspects any object's signature,
  methods and docstring.

Usage:
    # Discover everything (modules + registered robot/teleop/camera/policy choices)
    use_lerobot(module="__discovery__", method="list_modules")

    # Describe a class (methods + init signature, no call)
    use_lerobot(module="cameras.opencv.OpenCVCamera", method="__describe__")

    # Find cameras
    use_lerobot(module="cameras.opencv.OpenCVCamera", method="find_cameras")

    # Get a registered config class by choice name
    use_lerobot(module="__registry__", method="robots")        # list robot choices
    use_lerobot(module="__registry__", method="policies")      # list policy choices

    # Get policy class
    use_lerobot(module="policies.factory", method="get_policy_class",
                parameters={"name": "act"})

    # Capture a frame -> returned as an IMAGE content block
    # (instantiate camera elsewhere, or use lerobot_camera tool for full flow)
"""

import importlib
import inspect
import json
import logging
import pkgutil
from typing import Any

from strands import tool

logger = logging.getLogger(__name__)

# Caps chosen to preserve content fully in practice while protecting the
# context window from pathological blobs. These are deliberately huge compared
# to the old [:200] - we keep contents "as is" for anything reasonable.
_MAX_STR = 200_000  # max chars for a single serialized string field
_MAX_LIST_ITEMS = 1_000  # max items rendered from a list/tuple
_MAX_DICT_ITEMS = 1_000  # max keys rendered from a dict
_IMAGE_MAX_DIM = 4096  # arrays larger than this on H or W are still treated as images

# Security hardening: module prefixes whose callables must not be invoked by
# the LLM.  These modules contain filesystem-write, network-push, or
# subprocess-spawn operations that could be weaponised via prompt injection.
_BLOCKED_MODULE_PREFIXES: tuple[str, ...] = (
    "lerobot.scripts",  # training scripts that spawn subprocesses
    "lerobot.common.datasets.push",  # HuggingFace Hub push / upload
)

# Methods on otherwise-safe modules that have dangerous side effects.
_BLOCKED_METHODS: set[str] = {
    "push_to_hub",
    "push_to_hub_revision",
    "upload_folder",
    "save_to_disk",
}


# Import resolution
class LeRobotResolveError(Exception):
    """Raised when a lerobot path cannot be resolved. Carries a hint about
    whether the path is genuinely missing or failed due to a dependency."""

    def __init__(self, message: str, real_error: Exception | None = None):
        super().__init__(message)
        self.real_error = real_error


def _import_from_lerobot(module_path: str):
    """Resolve a dotted path into lerobot to a module / class / function / attr.

    Surfaces the *real* ImportError when a module exists on disk but fails to
    import (missing optional dep), instead of a misleading generic message.
    """
    full_path = f"lerobot.{module_path}" if not module_path.startswith("lerobot.") else module_path

    segments = full_path.split(".")
    if not segments or segments == [""]:
        raise LeRobotResolveError(f"Cannot resolve '{module_path}' in lerobot")

    # Walk from the longest importable module prefix down to the shortest. The
    # first prefix that imports cleanly is the module; remaining segments are
    # attribute lookups (Class, function, nested attr). If a prefix exists on
    # disk but its import raises due to a *third-party* dependency, we remember
    # that real error and surface it rather than a misleading "cannot resolve".
    last_real_error: Exception | None = None
    for i in range(len(segments), 0, -1):
        modname = ".".join(segments[:i])
        attrs = segments[i:]
        try:
            mod = importlib.import_module(modname)
        except ImportError as e:
            # Classify: is the *missing* module lerobot's own (genuine not-found)
            # or a third-party optional dependency (helpful "install extra")?
            missing = getattr(e, "name", "") or ""
            is_self_missing = missing == modname or modname.startswith(missing + ".") or missing.startswith("lerobot")
            if not is_self_missing:
                # A real third-party dependency is missing -> surface it.
                last_real_error = e
            continue
        except Exception as e:  # non-Import errors during import are real too
            last_real_error = e
            continue
        # Imported OK - now walk attributes
        obj = mod
        try:
            for attr in attrs:
                obj = getattr(obj, attr)
            return obj
        except AttributeError:
            # Maybe the attr is itself a submodule that needs importing, or the
            # split point is wrong - keep trying shorter prefixes.
            continue

    # Nothing resolved. If we saw a real dependency/import error, surface it.
    if last_real_error is not None:
        raise LeRobotResolveError(
            f"'{module_path}' exists but failed to import: {last_real_error}",
            real_error=last_real_error,
        )
    raise LeRobotResolveError(f"Cannot resolve '{module_path}' in lerobot")


# Packages whose submodule tree we've already walked this process. The actual
# module objects are cached by Python in sys.modules; this avoids re-walking the
# (recursive) package tree on every discovery/registry call.
_DEEP_IMPORTED: set = set()


def _deep_import(pkg_name: str, _seen: set | None = None) -> None:
    """Recursively import all submodules of a package so that draccus
    ``register_subclass`` decorators run and populate the choice registries.

    lerobot registers robot/teleop/camera/policy configs lazily at import time,
    so discovery is empty until the modules are imported. This walks the tree.
    """
    _seen = _seen if _seen is not None else _DEEP_IMPORTED
    if pkg_name in _seen:
        return
    _seen.add(pkg_name)
    try:
        mod = importlib.import_module(pkg_name)
    except Exception:
        return
    if not hasattr(mod, "__path__"):
        return
    for _, sub, _ispkg in pkgutil.iter_modules(mod.__path__):
        if sub.startswith("_") or sub == "tests":
            continue
        _deep_import(f"{pkg_name}.{sub}", _seen)


# Registry discovery (the un-hardcoded part)
# Each entry: choice-name -> (package to deep-import, dotted ChoiceRegistry class)
_REGISTRIES = {
    "robots": ("lerobot.robots", "lerobot.robots.config.RobotConfig"),
    "teleoperators": ("lerobot.teleoperators", "lerobot.teleoperators.config.TeleoperatorConfig"),
    "cameras": ("lerobot.cameras", "lerobot.cameras.configs.CameraConfig"),
    "policies": ("lerobot.policies", "lerobot.configs.policies.PreTrainedConfig"),
}


def _get_registry_choices(kind: str) -> dict[str, str]:
    """Return {choice_name: fully.qualified.ConfigClass} for a registry kind.

    Triggers a deep import first so all subclasses are registered, then reads
    lerobot's own ``ChoiceRegistry.get_known_choices()`` - no hardcoding.
    """
    if kind not in _REGISTRIES:
        return {}
    pkg, cls_path = _REGISTRIES[kind]
    _deep_import(pkg)
    try:
        cls = _import_from_lerobot(cls_path)
    except Exception as e:  # pragma: no cover
        logger.debug(f"registry {kind}: cannot import {cls_path}: {e}")
        return {}
    if not hasattr(cls, "get_known_choices"):
        return {}
    out: dict[str, str] = {}
    try:
        for name, klass in cls.get_known_choices().items():
            out[name] = f"{klass.__module__}.{klass.__qualname__}"
    except Exception as e:  # pragma: no cover
        logger.debug(f"registry {kind}: get_known_choices failed: {e}")
    return out


def _discover_modules() -> dict[str, Any]:
    """Discover lerobot submodules and ALL registered config choices dynamically."""
    try:
        import lerobot
    except ImportError:
        return {"error": "lerobot not installed"}

    result: dict[str, Any] = {"packages": [], "modules": [], "registries": {}}

    for _importer, modname, ispkg in pkgutil.iter_modules(lerobot.__path__):
        if modname.startswith("_") or modname == "tests":
            continue
        (result["packages"] if ispkg else result["modules"]).append(modname)

    # Dynamic registry discovery - this replaces the old hardcoded key_apis
    for kind in _REGISTRIES:
        result["registries"][kind] = _get_registry_choices(kind)

    # A few stable, genuinely useful entry points (functions/paths, not configs).
    # These are call targets, not enumerable via a registry.
    result["entry_points"] = {
        "cameras.opencv.OpenCVCamera.find_cameras": "Discover connected cameras",
        "datasets.lerobot_dataset.LeRobotDataset": "Dataset create/load/push/pull",
        "datasets.lerobot_dataset.LeRobotDataset.create": "Create a new dataset",
        "policies.factory.get_policy_class": "Get policy class by registered name",
        "policies.factory.make_policy": "Build a policy instance from config",
        "robots.utils.make_robot_from_config": "Instantiate a robot from its config",
        "teleoperators.utils.make_teleoperator_from_config": "Instantiate a teleoperator",
        "scripts.lerobot_train.train": "Train a policy on a dataset",
        "model.kinematics.RobotKinematics": "Forward/inverse kinematics",
        "envs.factory.make_env": "Create a gym environment",
        "utils.constants.HF_LEROBOT_CALIBRATION": "Calibration directory path",
    }
    return result


# Introspection
def _describe_object(obj) -> dict[str, Any]:
    """Describe a Python object - methods, signature, docstring (full, untruncated)."""
    info: dict[str, Any] = {
        "type": type(obj).__name__,
        "name": getattr(obj, "__name__", str(obj)),
    }

    if inspect.isclass(obj):
        # Use getattr_static so we never trigger descriptor/property side-effects
        # (some lerobot classes have properties that touch hardware on access).
        methods, class_methods, properties = [], [], []
        for m in dir(obj):
            if m.startswith("_"):
                continue
            static_attr = inspect.getattr_static(obj, m, None)
            if isinstance(static_attr, (classmethod, staticmethod)):
                class_methods.append(m)
            elif isinstance(static_attr, property):
                properties.append(m)
            elif callable(static_attr):
                methods.append(m)
        info["methods"] = methods
        info["class_methods"] = class_methods
        if properties:
            info["properties"] = properties
        if getattr(obj, "__init__", None) and obj.__init__.__doc__:
            info["init_doc"] = obj.__init__.__doc__
        try:
            sig = inspect.signature(obj.__init__)
            info["init_params"] = [p for p in sig.parameters if p != "self"]
        except (ValueError, TypeError):
            # Builtins / C-extension types expose no introspectable signature;
            # skip init_params rather than failing the describe call.
            pass

    elif callable(obj):
        try:
            sig = inspect.signature(obj)
            info["params"] = {
                name: {
                    "default": (str(p.default) if p.default is not inspect.Parameter.empty else "REQUIRED"),
                    "annotation": (str(p.annotation) if p.annotation is not inspect.Parameter.empty else None),
                }
                for name, p in sig.parameters.items()
            }
        except (ValueError, TypeError):
            # Builtins / C-extension callables expose no introspectable
            # signature; describe them without a params map.
            pass
        if obj.__doc__:
            info["doc"] = obj.__doc__

    elif inspect.ismodule(obj):
        info["public_names"] = [n for n in dir(obj) if not n.startswith("_")]
        if obj.__doc__:
            info["doc"] = obj.__doc__

    else:
        info["value"] = str(obj)

    return info


# Image detection + encoding -> Strands content blocks
def _is_image_array(arr) -> bool:
    """Heuristic: is this numpy array an image frame?

    Accepts HxW (grayscale) or HxWxC (C in {1,3,4}) uint8/uint16/float arrays
    with sane spatial dims.
    """
    try:
        import numpy as np
    except ImportError:
        return False
    if not isinstance(arr, np.ndarray):
        return False
    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        h, w, c = arr.shape
        if c not in (1, 3, 4):
            return False
    else:
        return False
    return 2 <= h <= _IMAGE_MAX_DIM and 2 <= w <= _IMAGE_MAX_DIM


def _array_to_image_block(arr) -> dict[str, Any] | None:
    """Encode a numpy image array to a Strands ``image`` content block (PNG/JPEG bytes).

    Returns None if cv2 is unavailable or encoding fails. Assumes RGB input
    (lerobot cameras default to RGB) and converts to BGR for cv2 encoding.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    try:
        frame = arr
        # Normalize float images to uint8
        if frame.dtype != np.uint8:
            if np.issubdtype(frame.dtype, np.floating):
                fmax = float(frame.max()) if frame.size else 1.0
                scale = 255.0 if fmax <= 1.0 else 255.0 / fmax
                frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # RGB(A)->BGR for cv2
        if frame.ndim == 3 and frame.shape[2] == 3:
            enc_src = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            enc_src = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGRA)
        else:
            enc_src = frame  # grayscale

        # Choose codec: JPEG for large opaque frames (far smaller in-context),
        # PNG for small frames, grayscale, or anything with an alpha channel.
        h, w = enc_src.shape[:2]
        has_alpha = enc_src.ndim == 3 and enc_src.shape[2] == 4
        use_jpeg = (not has_alpha) and (h * w) > (320 * 240)
        if use_jpeg:
            ok, buf = cv2.imencode(".jpg", enc_src, [cv2.IMWRITE_JPEG_QUALITY, 85])
            fmt = "jpeg"
            if not ok:  # fall back to PNG if JPEG encoding refuses
                ok, buf = cv2.imencode(".png", enc_src)
                fmt = "png"
        else:
            ok, buf = cv2.imencode(".png", enc_src)
            fmt = "png"
        if not ok:
            return None
        return {"image": {"format": fmt, "source": {"bytes": buf.tobytes()}}}
    except Exception as e:  # pragma: no cover
        logger.debug(f"image encode failed: {e}")
        return None


# Result serialization (full fidelity, image-aware)
def _collect_images(result: Any, _blocks: list[dict[str, Any]], _depth: int = 0) -> None:
    """Walk a result and collect any image arrays as Strands image content blocks.

    Looks at the top-level value, and one level into lists/tuples/dicts (e.g. a
    dict of {camera_name: frame} or a list of frames).
    """
    if _depth > 2 or len(_blocks) >= 16:
        return
    if _is_image_array(result):
        blk = _array_to_image_block(result)
        if blk:
            _blocks.append(blk)
        return
    if isinstance(result, (list, tuple)):
        for item in result[:16]:
            _collect_images(item, _blocks, _depth + 1)
    elif isinstance(result, dict):
        for v in list(result.values())[:16]:
            _collect_images(v, _blocks, _depth + 1)


def _serialize_value(value: Any, _seen: set | None = None, _depth: int = 0) -> Any:
    """Recursively convert a value to a JSON-friendly structure, full fidelity.

    Unlike the old version, this does NOT truncate individual items to 200 chars.
    Strings are kept whole (up to a large safety cap); numpy arrays are described
    structurally (shape/dtype) since raw pixel dumps are useless as text.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value

    # Depth guard - prevent runaway nesting blowing the stack.
    if _depth > 50:
        return "...[max depth exceeded]"
    _seen = _seen if _seen is not None else set()
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR else value[:_MAX_STR] + f"...[+{len(value) - _MAX_STR} chars]"

    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": True, "length": len(value), "preview_hex": value[:64].hex()}

    # numpy arrays -> structural summary (image bytes handled separately)
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            summary = {"__ndarray__": True, "shape": list(value.shape), "dtype": str(value.dtype)}
            if _is_image_array(value):
                summary["is_image"] = True
                summary["note"] = "returned as an image content block"
            elif value.size <= 64:
                summary["values"] = value.tolist()
            return summary
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
    except ImportError:
        # numpy absent: fall through to the plain-Python serialization below.
        pass

    if isinstance(value, (list, tuple)):
        vid = id(value)
        if vid in _seen:
            return "...[circular ref]"
        _seen = _seen | {vid}
        items = [_serialize_value(v, _seen, _depth + 1) for v in value[:_MAX_LIST_ITEMS]]
        if len(value) > _MAX_LIST_ITEMS:
            items.append(f"...[+{len(value) - _MAX_LIST_ITEMS} more items]")
        return items

    if isinstance(value, dict):
        vid = id(value)
        if vid in _seen:
            return "...[circular ref]"
        _seen = _seen | {vid}
        out = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= _MAX_DICT_ITEMS:
                out["__truncated__"] = f"+{len(value) - _MAX_DICT_ITEMS} more keys"
                break
            out[str(k)] = _serialize_value(v, _seen, _depth + 1)
        return out

    # dataclass / config objects -> dict of fields
    if hasattr(value, "__dataclass_fields__"):
        try:
            from dataclasses import asdict

            return {"__dataclass__": type(value).__name__, **_serialize_value(asdict(value), _seen, _depth + 1)}
        except Exception:
            pass

    # Fallback: describe objects (classes/instances/functions/modules)
    if inspect.isclass(value) or inspect.isfunction(value) or inspect.ismethod(value) or inspect.ismodule(value):
        return _describe_object(value)

    # Last resort - full repr (capped)
    s = str(value)
    return s if len(s) <= _MAX_STR else s[:_MAX_STR] + f"...[+{len(s) - _MAX_STR} chars]"


def _serialize_result(result: Any) -> str:
    """Convert any result to a full-fidelity JSON string (no per-item 200-char cap)."""
    try:
        return json.dumps(_serialize_value(result), indent=2, default=str)
    except Exception as e:  # pragma: no cover
        return f"<unserializable {type(result).__name__}: {e}>"


# The tool
@tool
def use_lerobot(
    module: str = "__discovery__",
    method: str = "list_modules",
    parameters: dict[str, Any] | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Universal LeRobot access - call any lerobot module, class, or function dynamically.

    Like use_aws wraps boto3.client[service].operation(**params), this wraps
    lerobot[module].method(**params). The agent discovers available APIs and
    registered configs (robots/teleoperators/cameras/policies) directly from
    lerobot's own draccus registries - nothing is hardcoded.

    Results are serialized at full fidelity (no lossy truncation), and any image
    frames (numpy HxW / HxWxC arrays) returned by a call are emitted as proper
    Strands ``image`` content blocks so the model can actually see them.

    Args:
        module: Dotted path into lerobot (e.g. "cameras.opencv.OpenCVCamera",
                "datasets.lerobot_dataset.LeRobotDataset", "policies.factory").
                Special values:
                  "__discovery__" - explore modules + all registered configs.
                  "__registry__"  - list a single registry; pass its name as
                                     ``method`` (robots|teleoperators|cameras|policies).
        method: Method/function/attribute name to call or read. Special values:
                  "list_modules" - discovery output (with module="__discovery__").
                  "__describe__" - inspect the object without calling it.
        parameters: Dict of kwargs to pass to the method. Omit for no-arg calls.
        label: Human-readable description of what this call does.

    Returns:
        Dict with status and content; content may include text + image blocks.

    Examples:
        # Discover everything (modules + registered choices)
        use_lerobot(module="__discovery__", method="list_modules")

        # List just the robot choices (dynamic, from registry)
        use_lerobot(module="__registry__", method="robots")

        # Describe a class
        use_lerobot(module="cameras.opencv.OpenCVCamera", method="__describe__")

        # Find cameras
        use_lerobot(module="cameras.opencv.OpenCVCamera", method="find_cameras")

        # Get a policy class
        use_lerobot(module="policies.factory", method="get_policy_class",
                    parameters={"name": "act"})

        # Read a calibration path constant
        use_lerobot(module="utils.constants", method="HF_LEROBOT_CALIBRATION")
    """
    params = parameters or {}

    try:
        # Discovery mode
        if module == "__discovery__":
            discovery = _discover_modules()
            if "error" in discovery:
                return {"status": "error", "content": [{"text": f"Error: {discovery['error']}"}]}

            lines = ["LeRobot API Discovery\n"]
            lines.append(f"Packages: {', '.join(discovery['packages'])}")
            lines.append(f"Modules: {', '.join(discovery['modules'])}")

            for kind, choices in discovery.get("registries", {}).items():
                lines.append(f"\n{kind} ({len(choices)}):")
                for name, cls in choices.items():
                    lines.append(f"  - {name}  ->  {cls}")

            lines.append("\nEntry points (callables / paths):")
            for path, desc in discovery.get("entry_points", {}).items():
                lines.append(f"  - lerobot.{path}\n    {desc}")

            lines.append("\nUsage:")
            lines.append('  use_lerobot(module="__registry__", method="robots")')
            lines.append('  use_lerobot(module="cameras.opencv.OpenCVCamera", method="find_cameras")')
            lines.append(
                '  use_lerobot(module="policies.factory", method="get_policy_class", parameters={"name":"act"})'
            )

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        # Single-registry listing
        if module == "__registry__":
            kind = method or "robots"
            choices = _get_registry_choices(kind)
            if not choices:
                return {
                    "status": "error",
                    "content": [
                        {"text": (f"Unknown or empty registry '{kind}'. Valid: {', '.join(_REGISTRIES.keys())}")}
                    ],
                }
            lines = [f"lerobot {kind} registry ({len(choices)} choices):\n"]
            for name, cls in choices.items():
                lines.append(f"  - {name}  ->  {cls}")
            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        # Resolve the target object
        target = _import_from_lerobot(module)

        # Describe mode (inspect without calling)
        if method == "__describe__":
            info = _describe_object(target)
            return {"status": "success", "content": [{"text": f"{module}\n{json.dumps(info, indent=2, default=str)}"}]}

        # Get the method/attribute
        if method:
            if hasattr(target, method):
                target = getattr(target, method)
            else:
                available = [a for a in dir(target) if not a.startswith("_")]
                return {
                    "status": "error",
                    "content": [{"text": (f"'{method}' not found on {module}\nAvailable: {', '.join(available)}")}],
                }

        # If target is not callable, just return its value
        if not callable(target):
            return {"status": "success", "content": [{"text": f"{module}.{method} = {_serialize_result(target)}"}]}

        # Security hardening: block callables in denied modules or methods.
        full_path = f"lerobot.{module}" if not module.startswith("lerobot.") else module
        for prefix in _BLOCKED_MODULE_PREFIXES:
            if full_path.startswith(prefix):
                return {
                    "status": "error",
                    "content": [{"text": f"Blocked: {full_path} is in a restricted module namespace"}],
                }
        if method in _BLOCKED_METHODS:
            return {
                "status": "error",
                "content": [{"text": f"Blocked: method '{method}' is not permitted via use_lerobot"}],
            }

        # Call it
        if label:
            logger.info(f" LeRobot: {label} - {module}.{method}({list(params.keys())})")

        result = target(**params)

        # Collect image blocks (full-frame, model-viewable)
        image_blocks: list[dict[str, Any]] = []
        _collect_images(result, image_blocks)

        serialized = _serialize_result(result)
        content: list[dict[str, Any]] = [{"text": f"{module}.{method}() ->\n{serialized}"}]
        if image_blocks:
            content[0]["text"] += f"\n\n({len(image_blocks)} image block(s) attached below)"
            content.extend(image_blocks)

        return {"status": "success", "content": content}

    except LeRobotResolveError as e:
        # Distinguish "doesn't exist" from "exists but a dependency is missing".
        if e.real_error is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"{e}\n\n"
                            f"This path exists in lerobot but an optional dependency is missing. "
                            f"Install the relevant extra, e.g.:\n"
                            f"  uv pip install 'lerobot[dataset]'   # datasets / training\n"
                            f"  uv pip install 'lerobot[feetech]'   # SO-100/101 motors"
                        )
                    }
                ],
            }
        return {
            "status": "error",
            "content": [{"text": (f'{e}\n\nTip: use module="__discovery__" to list valid modules and registries.')}],
        }

    except ImportError as e:
        return {"status": "error", "content": [{"text": f"Import error: {e}\n\nInstall: pip install lerobot"}]}

    except TypeError as e:
        try:
            sig = inspect.signature(target)
            param_info = {name: str(p) for name, p in sig.parameters.items()}
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"TypeError: {e}\n\n"
                            f"Expected signature for {module}.{method}:\n"
                            f"{json.dumps(param_info, indent=2)}"
                        )
                    }
                ],
            }
        except Exception:
            return {"status": "error", "content": [{"text": f"TypeError: {e}"}]}

    except Exception as e:
        logger.error(f"use_lerobot({module}.{method}) failed: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"{type(e).__name__}: {e}"}]}
