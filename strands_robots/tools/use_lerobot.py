#!/usr/bin/env python3
"""
Universal LeRobot integration — like use_aws wraps boto3.client[*], this wraps lerobot[*].

Instead of hardcoding 41 actions, dynamically access ANY lerobot module, class,
or function. The agent discovers what's available and calls it directly.

Usage:
    # Discover cameras
    use_lerobot(module="cameras.opencv.OpenCVCamera", method="find_cameras")

    # Create a dataset
    use_lerobot(module="datasets.lerobot_dataset.LeRobotDataset", method="create",
                parameters={"repo_id": "user/data", "fps": 30, ...})

    # Train a policy
    use_lerobot(module="scripts.lerobot_train", method="train",
                parameters={"...": "..."})

    # Get policy class
    use_lerobot(module="policies.factory", method="get_policy_class",
                parameters={"name": "act"})

    # List available services (like use_aws lists services)
    use_lerobot(module="__discovery__", method="list_modules")
"""

import importlib
import inspect
import json
import logging
import pkgutil
from typing import Any, Dict

from strands import tool

logger = logging.getLogger(__name__)

# Safety: block module paths that could escape the lerobot namespace
_BLOCKED_PATTERNS = frozenset(
    {
        "__builtins__",
        "__import__",
        "__subclasses__",
        "os.",
        "sys.",
        "subprocess.",
        "shutil.",
        "pathlib.",
    }
)

# Methods that require explicit confirmation (destructive operations)
# TODO: Wire this into the dispatch logic to gate destructive operations,
# similar to _DANGEROUS_ACTIONS in use_unitree.py
_SENSITIVE_METHODS = frozenset(
    {
        "delete",
        "remove",
        "drop",
        "push_to_hub",
        "rmtree",
    }
)


def _import_from_lerobot(module_path: str):
    """Import a module/class/function from lerobot.

    Examples:
        _import_from_lerobot("cameras.opencv.OpenCVCamera")
        → <class 'lerobot.cameras.opencv.OpenCVCamera'>

        _import_from_lerobot("datasets.lerobot_dataset")
        → <module 'lerobot.datasets.lerobot_dataset'>

        _import_from_lerobot("policies.factory.get_policy_class")
        → <function get_policy_class>
    """
    # Safety: reject paths that could escape the lerobot namespace
    if any(pat in module_path.lower() for pat in _BLOCKED_PATTERNS):
        raise ImportError(f"Access to '{module_path}' is blocked for safety")

    full_path = f"lerobot.{module_path}" if not module_path.startswith("lerobot.") else module_path

    # Try importing as a module first
    try:
        return importlib.import_module(full_path)
    except ImportError:
        pass

    # Split off the last part and try as an attribute
    parts = full_path.rsplit(".", 1)
    if len(parts) == 2:
        try:
            mod = importlib.import_module(parts[0])
            return getattr(mod, parts[1])
        except (ImportError, AttributeError):
            pass

    # Try splitting at multiple levels (e.g. module.Class.method)
    segments = full_path.split(".")
    for i in range(len(segments), 0, -1):
        try:
            mod = importlib.import_module(".".join(segments[:i]))
            obj = mod
            for attr in segments[i:]:
                obj = getattr(obj, attr)
            return obj
        except (ImportError, AttributeError):
            continue

    raise ImportError(f"Cannot resolve '{module_path}' in lerobot")


def _discover_modules() -> Dict[str, Any]:
    """Discover all lerobot submodules and their public APIs."""
    try:
        import lerobot
    except ImportError:
        return {"error": "lerobot not installed"}

    result = {"packages": [], "modules": [], "key_apis": {}}

    # Walk top-level
    for importer, modname, ispkg in pkgutil.iter_modules(lerobot.__path__):
        if modname.startswith("_") or modname == "tests":
            continue
        if ispkg:
            result["packages"].append(modname)
        else:
            result["modules"].append(modname)

    # Key APIs — the most useful entry points
    key_apis = {
        "cameras.opencv.OpenCVCamera": "Camera capture, discovery, streaming",
        "cameras.opencv.OpenCVCamera.find_cameras": "Discover connected cameras",
        "datasets.lerobot_dataset.LeRobotDataset": "Dataset create/load/push/pull",
        "datasets.lerobot_dataset.LeRobotDataset.create": "Create new dataset",
        "policies.factory.get_policy_class": "Get policy class by name (act, pi0, smolvla...)",
        "robots.utils.make_robot_from_config": "Create robot from config",
        "robots.config.RobotConfig": "Base robot configuration",
        "teleoperators.utils.make_teleoperator_from_config": "Create teleoperator",
        "scripts.lerobot_train.train": "Train a policy on a dataset",
        "configs.train.TrainPipelineConfig": "Training configuration",
        "configs.policies.PreTrainedConfig": "Pretrained model configuration",
        "configs.default.DatasetConfig": "Dataset configuration for training",
        "processor.pipeline.DataProcessorPipeline": "Observation/action transforms",
        "model.kinematics.RobotKinematics": "Forward/inverse kinematics",
        "envs.factory.make_env": "Create gym environment",
        "utils.constants.HF_LEROBOT_CALIBRATION": "Calibration file path",
    }
    result["key_apis"] = key_apis

    # Discover available robots
    try:
        import lerobot.robots as lr_robots
        from lerobot.robots.config import RobotConfig

        robot_configs = []
        for _, modname, _ in pkgutil.iter_modules(lr_robots.__path__):
            if modname in ("config", "robot", "utils"):
                continue
            try:
                mod = importlib.import_module(f"lerobot.robots.{modname}")
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    if (
                        isinstance(obj, type)
                        and attr.endswith("Config")
                        and issubclass(obj, RobotConfig)
                        and obj is not RobotConfig
                    ):
                        robot_configs.append(f"robots.{modname}.{attr}")
            except Exception:
                continue
        result["robot_configs"] = robot_configs
    except Exception:
        pass

    # Discover available teleoperators
    try:
        import lerobot.teleoperators as lr_teleops
        from lerobot.teleoperators.config import TeleoperatorConfig

        teleop_configs = []
        for _, modname, _ in pkgutil.iter_modules(lr_teleops.__path__):
            if modname in ("config", "teleoperator", "utils"):
                continue
            try:
                mod = importlib.import_module(f"lerobot.teleoperators.{modname}")
                for attr in dir(mod):
                    obj = getattr(mod, attr)
                    if (
                        isinstance(obj, type)
                        and attr.endswith("Config")
                        and issubclass(obj, TeleoperatorConfig)
                        and obj is not TeleoperatorConfig
                    ):
                        teleop_configs.append(f"teleoperators.{modname}.{attr}")
            except Exception:
                continue
        result["teleop_configs"] = teleop_configs
    except Exception:
        pass

    return result


def _describe_object(obj) -> Dict[str, Any]:
    """Describe a Python object for the agent — methods, signature, docstring."""
    info = {
        "type": type(obj).__name__,
        "name": getattr(obj, "__name__", str(obj)),
    }

    if inspect.isclass(obj):
        info["methods"] = [m for m in dir(obj) if not m.startswith("_") and callable(getattr(obj, m, None))]
        info["class_methods"] = [
            m
            for m in dir(obj)
            if not m.startswith("_") and isinstance(inspect.getattr_static(obj, m, None), (classmethod, staticmethod))
        ]
        if obj.__init__.__doc__:
            info["init_doc"] = obj.__init__.__doc__[:500]
        try:
            sig = inspect.signature(obj.__init__)
            info["init_params"] = [p for p in sig.parameters if p != "self"]
        except (ValueError, TypeError):
            pass

    elif callable(obj):
        try:
            sig = inspect.signature(obj)
            info["params"] = {
                name: {
                    "default": str(p.default) if p.default is not inspect.Parameter.empty else "REQUIRED",
                    "annotation": str(p.annotation) if p.annotation is not inspect.Parameter.empty else None,
                }
                for name, p in sig.parameters.items()
            }
        except (ValueError, TypeError):
            pass
        if obj.__doc__:
            info["doc"] = obj.__doc__[:500]

    elif inspect.ismodule(obj):
        info["public_names"] = [n for n in dir(obj) if not n.startswith("_")][:30]
        if obj.__doc__:
            info["doc"] = obj.__doc__[:300]

    else:
        info["value"] = str(obj)[:200]

    return info


def _serialize_result(result: Any) -> str:
    """Convert any result to a JSON-friendly string."""
    if result is None:
        return "None"
    if isinstance(result, (str, int, float, bool)):
        return str(result)
    if isinstance(result, (list, tuple)):
        return json.dumps([str(x)[:200] for x in result[:50]], indent=2)
    if isinstance(result, dict):
        safe = {}
        for k, v in list(result.items())[:50]:
            try:
                json.dumps(v)
                safe[str(k)] = v
            except (TypeError, ValueError):
                safe[str(k)] = str(v)[:200]
        return json.dumps(safe, indent=2, default=str)
    # For objects — describe them
    return json.dumps(_describe_object(result), indent=2, default=str)


@tool
def use_lerobot(
    module: str = "__discovery__",
    method: str = "list_modules",
    parameters: Dict[str, Any] = None,
    label: str = "",
) -> Dict[str, Any]:
    """Universal LeRobot access — call any lerobot module, class, or function dynamically.

    Like use_aws wraps boto3.client[service].operation(**params), this wraps
    lerobot[module].method(**params). The agent can discover available APIs
    and call them directly without any hardcoded action mapping.

    Args:
        module: Dotted path into lerobot (e.g. "cameras.opencv.OpenCVCamera",
                "datasets.lerobot_dataset.LeRobotDataset", "policies.factory").
                Use "__discovery__" to explore available modules.
        method: Method or function name to call (e.g. "find_cameras", "create",
                "get_policy_class"). Use "list_modules" for discovery.
                Use "__describe__" to inspect the object without calling it.
        parameters: Dict of kwargs to pass to the method. Omit for no-arg calls.
        label: Human-readable description of what this call does.

    Returns:
        Dict with status and result content

    Examples:
        # Discover what's available
        use_lerobot(module="__discovery__", method="list_modules")

        # Describe a class (see its methods + init signature)
        use_lerobot(module="cameras.opencv.OpenCVCamera", method="__describe__")

        # Find cameras
        use_lerobot(module="cameras.opencv.OpenCVCamera", method="find_cameras")

        # Create a dataset
        use_lerobot(module="datasets.lerobot_dataset.LeRobotDataset", method="create",
                    parameters={"repo_id": "user/my_data", "fps": 30,
                                "features": {...}, "robot_type": "so100"})

        # Load a dataset
        use_lerobot(module="datasets.lerobot_dataset", method="LeRobotDataset",
                    parameters={"repo_id": "lerobot/aloha_sim_transfer_cube_human"})

        # Get policy class
        use_lerobot(module="policies.factory", method="get_policy_class",
                    parameters={"name": "act"})

        # Get calibration path constant
        use_lerobot(module="utils.constants", method="HF_LEROBOT_CALIBRATION")

        # Train
        use_lerobot(module="scripts.lerobot_train", method="train",
                    parameters={"cfg": "<TrainPipelineConfig object>"})
    """
    params = parameters or {}

    try:
        # ── Discovery mode ──
        if module == "__discovery__":
            discovery = _discover_modules()
            if "error" in discovery:
                return {"status": "error", "content": [{"text": f"❌ {discovery['error']}"}]}

            lines = ["🔍 LeRobot API Discovery\n"]
            lines.append(f"📦 Packages: {', '.join(discovery['packages'])}")
            lines.append(f"📄 Modules: {', '.join(discovery['modules'])}")

            lines.append("\n🔑 Key APIs:")
            for path, desc in discovery.get("key_apis", {}).items():
                lines.append(f"  • lerobot.{path}")
                lines.append(f"    {desc}")

            if discovery.get("robot_configs"):
                lines.append(f"\n🤖 Robot Configs ({len(discovery['robot_configs'])}):")
                for rc in discovery["robot_configs"]:
                    lines.append(f"  • {rc}")

            if discovery.get("teleop_configs"):
                lines.append(f"\n🎮 Teleop Configs ({len(discovery['teleop_configs'])}):")
                for tc in discovery["teleop_configs"]:
                    lines.append(f"  • {tc}")

            lines.append("\n💡 Usage:")
            lines.append('  use_lerobot(module="cameras.opencv.OpenCVCamera", method="find_cameras")')
            lines.append('  use_lerobot(module="datasets.lerobot_dataset.LeRobotDataset", method="__describe__")')

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        # ── Resolve the target object ──
        target = _import_from_lerobot(module)

        # ── Describe mode (inspect without calling) ──
        if method == "__describe__":
            info = _describe_object(target)
            return {
                "status": "success",
                "content": [{"text": f"🔍 {module}\n{json.dumps(info, indent=2, default=str)}"}],
            }

        # ── Get the method/attribute ──
        if method:
            if hasattr(target, method):
                target = getattr(target, method)
            else:
                # Maybe the method IS the target (e.g. module="utils.constants", method="HF_LEROBOT_CALIBRATION")
                # Already resolved, check if it's callable
                available = [a for a in dir(target) if not a.startswith("_")][:20]
                return {
                    "status": "error",
                    "content": [
                        {"text": (f"❌ '{method}' not found on {module}\n" f"Available: {', '.join(available)}")}
                    ],
                }

        # ── If target is not callable, just return its value ──
        if not callable(target):
            return {"status": "success", "content": [{"text": f"📋 {module}.{method} = {_serialize_result(target)}"}]}

        # ── Call it ──
        if label:
            logger.info(f"🤖 LeRobot: {label} — {module}.{method}({list(params.keys())})")

        result = target(**params)
        serialized = _serialize_result(result)

        return {"status": "success", "content": [{"text": f"✅ {module}.{method}() → {serialized[:3000]}"}]}

    except ImportError as e:
        return {"status": "error", "content": [{"text": f"❌ Import error: {e}\n\nInstall: pip install lerobot"}]}

    except TypeError as e:
        # Wrong params — try to show the expected signature
        try:
            sig = inspect.signature(target)
            param_info = {name: str(p) for name, p in sig.parameters.items()}
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"❌ TypeError: {e}\n\n"
                            f"Expected signature for {module}.{method}:\n"
                            f"{json.dumps(param_info, indent=2)}"
                        )
                    }
                ],
            }
        except Exception:
            return {"status": "error", "content": [{"text": f"❌ TypeError: {e}"}]}

    except Exception as e:
        logger.error(f"use_lerobot({module}.{method}) failed: {e}", exc_info=True)
        return {"status": "error", "content": [{"text": f"❌ {type(e).__name__}: {e}"}]}
