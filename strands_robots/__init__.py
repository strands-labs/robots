"""
Strands Robots — AI-powered robot control, simulation, and training.

Control robots with natural language through Strands Agents. Unified interface
across NVIDIA GR00T, Isaac Sim/Lab, HuggingFace LeRobot, and MuJoCo with
18 policy providers and 35+ bundled robot models.

Quick Start:
    >>> from strands_robots import Robot
    >>> sim = Robot("so100")  # MuJoCo sim (auto-detected)

    >>> from strands import Agent
    >>> agent = Agent(tools=[sim])
    >>> agent("Pick up the red block with the mock policy")

Robot Factory:
    Robot("so100")           → MuJoCo sim (auto-detected, no USB hardware)
    Robot("so100", mode="real", cameras={...})  → Real hardware
    Robot("unitree_go2", backend="isaac", num_envs=4096)  → Isaac Sim GPU
    Robot("so100", backend="newton", num_envs=4096)  → Newton differentiable

Policy Providers:
    - lerobot_local: Direct HuggingFace inference (ACT, Pi0, SmolVLA, ...)
    - lerobot_async: gRPC to LeRobot PolicyServer
    - groot: NVIDIA GR00T N1.5/N1.6 (ZMQ or local GPU)
    - dreamgen: GR00T-Dreams IDM + VLA pipeline
    - openvla, internvla, rdt, magma, omnivla, unifolm, alpamayo,
      robobrain, cogact, dreamzero, cosmos_predict, gear_sonic, go1, mock

See Also:
    - https://github.com/strands-labs/robots
"""

import logging

logger = logging.getLogger(__name__)

__all__: list[str] = []

# --- Tier 1: Core (no external deps beyond stdlib) ---
# Policy ABC, registry, factory, resolver — never depend on strands SDK
try:
    from strands_robots.factory import Robot, list_robots  # noqa: F401
    from strands_robots.policies import (  # noqa: F401
        MockPolicy,
        Policy,
        create_policy,
        list_providers,
        register_policy,
    )
    from strands_robots.policy_resolver import resolve_policy  # noqa: F401

    __all__.extend(
        [
            "Robot",
            "Policy",
            "MockPolicy",
            "create_policy",
            "register_policy",
            "list_providers",
            "list_robots",
            "resolve_policy",
        ]
    )
except ImportError as e:
    logger.debug("Could not import core components: %s", e)

# --- Tier 2: Hardware Robot (needs strands AgentTool) ---
try:
    from strands_robots.robot import Robot as HardwareRobot  # noqa: F401

    __all__.append("HardwareRobot")
except ImportError:
    pass

# --- Tier 3: Tools (need strands @tool decorator) ---
_TOOL_IMPORTS = {
    "gr00t_inference": ("strands_robots.tools.gr00t_inference", "gr00t_inference"),
    "lerobot_calibrate": ("strands_robots.tools.lerobot_calibrate", "lerobot_calibrate"),
    "lerobot_camera": ("strands_robots.tools.lerobot_camera", "lerobot_camera"),
    "lerobot_teleoperate": ("strands_robots.tools.lerobot_teleoperate", "lerobot_teleoperate"),
    "pose_tool": ("strands_robots.tools.pose_tool", "pose_tool"),
    "serial_tool": ("strands_robots.tools.serial_tool", "serial_tool"),
}

for _export_name, (_module_path, _attr_name) in _TOOL_IMPORTS.items():
    try:
        import importlib as _importlib

        _mod = _importlib.import_module(_module_path)
        _obj = getattr(_mod, _attr_name)
        globals()[_export_name] = _obj
        __all__.append(_export_name)
    except ImportError:
        pass

# Clean up loop variables
for _name in ("_export_name", "_module_path", "_attr_name", "_mod", "_obj", "_importlib"):
    globals().pop(_name, None)
del _name

# --- Tier 4: Optional policy providers ---
try:
    from strands_robots.policies.groot import Gr00tPolicy  # noqa: F401

    __all__.append("Gr00tPolicy")
except ImportError as e:
    logger.debug("GR00T policy not available: %s", e)

try:
    from strands_robots.policies.lerobot_async import LerobotAsyncPolicy  # noqa: F401

    __all__.append("LerobotAsyncPolicy")
except ImportError:
    pass

try:
    from strands_robots.policies.dreamgen import DreamgenPolicy  # noqa: F401

    __all__.append("DreamgenPolicy")
except ImportError:
    pass
