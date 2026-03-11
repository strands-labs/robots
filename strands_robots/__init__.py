"""Strands Robots — Robot control, simulation, and training.

Control robots with natural language through Strands Agents. Unified interface
across NVIDIA GR00T, Isaac Sim/Lab, HuggingFace LeRobot, and MuJoCo with
18 policy providers and 35 bundled robot models.

Quick Start:
    >>> from strands_robots import Robot
    >>> sim = Robot("so100")  # MuJoCo sim (auto-detected)

    >>> from strands import Agent
    >>> agent = Agent(tools=[sim])
    >>> agent("Pick up the red block with the mock policy")

Policy Providers:
    - lerobot_local: Direct HuggingFace inference (ACT, Pi0, SmolVLA, ...)
    - lerobot_async: gRPC to LeRobot PolicyServer
    - groot: NVIDIA GR00T N1.5/N1.6 (ZMQ or local GPU)
    - dreamgen: GR00T-Dreams IDM + VLA pipeline
    - dreamzero: DreamZero world action model (WebSocket)
    - gear_sonic: NVIDIA GEAR-SONIC humanoid control (135Hz ONNX)
    - cosmos_predict: NVIDIA Cosmos world model policy
    - mock: Sinusoidal actions for testing

See Also:
    - https://github.com/strands-labs/robots
    - https://strands-labs.github.io/robots/
"""

import logging

logger = logging.getLogger(__name__)

__all__: list[str] = []

# --- Tier 1: Core (no external deps beyond stdlib) ---
# Policy ABC, registry, factory, resolver — never depend on strands SDK
try:
    from strands_robots.factory import Robot, list_robots
    from strands_robots.policies import (
        MockPolicy,
        Policy,
        create_policy,
        list_providers,
        register_policy,
    )
    from strands_robots.policy_resolver import resolve_policy

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
    from strands_robots.robot import Robot as HardwareRobot

    __all__.append("HardwareRobot")
except (ImportError, AttributeError, OSError):
    pass

# --- Tier 3: Tools (need strands @tool decorator) ---
# Each tool imported individually so one failure doesn't cascade
_TOOL_IMPORTS = {
    "gr00t_inference": ("strands_robots.tools.gr00t_inference", "gr00t_inference"),
    "lerobot_camera": ("strands_robots.tools.lerobot_camera", "lerobot_camera"),
    "pose_tool": ("strands_robots.tools.pose_tool", "pose_tool"),
    "reachy_mini_tool": ("strands_robots.tools.reachy_mini_tool", "reachy_mini"),
    "serial_tool": ("strands_robots.tools.serial_tool", "serial_tool"),
    "download_assets": ("strands_robots.assets.download", "download_assets"),
    "teleoperator": ("strands_robots.tools.teleoperator", "teleoperator"),
    "lerobot_dataset": ("strands_robots.tools.lerobot_dataset", "lerobot_dataset"),
    "newton_sim": ("strands_robots.tools.newton_sim", "newton_sim"),
    "marble_tool": ("strands_robots.tools.marble_tool", "marble_tool"),
    "stream": ("strands_robots.tools.stream", "stream"),
    "stereo_depth": ("strands_robots.tools.stereo_depth", "stereo_depth"),
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

# Clean up loop variables from module namespace
for _name in (
    "_export_name",
    "_module_path",
    "_attr_name",
    "_mod",
    "_obj",
    "_importlib",
):
    globals().pop(_name, None)
del _name

# --- Tier 4: Optional heavy integrations ---

# MujocoBackend (MuJoCo — optional dependency)
try:
    from strands_robots.mujoco import MujocoBackend

    __all__.append("MujocoBackend")
except (ImportError, AttributeError, OSError):
    pass

# Gymnasium env wrapper (optional)
try:
    from strands_robots.envs import StrandsSimEnv

    __all__.append("StrandsSimEnv")
except (ImportError, AttributeError, OSError):
    pass

# Isaac Sim / Isaac Lab integration (optional — requires NVIDIA GPU)
try:
    from strands_robots.isaac import (
        IsaacGymEnv,
        IsaacLabEnv,
        IsaacLabTrainer,
        IsaacLabTrainerConfig,
        IsaacSimBackend,
        convert_mjcf_to_usd,
        convert_usd_to_mjcf,
        create_isaac_env,
        list_isaac_tasks,
    )

    __all__.extend(
        [
            "IsaacSimBackend",
            "IsaacGymEnv",
            "IsaacLabEnv",
            "IsaacLabTrainer",
            "IsaacLabTrainerConfig",
            "create_isaac_env",
            "list_isaac_tasks",
            "convert_mjcf_to_usd",
            "convert_usd_to_mjcf",
        ]
    )
except (ImportError, AttributeError, OSError):
    pass

# Processor pipeline bridge (optional)
try:
    from strands_robots.processor import (
        ProcessedPolicy,
        ProcessorBridge,
        create_processor_bridge,
    )

    __all__.extend(["ProcessorBridge", "ProcessedPolicy", "create_processor_bridge"])
except (ImportError, AttributeError, OSError):
    pass

# Kinematics (optional)
try:
    from strands_robots.kinematics import (
        Kinematics,
        MuJoCoKinematics,
        ONNXKinematics,
        create_kinematics,
    )

    __all__.extend(
        ["create_kinematics", "MuJoCoKinematics", "ONNXKinematics", "Kinematics"]
    )
except (ImportError, AttributeError, OSError):
    pass

# Video encoding (optional)
try:
    from strands_robots.video import (
        VideoEncoder,
        encode_frames,
        get_video_info,
    )

    __all__.extend(["encode_frames", "VideoEncoder", "get_video_info"])
except (ImportError, AttributeError, OSError):
    pass

# Motion library (optional)
try:
    from strands_robots.motion_library import Motion, MotionLibrary

    __all__.extend(["MotionLibrary", "Motion"])
except (ImportError, AttributeError, OSError):
    pass

# Dataset recorder (LeRobotDataset bridge — optional)
try:
    from strands_robots.dataset_recorder import DatasetRecorder

    __all__.append("DatasetRecorder")
except (ImportError, AttributeError, OSError):
    pass

# Recording pipeline (teleop + policy → LeRobotDataset)
try:
    from strands_robots.record import (
        EpisodeStats,
        RecordMode,
        RecordSession,
    )

    __all__.extend(["RecordSession", "RecordMode", "EpisodeStats"])
except (ImportError, AttributeError, OSError):
    pass

# Live recording visualizer
try:
    from strands_robots.visualizer import (
        RecordingStats,
        RecordingVisualizer,
    )

    __all__.extend(["RecordingVisualizer", "RecordingStats"])
except (ImportError, AttributeError, OSError):
    pass

# Training abstraction
try:
    from strands_robots.training import (
        CosmosTrainer,
        CosmosTransferTrainer,
        TrainConfig,
        Trainer,
        create_trainer,
        evaluate,
    )

    __all__.extend(
        [
            "Trainer",
            "TrainConfig",
            "create_trainer",
            "evaluate",
            "CosmosTrainer",
            "CosmosTransferTrainer",
        ]
    )
except (ImportError, AttributeError, OSError):
    pass

# DreamGen pipeline
try:
    from strands_robots.dreamgen import (
        DreamGenConfig,
        DreamGenPipeline,
        NeuralTrajectory,
    )

    __all__.extend(["DreamGenPipeline", "DreamGenConfig", "NeuralTrajectory"])
except (ImportError, AttributeError, OSError):
    pass

# Newton Physics backend (GPU-accelerated, differentiable)
try:
    from strands_robots.newton import NewtonBackend, NewtonConfig

    __all__.extend(["NewtonBackend", "NewtonConfig"])
except (ImportError, AttributeError, OSError):
    pass

# Newton Gymnasium wrapper (GPU-parallel RL environments)
try:
    from strands_robots.newton.newton_gym_env import NewtonGymEnv

    __all__.append("NewtonGymEnv")
except (ImportError, AttributeError, OSError):
    pass

# RL Training (PPO/SAC via stable-baselines3)
try:
    from strands_robots.rl_trainer import (
        RewardFunction,
        RLConfig,
        SB3Trainer,
        create_rl_trainer,
    )

    __all__.extend(["RLConfig", "SB3Trainer", "RewardFunction", "create_rl_trainer"])
except (ImportError, AttributeError, OSError):
    pass

# Cosmos Transfer 2.5 (sim-to-real visual augmentation)
try:
    from strands_robots.cosmos_transfer import (
        CosmosTransferConfig,
        CosmosTransferPipeline,
    )

    __all__.extend(["CosmosTransferPipeline", "CosmosTransferConfig"])
except (ImportError, AttributeError, OSError):
    pass

# Marble 3D World Generation (World Labs — diverse training environments)
try:
    from strands_robots.marble import (
        MARBLE_PRESETS,
        MarbleConfig,
        MarblePipeline,
        MarbleScene,
    )

    __all__.extend(["MarblePipeline", "MarbleConfig", "MarbleScene", "MARBLE_PRESETS"])
except (ImportError, AttributeError, OSError):
    pass



# LeIsaac × LeRobot EnvHub (Lightwheel AI — IsaacSim environments)
try:
    from strands_robots.leisaac import (
        LEISAAC_TASKS,
        LeIsaacEnv,
        create_leisaac_env,
        format_task_table,
        leisaac_env,
    )
    from strands_robots.leisaac import (
        list_tasks as list_leisaac_tasks,
    )

    __all__.extend(
        [
            "LeIsaacEnv",
            "create_leisaac_env",
            "leisaac_env",
            "list_leisaac_tasks",
            "LEISAAC_TASKS",
        ]
    )
except (ImportError, AttributeError, OSError):
    pass

# Optional policy providers
try:
    from strands_robots.policies.groot import Gr00tPolicy

    __all__.append("Gr00tPolicy")
except ImportError as e:
    logger.debug("GR00T policy not available: %s", e)

try:
    from strands_robots.policies.lerobot_async import LerobotAsyncPolicy

    __all__.append("LerobotAsyncPolicy")
except (ImportError, AttributeError, OSError):
    pass

try:
    from strands_robots.policies.dreamgen import DreamgenPolicy

    __all__.append("DreamgenPolicy")
except (ImportError, AttributeError, OSError):
    pass

# Zenoh Robot Mesh (peer-to-peer — every Robot is a peer by default)
try:
    from strands_robots.tools.robot_mesh import robot_mesh
    from strands_robots.zenoh_mesh import (
        Mesh,
        PeerInfo,
        get_peer,
        get_peers,
    )

    __all__.extend(["Mesh", "get_peers", "get_peer", "PeerInfo", "robot_mesh"])
except (ImportError, AttributeError, OSError):
    pass

# Stereo depth estimation (Fast-FoundationStereo — NVIDIA GPU)
try:
    from strands_robots.stereo import (
        StereoConfig,
        StereoDepthPipeline,
        StereoResult,
        estimate_depth,
    )

    __all__.extend(
        ["StereoDepthPipeline", "StereoConfig", "StereoResult", "estimate_depth"]
    )
except (ImportError, AttributeError, OSError):
    pass

# Telemetry streaming (no heavy deps — stdlib only)
try:
    from strands_robots.telemetry import (
        BatchConfig,
        EventCategory,
        StreamTier,
        TelemetryEvent,
        TelemetryStream,
    )

    __all__.extend(
        [
            "TelemetryStream",
            "TelemetryEvent",
            "EventCategory",
            "StreamTier",
            "BatchConfig",
        ]
    )
except (ImportError, AttributeError, OSError):
    pass
