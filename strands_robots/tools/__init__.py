#!/usr/bin/env python3
"""
Strands Robotics Tools

Collection of specialized tools for robot control, camera management,
teleoperation, inference services, and serial communication.

All tools are lazy-loaded to avoid pulling in numpy, pyserial, psutil, etc.
at ``import strands_robots.tools`` time.
"""

import importlib as _importlib

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "create_judge_agent": (".episode_judge", "create_judge_agent"),
    "download_assets": (".download_assets", "download_assets"),
    "gr00t_inference": (".gr00t_inference", "gr00t_inference"),
    "harness_memory": (".harness_memory", "harness_memory"),
    "load_episode": (".episode_judge", "load_episode"),
    "lerobot_calibrate": (".lerobot_calibrate", "lerobot_calibrate"),
    "lerobot_camera": (".lerobot_camera", "lerobot_camera"),
    "lerobot_teleoperate": (".lerobot_teleoperate", "lerobot_teleoperate"),
    "lerobot_train": (".lerobot_train", "lerobot_train"),
    "read_predicate_verdict": (".episode_judge", "read_predicate_verdict"),
    "run_policy": (".run_policy", "run_policy"),
    "pose_tool": (".pose_tool", "pose_tool"),
    "robot_mesh": (".robot_mesh", "robot_mesh"),
    "sample_frames": (".episode_judge", "sample_frames"),
    "serial_tool": (".serial_tool", "serial_tool"),
    "train_policy": (".train_policy", "train_policy"),
    "use_lerobot": (".use_lerobot", "use_lerobot"),
    "use_ros": (".use_ros", "use_ros"),
    "use_rosbridge": (".use_rosbridge", "use_rosbridge"),
    "use_rtps": (".use_rtps", "use_rtps"),
    "write_label": (".episode_judge", "write_label"),
}

__all__ = list(_LAZY_IMPORTS.keys())


def __getattr__(name: str):  # noqa: N807
    if name in _LAZY_IMPORTS:
        rel_module, attr_name = _LAZY_IMPORTS[name]
        module = _importlib.import_module(rel_module, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
