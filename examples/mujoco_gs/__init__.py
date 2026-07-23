# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo + 3D Gaussian Splatting hybrid render demo for ``strands-robots``.

Inspired by `MuJoCo-GS-Web <https://vector-wangel.github.io/MuJoCo-GS-Web/>`_,
ported to Python on top of the upstream ``strands_robots.simulation.Simulation``
AgentTool. See ``README.md`` in this directory for run instructions.
"""

from .agent import MujocoGsAgent, build
from .backgrounds import BackgroundRenderer, GsplatBackground, PanoramaBackground
from .camera_utils import CameraParams, get_camera_params, render_rgb_and_depth
from .compositor import CompositeFrame, HybridCompositor
from .scene import SCENE_DESCRIPTION, build_default_scene

__all__ = [
    "BackgroundRenderer",
    "CameraParams",
    "CompositeFrame",
    "GsplatBackground",
    "HybridCompositor",
    "MujocoGsAgent",
    "PanoramaBackground",
    "SCENE_DESCRIPTION",
    "build",
    "build_default_scene",
    "get_camera_params",
    "render_rgb_and_depth",
]
