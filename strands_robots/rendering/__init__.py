# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid rendering: photoreal backgrounds, depth-aware compositing, media utils.

The library home of the Gaussian-splat hybrid-render layer (issue #1537):

* :class:`CameraParams` -- backend-agnostic pinhole camera description,
  produced by every backend's ``SimEngine.get_camera_params``.
* :class:`BackgroundRenderer` protocol with :class:`PanoramaBackground`
  (zero ML deps) and :class:`GsplatBackground` (3D Gaussian Splatting via the
  ``sim-gs`` extra), plus the downloadable scene presets.
* :class:`HybridCompositor` -- pure-numpy per-pixel z-compare of a simulation
  foreground (``SimEngine.get_frame``) over a photoreal background, with an
  optional shadow-catcher pass (:func:`plane_depth`) that grounds the robot
  with a contact shadow on the backdrop.
* :func:`bake_environment_map` / :func:`derive_key_light` -- image-based
  lighting derived from the background scene (issue #2323), plus the
  :mod:`~strands_robots.rendering.color` helpers that pin the layer
  color-space contract.
* :func:`encode_clip` / :func:`mjpeg_frames` -- shared media utilities.
"""

from .backgrounds import (
    GSPLAT_SCENES,
    GSPLAT_SKYBOX_ALIGN,
    BackgroundRenderer,
    GsplatBackground,
    PanoramaBackground,
    bake_gsplat_panorama,
    download_gsplat_scene,
    gsplat_rasterizer_available,
    gsplat_scene_names,
    gsplat_skybox_align_for,
    gsplat_skybox_scene_names,
)
from .camera import CameraParams
from .color import linear_to_srgb, relative_luminance, srgb_to_linear
from .compositor import CompositeFrame, FrameSource, HybridCompositor, feather_mask, plane_depth
from .ibl import (
    KeyLightEstimate,
    bake_environment_map,
    derive_key_light,
    environment_map_cache_path,
    render_environment_map,
)
from .video import encode_clip, mjpeg_frames

__all__ = [
    "GSPLAT_SCENES",
    "GSPLAT_SKYBOX_ALIGN",
    "BackgroundRenderer",
    "CameraParams",
    "CompositeFrame",
    "FrameSource",
    "GsplatBackground",
    "HybridCompositor",
    "KeyLightEstimate",
    "PanoramaBackground",
    "bake_environment_map",
    "bake_gsplat_panorama",
    "derive_key_light",
    "download_gsplat_scene",
    "encode_clip",
    "environment_map_cache_path",
    "feather_mask",
    "gsplat_rasterizer_available",
    "gsplat_scene_names",
    "gsplat_skybox_align_for",
    "gsplat_skybox_scene_names",
    "linear_to_srgb",
    "mjpeg_frames",
    "plane_depth",
    "relative_luminance",
    "render_environment_map",
    "srgb_to_linear",
]
