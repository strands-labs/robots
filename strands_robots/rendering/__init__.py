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
  foreground (``SimEngine.get_frame``) over a photoreal background.
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
from .compositor import CompositeFrame, FrameSource, HybridCompositor, feather_mask
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
    "PanoramaBackground",
    "bake_gsplat_panorama",
    "download_gsplat_scene",
    "encode_clip",
    "feather_mask",
    "gsplat_rasterizer_available",
    "gsplat_scene_names",
    "gsplat_skybox_align_for",
    "gsplat_skybox_scene_names",
    "mjpeg_frames",
]
