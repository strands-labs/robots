# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Photoreal background renderers -- now provided by the library.

The generic background layer this example pioneered (the
:class:`BackgroundRenderer` protocol, the panorama + 3DGS renderers, the
downloadable scene presets and their skybox alignment metadata) moved to
:mod:`strands_robots.rendering` (issue #1537) so the Isaac hybrid demo and
future backends share one implementation. This module re-exports the public
names so the example's imports keep reading naturally; new code should import
from ``strands_robots.rendering`` directly.
"""

from __future__ import annotations

from strands_robots.rendering import (
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

__all__ = [
    "GSPLAT_SCENES",
    "GSPLAT_SKYBOX_ALIGN",
    "BackgroundRenderer",
    "GsplatBackground",
    "PanoramaBackground",
    "bake_gsplat_panorama",
    "download_gsplat_scene",
    "gsplat_rasterizer_available",
    "gsplat_scene_names",
    "gsplat_skybox_align_for",
    "gsplat_skybox_scene_names",
]
