"""Background resolution for the isaac_gs demo -- defaults to a real 3DGS scene.

Mirrors ``mujoco_gs``'s default: the demo composites the robot against a
real 3D Gaussian Splatting scene (the ``tabletop`` preset from
MuJoCo-GS-Web) when ``gsplat`` is available, and falls back to the
procedural panorama otherwise so it still runs with zero ML deps.

All the heavy lifting (preset download + cache, ``.spz`` loading,
skybox alignment, the ``gsplat`` rasterizer, the CUDA-rasterizer
capability probe) lives in ``strands_robots.rendering`` (issue #1537);
this module only picks which renderer to construct from the CLI / UI
options.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("isaac_gs.background")

# Default 3DGS scene: MuJoCo-GS-Web's purpose-built tabletop room. Open
# floor, clean from every angle, has curated skybox alignment metadata
# (``GSPLAT_SKYBOX_ALIGN["tabletop"]``) so it sits behind the robot
# correctly without per-camera tuning.
DEFAULT_GS_SCENE = "tabletop (indoor room)"

# Where the IBL environment map is baked from: the world frame is authored so
# the robot's support surface is z=0 with the arm at the origin, so a point
# ~0.4 m above the base (mid-arm height, inside the captured room shell) sees
# the environment the way the robot does.
ENV_BAKE_ORIGIN = (0.0, 0.0, 0.4)


def resolve_ibl_env_map(
    background: object,
    gsplat_ply: Optional[str] = None,
    gsplat_scene: Optional[str] = None,
    panorama: Optional[str] = None,
) -> Optional[str]:
    """Resolve the equirect environment map that should light the robot.

    Companion to :func:`resolve_background` (issue #2323, stage 1): given the
    background it returned (and the same CLI options), produce the image for
    the scene's dome light:

    1. A live 3DGS background -> bake
       (:func:`strands_robots.rendering.bake_environment_map`) from
       :data:`ENV_BAKE_ORIGIN` through the background's own aligned
       ``render``, cached next to the scene file with the bake geometry in
       the name.
    2. A user panorama -> the panorama image itself (it already *is* an
       equirect environment map in the same direction convention).
    3. The procedural panorama fallback -> ``None`` (a synthetic gradient is
       no more the robot's room than the hardcoded lights; keep those).

    Returns the image path, or ``None`` when the demo should keep its
    default lighting rig (any bake failure logs a warning and falls back --
    same always-renders-something posture as :func:`resolve_background`).
    """
    from strands_robots.rendering import (
        GsplatBackground,
        bake_environment_map,
        download_gsplat_scene,
        environment_map_cache_path,
    )

    if isinstance(background, GsplatBackground):
        try:
            scene_file = gsplat_ply or str(download_gsplat_scene(gsplat_scene or DEFAULT_GS_SCENE))
            out = environment_map_cache_path(scene_file, origin_world=ENV_BAKE_ORIGIN)
            path = bake_environment_map(background, out, origin_world=ENV_BAKE_ORIGIN)
            logger.info("IBL: environment map %s (baked from %s)", path, scene_file)
            return str(path)
        except Exception as exc:  # noqa: BLE001 - any failure falls back to the default lights
            logger.warning(
                "IBL: environment-map bake failed (%s: %s); keeping the default lighting rig.",
                type(exc).__name__,
                exc,
            )
            return None
    if panorama:
        logger.info("IBL: using the panorama image %s as the dome environment map", panorama)
        return panorama
    return None


def resolve_background(
    gsplat_ply: Optional[str] = None,
    gsplat_scene: Optional[str] = None,
    panorama: Optional[str] = None,
    prefer_gs: bool = True,
):
    """Pick a ``BackgroundRenderer`` from the demo's background options.

    Precedence:

    1. ``gsplat_ply`` -> live 3DGS skybox from that ``.ply`` / ``.spz``.
    2. ``panorama`` -> ``PanoramaBackground`` from that image.
    3. default (``prefer_gs``) -> the ``gsplat_scene`` preset (or
       :data:`DEFAULT_GS_SCENE`), downloaded + skybox-aligned. If
       ``gsplat`` isn't importable (no CUDA build) or the scene fails to
       load, **fall back to the procedural panorama** with a logged
       note -- so the demo always renders something.

    Returns a renderer satisfying ``strands_robots.rendering.BackgroundRenderer``.
    """
    from strands_robots.rendering import GsplatBackground, PanoramaBackground, gsplat_rasterizer_available

    if gsplat_ply:
        ok, reason = gsplat_rasterizer_available()
        if ok:
            logger.info("background: live 3DGS skybox from %s", gsplat_ply)
            return GsplatBackground(ply_path=gsplat_ply, skybox=True)
        logger.warning(
            "background: uploaded 3DGS %s requested but the gsplat CUDA rasterizer is "
            "unavailable (%s); falling back to procedural panorama. Install a pre-built "
            "`gsplat` wheel -- see requirements.txt.",
            gsplat_ply,
            reason,
        )
        return PanoramaBackground()

    if panorama:
        logger.info("background: panorama image %s", panorama)
        return PanoramaBackground(image_path=panorama)

    if not prefer_gs:
        return PanoramaBackground()

    scene = gsplat_scene or DEFAULT_GS_SCENE
    try:
        ok, reason = gsplat_rasterizer_available()
        if not ok:
            raise RuntimeError(reason)

        from strands_robots.rendering import download_gsplat_scene, gsplat_skybox_align_for

        logger.info("background: downloading + loading default 3DGS scene %r ...", scene)
        ply = download_gsplat_scene(scene)
        align = gsplat_skybox_align_for(scene)
        bg = GsplatBackground(ply_path=str(ply), skybox=True, **align)
        logger.info("background: live 3DGS skybox %r%s", scene, "" if align else " (uncurated alignment)")
        return bg
    except Exception as exc:  # noqa: BLE001 - any failure falls back to panorama
        logger.warning(
            "background: 3DGS scene %r unavailable (%s: %s); falling back to procedural panorama. "
            "Install a pre-built `gsplat` wheel (CUDA) for the real captured scene -- see requirements.txt.",
            scene,
            type(exc).__name__,
            exc,
        )
        return PanoramaBackground()
