"""Background resolution for the isaac_gs demo -- defaults to a real 3DGS scene.

Mirrors ``mujoco_gs``'s default: the demo composites the robot against a
real 3D Gaussian Splatting scene (the ``tabletop`` preset from
MuJoCo-GS-Web) when ``gsplat`` is available. The photoreal path is the
product: when it cannot initialize (gsplat missing / CUDA rasterizer
disabled / scene download or load failure) the resolver **raises** with an
actionable install hint rather than silently demoting to the procedural
panorama (issue #2321; AGENTS.md rules 5/6 -- raise on fatal, no silent
defaults). Demo contexts that genuinely want the zero-ML-deps demotion opt
in with ``allow_fallback=True`` (``--allow-fallback`` on the CLIs).

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

# The most common GS failure is environmental and known (README.md): a plain
# `pip install gsplat` imports fine in the Isaac container but cannot
# CUDA-rasterize (no nvcc for the JIT build). The fix is a pre-built wheel.
_GSPLAT_INSTALL_HINT = (
    "Install a pre-built `gsplat` wheel matching your torch + CUDA build "
    "(e.g. `pip install --index-url https://docs.gsplat.studio/whl/pt24cu118 "
    "'gsplat==1.5.3+pt24cu118'`) -- a plain `pip install gsplat` imports fine "
    "but cannot CUDA-rasterize where the CUDA toolkit (nvcc) is absent, such "
    "as the Isaac container. See examples/isaac_gs/README.md. To demote to "
    "the procedural panorama instead, pass --allow-fallback "
    "(resolve_background(allow_fallback=True))."
)

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
    a dome light is a lighting nicety, so demoting it is safe; note this is
    the opposite posture from :func:`resolve_background`, which **raises**
    when the photoreal background itself cannot initialize, issue #2321).
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
    allow_fallback: bool = False,
):
    """Pick a ``BackgroundRenderer`` from the demo's background options.

    Precedence:

    1. ``gsplat_ply`` -> live 3DGS skybox from that ``.ply`` / ``.spz``.
    2. ``panorama`` -> ``PanoramaBackground`` from that image.
    3. default (``prefer_gs``) -> the ``gsplat_scene`` preset (or
       :data:`DEFAULT_GS_SCENE`), downloaded + skybox-aligned.

    On the GS paths (1 and 3) a failure -- ``gsplat`` not importable, the
    CUDA rasterizer disabled, a download/load error -- **raises**, because
    the caller asked for the photoreal background and must not silently get
    the procedural gradient (issue #2321). Capability and preset failures
    raise ``RuntimeError`` with the pre-built-wheel install hint; a
    caller-supplied ``gsplat_ply`` that fails to construct re-raises as-is
    (e.g. ``FileNotFoundError`` for a typo'd path), since the install hint
    is the wrong diagnosis there. Pass ``allow_fallback=True`` to restore
    the old demotion (logged warning + ``PanoramaBackground``) for demo
    contexts that prefer rendering *something* over failing; it covers
    every failure mode on both GS paths, construction included.

    Args:
        gsplat_ply: path to a user-supplied ``.ply`` / ``.spz`` 3DGS capture.
        gsplat_scene: named built-in 3DGS preset (see ``GSPLAT_SCENES``).
        panorama: path to an equirectangular panorama image.
        prefer_gs: when no explicit background is given, resolve the default
            3DGS preset (``True``) or the procedural panorama (``False``).
        allow_fallback: demote GS failures to ``PanoramaBackground`` with a
            logged warning instead of raising.

    Returns:
        A renderer satisfying ``strands_robots.rendering.BackgroundRenderer``.

    Raises:
        RuntimeError: a GS background was requested (explicitly or via the
            ``prefer_gs`` default) and could not initialize for a capability
            or preset reason (rasterizer unavailable, download/load failure),
            and ``allow_fallback`` is ``False``.
        FileNotFoundError: ``gsplat_ply`` names a path that does not exist
            (or is not a file) and ``allow_fallback`` is ``False``.
        PermissionError: ``gsplat_ply`` names a file the process cannot
            read and ``allow_fallback`` is ``False``.
    """
    from strands_robots.rendering import GsplatBackground, PanoramaBackground, gsplat_rasterizer_available

    if gsplat_ply:
        ok, reason = gsplat_rasterizer_available()
        if not ok:
            if not allow_fallback:
                raise RuntimeError(
                    f"3DGS background {gsplat_ply!r} was requested but the gsplat CUDA "
                    f"rasterizer is unavailable ({reason}). {_GSPLAT_INSTALL_HINT}"
                )
            logger.warning(
                "background: uploaded 3DGS %s requested but the gsplat CUDA rasterizer is "
                "unavailable (%s); falling back to procedural panorama (allow_fallback=True). "
                "Install a pre-built `gsplat` wheel -- see requirements.txt.",
                gsplat_ply,
                reason,
            )
            return PanoramaBackground()
        logger.info("background: live 3DGS skybox from %s", gsplat_ply)
        try:
            return GsplatBackground(ply_path=gsplat_ply, skybox=True)
        except Exception as exc:  # noqa: BLE001 - re-raised as-is unless allow_fallback opts in
            if not allow_fallback:
                # Deliberately a bare raise, not a RuntimeError wrap: the
                # install hint diagnoses a missing CUDA wheel, which is the
                # wrong remedy for a typo'd caller-supplied path --
                # `FileNotFoundError: Gaussian Splat not found: <path>` is
                # already the better message.
                raise
            logger.warning(
                "background: uploaded 3DGS %s failed to initialize (%s: %s); falling back to "
                "procedural panorama (allow_fallback=True).",
                gsplat_ply,
                type(exc).__name__,
                exc,
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
    except Exception as exc:  # noqa: BLE001 - re-raised with the install hint unless allow_fallback opts in
        if not allow_fallback:
            raise RuntimeError(
                f"the photoreal 3DGS background {scene!r} failed to initialize "
                f"({type(exc).__name__}: {exc}). {_GSPLAT_INSTALL_HINT}"
            ) from exc
        logger.warning(
            "background: 3DGS scene %r unavailable (%s: %s); falling back to procedural panorama "
            "(allow_fallback=True). Install a pre-built `gsplat` wheel (CUDA) for the real captured "
            "scene -- see requirements.txt.",
            scene,
            type(exc).__name__,
            exc,
        )
        return PanoramaBackground()
