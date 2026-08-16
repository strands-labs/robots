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
    CUDA rasterizer disabled, a download/load error -- **raises**
    ``RuntimeError`` with the pre-built-wheel install hint, because the
    caller asked for the photoreal background and must not silently get the
    procedural gradient (issue #2321). Pass ``allow_fallback=True`` to
    restore the old demotion (logged warning + ``PanoramaBackground``) for
    demo contexts that prefer rendering *something* over failing.

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
            ``prefer_gs`` default) and could not initialize, and
            ``allow_fallback`` is ``False``.
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
        return GsplatBackground(ply_path=gsplat_ply, skybox=True)

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
