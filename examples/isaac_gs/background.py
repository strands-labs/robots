"""Background resolution for the isaac_gs demo -- defaults to a real 3DGS scene.

Mirrors ``mujoco_gs``'s default: the demo composites the robot against a
real 3D Gaussian Splatting scene (the ``tabletop`` preset from
MuJoCo-GS-Web) when ``gsplat`` is available, and falls back to the
procedural panorama otherwise so it still runs with zero ML deps.

All the heavy lifting (preset download + cache, ``.spz`` loading,
skybox alignment, the ``gsplat`` rasterizer) is **reused verbatim**
from ``examples.mujoco_gs.backgrounds``; this module only picks which
renderer to construct from the CLI / UI options.
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


def _gsplat_rasterizer_available() -> "tuple[bool, str]":
    """Check that gsplat can actually CUDA-rasterize, not just import.

    A plain ``pip install gsplat`` is *importable* even when its CUDA
    kernels are unavailable: gsplat JIT-builds them from source on first
    use via ``nvcc``, and the Isaac Sim container ships the CUDA runtime
    (for RTX) but no CUDA *toolkit*. gsplat then disables its CUDA
    backend ("No CUDA toolkit found") and the first
    :func:`gsplat.rasterization` call raises
    ``AttributeError: 'NoneType' object has no attribute 'CameraModelType'``.

    Importing alone can't catch this, so we probe with a trivial
    one-gaussian rasterization. A broken backend then falls back to the
    procedural panorama *up front* instead of erroring on every render.
    Install a pre-built gsplat wheel (ships compiled kernels, no nvcc) to
    get the real captured scene -- see ``requirements.txt``.
    """
    try:
        import torch
        from gsplat import rasterization
    except Exception as exc:  # noqa: BLE001
        return False, f"gsplat/torch not importable ({type(exc).__name__}: {exc})"
    if not torch.cuda.is_available():
        return False, "no CUDA device available to torch"
    try:
        dev = "cuda"
        means = torch.tensor([[0.0, 0.0, 2.0]], device=dev)
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
        scales = torch.full((1, 3), 0.1, device=dev)
        opacities = torch.ones(1, device=dev)
        colors = torch.ones(1, 3, device=dev)
        viewmats = torch.eye(4, device=dev)[None]
        Ks = torch.tensor([[[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]]], device=dev)
        rasterization(means, quats, scales, opacities, colors, viewmats, Ks, width=16, height=16)
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - a disabled CUDA backend surfaces here
        return False, f"gsplat CUDA rasterizer unavailable ({type(exc).__name__}: {exc})"


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

    Returns a renderer satisfying ``mujoco_gs.backgrounds.BackgroundRenderer``.
    """
    from examples.mujoco_gs.backgrounds import GsplatBackground, PanoramaBackground

    if gsplat_ply:
        ok, reason = _gsplat_rasterizer_available()
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
        ok, reason = _gsplat_rasterizer_available()
        if not ok:
            raise RuntimeError(reason)

        from examples.mujoco_gs.backgrounds import download_gsplat_scene, gsplat_skybox_align_for

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
