"""Default Isaac scene for the 3DGS hybrid-render demo.

A real Franka Panda (loaded from Isaac's bundled USD, *not* the
procedural stick-figure -- see ``examples/libero/run_isaac.py`` for
why) plus a small red cube on the ground, and an over-the-shoulder
RTX camera. The robot + cube are the RTX foreground the compositor
z-composites over the captured-real 3DGS background.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SceneBuild:
    """Summary of what ``build_default_scene`` actually loaded."""

    robot_name: str
    robot_joint_count: int
    camera_name: str
    object_names: list[str]


# Hero camera presets (pos, target, fov_deg) framing the Franka in the 3DGS
# room. Adapted from the MuJoCo-GS demo's authored cameras (the tabletop scene
# + its skybox alignment were tuned together with those), then pulled back and
# aimed higher (target z~0.55 rather than the workspace floor) so the WHOLE arm
# fits with margin.
#
# Framing was validated against actual RTX renders (a plain-RGB framing probe,
# not just geometry math): the earlier "front" pose ([0.05,-1.25,0.75] aimed at
# z=0.40, fov 58) put the eye too close and too low, so the Franka's default
# upright pose pushed the wrist + gripper off the TOP-RIGHT of the frame. The
# pose below pulls the eye back to y=-1.95 / up to z=1.05 and aims at z=0.55
# with a narrower 48 deg fov, which keeps the full arm base->gripper in frame
# with comfortable margin (so it also stays framed across the --wave joint-0
# sweep). Eyes stay INSIDE the ~2 m captured shell (z < the ~1.6 m ceiling) so
# the splat still fills the frame; aiming higher captures the arm without
# grazing the unobserved ceiling.
CAMERA_PRESETS: "dict[str, tuple[list[float], list[float], float]]" = {
    "oblique": ([0.96, -0.93, 0.88], [0.05, 0.05, 0.40], 55.0),
    "front": ([0.05, -1.95, 1.05], [0.05, 0.05, 0.55], 48.0),
    # A high, slightly-offset eye (a perfectly vertical look-at is degenerate
    # for a +Z-up roll axis); kept under the ~1.6 m shell ceiling.
    "topdown": ([0.05, -0.61, 1.56], [0.05, 0.05, 0.30], 62.0),
}

# SO-101 (SO-ARM100) tabletop arm: a much smaller (~0.4 m) robot than the
# Franka, imported from the MuJoCo Menagerie MJCF (the URDF import doesn't
# render in RTX). The display USD holds the upright "home" pose via joint
# drives (centre ~[0,-0.12,0.12]). It renders empty at close range, so instead
# of moving in we keep the eye ~1.1 m back and use a NARROW fov (~30 deg) to
# zoom -- that enlarges the small arm to a centred hero shot while the 3DGS
# room still fills the frame.
#
# Azimuth note: the arm's base sits on the -Y (room) side and its working face
# (gripper opening + coloured servo detail) points toward the -X/-Y quadrant.
# A camera due -Y therefore looks at the arm's BACK, so "front" is authored
# from the -X,-Y side (the arm's actual front) and "oblique" from +X,-Y (a 3/4
# from the opposite side) so the two presets show genuinely different angles.
# "topdown" note: the 3DGS room was captured at eye level, so a true nadir
# floats the arm over the (downward-ray) floor and the base projects past the
# counter edge -- it can't composite cleanly. A steep angle that looks down the
# arm's length also self-occludes the base (only the gripper shows). So
# "topdown" is an ELEVATED BROADSIDE view (oblique's +X,-Y azimuth, raised to
# z~0.70): high enough to read as looking down, but it still sees the whole arm
# base->gripper grounded on the counter.
SO101_CAMERA_PRESETS: "dict[str, tuple[list[float], list[float], float]]" = {
    "oblique": ([0.8, -0.9, 0.58], [0.0, -0.12, 0.12], 34.0),
    "front": ([-0.6, -1.0, 0.46], [0.0, -0.12, 0.12], 33.0),
    "topdown": ([0.84, -0.88, 0.70], [0.0, -0.11, 0.15], 36.0),
}


def _add_lighting(sim: "object") -> None:
    """Add explicit key + dome lights to the stage.

    The digital-twin composite uses ``ground_plane=False`` (so the
    background provides the floor, not occluded by a sim ground). But
    Isaac's default lighting is part of the default-ground-plane
    subtree, so without it the robot renders as an unlit black
    silhouette. Author a distant key light + a dome fill light directly
    so lighting is independent of the (absent) floor.

    No-op if ``pxr`` / the stage aren't available.
    """
    try:
        import omni.usd  # type: ignore[import-not-found]
        from pxr import Gf, Sdf, UsdGeom, UsdLux  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        return

    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/KeyLight"))
    key.CreateIntensityAttr(1500.0)
    key.CreateAngleAttr(1.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))  # slightly warm, kitchen-ish
    UsdGeom.Xformable(key.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-50.0, 10.0, 0.0))

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/DomeLight"))
    dome.CreateIntensityAttr(450.0)
    logger.info("Added key + dome lights (ground-plane lighting unavailable with ground_plane=False)")


# Default Franka Panda USD sub-paths relative to the Isaac assets root.
# NVIDIA relocated the asset under a vendor folder in Isaac Sim 6.0, so the
# layout differs across releases. Probe the 6.0 path first (the current
# target runtime), then fall back to the legacy 4.x path. See
# strands-labs/robots-sim#110.
_FRANKA_USD_SUBPATHS: tuple[str, ...] = (
    # Isaac Sim 6.0+ (vendor folder).
    "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
    # Isaac Sim 4.x and earlier.
    "Isaac/Robots/Franka/franka.usd",
)


def _asset_exists(url: str) -> "bool | None":
    """Best-effort HEAD-probe for an asset URL.

    Returns ``True`` / ``False`` when the probe is conclusive, or
    ``None`` when it can't be determined (non-HTTP URL, e.g. an
    ``omniverse://`` Nucleus path, or a network error). A ``None`` result
    means "don't use this to rule a candidate in or out" so we degrade to
    the first candidate rather than failing on a transient hiccup.
    """
    if not url.lower().startswith(("http://", "https://")):
        return None
    import urllib.error
    import urllib.request

    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310
            return 200 <= resp.status < 400
    except urllib.error.HTTPError as exc:
        return exc.code < 400
    except Exception:  # noqa: BLE001
        return None


def _resolve_franka_usd(assets_root: str) -> str:
    """Pick the Franka USD candidate that exists under ``assets_root``.

    HEAD-probes each candidate in :data:`_FRANKA_USD_SUBPATHS` order and
    returns the first that resolves. If no probe is conclusive (e.g. a
    Nucleus ``omniverse://`` root that can't be HEAD-probed over HTTP),
    falls back to the first (6.0) candidate. Raises with an actionable
    hint only when every HTTP candidate definitively 404s.
    """
    candidates = [f"{assets_root}/{sub}" for sub in _FRANKA_USD_SUBPATHS]
    saw_definitive_miss = False
    for url in candidates:
        exists = _asset_exists(url)
        if exists is True:
            return url
        if exists is False:
            saw_definitive_miss = True
    if saw_definitive_miss:
        raise RuntimeError(
            "Default Franka USD not found under the Isaac assets root "
            f"({assets_root}); tried {candidates}. The asset layout changed "
            "between Isaac Sim 4.x and 6.0 -- pass robot_usd=... with an "
            "explicit asset path."
        )
    # Inconclusive (non-HTTP root / network hiccup): assume the current
    # 6.0 layout and let add_robot surface a clear error if it's wrong.
    return candidates[0]


def _default_franka_usd(sim: "object") -> str:
    """Resolve Isaac's bundled Franka Panda USD from the assets root.

    Reachable over HTTPS from the Omniverse CDN (no local Nucleus
    required). Same default as ``examples/libero/run_isaac.py``.

    Tries the modern ``isaacsim.storage.native`` namespace first (Isaac
    Sim 6.0 supported path) and falls back to the legacy
    ``omni.isaac.nucleus`` shim. The asset sub-path moved under a vendor
    folder in Isaac Sim 6.0, so :func:`_resolve_franka_usd` HEAD-probes
    the 6.0 and legacy 4.x layouts and returns whichever exists.
    """
    try:
        from isaacsim.storage.native import (  # type: ignore[import-not-found]
            get_assets_root_path,
        )
    except ImportError:
        from omni.isaac.nucleus import (  # type: ignore[import-not-found]
            get_assets_root_path,
        )

    root = get_assets_root_path()
    if not root:
        raise RuntimeError(
            "Could not resolve the Isaac assets root for the default Franka USD. " "Pass robot_usd=... explicitly."
        )
    return _resolve_franka_usd(root)


def build_default_scene(
    sim: "object",
    robot_usd: "str | None" = None,
    camera_name: str = "front",
    camera_position: "list[float] | None" = None,
    camera_target: "list[float] | None" = None,
    camera_width: int = 640,
    camera_height: int = 480,
    camera_fov: float = 48.0,
) -> SceneBuild:
    """Build the demo scene on a fresh ``IsaacSimulation``.

    Steps: ``create_world`` -> load a real Franka USD via
    ``add_robot(usd_path=...)`` (real Articulation, observable joints)
    -> add a red cube -> add the RTX camera the compositor renders
    from. Verifies each step's status so the caller never composites
    an empty stage.

    Parameters
    ----------
    sim : IsaacSimulation
        Fresh instance (``create_world`` is called here).
    robot_usd : str, optional
        Override the robot asset. Default: bundled Franka Panda USD.
    camera_name : str
        Name for the RTX camera (the compositor renders this).
    camera_position, camera_target : list[float], optional
        Camera placement. Defaults frame the arm over-the-shoulder.

    Returns
    -------
    SceneBuild
        What loaded (robot name + joint count, camera, objects).
    """
    # No ground plane in the digital-twin composite: the background
    # (captured-real 3DGS scene / panorama) is the visible floor, so a
    # sim ground plane would occlude it everywhere. The Franka USD is
    # fixed-base (stays up without one); the cube is static (below) so
    # it doesn't fall through. Lighting is added explicitly via
    # _add_lighting since Isaac's default light rides with the ground
    # plane we're omitting.
    cw = sim.create_world(ground_plane=False)
    if cw.get("status") != "success":
        raise RuntimeError(f"create_world failed: {cw}")
    _add_lighting(sim)

    usd = robot_usd or _default_franka_usd(sim)
    # Name "robot" is not a procedural alias, so the usd_path branch is
    # taken (real Articulation), not the procedural builder.
    rr = sim.add_robot(name="robot", usd_path=usd)
    if rr.get("status") != "success":
        raise RuntimeError(f"add_robot(usd_path={usd!r}) failed: {rr}")
    robot_info = rr.get("content", [{}])[0].get("json", {})
    joint_count = int(robot_info.get("joint_count") or 0)

    # A small red cube in front of the arm -- a second RTX foreground
    # element so the composite shows depth ordering between two objects
    # + the background. Static (is_static=True) so it doesn't fall
    # through the absent ground plane.
    obj_names: list[str] = []
    co = sim.add_object(
        name="cube",
        shape="box",
        position=[0.4, 0.0, 0.4],
        size=[0.05, 0.05, 0.05],
        color=[1.0, 0.0, 0.0],
        mass=0.1,
        is_static=True,
    )
    if co.get("status") == "success":
        obj_names.append("cube")
    else:
        logger.warning("add_object(cube) failed (non-fatal): %s", co)

    # Default camera == the "front" CAMERA_PRESETS pose. The app skips
    # re-adding "front" since this creates it, so this default must stay in
    # sync with CAMERA_PRESETS["front"]; render_demo renders it too.
    pos = camera_position or [0.05, -1.95, 1.05]
    tgt = camera_target or [0.05, 0.05, 0.55]
    ca = sim.add_camera(
        name=camera_name,
        position=pos,
        target=tgt,
        width=camera_width,
        height=camera_height,
        fov=camera_fov,
    )
    if ca.get("status") != "success":
        raise RuntimeError(f"add_camera({camera_name!r}) failed: {ca}")

    # Settle physics so the first composited frame isn't mid-drop.
    sim.step(20)

    build = SceneBuild(
        robot_name="robot",
        robot_joint_count=joint_count,
        camera_name=camera_name,
        object_names=obj_names,
    )
    logger.info(
        "Scene built: robot=%s (%d joints), camera=%s, objects=%s",
        build.robot_name,
        build.robot_joint_count,
        build.camera_name,
        build.object_names,
    )
    return build


def add_preset_cameras(
    sim: "object",
    width: int = 640,
    height: int = 480,
    presets: "dict[str, tuple[list[float], list[float]]] | None" = None,
) -> "list[str]":
    """Add the hero camera presets (``CAMERA_PRESETS``) to a built scene.

    Used by the Gradio app so the camera dropdown can switch angles
    without re-adding cameras per render. Skips any preset name already
    present. Returns the list of camera names available.
    """
    presets = presets or CAMERA_PRESETS
    added: list[str] = []
    for name, (pos, tgt, fov) in presets.items():
        if name in sim._cameras:
            added.append(name)
            continue
        r = sim.add_camera(name=name, position=list(pos), target=list(tgt), width=width, height=height, fov=float(fov))
        if r.get("status") == "success":
            added.append(name)
        else:
            logger.warning("add_camera(%s) failed: %s", name, r)
    return added
