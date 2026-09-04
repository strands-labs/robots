# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cached background render belongs to the camera it was rendered for.

:meth:`HybridCompositor.render` renders the photoreal backdrop once per camera
and reuses it while only the robot moves, so the cache key has to identify the
camera completely: ``BackgroundRenderer.render(cam)`` reads nothing but ``cam``,
and every :class:`CameraParams` field is an input to it. The clip planes are the
easy ones to overlook and the ones a backend moves without touching anything
else -- MuJoCo derives both from ``model.stat.extent``, which the compiler
recomputes from the scene bounds, so any scene change (``add_object``,
``attach_bodies``, ``load_scene``) moves ``znear``/``zfar`` while a fixed named
camera keeps its pose, its intrinsics and its image size. The panorama backdrop
fills its whole depth buffer with ``cam.zfar`` and the gsplat one hands both
planes to the rasterizer, so serving the previous entry composites the scene
against a far plane the camera no longer has.

The field table below is asserted to cover the dataclass, so a field added to
:class:`CameraParams` later is graded here on arrival rather than silently
widening what one cache entry stands for.
"""

from dataclasses import fields, replace

import numpy as np
import pytest

from strands_robots.rendering import CameraParams, HybridCompositor

W, H = 16, 12
# The measured MuJoCo pair: a tabletop world (``stat.extent`` 1.0) gaining one
# object 6 m away recompiles to ``extent`` 6.673205, and the clip planes are
# ``extent * model.vis.map.{znear,zfar}``. The camera's pose, K and size do not
# move with it.
NEAR_BEFORE, FAR_BEFORE = 0.01, 50.0
NEAR_AFTER, FAR_AFTER = 0.066732, 333.6603
# Foreground geometry outside the first far plane and inside the second, i.e.
# the depths whose composite verdict the two cameras disagree about.
FAR_GEOMETRY_M = 120.0
NEAR_GEOMETRY_M = 1.0


def _camera(**overrides) -> CameraParams:
    """A pinhole camera at ``W x H``, with ``overrides`` applied."""
    fy = 0.5 * H / np.tan(np.deg2rad(45.0) / 2.0)
    base = CameraParams(
        K=np.array([[fy, 0.0, W / 2.0], [0.0, fy, H / 2.0], [0.0, 0.0, 1.0]]),
        T_world_cam=np.eye(4),
        width=W,
        height=H,
        znear=NEAR_BEFORE,
        zfar=FAR_BEFORE,
    )
    return replace(base, **overrides) if overrides else base


def _moved_pose() -> np.ndarray:
    moved = np.eye(4)
    moved[0, 3] = 0.5  # half a metre along +x
    return moved


def _widened_intrinsics() -> np.ndarray:
    K = _camera().K.copy()
    K[0, 0] *= 1.5  # a different horizontal focal length
    return K


# One differing value per CameraParams field. Two cameras differing only in the
# named field are different cameras, so they cannot share a background entry.
DIFFERING_VALUE = {
    "K": _widened_intrinsics(),
    "T_world_cam": _moved_pose(),
    "width": W + 4,
    "height": H + 2,
    "znear": NEAR_AFTER,
    "zfar": FAR_AFTER,
}


class CountingBackground:
    """Backdrop at the camera's far plane, counting renders (as the panorama does)."""

    name = "counting-bg"

    def __init__(self) -> None:
        self.render_calls = 0

    def render(self, cam: CameraParams) -> tuple[np.ndarray, np.ndarray]:
        self.render_calls += 1
        rgb = np.zeros((cam.height, cam.width, 3), dtype=np.uint8)
        rgb[..., 2] = 255  # blue backdrop
        return rgb, np.full((cam.height, cam.width), cam.zfar, dtype=np.float32)


class ProgrammableSim:
    """Frame source whose camera the test replaces, as a recompile would."""

    def __init__(self, cam: CameraParams) -> None:
        self.cam = cam

    def get_camera_params(self, camera_name="default", width=None, height=None) -> CameraParams:
        return self.cam

    def get_frame(self, camera_name="default", width=None, height=None):
        """White geometry: a near patch, plus a patch beyond the first far plane."""
        cam = self.cam
        rgb = np.full((cam.height, cam.width, 3), 255, dtype=np.uint8)
        depth = np.full((cam.height, cam.width), cam.zfar, dtype=np.float32)
        depth[1:3, 1:4] = NEAR_GEOMETRY_M
        depth[6:9, 6:10] = FAR_GEOMETRY_M
        return rgb, depth


def _compositor(cam: CameraParams) -> tuple[HybridCompositor, CountingBackground, ProgrammableSim]:
    bg = CountingBackground()
    sim = ProgrammableSim(cam)
    return HybridCompositor(sim, background=bg, feather_pixels=0), bg, sim


def test_the_differing_value_table_covers_every_camera_field() -> None:
    """A field added to CameraParams has to be given a differing value here."""
    assert {field.name for field in fields(CameraParams)} == set(DIFFERING_VALUE)


@pytest.mark.parametrize("field_name", sorted(DIFFERING_VALUE))
def test_a_camera_differing_in_one_field_is_not_served_the_other_camera_s_background(field_name: str) -> None:
    comp, bg, sim = _compositor(_camera())
    comp.render("cam")
    sim.cam = _camera(**{field_name: DIFFERING_VALUE[field_name]})
    comp.render("cam")
    assert bg.render_calls == 2, (
        f"the background rendered for a camera with {field_name}="
        f"{getattr(_camera(), field_name)!r} was served for one with "
        f"{field_name}={DIFFERING_VALUE[field_name]!r}"
    )


def test_geometry_inside_the_new_far_plane_is_not_composited_away_by_the_old_one() -> None:
    """The scene-change case end to end: the far patch is the disagreement."""
    comp, bg, sim = _compositor(_camera())
    before = comp.render("cam")
    # Outside the first camera's frustum, so the backdrop legitimately owns it.
    assert not before.foreground_mask[7, 7]
    assert before.foreground_mask[2, 2]

    # The recompile: both clip planes move, nothing else does.
    sim.cam = _camera(znear=NEAR_AFTER, zfar=FAR_AFTER)
    after = comp.render("cam")

    assert bg.render_calls == 2, "the backdrop was served from the pre-recompile camera's entry"
    assert after.foreground_mask[7, 7], (
        f"geometry at {FAR_GEOMETRY_M} m is inside the camera's far plane "
        f"({FAR_AFTER} m) but lost the depth test to a backdrop parked at the "
        f"previous far plane ({FAR_BEFORE} m)"
    )
    assert tuple(after.rgb[7, 7]) == (255, 255, 255)
    assert after.foreground_mask[2, 2], "the near patch is unaffected by either far plane"


def test_an_unchanged_camera_is_still_served_from_the_cache() -> None:
    """The caching this key exists for: a still camera renders the backdrop once."""
    comp, bg, _sim = _compositor(_camera())
    for _ in range(3):
        comp.render("cam")
    assert bg.render_calls == 1
