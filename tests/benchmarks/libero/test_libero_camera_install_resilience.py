"""Resilience contract of :meth:`LiberoAdapter._install_libero_cameras`.

The adapter installs its configured LIBERO cameras (``image`` / ``wrist_image``)
onto the sim before an eval. Two failure modes must NOT abort the run:

* ``sim.add_camera(...)`` raising an exception - one bad camera cannot kill the
  whole eval, so the failure is logged at WARNING and the loop continues to the
  next camera.
* ``sim.add_camera(...)`` returning an ``{"status": "error"}`` dict whose message
  says the camera "already exists" - benign, because the scene MJCF beat the
  adapter to it; the adapter still publishes the configured render dimensions to
  ``world.cameras`` so :meth:`_get_sim_observation` reads 256x256 instead of the
  renderer's 480x640 default.

These tests pin both behaviours with a fake sim whose ``add_camera`` is under
test control.
"""

from __future__ import annotations

import logging
from typing import Any

from strands_robots.benchmarks.libero import LiberoAdapter

PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the cube")
  (:objects cube_1 - object)
  (:goal (grasped cube_1)))
"""


class _FakeWorld:
    """Minimal SimWorld stand-in: a registry-only camera store, no MjModel.

    ``_model = None`` routes :meth:`LiberoAdapter._existing_camera_names` down
    its registry-only fallback, so the set of already-present cameras is exactly
    whatever is in ``cameras`` - letting the test start from an empty registry
    and observe every install attempt.
    """

    def __init__(self) -> None:
        self.cameras: dict[str, Any] = {}
        self._model = None


class _FakeSim:
    """SimEngine stand-in with a caller-supplied ``add_camera`` behaviour."""

    def __init__(self, add_camera: Any) -> None:
        self._world = _FakeWorld()
        self.add_camera = add_camera


def _adapter() -> LiberoAdapter:
    return LiberoAdapter.from_text(
        PICK_CUBE_BDDL,
        scene_path="/tmp/libero_scene.xml",
        install_cameras=True,
        auto_generate_scene=False,
    )


def test_install_cameras_swallows_add_camera_exception(caplog):
    """A raising ``add_camera`` is logged at WARNING and never propagates, and
    the loop still attempts every configured camera - one bad camera must not
    kill the eval."""
    adapter = _adapter()
    attempted: list[str] = []

    def _raising_add_camera(*, name: str, **_kwargs: Any) -> None:
        attempted.append(name)
        raise RuntimeError(f"boom installing {name}")

    sim = _FakeSim(_raising_add_camera)

    with caplog.at_level(logging.WARNING, logger="strands_robots.benchmarks.libero.adapter"):
        result = adapter._install_libero_cameras(sim)  # type: ignore[arg-type]

    assert result is None
    # Both configured cameras attempted despite the first raising.
    assert set(attempted) == set(adapter._cameras)
    warnings = [r for r in caplog.records if "add_camera" in r.getMessage() and "raised" in r.getMessage()]
    assert len(warnings) == len(adapter._cameras)


def test_install_cameras_publishes_dims_on_already_exists(caplog):
    """When ``add_camera`` reports the camera already exists (declared by the
    scene MJCF), the adapter publishes the configured render dimensions to
    ``world.cameras`` so observations render at the trained 256x256 resolution
    rather than the renderer's 480x640 default."""
    adapter = _adapter()

    def _already_exists(*, name: str, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "error", "content": [{"text": f"camera {name!r} already exists"}]}

    sim = _FakeSim(_already_exists)

    with caplog.at_level(logging.DEBUG, logger="strands_robots.benchmarks.libero.adapter"):
        adapter._install_libero_cameras(sim)  # type: ignore[arg-type]

    # Every configured camera got a config-only entry with the trained dims.
    for cam_name, cam_kwargs in adapter._cameras.items():
        assert cam_name in sim._world.cameras
        published = sim._world.cameras[cam_name]
        assert published.width == cam_kwargs["width"]
        assert published.height == cam_kwargs["height"]
