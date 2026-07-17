"""Fault tolerance of ``Simulation.get_observation``'s per-camera render loop.

``get_observation`` returns a robot's joint state plus one image entry per scene
camera. The joint-state block is the signal a control policy actually closes the
loop on, so a single camera's render failure must never take it down. Two
failure modes are swallowed inside the per-camera loop:

* the offscreen renderer cannot be created (headless without EGL/OSMesa) - the
  camera is skipped and the observation still carries joint state;
* an individual ``render`` raises (a common cause is a camera id gone stale
  after a scene recompile) - only that camera is dropped; every healthy camera
  and the joint state are still returned, and nothing propagates out of
  ``get_observation`` to stall the control loop.

These pin that contract directly by injecting each failure at the renderer
boundary (the same stubbing approach ``test_render_all_multicamera`` uses) and
asserting the shape of the returned observation.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj  # noqa: E402

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A minimal single-hinge arm so the observation carries a known joint key.
_ARM_XML = """
<mujoco model="arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link0" pos="0 0 0.1">
      <joint name="pan" type="hinge" axis="0 0 1"/>
      <geom type="cylinder" size="0.05 0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim(tmp_path):
    s = Simulation(tool_name="obs_cam_resilience", mesh=False)
    s.create_world()
    arm = tmp_path / "arm.xml"
    arm.write_text(_ARM_XML)
    s.add_robot(name="arm", urdf_path=str(arm))
    try:
        yield s
    finally:
        s.cleanup(policy_stop_timeout=0.2)


class _FakeRenderer:
    """Stand-in offscreen renderer that fails ``update_scene`` for one camera id.

    Mirrors the two production call shapes: ``update_scene(data, camera=<id>,
    scene_option=...)`` for a named camera and ``update_scene(data,
    scene_option=...)`` for the default view.
    """

    def __init__(self, fail_camera_id: int) -> None:
        self._fail = fail_camera_id

    def update_scene(self, data, camera: int = -1, scene_option=None) -> None:
        if camera == self._fail:
            raise RuntimeError("simulated camera render failure (stale camera id)")

    def render(self) -> np.ndarray:
        return np.zeros((8, 8, 3), dtype=np.uint8)


def test_camera_render_failure_keeps_joint_state_and_isolates_bad_camera(sim, monkeypatch):
    """One camera raising in ``render`` drops only that camera; joint state and
    every healthy camera still come back, and nothing propagates out."""
    assert sim.add_camera("good_cam", position=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.0])["status"] == "success"
    assert sim.add_camera("bad_cam", position=[-0.5, 0.5, 0.5], target=[0.0, 0.0, 0.0])["status"] == "success"

    world = sim._world
    assert world is not None
    bad_id = mj.mj_name2id(world._model, mj.mjtObj.mjOBJ_CAMERA, "bad_cam")
    assert bad_id >= 0

    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: _FakeRenderer(fail_camera_id=bad_id))

    obs = sim.get_observation("arm", skip_images=False)

    # Joint state is always present - the failing camera never took it down.
    assert "pan" in obs
    assert "pan.vel" in obs
    # The healthy camera rendered; the failing one was dropped, not raised.
    assert "good_cam" in obs
    assert "bad_cam" not in obs


def test_renderer_unavailable_keeps_joint_state_without_images(sim, monkeypatch):
    """When the offscreen renderer cannot be created (``None``), cameras are
    skipped but the joint-state observation is still returned intact."""
    assert sim.add_camera("cam", position=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.0])["status"] == "success"

    monkeypatch.setattr(sim, "_get_renderer", lambda w, h: None)

    obs = sim.get_observation("arm", skip_images=False)

    assert "pan" in obs
    assert "pan.vel" in obs
    assert "cam" not in obs
