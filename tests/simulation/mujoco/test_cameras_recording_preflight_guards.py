"""Pre-flight validation contracts for the plain-MP4 camera recorders.

Both plain-MP4 recorder entry points on
:class:`strands_robots.simulation.mujoco.rendering.RenderingMixin` -
``start_cameras_recording`` (daemon-thread) and
``start_cameras_recording_synchronous`` (``(on_frame, finalize)`` closures) -
run the same pre-flight guards before touching the filesystem or spawning a
capture thread:

* an empty resolved camera set fails loudly with ``"No cameras to record."``
  rather than silently starting a recording that would only ever write empty
  MP4 files, and
* an ``output_dir`` that fails path validation (traversal / metacharacters) is
  rejected with a ``"cameras_recording: ..."`` error instead of being passed
  through to ``os.makedirs``.

These are LLM-facing tool contracts, so the guards return the structured
``{"status": "error", ...}`` shape rather than raising. Pinned here so a
regression that lets a zero-camera or traversal-carrying request slip through
is caught immediately.
"""

from __future__ import annotations

import os
import shutil
import tempfile

import pytest

pytest.importorskip("mujoco")

os.environ.setdefault("MUJOCO_GL", "glfw")

# Inline MJCF avoids a network-dependent model download and keeps the world
# deterministic: three hinge joints, matching the fixtures used by the other
# recorder tests in this package.
_ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="shoulder_lift_act" joint="shoulder_lift" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim():
    from strands_robots.simulation import Simulation

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(_ROBOT_XML)

    s = Simulation()
    s.create_world()
    s.add_robot("arm", urdf_path=path, position=[0.0, 0.0, 0.0])
    yield s
    s.destroy()
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestNoCamerasGuard:
    """An empty resolved camera set is a fail-loud error, not a silent no-op.

    Passing ``cameras=[]`` explicitly resolves to zero camera names even though
    the compiled world carries a ``default`` camera, so it exercises the guard
    without depending on the scene having no cameras at all.
    """

    def test_start_cameras_recording_empty_list_errors(self, sim):
        result = sim.start_cameras_recording(cameras=[])
        assert result["status"] == "error"
        assert result["content"][0]["text"] == "No cameras to record."
        # The guard fires before any recorder thread is registered.
        assert sim.get_cameras_recording_status()["status"] == "success"

    def test_start_cameras_recording_synchronous_empty_list_errors(self, sim):
        result = sim.start_cameras_recording_synchronous(cameras=[])
        assert result["status"] == "error"
        assert result["content"][0]["text"] == "No cameras to record."


class TestOutputDirValidation:
    """A traversal-carrying ``output_dir`` is rejected before ``os.makedirs``."""

    def test_start_cameras_recording_synchronous_rejects_traversal(self, sim):
        sim.add_camera("cam_a", position=[-0.3, -0.3, 0.4], target=[0.0, 0.0, 0.1])
        result = sim.start_cameras_recording_synchronous(cameras=["cam_a"], output_dir="../../etc/evil")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text.startswith("cameras_recording:")
        assert "output_dir" in text
        # Rejected at the pre-flight stage: no synchronous session started.
        assert sim.get_cameras_recording_status()["status"] == "success"
