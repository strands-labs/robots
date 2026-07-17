"""Regression test: start_recording skips anonymous (unnamed) scene cameras when
deriving the LeRobotDataset schema.

MJCF models pulled from the wild routinely declare cameras with no ``name``
attribute (``<camera pos=.../>``). MuJoCo assigns those a slot in ``model.ncam``
but ``mj_id2name`` returns an empty name for them. The recording schema derivation
walks every scene camera to build the raw-name -> schema-safe-name map; an
anonymous camera must be dropped, not turned into a degenerate
``observation.images.`` feature key (a LeRobot feature name cannot be empty, and
the collapse ``name.replace("/", "__")`` would raise on a ``None`` name). Named
cameras are still recorded, with their ROS-unsafe ``/`` separator collapsed to
``__`` for the dataset schema.
"""

import os
import tempfile

import pytest

from strands_robots.simulation.mujoco.simulation import Simulation

# A single fixed-base arm carrying ONE named camera and ONE anonymous camera.
# After add_robot attaches the model under its instance namespace the named
# camera surfaces as ``arm/wrist`` (raw MuJoCo form); the anonymous camera keeps
# an empty name and must be skipped from the dataset schema.
NAMED_AND_UNNAMED_CAM_XML = """
<mujoco model="cam_mix">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link" pos="0 0 0.1">
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <camera name="wrist" pos="0 -0.3 0.2" mode="fixed"/>
      <camera pos="0.3 0 0.2" mode="fixed"/>
    </body>
  </worldbody>
  <actuator>
    <position name="j0_act" joint="j0" kp="10"/>
  </actuator>
</mujoco>
"""


def _write(xml: str) -> str:
    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    return path


@pytest.fixture
def sim():
    pytest.importorskip("lerobot")  # start_recording produces a LeRobotDataset
    s = Simulation(tool_name="test_unnamed_cam", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    try:
        s.cleanup()
    except Exception:
        # Best-effort teardown: cleanup failures must not mask the test result.
        pass


def _named_scene_cameras(sim) -> list[str]:
    """Raw MuJoCo names of every camera that actually carries a name."""
    import mujoco as mj

    model = sim._world._model
    names = []
    for i in range(model.ncam):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
        if name:
            names.append(name)
    return names


def _dataset_image_keys(sim) -> list[str]:
    rec = sim._world._backend_state["dataset_recorder"]
    return [k for k in rec.dataset.features if k.startswith("observation.images.")]


def test_unnamed_camera_excluded_from_dataset_schema(sim):
    """A scene mixing a named and an anonymous camera records cleanly: the named
    camera becomes a schema-safe image feature and the anonymous one is dropped
    (no empty ``observation.images.`` key, no crash on a ``None`` camera name)."""
    sim.add_robot("arm", urdf_path=_write(NAMED_AND_UNNAMED_CAM_XML))

    named = _named_scene_cameras(sim)
    wrist = next(n for n in named if n.endswith("wrist"))

    root = tempfile.mkdtemp(prefix="unnamed_cam_")
    res = sim.start_recording(
        repo_id="local/unnamed_cam_test",
        task="t",
        fps=20,
        root=root,
        cameras=[wrist],
    )
    assert res["status"] == "success", res

    image_keys = _dataset_image_keys(sim)
    # The named camera is recorded, with its ROS-unsafe '/' collapsed to '__'.
    expected = f"observation.images.{wrist.replace('/', '__')}"
    assert image_keys == [expected], image_keys
    # The anonymous camera produced no degenerate empty-named feature key.
    assert "observation.images." not in image_keys
    assert not any(k == "observation.images." for k in sim._world._backend_state["dataset_recorder"].dataset.features)

    sim.stop_recording()
