"""Regression tests for ``start_recording(cameras=...)`` camera scoping.

By default every scene camera is recorded into the LeRobotDataset, which
silently includes the implicit ``default`` free camera and any view the trained
policy never declared. ``cameras=`` lets a caller record exactly the views the
policy expects (e.g. the three SmolVLA cameras) so the dataset schema matches
the policy's ``input_features`` and is not bloated by stray cameras.

Covers:
* default (``cameras=None``) records every camera including ``default``;
* ``cameras=[subset]`` declares exactly that subset in the dataset schema;
* an unknown camera name fails loudly (no silent drop);
* the ``_drop_unrecorded_cameras`` frame filter keeps state, drops excluded
  cameras, and is a no-op when nothing is scoped.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("lerobot")

os.environ.setdefault("MUJOCO_GL", "egl")

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
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim_with_cameras():
    from strands_robots.simulation import Simulation

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(_ROBOT_XML)

    s = Simulation()
    s.create_world()
    s.add_robot("arm", urdf_path=path)
    s.add_camera(name="cam_a", position=[0.5, 0.0, 0.3], target=[0.0, 0.0, 0.1], width=64, height=64)
    s.add_camera(name="cam_b", position=[0.0, 0.0, 0.7], target=[0.0, 0.0, 0.0], width=64, height=64)
    yield s
    s.destroy()


def _recorder_image_features(sim) -> set[str]:
    rec = sim._world._backend_state["dataset_recorder"]
    feats = rec.dataset.features
    return {k for k in feats if k.startswith("observation.images.")}


def test_default_records_every_camera_including_default(sim_with_cameras, tmp_path):
    sim = sim_with_cameras
    res = sim.start_recording(repo_id="local/scope_all", root=str(tmp_path / "all"), overwrite=True)
    assert res["status"] == "success"
    img = _recorder_image_features(sim)
    assert img == {
        "observation.images.default",
        "observation.images.cam_a",
        "observation.images.cam_b",
    }
    assert sim._world._backend_state["recording_cameras"] is None


def test_cameras_subset_scopes_dataset_schema(sim_with_cameras, tmp_path):
    sim = sim_with_cameras
    res = sim.start_recording(
        repo_id="local/scope_subset",
        root=str(tmp_path / "subset"),
        cameras=["cam_a", "cam_b"],
        overwrite=True,
    )
    assert res["status"] == "success"
    # The stray implicit ``default`` camera must NOT be in the schema.
    assert _recorder_image_features(sim) == {
        "observation.images.cam_a",
        "observation.images.cam_b",
    }
    assert sim._world._backend_state["recording_cameras"] == {"cam_a", "cam_b"}


def test_cameras_single_camera(sim_with_cameras, tmp_path):
    sim = sim_with_cameras
    res = sim.start_recording(
        repo_id="local/scope_one",
        root=str(tmp_path / "one"),
        cameras=["cam_a"],
        overwrite=True,
    )
    assert res["status"] == "success"
    assert _recorder_image_features(sim) == {"observation.images.cam_a"}


def test_unknown_camera_fails_loudly(sim_with_cameras, tmp_path):
    sim = sim_with_cameras
    res = sim.start_recording(
        repo_id="local/scope_bad",
        root=str(tmp_path / "bad"),
        cameras=["cam_a", "nope"],
        overwrite=True,
    )
    assert res["status"] == "error"
    text = res["content"][0]["text"]
    assert "nope" in text
    assert "cam_a" in text  # available cameras are listed
    # A failed start must not leave the engine half-armed.
    assert sim._world._backend_state.get("recording") is False


def test_drop_unrecorded_cameras_helper():
    from strands_robots.simulation.mujoco.simulation import _drop_unrecorded_cameras

    obs = {
        "shoulder_pan": 0.1,
        "shoulder_pan.vel": 0.0,
        "cam_a": np.zeros((8, 8, 3), dtype=np.uint8),
        "cam_b": np.zeros((8, 8, 3), dtype=np.uint8),
        "default": np.zeros((8, 8, 3), dtype=np.uint8),
    }
    # None -> unchanged (legacy default: record all).
    assert _drop_unrecorded_cameras(obs, None) is obs

    out = _drop_unrecorded_cameras(obs, {"cam_a"})
    assert set(out) == {"shoulder_pan", "shoulder_pan.vel", "cam_a"}
    # Scalars preserved, excluded cameras dropped.
    assert out["shoulder_pan"] == 0.1
    assert "cam_b" not in out
    assert "default" not in out


def test_recording_default_camera_alongside_sensors_warns(sim_with_cameras, tmp_path, caplog):
    """Record-all (``cameras=None``) must not silently sweep in ``default``.

    The implicit ``default`` overview camera that ``create_world`` adds is not a
    sensor any policy declares; recording it bloats the dataset with an
    ``observation.images.default`` view. Recording still happens (back-compat),
    but a one-time warning must surface it instead of including it silently.
    """
    sim = sim_with_cameras
    with caplog.at_level("WARNING"):
        res = sim.start_recording(
            repo_id="local/scope_warn",
            root=str(tmp_path / "warn"),
            overwrite=True,
        )
    assert res["status"] == "success"
    # Behavior unchanged: default is still recorded alongside the sensors.
    assert _recorder_image_features(sim) == {
        "observation.images.default",
        "observation.images.cam_a",
        "observation.images.cam_b",
    }
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("default" in m and "observation.images.default" in m and "cameras=" in m for m in warnings), (
        f"expected a stray-default-camera warning, got: {warnings}"
    )


def test_recording_scoped_cameras_does_not_warn(sim_with_cameras, tmp_path, caplog):
    """Scoping with ``cameras=`` drops ``default`` and must emit no stray warning."""
    sim = sim_with_cameras
    with caplog.at_level("WARNING"):
        res = sim.start_recording(
            repo_id="local/scope_quiet",
            root=str(tmp_path / "quiet"),
            cameras=["cam_a", "cam_b"],
            overwrite=True,
        )
    assert res["status"] == "success"
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert not any("observation.images.default" in m for m in warnings), (
        f"scoped recording must not warn about default, got: {warnings}"
    )


@pytest.fixture
def sim_with_namespaced_camera():
    """A sim whose sensor camera carries a namespaced MuJoCo name (``arm0/wrist_cam``).

    Robots that inject a namespace prefix produce camera names containing ``/``,
    which LeRobot cannot use as a feature key (``/`` is reserved for nested
    features), so ``start_recording`` collapses the separator to ``__`` for the
    dataset schema. This fixture reproduces that shape so the raw name and the
    schema-safe name genuinely differ.
    """
    from strands_robots.simulation import Simulation

    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(_ROBOT_XML)

    s = Simulation()
    s.create_world()
    s.add_robot("arm", urdf_path=path)
    s.add_camera(name="arm0/wrist_cam", position=[0.5, 0.0, 0.3], target=[0.0, 0.0, 0.1], width=64, height=64)
    yield s
    s.destroy()


def test_cameras_selectable_by_schema_safe_name(sim_with_namespaced_camera, tmp_path):
    """A namespaced camera can be scoped by its schema-safe (``__``) name.

    ``start_recording`` documents that ``cameras=`` names may be given in either
    the raw MuJoCo form (``arm0/wrist_cam``) or the collapsed schema-safe form
    (``arm0__wrist_cam``). This pins the schema-safe alias path: requesting the
    ``__`` form resolves to the same camera and produces the collapsed feature
    key, and the raw name is what gets stashed for the frame filter.
    """
    sim = sim_with_namespaced_camera
    res = sim.start_recording(
        repo_id="local/scope_safe_name",
        root=str(tmp_path / "safe"),
        cameras=["arm0__wrist_cam"],
        overwrite=True,
    )
    assert res["status"] == "success"
    # The schema-safe key is what lands in the dataset; the stray implicit
    # ``default`` overview camera is excluded.
    assert _recorder_image_features(sim) == {"observation.images.arm0__wrist_cam"}
    # The frame filter is keyed on the RAW MuJoCo name (what observations carry).
    assert sim._world._backend_state["recording_cameras"] == {"arm0/wrist_cam"}


def test_cameras_raw_and_schema_safe_names_are_equivalent(sim_with_namespaced_camera, tmp_path):
    """Selecting a namespaced camera by raw or schema-safe name yields the same schema."""
    sim = sim_with_namespaced_camera
    res = sim.start_recording(
        repo_id="local/scope_raw_name",
        root=str(tmp_path / "raw"),
        cameras=["arm0/wrist_cam"],
        overwrite=True,
    )
    assert res["status"] == "success"
    # Raw request collapses to the same schema-safe feature key as the ``__`` form.
    assert _recorder_image_features(sim) == {"observation.images.arm0__wrist_cam"}
    assert sim._world._backend_state["recording_cameras"] == {"arm0/wrist_cam"}
