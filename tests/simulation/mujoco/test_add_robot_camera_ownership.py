"""``add_robot`` registers the cameras of the robot it is adding, and no others.

A robot's source MJCF may declare its own cameras (a wrist cam, a chest cam).
``Simulation.add_robot`` picks those up after the attach by walking the compiled
model's camera table and recording each one in ``world.cameras`` with
``origin_robot`` set to the robot being added - the field
:func:`~strands_robots.simulation.mujoco.scene_ops.eject_robot_from_scene`
later cleans up by, so that removing a robot removes exactly its cameras.

The probe walks the MERGED model, which holds every camera in the scene: the
overview camera, cameras a caller added, and the cameras of robots already
attached. It stripped only the namespace of the robot being added now, so a
camera contributed by an EARLIER robot failed that strip, kept its qualified
name, and - because that qualified name was free as a registry key - was
registered a second time as belonging to the robot being added. One physical
camera, two entries, the second one owned by a robot that does not have it.

Three observable consequences, all pinned below:

* The registry grows an entry per already-attached robot camera on every
  subsequent ``add_robot``, so it stops agreeing with the compiled model.
* ``remove_robot`` cleans up by ``origin_robot``, so it takes the first robot's
  camera entry out with the SECOND robot, and leaves the duplicate behind when
  the first robot leaves - stranding an entry that names a camera the model no
  longer has. That is exactly the drift
  ``tests/simulation/mujoco/test_remove_object_camera_cascade.py`` pins for the
  object path, whose premise is that the robot path already gets this right.
* When two robots declare a camera with the SAME short name (two arms both with
  ``wrist``), the second robot's own camera was dropped from the registry
  entirely: the short key was taken, and the only entry the add produced was the
  duplicate of the FIRST robot's camera. The robot's camera had no owner, so no
  ``remove_robot`` would ever clean it up, and nothing recorded its dimensions.

The short alias stays first-come - it is what a single-robot caller addresses a
camera by, and the control tests below pin that a lone robot is unaffected. A
second claimant is registered under its qualified name, which is unique per
robot by construction, rather than being dropped.

GL-free: ``mesh=False`` and no rendering, so this runs without a GPU. Robots are
inline MJCF written to ``tmp_path``, so no asset download is needed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import simulation as sim_mod  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

_SIM_LOGGER = sim_mod.__name__

# A one-joint robot carrying exactly one camera, so the camera table stays small
# enough to read directly. No meshes and no <compiler> asset references, so the
# spec compiles with nothing downloaded.
_ROBOT_XML = """
<mujoco model="{model}">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link" pos="0 0 0.2">
      <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
      <geom name="body_geom" type="box" size="0.05 0.05 0.05" rgba="{rgba}"/>
      <camera name="{camera}" pos="0.3 0 0.1" xyaxes="0 -1 0 0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="10"/>
  </actuator>
</mujoco>
"""


def _write_robot(tmp_path: Path, model: str, camera: str, rgba: str = "0.8 0.2 0.2 1") -> str:
    path = tmp_path / f"{model}.xml"
    path.write_text(_ROBOT_XML.format(model=model, camera=camera, rgba=rgba))
    return str(path)


def _model_camera_names(sim: Any) -> list[str]:
    """Camera names the compiled model actually carries."""
    import mujoco as mj

    model = sim._world._model
    names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    return [n for n in names if n]


def _registry(sim: Any) -> dict[str, tuple[str, str]]:
    """``world.cameras`` as ``{key: (camera name, owning robot)}``."""
    return {key: (cam.name, cam.origin_robot) for key, cam in sim._world.cameras.items()}


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_add_robot_camera_ownership_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    yield s
    s.cleanup(policy_stop_timeout=0.5)


@pytest.fixture
def two_robots(sim, tmp_path):
    """``rover`` (camera ``front``) then ``arm`` (camera ``wrist``)."""
    assert sim.add_robot("rover", urdf_path=_write_robot(tmp_path, "rover_model", "front"))["status"] == "success"
    assert (
        sim.add_robot("arm", urdf_path=_write_robot(tmp_path, "arm_model", "wrist", "0.2 0.8 0.2 1"))["status"]
        == "success"
    )
    return sim


class TestEachCameraIsRegisteredOnceToTheRobotThatHasIt:
    def test_adding_a_second_robot_does_not_re_register_the_first_robots_camera(self, two_robots) -> None:
        registry = _registry(two_robots)
        entries_per_camera: dict[str, list[str]] = {}
        for key, (cam_name, _owner) in registry.items():
            entries_per_camera.setdefault(cam_name, []).append(key)
        duplicated = {cam: keys for cam, keys in entries_per_camera.items() if len(keys) > 1}
        assert duplicated == {}, f"one camera under several registry keys: {duplicated} (registry {registry})"

    def test_no_robot_owns_a_camera_from_another_robots_namespace(self, two_robots) -> None:
        registry = _registry(two_robots)
        misowned = {
            key: (cam_name, owner)
            for key, (cam_name, owner) in registry.items()
            if owner and not cam_name.startswith(f"{owner}/")
        }
        assert misowned == {}, f"entries owned by a robot outside their namespace: {misowned}"

    def test_the_registry_holds_one_entry_per_camera_in_the_model(self, two_robots) -> None:
        assert sorted(cam for cam, _ in _registry(two_robots).values()) == sorted(_model_camera_names(two_robots))


class TestRemovingARobotRemovesExactlyItsOwnCameras:
    def test_the_departed_robots_camera_leaves_the_registry(self, two_robots) -> None:
        assert two_robots.remove_robot("rover")["status"] == "success"
        assert [cam for cam, _ in _registry(two_robots).values() if cam.startswith("rover/")] == []

    def test_no_entry_survives_naming_a_camera_the_model_no_longer_has(self, two_robots) -> None:
        two_robots.remove_robot("rover")
        present = set(_model_camera_names(two_robots))
        stranded = {key: cam for key, (cam, _owner) in _registry(two_robots).items() if cam not in present}
        assert stranded == {}, f"registry advertises cameras the compiled model does not have: {stranded}"

    def test_the_remaining_robot_keeps_its_own_camera(self, two_robots) -> None:
        two_robots.remove_robot("rover")
        assert ("arm/wrist", "arm") in _registry(two_robots).values()


class TestASharedShortNameKeepsBothCamerasOwned:
    @pytest.fixture
    def colliding(self, sim, tmp_path):
        """Both robots declare a camera named ``front``."""
        sim.add_robot("rover", urdf_path=_write_robot(tmp_path, "rover_c", "front"))
        sim.add_robot("arm", urdf_path=_write_robot(tmp_path, "arm_c", "front", "0.2 0.8 0.2 1"))
        return sim

    def test_the_second_robots_camera_is_registered_and_owned(self, colliding) -> None:
        assert ("arm/front", "arm") in _registry(colliding).values(), _registry(colliding)

    def test_the_first_robot_keeps_the_short_alias(self, colliding) -> None:
        assert _registry(colliding)["front"] == ("rover/front", "rover")

    def test_removing_the_first_robot_keeps_the_second_robots_camera(self, colliding) -> None:
        colliding.remove_robot("rover")
        assert ("arm/front", "arm") in _registry(colliding).values()
        present = set(_model_camera_names(colliding))
        assert all(cam in present for cam, _ in _registry(colliding).values()), _registry(colliding)

    def test_the_qualified_key_is_reported_with_the_name_that_took_the_alias(self, sim, tmp_path, caplog) -> None:
        sim.add_robot("rover", urdf_path=_write_robot(tmp_path, "rover_l", "front"))
        with caplog.at_level(logging.INFO, logger=_SIM_LOGGER):
            sim.add_robot("arm", urdf_path=_write_robot(tmp_path, "arm_l", "front", "0.2 0.8 0.2 1"))
        reported = [r.getMessage() for r in caplog.records if "already taken" in r.getMessage()]
        assert len(reported) == 1, [r.getMessage() for r in caplog.records]
        for token in ("'arm'", "'arm/front'", "'front'", "'rover/front'"):
            assert token in reported[0], reported[0]


class TestTheSingleRobotAndUserCameraPathsAreUnchanged:
    """Controls: these hold both before and after the ownership fix."""

    def test_a_lone_robots_camera_is_addressed_by_its_short_name(self, sim, tmp_path) -> None:
        sim.add_robot("rover", urdf_path=_write_robot(tmp_path, "rover_solo", "front"))
        assert _registry(sim)["front"] == ("rover/front", "rover")

    def test_a_robot_add_leaves_a_user_camera_unowned(self, sim, tmp_path) -> None:
        assert sim.add_camera(name="overview", position=[0.9, -0.9, 0.6], target=[0.0, 0.0, 0.1])["status"] == "success"
        sim.add_robot("rover", urdf_path=_write_robot(tmp_path, "rover_user", "front"))
        assert _registry(sim)["overview"][1] == ""

    def test_a_robot_add_does_not_claim_the_scene_overview_camera(self, sim, tmp_path) -> None:
        sim.add_robot("rover", urdf_path=_write_robot(tmp_path, "rover_default", "front"))
        assert _registry(sim)["default"][1] == ""
