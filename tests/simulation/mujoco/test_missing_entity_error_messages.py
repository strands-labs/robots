"""Regression tests: the ``move_object`` / ``remove_object`` / ``remove_camera`` / robot-name
facade paths return an *actionable* error for an unknown entity instead of a
dead-end ``"<Kind> 'X' not found."``.

Before this change these three paths returned a bare ``"Object 'X' not found."``
or ``"Camera 'X' not found."`` with no list of what *is* available and no
close-match suggestion - forcing an agent driving the API blind into a discovery
round-trip on every typo. The camera *render*/*record* paths already listed
``Available: [...]`` and ``add_robot`` (#1299) already offered a difflib
close-match; these tests pin that the same actionable shape now covers the
remove/move-by-name paths: the message names the entity, offers a close match,
lists the available names, and points at the discovery action
(``list_objects`` / ``list_cameras`` / ``list_robots``).

The messages keep the ``"<Kind> 'X' not found."`` prefix so the consistent
error shape (T15 in ``test_agenttool_contract``) is preserved. GL-free
(``mesh=False``, no rendering) so it runs in CI without a GPU.
"""

import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_missing_entity_msgs_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    s.add_object("cube", shape="box", size=[0.03, 0.03, 0.03], position=[0.15, -0.12, 0.03], is_static=False)
    s.add_camera(name="front_cam", position=[0.5, 0.0, 0.35], target=[0.15, 0.0, 0.05])
    yield s
    s.cleanup()


def _err_text(result):
    assert result["status"] == "error", result
    return result["content"][0]["text"]


def test_move_object_unknown_is_actionable(sim):
    text = _err_text(sim.move_object("crube", position=[0.2, -0.1, 0.03]))
    assert "Object 'crube' not found" in text  # preserved prefix (T15 shape)
    assert "Did you mean: cube" in text  # close-match
    assert "cube" in text  # names the available object
    assert "list_objects" in text  # discovery surface


def test_remove_object_unknown_is_actionable(sim):
    text = _err_text(sim.remove_object("cubee"))
    assert "Object 'cubee' not found" in text
    assert "Did you mean: cube" in text
    assert "list_objects" in text


def test_remove_camera_unknown_lists_available_and_suggests(sim):
    text = _err_text(sim.remove_camera("frnt_cam"))
    assert "Camera 'frnt_cam' not found" in text
    assert "Did you mean: front_cam" in text
    assert "front_cam" in text  # names the available camera
    # Recovery hint must name the canonical action the discovery surface
    # teaches (tool_spec.json + describe() both list 'list_cameras'), not the
    # internal 'list_cameras_info' method the dispatcher aliases it to.
    assert "action='list_cameras'" in text
    assert "list_cameras_info" not in text


def test_missing_object_in_empty_scene_points_to_add_object():
    s = Simulation(tool_name="test_missing_entity_empty_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    try:
        text = _err_text(s.move_object("cube", position=[0, 0, 0.1]))
        assert "Object 'cube' not found" in text
        assert "add_object" in text  # no objects -> point at how to add one
    finally:
        s.cleanup()


def test_valid_move_and_remove_unaffected(sim):
    # No-regression guard: the happy paths still succeed and are not intercepted
    # by the new missing-entity messaging.
    assert sim.move_object("cube", position=[0.2, -0.1, 0.03])["status"] == "success"
    assert sim.remove_camera("front_cam")["status"] == "success"
    assert sim.remove_object("cube")["status"] == "success"


_ARM_XML = """<mujoco model="ded_arm">
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05"/>
      <joint name="pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2"/>
        <joint name="lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="pan_act" joint="pan" kp="50"/>
    <position name="lift_act" joint="lift" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim_with_robot(tmp_path):
    """A GL-free world holding one real robot named ``arm1`` (inline MJCF, no
    downloaded assets) so the populated ``_unknown_robot_msg`` branch (close-match
    + available-list) is exercised for real."""
    xml = tmp_path / "ded_arm.xml"
    xml.write_text(_ARM_XML)
    s = Simulation(tool_name="test_missing_robot_msgs_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    assert s.add_robot("arm1", urdf_path=str(xml), position=[0, 0, 0])["status"] == "success"
    yield s
    s.cleanup()


def test_get_robot_state_unknown_robot_is_actionable(sim_with_robot):
    # simulation.py facade path (get_robot_state).
    text = _err_text(sim_with_robot.get_robot_state("armm1"))
    assert "Robot 'armm1' not found" in text  # preserved prefix (T15 shape)
    assert "Did you mean: arm1" in text  # difflib close-match
    assert "arm1" in text  # names the available robot
    assert "list_robots" in text  # discovery surface


def test_set_joint_positions_unknown_robot_is_actionable(sim_with_robot):
    # physics.py facade path (set_joint_positions) shares the same helper.
    text = _err_text(sim_with_robot.set_joint_positions([0.0, 0.0], robot_name="armm1"))
    assert "Robot 'armm1' not found" in text
    assert "Did you mean: arm1" in text
    assert "list_robots" in text


def test_remove_robot_unknown_in_empty_scene_points_to_add_robot():
    # No robots in the world -> point at how to add one, not a dead-end.
    s = Simulation(tool_name="test_missing_robot_empty_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    try:
        text = _err_text(s.remove_robot("arm1"))
        assert "Robot 'arm1' not found" in text
        assert "add_robot" in text
    finally:
        s.cleanup()


def test_valid_robot_query_unaffected(sim_with_robot):
    # No-regression guard: a correct robot name is not intercepted.
    assert sim_with_robot.get_robot_state("arm1")["status"] == "success"


# --- physics.py Body / Site / Geom / Sensor lookups (get_body_state, get_jacobian,
# set_body_properties, apply_force, set_geom_properties, get_sensor_data) ------------
# These shared _resolve_mj_name lookups previously returned a dead-end
# "<Kind> 'X' not found." with no available-list and no close-match - the last
# holdouts of the actionable-error class the camera/object/robot paths already
# cover. They now route through ``_unknown_mj_entity_msg`` (same shape:
# preserved prefix + difflib close-match + available names, plus a
# ``list_bodies`` discovery hint for bodies).


def test_get_body_state_unknown_body_is_actionable(sim):
    # sim fixture has an object 'cube' -> body 'cube'.
    text = _err_text(sim.get_body_state("crube"))
    assert "Body 'crube' not found" in text  # preserved prefix (T15 shape)
    assert "Did you mean: cube" in text  # difflib close-match
    assert "cube" in text  # names the available body
    assert "list_bodies" in text  # discovery surface (a real action)


def test_get_jacobian_unknown_body_is_actionable(sim):
    # A different physics.py call site sharing the same helper.
    text = _err_text(sim.get_jacobian(body_name="crube"))
    assert "Body 'crube' not found" in text
    assert "Did you mean: cube" in text
    assert "list_bodies" in text


def test_set_body_properties_unknown_body_is_actionable(sim):
    text = _err_text(sim.set_body_properties("crube", mass=1.0))
    assert "Body 'crube' not found" in text
    assert "Did you mean: cube" in text
    assert "list_bodies" in text


def test_set_geom_properties_unknown_geom_is_actionable(sim):
    # add_object names the geom '<object>_geom'.
    text = _err_text(sim.set_geom_properties(geom_name="cube_geeom", color=[1, 0, 0, 1]))
    assert "Geom 'cube_geeom' not found" in text  # preserved prefix
    assert "Did you mean: cube_geom" in text  # close-match
    assert "cube_geom" in text  # names the available geom
    # geoms have no dedicated list_* action, so no discovery hint is emitted.
    assert "list_bodies" not in text


def test_get_sensor_data_no_sensors_message_preserved(sim):
    # The sim fixture has no sensors: the informative "Model has no sensors."
    # branch must be preserved (not replaced with a generic available-list).
    text = _err_text(sim.get_sensor_data(sensor_name="anything"))
    assert "Sensor 'anything' not found" in text
    assert "Model has no sensors" in text


_SITE_SENSOR_XML = """<mujoco model="probe">
  <worldbody>
    <body name="link" pos="0 0 0.3">
      <joint name="j0" type="hinge" axis="0 0 1"/>
      <geom name="pad" type="box" size="0.02 0.02 0.02"/>
      <site name="tip" pos="0 0 0.05"/>
    </body>
  </worldbody>
  <sensor>
    <framepos name="tip_pos" objtype="site" objname="tip"/>
  </sensor>
</mujoco>
"""


@pytest.fixture
def sim_with_site_sensor(tmp_path):
    """A GL-free world holding a robot named ``probe`` with a named site and a
    framepos sensor (inline MJCF), so the Site/Sensor branches of
    ``_unknown_mj_entity_msg`` are exercised on real named entities."""
    xml = tmp_path / "probe.xml"
    xml.write_text(_SITE_SENSOR_XML)
    s = Simulation(tool_name="test_missing_site_sensor_sim", mesh=False)
    s.create_world(gravity=[0, 0, -9.81])
    assert s.add_robot("probe", urdf_path=str(xml), position=[0, 0, 0])["status"] == "success"
    yield s
    s.cleanup()


def test_get_jacobian_unknown_site_lists_available(sim_with_site_sensor):
    # add_robot namespaces the site as 'probe/tip'; the available-list surfaces
    # the exact (namespaced) name the caller must use.
    text = _err_text(sim_with_site_sensor.get_jacobian(site_name="probe/tpi"))
    assert "Site 'probe/tpi' not found" in text  # preserved prefix
    assert "Did you mean: probe/tip" in text  # close-match on the real name
    assert "probe/tip" in text  # names the available site


def test_get_sensor_data_unknown_sensor_is_actionable(sim_with_site_sensor):
    text = _err_text(sim_with_site_sensor.get_sensor_data(sensor_name="tip_poss"))
    assert "Sensor 'tip_poss' not found" in text  # preserved prefix
    assert "Did you mean: probe/tip_pos" in text  # close-match (namespaced)
    assert "probe/tip_pos" in text  # names the available sensor


def test_valid_physics_queries_unaffected(sim):
    # No-regression guard: correct body/geom names are not intercepted.
    assert sim.get_body_state("cube")["status"] == "success"
    assert sim.set_geom_properties(geom_name="cube_geom", color=[0, 1, 0, 1])["status"] == "success"
