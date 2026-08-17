"""``add_robot`` reports where the robot landed, not where it was asked to land.

``position`` is written as the attach frame's translation and MuJoCo COMPOSES
that frame with the ``pos`` the model's root body declares - it does not replace
it. A ground-bolted arm declares ``pos="0 0 0"``, so for those the offset is the
world position; a locomotion model is authored standing, so it is not. 30 of the
55 single-root robots in the built-in registry declare a non-zero root ``pos``
(the Unitree Go2 base at ``z=0.445``, the JVRC pelvis at ``z=1.4``), and for
every one of them the result echoed the requested vector back as the robot's
placement::

    add_robot(name="dog", data_config="unitree_go2", position=[0, 0, 0.4])
    # Position: [0.0, 0.0, 0.4]     <- the base compiled at z=0.845

The sibling call ``add_object(position=...)`` does place its body at exactly the
world point it is given, so one parameter name meant two different things
depending on which entity it addressed, with nothing in either result to say so.
The requested vector was the only number the caller had, and it named a place
the robot was not.

These tests pin that the reported placement is the MEASURED world position of
the robot's root body, that the request and the model's own offset are named
beside it so the difference is visible rather than having to be measured with
``get_body_state``, and that the offset is the authored root ``pos`` and not the
model's keyframe root - a distinction that matters because the two differ (the
Go2 authors its base at ``z=0.445`` and its ``home`` keyframe root at
``z=0.27``), so a repair aimed at keyframes would not move this number at all.

The three controls carry as much weight as the assertions: a robot whose root
offset is zero, a robot whose roots cannot be reduced to one pose, and
``add_object`` itself must all report exactly what they reported before, so the
change is scoped to the case that was misreporting.

Hermetic: inline MJCF written to ``tmp_path``, so no asset download. GL-free:
``mesh=False`` and no rendering.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

#: A floating-base model authored standing: its root body declares ``z=0.5``, so
#: ``position`` composes with that. The ``crouch`` keyframe puts the free root at
#: ``z=0.2`` instead, so a test can tell the authored root pose apart from the
#: keyframe root - the Go2 has exactly this shape (root ``0.445``, home ``0.27``).
_STANDING_XML = """
<mujoco model="stander">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.05 0.04"/>
      <body name="leg" pos="0 0 -0.1">
        <joint name="knee" type="hinge" axis="0 1 0" range="-1 1" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.2" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key name="crouch" qpos="0 0 0.2 1 0 0 0 0.3"/>
  </keyframe>
</mujoco>
"""

#: A ground-bolted arm: root ``pos="0 0 0"``, so the requested vector already IS
#: the world position and the report must be unchanged.
_BOLTED_XML = """
<mujoco model="bolted">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="box" size="0.04 0.04 0.02"/>
      <body name="link" pos="0 0 0.05">
        <joint name="pan" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.12 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

#: Two independent roots, each with its own non-zero pose - the shape an
#: ``aloha`` (two arm bases) or an ``rby1`` (six) attaches. A set of roots has no
#: one base pose, so there is nothing to measure and the report stays as it was.
_TWO_ROOT_XML = """
<mujoco model="pair">
  <compiler angle="radian"/>
  <worldbody>
    <body name="left" pos="-0.4 0 0.02">
      <geom type="box" size="0.04 0.04 0.02"/>
      <body name="left_link" pos="0 0 0.05">
        <joint name="left_pan" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02"/>
      </body>
    </body>
    <body name="right" pos="0.4 0 0.02">
      <geom type="box" size="0.04 0.04 0.02"/>
      <body name="right_link" pos="0 0 0.05">
        <joint name="right_pan" type="hinge" axis="0 0 1" range="-2 2" damping="1"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _spawn(tmp_path, xml: str, label: str, position: list[float], **kwargs) -> tuple[Simulation, str]:
    """Add a robot from *xml* and return the sim plus its reported ``Position:`` line."""
    model = tmp_path / f"{label}.xml"
    model.write_text(xml)
    sim = Simulation(tool_name=f"test_add_robot_placement_{label}", mesh=False)
    sim.create_world(gravity=[0, 0, -9.81])
    result = sim.add_robot(name=label, urdf_path=str(model), position=position, **kwargs)
    assert result["status"] == "success", result
    lines = [line for line in result["content"][0]["text"].splitlines() if line.startswith("Position:")]
    assert len(lines) == 1, f"premise: one Position line, got {lines}"
    return sim, lines[0]


def _measured_z(sim: Simulation, body: str) -> float:
    """The world z of *body*, read through the public state surface."""
    state = sim.get_body_state(body)
    assert state["status"] == "success", state
    return float(state["content"][1]["json"]["position"][2])


class TestTheReportedPlacementIsTheCompiledOne:
    def test_the_reported_position_is_where_the_root_body_actually_is(self, tmp_path) -> None:
        sim, line = _spawn(tmp_path, _STANDING_XML, "stander", [0.0, 0.0, 0.4])
        try:
            landed = _measured_z(sim, "stander/base")
            assert landed == pytest.approx(0.9), "premise: the model's own 0.5 root composes with the 0.4 request"
            assert "0.9" in line, (
                f"add_robot reported {line!r} for a robot whose base compiled at z={landed}: the requested "
                "0.4 names a place the robot is not, and it was the only number the caller was given"
            )
        finally:
            sim.cleanup()

    def test_the_report_names_the_request_and_the_offset_the_model_added(self, tmp_path) -> None:
        sim, line = _spawn(tmp_path, _STANDING_XML, "stander", [0.0, 0.0, 0.4])
        try:
            assert "0.4" in line, f"the request is not named, so the difference is invisible: {line!r}"
            assert "0.5" in line, f"the model's own root offset is not named: {line!r}"
        finally:
            sim.cleanup()

    def test_the_offset_is_the_authored_root_pose_not_the_keyframe_root(self, tmp_path) -> None:
        # The keyframe puts the free root at z=0.2 and the body is authored at
        # z=0.5. Only the authored pose composes with ``position``, so a repair
        # aimed at the keyframe root would leave this number untouched.
        sim, line = _spawn(tmp_path, _STANDING_XML, "stander", [0.0, 0.0, 0.4], keyframe="crouch")
        try:
            assert _measured_z(sim, "stander/base") == pytest.approx(0.9)
            assert "0.5" in line, f"the offset is the authored root pos (0.5), not the keyframe root (0.2): {line!r}"
        finally:
            sim.cleanup()


class TestTheUnaffectedReportsAreUnchanged:
    def test_a_bolted_robot_reports_exactly_what_was_requested(self, tmp_path) -> None:
        sim, line = _spawn(tmp_path, _BOLTED_XML, "bolted", [0.0, 0.5, 0.0])
        try:
            assert _measured_z(sim, "bolted/base") == pytest.approx(0.0)
            assert line == "Position: [0.0, 0.5, 0.0]", f"an arm with a zero root offset must read as before: {line!r}"
        finally:
            sim.cleanup()

    def test_a_multi_root_robot_keeps_the_single_vector_form(self, tmp_path) -> None:
        sim, line = _spawn(tmp_path, _TWO_ROOT_XML, "pair", [0.0, 0.0, 0.0])
        try:
            assert line == "Position: [0.0, 0.0, 0.0]", f"a set of roots has no one base pose to report: {line!r}"
        finally:
            sim.cleanup()

    def test_add_object_still_reports_the_world_point_it_was_given(self, tmp_path) -> None:
        # The other half of the asymmetry: an object's ``position`` IS the world
        # point, and that is what made the robot report misleading rather than
        # merely terse. Making the robot honest must not disturb it.
        sim = Simulation(tool_name="test_add_robot_placement_object", mesh=False)
        try:
            sim.create_world(gravity=[0, 0, -9.81])
            assert (
                sim.add_object("cube", shape="box", size=[0.05, 0.05, 0.05], position=[0.1, 0.2, 0.7])["status"]
                == "success"
            )
            assert _measured_z(sim, "cube") == pytest.approx(0.7)
        finally:
            sim.cleanup()
