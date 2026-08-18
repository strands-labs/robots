"""``actuate_robot``'s two posture flags must be booleans, not anything truthy.

``actuate_robot`` takes six parameters. Four scale a quantity - ``kp``,
``damping``, ``armature`` (and ``torquescale`` on the sibling
``attach_bodies``) - and each has been checked on a shared numeric domain,
which rejects a string and rejects ``bool`` because ``bool`` is an ``int``
subclass that would pass as a silent ``1``. The other two select a *posture*,
and both were read by truthiness after a ``bool()`` that laundered whatever
arrived into a well-typed ``True`` before it reached the spec:

* ``disable_self_collision="false"`` (also ``"no"``, ``"off"``, ``"0"``, and
  ``math.nan``) zeroed ``contype``/``conaffinity`` on *every* geom of the
  robot - measured on MuJoCo through the agent-facing router, which publishes
  the parameter as ``"type": "boolean"`` and binds it against the method
  signature without checking the value. The caller spelling the opt-out got the
  branch its own docstring warns "disables ALL collision on the robot's geoms",
  the change lives on the spec so it outlives the recompile, and undoing it
  needs a separate ``patch_scene_mjcf`` call. ``status`` was ``"success"``.
* ``gravity_compensation="no"`` reached the spec as True and set
  ``gravcomp=1`` on the robot's bodies.

``None`` and ``0.0`` took the other branch just as silently, without ever
being a declared spelling of it.

Both are now checked on :func:`~strands_robots.utils.boolean_flag_error` - the
domain the sibling ``set_joint_positions``' ``hold`` flag in this same backend
already applies, and the one ``start_recording``'s ``overwrite`` /
``push_to_hub`` were moved onto for the same reason - ahead of the recompile
and ahead of anything written to the spec, so a rejected call leaves the robot
exactly as it found it.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from strands_robots.utils import boolean_flag_error

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

from .test_actuate_robot import MINI_ARM_URDF  # noqa: E402

# Spellings a caller reaches for when opting out, plus the two values that take
# a branch without being a declared spelling of either posture. Every one is
# accepted by ``bool()``: the strings and ``nan`` as True, ``None``/``0.0`` as
# False.
NOT_BOOLEANS: tuple[Any, ...] = ("false", "no", "off", "0", "1", math.nan, None, 0.0, 1, [], "true")

POSTURE_FLAGS = ("gravity_compensation", "disable_self_collision")


@pytest.fixture
def arm_sim(tmp_path: Path):
    """A world holding an actuator-less URDF arm, ready for ``actuate_robot``."""
    sim = Simulation(tool_name="test_actuate_posture_flags", mesh=False)
    sim.create_world(gravity=[0, 0, -9.81])
    urdf = tmp_path / "mini_arm.urdf"
    urdf.write_text(MINI_ARM_URDF)
    _ok(sim.add_robot(name="arm", urdf_path=str(urdf)), "premise: the URDF arm must load")
    if int(_model(sim).nu) != 0:
        raise AssertionError("premise: a URDF arm must load actuator-less")
    yield sim
    sim.cleanup(policy_stop_timeout=0.5)


def _ok(result: dict[str, Any], what: str) -> dict[str, Any]:
    """Return *result*, raising when it is not a success.

    A call that mutates the scene must not sit inside an ``assert``: ``python
    -O`` strips the statement, and with it the setup the test is about.
    """
    if result["status"] != "success":
        raise AssertionError(f"{what}: {result}")
    return result


def _model(sim: Simulation) -> Any:
    """The compiled ``MjModel``, past the ``SimWorld | None`` on the engine."""
    assert sim._world is not None and sim._world._model is not None
    return sim._world._model


def _robot_collision(sim: Simulation) -> list[tuple[int, int]]:
    """``(contype, conaffinity)`` for every geom belonging to robot ``arm``."""
    model = _model(sim)
    pairs = []
    for geom in range(int(model.ngeom)):
        body = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])) or ""
        if body.startswith("arm/"):
            pairs.append((int(model.geom_contype[geom]), int(model.geom_conaffinity[geom])))
    if not pairs:
        raise AssertionError("premise: the arm must own at least one geom")
    return pairs


def _robot_gravcomp(sim: Simulation) -> list[float]:
    """``body_gravcomp`` for every body belonging to robot ``arm``."""
    model = _model(sim)
    values = [
        float(model.body_gravcomp[body])
        for body in range(int(model.nbody))
        if (mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body) or "").startswith("arm/")
    ]
    if not values:
        raise AssertionError("premise: the arm must own at least one body")
    return values


class TestATruthyValueDoesNotSelectAPosture:
    """The failure that motivated the guard, measured on the compiled model."""

    @pytest.mark.parametrize("value", NOT_BOOLEANS)
    def test_disable_self_collision_leaves_the_robot_colliding(self, arm_sim, value: Any) -> None:
        """A non-boolean must not disable collision on the robot's geoms.

        Pre-fix every truthy spelling reached the spec as True and left the arm
        with ``contype == conaffinity == 0`` everywhere, which is what
        ``mode="kinematic"`` grasp assist and any planner-driven rollout then
        run against: nothing on the robot collides with anything.
        """
        before = _robot_collision(arm_sim)
        result = arm_sim.actuate_robot(robot_name="arm", disable_self_collision=value)

        assert result["status"] == "error", (
            f"disable_self_collision={value!r} was accepted; the robot's geoms are now "
            f"{_robot_collision(arm_sim)} (was {before})"
        )
        assert _robot_collision(arm_sim) == before, (
            f"disable_self_collision={value!r} was refused but the robot's collision state "
            f"changed to {_robot_collision(arm_sim)}"
        )
        assert int(_model(arm_sim).nu) == 0, "a refused call must not add actuators"

    @pytest.mark.parametrize("value", NOT_BOOLEANS)
    def test_gravity_compensation_is_not_selected_by_a_string(self, arm_sim, value: Any) -> None:
        """A non-boolean must not decide whether gravity is compensated."""
        before = _robot_gravcomp(arm_sim)
        result = arm_sim.actuate_robot(robot_name="arm", gravity_compensation=value)

        assert result["status"] == "error", (
            f"gravity_compensation={value!r} was accepted; body_gravcomp is now "
            f"{_robot_gravcomp(arm_sim)} (was {before})"
        )
        assert _robot_gravcomp(arm_sim) == before, (
            f"gravity_compensation={value!r} was refused but body_gravcomp changed to {_robot_gravcomp(arm_sim)}"
        )
        assert int(_model(arm_sim).nu) == 0, "a refused call must not add actuators"

    def test_the_agent_facing_router_refuses_it_too(self, arm_sim) -> None:
        """The tool call is the reachable route, and it publishes a boolean.

        ``tool_spec.json`` declares both flags ``"type": "boolean"`` and the
        router binds an action's parameters against the method signature without
        checking values, so the schema's own declaration was the only thing
        saying the value had to be one.
        """
        before = _robot_collision(arm_sim)
        result = arm_sim(action="actuate_robot", robot_name="arm", disable_self_collision="false")

        assert result["status"] == "error", result
        assert _robot_collision(arm_sim) == before


class TestTheRefusalNamesWhatToSupply:
    @pytest.mark.parametrize("flag", POSTURE_FLAGS)
    def test_the_message_names_the_flag_and_the_method(self, arm_sim, flag: str) -> None:
        result = arm_sim.actuate_robot(robot_name="arm", **{flag: "false"})
        text = result["content"][0]["text"]

        assert flag in text, text
        assert "actuate_robot" in text, text

    @pytest.mark.parametrize("flag", POSTURE_FLAGS)
    @pytest.mark.parametrize("value", NOT_BOOLEANS)
    def test_both_flags_answer_exactly_the_shared_domain(self, arm_sim, flag: str, value: Any) -> None:
        """Keyed on the shared helper, so the domain cannot drift here alone."""
        expected = boolean_flag_error(value, flag, "actuate_robot")
        if expected is None:
            raise AssertionError(f"premise: {value!r} must be outside the shared boolean domain")

        result = arm_sim.actuate_robot(robot_name="arm", **{flag: value})

        assert result["status"] == "error"
        assert result["content"][0]["text"] == expected


class TestARealBooleanStillSelectsItsPosture:
    """Controls: the capability is unchanged for every value that was usable."""

    @pytest.mark.parametrize("flag", POSTURE_FLAGS)
    @pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
    def test_a_python_or_numpy_boolean_is_accepted(self, arm_sim, flag: str, value: Any) -> None:
        result = arm_sim.actuate_robot(robot_name="arm", **{flag: value})

        assert result["status"] == "success", result
        assert int(_model(arm_sim).nu) == 2, "both hinges must still gain a servo"

    def test_true_disables_collision_and_false_keeps_it(self, arm_sim, tmp_path: Path) -> None:
        """Both branches still reachable, and they still differ."""
        _ok(arm_sim.actuate_robot(robot_name="arm", disable_self_collision=False), "actuate with False")
        kept = _robot_collision(arm_sim)

        other = Simulation(tool_name="test_actuate_posture_flags_b", mesh=False)
        try:
            other.create_world(gravity=[0, 0, -9.81])
            urdf = tmp_path / "mini_arm.urdf"
            _ok(other.add_robot(name="arm", urdf_path=str(urdf)), "add_robot")
            _ok(other.actuate_robot(robot_name="arm", disable_self_collision=True), "actuate with True")
            disabled = _robot_collision(other)
        finally:
            other.cleanup(policy_stop_timeout=0.5)

        assert all(pair == (1, 1) for pair in kept), kept
        assert all(pair == (0, 0) for pair in disabled), disabled

    def test_true_compensates_gravity_and_false_does_not(self, arm_sim, tmp_path: Path) -> None:
        _ok(arm_sim.actuate_robot(robot_name="arm", gravity_compensation=False), "actuate with False")
        assert _robot_gravcomp(arm_sim) == pytest.approx([0.0] * len(_robot_gravcomp(arm_sim)))

        other = Simulation(tool_name="test_actuate_posture_flags_c", mesh=False)
        try:
            other.create_world(gravity=[0, 0, -9.81])
            urdf = tmp_path / "mini_arm.urdf"
            _ok(other.add_robot(name="arm", urdf_path=str(urdf)), "add_robot")
            _ok(other.actuate_robot(robot_name="arm", gravity_compensation=True), "actuate with True")
            assert _robot_gravcomp(other) == pytest.approx([1.0] * len(_robot_gravcomp(other)))
        finally:
            other.cleanup(policy_stop_timeout=0.5)

    def test_the_success_text_still_names_both_postures(self, arm_sim) -> None:
        result = arm_sim.actuate_robot(robot_name="arm", gravity_compensation=True, disable_self_collision=True)

        text = result["content"][0]["text"]
        assert "gravcomp=on" in text, text
        assert "self_collision=off" in text, text

    @pytest.mark.parametrize(
        ("param", "value"),
        [("damping", "2.0"), ("armature", "0.01"), ("kp", "100.0"), ("damping", True), ("kp", False)],
    )
    def test_the_numeric_siblings_keep_their_own_domain(self, arm_sim, param: str, value: Any) -> None:
        """A number is still a number: these four were never the defect.

        They also still reject ``bool``, which the posture domain requires -
        the two domains are inverses and must not be conflated.
        """
        result = arm_sim.actuate_robot(robot_name="arm", **{param: value})

        assert result["status"] == "error"
        assert f"'{param}'" in result["content"][0]["text"]
        assert "finite number" in result["content"][0]["text"]
