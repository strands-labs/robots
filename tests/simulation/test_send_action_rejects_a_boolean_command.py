"""A boolean is not an actuator command, on either accepted action shape.

``bool`` is an ``int`` subclass and ``numpy.bool_`` coerces identically, so
``send_action``'s scalar-coercion check admitted both and they reached the
actuator as a silent ``1.0``/``0.0``. That is not one command: 1.0 is a
1-radian target on a joint-position drive, a full-travel command on a
normalized or tendon drive, and an out-of-range value that is silently clamped
where ``ctrlrange`` excludes 1 - so the same ``True`` commands a different pose
on every actuator. A boolean is the conventional binary-gripper action, so the
value arrives at this surface routinely rather than as a typo.

The teleop wire validator in :mod:`strands_robots.mesh.security` already refuses
a boolean so it cannot "masquerade as a 1.0/0.0 command", and it applies frames
through ``send_action`` - so the remote surface was held to a stricter domain
than the local call it delegates to. These tests pin the local call to the same
rule, on both accepted action shapes, and pin the numeric spellings that must
keep working so the guard cannot over-reach.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.base import (
    SimEngine,
    _unwrap_single_element_action_value,
)
from strands_robots.utils import is_boolean

pytest.importorskip("mujoco")

# Two drives with deliberately different unit conventions, so "the same 1.0
# means different things" is a property of the fixture rather than a claim:
# ``lift_act`` is a joint-position drive in radians and ``grip_act`` is a fixed
# tendon whose ctrlrange is the [0, 255] gripper convention.
_ARM_XML = """
<mujoco model="two_drive_arm">
  <compiler angle="radian"/>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <joint name="lift" type="hinge" axis="0 1 0" range="-2 2" limited="true" damping="4"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
      <body name="hand" pos="0.2 0 0">
        <joint name="finger1" type="slide" axis="0 1 0" range="0 0.04" limited="true" damping="2"/>
        <geom type="box" size="0.01 0.005 0.02" pos="0 0.02 0"/>
        <body name="finger2_link">
          <joint name="finger2" type="slide" axis="0 -1 0" range="0 0.04" limited="true" damping="2"/>
          <geom type="box" size="0.01 0.005 0.02" pos="0 -0.02 0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <tendon>
    <fixed name="grip">
      <joint joint="finger1" coef="1"/>
      <joint joint="finger2" coef="1"/>
    </fixed>
  </tendon>
  <actuator>
    <position name="lift_act" joint="lift" kp="30" ctrlrange="-2 2"/>
    <position name="grip_act" tendon="grip" kp="40" ctrlrange="0 255"/>
  </actuator>
</mujoco>
"""

# ``0.5`` is a plain numeric command; the string spelling is accepted by
# ``send_action`` by documented design and is therefore excluded from the wire
# parity check below rather than smuggled into it.
_ACCEPTED_NUMERIC: list[tuple[str, Any]] = [
    ("float", 1.0),
    ("zero float", 0.0),
    ("int", 0),
    ("numpy float scalar", np.float64(0.5)),
    ("numpy int scalar", np.int64(1)),
    ("single-element list", [0.5]),
    ("single-element array", np.array([0.5])),
    ("0-d numpy array", np.array(0.5)),
]

_BOOLEAN_SPELLINGS: list[tuple[str, Any]] = [
    ("python True", True),
    ("python False", False),
    ("numpy bool scalar", np.True_),
    ("numpy bool false", np.bool_(False)),
    ("0-d numpy bool array", np.array(True)),
    ("single-element list of bool", [True]),
    ("single-element bool array", np.array([True])),
]


@pytest.fixture
def sim(tmp_path):  # noqa: ANN001, ANN201 - annotating this as the engine types _world as optional
    """A two-drive arm loaded from an inline MJCF (no asset download)."""
    from strands_robots.simulation import Simulation

    path = tmp_path / "two_drive_arm.xml"
    path.write_text(_ARM_XML, encoding="utf-8")
    engine = Simulation(backend="mujoco", tool_name="bool_action_test", mesh=False)
    engine.create_world()
    added = engine.add_robot(name="arm", urdf_path=str(path))
    assert added["status"] == "success", added
    yield engine
    engine.cleanup()


def _text(result: dict[str, Any]) -> str:
    return next(block["text"] for block in result["content"] if "text" in block)


def _ctrl(engine: Any) -> list[float]:
    data = engine._world._data
    return [float(v) for v in data.ctrl[: engine._world._model.nu]]


class TestBooleanMappingValueRefused:
    """A boolean mapping value is refused, naming the key that carried it."""

    @pytest.mark.parametrize(("label", "value"), _BOOLEAN_SPELLINGS, ids=[s[0] for s in _BOOLEAN_SPELLINGS])
    def test_every_boolean_spelling_is_refused(self, sim, label: str, value: Any) -> None:
        result = sim.send_action({"grip_act": value}, robot_name="arm")
        assert result["status"] == "error", f"{label} was accepted as a command"
        message = _text(result)
        assert "not a bool" in message
        assert "grip_act" in message

    def test_the_refusal_names_the_units_the_caller_should_use(self, sim) -> None:
        """The remedy has to say what to send instead, not merely that True is wrong."""
        message = _text(sim.send_action({"grip_act": True}, robot_name="arm"))
        assert "actuator's own units" in message
        assert "binary gripper" in message

    def test_a_boolean_leaves_the_actuators_and_the_clock_untouched(self, sim) -> None:
        """The guard precedes the ctrl write and the physics step, so nothing is half-applied."""
        applied = sim.send_action({"lift_act": 0.7, "grip_act": 12.0}, robot_name="arm")
        assert applied["status"] == "success", applied
        before_ctrl = _ctrl(sim)
        before_time = float(sim._world._data.time)
        assert before_ctrl != [0.0, 0.0], "fixture must establish a command a refusal could overwrite"

        refused = sim.send_action({"lift_act": 0.9, "grip_act": True}, robot_name="arm", n_substeps=5)
        assert refused["status"] == "error"
        assert _ctrl(sim) == before_ctrl, "a refused action must not apply its other keys"
        assert float(sim._world._data.time) == before_time, "a refused action must not advance physics"


class TestBooleanVectorEntryRefused:
    """A boolean vector entry is refused, naming its position and actuator key."""

    def test_an_all_boolean_vector_is_refused(self, sim) -> None:
        result = sim.send_action([True, True], robot_name="arm")
        assert result["status"] == "error"
        message = _text(result)
        assert "action vector entry 0" in message
        assert "lift_act" in message
        assert "not a bool" in message

    def test_one_boolean_among_floats_names_its_own_index(self, sim) -> None:
        """The gripper axis is the realistic carrier, and it is the last entry."""
        result = sim.send_action([0.3, True], robot_name="arm")
        assert result["status"] == "error"
        message = _text(result)
        assert "action vector entry 1" in message
        assert "grip_act" in message

    def test_a_numpy_boolean_vector_is_refused(self, sim) -> None:
        result = sim.send_action(np.array([True, False]), robot_name="arm")
        assert result["status"] == "error"
        assert "not a bool" in _text(result)

    def test_a_refused_vector_leaves_every_actuator_untouched(self, sim) -> None:
        assert sim.send_action([0.4, 20.0], robot_name="arm")["status"] == "success"
        before = _ctrl(sim)
        assert sim.send_action([0.5, True], robot_name="arm")["status"] == "error"
        assert _ctrl(sim) == before


class TestNumericCommandsStillAccepted:
    """The guard must not narrow the numeric spellings ``send_action`` accepts."""

    @pytest.mark.parametrize(("label", "value"), _ACCEPTED_NUMERIC, ids=[s[0] for s in _ACCEPTED_NUMERIC])
    def test_numeric_mapping_values_are_accepted(self, sim, label: str, value: Any) -> None:
        result = sim.send_action({"grip_act": value}, robot_name="arm")
        assert result["status"] == "success", f"{label} was refused: {_text(result)}"

    def test_a_numeric_string_remains_an_accepted_spelling(self, sim) -> None:
        """Documented behaviour: the domain is the value, not its python type."""
        result = sim.send_action({"grip_act": "0.5"}, robot_name="arm")
        assert result["status"] == "success", _text(result)
        assert _ctrl(sim)[1] == pytest.approx(127.5)

    @pytest.mark.parametrize("vector", [[0.3, 0.4], (0.3, 0.4), np.array([0.3, 0.4])], ids=["list", "tuple", "array"])
    def test_numeric_vectors_are_accepted(self, sim, vector: Any) -> None:
        assert sim.send_action(vector, robot_name="arm")["status"] == "success"


class TestZeroDimensionalArrayReachesTheValueChecks:
    """A 0-d numpy array raised ``len() of unsized object`` past the tool contract.

    It declares ``__len__`` and ``__getitem__`` but raises on ``len()``, so the
    single-element unwrap could not probe it - and the boolean gate could not see
    the 0-d boolean a policy comparison produces.
    """

    def test_a_zero_dimensional_numeric_array_is_a_usable_command(self, sim) -> None:
        result = sim.send_action({"grip_act": np.array(0.5)}, robot_name="arm")
        assert result["status"] == "success", _text(result)
        assert _ctrl(sim)[1] == pytest.approx(127.5)

    def test_a_zero_dimensional_boolean_array_is_refused_not_raised(self, sim) -> None:
        result = sim.send_action({"grip_act": np.array(True)}, robot_name="arm")
        assert result["status"] == "error"
        assert "not a bool" in _text(result)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ([0.05], 0.05),
            (np.array([0.05]), 0.05),
            ("0.5", "0.5"),
            (0.5, 0.5),
        ],
        ids=["single-element list", "single-element array", "string", "scalar"],
    )
    def test_the_unwrap_yields_the_scalar_or_the_value_itself(self, value: Any, expected: Any) -> None:
        assert _unwrap_single_element_action_value(value) == expected

    def test_the_unwrap_returns_a_zero_dimensional_array_unchanged(self) -> None:
        value = np.array(0.5)
        assert _unwrap_single_element_action_value(value) is value

    def test_the_unwrap_leaves_a_multi_element_sequence_for_the_length_check(self) -> None:
        value = [0.1, 0.2]
        assert _unwrap_single_element_action_value(value) is value


class TestBooleanPredicate:
    """``numpy.bool_`` is not a ``bool`` subclass, so the predicate cannot rely on isinstance."""

    @pytest.mark.parametrize(
        "value", [True, False, np.True_, np.bool_(False), np.array(True)], ids=["True", "False", "np", "np2", "0d"]
    )
    def test_boolean_values_are_reported_as_boolean(self, value: Any) -> None:
        assert is_boolean(value) is True

    @pytest.mark.parametrize(
        "value",
        [0, 1, 0.0, 1.0, np.float64(1.0), np.int64(1), np.uint8(1), "0.5", [1.0], np.array([1.0]), None],
        ids=["0", "1", "0.0", "1.0", "npf", "npi", "npu", "str", "list", "arr", "None"],
    )
    def test_non_boolean_values_are_not_reported_as_boolean(self, value: Any) -> None:
        assert is_boolean(value) is False


class TestWireAndLocalDomainsAgreeOnBooleans:
    """The wire validator and the call it applies frames through must not diverge.

    ``validate_input_frame`` sanitises a teleop frame and ``InputReceiver``
    applies the result via ``send_action``, so a boolean the wire refuses must
    not be a value the local call accepts. Only the boolean question is compared:
    ``send_action`` accepts a numeric string by documented design while the wire
    validator requires an ``int``/``float``, and that divergence is deliberate.
    """

    def test_both_surfaces_refuse_every_boolean_spelling(self, sim) -> None:
        from strands_robots.mesh.security import ValidationError, validate_input_frame

        for label, value in _BOOLEAN_SPELLINGS:
            if isinstance(value, (list, np.ndarray)) and getattr(value, "ndim", 1) != 0:
                # A sequence value is not a teleop frame shape; the wire
                # validator rejects it for its shape rather than its booleanness.
                continue
            with pytest.raises(ValidationError, match="not bool"):
                validate_input_frame({"grip_act": value})
            assert sim.send_action({"grip_act": value}, robot_name="arm")["status"] == "error", label

    def test_both_surfaces_accept_a_plain_numeric_command(self, sim) -> None:
        from strands_robots.mesh.security import validate_input_frame

        assert validate_input_frame({"grip_act": 12.0}) == {"grip_act": 12.0}
        assert sim.send_action({"grip_act": 12.0}, robot_name="arm")["status"] == "success"


class TestTheSameOneCommandsADifferentPose:
    """The premise the refusal message states, measured on the fixture.

    ``float(True)`` is 1.0 for both drives, and 1.0 is a different physical
    command on each: the tendon drive reads it as the normalized full-travel
    fraction and maps it onto its own ctrlrange, while the joint drive takes it
    as a radian target verbatim.
    """

    def test_one_means_full_travel_on_a_tendon_drive_and_a_radian_on_a_joint_drive(self, sim) -> None:
        model = sim._world._model
        lift_range = [float(v) for v in model.actuator_ctrlrange[0]]
        grip_range = [float(v) for v in model.actuator_ctrlrange[1]]
        assert lift_range != grip_range, "fixture must give the two drives different unit conventions"

        assert sim.send_action({"lift_act": 1.0, "grip_act": 1.0}, robot_name="arm")["status"] == "success"
        lift_ctrl, grip_ctrl = _ctrl(sim)
        assert lift_ctrl == pytest.approx(1.0), "a joint drive takes 1.0 as a radian target"
        assert grip_ctrl == pytest.approx(grip_range[1]), "a tendon drive reads 1.0 as full travel"


class TestEveryBackendInheritsTheGuard:
    """The guard lives in the shared coercion, so no backend can ship without it."""

    def test_send_action_is_defined_once_and_routes_through_the_shared_coercion(self) -> None:
        from strands_robots.simulation.isaac import simulation as isaac_simulation
        from strands_robots.simulation.mujoco import simulation as mujoco_simulation
        from strands_robots.simulation.newton import simulation as newton_simulation

        # Every backend, not the two that happened to be written first: this
        # class asserts that no backend can ship without the guard, and the
        # Isaac backend shipped its own conversion for exactly as long as it was
        # missing from this loop.
        for module in (mujoco_simulation, newton_simulation, isaac_simulation):
            tree = ast.parse(inspect.getsource(module))
            sends = [
                node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "send_action"
            ]
            assert sends, f"{module.__name__} defines no send_action"
            for node in sends:
                calls = {
                    child.func.attr
                    for child in ast.walk(node)
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                }
                assert "_coerce_action" in calls, (
                    f"{module.__name__}.send_action does not route through _coerce_action, "
                    "so it does not inherit the action-value domain"
                )

    def test_the_shared_coercion_is_defined_on_the_abstract_engine(self) -> None:
        assert "_coerce_action" in vars(SimEngine)
