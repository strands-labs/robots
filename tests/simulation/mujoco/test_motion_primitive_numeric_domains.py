"""Numeric-field domains of the analytic motion primitives.

``move_to`` / ``set_gripper`` / ``rotate_wrist`` each document a numeric domain
for their tolerance and their control-tick budget, and each promises "Never
raises" - an unusable value must come back as a structured tool error naming
the field, not as an exception out of a primitive an agent called. Two shared
choke points enforce that: ``_is_finite_real`` for the continuous fields
(``move_to``'s ``tol``, ``rotate_wrist``'s ``tol`` and ``target_yaw``) and
``_validate_step_budget`` for the discrete ones (``move_to``'s ``max_steps``,
``set_gripper``'s ``steps``, ``rotate_wrist``'s ``max_steps``).

The continuous guards read ``not _is_finite_real(x) or float(x) <= 0.0``, and
only the second half of that ``or`` was exercised: the suite drove
``move_to(tol=0.0)`` and ``tol=-1``, which the comparison rejects, so
``_is_finite_real`` never once returned ``False`` and ``_validate_step_budget``
never once took its type branch. A ``nan`` tolerance is the value that costs
the most if the guard regresses - every ``residual < tol`` comparison is then
false, so the primitive burns its whole budget and reports "did not reach" for
a target it is sitting on - and it was the half nobody covered.

These tests pin the contract behaviourally on a genuine MuJoCo scene: the
refusal happens, it names the field and the value, it precedes every write to
the model (a refused call leaves ``qpos``, ``ctrl`` and the sim clock
bit-identical), and the two documented spellings that are *not* domain errors -
an omitted ``target_yaw`` and a NumPy integer budget - keep working.
"""

import math
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.motion_primitives_base import _is_finite_real  # noqa: E402
from strands_robots.simulation.mujoco.motion_primitives import MotionPrimitivesMixin  # noqa: E402
from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

from .test_motion_primitives import ARM_XML, REACHABLE  # noqa: E402

# Values no finite-real field can be built from. ``None`` is deliberately
# absent: it is the documented "not supplied" spelling for ``target_yaw`` and
# reports as a missing required field rather than a domain error, which
# ``TestTheOmittedTargetIsNotADomainError`` pins separately.
NOT_A_FINITE_NUMBER = [
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="inf"),
    pytest.param(-math.inf, id="-inf"),
    pytest.param(True, id="True"),
    pytest.param(False, id="False"),
    pytest.param("0.05", id="str"),
    pytest.param([0.05], id="list"),
    pytest.param(np.bool_(True), id="np.bool_"),
]

# Values no control-tick budget can be built from. A budget is a count, so an
# integral float is refused too: ``range()`` and the tick loop want an int.
NOT_A_STEP_BUDGET = [
    *NOT_A_FINITE_NUMBER,
    pytest.param(2.7, id="2.7"),
    pytest.param(3.0, id="integral-float"),
    pytest.param(None, id="None"),
]

# Continuous fields, with the other required fields of that call and the unit
# word the message is documented to carry.
CONTINUOUS_FIELDS = [
    pytest.param("move_to", "tol", {"position": REACHABLE}, "meters", id="move_to-tol"),
    pytest.param("rotate_wrist", "tol", {"target_yaw": 0.3}, "radians", id="rotate_wrist-tol"),
    pytest.param("rotate_wrist", "target_yaw", {}, "radians", id="rotate_wrist-target_yaw"),
]

# Discrete tick budgets, one per primitive.
STEP_BUDGET_FIELDS = [
    pytest.param("move_to", "max_steps", {"position": REACHABLE, "tol": 0.05}, id="move_to-max_steps"),
    pytest.param("set_gripper", "steps", {"state": "open"}, id="set_gripper-steps"),
    pytest.param("rotate_wrist", "max_steps", {"target_yaw": 0.3}, id="rotate_wrist-max_steps"),
]


@pytest.fixture
def sim(tmp_path):
    """A genuine MuJoCo arm - the primitives' own inline MJCF, no asset fetch."""
    path = tmp_path / "prim_arm.xml"
    path.write_text(ARM_XML)
    s = Simulation(tool_name="test_motion_primitive_numeric_domains", mesh=False)
    assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
    assert s.add_robot("arm", urdf_path=str(path))["status"] == "success"
    yield s
    s.cleanup(policy_stop_timeout=2.0)


def _call(s: Simulation, action: str, **fields: Any) -> dict[str, Any]:
    """Drive a primitive through the agent-facing dispatch router."""
    return s._dispatch_action(action, {"action": action, **fields})


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result["content"] if isinstance(block, dict))


class TestTheSharedFinitenessPredicate:
    """``_is_finite_real`` is the numeric half of all three continuous guards."""

    @pytest.mark.parametrize("value", [*NOT_A_FINITE_NUMBER, pytest.param(None, id="None")])
    def test_a_value_no_number_can_be_built_from_is_rejected(self, value):
        assert _is_finite_real(value) is False

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param(0.05, id="float"),
            pytest.param(1, id="int"),
            pytest.param(-0.5, id="negative"),
            pytest.param(0.0, id="zero"),
            pytest.param(np.float64(0.05), id="np.float64"),
            pytest.param(np.int64(1), id="np.int64"),
        ],
    )
    def test_a_real_finite_scalar_is_accepted(self, value):
        """Sign and magnitude are the caller's business; this predicate is not."""
        assert _is_finite_real(value) is True


class TestEveryPrimitiveRefusesAnUnusableContinuousField:
    """A tolerance or set-point no number can be built from is a tool error."""

    @pytest.mark.parametrize(("action", "field", "rest", "unit"), CONTINUOUS_FIELDS)
    @pytest.mark.parametrize("value", NOT_A_FINITE_NUMBER)
    def test_the_refusal_names_the_field_the_unit_and_the_value(self, sim, action, field, rest, unit, value):
        # Returning at all is the "Never raises" half of the contract.
        result = _call(sim, action, robot_name="arm", **rest, **{field: value})
        assert result["status"] == "error"
        text = _text(result)
        assert f"'{field}'" in text
        assert unit in text
        assert repr(value) in text


class TestEveryPrimitiveRefusesAnUnusableStepBudget:
    """A control-tick budget that is not an integer is a tool error."""

    @pytest.mark.parametrize(("action", "field", "rest"), STEP_BUDGET_FIELDS)
    @pytest.mark.parametrize("value", NOT_A_STEP_BUDGET)
    def test_the_refusal_names_the_field_and_the_type(self, sim, action, field, rest, value):
        result = _call(sim, action, robot_name="arm", **rest, **{field: value})
        assert result["status"] == "error"
        text = _text(result)
        assert f"'{field}'" in text
        assert f"got {type(value).__name__}." in text

    def test_a_numpy_integer_budget_is_honoured(self, sim):
        """The rule refuses non-integers, not every spelling of an integer."""
        result = _call(sim, "set_gripper", robot_name="arm", state="open", steps=np.int64(5))
        assert result["status"] == "success"


class TestARefusedNumericFieldActuatesNothing:
    """Every numeric guard runs before the primitive touches the model."""

    def test_a_refused_call_leaves_the_pose_the_command_and_the_clock_untouched(self, sim):
        # Drive the arm somewhere distinctive first, so a partially applied
        # primitive - a written ctrl, a stepped clock - would be visible.
        assert (
            _call(sim, "rotate_wrist", robot_name="arm", target_yaw=0.4, tol=0.02, max_steps=300)["status"] == "success"
        )
        world = sim._world
        assert world is not None
        qpos = np.array(world._data.qpos, copy=True)
        ctrl = np.array(world._data.ctrl, copy=True)
        clock = float(world._data.time)
        assert not np.allclose(ctrl, 0.0), "fixture must leave a non-default command to detect a write"

        refusals = [
            ("rotate_wrist", {"target_yaw": math.nan}),
            ("rotate_wrist", {"target_yaw": 0.1, "tol": math.nan}),
            ("rotate_wrist", {"target_yaw": 0.1, "max_steps": 2.7}),
            ("move_to", {"position": REACHABLE, "tol": math.inf}),
            ("set_gripper", {"state": "open", "steps": True}),
        ]
        for action, fields in refusals:
            result = _call(sim, action, robot_name="arm", **fields)
            assert result["status"] == "error", f"{action} {fields}"
            assert np.array_equal(world._data.qpos, qpos), f"{action} moved the arm"
            assert np.array_equal(world._data.ctrl, ctrl), f"{action} wrote a command"
            assert float(world._data.time) == clock, f"{action} stepped the clock"


class TestTheOmittedTargetIsNotADomainError:
    """``target_yaw=None`` means "not supplied", not "outside the domain"."""

    def test_an_omitted_target_reports_the_missing_field(self, sim):
        text = _text(_call(sim, "rotate_wrist", robot_name="arm", target_yaw=None))
        assert "requires 'target_yaw'" in text
        assert "must be a finite number" not in text

    def test_a_usable_target_still_converges(self, sim):
        result = _call(sim, "rotate_wrist", robot_name="arm", target_yaw=0.3, tol=0.02, max_steps=300)
        assert result["status"] == "success"
        assert "reached" in _text(result)


class TestTheStepBudgetRuleHasOneOwner:
    """All three tick budgets route through ``_validate_step_budget``."""

    @pytest.mark.parametrize(("action", "field", "rest"), STEP_BUDGET_FIELDS)
    def test_each_primitive_reports_the_shared_wording(self, sim, action, field, rest):
        shared = _text(MotionPrimitivesMixin._validate_step_budget(action, field, "not-a-count") or {"content": []})
        assert shared, "the shared validator must produce a message"
        assert _text(_call(sim, action, robot_name="arm", **rest, **{field: "not-a-count"})) == shared
