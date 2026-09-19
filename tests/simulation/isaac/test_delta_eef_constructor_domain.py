"""Every refusal ``IsaacDeltaEEFController.__init__`` makes, end to end (#1812).

The controller's constructor takes five numeric knobs, and each one either
multiplies the DLS differential-IK solve (``pos_scale`` / ``rot_scale`` /
``damping``) or is written straight into a joint target (``gripper_open`` /
``gripper_close``), plus a ``joint_limits`` table every target is clipped
into. Three of them were bounded by an **order comparison** (``> 0``,
``lower > upper``); an order comparison cannot reject ``inf`` (``inf > 0``
is ``True``) and two of them were not bounded at all. Every such value was
accepted at construction and then made ``compute_joint_targets`` return
``nan`` for every joint, which ``send_action``'s action-value domain refuses
one action at a time -- naming the *joint* it was handed, never the
constructor parameter the value came from.

This module pins that whole matrix:

* :class:`TestNumericConstructorDomain` -- each of the five numeric knobs
  reports the shared scalar domain's verdict verbatim, so the accepted
  domain cannot drift from the one every other surface enforces.
* :class:`TestAnOrderComparisonCannotSeeInf` -- the premise, executable:
  ``inf`` satisfies ``> 0`` and ``nan > nan`` is ``False``, which is why
  neither shape can bound these values.
* :class:`TestJointLimitsMustBeFinite` -- the table's finiteness check runs
  *before* its lower/upper comparison, which the premise above cannot see.
* :class:`TestGripperReferenceIsLoadBearing` -- an unguarded non-finite
  gripper reference is latent: approach and hold are unaffected and only
  the first *close* produces ``nan``.
* :class:`TestRefusalsThatHadNoTest` -- the four refusals the constructor
  already made that nothing exercised (duplicate arm names, a
  non-callable injected read, a non-positive scale, an inverted limit row).
* :class:`TestAnUnreadableChannelIsRefusedByDefault`,
  :class:`TestTheDegradedPostureIsStillAvailable` and
  :class:`TestANonFiniteChannelIsRefusedInBothPostures` -- the coercion branch,
  which used to log a WARNING and substitute a zero unconditionally. The
  degrade-one-axis posture is preserved under ``strict=False``; a non-finite
  channel is refused either way, because it solved to ``nan`` targets for every
  arm joint rather than holding one axis.
* :class:`TestAnAbsentChannelIsNotAnError` -- absence means "hold this axis" and
  is the distinction that lets the coercion refuse a value it cannot read.
* :class:`TestTheStrictFlagIsCheckedNotTruthy` -- the posture flag's domain.
* :class:`TestARefusedValueNeverBecomesAPerActionEnvelope` -- the engine
  seam, and the reason the constructor is the right place to refuse.
* :class:`TestEveryNumericKnobIsJudged` -- a drift guard, so a sixth
  numeric knob cannot ship unbounded.

Fully mocked articulation: no Isaac Sim install is needed.
"""

from __future__ import annotations

import ast
import inspect
import logging
import math
import pathlib
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac import delta_eef as delta_eef_mod
from strands_robots.simulation.isaac.delta_eef import (
    DEFAULT_POS_SCALE,
    IsaacDeltaEEFController,
    _to_scalar,
)
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.utils import finite_number_error, positive_finite_number_error

from .test_backend_parity import (  # noqa: F401 - fake_isaacsim_types is a fixture
    _FakeArticulation,
    _seed_running_world,
    fake_isaacsim_types,
)
from .test_delta_eef_controller import ARM, DAMPING, GRIP

JOINTS = ARM + GRIP

#: Values no scalar domain in this repository accepts. ``inf`` is the one the
#: replaced order comparisons let through the front door; ``nan`` they refused
#: only because ``nan > 0`` is ``False``, and ``True`` because ``bool`` is an
#: ``int`` subclass whose ``1`` would be a silent 20x scale.
UNUSABLE = [
    pytest.param(0.0, id="zero"),
    pytest.param(-1.0, id="negative"),
    pytest.param(math.nan, id="nan"),
    pytest.param(math.inf, id="inf"),
    pytest.param(-math.inf, id="neg-inf"),
    pytest.param(True, id="bool"),
    pytest.param("0.05", id="numeric-string"),
    pytest.param(None, id="none"),
    pytest.param([0.05], id="list"),
]

#: Values only the signed domain accepts -- a gripper reference is a joint
#: position, so a negative reference is a legitimate configuration.
SIGNED_ONLY = [pytest.param(0.0, id="zero"), pytest.param(-0.01, id="negative")]

#: The three knobs that scale or damp the solve, and the shared domain each
#: one is judged by. ``joint_limits`` is a table rather than a scalar and has
#: :class:`TestJointLimitsMustBeFinite` to itself.
POSITIVE_KNOBS = ("pos_scale", "rot_scale", "damping")
SIGNED_KNOBS = ("gripper_open", "gripper_close")


def _build(**overrides: Any) -> IsaacDeltaEEFController:
    """Construct a controller, funnelling deliberately off-type overrides.

    The numeric parameters are annotated ``float``, so a test that hands one
    a string or ``None`` -- the whole point here -- is an ``arg-type`` error
    at the call site. Splatting through one ``**kwargs: Any`` funnel keeps
    the deliberate part in the parametrization instead of in a suppression.
    """
    kwargs: dict[str, Any] = {
        "arm_joint_names": ARM,
        "gripper_joint_names": GRIP,
        "joint_positions_fn": lambda: np.zeros(7),
        "jacobian_fn": lambda: np.eye(6, 7),
    }
    kwargs.update(overrides)
    return IsaacDeltaEEFController(**kwargs)


class TestNumericConstructorDomain:
    """Each numeric knob reports the shared domain's verdict verbatim."""

    @pytest.mark.parametrize("param", POSITIVE_KNOBS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_positive_knob_reports_the_shared_domain_verbatim(self, param: str, value: Any) -> None:
        expected = positive_finite_number_error(value, param, "IsaacDeltaEEFController")
        assert expected is not None, "probe value must be outside the domain"
        with pytest.raises(ValueError) as excinfo:
            _build(**{param: value})
        assert str(excinfo.value) == expected

    @pytest.mark.parametrize("param", SIGNED_KNOBS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_signed_knob_reports_the_shared_domain_verbatim(self, param: str, value: Any) -> None:
        expected = finite_number_error(value, param, "IsaacDeltaEEFController")
        if expected is None:
            # A gripper reference may legitimately be zero or negative; the
            # signed domain accepts it and so must the constructor.
            _build(**{param: value})
            return
        with pytest.raises(ValueError) as excinfo:
            _build(**{param: value})
        assert str(excinfo.value) == expected

    @pytest.mark.parametrize("param", SIGNED_KNOBS)
    @pytest.mark.parametrize("value", SIGNED_ONLY)
    def test_a_gripper_reference_may_be_zero_or_negative(self, param: str, value: float) -> None:
        controller = _build(**{param: value})
        assert getattr(controller, f"_{param}") == pytest.approx(value)

    @pytest.mark.parametrize("param", POSITIVE_KNOBS + SIGNED_KNOBS)
    def test_the_default_is_inside_its_own_domain(self, param: str) -> None:
        """The shipped default must satisfy the domain it is judged by."""
        default = inspect.signature(IsaacDeltaEEFController).parameters[param].default
        judge = positive_finite_number_error if param in POSITIVE_KNOBS else finite_number_error
        assert judge(default, param, "IsaacDeltaEEFController") is None

    def test_a_refused_knob_is_named_not_the_joint_it_would_have_poisoned(self) -> None:
        """The refusal names the parameter, which the per-action one cannot."""
        with pytest.raises(ValueError, match=r"^IsaacDeltaEEFController: pos_scale must be > 0"):
            _build(pos_scale=math.inf)


class TestAnOrderComparisonCannotSeeInf:
    """The premise the replaced guards rested on, measured rather than asserted."""

    def test_inf_satisfies_a_positive_order_comparison(self) -> None:
        assert math.inf > 0.0, "an order comparison cannot reject inf"

    def test_nan_fails_both_directions_of_an_order_comparison(self) -> None:
        assert not (math.nan > 0.0)
        assert not (math.nan <= 0.0)

    def test_nan_bounds_pass_an_inverted_limit_comparison(self) -> None:
        limits = np.full((len(ARM), 2), math.nan)
        assert not np.any(limits[:, 0] > limits[:, 1]), "lower>upper cannot see a nan row"

    def test_the_domain_refuses_what_the_order_comparison_accepted(self) -> None:
        for param in POSITIVE_KNOBS:
            with pytest.raises(ValueError, match=r"must be > 0, got inf"):
                _build(**{param: math.inf})


class TestJointLimitsMustBeFinite:
    """The clip table is checked for finiteness before it is checked for order."""

    @pytest.mark.parametrize(
        "row",
        [
            pytest.param([math.nan, math.nan], id="both-nan"),
            pytest.param([0.0, math.nan], id="upper-nan"),
            pytest.param([math.nan, 1.0], id="lower-nan"),
            pytest.param([-math.inf, 1.0], id="lower-neg-inf"),
            pytest.param([0.0, math.inf], id="upper-inf"),
        ],
    )
    def test_a_non_finite_bound_is_refused(self, row: list[float]) -> None:
        with pytest.raises(ValueError, match=r"joint_limits must be finite \(no nan/inf\)"):
            _build(joint_limits=[list(row)] * len(ARM))

    def test_the_inverted_row_check_is_still_reachable(self) -> None:
        """Finiteness runs first, so the order check keeps its own message."""
        with pytest.raises(ValueError, match="lower bound above its upper bound"):
            _build(joint_limits=[[1.0, 0.0]] * len(ARM))

    def test_a_usable_table_still_clips(self) -> None:
        controller = _build(joint_limits=[[-0.01, 0.01]] * len(ARM))
        targets = controller.compute_joint_targets({"x": 1.0})
        assert targets[ARM[0]] == pytest.approx(0.01)

    def test_none_still_disables_clipping(self) -> None:
        controller = _build(joint_limits=None)
        targets = controller.compute_joint_targets({"x": 1.0})
        assert targets[ARM[0]] == pytest.approx(DEFAULT_POS_SCALE / (1.0 + DAMPING**2), abs=1e-6)


class TestGripperReferenceIsLoadBearing:
    """A gripper reference reaches a target, so a bad one is latent."""

    def test_open_and_close_write_their_configured_references(self) -> None:
        controller = _build(gripper_open=0.031, gripper_close=-0.002)
        opened = controller.compute_joint_targets({"gripper": 1.0})
        closed = controller.compute_joint_targets({"gripper": 0.0})
        for joint in GRIP:
            assert opened[joint] == pytest.approx(0.031)
            assert closed[joint] == pytest.approx(-0.002)

    def test_a_close_reference_is_unread_until_the_first_close(self) -> None:
        """Why an unguarded reference was latent, not loud.

        ``gripper_close`` is read only by a close command, so an unusable one
        left approach and hold actions untouched and surfaced at the grasp.
        The reference is now refused at construction; this pins the read
        pattern that made it latent.
        """
        controller = _build(gripper_close=-0.002)
        assert set(controller.compute_joint_targets({"gripper": 1.0})) >= set(GRIP)
        assert not set(controller.compute_joint_targets({"gripper": 0.5})) & set(GRIP)
        assert set(controller.compute_joint_targets({"gripper": 0.0})) >= set(GRIP)


class TestRefusalsThatHadNoTest:
    """The four refusals the constructor already made and nothing exercised."""

    def test_duplicate_arm_joint_names_are_refused(self) -> None:
        with pytest.raises(ValueError, match="contains duplicates"):
            _build(arm_joint_names=[ARM[0]] + ARM)

    @pytest.mark.parametrize("param", ["joint_positions_fn", "jacobian_fn"])
    @pytest.mark.parametrize("value", [None, 42, "callable"])
    def test_a_non_callable_injected_read_is_refused(self, param: str, value: Any) -> None:
        with pytest.raises(TypeError, match="must be callables"):
            _build(**{param: value})

    @pytest.mark.parametrize("param", POSITIVE_KNOBS)
    def test_a_non_positive_scale_is_refused(self, param: str) -> None:
        with pytest.raises(ValueError, match=rf"{param} must be > 0, got 0\.0"):
            _build(**{param: 0.0})

    def test_an_inverted_limit_row_is_refused(self) -> None:
        with pytest.raises(ValueError, match="lower bound above its upper bound"):
            _build(joint_limits=[[0.1, -0.1]] * len(ARM))


#: The values that cannot be read as a scalar at all.
_UNREADABLE = [
    pytest.param(None, id="none"),
    pytest.param([], id="empty-list"),
    pytest.param({}, id="dict"),
    pytest.param("abc", id="non-numeric-string"),
    pytest.param([None], id="list-of-none"),
    pytest.param(np.array([]), id="empty-array"),
]


class TestAnUnreadableChannelIsRefusedByDefault:
    """This branch used to substitute a zero and log a WARNING, always.

    It replaces ``TestToScalarFallback``, whose reasoning - "a malformed channel
    degrades that one axis, it does not abort the step" - is a defensible
    posture and is now what ``strict=False`` selects. What changed is which
    posture is the *default*, on two grounds: it returned a zero-valued action
    on failure, and the substitution was not the "degrade one axis" it reads as
    for the gripper, where ``0.5`` becomes ``-sign(0.0)`` -> ``-0.0`` and is
    dropped by the ``command != 0.0`` guard, so a grasp silently did nothing.
    """

    @pytest.mark.parametrize("value", _UNREADABLE)
    def test_the_coercion_raises(self, value: Any) -> None:
        with pytest.raises(ValueError, match="cannot be read as a scalar"):
            _to_scalar(value, "x", strict=True)

    @pytest.mark.parametrize("value", _UNREADABLE)
    def test_the_controller_refuses_the_action(self, value: Any) -> None:
        controller = _build()

        with pytest.raises(ValueError, match="cannot be read as a scalar"):
            controller.compute_joint_targets({"x": 1.0, "y": value})

    def test_the_refusal_names_the_channel(self) -> None:
        """So the caller fixes the axis rather than searching the action."""
        with pytest.raises(ValueError, match="'roll'"):
            _to_scalar("abc", "roll", strict=True)

    def test_the_refusal_names_the_escape_hatch(self) -> None:
        with pytest.raises(ValueError, match="strict=False"):
            _to_scalar("abc", "x", strict=True)

    def test_an_unreadable_gripper_is_refused_too(self) -> None:
        """The case the old substitution handled worst: 0.5 -> -0.0 -> dropped,
        so a commanded grasp or release silently did not happen."""
        controller = _build()

        with pytest.raises(ValueError, match="'gripper'"):
            controller.compute_joint_targets({"gripper": "abc"})


class TestTheDegradedPostureIsStillAvailable:
    """``strict=False`` restores what this controller shipped with."""

    @pytest.mark.parametrize("value", _UNREADABLE)
    def test_an_unreadable_channel_falls_back_to_the_default(self, value: Any) -> None:
        assert _to_scalar(value, "x", strict=False, default=0.0) == 0.0
        assert _to_scalar(value, "x", strict=False, default=0.25) == 0.25

    def test_the_fallback_logs_a_warning_naming_the_value(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=delta_eef_mod.__name__):
            assert _to_scalar("abc", "x", strict=False) == 0.0
        assert any("could not coerce action value" in r.getMessage() for r in caplog.records)
        assert any("'abc'" in r.getMessage() for r in caplog.records)

    def test_an_unreadable_channel_is_a_no_op_delta_not_a_raise(self) -> None:
        """The original assertion, verbatim in substance, under the flag that
        now selects it: the rest of the action still applies."""
        controller = _build(strict=False)

        targets = controller.compute_joint_targets({"x": 1.0, "y": None, "roll": "abc"})

        assert all(np.isfinite(list(targets.values())))
        assert targets[ARM[0]] == pytest.approx(DEFAULT_POS_SCALE / (1.0 + DAMPING**2), abs=1e-6)


class TestNaNIsRefusedInBothPostures:
    """The case no posture can honour, and the one the old code never caught.

    ``float("nan")`` coerces successfully, so it never reached the fallback at
    all, and ``_solve_arm_targets``' finiteness guard covers only the injected
    callables. The solve returned non-finite targets for EVERY arm joint - not a
    held axis - and those reach PhysX, which reports them from a later step as
    "Illegal BroadPhaseUpdateData - non-finite bounds", attributed to whatever is
    running by then. The replaced test asserted ``all(np.isfinite(...))`` of the
    targets and never passed a ``nan`` in, so its own invariant was the thing this
    violated.
    """

    @pytest.mark.parametrize("strict", [True, False])
    def test_the_coercion_raises(self, strict: bool) -> None:
        with pytest.raises(ValueError, match="is NaN"):
            _to_scalar(math.nan, "x", strict=strict)

    @pytest.mark.parametrize("strict", [True, False])
    @pytest.mark.parametrize("channel", ["x", "y", "z", "roll", "pitch", "yaw", "gripper"])
    def test_every_channel_refuses_it(self, strict: bool, channel: str) -> None:
        """Parametrized over the channel, because a guard that covered only the
        arm axes would leave the gripper coercion free to reintroduce a
        substitution - and a class that tested one channel could not see it."""
        controller = _build(strict=strict)

        with pytest.raises(ValueError, match="is NaN"):
            controller.compute_joint_targets({channel: math.nan})

    def test_the_refusal_says_strict_does_not_relax_it(self) -> None:
        with pytest.raises(ValueError, match="strict=False does not relax this"):
            _to_scalar(math.nan, "x", strict=True)

    @pytest.mark.parametrize("strict", [True, False])
    def test_no_non_finite_target_is_ever_returned(self, strict: bool) -> None:
        """The invariant the class exists for, stated directly."""
        controller = _build(strict=strict)

        for channel in ("x", "y", "z", "roll", "pitch", "yaw"):
            with pytest.raises(ValueError):
                controller.compute_joint_targets({channel: math.nan})


class TestInfinityIsAcceptedAndSaturates:
    """``+-inf`` is NOT refused, and that is deliberate and measured.

    ``np.clip(+-inf, -1, 1)`` is ``+-1.0``, so an infinite channel saturates to the
    maximum per-step delta - precisely the clip-then-scale contract this controller
    documents. Measured against clean main, ``x=+inf`` produced a finite
    ``{"j1": 0.0499, ...}`` while ``x=nan`` produced ``{"j1": nan, "j2": nan}``.

    An earlier version of this fix refused every non-finite value on the stated
    grounds that it "solves to non-finite targets for every arm joint". True of
    ``nan``, false of ``+-inf`` - and refusing inf converts a correct saturation
    into a hard failure for any policy head that saturates. Adversarial review
    caught it; this class stops it coming back.
    """

    @pytest.mark.parametrize("value", [math.inf, -math.inf])
    @pytest.mark.parametrize("strict", [True, False])
    def test_the_coercion_accepts_it(self, value: float, strict: bool) -> None:
        assert _to_scalar(value, "x", strict=strict) == value

    def test_it_saturates_to_the_maximum_positive_delta(self) -> None:
        targets = _build().compute_joint_targets({"x": math.inf})

        assert targets[ARM[0]] == pytest.approx(DEFAULT_POS_SCALE / (1.0 + DAMPING**2), abs=1e-6)
        assert all(math.isfinite(v) for v in targets.values())

    def test_it_saturates_to_the_maximum_negative_delta(self) -> None:
        targets = _build().compute_joint_targets({"x": -math.inf})

        assert targets[ARM[0]] == pytest.approx(-DEFAULT_POS_SCALE / (1.0 + DAMPING**2), abs=1e-6)
        assert all(math.isfinite(v) for v in targets.values())

    def test_it_matches_a_saturated_finite_input(self) -> None:
        """The equivalence that makes accepting it correct rather than lenient:
        inf and +1.0 are the same request after the documented clip."""
        assert _build().compute_joint_targets({"x": math.inf}) == _build().compute_joint_targets({"x": 1.0})

    @pytest.mark.parametrize("value", [math.inf, -math.inf])
    def test_an_infinite_gripper_is_well_defined_too(self, value: float) -> None:
        """``-sign(2*(+-inf) - 1)`` is ``-+1``, so it is a definite open/close
        rather than the dropped ``-0.0`` an unreadable value used to become."""
        targets = _build().compute_joint_targets({"gripper": value})

        assert set(GRIP) <= set(targets)
        assert all(math.isfinite(v) for v in targets.values())


class TestAnAbsentChannelIsNotAnError:
    """Absence means "hold this axis" and is the documented default.

    Keeping this separate from the unreadable case is what lets the coercion
    refuse: the zero for a missing axis is a documented default, and the zero
    that used to stand in for an unreadable one was a fabricated command.
    """

    def test_an_empty_action_yields_no_targets(self) -> None:
        assert _build().compute_joint_targets({}) == {}

    def test_a_partial_action_moves_only_what_it_names(self) -> None:
        targets = _build().compute_joint_targets({"x": 1.0})

        assert targets[ARM[0]] == pytest.approx(DEFAULT_POS_SCALE / (1.0 + DAMPING**2), abs=1e-6)

    def test_an_absent_gripper_writes_no_finger_target(self) -> None:
        assert not set(GRIP) & set(_build().compute_joint_targets({"x": 1.0}))


class TestTheStrictFlagIsCheckedNotTruthy:
    """A posture flag read by truthiness inverts exactly the spellings an
    operator reaches for: ``strict="no"`` would select the permissive posture
    while reading as the strict one."""

    @pytest.mark.parametrize("value", ["no", "false", "0", "off", 0, 1, None, "", [], "yes"])
    def test_a_non_boolean_is_refused(self, value: Any) -> None:
        with pytest.raises(ValueError, match="strict"):
            _build(strict=value)

    @pytest.mark.parametrize("value", [True, False])
    def test_a_boolean_is_accepted(self, value: bool) -> None:
        assert _build(strict=value) is not None


class _NanController:
    """Stand-in whose conversion returns the non-finite targets a bad knob made."""

    def compute_joint_targets(self, action: Any) -> dict[str, float]:  # noqa: ARG002 - fixed output
        return dict.fromkeys(JOINTS, math.nan)

    def reset(self) -> None:
        return None


class TestARefusedValueNeverBecomesAPerActionEnvelope:
    """The engine seam: why the constructor is the right place to refuse."""

    def test_a_non_finite_target_is_reported_against_a_joint_name(self, fake_isaacsim_types) -> None:  # noqa: F811, ARG002
        """The failure a bad knob used to produce, once per action.

        ``send_action``'s action-value domain sees only the converted target,
        so it names the joint it was handed. That message is correct about
        the value it has and cannot name the constructor parameter the value
        came from -- which is the whole reason the knob is bounded up front.
        """
        sim = IsaacSimulation()
        articulation = _FakeArticulation()
        _seed_running_world(sim, JOINTS, articulation)
        assert sim.install_action_controller("arm", _NanController())["status"] == "success"

        result = sim.send_action({"x": 1.0, "gripper": 1.0}, robot_name="arm")
        text = " ".join(block.get("text", "") for block in result["content"])
        assert result["status"] == "error", result
        assert "must be finite" in text
        assert ARM[0] in text
        assert "pos_scale" not in text
        assert articulation.last_action is None, "a non-finite target must not be applied"

    def test_a_usable_configuration_still_applies(self, fake_isaacsim_types) -> None:  # noqa: F811, ARG002
        sim = IsaacSimulation()
        articulation = _FakeArticulation()
        _seed_running_world(sim, JOINTS, articulation)
        assert sim.install_action_controller("arm", _build())["status"] == "success"

        result = sim.send_action({"x": 1.0, "gripper": 1.0}, robot_name="arm")
        assert result["status"] == "success", result
        assert articulation.last_action is not None

    def test_the_refusal_precedes_any_use_of_the_injected_reads(self) -> None:
        """A refused knob never calls the callables it was handed."""
        calls: list[str] = []

        def positions() -> Any:
            calls.append("q")
            return np.zeros(7)

        def jacobian() -> Any:
            calls.append("jac")
            return np.eye(6, 7)

        with pytest.raises(ValueError):
            _build(joint_positions_fn=positions, jacobian_fn=jacobian, damping=math.inf)
        assert calls == []


class TestEveryNumericKnobIsJudged:
    """Drift guard: a sixth numeric knob cannot ship unbounded."""

    @staticmethod
    def _judged_params(source: str) -> set[str]:
        """Parameter names appearing in a domain-guard tuple in ``__init__``."""
        tree = ast.parse(source)
        init = next(
            node
            for cls in tree.body
            if isinstance(cls, ast.ClassDef) and cls.name == "IsaacDeltaEEFController"
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        judged: set[str] = set()
        for node in ast.walk(init):
            if not isinstance(node, ast.Tuple) or len(node.elts) != 2:
                continue
            label, value = node.elts
            if (
                isinstance(label, ast.Constant)
                and isinstance(label.value, str)
                and isinstance(value, ast.Name)
                and label.value == value.id
            ):
                judged.add(label.value)
        return judged

    @staticmethod
    def _float_params() -> set[str]:
        """Keyword-only ``float`` parameters of the constructor."""
        signature = inspect.signature(IsaacDeltaEEFController)
        return {name for name, parameter in signature.parameters.items() if parameter.annotation in ("float", float)}

    def test_the_scanner_finds_the_known_knobs(self) -> None:
        """Non-vacuity: an empty scan must not read as a clean sweep."""
        source = pathlib.Path(inspect.getfile(delta_eef_mod)).read_text(encoding="utf-8")
        assert self._judged_params(source) == set(POSITIVE_KNOBS) | set(SIGNED_KNOBS)
        assert self._float_params() == set(POSITIVE_KNOBS) | set(SIGNED_KNOBS)

    def test_every_float_parameter_is_judged_by_a_shared_domain(self) -> None:
        source = pathlib.Path(inspect.getfile(delta_eef_mod)).read_text(encoding="utf-8")
        adrift = sorted(self._float_params() - self._judged_params(source))
        assert not adrift, f"numeric knob(s) reaching the solve unbounded: {adrift}"

    def test_the_scanner_sees_a_planted_unbounded_knob(self) -> None:
        """Meta: an added knob outside every guard tuple is reported."""
        source = pathlib.Path(inspect.getfile(delta_eef_mod)).read_text(encoding="utf-8")
        planted = source.replace(
            "        gripper_close: float = 0.0,\n",
            "        gripper_close: float = 0.0,\n        stiffness: float = 1.0,\n",
            1,
        )
        assert planted != source
        assert "stiffness" not in self._judged_params(planted)
