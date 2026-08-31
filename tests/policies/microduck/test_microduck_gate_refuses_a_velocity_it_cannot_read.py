"""The velocity gate refuses a goal it cannot read, before the selection moves.

``MicroduckPolicyBundle``'s ``switch_on_velocity`` gate reads the per-tick
``target_velocity`` and picks ``move_key`` or ``idle_key`` from its magnitude.
It was the third reader of that well-known key in this family and the only one
not held to a domain. The other operand of its own comparison is: the threshold
``switch_on_velocity`` goes through ``positive_finite_number_error`` at
construction. One comparison, two operands, one guarded.

The child's ``MicroduckPolicy._apply_command_kwargs`` states the two reasons for
its own guard, and both of them applied here one layer up:

* A non-numeric value surfaced as a bare ``could not convert string to float``
  out of the gate's own ``np.asarray`` - and a ``TypeError`` for a mapping, where
  this family documents ``ValueError`` - naming neither the bundle the caller
  called nor the parameter it passed.
* A non-finite value made the magnitude ``nan``, and ``nan`` is ``>=`` nothing,
  so the gate silently selected ``idle_key``. A caller asking the robot to move
  at a ``nan`` velocity got the standing skill. The child then refused the tick
  against a name the caller never used - and the moved selection outlived that
  failed call, so a caller that handled the error and kept ticking was running a
  skill it never asked for.

A width outside the two the child documents behaved the same way: the gate read
a magnitude out of ``[0.5]`` or the first three of ``[0.5, 0, 0, 99]``, moved the
selection, and only then did the tick refuse the width.

Nothing graded it because every cell that exercises the gate passes a velocity
both readers accept, and every cell that exercises a bad velocity drives the
child directly - so no cell held a bundle and a value the tick would refuse. The
invariant below is the one that was missing: a refused tick leaves the active
skill exactly as it was.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import math
import textwrap
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.microduck import MicroduckPolicy, MicroduckPolicyBundle
from strands_robots.policies.microduck import composite as composite_mod
from strands_robots.policies.microduck import policy as policy_mod
from strands_robots.policies.microduck.policy import TARGET_VELOCITY_WIDTHS

from .test_microduck_gate_arbitrates_between_the_pair import (
    _GATE_THRESHOLD,
    _IDLE_KEY,
    _MOVE_KEY,
    _bundle,
)
from .test_microduck_policy import _obs_dict, _StubSession

#: Velocities neither the gate nor the tick can honor, one per failure kind the
#: two guards cover: a non-finite component, a non-numeric one, and a component
#: count outside the pair the child documents.
_REFUSED: tuple[Any, ...] = (
    pytest.param([float("nan"), 0.0, 0.0], id="nan-in-vx"),
    pytest.param([0.0, 0.0, float("nan")], id="nan-in-omega"),
    pytest.param([float("inf"), 0.0, 0.0], id="positive-inf"),
    pytest.param([float("-inf"), 0.0, 0.0], id="negative-inf"),
    pytest.param([0.5, None, 0.0], id="none-component"),
    pytest.param("fast", id="string"),
    pytest.param({"vx": 1.0}, id="mapping"),
    pytest.param(["a", "b", "c"], id="string-components"),
    pytest.param(0.5, id="bare-scalar"),
    pytest.param([0.5], id="one-component"),
    pytest.param([0.5, 0.0, 0.0, 99.0], id="four-components"),
)

#: Velocities both readers accept, with the skill the magnitude selects. The
#: two-component spelling is documented and deliberately kept.
_ACCEPTED: tuple[Any, ...] = (
    pytest.param([0.5, 0.0, 0.0], _MOVE_KEY, id="three-components-moving"),
    pytest.param([0.0, 0.0, 0.0], _IDLE_KEY, id="three-components-still"),
    pytest.param([0.5, 0.0], _MOVE_KEY, id="two-components-moving"),
    pytest.param([0.0, 0.0], _IDLE_KEY, id="two-components-still"),
    pytest.param([1, 0, 0], _MOVE_KEY, id="integer-components"),
    pytest.param(np.array([0.4, 0.0, 0.0], dtype=np.float32), _MOVE_KEY, id="numpy-vector"),
)

#: The prefix the bundle's own refusals carry. Matched as a prefix rather than a
#: containment: ``MicroduckPolicy`` is a substring of ``MicroduckPolicyBundle``,
#: so "the child is not named" cannot be spelled with ``not in``.
_BUNDLE_PREFIX = f"{MicroduckPolicyBundle.__name__}.get_actions:"


def _tick(bundle: MicroduckPolicyBundle, **kwargs: Any) -> None:
    asyncio.run(bundle.get_actions(_obs_dict(), "", **kwargs))


def _tick_error(bundle: MicroduckPolicyBundle, **kwargs: Any) -> Exception | None:
    """Run one tick; return the exception it raised, or ``None`` on success.

    Deliberately not ``BaseException``: every refusal on this path is a
    ``ValueError`` and the pre-fix ones a ``ValueError`` or ``TypeError``, so the
    narrowest superset that can hold the measurement is ``Exception``.
    """
    try:
        _tick(bundle, **kwargs)
    except Exception as exc:  # noqa: BLE001 - the exception type is the measurement
        return exc
    return None


def _child_refuses(value: Any) -> bool:
    """Whether the active child refuses ``value`` on its own, through its own path.

    Measured by driving a bare :class:`MicroduckPolicy`, not by reading the
    child's source, so the parity cell below grades behaviour rather than
    spelling.
    """
    policy = MicroduckPolicy(session=_StubSession())  # type: ignore[arg-type]
    asyncio.run(policy.get_actions(_obs_dict(), ""))  # configure from metadata
    try:
        asyncio.run(policy.get_actions(_obs_dict(), "", target_velocity=value))
    except Exception:
        return True
    return False


def _auto_switch_body() -> list[ast.stmt]:
    source = textwrap.dedent(inspect.getsource(MicroduckPolicyBundle._auto_switch))
    function = ast.parse(source).body[0]
    assert isinstance(function, ast.FunctionDef)
    return function.body


def _statement_index(predicate) -> int:
    """The index of the first top-level statement in ``_auto_switch`` matching."""
    for index, statement in enumerate(_auto_switch_body()):
        if predicate(ast.unparse(statement)):
            return index
    raise AssertionError("no statement in _auto_switch matched the predicate")


class TestARefusedTickDoesNotMoveTheSelection:
    """The invariant: a velocity the tick cannot honor leaves the skill alone."""

    @pytest.mark.parametrize("value", _REFUSED)
    def test_the_tick_is_refused(self, value):
        assert isinstance(_tick_error(_bundle(active=_MOVE_KEY), target_velocity=value), ValueError)

    @pytest.mark.parametrize("value", _REFUSED)
    def test_the_active_skill_is_unchanged_from_move(self, value):
        bundle = _bundle(active=_MOVE_KEY)
        _tick_error(bundle, target_velocity=value)
        assert bundle.active == _MOVE_KEY

    @pytest.mark.parametrize("value", _REFUSED)
    def test_the_active_skill_is_unchanged_from_idle(self, value):
        # The other direction: a magnitude the gate would have read as moving
        # must not pull the bundle out of idle on a tick that fails.
        bundle = _bundle(active=_IDLE_KEY)
        _tick_error(bundle, target_velocity=value)
        assert bundle.active == _IDLE_KEY

    def test_the_selection_does_not_move_across_a_later_tick_either(self):
        # The failure mode that made this a defect rather than a lost tick: the
        # gate is skipped when no velocity is carried, so a flip it made on a
        # refused tick was still commanding the robot on every tick after.
        bundle = _bundle(active=_MOVE_KEY)
        assert isinstance(_tick_error(bundle, target_velocity=[float("nan"), 0.0, 0.0]), ValueError)
        _tick(bundle)
        assert bundle.active == _MOVE_KEY

    @pytest.mark.parametrize("value", _REFUSED)
    def test_no_held_skill_ran(self, value):
        # A refused tick must not have handed the value to a policy either.
        bundle = _bundle(active=_MOVE_KEY)
        _tick_error(bundle, target_velocity=value)
        assert all(pol._session.last_input is None for pol in bundle._policies.values())  # type: ignore[union-attr]


class TestTheRefusalNamesTheBundleAndTheParameter:
    """The diagnosis half: a bare numpy message named neither."""

    @pytest.mark.parametrize("value", _REFUSED)
    def test_the_message_names_the_surface_the_caller_called(self, value):
        error = _tick_error(_bundle(active=_MOVE_KEY), target_velocity=value)
        assert str(error).startswith(_BUNDLE_PREFIX), str(error)

    @pytest.mark.parametrize("value", _REFUSED)
    def test_the_message_names_the_parameter(self, value):
        error = _tick_error(_bundle(active=_MOVE_KEY), target_velocity=value)
        assert "target_velocity" in str(error), str(error)

    @pytest.mark.parametrize("value", [pytest.param({"vx": 1.0}, id="mapping")])
    def test_a_mapping_is_the_documented_error_type(self, value):
        # This one used to be a ``TypeError`` out of ``float()`` inside numpy,
        # where every documented refusal in this family is a ``ValueError``.
        assert type(_tick_error(_bundle(active=_MOVE_KEY), target_velocity=value)) is ValueError

    def test_a_wrong_width_says_which_widths_it_expects(self):
        error = _tick_error(_bundle(active=_MOVE_KEY), target_velocity=[0.5])
        text = " ".join(str(error).split())
        assert "target_velocity has 1 component(s)" in text, text
        for width in TARGET_VELOCITY_WIDTHS:
            assert str(width) in text, text


class TestTheGateRefusesExactlyWhatTheTickRefuses:
    """Parity, derived: the gate's verdict equals the child's on every value.

    This is what makes the width question the gate's business as well as the
    child's. Refusing less leaves a refused tick moving the selection; refusing
    more would reject a goal a caller may legitimately ask for. Asserted as an
    equality over both grids rather than as two lists, so a future widening on
    either side has to move both.
    """

    @pytest.mark.parametrize("value", _REFUSED)
    def test_the_gate_refuses_a_value_the_tick_refuses(self, value):
        gate_refuses = composite_mod._target_velocity_error(_BUNDLE_PREFIX, value) is not None
        assert gate_refuses is _child_refuses(value) is True

    @pytest.mark.parametrize("value,expected", _ACCEPTED)
    def test_the_gate_accepts_a_value_the_tick_accepts(self, value, expected):
        gate_refuses = composite_mod._target_velocity_error(_BUNDLE_PREFIX, value) is not None
        assert gate_refuses is _child_refuses(value) is False


class TestTheChildsRefusalSetIsThePremise:
    """What the child does on its own. Holds on the pre-fix code too.

    The gate is being held to this set, so the set itself is worth stating: it
    is measured by driving a bare :class:`MicroduckPolicy`, not by reading the
    child's source, so the parity above grades behaviour rather than spelling.
    """

    @pytest.mark.parametrize("value", _REFUSED)
    def test_the_tick_alone_refuses_it(self, value):
        assert _child_refuses(value)

    @pytest.mark.parametrize("value,expected", _ACCEPTED)
    def test_the_tick_alone_accepts_it(self, value, expected):
        assert not _child_refuses(value)

    def test_the_refused_set_covers_every_kind_the_two_guards_answer(self):
        # Non-vacuity: the grid above must keep reaching all three failure kinds,
        # so it cannot shrink to one and still look exhaustive.
        values = [param.values[0] for param in _REFUSED]
        non_finite = [
            v for v in values if isinstance(v, list) and any(isinstance(c, float) and not math.isfinite(c) for c in v)
        ]
        non_numeric = [v for v in values if isinstance(v, str | dict)]
        wrong_width = [v for v in values if isinstance(v, list) and len(v) not in TARGET_VELOCITY_WIDTHS]
        assert non_finite and non_numeric and wrong_width


class TestWhatIsUnchangedEitherWay:
    """The boundary. Every cell here holds on the pre-fix code too."""

    @pytest.mark.parametrize("value,expected", _ACCEPTED)
    def test_a_readable_velocity_still_arbitrates(self, value, expected):
        bundle = _bundle(active=_MOVE_KEY if expected == _IDLE_KEY else _IDLE_KEY)
        _tick(bundle, target_velocity=value)
        assert bundle.active == expected

    def test_the_threshold_is_still_inclusive_at_its_own_value(self):
        bundle = _bundle(active=_IDLE_KEY)
        _tick(bundle, target_velocity=[_GATE_THRESHOLD, 0.0, 0.0])
        assert bundle.active == _MOVE_KEY

    def test_an_absent_velocity_is_still_no_goal_rather_than_a_refusal(self):
        # ``None`` reaches the shared domain as a refusal, so the early return
        # for it has to stay above the guard: a tick that carries no goal is the
        # ordinary case, not a bad velocity.
        bundle = _bundle(active=_MOVE_KEY)
        assert _tick_error(bundle, target_velocity=None) is None
        assert bundle.active == _MOVE_KEY

    def test_a_tick_with_no_velocity_kwarg_at_all_is_still_accepted(self):
        bundle = _bundle(active=_MOVE_KEY)
        assert _tick_error(bundle) is None
        assert bundle.active == _MOVE_KEY

    def test_the_gate_being_off_leaves_the_refusal_to_the_child(self):
        # With no threshold ``_auto_switch`` is never called, so the value is
        # forwarded untouched and the child is the one that refuses it - the
        # selection was never at risk on that path, and still is not.
        bundle = _bundle(active=_MOVE_KEY, gate=None)
        assert isinstance(_tick_error(bundle, target_velocity=[float("nan"), 0.0, 0.0]), ValueError)
        assert bundle.active == _MOVE_KEY

    def test_a_third_skill_active_is_still_left_alone(self):
        bundle = _bundle(active="sitstand")
        _tick_error(bundle, target_velocity=[float("nan"), 0.0, 0.0])
        assert bundle.active == "sitstand"


class TestTheGuardIsAskedBeforeTheSelectionMoves:
    """Structural: ordering is the whole point, so it is asserted directly."""

    def test_the_guard_precedes_the_magnitude(self):
        guard = _statement_index(lambda text: "_target_velocity_error" in text)
        magnitude = _statement_index(lambda text: "linalg.norm" in text)
        assert guard < magnitude

    def test_the_guard_precedes_the_assignment_that_moves_the_selection(self):
        guard = _statement_index(lambda text: "_target_velocity_error" in text)
        assignment = _statement_index(lambda text: "self._active =" in text)
        assert guard < assignment

    def test_the_absent_velocity_return_precedes_the_guard(self):
        early = _statement_index(lambda text: "target_velocity is None" in text)
        guard = _statement_index(lambda text: "_target_velocity_error" in text)
        assert early < guard


class TestTheDomainIsTheChildsOwn:
    """Single-sourced: the gate and the tick read one helper and one constant."""

    def test_the_helper_consults_the_shared_vector_domain(self):
        assert "finite_vector_error" in inspect.getsource(composite_mod._target_velocity_error)

    def test_the_width_refusal_reads_the_shared_constant(self):
        # Not a restated literal: widening the child's accepted spellings widens
        # the gate's, so the two cannot drift apart on what a velocity is.
        assert "TARGET_VELOCITY_WIDTHS" in inspect.getsource(composite_mod._target_velocity_error)

    def test_the_constant_is_the_childs_own_object(self):
        assert composite_mod.TARGET_VELOCITY_WIDTHS is policy_mod.TARGET_VELOCITY_WIDTHS
