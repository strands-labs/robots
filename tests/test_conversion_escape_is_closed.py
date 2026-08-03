"""Regression tests: the ``float()`` conversion escape is closed (#1874).

Four scalar guards in :mod:`strands_robots.utils` establish their domain by
converting the caller's value with ``float()``. That conversion raises
``OverflowError`` for a real whose magnitude exceeds ``sys.float_info.max``, and
it runs *before* any message is rendered - so #1873, which made the rendering
total, could not reach it. This is the other half of that pair, and with it the
scalar numeric family answers every real scalar it is given.

Measured on ``2c254c7`` (i.e. with #1873 already merged), each guard called with
its real signature. ``R:`` is a raise:

| guard | ``10**400`` | ``10**5000`` | ``Fraction(10**400, 3)`` | ``RealNoFloat()`` |
| --- | --- | --- | --- | --- |
| ``positive_finite_number_error`` | R:Overflow | R:Overflow | R:Overflow | R:TypeError |
| ``finite_number_error`` | R:Overflow | R:Overflow | R:Overflow | R:TypeError |
| ``positive_whole_number_error`` | R:Overflow | R:Overflow | R:Overflow | R:TypeError |
| ``camera_fov_error`` | R:Overflow | R:Overflow | R:Overflow | R:TypeError |

Two columns are not in #1874's description of the defect and change what the fix
has to be:

* **It is not an ``int`` problem.** ``Fraction(10**400, 3)`` is a registered
  :class:`numbers.Real` that overflows identically, so the guard has to ask the
  question of the *conversion* rather than of the type. A fix keyed on
  ``isinstance(value, int)`` would have left the ``Fraction`` column raising.
* **The escape has two exception classes.** A ``numbers.Real`` registration
  guarantees no working ``__float__``, so ``float()`` can raise ``TypeError``
  here as well - and that is not a magnitude complaint and must not be reported
  as one. :class:`TestAValueNoNumberCanBeReadFromKeepsTheOldReason` pins the
  difference.

## The verdict is preserved; only the raise becomes an answer

Every value these guards accepted is still accepted and every value they refused
is still refused with byte-identical text - the whole change is that 24 probes
that used to raise now return a string. That is measured rather than asserted:
:class:`TestNoExistingVerdictOrMessageMoved` walks a control matrix per guard.

## Why the refusal names the float64 range

Three of the four needed new text, because their own reason would have been a
false statement about the value: ``10**400`` *is* positive, *is* finite, and *is*
a positive whole number. The new reason names the float64 range, and that is not
a bound this change invents - it is the bound the guards already enforced. Each
accepts ``1e300`` and ``10**308`` today and raised one step past
``sys.float_info.max``, so the range is where the accepted domain already ended;
all that changes is that the edge now answers instead of raising.
:class:`TestTheRangeIsTheBoundTheGuardAlreadyHad` is that argument as a test.

``camera_fov_error`` is the exception and needed no new text: its domain is
bounded above at 180 degrees, so "outside the open interval (0, 180)" is already
a true statement about a value past the float64 range, and the overflow alone
establishes it without a comparison.

## The one deliberate asymmetry

``positive_whole_number_error`` refuses an outsized value where its sibling
``non_negative_whole_number_error`` accepts one, and the two are documented as
the same policy with the floor moved. The difference is the consumer: MuJoCo's
``_MAX_STEPS_PER_CALL`` bounds a step count with a reason of its own, while this
domain's callers include the mesh robots' ``drive(count=...)``, which repeats an
actuation command - so an unbounded count is an unbounded actuation loop against
a physical robot. :class:`TestTheAsymmetryWithTheSiblingIsDeliberate` pins it so
a later reader finds a decision rather than an inconsistency.
"""

from __future__ import annotations

import ast
import inspect
import math
import numbers
import sys
from collections.abc import Callable
from fractions import Fraction
from typing import Any, NamedTuple

import numpy as np
import pytest

from strands_robots import utils
from strands_robots.utils import (
    _beyond_float_range,
    camera_fov_error,
    finite_number_error,
    non_negative_whole_number_error,
    positive_finite_number_error,
    positive_whole_number_error,
)

# --------------------------------------------------------------------------- #
# Probes                                                                      #
# --------------------------------------------------------------------------- #
#: Past ``sys.float_info.max`` but inside the interpreter's digit limit, so its
#: own ``repr`` works. Isolates the conversion from #1873's rendering defect.
BEYOND_FLOAT_RANGE = 10**400

#: Past the digit limit too, so it exercises the conversion *and* #1873's
#: fallback in one value - the two fixes have to compose.
BEYOND_INT_STR_LIMIT = 10**5000

#: The largest value the conversion still accepts. The boundary is inclusive.
FLOAT_MAX = sys.float_info.max

#: An ``int`` inside the float64 range, to show the refusal is about magnitude
#: and not about being an ``int`` or being large.
LARGE_BUT_CONVERTIBLE = 10**308

NAN = float("nan")
INF = float("inf")


class RealNoFloat:
    """A registered :class:`numbers.Real` from which no float can be read.

    Registered with ``numbers.Real.register`` rather than subclassed, which is
    not a shortcut - it is the property under test. ``numbers.Real`` is a
    registration rather than an inheritance, so a value can satisfy a guard's
    ``isinstance`` check while owing it nothing else, and subclassing the ABC
    would have forced this double to implement two dozen operators it never uses
    and to supply the very ``__float__`` whose absence is the point.

    With no ``__float__`` at all, ``float()`` raises ``TypeError`` of its own
    accord. That is the conversion's second exception class, and unlike an
    outsized magnitude it is not a range complaint: no number can be read from
    this value, so it must not be told about the float64 range. Its ``repr``
    works, which isolates the conversion from #1873's rendering defect.
    """

    def __repr__(self) -> str:
        return "RealNoFloat()"


numbers.Real.register(RealNoFloat)


#: Every value that used to raise out of a converting guard. The ``Fraction`` and
#: the ``RealNoFloat`` are the two cells #1874's description does not cover.
OVERFLOWING_PROBES: tuple[tuple[str, Any], ...] = (
    ("10**400", BEYOND_FLOAT_RANGE),
    ("-10**400", -BEYOND_FLOAT_RANGE),
    ("10**5000", BEYOND_INT_STR_LIMIT),
    ("-10**5000", -BEYOND_INT_STR_LIMIT),
    ("Fraction(10**400, 3)", Fraction(BEYOND_FLOAT_RANGE, 3)),
)
OVERFLOWING_IDS = tuple(name for name, _ in OVERFLOWING_PROBES)
OVERFLOWING_VALUES = tuple(value for _, value in OVERFLOWING_PROBES)


class Converting(NamedTuple):
    """One guard that classifies by converting with ``float()``.

    Attributes:
        name: The function's name, used as the test ID.
        call: The guard with its label arguments bound, so a probe goes to the
            value position - first for the numeric family, third for
            ``camera_fov_error``.
        range_refusal: The message the guard gives for a value past the float64
            range, as a callable taking the value. ``camera_fov_error`` is the
            one member that reuses a message it already had.
        old_reason: The message the guard gives for a value no number can be read
            from - text that predates this change on every guard.
        controls: ``(value, expected)`` pairs, where ``expected`` is ``None`` for
            an accepted value and the exact message otherwise. Per guard, because
            the family does not share a domain.
    """

    name: str
    call: Callable[[Any], str | None]
    range_refusal: Callable[[Any], str]
    old_reason: str
    controls: tuple[tuple[Any, str | None], ...]


CONVERTING: tuple[Converting, ...] = (
    Converting(
        "positive_finite_number_error",
        lambda v: positive_finite_number_error(v, "hz", "teleoperate"),
        lambda v: f"teleoperate: hz must be within the range of a 64-bit float, got {v!r}.",
        "teleoperate: hz must be > 0, got RealNoFloat().",
        (
            (2.5, None),
            (30.0, None),
            (FLOAT_MAX, None),
            (LARGE_BUT_CONVERTIBLE, None),
            (np.float32(58.0), None),
            (np.int64(30), None),
            (0, "teleoperate: hz must be > 0, got 0."),
            (-5, "teleoperate: hz must be > 0, got -5."),
            (NAN, "teleoperate: hz must be > 0, got nan."),
            (INF, "teleoperate: hz must be > 0, got inf."),
            (-INF, "teleoperate: hz must be > 0, got -inf."),
            (True, "teleoperate: hz must be > 0, got True."),
            ("x", "teleoperate: hz must be > 0, got 'x'."),
        ),
    ),
    Converting(
        "finite_number_error",
        lambda v: finite_number_error(v, "vx", "drive"),
        lambda v: f"drive: vx must be within the range of a 64-bit float, got {v!r}.",
        "drive: vx must be a finite number, got RealNoFloat().",
        (
            (2.5, None),
            (0, None),
            (-5, None),
            (FLOAT_MAX, None),
            (-FLOAT_MAX, None),
            (LARGE_BUT_CONVERTIBLE, None),
            (np.float32(58.0), None),
            (NAN, "drive: vx must be a finite number, got nan."),
            (INF, "drive: vx must be a finite number, got inf."),
            (-INF, "drive: vx must be a finite number, got -inf."),
            (True, "drive: vx must be a finite number, got True."),
            ("x", "drive: vx must be a finite number, got 'x'."),
        ),
    ),
    Converting(
        "positive_whole_number_error",
        lambda v: positive_whole_number_error(v, "fps", "video"),
        lambda v: f"video: fps must be within the range of a 64-bit float, got {v!r}.",
        "video: fps must be a positive whole number, got RealNoFloat().",
        (
            (30.0, None),
            (FLOAT_MAX, None),
            (LARGE_BUT_CONVERTIBLE, None),
            (np.int64(30), None),
            (2.5, "video: fps must be a positive whole number, got 2.5."),
            (0, "video: fps must be a positive whole number, got 0."),
            (-5, "video: fps must be a positive whole number, got -5."),
            (NAN, "video: fps must be a positive whole number, got nan."),
            (INF, "video: fps must be a positive whole number, got inf."),
            (True, "video: fps must be a positive whole number, got True."),
            ("x", "video: fps must be a positive whole number, got 'x'."),
        ),
    ),
    Converting(
        "camera_fov_error",
        lambda v: camera_fov_error("add_camera", "fov", v),
        # The one member that needed no new text: its domain is bounded above, so
        # the interval message it already had is true of an outsized value. Note
        # ``str`` rather than ``repr`` - that branch renders plainly, and #1873
        # pins why converting it would change an ``np.float32`` fov's text.
        lambda v: f"add_camera: 'fov' must be in the open interval (0, 180) degrees, got {v!s}.",
        "add_camera: 'fov' must be a finite number in degrees, got RealNoFloat().",
        (
            (2.5, None),
            (58.0, None),
            (np.float32(58.0), None),
            (np.int64(30), None),
            (0, "add_camera: 'fov' must be in the open interval (0, 180) degrees, got 0."),
            (-5, "add_camera: 'fov' must be in the open interval (0, 180) degrees, got -5."),
            (200.0, "add_camera: 'fov' must be in the open interval (0, 180) degrees, got 200.0."),
            (NAN, "add_camera: 'fov' must be a finite number in degrees, got nan."),
            (INF, "add_camera: 'fov' must be a finite number in degrees, got inf."),
            (True, "add_camera: 'fov' must be a finite number in degrees, got True."),
            ("x", "add_camera: 'fov' must be a finite number in degrees, got 'x'."),
        ),
    ),
)
CONVERTING_IDS = tuple(guard.name for guard in CONVERTING)


# --------------------------------------------------------------------------- #
# The helper                                                                  #
# --------------------------------------------------------------------------- #
class TestTheSharedPredicate:
    """``_beyond_float_range`` answers one question and does not widen it."""

    @pytest.mark.parametrize("value", OVERFLOWING_VALUES[:4], ids=OVERFLOWING_IDS[:4])
    def test_it_is_true_for_a_magnitude_past_the_float_range(self, value: Any) -> None:
        assert _beyond_float_range(value) is True

    def test_it_is_true_for_a_non_int_real_that_overflows(self) -> None:
        """Keyed on the conversion, not on the type - see the module docstring."""
        assert _beyond_float_range(Fraction(BEYOND_FLOAT_RANGE, 3)) is True

    @pytest.mark.parametrize(
        "value",
        [0, 1, -1, 2.5, FLOAT_MAX, -FLOAT_MAX, LARGE_BUT_CONVERTIBLE, NAN, INF, -INF, np.float32(1.0), np.int64(3)],
    )
    def test_it_is_false_for_everything_a_float_can_hold(self, value: Any) -> None:
        assert _beyond_float_range(value) is False

    def test_it_is_false_when_the_conversion_fails_some_other_way(self) -> None:
        """A ``TypeError`` is not a magnitude complaint, so it is not this one.

        Reporting "must be within the range of a 64-bit float" for a value that
        has no ``__float__`` at all would be as false as the reasons this change
        exists to stop giving.
        """
        assert _beyond_float_range(RealNoFloat()) is False
        assert _beyond_float_range("x") is False
        assert _beyond_float_range(None) is False

    def test_the_boundary_is_inclusive(self) -> None:
        """``float_info.max`` itself converts; only past it overflows."""
        assert _beyond_float_range(FLOAT_MAX) is False
        assert _beyond_float_range(int(FLOAT_MAX) * 2) is True

    def test_it_does_not_swallow_a_base_exception(self) -> None:
        """Same boundary #1873 drew on the renderers, and for the same reason.

        Swallowing a ``KeyboardInterrupt`` to classify a number would be a worse
        failure than the one being reported.
        """

        class Interrupting:
            """Registered below, for the reason given on :class:`RealNoFloat`."""

            def __float__(self) -> float:
                raise KeyboardInterrupt

        numbers.Real.register(Interrupting)

        with pytest.raises(KeyboardInterrupt):
            _beyond_float_range(Interrupting())


# --------------------------------------------------------------------------- #
# The closure                                                                 #
# --------------------------------------------------------------------------- #
class TestEveryConvertingGuardAnswersAValueItCannotConvert:
    """The invariant, over all four guards and every probe that used to raise."""

    @pytest.mark.parametrize("guard", CONVERTING, ids=CONVERTING_IDS)
    @pytest.mark.parametrize("value", OVERFLOWING_VALUES, ids=OVERFLOWING_IDS)
    def test_it_returns_a_message_instead_of_raising(self, guard: Converting, value: Any) -> None:
        result = guard.call(value)
        assert isinstance(result, str)
        assert result

    @pytest.mark.parametrize("guard", CONVERTING, ids=CONVERTING_IDS)
    def test_the_message_names_the_parameter_and_the_surface(self, guard: Converting) -> None:
        """A refusal that does not say what was refused is not an answer."""
        assert guard.call(BEYOND_FLOAT_RANGE) == guard.range_refusal(BEYOND_FLOAT_RANGE)

    @pytest.mark.parametrize("guard", CONVERTING, ids=CONVERTING_IDS)
    def test_it_composes_with_the_rendering_fix(self, guard: Converting) -> None:
        """A value past *both* boundaries needs #1873 and #1874 at once.

        ``10**5000`` overflows the conversion and cannot render itself either, so
        the refusal is only producible if the range branch routes through the
        shared renderer rather than interpolating the value.
        """
        result = guard.call(BEYOND_INT_STR_LIMIT)
        assert isinstance(result, str)
        assert "<int of 16610 bits>" in result


class TestAValueNoNumberCanBeReadFromKeepsTheOldReason:
    """The second exception class is not a range complaint (module docstring).

    ``float()`` raises ``TypeError`` for a registered :class:`numbers.Real` with
    no working ``__float__``. Such a value used to raise out of all four guards
    too, so it is fixed here - but reporting a *range* for it would be exactly
    the kind of false reason this change exists to remove.
    """

    @pytest.mark.parametrize("guard", CONVERTING, ids=CONVERTING_IDS)
    def test_it_is_refused_with_the_text_that_predates_this_change(self, guard: Converting) -> None:
        assert guard.call(RealNoFloat()) == guard.old_reason

    @pytest.mark.parametrize("guard", CONVERTING, ids=CONVERTING_IDS)
    def test_it_is_not_told_about_the_float_range(self, guard: Converting) -> None:
        result = guard.call(RealNoFloat())
        assert result is not None
        assert "64-bit float" not in result


# --------------------------------------------------------------------------- #
# Why the new reason is the honest one                                        #
# --------------------------------------------------------------------------- #
class TestTheRangeIsTheBoundTheGuardAlreadyHad:
    """The new message names an existing boundary rather than inventing one.

    This is the argument for the wording. If these guards refused well below the
    float64 range, naming it would be arbitrary; they do not - they accept right
    up to ``sys.float_info.max`` and used to raise one step past it.
    """

    @pytest.mark.parametrize("guard", CONVERTING[:3], ids=CONVERTING_IDS[:3])
    def test_the_largest_convertible_value_is_still_accepted(self, guard: Converting) -> None:
        assert guard.call(FLOAT_MAX) is None

    @pytest.mark.parametrize("guard", CONVERTING[:3], ids=CONVERTING_IDS[:3])
    def test_a_large_int_inside_the_range_is_still_accepted(self, guard: Converting) -> None:
        """So the refusal is about magnitude, not about being an ``int``."""
        assert guard.call(LARGE_BUT_CONVERTIBLE) is None

    @pytest.mark.parametrize("guard", CONVERTING[:3], ids=CONVERTING_IDS[:3])
    def test_the_refusal_begins_exactly_where_the_conversion_fails(self, guard: Converting) -> None:
        just_past = int(FLOAT_MAX) * 2
        assert guard.call(FLOAT_MAX) is None
        assert guard.call(just_past) == guard.range_refusal(just_past)

    def test_the_reason_would_have_been_false_of_the_value(self) -> None:
        """Why new text was needed at all, stated as three true propositions."""
        assert BEYOND_FLOAT_RANGE > 0, "so 'must be > 0' would be false of it"
        assert BEYOND_FLOAT_RANGE == int(BEYOND_FLOAT_RANGE), "so 'positive whole number' would be false"
        assert not isinstance(BEYOND_FLOAT_RANGE, float), "and it is finite: no float64 is involved"


class TestTheFovGuardNeededNoNewText:
    """The one member with a bounded domain, so it already owned a true reason."""

    @pytest.mark.parametrize("value", OVERFLOWING_VALUES, ids=OVERFLOWING_IDS)
    def test_an_outsized_angle_is_refused_as_outside_the_interval(self, value: Any) -> None:
        result = camera_fov_error("add_camera", "fov", value)
        assert result is not None
        assert "open interval (0, 180) degrees" in result

    @pytest.mark.parametrize("value", OVERFLOWING_VALUES, ids=OVERFLOWING_IDS)
    def test_it_never_mentions_the_float_range(self, value: Any) -> None:
        """New text here would have been a worse message, not a better one."""
        result = camera_fov_error("add_camera", "fov", value)
        assert result is not None
        assert "64-bit float" not in result

    def test_it_is_the_same_message_an_ordinary_out_of_range_angle_gets(self) -> None:
        """Byte-identical but for the value, so the two are one reason not two."""
        outsized = camera_fov_error("add_camera", "fov", BEYOND_FLOAT_RANGE)
        ordinary = camera_fov_error("add_camera", "fov", 200.0)
        assert outsized is not None and ordinary is not None
        prefix = "add_camera: 'fov' must be in the open interval (0, 180) degrees, got "
        assert outsized.startswith(prefix)
        assert ordinary.startswith(prefix)

    def test_the_claim_the_reason_rests_on(self) -> None:
        """A magnitude past the float64 range really is outside ``(0, 180)``.

        The guard asserts this without comparing, so the comparison lives here.
        """
        assert not (0 < BEYOND_FLOAT_RANGE < 180)
        assert not (0 < -BEYOND_FLOAT_RANGE < 180)
        assert not (0 < Fraction(BEYOND_FLOAT_RANGE, 3) < 180)


# --------------------------------------------------------------------------- #
# Nothing else moved                                                          #
# --------------------------------------------------------------------------- #
class TestNoExistingVerdictOrMessageMoved:
    """The change is additive: 24 raises became answers and nothing else.

    Without this the class above is satisfied by a guard that started refusing
    everything, which is why an accepted-value control sits in every row.
    """

    @pytest.mark.parametrize("guard", CONVERTING, ids=CONVERTING_IDS)
    def test_every_control_keeps_its_exact_verdict_and_text(self, guard: Converting) -> None:
        for value, expected in guard.controls:
            assert guard.call(value) == expected, f"{guard.name} moved on {value!r}"

    @pytest.mark.parametrize("guard", CONVERTING, ids=CONVERTING_IDS)
    def test_at_least_one_control_is_accepted_and_one_refused(self, guard: Converting) -> None:
        """Non-vacuity of the row above."""
        verdicts = {expected is None for _, expected in guard.controls}
        assert verdicts == {True, False}

    def test_the_sibling_guards_this_change_does_not_touch_are_unmoved(self) -> None:
        """#1873's family is wider than #1874's; the rest must be untouched."""
        assert non_negative_whole_number_error(2.7, "n_steps", "step") == (
            "step: n_steps must be a non-negative whole number, got 2.7."
        )
        assert non_negative_whole_number_error(0, "n_steps", "step") is None


class TestTheAsymmetryWithTheSiblingIsDeliberate:
    """``positive_`` refuses an outsized value where ``non_negative_`` accepts one.

    The two are documented as the same scalar policy with the floor moved, so a
    reader who measures this will find a contradiction unless the exception is
    recorded. It is a decision, not an oversight: the consumer of a step count
    owns a ceiling (MuJoCo's ``_MAX_STEPS_PER_CALL``) and no consumer of an
    ``fps`` / ``width`` / ``drive(count=...)`` does, and that last one repeats a
    command to a robot.
    """

    def test_the_positive_guard_refuses_an_outsized_value(self) -> None:
        assert positive_whole_number_error(BEYOND_FLOAT_RANGE, "fps", "video") is not None

    def test_the_non_negative_sibling_still_accepts_one(self) -> None:
        assert non_negative_whole_number_error(BEYOND_FLOAT_RANGE, "n_steps", "step") is None

    def test_they_agree_on_every_other_probe(self) -> None:
        """The asymmetry is exactly one input class wide, not a general drift.

        The floor is excluded because it is the *other*, already-documented
        difference between the two guards - ``0`` is accepted by the
        ``non_negative`` one by definition. Everything else must match.
        """
        for value in (1, 2.5, 30.0, NAN, INF, -1, True, "x", np.int64(7)):
            positive = positive_whole_number_error(value, "n", "ctx") is None
            non_negative = non_negative_whole_number_error(value, "n", "ctx") is None
            assert positive == non_negative, f"the two guards disagree on {value!r}"

    def test_the_floor_is_the_other_difference_and_is_excluded_deliberately(self) -> None:
        """So the exclusion above is a stated decision, not a probe quietly dropped."""
        assert positive_whole_number_error(0, "n", "ctx") is not None
        assert non_negative_whole_number_error(0, "n", "ctx") is None

    def test_the_reason_for_the_asymmetry_is_recorded_where_a_reader_will_look(self) -> None:
        """Both docstrings must carry it, since either is the one being read."""
        assert "_MAX_STEPS_PER_CALL" in (positive_whole_number_error.__doc__ or "")
        assert "drive(count=" in (positive_whole_number_error.__doc__ or "")
        assert "the one place the two guards differ" in (non_negative_whole_number_error.__doc__ or "")


# --------------------------------------------------------------------------- #
# Drift: no guard may convert without a guard                                 #
# --------------------------------------------------------------------------- #
def _unguarded_float_calls_in(tree: ast.AST) -> list[str]:
    """Names converted by a ``float()`` call that no ``try`` protects.

    The scan is what makes this the last pass rather than a first: a fifth guard,
    or a later edit to one of these four, reintroduces the defect silently
    otherwise. It reports the *argument* of each unprotected conversion so a
    failure says which value is exposed rather than only that one is.

    A call is protected when it is lexically inside a ``try`` body, or when it is
    the argument to :func:`_beyond_float_range` - which is itself protected, and
    is the shared spelling this change introduces.

    Args:
        tree: A parsed function, or the ``ast`` node for one.
    """
    protected: set[ast.AST] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for stmt in node.body:
                for inner in ast.walk(stmt):
                    if isinstance(inner, ast.Call):
                        protected.add(inner)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "_beyond_float_range":
            for arg in node.args:
                for inner in ast.walk(arg):
                    if isinstance(inner, ast.Call):
                        protected.add(inner)

    exposed = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "float"
            and node not in protected
            and node.args
        ):
            arg = node.args[0]
            exposed.append(arg.id if isinstance(arg, ast.Name) else ast.dump(arg))
    return exposed


def _unguarded_float_calls(func: Any) -> list[str]:
    """:func:`_unguarded_float_calls_in` for a live function object."""
    return _unguarded_float_calls_in(ast.parse(inspect.getsource(func).lstrip()))


class TestNoConvertingGuardConvertsUnprotected:
    """The invariant is scanned in the source, not enumerated in a list."""

    @pytest.mark.parametrize("guard", CONVERTING, ids=CONVERTING_IDS)
    def test_the_guard_has_no_unprotected_conversion(self, guard: Converting) -> None:
        func = getattr(utils, guard.name)
        assert _unguarded_float_calls(func) == []

    def test_the_helper_itself_is_protected(self) -> None:
        """It is the one place the conversion is allowed to be attempted."""
        assert _unguarded_float_calls(_beyond_float_range) == []

    @pytest.mark.parametrize("guard", CONVERTING, ids=CONVERTING_IDS)
    def test_the_guard_asks_the_shared_predicate(self, guard: Converting) -> None:
        """The positive form: absence of a bare ``float()`` is also satisfied by
        a guard that stopped classifying magnitude at all."""
        source = inspect.getsource(getattr(utils, guard.name))
        assert "_beyond_float_range(" in source

    def test_the_scanner_reports_a_planted_unprotected_conversion(self) -> None:
        """Control: an always-passing scan would satisfy every test above."""

        def planted(value: Any) -> bool:
            return math.isfinite(float(value))

        assert _unguarded_float_calls(planted) == ["value"]

    def test_the_scanner_accepts_a_conversion_inside_a_try(self) -> None:
        """Control in the other direction, so the scan is not simply strict."""

        def planted(value: Any) -> float:
            try:
                return float(value)
            except OverflowError:
                return 0.0

        assert _unguarded_float_calls(planted) == []


class TestTheModuleConvertsOnlyInsideAGuardedRead:
    """The scan run over the whole module, not over a list of names.

    Replaces ``TestTheRemainingConversionsRunOnlyOnTheAcceptedPath`` rather than
    deleting it, which is the second time this boundary has moved rather than gone:
    that class replaced ``TestTheModuleHasExactlyOneRemainingConversionSurface``,
    which reported the four container guards and pinned them raising (#1875).

    Its statement was that three names remain - ``coerce_pose_vector``,
    ``coerce_rgba`` and ``coerce_size_vector`` - whose ``float()`` the lexical scan
    reports and whose safety is an **upstream** guarantee: ``finite_vector_error``
    had already converted every element successfully, so theirs provably could not
    raise. Since the scan could not see that, the three were pinned with the
    validating call asserted to come first.

    The three no longer convert at all. #1906 moved the element read into
    ``_read_finite_vector``, which returns the floats it built inside its own
    ``try``, so each coercion now keeps a list this module made instead of reading
    the caller's value a second time - which also closed the escape that second
    read was: a value that answered the checked read and refused the next one
    raised out of the coercion, including out of ``coerce_rgba``'s wrong-length
    refusal.

    So the scan's output is empty, and an empty output is also what a scanner
    matching nothing produces. Two statements below are what keep it a
    measurement: the conversions are gone, *and* the floats still come from the
    guarded read - a coercion that stopped producing floats would satisfy the
    first alone.
    """

    #: Not names this change chose to skip - this is the scan's output, asserted
    #: so it can neither grow nor be quietly narrowed. Empty since #1906.
    EXPECTED_REMAINING: frozenset[str] = frozenset()

    #: The coercions that used to convert, and the guarded reads they now take
    #: their floats from. Two spellings, because ``coerce_pose_vector`` reaches the
    #: element read through the fixed-length wrapper.
    COERCIONS = ("coerce_pose_vector", "coerce_rgba", "coerce_size_vector")
    GUARDED_READS = frozenset({"_read_finite_vector", "_read_pose_vector"})

    @staticmethod
    def _converting_unprotected() -> set[str]:
        tree = ast.parse(inspect.getsource(utils))
        return {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _unguarded_float_calls_in(node)
        }

    def test_no_function_in_the_module_converts_unprotected(self) -> None:
        assert self._converting_unprotected() == set(self.EXPECTED_REMAINING)

    def test_no_scalar_guard_is_among_them(self) -> None:
        """#1874's claim, stated over the module rather than over a list."""
        assert self._converting_unprotected().isdisjoint({guard.name for guard in CONVERTING})

    def test_the_vector_guard_no_longer_converts_unprotected(self) -> None:
        """#1875's half of the claim, followed to where the conversion now lives.

        Asserting it of ``finite_vector_error`` alone would be vacuous since #1906:
        that function converts nothing at all now, so the guarantee has to be
        asserted of the helper it delegates its read to, which is the function that
        classifies magnitude and then converts inside a ``try``.
        """
        converting = self._converting_unprotected()
        assert "finite_vector_error" not in converting
        assert "_read_finite_vector" not in converting

    @pytest.mark.parametrize("name", COERCIONS)
    def test_each_coercion_takes_its_floats_from_the_guarded_read(self, name: str) -> None:
        """What replaces the ordering pin, and why the empty scan is not vacuous.

        The ordering assertion said: the validating call is present, and earlier
        than the conversion. There is no longer a conversion to order against, so
        the two halves are stated directly - the coercion calls a guarded read, and
        makes no ``float()`` call of its own. Dropping the read, or converting the
        caller's value again beside it, fails here rather than at a caller.
        """
        fn = ast.parse(inspect.getsource(getattr(utils, name)).lstrip())
        called = {
            node.func.id for node in ast.walk(fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert called & self.GUARDED_READS, f"{name} builds its floats without a guarded read"
        assert "float" not in called, f"{name} converts the caller's value itself"

    def test_the_module_scan_still_reports_a_planted_conversion(self) -> None:
        """Control for the empty output above, which no other row can supply.

        The behavioural scanner controls sit on ``_unguarded_float_calls``; this is
        the node-level form the module scan actually runs, so an
        ``_unguarded_float_calls_in`` that had stopped matching would leave
        ``EXPECTED_REMAINING`` empty and every assertion above passing.
        """
        planted = ast.parse("def planted(value: Any) -> float:\n    return float(value)\n")
        fn = next(node for node in ast.walk(planted) if isinstance(node, ast.FunctionDef))
        assert _unguarded_float_calls_in(fn)

    #: Each container guard called with an outsized element, through the public
    #: entry point a caller actually reaches: ``coerce_pose_vector`` is reached
    #: via ``pose_vector_error``. Every one of them takes the value *last*, after
    #: the method and parameter labels.
    CONTAINER_PROBES: tuple[tuple[str, Callable[[], Any]], ...] = (
        ("finite_vector_error", lambda: utils.finite_vector_error("raycast", "origin", [BEYOND_FLOAT_RANGE])),
        ("pose_vector_error", lambda: utils.pose_vector_error("add_object", "position", [BEYOND_FLOAT_RANGE] * 3, 3)),
        ("coerce_rgba", lambda: utils.coerce_rgba("add_object", "colour", [BEYOND_FLOAT_RANGE] * 4)),
        ("coerce_size_vector", lambda: utils.coerce_size_vector("add_object", "size", [BEYOND_FLOAT_RANGE] * 3)),
    )

    @pytest.mark.parametrize(
        "call",
        [probe for _, probe in CONTAINER_PROBES],
        ids=[name for name, _ in CONTAINER_PROBES],
    )
    def test_every_container_guard_now_answers_an_outsized_element(self, call: Callable[[], Any]) -> None:
        """The inverse of the row this replaces, which pinned all four raising.

        Behaviour is what the invariant is ultimately about, so it is asserted
        directly rather than inferred from the scan: whether a guard's conversion
        is protected locally or upstream, no caller may see an ``OverflowError``.
        """
        result = call()
        message = result if isinstance(result, str) else result[1]
        assert message is not None
        assert "within the range of a 64-bit float" in message

    def test_the_probes_reach_the_conversion_rather_than_a_label_check(self) -> None:
        """Control: each probe must fail *on its element*, not on its own shape.

        These guards take the value in their last position, after two ``str``
        labels. A probe that passed the container in a label position would be
        refused for having the wrong type there - which looks like evidence about
        the element and is not. So the same guards are called with a
        well-formed container and must return an accepted result, proving the
        argument order is right and the outsized element is what the row above
        measures.
        """
        assert utils.finite_vector_error("raycast", "origin", [0.0, 0.0, 1.0]) is None
        assert utils.pose_vector_error("add_object", "position", [0.0, 0.0, 1.0], 3) is None
        assert utils.coerce_rgba("add_object", "colour", [0.5, 0.5, 0.5, 1.0])[1] is None
        assert utils.coerce_size_vector("add_object", "size", [1.0, 1.0, 1.0])[1] is None
