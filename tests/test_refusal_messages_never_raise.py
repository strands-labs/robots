"""Regression tests: a refusal message must not raise while it is being built (#1873).

The guards in :mod:`strands_robots.utils` exist so that a caller's bad value is
reported through a structured ``{status, content}`` result - every one of their
callers documents that result as the only channel a rejected value is reported
on. Building the message is part of that answer, and it was not a total
operation: the guards interpolated the caller's value directly, and rendering a
value can raise.

Two values reach it, neither hypothetical. Rendering an ``int`` wider than
``sys.get_int_max_str_digits()`` (4300 digits by default) raises ``ValueError``,
and ``device_connect``'s ``@rpc()`` surfaces forward a remote caller's number
unchanged while Python integers are arbitrary-precision. And a third-party type
may raise anything from its own ``__repr__``: ``numbers.Real`` is a registration
rather than an inheritance, so a scalar that satisfies a guard's type test owes
it nothing else.

Measured on ``b28c911``, each guard called with its real signature. ``R:`` is a
raise; ``Unprintable()`` is a plain object whose ``__repr__`` raises:

| guard | ``10**400`` | ``-10**400`` | ``10**5000`` | ``-10**5000`` | ``Unprintable()`` |
| --- | --- | --- | --- | --- | --- |
| ``positive_finite_number_error`` | R:Overflow | R:Overflow | R:Overflow | R:Overflow | **R:RuntimeError** |
| ``finite_number_error`` | R:Overflow | R:Overflow | R:Overflow | R:Overflow | **R:RuntimeError** |
| ``positive_whole_number_error`` | R:Overflow | R:Overflow | **R:ValueError** | **R:ValueError** | **R:RuntimeError** |
| ``non_negative_whole_number_error`` | accept | refuse | accept | refuse | refuse |
| ``positive_count_error`` | accept | refuse | accept | **R:ValueError** | **R:RuntimeError** |
| ``non_negative_count_error`` | accept | refuse | accept | **R:ValueError** | **R:RuntimeError** |
| ``tcp_port_error`` | refuse | refuse | **R:ValueError** | **R:ValueError** | **R:RuntimeError** |
| ``camera_fov_error`` | R:Overflow | R:Overflow | R:Overflow | R:Overflow | **R:RuntimeError** |
| ``entity_name_error`` | refuse | refuse | **R:ValueError** | **R:ValueError** | **R:RuntimeError** |

The last column is what makes this one uniform defect rather than a handful of
outsized-integer curiosities: **every guard except the one #1869 rewrote raised
on a value it had already decided to refuse.** That column is
:class:`TestEveryScalarGuardAnswersAValueItCannotRender`, and it fails on pre-fix
code for all eight.

The ``R:ValueError`` cells are the same defect reached by a plain Python ``int``
today, and they are narrower than they look: those guards never convert, so they
raised only on a value they had *refused*, never on one they accepted. That is
why they read as clean - including to #1872, which lists the count pair as out of
scope.

Two further sites render a value with ``str`` rather than ``repr`` and reach the
same escape, since ``str`` falls back to ``__repr__`` when a type defines no
``__str__`` and performs the same decimal conversion for an ``int``:

* ``camera_fov_error``'s open-interval branch. It looks unreachable, because it
  runs only after ``math.isfinite(float(value))`` has succeeded - but a
  registered ``numbers.Real`` whose ``float()`` is finite and outside ``(0,
  180)`` passes that test and is refused here. Pinned in
  :class:`TestTheOpenIntervalBranchRendersPlainlyAndStillCannotRaise`.
* ``validation_split_error``, which renders a ``total_tasks`` read straight out
  of a dataset's ``meta/info.json`` - a JSON integer, so arbitrary-precision.
  Pinned in :class:`TestValidationSplitRendersATaskCountItCannotRender`.

Those two keep ``str`` rather than moving to ``repr``: NumPy 2 reprs a scalar
with its type, so a documented-as-accepted ``np.float32`` fov would start
reporting ``got np.float32(200.0)`` in text an agent reads.

The fix routes all of them through ``_refusal_repr`` / ``_refusal_str``, which
defer to ``repr`` / ``str`` wherever those work. So no verdict and no
agent-visible text changes, which is what
:class:`TestTheTextIsUnchangedForAValueThatCanBeRendered` pins.

The ``R:Overflow`` cells are a *different* escape - the ``float()`` conversion,
which happens before any message is rendered - and survived this change
untouched. They were #1874 and are now closed, so what was a pin of deliberate
scope has become :class:`TestTheConversionEscapeIsClosed`: the class is replaced
rather than deleted, because the boundary it drew is still the useful statement
and simply moved. The
container guards had the same rendering defect but needed a rendering rather than
a fallback, since ``<unrepresentable list>`` erases the elements that print fine.
That was #1875 and is now closed too, so the scope pin here is likewise replaced
by its inverse, :class:`TestTheContainerGuardsAnswerAnUnrenderableElement`; the
rendering itself lives in ``tests/test_container_refusals_render_elementwise.py``.

:class:`TestNoGuardRendersACallerValueDirectly` is what makes this the last pass
over the question rather than a first. It keys on the parameter annotated
``Any``, which is how every guard in this module spells "the caller's value", and
it reports the plain ``{value}`` form as well as ``{value!r}`` - a scanner that
looked only for ``!r`` would have missed both ``str`` sites above, which is how
they went unnoticed until the scan was written.
"""

from __future__ import annotations

import ast
import inspect
import numbers
import pathlib
import sys
import textwrap
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import numpy as np
import pytest

import strands_robots.utils as utils
from strands_robots.utils import (
    _describe_unrenderable,
    _refusal_repr,
    _refusal_str,
    camera_fov_error,
    entity_name_error,
    finite_number_error,
    finite_vector_error,
    name_list_error,
    non_negative_count_error,
    non_negative_whole_number_error,
    pose_vector_error,
    positive_count_error,
    positive_finite_number_error,
    positive_whole_number_error,
    tcp_port_error,
    validation_split_error,
)

NAN = float("nan")
INF = float("inf")

#: An ``int`` wider than the float range. ``float(10**400)`` raises
#: ``OverflowError`` - the conversion escape, which is #1874's and not this
#: change's, because it fires before any message is rendered.
BEYOND_FLOAT_RANGE = 10**400

#: An ``int`` too wide to render: past :func:`sys.get_int_max_str_digits` (4300
#: digits by default), so both ``repr`` and ``str`` raise ``ValueError``. This is
#: the rendering escape this change closes.
BEYOND_INT_STR_LIMIT = 10**5000

#: ``BEYOND_INT_STR_LIMIT`` has this many decimal digits, by construction.
#: Spelled out rather than measured with ``len(str(...))``, which is the very
#: conversion that raises.
BEYOND_INT_STR_LIMIT_DIGITS = 5001


class Unprintable:
    """A plain object whose ``repr`` raises.

    Refused by every guard on type alone, which is why the last column of the
    module docstring's table is uniform. An outsized ``int`` only reaches the
    guards that do not convert first; this reaches all of them, and it is the
    reason the renderer's guarantee is unconditional rather than a list of the
    exceptions known today.
    """

    def __repr__(self) -> str:
        raise RuntimeError("this type cannot render itself")


@numbers.Real.register
class UnrenderableReal:
    """A registered ``numbers.Real``, refused on its value, that cannot render itself.

    ``numbers.Real`` is a registration rather than an inheritance, so this
    satisfies the scalar type test the numeric guards lead with and still owes
    them nothing. ``float()`` is ``nan`` so every guard in the family refuses it
    on its value rather than its type - the path an ``Unprintable`` never takes,
    since that one is turned away by the type test first.
    """

    def __float__(self) -> float:
        return NAN

    def __int__(self) -> int:
        return -1

    def __repr__(self) -> str:
        raise RuntimeError("this scalar cannot render itself")


@numbers.Real.register
class UnrenderableFov:
    """A registered ``numbers.Real`` that is finite and outside ``(0, 180)``.

    The probe that reaches ``camera_fov_error``'s open-interval branch: it passes
    the finiteness branch above it, which is what made that branch look
    unreachable by anything unrenderable.
    """

    def __float__(self) -> float:
        return 200.0

    def __int__(self) -> int:
        return 200

    def __repr__(self) -> str:
        raise RuntimeError("this fov cannot render itself")


class UnrenderableName(str):
    """A ``str`` subclass whose ``repr`` raises.

    ``entity_name_error``'s NUL branch runs only after ``isinstance(name, str)``
    has passed, and a subclass passes it - so "a ``str``'s ``repr`` cannot raise"
    is true of ``str`` and not of the type test guarding that branch.
    """

    def __repr__(self) -> str:
        raise RuntimeError("this name cannot render itself")


class Guard(NamedTuple):
    """One scalar guard, with everything needed to probe it.

    Attributes:
        name: The function's name, used as the test ID.
        param: The parameter label the message must still carry.
        call: The guard with its non-value arguments already bound, so a probe
            can be sent to its value position - first for the numeric family and
            third for the two that lead with the method name.
        refused: Values this guard refuses whose rendering cannot raise. Per
            guard rather than shared, because the family does not agree on a
            domain: ``finite_number_error`` accepts every negative the others
            refuse, and ``entity_name_error`` accepts the string ``"3"``.
    """

    name: str
    param: str
    call: Callable[[Any], str | None]
    refused: tuple[Any, ...]


SCALAR_GUARDS: tuple[Guard, ...] = (
    Guard(
        "positive_finite_number_error",
        "hz",
        lambda v: positive_finite_number_error(v, "hz", "teleoperate"),
        (0, -1.0, NAN, INF, None, "3", [3], True, np.float32(-1.0), np.int64(-1)),
    ),
    Guard(
        "finite_number_error",
        "linear",
        lambda v: finite_number_error(v, "linear", "drive"),
        (NAN, INF, -INF, None, "3", [3], True),
    ),
    Guard(
        "positive_whole_number_error",
        "fps",
        lambda v: positive_whole_number_error(v, "fps", "video"),
        (0, -1, 2.7, NAN, INF, None, "3", [3], True, np.int64(-1)),
    ),
    Guard(
        "non_negative_whole_number_error",
        "n_steps",
        lambda v: non_negative_whole_number_error(v, "n_steps", "step"),
        (-1, 2.7, NAN, INF, None, "3", [3], True, np.int64(-1)),
    ),
    Guard(
        "positive_count_error",
        "width",
        lambda v: positive_count_error(v, "width", "add_camera"),
        (0, -1, 2.7, NAN, None, "3", [3], True, np.int64(640)),
    ),
    Guard(
        "non_negative_count_error",
        "steps",
        lambda v: non_negative_count_error(v, "steps", "handshake"),
        (-1, 2.7, NAN, None, "3", [3], True, np.int64(0)),
    ),
    Guard(
        "tcp_port_error",
        "port",
        lambda v: tcp_port_error(v, "port", "RosbridgeRobot"),
        (0, -1, 70000, 2.7, NAN, None, "3", [3], True, np.int64(9090)),
    ),
    Guard(
        "camera_fov_error",
        "fov",
        lambda v: camera_fov_error("add_camera", "fov", v),
        (NAN, INF, None, "3", [3], True),
    ),
    Guard(
        "entity_name_error",
        "name",
        lambda v: entity_name_error("add_object", "name", v),
        (7, 2.7, None, [3], True, np.int64(7)),
    ),
)

GUARD_IDS = tuple(guard.name for guard in SCALAR_GUARDS)

#: The guards that reach their message for an outsized ``int``: they refuse it on
#: type or range without converting, so the eager rendering was the only thing
#: between them and a structured answer.
ANSWERS_AN_OUTSIZED_INT = frozenset(
    {
        "non_negative_whole_number_error",
        "positive_count_error",
        "non_negative_count_error",
        "tcp_port_error",
        "entity_name_error",
    }
)

#: The guards that establish their domain by converting with ``float()``. That
#: conversion used to raise ``OverflowError`` before any rendering (#1874); it is
#: now guarded, so this is a statement about *how* they classify a value rather
#: than about an escape. The name changed with the fact: membership is still the
#: line that divides the family, which is why the set outlived the defect.
CONVERTS_THROUGH_FLOAT = frozenset(
    {
        "positive_finite_number_error",
        "finite_number_error",
        "positive_whole_number_error",
        "camera_fov_error",
    }
)

OUTSIZED_GUARDS = tuple(g for g in SCALAR_GUARDS if g.name in ANSWERS_AN_OUTSIZED_INT)
OUTSIZED_IDS = tuple(g.name for g in OUTSIZED_GUARDS)
CONVERTING_GUARDS = tuple(g for g in SCALAR_GUARDS if g.name in CONVERTS_THROUGH_FLOAT)
CONVERTING_IDS = tuple(g.name for g in CONVERTING_GUARDS)


# --------------------------------------------------------------------------- #
# The shared renderers                                                        #
# --------------------------------------------------------------------------- #
class TestTheSharedRenderers:
    """``_refusal_repr`` / ``_refusal_str`` are the only ways a refused value is rendered."""

    def test_a_renderable_value_renders_exactly_as_repr(self) -> None:
        """The whole reason no message text changes: it defers wherever it can."""
        for value in (-1, -1.0, None, "3", [3], True, np.float32(-1.0), np.int64(-1), NAN):
            assert _refusal_repr(value) == repr(value)

    def test_a_renderable_value_renders_exactly_as_str(self) -> None:
        for value in (-1, -1.0, None, "3", [3], True, np.float32(200.0), np.int64(200)):
            assert _refusal_str(value) == str(value)

    def test_the_two_forms_are_not_interchangeable(self) -> None:
        """Why the ``str`` sites keep ``str``: NumPy 2 reprs a scalar with its type."""
        assert _refusal_str(np.float32(200.0)) == "200.0"
        assert _refusal_repr(np.float32(200.0)) == "np.float32(200.0)"

    def test_an_outsized_integer_is_described_by_its_magnitude(self) -> None:
        """``int.bit_length`` needs no decimal conversion, so this stays possible."""
        expected = f"<int of {BEYOND_INT_STR_LIMIT.bit_length()} bits>"
        with pytest.raises(ValueError):
            repr(BEYOND_INT_STR_LIMIT)
        with pytest.raises(ValueError):
            str(BEYOND_INT_STR_LIMIT)
        assert _refusal_repr(BEYOND_INT_STR_LIMIT) == expected
        assert _refusal_str(BEYOND_INT_STR_LIMIT) == expected

    def test_both_forms_describe_an_unrenderable_value_identically(self) -> None:
        """One describer, so the two render forms cannot disagree about one value."""
        for value in (BEYOND_INT_STR_LIMIT, Unprintable(), UnrenderableReal()):
            assert _refusal_repr(value) == _describe_unrenderable(value)
            assert _refusal_str(value) == _describe_unrenderable(value)

    def test_the_description_of_an_outsized_integer_does_not_carry_its_sign(self) -> None:
        """A known limit of the description, pinned rather than left to be discovered.

        ``int.bit_length`` is the magnitude, so the two spellings describe
        identically and the message cannot say which arrived. It is not worth a
        format change: magnitude is inside these guards' domains, so the sign is
        the only reason left to refuse an outsized integer, and the guard's own
        text states the domain it was refused against.
        """
        assert _describe_unrenderable(-BEYOND_INT_STR_LIMIT) == _describe_unrenderable(BEYOND_INT_STR_LIMIT)
        refusal = positive_count_error(-BEYOND_INT_STR_LIMIT, "width", "add_camera")
        assert refusal is not None
        assert "must be a positive integer" in refusal

    def test_a_value_that_cannot_render_itself_is_described_by_its_type(self) -> None:
        assert _refusal_repr(Unprintable()) == "<unrepresentable Unprintable>"
        assert _refusal_repr(UnrenderableReal()) == "<unrepresentable UnrenderableReal>"
        assert _refusal_str(UnrenderableFov()) == "<unrepresentable UnrenderableFov>"
        assert _refusal_repr(UnrenderableName("arm")) == "<unrepresentable UnrenderableName>"

    def test_the_fallback_is_unconditional_rather_than_a_known_exception_list(self) -> None:
        """A third-party type may raise anything, so the guarantee cannot enumerate."""

        class RaisesUnusual:
            def __repr__(self) -> str:
                raise MemoryError("out of memory rendering myself")

        class RaisesBaseException:
            def __repr__(self) -> str:
                raise KeyboardInterrupt

        assert _refusal_repr(RaisesUnusual()) == "<unrepresentable RaisesUnusual>"
        # ``except Exception`` does not cover a ``BaseException`` and must not:
        # swallowing a ``KeyboardInterrupt`` to render an error message would be a
        # worse failure than the one being reported. Pinned so the boundary is a
        # decision rather than an oversight.
        with pytest.raises(KeyboardInterrupt):
            _refusal_repr(RaisesBaseException())

    def test_every_rendering_is_ascii(self) -> None:
        for value in (-1, NAN, None, [3], BEYOND_INT_STR_LIMIT, Unprintable()):
            _refusal_repr(value).encode("ascii")
            _refusal_str(value).encode("ascii")


# --------------------------------------------------------------------------- #
# The invariant: no guard raises while refusing                               #
# --------------------------------------------------------------------------- #
class TestEveryScalarGuardAnswersAValueItCannotRender:
    """The uniform defect: eight of nine guards raised here before this change.

    Every row fails on pre-fix code with the probe's own ``RuntimeError`` escaping
    the guard, in place of the refusal string it had already decided to return.
    """

    @pytest.mark.parametrize("guard", SCALAR_GUARDS, ids=GUARD_IDS)
    def test_an_unrenderable_value_refused_on_its_type_is_answered(self, guard: Guard) -> None:
        refusal = guard.call(Unprintable())
        assert refusal is not None, "an unrenderable value is not a usable one"
        assert "<unrepresentable Unprintable>" in refusal
        assert guard.param in refusal, "the message must still name the parameter"
        refusal.encode("ascii")

    @pytest.mark.parametrize("guard", SCALAR_GUARDS, ids=GUARD_IDS)
    def test_an_unrenderable_value_refused_on_its_value_is_answered(self, guard: Guard) -> None:
        """Past the type test, refused on its value - the path ``Unprintable`` skips."""
        refusal = guard.call(UnrenderableReal())
        assert refusal is not None
        assert "<unrepresentable UnrenderableReal>" in refusal
        assert guard.param in refusal

    def test_an_unrenderable_name_subclass_is_answered(self) -> None:
        """``entity_name_error``'s NUL branch runs after the ``str`` test passes."""
        refusal = entity_name_error("add_object", "name", UnrenderableName("a\x00b"))
        assert refusal is not None
        assert "<unrepresentable UnrenderableName>" in refusal
        assert "NUL" in refusal

    @pytest.mark.parametrize("guard", SCALAR_GUARDS, ids=GUARD_IDS)
    def test_the_verdict_is_a_string_or_none_for_every_probe(self, guard: Guard) -> None:
        """The contract as one assertion, over every value this module names."""
        probes = (*guard.refused, Unprintable(), UnrenderableReal(), UnrenderableFov(), UnrenderableName("a\x00b"))
        for probe in probes:
            verdict = guard.call(probe)
            assert verdict is None or isinstance(verdict, str)


class TestOutsizedIntegersAreAnsweredWhereTheGuardReachesItsMessage:
    """The ``R:ValueError`` cells: reachable today with a plain Python ``int``.

    These five refuse on type or range without converting, so the eager rendering
    was the only thing between them and a structured answer. A remote caller's
    number arrives through ``device_connect``'s ``@rpc()`` unchanged, and Python
    integers are arbitrary-precision.
    """

    @pytest.mark.parametrize("guard", OUTSIZED_GUARDS, ids=OUTSIZED_IDS)
    def test_an_outsized_integer_outside_the_domain_is_refused(self, guard: Guard) -> None:
        refusal = guard.call(-BEYOND_INT_STR_LIMIT)
        assert refusal is not None
        assert f"<int of {BEYOND_INT_STR_LIMIT.bit_length()} bits>" in refusal
        assert guard.param in refusal
        refusal.encode("ascii")

    def test_the_port_range_refuses_an_outsized_integer_of_either_sign(self) -> None:
        """``tcp_port_error`` bounds above too, so both signs reach the message."""
        for probe in (BEYOND_INT_STR_LIMIT, -BEYOND_INT_STR_LIMIT):
            refusal = tcp_port_error(probe, "port", "RosbridgeRobot")
            assert refusal is not None
            assert "expected 1-65535" in refusal

    def test_a_name_that_is_an_outsized_integer_is_refused(self) -> None:
        refusal = entity_name_error("add_object", "name", BEYOND_INT_STR_LIMIT)
        assert refusal is not None
        assert f"<int of {BEYOND_INT_STR_LIMIT.bit_length()} bits>" in refusal
        assert "(int)" in refusal, "the message names the type as well as the value"

    def test_an_outsized_integer_inside_the_domain_is_still_accepted(self) -> None:
        """Magnitude is not these guards' question, and this change does not make it one.

        ``positive_count_error`` and ``non_negative_count_error`` accepted
        ``10**400`` and ``10**5000`` before this change and still do, which is also
        the evidence #1872 turns on: refusing an outsized positive in
        ``positive_whole_number_error`` would break the documented 2x2 against both
        of them, not only against the cell #1869 rewrote.
        """
        for probe in (BEYOND_FLOAT_RANGE, BEYOND_INT_STR_LIMIT):
            assert positive_count_error(probe, "width", "add_camera") is None
            assert non_negative_count_error(probe, "steps", "handshake") is None
            assert non_negative_whole_number_error(probe, "n_steps", "step") is None


class TestTheOpenIntervalBranchRendersPlainlyAndStillCannotRaise:
    """``camera_fov_error``'s second branch renders with ``str``, and keeps doing so.

    It looks unreachable by anything unrenderable, because it runs only after
    ``math.isfinite(float(value))`` has succeeded. A registered ``numbers.Real``
    whose ``float()`` is finite and outside the interval passes that test and is
    refused here, so the branch needed the guarantee too - and it needed it in the
    ``str`` form, because converting to ``repr`` would silently change
    agent-visible text for a documented-as-accepted NumPy fov.
    """

    def test_the_interval_message_renders_a_numpy_fov_without_its_type(self) -> None:
        refusal = camera_fov_error("add_camera", "fov", np.float32(200.0))
        assert refusal == "add_camera: 'fov' must be in the open interval (0, 180) degrees, got 200.0."
        assert "np.float32" not in refusal

    def test_a_finite_fov_outside_the_interval_that_cannot_render_is_answered(self) -> None:
        refusal = camera_fov_error("add_camera", "fov", UnrenderableFov())
        assert refusal is not None
        assert "must be in the open interval (0, 180) degrees" in refusal
        assert "<unrepresentable UnrenderableFov>" in refusal

    def test_the_finiteness_branch_is_the_one_that_takes_a_non_finite_fov(self) -> None:
        """The two branches carry different reasons; neither may raise."""
        refusal = camera_fov_error("add_camera", "fov", UnrenderableReal())
        assert refusal is not None
        assert "must be a finite number in degrees" in refusal


class TestValidationSplitRendersATaskCountItCannotRender:
    """``validation_split_error`` renders a task count read out of a dataset.

    ``total_tasks`` comes from ``meta/info.json``, so it is a JSON integer and
    arbitrary-precision. It is rendered plainly rather than quoted, which is the
    other reason the drift scan had to cover the ``{value}`` form: this site
    carries no ``!r`` and was invisible to a scan that looked for one.
    """

    def test_an_outsized_task_count_is_answered(self) -> None:
        refusal = validation_split_error(1, BEYOND_INT_STR_LIMIT, "train", passthrough_param="extra")
        assert refusal is not None
        assert f"<int of {BEYOND_INT_STR_LIMIT.bit_length()} bits>" in refusal
        refusal.encode("ascii")

    def test_an_ordinary_task_count_still_renders_as_a_plain_number(self) -> None:
        refusal = validation_split_error(1, 5, "train", passthrough_param="extra")
        assert refusal is not None
        assert "dataset with 5 tasks" in refusal

    def test_a_single_task_dataset_is_still_accepted(self) -> None:
        assert validation_split_error(1, 1, "train", passthrough_param="extra") is None
        assert validation_split_error(1, None, "train", passthrough_param="extra") is None


# --------------------------------------------------------------------------- #
# No agent-visible change                                                     #
# --------------------------------------------------------------------------- #
class TestTheTextIsUnchangedForAValueThatCanBeRendered:
    """The renderers defer, so every existing message is byte-identical.

    This is the claim that makes the change verdict- and text-preserving, so it is
    pinned rather than asserted in a description.
    """

    @pytest.mark.parametrize("guard", SCALAR_GUARDS, ids=GUARD_IDS)
    def test_the_message_still_contains_the_plain_repr(self, guard: Guard) -> None:
        for value in guard.refused:
            refusal = guard.call(value)
            assert refusal is not None, f"{value!r} must be refused by {guard.name}"
            assert repr(value) in refusal, f"{guard.name} no longer renders {value!r} as repr"

    def test_the_exact_text_of_one_message_per_guard(self) -> None:
        """Spelled out, so a reworded message is a diff here rather than a surprise."""
        assert positive_finite_number_error(0, "hz", "teleoperate") == "teleoperate: hz must be > 0, got 0."
        assert finite_number_error("x", "linear", "drive") == "drive: linear must be a finite number, got 'x'."
        assert positive_whole_number_error(0, "fps", "video") == "video: fps must be a positive whole number, got 0."
        assert (
            non_negative_whole_number_error(-5, "n_steps", "step")
            == "step: n_steps must be a non-negative whole number, got -5."
        )
        assert positive_count_error(0, "width", "add_camera") == "add_camera: width must be a positive integer, got 0."
        assert (
            non_negative_count_error(-1, "steps", "handshake")
            == "handshake: steps must be a non-negative integer, got -1."
        )
        assert tcp_port_error(0, "port", "RosbridgeRobot") == "RosbridgeRobot: invalid port: 0 (expected 1-65535)"
        assert (
            camera_fov_error("add_camera", "fov", NAN)
            == "add_camera: 'fov' must be a finite number in degrees, got nan."
        )

    def test_an_accepted_value_is_still_accepted(self) -> None:
        """The over-reach control: rendering changed, no domain did."""
        assert positive_finite_number_error(30.0, "hz", "teleoperate") is None
        assert finite_number_error(-1.5, "linear", "drive") is None
        assert positive_whole_number_error(30.0, "fps", "video") is None
        assert non_negative_whole_number_error(0, "n_steps", "step") is None
        assert positive_count_error(640, "width", "add_camera") is None
        assert non_negative_count_error(0, "steps", "handshake") is None
        assert tcp_port_error(9090, "port", "RosbridgeRobot") is None
        assert camera_fov_error("add_camera", "fov", np.float32(58.0)) is None
        assert entity_name_error("add_object", "name", "cube") is None


# --------------------------------------------------------------------------- #
# Boundary: the escapes this change does not close                            #
# --------------------------------------------------------------------------- #
class TestTheConversionEscapeIsClosed:
    """The replacement for this file's #1874 scope pin, now that #1874 has landed.

    It is a replacement rather than a deletion, per the premise-test guidance in
    ``AGENTS.md``: the conclusion the old pin supported - that the family divides
    into guards that convert with ``float()`` and guards that do not - still
    holds, and is still the line worth drawing. What changed is that converting no
    longer means raising.

    Where the old class asserted ``pytest.raises(OverflowError)`` for these four,
    each assertion below is the same probe with the opposite expectation, so the
    two classes read against each other. The detailed closure - the reason each
    guard gives, and the fact that none of the existing reasons changed - is
    :mod:`tests.test_conversion_escape_is_closed`; what is pinned here is only
    that this file's stated boundary moved.
    """

    @pytest.mark.parametrize("guard", CONVERTING_GUARDS, ids=CONVERTING_IDS)
    def test_an_integer_wider_than_a_float_is_answered(self, guard: Guard) -> None:
        assert isinstance(guard.call(BEYOND_FLOAT_RANGE), str)

    def test_positive_whole_number_error_has_no_escape_left(self) -> None:
        """Both outsized spellings are answered, which is what #1872 asked for.

        This guard is where the two escapes met: its eager rendering raised
        ``ValueError`` on a value past the digit limit *ahead of* the conversion,
        so closing only one of them would have left the class open. #1873 closed
        the rendering and #1874 the conversion, and this is the assertion that
        neither left a residue at the other's boundary.
        """
        for probe in (BEYOND_FLOAT_RANGE, BEYOND_INT_STR_LIMIT):
            assert isinstance(positive_whole_number_error(probe, "fps", "video"), str)

    def test_the_two_halves_of_the_family_have_merged(self) -> None:
        """The partition survives as a description; the escape it described does not.

        The sets are still disjoint and still cover the family - that is a fact
        about how each guard classifies a value, and it is why the two escapes
        could be settled separately at all. What no longer holds is that
        membership predicts a raise: every guard now answers.
        """
        assert CONVERTS_THROUGH_FLOAT.isdisjoint(ANSWERS_AN_OUTSIZED_INT)
        assert CONVERTS_THROUGH_FLOAT | ANSWERS_AN_OUTSIZED_INT == set(GUARD_IDS)
        for guard in SCALAR_GUARDS:
            assert isinstance(guard.call(BEYOND_FLOAT_RANGE), str) or guard.call(BEYOND_FLOAT_RANGE) is None


class TestTheContainerGuardsAnswerAnUnrenderableElement:
    """The inverse of the scope pin this class replaces (#1875).

    ``repr`` of a container recurses into its elements, so a container was
    unrenderable whenever any one element was, and these guards raised out of a
    refusal they had already decided. They now answer, through
    ``_refusal_container_repr``: the container is rendered elementwise and only
    the components that cannot print are substituted, so the shape and the
    element count - often the refusal's entire reason - survive.

    Kept here, in the class the scope pin occupied, so the boundary this module
    drew is visibly *moved* rather than dropped. The rendering's own contract is
    pinned in ``tests/test_container_refusals_render_elementwise.py``.
    """

    def test_a_vector_with_an_unrenderable_element_answers(self) -> None:
        for message in (
            finite_vector_error("raycast", "origin", [BEYOND_INT_STR_LIMIT]),
            pose_vector_error("add_object", "position", [BEYOND_INT_STR_LIMIT], 3),
        ):
            assert message is not None
            assert f"<int of {BEYOND_INT_STR_LIMIT.bit_length()} bits>" in message

    def test_a_name_list_with_an_unrenderable_entry_answers(self) -> None:
        message = name_list_error([BEYOND_INT_STR_LIMIT], "cameras", "render_all")
        assert message is not None
        assert f"<int of {BEYOND_INT_STR_LIMIT.bit_length()} bits>" in message

    def test_the_elements_that_render_are_not_erased_with_the_one_that_cannot(self) -> None:
        """The whole reason a container needed a rendering and not a fallback."""
        message = finite_vector_error("raycast", "origin", [1.0, BEYOND_INT_STR_LIMIT, 3.0])
        assert message is not None
        assert f"[1.0, <int of {BEYOND_INT_STR_LIMIT.bit_length()} bits>, 3.0]" in message
        assert "<unrepresentable list>" not in message


# --------------------------------------------------------------------------- #
# Drift: no caller value may be rendered directly                             #
# --------------------------------------------------------------------------- #
def _scan_direct_renders(source: str) -> dict[str, tuple[tuple[str, str], ...]]:
    """Map each function in ``source`` to the caller values it renders directly.

    "The caller's value" is spelled ``Any`` in every guard in this module - the
    other parameters are the ``str`` labels the *call site* supplies, which are
    literals and cannot raise. Keying on the annotation rather than on a list of
    parameter names is what lets the scan cover a guard nobody has written yet.

    Both render forms are reported. A scan for ``!r`` alone would have passed
    ``camera_fov_error``'s interval branch and ``validation_split_error``, which
    render plainly and reach the same escape.

    Text a function *raises* is excluded: ``_safe_join`` renders an untrusted path
    into a deliberate ``ValueError`` rather than answering a caller through a
    return value, so it is not in this contract.

    Args:
        source: The contents of a Python module.

    Returns:
        Every function that renders a caller value directly, mapped to
        ``(parameter, form)`` pairs where form is ``"!r"`` or ``"plain"``. A guard
        on the shared renderers contributes nothing, because
        ``_refusal_repr(value)`` is a call rather than a bare name.
    """
    tree = ast.parse(source)
    found: dict[str, tuple[tuple[str, str], ...]] = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]:
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        carries_value = {arg.arg for arg in args if arg.annotation is not None and ast.unparse(arg.annotation) == "Any"}
        if not carries_value:
            continue
        raised = {id(n) for stmt in ast.walk(fn) if isinstance(stmt, ast.Raise) for n in ast.walk(stmt)}
        rendered = set()
        for node in ast.walk(fn):
            if id(node) in raised or not isinstance(node, ast.FormattedValue):
                continue
            if isinstance(node.value, ast.Name) and node.value.id in carries_value:
                rendered.add((node.value.id, "!r" if node.conversion == ord("r") else "plain"))
        if rendered:
            found[fn.name] = tuple(sorted(rendered))
    return found


#: Empty, and that is the assertion: **no** function in ``utils.py`` renders a
#: caller value directly any more. The five container guards that used to be
#: listed here were #1875 and now route through ``_refusal_container_repr``, so
#: the table has nothing left to hold. Any entry appearing here is a new guard
#: that skipped the shared renderers.
KNOWN_DIRECT_RENDERS: dict[str, tuple[tuple[str, str], ...]] = {}

#: The guards that render a *container*, which need the elementwise renderer
#: rather than the whole-value one. Named so the scan below is shown to reach
#: them: an empty ``KNOWN_DIRECT_RENDERS`` would otherwise be satisfied just as
#: well by a scanner that had stopped looking at them.
CONTAINER_GUARDS = frozenset(
    {
        "coerce_rgba",
        "coerce_size_vector",
        "finite_vector_error",
        "name_list_error",
        "pose_vector_error",
    }
)


#: Builtin calls that *read* the value handed to them, as opposed to inspecting it.
#: ``isinstance`` and ``type`` ask a question of the object's class and cannot run
#: the value's own code; every name here dispatches to a dunder the value owns.
READING_BUILTINS = frozenset(
    {
        "all",
        "any",
        "dict",
        "enumerate",
        "frozenset",
        "iter",
        "len",
        "list",
        "max",
        "min",
        "next",
        "repr",
        "reversed",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
    }
)

#: The helpers that perform a read *inside* a ``try`` and hand back an ordinary
#: value. A name bound from one of these is this module's own object rather than
#: the caller's, so reading it further cannot run caller code and the scan below
#: stops following it. Without this the scan would report
#: ``len(characters)`` - a plain ``list`` produced by a guarded read - as loudly
#: as it reports ``len(value)``.
GUARDED_READERS = frozenset(
    {
        "_describe_failed_read",
        "_describe_unrenderable",
        "_read_finite_vector",
        "_read_name_list",
        "_read_pose_vector",
        "_read_to_quote",
        "_refusal_container_repr",
        "_refusal_repr",
        "_refusal_str",
        "sequence_length",
    }
)


def _scan_unguarded_message_reads(source: str) -> dict[str, tuple[tuple[str, str], ...]]:
    """Map each public guard in ``source`` to the caller reads it makes outside a ``try``.

    The companion to :func:`_scan_direct_renders`, and the escape that scan is
    blind to. It reports a caller value *rendered* without a shared renderer,
    which is why ``_refusal_container_repr(list(value))`` satisfied it: the value
    reaching the renderer is a ``list`` the guard built, and building it ran the
    caller's ``__iter__``. So the render was guarded and the read feeding it was
    not, and the whole of #1903 lived in that gap - as did the four escapes on
    ``name_list_error``'s string branch that #1903 did not have.

    "Reads the caller's value" is followed rather than assumed. The parameters
    annotated ``Any`` are the caller's, and so is any local assigned from one:
    ``name_list_error``'s ``shown`` is the caller's own ``str`` under a different
    name, and reading *it* is what escaped. The chain stops at
    :data:`GUARDED_READERS`, whose members read inside a ``try`` and return an
    ordinary value - so a local bound from one is no longer the caller's object
    and reading it cannot run their code.

    Only public functions are scanned. The private helpers below them are the
    guarded layer a guard is built from, so a read inside one is the point rather
    than a defect, and reporting them would make the table a list of the fixes
    instead of a list of the escapes.

    Args:
        source: The contents of a Python module.

    Returns:
        Every public guard that reads a caller value outside a ``try``, mapped to
        ``(name, kind)`` pairs, where kind names the read form - a builtin, an
        attribute call, a comprehension or a subscript.
    """
    tree = ast.parse(source)
    found: dict[str, tuple[tuple[str, str], ...]] = {}
    for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]:
        if fn.name.startswith("_"):
            continue
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        tainted = {arg.arg for arg in args if arg.annotation is not None and ast.unparse(arg.annotation) == "Any"}
        if not tainted:
            continue

        # Follow the caller's value through assignments until nothing new is
        # bound. A guarded reader breaks the chain; anything else propagates,
        # because a slice or a decode of the caller's object is still theirs.
        for _ in range(len(list(ast.walk(fn)))):
            grown = set(tainted)
            for node in ast.walk(fn):
                if not isinstance(node, ast.Assign | ast.AnnAssign) or node.value is None:
                    continue
                calls = {
                    call.func.id
                    for call in ast.walk(node.value)
                    if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                }
                if calls & GUARDED_READERS:
                    continue
                sources = {n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)}
                if not sources & tainted:
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    grown |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
            if grown == tainted:
                break
            tainted = grown

        guarded = {
            id(n) for stmt in ast.walk(fn) if isinstance(stmt, ast.Try) for body in stmt.body for n in ast.walk(body)
        }
        raised = {id(n) for stmt in ast.walk(fn) if isinstance(stmt, ast.Raise) for n in ast.walk(stmt)}
        reads: set[tuple[str, str]] = set()
        for node in ast.walk(fn):
            if id(node) in guarded or id(node) in raised:
                continue
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in READING_BUILTINS:
                for arg in node.args:
                    reads |= {
                        (n.id, f"{node.func.id}()")
                        for n in ast.walk(arg)
                        if isinstance(n, ast.Name) and n.id in tainted
                    }
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                target = node.func.value
                if isinstance(target, ast.Name) and target.id in tainted:
                    reads.add((target.id, f".{node.func.attr}()"))
            elif isinstance(node, ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp):
                for generator in node.generators:
                    reads |= {
                        (n.id, "comprehension")
                        for n in ast.walk(generator.iter)
                        if isinstance(n, ast.Name) and n.id in tainted
                    }
            elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in tainted:
                reads.add((node.value.id, "subscript"))
        if reads:
            found[fn.name] = tuple(sorted(reads))
    return found


#: The reads still outstanding, which is what this scan was written to surface.
#: Empty, and both entries came off it the same way. ``name_list_error`` went with
#: #1903 - five reads across two branches, one of them filed and four of them not -
#: and the three coercions the scan found beside it went with #1906: each validated
#: its value with a guard that reads it and then read it again, unguarded, to build
#: the floats. That read now lives in ``_read_finite_vector`` / ``_read_pose_vector``,
#: which return the floats they read, so a coercion's result and the components its
#: refusal quotes are one list rather than two independent reads of the same value.
#:
#: An empty table is the load-bearing state here rather than a satisfied one: a new
#: guard that reads a caller value outside a ``try`` shows up as an entry nobody
#: wrote, which is why the assertion is stated over the module instead of over a
#: list of guards. :class:`TestTheCoercionsReadTheirValueOnce` holds the behaviour
#: this emptiness is the scan-level statement of.
KNOWN_MESSAGE_READS: dict[str, tuple[tuple[str, str], ...]] = {}


class TestNoGuardRendersACallerValueDirectly:
    """A guard must render a refused value through a shared renderer.

    A fixed list of guards would not survive a tenth being added, so this scans
    the module instead: the assertion is over the whole file, and a new function
    that renders a value it returns shows up as a new entry.
    """

    def _source(self) -> str:
        return pathlib.Path(inspect.getfile(utils)).read_text(encoding="utf-8")

    def test_no_scalar_guard_renders_a_caller_value_directly(self) -> None:
        found = _scan_direct_renders(self._source())
        adrift = {name: sites for name, sites in found.items() if name not in KNOWN_DIRECT_RENDERS}
        assert adrift == {}, f"these render a caller value without a shared renderer: {adrift}"
        assert found == KNOWN_DIRECT_RENDERS, f"the set of direct renders changed: {found}"

    def test_the_scan_actually_reaches_the_guards_this_change_owns(self) -> None:
        """An empty scan would satisfy the assertion above just as well."""
        tree = ast.parse(self._source())
        scanned = set()
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
            args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
            if any(a.annotation is not None and ast.unparse(a.annotation) == "Any" for a in args):
                scanned.add(fn.name)
        assert set(GUARD_IDS) <= scanned, f"guards invisible to the scan: {set(GUARD_IDS) - scanned}"
        assert CONTAINER_GUARDS <= scanned, f"container guards invisible to the scan: {CONTAINER_GUARDS - scanned}"
        assert "validation_split_error" in scanned

    def test_every_guard_calls_a_shared_renderer(self) -> None:
        """The positive form: absence from the table is not enough on its own.

        A guard that stopped naming the refused value at all would also be absent,
        and would return a message that no longer says what was refused. Now that
        the table is empty this is the whole of the load-bearing half, and it
        covers the container guards as well as the scalar ones.

        Followed through the module's own calls rather than stopping at the guard's
        body, because a guard may compute its verdict from a read helper that does
        the rendering: since #1906 ``finite_vector_error`` and ``pose_vector_error``
        return the message ``_read_finite_vector`` / ``_read_pose_vector`` built, and
        a body-only reading calls that a guard naming nothing. Transitive is the
        weaker statement of the two, so the reachable set is what the assertion is
        over - a guard whose whole call graph renders nothing still fails.
        """
        tree = ast.parse(self._source())
        defined = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        owned = set(GUARD_IDS) | CONTAINER_GUARDS | {"validation_split_error"}
        renderers = {"_refusal_repr", "_refusal_str", "_refusal_container_repr"}
        for name in sorted(owned & set(defined)):
            reached: set[str] = set()
            pending = [name]
            while pending:
                current = pending.pop()
                if current in reached:
                    continue
                reached.add(current)
                body = defined.get(current)
                if body is None:
                    continue
                pending += [
                    node.func.id
                    for node in ast.walk(body)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                ]
            assert reached & renderers, f"{name} renders no value through a shared renderer"

    def test_the_scanner_reports_a_planted_repr_omission(self) -> None:
        """Without this, an empty result could mean a scanner matching nothing."""
        planted = textwrap.dedent(
            """
            def new_guard_error(value: Any, param: str, context: str) -> str | None:
                if value < 0:
                    return f"{context}: {param} must be positive, got {value!r}."
                return None
            """
        )
        assert _scan_direct_renders(planted) == {"new_guard_error": (("value", "!r"),)}

    def test_the_scanner_reports_a_planted_plain_omission(self) -> None:
        """The form that hid both ``str`` sites from an ``!r``-only reading."""
        planted = textwrap.dedent(
            """
            def new_guard_error(value: Any, param: str, context: str) -> str | None:
                if value < 0:
                    return f"{context}: {param} must be positive, got {value}."
                return None
            """
        )
        assert _scan_direct_renders(planted) == {"new_guard_error": (("value", "plain"),)}

    def test_the_scanner_accepts_a_guard_on_a_shared_renderer(self) -> None:
        """The control for the control: a converted guard must not be reported."""
        converted = textwrap.dedent(
            """
            def new_guard_error(value: Any, param: str, context: str) -> str | None:
                if value < 0:
                    return f"{context}: {param} must be positive, got {_refusal_repr(value)}."
                return None
            """
        )
        assert _scan_direct_renders(converted) == {}

    def test_the_scanner_ignores_the_call_site_labels(self) -> None:
        """``param`` and ``context`` are the call site's own literals, not values."""
        labels_only = textwrap.dedent(
            """
            def new_guard_error(value: Any, param: str, context: str) -> str | None:
                return f"{context}: {param} is wrong ({context!r})."
            """
        )
        assert _scan_direct_renders(labels_only) == {}

    def test_the_scanner_ignores_a_value_rendered_into_a_raise(self) -> None:
        """``_safe_join`` raises by design; this contract is about returned text."""
        raising = textwrap.dedent(
            """
            def _safe_join(base: Path, untrusted: Any) -> Path:
                raise ValueError(f"Path traversal blocked: {untrusted!r} escapes {base}")
            """
        )
        assert _scan_direct_renders(raising) == {}
        assert "_safe_join" not in KNOWN_DIRECT_RENDERS


class TestNoGuardReadsACallerValueOutsideATry:
    """A guard must not run the caller's own code while building its refusal (#1903).

    ``TestNoGuardRendersACallerValueDirectly`` above holds the *rendering* of a
    refused value, and that is not the same property. Its table is empty because
    every guard renders through a shared renderer - and
    ``_refusal_container_repr(list(value))`` satisfied it completely, because the
    value handed to the renderer was a ``list`` the guard had already built. The
    render could not raise; building its argument ran the caller's ``__iter__``,
    which could.

    So this is the read-side scan, stated over the module rather than over a list
    of guards, for the reason the sibling gives: a fixed list would not survive a
    tenth guard being added. It is what closing #1903 needed in order not to be a
    one-branch patch - the issue described the mapping's ``list(value)``, and the
    scan found four more on the string branch beside it.
    """

    def _source(self) -> str:
        return pathlib.Path(inspect.getfile(utils)).read_text(encoding="utf-8")

    def test_no_public_guard_reads_a_caller_value_outside_a_try(self) -> None:
        found = _scan_unguarded_message_reads(self._source())
        adrift = {name: sites for name, sites in found.items() if name not in KNOWN_MESSAGE_READS}
        assert adrift == {}, f"these run caller code outside a try: {adrift}"
        assert found == KNOWN_MESSAGE_READS, f"the set of unguarded reads changed: {found}"

    def test_the_scan_reaches_the_guards_it_is_asserted_over(self) -> None:
        """An empty result and a scanner that looks at nothing are the same table."""
        scanned = set(_scan_unguarded_message_reads(self._source() + _PLANTED_UNGUARDED_READ))
        assert scanned == set(KNOWN_MESSAGE_READS) | {"new_guard_error"}, (
            f"the planted guard was not reached: {scanned}"
        )

    def test_the_scanner_reports_the_read_1903_was_filed_for(self) -> None:
        """The mapping remedy, verbatim, as the escape it was."""
        assert _scan_unguarded_message_reads(
            textwrap.dedent(
                """
                def new_guard_error(value: Any, param: str, context: str) -> str | None:
                    if isinstance(value, Mapping):
                        return f"{param} must be a list: {_refusal_container_repr(list(value))}."
                    return None
                """
            )
        ) == {"new_guard_error": (("value", "list()"),)}

    def test_the_scanner_reports_the_four_reads_1903_did_not_have(self) -> None:
        """The string branch as it stood: a comprehension, a length, and a decode.

        All four were invisible to every existing assertion - the rendering scan
        sees a shared renderer, and no test passed a hostile ``str`` subclass.
        """
        assert _scan_unguarded_message_reads(
            textwrap.dedent(
                """
                def new_guard_error(value: Any, param: str, context: str) -> str | None:
                    shown = value.decode(errors="replace") if isinstance(value, bytes) else value
                    return (
                        f"{param}: read as {[c for c in shown][:6]} ({len(shown)} name(s)). "
                        f"Wrap it in a list: [{_refusal_repr(shown)}]."
                    )
                """
            )
        ) == {
            "new_guard_error": (
                ("shown", "comprehension"),
                ("shown", "len()"),
                ("value", ".decode()"),
            )
        }

    def test_the_scanner_accepts_a_read_moved_inside_a_try(self) -> None:
        """The control for the control: the shape this change lands must not be reported."""
        assert (
            _scan_unguarded_message_reads(
                textwrap.dedent(
                    """
                    def new_guard_error(value: Any, param: str, context: str) -> str | None:
                        try:
                            names = list(value)
                        except Exception:
                            return f"{param}: could not be read."
                        return f"{param} must be a list: {_refusal_container_repr(names)}."
                    """
                )
            )
            == {}
        )

    def test_the_scanner_accepts_a_read_of_a_guarded_readers_result(self) -> None:
        """``len(characters)`` is this module's own list, not the caller's value.

        Without this the scan would report the fix as loudly as the defect, and an
        unfollowable chain is what makes a taint scan get switched off.
        """
        assert (
            _scan_unguarded_message_reads(
                textwrap.dedent(
                    """
                    def new_guard_error(value: Any, param: str, context: str) -> str | None:
                        characters, unreadable = _read_to_quote(value)
                        if characters is None:
                            return f"{param}: could not be read ({unreadable})."
                        return f"{param}: read as {characters[:6]} ({len(characters)} name(s))."
                    """
                )
            )
            == {}
        )

    def test_the_scanner_ignores_a_type_test(self) -> None:
        """``isinstance`` asks the class, so it cannot run the value's own code.

        A scan that reported it would fire on every guard in the file and be
        deleted within a week, which is the failure mode worth a control of its own.
        """
        assert (
            _scan_unguarded_message_reads(
                textwrap.dedent(
                    """
                    def new_guard_error(value: Any, param: str, context: str) -> str | None:
                        if isinstance(value, Mapping) or type(value) is dict:
                            return f"{param} must be a list of names."
                        return None
                    """
                )
            )
            == {}
        )

    def test_the_scanner_ignores_a_private_helper(self) -> None:
        """A read inside the guarded layer is the point, not the defect.

        ``_read_to_quote`` reads the caller's value on purpose; a scan that
        reported it would hold the table open on the very functions that close it.
        """
        assert (
            _scan_unguarded_message_reads(
                textwrap.dedent(
                    """
                    def _read_to_quote(value: Any) -> tuple[list[Any] | None, str | None]:
                        return list(value), None
                    """
                )
            )
            == {}
        )


class FailsOnItsSecondNumericRead(Sequence[Any]):
    """Answers one full read and refuses the next, the #1897 probe shape.

    Distinct from that module's ``FailsOnItsSecondRead`` in carrying numbers, so
    it reaches the numeric coercions rather than being refused as a non-name
    before the second read happens.
    """

    def __init__(self, *values: Any) -> None:
        self.values = list(values)
        self.reads = 0

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: Any) -> Any:
        if index == 0:
            self.reads += 1
            if self.reads > 1:
                raise RuntimeError("no second read for you")
        return self.values[index]


class TestTheCoercionsReadTheirValueOnce:
    """Each coercion's floats come from the read its guard already made (#1906).

    Replaces ``TestTheCoercionsSecondReadIsAStatedBoundary`` rather than deleting
    it, per the guidance that a premise test invalidated by a fix is re-pinned
    instead of dropped: what it measured was a value read exactly once, asserted in
    the raising direction because the second read escaped. The boundary is gone, so
    the same probe is asserted in the accepting direction - the value it raised on
    is the value it now returns floats for.

    Two of the three reads are on the **accepted** path and build the return value,
    which is why two of these assertions are about a returned vector rather than a
    refusal. ``coerce_rgba``'s is the one a refusal quotes, and it gets the
    refusal-shaped assertion the contract asks for.
    """

    def test_the_three_coercions_accept_a_value_that_refuses_a_second_read(self) -> None:
        """Called one by one rather than looped over: the three signatures differ.

        ``coerce_pose_vector`` takes the component count the others infer, so a
        loop over ``(guard, args)`` pairs would hide that behind a ``*args`` splat
        no reader - and no static analysis - can check against either signature.
        """
        assert utils.coerce_pose_vector("add_object", "position", FailsOnItsSecondNumericRead(0.1, 0.2, 0.3), 3) == (
            [0.1, 0.2, 0.3],
            None,
        )
        assert utils.coerce_rgba("add_object", "color", FailsOnItsSecondNumericRead(0.1, 0.2, 0.3)) == (
            [0.1, 0.2, 0.3, 1.0],
            None,
        )
        assert utils.coerce_size_vector("add_object", "size", FailsOnItsSecondNumericRead(0.1, 0.2, 0.3)) == (
            [0.1, 0.2, 0.3],
            None,
        )

    def test_each_coercion_reads_the_value_exactly_once(self) -> None:
        """The read count itself, which is the property rather than its symptom.

        A coercion that read twice and tolerated a failing second read would
        satisfy the assertions above while still computing its floats from a read
        nothing obliged to agree with the checked one - the half of #1906 that is
        about agreement rather than about the escape.
        """
        position = FailsOnItsSecondNumericRead(0.1, 0.2, 0.3)
        utils.coerce_pose_vector("add_object", "position", position, 3)
        assert position.reads == 1
        color = FailsOnItsSecondNumericRead(0.1, 0.2, 0.3)
        utils.coerce_rgba("add_object", "color", color)
        assert color.reads == 1
        size = FailsOnItsSecondNumericRead(0.1, 0.2, 0.3)
        utils.coerce_size_vector("add_object", "size", size)
        assert size.reads == 1

    def test_coerce_rgba_answers_on_the_path_that_exists_to_return_text(self) -> None:
        """The one of the three that broke a stated contract, which #1906 turned on.

        Two components is a refusal, and the value never reached it: the second read
        happened before the length was compared, so the guard raised on exactly the
        path documented never to. It now returns that refusal, and the components it
        quotes are the ones the domain checks examined rather than a second read's.
        """
        floats, err = utils.coerce_rgba("add_object", "color", FailsOnItsSecondNumericRead(0.1, 0.2))
        assert floats is None
        assert err is not None
        assert "must have exactly 3 or 4 component(s)" in err
        assert "got 2: [0.1, 0.2]" in err

    def test_the_probe_still_refuses_a_second_read(self) -> None:
        """Non-vacuity: a probe that had stopped refusing would pass everything above.

        It is the same fixture #1897 used, so this also states what the read count
        above is counting - one full read, and the next one raising.
        """
        probe = FailsOnItsSecondNumericRead(0.1, 0.2, 0.3)
        assert [float(component) for component in probe] == [0.1, 0.2, 0.3]
        with pytest.raises(RuntimeError, match="no second read for you"):
            [float(component) for component in probe]  # noqa: B018

    def test_the_guard_read_once_does_not_escape(self) -> None:
        """The family control: #1897's guard on the same probe, unchanged."""
        assert utils.name_list_error(FailsOnItsSecondNumericRead("top", "wrist"), "cameras", "render_all") is None


class OverstatesItsLength(Sequence[Any]):
    """``__len__`` reports one more component than the read produces.

    A ``Sequence`` owes its two protocols no agreement, and this is the shape the
    coercions' two reads disagreed on (#1909): the count came from ``__len__`` and
    the components from the element read.
    """

    def __init__(self, *values: Any) -> None:
        self.values = list(values)
        self.element_reads = 0

    def __len__(self) -> int:
        return len(self.values) + 1

    def __getitem__(self, index: Any) -> Any:
        self.element_reads += 1
        return self.values[index]


class UnderstatesItsLength(Sequence[Any]):
    """``__len__`` reports two components; the read produces however many it holds."""

    def __init__(self, *values: Any) -> None:
        self.values = list(values)
        self.element_reads = 0

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: Any) -> Any:
        self.element_reads += 1
        return self.values[index]


class YieldsNoComponent(Sequence[Any]):
    """``__len__`` reports three components and the read produces none."""

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: Any) -> Any:
        raise IndexError(index)


class TestTheCoercionsCountTheComponentsTheyRead:
    """A coercion's component count comes from the read that produced them (#1909).

    #1906 made each coercion read its value's *components* once. The count beside
    them was still a second, independent read - ``sequence_length(value)``, i.e.
    ``__len__`` - and a ``Sequence`` is not obliged to answer the two the same way.
    Two of the rows below are contract breaks rather than message defects, which is
    why this is a change in verdict and not only in text.

    :func:`~strands_robots.utils.sequence_length` stays where it answers its own
    question - *is this a sized sequence at all*, which is what refuses a generator
    on both coercions - and the pose path keeps its length gate ahead of the element
    read, so a wrong reported length is still refused without producing a component.
    """

    def test_rgba_returns_four_components_for_a_value_that_overstates_its_length(self) -> None:
        """The documented promise, previously true only of a value that agreed with itself.

        ``coerce_rgba`` returns "exactly 4 finite floats" so the ``color[:3]`` reads
        the shape builders do are well-defined by construction. The alpha completion
        was gated on the *reported* count, so a ``__len__`` of 4 over a read yielding
        3 skipped it and returned a 3-element rgba under a success result.
        """
        assert utils.coerce_rgba("add_object", "color", OverstatesItsLength(0.1, 0.2, 0.3)) == (
            [0.1, 0.2, 0.3, 1.0],
            None,
        )

    def test_rgba_accepts_the_components_it_read_when_the_length_understates_them(self) -> None:
        """The other direction, and a deliberate change of verdict.

        Three readable RGB components are a colour whatever the value's ``__len__``
        says; refusing them "got 2" while quoting three was the refusal that
        contradicted itself in one sentence.
        """
        assert utils.coerce_rgba("add_object", "color", UnderstatesItsLength(0.1, 0.2, 0.3)) == (
            [0.1, 0.2, 0.3, 1.0],
            None,
        )

    def test_every_rgba_this_guard_accepts_carries_four_components(self) -> None:
        """The promise as a property over the disagreeing shapes, not one example."""
        for color in (
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 0.5],
            OverstatesItsLength(0.1, 0.2, 0.3),
            OverstatesItsLength(0.1, 0.2, 0.3, 0.5),
            UnderstatesItsLength(0.1, 0.2, 0.3),
        ):
            floats, err = utils.coerce_rgba("add_object", "color", color)
            assert err is None
            assert floats is not None
            assert len(floats) == 4

    def test_the_rgba_refusal_counts_the_components_it_quotes(self) -> None:
        """The count named and the components quoted are one read's.

        A refusal reading ``got 2: [0.1, 0.2, 0.3, 0.4, 0.5]`` names a count from
        one read beside components from the other, and a reader cannot tell which
        of the two the guard acted on.
        """
        floats, err = utils.coerce_rgba("add_object", "color", UnderstatesItsLength(0.1, 0.2, 0.3, 0.4, 0.5))
        assert floats is None
        assert err is not None
        assert "got 5: [0.1, 0.2, 0.3, 0.4, 0.5]" in err

    def test_a_pose_vector_shorter_than_its_reported_length_is_refused(self) -> None:
        """A 3-component quaternion reached ``data.qpos`` through the guard.

        The length gate accepted the reported 4, the read produced 3, and the
        coercion returned them - the bare ``ValueError`` inside the numpy
        assignment that :func:`~strands_robots.utils.pose_vector_error` exists to
        prevent, reached through the guard rather than around it.
        """
        floats, err = utils.coerce_pose_vector("add_object", "position", OverstatesItsLength(0.1, 0.2, 0.3), 4)
        assert floats is None
        assert err is not None
        assert "'position' must be a 4-element vector, got 3: [0.1, 0.2, 0.3]" in err
        assert "length reported 4" in err

    def test_no_accepted_pose_vector_has_the_wrong_component_count(self) -> None:
        """The property behind the row above, over both directions of disagreement."""
        for vec in (
            [0.1, 0.2, 0.3],
            OverstatesItsLength(0.1, 0.2, 0.3),
            UnderstatesItsLength(0.1, 0.2, 0.3),
            YieldsNoComponent(),
        ):
            for expected_len in (3, 4):
                floats, err = utils.coerce_pose_vector("add_object", "position", vec, expected_len)
                assert (floats is None) is (err is not None)
                if floats is not None:
                    assert len(floats) == expected_len

    def test_a_wrong_reported_length_is_still_refused_before_any_element_is_read(self) -> None:
        """The gate's position is deliberate and unchanged.

        Refusing a wrong reported length without producing a component is worth
        keeping - the re-check above is an addition to it, not a replacement, and a
        re-check alone would read every element of a vector already known to be the
        wrong length.
        """
        vec = UnderstatesItsLength(0.1, 0.2, 0.3)
        floats, err = utils.coerce_pose_vector("add_object", "position", vec, 3)
        assert floats is None
        assert err is not None
        assert "must be a 3-element vector, got 2" in err
        assert vec.element_reads == 0

    def test_size_takes_its_component_count_from_the_read(self) -> None:
        """``size`` counts extents, and an extent it cannot read is not one it has.

        The per-shape count is #1858's decision and not this one; what is decided
        here is only which read the count comes from. A value whose length reports
        three extents and whose read yields none has no extent to write, so it is
        the empty-vector refusal - not an accepted ``size`` of three.
        """
        assert utils.coerce_size_vector("add_object", "size", OverstatesItsLength(0.1, 0.2, 0.3)) == (
            [0.1, 0.2, 0.3],
            None,
        )
        floats, err = utils.coerce_size_vector("add_object", "size", YieldsNoComponent())
        assert floats is None
        assert err is not None
        assert "must have at least one component" in err

    def test_the_probes_disagree_with_themselves(self) -> None:
        """Non-vacuity: probes that had started agreeing would pass everything above."""
        over = OverstatesItsLength(0.1, 0.2, 0.3)
        assert len(over) == 4
        assert [float(component) for component in over] == [0.1, 0.2, 0.3]
        under = UnderstatesItsLength(0.1, 0.2, 0.3)
        assert len(under) == 2
        assert [float(component) for component in under] == [0.1, 0.2, 0.3]
        nothing = YieldsNoComponent()
        assert len(nothing) == 3
        assert list(nothing) == []


#: A guard that reads the caller's value straight into its message, appended to
#: ``utils.py``'s own source so the scan is shown to reach a function it has never
#: seen. Kept beside the tests that use it rather than inlined, because both the
#: reachability check and its own report read the same planted text.
_PLANTED_UNGUARDED_READ = textwrap.dedent(
    """

def new_guard_error(value: Any, param: str, context: str) -> str | None:
    return f"{context}: {param} must be a list of names, got {len(value)}."
"""
)


def test_the_digit_limit_this_module_turns_on_is_below_its_probe() -> None:
    """``BEYOND_INT_STR_LIMIT`` is only unrenderable while the limit is below it.

    ``sys.set_int_max_str_digits`` is process-global and a test elsewhere could
    raise it, which would quietly turn every probe here into an ordinary integer
    and every assertion above into a tautology.
    """
    assert 0 < sys.get_int_max_str_digits() < BEYOND_INT_STR_LIMIT_DIGITS
