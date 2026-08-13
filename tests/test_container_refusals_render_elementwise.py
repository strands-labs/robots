"""A container guard must not raise while rendering the container it refuses (#1875).

``strands_robots.utils``' scalar guards were made total in two passes: #1873 gave
them a rendering that cannot raise (``_refusal_repr`` / ``_refusal_str``), and
#1874 gave them a *conversion* that cannot raise. Both passes deliberately
stopped at the container guards - ``finite_vector_error``, ``pose_vector_error``,
``coerce_pose_vector``, ``coerce_rgba`` and ``name_list_error`` - and this module
is the pass that finishes them.

Why a container could not simply reuse the scalar fallback
----------------------------------------------------------
``repr`` of a list recurses into its elements, so a container is unrenderable
whenever any *one* of its elements is. Applying ``_refusal_repr`` to the whole
container answers that with ``<unrepresentable list>``, which erases:

* every element that rendered perfectly well, and
* the element **count** - which is frequently the refusal's entire reason, as in
  ``must be a 3-element vector, got 4``.

So a container needs a *rendering* rather than a fallback:
``_refusal_container_repr`` describes it component by component and substitutes
only the components that cannot print, keeping the shape legible::

    raycast: 'origin' must contain finite numbers (no nan/inf), got
    [1.0, <int of 16610 bits>, 3.0]

That is the difference this module pins, in both directions: the rendering
survives an element that cannot print, *and* it does not throw away the ones that
can. :class:`TestTheWholeValueFallbackWouldHaveBeenWrong` states the second half
as a measurement rather than as a claim in prose.

Two escapes, one boundary
-------------------------
The same guards also raised on the *conversion*: ``finite_vector_error`` called
``float(element)`` unprotected to test finiteness, so an element past
``sys.float_info.max`` raised ``OverflowError`` before any message was rendered -
#1874's defect, one level down. Both halves are closed here because both had to
be: a rendering that survives an unprintable element is no use if the guard
raises on an outsized one first, and the two arrive in the same value
(``10**5000`` is both).

Where the module-wide invariants live
-------------------------------------
Two scans outside this file assert that no *sixth* guard can reintroduce either
escape silently, so they are stated over ``utils.py`` as a whole rather than over
a list of names this file happens to know:

* ``tests/test_refusal_messages_never_raise.py`` -
  ``TestNoGuardRendersACallerValueDirectly``, whose ``KNOWN_DIRECT_RENDERS`` table
  is now **empty**: no function in the module renders a caller value without a
  shared renderer.
* ``tests/test_conversion_escape_is_closed.py`` -
  ``TestTheRemainingConversionsRunOnlyOnTheAcceptedPath``, which reports the
  remaining unprotected conversions exactly.

Reachability
------------
These are not hypothetical inputs. ``raycast(origin=...)``,
``add_object(position=..., size=..., color=...)`` and the recorder / render
``cameras`` lists all arrive from an agent tool call or a ``device_connect``
``@rpc()`` payload, where a list of JSON numbers is the normal shape and Python
integers are arbitrary-precision. The guards are exercised here directly, as
their siblings' tests are, so the contract is pinned at the one place every
backend shares rather than once per caller.
"""

from __future__ import annotations

import ast
import inspect
import math
import numbers
import pathlib
import sys
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from strands_robots import utils
from strands_robots.utils import (
    _describe_unrenderable,
    _refusal_container_repr,
    _refusal_repr,
    coerce_pose_vector,
    coerce_rgba,
    coerce_size_vector,
    finite_vector_error,
    name_list_error,
    pose_vector_error,
)

# --------------------------------------------------------------------------- #
# Probes                                                                      #
# --------------------------------------------------------------------------- #
#: An ``int`` too wide to render: past :func:`sys.get_int_max_str_digits` (4300
#: digits by default), so ``repr`` raises ``ValueError`` - and so does ``repr`` of
#: any list containing it. This is the rendering escape.
UNRENDERABLE_INT = 10**5000

#: How :func:`_describe_unrenderable` names the value above. ``bit_length``
#: needs no decimal conversion, so the magnitude survives where the digits cannot.
UNRENDERABLE_INT_SHOWN = f"<int of {UNRENDERABLE_INT.bit_length()} bits>"

#: Past ``sys.float_info.max`` but *inside* the digit limit, so its own ``repr``
#: works and it isolates the conversion escape from the rendering one.
OUTSIZED_INT = 10**400

#: Inside the float64 range, to show a refusal is about magnitude rather than
#: about being an ``int`` or being large.
LARGE_BUT_CONVERTIBLE = 10**308

NAN = float("nan")
INF = float("inf")


class Unprintable:
    """A plain object whose ``repr`` raises.

    ``int`` past the digit limit is the case that actually arrives, but the
    guarantee is unconditional rather than a list of today's known exceptions, so
    an arbitrary ``__repr__`` failure is probed as well.
    """

    def __repr__(self) -> str:
        raise RuntimeError("no repr for you")


class UnprintableReal:
    """A registered :class:`numbers.Real` whose ``repr`` raises.

    Registered with ``numbers.Real.register`` rather than subclassed, which is not
    a shortcut - it is the property under test, and matches the double in
    ``tests/test_conversion_escape_is_closed.py``. :class:`numbers.Real` is a
    registration rather than an inheritance, so a component can satisfy a vector
    guard's ``isinstance`` check while owing it nothing else - including a working
    ``__repr__``. Subclassing the ABC would have forced two dozen operators this
    double never uses.

    ``__float__`` answers ``nan``, so the element reaches the guard's *finiteness*
    branch - the last one a value crosses before acceptance - rather than its type
    branch, and the message still has to render the container.
    """

    def __repr__(self) -> str:
        raise RuntimeError("no repr for you")

    def __float__(self) -> float:
        return NAN


numbers.Real.register(UnprintableReal)


class RealNoFloat:
    """A registered :class:`numbers.Real` from which no float can be read.

    Registered with ``numbers.Real.register`` rather than subclassed, so a value
    can satisfy a guard's ``isinstance`` check while owing it nothing else.
    With no ``__float__`` at all, ``float()`` raises ``TypeError`` of its own
    accord. That is the conversion's second exception class: unlike an outsized
    magnitude it is not a range complaint - no number can be read from this value,
    so it must not be told about the float64 range. Its ``repr`` works, which
    isolates the conversion from the rendering defect.
    """

    def __repr__(self) -> str:
        return "RealNoFloat()"


numbers.Real.register(RealNoFloat)


class UnprintableContainer(list):  # type: ignore[type-arg]
    """A list whose own ``repr`` raises although every element renders fine.

    Separates the two reasons a container can fail to render: the container
    itself, and one of its elements. The elementwise path has to answer both, and
    for this one it can report every component exactly.
    """

    def __repr__(self) -> str:
        raise RuntimeError("no repr for you")


class UniterableContainer:
    """A value whose ``repr`` fails and which is not iterable.

    The floor of the rendering: with no ``repr`` and no elements to walk, the only
    honest answer left is the whole-value description, which is what
    :func:`_refusal_repr` would have given.

    ``__iter__`` raises ``TypeError``, which is what "not iterable" means in
    Python and what every guard here already routes to a refusal. A ``__iter__``
    raising something *else* was a third escape mechanism, neither rendering nor
    conversion; it is closed by #1878 and gets its own verdict, which is what
    :class:`TestTheIterationIsAnsweredNotEscaped` measures against this probe.
    """

    def __repr__(self) -> str:
        raise RuntimeError("no repr for you")

    def __iter__(self) -> Iterator[Any]:
        raise TypeError("not iterable")


class HostileIteration:
    """A value whose ``repr`` fails and whose ``__iter__`` raises a non-``TypeError``.

    Used against the renderer, which recovers any ``Exception``, and against
    ``finite_vector_error``, whose own ``iter()`` is reached before the renderer
    and answers this value with its own refusal since #1878;
    :class:`TestTheIterationIsAnsweredNotEscaped` states what that answer is.
    """

    def __repr__(self) -> str:
        raise RuntimeError("no repr for you")

    def __iter__(self) -> Iterator[Any]:
        raise RuntimeError("no iteration for you")


class UnprintableFailureError(Exception):
    """An exception whose own ``str`` raises.

    The fix for #1878 reports the exception that stopped the iteration, so the
    message now interpolates a value supplied by the same hostile type. Without
    this probe that interpolation is the #1873 rendering escape again, one level
    inside the fix for the iteration escape, and nothing would measure it.
    """

    def __str__(self) -> str:
        raise RuntimeError("no str for you")


class UnprintableFailure:
    """A value whose ``__iter__`` raises an exception that cannot be rendered."""

    def __repr__(self) -> str:
        raise RuntimeError("no repr for you")

    def __iter__(self) -> Iterator[Any]:
        raise UnprintableFailureError


class UnprintableName(str):
    """A ``str`` subclass whose ``repr`` raises.

    ``name_list_error`` reaches several of its messages only after an element has
    passed ``isinstance(entry, str)``, so a raising ``__repr__`` on a genuine
    ``str`` is the way those branches are reached at all.
    """

    def __repr__(self) -> str:
        raise RuntimeError("no repr for you")


class InterruptingRepr:
    """A value whose ``repr`` raises a :class:`BaseException`.

    ``KeyboardInterrupt`` and ``SystemExit`` are not errors to recover from, and a
    renderer that swallowed them would make a container guard uninterruptible.
    """

    def __repr__(self) -> str:
        raise KeyboardInterrupt


# --------------------------------------------------------------------------- #
# The rendering escape, per guard                                             #
# --------------------------------------------------------------------------- #
#: Every container guard, called with a container holding one element that cannot
#: render. Each returns its message differently - the ``*_error`` guards return a
#: ``str | None`` and the ``coerce_*`` guards a ``(value, error)`` pair - so the
#: probe normalises to the message and the table stays one row per guard.
#:
#: The value is in the **last** position for all of them, after the two ``str``
#: labels. Passing it anywhere else is refused for the wrong reason and would
#: look like evidence about the element; ``test_the_probes_reach_the_element``
#: below is the control for that.
UNRENDERABLE_PROBES: tuple[tuple[str, Any], ...] = (
    ("finite_vector_error", lambda v: finite_vector_error("raycast", "origin", v)),
    ("pose_vector_error", lambda v: pose_vector_error("add_object", "position", v, 1)),
    ("coerce_pose_vector", lambda v: coerce_pose_vector("add_camera", "position", v, 1)[1]),
    ("coerce_rgba", lambda v: coerce_rgba("add_object", "color", v)[1]),
    ("coerce_size_vector", lambda v: coerce_size_vector("add_object", "size", v)[1]),
    ("name_list_error", lambda v: name_list_error(v, "cameras", "render_all")),
)

PROBE_IDS = [name for name, _ in UNRENDERABLE_PROBES]
PROBE_CALLS = [call for _, call in UNRENDERABLE_PROBES]

#: A container each guard accepts, for the control that the probes above reach the
#: element. These are not interchangeable: ``coerce_rgba`` refuses any component
#: count but 3 or 4, and ``name_list_error`` wants names rather than numbers.
ACCEPTABLE: dict[str, Any] = {
    "finite_vector_error": [0.5],
    "pose_vector_error": [0.5],
    "coerce_pose_vector": [0.5],
    "coerce_rgba": [0.5, 0.5, 0.5],
    "coerce_size_vector": [0.5],
    "name_list_error": ["wrist"],
}


class TestEveryContainerGuardAnswersAnUnrenderableElement:
    """The defect, closed: a refusal already decided must be returned, not lost.

    Each guard runs only on the path whose entire purpose is to answer an unusable
    input with a structured ``{"status": "error"}`` message instead of an
    exception. Raising while building that message fails on exactly the path that
    exists so it does not, which is why every one of these is asserted to return
    text rather than merely "not raise".
    """

    @pytest.mark.parametrize("call", PROBE_CALLS, ids=PROBE_IDS)
    def test_an_int_past_the_digit_limit_is_answered(self, call: Any) -> None:
        message = call([UNRENDERABLE_INT])
        assert message is not None
        assert UNRENDERABLE_INT_SHOWN in message

    @pytest.mark.parametrize("call", PROBE_CALLS, ids=PROBE_IDS)
    def test_an_element_with_an_arbitrary_repr_failure_is_answered(self, call: Any) -> None:
        """The guarantee is unconditional, not a list of the exceptions known today."""
        message = call([Unprintable()])
        assert message is not None
        assert "<unrepresentable Unprintable>" in message

    @pytest.mark.parametrize("call", PROBE_CALLS, ids=PROBE_IDS)
    def test_a_container_whose_own_repr_fails_is_answered(self, call: Any) -> None:
        """The container, not an element, is what cannot render here."""
        message = call(UnprintableContainer([UNRENDERABLE_INT]))
        assert message is not None
        assert UNRENDERABLE_INT_SHOWN in message

    @pytest.mark.parametrize("call", PROBE_CALLS, ids=PROBE_IDS)
    def test_a_value_that_can_neither_render_nor_be_walked_is_answered(self, call: Any) -> None:
        message = call(UniterableContainer())
        assert message is not None
        assert "<unrepresentable UniterableContainer>" in message

    def test_a_numeric_element_reaches_the_finiteness_branch_and_still_renders(self) -> None:
        """A registered ``numbers.Real`` passes the type test and owes nothing else.

        Without this the unrenderable element would only ever be probed on the
        *type* branch, and the branch nearest acceptance - the last one a value
        crosses before the guard returns ``None`` - would go untested.
        """
        message = finite_vector_error("raycast", "origin", [UnprintableReal()])
        assert message is not None
        assert "must contain finite numbers (no nan/inf)" in message
        assert "<unrepresentable UnprintableReal>" in message

    def test_an_unrenderable_name_is_answered_on_every_name_list_branch(self) -> None:
        """``name_list_error`` renders its value on more branches than the others."""
        blank = name_list_error([UnprintableName("  ")], "cameras", "render_all")
        assert blank is not None and "must be a non-blank name" in blank
        assert "<unrepresentable UnprintableName>" in blank

        repeated = name_list_error([UnprintableName("a"), UnprintableName("a")], "cameras", "render_all")
        assert repeated is not None and "must not repeat a name" in repeated
        assert "<unrepresentable UnprintableName>" in repeated

        mapping = name_list_error({UNRENDERABLE_INT: "camera"}, "cameras", "render_all")
        assert mapping is not None and "not a mapping" in mapping
        assert UNRENDERABLE_INT_SHOWN in mapping

        scalar = name_list_error(UNRENDERABLE_INT, "cameras", "render_all")
        assert scalar is not None and "must be a list of names" in scalar
        assert UNRENDERABLE_INT_SHOWN in scalar

    @pytest.mark.parametrize(
        ("call", "acceptable"),
        [(call, ACCEPTABLE[name]) for name, call in UNRENDERABLE_PROBES],
        ids=PROBE_IDS,
    )
    def test_the_probes_reach_the_element(self, call: Any, acceptable: Any) -> None:
        """Control: a well-formed container must be accepted by the same call.

        Every guard here takes the value last, after two ``str`` labels. A probe
        that passed the container in a label position would be refused for having
        the wrong type *there*, which reads exactly like a verdict about the
        element and is not one. So the same probes are run against a container each
        guard accepts and must answer ``None``, which proves the rows above measure
        the element rather than the shape of the call.

        The accepted value differs per guard because their domains do -
        ``coerce_rgba`` needs 3 or 4 components and ``name_list_error`` needs names
        - and a single shared value would silently be refused for its shape by
        those two, which is the exact confusion this control exists to rule out.
        """
        assert call(acceptable) is None


class TestTheConversionEscapeIsClosedToo:
    """The other half of #1875: an element no float64 can hold.

    ``finite_vector_error`` tested finiteness with a bare ``float(element)``, so an
    element past ``sys.float_info.max`` raised ``OverflowError`` before any message
    was rendered. It now asks ``_beyond_float_range`` first and converts inside a
    ``try``, which is the shape #1874 established for the scalar guards, and the
    three ``coerce_*`` guards inherit the fix because they reach it through this
    one.
    """

    @pytest.mark.parametrize("call", PROBE_CALLS, ids=PROBE_IDS)
    def test_an_outsized_element_is_answered(self, call: Any) -> None:
        message = call([OUTSIZED_INT])
        assert message is not None

    def test_the_reason_given_is_the_magnitude(self) -> None:
        """Not "not a number": an integer past the float range *is* a number.

        The wording follows the scalar guards' own range text so an agent reading
        either is told the same thing, and is a vector's plural of it.
        """
        message = finite_vector_error("raycast", "origin", [1.0, OUTSIZED_INT, 3.0])
        assert message is not None
        assert "must contain numbers within the range of a 64-bit float" in message
        assert f"[1.0, {OUTSIZED_INT}, 3.0]" in message

    def test_an_element_no_number_can_be_read_from_keeps_the_not_a_number_reason(self) -> None:
        """The two exception classes are two honest reasons, not one.

        ``_beyond_float_range`` answers ``OverflowError`` only, so a registered
        ``numbers.Real`` whose ``__float__`` does not work falls through to the
        pre-existing not-a-number text rather than being mis-reported as a range
        complaint about a magnitude nobody could read.
        """
        message = finite_vector_error("raycast", "origin", [RealNoFloat()])
        assert message is not None
        assert "elements must be numbers" in message
        assert "range of a 64-bit float" not in message

    def test_the_boundary_is_where_the_float_range_is(self) -> None:
        """A control matrix: the refusal is about magnitude, and only about that."""
        assert finite_vector_error("raycast", "origin", [sys.float_info.max]) is None
        assert finite_vector_error("raycast", "origin", [LARGE_BUT_CONVERTIBLE]) is None
        assert finite_vector_error("raycast", "origin", [-OUTSIZED_INT]) is not None

    def test_the_two_escapes_compose_in_one_value(self) -> None:
        """``10**5000`` is both outsized *and* unrenderable, and arrives as one value.

        Neither fix is any use without the other here: the conversion has to be
        recovered for a message to be reached at all, and the rendering has to be
        elementwise for that message to be buildable.
        """
        message = finite_vector_error("raycast", "origin", [1.0, UNRENDERABLE_INT, 3.0])
        assert message is not None
        assert "must contain numbers within the range of a 64-bit float" in message
        assert f"[1.0, {UNRENDERABLE_INT_SHOWN}, 3.0]" in message


# --------------------------------------------------------------------------- #
# The renderer's own contract                                                 #
# --------------------------------------------------------------------------- #
class TestTheElementwiseRenderer:
    """``_refusal_container_repr``: ``repr`` where it works, elementwise where it cannot."""

    @pytest.mark.parametrize(
        "value",
        [
            [1.0, 2.0, 3.0],
            (1.0, 2.0),
            [],
            ["a", None, {"k": 1}],
            {"a": 1},
            np.array([1.0, 2.0]),
            0.5,
            "wrist",
            None,
        ],
        ids=["list", "tuple", "empty", "mixed", "dict", "ndarray", "scalar", "str", "none"],
    )
    def test_a_value_that_can_render_is_reported_exactly_as_repr(self, value: Any) -> None:
        """The fast path, and the reason no existing message text moved.

        A ``tuple``, a ``dict`` and a NumPy array all have reprs no elementwise
        form reproduces (``(1.0, 2.0)``, ``{'a': 1}``, ``array([1., 2.])``), so
        trying the whole container first is not merely an optimisation - it is what
        keeps those containers reported in their own notation.
        """
        assert _refusal_container_repr(value) == repr(value)

    def test_only_the_components_that_cannot_print_are_substituted(self) -> None:
        assert _refusal_container_repr([1.0, UNRENDERABLE_INT, 3.0]) == f"[1.0, {UNRENDERABLE_INT_SHOWN}, 3.0]"

    def test_the_offending_component_is_located_by_position_not_by_an_index(self) -> None:
        """A decision #1875 asked to be settled, so it is stated rather than implied.

        ``finite_vector_error`` and ``name_list_error`` already report a per-element
        index in their own text (``cameras[0] must be a name``). An index inserted
        by the renderer as well would state it twice in two forms that could
        disagree, and would not match the ``repr`` this stands in for. Position
        alone locates the component, so position is what is used.
        """
        rendered = _refusal_container_repr([1.0, UNRENDERABLE_INT])
        assert rendered == f"[1.0, {UNRENDERABLE_INT_SHOWN}]"
        assert "[0]" not in rendered and "[1]" not in rendered

        # The guards that *do* name an index still do, unchanged.
        message = name_list_error(["ok", UNRENDERABLE_INT], "cameras", "render_all")
        assert message is not None and "cameras[1]" in message

    def test_nothing_is_elided(self) -> None:
        """Truncating would repeat the failure this renderer exists to avoid.

        A cap would erase elements that rendered fine - exactly what
        ``<unrepresentable list>`` does - and ``repr`` of a long container, the
        text this stands in for, is not truncated either.
        """
        values: list[Any] = [float(i) for i in range(50)]
        values[25] = UNRENDERABLE_INT
        rendered = _refusal_container_repr(values)
        assert rendered.count(",") == 49
        assert rendered.startswith("[0.0, 1.0,") and rendered.endswith("49.0]")
        assert UNRENDERABLE_INT_SHOWN in rendered
        assert "..." not in rendered

    def test_the_element_count_survives(self) -> None:
        """The count is often the refusal's whole reason, so it may not be lost."""
        message = pose_vector_error("add_object", "position", [1.0, UNRENDERABLE_INT], 3)
        assert message is not None
        assert "must be a 3-element vector, got 2" in message
        assert _refusal_container_repr([UNRENDERABLE_INT] * 4).count(UNRENDERABLE_INT_SHOWN) == 4

    def test_a_mapping_is_rendered_as_a_mapping(self) -> None:
        """``name_list_error`` refuses a mapping *for* discarding its values.

        Rendering it as the list of its keys would perform, in the message, the
        very discarding the message is complaining about.
        """
        rendered = _refusal_container_repr({UNRENDERABLE_INT: "camera"})
        assert rendered == f"{{{UNRENDERABLE_INT_SHOWN}: 'camera'}}"

    def test_an_unrenderable_value_inside_a_mapping_is_substituted_too(self) -> None:
        assert _refusal_container_repr({"k": Unprintable()}) == "{'k': <unrepresentable Unprintable>}"

    def test_a_container_whose_own_repr_fails_still_reports_every_element(self) -> None:
        assert _refusal_container_repr(UnprintableContainer([1.0, 2.0])) == "[1.0, 2.0]"

    def test_a_value_that_cannot_be_walked_falls_back_to_the_whole_value_description(self) -> None:
        """With no repr and no elements, the scalar answer is the only honest one."""
        value = UniterableContainer()
        assert _refusal_container_repr(value) == _describe_unrenderable(value)
        assert _refusal_container_repr(value) == "<unrepresentable UniterableContainer>"

    def test_an_arbitrary_iteration_failure_is_recovered_too(self) -> None:
        """``TypeError`` is what "not iterable" means, but the renderer promises more.

        It runs on a path that must not raise, so it recovers any ``Exception`` from
        the walk rather than only the one a well-behaved type would raise.
        """
        assert _refusal_container_repr(HostileIteration()) == "<unrepresentable HostileIteration>"

    def test_a_mapping_whose_items_cannot_be_walked_is_described(self) -> None:
        class HostileMapping(dict):  # type: ignore[type-arg]
            def __repr__(self) -> str:
                raise RuntimeError("no repr for you")

            def items(self) -> Any:
                raise RuntimeError("no items for you")

        assert _refusal_container_repr(HostileMapping()) == "<unrepresentable HostileMapping>"

    def test_a_non_container_is_answered_as_the_scalar_renderer_would(self) -> None:
        """Every one of these guards accepts ``Any``, so a scalar reaches them."""
        assert _refusal_container_repr(UNRENDERABLE_INT) == _refusal_repr(UNRENDERABLE_INT)
        assert _refusal_container_repr(UNRENDERABLE_INT) == UNRENDERABLE_INT_SHOWN

    def test_the_rendering_is_one_level_deep(self) -> None:
        """A nested container is described, not recursed into, and that is deliberate.

        These vectors carry *scalars* by contract - a nested list is itself the
        refusal's reason, reported as ``elements must be numbers`` - so the inner
        container's contents are never what the message is about, and one level is
        exactly the depth the guards need.

        Recursing would also have to detect cycles: ``a = []; a.append(a)`` is
        renderable only because the interpreter's own ``repr`` tracks what it has
        already visited, and a hand-written recursion without that would not
        terminate. The whole-value description is the right answer for an inner
        element, because at that point the element *is* the value.
        """
        assert _refusal_container_repr([[UNRENDERABLE_INT], 2.0]) == "[<unrepresentable list>, 2.0]"
        assert finite_vector_error("m", "p", [[UNRENDERABLE_INT], 2.0]) == (
            "m: 'p' elements must be numbers, got [<unrepresentable list>, 2.0]"
        )

    def test_a_self_referential_container_is_still_rendered(self) -> None:
        """The fast path handles it, which is the other reason not to recurse."""
        cyclic: list[Any] = []
        cyclic.append(cyclic)
        assert _refusal_container_repr(cyclic) == "[[...]]"

    def test_it_does_not_swallow_a_base_exception(self) -> None:
        """``KeyboardInterrupt`` is not an error to recover from."""
        with pytest.raises(KeyboardInterrupt):
            _refusal_container_repr([InterruptingRepr()])
        with pytest.raises(KeyboardInterrupt):
            _refusal_container_repr({"k": InterruptingRepr()})


class TestTheWholeValueFallbackWouldHaveBeenWrong:
    """Why this is a new renderer and not a call to the existing one (#1875).

    Stated as a measurement rather than as a claim in the docstrings: the scalar
    renderer is applied to the same containers and shown to erase what the
    elementwise one keeps. Without this, "a fallback would not do" is an assertion
    no test would notice becoming false.
    """

    def test_the_scalar_renderer_erases_the_elements_that_print(self) -> None:
        container = [1.0, UNRENDERABLE_INT, 3.0]
        assert _refusal_repr(container) == "<unrepresentable list>"
        assert "1.0" not in _refusal_repr(container)
        assert "1.0" in _refusal_container_repr(container)

    def test_the_scalar_renderer_erases_the_element_count(self) -> None:
        """And the count is frequently the entire reason for the refusal."""
        assert _refusal_repr([UNRENDERABLE_INT] * 4) == _refusal_repr([UNRENDERABLE_INT])
        assert _refusal_container_repr([UNRENDERABLE_INT] * 4) != _refusal_container_repr([UNRENDERABLE_INT])

    def test_no_guard_message_shows_the_whole_value_form_for_a_container(self) -> None:
        """The negative form, over every guard: none of them took the shortcut."""
        for name, call in UNRENDERABLE_PROBES:
            message = call([1.0, UNRENDERABLE_INT])
            assert message is not None, name
            assert "<unrepresentable list>" not in message, name


# --------------------------------------------------------------------------- #
# No verdict and no existing text moved                                       #
# --------------------------------------------------------------------------- #
#: Every branch of every container guard that renders a value, called with an
#: input that renders perfectly well, paired with the exact message it produced
#: before this change. Asserted as equality rather than as a substring: routing a
#: message through a renderer is only safe if the text is identical, and a
#: substring check would not notice a widened or reordered sentence.
UNCHANGED_TEXT: tuple[tuple[str, Any, str], ...] = (
    (
        "finite_vector_error/not-iterable",
        lambda: finite_vector_error("m", "size", 0.5),
        "m: 'size' must be a list/tuple of numbers, got 0.5",
    ),
    (
        "finite_vector_error/not-a-number",
        lambda: finite_vector_error("m", "size", ["a", "b"]),
        "m: 'size' elements must be numbers, got ['a', 'b']",
    ),
    (
        "finite_vector_error/nan",
        lambda: finite_vector_error("m", "size", [NAN, 1.0]),
        "m: 'size' must contain finite numbers (no nan/inf), got [nan, 1.0]",
    ),
    (
        "finite_vector_error/inf",
        lambda: finite_vector_error("m", "size", [INF]),
        "m: 'size' must contain finite numbers (no nan/inf), got [inf]",
    ),
    (
        "finite_vector_error/tuple",
        lambda: finite_vector_error("m", "size", ("a",)),
        "m: 'size' elements must be numbers, got ('a',)",
    ),
    (
        "pose_vector_error/unsized",
        lambda: pose_vector_error("add_object", "position", 0.5, 3),
        "add_object: 'position' must be a list/tuple of 3 numbers, got 0.5",
    ),
    (
        "pose_vector_error/wrong-length",
        lambda: pose_vector_error("add_object", "position", [0.4, 0.9], 3),
        "add_object: 'position' must be a 3-element vector, got 2 ([0.4, 0.9])",
    ),
    (
        "coerce_rgba/unsized",
        lambda: coerce_rgba("add_object", "color", 5)[1],
        "add_object: 'color' must be a sequence of numbers, got 5",
    ),
    (
        "coerce_size_vector/empty",
        lambda: coerce_size_vector("add_object", "size", [])[1],
        "add_object: 'size' must have at least one component, got an empty vector ([]). "
        "An empty 'size' is a component count, not an omission - omit 'size' to take the default extent.",
    ),
    (
        "name_list_error/bare-string",
        lambda: name_list_error("wrist", "image_keys", "Surface"),
        "Surface: image_keys must be a list of names, not a single string, got 'wrist'. "
        "A string is iterable per character, so this would be read as "
        "['w', 'r', 'i', 's', 't'] (5 name(s)). Wrap it in a list: ['wrist'].",
    ),
    (
        "name_list_error/mapping",
        lambda: name_list_error({"a": 1}, "image_keys", "Surface"),
        "Surface: image_keys must be a list of names, not a mapping, got {'a': 1}. "
        "A mapping is iterable over its keys, so its values would be discarded - "
        "pass the names as a list: ['a'].",
    ),
    (
        "name_list_error/not-a-name",
        lambda: name_list_error(["a", 1], "image_keys", "Surface"),
        "Surface: image_keys[1] must be a name (str), got int (1).",
    ),
    (
        "name_list_error/blank",
        lambda: name_list_error(["a", " "], "image_keys", "Surface"),
        "Surface: image_keys[1] must be a non-blank name, got ' '.",
    ),
    (
        "name_list_error/repeated",
        lambda: name_list_error(["a", "a"], "image_keys", "Surface"),
        "Surface: image_keys must not repeat a name, got ['a', 'a'] (['a'] appears more than once).",
    ),
)


class TestNoExistingVerdictOrMessageMoved:
    """The change is additive: inputs that answered before answer identically.

    Every message these guards produce is agent-visible, several are asserted
    verbatim by other suites, and two are quoted in the docs. So the text is
    pinned here as equality rather than left to those callers to discover.
    """

    @pytest.mark.parametrize(
        ("call", "expected"),
        [(call, expected) for _, call, expected in UNCHANGED_TEXT],
        ids=[name for name, _, _ in UNCHANGED_TEXT],
    )
    def test_the_message_is_unchanged(self, call: Any, expected: str) -> None:
        assert call() == expected

    def test_the_bool_component_reason_is_unchanged(self) -> None:
        """Kept out of the table above because it quotes a module constant."""
        message = finite_vector_error("m", "size", [True, 1.0])
        assert message == (
            "m: 'size' elements must be numbers, not a bool (got [True, 1.0]). " + utils.BOOLEAN_VECTOR_REASON
        )

    def test_the_component_count_refusal_still_shows_the_coerced_floats(self) -> None:
        """``coerce_rgba`` renders its *converted* list, which cannot fail to print."""
        assert coerce_rgba("add_object", "color", [0.5, 0.5])[1] == (
            "add_object: 'color' must have exactly 3 or 4 component(s) (RGB, or RGBA with alpha), "
            "got 2: [0.5, 0.5]. Pass every component - a partial 'color' cannot be applied "
            "without inventing the missing values."
        )

    @pytest.mark.parametrize(
        ("call", "accepted"),
        [
            (lambda v: finite_vector_error("m", "p", v), [0.0, 1.0, 2.0]),
            (lambda v: pose_vector_error("m", "p", v, 3), [0.0, 1.0, 2.0]),
            (lambda v: coerce_pose_vector("m", "p", v, 3)[1], [0.0, 1.0, 2.0]),
            (lambda v: coerce_rgba("m", "p", v)[1], [0.5, 0.5, 0.5]),
            (lambda v: coerce_size_vector("m", "p", v)[1], [0.1, 0.2, 0.3]),
            (lambda v: name_list_error(v, "p", "m"), ["wrist", "top"]),
        ],
        ids=PROBE_IDS,
    )
    def test_an_acceptable_value_is_still_accepted(self, call: Any, accepted: Any) -> None:
        """The other direction: nothing was refused that used to pass."""
        assert call(accepted) is None

    def test_the_numpy_component_types_are_still_accepted(self) -> None:
        """A documented acceptance that a stricter conversion could have broken."""
        assert finite_vector_error("m", "p", np.array([0.0, 1.0])) is None
        assert finite_vector_error("m", "p", [np.float32(1.5), np.int64(2)]) is None
        assert coerce_pose_vector("m", "p", np.array([0.0, 1.0, 2.0]), 3)[0] == [0.0, 1.0, 2.0]

    def test_the_coerced_output_is_still_plain_floats(self) -> None:
        """Normalisation is the reason three of these guards exist at all."""
        floats, error = coerce_pose_vector("m", "p", np.array([0.0, 1.0, 2.0]), 3)
        assert error is None and floats is not None
        assert all(type(component) is float for component in floats)


# --------------------------------------------------------------------------- #
# The invariant, not the four call sites                                      #
# --------------------------------------------------------------------------- #
class TestEveryContainerGuardRoutesThroughTheRenderer:
    """A structural scan, so a new branch cannot skip the renderer silently.

    The behavioural tests above cover the branches that exist today. A branch
    added tomorrow would not be in them, so the guards' source is scanned as well:
    a container guard may not interpolate a bare parameter name into a message.

    The module-wide form of this - over *every* function in ``utils.py``, not only
    these five - is ``TestNoGuardRendersACallerValueDirectly`` in
    ``tests/test_refusal_messages_never_raise.py``, whose table is now empty.
    """

    #: The guards this module owns. ``coerce_pose_vector`` renders nothing itself
    #: (every message it returns comes from ``pose_vector_error``), so it is
    #: absent here and covered by the behavioural rows instead. The two read
    #: helpers are here because since #1906 they are where the rendering happens:
    #: they hold the element read the two vector guards return a verdict from.
    RENDERING_GUARDS = (
        "_read_finite_vector",
        "_read_pose_vector",
        "coerce_rgba",
        "coerce_size_vector",
        "finite_vector_error",
        "name_list_error",
        "pose_vector_error",
    )

    #: The helpers a guard may compute its verdict from, whose own body renders.
    READ_HELPERS = frozenset({"_read_finite_vector", "_read_pose_vector"})

    @staticmethod
    def _function(name: str) -> ast.FunctionDef:
        source = pathlib.Path(inspect.getfile(utils)).read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                return node
        raise AssertionError(f"{name} not found in utils.py")

    @classmethod
    def _calls(cls, name: str) -> set[str]:
        return {
            node.func.id
            for node in ast.walk(cls._function(name))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

    @pytest.mark.parametrize("name", RENDERING_GUARDS)
    def test_the_guard_calls_the_elementwise_renderer(self, name: str) -> None:
        """Followed one hop into the read helper a guard delegates its verdict to.

        Since #1906 ``finite_vector_error`` and ``pose_vector_error`` return the
        message ``_read_finite_vector`` / ``_read_pose_vector`` built, so scanning
        the guard's own body alone would read that delegation as a guard that names
        nothing it refused. One hop is enough because the helpers are rows here in
        their own right, so nothing is taken on trust from the hop.
        """
        called = self._calls(name)
        for helper in called & self.READ_HELPERS:
            called |= self._calls(helper)
        assert "_refusal_container_repr" in called

    @pytest.mark.parametrize("name", RENDERING_GUARDS)
    def test_the_guard_renders_no_parameter_directly(self, name: str) -> None:
        fn = self._function(name)
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        carries_value = {a.arg for a in args if a.annotation is not None and ast.unparse(a.annotation) == "Any"}
        direct = {
            node.value.id
            for node in ast.walk(fn)
            if isinstance(node, ast.FormattedValue)
            and isinstance(node.value, ast.Name)
            and node.value.id in carries_value
        }
        assert direct == set(), f"{name} renders {sorted(direct)} without the shared renderer"

    def test_the_scan_reports_a_planted_direct_render(self) -> None:
        """Without this, a scanner that matched nothing would pass just as well."""
        planted = ast.parse(
            "def planted(vec: Any, method: str) -> str:\n    return f'{method}: bad, got {vec!r}'\n",
        )
        fn = next(n for n in ast.walk(planted) if isinstance(n, ast.FunctionDef))
        args = fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
        carries_value = {a.arg for a in args if a.annotation is not None and ast.unparse(a.annotation) == "Any"}
        direct = {
            node.value.id
            for node in ast.walk(fn)
            if isinstance(node, ast.FormattedValue)
            and isinstance(node.value, ast.Name)
            and node.value.id in carries_value
        }
        assert direct == {"vec"}
        assert "method" not in direct, "a str label is not a caller value and must not be reported"

    def test_the_digit_limit_this_module_relies_on_is_below_its_probe(self) -> None:
        """The probes are only unrenderable while the interpreter limit holds.

        ``sys.set_int_max_str_digits`` is settable, so a future default - or
        another test - could lift it past this module's probe and quietly turn
        every rendering assertion above into a test of nothing.
        """
        assert sys.get_int_max_str_digits() < UNRENDERABLE_INT.bit_length() * math.log10(2)
        with pytest.raises(ValueError):
            repr(UNRENDERABLE_INT)


# --------------------------------------------------------------------------- #
# The boundary of this change, stated rather than omitted                      #
# --------------------------------------------------------------------------- #
class TestTheIterationIsAnsweredNotEscaped:
    """``finite_vector_error`` answers a hostile iteration instead of raising (#1878).

    Replaces ``TestTheIterationEscapeStaysOutOfScope``, which pinned the opposite
    for as long as the escape was a stated boundary rather than a fixed defect.
    The conclusion that class supported still holds and is still measured here -
    only one guard ever reached an iteration, so the surface this closes is one
    ``iter()`` call - but the verdict on that call is now a message.

    This is the fourth escape in the family: rendering (#1873), scalar conversion
    (#1874), container conversion and rendering (#1875), and iteration here. All
    four are the same defect - a guard whose entire purpose is to answer an
    unusable input with a structured refusal, raising instead.

    It was recorded as the *last* of them, and that did not hold twice over. The
    shared length probe every vector guard runs first carried the same shape and
    was closed by #1888, and the element access one level inside this guard's own
    loop carried it until #1889, now measured in
    :class:`TestElementProductionIsAnsweredNotEscaped`. "Last" is an assertion
    about everything that was not measured, so the family is named here by what it
    contains rather than by being finished.
    """

    def test_a_hostile_iteration_is_refused_rather_than_raised(self) -> None:
        message = finite_vector_error("raycast", "origin", HostileIteration())
        assert message is not None
        assert "raycast: 'origin' could not be iterated" in message

    def test_the_refusal_names_the_exception_that_stopped_the_iteration(self) -> None:
        """The type and text are what make the message actionable.

        Without them the caller learns only that *something* went wrong inside a
        value they own, which is the diagnostic content of the traceback this
        replaces minus the traceback.
        """
        message = finite_vector_error("raycast", "origin", HostileIteration())
        assert message is not None
        assert "RuntimeError" in message
        assert "no iteration for you" in message

    def test_it_does_not_claim_the_value_was_not_a_list_of_numbers(self) -> None:
        """The two verdicts are different measurements and must not share text.

        A value whose ``__iter__`` raised may well have held numbers; this guard
        never found out. Reusing the ``TypeError`` text would report a domain
        check that never ran - which is why #1878 was not folded into #1875.
        """
        hostile = finite_vector_error("raycast", "origin", HostileIteration())
        not_iterable = finite_vector_error("raycast", "origin", UniterableContainer())
        assert hostile is not None and not_iterable is not None
        assert "must be a list/tuple of numbers" in not_iterable
        assert "must be a list/tuple of numbers" not in hostile
        assert hostile != not_iterable

    def test_a_plain_non_iterable_still_gets_the_type_error_text(self) -> None:
        """The ``TypeError`` branch is the common case and is unchanged."""
        message = finite_vector_error("raycast", "origin", 1.0)
        assert message is not None
        assert "must be a list/tuple of numbers" in message

    def test_coerce_size_vector_inherits_the_answer_it_inherited_the_raise_from(self) -> None:
        """``coerce_size_vector`` reaches this guard, so the fix reaches it too.

        It was the only other exposed surface, by inheritance rather than by its
        own ``iter()`` call, so it is the one place a fix could have been missed.
        """
        _value, error = coerce_size_vector("add_object", "size", HostileIteration())
        assert error is not None
        assert "could not be iterated" in error

    def test_an_exception_whose_own_str_raises_does_not_reescape(self) -> None:
        """The fix must not reintroduce #1873 inside its own message.

        A value hostile enough to raise a non-``TypeError`` from ``__iter__`` is
        not one whose exception is assumed to render, so the exception text goes
        through the same safe renderer every other refusal uses.
        """
        message = finite_vector_error("raycast", "origin", UnprintableFailure())
        assert message is not None
        assert "could not be iterated" in message
        assert "UnprintableFailureError" in message

    def test_the_other_guards_answer_this_value_before_they_reach_an_iteration(self) -> None:
        """Unchanged, and the reason the surface closed by *this* call is one ``iter()``.

        ``pose_vector_error`` asks for a length first - through the shared probe
        since #1888, with its own ``len()`` before that - and ``name_list_error``
        tests ``isinstance(value, Sequence)``, so this probe is refused by those
        guards before either reaches an iteration.

        That is a property of the probe, not a general exemption: it carries no
        length and is no ``Sequence``. A value that clears those first checks does
        reach the iteration through them, which is why the element-production
        escape had a wider surface than this one - see
        :class:`TestElementProductionIsAnsweredNotEscaped`.
        """
        assert pose_vector_error("add_object", "position", HostileIteration(), 3) is not None
        assert name_list_error(HostileIteration(), "cameras", "render_all") is not None

    def test_the_rendering_half_was_already_closed(self) -> None:
        """The renderer was never what left this open, and still is not."""
        assert _refusal_container_repr(HostileIteration()) == "<unrepresentable HostileIteration>"


class LazySizedVector:
    """A legacy sequence: a readable ``__len__`` and components read one at a time.

    ``GetItemOnly`` below is the failing counterpart. This one succeeds, so it is
    what shows the element loop accepting a complete lazy read - the property a
    generator used to carry here before an unreadable length became part of the
    verdict (#2200).
    """

    def __init__(self, *values: float) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> float:
        return self._values[index]


class GetItemOnly:
    """A sequence by the legacy protocol: ``__getitem__``, no ``__iter__``.

    CPython synthesises an iterator for such a value *without* calling
    ``__getitem__``, so ``iter()`` succeeds and the first ``next()`` is what
    raises. This is the 0-d array's own shape - ``simulation/base.py`` notes such
    a value "declares ``__len__`` and ``__getitem__``" - so it is a shape the
    library documents itself as receiving, not an invented one.
    """

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> float:
        raise RuntimeError("backing store unavailable")


def failing_generator() -> Iterator[float]:
    """Yields once, then fails. ``iter()`` on a generator cannot fail at all."""
    yield 0.1
    raise RuntimeError("stream truncated")


class MutatedWhileRead:
    """Grows during its own iteration, so the stdlib raises, not the value."""

    def __init__(self) -> None:
        self._items = {"a": 1.0, "b": 2.0}

    def __iter__(self) -> Iterator[float]:
        for key in self._items:
            self._items[key + "x"] = 1.0
            yield self._items[key]


class UnprintableItemFailure:
    """A legacy sequence whose item access raises an exception that cannot be rendered.

    The ``__next__`` counterpart of :class:`UnprintableFailure`. The refusal for a
    failed read interpolates an exception supplied by the same hostile value, so
    without this probe that interpolation is the #1873 rendering escape again, one
    level inside the fix for #1889.
    """

    def __repr__(self) -> str:
        raise RuntimeError("no repr for you")

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> float:
        raise UnprintableFailureError


class CountedRead:
    """Yields ``values`` one at a time, recording how many were produced.

    The guard's laziness is a documented property - it reads a component at a
    time rather than materialising the vector - and a property nothing measures is
    one a later rewrite can drop without noticing.
    """

    def __init__(self, *values: Any) -> None:
        self.values = values
        self.produced = 0

    def __iter__(self) -> Iterator[Any]:
        for value in self.values:
            self.produced += 1
            yield value


class TestElementProductionIsAnsweredNotEscaped:
    """A read that fails part-way is refused, not raised (#1889).

    Replaces ``TestElementAccessStaysOutOfScope``, which pinned the opposite while
    the escape was a stated boundary rather than a fixed defect. ``iter(vec)`` was
    probed inside a ``try`` and the iterator then walked by an unguarded ``for``,
    so an exception raised while *producing* an element escaped exactly as one
    from ``__iter__`` used to - the same defect as #1873, #1874, #1875 and #1878,
    one level in.

    Three routes reach it and none needs a hostile type. ``iter()`` of a generator
    cannot fail, so the probe was structurally blind to a generator that fails
    after its first yield; CPython synthesises the iterator for a legacy
    ``__getitem__`` sequence *without* calling ``__getitem__``, which is the 0-d
    array's own shape this library documents itself as receiving; and a container
    mutated during its own read raises from the stdlib rather than from the value.

    The verdict is not the ``__iter__`` one reworded. A read that stopped at
    element 4 had four components read and found finite, which is a different
    measurement from a value whose iteration never began, so it names the element
    it stopped at - #1878's own lesson, that a refusal must not state what was
    never measured.
    """

    def test_a_legacy_sequence_whose_item_access_raises_is_refused(self) -> None:
        """``iter()`` succeeds on this value, so only a guarded read answers it."""
        assert iter(GetItemOnly()) is not None
        message = finite_vector_error("raycast", "origin", GetItemOnly())
        assert message is not None
        assert "raycast: 'origin[0]' could not be read" in message

    def test_the_refusal_names_the_exception_that_stopped_the_read(self) -> None:
        """The type and text are what make it actionable, as on the ``__iter__`` half."""
        message = finite_vector_error("raycast", "origin", GetItemOnly())
        assert message is not None
        assert "RuntimeError" in message
        assert "backing store unavailable" in message

    def test_a_generator_that_fails_after_its_first_yield_names_where_it_stopped(self) -> None:
        """The index is the measurement: element 0 was read and was finite.

        This is the whole reason the verdict is not the ``__iter__`` text - a
        value that produced good components before failing is not one that held
        nothing readable.
        """
        message = finite_vector_error("raycast", "origin", failing_generator())
        assert message is not None
        assert "'origin[1]' could not be read" in message
        assert "stream truncated" in message

    def test_a_container_mutated_during_its_own_read_is_refused(self) -> None:
        """The exception is the stdlib's; the value raises nothing of its own."""
        message = finite_vector_error("raycast", "origin", MutatedWhileRead())
        assert message is not None
        assert "could not be read" in message
        assert "changed size during iteration" in message

    def test_it_does_not_claim_the_value_was_not_a_list_of_numbers(self) -> None:
        """The domain check never ran on the element that could not be produced."""
        message = finite_vector_error("raycast", "origin", GetItemOnly())
        assert message is not None
        assert "must be a list/tuple of numbers" not in message

    def test_the_two_iteration_verdicts_are_not_the_same_text(self) -> None:
        """Distinct measurements, so distinct messages.

        A shared verdict would be the failure #1878 avoided when it declined to
        fold itself into #1875: reporting a check that did not run.
        """
        part_way = finite_vector_error("raycast", "origin", GetItemOnly())
        never_started = finite_vector_error("raycast", "origin", HostileIteration())
        assert part_way is not None and never_started is not None
        assert "could not be iterated" in never_started
        assert "could not be iterated" not in part_way
        assert part_way != never_started

    def test_every_guard_reaching_this_iteration_inherits_the_answer(self) -> None:
        """The surface is four calls, not the one ``iter()`` #1878 closed.

        A legacy sequence carries a readable ``__len__``, so it clears the length
        check that refused #1878's probe and reaches the iteration through
        ``pose_vector_error`` and ``coerce_rgba`` as well - which is why the fix
        belongs in the shared guard and not at a call site.
        """
        assert pose_vector_error("add_object", "position", GetItemOnly(), 3) is not None
        assert coerce_pose_vector("add_object", "position", GetItemOnly(), 3)[1] is not None
        assert coerce_rgba("add_object", "color", GetItemOnly())[1] is not None
        assert coerce_size_vector("add_object", "size", GetItemOnly())[1] is not None

    def test_an_exception_whose_own_str_raises_does_not_reescape(self) -> None:
        """The fix must not reintroduce #1873 inside its own message."""
        message = finite_vector_error("raycast", "origin", UnprintableItemFailure())
        assert message is not None
        assert "could not be read" in message
        assert "UnprintableFailureError" in message

    def test_the_iter_half_still_answers(self) -> None:
        """Non-vacuity: the ``__iter__`` verdict is unchanged, not absorbed."""
        message = finite_vector_error("raycast", "origin", HostileIteration())
        assert message is not None
        assert "could not be iterated" in message

    def test_an_acceptable_lazy_vector_is_still_accepted(self) -> None:
        """The rewritten loop must not refuse what the ``for`` accepted.

        The ``StopIteration`` that ends the read is the read finishing, not a
        failure, and this is the property that pins it. The vehicle is a legacy
        sequence rather than a generator because the two are not the same claim: a
        generator also has no readable length, and the guard refuses that
        separately - its callers count the components by reading the value again,
        which a consumed value cannot answer (#2200). ``LazySizedVector`` is read
        one component at a time through ``__getitem__``, so it exercises the same
        loop and the same ``StopIteration`` while leaving that second question
        answerable. An empty one is here for the same reason it was before: zero
        components is a complete read.
        """
        assert finite_vector_error("raycast", "origin", LazySizedVector(0.1, 0.2, 0.3)) is None
        assert finite_vector_error("raycast", "origin", LazySizedVector()) is None

    def test_the_index_is_a_position_and_not_a_constant(self) -> None:
        """Non-vacuity of the index: three values, three different positions.

        Without this, a hard-coded ``[0]`` would satisfy every assertion above
        while naming the wrong element for every value but the first.
        """

        def fails_at_element(position: int) -> Iterator[float]:
            for _ in range(position):
                yield 0.1
            raise RuntimeError("halted")

        for position in (0, 1, 2):
            message = finite_vector_error("raycast", "origin", fails_at_element(position))
            assert message is not None
            assert f"'origin[{position}]' could not be read" in message

    def test_the_read_is_still_one_component_at_a_time(self) -> None:
        """The laziness the guard documents, measured rather than asserted in prose.

        A materialising ``list(vec)`` would answer for ``__next__`` too, and would
        read every component before examining any - so it could neither stop at the
        first unusable one nor say how far the read got.
        """
        vector = CountedRead(0.1, "not a number", 0.3)
        assert finite_vector_error("raycast", "origin", vector) is not None
        assert vector.produced == 2


class HostileNameSeq(Sequence[Any]):
    """A registered ``Sequence`` whose item access raises - #1897's own probe.

    ``name_list_error`` tests ``isinstance(value, Sequence)`` and then walked the
    value, so this cleared the type check and escaped from the walk.
    ``collections.abc.Sequence`` is what a proxy or a lazily-backed name list
    subclasses in order to be accepted here at all, which is why the type check
    passing says nothing about the read succeeding.
    """

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: Any) -> Any:
        raise RuntimeError("seq item exploded")


class NamesThenFailure(Sequence[Any]):
    """Produces two usable names and then fails, so the index is a measurement."""

    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: Any) -> Any:
        if index < 2:
            return ("top", "wrist")[index]
        raise RuntimeError("backing store unavailable")


class HostileNameIteration(Sequence[Any]):
    """A ``Sequence`` whose ``__iter__`` raises, so no entry is ever produced."""

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: Any) -> Any:
        return "wrist"

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("no iteration for you")


class UnprintableNameFailure(Sequence[Any]):
    """Item access raises an exception that cannot itself be rendered.

    The refusal for a failed read interpolates an exception supplied by the same
    hostile value, so without this probe that interpolation is #1873's rendering
    escape again, one level inside the fix for #1897.
    """

    def __repr__(self) -> str:
        raise RuntimeError("no repr for you")

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: Any) -> Any:
        raise UnprintableFailureError


class CountedNameSeq(Sequence[Any]):
    """A ``Sequence`` recording how many times it was read from start to end.

    The read count is the invariant #1897 turns on. It is a property of the
    guard, not of any one message, so nothing in the refusal text would reveal a
    regression to two reads.
    """

    def __init__(self, *names: Any) -> None:
        self.names = names
        self.reads = 0

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> Iterator[Any]:
        self.reads += 1
        return iter(self.names)

    def __getitem__(self, index: Any) -> Any:
        return self.names[index]


class DifferentOnItsSecondRead(Sequence[Any]):
    """Answers its first read with distinct names and its second with a repeat.

    Nothing obliges a ``Sequence`` to answer two reads the same way, and the
    guard used to run its per-entry checks against one read and its duplicate
    check against another.
    """

    def __init__(self) -> None:
        self.reads = 0

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[str]:
        self.reads += 1
        return iter(("top", "wrist") if self.reads == 1 else ("top", "top"))

    def __getitem__(self, index: Any) -> Any:
        return ("top", "wrist")[index]


class FailsOnItsSecondRead(Sequence[Any]):
    """One clean read, then raises - so only the duplicate walk reached it."""

    def __init__(self) -> None:
        self.reads = 0

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[str]:
        self.reads += 1
        if self.reads > 1:
            raise RuntimeError("second read refused")
        return iter(("top", "wrist"))

    def __getitem__(self, index: Any) -> Any:
        return ("top", "wrist")[index]


class RepeatedThenFailsOnItsThirdRead(Sequence[Any]):
    """Two clean reads naming a repeat, then raises - the refusal's own read."""

    def __init__(self) -> None:
        self.reads = 0

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterator[str]:
        self.reads += 1
        if self.reads > 2:
            raise RuntimeError("third read refused")
        return iter(("top", "top"))

    def __getitem__(self, index: Any) -> Any:
        return "top"


class HostileKeyMapping(Mapping[str, str]):
    """A ``Mapping`` whose key iteration raises, reached by the remedy's read."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> str:
        return "camera"

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("no keys for you")


class UnprintableKeyFailureMapping(Mapping[str, str]):
    """A ``Mapping`` whose key read raises an exception that cannot print itself."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, key: str) -> str:
        return "camera"

    def __iter__(self) -> Iterator[str]:
        raise UnprintableFailureError


class HostileCharacterStr(str):
    """A ``str`` whose own iteration raises, reached by the consequence clause's read.

    A ``str`` subclass is the shape a name arrives as when a caller wraps one -
    an interned label, a path-like, an enum member's value - so overriding
    ``__iter__`` is not the only route here, merely the clearest.
    """

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("no characters for you")


class HostileLengthStr(str):
    """A ``str`` whose ``__len__`` raises, which the character count used to call."""

    def __len__(self) -> int:
        raise RuntimeError("no length for you")


class UnprintableCharacterStr(str):
    """A ``str`` that iterates into elements whose ``repr`` raises.

    The container-renderer half of the same branch: reading the characters can
    succeed and *quoting* them still escape, which is #1875's defect one level
    inside the clause that quotes them.
    """

    def __iter__(self) -> Iterator[Any]:
        return iter([Unprintable()])


class UndecodableBytes(bytes):
    """``bytes`` whose ``decode`` raises, which the branch calls before any read.

    ``bytes`` is the one input class this branch *transforms* before quoting it,
    so producing the text to quote is itself a read of the caller's value.
    """

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        raise RuntimeError("no decoding for you")


class TestTheNameListIsReadOnceAndAnsweredNotEscaped:
    """``name_list_error`` reads the caller's value once, and answers a read that fails (#1897).

    The seventh escape in the family - rendering (#1873), scalar conversion
    (#1874), container conversion (#1875), ``__iter__`` (#1878), the shared length
    probe (#1888), element production (#1889) - and the one member
    :class:`TestElementProductionIsAnsweredNotEscaped` could not reach, because
    ``name_list_error`` refuses that class's probes on ``isinstance(value,
    Sequence)`` before arriving at a walk.

    The defect was wider than a walk. The guard read the caller's value on every
    branch that needed it and guarded none of them: the per-entry walk, the
    duplicate walk, and the duplicate refusal's own ``list(value)``. So an
    acceptable value was read **twice** and a repeated one **three times**, and a
    value that answers one read could break the guard on any of the others.

    Reading once also settles a verdict that had nothing to do with an exception.
    The reads were independent, so a value whose contents differ between two of
    them was refused against a list no check had examined - measured by
    :meth:`test_the_duplicate_verdict_is_about_the_entries_that_were_checked`.

    The read is still entry by entry rather than ``list(value)``, so the refusal
    can name where it stopped. That is the same reason #1889 declined to
    materialise, reached from the opposite direction: this guard collects the
    whole value either way, because a repeat cannot be ruled out without reading
    every entry, so laziness buys it no earlier verdict - only the index.
    """

    def test_a_registered_sequence_whose_item_access_raises_is_refused(self) -> None:
        """The escape #1897 filed, with the probe it filed it with."""
        assert isinstance(HostileNameSeq(), Sequence)
        message = name_list_error(HostileNameSeq(), "cameras", "render_all")
        assert message is not None
        assert "render_all: cameras[0] could not be read" in message

    def test_the_refusal_names_the_exception_that_stopped_the_read(self) -> None:
        """The type and text are the diagnostic content of the traceback it replaces."""
        message = name_list_error(HostileNameSeq(), "cameras", "render_all")
        assert message is not None
        assert "RuntimeError" in message
        assert "seq item exploded" in message

    def test_a_read_that_fails_part_way_names_the_entry_it_stopped_at(self) -> None:
        """Two names were read and found usable, which the index states."""
        message = name_list_error(NamesThenFailure(), "cameras", "render_all")
        assert message is not None
        assert "cameras[2] could not be read" in message
        assert "backing store unavailable" in message

    def test_the_index_is_a_position_and_not_a_constant(self) -> None:
        """Non-vacuity of the index: a hard-coded ``[0]`` would pass everything above."""

        class FailsAt(Sequence[Any]):
            def __init__(self, position: int) -> None:
                self.position = position

            def __len__(self) -> int:
                return self.position + 1

            def __getitem__(self, index: Any) -> Any:
                if index < self.position:
                    return f"cam{index}"
                raise RuntimeError("halted")

        for position in (0, 1, 2):
            message = name_list_error(FailsAt(position), "cameras", "render_all")
            assert message is not None
            assert f"cameras[{position}] could not be read" in message

    def test_a_read_that_never_began_is_reported_without_an_index(self) -> None:
        """No entry was produced, so there is none to name - the index is the measurement.

        The two stems are ``finite_vector_error``'s own, so this introduces no
        vocabulary; what differs is the remedy, which is worded for names.
        """
        message = name_list_error(HostileNameIteration(), "cameras", "render_all")
        assert message is not None
        assert "render_all: cameras could not be iterated" in message
        assert "no iteration for you" in message
        assert "could not be read" not in message
        assert "cameras[0]" not in message

    def test_it_does_not_claim_the_entry_was_not_a_name(self) -> None:
        """The domain check never ran on an entry that could not be produced.

        Reusing the per-entry text would report a check that did not run, which
        is #1878's lesson and the reason that fix was not folded into #1875.
        """
        unread = name_list_error(HostileNameSeq(), "cameras", "render_all")
        not_a_name = name_list_error([1], "cameras", "render_all")
        assert unread is not None and not_a_name is not None
        assert "must be a name (str)" in not_a_name
        assert "must be a name (str)" not in unread
        assert "must be a list of names" not in unread

    def test_an_exception_whose_own_str_raises_does_not_reescape(self) -> None:
        """The fix must not reintroduce #1873 inside its own message."""
        message = name_list_error(UnprintableNameFailure(), "cameras", "render_all")
        assert message is not None
        assert "could not be read" in message
        assert "UnprintableFailureError" in message

    def test_the_value_is_read_exactly_once(self) -> None:
        """The invariant no message would reveal, on all three of its outcomes.

        An accepted value was read twice before this and a repeated one three
        times, so a count is the only assertion that can hold the guard to one.
        """
        accepted = CountedNameSeq("top", "wrist")
        assert name_list_error(accepted, "cameras", "render_all") is None
        assert accepted.reads == 1

        per_entry = CountedNameSeq("top", 1)
        assert name_list_error(per_entry, "cameras", "render_all") is not None
        assert per_entry.reads == 1

        repeated = CountedNameSeq("top", "top")
        assert name_list_error(repeated, "cameras", "render_all") is not None
        assert repeated.reads == 1

    def test_the_duplicate_verdict_is_about_the_entries_that_were_checked(self) -> None:
        """A refusal must not be computed from a read no check examined.

        This value has no hostile behaviour and raises nothing. It was refused as
        a repeat on the strength of its *second* read, having cleared the
        per-entry checks against its first, in a message rendering a third - so
        the verdict, the check behind it and the text quoting it described three
        different lists. One read makes them the same list by construction.
        """
        changing = DifferentOnItsSecondRead()
        assert name_list_error(changing, "cameras", "render_all") is None
        assert changing.reads == 1

    def test_a_value_that_refuses_its_second_read_is_never_asked_for_one(self) -> None:
        """The duplicate walk was a second read and could fail on its own."""
        value = FailsOnItsSecondRead()
        assert name_list_error(value, "cameras", "render_all") is None
        assert value.reads == 1

    def test_the_duplicate_refusal_does_not_read_the_value_again_to_quote_it(self) -> None:
        """``list(value)`` inside the message was a third read, and the last escape.

        The repeat is still reported, and still quotes the entries - from the one
        read, so a value that cannot answer a third is refused for the repeat it
        has rather than for the read the message wanted.
        """
        value = RepeatedThenFailsOnItsThirdRead()
        message = name_list_error(value, "cameras", "render_all")
        assert message is not None
        assert "must not repeat a name" in message
        assert "['top', 'top']" in message
        assert value.reads == 1

    def test_acceptable_and_empty_name_lists_are_unchanged(self) -> None:
        """The rewritten read must not refuse what the two walks accepted.

        Emptiness means "not supplied" to every caller, so it is not this
        function's to reject - the ``StopIteration`` that ends the read is the
        read finishing, not a failure.
        """
        assert name_list_error(["top", "wrist"], "cameras", "render_all") is None
        assert name_list_error(("top", "wrist"), "cameras", "render_all") is None
        assert name_list_error([], "cameras", "render_all") is None
        assert name_list_error((), "cameras", "render_all") is None

    def test_the_sibling_guards_verdicts_are_untouched(self) -> None:
        """Non-vacuity in the other direction: this fix is local to one guard.

        ``finite_vector_error``'s two read verdicts are the text these reuse, so a
        change made in the wrong place would show up as one of them moving.
        """
        assert "could not be read" in str(finite_vector_error("raycast", "origin", GetItemOnly()))
        assert "could not be iterated" in str(finite_vector_error("raycast", "origin", HostileIteration()))

    def test_a_mapping_whose_keys_cannot_be_read_keeps_its_verdict(self) -> None:
        """Replaces the #1903 boundary this class stated: the mapping verdict survives.

        The pin this replaces asserted the escape - ``pytest.raises`` on the
        remedy's own ``list(value)``. What made it a boundary rather than a bug
        to absorb was that the verdict was never in doubt on this branch, so the
        question was what a refusal says when only its *advice* is unmeasurable,
        and that is a message-design decision rather than a guarded read.

        It is answered by keeping the verdict and degrading the remedy, which is
        the only one of #1903's three candidates that states both what was
        measured and what was not. Dropping the verdict would stop naming the
        mistake the caller actually made, and dropping the remedy silently would
        omit advice on one path without saying why.
        """
        assert isinstance(HostileKeyMapping(), Mapping)
        message = name_list_error(HostileKeyMapping(), "cameras", "render_all")
        assert message is not None
        assert "must be a list of names, not a mapping" in message
        assert "its own keys could not be read to quote them here" in message
        assert "RuntimeError: no keys for you" in message


class TestARefusalDegradesWhenItCannotQuoteWhatItRefuses:
    """``name_list_error``'s text reads the caller's value, and those reads are guarded (#1903).

    The eighth and last member of the family - rendering (#1873), scalar
    conversion (#1874), container conversion (#1875), ``__iter__`` (#1878), the
    shared length probe (#1888), element production (#1889), the name-list read
    count (#1897) - and the only one where the read was never load-bearing.

    Every other member guarded a read the *verdict* needed, so a read that failed
    became the verdict. Here the verdict is settled before the read happens: a
    mapping is refused whatever its keys say, and a string whatever its
    characters are. So the read exists only to quote something, and the decision
    #1903 asked for is what the refusal says when it cannot. It keeps the verdict
    and degrades the clause that wanted the quotation, naming the read failure
    with ``_describe_failed_read`` - the stem ``_read_name_list`` already uses -
    so a refusal degraded here reads like every other read failure in the module.

    #1903 described one read, on the mapping branch. There are five, across two
    branches, and the four the issue did not have are asserted below: the string
    branch produced its characters with a comprehension, counted them with a
    second ``len`` the first was not obliged to agree with, rendered them with a
    bare f-string rather than the elementwise renderer, and decoded ``bytes``
    with an overridable method before any of that.
    """

    def test_a_string_whose_characters_cannot_be_read_keeps_its_verdict(self) -> None:
        """Not in #1903, and the same shape one branch up.

        ``[c for c in shown]`` built the consequence clause and was as unguarded
        as the mapping's ``list(value)``.
        """
        message = name_list_error(HostileCharacterStr("wrist"), "cameras", "render_all")
        assert message is not None
        assert "must be a list of names, not a single string" in message
        assert "its own characters could not be read to quote them here" in message
        assert "RuntimeError: no characters for you" in message

    def test_the_character_count_comes_from_the_read_that_produced_them(self) -> None:
        """``len(shown)`` was a second read, so a hostile ``__len__`` escaped too.

        This is #1897's property applied to a message rather than to a verdict:
        the quotation and the count beside it now come from one read, so they
        cannot describe different values.
        """
        message = name_list_error(HostileLengthStr("wrist"), "cameras", "render_all")
        assert message is not None
        assert "RuntimeError: no length for you" in message

    def test_the_quoted_characters_go_through_the_elementwise_renderer(self) -> None:
        """Reading the characters can succeed and quoting them still escape.

        The clause interpolated the list directly, so ``repr`` recursed into an
        element that cannot print - #1875's defect, inside the fix for #1903's.
        """
        message = name_list_error(UnprintableCharacterStr("w"), "cameras", "render_all")
        assert message is not None
        assert "<unrepresentable Unprintable>" in message
        assert "(1 name(s))" in message

    def test_bytes_whose_decode_raises_keeps_its_verdict(self) -> None:
        """``bytes`` is the one input class this branch transforms before quoting it.

        So producing the text is a read of the caller's value, and it ran outside
        the guard - the earliest of the five escapes on this branch.
        """
        message = name_list_error(UndecodableBytes(b"wrist"), "cameras", "render_all")
        assert message is not None
        assert "must be a list of names, not a single string" in message
        assert "RuntimeError: no decoding for you" in message

    def test_a_degraded_remedy_still_offers_the_remedy(self) -> None:
        """Candidate 3 - drop the clause - is excluded by measurement, not by prose.

        The advice is what the caller acts on, so the refusal keeps saying to
        pass a list and reports only that it could not quote the names.
        """
        mapping = name_list_error(HostileKeyMapping(), "cameras", "render_all")
        string = name_list_error(HostileCharacterStr("wrist"), "cameras", "render_all")
        assert mapping is not None and string is not None
        assert "pass the names as a list" in mapping
        assert "Wrap it in a list: ['wrist']." in string

    def test_a_degraded_refusal_does_not_claim_a_check_that_did_not_run(self) -> None:
        """#1878's lesson: the entries were never read, so no entry verdict may appear."""
        message = name_list_error(HostileKeyMapping(), "cameras", "render_all")
        assert message is not None
        assert "must be a name (str)" not in message
        assert "must not repeat a name" not in message
        assert "could not be iterated" not in message

    def test_an_exception_whose_own_str_raises_does_not_reescape(self) -> None:
        """The fix must not reintroduce #1873 inside its own degraded clause."""
        message = name_list_error(UnprintableKeyFailureMapping(), "cameras", "render_all")
        assert message is not None
        assert "could not be read to quote them here" in message
        assert "UnprintableFailureError" in message

    def test_no_accepted_or_refused_message_moved(self) -> None:
        """The whole point of degrading is that nothing else changes.

        Asserted as exact text rather than as substrings, because a rewrite of
        two message branches is precisely the change that moves a word nobody
        looks at. The ``(5 name(s))`` count and the ``['top', 'wrist']`` remedy
        are the two values now taken from the single read.
        """
        assert name_list_error(["top", "wrist"], "cameras", "render_all") is None
        assert name_list_error("wrist", "cameras", "render_all") == (
            "render_all: cameras must be a list of names, not a single string, got 'wrist'. "
            "A string is iterable per character, so this would be read as "
            "['w', 'r', 'i', 's', 't'] (5 name(s)). Wrap it in a list: ['wrist']."
        )
        assert name_list_error(b"wrist", "cameras", "render_all") == (
            "render_all: cameras must be a list of names, not a single string, got b'wrist'. "
            "A string is iterable per character, so this would be read as "
            "['w', 'r', 'i', 's', 't'] (5 name(s)). Wrap it in a list: ['wrist']."
        )
        assert name_list_error("shoulder", "cameras", "render_all") == (
            "render_all: cameras must be a list of names, not a single string, got 'shoulder'. "
            "A string is iterable per character, so this would be read as "
            "['s', 'h', 'o', 'u', 'l', 'd'] ... (8 name(s)). Wrap it in a list: ['shoulder']."
        )
        assert name_list_error({"top": 1, "wrist": 2}, "cameras", "render_all") == (
            "render_all: cameras must be a list of names, not a mapping, "
            "got {'top': 1, 'wrist': 2}. A mapping is iterable over its keys, so its values "
            "would be discarded - pass the names as a list: ['top', 'wrist']."
        )

    def test_the_probes_are_the_types_the_branches_dispatch_on(self) -> None:
        """Non-vacuity: a probe that missed its branch would pass every test above.

        ``UnprintableCharacterStr`` is checked for the property it is *for* -
        readable characters that cannot be quoted - since a probe that raised on
        the read instead would satisfy this class's other assertions by accident.
        """
        assert isinstance(HostileCharacterStr("w"), str)
        assert isinstance(HostileLengthStr("w"), str)
        assert isinstance(UndecodableBytes(b"w"), bytes)
        assert isinstance(UnprintableKeyFailureMapping(), Mapping)
        assert len(list(UnprintableCharacterStr("w"))) == 1
        with pytest.raises(RuntimeError, match="no repr for you"):
            repr(list(UnprintableCharacterStr("w")))
