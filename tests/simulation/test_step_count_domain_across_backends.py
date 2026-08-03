"""Regression tests: every backend validates the ``step`` count it is given.

``step(n_steps)`` is the most-called method on the simulation surface, and it
was the last public numeric input with three different domains. The MuJoCo
backend has documented one since its numeric inputs were hardened - "Non-negative
step count (``0`` is an accepted no-op)" - but implemented it by hand, and
Newton and Isaac validated nothing at all.

Measured on the probe set below, one ``step`` per cell, with no ``newton`` /
``warp`` / ``isaacsim`` installed. Every guard precedes the point a backend
touches a solver, a stage or its lock, which is what lets these run solver-free.

| ``n_steps=`` | MuJoCo | Newton | Isaac |
| --- | --- | --- | --- |
| ``0`` | success, no-op (documented) | success, **advanced 1** | success, 0 steps |
| ``-5`` | error | success, **advanced 1** | **success, advanced 0** |
| ``True`` | **success, advanced 1** | success, **advanced 1** | **success, advanced 1** |
| ``np.True_`` | **success, advanced 1** | success, **advanced 1** | raised ``TypeError`` |
| ``2.7`` | **success, advanced 2** | ``step_count`` became **2.7** / raised | raised ``TypeError`` |
| ``"3"`` | **success, advanced 3** | raised ``TypeError`` | raised ``TypeError`` |
| ``nan`` | error | success, **advanced 1** | raised ``TypeError`` |
| ``inf`` | **raised ``OverflowError``** | ``step_count`` became **inf** / raised | raised ``TypeError`` |
| ``None`` / ``[3]`` | error | raised ``TypeError`` | raised ``TypeError`` |

Five findings, in the order they cost something:

* **A negative count was a silent no-op on Isaac.** ``range(-5)`` is empty, so
  the call stepped nothing and reported SUCCESS - then divided the elapsed wall
  time by that negative count and reported the rate as
  ``-11876485 steps/sec``. An agent reading "Stepped -5x" has been told the
  world advanced when it did not.
* **``step(0)`` advanced the world on Newton.** ``_advance`` floors its count at
  ``max(1, n_steps)``, so a zero - the value MuJoCo documents as a no-op -
  stepped once while the result text said ``Stepped 0 step(s).``. The report and
  the world disagreed, and the same call was a no-op on one backend and a step
  on another. ``-5`` and ``nan`` reached that same floor and also advanced 1,
  under ``Stepped -5 step(s).`` / ``Stepped nan step(s).``.
* **``inf`` escaped MuJoCo's own envelope.** ``int(float("inf"))`` raises
  ``OverflowError``, which the hand-rolled ``except (TypeError, ValueError)``
  did not catch - so the one backend that documented this domain raised a bare
  exception through the structured result these methods document as their only
  failure channel. On Newton's solver-free path the same value left
  ``step_count`` and ``sim_time`` as ``inf`` PERMANENTLY, so every later
  ``get_state()`` reported ``t=inf``.
* **A boolean was read as a count of one on all three.** The defect the runtime
  writers removed for their own inputs (``_ANY_NUMBERS`` in
  ``test_input_validators_refuse_a_boolean``): ``bool`` is an ``int`` subclass,
  so ``True`` survived every gate as one physics step.
* **MuJoCo truncated a fractional count under a success result.** ``int(2.7)``
  is ``2``, so a caller asking for 2.7 steps got 2 and was told it succeeded,
  while its docstring promised an error when ``n_steps`` is "not an integer".
  Newton's behaviour for the same value depended on whether a solver had been
  constructed: solver-free it wrote the FLOAT ``2.7`` into the integer
  ``step_count``, and with a solver it raised out of ``range()``.

The fix is the shared :func:`~strands_robots.utils.non_negative_whole_number_error`
domain applied by all three, then a single ``int()`` coercion that is safe
because the guard has already performed that same conversion and compared the
result back.

That guard returns a verdict for every real scalar and raises for none, which is
the contract rather than a detail: all three ``step`` docstrings name the
structured result as the only channel an out-of-domain count is reported on, and
one caller takes its count from a remote process. Two values are the reason the
implementation looks the way it does, and both are pinned by
``TestEveryRealGetsAVerdictAndNothingRaises``. ``float(10**400)`` raises
``OverflowError``, so integrality is tested with an ``int()`` in a ``try`` rather
than a ``float()`` round-trip. And ``repr`` of an ``int`` past
:func:`sys.get_int_max_str_digits` raises ``ValueError`` while that same count is
*accepted*, so the refusal text is rendered only when a refusal is returned -
work on the accept path that only the refuse path needs is the same shape as the
defects above.
That helper is the missing cell of an existing 2x2 - the same scalar policy as
:func:`~strands_robots.utils.positive_whole_number_error` with the floor at
``0``, standing to it exactly as
:func:`~strands_robots.utils.non_negative_count_error` stands to
:func:`~strands_robots.utils.positive_count_error`. Reusing either existing
non-negative or whole-number helper would have regressed the reference backend:
``non_negative_count_error`` accepts only a true ``int``, and MuJoCo honors
``3.0`` and ``np.int64(3)`` today.

What is NOT in scope, and is asserted to be unchanged by
``TestNeighbouringStepSurfacesStayOutOfScope``:

* The magnitude of a count on ``step`` (below). ``send_action(n_substeps=)`` is
  no longer listed here: it was settled separately as #1870, on
  ``positive_whole_number_error`` - the same scalar policy with the floor at
  ``1``, because that surface writes an actuator target before it advances and a
  ``0`` there leaves the target written and never integrated. ``step`` still
  owns the honored zero, which is why it answers its own rather than moving
  Newton's ``_advance`` floor; that floor is now unreachable from either public
  surface. See ``test_send_action_substep_domain_across_backends.py``.
* The magnitude of a count, which belongs to that ceiling and not to this
  domain. ``10**400`` is a non-negative whole number, so the domain accepts it
  and MuJoCo refuses it with its own ceiling error - exactly as it did before
  this change, where a true ``int`` skipped its ``int()`` coercion. Refusing it
  in the guard would have put a silent boundary at the float range while still
  accepting ``10**300``, which no backend can advance either.
* The per-call ceiling. MuJoCo refuses ``n_steps > _MAX_STEPS_PER_CALL``
  (100_000); Isaac and Newton have no equivalent. That is a resource policy
  rather than an input domain, and picking one ceiling for three backends with
  different per-step costs is a decision rather than a defect, so #1871 stays
  open on it.

  The *lock hold* that ceiling was partly justified by is no longer part of the
  asymmetry: all three backends now batch their loop on the shared
  ``SimEngine._STEPS_PER_BATCH`` and re-check the world on each boundary. Before
  that, Isaac accepted ``100_001`` and called ``world.step`` that many times
  inside one ``self._lock`` hold. See
  ``test_step_lock_hold_across_backends.py``, which also records that the
  batching MuJoCo already had was itself unsafe at the boundaries: a concurrent
  ``cleanup`` world handoff between two batches made ``step`` raise
  ``AttributeError`` past its structured envelope.
"""

from __future__ import annotations

import ast
import inspect
import math
import numbers
import pathlib
import textwrap
import threading
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.simulation import IsaacSimulation
from strands_robots.simulation.models import SimWorld
from strands_robots.simulation.mujoco.simulation import MuJoCoSimEngine
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from strands_robots.utils import non_negative_whole_number_error, positive_whole_number_error

NAN = float("nan")
INF = float("inf")

#: An ``int`` wider than the float range: ``float(10**400)`` raises
#: ``OverflowError``, which is how the guard's own integrality test used to
#: crash on it.
BEYOND_FLOAT_RANGE = 10**400
#: An ``int`` whose own ``repr`` raises: wider than
#: :func:`sys.get_int_max_str_digits` (4300 digits by default), so *rendering*
#: the refusal crashed before the value was even classified.
BEYOND_INT_STR_LIMIT = 10**5000

#: Every count no backend can advance by. Each row is a measured pre-fix
#: acceptance or a bare raise on at least one backend (see the module docstring).
UNUSABLE_COUNTS: tuple[Any, ...] = (
    -5,
    -1,
    True,
    False,
    np.bool_(True),
    2.7,
    "3",
    NAN,
    INF,
    None,
    [3],
    np.array([3]),
    -BEYOND_FLOAT_RANGE,
    -BEYOND_INT_STR_LIMIT,
)


def _count_id(value: Any) -> str | None:
    """Test ID for a probe count, or ``None`` to let pytest name it.

    Only the outsized integers need naming, and they need it to keep the module
    runnable: pytest derives an ID for an ``int`` parameter with ``str()``, which
    raises for one wider than :func:`sys.get_int_max_str_digits`. That is a
    collection error, not a test failure - it takes down every class in this
    file, the parity matrix included, and reports as an error rather than as
    coverage silently lost. The defect this module documents on the guard
    (rendering a value it was only asked to classify) has the same shape in the
    harness.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        if value == -BEYOND_FLOAT_RANGE:
            return "negative_beyond_float_range"
        if value == -BEYOND_INT_STR_LIMIT:
            return "negative_beyond_int_str_limit"
    return None


#: Counts every backend must honor, paired with the number of steps each must
#: advance. ``3.0`` and ``np.int64(3)`` are the load-bearing rows: MuJoCo
#: accepted both before this change (its ``int()`` coercion took them), so a
#: domain that refused them would be a regression on the reference backend, and
#: a step count computed as ``int(duration / dt)`` is an ``np.int64`` whenever
#: ``dt`` came from NumPy.
USABLE_COUNTS: tuple[tuple[Any, int], ...] = (
    (0, 0),
    (1, 1),
    (3, 3),
    (3.0, 3),
    (np.int64(3), 3),
    (np.uint8(2), 2),
    (np.float64(4.0), 4),
)


def _text(result: dict[str, Any]) -> str:
    return str(result["content"][0]["text"])


# --------------------------------------------------------------------------- #
# The shared domain                                                           #
# --------------------------------------------------------------------------- #
class TestTheSharedDomain:
    """``non_negative_whole_number_error`` is the single definition all three share."""

    @pytest.mark.parametrize("count", UNUSABLE_COUNTS, ids=_count_id)
    def test_an_unusable_count_is_refused(self, count: Any) -> None:
        error = non_negative_whole_number_error(count, "n_steps", "step")
        assert error is not None, count
        assert "n_steps" in error

    @pytest.mark.parametrize(("count", "_expected"), USABLE_COUNTS)
    def test_a_usable_count_is_accepted(self, count: Any, _expected: int) -> None:
        assert non_negative_whole_number_error(count, "n_steps", "step") is None, count

    def test_zero_is_accepted_and_one_below_it_is_not(self) -> None:
        """The floor is the whole reason this is not ``positive_whole_number_error``."""
        assert non_negative_whole_number_error(0, "n_steps", "step") is None
        assert non_negative_whole_number_error(-1, "n_steps", "step") is not None

    def test_an_accepted_count_survives_the_int_coercion_it_is_paired_with(self) -> None:
        """The guard exists so the ``int()`` that follows it cannot raise."""
        for count, expected in USABLE_COUNTS:
            assert non_negative_whole_number_error(count, "n_steps", "step") is None
            assert int(count) == expected

    def test_a_non_finite_count_is_refused_rather_than_raising(self) -> None:
        """``int(nan)`` raises ``ValueError`` and ``int(inf)`` ``OverflowError``."""
        for count in (NAN, INF, -INF):
            assert non_negative_whole_number_error(count, "n_steps", "step") is not None

    def test_the_message_names_the_parameter_and_the_value(self) -> None:
        error = non_negative_whole_number_error(-5, "n_steps", "step")
        assert error == "step: n_steps must be a non-negative whole number, got -5."

    def test_the_message_is_ascii(self) -> None:
        for count in UNUSABLE_COUNTS:
            error = non_negative_whole_number_error(count, "n_steps", "step")
            assert error is not None
            error.encode("ascii")


# --------------------------------------------------------------------------- #
# The shared domain: every real scalar gets a verdict, nothing raises         #
# --------------------------------------------------------------------------- #
@numbers.Real.register
class _RealWithoutAnInt:
    """A registered ``numbers.Real`` that cannot be converted to an ``int``.

    Registration makes ``isinstance(x, numbers.Real)`` true without inheriting
    the ABC's implementations, so this is what reaches the guard's ``TypeError``
    branch: ``int()`` has nothing to call. It is the one probe in this module
    with no real-world counterpart, and it is here so that branch is a covered
    decision rather than an unreachable line a later reader deletes.
    """


@numbers.Real.register
class _RealWithAnUnprintableRepr:
    """A registered ``numbers.Real`` whose ``repr`` raises.

    ``int()`` yields a negative count, so it is refused - and rendering that
    refusal has to survive its ``__repr__``. The real case is an outsized
    ``int``, whose ``repr`` raises ``ValueError``; a third-party scalar can
    raise anything, which is why the renderer's guarantee is unconditional
    rather than a list of the exceptions known today.
    """

    def __int__(self) -> int:
        return -1

    def __repr__(self) -> str:
        raise RuntimeError("this type cannot render itself")


#: Every probe in this module, plus the two synthetic scalars above.
ALL_PROBES: tuple[Any, ...] = (
    *UNUSABLE_COUNTS,
    *(count for count, _expected in USABLE_COUNTS),
    _RealWithoutAnInt(),
    _RealWithAnUnprintableRepr(),
)


class TestEveryRealGetsAVerdictAndNothingRaises:
    """The guard reports through its return value, never through an exception.

    All three ``step`` docstrings name the structured ``{status, content}``
    result as the only channel an out-of-domain count is reported on, and
    ``device_connect/sim_driver.py``'s ``@rpc()`` ``step`` forwards a remote
    caller's count unchanged - Python integers are arbitrary-precision, so an
    ``int`` of any width is one request away and has to be answered rather than
    raised on.
    """

    def test_a_count_wider_than_a_float_is_answered(self) -> None:
        """``float(10**400)`` raises ``OverflowError``, so the guard cannot use it.

        Accepted, because it *is* a non-negative whole number - magnitude is the
        per-call ceiling's question, pinned in
        ``TestNeighbouringStepSurfacesStayOutOfScope``.
        """
        assert non_negative_whole_number_error(BEYOND_FLOAT_RANGE, "n_steps", "step") is None
        error = non_negative_whole_number_error(-BEYOND_FLOAT_RANGE, "n_steps", "step")
        assert error is not None
        assert "n_steps" in error

    def test_a_count_whose_repr_raises_is_still_refusable(self) -> None:
        """Rendering happens on demand, so an accepted count is never rendered.

        ``repr`` of an ``int`` past ``sys.get_int_max_str_digits()`` raises
        ``ValueError``, and the message used to be built before the value was
        classified - so the guard failed on the accept path, doing work only the
        refuse path needs.
        """
        assert non_negative_whole_number_error(BEYOND_INT_STR_LIMIT, "n_steps", "step") is None
        error = non_negative_whole_number_error(-BEYOND_INT_STR_LIMIT, "n_steps", "step")
        assert error is not None
        assert f"<int of {BEYOND_INT_STR_LIMIT.bit_length()} bits>" in error
        assert "n_steps" in error
        error.encode("ascii")

    def test_a_real_that_cannot_be_converted_is_refused_rather_than_raising(self) -> None:
        assert non_negative_whole_number_error(_RealWithoutAnInt(), "n_steps", "step") is not None

    def test_a_real_that_cannot_render_itself_is_refused_rather_than_raising(self) -> None:
        error = non_negative_whole_number_error(_RealWithAnUnprintableRepr(), "n_steps", "step")
        assert error is not None
        assert "_RealWithAnUnprintableRepr" in error

    @pytest.mark.parametrize("count", ALL_PROBES, ids=_count_id)
    def test_the_verdict_is_a_string_or_none_for_every_probe(self, count: Any) -> None:
        """The contract as one assertion, over every value this module names."""
        verdict = non_negative_whole_number_error(count, "n_steps", "step")
        assert verdict is None or isinstance(verdict, str)

    def test_the_count_is_coerced_with_int_because_numpy_has_no_trunc(self) -> None:
        """``math.trunc`` is the conversion the ABC guarantees and is unusable here.

        Measured, so that a later "use the abstract method" refactor fails here
        rather than in the field: ``np.int64`` and ``np.uint8`` are load-bearing
        rows of ``USABLE_COUNTS`` - MuJoCo honors both today - and they implement
        ``__int__`` but not ``__trunc__``.
        """
        numpy_rows: tuple[tuple[Any, int], ...] = ((np.int64(3), 3), (np.uint8(2), 2))
        for count, expected in numpy_rows:
            with pytest.raises(TypeError):
                math.trunc(count)
            assert int(count) == expected
            assert non_negative_whole_number_error(count, "n_steps", "step") is None


# --------------------------------------------------------------------------- #
# Backend stand-ins (the guards precede every solver, stage and lock)         #
# --------------------------------------------------------------------------- #
def _mujoco_stub() -> tuple[Any, dict[str, int]]:
    """A MuJoCo stand-in counting ``mj_step`` calls, with no model compiled."""
    calls = {"n": 0}

    def mj_step(model: Any, data: Any) -> None:
        calls["n"] += 1
        data.time += 0.002

    stub = types.SimpleNamespace(
        _world=types.SimpleNamespace(
            _model=object(),
            _data=types.SimpleNamespace(time=0.0),
            sim_time=0.0,
            step_count=0,
            _backend_state={},
        ),
        _mj=types.SimpleNamespace(mj_step=mj_step, mj_forward=lambda model, data: None),
        _lock=threading.RLock(),
        _MAX_STEPS_PER_CALL=MuJoCoSimEngine._MAX_STEPS_PER_CALL,
        _STEPS_PER_BATCH=MuJoCoSimEngine._STEPS_PER_BATCH,
        _apply_kinematic_attachments=lambda: None,
        _publish_ros_telemetry=lambda: None,
    )
    return stub, calls


def _isaac_stub() -> tuple[Any, dict[str, int]]:
    calls = {"n": 0}
    stub: Any = types.SimpleNamespace(
        _lock=threading.RLock(),
        _world_created=True,
        _STEPS_PER_BATCH=IsaacSimulation._STEPS_PER_BATCH,
        _config=IsaacConfig(),
        _sim_time=0.0,
        _step_count=0,
        _world=types.SimpleNamespace(step=lambda render=False: calls.__setitem__("n", calls["n"] + 1)),
        # Main-thread-affinity state (#1896): the stub is "created" on the
        # test's own thread with no pump, so the genuinely-bound marshal
        # helper takes the inline path - the same seam the real engine
        # exercises on the headless smoke path.
        _main_tid=threading.get_ident(),
        _pump_running=False,
    )
    stub._on_main_thread = lambda: IsaacSimulation._on_main_thread(stub)
    stub._marshal_main_thread_affine = lambda name, fn: IsaacSimulation._marshal_main_thread_affine(stub, name, fn)
    return stub, calls


def _newton_stub() -> tuple[Any, SimWorld]:
    """A Newton stand-in on the real inherited ``_advance``, solver-free.

    ``_advance`` is bound genuinely rather than replaced: its ``max(1, n_steps)``
    floor is the thing ``step`` has to answer for, so a stand-in that stubbed it
    out would make that guard look absent rather than exercised. Since #1870 the
    floor is unreachable from either public surface - ``step`` answers ``0``
    before calling it and ``send_action`` refuses a non-positive count - so it is
    retained as a defensive no-op rather than as a contract, and this stub still
    binds the real method so that stays a measurement.
    """
    world = SimWorld()
    stub: Any = types.SimpleNamespace(
        _world=world,
        _model=types.SimpleNamespace(body_label=["ground"]),
        _lock=threading.RLock(),
        _solver=None,
        _sync_viewer=lambda: None,
        _STEPS_PER_BATCH=NewtonSimEngine._STEPS_PER_BATCH,
        substeps=1,
    )
    stub._advance = lambda n_steps: NewtonSimEngine._advance(stub, n_steps)
    return stub, world


# --------------------------------------------------------------------------- #
# MuJoCo                                                                      #
# --------------------------------------------------------------------------- #
class TestMuJoCoStep:
    @pytest.mark.parametrize("count", UNUSABLE_COUNTS, ids=_count_id)
    def test_an_unusable_count_is_refused(self, count: Any) -> None:
        stub, _calls = _mujoco_stub()
        result = MuJoCoSimEngine.step(stub, count)
        assert result["status"] == "error", (count, result)
        assert "n_steps" in _text(result)

    @pytest.mark.parametrize("count", UNUSABLE_COUNTS, ids=_count_id)
    def test_a_refused_count_advances_nothing(self, count: Any) -> None:
        stub, calls = _mujoco_stub()
        assert MuJoCoSimEngine.step(stub, count)["status"] == "error"
        assert calls["n"] == 0
        assert stub._world.step_count == 0
        assert stub._world.sim_time == 0.0

    @pytest.mark.parametrize(("count", "expected"), USABLE_COUNTS)
    def test_a_usable_count_advances_exactly_that_many(self, count: Any, expected: int) -> None:
        stub, calls = _mujoco_stub()
        assert MuJoCoSimEngine.step(stub, count)["status"] == "success", count
        assert calls["n"] == expected

    def test_an_infinite_count_is_refused_rather_than_raising_overflow(self) -> None:
        """``int(inf)`` raises ``OverflowError``, which the old guard let through."""
        stub, calls = _mujoco_stub()
        infinite: Any = INF
        result = MuJoCoSimEngine.step(stub, infinite)
        assert result["status"] == "error"
        assert calls["n"] == 0

    def test_a_fractional_count_is_refused_rather_than_truncated(self) -> None:
        """``int(2.7)`` stepped twice and called it success."""
        stub, calls = _mujoco_stub()
        fractional: Any = 2.7
        assert MuJoCoSimEngine.step(stub, fractional)["status"] == "error"
        assert calls["n"] == 0

    def test_the_documented_zero_no_op_still_holds(self) -> None:
        stub, calls = _mujoco_stub()
        result = MuJoCoSimEngine.step(stub, 0)
        assert result["status"] == "success"
        assert calls["n"] == 0
        assert "no-op" in _text(result)

    def test_the_per_call_ceiling_still_holds(self) -> None:
        """The ceiling is out of scope but must not be lost to the refactor."""
        stub, calls = _mujoco_stub()
        result = MuJoCoSimEngine.step(stub, MuJoCoSimEngine._MAX_STEPS_PER_CALL + 1)
        assert result["status"] == "error"
        assert "exceeds max" in _text(result)
        assert calls["n"] == 0


# --------------------------------------------------------------------------- #
# Newton                                                                      #
# --------------------------------------------------------------------------- #
class TestNewtonStep:
    @pytest.mark.parametrize("count", UNUSABLE_COUNTS, ids=_count_id)
    def test_an_unusable_count_is_refused(self, count: Any) -> None:
        stub, _world = _newton_stub()
        result = NewtonSimEngine.step(stub, count)
        assert result["status"] == "error", (count, result)
        assert "n_steps" in _text(result)

    @pytest.mark.parametrize("count", UNUSABLE_COUNTS, ids=_count_id)
    def test_a_refused_count_advances_nothing(self, count: Any) -> None:
        """Pre-fix, ``-5`` / ``nan`` / ``True`` each advanced one step."""
        stub, world = _newton_stub()
        assert NewtonSimEngine.step(stub, count)["status"] == "error"
        assert world.step_count == 0
        assert world.sim_time == 0.0

    @pytest.mark.parametrize(("count", "expected"), USABLE_COUNTS)
    def test_a_usable_count_advances_exactly_that_many(self, count: Any, expected: int) -> None:
        stub, world = _newton_stub()
        assert NewtonSimEngine.step(stub, count)["status"] == "success", count
        assert world.step_count == expected

    def test_zero_is_a_no_op_rather_than_one_step(self) -> None:
        """``_advance``'s ``max(1, n_steps)`` floor stepped once for a zero."""
        stub, world = _newton_stub()
        result = NewtonSimEngine.step(stub, 0)
        assert result["status"] == "success"
        assert world.step_count == 0
        assert world.sim_time == 0.0
        assert "no-op" in _text(result)

    def test_the_step_counter_stays_an_integer(self) -> None:
        """``step_count`` took the float ``2.7`` and the ``inf`` verbatim."""
        stub, world = _newton_stub()
        non_integral: tuple[Any, ...] = (2.7, INF, NAN)
        for count in non_integral:
            assert NewtonSimEngine.step(stub, count)["status"] == "error"
        assert type(world.step_count) is int
        assert world.step_count == 0
        assert NewtonSimEngine.step(stub, 2)["status"] == "success"
        assert type(world.step_count) is int

    def test_the_reported_count_is_the_count_advanced(self) -> None:
        """``Stepped -5 step(s).`` was reported for a world that advanced 1."""
        stub, world = _newton_stub()
        assert NewtonSimEngine.step(stub, 4)["status"] == "success"
        assert "4" in _text(NewtonSimEngine.step(_newton_stub()[0], 4))
        assert world.step_count == 4

    def test_a_no_world_engine_still_reports_that_first(self) -> None:
        """The world check precedes the domain, so the actionable error wins."""
        stub: Any = types.SimpleNamespace(_world=None, _model=None)
        result = NewtonSimEngine.step(stub, -5)
        assert result["status"] == "error"
        assert "create_world" in _text(result)


# --------------------------------------------------------------------------- #
# Isaac                                                                       #
# --------------------------------------------------------------------------- #
class TestIsaacStep:
    @pytest.mark.parametrize("count", UNUSABLE_COUNTS, ids=_count_id)
    def test_an_unusable_count_is_refused(self, count: Any) -> None:
        stub, _calls = _isaac_stub()
        result = IsaacSimulation.step(stub, count)
        assert result["status"] == "error", (count, result)
        assert "n_steps" in _text(result)

    @pytest.mark.parametrize("count", UNUSABLE_COUNTS, ids=_count_id)
    def test_a_refused_count_advances_nothing(self, count: Any) -> None:
        stub, calls = _isaac_stub()
        assert IsaacSimulation.step(stub, count)["status"] == "error"
        assert calls["n"] == 0
        assert stub._step_count == 0
        assert stub._sim_time == 0.0

    @pytest.mark.parametrize(("count", "expected"), USABLE_COUNTS)
    def test_a_usable_count_advances_exactly_that_many(self, count: Any, expected: int) -> None:
        stub, calls = _isaac_stub()
        assert IsaacSimulation.step(stub, count)["status"] == "success", count
        assert calls["n"] == expected
        assert stub._step_count == expected

    def test_a_negative_count_is_refused_rather_than_reported_as_success(self) -> None:
        """``range(-5)`` was empty, so it reported success having stepped nothing."""
        stub, calls = _isaac_stub()
        result = IsaacSimulation.step(stub, -5)
        assert result["status"] == "error"
        assert calls["n"] == 0

    def test_no_result_text_can_report_a_negative_rate(self) -> None:
        """``-5`` divided the wall time by a negative count: ``-11876485 steps/sec``."""
        for count in (-5, -1):
            stub, _calls = _isaac_stub()
            result = IsaacSimulation.step(stub, count)
            assert result["status"] == "error"
            assert "steps/sec" not in _text(result)

    def test_a_non_integral_count_is_refused_rather_than_raising(self) -> None:
        """``range(2.7)`` raised ``TypeError`` past the structured envelope."""
        non_integral: tuple[Any, ...] = (2.7, "3", NAN, INF, None, [3])
        for count in non_integral:
            stub, _calls = _isaac_stub()
            result = IsaacSimulation.step(stub, count)
            assert result["status"] == "error", count

    def test_the_guard_precedes_the_world_tick_on_an_uncreated_world(self) -> None:
        """A refused count needs no world, so the domain answers without one."""
        stub, calls = _isaac_stub()
        stub._world_created = False
        assert IsaacSimulation.step(stub, -5)["status"] == "error"
        assert calls["n"] == 0


# --------------------------------------------------------------------------- #
# Cross-backend parity                                                        #
# --------------------------------------------------------------------------- #
class TestEveryBackendGivesTheSameVerdict:
    """A count one backend refuses is refused by all of them, in the same words."""

    @staticmethod
    def _all_three(count: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        return (
            MuJoCoSimEngine.step(_mujoco_stub()[0], count),
            NewtonSimEngine.step(_newton_stub()[0], count),
            IsaacSimulation.step(_isaac_stub()[0], count),
        )

    @pytest.mark.parametrize("count", UNUSABLE_COUNTS, ids=_count_id)
    def test_an_unusable_count_is_refused_everywhere(self, count: Any) -> None:
        mj, nt, ic = self._all_three(count)
        assert mj["status"] == nt["status"] == ic["status"] == "error", (count, mj, nt, ic)

    @pytest.mark.parametrize("count", UNUSABLE_COUNTS, ids=_count_id)
    def test_the_shared_refusal_has_one_wording(self, count: Any) -> None:
        """Two spellings of one verdict is how backend domains start to drift."""
        mj, nt, ic = self._all_three(count)
        assert {_text(mj), _text(nt), _text(ic)} == {_text(mj)}, (count, _text(mj), _text(nt), _text(ic))

    @pytest.mark.parametrize(("count", "expected"), USABLE_COUNTS)
    def test_a_usable_count_is_accepted_everywhere(self, count: Any, expected: int) -> None:
        """The parity is two-way: no backend refuses a count another honors."""
        mj, nt, ic = self._all_three(count)
        assert mj["status"] == nt["status"] == ic["status"] == "success", (count, mj, nt, ic)

    @pytest.mark.parametrize(("count", "expected"), USABLE_COUNTS)
    def test_the_same_count_advances_the_same_number_of_steps(self, count: Any, expected: int) -> None:
        """The verdict agreeing is not enough: the effect has to agree too."""
        mj_stub, mj_calls = _mujoco_stub()
        nt_stub, nt_world = _newton_stub()
        ic_stub, ic_calls = _isaac_stub()
        MuJoCoSimEngine.step(mj_stub, count)
        NewtonSimEngine.step(nt_stub, count)
        IsaacSimulation.step(ic_stub, count)
        assert mj_calls["n"] == nt_world.step_count == ic_calls["n"] == expected, count

    def test_zero_is_a_no_op_on_every_backend(self) -> None:
        """The row that used to disagree: MuJoCo no-op, Newton one step, Isaac none."""
        mj_stub, mj_calls = _mujoco_stub()
        nt_stub, nt_world = _newton_stub()
        ic_stub, ic_calls = _isaac_stub()
        results = (
            MuJoCoSimEngine.step(mj_stub, 0),
            NewtonSimEngine.step(nt_stub, 0),
            IsaacSimulation.step(ic_stub, 0),
        )
        assert all(result["status"] == "success" for result in results), results
        assert mj_calls["n"] == nt_world.step_count == ic_calls["n"] == 0


class TestTheParityHoldsOnACompiledModel:
    """The stand-ins are only honest if a real engine agrees with them."""

    @pytest.fixture
    def mj_sim(self) -> Any:
        pytest.importorskip("mujoco")
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation(tool_name="test_step_count_domain_parity_sim", mesh=False)
        assert sim.create_world()["status"] == "success"
        yield sim
        sim.cleanup()

    @pytest.mark.parametrize("count", UNUSABLE_COUNTS, ids=_count_id)
    def test_an_unusable_count_is_refused_on_a_real_world(self, mj_sim: Any, count: Any) -> None:
        result = mj_sim.step(count)
        assert result["status"] == "error", (count, result)
        assert "n_steps" in _text(result)

    @pytest.mark.parametrize(("count", "expected"), USABLE_COUNTS)
    def test_a_usable_count_advances_a_real_world_exactly_that_far(
        self, mj_sim: Any, count: Any, expected: int
    ) -> None:
        before = mj_sim._world.step_count
        assert mj_sim.step(count)["status"] == "success", count
        assert mj_sim._world.step_count - before == expected

    @pytest.mark.parametrize("count", UNUSABLE_COUNTS, ids=_count_id)
    def test_a_refused_count_leaves_a_real_clock_untouched(self, mj_sim: Any, count: Any) -> None:
        """The ``inf`` row left Newton's clock at ``inf`` for the world's lifetime."""
        assert mj_sim.step(2)["status"] == "success"
        steps, elapsed = mj_sim._world.step_count, mj_sim._world.sim_time
        assert mj_sim.step(count)["status"] == "error"
        assert mj_sim._world.step_count == steps
        assert mj_sim._world.sim_time == elapsed


# --------------------------------------------------------------------------- #
# Structural: no step-count surface drifts off the shared domain              #
# --------------------------------------------------------------------------- #
#: Every public engine method taking an ``n_steps``, and the shared domain it
#: routes the count through. The three MuJoCo rollout entries are a DIFFERENT
#: domain and deliberately so: a rollout horizon has a floor of 1 (a zero-step
#: rollout collects nothing) and is resolved through ``_resolve_horizon`` /
#: ``_validate_action_horizon`` on ``positive_count_error``. They are listed so
#: the count of ``n_steps`` surfaces cannot grow unnoticed, not because they
#: share this change's floor.
_KNOWN_STEP_COUNT_SURFACES: dict[tuple[str, str], tuple[str, ...]] = {
    ("mujoco", "step"): ("non_negative_whole_number_error",),
    ("mujoco", "start_policy"): ("_resolve_horizon",),
    ("mujoco", "run_policy"): (),
    ("mujoco", "run_multi_policy"): ("_resolve_horizon",),
    ("newton", "step"): ("non_negative_whole_number_error",),
    ("isaac", "step"): ("non_negative_whole_number_error",),
}

#: The surfaces this change owns: every backend's public ``step``.
_STEP_METHODS = {key for key in _KNOWN_STEP_COUNT_SURFACES if key[1] == "step"}

#: Any of these, called on the count, means the surface is on a shared domain.
_SHARED_STEP_VALIDATORS = ("non_negative_whole_number_error", "_resolve_horizon")


def _scan_step_count_surfaces(
    root: pathlib.Path,
) -> tuple[dict[tuple[str, str], tuple[str, ...]], list[str]]:
    """Find public engine-class methods taking ``n_steps``, and which skip a domain.

    Scoped to public methods deliberately: ``_advance`` and ``_warmup_camera``
    also take a step count, but they receive an already-validated one and are
    not caller-facing.

    Args:
        root: The ``strands_robots/simulation`` package directory.

    Returns:
        ``(found, adrift)`` - every ``(backend, method)`` pair mapped to the
        shared validators it calls, and the ones that call none.
    """
    found: dict[tuple[str, str], tuple[str, ...]] = {}
    adrift: list[str] = []
    for backend in ("mujoco", "newton", "isaac"):
        for path in sorted((root / backend).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                for fn in [n for n in ast.iter_child_nodes(cls) if isinstance(n, ast.FunctionDef)]:
                    if fn.name.startswith("_"):
                        continue
                    if "n_steps" not in [a.arg for a in fn.args.args + fn.args.kwonlyargs]:
                        continue
                    called = {
                        node.func.id
                        for node in ast.walk(fn)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    } | {
                        node.func.attr
                        for node in ast.walk(fn)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    }
                    validators = tuple(sorted(name for name in _SHARED_STEP_VALIDATORS if name in called))
                    found[(backend, fn.name)] = validators
                    if not validators and fn.name == "step":
                        adrift.append(f"{backend}/{path.name}:{fn.lineno} {cls.name}.{fn.name}")
    return found, adrift


class TestNoStepCountSurfaceDrifts:
    """A backend ``step`` must route its count through the shared domain."""

    def test_every_backend_step_validates(self) -> None:
        root = pathlib.Path(inspect.getfile(NewtonSimEngine)).parent.parent
        found, adrift = _scan_step_count_surfaces(root)
        assert adrift == [], "these advance a step count without a shared domain: " + ", ".join(adrift)
        assert found == _KNOWN_STEP_COUNT_SURFACES, f"the set of n_steps surfaces changed: {found}"

    def test_all_three_backends_are_covered(self) -> None:
        """An empty scan would satisfy the assertion above just as well."""
        assert {backend for backend, _method in _STEP_METHODS} == {"mujoco", "newton", "isaac"}

    def test_every_step_is_on_this_changes_domain(self) -> None:
        root = pathlib.Path(inspect.getfile(NewtonSimEngine)).parent.parent
        found, _adrift = _scan_step_count_surfaces(root)
        for key in _STEP_METHODS:
            assert found[key] == ("non_negative_whole_number_error",), key

    def test_the_scanner_reports_a_planted_omission(self, tmp_path: pathlib.Path) -> None:
        """Without this, an empty result could mean a scanner matching nothing."""
        backend = tmp_path / "newton"
        backend.mkdir()
        (backend / "simulation.py").write_text(
            textwrap.dedent(
                """
                class Engine:
                    def step(self, n_steps=1):
                        return {"status": "success"}
                """
            ),
            encoding="utf-8",
        )
        found, adrift = _scan_step_count_surfaces(tmp_path)
        assert found == {("newton", "step"): ()}
        assert len(adrift) == 1


# --------------------------------------------------------------------------- #
# Boundary: the neighbouring surfaces this change does not settle             #
# --------------------------------------------------------------------------- #
class TestNeighbouringStepSurfacesStayOutOfScope:
    """Pins of behaviour left unchanged, so the boundary is stated not omitted.

    Replace these when the surfaces they describe are settled, rather than
    deleting them: the scope statement stays useful and simply narrows.
    """

    def test_the_advance_floor_survives_but_no_public_surface_can_reach_it(self) -> None:
        """Replaces the pin that ``send_action(n_substeps=)`` was unvalidated.

        That surface was settled as #1870 on ``positive_whole_number_error``, so
        the boundary this class draws has moved rather than gone: the
        ``max(1, n_steps)`` floor is still in ``_advance`` and still turns a
        ``0`` into one control step when called directly, which is why ``step``
        answers its own zero above instead of moving it. What changed is that
        neither public caller can hand it a non-positive count any more -
        ``step`` returns before calling it and ``send_action`` refuses - so the
        floor is a defensive no-op rather than a contract either surface reads.

        Both halves are asserted, because only the pair is the statement: drop
        the first and a later reader deletes a floor believing nothing depends on
        it; drop the second and the floor still reads as reachable behaviour.
        """
        stub, world = _newton_stub()
        stub._advance(0)
        assert world.step_count == 1, "the floor itself is unchanged"

        assert non_negative_whole_number_error(0, "n_steps", "step") is None
        assert positive_whole_number_error(0, "n_substeps", "send_action") is not None

    def test_the_per_call_ceiling_is_still_mujoco_only(self) -> None:
        """The ceiling stays MuJoCo's; only the lock hold became shared.

        Replaces the pin that Isaac and Newton had *neither* a ceiling nor a
        batched lock release. The batching half is settled - all three now
        release the lock every ``SimEngine._STEPS_PER_BATCH`` steps and re-check
        the world on each boundary, pinned by
        ``test_step_lock_hold_across_backends.py`` - so the boundary this class
        draws has narrowed rather than gone.

        Still out of scope is the *ceiling*, which is a different quantity: the
        batching bounds how long the lock is held, the ceiling bounds how much
        work is accepted at all. A count above ``_MAX_STEPS_PER_CALL`` is
        refused by MuJoCo and accepted by the other two, because one number
        cannot express one resource policy across backends whose per-step cost
        differs by an order of magnitude - and because a Newton step is a
        control step of ``substeps`` solver steps, so the same number is not
        even the same quantity. Asserting the acceptance rather than merely
        omitting it is the point: #1871 stays open on exactly this row.
        """
        over = MuJoCoSimEngine._MAX_STEPS_PER_CALL + 1
        assert MuJoCoSimEngine.step(_mujoco_stub()[0], over)["status"] == "error"

        isaac_stub, isaac_calls = _isaac_stub()
        assert IsaacSimulation.step(isaac_stub, over)["status"] == "success"
        assert isaac_calls["n"] == over

    def test_an_outsized_positive_count_is_left_to_the_per_call_ceiling(self) -> None:
        """Magnitude is the ceiling's question, not this domain's.

        ``10**400`` is a non-negative whole number, so the domain accepts it and
        says so. MuJoCo then refuses it with its own ``_MAX_STEPS_PER_CALL``
        error - exactly as it did before this change, where a true ``int``
        skipped its ``int()`` coercion and reached the ceiling. Newton and Isaac
        have no ceiling, so they would attempt it - unbounded *work*, which is
        #1871 and is not pinned here because pinning it means running it. The
        unbounded *lock hold* it used to imply is gone: all three now batch on
        ``SimEngine._STEPS_PER_BATCH``, so an outsized count is a long run rather
        than a wedged engine. Refusing the value in the guard instead would have put a
        silent boundary at the float range - accepting ``10**300``, which no
        backend can advance either - which is the boundary-by-accident this
        change exists to remove.
        """
        assert non_negative_whole_number_error(BEYOND_FLOAT_RANGE, "n_steps", "step") is None
        result = MuJoCoSimEngine.step(_mujoco_stub()[0], BEYOND_FLOAT_RANGE)
        assert result["status"] == "error"
        assert "exceeds max" in _text(result)

    def test_a_boolean_is_refused_by_the_count_domain_not_the_shared_predicate(self) -> None:
        """``is_boolean`` covers the float writers; this domain has its own check.

        Recorded because the two look interchangeable: ``numpy.bool_`` is not
        registered as ``numbers.Real``, so it is refused here by the scalar gate
        rather than by an explicit boolean test, and only python ``bool`` needs
        the explicit one (it is an ``int`` subclass).
        """
        assert non_negative_whole_number_error(True, "n_steps", "step") is not None
        assert non_negative_whole_number_error(np.bool_(True), "n_steps", "step") is not None
