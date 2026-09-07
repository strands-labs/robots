"""What the ``observer`` guard on :meth:`PolicyRunner.run` may absorb, and what it may not.

The guard exists for one class. :class:`CooperativeStop` is a ``BaseException``
*precisely* so a hook's broad ``except Exception`` cannot swallow a graceful
stop, so a guard written as ``except Exception`` would let an observer fire one
and cancel a rollout it is only supposed to watch. That is the defect the guard
prevents, and the sibling module pins it.

Written as ``except BaseException`` the guard also covered three signals that
are not an observer's to absorb, and two of them reached a caller as a *counted
telemetry failure* on a rollout reported ``success``:

======================  =====================  ==========================
observer raises         ``BaseException``      ``(CooperativeStop, Exception)``
======================  =====================  ==========================
``CooperativeStop``     contained              contained
``ValueError``          contained, counted     contained, counted
``GeneratorExit``       contained, counted     **propagates**
``CancelledError``      contained, counted     **propagates**
``KeyboardInterrupt``   propagates             propagates
``SystemExit``          propagates             propagates
======================  =====================  ==========================

The last two propagated only because the clause above the guard re-raises them
by name. ``GeneratorExit`` and :class:`asyncio.CancelledError` had no such
clause, so a generator being closed underneath a visualiser, or a task cancelled
while one was drawing, was reported as "the observer is down, the rollout is
fine" and the rollout ran to its budget. None of those four is an ``Exception``
subclass, which is what makes the narrower tuple both sufficient for the one
class the guard is for and silent about the three it is not.

That tuple is also what ``py/catch-base-exception`` reads, and the rule's
disposition in ``AGENTS.md`` does not reach this site: it turns on "which thread
you marshal onto", and this guard marshals onto none - it is a synchronous call
to a caller's callback on the rollout's own thread. The census in
``tests/test_codeql_query_filters.py`` says so from the other side, refusing a
second non-reraising handler that is not the cross-thread marshal box.

The second cell group is about *where* the per-step event is built rather than
what it contains. The emission sits in a ``finally`` on purpose - that is what
reports the action a cancelling hook aborted on - and a ``lambda`` written there
is read by ``py/exit-from-finally`` as a ``return`` leaving the block. The block
holds no ``return`` / ``break`` / ``continue`` of its own, measured below, so
naming the builder is what answers the report; the event it produces is
unchanged, which the field-parity cell holds.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from typing import Any

import pytest

from strands_robots.simulation import policy_runner as policy_runner_module
from strands_robots.simulation.observers import RunPolicyStep
from strands_robots.simulation.policy_runner import CooperativeStop
from tests.simulation.test_run_policy_observer import (
    _json,
    _run,
    _runner_and_policy,
)

# Bound here rather than reached through the module under test so a rename in
# production cannot quietly turn a cell below into an assertion about nothing.
_GUARD_OWNER = "_emit_event"
_STEP_EMITTER = "_emit_step"


def _module_tree() -> ast.Module:
    """Parse the production module from disk.

    Read as a file rather than through :func:`inspect.getsource` on the nested
    function: the guard and the step emitter are closures inside ``run``, and a
    dedented fragment loses the enclosing ``try`` the second group is about.
    """
    path = Path(inspect.getfile(policy_runner_module))
    return ast.parse(path.read_text(encoding="utf-8"))


def _handlers_in(owner: str) -> list[ast.ExceptHandler]:
    """Every ``except`` clause owned by the named function."""
    tree = _module_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == owner:
            return [h for h in ast.walk(node) if isinstance(h, ast.ExceptHandler)]
    raise AssertionError(f"no function named {owner!r} in {inspect.getfile(policy_runner_module)}")


def _finally_blocks_calling(name: str) -> list[ast.Try]:
    """Every ``try`` whose ``finally`` calls the named function."""
    out = []
    for node in ast.walk(_module_tree()):
        if not (isinstance(node, ast.Try) and node.finalbody):
            continue
        for statement in node.finalbody:
            for call in ast.walk(statement):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == name:
                    out.append(node)
                    break
            else:
                continue
            break
    return out


def _observer_raising(exc: type[BaseException], *, only_step: bool = True):
    """An observer that raises ``exc`` once, on the first Step event by default."""
    state = {"fired": False}

    def rogue(event: Any) -> None:
        if only_step and not isinstance(event, RunPolicyStep):
            return
        if not state["fired"]:
            state["fired"] = True
            raise exc("observer misbehaved")

    return rogue


class TestPremises:
    """The class facts that make the narrower tuple the right shape. True either way."""

    def test_cooperative_stop_is_a_base_exception_that_is_not_an_exception(self):
        """Why the guard cannot be ``except Exception``, and why the tuple is not redundant.

        Both halves matter. ``CooperativeStop`` being outside ``Exception`` is
        what a bare ``except Exception`` would miss, and it is also what makes
        ``(CooperativeStop, Exception)`` two disjoint members rather than a
        narrow class beside its own superclass - the shape
        ``tests/test_except_tuples_state_their_real_scope.py`` refuses.
        """
        assert issubclass(CooperativeStop, BaseException)
        assert not issubclass(CooperativeStop, Exception), (
            "CooperativeStop must stay outside the Exception tree: that is the whole reason a "
            "hook's broad `except Exception` cannot swallow a graceful stop, and the reason the "
            "observer guard has to name it"
        )

    @pytest.mark.parametrize(
        "exc",
        [GeneratorExit, asyncio.CancelledError, KeyboardInterrupt, SystemExit],
        ids=["GeneratorExit", "CancelledError", "KeyboardInterrupt", "SystemExit"],
    )
    def test_the_propagating_classes_are_not_exception_subclasses(self, exc):
        """Why ``except BaseException`` reached them and the narrower tuple does not."""
        assert not issubclass(exc, Exception)

    def test_no_finally_in_the_module_carries_a_statement_level_exit(self):
        """``py/exit-from-finally`` reported a closure's position, not any control flow.

        True on both shapes, and that is the point: it is the reason the answer
        is to name the builder rather than to restructure the block. Had any
        ``finally`` here really carried a ``return``, moving a ``lambda`` out of
        one would have fixed nothing and the report would have been about a real
        swallowed exception instead.
        """
        blocks = [node for node in ast.walk(_module_tree()) if isinstance(node, ast.Try) and node.finalbody]
        assert blocks, "no `finally` blocks found; the scan stopped reading the module"
        exits = [
            f"{type(node).__name__}@{node.lineno}"
            for block in blocks
            for statement in block.finalbody
            for node in ast.walk(statement)
            if isinstance(node, (ast.Return, ast.Break, ast.Continue))
        ]
        assert exits == [], (
            f"a `finally` now carries statement-level exits {exits}. A `return` there really "
            "would swallow the in-flight exception the step-emitting block exists to preserve, "
            "which is a different defect from the one this file is about"
        )


class TestTheGuardAbsorbsOnlyWhatItIsFor:
    """The behavioural split: two classes stop being absorbed, four are unchanged."""

    @pytest.mark.parametrize(
        "exc",
        [GeneratorExit, asyncio.CancelledError],
        ids=["GeneratorExit", "CancelledError"],
    )
    def test_a_signal_that_is_not_the_observers_propagates(self, exc):
        """The regression: neither is an observer's to absorb.

        Absorbed, each one reached the caller as a counted telemetry failure on
        a rollout reported ``success`` and run to its full budget - a generator
        closed underneath a visualiser, or a task cancelled while one was
        drawing, reported as "the observer is down".
        """
        runner, policy, _ = _runner_and_policy()
        with pytest.raises(exc):
            _run(runner, policy, observer=_observer_raising(exc))

    def test_a_cooperative_stop_from_the_observer_is_still_contained(self):
        """The guard's whole purpose, and the property the narrowing must not cost.

        Holds either way. It is here because it is the assertion a reader
        reaches for to argue the guard cannot be narrowed at all, and the tuple
        is exactly what keeps it true.
        """
        runner, policy, _ = _runner_and_policy()
        result = _run(runner, policy, observer=_observer_raising(CooperativeStop))
        assert result["status"] == "success"
        payload = _json(result)
        assert payload["stopped_reason"] == "budget", "an observer must not be able to cancel"
        assert payload["observer_failures"] == 1

    def test_an_ordinary_exception_from_the_observer_is_still_contained_and_counted(self):
        """A visualiser that cannot draw is still not a rollout failure."""
        runner, policy, _ = _runner_and_policy()
        result = _run(runner, policy, observer=_observer_raising(ValueError))
        assert result["status"] == "success"
        assert _json(result)["observer_failures"] == 1

    @pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit], ids=["KeyboardInterrupt", "SystemExit"])
    def test_the_operators_signals_still_propagate(self, exc):
        """Unchanged: these were re-raised by name before and are outside the tuple now."""
        runner, policy, _ = _runner_and_policy()
        with pytest.raises(exc):
            _run(runner, policy, observer=_observer_raising(exc))


class TestTheShapeThatKeepsBothAlertsClosed:
    """Structural pins, so neither report returns without a cell naming it."""

    def test_the_guard_names_no_base_exception(self):
        """``py/catch-base-exception`` and the AGENTS.md census read the same clause."""
        named: list[str] = []
        for handler in _handlers_in(_GUARD_OWNER):
            if handler.type is None:
                named.append("<bare except>")
                continue
            members = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
            named.extend(ast.unparse(member) for member in members)
        assert named, f"{_GUARD_OWNER}() carries no `except` clause; the guard is the whole point"
        assert "BaseException" not in named, (
            f"{_GUARD_OWNER}() names BaseException again (clauses: {named}). That is a "
            "py/catch-base-exception alert whose review thread gates the merge, and a second "
            "non-reraising handler in the AGENTS.md census, whose recorded disposition turns on "
            "marshalling across a thread - which this synchronous callback does not do. The "
            "smallest superset that keeps a CooperativeStop from escaping an observer is "
            "(CooperativeStop, Exception)"
        )

    def test_the_guard_still_names_cooperative_stop(self):
        """The other half of the clause, and the non-vacuity partner of the cell above.

        ``except Exception`` satisfies the cell above and reintroduces the
        defect, so "names no BaseException" is only half a contract. Before the
        narrowing the guard named ``CooperativeStop`` in its prose and not in
        its clause, which is exactly the gap this closes.
        """
        named = [
            ast.unparse(member)
            for handler in _handlers_in(_GUARD_OWNER)
            if handler.type is not None
            for member in (handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type])
        ]
        assert "CooperativeStop" in named, (
            f"{_GUARD_OWNER}() no longer names CooperativeStop (clauses: {named}). Dropping it "
            "satisfies the cell above and reintroduces the defect the guard exists for: "
            "CooperativeStop is outside the Exception tree, so an observer could raise one and "
            "cancel a rollout it is only supposed to watch"
        )

    def test_the_step_emit_is_not_a_closure_inside_the_finally(self):
        """``py/exit-from-finally`` reads a lambda body's implicit return as a block exit."""
        blocks = _finally_blocks_calling(_STEP_EMITTER)
        assert len(blocks) == 1, (
            f"expected exactly one `finally` calling {_STEP_EMITTER}(), found {len(blocks)}. The "
            "per-step event is emitted from a `finally` on purpose - that is what reports the "
            "action a cancelling hook aborted on - so the block has to still be there, and the "
            "builder has to still be named rather than written inline as a closure"
        )
        offenders = []
        for node in ast.walk(_module_tree()):
            if not (isinstance(node, ast.Try) and node.finalbody):
                continue
            offenders += [
                lam.lineno for statement in node.finalbody for lam in ast.walk(statement) if isinstance(lam, ast.Lambda)
            ]
        assert offenders == [], (
            f"a lambda sits inside a `finally` at line(s) {offenders}. py/exit-from-finally reads "
            "its implicit return as a `return` leaving the block and opens a review thread that "
            "gates the merge, even where the block carries no control flow of its own. Name the "
            f"builder instead, as {_STEP_EMITTER}() does"
        )

    def test_the_step_event_still_carries_the_same_fields(self):
        """Over-reach guard: naming the builder moved the closure, not the event.

        The three values the builder reads from the enclosing scope - the
        applied-action index, the elapsed duration and the sim clock - are the
        ones that would change if the hoist had turned a lazily-sampled field
        into an eagerly-passed argument.
        """
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        result = _run(runner, policy, observer=events.append)
        assert result["status"] == "success"

        steps = [event for event in events if isinstance(event, RunPolicyStep)]
        assert steps, "a 3-step rollout must emit Step events for the parity check to mean anything"
        for index, step in enumerate(steps):
            assert step.applied_action_index == index, (
                "applied_action_index is read inside the builder, so it must still count applied "
                f"actions: step {index} reported {step.applied_action_index}"
            )
            assert step.elapsed_s > 0.0, "elapsed_s is sampled when the event is built"
            assert step.action_resolution in {"full", "partial", "none", "unknown"}
            assert step.legacy_hook_outcome is None or isinstance(step.legacy_hook_outcome, str)
