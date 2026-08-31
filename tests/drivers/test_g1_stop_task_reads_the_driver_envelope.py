"""``g1_stop_task`` returns exactly what ``G1Driver.stop_task`` gives it.

``g1_stop_task`` is the sibling write-side verb to ``g1_get_task_status``:
where that one reads the loop's snapshot without changing anything, this
one signals the driver's control loop to exit, joins its thread, and lets
the loop publish :func:`_build_zero_torque_lowcmd` on the way out.  The
driver's method (:meth:`G1Driver.stop_task`) returns one of three shapes
- a "no task is running" text sentinel, a joined-within-budget snapshot,
or a join-outlasted-budget snapshot carrying ``status="error"`` - and the
tests here fix the flat dict the verb reshapes each of them into.

The snapshot field names are read here off the driver's own writer
(:meth:`_ControlLoop.snapshot`) rather than being restated: a rename on
the driver side moves both the write path and this verb together.  What
the tests do restate is the SDK-load-hygiene contract every file under
:mod:`strands_robots.tools.g1` carries: importing the module must not
pull any ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_stop_task import g1_stop_task


class _StubG1Driver:
    """A driver double whose ``stop_task`` returns a fixed envelope.

    ``g1_stop_task`` calls ``driver.stop_task()`` and reads the
    ``content[0]`` payload.  This double sits under the same interface
    without pulling the real driver's imports (the real class reaches
    CycloneDDS at construction time in some paths), so a test can hand
    a wired-shape envelope to the verb without a bus.  ``calls`` counts
    the invocations so a test can pin "the verb writes the driver
    exactly once" without asking the driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls = 0

    def stop_task(self) -> dict[str, Any]:
        self.calls += 1
        return self._envelope


def _call(driver: _StubG1Driver) -> dict[str, Any]:
    """Call the ``@tool``-decorated verb and return its dict.

    The ``strands`` ``@tool`` wrapper defers to the wrapped function
    directly when called in-process, but a caller cannot rely on that:
    the wrapper's contract is that it returns the wrapped function's
    return value verbatim.  This helper is where a shape drift would
    surface once, rather than at every call site.
    """
    return g1_stop_task(driver=driver)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import time
    would break every headless CI runner and Thor before an office
    bring-up.  The driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only
    path that loads the SDK); this cell holds the stop-task verb to it
    too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_stop_task")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_stop_task imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_with_no_task_running_reports_absent() -> None:
    """The driver's "no task is running" envelope becomes ``present=False``.

    :meth:`G1Driver.stop_task` returns this shape when ``self._loop`` is
    ``None`` or the loop has already finished - the idempotent branch a
    supervisor polling stop hits when it races the loop's own
    self-terminating exit.  The verb reports that decidably:
    ``present=False``, ``stopped=None`` (there was nothing to stop),
    ``running=False`` (nothing is running), and ``reason`` carries the
    driver's own text verbatim so a caller reading the flat dict sees
    the same string the driver wrote.
    """
    driver = _StubG1Driver(
        envelope={
            "status": "success",
            "content": [{"text": "stop_task: no task is running"}],
        }
    )
    result = _call(driver)

    assert result["status"] == "success"
    assert result["present"] is False
    assert result["stopped"] is None
    assert result["running"] is False
    assert result["reason"] == "stop_task: no task is running"
    # Every snapshot field is ``None`` on the "no task" shape; the flat
    # dict does not fabricate a zero for a loop that never ran.
    for field in (
        "steps",
        "refusals",
        "elapsed_s",
        "duration_budget_s",
        "n_steps_budget",
        "exit_reason",
        "exit_detail",
        "hz",
        "fsm_refresh_hz",
        "fsm_reads",
    ):
        assert result[field] is None, f"expected {field} None on no-task shape, got {result[field]!r}"


def test_a_joined_loop_reports_stopped_true_running_false() -> None:
    """A loop that joined within budget reshapes to ``stopped=True``.

    :meth:`G1Driver.stop_task` calls ``loop.stop("stop_task")``, joins
    the thread, reads the loop's ``snapshot`` and stamps ``stopped``
    on it before returning.  ``running`` is ``False`` (the thread has
    joined) and ``exit_reason`` names ``stop_task`` (the sixth
    self-terminating reason, added to the five the loop's own
    finally-block writes).  The verb surfaces every snapshot field flat.
    """
    snapshot = {
        "running": False,
        "steps": 137,
        "refusals": [],
        "elapsed_s": 0.274,
        "duration_budget_s": 2.0,
        "n_steps_budget": 1000,
        "exit_reason": "stop_task",
        "exit_detail": "stop requested by stop_task",
        "hz": 500,
        "fsm_refresh_hz": 20,
        "fsm_reads": 5,
        "stopped": True,
    }
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": snapshot}]})
    result = _call(driver)

    assert result["status"] == "success"
    assert result["present"] is True
    assert result["stopped"] is True
    assert result["running"] is False
    # Every snapshot field round-trips verbatim - the verb reshapes, it
    # does not recompute.
    assert result["steps"] == 137
    assert result["refusals"] == []
    assert result["elapsed_s"] == 0.274
    assert result["duration_budget_s"] == 2.0
    assert result["n_steps_budget"] == 1000
    assert result["exit_reason"] == "stop_task"
    assert result["exit_detail"] == "stop requested by stop_task"
    assert result["hz"] == 500
    assert result["fsm_refresh_hz"] == 20
    assert result["fsm_reads"] == 5
    # ``reason`` is a field of the "no task" text branch; the snapshot
    # shape does not carry it, so the flat dict reports ``None`` rather
    # than raising.
    assert result["reason"] is None


def test_a_join_that_outlasts_the_budget_surfaces_status_error() -> None:
    """A blocking policy surfaces as ``status="error"`` with ``stopped=False``.

    :meth:`G1Driver.stop_task`'s docstring names this the ordinary case
    for a remote-inference policy: the loop was signalled to stop but the
    ``_call_policy`` call has not returned, so the join times out.  The
    driver stamps ``stopped=False`` on the snapshot, ``running`` may
    still be ``True`` (the thread is still writing frames), and the
    outer envelope returns ``status="error"``.  A caller reading only
    ``status`` on the flat dict must see the ``"error"`` verbatim - the
    verb cannot flatten this to ``"success"`` because the loop is still
    holding the wire.
    """
    snapshot = {
        "running": True,
        "steps": 42,
        "refusals": [],
        "elapsed_s": 0.084,
        "duration_budget_s": None,
        "n_steps_budget": None,
        "exit_reason": None,
        "exit_detail": None,
        "hz": 500,
        "fsm_refresh_hz": 20,
        "fsm_reads": 2,
        "stopped": False,
        "reason": (
            "stop_task: control loop did not join within timeout; "
            "policy is likely blocking - the loop will publish the "
            "zero-torque frame when it exits"
        ),
    }
    driver = _StubG1Driver(envelope={"status": "error", "content": [{"json": snapshot}]})
    result = _call(driver)

    # The envelope's ``status`` is authoritative and surfaces verbatim -
    # a caller reading only ``status`` cannot count the task as stopped
    # while the payload says the loop is still running.
    assert result["status"] == "error"
    assert result["present"] is True
    assert result["stopped"] is False
    assert result["running"] is True
    assert result["steps"] == 42
    # ``reason`` here quotes the driver's own text verbatim so a caller
    # that logs it can point at the timeout without paraphrasing.
    assert result["reason"] is not None
    assert "did not join within timeout" in result["reason"]


def test_the_verb_writes_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.stop_task``.

    ``stop_task`` is a write (it signals the loop to exit and joins the
    thread) so a double-call from a wrapper that polled twice would
    trigger the driver's idempotent no-task branch on the second read,
    silently masking a caller who saw an inconsistent "the loop just
    stopped between our two reads" gap.  This cell pins the verb to a
    single driver write per invocation.
    """
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"text": "stop_task: no task is running"}]})
    _call(driver)
    assert driver.calls == 1


def test_the_envelope_status_is_reported_verbatim() -> None:
    """A non-success envelope from the driver surfaces on the returned dict.

    :meth:`G1Driver.stop_task` returns ``status="error"`` when the join
    outlasts the budget (the loop is still writing frames), and the verb
    must surface that verbatim rather than flattening it to a success.
    This cell fixes the flow from the envelope's ``status`` to the
    returned dict's ``status``, so a caller reading only ``status``
    cannot read "success" while the payload's ``stopped`` field says
    the loop did not stop.
    """
    snapshot = {
        "running": True,
        "steps": 5,
        "refusals": [],
        "elapsed_s": 0.01,
        "duration_budget_s": None,
        "n_steps_budget": None,
        "exit_reason": None,
        "exit_detail": None,
        "hz": 500,
        "fsm_refresh_hz": 20,
        "fsm_reads": 0,
        "stopped": False,
        "reason": "stop_task: control loop did not join within timeout",
    }
    driver = _StubG1Driver(envelope={"status": "error", "content": [{"json": snapshot}]})
    result = _call(driver)
    assert result["status"] == "error"
    assert result["stopped"] is False
