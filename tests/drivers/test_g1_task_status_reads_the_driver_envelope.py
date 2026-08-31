"""``g1_get_task_status`` returns exactly what ``G1Driver.get_task_status`` gives it.

``g1_get_task_status`` is the third driver-instance-taking verb in
:mod:`strands_robots.tools.g1`, after ``g1_get_state`` and ``g1_battery``.
Where those two reach into the driver's cached DDS snapshots (the status
envelope and the BMS decode), this one wraps the driver's control-loop
task readout - the same method the mesh's status wire uses to publish loop
progress.  The tests here fix that contract by handing a hand-rolled
driver double to the verb and asserting the returned dict names each field
:meth:`G1Driver._ControlLoop.snapshot` writes (plus the ``reason`` field
the "no task has been started" branch writes when the driver has never
been asked to roll out).

The snapshot field names are read here off the driver's own writer
(:meth:`G1Driver.get_task_status` returns ``{"content": [{"json": ...}]}``
where the inner dict is either :meth:`_ControlLoop.snapshot`'s output or
the "no task" sentinel) rather than being restated in the tests, so a
rename on the driver side moves both the write path and this verb
together.  What the tests do restate is the SDK-load-hygiene contract
every file under :mod:`strands_robots.tools.g1` carries: importing the
module must not pull any ``unitree_sdk2py`` submodule.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from strands_robots.tools.g1.g1_task_status import g1_get_task_status


class _StubG1Driver:
    """A driver double whose ``get_task_status`` returns a fixed envelope.

    ``g1_get_task_status`` calls ``driver.get_task_status()`` and reads
    the ``content[0]["json"]`` payload.  This double sits under the same
    interface without pulling the real driver's imports (the real class
    reaches CycloneDDS at construction time in some paths), so a test can
    hand a wired-shape envelope to the verb without a bus.  ``calls``
    counts the invocations so a test can pin "the verb reads the driver
    exactly once" without asking the driver method itself to record.
    """

    def __init__(self, envelope: dict[str, Any]) -> None:
        self._envelope = envelope
        self.calls = 0

    def get_task_status(self) -> dict[str, Any]:
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
    return g1_get_task_status(driver=driver)


def test_the_import_pulls_no_sdk_module() -> None:
    """The tool module is loadable on a host without ``unitree_sdk2py``.

    Every file under :mod:`strands_robots.tools.g1` must be importable
    with the SDK absent; a module that pulled a submodule at import time
    would break every headless CI runner and Thor before an office
    bring-up.  The driver enforces the same rule against itself
    (:func:`~strands_robots.tools.g1._g1_common.ensure_dds` is the only
    path that loads the SDK); this cell holds the task-status verb to it
    too.
    """
    before = set(sys.modules)
    importlib.import_module("strands_robots.tools.g1.g1_task_status")
    after = set(sys.modules)
    leaked = {name for name in after - before if "unitree" in name.lower()}
    assert leaked == set(), (
        f"strands_robots.tools.g1.g1_task_status imports pulled SDK submodules: {leaked}. "
        "The rule for this package is that the SDK loads only inside function "
        "bodies (refs strands-labs/robots#358)."
    )


def test_a_driver_with_no_task_yet_reports_absent() -> None:
    """The driver's "no task has been started" envelope becomes ``present=False``.

    ``G1Driver.get_task_status`` returns this shape when ``self._loop`` is
    ``None`` and ``self._last_task_snapshot`` is ``None`` - the state
    every just-connected driver is in before its first ``run_policy``
    call.  The verb must report that decidably: ``present=False`` and
    every snapshot field ``None`` except ``running`` (which is ``False``
    unconditionally) and ``reason`` (which carries the driver's own text
    verbatim, so a caller reading the flat dict sees the same words the
    driver's method would give a JSON consumer).
    """
    driver = _StubG1Driver(
        envelope={
            "status": "success",
            "content": [
                {
                    "json": {
                        "running": False,
                        "reason": "no task has been started on this driver",
                    }
                }
            ],
        }
    )
    result = _call(driver)

    assert result["status"] == "success"
    assert result["present"] is False
    assert result["running"] is False
    assert result["reason"] == "no task has been started on this driver"
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
        assert result[field] is None, f"expected {field} None on absent-task shape, got {result[field]!r}"


def test_a_running_loop_snapshot_flattens_to_present() -> None:
    """A live-loop snapshot from the driver reshapes without dropping fields.

    :meth:`_ControlLoop.snapshot` writes ``running=True`` while the loop
    thread is alive and carries positive ``steps`` (the loop has published
    frames), a possibly-empty ``refusals`` list and an ``elapsed_s`` that
    grows.  ``exit_reason`` / ``exit_detail`` are ``None`` on a running
    loop; ``fsm_refresh_hz`` and ``fsm_reads`` name the refresher thread
    filling the cache the loop's re-gate reads.
    """
    snapshot = {
        "running": True,
        "steps": 250,
        "refusals": [],
        "elapsed_s": 0.5,
        "duration_budget_s": 2.0,
        "n_steps_budget": 1000,
        "exit_reason": None,
        "exit_detail": None,
        "hz": 500,
        "fsm_refresh_hz": 20,
        "fsm_reads": 10,
    }
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": snapshot}]})
    result = _call(driver)

    assert result["status"] == "success"
    assert result["present"] is True
    assert result["running"] is True
    # Every snapshot field round-trips verbatim - the verb reshapes, it
    # does not recompute.
    assert result["steps"] == 250
    assert result["refusals"] == []
    assert result["elapsed_s"] == 0.5
    assert result["duration_budget_s"] == 2.0
    assert result["n_steps_budget"] == 1000
    assert result["exit_reason"] is None
    assert result["exit_detail"] is None
    assert result["hz"] == 500
    assert result["fsm_refresh_hz"] == 20
    assert result["fsm_reads"] == 10
    # ``reason`` is a field of the "no task" branch; a live snapshot does
    # not carry it, so the flat dict reports ``None`` rather than raising.
    assert result["reason"] is None


def test_a_finished_loop_snapshot_reports_present_running_false() -> None:
    """A stashed ``_last_task_snapshot`` still reports ``present=True``.

    The driver's ``_ControlLoop._run`` finally-block writes
    ``_last_task_snapshot`` under ``_task_admission`` before clearing
    ``self._loop`` to ``None``, so a poller that missed the running
    window sees the loop's terminal snapshot rather than the "no task"
    sentinel.  ``running`` is ``False`` (the thread has joined) but the
    ``exit_reason`` / ``exit_detail`` fields name why - which is the
    whole point of the stash.
    """
    terminal = {
        "running": False,
        "steps": 1000,
        "refusals": [],
        "elapsed_s": 2.0,
        "duration_budget_s": 2.0,
        "n_steps_budget": 1000,
        "exit_reason": "n_steps",
        "exit_detail": "reached n_steps_budget=1000",
        "hz": 500,
        "fsm_refresh_hz": 20,
        "fsm_reads": 40,
    }
    driver = _StubG1Driver(envelope={"status": "success", "content": [{"json": terminal}]})
    result = _call(driver)

    # ``present`` is decided on the presence of ``steps``, not on
    # ``running`` - so a finished loop reads ``present=True`` and a caller
    # can tell it apart from the "no task" sentinel that reports both as
    # falsey.
    assert result["present"] is True
    assert result["running"] is False
    assert result["steps"] == 1000
    assert result["exit_reason"] == "n_steps"
    assert result["exit_detail"] == "reached n_steps_budget=1000"
    assert result["elapsed_s"] == 2.0


def test_the_verb_reads_the_driver_exactly_once() -> None:
    """One call to the verb makes one call to ``driver.get_task_status``.

    The task status envelope is a point-in-time read; if the verb polled
    twice a caller reading it could see two different snapshots on the
    same call, and a caller reading ``elapsed_s`` twice could think the
    loop has ticked while it has not.  This cell pins the verb to a
    single driver read per invocation.
    """
    driver = _StubG1Driver(
        envelope={
            "status": "success",
            "content": [{"json": {"running": False, "reason": "no task has been started on this driver"}}],
        }
    )
    _call(driver)
    assert driver.calls == 1


def test_the_envelope_status_is_reported_verbatim() -> None:
    """A non-success envelope from the driver surfaces on the returned dict.

    The driver's ``get_task_status`` returns ``status="success"`` on every
    shape today (both "no task" and the live/stashed snapshot), but the
    verb must not hard-code that: the driver's contract is that ``status``
    is authoritative, and a future refusal on this path (an admission
    lock that could not be acquired within a bound, say) has to reach the
    caller as an error rather than being masked to success by the verb.
    """
    driver = _StubG1Driver(
        envelope={
            "status": "error",
            "content": [{"json": {"running": False, "reason": "task admission lock timed out"}}],
        }
    )
    result = _call(driver)
    assert result["status"] == "error"
    assert result["reason"] == "task admission lock timed out"
