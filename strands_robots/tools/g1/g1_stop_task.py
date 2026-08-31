"""Agent-facing wrapper for ``G1Driver.stop_task``.

``G1Driver.stop_task`` signals the driver's running control loop to exit,
joins its thread, and lets the loop publish
:func:`_build_zero_torque_lowcmd` on the way out - a soft *controlled*
stop rather than a Disable that would let the named joints fall freely.
Idempotent: a call against a driver whose ``_loop`` is ``None`` or has
already finished returns a success envelope naming the state (`"stop_task:
no task is running"`) instead of raising, so a caller polling this verb
from a supervisor cannot get a spurious refusal by racing the loop's own
self-terminating exit.

The driver's method reports the join outcome honestly.  A caller-supplied
policy that outlasts the join budget (a remote inference call is the
ordinary case) surfaces as ``status="error"`` naming the timeout, with the
snapshot's ``stopped=False`` and ``running=True`` in the payload, so a
caller that reads only ``status`` cannot count the task as stopped while
the payload's own fields say the loop is still writing frames.

This module is the agent-facing side of that write.  It calls
:meth:`~strands_robots.drivers.g1.G1Driver.stop_task` once, unwraps the
envelope the driver returns (either the ``content[0]["text"]`` no-task
sentinel or the ``content[0]["json"]`` snapshot dict) and reshapes it into
a flat dict the ``@tool`` contract exposes.  The driver's own admission
lock (``_task_admission``) serialises this against ``run_policy`` and
``get_task_status``, so the verb does not need its own lock.  No DDS is
subscribed, no bus is touched, no motion switcher is opened; ``stop_task``
publishes the zero-torque frame through the loop's already-open publisher
on the way out, and this verb is that request's shape.

The FSM gate is not consulted here.  Stopping a task is not a positive
motion write; the driver's ``run_policy`` already ran through
:meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates` to admit
the loop, and the loop's ``finally`` publishes zero-torque unconditionally
- a stop-under-gate-refusal would trap the loop until it self-terminated,
which is exactly what the driver's contract refuses to do.  ``import
strands_robots.tools.g1.g1_stop_task`` still pulls no ``unitree_sdk2py``
submodule (the package's SDK-load-hygiene contract, refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather than
as ``G1Driver`` for the same reason the ``g1_task_status`` module gives:
the driver module imports ``ensure_dds`` from this package at load, so a
runtime import of ``G1Driver`` here would close a cycle, and ``@tool``
calls :func:`typing.get_type_hints` at decoration time so a string forward
reference cannot resolve without pulling the driver at import.  The verb
is duck-typed on ``stop_task``; any object with a synchronous
``stop_task`` returning the driver's envelope satisfies it, which is also
how the tests hand it a hand-rolled double.

What this module does not do.

* Start or observe a task.  The driver's ``run_policy`` starts the loop
  (motion-scoped through :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`),
  and ``get_task_status`` reports its live snapshot.  Both are separate
  verbs; this one only requests the loop's exit and reports the join.
* Fall the robot.  The driver's ``_ControlLoop._run`` finally-block
  publishes :func:`_build_zero_torque_lowcmd` before returning; the joints
  hold their weight rather than dropping.  A caller that wants a hard
  Disable calls the motion switcher directly, not this verb.
* Decode ``exit_reason`` into a friendlier label.  The driver's
  ``_ControlLoop._run`` finally-block names the five self-terminating
  exit reasons (``n_steps``, ``duration``, ``gate``, ``policy``,
  ``publish``); a ``stop_task`` request adds ``stop_task`` as the sixth.
  Restating them here would be a second source of truth for a domain the
  driver's own snapshot already carries verbatim.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal


@tool
def g1_stop_task(driver: Any) -> dict[str, Any]:
    """Signal the driver's control loop to stop and report the join.

    Calls :meth:`~strands_robots.drivers.g1.G1Driver.stop_task` once and
    reshapes the returned envelope into a flat agent-facing dict.  The
    driver's method returns one of three shapes:

    * ``{"status": "success", "content": [{"text": "stop_task: no task
      is running"}]}`` when no loop is running or the loop has already
      exited.  The verb surfaces this as ``present=False`` and
      ``running=False`` with every snapshot field ``None`` except
      ``reason`` (which quotes the driver's text verbatim so a caller
      that logs ``reason`` sees the same string the driver wrote).
    * ``{"status": "success", "content": [{"json": snap}]}`` when the
      loop joined within budget.  The snapshot's ``stopped`` field is
      ``True`` and ``running`` is ``False``; the verb surfaces every
      snapshot field flat with ``present=True``.
    * ``{"status": "error", "content": [{"json": snap}]}`` when the
      loop's join outlasted the budget (a policy blocking on inference
      is the ordinary case).  The snapshot's ``stopped`` field is
      ``False`` and ``running`` may still be ``True``; the verb
      surfaces ``status="error"`` and every snapshot field flat with
      ``present=True``, so a caller reading only ``status`` cannot
      read "success" while the payload says the loop is still holding
      the wire.

    Args:
        driver: An object with a synchronous ``stop_task`` method
            returning the driver's stop envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep this
            module out of the import cycle the driver's own
            ``ensure_dds`` reach into this package would close - see the
            module docstring's SDK-load-hygiene note.  The verb is
            duck-typed on ``stop_task``; any object with that method
            returning the envelope shape the driver writes will satisfy
            it.

    Returns:
        A dict with ``status`` (the envelope's own ``status`` value, so
        the join-outlasted-budget shape surfaces its ``"error"`` verbatim
        rather than being flattened to a success), a ``present`` flag
        naming whether the driver returned a snapshot dict (``True`` on
        both the joined and the timed-out shape, ``False`` on the "no
        task is running" text sentinel), the loop's ``stopped`` flag
        (``True`` if the loop joined within budget, ``False`` if the
        join timed out, ``None`` on the "no task is running" shape),
        the ``running`` flag (``True`` only on the timed-out shape where
        the loop is still writing frames), and the eight snapshot fields
        the driver's ``_ControlLoop.snapshot`` writes: ``steps``,
        ``refusals``, ``elapsed_s``, ``duration_budget_s``,
        ``n_steps_budget``, ``exit_reason``, ``exit_detail``, ``hz``,
        and the two FSM-refresher fields ``fsm_refresh_hz`` /
        ``fsm_reads``.  On the "no task is running" shape ``reason``
        quotes the driver's own text verbatim and every snapshot field
        is ``None``.
    """
    # The handle is a live Python object typed :class:`~typing.Any` (see the
    # module docstring's import-cycle note), so the tool schema carries no
    # signal that ``None`` or a robot *name* is refused.  The shared
    # ``live_handle_refusal`` guard is the one implementation of that judgement
    # for this package; it is keyed on the accessor the verb reads, which for
    # this verb is ``stop_task`` (a callable that requests the loop's exit and
    # returns the driver's stop envelope) rather than the sensor verbs'
    # ``_snapshot``.  Returning its refusal envelope here rather than raising
    # keeps the four invariants every ``@tool`` handler owes a caller
    # (envelope not exception, names the verb, names ``driver``, names the
    # type on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_stop_task",
        driver,
        accessor="stop_task",
        reads=(
            "the verb signals the driver's own control-loop thread to exit and "
            "reads back the join outcome the driver produced"
        ),
        expected=(
            "a callable ``stop_task`` returning the driver's stop envelope - "
            "pass the live G1Driver handle the orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    envelope = driver.stop_task()
    payload = envelope["content"][0]

    # The driver's ``stop_task`` returns one of three shapes:
    #
    # * ``{"content": [{"text": "stop_task: no task is running"}]}`` -
    #   the idempotent no-task branch, no snapshot to unwrap.
    # * ``{"content": [{"json": snap}]}`` where ``snap`` carries
    #   ``stopped=True`` - the loop joined within budget.
    # * ``{"content": [{"json": snap}]}`` where ``snap`` carries
    #   ``stopped=False`` and ``status="error"`` on the outer envelope -
    #   the join outlasted the budget.
    #
    # ``present`` is ``True`` iff the driver returned a snapshot dict,
    # decided on the presence of the ``json`` key on the payload rather
    # than on the envelope's ``status`` (a future refusal path that
    # returned a snapshot would still carry ``json``; the "no task"
    # sentinel is the only shape that carries ``text``).
    if "json" in payload:
        snap: dict[str, Any] = payload["json"]
        return {
            "status": envelope["status"],
            "present": True,
            "stopped": snap.get("stopped"),
            "running": snap.get("running", False),
            "steps": snap.get("steps"),
            "refusals": snap.get("refusals"),
            "elapsed_s": snap.get("elapsed_s"),
            "duration_budget_s": snap.get("duration_budget_s"),
            "n_steps_budget": snap.get("n_steps_budget"),
            "exit_reason": snap.get("exit_reason"),
            "exit_detail": snap.get("exit_detail"),
            "hz": snap.get("hz"),
            "fsm_refresh_hz": snap.get("fsm_refresh_hz"),
            "fsm_reads": snap.get("fsm_reads"),
            "reason": snap.get("reason"),
        }

    # The "no task is running" text sentinel.  The driver writes the
    # exact string ``"stop_task: no task is running"`` and this verb
    # surfaces it verbatim on ``reason`` so a caller that logs the
    # field sees the driver's own text, not a paraphrase.
    return {
        "status": envelope["status"],
        "present": False,
        "stopped": None,
        "running": False,
        "steps": None,
        "refusals": None,
        "elapsed_s": None,
        "duration_budget_s": None,
        "n_steps_budget": None,
        "exit_reason": None,
        "exit_detail": None,
        "hz": None,
        "fsm_refresh_hz": None,
        "fsm_reads": None,
        "reason": payload.get("text"),
    }
