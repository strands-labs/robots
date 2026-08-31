"""Agent-facing wrapper for ``G1Driver.get_task_status``.

``G1Driver.get_task_status`` returns a JSON envelope naming whether the
driver's control loop is currently rolling out a policy, and if it is, the
loop's ``running`` / ``steps`` / ``refusals`` / ``elapsed_s`` fields from
``_ControlLoop.snapshot`` (plus the ``duration_budget_s`` / ``n_steps_budget``
budgets the loop was started with and the ``fsm_refresh_hz`` /
``fsm_reads`` fields that name who is filling the FSM cache the re-gate
reads).  A poller that missed the running window still sees the loop's
final snapshot: the driver stashes ``_last_task_snapshot`` under the
admission lock right before its ``finally`` clears ``self._loop``, so every
self-terminating exit reason (``n_steps``, ``duration``, ``gate``,
``policy``, ``publish``) round-trips to the caller instead of collapsing to
"no task has been started" once the thread joins.

This module is the agent-facing side of that read.  It calls
:meth:`~strands_robots.drivers.g1.G1Driver.get_task_status` once, unwraps
the ``content[0]["json"]`` envelope the driver returns and reshapes it into
a flat dict the ``@tool`` contract exposes.  Read-only.  Every field is a
driver-side snapshot; no DDS is subscribed, no bus is touched, no
locomotion client is called (the driver's method takes only its own
``_task_admission`` lock, which the same admission on the driver already
serialises).  The verb does not consult the FSM gate: reading the loop's
snapshot is not a motion write, and the driver's own method returns a
success envelope whether or not the gate would admit a new write today -
"no task running" and "loop finished with exit_reason=n_steps" are honest
answers, not motion refusals, and the driver's own suite pins both as
successes rather than as refusals.

The driver argument is typed :class:`~typing.Any` at runtime rather than
as ``G1Driver`` for the same reason the ``g1_state`` module gives: the
driver module imports ``ensure_dds`` from this package at load, so a
runtime import of ``G1Driver`` here would close a cycle, and ``@tool``
calls :func:`typing.get_type_hints` at decoration time so a string forward
reference cannot resolve without pulling the driver at import.  The verb
is duck-typed on ``get_task_status`` (any object with a synchronous
``get_task_status`` returning the driver's envelope answers), which is
also how the tests hand it a hand-rolled double.  ``import
strands_robots.tools.g1.g1_task_status`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene contract,
refs strands-labs/robots#358).

What this module does not do.

* Start or stop a task.  The driver's ``run_policy`` starts the loop
  (motion-scoped through :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`),
  and ``stop_task`` joins it and publishes the zero-torque frame on the
  way out.  Both are writes and both belong on their own verbs; this one
  reads what the loop is doing right now and cannot change it.
* Poll the loop on a schedule.  A caller who wants a rate calls this
  verb on their own timer; the driver's ``_control_loop`` publishes at
  :data:`~strands_robots.drivers.g1._CONTROL_LOOP_HZ` and this verb is a
  point read at whatever cadence the caller picks.
* Decode ``exit_reason`` into a friendlier label.  The driver's
  ``_ControlLoop._run`` finally-block names the five self-terminating
  exit reasons (``n_steps``, ``duration``, ``gate``, ``policy``,
  ``publish``) as the ``exit_reason`` field in the loop's snapshot;
  restating them here would be a second source of truth for a domain the
  driver's own snapshot already carries verbatim.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal


@tool
def g1_get_task_status(driver: Any) -> dict[str, Any]:
    """Return the driver's control-loop task snapshot.

    Read-only.  Calls
    :meth:`~strands_robots.drivers.g1.G1Driver.get_task_status` once and
    reshapes its ``content[0]["json"]`` payload into a flat agent-facing
    dict.  The driver's method already reports two shapes: the live loop's
    ``snapshot`` while ``_loop`` is set (with ``running`` / ``steps`` /
    ``refusals`` / ``elapsed_s`` and the loop's exit-reason fields once
    finished) or a ``running=False`` sentinel with a ``reason`` field
    (either "no task has been started on this driver" or the stashed
    ``_last_task_snapshot`` from the loop's ``finally``).  This verb
    surfaces both shapes on the same envelope and adds a ``present``
    boolean naming whether the driver has ever run a loop, so a caller
    reading the flat dict can tell "the driver has just started, no task
    has been submitted yet" from "the last task finished with
    exit_reason=n_steps two seconds ago" without parsing the ``reason``
    string.

    Args:
        driver: An object with a synchronous ``get_task_status`` method
            returning the driver's task envelope (in practice a
            :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep this
            module out of the import cycle the driver's own
            ``ensure_dds`` reach into this package would close - see the
            module docstring's SDK-load-hygiene note.  The verb is
            duck-typed on ``get_task_status``; any object with that
            method returning the envelope shape the driver writes will
            satisfy it.

    Returns:
        A dict with ``status`` (the envelope's own ``status`` value, so a
        driver method that ever refuses on this path surfaces the refusal
        verbatim - today it does not; the driver returns
        ``status="success"`` on both shapes), a ``present`` flag naming
        whether the driver has ever started a loop (``False`` only on the
        just-connected "no task has been started" shape), the loop's
        ``running`` flag, and the eight snapshot fields the driver's
        ``_ControlLoop.snapshot`` writes: ``steps`` (integer count of
        published frames), ``refusals`` (a list of per-step refusal
        records the re-gate accumulated), ``elapsed_s`` (float seconds
        since the loop started, or since it finished if it has), the two
        budgets the loop was started with (``duration_budget_s`` /
        ``n_steps_budget``, either float / int or ``None`` when
        open-ended), ``exit_reason`` (one of the five self-terminating
        reasons named in the module docstring, or ``None`` while running),
        ``exit_detail`` (a per-reason free-text field the loop's exit
        branch writes), ``hz`` (the loop's target publish rate, from
        :data:`~strands_robots.drivers.g1._CONTROL_LOOP_HZ`), and the two
        fields the FSM refresher writes (``fsm_refresh_hz`` /
        ``fsm_reads``, naming who fills the FSM cache the re-gate reads).
        On the "no task has been started" shape ``present`` is ``False``
        and every snapshot field is ``None`` except ``running`` (which is
        ``False``) and ``reason`` (which quotes the driver's own text
        verbatim).
    """
    # The handle is a live object typed ``Any``, so the generated tool schema
    # carries nothing telling a model the argument cannot be synthesized: a
    # caller reaches this verb with ``None`` or with a robot *name*, and the
    # accessor call below would surface that as ``AttributeError`` naming a
    # method rather than as the refusal a ``@tool`` owes its caller.  One
    # owner for that judgement, shared with every sibling verb in this
    # package: the refusal is an error envelope naming this verb, the
    # ``driver`` parameter and the type it received, never an exception.
    refusal = live_handle_refusal(
        "g1_get_task_status",
        driver,
        accessor="get_task_status",
        reads=("the verb reads the task snapshot the driver's own control loop writes"),
        expected=(
            "the task-status accessor this verb reads. Pass a strands_robots "
            "G1Driver, or an object with a callable `get_task_status()` "
            "answering the driver's task envelope."
        ),
    )
    if refusal is not None:
        return refusal

    envelope = driver.get_task_status()
    inner: dict[str, Any] = envelope["content"][0]["json"]

    # The driver's ``get_task_status`` returns one of two shapes:
    #
    # * ``{"running": False, "reason": "no task has been started on this
    #   driver"}`` - the just-connected shape, no loop has been submitted
    #   and ``_last_task_snapshot`` is ``None``.
    # * The loop's snapshot dict (live or stashed under ``_last_task_snapshot``
    #   in the ``finally``), which always carries ``steps`` and
    #   ``refusals`` because ``_ControlLoop.snapshot`` writes them
    #   unconditionally.
    #
    # ``present`` is ``True`` only on the second shape, decided on the
    # presence of the ``steps`` field - not on ``running``, which is
    # ``False`` on both a just-connected driver and a finished loop.  The
    # driver names its "no task" shape with the ``reason`` field and no
    # snapshot fields, so a caller reading ``present`` can tell those two
    # apart without parsing the free-text reason.
    is_snapshot = "steps" in inner

    return {
        "status": envelope["status"],
        "present": is_snapshot,
        "running": inner.get("running", False),
        "steps": inner.get("steps"),
        "refusals": inner.get("refusals"),
        "elapsed_s": inner.get("elapsed_s"),
        "duration_budget_s": inner.get("duration_budget_s"),
        "n_steps_budget": inner.get("n_steps_budget"),
        "exit_reason": inner.get("exit_reason"),
        "exit_detail": inner.get("exit_detail"),
        "hz": inner.get("hz"),
        "fsm_refresh_hz": inner.get("fsm_refresh_hz"),
        "fsm_reads": inner.get("fsm_reads"),
        "reason": inner.get("reason"),
    }
