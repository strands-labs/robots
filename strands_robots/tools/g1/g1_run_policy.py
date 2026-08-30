"""Agent-facing wrapper for ``G1Driver.run_policy``.

``G1Driver.run_policy`` starts the driver's 500 Hz control loop against
an already-built policy: it spawns a dedicated thread that re-gates
through :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
with scope ``"motion"`` on every step, calls ``policy_object.step(obs)``
(or the object itself, if it is a bare callable) for a joint-name-keyed
action dict, and publishes one :class:`LowCmd_` frame per step on
``rt/lowcmd`` through the same publisher :meth:`send_action` writes.  A
gate flip mid-rollout refuses the step and the loop publishes a
zero-torque frame before exiting rather than freezing with the last
commanded posture (the driver's own five-reason exit table: ``n_steps``,
``duration``, ``gate``, ``policy``, ``publish``).

This module is the agent-facing side of that write.  It calls
:meth:`~strands_robots.drivers.g1.G1Driver.run_policy` once and returns
the envelope the driver produced verbatim.  A caller who wants to
observe the rollout reaches :meth:`get_task_status` (the ``g1_task_status``
verb); a caller who wants to end it reaches :meth:`stop_task` (the
``g1_stop_task`` verb).  The driver's own :class:`_task_admission` lock
serialises this verb against those two, so this module needs no lock of
its own.  No DDS is subscribed, no bus is touched, no motion switcher
is opened; the loop's publisher is opened by the driver on start and
the zero-torque frame on exit publishes through it.

The FSM gate is not consulted here.  The driver's
:meth:`~strands_robots.drivers.g1.G1Driver.run_policy` runs the arm-SDK
admission gate before spawning the loop and re-runs it inside the loop
on every step - a second gate call here would double the read against a
cache the driver's FSM refresher fills, and a caller who saw the first
gate answer ``None`` and the second refuse could not tell which read it
should trust.  Restating any of that on this side would be a second
source of truth for a rule the driver's own path already enforces
(refs strands-labs/robots#358, strands-labs/robots#2916).  ``import
strands_robots.tools.g1.g1_run_policy`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene contract,
refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_stop_task`` and
``g1_send_action`` give: the driver module imports
:func:`~strands_robots.tools.g1._g1_common.ensure_dds` from this
package at load, so a runtime import of ``G1Driver`` here would close
a cycle, and ``@tool`` calls :func:`typing.get_type_hints` at
decoration time so a string forward reference cannot resolve without
pulling the driver at import.  The verb is duck-typed on
``run_policy``; any object with a synchronous
``run_policy(policy_object, instruction=..., duration=..., n_steps=...)``
returning the driver's start envelope satisfies it, which is also how
the tests hand it a hand-rolled double.

``policy_object`` is a live Python object - a built
:class:`~strands_robots.policies.Policy` instance or a bare callable
returning a joint-name-keyed action dict per step.  The tool schema
carries no signal that ``None`` or a robot *name* is refused, so this
verb gates the argument shape here before reaching the driver (with a
refusal envelope naming ``policy_object`` and the remedy) rather than
letting the driver's own admission surface the same refusal through a
call site the caller cannot map back to the parameter.  The driver's
:meth:`~strands_robots.drivers.g1.G1Driver.run_policy` also refuses a
non-callable / no-``.step()`` object verbatim once its own path is
reached; the verb's early refusal has the same effect for a caller who
never called the driver in the first place.

What this module does not do.

* Build the policy.  The driver's ``run_policy`` accepts an already-built
  policy or a bare callable; a second construction path here would fork
  the policy-registry lookup the ``strands_robots.policies`` module owns
  today, and a caller who wanted a specific policy would have to know
  which of the two paths the verb chose.  Building the policy is the
  orchestrator's shape, not the verb's.
* Poll the loop.  The driver's ``get_task_status`` is the one reader
  for the loop's live snapshot; a caller who wants ``steps``,
  ``refusals``, ``elapsed_s`` or the loop's ``exit_reason`` reaches that
  verb (``g1_task_status``).  This verb only reports the start.
* Stop the loop.  The driver's ``stop_task`` signals the loop's exit
  and joins its thread; a caller who wants a controlled stop reaches
  that verb (``g1_stop_task``).  This verb does not carry a stop knob.
* Restate the driver's five exit-reason strings.  The driver's
  :class:`_ControlLoop._run` finally-block names ``n_steps``,
  ``duration``, ``gate``, ``policy``, ``publish`` verbatim, and
  ``get_task_status`` surfaces them.  Restating them here would be a
  second source of truth for a domain the driver's own snapshot
  already carries.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal


def _refusal_envelope(text: str) -> dict[str, Any]:
    """Return the ``@tool`` error envelope with ``text`` inside.

    Kept as a small free helper so every ``policy_object``-refusal path
    in this module renders the same shape a caller can grep for, matching
    the driver's own :func:`~strands_robots.drivers.g1._refuse` free
    function on the write side.
    """
    return {"status": "error", "content": [{"text": text}]}


@tool
def g1_run_policy(
    driver: Any,
    policy_object: Any = None,
    instruction: str = "",
    duration: float = 30.0,
    n_steps: int | None = None,
) -> dict[str, Any]:
    """Start the driver's 500 Hz control loop against ``policy_object``.

    Calls :meth:`~strands_robots.drivers.g1.G1Driver.run_policy` once
    and returns the envelope the driver produced verbatim.  The
    driver's method re-gates through
    :meth:`~strands_robots.drivers.g1.G1Driver._check_motion_gates`
    with scope ``"motion"`` on every step, so a caller whose FSM leaves
    the arm-SDK admission set mid-rollout has the loop publish a
    zero-torque frame and exit with ``exit_reason="gate"`` rather than
    freezing with the last commanded posture (refs
    strands-labs/robots#2916).  The call returns immediately once the
    loop's thread has started; the caller polls
    :meth:`~strands_robots.drivers.g1.G1Driver.get_task_status` (the
    ``g1_task_status`` verb) to observe progress and reaches
    :meth:`~strands_robots.drivers.g1.G1Driver.stop_task` (the
    ``g1_stop_task`` verb) to end it.

    Args:
        driver: An object with a synchronous
            ``run_policy(policy_object, instruction=..., duration=...,
            n_steps=...)`` returning the driver's start envelope (in
            practice a :class:`~strands_robots.drivers.g1.G1Driver`).
            Typed :class:`~typing.Any` rather than as ``G1Driver`` to
            keep this module out of the import cycle the driver's own
            :func:`~strands_robots.tools.g1._g1_common.ensure_dds` reach
            into this package would close - see the module docstring's
            SDK-load-hygiene note.  The verb is duck-typed on
            ``run_policy``; any object with that method returning the
            envelope shape the driver writes will satisfy it.
        policy_object: An already-built policy - either a
            :class:`~strands_robots.policies.Policy` instance with a
            ``.step(obs)`` method or a bare callable that returns a
            joint-name-keyed action dict per step.  The driver's own
            :meth:`~strands_robots.drivers.g1.G1Driver.run_policy`
            refuses a ``None`` or non-callable / no-``.step()`` object
            verbatim once its own path is reached; this verb refuses
            the same shape here before reaching the driver so the
            refusal envelope names ``policy_object`` and the remedy
            rather than surfacing the driver's own message through a
            call site the caller cannot map back to the parameter.
        instruction: A free-form conditioning string.  The driver's
            :meth:`~strands_robots.drivers.g1.G1Driver.run_policy`
            discards it (``del instruction  # policies own their own
            conditioning``); a policy that reads instructions carries
            its own state and does not need the driver to route them.
            The parameter is retained on this verb for the shape
            :meth:`send_action`'s wire frame and every language-driven
            manipulation demo already exposes; the driver's discard
            is documented and stable.
        duration: Wall-clock budget for the rollout in seconds.  The
            driver's method validates this against
            :func:`~strands_robots.utils.positive_finite_number_error`
            (``nan`` poisons every ``deadline`` comparison in the loop;
            ``inf`` collapses the exit test to always-false; a
            non-numeric string raises out of a method that must return
            an envelope).  Defaults to ``30.0`` seconds; the loop
            exits with ``exit_reason="duration"`` when
            ``time.monotonic() - started_at >= duration``.
        n_steps: Optional step-count budget.  ``None`` (the default)
            means "no step cap"; a positive integer caps the loop at
            that many steps and exits with ``exit_reason="n_steps"``
            when ``self._steps >= self._n_steps``.  The driver's
            method validates this against
            :func:`~strands_robots.utils.positive_count_error` when
            not ``None`` (a ``bool`` silently caps at 1, a fractional
            applies a cap the caller never named, and ``0`` / negative
            exits instantly with ``exit_reason="n_steps"`` on a rollout
            that commanded nothing).

    Returns:
        The envelope :meth:`G1Driver.run_policy` returned.  On the
        success path this is ``{"status": "success", "content":
        [{"json": {"tool_name": ..., "task_running": True,
        "duration": ..., "n_steps": ..., "hz": 500}}]}``; on the
        driver's refusal path (``duration`` / ``n_steps`` validation,
        gate flip, ``policy_object`` refused by the driver, a task
        already running) it is ``{"status": "error", "content":
        [{"text": "..."}]}`` with the driver's own reason inside.
        The verb does not reshape either shape - a future field the
        driver adds on the success path reaches a caller the moment
        the driver writes it, because this verb passes the envelope
        through.
    """
    # The handle is a live Python object typed :class:`~typing.Any` (see
    # the module docstring's import-cycle note), so the tool schema
    # carries no signal that ``None`` or a robot *name* is refused.  The
    # shared ``live_handle_refusal`` guard is the one implementation of
    # that judgement for this package; it is keyed on the accessor the
    # verb reads, which for this verb is ``run_policy`` (a callable that
    # starts the 500 Hz control loop and returns the driver's start
    # envelope) rather than the sensor verbs' ``_snapshot``.  Returning
    # its refusal envelope here rather than raising keeps the four
    # invariants every ``@tool`` handler owes a caller (envelope not
    # exception, names the verb, names ``driver``, names the type on
    # wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_run_policy",
        driver,
        accessor="run_policy",
        reads=(
            "the verb starts the driver's own 500 Hz control-loop thread "
            "against a caller-supplied policy and reads back the start "
            "envelope the driver produced"
        ),
        expected=(
            "a callable ``run_policy(policy_object, instruction=..., "
            "duration=..., n_steps=...)`` returning the driver's start "
            "envelope - pass the live G1Driver handle the orchestrator "
            "constructed"
        ),
    )
    if refusal is not None:
        return refusal

    # ``policy_object`` is a live Python object the tool schema *does*
    # describe (a callable or a ``.step()``-exposing object), so a model
    # can synthesize the wrong shape here as easily as it can reach the
    # verb with the right one.  These two refusals cover the shapes the
    # driver's own path would surface as an inner refusal - a ``None``
    # and a non-callable / no-``.step()`` object.  Naming ``policy_object``
    # here keeps the four invariants for the parameter: envelope not
    # exception, names the verb, names the parameter, names the shape
    # received.  The driver's own refusal fires for the same shapes if
    # a caller reached its method directly; the verb's refusal only
    # differs in the call site (a call site the caller can map back to
    # ``policy_object`` on the tool's own signature).
    if policy_object is None:
        return _refusal_envelope(
            "g1_run_policy: `policy_object` is required. Pass an "
            "already-built policy - either a Policy instance with a "
            "`.step(obs)` method or a bare callable that returns a "
            "joint-name-keyed action dict per step - see "
            "G1Driver.run_policy for the shape "
            "(refs strands-labs/robots#361)."
        )
    step_fn = getattr(policy_object, "step", None)
    if not callable(step_fn) and not callable(policy_object):
        return _refusal_envelope(
            f"g1_run_policy: `policy_object` of type "
            f"{type(policy_object).__name__!r} is neither callable nor "
            "exposes a callable `.step()`. Pass an already-built policy "
            "- either a Policy instance with a `.step(obs)` method or a "
            "bare callable that returns a joint-name-keyed action dict "
            "per step - see G1Driver.run_policy for the shape "
            "(refs strands-labs/robots#361)."
        )

    return driver.run_policy(
        policy_object,
        instruction=instruction,
        duration=duration,
        n_steps=n_steps,
    )
