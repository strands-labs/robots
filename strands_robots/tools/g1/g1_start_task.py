"""Agent-facing wrapper for ``G1Driver.start_task``.

``G1Driver.start_task`` is the provider-registry entry point for the
driver's 500 Hz control loop: a caller passes an ``instruction`` and a
``policy_provider`` name (``"groot"`` today, more to come) and the
driver looks the provider up in :mod:`strands_robots.policies`, builds
the policy inline and hands it to the same
:meth:`~strands_robots.drivers.g1.G1Driver.run_policy` path.  The
provider registry is not yet plumbed to this driver, so the driver's
method refuses every call after the FSM/battery gate admits with a
verbatim ``start_task: provider registry not wired yet; use
run_policy(policy_object=...) to drive the control loop today``
message; that separation lets the arm-SDK write path
(:meth:`send_action`) and the loop path (:meth:`run_policy`) land
without waiting on the provider table (refs strands-labs/robots#358).

This module is the agent-facing side of that call.  It reaches
:meth:`~strands_robots.drivers.g1.G1Driver.start_task` once and
returns the envelope the driver produced verbatim.  A caller who
wants to drive the loop today reaches
:meth:`~strands_robots.drivers.g1.G1Driver.run_policy` (the
``g1_run_policy`` verb) with an already-built policy; a caller who
wants to observe or end a loop the driver is already running reaches
:meth:`get_task_status` (``g1_task_status``) or :meth:`stop_task`
(``g1_stop_task``).  The driver's own
:class:`_task_admission` lock serialises this verb against those
three, so this module needs no lock of its own.  No DDS is
subscribed, no bus is touched, no motion switcher is opened; the
driver's method returns a refusal envelope today and (once the
registry lands) will spawn the loop on its own thread and return
verbatim what :meth:`run_policy` returns.

The FSM gate is not consulted here.  The driver's
:meth:`~strands_robots.drivers.g1.G1Driver.start_task` runs the
arm-SDK admission gate with scope ``"motion"`` before the refusal
returns, so a caller whose FSM leaves the admission set surfaces the
gate's own refusal string rather than the registry-not-wired one; a
second gate call on this side would double the read against a cache
the driver's FSM refresher fills, and a caller who saw the first
gate answer ``None`` and the second refuse could not tell which read
to trust.  Restating that here would be a second source of truth for
a rule the driver's own path already enforces (refs
strands-labs/robots#2916).  ``import
strands_robots.tools.g1.g1_start_task`` still pulls no
``unitree_sdk2py`` submodule (the package's SDK-load-hygiene
contract, refs strands-labs/robots#358).

The driver argument is typed :class:`~typing.Any` at runtime rather
than as ``G1Driver`` for the same reason ``g1_stop_task``,
``g1_run_policy`` and ``g1_send_action`` give: the driver module
imports :func:`~strands_robots.tools.g1._g1_common.ensure_dds` from
this package at load, so a runtime import of ``G1Driver`` here would
close a cycle, and ``@tool`` calls :func:`typing.get_type_hints` at
decoration time so a string forward reference cannot resolve without
pulling the driver at import.  The verb is duck-typed on
``start_task``; any object with a synchronous
``start_task(instruction, policy_port=..., policy_host=...,
policy_provider=..., duration=..., **kwargs)`` returning the
driver's envelope satisfies it, which is also how the tests hand it
a hand-rolled double.

What this module does not do.

* Build a policy.  The driver's ``start_task`` looks the provider up
  in :mod:`strands_robots.policies` (once wired); a caller who
  already holds a built policy reaches ``g1_run_policy``, not this
  verb.  Restating the registry on this side would fork the
  lookup :mod:`strands_robots.policies` owns and hand a caller two
  paths to the same provider.
* Poll or stop the loop.  The driver's ``get_task_status`` reports
  the loop's live snapshot and :meth:`stop_task` signals its exit;
  both are separate verbs (``g1_task_status`` / ``g1_stop_task``).
  This verb only requests the start.
* Restate the driver's refusal wording.  The registry-not-wired
  message is exactly the driver's current shape; a wording drift on
  the driver side moves this verb with it because the verb passes
  the envelope through.  A verbatim quote here would trap the verb
  to one release's prose (refs strands-labs/robots#2874).
* Refuse ``policy_provider`` names the driver does not know.  The
  driver's own registry lookup is the source of truth for the
  admission set; a second admission on this side would deny a name
  the registry knows about but this module was written before or
  admit a name the registry has since removed.
"""

from __future__ import annotations

from typing import Any

from strands import tool

from strands_robots.tools.g1._g1_common import live_handle_refusal


@tool
def g1_start_task(
    driver: Any,
    instruction: str = "",
    policy_port: int | None = None,
    policy_host: str = "localhost",
    policy_provider: str = "groot",
    duration: float = 30.0,
) -> dict[str, Any]:
    """Start a provider-driven task on the driver's 500 Hz control loop.

    Calls :meth:`~strands_robots.drivers.g1.G1Driver.start_task` once
    and returns the envelope the driver produced verbatim.  The
    driver's method runs the arm-SDK admission gate (scope
    ``"motion"``) and, on today's driver, refuses with a
    ``start_task: provider registry not wired yet; use
    run_policy(policy_object=...) to drive the control loop today``
    message because the registry in :mod:`strands_robots.policies`
    is not yet plumbed to this driver.  A caller who wants to drive
    the loop today reaches
    :meth:`~strands_robots.drivers.g1.G1Driver.run_policy` (the
    ``g1_run_policy`` verb) with an already-built policy; this verb
    is the shape the provider-registry landing will surface once it
    ships.  Once the registry lands the same call returns the
    driver's start envelope verbatim (the shape ``run_policy``
    returns today).

    Args:
        driver: An object with a synchronous
            ``start_task(instruction, policy_port=...,
            policy_host=..., policy_provider=..., duration=...,
            **kwargs)`` returning the driver's envelope (in practice
            a :class:`~strands_robots.drivers.g1.G1Driver`).  Typed
            :class:`~typing.Any` rather than as ``G1Driver`` to keep
            this module out of the import cycle the driver's own
            :func:`~strands_robots.tools.g1._g1_common.ensure_dds`
            reach into this package would close - see the module
            docstring's SDK-load-hygiene note.  The verb is
            duck-typed on ``start_task``; any object with that
            method returning the envelope shape the driver writes
            will satisfy it.
        instruction: A free-form conditioning string handed to the
            provider once the registry lands.  The driver's
            :meth:`~strands_robots.drivers.g1.G1Driver.start_task`
            discards it today (``del instruction, ...``) alongside
            every other provider-facing argument, because there is
            no provider to receive them; the parameter is retained
            on this verb for the shape every language-driven
            manipulation demo already exposes and for parity with
            :meth:`send_action`'s wire frame.  The driver's discard
            is documented and stable.
        policy_port: TCP port a remote inference server listens on,
            passed through to the provider once the registry lands.
            ``None`` (the default) lets the provider pick its own
            default.  Discarded on today's driver alongside
            ``instruction``.
        policy_host: Hostname of the remote inference server;
            ``"localhost"`` is the default the lerobot driver uses
            in the same shape.  Passed through to the provider once
            the registry lands, discarded on today's driver.
        policy_provider: Provider name looked up in
            :mod:`strands_robots.policies` once the registry lands.
            ``"groot"`` is the neon reference stack's default and
            matches the driver's own signature default.  The
            registry is the source of truth for the admission set;
            this verb does not gate the name on this side (see the
            module docstring's "does not refuse" note).
        duration: Wall-clock budget for the rollout in seconds.
            Handed to the same loop :meth:`run_policy` starts, so
            the ``deadline = started_at + duration`` shape and
            ``exit_reason="duration"`` exit apply once the provider
            registry lands.  Defaults to ``30.0`` seconds; discarded
            on today's driver.

    Returns:
        The envelope :meth:`G1Driver.start_task` returned.  On
        today's driver this is ``{"status": "error", "content":
        [{"text": "start_task: provider registry not wired yet; ...
        "}]}`` after the FSM/battery gate admits, or the gate's own
        refusal envelope if it did not.  Once the provider registry
        lands the same call returns
        :meth:`~strands_robots.drivers.g1.G1Driver.run_policy`'s
        start envelope (``{"status": "success", "content": [{"json":
        {...}}]}``) verbatim, because the driver's ``start_task``
        forwards to the same loop path.  The verb does not reshape
        either shape - a future field the driver adds reaches a
        caller the moment the driver writes it.
    """
    # The handle is a live Python object typed :class:`~typing.Any`
    # (see the module docstring's import-cycle note), so the tool
    # schema carries no signal that ``None`` or a robot *name* is
    # refused.  The shared ``live_handle_refusal`` guard is the one
    # implementation of that judgement for this package; it is keyed
    # on the accessor the verb reads, which for this verb is
    # ``start_task`` (a callable that runs the driver's provider
    # lookup and either spawns the control loop or refuses with a
    # registry-not-wired message) rather than the sensor verbs'
    # ``_snapshot``.  Returning its refusal envelope here rather
    # than raising keeps the four invariants every ``@tool`` handler
    # owes a caller (envelope not exception, names the verb, names
    # ``driver``, names the type on wrong-type inputs).
    refusal = live_handle_refusal(
        "g1_start_task",
        driver,
        accessor="start_task",
        reads=(
            "the verb requests a provider-driven task on the driver's "
            "own 500 Hz control loop and reads back either the loop's "
            "start envelope or the driver's registry-not-wired refusal"
        ),
        expected=(
            "a callable ``start_task(instruction, policy_port=..., "
            "policy_host=..., policy_provider=..., duration=..., "
            "**kwargs)`` returning the driver's envelope - pass the "
            "live G1Driver handle the orchestrator constructed"
        ),
    )
    if refusal is not None:
        return refusal

    return driver.start_task(
        instruction,
        policy_port=policy_port,
        policy_host=policy_host,
        policy_provider=policy_provider,
        duration=duration,
    )
