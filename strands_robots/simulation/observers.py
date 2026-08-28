"""Read-only rollout events for :meth:`PolicyRunner.run`.

Why this is a second lane rather than a use of ``on_frame``
----------------------------------------------------------

``on_frame`` looks like the observation seam and is not one. It is owned by the
backend for the duration of a rollout: MuJoCo's hook raises
:class:`~strands_robots.simulation.policy_runner.CooperativeStop` so
``stop_policy`` can interrupt, appends the trajectory mirror, publishes mesh step
telemetry and drives the LeRobot dataset recorder; Isaac's and Newton's do the
recording half of the same job. There is exactly one of them -
:meth:`~strands_robots.simulation.base.SimEngine.run_policy` obtains it from
``_make_run_policy_hook`` and does not accept one from the caller - so a consumer
that supplied its own would not *add* observation, it would silently remove
cancellation and recording.

These events are therefore emitted beside that hook, and deliberately report the
three things the hook's own signature cannot carry:

* **what the backend answered** - ``send_action``'s per-key verdict, normalised
  to :data:`ACTION_RESOLUTIONS` so a consumer never parses a backend envelope;
* **how fresh the observation was** - open-loop chunk replay feeds one
  observation to a whole chunk, so ``observation_is_chunk_reused`` says which
  actions were taken from a stale snapshot;
* **what the legacy hook did** - including the step it aborted on, which the
  legacy step accounting excludes (see :class:`RunPolicyStep`).

The payload-ownership rule
--------------------------

``RunPolicyStep.observation`` and ``RunPolicyStep.action`` are **borrowed**: the
same objects the legacy hook received, not copies. Copying them per step would
put an image-sized deep copy on the control path of every rollout that enables
the lane, which is the opposite of what an observability lane should cost. So the
contract is placed on the consumer instead, and it is narrow:

* Treat both as **read-only**. A backend may reuse the same buffers next step,
  and the dataset recorder reads them after you do.
* Do not retain them past the call. Snapshot the few fields you need
  (synchronously, inside the callback) if you intend to hand them to a queue,
  a thread or a socket.

An event is dispatched **synchronously** on the rollout thread. That makes the
lane cheap and ordered, and it means a consumer that blocks, blocks the robot.
Exceptions are contained - a raise never changes the rollout's outcome, and never
reaches the ``on_frame`` consecutive-failure watchdog, which exists for a
recorder losing dataset frames (GH #117) rather than for a visualiser that cannot
draw - but containment is not isolation: this is telemetry, not a sandbox. Keep
the callback short and non-blocking.

Ordering guarantees
-------------------

``event_seq`` is dense and 0-based within one ``run_id``, so a gap is
observable. ``monotonic_ns`` is the ordering clock (a ``date -s`` or an NTP
correction cannot move it); ``utc_ns`` is derived from a single rollout anchor so
a wall-clock label never reorders the stream.

Scope
-----

Single-policy simulation rollouts through :meth:`PolicyRunner.run` and the
``n_episodes=1`` fast path of
:meth:`~strands_robots.simulation.base.SimEngine.run_policy`. ``eval_policy``,
``evaluate_benchmark``, ``run_multi_policy`` and hardware carry no observer yet -
they are separate loops with different step semantics, and claiming them here
would promise coverage this module does not have.

Example::

    from strands_robots.simulation.observers import RunPolicyStep

    def watch(event):
        if isinstance(event, RunPolicyStep) and event.action_resolution != "full":
            print(event.applied_action_index, event.unresolved_action_keys)

    sim.run_policy(robot_name="arm", observer=watch)
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ACTION_RESOLUTIONS",
    "LEGACY_HOOK_OUTCOMES",
    "STOPPED_REASONS",
    "RunPolicyEnded",
    "RunPolicyEvent",
    "RunPolicyObserver",
    "RunPolicyStarted",
    "RunPolicyStep",
    "SCHEMA_VERSION",
]

#: Version of the event shape. Bumped when a field changes meaning or is
#: removed, so a consumer can refuse a stream it does not understand instead of
#: reading a renamed field as a missing one.
SCHEMA_VERSION = 1

#: The complete set of :attr:`RunPolicyStep.action_resolution` values.
#:
#: ``"full"`` every emitted key drove an actuator. ``"partial"`` some did, and
#: the rollout continues - this is the silent-degradation case that a single
#: aggregate error count hides, because the robot *does* move. ``"none"`` the
#: policy emitted keys and none resolved, so the robot did not move at all.
#: ``"unknown"`` the backend reported an error with no per-key breakdown (a
#: missing world, a vector-length mismatch), so which keys landed is not
#: knowable from the envelope.
ACTION_RESOLUTIONS = ("full", "partial", "none", "unknown")

#: The complete set of :attr:`RunPolicyStep.legacy_hook_outcome` values.
#:
#: ``"absent"`` no ``on_frame`` was installed. ``"ok"`` it returned normally.
#: ``"cancelled"`` it raised ``CooperativeStop`` (the documented graceful stop).
#: ``"recording_error"`` it raised ``RecordingFrameError``, which is dataset
#: loss and fatal on its first occurrence. ``"error"`` it raised anything else,
#: which is tolerated per call and fatal only after the consecutive-failure
#: limit.
LEGACY_HOOK_OUTCOMES = ("absent", "ok", "cancelled", "recording_error", "error")

#: The complete set of :attr:`RunPolicyEnded.stopped_reason` values, matching the
#: ``stopped_reason`` field of the rollout's own result payload.
STOPPED_REASONS = ("budget", "predicate", "cancelled", "error")


@dataclass(frozen=True)
class RunPolicyStarted:
    """Opens a rollout. Emitted once, after setup, before the first observation.

    Emitted only for a rollout that actually begins: a request refused in
    pre-flight (a bad horizon, an unusable seed, a video path that cannot be
    opened) raises or returns before this event, so no lifecycle is opened and
    none has to be closed.

    Attributes:
        schema_version: :data:`SCHEMA_VERSION` at emission time.
        run_id: Identifies this rollout. Every event of one
            :meth:`PolicyRunner.run` call shares it; a multi-episode
            ``run_policy`` produces one ``run_id`` per episode, because each
            episode is its own runner call.
        event_seq: ``0`` - the first event of the rollout.
        monotonic_ns: Ordering clock, from :func:`time.monotonic_ns`.
        utc_ns: Wall-clock label for the same instant, derived from the rollout's
            single ``(utc, monotonic)`` anchor.
        robot_name: Robot being driven.
        policy: Class name of the driving policy (e.g. ``"MockPolicy"``).
        instruction: Natural-language instruction forwarded to the policy.
        control_frequency: Target Hz of the control loop.
        action_horizon: Max actions consumed per policy call, as requested. The
            effective chunk may be longer when the policy declares a larger
            ``actions_per_step``.
        total_steps: Step budget resolved for this rollout.
        async_rtc: Whether inference is overlapped with action execution. This
            is the resolved value, so a rollout that auto-detected a
            chunk-emitting policy reports ``True`` even though the caller passed
            ``None``.
    """

    schema_version: int
    run_id: str
    event_seq: int
    monotonic_ns: int
    utc_ns: int
    robot_name: str
    policy: str
    instruction: str
    control_frequency: float
    action_horizon: int
    total_steps: int
    async_rtc: bool


@dataclass(frozen=True)
class RunPolicyStep:
    """One physically applied action.

    Emitted after ``send_action`` has driven the world and after the legacy
    ``on_frame`` hook has run, whatever that hook did. That ordering is the
    reason this event exists in the shape it does: the hook runs *after* the
    action is applied, and ``step_count`` is incremented *after* the hook, so an
    action the hook aborts on - a cancellation, a lost dataset frame - has
    already moved the robot while being excluded from ``steps_used``, from the
    video cadence and from the resolution denominator. :attr:`applied_action_index`
    counts what the world did; :attr:`legacy_step_index` mirrors what the hook
    was told. They differ by one on exactly that final step, which is the step a
    user debugging a cancellation most needs to see.

    Attributes:
        schema_version: :data:`SCHEMA_VERSION` at emission time.
        run_id: The rollout this step belongs to.
        event_seq: Dense, monotonic position in the rollout's event stream.
        monotonic_ns: Ordering clock, sampled after the action was applied.
        utc_ns: Wall-clock label for the same instant.
        applied_action_index: 0-based count of actions physically applied,
            including this one. Dense across the whole rollout.
        legacy_step_index: The index this step's ``on_frame`` call received, or
            the index it would have received had a hook been installed.
        observation: **Borrowed** pre-action observation - the same object the
            legacy hook received. Read-only; do not retain past the call. See the
            module docstring.
        action: **Borrowed** action sent to the backend. Usually a
            ``dict[str, float]``; a numeric vector for policies that bind
            positionally. Same ownership rule as ``observation``.
        observation_is_chunk_reused: ``True`` when this action was replayed from
            an action chunk and ``observation`` is the chunk-start snapshot
            rather than the state this action was taken from. Open-loop chunk
            replay makes that the normal case for every action after the first,
            unless an active dataset recording forced a per-step refresh.
        action_resolution: One of :data:`ACTION_RESOLUTIONS`.
        applied_action_keys: Keys that drove an actuator this step.
        unresolved_action_keys: Keys the backend could not absorb. Empty on the
            success path and on ``"unknown"`` resolutions without a breakdown.
        elapsed_s: Seconds since the rollout's monotonic start.
        sim_time_s: Backend simulation clock after the action, when the backend
            exposes one cheaply; ``None`` otherwise. Never fetched at the cost of
            an extra backend call.
        legacy_hook_outcome: One of :data:`LEGACY_HOOK_OUTCOMES`.
    """

    schema_version: int
    run_id: str
    event_seq: int
    monotonic_ns: int
    utc_ns: int
    applied_action_index: int
    legacy_step_index: int
    observation: dict[str, Any]
    action: Any
    observation_is_chunk_reused: bool
    action_resolution: str
    applied_action_keys: tuple[str, ...]
    unresolved_action_keys: tuple[str, ...]
    elapsed_s: float
    sim_time_s: float | None
    legacy_hook_outcome: str


@dataclass(frozen=True)
class RunPolicyEnded:
    """Closes a rollout. Emitted once, if and only if :class:`RunPolicyStarted` was.

    Attempted on every exit path a started rollout can take - budget exhausted,
    predicate fired, cooperative stop, or any error - so a consumer can always
    pair an open with a close. The one thing it cannot survive is the process
    dying under it (``SIGKILL``, OOM, a consumer that blocks forever), which is
    why this lane is telemetry rather than a durable record.

    Attributes:
        schema_version: :data:`SCHEMA_VERSION` at emission time.
        run_id: The rollout being closed.
        event_seq: Final position in the rollout's event stream.
        monotonic_ns: Ordering clock at close.
        utc_ns: Wall-clock label for the same instant.
        outcome: ``"success"`` or ``"error"``, matching the rollout result's
            ``status``.
        stopped_reason: One of :data:`STOPPED_REASONS`, matching the result
            payload's own field.
        applied_actions: Total actions physically applied. Equals the number of
            :class:`RunPolicyStep` events emitted, and may exceed
            :attr:`legacy_steps_used` by one when the legacy hook aborted the
            final step.
        legacy_steps_used: The rollout's own ``steps_used`` / ``n_steps``.
        action_errors: Steps whose ``send_action`` reported an error, partial
            resolutions included.
        elapsed_s: Rollout duration, measured on the monotonic clock.
        error_type: Exception class name when ``outcome == "error"``, else
            ``None``.
        error_message: Exception message when ``outcome == "error"``, else
            ``None``. Truncated; the full traceback stays in the log.
        observer_failures: How many times this observer raised during the
            rollout, across all three event kinds. Non-zero means the stream you
            are reading has holes.
    """

    schema_version: int
    run_id: str
    event_seq: int
    monotonic_ns: int
    utc_ns: int
    outcome: str
    stopped_reason: str
    applied_actions: int
    legacy_steps_used: int
    action_errors: int
    elapsed_s: float
    error_type: str | None
    error_message: str | None
    observer_failures: int = field(default=0)


#: Union of everything the lane can deliver. Dispatch with ``isinstance``: the
#: three kinds carry different fields, and a consumer that only wants one should
#: not have to guess which.
RunPolicyEvent = RunPolicyStarted | RunPolicyStep | RunPolicyEnded

#: A rollout observer: any callable taking one :data:`RunPolicyEvent`.
#:
#: Spelled as a callable alias rather than a protocol class for the same reason
#: :data:`~strands_robots.simulation.policy_runner.OnFrame` is - a plain function
#: or a bound method is the common case, and neither should have to satisfy an
#: interface to watch a rollout. Called synchronously on the rollout thread; see
#: the module docstring for the ownership and blocking rules.
RunPolicyObserver = Callable[[RunPolicyEvent], None]
