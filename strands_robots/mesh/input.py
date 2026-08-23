"""Input device streaming over the mesh - publish and receive teleoperator actions.

Enables remote teleoperation: a leader arm on machine A publishes its joint
positions via :class:`InputPublisher`, and the follower arm on machine B
receives and applies them via :class:`InputReceiver`.

Topic schema for ``strands/{peer_id}/input/{device_name}``::

    {
        "peer_id": "<publisher-peer-id>",
        "device": "<device-name>",
        "method": "arm" | "gamepad" | "keyboard" | "phone",
        "t": <unix-timestamp>,
        "seq": <monotonic-frame-counter>,
        "action": {"motor.pos": float, ...},
        "events": {"terminate_episode": bool, ...} | null
    }

``events`` is ``null`` both when the teleoperator exposes no
``get_teleop_events()`` surface and when reading it failed, so the publisher
side reports a failed read through ``InputPublisher.stats``
(``event_read_errors``) and a log line rather than only on the wire.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from strands_robots.bus_access import write_action
from strands_robots.mesh.pacing import Ticker
from strands_robots.mesh.security import (
    ValidationError,
    input_frame_slew_violation,
    merge_slew_baseline,
    validate_input_frame,
    validate_mesh_identifier,
)
from strands_robots.mesh.session import hz_from_env
from strands_robots.utils import partial_construction_repr, positive_finite_number_error

_log_safety_event: Callable[..., None] | None
try:  # audit is best-effort; never let an import issue break teleop apply
    from strands_robots.mesh.audit import log_safety_event as _log_safety_event
except Exception:  # pragma: no cover - defensive
    _log_safety_event = None

if TYPE_CHECKING:
    from strands_robots.mesh.core import Mesh

logger = logging.getLogger(__name__)

INPUT_HZ_DEFAULT = 50.0

#: How many times each :class:`InputPublisher` loop failure is logged before
#: going quiet. The loop runs at ``hz`` (50 Hz by default), so an unbounded
#: log of a persistent fault - a dead publish transport, a teleoperator whose
#: event surface stopped answering - floods the operator's console at the
#: control rate. The counters in :attr:`InputPublisher.stats` stay exact.
_MAX_LOGGED_LOOP_ERRORS = 5

#: Default ceiling on the rate at which an InputReceiver will
#: APPLY inbound teleop frames to the robot. The publisher streams at
#: INPUT_HZ_DEFAULT (50Hz); a malicious peer can stream far faster to slam
#: the servos (overcurrent / thermal / gear-strip). Frames arriving faster
#: than this cap are dropped-and-counted (``_rate_dropped``). Generous 2x
#: headroom over the default publish rate so legitimate jitter is never
#: rejected. Operator-tunable via ``STRANDS_MESH_INPUT_MAX_HZ`` (0 disables
#: the cap for trusted closed networks).
INPUT_MAX_HZ_DEFAULT = 100.0


#: How many refused frames of ONE cause are logged before that cause goes quiet.
#: Spent per cause rather than per receiver: a cause that keeps refusing must not
#: silence the FIRST refusal of a different one, because the log line is the only
#: place a refusal states which value it refused and against which bound.
_REFUSAL_LOG_BUDGET = 5

#: The refusal causes that share the ``rejected`` total, in the order
#: :meth:`InputReceiver._on_input` checks them. Each is reported as
#: ``rejected_<cause>`` in :attr:`InputReceiver.stats` and spends its own share of
#: :data:`_REFUSAL_LOG_BUDGET`.
_REJECTION_CAUSES: tuple[str, ...] = ("lockout", "freshness", "invalid")


def _input_max_hz() -> float:
    """Resolve ``STRANDS_MESH_INPUT_MAX_HZ`` (lazy, restart-free).

    Bad / missing input falls back to the default ceiling; an explicit
    non-positive value (0) disables the cap for trusted closed networks.
    A non-finite override falls back too: ``inf`` makes the caller's
    ``1.0 / max_hz`` interval zero and ``nan`` makes ``max_hz > 0`` false, so
    either one would silently switch the ceiling off -- the opposite of what an
    operator raising a rate limit is asking for.

    This resolver is evaluated per applied frame in a 50 Hz-plus loop, so an
    unusable value is not logged here; it resolves to the default ceiling,
    which keeps the servos protected.
    """
    hz, reason = hz_from_env("STRANDS_MESH_INPUT_MAX_HZ")
    if hz is None or reason is not None or hz < 0:
        return INPUT_MAX_HZ_DEFAULT
    return hz  # 0 => disabled


#: M-5: the teleop input path is high-rate (up to 50Hz),
#: so we cannot audit every applied frame without flooding the log. Instead we
#: record one ``input_stream_applied`` audit event every N applied frames (a
#: heartbeat that proves the stream was live + actuating, for post-incident
#: forensics of the "Invisible Puppeteer" chain). Operator-tunable via
#: ``STRANDS_MESH_INPUT_AUDIT_EVERY`` (0 disables input audit entirely).
INPUT_AUDIT_EVERY_DEFAULT = 100


def _input_audit_every() -> int:
    """Resolve ``STRANDS_MESH_INPUT_AUDIT_EVERY`` (lazy, restart-free).

    Bad/missing input falls back to the default sampling interval; an
    explicit non-positive value disables input-stream auditing.
    """
    import os

    raw = os.getenv("STRANDS_MESH_INPUT_AUDIT_EVERY")
    if raw is None:
        return INPUT_AUDIT_EVERY_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return INPUT_AUDIT_EVERY_DEFAULT
    return val if val > 0 else 0


class InputPublisher:
    """Publishes teleoperator actions to the mesh at a fixed rate.

    Runs in a background thread, polling the teleoperator and publishing
    normalized action dicts.

    Raises:
        ValidationError: ``device_name`` is not a valid mesh identifier
            (see :func:`~strands_robots.mesh.security.validate_mesh_identifier`);
            it is interpolated into the published key expression.
    """

    def __init__(
        self,
        mesh: Mesh,
        teleoperator: Any,
        device_name: str = "leader",
        method: str = "arm",
        hz: float = INPUT_HZ_DEFAULT,
    ) -> None:
        """Bind a teleoperator to a mesh topic at a fixed publish rate.

        Args:
            mesh: Live mesh used as the single publish chokepoint.
            teleoperator: Any object exposing ``get_action() -> dict``.
            device_name: Input-stream name; becomes the last topic segment.
            method: Input-method label ("arm", "gamepad", "keyboard", "phone").
            hz: Publish rate. Must be a positive finite number - the loop
                period is ``1 / hz``, so ``0`` raises inside the background
                thread and a negative/``nan``/``inf`` rate leaves the loop
                unthrottled, flooding every subscribed peer.

        Raises:
            ValueError: If ``hz`` is not a positive finite number. Refusing at
                construction is what keeps the rate a contract:
                :meth:`_publish_loop` runs on a background thread, where the
                same mistake would surface as a dead publisher that still
                reports ``running``.
        """
        error = positive_finite_number_error(hz, "hz", "InputPublisher")
        if error:
            raise ValueError(error)
        self.mesh = mesh
        self.teleoperator = teleoperator
        # ``device_name`` is interpolated into this publisher's key expression
        # (see ``topic``), so it carries the same identifier discipline as the
        # wire ``teleop_receive`` surface -- a wildcard or an extra ``/``
        # segment here publishes actuator data to a key no receiver named.
        self.device_name = validate_mesh_identifier(device_name, "InputPublisher.device_name")
        self.method = method
        self.hz = hz
        self._running = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._seq = 0
        self._error_count = 0
        self._event_read_error_count = 0
        self._frame_count = 0
        self._start_mono = 0.0

    def __repr__(self) -> str:
        try:
            state = "running" if self._running else "stopped"
            return f"InputPublisher(device={self.device_name!r}, method={self.method!r}, {state})"
        except AttributeError:
            return partial_construction_repr(self)

    @property
    def topic(self) -> str:
        """Mesh key this publisher writes to:
        ``strands/{own_peer_id}/input/{device_name}``. A remote
        :class:`InputReceiver` subscribes to this exact key to mirror the
        actions locally.
        """
        return f"strands/{self.mesh.peer_id}/input/{self.device_name}"

    @property
    def stats(self) -> dict[str, Any]:
        """Live publishing counters: the target device/method, whether the
        loop is ``running``, cumulative ``frames`` published and ``errors``
        hit, ``event_read_errors`` (frames published with ``events: null``
        because ``get_teleop_events()`` raised, rather than because the
        teleoperator has no event surface), and the achieved vs. requested rate
        (``hz_actual`` / ``hz_target``).
        """
        elapsed = time.monotonic() - self._start_mono if self._start_mono else 0
        return {
            "device": self.device_name,
            "method": self.method,
            "running": self._running,
            "frames": self._frame_count,
            "errors": self._error_count,
            "event_read_errors": self._event_read_error_count,
            "hz_actual": self._frame_count / elapsed if elapsed > 0 else 0,
            "hz_target": self.hz,
        }

    def start(self) -> None:
        """Start the input publishing loop."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._start_mono = time.monotonic()
        self._thread = threading.Thread(
            target=self._publish_loop,
            name=f"mesh-input-{self.device_name}",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "[mesh] input publisher started: %s (%s @ %.0fHz)",
            self.device_name,
            self.method,
            self.hz,
        )

    def stop(self) -> dict[str, Any]:
        """Stop the input publishing loop and return stats."""
        if not self._running:
            return self.stats
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info(
            "[mesh] input publisher stopped: %s (%d frames)",
            self.device_name,
            self._frame_count,
        )
        return self.stats

    def _publish_loop(self) -> None:
        """Read the leader device and publish one input frame per tick.

        Paced by :class:`~strands_robots.mesh.pacing.Ticker`. This loop is one of
        two that already did the deadline arithmetic by hand -- it measured its own
        body with ``perf_counter`` and waited ``period - elapsed`` -- so unlike the
        state, camera and sensor loops it was already achieving its requested rate
        where the wait itself is honest. Two things change. The subtraction now has
        ONE owner instead of being duplicated in every loop that needs it, so two
        copies cannot drift; and the wait it fed is gone, which matters on a host
        that inflates ``Event.wait`` (see the module docstring) -- there this loop
        ran at ``1 / (period + penalty)`` no matter how good its arithmetic was.
        """
        # ``hz`` is validated in __init__, so the division is safe.
        period = 1.0 / float(self.hz)
        with Ticker(period, self._stop_event) as ticker:
            self._publish_ticks(ticker)

    def _publish_ticks(self, ticker: Ticker) -> None:
        """Run the publish loop until stopped, pacing on ``ticker``.

        Split out so :meth:`_publish_loop` owns the ticker's lifetime in a
        ``finally``: the selector and its self-pipe must be released even when a
        frame read raises out of the loop.

        Args:
            ticker: The ticker to pace on, owned by the caller.
        """
        while self._running and not self._stop_event.is_set():
            try:
                action = self.teleoperator.get_action()
                action_dict = self._normalize_action(action)

                events = None
                if hasattr(self.teleoperator, "get_teleop_events"):
                    try:
                        events = self.teleoperator.get_teleop_events()
                    except Exception as event_err:  # noqa: BLE001
                        # The operator's control signals (terminate_episode /
                        # success / rerecord_episode) are secondary to the joint
                        # stream, so a failure reading them must not stop the
                        # arm: letting it reach the handler below would drop the
                        # whole frame, action included. It must not be silent
                        # either - ``events: null`` is also what a teleoperator
                        # with no event surface publishes, so an unreported
                        # failure is indistinguishable from "the operator
                        # signalled nothing" while joint commands keep flowing.
                        self._event_read_error_count += 1
                        if self._event_read_error_count <= _MAX_LOGGED_LOOP_ERRORS:
                            logger.warning(
                                "[mesh] input teleop-event read failed (%s): %s - "
                                "publishing events=None, so operator signals are "
                                "not reaching subscribers",
                                self.device_name,
                                event_err,
                            )

                payload = {
                    "peer_id": self.mesh.peer_id,
                    "device": self.device_name,
                    "method": self.method,
                    "t": time.time(),
                    "seq": self._seq,
                    "action": action_dict,
                    "events": events,
                }
                # Route through Mesh.publish() -- the documented single
                # publish chokepoint -- so this teleop actuation stream is
                # covered by any audit/telemetry/compression hook landing
                # there, exactly like sensor/state/command publishers. The
                # receiver side already goes through self.mesh.subscribe().
                self.mesh.publish(self.topic, payload)
                self._seq += 1
                self._frame_count += 1
            except Exception as exc:
                self._error_count += 1
                if self._error_count <= _MAX_LOGGED_LOOP_ERRORS:
                    logger.warning("[mesh] input publish error (%s): %s", self.device_name, exc)

            if ticker.wait():
                break

    @staticmethod
    def _normalize_action(action: Any) -> dict[str, float]:
        """Convert action from any teleoperator format to a flat dict."""
        if isinstance(action, dict):
            result = {}
            for k, v in action.items():
                if hasattr(v, "item"):
                    result[k] = float(v.item())
                elif isinstance(v, (int, float)):
                    result[k] = float(v)
                else:
                    result[k] = float(v)
            return result
        elif hasattr(action, "tolist"):
            # A numpy/torch scalar or 0-d array returns a bare Python number
            # from ``tolist()`` (not a list); enumerating it raises
            # ``'float' object is not iterable``. Treat a non-list result as a
            # single-DOF scalar so a 1-DOF leader does not crash the stream.
            arr = action.tolist()
            if isinstance(arr, list):
                return {f"j{i}": float(v) for i, v in enumerate(arr)}
            return {"raw": float(arr)}
        else:
            return {"raw": float(action)}


class InputReceiver:
    """Subscribes to a remote peer's input stream and applies actions locally.

    Listens on ``strands/{source_peer_id}/input/{device_name}`` and calls
    ``robot.send_action(action)`` for each received frame.

    Raises:
        ValidationError: ``source_peer_id`` or ``device_name`` is not a valid
            mesh identifier (see
            :func:`~strands_robots.mesh.security.validate_mesh_identifier`).
            Both are interpolated into the subscribed key expression, where a
            Zenoh wildcard would widen this stream to every publishing peer.
    """

    def __init__(
        self,
        mesh: Mesh,
        robot: Any,
        source_peer_id: str,
        device_name: str = "leader",
        apply_fn: Callable[[Any, dict[str, float]], None] | None = None,
    ) -> None:
        self.mesh = mesh
        self.robot = robot
        # Source scoping is the only thing that makes this a point-to-point
        # stream: both identifiers are interpolated into the subscribed key
        # expression (see ``topic``). Zenoh reads ``*`` / ``**`` as wildcards,
        # so an unvalidated ``source_peer_id`` turns "follow this leader" into
        # "apply joint commands from any peer that publishes an input frame".
        # Same validator the wire ``teleop_receive`` path uses, so the accepted
        # domains cannot diverge.
        self.source_peer_id = validate_mesh_identifier(source_peer_id, "InputReceiver.source_peer_id")
        self.device_name = validate_mesh_identifier(device_name, "InputReceiver.device_name")
        self._apply_fn = apply_fn or self._default_apply
        self._running = False
        self._sub_name: str | None = None
        self._frame_count = 0
        self._error_count = 0
        self._last_seq = -1
        self._drops = 0
        self._rejected = 0
        # Per-cause breakdown of ``_rejected``. Every cause that shares the total
        # keeps its own count, so a report can say WHICH guard refused the stream
        # and each cause spends its own log budget.
        self._rejected_by_cause: dict[str, int] = dict.fromkeys(_REJECTION_CAUSES, 0)
        self._rate_dropped = 0
        self._slew_rejected = 0
        self._last_rate_gate_mono = 0.0
        # Baseline for the per-joint slew bound: for each joint, the last
        # value actually applied and when. Updated only on a successful apply,
        # so a refused frame never becomes the reference for the next one, and
        # merged rather than replaced, so a frame carrying a subset of the
        # joints cannot erase the baseline of the ones it omits.
        self._last_applied: dict[str, tuple[float, float]] = {}
        self._start_mono = 0.0

    def __repr__(self) -> str:
        try:
            state = "running" if self._running else "stopped"
            return f"InputReceiver(source={self.source_peer_id!r}, device={self.device_name!r}, {state})"
        except AttributeError:
            return partial_construction_repr(self)

    @property
    def topic(self) -> str:
        """Mesh key this receiver subscribes to:
        ``strands/{source_peer_id}/input/{device_name}`` - the stream the
        remote peer's :class:`InputPublisher` writes to.
        """
        return f"strands/{self.source_peer_id}/input/{self.device_name}"

    @property
    def stats(self) -> dict[str, Any]:
        """Live receive counters: the ``source`` peer and device, whether the
        subscription is ``running``, ``frames_received``, ``errors``, and the
        loss/back-pressure breakdown - out-of-order ``drops``, ``rejected``
        frames, and ``rate_dropped`` frames (shed to hold the apply-rate cap),
        and ``slew_rejected`` frames (refused for commanding a joint faster
        than the per-joint slew bound) -
        plus the achieved ``hz_actual``.

        ``rejected`` is the total of a breakdown that names which guard refused
        the frame, so a report does not have to recover the reason from the
        log: ``rejected_lockout`` (arrived during an E-stop lockout),
        ``rejected_freshness`` (the frame's ``t`` is missing, non-numeric, stale
        or too far in the future - the replay defence), and ``rejected_invalid``
        (``validate_input_frame`` refused the frame's shape or a value: too many
        keys, an illegal key, or a value that is non-scalar, non-numeric,
        non-finite or past the magnitude bound). ``rejected`` always equals
        their sum.
        """
        elapsed = time.monotonic() - self._start_mono if self._start_mono else 0
        return {
            "source": self.source_peer_id,
            "device": self.device_name,
            "running": self._running,
            "frames_received": self._frame_count,
            "errors": self._error_count,
            "drops": self._drops,
            "rejected": self._rejected,
            # Derived from the declared causes, so a cause added to the
            # vocabulary is reported without a second list to update here.
            **{f"rejected_{cause}": n for cause, n in self._rejected_by_cause.items()},
            "rate_dropped": self._rate_dropped,
            "slew_rejected": self._slew_rejected,
            "hz_actual": self._frame_count / elapsed if elapsed > 0 else 0,
        }

    def start(self) -> None:
        """Start receiving input actions from the remote peer."""
        if self._running:
            return
        self._running = True
        self._start_mono = time.monotonic()
        self._sub_name = self.mesh.subscribe(
            self.topic,
            callback=self._on_input,
            name=f"input:{self.source_peer_id}/{self.device_name}",
        )
        if self._sub_name:
            logger.info(
                "[mesh] input receiver started: %s from %s",
                self.device_name,
                self.source_peer_id,
            )
        else:
            logger.warning("[mesh] input receiver failed to subscribe: %s", self.topic)
            self._running = False

    def stop(self) -> dict[str, Any]:
        """Stop receiving and return stats."""
        if not self._running:
            return self.stats
        self._running = False
        if self._sub_name:
            self.mesh.unsubscribe(self._sub_name)
        logger.info(
            "[mesh] input receiver stopped: %d frames from %s",
            self._frame_count,
            self.source_peer_id,
        )
        return self.stats

    def _refuse(self, cause: str, message: str, *args: Any) -> None:
        """Count one refused frame under ``cause`` and log the first few of it.

        The one place refusals on this path are accounted for. ``cause`` is a
        member of :data:`_REJECTION_CAUSES`; the frame is counted under it and in
        the ``rejected`` total, and :data:`_REFUSAL_LOG_BUDGET` is spent per
        cause. Reasons this path refuses a frame outnumber the counters that
        carried them, so a stream refused for one reason used to exhaust the
        shared budget and leave the next reason -- stated nowhere else -- unlogged.

        Args:
            cause: Which guard refused the frame. Must be in
                :data:`_REJECTION_CAUSES`; an unknown cause raises ``KeyError``
                rather than being counted under a name no report enumerates.
            message: ``logger.warning`` format string for this refusal.
            *args: Interpolated into ``message`` only if the budget allows.
        """
        seen = self._rejected_by_cause[cause] + 1
        self._rejected_by_cause[cause] = seen
        self._rejected = getattr(self, "_rejected", 0) + 1
        if seen <= _REFUSAL_LOG_BUDGET:
            logger.warning(message, *args)

    def _on_input(self, topic: str, data: dict[str, Any]) -> None:
        if not self._running:
            return
        # E-stop lockout MUST gate the teleop input path the
        # same way it gates the command path (see Mesh._dispatch). Without
        # this check a LAN-adjacent peer could keep driving the follower's
        # joints via send_action() while an operator believes the robot is
        # safely locked out -- the "Safe Mode Illusion" / "Oscillation Kill"
        # exploit chains. The CMD path raises LockoutError; the input path is
        # a high-rate streaming loop, so we drop-and-count instead of raising
        # to avoid log/exception spam at 50Hz. Rejected frames are surfaced
        # via the ``rejected`` stat and a rate-limited warning.
        lockout = getattr(self.mesh, "_estop_lockout", None)
        if lockout is not None and lockout.is_set():
            self._refuse(
                "lockout",
                "[mesh] input frame rejected during E-stop lockout from %s",
                self.source_peer_id,
            )
            return

        # Cross-session teleop replay defence. The CMD path got
        # (sender, turn_id) replay dedup and presence got freshness,
        # but this streaming input path had NO freshness check: an attacker who
        # eavesdrops a teleop stream can store frames and replay them hours/days
        # later (different session/ZID, stale timestamps) and the follower
        # repeats the captured motion -- the rate cap (100Hz) and value bound
        # still pass because the replayed frames are legitimate-shaped.
        # Every frame already carries a wall-clock ``t`` (set by
        # InputPublisher._publish_loop), so we just have to CHECK it. We reuse
        # the same freshness/forward-skew env knobs as the resume/e-stop replay
        # defence so operators tune one set of clock-drift bounds for the
        # whole mesh. Frames with a missing/non-numeric ``t`` are rejected too
        # (the publisher always sets it; absence means malformed or a
        # hand-crafted replay envelope) -- identical posture to M-3. Dropped
        # frames are counted + rate-limited-logged, never raised, because this
        # is a 50Hz+ hot loop.
        from strands_robots.mesh.core import (
            _resume_forward_skew_s,
            _resume_freshness_window_s,
        )

        _frame_t = data.get("t")
        if not isinstance(_frame_t, (int, float)) or isinstance(_frame_t, bool):
            self._refuse(
                "freshness",
                "[mesh] input frame rejected (missing/invalid timestamp) from %s",
                self.source_peer_id,
            )
            return
        _age = time.time() - float(_frame_t)
        if _age > _resume_freshness_window_s() or _age < -_resume_forward_skew_s():
            self._refuse(
                "freshness",
                "[mesh] input frame rejected (stale/future t=%.1fs) from %s",
                _age,
                self.source_peer_id,
            )
            return

        try:
            action = data.get("action")
            if action is None:
                return
            seq = data.get("seq", 0)
            if self._last_seq >= 0 and seq > self._last_seq + 1:
                self._drops += seq - self._last_seq - 1
            self._last_seq = seq

            # Apply-rate ceiling. A peer streaming teleop far
            # above the nominal publish rate can slam servos into overcurrent
            # / thermal / gear damage. Enforce a minimum inter-apply interval
            # using a monotonic clock (immune to wall-clock/NTP skew). Frames
            # over the cap are dropped-and-counted (``_rate_dropped``) rather
            # than raising -- this is a 50Hz+ hot loop. 0 disables the cap.
            max_hz = _input_max_hz()
            if max_hz > 0:
                now_mono = time.perf_counter()
                min_interval = 1.0 / max_hz
                if self._last_rate_gate_mono and (now_mono - self._last_rate_gate_mono) < min_interval:
                    self._rate_dropped = getattr(self, "_rate_dropped", 0) + 1
                    if self._rate_dropped <= 5:
                        logger.warning(
                            "[mesh] input frame rate-limited from %s (> %.0fHz)",
                            self.source_peer_id,
                            max_hz,
                        )
                    return
                self._last_rate_gate_mono = now_mono
            # B-04 / F-02: validate the teleop frame before it reaches
            # send_action(). A LAN-adjacent peer that discovers this
            # source peer_id could otherwise drive the follower's joints
            # directly with unbounded / non-finite values. validate_input_frame
            # bounds key count, key charset, and clamps each value to a
            # finite magnitude. Rejected frames are counted + logged and
            # dropped (never applied) rather than crashing the receiver.
            try:
                safe_action = validate_input_frame(action)
            except ValidationError as verr:
                self._refuse(
                    "invalid",
                    "[mesh] input frame rejected from %s: %s",
                    self.source_peer_id,
                    verr,
                )
                return
            # Per-joint slew bound. The guards above bound each frame in
            # isolation - who sent it, how fresh it is, how densely frames
            # arrive, how large a value may be - but none bounds the distance
            # between consecutive commands for one joint. A stream inside every
            # one of those caps can still reverse a joint full-scale on every
            # frame (1.8 units at 50 Hz is 90 units/s, over an order of
            # magnitude past what the leader's own servos can travel), which is
            # the overcurrent / gear-strip trajectory the rate cap exists to
            # prevent in the time domain. Refuse-and-count, matching every
            # other guard on this path: clamping toward the commanded value
            # would silently alter an actuator command. Because the bound is a
            # speed measured per joint from that joint's last applied value,
            # the allowance grows while a joint is not moving, so a refused
            # stream resumes by itself once the commanded pose is reachable
            # safely - no resync handshake - and a joint that pauses while
            # others move is not over-refused when it starts moving again.
            # The baseline is per joint, and merged rather than replaced,
            # because the frame shape is the sender's choice: a stream that
            # interleaved single-joint frames could otherwise erase the
            # baseline of the joint it was about to reverse, arriving with no
            # reference every time and never tripping the bound at all.
            # Frames can reach this point faster than real time - a batched
            # or buffered delivery, or a legitimate burst on a network where
            # the operator disabled the apply-rate cap. The intermediate
            # commands of such a burst are superseded before an actuator can
            # act on them, so the interval charged to the move is floored at
            # the minimum inter-apply interval the rate cap guarantees; frames
            # spaced closer than that are the rate cap's business, which lets
            # the two guards compose instead of contradicting each other. With
            # the cap disabled there is no guaranteed spacing, so the nominal
            # publish period stands in.
            apply_mono = time.perf_counter()
            min_apply_interval = 1.0 / max_hz if max_hz > 0 else 1.0 / INPUT_HZ_DEFAULT
            slew_reason = input_frame_slew_violation(
                safe_action,
                self._last_applied,
                apply_mono,
                min_apply_interval,
            )
            if slew_reason is not None:
                self._slew_rejected += 1
                if self._slew_rejected <= 5:
                    logger.warning(
                        "[mesh] input frame refused from %s: %s",
                        self.source_peer_id,
                        slew_reason,
                    )
                return
            self._apply_fn(self.robot, safe_action)
            self._last_applied = merge_slew_baseline(self._last_applied, safe_action, apply_mono)
            self._frame_count += 1
            # M-5: sampled positive audit of the live teleop stream so a
            # successful remote actuation is not invisible to forensics.
            _audit_every = _input_audit_every()
            if _log_safety_event is not None and _audit_every > 0 and self._frame_count % _audit_every == 0:
                try:
                    _log_safety_event(
                        "input_stream_applied",
                        getattr(self.mesh, "peer_id", "?"),
                        {
                            "source": self.source_peer_id,
                            "device": self.device_name,
                            "frames": self._frame_count,
                        },
                    )
                except (TypeError, ValueError, OSError) as audit_exc:
                    logger.debug("[mesh] input audit unavailable: %s", audit_exc)
        except Exception as exc:
            self._error_count += 1
            if self._error_count <= 5:
                logger.warning("[mesh] input apply error: %s", exc)

    @staticmethod
    def _default_apply(robot: Any, action: dict[str, float]) -> None:
        """Default: calls robot.send_action() under the device's bus lock."""
        # The same lock the readers take: a write that interleaves with a
        # sync-read corrupts both halves of the exchange, and teleop moving an
        # arm while a probe reads its position is the common case.
        if hasattr(robot, "send_action"):
            write_action(robot, action)
        elif hasattr(robot, "robot") and hasattr(robot.robot, "send_action"):
            write_action(robot.robot, action)
