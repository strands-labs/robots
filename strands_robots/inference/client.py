"""Remote policy inference client (robot side, typically a CPU/edge host).

:class:`RemotePolicy` is a drop-in :class:`~strands_robots.policies.base.Policy`
that forwards every observation to a :class:`PolicyServer` over a WebSocket and
returns the action chunk the server computes. Because it satisfies the ``Policy``
ABC, it works anywhere a local policy does: ``sim.run_policy(policy_provider=...)``,
``sim.eval_policy(...)``, or a hardware control loop that calls
:func:`~strands_robots.policies.create_policy`.

Usage::

    from strands_robots import create_policy

    policy = create_policy("remote", endpoint="ws://gpu-box:8765")
    # or via smart string:
    policy = create_policy("ws://gpu-box:8765")

The client mirrors the server policy's introspection metadata
(``requires_images``, ``execution_horizon``, ``actions_per_step``,
``supports_rtc``) so the local runtime sizes chunks and skips camera rendering
exactly as it would for the real policy - and the Real-Time Chunking contract is
preserved end-to-end: the runner-counted ``rtc_observed_delay_steps`` is
forwarded on every request and applied to the wrapped policy before it blends
chunk seams server-side.

The connection is established lazily on first use so constructing the policy
does not require the server to already be up.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from strands_robots.inference import protocol
from strands_robots.policies.base import Policy, chunk_count_error, required_bodies_error
from strands_robots.utils import (
    dial_host_error,
    name_list_error,
    positive_finite_number_error,
    tcp_port_error,
)

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection

logger = logging.getLogger(__name__)

#: Default request/receive timeout (seconds). A remote VLA can take a while, so
#: this is generous; override via the ``request_timeout`` kwarg.
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_CONNECT_TIMEOUT = 10.0


#: Metadata fields the ``ready`` handshake advertises as a per-inference chunk
#: count. Both are consumed as slice bounds over the action chunk, so they share
#: :func:`~strands_robots.policies.base.chunk_count_error`'s domain with the
#: constructor parameters a locally-loaded checkpoint is held to.
_MIRRORED_CHUNK_COUNTS = ("execution_horizon", "actions_per_step")

#: Metadata fields advertised as a capability flag. JSON spells a boolean
#: ``true``/``false``, so anything else is a peer that does not speak this
#: protocol - and a non-empty string is TRUTHY, which is how a peer answering
#: ``"no"`` used to turn a capability ON.
_MIRRORED_FLAGS = ("requires_images", "supports_rtc")


def _metadata_refusal(metadata: Mapping[str, Any]) -> str | None:
    """Report why advertised policy metadata cannot be mirrored.

    The handshake is the one place a peer's own numbers become this policy's
    introspection answers, so it is where they have to be checked. Reading them
    unchecked is not merely lenient, it is silent: the chunk counts land behind
    :attr:`Policy.execution_horizon`'s ``max(1, int(...))``, which is documented
    on :func:`~strands_robots.policies.base.chunk_count_error` as "silently
    destructive" for exactly this reason - a count no consumer can execute
    becomes ``1``, so an advertised ``0`` turns a chunk-emitting remote policy
    into a single-step one and :meth:`Policy.is_chunk_emitting` reports
    ``False``, which takes the rollout off the async-RTC path with nothing said.

    Holding the wire to ``chunk_count_error``'s domain is what that function
    asks for: it exists so "the same chunk count cannot be refused by a local
    checkpoint and accepted by the server serving it", and this handshake is the
    place the server does the accepting.

    ``required_bodies`` is held to
    :func:`~strands_robots.policies.base.required_bodies_error` for the same
    reason, and it is the field where reading unchecked was least visible: the
    mirror used to KEEP the entries it could use and drop the rest, so a peer
    advertising ``["torso_link", 42]`` became a proxy declaring
    ``("torso_link",)`` - a declaration nobody made. The robot host then resolved
    that shorter set against its scene, merged poses for it, and reported a
    successful rollout, while the served tracker's second anchor link never
    arrived. Dropping a body name is not a smaller request; it is a pose the
    observation never carries, and the served policy reads ``base_quat`` - the
    pelvis - in its place. A declaration the local owner refuses by name has to
    be refused here too, which is the whole of what
    :func:`~strands_robots.policies.base.collect_required_bodies` means by "two
    surfaces ask it and must not disagree".

    A field the handshake omits is not refused - the client keeps its own
    default for it, which is what makes a peer advertising a subset of the
    metadata (an older server, a third-party implementation) still usable.

    Args:
        metadata: The ``metadata`` payload of a ``ready`` or ``reset`` reply.

    Returns:
        Refusal text naming the field and the value, or ``None`` when every
        advertised field is one this client can mirror.
    """
    for param in _MIRRORED_CHUNK_COUNTS:
        if param in metadata and (error := chunk_count_error(metadata[param], param, "the served policy")):
            return error
    for param in _MIRRORED_FLAGS:
        value = metadata.get(param, False)
        if param in metadata and not isinstance(value, bool):
            return (
                f"the served policy advertised {param}={value!r} ({type(value).__name__}), which is not a JSON boolean."
            )
    if "provider_name" in metadata and not isinstance(metadata["provider_name"], str):
        name = metadata["provider_name"]
        return f"the served policy advertised provider_name={name!r} ({type(name).__name__}), which is not a string."
    if "required_bodies" in metadata and (
        error := required_bodies_error(metadata["required_bodies"], "required_bodies", "the served policy")
    ):
        return error
    return None


class RemotePolicy(Policy):
    """Client-side policy that runs inference on a remote :class:`PolicyServer`.

    Args:
        endpoint: Full server URL, e.g. ``ws://gpu-box:8765``. When given it
            takes precedence over ``host``/``port``.
        host: Server host (used when ``endpoint`` is not given). Must be a bare
            hostname or IP literal a URI can carry - no ``/``, ``:``, scheme or
            credentials, and IPv6 bracketed (``"[::1]"``) - because it is
            interpolated into ``ws://<host>:<port>`` and the parse gives a
            delimiter to a later component, taking the validated ``port`` with
            it. Pass a full URL as ``endpoint`` instead. ``"0.0.0.0"`` reaches a
            server bound on every interface.
        port: Server port (used when ``endpoint`` is not given). Must be an
            ``int`` in ``[1, 65535]``: this client has to dial the port, so
            unlike :class:`~strands_robots.inference.PolicyServer` - which
            binds one - it cannot accept ``0``, the request for an ephemeral
            port that only the server side can make.
        connect_timeout: Seconds to wait for the WebSocket handshake. Only a
            positive finite number names a budget. The value is handed to
            ``open_timeout`` on ``connect`` and to the handshake ``recv``, where
            ``0``, a negative and ``True`` time out against a server that is
            running and reachable - reported below as a ``ConnectionError``
            naming the server, not the timeout - while ``nan`` and ``inf`` raise
            ``ValueError`` / ``OverflowError`` out of ``websockets`` itself.
        request_timeout: Seconds to wait for a reply to each request. Same
            domain and the same ``recv`` deadline as ``connect_timeout``.

    ``inf`` is refused rather than read as "no deadline": ``websockets`` raises
    ``OverflowError`` computing the deadline from it, so an unbounded wait is not
    something either of these knobs can currently express.

    Unrecognized kwargs are ignored (for forward-compatible ``policy_config``
    passthrough via :func:`~strands_robots.policies.create_policy`) but logged
    at WARNING, since a mistyped connection kwarg (e.g. ``uri=``) would
    otherwise leave the client silently connected to the default endpoint.

    Raises:
        ValueError: If ``host`` or ``port`` cannot address a server to dial, or if
            ``connect_timeout`` / ``request_timeout`` is not a positive finite
            number.
        ConnectionError: On first use, if the server cannot be reached.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        **ignored_kwargs: Any,
    ) -> None:
        # ``port`` is interpolated into the URI verbatim, and a WebSocket target
        # is only resolved on first use - so an unusable value is not refused by
        # the transport, it fails much later as an unreachable server and
        # implicates the service the caller was trying to reach. Refuse it while
        # the caller still holds the value, before ``uri`` exists at all.
        # ``endpoint`` supersedes ``host``/``port``, so the port is validated
        # only when it is the effective spelling.
        # ``host`` is the other half of that same URI and is carried into it
        # verbatim, so it is graded on the same terms and refused first: a
        # delimiter in the host gives the path everything after it, ``port``
        # among it, and the parse then dials :80 - which makes the port's own
        # verdict unreadable rather than wrong.
        if not endpoint:
            if (host_error := dial_host_error(host, "host", type(self).__name__)) is not None:
                raise ValueError(host_error)
            if (port_error := tcp_port_error(port, "port", type(self).__name__)) is not None:
                raise ValueError(port_error)
        # A timeout that names no budget is refused here, while the caller still
        # holds the value, because the transport's own reaction to one is
        # indistinguishable from an absent server: ``0``, a negative and ``True``
        # all raise ``TimeoutError`` out of ``connect`` against a server that is
        # running, and ``_connect``'s ``except OSError`` covers it (a
        # ``TimeoutError`` is an ``OSError``) to report "could not
        # reach a PolicyServer ... Start one first" - pointing the operator at the
        # one thing that is not wrong. ``nan``, ``inf`` and a numeric string
        # instead escape that clause as a ``ValueError`` / ``OverflowError`` /
        # ``TypeError`` from inside ``websockets``, and since ``_connect`` is
        # reached lazily they land mid-rollout on the first ``predict``, naming no
        # parameter. Refused after ``port`` so the more specific "this address
        # cannot be dialled" still wins when both are wrong.
        for _param, _value in (("connect_timeout", connect_timeout), ("request_timeout", request_timeout)):
            if error := positive_finite_number_error(_value, _param, type(self).__name__):
                raise ValueError(error)
        self.uri = endpoint if endpoint else f"ws://{host}:{port}"
        if not self.uri.startswith(("ws://", "wss://")):
            self.uri = f"ws://{self.uri}"
        # A remote client forwards a shared policy_config superset (via
        # create_policy), so unrecognized kwargs are tolerated rather than
        # rejected - the cross-provider "ignore unknown constructor kwargs"
        # contract. But dropping them SILENTLY hides a connection
        # misconfiguration: the server endpoint is set via ``endpoint`` (with
        # a ``host``/``port`` fallback), and passing it under any other name
        # (e.g. ``uri=`` - also this object's own attribute name, an easy
        # slip) leaves the client silently pointed at the default
        # ws://127.0.0.1:8765, surfacing only as a confusing "connection
        # refused" to a port the user never chose. Warn so the endpoint
        # actually in use is visible (mirrors the #317 no-silent-localhost
        # -default fix for cosmos3:// URLs).
        if ignored_kwargs:
            logger.warning(
                "RemotePolicy ignoring unexpected constructor kwarg(s) %s; "
                "connecting to %s. Set the server endpoint via endpoint= "
                "(or host=/port=); server-side policy config belongs on the "
                "PolicyServer, not the client.",
                sorted(ignored_kwargs),
                self.uri,
            )
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout

        self._ws: ClientConnection | None = None
        self._lock = threading.Lock()

        # Config that may be set before the connection exists; flushed on connect.
        self._robot_state_keys: list[str] = []
        self._reset_pending: bool = False
        self._reset_seed: int | None = None

        # Mirrored server metadata (defaults until the ``ready`` handshake).
        self._remote_provider_name: str = "unknown"
        self._requires_images: bool = True
        self._execution_horizon: int = 1
        self.actions_per_step: int = 1
        self.supports_rtc: bool = False
        self._required_bodies: tuple[str, ...] = ()

    # -- connection lifecycle -------------------------------------------------

    def _connect(self) -> None:
        """Open the WebSocket, read the handshake, and flush pending config."""
        from websockets.sync.client import connect

        try:
            self._ws = connect(
                self.uri,
                open_timeout=self.connect_timeout,
                max_size=None,
                compression=None,
            )
        except OSError as exc:
            raise ConnectionError(
                f"RemotePolicy could not reach a PolicyServer at {self.uri}. "
                f"Start one first, e.g.:\n"
                f"  python -m strands_robots.inference.server --provider <name> "
                f"--host 0.0.0.0 --port {self.uri.rsplit(':', 1)[-1]}\n"
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc

        # Everything from here on runs with ``self._ws`` already live, so a
        # failure has to un-cache it: ``_ensure_connected`` short-circuits on a
        # non-``None`` ``self._ws``, and a connection whose handshake this
        # client *rejected* would otherwise be handed to the next request -
        # raising the mismatch once and then serving on it silently.
        established = False
        try:
            ready = protocol.loads(self._ws.recv(timeout=self.connect_timeout))
            if ready.get("type") != protocol.MSG_READY:
                raise ConnectionError(f"expected a '{protocol.MSG_READY}' handshake, got {ready.get('type')!r}")
            server_version = ready.get("protocol_version")
            if server_version != protocol.PROTOCOL_VERSION:
                raise ConnectionError(
                    f"protocol version mismatch: client speaks {protocol.PROTOCOL_VERSION}, "
                    f"server speaks {server_version}. Upgrade the older peer."
                )
            self._apply_metadata(ready.get("metadata", {}))
            logger.info("RemotePolicy connected to %s (remote provider=%s)", self.uri, self._remote_provider_name)

            # Replay config that was set before the connection existed.
            if self._robot_state_keys:
                self._request({"type": protocol.MSG_SET_STATE_KEYS, "keys": self._robot_state_keys})
            if self.control_frequency is not None:
                self._request({"type": protocol.MSG_SET_CONTROL_FREQUENCY, "hz": self.control_frequency})
            if self._reset_pending:
                reply = self._request({"type": protocol.MSG_RESET, "seed": self._reset_seed})
                self._apply_metadata(reply.get("metadata", {}))
                self._reset_pending = False
            established = True
        finally:
            if not established:
                self._discard_connection()

    def _ensure_connected(self) -> None:
        """Open the connection if it is not open yet, serialised with the other wire users.

        Holds ``self._lock`` across :meth:`_connect` because the connect
        sequence is itself a wire user, and the widest one: it reads the
        handshake and then replays up to three pending-config requests through
        :meth:`_request`, every one of them after ``self._ws`` is already live.
        Left unlocked, those sends interleave with a concurrent :meth:`reset`,
        :meth:`set_robot_state_keys` or :meth:`get_actions` on the same
        connection, and ``websockets`` refuses the overlapping read with a
        ``ConcurrencyError`` naming its own internals rather than anything the
        caller passed - a report no ``RemotePolicy`` caller can act on.

        Two threads on one policy is the ordinary case here, not a contrived
        one: every policy coroutine resolves through
        :mod:`strands_robots._async_utils`' reused worker thread, and the
        async-RTC path in :mod:`strands_robots.simulation.policy_runner`
        submits prefetch inference to its own ``rtc-prefetch`` worker while the
        rollout thread carries on stepping.

        The unlocked fast path is a benign double check. A thread that already
        sees a live connection has nothing to serialise against; one that sees
        ``None`` re-checks under the lock before connecting, so two racing
        first-callers open one connection rather than two.
        """
        if self._ws is not None:
            return
        with self._lock:
            # Re-checked under the lock: another thread may have connected
            # while this one waited, and a second connect would leak the first.
            if self._ws is None:
                self._connect()

    def _apply_metadata(self, metadata: dict[str, Any]) -> None:
        """Mirror the server policy's introspection metadata locally.

        Every advertised field is checked before ANY is applied, so a refusal
        leaves the mirror exactly as it was rather than half-updated with the
        fields that happened to be read before the offending one.

        The coercions this used to apply are gone with the check that replaces
        them: ``int()`` truncated an advertised ``8.9`` to ``8`` and parsed a
        ``"16"`` that no local checkpoint would be allowed to pass, ``bool()``
        turned the truthy string ``"no"`` into ``True``, and the
        ``required_bodies`` filter kept the entries it could use while dropping
        the rest, mirroring a declaration the peer never advertised.

        Args:
            metadata: The ``metadata`` payload of a ``ready`` handshake or of a
                ``reset`` reply, which re-advertises it once the server policy
                has firmed up.

        Raises:
            ConnectionError: If a field the peer advertised is not one this
                client can mirror, per :func:`_metadata_refusal`.
        """
        if not metadata:
            return
        if refusal := _metadata_refusal(metadata):
            raise ConnectionError(
                f"PolicyServer at {self.uri} advertised metadata this client cannot mirror: {refusal}"
            )
        self._remote_provider_name = metadata.get("provider_name", self._remote_provider_name)
        self._requires_images = metadata.get("requires_images", self._requires_images)
        self.actions_per_step = metadata.get("actions_per_step", self.actions_per_step)
        self.supports_rtc = metadata.get("supports_rtc", self.supports_rtc)
        self._execution_horizon = metadata.get("execution_horizon", self._execution_horizon)
        # A JSON array of names, applied verbatim: the refusal above already held
        # it to the domain the robot host's runtime validates, so there is
        # nothing left to coerce and no entry to drop. Mirroring it exactly is
        # what makes this proxy the declaring class for the set the peer really
        # advertised - see ``collect_required_bodies``.
        if "required_bodies" in metadata:
            self._required_bodies = tuple(metadata["required_bodies"])

    def close(self) -> None:
        """Close the WebSocket connection. Safe to call more than once."""
        with self._lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                finally:
                    self._ws = None

    # -- wire helpers (call while holding ``self._lock``) ---------------------

    def _discard_connection(self) -> None:
        """Drop the connection, without taking ``self._lock``.

        Every caller is a wire helper that already holds it, and :meth:`close`
        takes the same lock - routing through ``close`` from here would deadlock
        on a plain ``Lock``. Clearing ``self._ws`` first means the next
        :meth:`_ensure_connected` reconnects even if the close itself fails.
        """
        ws, self._ws = self._ws, None
        if ws is None:
            return
        try:
            ws.close()
        except OSError as exc:
            # The connection is being abandoned either way; a close that fails
            # on an already-broken socket is not news the caller can act on.
            logger.debug("RemotePolicy: closing an abandoned connection raised %s", exc)

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a message and return the decoded reply, raising on server error.

        Discards the connection unless the exchange completes. A request whose
        reply never arrived leaves that reply queued on the socket, so the next
        request reads the previous one's answer - and keeps reading one behind
        for the life of the connection. Neither the reply's own type nor the
        callers can see it: a stale action chunk carries the very
        ``MSG_ACTIONS`` type this call expects, and a ``MSG_OK`` read as an
        action chunk yields ``[]`` through
        :meth:`_get_actions_blocking`'s ``reply.get("actions", [])``, which is
        a legitimate "the policy emitted nothing". A robot then executes a
        chunk computed for an observation it has already moved past.

        A ``MSG_ERROR`` reply is deliberately *not* a failed exchange: the
        server marshals any dispatch failure back on this connection and
        carries on serving, so the stream stays in step and the connection is
        still good for the next request.

        The bookkeeping is a ``finally`` rather than an ``except`` so a
        ``BaseException`` - a cancellation between the send and the receive
        leaves the same undelivered reply behind - discards the connection too.
        """
        if self._ws is None:
            # Reachable from a caller that does not re-check under the lock: a
            # sibling's failed exchange discards the connection, so a caller
            # holding a reference from before that arrives with nothing to send
            # on. Naming the condition is what a bare assert could not do - and
            # under ``python -O`` there is no assert at all, leaving
            # ``AttributeError: 'NoneType' object has no attribute 'send'``.
            raise ConnectionError("the connection was discarded after a failed exchange; retry to open a fresh one")
        exchanged = False
        try:
            self._ws.send(protocol.dumps(message))
            reply = protocol.loads(self._ws.recv(timeout=self.request_timeout))
            exchanged = True
        finally:
            if not exchanged:
                self._discard_connection()
        if reply.get("type") == protocol.MSG_ERROR:
            detail = reply.get("traceback") or reply.get("error", "unknown error")
            raise RuntimeError(f"remote policy server error:\n{detail}")
        return reply

    # -- Policy ABC -----------------------------------------------------------

    @property
    def provider_name(self) -> str:
        """Identify this policy as the ``remote`` provider (the client half).

        Always ``"remote"`` regardless of which policy runs on the server;
        read :attr:`remote_provider_name` for the server-side policy's own
        provider name.
        """
        return "remote"

    @property
    def remote_provider_name(self) -> str:
        """Provider name of the policy running on the server (for identification)."""
        return self._remote_provider_name

    @property
    def requires_images(self) -> bool:
        """Mirror the server policy: skip camera rendering when it does not need frames."""
        self._ensure_connected()
        return self._requires_images

    @property
    def execution_horizon(self) -> int:
        """Mirror the server policy's re-query interval so RTC/chunking stays correct."""
        self._ensure_connected()
        return max(1, self._execution_horizon)

    @property
    def required_bodies(self) -> tuple[str, ...]:
        """Mirror the served tree's declared bodies so the robot host supplies them.

        ``required_bodies`` is read on the machine driving the rollout, not on
        the inference host: the simulation runtime resolves the names once
        before the rollout, refuses one the scene does not contain, and merges
        each body's pose into every observation it sends. The policy that needs
        them is behind the wire, so unless this half declares them too both
        halves of that contract are skipped - the rollout reports success having
        never supplied the key, and a name the scene lacks is never reported.

        Empty until the ``ready`` handshake has been read, so this connects
        first, exactly as :attr:`requires_images` and :attr:`execution_horizon`
        do.
        """
        self._ensure_connected()
        return self._required_bodies

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Store the robot state keys and forward them to the server policy.

        Unlike the base no-op, the remote client mirrors the keys to the
        server over the open connection so the server-side policy maps
        observations onto the same joints; when not yet connected, they are
        replayed on the next connect handshake.

        Validated here rather than relying on the server: the ``list(...)``
        below would otherwise flatten a bare string into one name per character
        and put a well-formed - but wrong - list on the wire, where no
        server-side check could recognise it as a mis-typed parameter.

        Raises:
            ValueError: If ``robot_state_keys`` is not an ordered list of
                distinct non-blank names, per
                :func:`~strands_robots.utils.name_list_error`. A single name
                passed as a bare string is the mistake this catches: ``str`` is
                iterable per character, so it would bind one joint per letter.
        """
        if robot_state_keys and (
            error := name_list_error(robot_state_keys, "robot_state_keys", "set_robot_state_keys")
        ):
            raise ValueError(error)
        self._robot_state_keys = list(robot_state_keys)
        with self._lock:
            if self._ws is not None:
                self._request({"type": protocol.MSG_SET_STATE_KEYS, "keys": self._robot_state_keys})

    def set_control_frequency(self, hz: float) -> None:
        """Set the control rate and forward it to the server policy.

        Delegates to the base to validate ``hz > 0`` and store it locally,
        then mirrors the rate to the server (when connected) so latency-aware
        server policies (RTC) slice chunk seams against the real loop rate.
        """
        super().set_control_frequency(hz)  # validates hz > 0 and sets the attribute
        with self._lock:
            if self._ws is not None:
                self._request({"type": protocol.MSG_SET_CONTROL_FREQUENCY, "hz": self.control_frequency})

    def reset(self, seed: int | None = None) -> None:
        """Forward the per-episode reset to the server policy.

        Without this, seeding only the client leaves the server's per-episode
        state (diffusion RNG, RTC chunk seams) drifting across episodes and
        breaks reproducibility - the same failure mode as a local service-mode
        policy that does not forward ``reset``.
        """
        with self._lock:
            if self._ws is None:
                self._reset_pending = True
                self._reset_seed = seed
                return
            reply = self._request({"type": protocol.MSG_RESET, "seed": seed})
            try:
                self._apply_metadata(reply.get("metadata", {}))
            except ConnectionError:
                # Same rule the handshake follows: a connection whose metadata
                # this client rejected must not be handed to the next request,
                # or the refusal is raised once and then served on silently.
                self._discard_connection()
                raise

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Forward the observation to the server and return the action chunk.

        The blocking WebSocket round-trip runs in a worker thread so this
        coroutine does not stall the event loop.
        """
        return await asyncio.to_thread(self._get_actions_blocking, observation_dict, instruction, kwargs)

    def _get_actions_blocking(
        self, observation_dict: dict[str, Any], instruction: str, kwargs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        self._ensure_connected()
        with self._lock:
            # Re-check: another thread may have discarded the connection (via
            # _discard_connection after a failed exchange) while this thread was
            # past _ensure_connected's unlocked fast path and waiting on the lock.
            if self._ws is None:
                self._connect()
            reply = self._request(
                {
                    "type": protocol.MSG_GET_ACTIONS,
                    "observation": observation_dict,
                    "instruction": instruction,
                    # Forwarded so the server applies the runner-counted RTC
                    # delay before inference - deterministic chunk seams across
                    # the wire (see Policy.set_rtc_observed_delay).
                    "rtc_observed_delay_steps": self.rtc_observed_delay_steps,
                    "kwargs": kwargs,
                }
            )
        actions = reply.get("actions", [])
        if not isinstance(actions, list):
            raise RuntimeError(f"server returned a non-list action chunk: {type(actions).__name__}")
        return actions
