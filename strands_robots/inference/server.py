"""Remote policy inference server (host side, typically a GPU box).

:class:`PolicyServer` wraps ANY :class:`~strands_robots.policies.base.Policy`
and exposes it over a WebSocket so a resource-constrained robot host can stream
observations in and receive action chunks back - the client/server split that
lets an edge device drive a big VLA (pi0 / SmolVLA / MolmoAct2) running on
remote GPU at control rate.

Usage::

    from strands_robots.inference import PolicyServer

    # Wrap a provider by name (built via create_policy on the server):
    PolicyServer(policy_provider="lerobot/act_so101", host="0.0.0.0").serve()

    # Or wrap an already-loaded policy object:
    PolicyServer(policy=my_policy, port=8765).serve()

The server binds ``127.0.0.1`` by default; set ``host="0.0.0.0"`` explicitly to
accept connections from other machines (wrap the link in tailscale/wireguard for
production - transport auth/TLS is intentionally out of scope for v1).

Concurrency: v1 serves ONE client at a time. The wrapped policy holds
per-episode state (RTC chunk seams, diffusion RNG), so concurrent clients would
corrupt each other; an internal lock serializes inference across connections.
"""

from __future__ import annotations

import logging
import threading
import traceback
from typing import TYPE_CHECKING, Any

from strands_robots.inference import protocol
from strands_robots.policies.base import Policy
from strands_robots.utils import tcp_port_error

if TYPE_CHECKING:
    from websockets.sync.server import Server, ServerConnection

logger = logging.getLogger(__name__)


def _bind_port_error(value: Any, param: str, context: str) -> str | None:
    """Error text when ``value`` cannot be bound as a server port.

    :func:`~strands_robots.utils.tcp_port_error` owns the range and the scalar
    policy; this wrapper decides only the floor, because a server *binds* a
    port rather than dialing one and so may ask the kernel for an ephemeral
    one. :class:`PolicyServer` documents ``0`` as exactly that, and reads the
    assigned port back onto :attr:`PolicyServer.port` once bound, so a genuine
    ``int`` zero is accepted here and every other value is deferred - the bind
    and dial domains cannot drift apart on what counts as an addressable port.

    ``bool`` is deliberately not spelled in the zero test: ``False == 0``, so a
    bare ``value == 0`` would read ``False`` as the ephemeral request. The type
    identity is checked first, and a boolean falls through to
    :func:`~strands_robots.utils.tcp_port_error`, which refuses it for the same
    reason it refuses one for a port the client dials - otherwise ``True``
    would bind privileged port 1.

    Args:
        value: The caller-supplied value.
        param: The parameter name it came from, used in the message.
        context: Message prefix identifying the surface that received it.

    Returns:
        An error message, or ``None`` when the value can be bound.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value == 0:
        return None
    return tcp_port_error(value, param, context)


class PolicyServer:
    """Serve a :class:`Policy` over a WebSocket for remote inference.

    Args:
        policy: A pre-built policy to serve. Mutually exclusive with
            ``policy_provider``.
        policy_provider: Provider name or smart string built server-side via
            :func:`~strands_robots.policies.create_policy`. Mutually exclusive
            with ``policy``.
        policy_config: Extra kwargs forwarded to ``create_policy`` when
            ``policy_provider`` is used.
        host: Bind address. Defaults to ``127.0.0.1`` (loopback only); use
            ``0.0.0.0`` to accept remote connections.
        port: Bind port. ``0`` asks the OS for a free port (read back from
            :attr:`port` after :meth:`start`); any other value must be an
            ``int`` in ``[1, 65535]``.

    Raises:
        ValueError: If neither or both of ``policy`` / ``policy_provider`` are
            given, or if ``port`` cannot be bound.
    """

    def __init__(
        self,
        policy: Policy | None = None,
        *,
        policy_provider: str | None = None,
        policy_config: dict[str, Any] | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        if (policy is None) == (policy_provider is None):
            raise ValueError("provide exactly one of 'policy' or 'policy_provider'")

        # ``port`` is the port this server binds, so a value it cannot bind is
        # refused here rather than at ``serve()``: building the policy first can
        # download a checkpoint, and the port is only actionable at the point the
        # caller named it. ``0`` stays valid - it is the documented request for an
        # ephemeral port, which ``start()``/``serve()`` read back onto ``port``.
        if (port_error := _bind_port_error(port, "port", type(self).__name__)) is not None:
            raise ValueError(port_error)

        if policy is None:
            from strands_robots.policies import create_policy

            # Guaranteed by the exactly-one check above (policy is None here
            # implies policy_provider is not None); assert so mypy narrows.
            assert policy_provider is not None
            policy = create_policy(policy_provider, **(policy_config or {}))

        self.policy: Policy = policy
        self.host = host
        self.port = port
        self._server: Server | None = None
        self._thread: threading.Thread | None = None
        # Set by stop() before it closes the listening socket, so the serving
        # loop can tell a teardown apart from a genuine socket failure. See
        # _run_accept_loop.
        self._stopping = threading.Event()
        # Serialize inference so per-episode policy state is never interleaved
        # across connections (v1 single-client contract).
        self._lock = threading.Lock()

    def _run_accept_loop(self, server: Server) -> None:
        """Run the sync server's accept loop, absorbing the teardown race only.

        :meth:`stop` ends the loop by closing the listening socket from another
        thread, and the socket is exactly what the loop is waiting on. The
        websockets sync server does not synchronize those two, so a stop that
        lands while the loop is coming up raises out of ``serve_forever()``.

        Both shapes come from the *same* call - ``serve_forever()`` registering
        the listening socket with its selector - and which one surfaces depends
        on where in that call the close lands, not on the release. ``selectors``
        reads ``fileno()`` to build its key and then hands the descriptor to
        ``epoll_ctl``: a close landing before the read leaves ``fileno()`` at
        ``-1`` and raises ``ValueError: Invalid file descriptor: -1``, while a
        close landing between the read and ``epoll_ctl`` leaves a live-looking
        descriptor and raises ``OSError: [Errno 9] Bad file descriptor``. 12.0
        wraps that call in nothing, so both escape it; 13.0 through 17.x wrap it
        in ``except ValueError: return``, which absorbs the first shape and
        leaves the second - so raising the dependency floor narrows this race
        but does not close it.

        Left unhandled that kills the serving thread, and a daemon thread reports
        its death nowhere: ``stop()`` returns, :attr:`_server` is cleared, and the
        server looks cleanly shut down. So the loop is only ever run through
        here, and the exception is absorbed *only* while a stop is in progress -
        the same failure without a stop pending is a real one and still
        propagates, which is why this keys on :attr:`_stopping` rather than on
        the exception type. Keying on the type would both swallow a genuine
        socket error and encode a release-to-exception mapping that is already
        untrue: one call raises either shape depending on scheduling, and every
        supported release can produce the ``OSError``.

        Args:
            server: The running sync server whose accept loop to drive.
        """
        try:
            server.serve_forever()
        except (OSError, ValueError):
            if not self._stopping.is_set():
                raise
            logger.debug(
                "PolicyServer: accept loop ended while stopping (the listening socket was closed under it)",
                exc_info=True,
            )

    def _metadata(self) -> dict[str, Any]:
        """Introspection payload advertised in the ``ready`` handshake."""
        return {
            "provider_name": self.policy.provider_name,
            "requires_images": bool(self.policy.requires_images),
            "actions_per_step": int(getattr(self.policy, "actions_per_step", 1)),
            "supports_rtc": bool(getattr(self.policy, "supports_rtc", False)),
            "execution_horizon": int(self.policy.execution_horizon),
        }

    def _handle(self, websocket: ServerConnection) -> None:
        """Serve one client connection: handshake, then dispatch each message."""
        peer = getattr(websocket, "remote_address", None)
        logger.info("PolicyServer: client connected %s", peer)
        websocket.send(
            protocol.dumps(
                {
                    "type": protocol.MSG_READY,
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "metadata": self._metadata(),
                }
            )
        )
        for raw in websocket:
            try:
                message = protocol.loads(raw)
                reply = self._dispatch(message)
            except Exception as exc:  # noqa: BLE001 - marshal ANY failure back to the client
                logger.exception("PolicyServer: error handling message")
                reply = {
                    "type": protocol.MSG_ERROR,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            websocket.send(protocol.dumps(reply))
        logger.info("PolicyServer: client disconnected %s", peer)

    def _dispatch(self, message: dict[str, Any]) -> dict[str, Any]:
        """Route one decoded message to the wrapped policy and build the reply.

        Raises:
            ValueError: On an unknown message type.
        """
        msg_type = message.get("type")

        if msg_type == protocol.MSG_GET_ACTIONS:
            with self._lock:
                # Preserve the Real-Time Chunking contract across the wire: the
                # runner-supplied step count is applied to the wrapped policy
                # BEFORE inference, exactly as a local runner would.
                self.policy.set_rtc_observed_delay(message.get("rtc_observed_delay_steps"))
                actions = self.policy.get_actions_sync(
                    message.get("observation", {}),
                    message.get("instruction", ""),
                    **(message.get("kwargs") or {}),
                )
            return {"type": protocol.MSG_ACTIONS, "actions": actions}

        if msg_type == protocol.MSG_SET_STATE_KEYS:
            with self._lock:
                # Forwarded verbatim, for the same reason as ``hz`` below:
                # coercing here (``list(...)``) lets the wire through a value
                # the in-process API refuses. ``list("wrist")`` is
                # ``['w', 'r', 'i', 's', 't']`` - five distinct, non-blank
                # names that pass every shape check, so the coercion would
                # launder a mis-typed parameter into a well-formed joint list
                # naming one joint per letter. The policy owns the domain.
                self.policy.set_robot_state_keys(message.get("keys", []))
            return {"type": protocol.MSG_OK}

        if msg_type == protocol.MSG_SET_CONTROL_FREQUENCY:
            with self._lock:
                # Forwarded verbatim: coercing here (``float(...)``) would let
                # the wire accept a rate the in-process API refuses - a JSON
                # ``true`` becomes ``1.0`` and installs a silent 1 Hz clock,
                # and a quoted ``"50"`` becomes a rate no local caller could
                # have set. The policy owns the accepted domain.
                self.policy.set_control_frequency(message["hz"])
            return {"type": protocol.MSG_OK}

        if msg_type == protocol.MSG_RESET:
            with self._lock:
                self.policy.reset(seed=message.get("seed"))
            # Metadata (e.g. execution_horizon) can only firm up after the
            # first reset for some providers; re-advertise so the client stays
            # in sync without reconnecting.
            return {"type": protocol.MSG_OK, "metadata": self._metadata()}

        raise ValueError(f"unknown message type: {msg_type!r}")

    def start(self) -> PolicyServer:
        """Start serving in a background thread and return once bound.

        After this returns, :attr:`port` holds the actual bound port (useful
        when constructed with ``port=0``). Idempotent per instance: calling it
        twice raises.

        Returns:
            ``self``, so callers can chain ``PolicyServer(...).start()``.

        Raises:
            RuntimeError: If the server is already running.
        """
        if self._server is not None:
            raise RuntimeError("PolicyServer is already running")

        # A previous stop() left the flag set; this run's teardown has not begun.
        self._stopping.clear()

        from websockets.sync.server import serve

        # Match the client's connect() options: an observation carrying camera
        # frames is large (a single 640x480 RGB frame base64-encodes to ~1.2 MiB,
        # and a multi-camera VLA observation is several MiB), so the default 1 MiB
        # frame limit must be lifted or the server 1009-closes every real image
        # observation. Compression is disabled too (base64 binary barely compresses
        # and deflate wastes CPU at control rate); the client already opts out.
        server = serve(self._handle, self.host, self.port, max_size=None, compression=None)
        # Read back the OS-assigned port BEFORE publishing ``_server``: a
        # background caller polling ``_server is not None`` as the "bound"
        # signal must then always observe the real port, never the
        # constructor default (see serve() for the same ordering).
        self.port = server.socket.getsockname()[1]
        self._server = server
        self._thread = threading.Thread(
            target=self._run_accept_loop,
            args=(server,),
            name=f"policy-server-{self.port}",
            daemon=True,
        )
        self._thread.start()
        logger.info("PolicyServer serving on ws://%s:%d", self.host, self.port)
        return self

    def stop(self) -> None:
        """Stop the background server and join its thread. Safe to call twice."""
        # Before the close, not after: the serving thread races this and reads
        # the flag to tell a teardown apart from a real socket failure.
        self._stopping.set()
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def serve(self) -> None:
        """Serve in the foreground (blocking) until interrupted.

        Convenience entry point for a standalone server process. Blocks the
        calling thread; use :meth:`start`/:meth:`stop` for programmatic control.
        """
        from websockets.sync.server import serve

        # See start(): lift the default 1 MiB frame limit (and disable compression)
        # so large multi-camera observations stream in, matching the client.
        with serve(self._handle, self.host, self.port, max_size=None, compression=None) as server:
            # Publish the bound port BEFORE marking the server as bound so a
            # concurrent poller of ``_server is not None`` never reads the
            # constructor default 0 in the window between the two writes.
            self.port = server.socket.getsockname()[1]
            self._server = server
            logger.info("PolicyServer serving on ws://%s:%d", self.host, self.port)
            try:
                self._run_accept_loop(server)
            finally:
                self._server = None

    def __enter__(self) -> PolicyServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def main(argv: list[str] | None = None) -> None:
    """CLI: serve a policy provider over a WebSocket.

    Example::

        python -m strands_robots.inference.server \\
            --provider lerobot/act_so101 --host 0.0.0.0 --port 8765
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="strands_robots.inference.server",
        description="Serve a strands-robots policy for remote inference over WebSocket.",
    )
    parser.add_argument(
        "--provider",
        required=True,
        help="Policy provider name or smart string (e.g. 'mock', 'lerobot/act_so101').",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Bind port, or 0 to ask the OS for a free one (default: 8765).",
    )
    args = parser.parse_args(argv)

    # Same bind domain as the constructor, so the CLI cannot refuse a port the
    # class it constructs accepts. The previous inline range did exactly that
    # for ``--port 0``, the documented ephemeral bind.
    if (port_error := _bind_port_error(args.port, "--port", parser.prog)) is not None:
        parser.error(port_error)

    logging.basicConfig(level=logging.INFO)
    PolicyServer(policy_provider=args.provider, host=args.host, port=args.port).serve()


if __name__ == "__main__":
    main()
