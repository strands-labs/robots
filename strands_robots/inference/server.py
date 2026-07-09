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

if TYPE_CHECKING:
    from websockets.sync.server import Server, ServerConnection

logger = logging.getLogger(__name__)


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
            :attr:`port` after :meth:`start`).

    Raises:
        ValueError: If neither or both of ``policy`` / ``policy_provider`` are
            given.
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
        # Serialize inference so per-episode policy state is never interleaved
        # across connections (v1 single-client contract).
        self._lock = threading.Lock()

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
                self.policy.set_robot_state_keys(list(message.get("keys", [])))
            return {"type": protocol.MSG_OK}

        if msg_type == protocol.MSG_SET_CONTROL_FREQUENCY:
            with self._lock:
                self.policy.set_control_frequency(float(message["hz"]))
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

        from websockets.sync.server import serve

        # Match the client's connect() options: an observation carrying camera
        # frames is large (a single 640x480 RGB frame base64-encodes to ~1.2 MiB,
        # and a multi-camera VLA observation is several MiB), so the default 1 MiB
        # frame limit must be lifted or the server 1009-closes every real image
        # observation. Compression is disabled too (base64 binary barely compresses
        # and deflate wastes CPU at control rate); the client already opts out.
        self._server = serve(self._handle, self.host, self.port, max_size=None, compression=None)
        # Read back the OS-assigned port when the caller passed 0.
        self.port = self._server.socket.getsockname()[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"policy-server-{self.port}",
            daemon=True,
        )
        self._thread.start()
        logger.info("PolicyServer serving on ws://%s:%d", self.host, self.port)
        return self

    def stop(self) -> None:
        """Stop the background server and join its thread. Safe to call twice."""
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
            self._server = server
            self.port = server.socket.getsockname()[1]
            logger.info("PolicyServer serving on ws://%s:%d", self.host, self.port)
            try:
                server.serve_forever()
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
    parser.add_argument("--port", type=int, default=8765, help="Bind port (default: 8765).")
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        parser.error(f"--port must be between 1 and 65535, got {args.port}")

    logging.basicConfig(level=logging.INFO)
    PolicyServer(policy_provider=args.provider, host=args.host, port=args.port).serve()


if __name__ == "__main__":
    main()
