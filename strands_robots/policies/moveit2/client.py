"""MoveIt2 ZMQ client - msgpack-encoded REQ/REP transport.

Mirrors the shape of :class:`~strands_robots.policies.groot.client.Gr00tInferenceClient`
so users familiar with the GR00T service-mode pattern can use the same mental
model. The only wire types are JSON-equivalent values plus 1-D / 2-D float
arrays (joint state and trajectory rows), so we keep msgpack handling
deliberately minimal - no custom ``__class__`` markers, no numpy probing on
the hot path.
"""

from __future__ import annotations

import logging
from typing import Any

from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)


def _load_zmq() -> Any:
    """Load ZMQ dependency."""
    return require_optional(
        "zmq",
        pip_install="pyzmq",
        extra="moveit2",
        purpose="MoveIt2 service inference",
    )


def _load_msgpack() -> Any:
    """Load msgpack dependency."""
    return require_optional(
        "msgpack",
        extra="moveit2",
        purpose="MoveIt2 service inference",
    )


class MsgSerializer:
    """(De)serialization helpers for ZMQ communication with the MoveIt2 sidecar.

    The wire format only contains JSON-shaped values (numbers, strings,
    bools, lists, and dicts); we therefore use plain msgpack with the
    default packer / unpacker and no custom hooks.
    """

    @staticmethod
    def to_bytes(data: dict[str, Any]) -> bytes:
        msgpack = _load_msgpack()
        # use_bin_type=True keeps str / bytes distinct on the wire
        # (matches msgpack >=1.0 default but explicit is better than
        # implicit for cross-language sidecars).
        return msgpack.packb(data, use_bin_type=True)

    @staticmethod
    def from_bytes(data: bytes) -> dict[str, Any]:
        msgpack = _load_msgpack()
        # raw=False decodes msgpack ``str`` types back to Python ``str``
        # (msgpack >=1.0 default); strict_map_key=False allows
        # numeric / bytes keys but we never emit those.
        return msgpack.unpackb(data, raw=False)


class MoveIt2InferenceClient:
    """ZMQ REQ client for the MoveIt2 sidecar.

    Args:
        host: Server hostname or IP. Default ``"127.0.0.1"`` - bind to
            loopback by default; users opt into network exposure.
        port: Server port.
        timeout_ms: Socket send/recv timeout in milliseconds.
        api_token: Optional token included in every request for
            authentication. When unset, no auth is sent. Sent in
            plaintext over TCP - use a TLS tunnel or SSH port-forward
            for non-localhost deployments (same caveat as
            :class:`~strands_robots.policies.groot.client.Gr00tInferenceClient`).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ) -> None:
        self._zmq = _load_zmq()
        self.context = self._zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.api_token = api_token

        if api_token and host not in ("localhost", "127.0.0.1", "::1"):
            logger.warning(
                "API token will be sent in plaintext over TCP to %s:%s. "
                "ZMQ does not encrypt traffic by default. Consider using a "
                "TLS tunnel or SSH port-forward for non-localhost deployments.",
                host,
                port,
            )

        self._init_socket()
        logger.debug(
            "MoveIt2InferenceClient initialised: %s:%s (timeout=%dms)",
            host,
            port,
            timeout_ms,
        )

    def _init_socket(self) -> None:
        """Create and connect the ZMQ REQ socket."""
        self.socket = self.context.socket(self._zmq.REQ)
        self.socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        # LINGER=0 so socket.close() / context.term() never block waiting to
        # flush undelivered requests to a dead sidecar. Without it the default
        # linger is infinite, so a queued request to an unreachable server
        # hangs teardown (and interpreter shutdown / GC of __del__) forever.
        self.socket.setsockopt(self._zmq.LINGER, 0)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def reconnect(self) -> None:
        """Close and re-create the socket connection."""
        logger.info("Reconnecting to %s:%s", self.host, self.port)
        try:
            self.socket.close()
        except Exception:  # noqa: BLE001 - socket close failures don't matter on reconnect
            pass
        self._init_socket()

    def ping(self) -> bool:
        """Check server connectivity. Returns True if the server responds."""
        try:
            self.call_endpoint("ping")
            return True
        except Exception as exc:  # noqa: BLE001 - any failure means "not reachable"
            logger.debug("Ping failed: %s", exc)
            return False

    def call_endpoint(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a request to the server and return the parsed response.

        Args:
            endpoint: Server endpoint name (``"ping"``, ``"plan"``, ``"reset"``).
            data: Optional request payload.

        Returns:
            Parsed response dict from the server.

        Raises:
            RuntimeError: If the server returns an ``error`` field.
        """
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)
        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def plan(
        self,
        joint_state: list[float] | None,
        planning_group: str,
        target_pose: list[float] | None = None,
        target_joints: dict[str, float] | None = None,
        world_update: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Request a plan from the sidecar.

        Args:
            joint_state: Current joint configuration (radians / metres).
                ``None`` lets the server use its own latest state estimate.
            planning_group: MoveIt2 planning-group name (e.g. ``"arm"``).
            target_pose: Cartesian goal ``[x, y, z, qw, qx, qy, qz]`` in
                the planning group's base frame. Mutually exclusive with
                ``target_joints`` (server enforces).
            target_joints: Joint-space goal keyed by joint name.
            world_update: Per-call world refresh for collision-aware
                planning (depth, mesh, ...). Server-defined schema.

        Returns:
            ``{"trajectory": [[t, q0, q1, ...], ...], "success": bool, "status": str}``.
        """
        payload: dict[str, Any] = {
            "joint_state": joint_state,
            "planning_group": planning_group,
        }
        if target_pose is not None:
            payload["target_pose"] = target_pose
        if target_joints is not None:
            payload["target_joints"] = target_joints
        if world_update is not None:
            payload["world_update"] = world_update
        return self.call_endpoint("plan", payload)

    def _teardown(self) -> None:
        """Best-effort ZMQ socket + context teardown.

        Extracted from ``__del__`` so the destructor stays one line - CodeQL
        flags non-trivial logic in ``__del__`` because exceptions raised
        during interpreter shutdown are swallowed silently and can mask
        resource leaks.

        The socket is created with ``LINGER=0`` (see ``_init_socket``) so
        ``close()`` discards any undelivered request immediately instead of
        blocking to flush it to a dead sidecar; ``term()`` then returns once
        the (now-closed) socket is gone. Without the zero linger, a request
        queued to an unreachable server would hang teardown - and the GC that
        drives ``__del__`` / interpreter shutdown - indefinitely.
        """
        try:
            if hasattr(self, "socket"):
                self.socket.close()
            if hasattr(self, "context"):
                self.context.term()
        except Exception:  # noqa: BLE001
            pass

    def __del__(self) -> None:
        self._teardown()


__all__ = [
    "MoveIt2InferenceClient",
    "MsgSerializer",
]
