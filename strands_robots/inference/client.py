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
from typing import TYPE_CHECKING, Any

from strands_robots.inference import protocol
from strands_robots.policies.base import Policy

if TYPE_CHECKING:
    from websockets.sync.client import ClientConnection

logger = logging.getLogger(__name__)

#: Default request/receive timeout (seconds). A remote VLA can take a while, so
#: this is generous; override via the ``request_timeout`` kwarg.
DEFAULT_REQUEST_TIMEOUT = 60.0
DEFAULT_CONNECT_TIMEOUT = 10.0


class RemotePolicy(Policy):
    """Client-side policy that runs inference on a remote :class:`PolicyServer`.

    Args:
        endpoint: Full server URL, e.g. ``ws://gpu-box:8765``. When given it
            takes precedence over ``host``/``port``.
        host: Server host (used when ``endpoint`` is not given).
        port: Server port (used when ``endpoint`` is not given).
        connect_timeout: Seconds to wait for the WebSocket handshake.
        request_timeout: Seconds to wait for a reply to each request.

    Unrecognized kwargs are ignored (for forward-compatible ``policy_config``
    passthrough via :func:`~strands_robots.policies.create_policy`) but logged
    at WARNING, since a mistyped connection kwarg (e.g. ``uri=``) would
    otherwise leave the client silently connected to the default endpoint.

    Raises:
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
        except (OSError, TimeoutError) as exc:
            raise ConnectionError(
                f"RemotePolicy could not reach a PolicyServer at {self.uri}. "
                f"Start one first, e.g.:\n"
                f"  python -m strands_robots.inference.server --provider <name> "
                f"--host 0.0.0.0 --port {self.uri.rsplit(':', 1)[-1]}\n"
                f"Underlying error: {type(exc).__name__}: {exc}"
            ) from exc

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

    def _ensure_connected(self) -> None:
        if self._ws is None:
            self._connect()

    def _apply_metadata(self, metadata: dict[str, Any]) -> None:
        """Mirror the server policy's introspection metadata locally."""
        if not metadata:
            return
        self._remote_provider_name = metadata.get("provider_name", self._remote_provider_name)
        self._requires_images = bool(metadata.get("requires_images", self._requires_images))
        self.actions_per_step = int(metadata.get("actions_per_step", self.actions_per_step))
        self.supports_rtc = bool(metadata.get("supports_rtc", self.supports_rtc))
        self._execution_horizon = int(metadata.get("execution_horizon", self._execution_horizon))

    def close(self) -> None:
        """Close the WebSocket connection. Safe to call more than once."""
        with self._lock:
            if self._ws is not None:
                try:
                    self._ws.close()
                finally:
                    self._ws = None

    # -- wire helpers (call while holding ``self._lock``) ---------------------

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        """Send a message and return the decoded reply, raising on server error."""
        assert self._ws is not None  # noqa: S101 - guarded by _ensure_connected
        self._ws.send(protocol.dumps(message))
        reply = protocol.loads(self._ws.recv(timeout=self.request_timeout))
        if reply.get("type") == protocol.MSG_ERROR:
            detail = reply.get("traceback") or reply.get("error", "unknown error")
            raise RuntimeError(f"remote policy server error:\n{detail}")
        return reply

    # -- Policy ABC -----------------------------------------------------------

    @property
    def provider_name(self) -> str:
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

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._robot_state_keys = list(robot_state_keys)
        with self._lock:
            if self._ws is not None:
                self._request({"type": protocol.MSG_SET_STATE_KEYS, "keys": self._robot_state_keys})

    def set_control_frequency(self, hz: float) -> None:
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
            self._apply_metadata(reply.get("metadata", {}))

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
