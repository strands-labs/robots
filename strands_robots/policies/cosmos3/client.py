"""WebSocket client for the Cosmos 3 RoboLab policy server.

Cosmos Framework ships a ready-made policy server
(``cosmos_framework.scripts.action_policy_server_robolab``) that serves
``nvidia/Cosmos3-Nano-Policy-DROID`` over a msgpack + NumPy WebSocket
protocol. This module ships a self-contained client — **no
``openpi-client`` dependency** — using only ``websockets`` + ``msgpack``
plus a vendored NumPy packer (``_msgpack_numpy``).

Why no ``openpi-client``? It pins ``numpy<2.0``, which is mutually
exclusive with ``lerobot`` (``numpy>=2.0``). Speaking the wire protocol
directly lets a Cosmos 3 rollout run alongside LeRobot dataset recording
in the same venv.

Wire contract (verified against the server source):

* request  = observation dict, keys in the ``observation/...`` namespace
             plus a top-level ``prompt`` string.
* response = ``{"action": np.ndarray[T, D], "video"?: np.ndarray, ...}``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class _RawWebsocketTransport:
    """msgpack + NumPy wire client using ``websockets`` + a vendored packer.

    This is the *only* transport. It speaks the exact same wire protocol the
    Cosmos 3 / OpenPI ``WebsocketPolicyServer`` expects (connect → recv
    msgpack metadata → send packed obs → recv packed action) and has zero
    NumPy-version constraints, so it composes cleanly with ``lerobot``
    (``numpy>=2``).

    Requires only ``websockets`` and ``msgpack``.
    """

    def __init__(self, host: str, port: int, api_key: str | None = None):
        self.uri = f"ws://{host}:{port}"
        self.api_key = api_key
        self._ws: Any = None
        from . import _msgpack_numpy as _mnp  # vendored, numpy-agnostic

        self._mnp = _mnp
        self._packer = _mnp.Packer()

    def _ensure(self) -> Any:
        if self._ws is None:
            import websockets.sync.client as _wsc

            headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
            self._ws = _wsc.connect(self.uri, compression=None, max_size=None, additional_headers=headers)
            self._mnp.unpackb(self._ws.recv())  # server metadata handshake
        return self._ws

    def get_server_metadata(self) -> dict[str, Any]:
        self._ensure()
        return {}

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        ws = self._ensure()
        ws.send(self._packer.pack(observation))
        resp = ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"Error in inference server:\n{resp}")
        return self._mnp.unpackb(resp)

    def reset(self) -> None:
        pass


class Cosmos3WebsocketClient:
    """Self-contained WebSocket client for the Cosmos 3 policy server.

    Args:
        host: Server hostname or IP.
        port: Server WebSocket port.
        api_key: Optional bearer token forwarded to the server, when set.
        transport: Accepted for backwards compatibility. The only supported
            transport is the vendored raw msgpack+websockets packer; any
            value is treated as ``"raw"`` and a deprecation warning is
            logged for the legacy ``"openpi"`` / ``"auto"`` selectors.

    The connection is established lazily on the first :meth:`infer` (or
    :meth:`get_server_metadata`) call so constructing a policy does not
    require the server to already be up — matching ``Gr00tInferenceClient``.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        api_key: str | None = None,
        transport: str = "raw",
    ):
        self.host = host
        self.port = port
        self.api_key = api_key
        if transport not in (None, "", "raw"):
            logger.warning(
                "Cosmos3WebsocketClient(transport=%r) is deprecated — the "
                "openpi-client dependency has been removed and the only "
                "supported transport is the vendored raw msgpack+websockets "
                "packer. Treating as transport='raw'.",
                transport,
            )
        self.transport = "raw"
        self._client: Any = None

    def _server_hint(self) -> str:
        """Actionable hint for starting the Cosmos 3 RoboLab policy server."""
        return (
            f"Could not reach the Cosmos 3 policy server at ws://{self.host}:{self.port}. "
            "Start it first (holds the GPU) from a Cosmos Framework checkout:\n"
            "  uv sync --all-extras --group=cu130-train --group=policy-server\n"
            "  python -m cosmos_framework.scripts.action_policy_server_robolab \\\n"
            "    --checkpoint-path nvidia/Cosmos3-Nano-Policy-DROID --port "
            f"{self.port}\n"
            f"Then confirm it is up:  curl http://{self.host}:{self.port}/healthz"
        )

    def _ensure_client(self) -> Any:
        """Connect on first use (lazy)."""
        if self._client is not None:
            return self._client
        try:
            self._client = _RawWebsocketTransport(self.host, self.port, self.api_key)
        except (ConnectionRefusedError, OSError) as e:
            raise ConnectionError(self._server_hint()) from e
        logger.info(
            "Cosmos3WebsocketClient ready for ws://%s:%s (transport=raw)",
            self.host,
            self.port,
        )
        return self._client

    def get_server_metadata(self) -> dict[str, Any]:
        """Return the metadata dict the server sends on connect."""
        client = self._ensure_client()
        try:
            return client.get_server_metadata()
        except (ConnectionRefusedError, OSError) as e:
            raise ConnectionError(self._server_hint()) from e

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send an observation dict and return the server response.

        Args:
            observation: Observation dict in the Cosmos 3 / OpenPI-compatible
                wire schema. Must contain ``prompt`` and at least one image
                plus the required state keys for the served action space.

        Returns:
            Response dict containing at least ``"action"`` (an ``[T, D]``
            NumPy array) and optionally ``"video"`` / ``"server_timing"``.
        """
        client = self._ensure_client()
        try:
            return client.infer(observation)
        except (ConnectionRefusedError, OSError) as e:
            raise ConnectionError(self._server_hint()) from e

    def reset(self) -> None:
        """Best-effort per-episode reset hint to the server.

        The raw transport is stateless on the client side — reset is a
        soft hint, never a correctness requirement (mirrors
        ``Gr00tPolicy.reset``). Any failure is swallowed.
        """
        try:
            client = self._ensure_client()
            reset_fn = getattr(client, "reset", None)
            if callable(reset_fn):
                reset_fn()
        except Exception as e:  # noqa: BLE001 - reset is best-effort
            logger.info("Cosmos3WebsocketClient.reset best-effort failed: %s", e)
