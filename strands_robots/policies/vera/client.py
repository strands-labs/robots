"""Self-contained WebSocket client for the VERA policy server.

Speaks VERA's websocket wire protocol (``vera/server/protocol/``) directly using
only ``websockets`` + ``msgpack`` + a vendored NumPy packer - **no dependency on
the ``vera`` package** at client time. This mirrors the ``cosmos3`` client: the
heavy model stack lives in the server subprocess; the client is import-light and
composes with any numpy version.

Wire contract (verified against ``vera.server.protocol.websocket_policy_client``
and ``vera.controller.run_mimicgen_eval.RemotePolicy``):

* On connect the server sends one msgpack metadata blob - the
  ``VeraServerConfig`` (``view_keys``, ``context_frames``, ``action_space``,
  ``action_dim``, ``action_horizon``, ``control_dt``, ``gripper_dim_index`` …).
* ``infer`` request = ``{"context_rgb": (T,H,W,3) uint8, "view_keys": [...],
  "view_widths": [...], "session_id": str, "prompt"?: str, "endpoint": "infer"}``.
* ``infer`` response = ``{"action": np.ndarray[H, D], ...}``.
* A *string* response is the error sentinel - raise.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# WAN forward passes are slow; keep the connection generous on open.
_OPEN_TIMEOUT_SECS = 600


class VeraWebsocketClient:
    """Lazy-connecting WebSocket client for ``vera.server.start_vera_server``.

    The connection (and the server metadata handshake) is established on the
    first :meth:`infer` / :meth:`get_server_metadata` call, so constructing a
    policy never requires the server to already be up.

    Args:
        host: Server hostname or IP.
        port: Server websocket port.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8800) -> None:
        self.host = host
        self.port = port
        self.uri = f"ws://{host}:{port}"
        self._ws: Any = None
        self._server_metadata: dict[str, Any] | None = None
        from . import _msgpack_numpy as _mnp  # vendored, numpy-agnostic

        self._mnp = _mnp
        self._packer = _mnp.Packer()

    # -- connection ---------------------------------------------------------

    def _server_hint(self) -> str:
        """Actionable hint for bringing the VERA policy server up."""
        return (
            f"Could not reach the VERA policy server at {self.uri}. "
            "Start it first (holds the GPU):\n"
            "  python -m vera.server.start_vera_server "
            f"--embodiment <pusht|mimicgen|allegro|droid> --port {self.port}\n"
            "Checkpoints download (~42 GB full / ~4 GB Wave-1):\n"
            "  hf download sizhe-lester-li/VERA --local-dir ./vera-ckpts\n"
            "Then set VERA_CKPT_ROOT to that directory."
        )

    def _ensure(self) -> Any:
        if self._ws is not None:
            return self._ws
        try:
            import websockets.sync.client as _wsc

            self._ws = _wsc.connect(
                self.uri,
                compression=None,
                max_size=None,
                open_timeout=_OPEN_TIMEOUT_SECS,
            )
            # Server sends its VeraServerConfig as the first message.
            self._server_metadata = self._mnp.unpackb(self._ws.recv())
        except OSError as e:
            raise ConnectionError(self._server_hint()) from e
        logger.info("VeraWebsocketClient connected to %s", self.uri)
        return self._ws

    def get_server_metadata(self) -> dict[str, Any]:
        """Return the ``VeraServerConfig`` dict the server sends on connect."""
        self._ensure()
        return dict(self._server_metadata or {})

    # -- endpoints ----------------------------------------------------------

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send one inference request; return the server response dict.

        Args:
            observation: Wire dict - must carry ``context_rgb`` (the rolling
                context window), ``view_keys``, ``view_widths`` and
                ``session_id``; ``prompt`` is optional (only when the server
                was launched with text conditioning).

        Returns:
            Response dict containing at least ``"action"`` - an ``[H, D]``
            NumPy array - the chunk the controller plays before refilling.
        """
        ws = self._ensure()
        msg = {**observation, "endpoint": "infer"}
        ws.send(self._packer.pack(msg))
        resp = ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"VERA inference server error:\n{resp}")
        return self._mnp.unpackb(resp)

    def reset(self, reset_info: dict[str, Any] | None = None) -> None:
        """Clear the server's per-episode state (history + KV caches)."""
        try:
            ws = self._ensure()
        except ConnectionError:
            # Reset is best-effort - never a correctness requirement (mirrors
            # Cosmos3WebsocketClient.reset / Gr00tPolicy.reset).
            return
        msg = {**(reset_info or {}), "endpoint": "reset"}
        ws.send(self._packer.pack(msg))
        resp = ws.recv()
        if isinstance(resp, str) and resp != "reset successful":
            raise RuntimeError(f"VERA reset server error:\n{resp}")

    def configure(self, params: dict[str, Any]) -> dict[str, Any]:
        """Live-tune runtime knobs (motion_plan_scale, sample_steps, guidance)
        without a model rebuild. Returns the server's ``{"applied": {...}}``."""
        ws = self._ensure()
        msg = {**params, "endpoint": "configure"}
        ws.send(self._packer.pack(msg))
        resp = ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"VERA configure server error:\n{resp}")
        return self._mnp.unpackb(resp)

    def close(self) -> None:
        """Close the underlying websocket if open (best-effort and idempotent)."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:  # noqa: BLE001 - close is best-effort
                pass
            self._ws = None
