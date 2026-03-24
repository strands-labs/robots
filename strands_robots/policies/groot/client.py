"""GR00T inference client — ZMQ client for inference-service communication.

Handles serialization of numpy arrays and ModalityConfig objects over ZMQ
using msgpack with custom encode/decode hooks.
"""

import io
import json
import logging
from typing import Any

import numpy as np

from strands_robots.utils import require_optional

from .data_config import ModalityConfig

logger = logging.getLogger(__name__)


def _load_zmq():
    """Load ZMQ dependency."""
    return require_optional("zmq", pip_install="pyzmq", extra="groot-service", purpose="GR00T service inference")


def _load_msgpack():
    """Load msgpack dependency."""
    return require_optional("msgpack", extra="groot-service", purpose="GR00T service inference")


class MsgSerializer:
    """(De)serialization helpers for ZMQ communication with GR00T services.

    Handles numpy ndarray and ModalityConfig types that cannot be directly
    serialized by msgpack.
    """

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        msgpack = _load_msgpack()
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        msgpack = _load_msgpack()
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)

    @staticmethod
    def _decode(obj):
        """Decode custom types from msgpack wire format."""
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return ModalityConfig(**json.loads(obj["as_json"]))
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def _encode(obj):
        """Encode custom types to msgpack wire format."""
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": obj.model_dump_json()}
        if isinstance(obj, np.ndarray):
            buffer = io.BytesIO()
            np.save(buffer, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": buffer.getvalue()}
        return obj


class Gr00tInferenceClient:
    """ZMQ REQ client for GR00T inference services.

    Handles socket lifecycle, timeout, and optional API-token authentication.

    Args:
        host: Server hostname or IP.
        port: Server port.
        timeout_ms: Socket timeout in milliseconds.
        api_token: Optional token included in every request for authentication.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str | None = None,
    ):
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
        logger.debug("Gr00tInferenceClient initialized: %s:%s (timeout=%dms)", host, port, timeout_ms)

    def _init_socket(self):
        """Create and connect the ZMQ REQ socket."""
        self.socket = self.context.socket(self._zmq.REQ)
        self.socket.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def reconnect(self):
        """Close and re-create the socket connection."""
        logger.info("Reconnecting to %s:%s", self.host, self.port)
        try:
            self.socket.close()
        except Exception:
            pass
        self._init_socket()

    def ping(self) -> bool:
        """Check server connectivity.

        Returns True if the server responds, False otherwise.
        Does NOT auto-reconnect — call :meth:`reconnect` explicitly if needed.
        """
        try:
            self.call_endpoint("ping")
            return True
        except Exception as exc:
            logger.debug("Ping failed: %s", exc)
            return False

    def call_endpoint(self, endpoint: str, data: dict | None = None) -> dict:
        """Send a request to the server and return the parsed response.

        Args:
            endpoint: Server endpoint name (e.g. "ping", "get_action").
            data: Optional request payload.

        Returns:
            Parsed response dict from the server.

        Raises:
            RuntimeError: If the server returns an error response.
        """
        request: dict = {"endpoint": endpoint}
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

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        """Send observations and receive an action chunk."""
        return self.call_endpoint("get_action", observations)

    def __del__(self):
        if hasattr(self, "socket"):
            self.socket.close()
        if hasattr(self, "context"):
            self.context.term()


__all__ = [
    "Gr00tInferenceClient",
    "MsgSerializer",
]
