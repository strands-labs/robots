"""GR00T Client Implementation — ZMQ client for inference service communication.

Dependencies (msgpack, zmq) are imported lazily so the module can be
imported even when they are not installed (local-mode does not need them).
"""

import io
import json
import logging
from typing import Any, Dict

import numpy as np

from .data_config import ModalityConfig

logger = logging.getLogger(__name__)

# Lazy imports
_zmq = None
_msgpack = None


def _ensure_deps():
    global _zmq, _msgpack
    if _zmq is None:
        try:
            import msgpack
            import zmq

            _zmq = zmq
            _msgpack = msgpack
        except ImportError as e:
            raise ImportError(
                "GR00T service client requires: pip install pyzmq msgpack"
            ) from e


class MsgSerializer:
    """Message serializer for ZMQ communication with GR00T inference service."""

    @staticmethod
    def to_bytes(data: dict) -> bytes:
        _ensure_deps()
        return _msgpack.packb(data, default=MsgSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> dict:
        _ensure_deps()
        return _msgpack.unpackb(data, object_hook=MsgSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if "__ModalityConfig_class__" in obj:
            obj = ModalityConfig(**json.loads(obj["as_json"]))
        if "__ndarray_class__" in obj:
            obj = np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, ModalityConfig):
            return {"__ModalityConfig_class__": True, "as_json": obj.model_dump_json()}
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class BaseInferenceClient:
    """Base client for communicating with GR00T inference services."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 15000,
        api_token: str = None,
    ):
        _ensure_deps()
        self.context = _zmq.Context()
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

    def _init_socket(self):
        self.socket = self.context.socket(_zmq.REQ)
        self.socket.setsockopt(_zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(_zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def ping(self) -> bool:
        try:
            self.call_endpoint("ping", requires_input=False)
            return True
        except Exception:
            self._init_socket()
            return False

    def call_endpoint(
        self, endpoint: str, data: dict | None = None, requires_input: bool = True
    ) -> dict:
        request: dict = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        if self.api_token:
            request["api_token"] = self.api_token
        self.socket.send(MsgSerializer.to_bytes(request))
        message = self.socket.recv()
        response = MsgSerializer.from_bytes(message)
        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        return response

    def __del__(self):
        if hasattr(self, "socket"):
            self.socket.close()
        if hasattr(self, "context"):
            self.context.term()


class ExternalRobotInferenceClient(BaseInferenceClient):
    """Client for GR00T inference services (ZMQ)."""

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        return self.call_endpoint("get_action", observations)


# Convenience alias
Gr00tClient = ExternalRobotInferenceClient

__all__ = [
    "ExternalRobotInferenceClient",
    "Gr00tClient",
    "BaseInferenceClient",
    "MsgSerializer",
]
