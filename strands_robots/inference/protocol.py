"""WS-JSON wire protocol for remote policy inference.

This module defines the message schema and the (de)serialization helpers used
by :class:`~strands_robots.inference.server.PolicyServer` and
:class:`~strands_robots.inference.client.RemotePolicy` to stream observations
to a remote inference host and receive action chunks back.

The transport is deliberately plain JSON over a WebSocket: it is trivially
portable (any language with a JSON + WebSocket client can drive the server) and
has no NumPy-version constraints, so a remote rollout composes cleanly with
``lerobot`` (``numpy>=2``) in the same environment. NumPy arrays that cannot be
represented natively in JSON (camera frames, the state vector) are wrapped in a
tagged envelope carrying the base64 raw buffer plus ``dtype`` and ``shape`` so
they round-trip byte-exact. Action chunks returned by a policy are already
JSON-native (``dict[str, float | list[float]]`` per the
:meth:`~strands_robots.policies.base.Policy.get_actions` contract) and pass
through untouched.

Message types (each message is one JSON object with a ``type`` field):

* ``ready`` (server -> client, once on connect): advertises the wrapped
  policy's introspection metadata so the client can mirror it. Payload key
  ``metadata`` -> ``provider_name``, ``requires_images``, ``actions_per_step``,
  ``supports_rtc``, ``execution_horizon``.
* ``set_state_keys`` (client -> server): forwards
  :meth:`Policy.set_robot_state_keys`. Payload key ``keys``.
* ``set_control_frequency`` (client -> server): forwards
  :meth:`Policy.set_control_frequency`. Payload key ``hz``.
* ``reset`` (client -> server): forwards :meth:`Policy.reset`. Payload key
  ``seed`` (int or null).
* ``get_actions`` (client -> server): one inference request. Payload keys
  ``observation`` (encoded), ``instruction``, ``rtc_observed_delay_steps``
  (int or null, applied server-side before inference to preserve the
  Real-Time Chunking contract across the wire) and ``kwargs``.
* ``actions`` (server -> client): the response to ``get_actions``. Payload key
  ``actions`` (a JSON list of action dicts).
* ``ok`` (server -> client): acknowledges a control message.
* ``error`` (server -> client): a failure. Payload keys ``error`` and
  optional ``traceback``. The client raises rather than silently substituting
  a zero action.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import numpy as np

#: Wire-protocol version. Bumped on any breaking change to the message schema.
#: The server advertises it in the ``ready`` handshake and the client refuses a
#: mismatch rather than mis-decoding a newer/older peer.
PROTOCOL_VERSION = 1

# Message-type tags.
MSG_READY = "ready"
MSG_SET_STATE_KEYS = "set_state_keys"
MSG_SET_CONTROL_FREQUENCY = "set_control_frequency"
MSG_RESET = "reset"
MSG_GET_ACTIONS = "get_actions"
MSG_ACTIONS = "actions"
MSG_OK = "ok"
MSG_ERROR = "error"

#: Envelope key marking a base64-encoded NumPy array in the JSON payload.
_NDARRAY_TAG = "__ndarray__"


def encode_ndarray(arr: np.ndarray) -> dict[str, Any]:
    """Wrap a NumPy array in a JSON-safe, byte-exact envelope.

    Args:
        arr: Array to encode. Copied to C-contiguous layout so the raw buffer
            matches the declared ``shape``/``dtype`` on decode.

    Returns:
        A dict with the base64 raw buffer plus ``dtype`` and ``shape``.
    """
    contiguous = np.ascontiguousarray(arr)
    return {
        _NDARRAY_TAG: base64.b64encode(contiguous.tobytes()).decode("ascii"),
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
    }


def decode_ndarray(envelope: dict[str, Any]) -> np.ndarray:
    """Reconstruct a NumPy array from an :func:`encode_ndarray` envelope.

    The reconstructed array is WRITABLE, matching what a locally-run policy
    receives: a freshly rendered observation (camera frame or state vector) is
    always writable, and a policy legitimately relies on that. VLA preprocessors
    routinely normalize the observation in place (``image /= 255``, joint-state
    rescaling) or hand it to ``torch.from_numpy`` (zero-copy, then mutated), both
    of which raise ``ValueError: output array is read-only`` or invoke undefined
    behavior on a read-only buffer. Because :func:`base64.b64decode` returns
    immutable ``bytes``, decoding through ``np.frombuffer`` directly would yield
    a read-only array and silently diverge from the local-inference contract; we
    wrap the decoded bytes in a mutable ``bytearray`` so the view is writable
    (one allocation, no extra copy of the array data, still byte-exact).

    Args:
        envelope: A dict produced by :func:`encode_ndarray`.

    Returns:
        The reconstructed writable array with the original ``dtype`` and
        ``shape``.
    """
    raw = bytearray(base64.b64decode(envelope[_NDARRAY_TAG]))
    dtype = np.dtype(envelope["dtype"])
    array = np.frombuffer(raw, dtype=dtype)
    return array.reshape(envelope["shape"])


def _encode(obj: Any) -> Any:
    """Recursively convert an object graph into a JSON-serializable form.

    NumPy arrays become tagged envelopes; NumPy scalars become python scalars;
    dicts/lists are walked; everything else passes through unchanged.
    """
    if isinstance(obj, np.ndarray):
        return encode_ndarray(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {key: _encode(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(value) for value in obj]
    return obj


def _decode(obj: Any) -> Any:
    """Inverse of :func:`_encode`: rebuild NumPy arrays from tagged envelopes."""
    if isinstance(obj, dict):
        if _NDARRAY_TAG in obj:
            return decode_ndarray(obj)
        return {key: _decode(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_decode(value) for value in obj]
    return obj


def dumps(message: dict[str, Any]) -> str:
    """Serialize a protocol message (with embedded NumPy arrays) to a JSON string."""
    return json.dumps(_encode(message))


def loads(text: str | bytes) -> dict[str, Any]:
    """Parse a protocol message JSON string, rebuilding embedded NumPy arrays.

    Args:
        text: The raw WebSocket frame (``str`` or ``bytes``).

    Returns:
        The decoded message dict.

    Raises:
        ValueError: If the frame is not a JSON object.
    """
    decoded = _decode(json.loads(text))
    if not isinstance(decoded, dict):
        raise ValueError(f"protocol message must be a JSON object, got {type(decoded).__name__}")
    return decoded
