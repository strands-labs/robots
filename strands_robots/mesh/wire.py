"""What travels between peers: one versioned envelope, and the one table of keys it travels on.

An envelope is ``{v, tenant, device_id, seq, t, kind, name, args|result|error}``, encoded as JSON. Arrays travel
as ``{"__nd__": [dtype, shape, base64]}`` so images and joint vectors survive the trip bit-exact; numpy scalars
become plain numbers. A peer that receives an envelope of another version refuses it rather than guess.
"""

from __future__ import annotations

import base64
import json
import re
import time
from typing import Any

import numpy as np

VERSION = 1
KINDS = ("rpc", "result", "error", "event", "state", "manifest")

KEYS: dict[str, str] = {
    "manifest": "strands/{tenant}/{device_id}/manifest",
    "state": "strands/{tenant}/{device_id}/state",
    "event": "strands/{tenant}/{device_id}/event/{name}",
    "rpc": "strands/{tenant}/{device_id}/rpc/{name}",
    "presence": "strands/{tenant}/@presence/{device_id}",
    "estop": "strands/{tenant}/@estop",
    "peer": "strands/{tenant}/@peer/{device_id}",
    "holder": "strands/{tenant}/@holder/{device_id}/{name}",
}
"""Every key the mesh uses (``peer`` is every Mesh's own liveliness, keyed by its peer id, which the device watchdog
watches; ``holder`` is each mounting peer's token for a device). Nothing else in the package spells a key.
``@presence``, ``@holder``, ``@estop`` and ``@peer`` are verbatim segments: only the literal matches them, so a
``strands/<tenant>/**`` subscriber never receives them."""

_ID = re.compile(r"^[A-Za-z0-9_.:-]+$")


def check_id(value: Any, what: str = "id") -> str:
    """A tenant, device id or name: one key segment, no wildcards or separators."""
    if not isinstance(value, str) or not _ID.match(value):
        raise ValueError(f"{what} {value!r} must be letters, digits or _ . : - (one key segment)")
    return value


def key(slot: str, tenant: str, device_id: str = "", name: str = "") -> str:
    """The key for ``slot`` (one of ``KEYS``). ``device_id="*"`` / ``name="*"`` give a wildcard."""
    if slot not in KEYS:
        raise ValueError(f"unknown key slot {slot!r}; known: {sorted(KEYS)}")
    pattern = KEYS[slot]
    if "{name}" in pattern and not name:
        raise ValueError(f"key slot {slot!r} needs a name")
    if "{name}" not in pattern and name:
        raise ValueError(f"key slot {slot!r} takes no name, got {name!r}")
    if "{device_id}" in pattern and device_id != "*":
        check_id(device_id, "device_id")
    if "{device_id}" not in pattern and device_id:
        raise ValueError(f"key slot {slot!r} takes no device_id, got {device_id!r}")
    if name and name != "*":
        check_id(name, "name")
    return pattern.format(tenant=check_id(tenant, "tenant"), device_id=device_id, name=name)


def parse_presence(k: str) -> tuple[str, str]:
    """``strands/<tenant>/@presence/<device_id>`` → ``(tenant, device_id)``."""
    parts = str(k).split("/")
    if len(parts) != 4 or parts[0] != "strands" or parts[2] != "@presence":
        raise ValueError(f"not a presence key: {k!r}")
    return parts[1], parts[3]


def envelope(
    tenant: str,
    device_id: str,
    kind: str,
    name: str,
    *,
    seq: int = 0,
    t: float | None = None,
    args: Any = None,
    result: Any = None,
    error: Any = None,
    peer: str | None = None,
) -> dict[str, Any]:
    """A new envelope; only the body fields that are set travel. ``peer`` is the sending Mesh's peer id (rpcs carry
    it so the device knows whose silence to watch)."""
    if kind not in KINDS:
        raise ValueError(f"unknown envelope kind {kind!r}; known: {KINDS}")
    env: dict[str, Any] = {"v": VERSION, "tenant": check_id(tenant, "tenant")}
    env.update(device_id=check_id(device_id, "device_id"), seq=int(seq), kind=kind, name=name)
    env["t"] = time.time() if t is None else float(t)
    if peer is not None:
        env["peer"] = check_id(peer, "peer")
    for field, value in (("args", args), ("result", result), ("error", error)):
        if value is not None:
            env[field] = value
    return env


def _default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            raise TypeError("cannot send object arrays over the mesh (numeric and bool arrays only)")
        if obj.dtype.kind in "fc" and not np.isfinite(obj).all():
            raise ValueError("cannot send NaN or infinity over the mesh (not valid JSON)")
        data = base64.b64encode(np.ascontiguousarray(obj).tobytes()).decode("ascii")
        return {"__nd__": [obj.dtype.str, list(obj.shape), data]}
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, (tuple, set, frozenset)):
        return list(obj)
    raise TypeError(f"cannot send {type(obj).__name__} over the mesh")


def _hook(d: dict[str, Any]) -> Any:
    nd = d.get("__nd__")
    if nd is not None and len(d) == 1:
        dtype, shape, data = nd
        if np.dtype(dtype).hasobject:
            raise ValueError("refusing object arrays from the mesh (numeric and bool arrays only)")
        return np.frombuffer(base64.b64decode(data), dtype=np.dtype(dtype)).reshape(shape).copy()
    return d


def _no_constant(name: str) -> Any:
    raise ValueError(f"refusing {name} from the mesh (not valid JSON)")


def encode(obj: Any) -> bytes:
    """JSON bytes; arrays and numpy scalars included. NaN and infinity are refused (they are not JSON). A dict that
    is exactly ``{"__nd__": [...]}`` is reserved: it decodes as an array."""
    try:
        return json.dumps(obj, default=_default, separators=(",", ":"), allow_nan=False).encode()
    except ValueError as exc:
        if "JSON compliant" in str(exc):
            raise ValueError("cannot send NaN or infinity over the mesh (not valid JSON)") from exc
        raise


def decode(raw: bytes | bytearray | memoryview | str) -> Any:
    """The inverse of ``encode``."""
    text = raw if isinstance(raw, str) else bytes(raw).decode()
    return json.loads(text, object_hook=_hook, parse_constant=_no_constant)


def decode_envelope(raw: bytes | bytearray | memoryview | str) -> dict[str, Any]:
    """``decode`` plus validation of every field a peer controls; another version is refused, never guessed at."""
    env = decode(raw)
    if not isinstance(env, dict) or env.get("v") != VERSION:
        got = env.get("v") if isinstance(env, dict) else type(env).__name__
        raise ValueError(f"mesh envelope version {got} is not supported (this peer speaks version {VERSION})")
    if env.get("kind") not in KINDS:
        raise ValueError(f"mesh envelope kind {env.get('kind')!r} is not one of {KINDS}")
    check_id(env.get("tenant"), "tenant")
    check_id(env.get("device_id"), "device_id")
    if "peer" in env:
        check_id(env["peer"], "peer")
    if not isinstance(env.get("name"), str):
        raise ValueError("mesh envelope name must be a string")
    if type(env.get("seq")) is not int or not isinstance(env.get("t"), (int, float)) or isinstance(env["t"], bool):
        raise ValueError("mesh envelope seq must be an int and t a number")
    return env
