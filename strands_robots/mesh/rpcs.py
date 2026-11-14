"""Device rpcs: commands a mounted device adds to the built-in ones, by name, with a checked argument schema.

``Mesh.mount(robot, rpcs={"gripper": set_gripper, "battery": device_rpc(read_battery, write=False)})`` serves
``strands/<tenant>/<device>/rpc/gripper`` and advertises each one in the device's manifest (signed with it on a mesh
with identities). A caller finds them there: ``mesh.invoke("arm", "battery")`` works from a process that never
imported the handler. A plain callable is the safe default, a write that arms the watchdog: an observe-only device
refuses it, it is refused while the device is e-stopped, it needs the caller's peer id, its caller is the mover the
watchdog watches, and a write to an id two peers hold is refused. ``device_rpc(fn, write=False)`` marks a read (it is
answered everywhere); ``arms=False`` a write the watchdog need not watch (a light, a beep). No device rpc may reuse a
built-in name: ``estop``, ``stop`` and ``hold`` stay built in, and ``estop`` is still answered inline.

A handler takes keyword arguments only, each a JSON value (``bool``, ``int``, ``float``, ``str``, ``list``,
``dict``): the device refuses an argument the handler does not name, a missing required one, or one of the wrong
JSON type, before the handler or the watchdog hears of the call. ``by`` (who asked) reaches it only if it names it.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..robot.safety import EStopped
from . import wire

RESERVED = ("run", "calibrate")
"""Robot verbs with no mesh rpc of their own (``run`` is the tool's, ``calibrate`` the operator's): not a device rpc
name either, so a manifest never shows two meanings for one name."""

_JSON = {"bool": "boolean", "int": "integer", "float": "number", "str": "string", "list": "array", "dict": "object"}
_PY: dict[str, tuple[type, ...]] = {
    "boolean": (bool,),
    "integer": (int,),
    "number": (int, float),
    "string": (str,),
    "array": (list, tuple),
    "object": (dict,),
}


@dataclass(frozen=True)
class DeviceRpc:
    """One device rpc: its handler, the JSON schema of its arguments, and whether it writes and arms the watchdog."""

    fn: Callable[..., Any]
    write: bool = True
    arms: bool = True
    doc: str = ""
    params: Mapping[str, Any] = field(default_factory=dict)

    def advert(self, name: str) -> dict[str, Any]:
        """What the manifest says about it: a caller resolves the name, arguments and kind from this."""
        return {"name": name, "params": dict(self.params), "doc": self.doc, "write": self.write, "arms": self.arms}


def device_rpc(fn: Callable[..., Any], *, write: bool = True, arms: bool | None = None, doc: str | None = None) -> Any:
    """Mark ``fn`` as a device rpc: ``write=False`` for a read; ``arms`` defaults to ``write`` (a read never arms)."""
    arms = write if arms is None else arms
    if arms and not write:
        raise ValueError("a device rpc that arms the watchdog is a write: write=False with arms=True is refused")
    return DeviceRpc(fn, bool(write), bool(arms), (doc if doc is not None else inspect.getdoc(fn) or "").split("\n")[0])


def _kind(annotation: Any) -> dict[str, Any]:
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    parts = [p.strip().split("[")[0] for p in str(text).split("|")]
    kinds = [_JSON[p] for p in parts if p in _JSON]
    return {"type": kinds, "nullable": "None" in parts} if kinds else {}


def _schema(name: str, fn: Callable[..., Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError) as exc:
        raise TypeError(f"device rpc {name!r}: its handler has no signature to check arguments against") from exc
    props: dict[str, Any] = {}
    required: list[str] = []
    for p in params:
        if p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
            raise TypeError(f"device rpc {name!r} takes {p}: a handler names each argument it accepts (no *args, **kw)")
        props[p.name] = _kind(p.annotation)
        if p.default is p.empty:
            required.append(p.name)
    return {"type": "object", "properties": props, "required": required}


def check_rpcs(rpcs: Mapping[str, Any] | None, builtin: tuple[str, ...]) -> dict[str, DeviceRpc]:
    """The device rpcs a mount was given, checked: mesh-id names that shadow no ``builtin`` rpc, callable handlers."""
    if rpcs is None:
        return {}
    if not isinstance(rpcs, Mapping):
        raise TypeError(f"rpcs= must be a mapping of name -> handler, got {type(rpcs).__name__}")
    out: dict[str, DeviceRpc] = {}
    for name, entry in rpcs.items():
        wire.check_id(name, "rpc name")
        if name in builtin or name in RESERVED:
            raise ValueError(f"device rpc {name!r} would shadow the built-in {name!r}; built-ins: {sorted(builtin)}")
        rpc = entry if isinstance(entry, DeviceRpc) else device_rpc(entry) if callable(entry) else None
        if rpc is None or not callable(rpc.fn):
            raise TypeError(f"device rpc {name!r} needs a callable handler, got {entry!r}")
        out[name] = DeviceRpc(rpc.fn, rpc.write, rpc.arms, rpc.doc, _schema(name, rpc.fn))
    return out


def check_args(device_id: str, name: str, rpc: DeviceRpc, args: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments the handler gets, after the device checked them against its schema; ``by`` only if named."""
    props, required = rpc.params.get("properties", {}), rpc.params.get("required", [])
    given = {k: v for k, v in args.items() if k != "by" or "by" in props}
    extra = sorted(set(given) - set(props))
    if extra:
        raise TypeError(f"{device_id}.{name} takes {sorted(props) or 'no arguments'}, not {extra}")
    missing = [k for k in required if k not in given]
    if missing:
        raise TypeError(f"{device_id}.{name} needs {missing}")
    for k, v in given.items():
        kinds = props[k].get("type")
        if not kinds or (v is None and props[k].get("nullable")):
            continue
        ok = any(isinstance(v, _PY[t]) and not (isinstance(v, bool) and t != "boolean") for t in kinds)
        if not ok:
            raise TypeError(f"{device_id}.{name} argument {k}= must be {' or '.join(kinds)}, got {type(v).__name__}")
    return given


def call(device_id: str, name: str, rpc: DeviceRpc, engaged: Callable[[], bool], /, **args: Any) -> Any:
    """Run a device rpc whose arguments passed: a write is refused while the device's e-stop is latched."""
    if rpc.write and engaged():
        raise EStopped(f"{device_id} is e-stopped; {name} is a write and waits for a resume")
    return rpc.fn(**args)


def served(
    device_id: str, extras: Mapping[str, DeviceRpc], engaged: Callable[[], bool]
) -> tuple[dict[str, Callable[..., Any]], tuple[str, ...], tuple[str, ...]]:
    """At the device: a handler per device rpc, and the names that write and that arm the watchdog."""
    handlers: dict[str, Callable[..., Any]] = {
        n: functools.partial(call, device_id, n, r, engaged) for n, r in extras.items()
    }
    return handlers, tuple(n for n, r in extras.items() if r.write), tuple(n for n, r in extras.items() if r.arms)


def advertised(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """The entry for ``name`` among the rpcs a discovered device's manifest advertises, if any."""
    for r in manifest.get("rpcs") or ():
        if isinstance(r, Mapping) and r.get("name") == name:
            return r
    return None


def kind_of(devices: Any, device_id: str, name: str, builtin: tuple[str, ...]) -> bool | None:
    """At the caller: ``None`` for a built-in, else whether the device's advertised rpc ``name`` writes (missing flag:
    a write). ``LookupError`` when the device is present but does not advertise it, or is not present at all."""
    if name in builtin:
        return None
    found = [d for d in devices if d.id == device_id]
    entry = advertised(found[0].manifest, name) if found else None
    if entry is None or name in RESERVED:
        rows = found[0].manifest.get("rpcs") or () if found else ()
        names = (r.get("name") for r in rows if isinstance(r, Mapping))
        known = sorted({*builtin, *(n for n in names if isinstance(n, str))})
        raise LookupError(f"{device_id} has no rpc {name!r} over the mesh; known: {known}")
    return entry.get("write") is not False
