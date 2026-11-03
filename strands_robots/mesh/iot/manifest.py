"""The ``manifest`` shadow's document: what a dashboard that has never met the device needs from one
``get_thing_shadow``, in a shape that fits every registry robot under IoT's 8 KB document cap.

The mesh record carries the full manifest (every control's unit, safe range, on_disconnect and interlocks, every
sensor, every rpc's JSON schema, the description) and travels on the mesh ``manifest`` key; on a humanoid it is
17–29 KB. The shadow keeps identity, each control's range, the units, the sensor count per kind and the rpc and
event NAMES, and says where the rest is.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .. import wire

_IDENTITY = ("device_id", "device_type", "tenant", "labels", "name", "real", "twin_of")


def manifest_doc(record: Mapping[str, Any]) -> dict[str, Any]:
    """The compact ``manifest`` shadow document for a device record (``mesh.record.record``)."""
    m = record.get("manifest") or {}
    controls = [c for c in (m.get("controls") or ()) if isinstance(c, Mapping) and "name" in c]
    units: dict[str, list[str]] = {}
    for c in controls:
        units.setdefault(str(c.get("unit") or "?"), []).append(str(c["name"]))
    kinds: dict[str, int] = {}
    for s in m.get("sensors") or ():
        if isinstance(s, Mapping):
            kinds[str(s.get("kind") or "?")] = kinds.get(str(s.get("kind") or "?"), 0) + 1
    doc: dict[str, Any] = {k: m[k] for k in _IDENTITY if k in m}
    doc.update(
        id=str(record["id"]),
        device_type=str(record.get("device_type") or m.get("device_type") or ""),
        tenant=str(record.get("tenant") or m.get("tenant") or ""),
        labels=dict(record.get("labels") or {}),
        controls={str(c["name"]): _range(c) for c in controls},
        units=units,
        sensors=kinds,
        rpcs=[str(r["name"]) for r in (m.get("rpcs") or ()) if isinstance(r, Mapping) and "name" in r],
        events=[str(e["name"]) for e in (m.get("events") or ()) if isinstance(e, Mapping) and "name" in e],
        dof=len(controls),
    )
    if record.get("tenant"):
        doc["full"] = wire.key("manifest", str(record["tenant"]), str(record["id"]))  # the mesh key with everything
    return doc


def _range(control: Mapping[str, Any]) -> list[float] | None:
    r = control.get("safe") or control.get("range")
    if not isinstance(r, (list, tuple)) or len(r) != 2:
        return None
    return [round(float(r[0]), 4), round(float(r[1]), 4)]
