"""A Robot is a device: one generated manifest of sensors, controls, rpcs and events."""

from __future__ import annotations

from .build import RPCS, manifest_for, manifest_of, position_hold_of, rpc_schema
from .events import EVENTS, Emitter, state_payload
from .manifest import ON_DISCONNECT, Control, Event, Manifest, Rpc, Sensor
from .surface import DeviceVerbs

__all__ = [
    "Manifest",
    "Sensor",
    "Control",
    "Rpc",
    "Event",
    "ON_DISCONNECT",
    "RPCS",
    "EVENTS",
    "manifest_for",
    "manifest_of",
    "position_hold_of",
    "rpc_schema",
    "state_payload",
    "Emitter",
    "DeviceVerbs",
]
