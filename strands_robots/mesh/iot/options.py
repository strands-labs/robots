"""``Mesh.mount(robot, iot=IotOptions(...))``: what a mounted device does on its IoT transports, beyond the mesh.

Nothing here runs unless one of the Mesh's transports is ``iot://``; ``Robot``, ``Driver``, ``Simulation`` and the
safety chain are untouched (a shadow target still goes through ``confirm_act``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..transports.base import Unsubscribe


@dataclass(frozen=True)
class IotOptions:
    """``shadows``: keep the ``manifest``/``state``/``control`` named shadows (``shadow_hz`` ≤ 1 for ``state``).
    ``telemetry_hz``: > 0 publishes the state envelope on the Basic Ingest topic of the tenant's telemetry rule
    (≤ 1 Hz; the rule must exist: ``rules.apply_rules``)."""

    shadows: bool = True
    shadow_hz: float = 0.5
    telemetry_hz: float = 0.0


def is_iot(transport: Any) -> bool:
    return str(getattr(transport, "url", "")).startswith("iot://")


def attach(
    transports: Sequence[Any],
    options: IotOptions | Mapping[str, Any],
    robot: Any,
    record: Mapping[str, Any],
    watchdog_ms: float | None,
    lockout: Any,
) -> Unsubscribe:
    """Start the IoT-native pieces for one mounted device on every IoT transport. ``lockout`` is the device's
    ``Lockout`` (its ``to_dict`` is reported into the ``state`` shadow). Returns the undo."""
    opts = options if isinstance(options, IotOptions) else IotOptions(**dict(options))
    iot_transports = [t for t in transports if is_iot(t)]
    if not iot_transports:
        raise ValueError("iot= was given but no transport of this Mesh is iot://")
    undos: list[Unsubscribe] = []
    tenant, did = str(record["tenant"]), str(record["id"])
    for t in iot_transports:
        link = t.link
        if opts.shadows:
            from .shadow import ShadowMirror

            mirror = ShadowMirror(
                link, robot, record, shadow_hz=opts.shadow_hz, watchdog_ms=watchdog_ms, lockout=lockout.to_dict
            ).start()
            undos.append(mirror.stop)
        if opts.telemetry_hz > 0 and hasattr(robot, "subscribe"):
            from .rules import telemetry

            undos.append(telemetry(link, tenant, did, robot, opts.telemetry_hz))

    def undo() -> None:
        for u in reversed(undos):
            u()

    return undo
