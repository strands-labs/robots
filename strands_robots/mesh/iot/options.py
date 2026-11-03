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
    ``stale_s``: a ``control`` target whose shadow stamp is older than this at delivery is dropped as stale
    (≥ 1 s, the stamp's resolution; ``None`` applies every delta). It is the shadow's own bound and leaves the
    mesh's ``watchdog_ms`` alone. ``telemetry_hz``: > 0 publishes the state envelope on the Basic Ingest topic of
    the tenant's telemetry rule (≤ 1 Hz; the rule must exist: ``rules.apply_rules``)."""

    shadows: bool = True
    shadow_hz: float = 0.5
    stale_s: float | None = 2.0
    telemetry_hz: float = 0.0

    def __post_init__(self) -> None:
        """Every bound is checked here, before a mount starts anything: a bad option is refused by name with no
        side effect (a mirror that had already written the manifest would otherwise outlive the failed mount)."""
        from .rules import MAX_TELEMETRY_HZ
        from .shadow import MAX_SHADOW_HZ, MIN_STALE_S

        if not 0 < self.shadow_hz <= MAX_SHADOW_HZ:
            raise ValueError(f"shadow_hz={self.shadow_hz} must be in (0, {MAX_SHADOW_HZ:g}] (IoT's per-thing quota)")
        if self.stale_s is not None and not self.stale_s >= MIN_STALE_S:
            raise ValueError(
                f"stale_s={self.stale_s} is under the {MIN_STALE_S:g} s stamp resolution; None applies every delta"
            )
        if self.telemetry_hz and not 0 < self.telemetry_hz <= MAX_TELEMETRY_HZ:
            raise ValueError(
                f"telemetry_hz={self.telemetry_hz} is over {MAX_TELEMETRY_HZ:g} Hz; the mesh state stream is live"
            )


def is_iot(transport: Any) -> bool:
    return str(getattr(transport, "url", "")).startswith("iot://")


def attach(
    transports: Sequence[Any],
    options: IotOptions | Mapping[str, Any],
    robot: Any,
    record: Mapping[str, Any],
    lockout: Any,
) -> Unsubscribe:
    """Start the IoT-native pieces for one mounted device on every IoT transport. ``lockout`` is the device's
    ``Lockout``; the ``state`` shadow reports it synced with the robot's latch, as the ``lockout`` rpc does.
    Returns the undo."""
    opts = options if isinstance(options, IotOptions) else IotOptions(**dict(options))
    iot_transports = [t for t in transports if is_iot(t)]
    if not iot_transports:
        raise ValueError("iot= was given but no transport of this Mesh is iot://")
    undos: list[Unsubscribe] = []
    tenant, did = str(record["tenant"]), str(record["id"])

    def undo() -> None:
        for u in reversed(undos):
            u()

    try:
        for t in iot_transports:
            link = t.link
            if opts.shadows:
                from .shadow import ShadowMirror

                mirror = ShadowMirror(
                    link,
                    robot,
                    record,
                    shadow_hz=opts.shadow_hz,
                    stale_s=opts.stale_s,
                    lockout=lambda: lockout.synced(robot).to_dict(),  # what the lockout rpc answers, not a stale copy
                )
                undos.append(mirror.stop)  # registered before start(): a failure inside start() is undone too
                mirror.start()
            if opts.telemetry_hz > 0 and hasattr(robot, "subscribe"):
                from .rules import telemetry

                undos.append(telemetry(link, tenant, did, robot, opts.telemetry_hz))
    except BaseException:  # nothing started here outlives a failed mount (STEER K2: release on every exit)
        undo()
        raise
    return undo
