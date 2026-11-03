"""A device's manifest: what it is, what it reports, what it accepts, and what it pushes.

A ``Robot`` is a device. Its manifest is generated (``device.build.manifest_for``), never hand-written,
and it is the one surface the mesh, the agent tools and the dashboard read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

OnDisconnect = Literal["hold", "zero", "safe", "apply"]
ON_DISCONNECT: tuple[str, ...] = ("hold", "zero", "safe", "apply")
Bound = tuple[float, float] | None


def _bound(b: Any) -> Bound:
    return None if b is None else (float(b[0]), float(b[1]))


@dataclass(frozen=True)
class Sensor:
    """A read-only value: ``<label>.pos``, ``<label>.vel``, ``<label>.effort``, a camera frame, a pose."""

    name: str
    unit: str
    kind: str = "joint"  # joint | camera | pose


@dataclass(frozen=True)
class Control:
    """A writable value: hard ``range``, soft ``safe`` bound, and what happens when the commander vanishes."""

    name: str
    unit: str
    range: Bound
    safe: Bound
    on_disconnect: OnDisconnect = "hold"
    disconnect_value: float | None = None
    interlocks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.on_disconnect not in ON_DISCONNECT:
            raise ValueError(f"{self.name}: on_disconnect {self.on_disconnect!r} is not one of {ON_DISCONNECT}")
        if self.on_disconnect == "apply" and self.disconnect_value is None:
            raise ValueError(f"{self.name}: on_disconnect='apply' needs a disconnect_value")
        r, s = self.range, self.safe
        if r is not None and s is not None and not (r[0] <= s[0] <= s[1] <= r[1]):
            raise ValueError(f"{self.name}: safe {list(s)} is not inside range {list(r)}")


@dataclass(frozen=True)
class Rpc:
    """A named command; ``params`` is the JSON schema of its arguments, generated from the signature."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    doc: str = ""


@dataclass(frozen=True)
class Event:
    """Something the device pushes (``state`` at ``hz``, ``fault``, ``estop``, ``run.started`` …)."""

    name: str
    hz: float | None = None


@dataclass(frozen=True)
class Manifest:
    """The machine-readable description of a device: identity plus every sensor, control, rpc and event."""

    device_id: str
    device_type: str
    name: str
    real: bool
    position_hold: bool = False  # unknown reads as "does not hold", the safe answer
    world: str | None = None
    tenant: str = "default"
    labels: Mapping[str, str] = field(default_factory=dict)
    twin: str | None = None
    twin_of: str | None = None
    sensors: tuple[Sensor, ...] = ()
    controls: tuple[Control, ...] = ()
    rpcs: tuple[Rpc, ...] = ()
    events: tuple[Event, ...] = ()
    description: Mapping[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.device_id

    def control(self, name: str) -> Control:
        for c in self.controls:
            if c.name == name:
                return c
        raise KeyError(f"{self.device_id} has no control {name!r}; controls are {[c.name for c in self.controls]}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe. Also carries ``id`` and the description summary, so it is what ``describe()`` returns."""
        d = asdict(self)
        d["labels"] = dict(self.labels)
        for c in d["controls"]:
            for k in ("range", "safe"):
                c[k] = None if c[k] is None else list(c[k])
            c["interlocks"] = list(c["interlocks"])
        d["id"] = self.device_id
        d.update({k: v for k, v in self.description.items() if k not in d})
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Manifest:
        return cls(
            device_id=d["device_id"],
            device_type=d["device_type"],
            name=d["name"],
            real=bool(d["real"]),
            position_hold=bool(d.get("position_hold", False)),
            world=d.get("world"),
            tenant=d.get("tenant", "default"),
            labels=dict(d.get("labels") or {}),
            twin=d.get("twin"),
            twin_of=d.get("twin_of"),
            sensors=tuple(Sensor(**s) for s in d.get("sensors", ())),
            controls=tuple(
                Control(
                    **{
                        **c,
                        "range": _bound(c["range"]),
                        "safe": _bound(c["safe"]),
                        "interlocks": tuple(c.get("interlocks", ())),
                    }
                )
                for c in d.get("controls", ())
            ),
            rpcs=tuple(Rpc(r["name"], dict(r.get("params") or {}), r.get("doc", "")) for r in d.get("rpcs", ())),
            events=tuple(Event(**e) for e in d.get("events", ())),
            description=dict(d.get("description") or {}),
        )
