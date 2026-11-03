"""``manifest_for(robot)``: a device's manifest generated from its Description and its Body, never by hand."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, cast

from .manifest import Control, Event, Manifest, OnDisconnect, Rpc, Sensor

if TYPE_CHECKING:
    from ...description import Description
    from ..robot import Robot

RPCS: tuple[str, ...] = ("describe", "observe", "act", "run", "stop", "hold", "estop", "resume", "calibrate")
STATE_HZ = 10.0
_JSON = (("bool", "boolean"), ("int", "integer"), ("float", "number"), ("str", "string"))


def _json_type(annotation: Any) -> dict[str, Any]:
    text = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", "")
    kinds = [j for py, j in _JSON if py in text.replace("Callable", "").split(" | ")]
    if not kinds:
        return {}
    out: dict[str, Any] = {"type": kinds[0] if len(kinds) == 1 else kinds}
    if "None" in text:
        out["nullable"] = True
    return out


def rpc_schema(fn: Any) -> dict[str, Any]:
    """JSON schema of a method's arguments (the receiver dropped), from its signature."""
    props: dict[str, Any] = {}
    required: list[str] = []
    params = list(inspect.signature(fn).parameters.values())
    for p in params if inspect.ismethod(fn) else params[1:]:  # a class attribute: drop the receiver
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        props[p.name] = _json_type(p.annotation)
        if p.default is p.empty:
            required.append(p.name)
    return {"type": "object", "properties": props, "required": required}


def _rpc(cls: type, name: str) -> Rpc:
    fn = getattr(cls, name)
    return Rpc(name, rpc_schema(fn), (inspect.getdoc(fn) or "").split("\n")[0])


def position_hold_of(desc: Description, body: object) -> bool:
    """Does this body keep its last goal under torque? Registry ``hardware.position_hold``, else the driver."""
    declared = desc.hardware.position_hold
    return bool(getattr(body, "supports_position_hold", False) if declared is None else declared)


def manifest_of(
    desc: Description,
    *,
    device_id: str,
    real: bool = False,
    position_hold: bool = True,
    world: str | None = None,
    cameras: tuple[str, ...] = (),
    tenant: str = "default",
    labels: dict[str, str] | None = None,
    twin: str | None = None,
    twin_of: str | None = None,
) -> Manifest:
    """A device surface from a Description: one control per actuated joint, sensors per joint, the rpcs."""
    from ..robot import Robot

    hw, keys = desc.hardware, desc.action_keys
    sensors = [Sensor(f"{k}.{q}", u) for k in keys for q, u in (("pos", "rad"), ("vel", "rad/s"), ("effort", "N*m"))]
    if desc.floating:
        sensors.append(Sensor("base.pose", "m,quat", kind="pose"))
    sensors += [Sensor(f"{c}.image", "rgb8", kind="camera") for c in cameras]
    on_disconnect = cast(OnDisconnect, hw.on_disconnect or "hold")
    controls = [
        Control(f"{k}.pos", "rad", desc.limits.get(k), (hw.safe or {}).get(k) or desc.limits.get(k), on_disconnect)
        for k in keys
    ]
    return Manifest(
        device_id=device_id,
        device_type=desc.device_type or desc.category,
        name=desc.name,
        real=real,
        position_hold=position_hold,
        world=world,
        tenant=tenant,
        labels=dict(labels or {}),
        twin=twin,
        twin_of=twin_of,
        sensors=tuple(sensors),
        controls=tuple(controls),
        rpcs=tuple(_rpc(Robot, n) for n in RPCS),
        events=(Event("state", STATE_HZ), Event("fault"), Event("estop"), Event("run.started"), Event("run.stopped")),
        description=desc.describe(),
    )


def manifest_key(robot: Robot) -> tuple[Any, ...]:
    """Everything a manifest is derived from that can change after the Robot is built (K4: cache on this)."""
    sim = robot.sim
    cams = () if sim is None else tuple(c.name for c in sim.scene.cameras if c.robot == robot.id)
    twin = None if robot.twin is None else robot.twin.id
    twin_of = None if robot.twin_of is None else robot.twin_of.id
    return (None if sim is None else sim.backend_name, cams, twin, twin_of, robot.supports_position_hold)


def manifest_for(robot: Robot, *, tenant: str = "default", labels: dict[str, str] | None = None) -> Manifest:
    """The robot's device surface, generated from its Description and where its body lives."""
    sim = robot.sim
    return manifest_of(
        robot.description,
        device_id=robot.id,
        real=robot.is_real,
        position_hold=robot.supports_position_hold,
        world=None if sim is None else sim.backend_name,
        cameras=() if sim is None else tuple(c.name for c in sim.scene.cameras if c.robot == robot.id),
        tenant=tenant,
        labels=labels,
        twin=None if robot.twin is None else robot.twin.id,
        twin_of=None if robot.twin_of is None else robot.twin_of.id,
    )
