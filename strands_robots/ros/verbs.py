"""What ``use_ros`` does, as plain functions over a RosBus and a CommandGate (no strands import here).

Reads: status, topics, nodes, services, actions, info, echo, latest, hz, params, tf, record, profiles.
Writes (gated): publish, call, goal, set_param. Never gated: stop (the profile's STOP sequence).
"""

from __future__ import annotations

import time
from typing import Any

from . import calls
from . import profile as _profile
from .bus import RosBus
from .convert import field_types, from_dict, message_class, to_data
from .gate import CommandGate
from .names import canonical, positive

READS = (
    "status",
    "topics",
    "nodes",
    "services",
    "actions",
    "info",
    "echo",
    "latest",
    "hz",
    "params",
    "tf",
    "record",
    "profiles",
    "stats",
    "heal",
)
WRITES = ("publish", "call", "goal", "set_param")
VERBS = READS + WRITES + ("stop",)
_TF: dict[int, Any] = {}


def _type_of(bus: RosBus, topic: str, given: str | None) -> str:
    if given:
        return given
    types = bus.graph()["topics"].get(canonical(topic))
    if not types:
        raise ValueError(f"{topic} is not on the graph; pass type= to read or write it anyway")
    return types[0]


def dispatch(bus: RosBus, gate: CommandGate, verb: str, a: dict[str, Any]) -> Any:
    if verb not in VERBS:
        raise ValueError(f"unknown verb {verb!r}; verbs: {', '.join(VERBS)}")
    if verb in WRITES:
        target = a.get("topic") or a.get("service") or a.get("action") or a.get("node")
        if not target:
            raise ValueError(f"{verb} needs topic=/service=/action=/node=")
        _required(verb, a)  # an incomplete call is refused before an operator is asked about it
        gate.owners = {**bus.owners, **dict(gate.owners)}  # a Robot's command topic is never a free write
        gate.check(
            verb, target, {k: a.get(k) for k in ("type", "fields", "request", "goal", "name", "value") if k in a}
        )
    return _VERB[verb](bus, a)


def _required(verb: str, a: dict[str, Any]) -> None:
    need = {
        "publish": ("topic",),
        "call": ("service", "type"),
        "goal": ("action", "type"),
        "set_param": ("node", "name", "value"),
    }[verb]
    missing = [k for k in need if a.get(k) in (None, "")]
    if missing:
        raise ValueError(f"{verb} needs {missing}")


def _status(bus: RosBus, a: dict[str, Any]) -> Any:
    g = bus.graph()
    return {
        "node": bus.node_name,
        "discovery": bus.discovery.__dict__,
        "topics": len(g["topics"]),
        "nodes": len(g["nodes"]),
        "services": len(g["services"]),
        "owned": dict(bus.owners),
    }


def _topics(bus: RosBus, a: dict[str, Any]) -> Any:
    flt = a.get("filter") or ""
    return {t: ty for t, ty in sorted(bus.graph()["topics"].items()) if flt in t}


def _info(bus: RosBus, a: dict[str, Any]) -> Any:
    if a.get("type") and not a.get("topic"):
        return {"type": a["type"], "fields": field_types(message_class(a["type"]))}
    topic = canonical(a["topic"])
    node = bus.start().node
    pubs = [
        {
            "node": i.node_name,
            "qos": {"reliability": str(i.qos_profile.reliability), "durability": str(i.qos_profile.durability)},
        }
        for i in node.get_publishers_info_by_topic(topic)
    ]
    subs = [i.node_name for i in node.get_subscriptions_info_by_topic(topic)]
    ty = _type_of(bus, topic, a.get("type"))
    return {
        "topic": topic,
        "type": ty,
        "fields": field_types(message_class(ty)),
        "publishers": pubs,
        "subscribers": subs,
    }


def _echo(bus: RosBus, a: dict[str, Any]) -> Any:
    topic = canonical(a["topic"])
    count = int(positive(a.get("count", 1), "count", integer=True))
    timeout = positive(a.get("timeout", 3.0), "timeout")
    sub = bus.subscribe(topic, _type_of(bus, topic, a.get("type")), qos=a.get("qos", "auto"))
    out: list[Any] = []
    seen, end = sub.count, time.monotonic() + timeout
    while len(out) < min(count, 20) and time.monotonic() < end:
        m = bus.wait_newer(topic, seen, end - time.monotonic())
        if m is None:
            break
        seen = sub.count
        out.append(to_data(m, max_list=int(a.get("max_list", 32))))
    return {"topic": topic, "qos": sub.qos, "messages": out, "timed_out": len(out) < min(count, 20)}


def _latest(bus: RosBus, a: dict[str, Any]) -> Any:
    topic = canonical(a["topic"])
    bus.subscribe(topic, _type_of(bus, topic, a.get("type")), qos=a.get("qos", "auto"))
    got = bus.wait_newer(topic, 0, positive(a.get("timeout", 2.0), "timeout"))
    held = bus.latest(topic)
    return {
        "topic": topic,
        "age_s": None if held is None else round(held[1], 3),
        "msg": None if got is None else to_data(got, max_list=int(a.get("max_list", 32))),
    }


def _hz(bus: RosBus, a: dict[str, Any]) -> Any:
    topic = canonical(a["topic"])
    sub = bus.subscribe(topic, _type_of(bus, topic, a.get("type")), qos=a.get("qos", "auto"))
    seconds = positive(a.get("seconds", 2.0), "seconds")
    c0, t0 = sub.count, time.monotonic()
    time.sleep(min(seconds, 30.0))
    n = sub.count - c0
    return {"topic": topic, "qos": sub.qos, "messages": n, "hz": round(n / (time.monotonic() - t0), 2)}


def _params(bus: RosBus, a: dict[str, Any]) -> Any:
    names = a.get("names") or calls.list_params(bus, a["node"])
    return calls.get_params(bus, a["node"], list(names))


def _tf(bus: RosBus, a: dict[str, Any]) -> Any:
    from .tf import TfReader

    reader = _TF.setdefault(id(bus), TfReader(bus))
    end = time.monotonic() + float(a.get("timeout", 2.0))
    while True:
        try:
            return reader.lookup(a["target"], a["source"])
        except KeyError:
            if time.monotonic() > end:
                raise
            time.sleep(0.05)


def _record(bus: RosBus, a: dict[str, Any]) -> Any:
    from .record import record

    topics = a.get("topics") or {}
    if isinstance(topics, list):
        topics = {t: _type_of(bus, t, None) for t in topics}
    return record(bus, topics, float(a.get("seconds", 5.0)), a.get("path", "./strands_ros_bag"))


def _publish(bus: RosBus, a: dict[str, Any]) -> Any:
    topic = canonical(a["topic"])
    ty = _type_of(bus, topic, a.get("type"))
    count = int(positive(a.get("count", 1), "count", integer=True))
    rate = positive(a.get("rate", 10.0), "rate")
    if count > 100:
        raise ValueError("publish count is capped at 100; a stream belongs in a driver, not a tool call")
    msg = from_dict(message_class(ty), a.get("fields") or {})
    for i in range(count):
        bus.publish(topic, ty, msg, qos=a.get("qos", "reliable"))
        if i + 1 < count:
            time.sleep(1.0 / rate)
    return {"published": topic, "type": ty, "count": count}


def _stop(bus: RosBus, a: dict[str, Any]) -> Any:
    from ..robot.drivers.ros.stop import send_stop

    prof = _profile.load(a.get("profile") or "ros2_control")
    return {"stop": send_stop(bus, prof, a.get("sender_id", "strands-use-ros")), "profile": prof.name}


_VERB = {
    "status": _status,
    "topics": _topics,
    "info": _info,
    "echo": _echo,
    "latest": _latest,
    "hz": _hz,
    "nodes": lambda b, a: b.graph()["nodes"],
    "services": lambda b, a: b.graph()["services"],
    "actions": lambda b, a: b.graph()["actions"],
    "params": _params,
    "tf": _tf,
    "record": _record,
    "profiles": lambda b, a: _profile.available(),
    "stats": lambda b, a: b.stats(),
    "heal": lambda b, a: b.heal(),
    "publish": _publish,
    "stop": _stop,
    "call": lambda b, a: calls.service_call(
        b, a["service"], a["type"], a.get("request"), timeout=a.get("timeout", 5.0)
    ),
    "goal": lambda b, a: calls.action_goal(b, a["action"], a["type"], a.get("goal"), timeout=a.get("timeout", 30.0)),
    "set_param": lambda b, a: calls.set_param(b, a["node"], a["name"], a["value"]),
}
