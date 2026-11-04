"""Request/reply on a RosBus: services, action goals, parameters, and request/response topic pairs.

Every wait has one ``time.monotonic()`` deadline. An action that runs out of time is cancelled before the
timeout is reported, so no goal keeps running untracked; feedback is capped (first 4 + latest).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

from .bus import RosBus
from .convert import action_class, from_dict, message_class, service_class, to_data
from .names import canonical, positive

FEEDBACK_KEEP = 5


def _wait_future(fut: Any, deadline: float) -> bool:
    done = threading.Event()
    fut.add_done_callback(lambda _f: done.set())
    return done.wait(max(0.0, deadline - time.monotonic())) or fut.done()


def service_call(
    bus: RosBus, name: str, type: str, request: Mapping[str, Any] | None = None, *, timeout: float = 5.0
) -> dict[str, Any]:
    name, timeout = canonical(name, what="service"), positive(timeout, "timeout")
    cls = service_class(type)
    node = bus.start().node
    deadline = time.monotonic() + timeout
    client = node.create_client(cls, name)
    try:
        if not client.wait_for_service(timeout_sec=timeout):
            raise TimeoutError(f"service {name} is not available after {timeout}s")
        fut = client.call_async(from_dict(cls.Request, request))
        if not _wait_future(fut, deadline):
            raise TimeoutError(f"service {name} did not answer within {timeout}s")
        return {"service": name, "response": to_data(fut.result())}
    finally:
        node.destroy_client(client)


def action_goal(
    bus: RosBus, name: str, type: str, goal: Mapping[str, Any] | None = None, *, timeout: float = 30.0
) -> dict[str, Any]:
    """Send one goal and wait for its result; cancel it if ``timeout`` runs out."""
    from rclpy.action import ActionClient

    name, timeout = canonical(name, what="action"), positive(timeout, "timeout")
    cls = action_class(type)
    node = bus.start().node
    deadline = time.monotonic() + timeout
    feedback: list[Any] = []
    client = ActionClient(node, cls, name)

    def on_feedback(fb: Any) -> None:
        data = to_data(getattr(fb, "feedback", fb), max_list=16)
        if len(feedback) < FEEDBACK_KEEP:
            feedback.append(data)
        else:
            feedback[-1] = data  # first 4 kept, the last slot always holds the latest

    try:
        if not client.wait_for_server(timeout_sec=timeout):
            raise TimeoutError(f"action server {name} is not available after {timeout}s")
        send = client.send_goal_async(from_dict(cls.Goal, goal), feedback_callback=on_feedback)
        if not _wait_future(send, deadline):
            raise TimeoutError(f"action {name} did not accept the goal within {timeout}s")
        handle = send.result()
        if not getattr(handle, "accepted", False):
            return {"action": name, "accepted": False}
        res = handle.get_result_async()
        if not _wait_future(res, deadline):
            cancel = handle.cancel_goal_async()
            _wait_future(cancel, time.monotonic() + 2.0)
            raise TimeoutError(f"action {name} ran past {timeout}s; the goal was cancelled")
        out = res.result()
        return {
            "action": name,
            "accepted": True,
            "status": int(getattr(out, "status", 0)),
            "result": to_data(getattr(out, "result", out)),
            "feedback": feedback,
        }
    finally:
        client.destroy()


def get_params(bus: RosBus, node_name: str, names: list[str], *, timeout: float = 3.0) -> dict[str, Any]:
    node_name = canonical(node_name, what="node")
    r = service_call(
        bus, f"{node_name}/get_parameters", "rcl_interfaces/srv/GetParameters", {"names": names}, timeout=timeout
    )
    values = r["response"].get("values", [])
    return {n: _param_value(v) for n, v in zip(names, values, strict=False)}


def set_param(bus: RosBus, node_name: str, name: str, value: Any, *, timeout: float = 3.0) -> dict[str, Any]:
    """Set one parameter and read it back; ``ok`` only when the read-back equals the value sent."""
    node_name = canonical(node_name, what="node")
    pv = _param_msg(value)
    r = service_call(
        bus,
        f"{node_name}/set_parameters",
        "rcl_interfaces/srv/SetParameters",
        {"parameters": [{"name": name, "value": pv}]},
        timeout=timeout,
    )
    back = get_params(bus, node_name, [name], timeout=timeout).get(name)
    return {
        "node": node_name,
        "name": name,
        "sent": value,
        "read_back": back,
        "ok": back == value,
        "results": r["response"].get("results"),
    }


def list_params(bus: RosBus, node_name: str, *, timeout: float = 3.0) -> list[str]:
    node_name = canonical(node_name, what="node")
    r = service_call(bus, f"{node_name}/list_parameters", "rcl_interfaces/srv/ListParameters", {}, timeout=timeout)
    return list((r["response"].get("result") or {}).get("names", []))


_PTYPES = {
    1: "bool_value",
    2: "integer_value",
    3: "double_value",
    4: "string_value",
    5: "byte_array_value",
    6: "bool_array_value",
    7: "integer_array_value",
    8: "double_array_value",
    9: "string_array_value",
}


def _param_value(v: Mapping[str, Any]) -> Any:
    field = _PTYPES.get(int(v.get("type", 0)))
    return None if field is None else v.get(field)


def _param_msg(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"type": 1, "bool_value": value}
    if isinstance(value, int):
        return {"type": 2, "integer_value": value}
    if isinstance(value, float):
        return {"type": 3, "double_value": value}
    if isinstance(value, str):
        return {"type": 4, "string_value": value}
    raise ValueError(f"parameter value must be bool, int, float or str, got {type(value).__name__}")


def request_topic(
    bus: RosBus,
    base: str,
    req_type: str,
    fields: Mapping[str, Any],
    *,
    ack_type: str,
    timeout: float = 3.0,
    id_field: str = "header.id",
    done: str | None = "complete",
) -> dict[str, Any]:
    """Vendor request/response/complete topic triplets (``<base>/request`` -> ``/response`` -> ``/complete``).

    Subscribes to the replies BEFORE publishing, stamps a fresh id, and returns only replies with that id.
    """
    base = canonical(base)
    rid = uuid.uuid4().hex[:16]
    subs = {k: bus.subscribe(f"{base}/{k}", ack_type, qos="reliable") for k in ("response", *(done and [done] or []))}
    counts = {k: s.count for k, s in subs.items()}
    payload = _set_path(dict(fields), id_field, rid)
    bus.publish(f"{base}/request", req_type, from_dict(message_class(req_type), payload))
    out: dict[str, Any] = {"request": f"{base}/request", "id": rid}
    deadline = time.monotonic() + timeout
    for k, s in subs.items():
        msg = bus.wait_newer(
            s.topic, counts[k], max(0.0, deadline - time.monotonic()), pred=lambda m: _get_path(m, id_field) == rid
        )
        out[k] = None if msg is None else to_data(msg, max_list=16)
    return out


def _set_path(d: dict[str, Any], path: str, value: Any) -> dict[str, Any]:
    cur = d
    *head, last = path.split(".")
    for p in head:
        cur = cur.setdefault(p, {})
    cur[last] = value
    return d


def _get_path(msg: Any, path: str) -> Any:
    for p in path.split("."):
        msg = getattr(msg, p, None)
    return msg
