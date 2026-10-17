"""The device side of the mesh: answer rpcs for one mounted robot.

Every motion rpc goes through the robot's safety chain and operator gate (``confirm_act``); a remote caller can
never confirm on the operator's behalf. ``estop`` has its own queryable, answered on the transport thread itself:
it never waits on the rpc pool, a lock or a queue. Errors travel as ``{"type", "message"}`` with the message
verbatim.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from . import wire
from .authority import Authority, Lockout, new_epoch
from .codec import observation_to_dict
from .transports.base import Transport, Unsubscribe

ARGS: dict[str, tuple[str, ...]] = {
    "describe": (),
    "observe": ("images",),
    "act": ("action", "steps"),
    "hold": (),
    "stop": (),
    "estop": ("by", "epoch"),
    "resume": ("approved_by", "assertion"),
    "lockout": (),
}
"""The rpcs a mounted device answers over the mesh, and the arguments each takes."""


_RENDER: dict[int, ThreadPoolExecutor] = {}


def _render_thread(robot: Any) -> ThreadPoolExecutor:
    """One thread per robot does every image observe: a GL context is bound to the thread that first rendered with
    it (measured on macOS: a render from a second thread never returns)."""
    pool = _RENDER.get(id(robot))
    if pool is None:
        pool = _RENDER[id(robot)] = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mesh-render")
    return pool


def _observe(robot: Any, images: bool) -> Any:
    if not images:
        return observation_to_dict(robot.observe(images=False))
    obs = _render_thread(robot).submit(robot.observe, images=True).result()
    return observation_to_dict(obs, images=True)


def _act(robot: Any, action: Any = None, steps: int | None = None) -> Any:
    if action is None:
        raise TypeError("act needs action=")
    return observation_to_dict(robot.confirm_act(action, steps=steps))


def _lockout(robot: Any, lockout: Lockout) -> Lockout:
    """The lockout as the robot's latch says it is (a local ``robot.estop()`` opens one too)."""
    engaged = bool(getattr(robot, "estopped", False))
    if engaged and not lockout.engaged:
        lockout.engaged, lockout.epoch, lockout.by = True, new_epoch(), None
    elif not engaged:
        lockout.engaged, lockout.epoch, lockout.by = False, None, None
    return lockout


def estop(robot: Any, lockout: Lockout, by: str | None = None, epoch: str | None = None) -> None:
    """Latch first, then record which stop and who: never waits on a lock, a pool or a queue."""
    robot.estop()
    if not lockout.engaged or epoch is not None:
        lockout.engaged, lockout.epoch, lockout.by = True, epoch or new_epoch(), by


def _resume(
    robot: Any, did: str, lockout: Lockout, authority: Authority, approved_by: Any, assertion: Any, local: bool
) -> None:
    current = _lockout(robot, lockout)
    if not current.engaged:
        return
    name = authority.check(did, current, approved_by, assertion, local=local)
    robot.resume(approved_by=name)
    _lockout(robot, lockout)


def _handlers(
    robot: Any, did: str, lockout: Lockout, authority: Callable[[], Authority], local: bool
) -> dict[str, Callable[..., Any]]:
    return {
        "describe": lambda: robot.describe(),
        "observe": lambda images=False: _observe(robot, bool(images)),
        "act": lambda action=None, steps=None: _act(robot, action, steps),
        "hold": lambda: robot.hold(),
        "stop": lambda: None if robot.stop() is None else "stopped",
        "estop": lambda by=None, epoch=None: estop(robot, lockout, by, epoch),
        "resume": lambda approved_by=None, assertion=None: _resume(
            robot, did, lockout, authority(), approved_by, assertion, local
        ),
        "lockout": lambda: _lockout(robot, lockout).to_dict(),
    }


def is_local(url: str) -> bool:
    """In-process (``mem://``, the fake IoT broker), or zenoh with explicit endpoints that are all loopback."""
    if url.startswith(("mem://", "iot://fake")):
        return True
    if not url.startswith("zenoh://") or "?" not in url:
        return False
    from urllib.parse import parse_qs

    q = parse_qs(url.split("?", 1)[1])
    eps = q.get("listen", []) + q.get("connect", [])
    hosts = [e.split("/", 1)[-1].rsplit(":", 1)[0] for e in eps]
    return bool(hosts) and all(h in ("127.0.0.1", "localhost", "[::1]") for h in hosts)


def _error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def serve(
    transport: Transport,
    tenant: str,
    robot: Any,
    device_id: str,
    lockout: Lockout,
    authority: Callable[[], Authority],
    on_rpc: Callable[[str, str | None], None] | None = None,
) -> Unsubscribe:
    """Answer ``strands/<tenant>/<device_id>/rpc/<name>`` for ``robot``; ``on_rpc(name, peer)`` sees every call
    before it runs (the watchdog). Returns the undo."""
    handlers = _handlers(robot, device_id, lockout, authority, is_local(transport.url))
    counter = itertools.count(1)

    def answer(key: str, payload: bytes) -> bytes:
        name = key.rsplit("/", 1)[-1]
        seq = next(counter)
        try:
            env = wire.decode_envelope(payload)
            args = env.get("args") or {}
            if on_rpc is not None and name != "estop":
                on_rpc(name, env.get("peer"))
            extra = sorted(set(args) - set(ARGS[name]))
            if extra:
                raise TypeError(f"{device_id}.{name} takes {list(ARGS[name]) or 'no arguments'}, not {extra}")
            body = {"result": handlers[name](**args)}
        except Exception as exc:
            body = {"error": _error(exc)}
        return wire.encode(
            wire.envelope(tenant, device_id, "result" if "result" in body else "error", name, seq=seq, **body)
        )

    undos = [
        transport.queryable(wire.key("rpc", tenant, device_id, name), answer, inline=(name == "estop")) for name in ARGS
    ]

    def undo() -> None:
        for u in undos:
            u()

    return undo
