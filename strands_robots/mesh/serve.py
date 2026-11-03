"""The device side of the mesh: answer rpcs for one mounted robot.

Every motion rpc goes through the robot's safety chain and operator gate (``confirm_act``); a remote caller can
never confirm on the operator's behalf. ``estop`` has its own queryable, answered on the transport thread itself:
it never waits on the rpc pool, a lock or a queue. Errors travel as ``{"type", "message"}`` with the message
verbatim. Every rpc leaves one record on the ``strands_robots.audit`` logger: tenant, device, rpc, who, peer, seq and
the outcome. ``by`` is the name the caller asserts; the transport proves the device, not the operator.
"""

from __future__ import annotations

import itertools
import logging
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..robot.safety import RequestRefused
from . import wire
from .authority import Authority, Lockout, new_epoch
from .codec import observation_to_dict
from .transports.base import Transport, Unsubscribe
from .watchdog import MOTION_RPCS

AUDIT = logging.getLogger("strands_robots.audit")
MAX_STEPS = 10_000
"""The most world steps one ``act`` may ask for: every robot in the world waits on its lock meanwhile."""
MAX_SECONDS = 10.0
"""The longest reach one ``act(seconds=)`` may ask for: the rpc holds a worker (and its invoker) that long."""

WRITE_RPCS = ("act", "hold")
"""The rpcs that physically move a body: an observe-only device (a hardware body mounted without a watchdog)
refuses them by name, so a mover can never leave an arm commanded with nothing to recall it."""

ARGS: dict[str, tuple[str, ...]] = {
    "describe": (),
    "observe": ("images",),
    "act": ("action", "steps", "seconds", "by"),
    "hold": ("by",),
    "stop": ("by",),
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


def act_wait(args: dict[str, Any]) -> float:
    """How much longer than one control step an ``act`` reply may take: its ``seconds``, if it is one we serve."""
    s = args.get("seconds")
    if isinstance(s, (int, float)) and not isinstance(s, bool) and 0 < s <= MAX_SECONDS:
        return float(s)
    return 0.0


def _act(robot: Any, action: Any = None, steps: int | None = None, seconds: float | None = None) -> Any:
    """``seconds``: reach the pose over that long (``Robot.act``'s ramp), not one control step toward it."""
    if action is None:
        raise TypeError("act needs action=")
    if steps is not None and (isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= MAX_STEPS):
        raise ValueError(f"act steps= must be an integer in [1, {MAX_STEPS}], got {steps!r}")
    if seconds is not None and steps is not None:  # before the operator is asked about a call that cannot run
        raise ValueError("act takes steps= or seconds=, not both")
    if seconds is not None and not act_wait({"seconds": seconds}):
        raise ValueError(f"act seconds= must be a number in (0, {MAX_SECONDS}], got {seconds!r}")
    return observation_to_dict(robot.confirm_act(action, steps=steps, seconds=seconds))


def _lockout(robot: Any, lockout: Lockout) -> Lockout:
    """The lockout as the robot's latch says it is (a local ``robot.estop()`` opens one too)."""
    return lockout.synced(robot)


def estop(robot: Any, lockout: Lockout, by: str | None = None, epoch: str | None = None) -> None:
    """Latch first, then record which stop and who: never waits on a lock, a pool or a queue.

    The device mints the lockout's epoch and the first stopper keeps it: a later stop (or the caller's ``epoch``,
    only a correlation id for the broadcast) never renames who stopped, nor picks the epoch an assertion signs."""
    if not _lockout(robot, lockout).engaged:  # before the body's stop, so a stop that raises still locks resume
        lockout.engaged, lockout.epoch, lockout.by, lockout.at = True, new_epoch(), by, time.time()
    robot.estop()


def estop_each(mounted: Iterable[tuple[Any, Any, Any, Lockout]], sel: Any, by: str, epoch: str, tenant: str) -> None:
    """The tenant-wide e-stop on every mounted device ``sel`` matches: one whose stop raises (a bus in a bad state,
    an object with no ``estop``) leaves an ``AUDIT`` warning naming it and does not keep the stop from the rest."""
    for robot, device, _undo, lockout in mounted:
        if sel.matches(device):
            try:
                estop(robot, lockout, by, epoch)
            except Exception as exc:
                AUDIT.warning("%s/%s estop by=%s: the device's stop raised %r", tenant, device.id, by, exc)


def _resume(
    robot: Any, did: str, lockout: Lockout, authority: Authority, approved_by: Any, assertion: Any, local: bool
) -> None:
    current = _lockout(robot, lockout)
    if not current.engaged:
        return
    name = authority.check(did, current, approved_by, assertion, local=local)
    robot.resume(approved_by=name)
    _lockout(robot, lockout)


def _hold(robot: Any) -> None:
    """A hold energises a limp arm (torque on at the measured pose): the safety chain and operator say go first."""
    if hasattr(robot, "gate"):
        robot.gate(robot.observe(images=False).joints)
    robot.hold()


def _handlers(
    robot: Any, did: str, lockout: Lockout, authority: Callable[[], Authority], local: bool
) -> dict[str, Callable[..., Any]]:
    return {
        "describe": lambda: robot.describe(),
        "observe": lambda images=False: _observe(robot, bool(images)),
        "act": lambda action=None, steps=None, seconds=None, by=None: _act(robot, action, steps, seconds),
        "hold": lambda by=None: _hold(robot),
        "stop": lambda by=None: None if robot.stop() is None else "stopped",
        "estop": lambda by=None, epoch=None: estop(robot, lockout, by, epoch),
        "resume": lambda approved_by=None, assertion=None: _resume(
            robot, did, lockout, authority(), approved_by, assertion, local
        ),
        "lockout": lambda: _lockout(robot, lockout).to_dict(),
    }


def in_process(url: str) -> bool:
    """True only when no peer outside this process can reach the transport: ``mem://`` and the fake IoT broker.

    A zenoh endpoint is never in-process, even one that only listens on loopback: a listener accepts connections
    from any other local process (another user, a container, an ``ssh -L`` tunnel), so a name claimed over it is
    not trusted without operator keys.
    """
    from urllib.parse import urlparse

    return url.startswith("mem://") or (url.startswith("iot://") and urlparse(url).netloc == "fake")


def _error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def serve(
    transport: Transport,
    tenant: str,
    robot: Any,
    device_id: str,
    lockout: Lockout,
    authority: Callable[[], Authority],
    on_rpc: Callable[[str, str | None], Callable[[], None] | None] | None = None,
) -> Unsubscribe:
    """Answer ``strands/<tenant>/<device_id>/rpc/<name>`` for ``robot``; ``on_rpc(name, peer)`` sees every call
    before it runs (the watchdog) and may return a callable run once the call ends. Returns the undo."""
    handlers = _handlers(robot, device_id, lockout, authority, in_process(transport.url))
    counter = itertools.count(1)
    # a hardware body with no mesh watchdog (on_rpc is None) is served observe-only: a mover that vanishes
    # must never leave an arm commanded with nothing to recall it; a simulated body is harmless, unchanged
    observe_only = on_rpc is None and bool(getattr(robot, "is_real", False))

    def answer(key: str, payload: bytes) -> bytes:
        name = key.rsplit("/", 1)[-1]
        seq = next(counter)
        env: dict[str, Any] = {}
        args: dict[str, Any] = {}
        done: Callable[[], None] | None = None
        try:
            env = wire.decode_envelope(payload)
            seq = int(env.get("seq", seq))  # M3: echo the request's seq so the caller can match the reply
            args = env.get("args") or {}
            if observe_only and name in WRITE_RPCS:
                raise RequestRefused(
                    f"{device_id} is mounted observe-only (a hardware device with watchdog_ms=None); "
                    f"{name} is refused. Mount it with watchdog_ms= so a vanished mover recalls it."
                )
            if on_rpc is not None and name in MOTION_RPCS and env.get("peer") is None:
                raise TypeError(f"{device_id}.{name} needs the caller's peer id: the watchdog arms on whoever moves it")
            if on_rpc is not None and name != "estop":
                done = on_rpc(name, env.get("peer"))
            extra = sorted(set(args) - set(ARGS[name]))
            if extra:
                raise TypeError(f"{device_id}.{name} takes {list(ARGS[name]) or 'no arguments'}, not {extra}")
            body = {"result": handlers[name](**args)}
        except Exception as exc:
            body = {"error": _error(exc)}
        finally:
            if done is not None:  # the watchdog's quiet clock restarts when the motion ends, not when it began
                done()
        outcome = "ok" if "result" in body else body["error"]["type"]
        who = (args.get("by") or args.get("approved_by")) if isinstance(args, dict) else None
        AUDIT.info(
            "%s/%s %s by=%s peer=%s seq=%s %s",
            tenant,
            device_id,
            name,
            who,
            env.get("peer"),
            seq,
            outcome,
            extra={
                "tenant": tenant,
                "device_id": device_id,
                "rpc": name,
                "by": who,
                "peer": env.get("peer"),
                "seq": seq,
                "outcome": outcome,
            },
        )
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
