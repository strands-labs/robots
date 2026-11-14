"""Skills: named policies a mounted device runs ON ITSELF when the mesh asks, so no control loop crosses the network.

``Mesh.mount(robot, skills={"walk": "microduck?skill=velstand&vx=0.3", "wave": my_policy})`` gives the device a
catalogue. A caller names a skill and a duration (``invoke(id, "skill", name="walk", seconds=4)``); the device starts
it with its own ``Robot.start`` and answers at once, and ``skill_status`` reports it. The caller never passes a
policy, a model id or an address: only a name from the device's own list. The policy then runs at the device's own
rate, where its cameras and its safety chain are, instead of one observe and one act rpc per control step from the
caller (measured over AWS IoT Core for a 50 Hz walk: 5.7 steps per second through the caller, full rate here).

The run is an agent's (``from_agent``), so every step passes the device's safety chain and a real body asks its own
``Robot(operator=)`` once, with the ``RunPlan``, on the device: an operator answers where the robot is, and the reply
to the caller never waits on a person. ``stop`` and ``estop`` end a running skill. A skill holds its caller's watchdog
for as long as it runs: the quiet clock pauses (the device is busy, the caller may poll slowly), but if the caller's
Mesh vanishes (closed, crashed, network gone) the device ends the skill and applies its ``on_disconnect``. A skill is
bounded by its own ``seconds``; a robot whose commander is gone stops rather than finishing work nobody is watching.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from . import wire

MAX_SKILL_SECONDS = 60.0
"""The longest one ``skill`` call may run: a bound the device enforces, whatever the caller asks."""

SKILL_ARGS = ("name", "seconds", "instruction", "by")


def check_catalogue(skills: Mapping[str, Any] | None) -> dict[str, Any]:
    """The catalogue a mount was given, checked: names are mesh ids (what a selector and a key can carry), every
    entry is a policy (a spec string, a ``PolicyLike`` or a callable), and nothing names a network address."""
    if skills is None:
        return {}
    if not isinstance(skills, Mapping):
        raise TypeError(f"skills= must be a mapping of name -> policy, got {type(skills).__name__}")
    out: dict[str, Any] = {}
    for name, policy in skills.items():
        wire.check_id(name, "skill name")
        if policy is None or isinstance(policy, bool):
            raise TypeError(f"skill {name!r} needs a policy (a spec string, a Policy or a callable), got {policy!r}")
        if isinstance(policy, str) and "://" in policy:
            raise ValueError(
                f"skill {name!r} names a network address ({policy}); a device skill runs on the device itself"
            )
        out[name] = policy
    return out


def _seconds(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"skill seconds= must be a number in (0, {MAX_SKILL_SECONDS}], got {value!r}")
    if not 0 < value <= MAX_SKILL_SECONDS:
        raise ValueError(f"skill seconds= must be a number in (0, {MAX_SKILL_SECONDS}], got {value!r}")
    return float(value)


def _moved(res: Any, refused: bool) -> bool:
    """Whether a finished skill run may have moved the body. Fail safe: only what is known to have moved nothing
    counts as unmoved, namely a refusal before the body was touched, or a result that took no step, held nothing and
    did not fault. A run that raised anything else (a policy error, a bus error) may have moved it mid-run, so it
    counts as moved and the watchdog stays armed for its caller."""
    if res is None:
        return not refused
    return bool(getattr(res, "steps", 0) or getattr(res, "holds", 0) or getattr(res, "fault", None))


class Skills:
    """One device's skills: start one by name, report on it, and end it when the watchdog fires."""

    def __init__(self, robot: Any, device_id: str, catalogue: Mapping[str, Any], dog: Any = None) -> None:
        self.robot, self.device_id, self.catalogue, self.dog = robot, device_id, dict(catalogue), dog
        self._lock = threading.Lock()
        self._task: Any = None
        self._state: dict[str, Any] = {"name": None, "state": "idle"}
        if dog is not None:
            dog.cuts.append(self.cut)

    @property
    def names(self) -> list[str]:
        return sorted(self.catalogue)

    def start(
        self, peer: str | None, name: Any = None, seconds: Any = None, instruction: Any = None, by: Any = None
    ) -> dict[str, Any]:
        """``start_for`` with a caller that showed no beat key, for a transport that proved it: the reply only."""
        return self.start_for(peer, None, True, name=name, seconds=seconds, instruction=instruction, by=by)[0]

    def start_for(
        self,
        peer: str | None,
        beat: str | None,
        proven: bool,
        name: Any = None,
        seconds: Any = None,
        instruction: Any = None,
        by: Any = None,
    ) -> tuple[dict[str, Any], str | None]:
        """Start ``name`` for ``seconds`` on this device and answer at once; ``skill_status`` reports how it went.

        Raises:
            LookupError: the device offers no skills, or none called ``name``.
            ValueError: ``seconds`` is not a number in (0, ``MAX_SKILL_SECONDS``].
            TypeError: ``instruction`` is not a string.
            RuntimeError: a skill (or another run) is already running on this robot.
            EStopped: the robot's e-stop is latched.
            RequestRefused: a real robot was mounted without ``operator=``, so no operator can allow the run.
        """
        if not self.catalogue:
            raise LookupError(f"{self.device_id} offers no skills; mount it with skills={{name: policy}}")
        if name not in self.catalogue:
            raise LookupError(f"{self.device_id} has no skill {name!r}; it offers {self.names}")
        secs = _seconds(seconds)
        if instruction is not None and not isinstance(instruction, str):
            raise TypeError(f"skill instruction= must be a string, got {type(instruction).__name__}")
        from ..robot.run.source import from_agent
        from ..robot.safety import RequestRefused
        from ..robot.safety.move import HINT

        latch = getattr(self.robot, "estop_latch", None)
        if latch is not None:  # said now, not from the run's thread after "started"
            latch.check(self.device_id)
        if getattr(self.robot, "is_real", False) and getattr(self.robot, "operator", None) is None:
            raise RequestRefused(
                f"{self.device_id} is a real robot and a skill is an agent's run: {HINT} on the device, so an "
                "operator at the robot can allow it"
            )
        with self._lock:
            if self._task is not None and self._task.running:
                raise RuntimeError(f"{self.device_id} is already running skill {self._state['name']!r}")
            with from_agent():  # an agent's run: the safety chain on every step, the device's own operator= first
                self._task = task = self.robot.start(self.catalogue[name], seconds=secs, instruction=instruction)
            release, key = (
                self.dog.busy(peer, beat, proven) if self.dog is not None and peer is not None else (None, None)
            )
            self._state = {"name": name, "state": "running", "seconds": secs, "by": by, "started": time.time()}
        threading.Thread(target=self._finish, args=(task, name, release), daemon=True, name="mesh-skill").start()
        return {"started": name, "device": self.device_id, "seconds": secs}, key

    def _finish(self, task: Any, name: str, release: Callable[..., None] | None) -> None:
        state: dict[str, Any] = {"name": name, "state": "done"}
        res, refused = None, False
        try:
            res = task._wait()
            state.update(steps=getattr(res, "steps", None), stopped_by=getattr(res, "stopped_by", None))
            if getattr(res, "fault", None):
                state.update(state="fault", fault=str(res.fault))
        except Exception as exc:  # noqa: BLE001 - a refusal or a policy error is the skill's outcome, reported
            refused = isinstance(exc, PermissionError)  # the operator said no before the body was touched
            state.update(state="refused" if refused else "error", error=f"{exc}")
        finally:
            if release is not None:
                release(_moved(res, refused))
        with self._lock:
            if self._task is task:
                self._state = {**self._state, **state}

    def status(self) -> dict[str, Any]:
        """The current or last skill: ``name``, ``state`` (idle, running, done, fault, refused, error), and once it
        has ended its ``steps`` and ``stopped_by`` (``seconds``, ``estop``, ``stop``, ...)."""
        with self._lock:
            return dict(self._state)

    def cut(self) -> None:
        """End the running skill now (its commander vanished): the watchdog then applies ``on_disconnect``."""
        with self._lock:
            task = self._task
        if task is not None and task.running:
            task.halt(2.0)
