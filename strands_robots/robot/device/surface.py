"""The device verbs every ``Robot`` has: ``manifest``, ``describe``, ``estop``, ``resume``, ``hold``, ``subscribe``."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

from .build import manifest_for, manifest_key
from .manifest import Manifest

if TYPE_CHECKING:
    from ..robot import Robot


class DeviceVerbs:
    """Mixed into ``Robot``: its device verbs, over the latch, validators and emitter ``Robot.__init__`` creates."""

    @property
    def _robot(self) -> Robot:
        return cast("Robot", self)

    @property
    def manifest(self) -> Manifest:
        """This robot's device manifest, regenerated whenever what it is derived from changes (K4).

        A camera or a twin can be added after the first read; a ``cached_property`` missed them. The cache is keyed
        on ``manifest_key`` (cheap), so the safety chain reading the manifest every step costs a tuple compare.
        """
        key = manifest_key(self._robot)
        cached = self.__dict__.get("_manifest")
        if cached is None or cached[0] != key:
            cached = (key, manifest_for(self._robot))
            self.__dict__["_manifest"] = cached
        return cached[1]

    @property
    def estopped(self) -> bool:
        return self._robot.estop_latch.engaged

    def describe(self) -> dict[str, Any]:
        """The manifest as JSON, plus the live ``task`` and ``estop`` status, and in a world (its own or its twin's)
        the ``reach``: where the farthest link is and how it moves per +1 rad of each joint."""
        d = self.manifest.to_dict()
        d["task"] = None if self._robot._task is None else self._robot._task.status
        d["estop"] = self._robot.estopped
        body = self._robot if self._robot.sim is not None else self._robot.twin
        if body is not None and body.sim is not None:
            with body.sim.lock:
                d["reach"] = body.sim.backend.reach(body.id)
        return d

    def estop(self, *, by: str | None = None) -> None:
        """Stop now: bypasses the safety chain, never waits for the world or the run loop; latches.

        ``by`` names the operator who pressed it; it is recorded on the latch and the ``estop`` event.
        """
        from ..safety import engage

        engage(self._robot, by=by)
        self._robot.events.emit("estop", {"device_id": self._robot.id, "engaged": True, "by": by})

    def resume(self, *, approved_by: str | None = None, stop: int | None = None) -> None:
        """Release the e-stop latch; refused without the operator's name, or for a stop they did not see.

        ``stop`` pins the engagement cleared to ``estop_latch.count``: pass the count shown to the operator
        and a yes typed for an earlier stop is refused once a later one is latched.
        """
        from ..safety import resume

        resume(self._robot, approved_by=approved_by, stop=stop)
        self._robot.events.emit("estop", {"device_id": self._robot.id, "engaged": False, "approved_by": approved_by})

    def hold(self) -> None:
        """Keep the current joint positions (under torque on hardware)."""
        with self._robot._lock:
            self._robot.body.hold()
            self._robot._held = None  # the measured pose is the target now

    def subscribe(
        self, event: str, cb: Callable[[dict[str, Any]], Any], *, hz: float | None = None
    ) -> Callable[[], None]:
        """Call ``cb(payload)`` on each ``event``; ``state`` samples at ``hz``. Returns ``unsubscribe``."""
        return self._robot.events.subscribe(event, cb, hz=hz)
