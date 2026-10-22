"""The device verbs every ``Robot`` has: ``manifest``, ``describe``, ``estop``, ``resume``, ``hold``, ``subscribe``."""

from __future__ import annotations

from collections.abc import Callable
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

from .build import manifest_for
from .manifest import Manifest

if TYPE_CHECKING:
    from ..robot import Robot


class DeviceVerbs:
    """Mixed into ``Robot``: its device verbs, over the latch, validators and emitter ``Robot.__init__`` creates."""

    @property
    def _robot(self) -> Robot:
        return cast("Robot", self)

    @cached_property
    def manifest(self) -> Manifest:
        """This robot's device manifest (generated once, cached)."""
        return manifest_for(self._robot)

    @property
    def estopped(self) -> bool:
        return self._robot.estop_latch.engaged

    def describe(self) -> dict[str, Any]:
        """The manifest as JSON, plus the live ``task`` and ``estop`` status."""
        d = self.manifest.to_dict()
        d["task"] = None if self._robot._task is None else self._robot._task.status
        d["estop"] = self._robot.estopped
        return d

    def estop(self, *, by: str | None = None) -> None:
        """Stop now: bypasses the safety chain, never waits for the world or the run loop; latches.

        ``by`` names the operator who pressed it; it is recorded on the latch and the ``estop`` event. A clock that
        raises while the latch is stamped never hides the stop: ``estop`` goes out, then a ``fault`` naming the error.
        """
        from ..safety import engage

        robot = self._robot
        clock_error = engage(robot, by=by)
        robot.events.emit("estop", {"device_id": robot.id, "engaged": True, "by": by})
        if clock_error is not None:  # latched and stopped; only estop_latch.engaged_at is missing
            why = f"estop: latched and stopped, but robot.clock raised {clock_error!r}; estop_latch.engaged_at is None"
            robot.events.emit("fault", {"device_id": robot.id, "reason": why})

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

    def subscribe(
        self, event: str, cb: Callable[[dict[str, Any]], Any], *, hz: float | None = None
    ) -> Callable[[], None]:
        """Call ``cb(payload)`` on each ``event``; ``state`` samples at ``hz``. Returns ``unsubscribe``."""
        return self._robot.events.subscribe(event, cb, hz=hz)
