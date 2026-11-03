"""A Body is something with joints you can read and targets you can write.

``SimBody`` is a handle into a Simulation; a ``Driver`` (robot.drivers) is the hardware Body. Every
loop is written once against ``Body`` and never knows which it has.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from ..types import JointState, Pose

if TYPE_CHECKING:
    from ..simulation import Simulation


@runtime_checkable
class Body(Protocol):
    """Something with joints you can read and targets you can write."""

    names: tuple[str, ...]

    def read(self) -> JointState: ...

    def write(self, targets: np.ndarray) -> None: ...

    def pose(self) -> Pose | None: ...

    def images(self) -> Mapping[str, np.ndarray]: ...

    def extras(self) -> Mapping[str, Any]: ...

    def hold(self) -> None: ...

    def estop(self) -> None: ...

    def close(self) -> None: ...


class SimBody:
    """A handle into a Simulation: the body named ``id`` in ``sim``."""

    def __init__(self, sim: Simulation, id: str, names: tuple[str, ...]) -> None:
        self.sim = sim
        self.id = id
        self.names = names

    def read(self) -> JointState:
        return self.sim.backend.joint_state(self.id)

    def write(self, targets: np.ndarray) -> None:
        self.sim.backend.set_targets(self.id, targets)

    def pose(self) -> Pose | None:
        return self.sim.backend.body_pose(self.id)

    def images(self) -> Mapping[str, np.ndarray]:
        return {c.name: self.sim.backend.render(c.name) for c in self.sim.scene.cameras if c.robot == self.id}

    def extras(self) -> Mapping[str, Any]:
        return {}

    def hold(self) -> None:
        """Targets := the measured joint positions."""
        self.sim.backend.set_targets(self.id, np.array(self.read().positions, dtype=float))

    def estop(self) -> None:
        """A simulated body's fastest stop is a hold; the latch lives on the Robot. Takes no lock."""
        self.hold()

    def close(self) -> None:
        pass
