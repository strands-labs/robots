"""``sim.tool``: the world's verbs as a device, generated from the Simulation's public methods.

Every public method whose arguments can all cross a wire is an rpc, except the lifecycle ones a model must not
drive (``close``, ``manual_time``, ``unique_id``). ``add_robot`` is ``Robot(name, sim=world, ...)`` with only the
placement arguments: an agent places simulated bodies; it never mounts hardware (port, driver, confirm, cameras).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .schema import NotOnTheWire, parameters

if TYPE_CHECKING:
    from ..simulation import Simulation

LIFECYCLE = frozenset({"close", "manual_time", "unique_id"})
ROBOT_PLACEMENT = ("name", "position", "orientation", "keyframe", "id")


@dataclass(frozen=True)
class _Rpc:
    name: str


@dataclass(frozen=True)
class _Verbs:
    rpcs: tuple[_Rpc, ...]


def world_verbs(sim: Simulation) -> tuple[str, ...]:
    names = ["add_robot"]
    for name, attr in inspect.getmembers(type(sim)):
        if name.startswith("_") or name in LIFECYCLE or not inspect.isfunction(attr):
            continue
        try:
            parameters(getattr(sim, name), joints=())
        except NotOnTheWire:
            continue
        names.append(name)
    return tuple(names)


class WorldDevice:
    """A Simulation seen as a device: ``id``, a generated rpc list, and its methods (plus ``add_robot``)."""

    action_keys: tuple[str, ...] = ()

    def __init__(self, sim: Simulation) -> None:
        self.sim = sim
        self.id = "world"
        self.manifest = _Verbs(tuple(_Rpc(n) for n in world_verbs(sim)))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sim, name)

    def add_robot(
        self,
        name: str,
        *,
        position: tuple[float, float, float] | None = None,
        orientation: tuple[float, float, float, float] | None = None,
        keyframe: str | None = None,
        id: str | None = None,
    ) -> dict[str, Any]:
        """Put a simulated robot from the registry into this world; returns its describe()."""
        from ..robot import Robot

        return Robot(
            name, sim=self.sim, position=position, orientation=orientation, keyframe=keyframe, id=id
        ).describe()
