"""``RemoteBody``: a Body whose joints live on another peer. ``Mesh.connect(id)`` wraps one in a ``Robot``.

``read`` and ``pose`` are one ``observe`` round trip; ``write`` is one ``act``, which advances the device's world
one control step and goes through the device's own safety chain. Images travel only when asked for.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from ..types import JointState, Observation, Pose

if TYPE_CHECKING:
    from .mesh import Mesh


class RemoteBody:
    """The device ``device_id`` on ``mesh``, as a Body (and a Driver: hold, estop)."""

    def __init__(
        self, mesh: Mesh, device_id: str, names: tuple[str, ...], *, images: bool = False, position_hold: bool = False
    ) -> None:
        self.supports_position_hold = position_hold  # what the device's manifest declares; False if it does not say
        self.mesh = mesh
        self.device_id = device_id
        self.names = names
        self.with_images = images
        self._last: Observation | None = None

    def _observe(self) -> Observation:
        obs: Observation = self.mesh.invoke(self.device_id, "observe", images=self.with_images)
        self._last = obs
        return obs

    def read(self) -> JointState:
        return self._observe().joints

    def write(self, targets: np.ndarray) -> None:
        self._last = self.mesh.invoke(self.device_id, "act", action=np.asarray(targets, dtype=float))

    def pose(self) -> Pose | None:
        return (self._last or self._observe()).pose

    def images(self) -> Mapping[str, np.ndarray]:
        return (self._last or self._observe()).images if self.with_images else {}

    def extras(self) -> Mapping[str, Any]:
        return {"device_id": self.device_id}

    def hold(self) -> None:
        self.mesh.invoke(self.device_id, "hold")

    def estop(self) -> None:
        self.mesh.invoke(self.device_id, "estop", by=self.mesh.operator)

    def calibrate(self, *, interactive: bool = True) -> Any:
        raise NotImplementedError(f"{self.device_id} is remote: calibrate it on the machine its bus is on")

    def close(self) -> None:
        self._last = None
