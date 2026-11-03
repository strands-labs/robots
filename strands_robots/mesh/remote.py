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

    def __init__(self, mesh: Mesh, device_id: str, names: tuple[str, ...], *, images: bool = False) -> None:
        self.where = f"remote on {mesh.tenant}"  # Robot's repr: a peer's body, whatever drives it there
        self.supports_position_hold = False  # the registry decides via its type; a peer's manifest cannot grant a hold
        self.calibration, self.resolution = None, None  # the device's own bus owns its calibration
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


def connect(mesh: Mesh, device_id: str, images: bool) -> Any:
    """``Mesh.connect``: a ``Robot`` driven by the device ``device_id``, shaped by its announced manifest."""
    from ..robot import Robot

    found = [d for d in mesh._present() if d.id == device_id]
    if not found:
        raise LookupError(f"no device {device_id!r} in tenant {mesh.tenant!r}")
    manifest = found[0].manifest
    name = str((manifest.get("description") or {}).get("name") or manifest.get("name"))
    names = tuple(str(k) for k in manifest.get("action_keys") or ())
    body = RemoteBody(mesh, device_id, names, images=images)
    return Robot(name, driver=body, twin=False, id=device_id)
