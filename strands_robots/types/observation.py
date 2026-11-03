"""What a body reports at time t."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .joints import JointState, _eq
from .pose import Pose


@dataclass(frozen=True, eq=False)
class Observation:
    """What a body reports at time ``t``: joints, base pose, camera images and per-body extras.

    ``==`` compares every field by value (images and extras too); ``hash`` reads ``t``, joints and pose.
    """

    t: float
    joints: JointState
    pose: Pose | None = None
    images: Mapping[str, np.ndarray] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)

    __eq__ = _eq

    def __hash__(self) -> int:
        return hash((self.t, self.joints, self.pose))

    def __getitem__(self, joint: str) -> float:
        return self.joints[joint]

    def with_joints(self, positions: np.ndarray) -> Observation:
        """The same observation with joint positions replaced (used for unit conversion before a policy)."""
        return Observation(
            t=self.t,
            joints=JointState(
                self.joints.names, np.asarray(positions, dtype=float), self.joints.velocities, self.joints.efforts
            ),
            pose=self.pose,
            images=self.images,
            extras=self.extras,
        )

    def to_lerobot(self) -> dict[str, Any]:
        """``observation.state`` (float32 vector) + ``observation.images.<cam>`` (HxWx3 uint8)."""
        out: dict[str, Any] = {"observation.state": self.joints.positions.astype(np.float32)}
        for name, img in self.images.items():
            out[f"observation.images.{name}"] = img
        return out
