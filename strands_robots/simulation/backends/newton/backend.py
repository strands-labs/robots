"""The MuJoCo world with MuJoCo-Warp as its solver: built, read and rendered by ``MuJoCoBackend``, stepped on the GPU.

``num_envs`` copies of the world step together; env 0 is mirrored into the CPU ``MjData`` (lazily, on the first
read after a step) so every read, render and contact query keeps MuJoCo's meaning. A private lock serialises the
mirror and the device buffers independently of ``sim.lock``: the e-stop, which takes no world lock, waits for at most
one step, never races a second ``mj_forward`` on the shared ``MjData``.
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from ....types import JointState, Pose
from ...physics import Contact, Physics
from ...scene import Scene, SceneCamera
from ..mujoco import MuJoCoBackend
from .stepper import NewtonStepper


class NewtonBackend(MuJoCoBackend):
    @property
    def deterministic(self) -> bool:  # type: ignore[override]
        return self.num_envs == 1  # one GPU world replays bit-exactly; a batch of envs does not

    def __init__(self, **cfg: Any) -> None:
        super().__init__(**cfg)
        self._stepper = NewtonStepper(self.device, self.num_envs)
        self._stale = False
        self._mirror = threading.RLock()

    def _fresh(self) -> None:
        with self._mirror:
            if self._stale and self.data is not None:
                self._stepper.sync_to(self.data)
                self._stale = False

    def load(self, scene: Scene) -> None:
        with self._mirror:
            self._fresh()
            super().load(scene)
            self._stepper.load(self.model, self.data)

    def step(self, n: int) -> None:
        for _ in range(n):  # one step per hold, so a lock-free e-stop waits for one step, not for n
            with self._mirror:
                self._stepper.step(1)
                self._stale = True

    def reset(self, keyframe: str | None) -> None:
        with self._mirror:
            super().reset(keyframe)
            self._stale = False
            self._stepper.sync_from(self.data)

    def time(self) -> float:
        """Read without the world lock (the watchdog clock): the device scalar, never a write to the CPU MjData."""
        with self._mirror:
            if self._stale and self._stepper.d is not None:
                return float(self._stepper.d.time.numpy()[0])
            return super().time()

    def joint_state(self, body: str) -> JointState:
        with self._mirror:
            self._fresh()
            return super().joint_state(body)

    def set_targets(self, body: str, targets: np.ndarray) -> None:
        with self._mirror:
            super().set_targets(body, targets)
            self._stepper.set_ctrl(self._index[body].actuator_ids, targets)

    def body_pose(self, body: str) -> Pose:
        with self._mirror:
            self._fresh()
            return super().body_pose(body)

    def reach(self, body: str) -> dict[str, Any]:
        with self._mirror:
            self._fresh()
            return super().reach(body)

    def move_camera(self, old: SceneCamera, cam: SceneCamera) -> bool:
        with self._mirror:
            self._fresh()
            return super().move_camera(old, cam)

    def set_gains(self, body: str, kp: np.ndarray, kd: np.ndarray) -> None:
        with self._mirror:
            self._fresh()
            super().set_gains(body, kp, kd)
            self._stepper.load(self.model, self.data)  # the device model is a copy: re-upload the edited gains

    def set_joints(self, body: str, positions: np.ndarray) -> None:
        with self._mirror:
            self._fresh()
            super().set_joints(body, positions)
            self._stepper.sync_from(self.data)

    def set_pose(self, body: str, pose: Pose, *, zero_velocity: bool = True) -> None:
        with self._mirror:
            self._fresh()
            super().set_pose(body, pose, zero_velocity=zero_velocity)
            self._stepper.sync_from(self.data)

    def render(self, camera: str) -> np.ndarray:
        with self._mirror:
            self._fresh()
            return super().render(camera)

    def outline(self) -> np.ndarray | None:
        with self._mirror:
            self._fresh()
            return super().outline()

    def contacts(self) -> list[Contact]:
        with self._mirror:
            self._fresh()
            return super().contacts()

    def physics(self) -> Physics:
        assert self.model is not None
        return Physics("newton", float(self.model.opt.timestep), 1)

    def close(self) -> None:
        with self._mirror:
            self._stepper.close()
            super().close()
