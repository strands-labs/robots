"""Unitree G1 / Go2 / H1 over DDS (unitree_sdk2py): the READ path, ``rt/lowstate`` -> JointState + IMU.

``unitree_sdk2py`` is imported lazily and only when no ``subscriber`` is injected. Commanding joints
(``rt/lowcmd`` with kp/kd) is deliberately not wired in v1 core: it needs a live robot to measure gains
against, and this release does not command one. ``write`` raises ``NotImplementedError`` saying so.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from ...description import Description
from ...types import JointState, Pose
from .calibration import Calibration

G1_ORDER: tuple[str, ...] = (
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
)
GO2_ORDER: tuple[str, ...] = (
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
)
# The H1 (not the H1-2) speaks the Go2's ``unitree_go`` IDL; its slots are the SDK's ``H1JointIndex``
# (example/h1/low_level in unitree_sdk2_python): the wire starts at right_hip_roll, slot 9 has no motor.
H1_SLOTS: dict[str, int] = {
    "right_hip_roll": 0,
    "right_hip_pitch": 1,
    "right_knee": 2,
    "left_hip_roll": 3,
    "left_hip_pitch": 4,
    "left_knee": 5,
    "torso": 6,
    "left_hip_yaw": 7,
    "right_hip_yaw": 8,
    "left_ankle": 10,
    "right_ankle": 11,
    "right_shoulder_pitch": 12,
    "right_shoulder_roll": 13,
    "right_shoulder_yaw": 14,
    "right_elbow": 15,
    "left_shoulder_pitch": 16,
    "left_shoulder_roll": 17,
    "left_shoulder_yaw": 18,
    "left_elbow": 19,
}
_HG, _GO = "unitree_sdk2py.idl.unitree_hg.msg.dds_", "unitree_sdk2py.idl.unitree_go.msg.dds_"
_IDL: dict[str, tuple[str, Mapping[str, int]]] = {
    "g1": (_HG, {n: i for i, n in enumerate(G1_ORDER)}),
    "go2": (_GO, {n: i for i, n in enumerate(GO2_ORDER)}),
    "h1": (_GO, H1_SLOTS),
}


def _family(description: Description) -> str:
    fam = description.name.lower().removeprefix("unitree_")
    if fam not in _IDL:
        raise ValueError(f"unitree driver: unknown robot {description.name!r} (know {sorted(_IDL)})")
    return fam


class DDSSubscriber:
    """Keeps the latest ``rt/lowstate`` sample. Real SDK behind ``connect``; tests inject an object with ``.read()``."""

    def __init__(self, interface: str, idl_module: str) -> None:
        try:
            from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        except ImportError as exc:
            raise ImportError(
                "the unitree driver needs unitree_sdk2py, which is not on PyPI: "
                "pip install git+https://github.com/unitreerobotics/unitree_sdk2_python"
            ) from exc
        import importlib

        ChannelFactoryInitialize(0, interface)
        low_state = importlib.import_module(idl_module).LowState_
        self._latest: Any = None
        self._lock = threading.Lock()
        self._sub = ChannelSubscriber("rt/lowstate", low_state)
        self._sub.Init(self._on, 10)

    def _on(self, msg: Any) -> None:
        with self._lock:
            self._latest = msg

    def read(self) -> Any:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self._sub.Close()


class UnitreeDriver:
    """Hardware Body for a Unitree robot: joints from ``rt/lowstate``, IMU + battery in ``extras``. Observe-only."""

    supports_position_hold = False  # observe-only: nothing of ours writes a goal
    calibration: Calibration | None = None  # the vendor firmware owns it
    resolution: int | None = None

    def __init__(
        self,
        description: Description,
        *,
        port: str | None = None,  # the network interface, e.g. "eth0" / "en7"
        cameras: Mapping[str, int | str] | None = None,
        subscriber: Any = None,
        timeout: float = 2.0,
    ) -> None:
        self.description = description
        self.names = description.action_keys
        fam = _family(description)
        idl, slots = _IDL[fam]
        unknown = [n for n in self.names if n.removesuffix("_joint") not in slots]
        if unknown:
            raise ValueError(f"unitree {fam}: joints not in the SDK motor order: {unknown}")
        self.index = [slots[n.removesuffix("_joint")] for n in self.names]
        self.sub = subscriber if subscriber is not None else DDSSubscriber(port or "eth0", idl)
        self.cameras_cfg = dict(cameras or {})
        end = time.monotonic() + timeout
        while self.sub.read() is None and time.monotonic() < end:
            time.sleep(0.02)
        if self.sub.read() is None:
            raise ConnectionError(f"unitree {fam}: no rt/lowstate sample within {timeout}s on {port or 'eth0'}")

    def _state(self) -> Any:
        msg = self.sub.read()
        if msg is None:
            raise ConnectionError("unitree: lowstate stream went silent")
        return msg

    def read(self) -> JointState:
        ms = self._state().motor_state
        q = np.array([float(ms[i].q) for i in self.index])
        dq = np.array([float(ms[i].dq) for i in self.index])
        tau = np.array([float(getattr(ms[i], "tau_est", 0.0)) for i in self.index])
        return JointState(self.names, q, dq, tau)

    def write(self, targets: np.ndarray) -> None:
        raise NotImplementedError(
            "unitree: commanding joints over rt/lowcmd is not wired in v1 core (needs a live robot to measure "
            "kp/kd against); this driver is observe-only"
        )

    def pose(self) -> Pose | None:
        return None  # DDS lowstate has no world position; orientation is in extras["imu"]["quaternion"]

    def images(self) -> Mapping[str, np.ndarray]:
        return {}

    def extras(self) -> Mapping[str, Any]:
        s = self._state()
        imu = getattr(s, "imu_state", None)
        out: dict[str, Any] = {}
        if imu is not None:
            out["imu"] = {k: [float(x) for x in getattr(imu, k)] for k in ("quaternion", "gyroscope", "accelerometer")}
        for f in ("power_v", "power_a", "mode_machine", "tick"):
            if hasattr(s, f):
                out[f] = float(getattr(s, f)) if f != "mode_machine" else int(getattr(s, f))
        return out

    def hold(self) -> None:
        """Observe-only: this driver never commands a joint, so there is nothing of ours to hold."""

    def estop(self) -> None:
        """Observe-only: nothing of ours is moving the body; the vendor's own stop stays in charge."""

    def calibrate(self, *, interactive: bool = True) -> Calibration:
        raise RuntimeError("unitree robots are factory-calibrated; nothing to calibrate")

    def close(self) -> None:
        self.sub.close()
