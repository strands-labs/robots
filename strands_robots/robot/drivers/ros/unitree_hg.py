"""Unitree ``unitree_hg`` (G1, H1-2) over unitree_ros2: LowState in, LowCmd out with Unitree's CRC.

LowCmd carries all 35 motor slots: joints this body names get (mode 1, q, kp, kd); every other slot is
sent disabled (mode 0, gains 0). ``mode_machine`` is echoed from LowState, as the MCU requires.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ....ros.convert import from_dict, message_class
from ....types import JointState, Pose
from .codec import Codec
from .crc import hg_lowcmd_crc


class UnitreeHgCodec(Codec):
    needs_gains = True

    def read(self, state: Any) -> JointState:
        ms = state.motor_state
        if max(self.ids) >= len(ms):
            raise ValueError(f"LowState has {len(ms)} motor slots; this body needs slot {max(self.ids)}")
        q = np.array([float(ms[i].q) for i in self.ids])
        dq = np.array([float(ms[i].dq) for i in self.ids])
        tau = np.array([float(getattr(ms[i], "tau_est", 0.0)) for i in self.ids])
        return JointState(self.names, q, dq, tau)

    def build(self, q: np.ndarray, kp: np.ndarray | None, kd: np.ndarray | None, state: Any, sender: str) -> Any:
        if kp is None or kd is None:
            raise ValueError("no gains: LowCmd needs kp/kd per motor (profile gains, policy, or kp=/kd=)")
        cfg = self.profile.need("command")
        n = int(cfg.get("motors", 35))
        rows = [[0, 0.0, 0.0, 0.0, 0.0, 0.0, 0] for _ in range(n)]
        for k, slot in enumerate(self.ids):
            rows[slot] = [1, float(q[k]), 0.0, 0.0, float(kp[k]), float(kd[k]), 0]
        mode_pr, mode_machine = int(cfg.get("mode_pr", 0)), int(getattr(state, "mode_machine", 0))
        msg = from_dict(
            message_class(cfg["type"]),
            {
                "mode_pr": mode_pr,
                "mode_machine": mode_machine,
                "motor_cmd": [{"mode": r[0], "q": r[1], "dq": r[2], "tau": r[3], "kp": r[4], "kd": r[5]} for r in rows],
            },
        )
        msg.crc = hg_lowcmd_crc(mode_pr, mode_machine, rows, list(getattr(msg, "reserve", (0, 0, 0, 0))))
        return msg

    def imu(self, state: Any) -> tuple[np.ndarray, np.ndarray] | None:
        imu = getattr(state, "imu_state", None)
        if imu is None:
            return None
        return np.array([float(x) for x in imu.quaternion]), np.array([float(x) for x in imu.gyroscope])

    def extras(self, state: Any, status: Any) -> dict[str, Any]:
        if state is None:
            return {}
        temps = [max(float(t) for t in m.temperature) for m in state.motor_state if getattr(m, "temperature", None)]
        return {
            "mode_machine": int(getattr(state, "mode_machine", 0)),
            "tick": int(getattr(state, "tick", 0)),
            "max_motor_temp_c": max(temps) if temps else None,
        }

    def positions(self, command: Any) -> np.ndarray | None:
        return np.array([float(command.motor_cmd[i].q) for i in self.ids])

    def pose(self, state: Any, odom: Any) -> Pose | None:
        """Position and world velocity from SportModeState (/odommodestate), attitude from the LowState IMU."""
        if odom is None:
            return super().pose(state, None)
        imu = self.imu(state) if state is not None else None
        q, g = imu if imu is not None else (np.array([1.0, 0, 0, 0]), None)
        return Pose(np.array([float(x) for x in odom.position]), q, np.array([float(x) for x in odom.velocity]), g)
