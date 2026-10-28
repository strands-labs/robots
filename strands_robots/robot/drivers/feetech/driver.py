"""The Feetech hardware Body: read Present_Position, write Goal_Position, torque, LeRobot-format calibration."""

from __future__ import annotations

import json
import math
import sys
import time
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ....description import Description
from ....types import JointState, Pose
from ..calibration import Calibration, JointCalibration, lerobot_calibration_candidates, lerobot_calibration_path
from ..cameras import Cameras
from .bus import (
    REG_GOAL_POSITION,
    REG_HOMING_OFFSET,
    REG_LOCK,
    REG_PRESENT_POSITION,
    RESOLUTION,
    FeetechBus,
    decode_sm,
    encode_sm,
    word,
)


class FeetechDriver:
    """A hardware Body for an SO-10x class arm. Joint values are radians, mapped through the calibration."""

    supports_position_hold = True  # an STS servo keeps Goal_Position while torque is on
    resolution = RESOLUTION  # encoder counts per turn

    def __init__(
        self,
        description: Description,
        *,
        port: str | None,
        cameras: Mapping[str, Any] | None = None,
        baud: int = 1_000_000,
        calibration: str | Path | Mapping[str, Any] | Calibration | None = None,
        robot_id: str | None = None,
        bus: FeetechBus | None = None,
    ) -> None:
        if port is None and bus is None:
            raise ValueError("FeetechDriver needs port= (e.g. /dev/tty.usbmodem...)")
        self.description = description
        self.names = description.action_keys
        # motor ids: the registry labels model joints "1".."6" -> the label; fall back to 1..n in action order
        self.ids = [
            int(description.unlabel(n)) if description.unlabel(n).isdigit() else i + 1 for i, n in enumerate(self.names)
        ]
        self.bus = bus or FeetechBus(port or "", baud=baud)
        self.cameras = Cameras(cameras)
        self._torque_on = False
        self.calibration: Calibration | None = None
        self.calibration_source: str | None = None
        self.calibration_looked: list[str] = []
        self._load_calibration(calibration, robot_id)
        if self.calibration is None:
            warnings.warn(
                f"{description.name} on {self.bus.port}: no calibration found (looked at "
                f"{', '.join(self.calibration_looked) or 'nothing'}); joints read as a full uncalibrated turn and "
                f"LeRobot degrees will not match the arm's own. Pass calibration=, robot_id= (Robot(id=)) or run "
                f"calibrate().",
                stacklevel=3,
            )
        missing = [i for i in self.ids if not self.bus.ping(i)]
        if missing:
            raise ConnectionError(f"{description.name} on {self.bus.port}: no reply from servo id(s) {missing}")

    def _load_calibration(self, cal: Any, robot_id: str | None) -> None:
        if isinstance(cal, Calibration):
            self.calibration, self.calibration_source = cal, "caller"
        elif isinstance(cal, Mapping):
            self.calibration, self.calibration_source = Calibration.from_json(cal), "caller"
        else:
            looked = [Path(cal).expanduser()] if cal is not None else self.calibration_candidates(robot_id)
            path = next((p for p in looked if p.is_file()), looked[0])
            self.calibration_looked = [str(p) for p in looked]
            if path.is_file():
                self.calibration = Calibration.from_json(json.loads(path.read_text()))
                self.calibration_source = str(path)
            elif cal is not None:
                raise FileNotFoundError(f"calibration file not found: {path}")

    def default_calibration_path(self, robot_id: str | None = None) -> Path:
        """Where ``calibrate()`` saves: LeRobot's ``<type>/<id>.json`` for this arm."""
        rtype = str(self.description.hardware.lerobot_type or self.description.name)
        return lerobot_calibration_path(rtype, robot_id or rtype)

    def calibration_candidates(self, robot_id: str | None = None) -> list[Path]:
        return lerobot_calibration_candidates(
            str(self.description.hardware.lerobot_type or self.description.name), robot_id
        )

    def _span(self, name: str) -> tuple[float, float, float, float, bool]:
        """(count_min, count_max, rad_lo, rad_hi, reversed) for a joint, or the uncalibrated full turn."""
        lo, hi = self.description.limits.get(name, (-math.pi, math.pi))
        if self.calibration is not None and name in self.calibration.joints:
            c = self.calibration.joints[name]
            return float(c.range_min), float(c.range_max), lo, hi, bool(c.drive_mode)
        return 0.0, float(RESOLUTION), -math.pi, math.pi, False  # 2048 counts = 0 rad, like lerobot DEGREES mode

    def counts_to_rad(self, name: str, counts: int) -> float:
        cmin, cmax, lo, hi, rev = self._span(name)
        f = (counts - cmin) / (cmax - cmin)
        if rev:
            f = 1.0 - f
        return lo + f * (hi - lo)

    def rad_to_counts(self, name: str, rad: float) -> int:
        if not math.isfinite(rad):  # NaN would round to range_min: a full-speed move to the end stop
            raise ValueError(f"{self.description.name}.{name}: non-finite goal {rad}")
        cmin, cmax, lo, hi, rev = self._span(name)
        f = (rad - lo) / (hi - lo) if hi != lo else 0.0
        f = min(1.0, max(0.0, 1.0 - f if rev else f))
        return int(round(cmin + f * (cmax - cmin)))

    # Body

    def read(self) -> JointState:
        raw = self.bus.sync_read(REG_PRESENT_POSITION, 6, self.ids)
        pos, vel, load = [], [], []
        for name, mid in zip(self.names, self.ids, strict=True):
            r = raw[mid]
            counts = decode_sm(r[0:2], 15)
            pos.append(self.counts_to_rad(name, counts))
            cmin, cmax, lo, hi, rev = self._span(name)
            vel.append(decode_sm(r[2:4], 15) * (hi - lo) / (cmax - cmin) * (-1 if rev else 1))
            load.append(decode_sm(r[4:6], 10) / 1000.0)  # fraction of max torque, signed
        return JointState(self.names, np.array(pos), np.array(vel), np.array(load))

    def write(self, targets: np.ndarray) -> None:
        # Every goal is computed (and a non-finite target refused) BEFORE torque comes on, so a refused
        # write leaves a limp arm limp instead of stiffening it against a stale Goal_Position.
        goals = {
            mid: word(self.rad_to_counts(n, float(t))) for n, mid, t in zip(self.names, self.ids, targets, strict=True)
        }
        if not self._torque_on:
            self.bus.torque(self.ids, True)
            self._torque_on = True
        self.bus.sync_write(REG_GOAL_POSITION, 2, goals)

    def pose(self) -> Pose | None:
        return None

    def images(self) -> Mapping[str, np.ndarray]:
        return self.cameras.read()

    def extras(self) -> Mapping[str, Any]:
        return {"torque": self._torque_on, "calibration": self.calibration_source or "none (raw full-turn counts)"}

    def hold(self) -> None:
        """Goal_Position := Present_Position, under torque (gravity-loaded joints keep their pose)."""
        self.write(np.array(self.read().positions, dtype=float))

    def estop(self) -> None:
        """Fastest stop that does not drop the arm: a hold if torque is on; a limp arm is left limp."""
        if self._torque_on:
            self.hold()

    def torque(self, on: bool) -> None:
        self.bus.torque(self.ids, on)
        self._torque_on = on

    def calibrate(
        self, *, interactive: bool = True, save: bool = True, prompt: Any = input, sweep_seconds: float = 20.0
    ) -> Calibration:
        """LeRobot's procedure: torque off, middle pose sets the homing offset, sweep records min/max.

        ``interactive=False`` returns the calibration already loaded (LeRobot's file) and moves nothing."""
        if not interactive:
            if self.calibration is None:
                raise RuntimeError(
                    f"no LeRobot calibration for {self.description.name}; looked at {self.calibration_looked}. "
                    "Pass Robot(..., calibration=<path to lerobot's json>), or record one at the arm with "
                    "calibrate(interactive=True)"
                )
            return self.calibration
        if prompt is input and not sys.stdin.isatty():  # torque off drops the arm: only a human at the arm says go
            raise RuntimeError(f"{self.description.name}: calibrate(interactive=True) needs a human at a terminal")
        self.torque(False)
        for i in self.ids:  # the firmware reports Present = Actual - Homing_Offset: read the middle unshifted
            self._write_homing(i, 0)
        prompt(f"Move {self.description.name} to the MIDDLE of every joint's range, then press Enter")
        mid = {i: decode_sm(self.bus.read(i, REG_PRESENT_POSITION, 2), 15) for i in self.ids}
        limit = (1 << 11) - 1  # sign-magnitude, bit 11: LeRobot clamps pos - 2047 the same way
        homing = {i: max(-limit, min(limit, m - (RESOLUTION - 1) // 2)) for i, m in mid.items()}
        for i, h in homing.items():
            self._write_homing(i, h)
        prompt(f"Now move EVERY joint through its full range for {sweep_seconds:.0f} s (recording); Enter to start")
        lo = {i: RESOLUTION for i in self.ids}
        hi = {i: 0 for i in self.ids}
        end = time.monotonic() + sweep_seconds
        while time.monotonic() < end:
            raw = self.bus.sync_read(REG_PRESENT_POSITION, 2, self.ids)
            for i in self.ids:
                v = decode_sm(raw[i], 15)
                lo[i], hi[i] = min(lo[i], v), max(hi[i], v)
            time.sleep(0.01)
        cal = Calibration(
            {n: JointCalibration(i, 0, homing[i], lo[i], hi[i]) for n, i in zip(self.names, self.ids, strict=True)}
        )
        self.calibration, self.calibration_source = cal, "calibrate()"
        if save:
            path = self.default_calibration_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(cal.to_json(), indent=4))
            self.calibration_source = str(path)
        return cal

    def _write_homing(self, motor_id: int, offset: int) -> None:
        self.bus.write(motor_id, REG_LOCK, b"\x00")
        self.bus.write(motor_id, REG_HOMING_OFFSET, encode_sm(offset, 11))
        self.bus.write(motor_id, REG_LOCK, b"\x01")

    def close(self) -> None:
        if self._torque_on:
            self.bus.torque(self.ids, False)
            self._torque_on = False
        self.cameras.close()
        self.bus.close()
