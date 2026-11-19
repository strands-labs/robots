"""The Feetech hardware Body: read Present_Position, write Goal_Position, torque, LeRobot-format calibration."""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ....description import Description
from ....types import JointState, Pose
from .. import calibration as calib
from ..calibration import Calibration, JointCalibration, lerobot_calibration_path
from ..cameras import Cameras
from ..slew import MAX_SPEED, Slew
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
from .readout import MOTION_BYTES, POSITION_BYTES, joint_state
from .sweep import sweep_ranges


class FeetechDriver:
    """A hardware Body for an SO-10x class arm. Joint values are radians, mapped through the calibration."""

    supports_position_hold = True  # an STS servo keeps Goal_Position while torque is on
    resolution = RESOLUTION  # encoder counts per turn
    num_read_retries = 2  # LeRobot's SO follower default, from huggingface/lerobot#4207

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
        max_speed: float | None = MAX_SPEED,
        disable_torque_on_disconnect: bool = False,
        read_motion: bool = False,
    ) -> None:
        """``max_speed`` (rad/s) caps one write's move from the last goal (``slew.Slew``); ``None``: as given.
        ``read_motion``: every read also asks for velocity and load (6 bytes per servo, not 2); else they are NaN."""
        if port is None and bus is None:
            raise ValueError("FeetechDriver needs port= (e.g. /dev/tty.usbmodem...)")
        self.slew = Slew(max_speed)
        self.read_motion = read_motion
        self.description = description
        self.names = description.action_keys
        # motor ids: the registry labels model joints "1".."6" -> the label; fall back to 1..n in action order
        unlabelled = [description.unlabel(n) for n in self.names]
        self.ids = [int(u) if u.isdigit() else i + 1 for i, u in enumerate(unlabelled)]
        self.bus = bus or FeetechBus(port or "", baud=baud)
        self.cameras = Cameras(cameras)
        self._torque_on = False
        self.disable_torque_on_disconnect = disable_torque_on_disconnect  # K8: default False never drops the arm
        self.calibration: Calibration | None = None
        self.calibration_source: str | None = None
        self.calibration_looked: list[str] = []
        self._save_to: Path | None = self.default_calibration_path(robot_id)  # calibrate()'s file; None: no safe one
        self._load_calibration(calibration, robot_id)
        calib.warn_if_missing(self.calibration, f"{description.name} on {self.bus.port}", self.calibration_looked)
        if missing := [i for i in self.ids if not self.bus.ping(i)]:
            raise ConnectionError(f"{description.name} on {self.bus.port}: no reply from servo id(s) {missing}")

    def _load_calibration(self, cal: Any, robot_id: str | None) -> None:
        if isinstance(cal, (Calibration, Mapping)):
            self.calibration = cal if isinstance(cal, Calibration) else Calibration.from_json(cal)
            self.calibration_source = "caller"
        else:
            rtype = str(self.description.hardware.lerobot_type or self.description.name)
            path, self._save_to, looked = calib._find_lerobot_calibration(rtype, robot_id, cal)
            self.calibration_looked = [str(p) for p in looked]
            if path is not None:
                self.calibration = Calibration.from_json(json.loads(path.read_text()))
                self.calibration_source = str(path)
        if self.calibration is not None:
            self._check(self.calibration, str(self.calibration_source))

    def _check(self, cal: Calibration, source: str) -> None:
        cal.check_bus(resolution=RESOLUTION, max_homing=2047, source=source)
        cal.check_limits(self.description.limits, resolution=RESOLUTION, source=source)

    def default_calibration_path(self, robot_id: str | None = None) -> Path:
        """Where ``calibrate()`` saves when no file was loaded: LeRobot's ``<type>/<id>.json`` for this arm."""
        rtype = str(self.description.hardware.lerobot_type or self.description.name)
        return lerobot_calibration_path(rtype, robot_id or rtype)

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
        """Present_Position only (2 bytes per servo); with ``read_motion`` velocity and load too (6 bytes)."""
        n = MOTION_BYTES if self.read_motion else POSITION_BYTES
        js = joint_state(self, self.bus.sync_read(REG_PRESENT_POSITION, n, self.ids, retries=self.num_read_retries))
        self.slew.seen(np.array(js.positions))
        return js

    def write(self, targets: np.ndarray) -> None:
        # K8: every goal is computed (a non-finite target refused) BEFORE torque comes on, so a refused write stays limp
        if not np.all(np.isfinite(np.asarray(targets, dtype=float))):
            raise ValueError(f"{self.description.name}: non-finite goal {np.asarray(targets).tolist()}")
        targets = self.slew.apply(targets, self.read)
        goals = {
            mid: word(self.rad_to_counts(n, float(t))) for n, mid, t in zip(self.names, self.ids, targets, strict=True)
        }
        self.torque(True)  # a no-op once on; the first time, every servo is held where it is before torque comes on
        self.bus.sync_write(REG_GOAL_POSITION, 2, goals)

    def pose(self) -> Pose | None:
        return None

    def images(self) -> Mapping[str, np.ndarray]:
        return self.cameras.read()

    def extras(self) -> Mapping[str, Any]:
        return {
            "torque": self._torque_on,
            "max_speed": self.slew.max_speed,
            "read_motion": self.read_motion,
            "slewed": self.slew.count,
            "calibration": self.calibration_source or "none (raw full-turn counts)",
            "calibration_sha256": self.calibration.digest() if self.calibration is not None else None,
        }

    def hold(self) -> None:
        """Goal_Position := Present_Position under torque: the measured pose, never speed-capped (K8)."""
        self.slew.reset(present := np.array(self.read().positions, dtype=float))
        self.write(present)

    def estop(self) -> None:
        """Fastest stop that does not drop the arm: a hold if torque is on; a limp arm is left limp."""
        if self._torque_on:
            self.hold()

    def torque(self, on: bool) -> None:
        if on and not self._torque_on:  # Goal := Present first (FeetechBus.engage): torque never meets a stale goal
            self.bus.engage(self.ids, retries=self.num_read_retries)
        elif not on:
            self.bus.torque(self.ids, False)
            self.slew.release()
        self._torque_on = on

    def calibrate(
        self, *, interactive: bool = True, save: bool = True, prompt: Any = input, sweep_seconds: float = 20.0
    ) -> Calibration:
        """LeRobot's procedure: torque off, middle pose sets the homing offset, sweep records min/max.

        ``interactive=False`` returns the calibration already loaded (LeRobot's file) and moves nothing. With one
        loaded, ``interactive=True`` first asks, as LeRobot does: Enter keeps it, ``c`` records a new one."""
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
        if self.calibration is not None:  # ask first, as LeRobot does; unlike LeRobot's, this Enter writes nothing
            said = f"{self.description.name} has a calibration ({self.calibration_source}). Enter keeps it"
            if str(prompt(said + ", c re-records (torque goes off)") or "").strip().lower() != "c":
                return self.calibration
        if save and self._save_to is None:  # over or beside a guess, the save could give another arm this calibration
            raise RuntimeError(f"with no id, {self.calibration_source} may be another arm's: pass id= or calibration=")
        self.torque(False)
        prompt(f"Move {self.description.name} to the MIDDLE of every joint's range, then press Enter")
        was = {i: decode_sm(self.bus.read(i, REG_HOMING_OFFSET, 2), 11) for i in self.ids}  # put back if refused
        for i in self.ids:  # only after that Enter; Present = Actual - Homing_Offset, so read the middle unshifted
            self._write_homing(i, 0)
        mid = {i: decode_sm(self.bus.read(i, REG_PRESENT_POSITION, 2), 15) for i in self.ids}
        homing = {i: max(-2047, min(2047, m - (RESOLUTION - 1) // 2)) for i, m in mid.items()}  # as LeRobot clamps
        for i, h in homing.items():
            self._write_homing(i, h)
        prompt(f"Now move every joint but wrist_roll through its full range for {sweep_seconds:.0f} s; Enter to start")
        lo, hi = sweep_ranges(self.bus, self.ids, self.names, sweep_seconds)
        try:  # the driver keeps the calibration it had: on a refusal the servos get their offsets back
            if unswept := [n for n, i in zip(self.names, self.ids, strict=True) if lo[i] >= hi[i]]:  # as LeRobot
                raise ValueError(
                    f"{self.description.name}: the sweep recorded no range for {unswept}: nothing saved. Run "
                    "calibrate() again and move each of them through its full range"
                )
            cal = Calibration(
                {n: JointCalibration(i, 0, homing[i], lo[i], hi[i]) for n, i in zip(self.names, self.ids, strict=True)}
            )
            self._check(cal, "calibrate()")
        except ValueError:
            for i, h in was.items():
                self._write_homing(i, h)
            raise
        self.calibration, self.calibration_source = cal, "calibrate()"
        if save and self._save_to is not None:  # the file it loaded, else <type>/<id>.json: the next connect's file
            self._save_to.parent.mkdir(parents=True, exist_ok=True)
            self._save_to.write_text(json.dumps(cal.to_json(), indent=4))
            self.calibration_source = str(self._save_to)
        return cal

    def _write_homing(self, motor_id: int, offset: int) -> None:
        self.bus.write(motor_id, REG_LOCK, b"\x00")
        self.bus.write(motor_id, REG_HOMING_OFFSET, encode_sm(offset, 11))
        self.bus.write(motor_id, REG_LOCK, b"\x01")

    def close(self) -> None:
        """Holds a torqued arm (K8; ``disable_torque_on_disconnect`` then releases); port + cameras close anyway."""
        try:
            if self._torque_on:
                self.hold()
                if self.disable_torque_on_disconnect:
                    self.torque(False)
        finally:
            self.cameras.close()
            self.bus.close()
