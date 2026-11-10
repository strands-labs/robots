"""Per-joint bus calibration, in LeRobot's calibration JSON format."""

from __future__ import annotations

import hashlib
import json
import math
import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JointCalibration:
    """One joint's bus calibration — field names and meaning match LeRobot's calibration JSON."""

    id: int
    drive_mode: int
    homing_offset: int
    range_min: int
    range_max: int

    def __post_init__(self) -> None:
        if not self.range_min < self.range_max:  # K8: equal/crossed ends divide by zero on every read
            raise ValueError(
                f"motor {self.id}: range_min={self.range_min} must be < range_max={self.range_max} "
                "(re-run calibrate(): the joint was not swept)"
            )


@dataclass(frozen=True)
class Calibration:
    """Per-joint calibration for a hardware body. ``to_json``/``from_json`` are LeRobot's file format,
    so a calibration written by ``lerobot-calibrate`` loads here unchanged."""

    joints: Mapping[str, JointCalibration]

    def to_json(self) -> dict[str, dict[str, int]]:
        return {name: dict(vars(jc)) for name, jc in self.joints.items()}

    @classmethod
    def from_json(cls, data: Mapping[str, Mapping[str, Any]]) -> Calibration:
        return cls({name: JointCalibration(**{k: int(v) for k, v in jc.items()}) for name, jc in data.items()})

    def check_bus(self, *, resolution: int, max_homing: int, source: str) -> Calibration:
        """This calibration, if every joint fits a bus whose encoder reads ``0..resolution-1`` and whose homing
        offset register holds ``-max_homing..max_homing``; else ``ValueError`` naming the joint, field and file.

        A span outside the encoder's counts would map a joint's limits to goals several turns past its end stops."""
        for name, c in self.joints.items():
            bad = [
                f"{field}={value} (allowed {lo}..{hi})"
                for field, value, lo, hi in (
                    ("range_min", c.range_min, 0, resolution - 1),
                    ("range_max", c.range_max, 0, resolution - 1),
                    ("homing_offset", c.homing_offset, -max_homing, max_homing),
                    ("drive_mode", c.drive_mode, 0, 1),
                )
                if not lo <= value <= hi
            ]
            if bad:
                raise ValueError(
                    f"calibration {source}: joint {name!r} has {', '.join(bad)}; this is not a calibration of "
                    "this arm. Re-run calibrate() or pass calibration=<path> to a file you trust"
                )
        return self

    def check_limits(
        self, limits: Mapping[str, tuple[float, float]], *, resolution: int, source: str, max_scale: float = 1.5
    ) -> Calibration:
        """This calibration, if no joint's span covers more than ``max_scale`` times the counts its ``limits`` take
        on the servo (``resolution`` counts a turn); else ``ValueError`` naming the joint, span and file.

        A driver maps a joint's limits onto its span, so a span wider than the limits put a goal inside them past the
        joint's end stop, and scaled its speed with it. A narrower span only moves the joint less than commanded."""
        for name, c in self.joints.items():
            lo, hi = limits.get(name, (-math.pi, math.pi))
            scale = (c.range_max - c.range_min) / ((hi - lo) * resolution / (2 * math.pi))
            if scale > max_scale:
                raise ValueError(
                    f"calibration {source}: joint {name!r} spans {c.range_min}..{c.range_max} counts, {scale:.2f}x "
                    f"what its limits {lo:.3f}..{hi:.3f} rad take on the servo (at most {max_scale}x): this is not "
                    "a calibration of this joint. Re-run calibrate(), or fix the description's limits"
                )
        return self

    def digest(self) -> str:
        """sha256 of the calibration's values (key order and whitespace do not count): names the calibration in use."""
        return hashlib.sha256(json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# LeRobot 0.5+ writes every SO-10x arm under one robot name (``so_follower`` / ``so_leader``)
_LEROBOT_RENAMED = {f"so10{v}_{r}": f"so_{r}" for v in "01" for r in ("follower", "leader")}


def lerobot_calibration_path(robot_type: str, robot_id: str) -> Path:
    """Where ``lerobot-calibrate --robot.type=<type> --robot.id=<id>`` writes (honours HF_LEROBOT_CALIBRATION)."""
    root = Path(os.environ.get("HF_LEROBOT_CALIBRATION", "~/.cache/huggingface/lerobot/calibration")).expanduser()
    return root / "robots" / robot_type / f"{robot_id}.json"


def lerobot_calibration_candidates(robot_type: str, robot_id: str | None) -> list[Path]:
    """Every file a LeRobot calibration of this arm may be, newest LeRobot naming first.

    Without ``robot_id``: ``<type>.json``, or the only ``*.json`` in the directory (``--robot.id=my_arm``)."""
    out: list[Path] = []
    for rtype in (_LEROBOT_RENAMED.get(robot_type), robot_type):
        if rtype is None:
            continue
        path = lerobot_calibration_path(rtype, robot_id or robot_type)
        only = sorted(path.parent.glob("*.json")) if robot_id is None and path.parent.is_dir() else []
        out += [path] + (only if len(only) == 1 and only[0] != path else [])
    return out


def _find_lerobot_calibration(
    robot_type: str, robot_id: str | None, path: str | Path | None = None
) -> tuple[Path | None, Path | None, list[Path]]:
    """``(found, save_to, looked)``: the file holding this arm's LeRobot calibration (None if none does), the file
    ``calibrate()`` saves to, and every file looked at.

    ``path`` (``calibration=<path>``) is the only file looked at, and a missing one is an error. ``calibrate()``
    saves over the file found, or to LeRobot's ``<type>/<id>.json`` when none is. A file found without an id, as
    the only ``*.json`` in its directory, is a guess and may be another arm's: a save over it would give that arm
    this one's calibration, and one beside it would hide it from every connect without an id, so ``save_to`` is
    None and ``calibrate()`` refuses to record. An id never loads another arm's file: with none of its own, a
    warning names the files that are there."""
    looked = [Path(path).expanduser()] if path is not None else lerobot_calibration_candidates(robot_type, robot_id)
    found = next((p for p in looked if p.is_file()), None)
    if found is not None:
        guessed = path is None and found.stem != (robot_id or robot_type)
        return found, None if guessed else found, looked
    if path is not None:
        raise FileNotFoundError(f"calibration file not found: {looked[0]}")
    if robot_id is not None and (there := sorted({str(f) for p in looked for f in p.parent.glob("*.json")})):
        warnings.warn(
            f"{robot_type}: no calibration file for id {robot_id!r}, only {there}; joints map raw full-turn counts. "
            "Pass calibration=<path>, or the id LeRobot calibrated this arm with",
            stacklevel=3,
        )
    return None, lerobot_calibration_path(robot_type, robot_id or robot_type), looked


def warn_if_missing(found: Calibration | None, who: str, looked: list[str]) -> None:
    """Warn when a driver found no calibration: its joints then read as a raw full turn."""
    if found is None:
        warnings.warn(
            f"{who}: no calibration found (looked at {', '.join(looked) or 'nothing'}); joints read as a full "
            "uncalibrated turn and LeRobot degrees will not match the arm's own. Pass calibration=, robot_id= "
            "(Robot(id=)) or run calibrate().",
            stacklevel=4,
        )
