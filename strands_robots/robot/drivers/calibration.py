"""Per-joint bus calibration, in LeRobot's calibration JSON format."""

from __future__ import annotations

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
