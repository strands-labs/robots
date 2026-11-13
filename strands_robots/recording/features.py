"""LeRobot features for one robot: ``<action key>.pos`` state and action, one video feature per camera, and the
measured capture time of every frame."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..description import Description
from ..types import Observation
from ..units import Units

STATE, ACTION, IMAGES = "observation.state", "action", "observation.images."
# LeRobot's own ``timestamp`` is frame_index / fps by construction (its video frames are found by it). This one is when
# each frame was read, so a stalled or skipped tick shows. No ``observation``/``action`` prefix: never a policy input.
TIME = "time.measured"


def lerobot_names(keys: tuple[str, ...] | list[str]) -> list[str]:
    return [f"{k}.pos" for k in keys]


def features_from(first: Observation, action_keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """The dataset schema, read off the first observation the policy saw."""
    feats: dict[str, dict[str, Any]] = {
        STATE: {"dtype": "float32", "shape": (len(first.joints.names),), "names": lerobot_names(first.joints.names)},
        ACTION: {"dtype": "float32", "shape": (len(action_keys),), "names": lerobot_names(action_keys)},
        TIME: {"dtype": "float64", "shape": (1,), "names": ["seconds"]},  # seconds from the episode's first frame
    }
    for cam, img in first.images.items():
        if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            raise ValueError(f"camera {cam!r} must be HxWx3 uint8 to record as video, got {img.shape} {img.dtype}")
        feats[IMAGES + cam] = {
            "dtype": "video",
            "shape": tuple(int(s) for s in img.shape),
            "names": ["height", "width", "channels"],
        }
    return feats


def units_meta(units: Units | None, description: Description) -> dict[str, Any]:
    """``meta/units.json``: the unit system state/action are stored in, per joint as its range next to radians and,
    when the scheme has modes (LeRobot's ``MotorNormMode``), the joint's mode: what a replay rebuilds."""
    if units is None:
        return {"units": "rad", "joints": {}}
    modes = units.modes or (None,) * len(units.keys)
    joints: dict[str, dict[str, Any]] = {}
    for k, a, b, lo, hi, mode in zip(units.keys, units.lo_u, units.hi_u, units.lo, units.hi, modes, strict=True):
        joints[f"{k}.pos"] = {"range": [float(a), float(b)], "rad": [float(lo), float(hi)]}
        if mode is not None:
            joints[f"{k}.pos"]["mode"] = mode
    return {"units": units.name, "joints": joints, "robot": description.name}
