"""LeRobot ``semantics`` labels -> strands units, read from ``semantics.json`` (data, never an ``if``).

A label the table does not know is read as words: LeRobot keeps no registry of these strings, every
server names its own, and the deployment was already checked joint by joint against this robot. What the
label says about the body's mode (``degrees`` / ``range``, ``normalized`` / ``radians``) overrides the
robot's own LeRobot convention (``hardware.lerobot_norm`` in the registry: the modes its calibration
JSON is written for); the rest comes from that convention. Refused only when neither says anything.
"""

from __future__ import annotations

import json
import re
import warnings
from functools import cache
from importlib import resources
from typing import Any

from ..description import Description
from .lerobot import lerobot_units
from .scheme import Units


@cache
def table() -> dict[str, dict[str, Any]]:
    raw = json.loads(resources.files(__package__).joinpath("semantics.json").read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


WORDS = {  # what a label's words say about the body's mode
    "degrees": "degrees",
    "deg": "degrees",
    "range": "range_m100_100",
    "normalized": "range_m100_100",
    "normalised": "range_m100_100",
    "norm": "range_m100_100",
    "radians": "rad",
    "rad": "rad",
}


def derive(semantics: str, description: Description) -> dict[str, Any] | None:
    """A table row for an unknown label, from its words and the robot's own convention; None when neither says."""
    words = [w for w in re.split(r"[-_./]", semantics.lower()) if w]
    said = [WORDS[w] for w in words if w in WORDS]
    if said and said[0] == "rad":
        return {"units": "rad"}
    base = dict(description.hardware.lerobot_norm or {})
    if not said and not base:
        return None
    norm = {**base, "*": said[0]} if said else base
    if "percent" in words and "gripper" in description.action_keys:
        norm["gripper"] = "range_0_100"
    return {"lerobot_norm": norm}


def units_for_semantics(semantics: str, description: Description) -> Units | None:
    """The converter for a LeRobot semantics label (None = radians): the table's row, else one derived from the
    label's words and the robot's own LeRobot convention (a warning says what was assumed); a refusal only when
    neither the label nor the registry says what the numbers are."""
    entry = table().get(semantics)
    entry = table().get(entry["same_as"]) if entry and "same_as" in entry else entry
    if entry is None:
        entry = derive(semantics, description)
        if entry is None:
            raise ValueError(
                f"unknown LeRobot semantics {semantics!r}, and {description.name} declares no hardware.lerobot_norm "
                f"to derive it from: add a row to strands_robots/units/semantics.json "
                f"(known: {', '.join(sorted(table()))}) or pass units= explicitly"
            )
        what = "radians" if entry.get("units") == "rad" else f"LeRobot modes {entry['lerobot_norm']}"
        warnings.warn(
            f"LeRobot semantics {semantics!r} is not in strands_robots/units/semantics.json; read as {what} "
            f"(from its words and {description.name}'s own convention). Add a row to make it data.",
            stacklevel=2,
        )
    if entry.get("units") == "rad":
        return None
    return lerobot_units(description, entry["lerobot_norm"])


def semantics_for(description: Description, norm: dict[str, str] | None = None) -> str:
    """The table's label for this robot's norm modes (registry default unless ``norm`` is given)."""
    norm = dict(norm if norm is not None else (description.hardware.lerobot_norm or {}))
    if not norm:
        return "radians-v1"
    hits = [k for k, v in table().items() if v.get("lerobot_norm") == norm and k.startswith(f"{description.name}-")]
    hits = hits or [k for k, v in table().items() if v.get("lerobot_norm") == norm]
    if not hits:
        raise ValueError(f"no semantics label for {description.name} with modes {norm}; add one to semantics.json")
    return sorted(hits)[0]
