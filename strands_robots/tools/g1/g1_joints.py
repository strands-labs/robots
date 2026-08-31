"""Agent-facing lookup for the joint names :class:`G1Driver.send_action` accepts.

``send_action`` refuses an action dict whose keys are not in
:data:`~strands_robots.drivers.g1._G1_JOINT_INDEX`; this module surfaces that
same map to an agent so the refusal is decidable before the write is attempted
rather than surfaced from the driver as an unknown-key error at rollout time.
The three tools here read the driver's own constants -
:data:`~strands_robots.drivers.g1._G1_JOINT_INDEX`,
:data:`~strands_robots.drivers.g1._SDK_KP` and
:data:`~strands_robots.drivers.g1._SDK_KD` - so a joint added to the driver's
contract moves both the write path and this lookup together. No SDK, no DDS,
no cache: this module is a pure reader over module-level tables. That is why
``import strands_robots.tools.g1.g1_joints`` pulls no ``unitree_sdk2py``
submodule - the same import-hygiene contract every other file in this package
carries.

Two of the driver's contract facts are not resolved here:

* Slots ``13`` and ``14`` (:data:`waist_roll`, :data:`waist_pitch`) and slots
  ``20 / 21 / 27 / 28`` (wrist pitch and yaw pairs) are named in the driver's
  map but not present on every G1 variant. The bundle port carries the driver's
  full 29-name list because ``send_action`` accepts every name in the map
  regardless of the physical build - a caller pointing at a joint the local
  robot does not have will receive a firmware refusal, not a name-error - and
  the per-build presence question is still open on ``refs #2765``.
* The ankle-pitch / ankle-roll rename question is on the same issue. This
  lookup returns the names the driver's map declares today
  (``left_ankle_pitch`` / ``left_ankle_roll``); if that issue lands a rename,
  it will land in the driver's map, and this lookup follows.
"""

from __future__ import annotations

import re
from typing import Any

from strands import tool

from strands_robots.drivers.g1 import (
    _G1_JOINT_INDEX,
    _G1_NAMED_JOINTS,
    _SDK_KD,
    _SDK_KP,
)

#: The five body groups the driver's joint map is organised around, indexed by
#: the same slot as :data:`~strands_robots.drivers.g1._G1_JOINT_INDEX`. Kept
#: here rather than in the driver because the driver has no use for the
#: grouping (its write path indexes by slot, not by group); the group name is a
#: label the agent-facing tool returns to make the map legible.
_GROUP_SLOTS: dict[str, tuple[int, ...]] = {
    "left_leg": (0, 1, 2, 3, 4, 5),
    "right_leg": (6, 7, 8, 9, 10, 11),
    "waist": (12, 13, 14),
    "left_arm": (15, 16, 17, 18, 19, 20, 21),
    "right_arm": (22, 23, 24, 25, 26, 27, 28),
}


def _slot_group(slot: int) -> str | None:
    """The group name for a slot, or ``None`` if the slot is out of range."""
    for group, slots in _GROUP_SLOTS.items():
        if slot in slots:
            return group
    return None


#: The name the caller supplied normalised to the driver's snake_case
#: convention. Splits on either an underscore boundary or a lower-to-upper
#: camel-case boundary so ``left_knee``, ``LeftKnee`` and ``leftKnee`` all
#: normalise to the same key, then lowercases the whole thing. This is the
#: only alias layer: the driver's write path indexes by the exact snake_case
#: key, so a wider tolerance here would let a caller reach the lookup with a
#: spelling that the driver would then refuse - which is the failure mode
#: this tool is meant to prevent.
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _canonicalise(name: str) -> str:
    """Normalise a joint name to the driver's snake_case key."""
    return _CAMEL_BOUNDARY.sub("_", name.strip()).lower()


def _slot_row(slot: int) -> dict[str, Any]:
    """The record this module returns for a single joint slot."""
    name = _NAMES_BY_SLOT[slot]
    return {
        "index": slot,
        "name": name,
        "group": _slot_group(slot),
        "kp": _SDK_KP[slot],
        "kd": _SDK_KD[slot],
    }


#: The driver's map reversed: slot -> name. Precomputed once at module load
#: rather than searched linearly on every lookup - the map is 29 entries, so
#: the O(1) here is not a hot-path win, it is a correctness win: a
#: lookup-by-slot that walked the dict would silently return the first name
#: any two slots collided on, and this reversal is where such a collision would
#: surface as a ``ValueError`` at import time rather than as a wrong answer at
#: call time.
_NAMES_BY_SLOT: tuple[str, ...] = tuple(name for name, _ in sorted(_G1_JOINT_INDEX.items(), key=lambda kv: kv[1]))
if len(_NAMES_BY_SLOT) != _G1_NAMED_JOINTS:
    raise ValueError(
        f"G1 joint map has {len(_G1_JOINT_INDEX)} names filling "
        f"{_G1_NAMED_JOINTS} slots - the reversal is not one-to-one, "
        "so the driver's map is malformed. Refs "
        "strands-labs/robots#2765."
    )


@tool
def g1_joint_reference(group: str = "") -> dict[str, Any]:
    """Return the joint-name / slot / gain table :class:`G1Driver` writes against.

    Read-only. Every field is a driver constant; no bus is touched. Useful
    before a ``run_policy`` or ``send_action`` call, to confirm the name a
    caller intends to send is a key the driver will honour.

    Args:
        group: Optional group filter. One of ``left_leg``, ``right_leg``,
            ``waist``, ``left_arm``, ``right_arm``. Empty returns every slot.

    Returns:
        A dict with ``status``, a ``count`` of returned rows, and a ``joints``
        list of records carrying ``index``, ``name``, ``group``, ``kp`` and
        ``kd`` for each slot. On an unknown group name the returned dict
        carries ``status="error"`` and a ``message`` naming the valid groups
        as a resolvable domain.
    """
    if group and group not in _GROUP_SLOTS:
        valid = sorted(_GROUP_SLOTS)
        return {
            "status": "error",
            "message": (f"unknown group {group!r}. Valid groups are {valid}. Refs strands-labs/robots#2765."),
        }
    slots = _GROUP_SLOTS[group] if group else tuple(range(_G1_NAMED_JOINTS))
    joints = [_slot_row(slot) for slot in slots]
    return {
        "status": "success",
        "count": len(joints),
        "group": group or None,
        "groups": sorted(_GROUP_SLOTS),
        "joints": joints,
    }


@tool
def g1_joint_name(index: int) -> dict[str, Any]:
    """Return the driver name for a joint slot.

    Read-only. Refuses a slot outside ``[0, 28]`` with a message that names
    the accepted range; the driver's map has 29 entries and neither the
    write path nor this lookup can address slot 29 or higher.

    Args:
        index: The 0-based joint slot the caller wants to name.

    Returns:
        A dict with ``status`` and the same record shape
        :func:`g1_joint_reference` returns for a single slot. Out-of-range
        indices carry ``status="error"``.
    """
    if not isinstance(index, int) or isinstance(index, bool):
        return {
            "status": "error",
            "message": (
                f"index must be an int in [0, {_G1_NAMED_JOINTS - 1}]; got "
                f"{type(index).__name__} {index!r}. Refs "
                "strands-labs/robots#2765."
            ),
        }
    if not 0 <= index < _G1_NAMED_JOINTS:
        return {
            "status": "error",
            "message": (f"index {index} is out of range [0, {_G1_NAMED_JOINTS - 1}]. Refs strands-labs/robots#2765."),
        }
    row = _slot_row(index)
    return {"status": "success", **row}


@tool
def g1_joint_index(name: str) -> dict[str, Any]:
    """Return the driver slot for a joint name.

    Read-only. Accepts the driver's canonical snake_case names verbatim, and
    also normalises a PascalCase or camelCase spelling to that key
    (``LeftKnee`` and ``leftKnee`` both resolve to ``left_knee``). This alias
    is one-way: the driver's ``send_action`` accepts only the snake_case
    spelling on the wire, and a caller who wants the exact write-path spelling
    reads the ``name`` field on the returned record rather than the input they
    supplied.

    Args:
        name: A joint name in the driver's snake_case (canonical) or in
            PascalCase / camelCase (alias).

    Returns:
        A dict with ``status`` and the same record shape
        :func:`g1_joint_reference` returns for a single slot. An unknown name
        carries ``status="error"`` and a message listing the driver's actual
        map so the caller sees the domain rather than a hint.
    """
    if not isinstance(name, str):
        return {
            "status": "error",
            "message": (f"name must be a string; got {type(name).__name__} {name!r}. Refs strands-labs/robots#2765."),
        }
    key = _canonicalise(name)
    if key not in _G1_JOINT_INDEX:
        return {
            "status": "error",
            "message": (
                f"no joint named {name!r}. The driver's map is "
                f"{sorted(_G1_JOINT_INDEX)}. Refs strands-labs/robots#2765."
            ),
        }
    return {"status": "success", **_slot_row(_G1_JOINT_INDEX[key])}
