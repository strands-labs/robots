"""Grade a silent joint stream: is the arm quiet, or is its state probe broken?"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

#: The line mesh.core writes when a probe degrades (``strands_robots.mesh.core._warn_read_state_once``).
_PROBE_LINE = re.compile(r"state probe '?\"?hw_joints'?\"?.*?(failed|still failing)", re.I)

# : The recovery line mesh/core emits when the probe works again.
_RECOVERED_LINE = re.compile(r"state probe '?\"?hw_joints'?\"?.*?recovered", re.I)

# : The cure's own fingerprint (bus_access logs it when it clears a stranded in-use flag).
_FLAG_CLEARED_LINE = re.compile(r"marked in-use by an exchange that never finished", re.I)

#: Said when that fingerprint IS present: the stale-flag explanation is spent, and the remaining
#: causes are a real owner or a bus that stopped answering.
_PORT_IN_USE_AFTER_SELF_HEAL = (
    "This arm has already un-stranded its own bus at least once this session, so a stale flag is no "
    "longer the explanation - the recovery ran and the port was busy again immediately. That means a "
    "REAL second owner or a bus that has stopped answering: `/usr/sbin/lsof /dev/cu.usbmodem*` names "
    "every holder (two means an orphaned child from an earlier spawn - stop that one), and if this "
    "process is the only holder, check the cable and the hub. A bus that keeps stranding is failing "
    "hardware, not bad luck - the card's \u201cbus healed\u201d count is the evidence. "
    "Respawning masks it for a while."
)

#: Ordered: the FIRST match wins, so the specific readings are tried before the generic one.
_SIGNATURES: tuple[tuple[str, str, str, str], ...] = (
    (
        "not_probed",
        "joint probe did not run",
        "this peer never attempted a joint read",
        "Nothing failed on the bus, so there is nothing in the log to find: the arm was built "
        "without a readable hardware object, or it reports itself disconnected. Respawn it from "
        "devices (check the port and that it is not a sim peer) - recalibrating and replugging "
        "change nothing here.",
    ),
    (
        "port_in_use",
        "Port is in use!",
        "this arm's serial port is held - by another process, or by its own aborted read",
        "Ask the OS who holds it first: `/usr/sbin/lsof /dev/cu.usbmodem*` names every holder. "
        "TWO holders means an orphaned child from an earlier spawn - stop that one. ONE holder, "
        "this arm's own process, means the port is busy INSIDE it: a read that died mid-exchange "
        "left the motor bus marked in-use. Current code CLEARS that by itself on the next read "
        ", so an arm that stays silent is either running an older build - respawn it from "
        "devices to pick up the self-healing - or its recovery was refused, which points at a real "
        "second owner rather than a stale flag. Replugging and recalibrating change nothing either "
        "way.",
    ),
    (
        "uncalibrated",
        "no calibration registered",
        "this board has no calibration, so its positions cannot be read in degrees",
        "Calibrate this arm (devices > calibrate). Nothing is contended and a restart will not help.",
    ),
    (
        "no_response",
        "after 3 tries",
        "the motors did not answer a position read",
        "Check the arm's power supply and its data cable - a board on USB logic power alone "
        "answers some reads and not others.",
    ),
)


def classify(
    log_lines: Any,
    calibrations: Any = None,
    *,
    robot_name: Any = None,
    robot_id: Any = None,
) -> dict[str, str] | None:
    """Explain the NEWEST degraded ``hw_joints`` probe in ``log_lines``, or return None."""
    if not isinstance(log_lines, (list, tuple)):
        return None
    for line in reversed([str(x) for x in log_lines]):
        if _RECOVERED_LINE.search(line):
            # Newest-first, so a recovery seen before any failure means the newest word on this probe
            # is "it works" -- stop, and report nothing.
            return None
        if not _PROBE_LINE.search(line):
            continue
        matched = _match(line)
        if matched is not None:
            if matched["kind"] == "port_in_use" and _self_healed(log_lines):
                matched = {**matched, "remedy": _PORT_IN_USE_AFTER_SELF_HEAL}
            if matched["kind"] == "uncalibrated":
                better = calibration_advice(calibrations, robot_name=robot_name, robot_id=robot_id)
                if better:
                    matched = {**matched, "remedy": better}
            return {**matched, "detail": _tail(line)}
        return {
            "kind": "probe_failed",
            "headline": "the joint read failed and the snapshot omits joints",
            "remedy": "Open this robot's log (devices > logs) - the exception the probe raised is "
            "recorded there in full.",
            "detail": _tail(line),
        }
    return None


def calibration_advice(available: Any, *, robot_name: Any = None, robot_id: Any = None) -> str | None:
    """A better remedy for ``uncalibrated`` when calibration files ALREADY EXIST on this machine."""
    if not isinstance(available, Mapping) or not available:
        return None
    narrow = _advice_for_this_arm(available, robot_name, robot_id)
    if narrow:
        return narrow
    where = sorted(
        f"{group}/{name}.json"
        for group, names in available.items()
        if isinstance(names, (list, tuple))
        for name in [str(n) for n in names]
    )
    if not where:
        return None
    shown = ", ".join(where[:6]) + (f" (+{len(where) - 6} more)" if len(where) > 6 else "")
    return (
        "Calibration files DO exist on this machine, so this is probably an id/path mismatch rather "
        "than an uncalibrated arm: lerobot looks for calibration/robots/<robot_type>/<id>.json, and "
        f"what exists is {shown}. Spawn this arm with the id whose file exists (or copy that file to "
        "the path lerobot wants) BEFORE recalibrating - re-teaching an arm that is already "
        "calibrated is physical work to fix a filename."
    )


def _self_healed(log_lines: Any) -> bool:
    return any(_FLAG_CLEARED_LINE.search(str(x)) for x in log_lines)


def recovered(log_lines: Any) -> bool:
    """Whether this child's log SAYS the joint probe is working again."""
    return any(_RECOVERED_LINE.search(str(x)) for x in log_lines)


def _tail(line: str, limit: int = 240) -> str:
    """The exception part of the log line, trimmed - a multi-page motor dump is not a badge."""
    text = line.strip()
    marker = "omitted (further failures logged at debug): "
    if marker in text:
        text = text.split(marker, 1)[1]
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _match(text: Any) -> dict[str, str] | None:
    """The signature this text carries, or None. Shared by the log and snapshot readers."""
    haystack = str(text).lower()
    for kind, needle, headline, remedy in _SIGNATURES:
        if needle.lower() in haystack:
            return {"kind": kind, "headline": headline, "remedy": remedy}
    return None


def classify_state(state: Any) -> dict[str, Any] | None:
    """Explain a joint fault the PEER ITSELF reported in its snapshot, or return None."""
    if not isinstance(state, Mapping):
        return None
    degraded = state.get("degraded")
    if not isinstance(degraded, Mapping):
        return None
    entry = degraded.get("hw_joints")
    if not isinstance(entry, Mapping):
        return None
    reason = str(entry.get("reason") or "").strip()
    if not reason:
        return None
    matched = _match(reason) or {
        "kind": "probe_failed",
        "headline": "the joint read failed and the snapshot omits joints",
        "remedy": "Open this robot's log (devices > logs) - the exception the probe raised is recorded there in full.",
    }
    if matched["kind"] == "port_in_use" and _published_recoveries(state) > 0:
        # The snapshot's own fingerprint, exactly parallel to the log's (see _FLAG_CLEARED_LINE):
        # bus_recoveries counts stranded flags this peer has already cleared, so the stale-flag
        # explanation is spent here for the same reason it is spent there.
        matched = {**matched, "remedy": _PORT_IN_USE_AFTER_SELF_HEAL}
    out: dict[str, Any] = {**matched, "detail": _tail(reason), "source": "peer"}
    for key in ("failures", "for_seconds"):
        value = entry.get(key)
        if isinstance(value, (int, float)):
            out[key] = value
    return out


def _published_recoveries(state: Any) -> int:
    """How many stranded in-use flags the peer says it has cleared (``bus_recoveries``)."""
    if not isinstance(state, Mapping):
        return 0
    value = state.get("bus_recoveries")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value) if value > 0 else 0


def has_joints(state: Any) -> bool:
    """Does this peer's published state carry any joint position at all?"""
    if not isinstance(state, Mapping):
        return False
    joints = state.get("joints")
    if isinstance(joints, Mapping) and len(joints) > 0:
        return True
    return any(str(k).endswith(".pos") for k in state)


def merge(peer: Mapping[str, Any], fields: Mapping[str, Any]) -> dict[str, Any]:
    """The annotation fields to apply to ``peer``, with a stale joint complaint removed."""
    out = dict(fields)
    state = peer.get("state")
    if has_joints(state):
        # Live joints beat every complaint, from either source.
        out.pop("joint_problem", None)
        return out
    reported = classify_state(state)
    if reported is not None:
        # The peer's own report wins over this module's reading of its log: same fault, better
        # evidence (see classify_state).
        out["joint_problem"] = reported
    return out


def _advice_for_this_arm(available: Mapping, robot_name: Any, robot_id: Any) -> str | None:
    """The same advice, but for THIS arm - the exact path lerobot wanted and the ids that would work."""
    name = str(robot_name or "").strip()
    rid = str(robot_id or "").strip()
    if not name or not rid:
        return None
    models = sorted(
        group.split("/", 1)[1]
        for group in available
        if isinstance(group, str)
        and group.startswith("robots/")
        and (group.split("/", 1)[1] == name or group.split("/", 1)[1].startswith(f"{name}_"))
    )
    if not models:
        return None
    ids = sorted(
        {
            str(n)
            for model in models
            for n in (available.get(f"robots/{model}") or [])
            if isinstance(available.get(f"robots/{model}"), (list, tuple))
        }
    )
    if rid in ids:
        # The file lerobot wants IS there. Whatever went wrong, "your calibration is missing" is not
        # it, and sending this operator to recalibrate would be re-teaching a calibrated arm.
        return (
            f"robots/{models[0]}/{rid}.json EXISTS, so this arm is calibrated and the fault is not a "
            "missing calibration - read the probe's own exception in this robot's log (devices > "
            "logs) before touching the hardware."
        )
    elsewhere = sorted(
        f"{group}/{rid}.json"
        for group, names in available.items()
        if isinstance(group, str)
        and not group.startswith("robots/")
        and isinstance(names, (list, tuple))
        and rid in [str(n) for n in names]
    )
    wanted = f"robots/{models[0]}/{rid}.json"
    if elsewhere:
        return (
            f"lerobot wanted {wanted} and it is not there - but {elsewhere[0]} IS, so this id was "
            "calibrated for the OTHER side of the pair. This is a filename, not an uncalibrated arm: "
            + (f"respawn it as one of {', '.join(ids)}, " if ids else "")
            + "or copy that file to the path above. Do NOT recalibrate - re-teaching a calibrated arm "
            "is physical work to fix a path."
        )
    return f"lerobot wanted {wanted}, which does not exist" + (
        f"; the ids calibrated for this robot are {', '.join(ids)}. Respawn this arm as one of "
        "them, or calibrate it under the id you are using."
        if ids
        else ". Calibrate this arm under the id you are using."
    )
