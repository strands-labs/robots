"""Finding and shaping calibrations, for the dashboard's calibration drawer.

Read-only. The files are lerobot's own (``<root>/<device_type>/<device_model>/
<device_id>.json``); writing one is the wizard's job (:mod:`calibration_run`),
which drives ``lerobot-calibrate`` itself so the two can never disagree.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Layout on disk: <root>/<device_type>/<device_model>/<device_id>.json
_SUFFIX = ".json"

#: Where lerobot puts the directory when its constants cannot be asked.
_FALLBACK_ROOT = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration"


def default_root() -> Path:
    """Where lerobot keeps calibrations, honouring the same env the CLI does.

    Read from ``lerobot.utils.constants`` when lerobot is installed, so a moved
    ``HF_LEROBOT_CALIBRATION`` is followed; otherwise the env variable, then
    lerobot's documented default. Listing never needs lerobot itself.
    """
    try:
        from lerobot.utils import constants

        return Path(str(constants.HF_LEROBOT_CALIBRATION))
    except Exception:  # noqa: BLE001 - lerobot is optional; a broken install must not hide the files
        pass
    explicit = os.environ.get("HF_LEROBOT_CALIBRATION", "").strip()
    return Path(explicit).expanduser() if explicit else _FALLBACK_ROOT


def _sorted_dirs(parent: Path) -> list[Path]:
    try:
        return sorted(p for p in parent.iterdir() if p.is_dir())
    except OSError:
        return []


def structure(root: Path | str | None = None) -> dict[str, dict[str, list[str]]]:
    """``{device_type: {device_model: [device_id, ...]}}`` for every calibration on disk."""
    base = Path(root) if root is not None else default_root()
    out: dict[str, dict[str, list[str]]] = {}
    if not base.is_dir():
        return out
    for type_dir in _sorted_dirs(base):
        models: dict[str, list[str]] = {}
        for model_dir in _sorted_dirs(type_dir):
            try:
                ids = sorted(f.stem for f in model_dir.iterdir() if f.is_file() and f.suffix == _SUFFIX)
            except OSError:
                ids = []
            if ids:
                models[model_dir.name] = ids
        if models:
            out[type_dir.name] = models
    return out


def calibration_info(
    device_type: str,
    device_model: str,
    device_id: str,
    *,
    root: Path | str | None = None,
) -> dict[str, Any] | None:
    """One calibration file's facts and contents, or None when it cannot be read.

    Each name must be a single path segment: they are joined into a path under
    the root, and a separator or ``..`` in any of them would read outside it.
    """
    for value in (device_type, device_model, device_id):
        if not value or "/" in value or "\\" in value or value in (".", "..") or value.startswith("."):
            return None
    base = Path(root) if root is not None else default_root()
    path = base / device_type / device_model / f"{device_id}{_SUFFIX}"
    try:
        stat = path.stat()
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.debug("calibration %s unreadable: %s", path, exc)
        return None
    info: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_time": datetime.fromtimestamp(stat.st_mtime),
        "device_type": device_type,
        "device_model": device_model,
        "device_id": device_id,
        "data": data,
    }
    if isinstance(data, dict):
        info["motor_count"] = len(data)
        info["motor_names"] = list(data.keys())
    return info


def listing(root: Path | str | None = None) -> dict[str, Any]:
    """Every calibration on this machine, as the drawer's ``{status, text, ...}`` payload.

    ``text`` is the markdown block the drawer parses (location line, one ``##``
    per device type, one ``###`` per model, one bullet per id); ``entries``
    carries the same rows as data for a client that prefers them.
    """
    base = Path(root) if root is not None else default_root()
    tree = structure(base)
    lines = ["**LeRobot Calibrations**", f"Location: `{base}`", ""]
    entries: list[dict[str, Any]] = []
    for dev_type, models in tree.items():
        lines.append(f"## **{dev_type.title()}**")
        for model, ids in models.items():
            lines.append(f"### **{model}** ({len(ids)} calibrations)")
            for calib_id in ids:
                info = calibration_info(dev_type, model, calib_id, root=base)
                if info is None:
                    lines.append(f"  - `{calib_id}` *(error reading file)*")
                    entries.append(
                        {"device_type": dev_type, "device_model": model, "device_id": calib_id, "unreadable": True}
                    )
                    continue
                modified = info["modified_time"].strftime("%Y-%m-%d %H:%M:%S")
                size_kb = info["size_bytes"] / 1024
                motor_info = f"{info.get('motor_count', 0)} motors" if info.get("motor_count") else ""
                lines.append(f"  - `{calib_id}` *({modified}, {size_kb:.1f}KB, {motor_info})*")
                entries.append(
                    {
                        "device_type": dev_type,
                        "device_model": model,
                        "device_id": calib_id,
                        "path": info["path"],
                        "modified": info["modified_time"].isoformat(timespec="seconds"),
                        "size_bytes": info["size_bytes"],
                        "motor_count": info.get("motor_count"),
                        "unreadable": False,
                    }
                )
            lines.append("")
    if not entries:
        lines.append("No calibrations found.")
    return {"status": "success", "text": "\n".join(lines), "root": str(base), "count": len(entries), "entries": entries}


def candidates(
    name: str,
    *,
    root: Path | str | None = None,
    device_type: str | None = None,
    device_model: str | None = None,
) -> list[dict[str, str]]:
    """Every calibration whose device_id is ``name``, narrowed by the filters."""
    base = Path(root) if root is not None else default_root()
    if not name or "/" in name or name.startswith("."):
        return []  # a device id is one path segment, never a traversal
    out: list[dict[str, str]] = []
    if not base.is_dir():
        return out
    for type_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if device_type and type_dir.name != device_type:
            continue
        for model_dir in sorted(p for p in type_dir.iterdir() if p.is_dir()):
            if device_model and model_dir.name != device_model:
                continue
            path = model_dir / f"{name}{_SUFFIX}"
            if path.is_file():
                out.append(
                    {
                        "device_type": type_dir.name,
                        "device_model": model_dir.name,
                        "device_id": name,
                        "path": str(path),
                    }
                )
    return out


def motors(data: Any) -> list[dict[str, Any]]:
    """The per-motor rows, as a LIST so the UI keeps the file's own order. A calibration's dict order
    is the motor order on the arm (shoulder_pan, shoulder_lift, elbow_flex...).
    """
    if not isinstance(data, dict):
        return []
    rows: list[dict[str, Any]] = []
    for motor_name, motor in data.items():
        row: dict[str, Any] = {"name": str(motor_name)}
        if isinstance(motor, dict):
            row.update({str(k): v for k, v in motor.items()})
        else:
            row["value"] = motor
        rows.append(row)
    return rows


def payload(info: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe view of the tool's ``calibration_info``."""
    modified = info.get("modified_time")
    if isinstance(modified, datetime):
        modified_iso: str | None = modified.isoformat(timespec="seconds")
        modified_epoch: float | None = modified.timestamp()
    else:
        modified_iso = str(modified) if modified else None
        modified_epoch = None
    rows = motors(info.get("data"))
    return {
        "device_type": info.get("device_type"),
        "device_model": info.get("device_model"),
        "device_id": info.get("device_id"),
        "path": info.get("path"),
        "size_bytes": info.get("size_bytes"),
        "modified": modified_iso,
        "modified_epoch": modified_epoch,
        # motor_count comes from the file, not from len(rows), so a mismatch
        # between the two stays visible instead of being smoothed over.
        "motor_count": info.get("motor_count"),
        "motors": rows,
    }


def robot_calibration_gap(
    robot_name: str,
    robot_id: str | None,
    *,
    root: Path | str | None = None,
) -> str | None:
    """Why a REAL robot spawned as ``robot_id`` will refuse to read its motors, or None."""
    if not robot_id or not robot_name:
        return None
    base = Path(root) if root is not None else default_root()
    if not base.is_dir():
        return None  # no cache to judge; the child will speak for itself
    robots_dir = base / "robots"
    if not robots_dir.is_dir():
        return None
    # lerobot's model directory is the robot's own name plus a role suffix (so101 ->
    # so101_follower), so match by prefix rather than hard-coding the suffix: a robot type this
    # dashboard has never seen must not produce a confident wrong sentence.
    models = sorted(
        p for p in robots_dir.iterdir() if p.is_dir() and (p.name == robot_name or p.name.startswith(f"{robot_name}_"))
    )
    if not models:
        return None  # unknown layout for this robot type - say nothing rather than guess
    for model in models:
        if (model / f"{robot_id}{_SUFFIX}").is_file():
            return None  # exactly where it will be looked for
    elsewhere = [c for c in candidates(robot_id, root=base) if c["device_type"] != "robots"]
    have = sorted({f.stem for model in models for f in model.glob(f"*{_SUFFIX}")})
    where = ", ".join(f"{m.name}" for m in models)
    if elsewhere:
        first = elsewhere[0]
        return (
            f"robot_id {robot_id!r} has a calibration, but as a "
            f"{first['device_type'].rstrip('s')}: {first['path']}. A robot in real mode loads "
            f"robots/{where}/{robot_id}{_SUFFIX}, which does not exist, so the bus will refuse "
            f"with 'has no calibration registered' and the arm will report presence with no "
            f"joints. Calibrate this id as a robot, or spawn it with one that already is"
            + (f": {', '.join(have)}" if have else "")
        )
    return (
        f"robot_id {robot_id!r} has no calibration under robots/{where}, so the bus will refuse "
        f"with 'has no calibration registered' and the arm will report presence with no joints. "
        + (f"Ids that do have one: {', '.join(have)}. " if have else "")
        + "Calibrate this arm from the devices screen, or spawn it under an id that is already "
        "calibrated."
    )
