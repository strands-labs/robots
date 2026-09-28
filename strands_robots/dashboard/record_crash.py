"""The on-disk crumb a live record session leaves so a restart can say it was interrupted."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def crumb_path() -> Path:
    """Where the breadcrumb lives. Beside auth.json, so one directory holds the dashboard's state."""
    override = os.getenv("STRANDS_DASH_RECORD_CRUMB")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".strands_dashboard" / "record_session.json"


def write_crumb(session: Mapping[str, Any], *, path: Path | None = None, now: float | None = None) -> None:
    """Remember that a session is open. Failure is silent: a breadcrumb is a courtesy, and a
    read-only home directory must not stop a recording from starting."""
    p = path or crumb_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "dataset": session.get("dataset"),
                    "task": session.get("task"),
                    "leader": session.get("leader"),
                    "follower": session.get("follower"),
                    "opened_at": now if now is not None else time.time(),
                    "pid": os.getpid(),
                }
            ),
            encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass


def clear_crumb(path: Path | None = None) -> None:
    """A session that closed properly leaves no trace."""
    try:
        (path or crumb_path()).unlink(missing_ok=True)
    except Exception:  # noqa: BLE001
        pass


def read_crumb(path: Path | None = None) -> dict[str, Any] | None:
    """The crumb an earlier session left behind, or None when there is none or it is unreadable."""
    p = path or crumb_path()
    try:
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data.get("dataset") else None
    except Exception:  # noqa: BLE001 - a corrupt crumb is no evidence, not an error
        return None


def _ago(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "at an unknown time"
    if seconds < 90:
        return "less than two minutes ago"
    if seconds < 5400:
        return f"about {round(seconds / 60)} minutes ago"
    return f"about {round(seconds / 3600, 1)} hours ago"


def interrupted_notice(
    crumb: Mapping[str, Any] | None,
    *,
    now: float | None = None,
    same_process: bool = False,
) -> dict[str, Any] | None:
    """The sentence the record screen shows when a previous session never closed."""
    if not crumb:
        return None
    dataset = str(crumb.get("dataset") or "").strip()
    if not dataset:
        return None
    opened = crumb.get("opened_at")
    age = (now if now is not None else time.time()) - float(opened) if isinstance(opened, (int, float)) else None
    arms = [str(crumb.get(k)) for k in ("leader", "follower") if crumb.get(k)]
    who = " and ".join(arms)
    cause = (
        "a session was opened and never closed"
        if same_process
        else "the dashboard stopped while a recording session was open"
    )
    return {
        "dataset": dataset,
        "task": crumb.get("task") or "",
        "arms": arms,
        "opened_ago": age,
        "text": (
            f"{cause}: “{dataset}” was opened {_ago(age)}"
            + (f", driving {who}" if who else "")
            + ". Episodes already written are on disk; the ones in flight were not flushed"
            + (f", and {who} were left despawned - respawn them from devices." if who else ".")
        ),
        "next": [
            f"record into “{dataset}” again to continue that dataset (the name is taken, so a new "
            "session must use another name)",
            f"or delete “{dataset}” if the take is worthless",
        ],
    }
