"""Judge a training output directory before a run is started into it."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

#: How many names to show. Enough to recognise a directory, few enough to read.
SAMPLE = 5


def classify_output_dir(
    *,
    exists: bool,
    is_dir: bool = True,
    has_checkpoint: bool = False,
    names: list[str] | None = None,
    total: int = 0,
    unreadable: str | None = None,
) -> dict[str, Any]:
    """Verdict for one candidate ``output_dir``. ``state`` is one of: ``free`` nothing there (or an
    empty directory) -- the run creates it.
    """
    if unreadable:
        return {
            "state": "unknown",
            "destructive": False,
            "needs_confirm": False,
            "detail": (
                f"cannot read that path ({unreadable}), so what a run would do to it is unknown -- "
                "fix the path or the permissions before training"
            ),
        }
    if not exists:
        return {
            "state": "free",
            "destructive": False,
            "needs_confirm": False,
            "detail": "does not exist yet -- the run creates it",
        }
    if not is_dir:
        return {
            "state": "not_a_dir",
            "destructive": False,
            "needs_confirm": False,
            "detail": "that path is a FILE, not a directory -- a run cannot be written there",
        }
    if has_checkpoint:
        return {
            "state": "resumable",
            "destructive": False,
            "needs_confirm": False,
            "detail": (
                "already holds a training checkpoint. It will NOT be deleted, but this dashboard "
                "cannot resume a run, and lerobot refuses a fresh run into an existing run's "
                "directory -- pick a new directory (the failure would otherwise appear only in "
                "the run's log, after the job says it started)"
            ),
        }
    shown = list(names or [])[:SAMPLE]
    if not shown and total == 0:
        return {
            "state": "free",
            "destructive": False,
            "needs_confirm": False,
            "detail": "exists but is empty -- the run uses it as it is",
        }
    more = max(0, total - len(shown))
    listing = ", ".join(shown) + (f" and {more} more" if more else "")
    return {
        "state": "occupied",
        "destructive": True,
        "needs_confirm": True,
        "entries": shown,
        "total": total,
        "detail": (
            f"holds {total} item(s) and NO training checkpoint, so starting a run here "
            f"DELETES the directory and everything in it ({listing}). This cannot be undone"
        ),
    }


def inspect_output_dir(path: str, *, has_checkpoint: Any = None) -> dict[str, Any]:
    """Read the path and classify it. ``has_checkpoint`` may be a callable(path) -> bool."""
    p = Path(path).expanduser()
    verdict_path = str(p)
    try:
        if not p.exists():
            out = classify_output_dir(exists=False)
            out["path"] = verdict_path
            return out
        if not p.is_dir():
            out = classify_output_dir(exists=True, is_dir=False)
            out["path"] = verdict_path
            return out
        entries = sorted(e.name for e in os.scandir(p))
        ckpt = False
        if has_checkpoint is not None:
            try:
                ckpt = bool(has_checkpoint(verdict_path))
            except Exception:  # noqa: BLE001 - a trainer hiccup must not silently mean "no ckpt"
                out = classify_output_dir(exists=True, unreadable="checkpoint probe failed")
                out["path"] = verdict_path
                return out
        out = classify_output_dir(exists=True, has_checkpoint=ckpt, names=entries, total=len(entries))
    except OSError as exc:
        out = classify_output_dir(exists=True, unreadable=type(exc).__name__)
    out["path"] = verdict_path
    return out


def default_checkpoint_probe(path: str) -> bool:
    """The trainer's own answer, when it can be imported (torch-less installs say False)."""
    try:
        from strands_robots.training.lerobot import LerobotTrainer

        return LerobotTrainer.latest_checkpoint(LerobotTrainer.__new__(LerobotTrainer), path) is not None
    except Exception:  # noqa: BLE001
        # Fall back to the on-disk shape lerobot writes: <output_dir>/checkpoints/<step>/…
        return Path(path, "checkpoints").is_dir()
