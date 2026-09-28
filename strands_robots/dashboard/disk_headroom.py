"""Is there room on disk for the episodes you are about to record?"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

MB = 1024 * 1024
GB = 1024 * MB

#: Below this, the disk itself is the story: refuse and say so.
CRITICAL_MB = 2 * 1024
# : Below this, warn.
TIGHT_MB = 12 * 1024


def _fmt(mb: float) -> str:
    """Sizes the way an operator reads them: 18Gi, 940Mi."""
    return f"{mb / 1024:.1f}Gi" if mb >= 1024 else f"{mb:.0f}Mi"


def headroom_verdict(
    *,
    free_mb: float | None,
    total_mb: float | None = None,
    where: str | None = None,
) -> dict[str, Any] | None:
    """Classify the space where a dataset is about to be written."""
    if free_mb is None or free_mb < 0:
        return None
    free = float(free_mb)
    if free >= TIGHT_MB:
        return None
    place = f" on {where}" if where else ""
    share = ""
    if total_mb and total_mb > 0:
        share = f" ({free / float(total_mb) * 100:.1f}% of the volume)"
    if free < CRITICAL_MB:
        return {
            "level": "critical",
            "free_mb": round(free),
            "headline": f"only {_fmt(free)} free{place}{share} - not enough to finish a recording",
            "advice": (
                "A dataset write that runs out of space leaves meta/info.json promising episodes "
                "that data/ cannot supply, and the error surfaces at TRAIN time naming a parquet "
                "file. Free space first: this volume also holds macOS swap, so a memory-hungry "
                "process costs disk too."
            ),
        }
    return {
        "level": "tight",
        "free_mb": round(free),
        "headline": f"{_fmt(free)} free{place}{share} - fine for a short session, tight for a long one",
        "advice": (
            "Cameras dominate the size: a multi-camera episode is far bigger than the joint stream. "
            "Watch the number if you plan many episodes - and remember this volume holds macOS swap, "
            "so free space can fall while you record even if nothing else is writing."
        ),
    }


def free_space(path: str | Path | None = None) -> dict[str, Any]:
    """Read free space where datasets land. Never raises."""
    try:
        target = Path(path) if path is not None else _dataset_home()
        # A RELATIVE path would walk up to "." and report the volume this process happens to be
        # running in - a reading for a disk nobody asked about, presented as the dataset's disk.
        if not target.is_absolute():
            return {}
        probe = target
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        return {
            "path": str(target),
            "measured_at": str(probe),
            "free_mb": usage.free / MB,
            "total_mb": usage.total / MB,
        }
    except Exception:  # noqa: BLE001 - no reading is a valid answer, an exception is not
        return {}


def _dataset_home() -> Path:
    try:
        from strands_robots.dataset_recorder import resolve_dataset_dir

        return resolve_dataset_dir("local/_headroom_probe").parent
    except Exception:  # noqa: BLE001
        return Path.home() / ".cache" / "huggingface" / "lerobot"
