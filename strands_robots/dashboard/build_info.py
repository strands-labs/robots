"""What build this dashboard process actually is - the one thing /api/health never said."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any


def read_commit(root: Path | str | None) -> str | None:
    """The short sha of ``root``'s checkout, or None when it cannot be read honestly."""
    if root is None:
        return None
    git = Path(root) / ".git"
    try:
        head = (git / "HEAD").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if head.startswith("ref:"):
        ref = head[4:].strip()
        # A ref path is attacker-irrelevant here (it is our own checkout) but still must not escape
        # .git: a HEAD containing "ref: ../../etc/passwd" should yield None, not a file read.
        if ".." in Path(ref).parts or Path(ref).is_absolute():
            return None
        try:
            head = (git / ref).read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError):
            return None
    head = head.split()[0] if head else ""
    if len(head) < 7 or any(c not in "0123456789abcdef" for c in head.lower()):
        return None
    return head[:12]


def package_version() -> str | None:
    """The installed strands-robots version, or None in a source checkout with nothing installed."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:  # pragma: no cover - stdlib always has it on 3.12
        return None
    for name in ("strands-robots", "strands_robots"):
        try:
            return version(name)
        except PackageNotFoundError:
            continue
        except Exception:  # pragma: no cover - a broken dist must not break /api/health
            return None
    return None


def stamp(*, commit: str | None, version: str | None, started: float) -> dict[str, Any]:
    """Assemble the payload. Pure, so the shape is testable without a filesystem or a clock."""
    return {"commit": commit, "version": version, "started": started}


@lru_cache(maxsize=1)
def build_info(root: str | None = None) -> dict[str, Any]:
    """Cached stamp for this process. Cached because it cannot change without a restart -
    that immutability IS the fact being reported, and /api/health is polled constantly."""
    here = Path(__file__).resolve()
    # this module (``strands_robots.dashboard.build_info``) -> repo root is three parents up.
    default_root = here.parents[2]
    return stamp(
        commit=read_commit(root if root is not None else default_root),
        version=package_version(),
        started=time.time(),
    )
