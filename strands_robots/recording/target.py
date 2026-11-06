"""Where a recording goes: a directory, or ``hf://owner/name`` (the only spelling that means the Hub)."""

from __future__ import annotations

import re
from pathlib import Path

HF_PREFIX = "hf://"
_HUB_NAME = re.compile(r"[A-Za-z0-9][\w.-]*")  # what the Hub accepts for an owner or a name


def is_repo_id(target: str | Path) -> bool:
    """Only ``hf://owner/name`` is a Hub repo id; any other string is a directory (``"data/duck"`` included)."""
    return isinstance(target, str) and target.startswith(HF_PREFIX)


def repo_id(target: str | Path) -> str:
    """``hf://owner/name`` -> ``owner/name``."""
    rid = str(target)[len(HF_PREFIX) :]
    parts = rid.split("/")
    if len(parts) != 2 or not all(_HUB_NAME.fullmatch(p) and ".." not in p for p in parts):
        raise ValueError(f"a Hub target is hf://owner/name, got {target!r}")
    return rid


def locate(target: str | Path) -> tuple[str, Path | None]:
    """``(repo_id, root)``: a directory is ``local/<dir name>`` at that path; a Hub id keeps lerobot's cache root."""
    if is_repo_id(target):
        return repo_id(target), None
    path = Path(target).expanduser().resolve()
    return f"local/{path.name}", path
