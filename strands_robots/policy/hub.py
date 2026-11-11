"""Which Hugging Face Hub checkpoints an agent may load: the ones the operator lists in ``STRANDS_POLICY_REPOS``.

Loading a LeRobot checkpoint runs what it names: its saved processor steps are rebuilt from its own JSON, and some
of them load further repos with remote code enabled. So the repo an agent names is code chosen by whoever publishes
it. The developer's own ``Policy("owner/repo")`` is their choice and is not checked here; an agent's is, against
``STRANDS_POLICY_REPOS=owner/repo,owner2/*`` (exact ids, or every repo of one publisher). Empty means none.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .forms import REPO, is_checkpoint, is_dataset, is_preset

ENV = "STRANDS_POLICY_REPOS"
_ENTRY = re.compile(r"^[A-Za-z0-9][\w.-]*/(\*|[\w.-]+)$")


def hub_repo(spec: Any) -> str | None:
    """The Hub repo id ``Policy(spec)`` would download, or None: a preset, a local checkpoint or dataset directory,
    a network address and a callable are not downloads (the same precedence ``make`` uses)."""
    if not isinstance(spec, (str, Path)):
        return None
    s = str(spec)
    if s.startswith(("zenoh://", "zenoh+")) or is_preset(s):
        return None
    local = Path(s.partition("?")[0]).expanduser()
    if is_dataset(local) or is_checkpoint(local):
        return None
    return s if REPO.match(s) else None


def allowed_repos() -> tuple[str, ...]:
    """The entries of ``STRANDS_POLICY_REPOS``, lower-cased; a malformed entry is a ``ValueError``, never a wildcard."""
    entries = tuple(e.strip().lower() for e in os.environ.get(ENV, "").split(",") if e.strip())
    if bad := [e for e in entries if not _ENTRY.match(e)]:
        raise ValueError(f"{ENV}: {bad} are not 'owner/repo' or 'owner/*'")
    return entries


def repo_allowed(repo: str) -> bool:
    owner, _, _ = (r := repo.lower()).partition("/")
    return any(e == r or e == f"{owner}/*" for e in allowed_repos())


def refuse_unlisted(spec: Any) -> None:
    """A ``PermissionError`` if ``spec`` downloads a Hub repo the operator did not list."""
    repo = hub_repo(spec)
    if repo is not None and not repo_allowed(repo):
        raise PermissionError(
            f"Hub checkpoint {repo!r} runs code its publisher chose, and the operator has not allowed it; "
            f"an agent may load only repos listed in {ENV} (now: {', '.join(allowed_repos()) or 'none'})"
        )
