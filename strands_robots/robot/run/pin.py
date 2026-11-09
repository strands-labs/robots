"""The files an operator approves are the files an agent's run on a real body reads.

``files_digest`` hashes what the policy loader would read from disk when the plan is shown; ``pinned`` then gives
the loader a private copy taken in one pass: every byte written into the copy is the byte that was hashed, so a
write to the original at any moment after the yes (during the dialog, the copy or the load) never reaches the robot.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .source import RunPlan

_CHUNK = 1 << 20


def files_digest(policy: object) -> str | None:
    """SHA-256 over the names and bytes of the files a policy spec reads from disk, or None if it names none."""
    root = _root(policy)
    return None if root is None else _snapshot(root, None)


@contextmanager
def pinned(plan: RunPlan | None, policy: Any) -> Iterator[Any]:
    """The spec to load: for a plan that pinned files, the same spec pointing at a private copy of exactly them.

    The copy is checked against ``plan.files`` before anything loads (``RequestRefused`` if they differ) and removed
    when the ``with`` ends, however it ends.
    """
    root = _root(policy) if plan is not None and plan.files is not None else None
    if plan is None or root is None:
        yield policy
        return
    from ..safety import RequestRefused

    tmp = Path(tempfile.mkdtemp(prefix="strands-run-"))  # 0700: no other user reads or writes it
    try:
        copy = tmp / (root.name or "policy")
        if _snapshot(root, copy) != plan.files:
            raise RequestRefused(
                f"{plan.robot}: {plan.policy} changed on disk after the operator approved it; run again"
            )
        _, sep, query = str(policy).partition("?")
        yield f"{copy}{sep}{query}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _root(policy: object) -> Path | None:
    from ...policy.factory import local_root

    return local_root(policy)


def _snapshot(root: Path, into: Path | None) -> str:
    """Hash every file under ``root``, each read once; with ``into``, write the same bytes there as they are hashed."""
    total = hashlib.sha256()
    if into is not None:
        into.mkdir()
    for f in sorted(_files(root, set())):
        rel = f.relative_to(root)
        one = hashlib.sha256()
        with f.open("rb") as src:
            dst = None
            if into is not None:
                (into / rel).parent.mkdir(parents=True, exist_ok=True)
                dst = (into / rel).open("xb")
            try:
                while chunk := src.read(_CHUNK):  # chunks, not hashlib.file_digest: that is 3.11+, the floor is 3.10
                    one.update(chunk)
                    if dst is not None:
                        dst.write(chunk)
            finally:
                if dst is not None:
                    dst.close()
        total.update(f"{rel.as_posix()}\0{one.hexdigest()}\n".encode())
    return total.hexdigest()


def _files(folder: Path, seen: set[Path]) -> list[Path]:
    """Every regular file under ``folder``, through symlinks as a loader reads them, each real folder once."""
    if (real := folder.resolve()) in seen:
        return []
    seen.add(real)
    found: list[Path] = []
    for entry in folder.iterdir():
        found += _files(entry, seen) if entry.is_dir() else [entry] if entry.is_file() else []
    return found
