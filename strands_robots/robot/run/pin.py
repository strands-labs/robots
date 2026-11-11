"""The files an operator approves are the files an agent's run on a real body reads.

``files_digest`` fixes, when the plan is shown, where the policy's bytes will come from: a SHA-256 of what it would
read from this machine's disk, or ``owner/repo@<commit>`` for a Hugging Face Hub checkpoint. ``pinned`` then loads
exactly that: a private copy of the disk files taken in one pass (every byte written into the copy is the byte that
was hashed, so a write to the original at any moment after the yes never reaches the robot), or the Hub commit the
operator saw (a commit names immutable content). The same holds for a policy object the developer registered: a
LeRobot checkpoint reads its bytes when it attaches, after the yes, so it is re-pointed at the pinned source.
Every disk file is also checked on the open file itself: owned by this user or root, not writable by group or
others (``policy.trust``).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...policy.trust import refuse_foreign, walk

if TYPE_CHECKING:
    from .source import RunPlan

_CHUNK = 1 << 20
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
Source = Path | tuple[str, str | None] | None
"""Where a policy reads its bytes after the yes: a folder on this disk, a Hub ``(repo, revision)``, or nowhere."""


def source(policy: object) -> Source:
    """The folder or Hub repo ``policy`` loads from when it attaches; None if it reads nothing then."""
    from ...policy.factory import local_root
    from ...policy.hub import hub_repo

    if isinstance(policy, str | Path):
        root = local_root(policy)
        repo = None if root is not None else hub_repo(policy)
        return root if root is not None else (repo, None) if repo is not None else None
    fn = getattr(policy, "load_source", None)  # a LeRobot checkpoint object (a ChunkedPolicy forwards it)
    return fn() if callable(fn) else None


def files_digest(policy: object) -> str | None:
    """What the operator approves the policy's bytes as: a SHA-256 over the names and bytes of the files it reads
    from disk, ``owner/repo@<commit>`` for a Hub checkpoint, or None if it reads none. A Hub revision that cannot be
    resolved to a commit is ``RequestRefused`` here, before anyone is asked: a yes is never for a moving name."""
    src = source(policy)
    if isinstance(src, tuple):
        return f"{src[0]}@{_commit(*src)}"
    return None if src is None else _snapshot(src, None)


@contextmanager
def pinned(plan: RunPlan | None, policy: Any) -> Iterator[Any]:
    """The policy to load: for a plan that pinned a source, the same policy reading exactly it. A disk copy is
    checked against ``plan.files`` before anything loads (``RequestRefused`` if they differ) and removed when the
    ``with`` ends, however it ends. The developer's own policy object is never changed: a fresh one is loaded.
    """
    if plan is None:
        yield policy
        return
    from ..safety import RequestRefused

    changed = RequestRefused(f"{plan.robot}: {plan.policy} changed on disk after the operator approved it; run again")
    src = source(policy)
    if (src is None) != (plan.files is None):  # the yes was for a source, or for none: what loads must agree
        raise changed
    if src is None:
        yield policy
        return
    if isinstance(src, tuple):
        repo, _, commit = (plan.files or "").rpartition("@")
        if repo != src[0] or not _COMMIT.match(commit):
            raise RequestRefused(f"{plan.robot}: {plan.policy} is not the Hub commit the operator approved; run again")
        yield _repoint(policy, revision=commit)
        return
    tmp = Path(tempfile.mkdtemp(prefix="strands-run-"))  # 0700: no other user reads or writes it
    try:
        copy = tmp / (src.name or "policy")
        if _snapshot(src, copy) != plan.files:
            raise changed
        yield _repoint(policy, checkpoint=copy)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _repoint(policy: Any, *, checkpoint: Path | None = None, revision: str | None = None) -> Any:
    if not isinstance(policy, str | Path):
        loaded = policy.loading(checkpoint=checkpoint, revision=revision)
        if type(loaded) is not type(policy):  # a wrapper that only forwards loading() would be dropped, not pinned
            from ..safety import RequestRefused

            raise RequestRefused(
                f"{type(policy).__name__} wraps a checkpoint but has no loading() of its own, so the approved run "
                "cannot load it from the approved bytes; give it a loading(**kw) that wraps the inner policy's"
            )
        return loaded
    if revision is not None:
        from ...policy.factory import make

        return make(str(policy), revision=revision)
    _, sep, query = str(policy).partition("?")
    return f"{checkpoint}{sep}{query}"


def _commit(repo: str, revision: str | None) -> str:
    """The commit ``revision`` (default branch if None) of Hub ``repo`` is now; a full commit id needs no lookup."""
    if revision is not None and _COMMIT.match(revision):
        return revision
    from ..safety import RequestRefused

    fix = f"pass revision=<40-hex commit> (Policy({repo!r}, revision=...)) or a local checkpoint"
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RequestRefused(f"{repo}: cannot pin the approved commit without huggingface_hub; {fix}") from exc
    try:
        sha = HfApi().model_info(repo, revision=revision).sha
    except Exception as exc:  # offline, unknown repo or revision: the yes would be for a name that can still move
        raise RequestRefused(
            f"{repo}@{revision or 'main'}: cannot resolve the commit to approve ({exc}); {fix}"
        ) from exc
    if not isinstance(sha, str) or not _COMMIT.match(sha):
        raise RequestRefused(f"{repo}: the Hub returned no commit id ({sha!r}); {fix}")
    return sha


def _snapshot(root: Path, into: Path | None) -> str:
    """Hash every file under ``root``, each read once; with ``into``, write the same bytes there as they are hashed."""
    total = hashlib.sha256()
    if into is not None:
        into.mkdir(mode=0o700)  # the private copy is this process's own, whatever the umask: 0700 folders, 0600 files
    for f in sorted(walk(root)[0]):
        rel = f.relative_to(root)
        one = hashlib.sha256()
        with f.open("rb") as src:
            refuse_foreign(f, os.fstat(src.fileno()))  # the bytes hashed and copied: on the open file, not its name
            dst = None
            if into is not None:
                for folder in reversed(rel.parents[:-1]):  # one by one: mkdir(parents=True) gives the parents no mode=
                    (into / folder).mkdir(mode=0o700, exist_ok=True)
                dst = os.fdopen(os.open(into / rel, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")
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
