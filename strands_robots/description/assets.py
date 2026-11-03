"""Robot assets that are not shipped by ``robot_descriptions``: one GitHub subdir, fetched once into a cache.

A registry entry may say ``asset.source == {"type": "github", "repo": "owner/name", "subdir": "path/in/repo"}``.
``resolve(entry)`` returns the model file under ``$STRANDS_ROBOTS_CACHE`` (default
``~/.cache/strands_robots``)``/<asset.dir>/``, downloading the repo's default-branch tarball and keeping only
that subdir the first time. Files the upstream stores in Git LFS come out of a tarball as ~130-byte text
pointers; those are replaced by the object they name (checked against the pointer's sha256 and size).
No git binary, no token, stdlib only.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

_REPO_RE = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_TARBALL_URL = "https://codeload.github.com/{repo}/tar.gz/HEAD"
_LFS_MEDIA_URL = "https://media.githubusercontent.com/media/{repo}/{sha}/{path}"
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"
_LFS_OID_RE = re.compile(rb"oid sha256:([0-9a-f]{64})")
_LFS_SIZE_RE = re.compile(rb"size (\d+)")


def cache_dir() -> Path:
    """Where fetched assets live: ``$STRANDS_ROBOTS_CACHE`` or ``~/.cache/strands_robots``."""
    return Path(os.environ.get("STRANDS_ROBOTS_CACHE") or Path.home() / ".cache" / "strands_robots").expanduser()


def resolve(entry: dict[str, Any], *, timeout: float = 120.0) -> Path:
    """Return the model file for a registry entry with a ``github`` source, fetching it on first use."""
    asset = entry["asset"]
    source = asset.get("source") or {}
    if source.get("type") != "github":
        raise ValueError(f"unsupported asset source type {source.get('type')!r} (only 'github' is fetched here)")
    repo = str(source.get("repo", ""))
    if not _REPO_RE.match(repo):
        raise ValueError(f"asset.source.repo must look like 'owner/name', got {repo!r}")
    subdir = str(source.get("subdir", "")).strip("/")
    if subdir and (".." in subdir.split("/") or subdir.startswith("/")):
        raise ValueError(f"asset.source.subdir may not escape the repository: {subdir!r}")
    dest = cache_dir() / asset["dir"]
    model = dest / asset["model_xml"]
    if not model.is_file():
        _fetch_github_subdir(repo, subdir, dest, timeout=timeout)
        if not model.is_file():
            raise FileNotFoundError(f"{repo}/{subdir} was fetched into {dest} but has no {asset['model_xml']}")
    return model


def _lfs_pointer(head: bytes) -> tuple[str, int] | None:
    """(oid, size) when ``head`` is a Git LFS pointer file, else None."""
    if not head.startswith(_LFS_MAGIC):
        return None
    oid, size = _LFS_OID_RE.search(head), _LFS_SIZE_RE.search(head)
    if oid is None or size is None:
        return None
    return oid.group(1).decode(), int(size.group(1))


def _fetch_github_subdir(repo: str, subdir: str, dest: Path, *, timeout: float = 120.0) -> int:
    """Download ``repo``'s default-branch tarball, keep only ``subdir``, move it into ``dest``; returns bytes."""
    url = _TARBALL_URL.format(repo=repo)
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        blob = resp.read()
    prefix = subdir + "/" if subdir else ""
    written = 0
    pointers: list[tuple[Path, str, str, int]] = []
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}-", dir=dest.parent))
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
            members = tar.getmembers()
            if not members:
                raise FileNotFoundError(f"empty tarball for {repo}")
            top = members[0].name.split("/", 1)[0]  # "<name>-<sha>"
            sha = top.rsplit("-", 1)[-1]
            found = False
            for m in members:
                rel = m.name.split("/", 1)[1] if "/" in m.name else ""
                if not rel.startswith(prefix) or not (m.isfile() or m.isdir()):
                    continue
                found = True
                sub = rel[len(prefix) :]
                if not sub:
                    continue
                target = tmp / sub
                if m.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                f = tar.extractfile(m)
                data = f.read() if f is not None else b""
                target.write_bytes(data)
                written += len(data)
                ptr = _lfs_pointer(data[:256])
                if ptr is not None:
                    pointers.append((target, rel, *ptr))
            if not found:
                raise FileNotFoundError(f"subdir {subdir!r} not found in {repo}")
        for target, rel, oid, size in pointers:
            written += _fetch_lfs_object(repo, sha, rel, target, oid, size, timeout=timeout)
        if dest.exists():
            shutil.rmtree(dest)
        tmp.rename(dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return written


def _fetch_lfs_object(repo: str, sha: str, rel: str, target: Path, oid: str, size: int, *, timeout: float) -> int:
    url = _LFS_MEDIA_URL.format(repo=repo, sha=sha, path=urllib.parse.quote(rel))
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        data = resp.read(size + 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != oid:
        raise OSError(f"{rel}: Git LFS object from {url} does not match its pointer (size/sha256)")
    target.write_bytes(data)
    return len(data)
