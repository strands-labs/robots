"""Robot assets that are not shipped by ``robot_descriptions``: one GitHub subdir, fetched once into a cache.

A registry entry may say ``asset.source == {"type": "github", "repo": "owner/name", "subdir": "path/in/repo",
"commit": "<40-hex sha>", "sha256": "<tree digest>"}``. ``resolve(entry)`` returns the model file under
``$STRANDS_ROBOTS_CACHE`` (default ``~/.cache/strands_robots``)``/<asset.dir>/``, downloading the tarball of that
exact commit and keeping only that subdir the first time. Files the upstream stores in Git LFS come out of a tarball
as ~130-byte text pointers; those are replaced by the object they name (checked against the pointer's sha256 and
size). The kept files must then hash to ``sha256`` (``tree_digest``) or nothing is cached: the joint limits the
safety chain enforces come from these files, so they only change through a reviewed registry edit. Member names that
are absolute or contain ``..`` refuse the whole tarball. The cache records the pin it was fetched at
(``.source.json``); a cache from another pin is fetched again. No git binary, no token, stdlib only.
"""

from __future__ import annotations

import hashlib
import io
import json
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
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_TARBALL_URL = "https://codeload.github.com/{repo}/tar.gz/{commit}"
_LFS_MEDIA_URL = "https://media.githubusercontent.com/media/{repo}/{sha}/{path}"
_LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"
_LFS_OID_RE = re.compile(rb"oid sha256:([0-9a-f]{64})")
_LFS_SIZE_RE = re.compile(rb"size (\d+)")
_MARKER = ".source.json"


class AssetMismatch(OSError):
    """A fetched asset is not the one the registry pins (wrong commit, wrong content, or a hostile member name)."""


def cache_dir() -> Path:
    """Where fetched assets live: ``$STRANDS_ROBOTS_CACHE`` or ``~/.cache/strands_robots``."""
    return Path(os.environ.get("STRANDS_ROBOTS_CACHE") or Path.home() / ".cache" / "strands_robots").expanduser()


def tree_digest(root: Path) -> str:
    """sha256 over every file under ``root`` (sorted ``relpath NUL sha256(bytes) LF`` lines; the marker excluded)."""
    h = hashlib.sha256()
    for p in sorted(q for q in root.rglob("*") if q.is_file() and q.name != _MARKER):
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode() + b"\0" + hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"\n")
    return h.hexdigest()


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
    commit, digest = str(source.get("commit", "")), str(source.get("sha256", ""))
    if not _COMMIT_RE.match(commit) or not _DIGEST_RE.match(digest):
        raise ValueError(
            f"asset.source for {repo} needs a pinned 'commit' (40 hex) and the subdir's 'sha256' "
            "(description.assets.tree_digest); an unpinned model is never fetched"
        )
    dest = cache_dir() / asset["dir"]
    model = dest / asset["model_xml"]
    pin = {"repo": repo, "subdir": subdir, "commit": commit, "sha256": digest}
    if not model.is_file() or _cached_pin(dest) != pin:
        _fetch_github_subdir(repo, subdir, dest, commit=commit, sha256=digest, timeout=timeout)
        if not model.is_file():
            raise FileNotFoundError(f"{repo}/{subdir} was fetched into {dest} but has no {asset['model_xml']}")
    return model


def _cached_pin(dest: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((dest / _MARKER).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _lfs_pointer(head: bytes) -> tuple[str, int] | None:
    """(oid, size) when ``head`` is a Git LFS pointer file, else None."""
    if not head.startswith(_LFS_MAGIC):
        return None
    oid, size = _LFS_OID_RE.search(head), _LFS_SIZE_RE.search(head)
    if oid is None or size is None:
        return None
    return oid.group(1).decode(), int(size.group(1))


def _member_path(name: str, prefix: str) -> str | None:
    """The path of a tar member below ``prefix`` (None when outside it); refuses absolute and ``..`` names."""
    parts = name.split("/")
    if name.startswith("/") or ".." in parts or "\\" in name:
        raise AssetMismatch(f"tarball member {name!r} is absolute or leaves its directory; refusing the archive")
    rel = "/".join(parts[1:])
    if not rel.startswith(prefix):
        return None
    sub = rel[len(prefix) :]
    if sub.startswith("/") or "//" in sub:
        raise AssetMismatch(f"tarball member {name!r} has an empty path segment; refusing the archive")
    return sub.rstrip("/")


def _fetch_github_subdir(
    repo: str, subdir: str, dest: Path, *, commit: str, sha256: str, timeout: float = 120.0
) -> int:
    """Download ``repo`` at ``commit``, keep only ``subdir``, check it against ``sha256``, move it into ``dest``."""
    url = _TARBALL_URL.format(repo=repo, commit=commit)
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
            if top.rsplit("-", 1)[-1] != commit:
                raise AssetMismatch(f"{url} unpacked to {top!r}, not commit {commit}")
            found = False
            for m in members:
                if m.name.split("/", 1)[0] != top:
                    raise AssetMismatch(f"tarball member {m.name!r} is outside {top!r}; refusing the archive")
                sub = _member_path(m.name, prefix)
                if sub is None or not (m.isfile() or m.isdir()):
                    continue
                found = True
                if not sub:
                    continue
                target = tmp / sub
                if not target.resolve().is_relative_to(tmp.resolve()):
                    raise AssetMismatch(f"tarball member {m.name!r} leaves the cache; refusing the archive")
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
                    pointers.append((target, "/".join(m.name.split("/")[1:]), *ptr))
            if not found:
                raise FileNotFoundError(f"subdir {subdir!r} not found in {repo}")
        for target, rel, oid, size in pointers:
            written += _fetch_lfs_object(repo, commit, rel, target, oid, size, timeout=timeout)
        got = tree_digest(tmp)
        if got != sha256:
            raise AssetMismatch(
                f"{repo}/{subdir} at {commit} hashes to {got}, not the registry's {sha256}; nothing was cached"
            )
        pin = {"repo": repo, "subdir": subdir, "commit": commit, "sha256": sha256}
        (tmp / _MARKER).write_text(json.dumps(pin, sort_keys=True))
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
