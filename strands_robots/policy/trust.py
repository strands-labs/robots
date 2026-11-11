"""A policy an agent names from this machine's disk loads only if no other user could have written it.

The agent picks the path; the bytes are whoever could write them. A LeRobot checkpoint rebuilds what its own JSON
names and a dataset is the motion a robot replays, so a folder another local user could write is that user's choice.
Every file the loader reads, the folders it walks and every folder above them must be owned by this process's user
or root and not writable by group or others. A sticky folder such as ``/tmp`` may sit above the policy (no one can
rename or replace an entry in it they do not own), never inside it. Then no other user can change a byte after the
check. The developer's own ``Policy(path)`` is not checked here.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any

_HOPS = 40  # symlinks followed on one path, as the kernel's own limit: a loop is refused, not walked forever


def walk(root: Path) -> tuple[list[Path], set[Path]]:
    """Every regular file under ``root`` and every real folder walked, through symlinks as a loader reads them."""
    files: list[Path] = []
    folders: set[Path] = set()

    def visit(folder: Path) -> None:
        if (real := folder.resolve()) in folders:
            return
        folders.add(real)
        for entry in folder.iterdir():
            if entry.is_dir():
                visit(entry)
            elif entry.is_file():
                files.append(entry)

    visit(root)
    return files, folders


def refuse_foreign(path: Path, st: os.stat_result, *, sticky_ok: bool = False) -> None:
    """A ``PermissionError`` if another user owns ``path`` or may write it (``st`` is its ``lstat`` or ``fstat``)."""
    if st.st_uid not in (os.geteuid(), 0):
        raise PermissionError(_why(path, f"is owned by uid {st.st_uid}, not this user ({os.geteuid()}) or root"))
    writers = st.st_mode & (0o002 if _private_group(st) else 0o022)
    if writers and not (sticky_ok and stat.S_ISDIR(st.st_mode) and st.st_mode & stat.S_ISVTX):
        raise PermissionError(_why(path, f"is writable by group or others (mode {stat.filemode(st.st_mode)})"))


def _private_group(st: os.stat_result) -> bool:
    """The file's group is its owner's own private group (Debian/Ubuntu's default, umask 002): no one else."""
    import grp
    import pwd

    try:
        owner, group = pwd.getpwuid(st.st_uid), grp.getgrgid(st.st_gid)
    except KeyError:
        return False
    return group.gr_gid == owner.pw_gid and group.gr_name == owner.pw_name and not group.gr_mem


def refuse_untrusted_local(spec: Any) -> None:
    """A ``PermissionError`` if ``spec`` is a checkpoint or dataset folder that another user could have written."""
    from .factory import local_root

    root = local_root(spec)
    if root is None:
        return
    files, folders = walk(root)
    for path in (root, *files):
        _check_path(path, folders)


def _check_path(path: Path, inside: set[Path]) -> None:
    """Each folder and the entry ``path`` passes through, one component at a time, symlinks followed hop by hop."""
    todo = list(Path(os.getcwd(), path).parts[1:])
    cur, hops = Path("/"), 0
    refuse_foreign(cur, os.lstat(cur), sticky_ok=bool(todo))
    while todo:
        name = todo.pop(0)
        if name == "..":
            cur = cur.parent
            continue
        nxt = cur / name
        st = os.lstat(nxt)
        if stat.S_ISLNK(st.st_mode):  # a sticky parent lets another plant their own link: check who owns it
            if st.st_uid not in (os.geteuid(), 0):
                raise PermissionError(_why(path, f"passes through {nxt}, a symlink owned by uid {st.st_uid}"))
            if (hops := hops + 1) > _HOPS:
                raise PermissionError(_why(path, "resolves through too many symlinks"))
            target = Path(os.readlink(nxt))
            cur = Path("/") if target.is_absolute() else cur
            todo = [p for p in target.parts if p != "/"] + todo
            continue
        refuse_foreign(nxt, st, sticky_ok=bool(todo) and nxt not in inside)
        cur = nxt


def _why(path: Path, reason: str) -> str:
    return (
        f"an agent may load a policy from this disk only if no other user could have written it: {path} {reason}; "
        "chown it to this user and chmod go-w (or keep it under your home folder), or pass it from your own code"
    )
