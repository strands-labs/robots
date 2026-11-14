"""The directory a dataset lives in, held so no other account can move it while a take writes or deletes it.

LeRobot writes a dataset by its path for the whole take, and an aborted take deletes it by path. A directory on that
path that another account can rename (a shared folder, a symlink swapped in after a check) would send those writes,
or that delete, somewhere else. So a dataset directory is used only when every directory on its path is a real
directory (no symlink) owned by this account or root, and writable by no one else unless sticky (a sticky
directory's entries can be renamed only by their owner, which the next step checks): the rule ssh applies to
``~/.ssh``. The walk goes one directory fd at a time with ``O_NOFOLLOW``, so a name cannot be swapped mid-walk.
"""

from __future__ import annotations

import os
import shutil
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path

_POSIX = hasattr(os, "getuid") and hasattr(os, "O_NOFOLLOW")
# O_PATH (Linux) needs only search permission on the parent, as a by-name mkdir/open does: a 711 /home still walks
_DIR = getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


class Place:
    """``root`` checked by :func:`claim`; :meth:`bind` once the dataset exists, :meth:`remove` to delete it."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._inode: tuple[int, int] | None = None

    def bind(self) -> None:
        """Note which directory the dataset is (it exists now): :meth:`remove` deletes only that one."""
        self._inode = _walk(self.root, create=False)

    def same(self) -> bool:
        """Whether ``root`` still names the directory :meth:`bind` saw, every step of the way still trusted."""
        if self._inode is None:
            return False
        try:
            return _walk(self.root, create=False) == self._inode
        except OSError:  # a symlink or an untrusted directory on the path now (PermissionError), or it is gone
            return False

    def remove(self) -> bool:
        """Delete the dataset directory if ``root`` still names the one :meth:`bind` saw; False (kept) otherwise."""
        if not self.same():
            return False
        shutil.rmtree(self.root, ignore_errors=True)  # no other account can rename a step of the path: it stays put
        return True


def claim(root: Path) -> Place:
    """Check every directory on ``root``'s path and create the missing parents (not ``root``: LeRobot creates it).

    Raises ``PermissionError`` naming the first directory another account could rename, or a symlink on the path.
    ``root`` is absolute and resolved (:func:`~strands_robots.recording.target.locate`)."""
    if not root.is_absolute():
        raise ValueError(f"a dataset directory is an absolute path, got {str(root)!r}")
    _walk(root.parent, create=True)
    if os.path.lexists(root):
        _walk(root, create=False)
    return Place(root)


def _walk(path: Path, *, create: bool) -> tuple[int, int]:
    """``(st_dev, st_ino)`` of ``path``, reached one trusted directory at a time; ``create`` makes missing ones."""
    if not _POSIX:  # no uid to own a directory or O_NOFOLLOW: create and stat by name, as before
        if create:
            path.mkdir(parents=True, exist_ok=True)
        st = os.stat(path)
        return st.st_dev, st.st_ino
    with ExitStack() as fds:
        fd = fds.enter_context(_opendir("/"))  # every fd closes when the walk unwinds, on any path out
        st = _trusted(os.fstat(fd), Path("/"))
        here = Path("/")
        for name in path.parts[1:]:
            here = here / name
            if create:
                with suppress(FileExistsError):  # an existing directory is checked below like a new one
                    os.mkdir(name, 0o755, dir_fd=fd)
            try:
                fd = fds.enter_context(_opendir(name, dir_fd=fd))
            except OSError:
                if os.path.islink(here):
                    raise PermissionError(f"{here} is a symlink: a dataset path may not go through one") from None
                raise
            st = _trusted(os.fstat(fd), here)
        return st.st_dev, st.st_ino


@contextmanager
def _opendir(name: str, *, dir_fd: int | None = None) -> Iterator[int]:
    """A directory fd opened with ``O_NOFOLLOW``, always closed when its context leaves."""
    fd = os.open(name, _DIR, dir_fd=dir_fd)
    try:
        yield fd
    finally:
        os.close(fd)


def _trusted(st: os.stat_result, where: Path) -> os.stat_result:
    if not stat.S_ISDIR(st.st_mode):
        raise PermissionError(f"{where} is not a directory")
    others_write = st.st_mode & (stat.S_IWOTH if _private_group(st) else stat.S_IWGRP | stat.S_IWOTH)
    if st.st_uid not in (0, os.getuid()) or (others_write and not st.st_mode & stat.S_ISVTX):
        raise PermissionError(
            f"{where} (owner uid {st.st_uid}, mode {stat.S_IMODE(st.st_mode):o}) can be changed by another account, "
            "so a dataset under it could be moved while it is written: record into a directory only you can change "
            "(chmod go-w), or one with the sticky bit"
        )
    return st


def _private_group(st: os.stat_result) -> bool:
    """The directory's group is its owner's own private group (Debian/Ubuntu's default, umask 002): no one else."""
    import grp
    import pwd

    try:
        owner, group = pwd.getpwuid(st.st_uid), grp.getgrgid(st.st_gid)
    except KeyError:
        return False
    return group.gr_gid == owner.pw_gid and group.gr_name == owner.pw_name and not group.gr_mem
