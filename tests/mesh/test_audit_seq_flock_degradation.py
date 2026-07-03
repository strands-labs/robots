"""Graceful-degradation contract for the cross-process seq flock.

``strands_robots.mesh.audit._seq_flock`` serialises per-peer sequence-number
allocation across processes with an ``flock``'d lockfile. Its documented
contract splits failure modes in two:

* An attacker pre-creating the lockfile as a symlink (``O_NOFOLLOW`` ->
  ``ELOOP``) is a HARD failure -- it raises ``SeqLockSymlinkError`` so the
  safety path records a poison record instead of silently downgrading the
  cross-process guarantee. That path is pinned elsewhere
  (``TestSeqFlockSymlinkRejection``).
* Ordinary operational failures -- the lockfile directory cannot be created,
  the lockfile cannot be opened for a non-symlink reason (``EACCES`` /
  ``ENOSPC``), or the best-effort unlock / close in the ``finally`` block
  fail -- must DEGRADE to yield-without-lock, never crash the caller. An
  audit-lock failure must not propagate into the safety code path.

These tests pin the degradation half of that contract.
"""

from __future__ import annotations

import errno
import pathlib

import pytest

from strands_robots.mesh import audit as mesh_audit

pytestmark = pytest.mark.skipif(
    not mesh_audit._HAS_FCNTL,
    reason="cross-process seq flock requires POSIX fcntl (not available here)",
)


@pytest.fixture
def audit_dir(tmp_path, monkeypatch):
    """Redirect the audit directory (and thus the seq lockfile) to tmp_path."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path))
    return tmp_path


def test_lockfile_dir_mkdir_failure_degrades_to_no_lock(audit_dir, monkeypatch):
    """An OSError creating the lockfile's parent dir yields without locking."""

    def boom(self, *args, **kwargs):
        raise OSError(errno.EACCES, "cannot create dir")

    monkeypatch.setattr(pathlib.Path, "mkdir", boom)

    entered = False
    with mesh_audit._seq_flock():
        entered = True
    # The block ran despite the mkdir failure -- degraded, did not raise.
    assert entered


def test_non_symlink_open_failure_degrades_to_no_lock(audit_dir, monkeypatch):
    """A non-ELOOP OSError opening the lockfile (e.g. EACCES) degrades, not raises."""
    real_open = mesh_audit.os.open

    def fake_open(path, flags, mode=0o600):
        if str(path).endswith("mesh_audit.seq.lock"):
            raise OSError(errno.EACCES, "permission denied")
        return real_open(path, flags, mode)

    monkeypatch.setattr(mesh_audit.os, "open", fake_open)

    entered = False
    with mesh_audit._seq_flock():
        entered = True
    assert entered


def test_finally_swallows_unlock_and_close_errors(audit_dir, monkeypatch):
    """Best-effort unlock + close failures in ``finally`` never surface to the caller."""
    ops: list[int] = []

    monkeypatch.setattr(mesh_audit.os, "open", lambda *a, **k: 4242)

    def fake_flock(fd, op):
        ops.append(op)
        if op == mesh_audit.fcntl.LOCK_UN:
            raise OSError(errno.EIO, "unlock failed")

    def fake_close(fd):
        raise OSError(errno.EIO, "close failed")

    monkeypatch.setattr(mesh_audit.fcntl, "flock", fake_flock)
    monkeypatch.setattr(mesh_audit.os, "close", fake_close)

    entered = False
    with mesh_audit._seq_flock():
        entered = True

    assert entered
    # Exclusive lock was taken on entry and unlock attempted on exit.
    assert mesh_audit.fcntl.LOCK_EX in ops
    assert mesh_audit.fcntl.LOCK_UN in ops


def test_symlink_open_still_hard_fails(audit_dir, monkeypatch):
    """Contract boundary: an ELOOP (symlink) open is NOT degraded -- it raises."""

    def fake_open(path, flags, mode=0o600):
        raise OSError(errno.ELOOP, "symlink refused")

    monkeypatch.setattr(mesh_audit.os, "open", fake_open)

    with pytest.raises(mesh_audit.SeqLockSymlinkError):
        with mesh_audit._seq_flock():
            pass
