"""Pin: TLS-key 0o600 mode check emits one-shot WARNING on non-POSIX.

``_resolve_tls_paths`` enforces mode 0o600 only under
``os.name == "posix"``. The README env-var matrix and the docstring on
``STRANDS_MESH_TLS_KEY`` promise mode-0o600 enforcement, so on Windows
the loader needs to surface that the documented contract is NOT being
checked -- otherwise an operator who reads the docs and ships a key
believes the loader is enforcing a guarantee it cannot.

The branch emits a one-shot WARNING naming the platform and the key
path so the operator can substitute filesystem ACLs (NTFS DACL).
One-shot via a module-level flag so the warning fires once per
process even though ``_resolve_tls_paths`` is invoked per-session
(potentially many times during a long-running peer's lifetime).

These tests pin the WARNING emit + one-shot semantics on the
non-POSIX branch and confirm the POSIX branch stays silent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from strands_robots.mesh import _zenoh_config as zc


@pytest.fixture
def reset_tls_warned():
    """Reset the per-key-path warning set around each test."""
    zc._NON_POSIX_TLS_WARNED_KEYS.clear()
    yield
    zc._NON_POSIX_TLS_WARNED_KEYS.clear()


def _make_tls_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create stub CA/cert/key files; key is mode 0o644 (would fail POSIX)."""
    ca = tmp_path / "ca.pem"
    cert = tmp_path / "cert.pem"
    key = tmp_path / "key.pem"
    for f in (ca, cert, key):
        f.write_text("---STUB---")
    key.chmod(0o644)  # would be rejected on POSIX; non-POSIX skips the check
    return ca, cert, key


def test_non_posix_emits_warning_once(monkeypatch, tmp_path, caplog, reset_tls_warned):
    """First call on non-POSIX emits WARNING; second call is silent."""
    ca, cert, key = _make_tls_files(tmp_path)
    monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
    monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
    monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
    # Force the non-POSIX branch.
    monkeypatch.setattr(zc, "_is_posix", lambda: False)

    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh._zenoh_config"):
        zc._resolve_tls_paths()
        first_warns = [r for r in caplog.records if "0o600" in r.getMessage()]
        assert len(first_warns) == 1, "first call must emit exactly one WARNING"

        caplog.clear()
        zc._resolve_tls_paths()
        second_warns = [r for r in caplog.records if "0o600" in r.getMessage()]
        assert second_warns == [], "second call must be silent (one-shot)"


def test_posix_does_not_emit_warning(monkeypatch, tmp_path, caplog, reset_tls_warned):
    """On POSIX, the actual mode check runs and the WARNING does NOT fire."""
    ca, cert, key = _make_tls_files(tmp_path)
    key.chmod(0o600)  # legit
    monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
    monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
    monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
    monkeypatch.setattr(zc, "_is_posix", lambda: True)

    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh._zenoh_config"):
        zc._resolve_tls_paths()
        warns = [r for r in caplog.records if "0o600" in r.getMessage()]
        assert warns == [], "POSIX must use the actual mode check, not the warning"


def test_warning_mentions_platform_and_path(monkeypatch, tmp_path, caplog, reset_tls_warned):
    """Warning surfaces enough context for an operator to act on it."""
    ca, cert, key = _make_tls_files(tmp_path)
    monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
    monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
    monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
    monkeypatch.setattr(zc, "_is_posix", lambda: False)

    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh._zenoh_config"):
        zc._resolve_tls_paths()
        msgs = [r.getMessage() for r in caplog.records if "0o600" in r.getMessage()]
        assert any("nt" in m for m in msgs), "warning must name the os.name value"
        assert any(str(key) in m for m in msgs), "warning must name the key path"
