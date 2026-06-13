"""Tests for Amazon Root CA1 pin verification + size-cap fetch.

The provisioner downloads ``AmazonRootCA1.pem`` over HTTPS and pins its
SHA-256 fingerprint before writing the file to disk. These tests exercise:

* :func:`_verify_ca_bytes` accepts the canonical bytes and rejects any
  one-byte modification.
* :func:`_ensure_ca` raises on a rogue download, on a tampered on-disk
  copy, and on responses larger than :data:`_CA_FETCH_MAX_BYTES`.
* The :envvar:`STRANDS_MESH_DISABLE_CA_PIN` break-glass env var bypasses
  the check (with a WARNING log) for proxy environments that legitimately
  re-encode the cert.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh.iot import provision

# Known-good copy of AmazonRootCA1.pem. If Amazon rotates this root the value
# below + provision._AMAZON_ROOT_CA1_PINS must both update together.
_REAL_CA = b"""-----BEGIN CERTIFICATE-----
MIIDQTCCAimgAwIBAgITBmyfz5m/jAo54vB4ikPmljZbyjANBgkqhkiG9w0BAQsF
ADA5MQswCQYDVQQGEwJVUzEPMA0GA1UEChMGQW1hem9uMRkwFwYDVQQDExBBbWF6
b24gUm9vdCBDQSAxMB4XDTE1MDUyNjAwMDAwMFoXDTM4MDExNzAwMDAwMFowOTEL
MAkGA1UEBhMCVVMxDzANBgNVBAoTBkFtYXpvbjEZMBcGA1UEAxMQQW1hem9uIFJv
b3QgQ0EgMTCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBALJ4gHHKeNXj
ca9HgFB0fW7Y14h29Jlo91ghYPl0hAEvrAIthtOgQ3pOsqTQNroBvo3bSMgHFzZM
9O6II8c+6zf1tRn4SWiw3te5djgdYZ6k/oI2peVKVuRF4fn9tBb6dNqcmzU5L/qw
IFAGbHrQgLKm+a/sRxmPUDgH3KKHOVj4utWp+UhnMJbulHheb4mjUcAwhmahRWa6
VOujw5H5SNz/0egwLX0tdHA114gk957EWW67c4cX8jJGKLhD+rcdqsq08p8kDi1L
93FcXmn/6pUCyziKrlA4b9v7LWIbxcceVOF34GfID5yHI9Y/QCB/IIDEgEw+OyQm
jgSubJrIqg0CAwEAAaNCMEAwDwYDVR0TAQH/BAUwAwEB/zAOBgNVHQ8BAf8EBAMC
AYYwHQYDVR0OBBYEFIQYzIU07LwMlJQuCFmcx7IQTgoIMA0GCSqGSIb3DQEBCwUA
A4IBAQCY8jdaQZChGsV2USggNiMOruYou6r4lK5IpDB/G/wkjUu0yKGX9rbxenDI
U5PMCCjjmCXPI6T53iHTfIUJrU6adTrCC2qJeHZERxhlbI1Bjjt/msv0tadQ1wUs
N+gDS63pYaACbvXy8MWy7Vu33PqUXHeeE6V/Uq2V8viTO96LXFvKWlJbYK8U90vv
o/ufQJVtMVT8QtPHRh8jrdkPSHCa2XV4cdFyQzR1bldZwgJcJmApzyMZFo6IQ6XU
5MsI+yMRQ+hDKXJioaldXgjUkK642M4UwtBV8ob2xJNDd2ZhwLnoQdeXeGADbkpy
rqXRfboQnoZsG4q5WTP468SQvvG5
-----END CERTIFICATE-----
"""


class TestPinVerification:
    def test_real_bytes_match(self):
        # Real bytes must match the constant in provision (else the constant
        # is wrong and every deployment will fail closed -- which is good).
        assert provision._verify_ca_bytes(_REAL_CA) is True

    def test_modified_bytes_rejected(self):
        # Flip one byte -> must fail.
        tampered = bytearray(_REAL_CA)
        tampered[100] ^= 0x01
        assert provision._verify_ca_bytes(bytes(tampered)) is False

    def test_empty_bytes_rejected(self):
        assert provision._verify_ca_bytes(b"") is False

    def test_breakglass_override_bypasses(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_DISABLE_CA_PIN", "true")
        assert provision._verify_ca_bytes(b"any garbage") is True

    def test_verify_ca_pin_path_helper(self, tmp_path):
        good = tmp_path / "ca.pem"
        good.write_bytes(_REAL_CA)
        assert provision.verify_ca_pin(good) is True

        bad = tmp_path / "rogue.pem"
        bad.write_bytes(b"-----FAKE CA-----")
        assert provision.verify_ca_pin(bad) is False

    def test_verify_missing_file_returns_false(self, tmp_path):
        assert provision.verify_ca_pin(tmp_path / "nope.pem") is False


class TestEnsureCA:
    def test_existing_clean_file_skipped(self, tmp_path):
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(_REAL_CA)
        # Should NOT make a network call
        with patch("strands_robots.mesh.iot.provision.urllib.request.urlopen") as mock_url:
            provision._ensure_ca(ca_path)
            mock_url.assert_not_called()

    def test_existing_clean_file_short_read_drains_correctly(self, tmp_path):
        # Issue #251: os.read may return short on interrupted syscalls /
        # unusual filesystems. The chunked-read loop in _ensure_ca must
        # drain the whole file so a valid on-disk CA is re-used (skipping
        # the download) rather than hashing a partial buffer and raising a
        # misleading "failed pin check".
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(_REAL_CA)
        real_os_read = provision.os.read

        def chunked_read(fd, n):
            # Force a one-byte-at-a-time short read to exercise the loop.
            return real_os_read(fd, 1)

        with patch("strands_robots.mesh.iot.provision.urllib.request.urlopen") as mock_url:
            with patch("strands_robots.mesh.iot.provision.os.read", side_effect=chunked_read):
                provision._ensure_ca(ca_path)
            mock_url.assert_not_called()

    def test_existing_tampered_file_raises(self, tmp_path):
        ca_path = tmp_path / "ca.pem"
        ca_path.write_bytes(b"rogue cert content")
        with patch("strands_robots.mesh.iot.provision.urllib.request.urlopen") as mock_url:
            with pytest.raises(RuntimeError, match="failed pin check"):
                provision._ensure_ca(ca_path)
            mock_url.assert_not_called()

    def test_download_writes_when_pin_matches(self, tmp_path):
        ca_path = tmp_path / "ca.pem"
        mock_resp = MagicMock()
        mock_resp.read.return_value = _REAL_CA
        mock_resp.__enter__ = lambda self: self
        mock_resp.__exit__ = lambda self, *a: None
        with patch("strands_robots.mesh.iot.provision.urllib.request.urlopen", return_value=mock_resp):
            provision._ensure_ca(ca_path)
        assert ca_path.read_bytes() == _REAL_CA

    def test_download_rogue_cert_rejected(self, tmp_path):
        # the download path is _download_with_per_socket_timeout,
        # which builds its own opener (no setdefaulttimeout). Patch the
        # helper directly so the test stays focused on the pin-mismatch
        # rejection rather than urllib internals.
        ca_path = tmp_path / "ca.pem"
        with patch(
            "strands_robots.mesh.iot.provision._download_with_per_socket_timeout",
            return_value=b"-----BEGIN ROGUE CERTIFICATE-----\n",
        ):
            with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
                provision._ensure_ca(ca_path)
        assert not ca_path.exists(), "rogue cert must NOT be written to disk"

    def test_download_oversized_rejected(self, tmp_path):
        # patch _download_with_per_socket_timeout directly. The
        # body-size cap is enforced after the download returns.
        ca_path = tmp_path / "ca.pem"
        big = b"X" * (provision._CA_FETCH_MAX_BYTES + 100)
        with patch(
            "strands_robots.mesh.iot.provision._download_with_per_socket_timeout",
            return_value=big,
        ):
            with pytest.raises(RuntimeError, match="exceeded"):
                provision._ensure_ca(ca_path)
        assert not ca_path.exists()


class TestVerifyCaPinSymlink:
    """verify_ca_pin must not follow symlinks.

    Symlink-following on the on-disk CA path is a TOCTOU gap: an attacker
    could race a symlink into place after _ensure_ca downloads but before
    verify_ca_pin reads.  Refusing O_NOFOLLOW at the verify layer closes
    the window.
    """

    def test_symlinked_ca_path_returns_false(self, tmp_path):
        from strands_robots.mesh.iot.provision import verify_ca_pin

        target = tmp_path / "real_ca.pem"
        target.write_bytes(b"-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----\n")
        symlink = tmp_path / "ca_link.pem"
        symlink.symlink_to(target)

        # verify_ca_pin must refuse to read through the symlink
        assert verify_ca_pin(symlink) is False


class TestMultiPinRotation:
    """Pin-tuple rotation regression: a follow-up cannot collapse the
    tuple back to a string without breaking this test.

    AGENTS.md > Review Learnings (#85) > "Pin regression tests for
    reviewed fixes" -- the move from a single ``str`` to a
    ``tuple[str, ...]`` exists *so that* a CA rotation can ship as a
    code-only deploy that adds the new pin alongside the old one.
    Without an explicit test that exercises the multi-entry path, every
    existing test still passes when someone "simplifies" the tuple back
    to a string -- and the rotation contract silently breaks.
    """

    def test_tuple_supports_multiple_pins(self, monkeypatch):
        import hashlib

        # Synthesize a second pin pointing at a fictional rotated CA.
        future_bytes = b"future-rotated-ca"
        future_pin = hashlib.sha256(future_bytes).hexdigest()

        # Append the new pin to the live tuple via monkeypatch to mirror
        # the rotation path: existing pin still accepted, new pin also
        # accepted. _hash_matches_pin reads the module-level constant
        # via _resolve_ca_pins, so the patch is visible.
        monkeypatch.setattr(
            provision,
            "_AMAZON_ROOT_CA1_PINS",
            provision._AMAZON_ROOT_CA1_PINS + (future_pin,),
        )

        # Original canonical CA bytes still pass.
        assert provision._hash_matches_pin(_REAL_CA) is True
        # New rotated CA bytes also pass.
        assert provision._hash_matches_pin(future_bytes) is True
        # Something that matches neither pin is still rejected.
        assert provision._hash_matches_pin(b"unrelated bytes") is False


class TestUnverifiedMarkerPermissions:
    """The CA-unverified sidecar marker must be owner-only (mode 0o600).

    AGENTS.md > Review Learnings (#85) > "Pin regression tests for
    reviewed fixes" + CodeQL py/overly-permissive-file-permission
    (alert #273): the marker is a local sentinel read only by this
    process via ``_ensure_ca`` to WARN about re-using a CA downloaded
    under the ``STRANDS_MESH_DISABLE_CA_PIN=true`` break-glass. No
    other user needs read access; world-readable mode (0o644) was
    flagged by CodeQL on the previous round and tightened to 0o600 in
    this commit.
    """

    def test_marker_written_owner_only_when_breakglass_active(self, tmp_path, monkeypatch):
        import stat

        ca_path = tmp_path / "ca.pem"
        monkeypatch.setenv("STRANDS_MESH_DISABLE_CA_PIN", "true")
        with patch(
            "strands_robots.mesh.iot.provision._download_with_per_socket_timeout",
            return_value=b"any-bytes-the-pin-bypass-accepts",
        ):
            provision._ensure_ca(ca_path)

        marker = ca_path.with_suffix(ca_path.suffix + ".unverified")
        assert marker.exists(), (
            "_ensure_ca must write the .unverified marker when the break-glass is active so subsequent runs can WARN."
        )

        mode = stat.S_IMODE(marker.stat().st_mode)
        assert mode == 0o600, (
            f"marker mode is 0o{mode:o}; must be 0o600 (owner-only). "
            "World-readable permissions on the unverified-CA sentinel "
            "were flagged by CodeQL py/overly-permissive-file-permission."
        )

    def test_marker_not_written_when_breakglass_inactive(self, tmp_path, monkeypatch):
        ca_path = tmp_path / "ca.pem"
        # Break-glass NOT set -- marker must not appear.
        monkeypatch.delenv("STRANDS_MESH_DISABLE_CA_PIN", raising=False)
        with patch(
            "strands_robots.mesh.iot.provision._download_with_per_socket_timeout",
            return_value=_REAL_CA,
        ):
            provision._ensure_ca(ca_path)

        marker = ca_path.with_suffix(ca_path.suffix + ".unverified")
        assert not marker.exists(), (
            "marker must only appear when the break-glass was active "
            "during the download; the canonical-CA path must not leak "
            "the sentinel."
        )


class TestUnverifiedReuseWarning:
    """Issue #261: re-using a CA whose origin was the break-glass must WARN.

    AGENTS.md > 'No silent defaults on error' -- the audit trail must
    survive past the single writing run. After a break-glass download
    leaves a ``.unverified`` sidecar, every later ``_ensure_ca`` re-use
    on a host where the env var is gone must emit exactly one WARNING
    per process so an operator can refresh via the canonical pinned
    path. These tests pin that the warning fires on re-use and that it
    is gated to one emission per CA path per process.
    """

    def _seed_breakglass_ca(self, ca_path, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_DISABLE_CA_PIN", "true")
        with patch(
            "strands_robots.mesh.iot.provision._download_with_per_socket_timeout",
            return_value=_REAL_CA,
        ):
            provision._ensure_ca(ca_path)
        monkeypatch.delenv("STRANDS_MESH_DISABLE_CA_PIN", raising=False)
        marker = ca_path.with_suffix(ca_path.suffix + ".unverified")
        assert marker.exists()

    def test_reuse_of_breakglass_ca_emits_warning(self, tmp_path, monkeypatch, caplog):
        ca_path = tmp_path / "ca.pem"
        self._seed_breakglass_ca(ca_path, monkeypatch)
        provision._UNVERIFIED_CA_WARNED.discard(ca_path)
        caplog.clear()
        with caplog.at_level("WARNING", logger="strands_robots.mesh.iot.provision"):
            provision._ensure_ca(ca_path)
        warnings = [
            r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "sidecar marker" in r.getMessage()
        ]
        assert len(warnings) == 1, (
            "re-using a CA written under the break-glass must emit one "
            f"WARNING about the unverified origin; got {warnings!r}"
        )

    def test_reuse_warning_fires_once_per_process(self, tmp_path, monkeypatch, caplog):
        ca_path = tmp_path / "ca.pem"
        self._seed_breakglass_ca(ca_path, monkeypatch)
        provision._UNVERIFIED_CA_WARNED.discard(ca_path)
        with caplog.at_level("WARNING", logger="strands_robots.mesh.iot.provision"):
            provision._ensure_ca(ca_path)
            provision._ensure_ca(ca_path)
            provision._ensure_ca(ca_path)
        warnings = [
            r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "sidecar marker" in r.getMessage()
        ]
        assert len(warnings) == 1, (
            f"the unverified-origin WARNING must be gated to one emission per CA path per process; got {len(warnings)}"
        )

    def test_reuse_without_marker_does_not_warn(self, tmp_path, monkeypatch, caplog):
        ca_path = tmp_path / "ca.pem"
        monkeypatch.delenv("STRANDS_MESH_DISABLE_CA_PIN", raising=False)
        with patch(
            "strands_robots.mesh.iot.provision._download_with_per_socket_timeout",
            return_value=_REAL_CA,
        ):
            provision._ensure_ca(ca_path)
        provision._UNVERIFIED_CA_WARNED.discard(ca_path)
        with caplog.at_level("WARNING", logger="strands_robots.mesh.iot.provision"):
            provision._ensure_ca(ca_path)
        warnings = [
            r.getMessage() for r in caplog.records if r.levelname == "WARNING" and "sidecar marker" in r.getMessage()
        ]
        assert warnings == [], (
            f"a canonically downloaded CA has no .unverified marker, so re-use must not WARN; got {warnings!r}"
        )


class TestEnsureCaV040Followups:
    """v0.4.0 mesh-iot follow-ups (#388) from the #228 review trail."""

    def test_ensure_ca_rejects_symlinked_path_with_explicit_message(self, tmp_path):
        """#312: a symlinked CA path must be rejected with an EXPLICIT 'SYMLINK'
        message (mirrors verify_ca_pin), not folded into a generic OSError text."""
        import os

        real = tmp_path / "real_ca.pem"
        real.write_bytes(_REAL_CA)
        link = tmp_path / "ca.pem"
        os.symlink(real, link)

        with pytest.raises(RuntimeError, match="SYMLINK"):
            provision._ensure_ca(link)

    def test_breakglass_marker_written_atomically_with_0600(self, tmp_path):
        """#311: the break-glass '.unverified' sidecar must be created with
        mode 0o600 atomically (no create-then-chmod world-readable window)."""
        import os
        import stat

        ca_path = tmp_path / "ca.pem"
        with patch("strands_robots.mesh.iot.provision._download_with_per_socket_timeout", return_value=_REAL_CA):
            with patch.dict(os.environ, {"STRANDS_MESH_DISABLE_CA_PIN": "true"}):
                provision._ensure_ca(ca_path)

        marker = ca_path.with_suffix(ca_path.suffix + ".unverified")
        assert marker.exists(), "break-glass download must write the .unverified marker"
        mode = stat.S_IMODE(marker.stat().st_mode)
        assert mode == 0o600, f"marker must be 0o600 (owner-only), got {oct(mode)}"

    def test_breakglass_marker_refuses_symlink(self, tmp_path):
        """#311: O_NOFOLLOW on the marker write refuses to follow a pre-planted
        symlink at the sidecar path (no write-through to an attacker target)."""
        import os

        ca_path = tmp_path / "ca.pem"
        marker = ca_path.with_suffix(ca_path.suffix + ".unverified")
        target = tmp_path / "attacker_target"
        target.write_text("original")
        os.symlink(target, marker)

        with patch("strands_robots.mesh.iot.provision._download_with_per_socket_timeout", return_value=_REAL_CA):
            with patch.dict(os.environ, {"STRANDS_MESH_DISABLE_CA_PIN": "true"}):
                # Marker write is best-effort: an O_NOFOLLOW refusal is caught
                # and logged, provisioning still succeeds. The attacker target
                # must NOT be overwritten through the symlink.
                provision._ensure_ca(ca_path)
        assert target.read_text() == "original", "O_NOFOLLOW must prevent write-through the symlink"
