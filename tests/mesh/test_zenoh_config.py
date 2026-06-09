"""Tests for :mod:`strands_robots.mesh._zenoh_config`.

These exercise the pure-function config builders (no Zenoh session
required). The integration smoke that the emitted JSON5 actually
parses cleanly through Zenoh's Rust ``Config`` validator lives in
``test_session_config.py``.
"""

from __future__ import annotations

import json
import os

import pytest

from strands_robots.mesh import _zenoh_config as zc


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Each test runs without inherited STRANDS_MESH_* env vars."""
    for key in [
        "STRANDS_MESH_NAMESPACE",
        "STRANDS_MESH_MULTICAST",
        "STRANDS_MESH_MAX_SESSIONS",
        "STRANDS_MESH_MAX_CMD_BYTES",
        "STRANDS_MESH_MAX_CAMERA_BYTES",
        "STRANDS_MESH_CMD_RATE_HZ",
        "STRANDS_MESH_AUTH_MODE",
        "STRANDS_MESH_TLS_CA",
        "STRANDS_MESH_TLS_CERT",
        "STRANDS_MESH_TLS_KEY",
        "STRANDS_MESH_LOCAL_DEV",
        "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE",
    ]:
        monkeypatch.delenv(key, raising=False)


# --- namespace ----------------------------------------------------------


class TestNamespace:
    def test_default(self):
        assert zc.resolve_namespace() == "strands"

    def test_override(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_NAMESPACE", "fleet_42")
        assert zc.resolve_namespace() == "fleet_42"

    def test_empty_falls_through_to_default(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_NAMESPACE", "   ")
        assert zc.resolve_namespace() == "strands"

    def test_namespace_block_returns_json5_string(self):
        path, value = zc.namespace_block()
        assert path == "namespace"
        assert json.loads(value) == "strands"


# --- auth mode ----------------------------------------------------------


class TestAuthMode:
    def test_default_is_mtls(self):
        assert zc.resolve_auth_mode() == "mtls"

    def test_explicit_none_with_optin(self, monkeypatch):
        # B2: "none" requires the second-factor env var.
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
        monkeypatch.setenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", "1")
        assert zc.resolve_auth_mode() == "none"

    def test_explicit_none_without_optin_raises(self, monkeypatch):
        # B2 pin: auth_mode=none without the second-factor env var
        # must raise. This is what prevents a typo / forgotten env / leaked
        # CI fixture from silently disabling wire auth.
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
        monkeypatch.delenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", raising=False)
        with pytest.raises(ValueError, match="STRANDS_MESH_I_KNOW_THIS_IS_INSECURE"):
            zc.resolve_auth_mode()

    def test_explicit_none_optin_accepts_truthy_strings(self, monkeypatch):
        # The opt-in is case-insensitive: 1, true, yes all work.
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
        for val in ("1", "true", "TRUE", "yes", "Yes"):
            monkeypatch.setenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", val)
            assert zc.resolve_auth_mode() == "none"

    def test_explicit_none_optin_rejects_garbage(self, monkeypatch):
        # Non-truthy values do NOT count as opt-in.
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
        for val in ("0", "false", "no", "maybe", ""):
            monkeypatch.setenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", val)
            with pytest.raises(ValueError, match="STRANDS_MESH_I_KNOW_THIS_IS_INSECURE"):
                zc.resolve_auth_mode()

    def test_typo_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtsl")
        with pytest.raises(ValueError, match="not supported"):
            zc.resolve_auth_mode()


class TestLocalDevAuthPreset:
    """STRANDS_MESH_LOCAL_DEV selects the one-variable localhost auth preset.

    Setting it alone defaults the mesh auth mode to ``none`` and acts as its
    own insecure-acknowledgement second factor; an explicit AUTH_MODE still wins.
    """

    def test_local_dev_defaults_to_none_without_second_factor(self, monkeypatch):
        # LOCAL_DEV alone defaults auth to 'none' AND is its own second factor:
        # no STRANDS_MESH_I_KNOW_THIS_IS_INSECURE required.
        monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
        monkeypatch.delenv("STRANDS_MESH_AUTH_MODE", raising=False)
        monkeypatch.delenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", raising=False)
        assert zc.resolve_auth_mode() == "none"

    def test_local_dev_accepts_truthy_strings(self, monkeypatch):
        monkeypatch.delenv("STRANDS_MESH_AUTH_MODE", raising=False)
        monkeypatch.delenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", raising=False)
        for val in ("1", "true", "TRUE", "yes", "Yes"):
            monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", val)
            assert zc.resolve_auth_mode() == "none"

    def test_local_dev_falsy_does_not_engage(self, monkeypatch):
        # A falsy LOCAL_DEV must NOT lower the default below mtls.
        monkeypatch.delenv("STRANDS_MESH_AUTH_MODE", raising=False)
        for val in ("0", "false", "no", ""):
            monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", val)
            assert zc.resolve_auth_mode() == "mtls"

    def test_explicit_mtls_overrides_local_dev(self, monkeypatch):
        # An explicit AUTH_MODE=mtls wins even under LOCAL_DEV.
        monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        assert zc.resolve_auth_mode() == "mtls"

    def test_explicit_none_under_local_dev_needs_no_second_factor(self, monkeypatch):
        # AUTH_MODE=none + LOCAL_DEV is fine without _I_KNOW_THIS_IS_INSECURE,
        # because LOCAL_DEV is the acknowledgement.
        monkeypatch.setenv("STRANDS_MESH_LOCAL_DEV", "1")
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
        monkeypatch.delenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", raising=False)
        assert zc.resolve_auth_mode() == "none"

    def test_none_without_local_dev_still_needs_second_factor(self, monkeypatch):
        # Regression guard: turning LOCAL_DEV off restores the strict gate.
        monkeypatch.delenv("STRANDS_MESH_LOCAL_DEV", raising=False)
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
        monkeypatch.delenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", raising=False)
        with pytest.raises(ValueError, match="STRANDS_MESH_I_KNOW_THIS_IS_INSECURE"):
            zc.resolve_auth_mode()


# --- scouting -----------------------------------------------------------


class TestScouting:
    def test_default_is_multicast_off_gossip_on(self):
        out = dict(zc.scouting_block())
        assert out["scouting/multicast/enabled"] == "false"
        assert out["scouting/gossip/enabled"] == "true"

    def test_multicast_can_be_enabled(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MULTICAST", "true")
        out = dict(zc.scouting_block())
        assert out["scouting/multicast/enabled"] == "true"

    def test_invalid_bool_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MULTICAST", "maybe")
        with pytest.raises(ValueError, match="boolean"):
            zc.scouting_block()


# --- transport caps -----------------------------------------------------


class TestTransportCaps:
    def test_default_max_sessions(self):
        out = dict(zc.transport_caps_block())
        assert out["transport/unicast/max_sessions"] == "256"

    def test_override(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_SESSIONS", "1024")
        out = dict(zc.transport_caps_block())
        assert out["transport/unicast/max_sessions"] == "1024"

    def test_oob_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_SESSIONS", "100000")
        with pytest.raises(ValueError, match="out of bounds"):
            zc.transport_caps_block()

    def test_zero_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_SESSIONS", "0")
        with pytest.raises(ValueError):
            zc.transport_caps_block()


# --- downsampling -------------------------------------------------------


class TestDownsampling:
    def test_default_freq(self):
        path, value = zc.downsampling_block()
        assert path == "downsampling"
        decoded = json.loads(value)
        assert decoded[0]["id"] == "strands_cmd_rate_cap"
        assert decoded[0]["messages"] == ["put"]
        assert decoded[0]["flows"] == ["ingress"]
        rules = {r["key_expr"]: r["freq"] for r in decoded[0]["rules"]}
        assert rules["**/cmd"] == zc.DEFAULT_CMD_RATE_HZ
        assert rules["**/broadcast"] == zc.DEFAULT_CMD_RATE_HZ

    def test_freq_override(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CMD_RATE_HZ", "5.0")
        _, value = zc.downsampling_block()
        decoded = json.loads(value)
        rules = {r["key_expr"]: r["freq"] for r in decoded[0]["rules"]}
        assert rules["**/cmd"] == 5.0

    def test_freq_oob_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_CMD_RATE_HZ", "9999999")
        with pytest.raises(ValueError):
            zc.downsampling_block()


# --- low_pass_filter ----------------------------------------------------


class TestLowPassFilter:
    def test_default_caps(self):
        path, value = zc.low_pass_filter_block()
        assert path == "low_pass_filter"
        decoded = json.loads(value)
        cmd, cam = decoded[0], decoded[1]
        assert cmd["id"] == "strands_cmd_size_cap"
        assert cmd["size_limit"] == zc.DEFAULT_MAX_CMD_BYTES
        assert "**/cmd" in cmd["key_exprs"]
        assert "**/broadcast" in cmd["key_exprs"]
        assert cam["id"] == "strands_camera_size_cap"
        assert cam["size_limit"] == zc.DEFAULT_MAX_CAMERA_BYTES
        assert "**/camera/**" in cam["key_exprs"]

    def test_default_omits_interfaces_for_wildcard_binding(self):
        """by default, ``interfaces`` MUST be absent so Zenoh applies
        the cap to every link via ``SubjectProperty::Wildcard`` (see
        zenoh/src/net/routing/interceptor/low_pass.rs:84-91 in 1.x).

        Pre-fix code enumerated NICs via psutil with a hardcoded
        fallback list (``lo, eth0, en0,...``); on hosts using
        ``enp0s3`` / ``wlp2s0`` / ``cni0`` / ``wg0`` without psutil,
        the cap silently bypassed because no listed NIC matched.
        """
        _, value = zc.low_pass_filter_block()
        decoded = json.loads(value)
        for rule in decoded:
            assert "interfaces" not in rule, (
                f"rule {rule['id']!r} carries `interfaces` by default; "
                "this re-introduces the F1 silent-bypass footgun on hosts "
                "with non-canonical NIC names."
            )

    def test_explicit_filter_interfaces_honoured(self, monkeypatch):
        """when STRANDS_MESH_FILTER_INTERFACES is set, the
        operator-supplied list is attached to every rule verbatim.
        """
        monkeypatch.setenv("STRANDS_MESH_FILTER_INTERFACES", "wlan0, br-mesh ,wg0")
        _, value = zc.low_pass_filter_block()
        decoded = json.loads(value)
        for rule in decoded:
            assert rule["interfaces"] == ["wlan0", "br-mesh", "wg0"]

    def test_empty_filter_interfaces_treated_as_unset(self, monkeypatch):
        """Whitespace / empty STRANDS_MESH_FILTER_INTERFACES falls through
        to the wildcard binding rather than emitting ``[]`` (which Zenoh
        rejects with ``Found empty interface value``).
        """
        monkeypatch.setenv("STRANDS_MESH_FILTER_INTERFACES", "   ,, ,")
        _, value = zc.low_pass_filter_block()
        decoded = json.loads(value)
        for rule in decoded:
            assert "interfaces" not in rule

    def test_override_cmd_bytes(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_CMD_BYTES", "8192")
        _, value = zc.low_pass_filter_block()
        decoded = json.loads(value)
        assert decoded[0]["size_limit"] == 8192

    def test_oversize_cap_rejected(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_CMD_BYTES", "999999999999")
        with pytest.raises(ValueError):
            zc.low_pass_filter_block()


# --- adminspace ---------------------------------------------------------


def test_adminspace_block_disabled():
    path, value = zc.adminspace_block()
    assert path == "adminspace"
    decoded = json.loads(value)
    assert decoded["enabled"] is False
    assert decoded["permissions"] == {"read": False, "write": False}


# --- mTLS ---------------------------------------------------------------


class TestTLSBlock:
    def test_missing_paths_raise(self):
        # No STRANDS_MESH_TLS_* env vars set.
        with pytest.raises(ValueError, match="STRANDS_MESH_TLS"):
            zc.tls_block()

    def test_nonexistent_files_raise(self, monkeypatch, tmp_path):
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(tmp_path / "missing.crt"))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(tmp_path / "missing.crt"))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(tmp_path / "missing.key"))
        with pytest.raises(FileNotFoundError):
            zc.tls_block()

    def test_valid_paths_emit_block(self, monkeypatch, tmp_path):
        ca = tmp_path / "ca.crt"
        cert = tmp_path / "peer.crt"
        key = tmp_path / "peer.key"
        for f in (ca, cert, key):
            f.write_text("dummy\n")
        # _resolve_tls_paths enforces mode 0o600 on the private key.
        key.chmod(0o600)

        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))

        path, value = zc.tls_block()
        assert path == "transport/link/tls"
        decoded = json.loads(value)
        assert decoded["enable_mtls"] is True
        assert decoded["verify_name_on_connect"] is True
        assert decoded["close_link_on_expiration"] is True
        assert decoded["root_ca_certificate"] == str(ca)
        assert decoded["listen_certificate"] == str(cert)
        assert decoded["connect_certificate"] == str(cert)
        assert decoded["listen_private_key"] == str(key)
        assert decoded["connect_private_key"] == str(key)


class TestTLSKeyMode:
    """R24-C: the mode 0o600 contract from docstring + README must be enforced."""

    def _make_tls_files(self, tmp_path, key_mode):
        ca = tmp_path / "ca.crt"
        cert = tmp_path / "peer.crt"
        key = tmp_path / "peer.key"
        for f in (ca, cert, key):
            f.write_text("dummy\n")
        key.chmod(key_mode)
        return ca, cert, key

    @pytest.mark.skipif(__import__("os").name != "posix", reason="POSIX file modes only")
    def test_world_readable_key_rejected(self, monkeypatch, tmp_path):
        """A 0o644 key (world-readable) raises ValueError naming the failure mode."""
        ca, cert, key = self._make_tls_files(tmp_path, 0o644)
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
        with pytest.raises(ValueError, match="refusing world/group readable"):
            zc.tls_block()

    @pytest.mark.skipif(__import__("os").name != "posix", reason="POSIX file modes only")
    def test_group_readable_key_rejected(self, monkeypatch, tmp_path):
        """A 0o640 key (group-readable) is also rejected -- shared-host exfiltration surface."""
        ca, cert, key = self._make_tls_files(tmp_path, 0o640)
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
        with pytest.raises(ValueError, match="0o640"):
            zc.tls_block()

    @pytest.mark.skipif(__import__("os").name != "posix", reason="POSIX file modes only")
    def test_owner_only_key_accepted(self, monkeypatch, tmp_path):
        """0o600 (the documented contract) is the only mode that passes."""
        ca, cert, key = self._make_tls_files(tmp_path, 0o600)
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
        # No raise.
        path, _value = zc.tls_block()
        assert path == "transport/link/tls"

    @pytest.mark.skipif(__import__("os").name != "posix", reason="POSIX file modes only")
    def test_owner_rwx_only_accepted(self, monkeypatch, tmp_path):
        """0o700 also passes (no group/world bits set); the gate is on group+world, not owner."""
        ca, cert, key = self._make_tls_files(tmp_path, 0o700)
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
        path, _value = zc.tls_block()
        assert path == "transport/link/tls"


def test_link_protocols_block_restricts_to_tls():
    path, value = zc.link_protocols_block()
    assert path == "transport/link/protocols"
    assert json.loads(value) == ["tls"]


# === TLS key/cert/CA symlink rejection ===


class TestTlsBlockSymlinkReject:
    """The mTLS file resolver must refuse symlinks for the key/cert/CA
    paths (prior pin). Without it, an operator setting
    ``STRANDS_MESH_TLS_KEY=/safe/key.pem`` pointing at an attacker-
    writable target whose mode is 0o600 silently passes.
    """

    @pytest.fixture
    def _tls_files(self, tmp_path):
        ca = tmp_path / "ca.pem"
        cert = tmp_path / "cert.pem"
        key = tmp_path / "key.pem"
        for p in (ca, cert, key):
            p.write_text("placeholder")
        os.chmod(key, 0o600)
        return ca, cert, key

    def test_symlinked_key_rejected(self, _tls_files, tmp_path, monkeypatch):
        ca, cert, key = _tls_files
        symlink = tmp_path / "key_link.pem"
        symlink.symlink_to(key)
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(symlink))
        with pytest.raises(ValueError, match=r"is a SYMLINK"):
            zc.tls_block()

    def test_symlinked_cert_rejected(self, _tls_files, tmp_path, monkeypatch):
        ca, cert, key = _tls_files
        symlink = tmp_path / "cert_link.pem"
        symlink.symlink_to(cert)
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(symlink))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
        with pytest.raises(ValueError, match=r"is a SYMLINK"):
            zc.tls_block()

    def test_symlinked_ca_rejected(self, _tls_files, tmp_path, monkeypatch):
        ca, cert, key = _tls_files
        symlink = tmp_path / "ca_link.pem"
        symlink.symlink_to(ca)
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(symlink))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
        with pytest.raises(ValueError, match=r"is a SYMLINK"):
            zc.tls_block()

    def test_real_files_pass(self, _tls_files, monkeypatch):
        ca, cert, key = _tls_files
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
        path, value = zc.tls_block()
        assert path == "transport/link/tls"
