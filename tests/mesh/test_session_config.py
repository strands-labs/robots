"""End-to-end tests for :func:`strands_robots.mesh.session._build_config`.

These tests exercise the full chain from env var -> ``_build_config`` ->
``zenoh.Config`` round-trip. They require ``eclipse-zenoh`` because the
emitted JSON5 must validate against Zenoh's Rust ``Config`` parser. If
the wheel is unavailable the tests skip cleanly.
"""

from __future__ import annotations

import json

import pytest

zenoh = pytest.importorskip("zenoh")


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
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
        "STRANDS_MESH_ACL_FILE",
        "ZENOH_CONNECT",
        "ZENOH_LISTEN",
    ]:
        monkeypatch.delenv(key, raising=False)


def _build():
    """Build a config in auth_mode=none (no TLS files needed)."""
    import os

    os.environ["STRANDS_MESH_AUTH_MODE"] = "none"
    os.environ["STRANDS_MESH_I_KNOW_THIS_IS_INSECURE"] = "1"
    from strands_robots.mesh.session import _build_config

    return _build_config()


def _build_mtls(tmp_path, monkeypatch):
    """Build a config in auth_mode=mtls with synthetic cert files."""
    ca = tmp_path / "ca.crt"
    cert = tmp_path / "peer.crt"
    key = tmp_path / "peer.key"
    for f in (ca, cert, key):
        f.write_text("dummy\n")
    # _resolve_tls_paths enforces mode 0o600 on the private key.
    key.chmod(0o600)
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
    monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
    monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
    from strands_robots.mesh.session import _build_config

    return _build_config()


# --- Default (no env, auth_mode=none) ----------------------------------


class TestDefaultBuild:
    def test_namespace_default_applied(self):
        cfg = _build()
        assert json.loads(cfg.get_json("namespace")) == "strands"

    def test_multicast_disabled_by_default(self):
        cfg = _build()
        assert cfg.get_json("scouting/multicast/enabled") == "false"

    def test_gossip_enabled_by_default(self):
        cfg = _build()
        assert cfg.get_json("scouting/gossip/enabled") == "true"

    def test_max_sessions_default_256(self):
        cfg = _build()
        assert cfg.get_json("transport/unicast/max_sessions") == "256"

    def test_downsampling_present(self):
        cfg = _build()
        ds = json.loads(cfg.get_json("downsampling"))
        assert any(rule["id"] == "strands_cmd_rate_cap" for rule in ds)

    def test_low_pass_filter_present(self):
        cfg = _build()
        lpf = json.loads(cfg.get_json("low_pass_filter"))
        assert any(rule["id"] == "strands_cmd_size_cap" for rule in lpf)
        assert any(rule["id"] == "strands_camera_size_cap" for rule in lpf)

    def test_adminspace_disabled(self):
        cfg = _build()
        admin = json.loads(cfg.get_json("adminspace"))
        assert admin["enabled"] is False


# --- Custom env overrides -----------------------------------------------


class TestEnvOverrides:
    def test_namespace_override_propagates(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_NAMESPACE", "fleet_42")
        cfg = _build()
        assert json.loads(cfg.get_json("namespace")) == "fleet_42"
        # Downsampling rules use ``**/cmd`` style globs (see comment in
        # `_zenoh_config.downsampling_block` for why). Namespace is the
        # routing-isolation primitive; the filter globs do not need to
        # mention it.
        ds = json.loads(cfg.get_json("downsampling"))
        rules = ds[0]["rules"]
        assert any(r["key_expr"].endswith("/cmd") for r in rules)

    def test_multicast_can_be_enabled(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MULTICAST", "true")
        cfg = _build()
        assert cfg.get_json("scouting/multicast/enabled") == "true"

    def test_max_sessions_override(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_MAX_SESSIONS", "1024")
        cfg = _build()
        assert cfg.get_json("transport/unicast/max_sessions") == "1024"


# --- mTLS path ----------------------------------------------------------


class TestMTLSBuild:
    def test_tls_block_present(self, tmp_path, monkeypatch):
        cfg = _build_mtls(tmp_path, monkeypatch)
        tls = json.loads(cfg.get_json("transport/link/tls"))
        assert tls["enable_mtls"] is True
        assert tls["verify_name_on_connect"] is True

    def test_link_protocols_restricted_to_tls(self, tmp_path, monkeypatch):
        cfg = _build_mtls(tmp_path, monkeypatch)
        protos = json.loads(cfg.get_json("transport/link/protocols"))
        assert protos == ["tls"]

    def test_acl_block_present_with_default_allow(self, tmp_path, monkeypatch):
        # Post-default ACL is permissive-by-design (default_permission='allow'
        # + empty rules), matching the documented behaviour without code-vs-doc drift.
        cfg = _build_mtls(tmp_path, monkeypatch)
        acl = json.loads(cfg.get_json("access_control"))
        assert acl["enabled"] is True
        assert acl["default_permission"] == "allow"
        assert acl["subjects"] == []

    def test_mtls_missing_cert_files_raises(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        monkeypatch.setenv("STRANDS_MESH_TLS_CA", "/nonexistent/ca.crt")
        monkeypatch.setenv("STRANDS_MESH_TLS_CERT", "/nonexistent/peer.crt")
        monkeypatch.setenv("STRANDS_MESH_TLS_KEY", "/nonexistent/peer.key")
        from strands_robots.mesh.session import _build_config

        with pytest.raises(FileNotFoundError):
            _build_config()

    def test_mtls_missing_env_vars_raises(self, monkeypatch):
        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        from strands_robots.mesh.session import _build_config

        with pytest.raises(ValueError, match="STRANDS_MESH_TLS"):
            _build_config()


# --- Endpoint env vars --------------------------------------------------


class TestEndpointEnvVars:
    def test_zenoh_connect_propagates(self, monkeypatch):
        monkeypatch.setenv("ZENOH_CONNECT", "tls/router.fleet.local:7447")
        cfg = _build()
        endpoints = json.loads(cfg.get_json("connect/endpoints"))
        assert endpoints == ["tls/router.fleet.local:7447"]

    def test_zenoh_listen_propagates(self, monkeypatch):
        monkeypatch.setenv("ZENOH_LISTEN", "tcp/0.0.0.0:7447")
        cfg = _build()
        endpoints = json.loads(cfg.get_json("listen/endpoints"))
        assert endpoints == ["tcp/0.0.0.0:7447"]


# --- Auth-mode=none warning ---------------------------------------------


def test_auth_mode_none_logs_error_on_open(caplog):
    """B2: auth_mode=none now logs at ERROR level (was WARNING)
    so production log volumes do not bury wire-auth-OFF events. Must
    fire at every session open (not once-and-forget)."""
    import logging
    import os

    os.environ["STRANDS_MESH_AUTH_MODE"] = "none"
    os.environ["STRANDS_MESH_I_KNOW_THIS_IS_INSECURE"] = "1"
    from strands_robots.mesh.session import _build_config

    with caplog.at_level(logging.ERROR):
        _build_config()
    error_messages = [r.message for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("WIRE SECURITY DISABLED" in m for m in error_messages), (
        f"expected ERROR log on auth_mode=none session open, got: {error_messages}"
    )


def test_auth_mode_none_requires_explicit_optin(monkeypatch):
    """B2 pin: auth_mode=none without the second-factor env var
    raises ValueError at config build. This prevents a typo / forgotten
    env / leaked CI fixture from silently disabling wire auth.
    """
    import pytest

    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
    monkeypatch.delenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", raising=False)
    from strands_robots.mesh.session import _build_config

    with pytest.raises(ValueError, match="STRANDS_MESH_I_KNOW_THIS_IS_INSECURE"):
        _build_config()


# --- Default ACL warning in mtls mode -----------------------------------


def test_mtls_default_acl_logs_warning(tmp_path, monkeypatch, caplog):
    """operators who forget STRANDS_MESH_ACL_FILE in mtls
    mode should get a WARNING on every session open."""
    ca = tmp_path / "ca.crt"
    cert = tmp_path / "peer.crt"
    key = tmp_path / "peer.key"
    for f in (ca, cert, key):
        f.write_text("dummy\n")
    # _resolve_tls_paths enforces mode 0o600 on the private key.
    key.chmod(0o600)
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.setenv("STRANDS_MESH_TLS_CA", str(ca))
    monkeypatch.setenv("STRANDS_MESH_TLS_CERT", str(cert))
    monkeypatch.setenv("STRANDS_MESH_TLS_KEY", str(key))
    # Explicitly unset STRANDS_MESH_ACL_FILE to simulate operator forgetting it
    monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)
    from strands_robots.mesh.session import _build_config

    with caplog.at_level("WARNING"):
        _build_config()
    assert any("STRANDS_MESH_ACL_FILE unset" in rec.message for rec in caplog.records)
    assert any("PERMISSIVE built-in" in rec.message for rec in caplog.records)
