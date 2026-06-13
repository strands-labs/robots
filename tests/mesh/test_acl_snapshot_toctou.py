"""ACL snapshot single-read TOCTOU fix.

Issue #218: closes the TOCTOU window between is_default_acl_in_use and resolve_acl.

The fix introduces ``snapshot_acl(namespace)`` which atomically returns
``(is_permissive, resolved_dict)`` from a single file read, plus
``acl_block_from(resolved)`` which builds the wire config block from
the snapshot dict. ``session.py`` is updated to use the snapshot pattern
so the refuse-to-start gate and the wire config builder share exactly
ONE ACL file read per ``Mesh.start()``.
"""

from __future__ import annotations


def test_snapshot_acl_returns_permissive_for_default():
    """No env var -> built-in default -> permissive=True, resolved=default_acl."""
    from strands_robots.mesh import _acl_config

    is_permissive, resolved = _acl_config.snapshot_acl("strands")
    assert is_permissive is True
    assert resolved == _acl_config.default_acl("strands")


def test_snapshot_acl_returns_non_permissive_for_role_separated_file(tmp_path, monkeypatch):
    """Operator-supplied role-separated ACL -> permissive=False, resolved=loaded dict."""
    import json

    from strands_robots.mesh import _acl_config

    acl_file = tmp_path / "acl.json5"
    acl_file.write_text(
        json.dumps(
            {
                "enabled": True,
                "default_permission": "deny",
                "rules": [
                    {
                        "id": "operator",
                        "permission": "allow",
                        "flows": ["egress"],
                        "messages": ["put"],
                        "key_exprs": ["strands/safety/estop"],
                    }
                ],
                "subjects": [{"id": "op", "cert_common_names": ["operator-1"]}],
                "policies": [{"rules": ["operator"], "subjects": ["op"]}],
            }
        )
    )

    monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(acl_file))
    is_permissive, resolved = _acl_config.snapshot_acl("strands")
    assert is_permissive is False
    assert resolved.get("default_permission") == "deny"


def test_snapshot_acl_single_file_read(tmp_path, monkeypatch):
    """Issue #218 core invariant: snapshot_acl reads the file ONCE.

    The previous two-call pattern (is_default_acl_in_use + resolve_acl)
    invalidated the identity-tuple cache and re-read the file. Pinning
    that snapshot_acl performs at most one _load_acl_file call.
    """
    import json

    from strands_robots.mesh import _acl_config

    acl_file = tmp_path / "acl.json5"
    acl_file.write_text(json.dumps({"enabled": True, "default_permission": "deny"}))

    monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(acl_file))
    # Clear the cache so we get a fresh read
    _acl_config._load_acl_cached.cache_clear() if hasattr(_acl_config._load_acl_cached, "cache_clear") else None

    call_count = [0]
    real_load_acl_file = _acl_config._load_acl_file

    def counted(path):
        call_count[0] += 1
        return real_load_acl_file(path)

    monkeypatch.setattr(_acl_config, "_load_acl_file", counted)
    # Also clear cache to ensure miss
    if hasattr(_acl_config._load_acl_cached, "cache_clear"):
        _acl_config._load_acl_cached.cache_clear()

    is_permissive, resolved = _acl_config.snapshot_acl("strands")
    # Sanity-check the return shape so the test fails loudly if the
    # signature changes (rather than silently passing on a refactor).
    assert isinstance(is_permissive, bool)
    assert isinstance(resolved, dict)
    # Core invariant: snapshot_acl performs at most ONE _load_acl_file call
    assert call_count[0] <= 1, f"snapshot_acl called _load_acl_file {call_count[0]} times; must be <= 1"


def test_acl_block_from_uses_provided_dict():
    """acl_block_from doesn't re-read the file -- it serialises the given dict."""
    import json

    from strands_robots.mesh import _acl_config

    custom = {"enabled": True, "default_permission": "deny", "marker": "from-snapshot"}
    key, value = _acl_config.acl_block_from(custom)
    assert key == "access_control"
    assert json.loads(value) == custom
    assert json.loads(value)["marker"] == "from-snapshot"


def test_mesh_start_reads_acl_file_once_end_to_end(tmp_path, monkeypatch, caplog):
    """Issue #218 acceptance criterion: ONE read of STRANDS_MESH_ACL_FILE per
    Mesh.start() call, measured end-to-end across the refuse-to-start gate and
    the wire-config builder.

    The prior pin (test_snapshot_acl_single_file_read) asserts a single
    snapshot_acl() call reads once. This pins the issue's exact criterion: the
    full Mesh.start() flow -- gate (_refuse_under_permissive_default_acl ->
    snapshot_acl) plus session._build_config (snapshot_acl -> acl_block_from) --
    performs at most ONE _load_acl_file call, so an attacker rewriting the file
    between gate and build cannot make the wire observe a different snapshot.
    """
    import logging
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from strands_robots.mesh import Mesh, _acl_config
    from strands_robots.mesh import core as mesh_core

    acl = tmp_path / "ops.json5"
    acl.write_text('{"rules": [], "subjects": [], "policies": [], "enabled": true, "default_permission": "deny"}\n')
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(acl))

    _acl_config._clear_acl_cache_for_test()
    _acl_config._clear_thread_snapshot()

    call_count = [0]
    real_load_acl_file = _acl_config._load_acl_file

    def counted(path):
        call_count[0] += 1
        return real_load_acl_file(path)

    monkeypatch.setattr(_acl_config, "_load_acl_file", counted)

    robot = SimpleNamespace(
        tool_name_str="r218",
        robot=SimpleNamespace(
            is_connected=True,
            name="r218_test",
            config=SimpleNamespace(cameras={}),
            get_observation=MagicMock(return_value={}),
        ),
    )

    class _StubDecl:
        def undeclare(self) -> None:
            pass

    class _StubSession:
        def declare_subscriber(self, *args, **kwargs):
            return _StubDecl()

    mesh = Mesh(robot, peer_id="test-218", peer_type="robot")
    with patch.object(mesh_core, "get_session", return_value=_StubSession()):
        with patch.object(mesh_core, "release_session"):
            with patch.object(mesh, "_heartbeat_loop"), patch.object(mesh, "_state_loop"):
                with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.core"):
                    mesh.start()
                mesh.stop()

    assert call_count[0] <= 1, (
        f"Mesh.start() read the ACL file {call_count[0]} times; the TOCTOU defence requires exactly one read per start"
    )
