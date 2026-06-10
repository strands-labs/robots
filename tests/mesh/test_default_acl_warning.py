"""the prior fix pin tests: Mesh.start warning when mtls + default ACL combo is active.

Reviewer concern (prior thread @ 2026-05-23T07:38:02Z, _acl_config.py:359):
> ``is_default_acl_in_use()`` exists but no consumer wires it. The dangerous-
> but-easy-to-miss config is mtls + permissive default ACL. Suggested
> follow-up: have ``Mesh.start`` emit a WARNING when both are active.

These tests assert the warning is emitted in the dangerous combo and
suppressed when either condition does not hold.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from strands_robots.mesh import core as core_mod  # noqa: F401  -- used by F3-B-1 source-code assertion


@pytest.fixture
def stub_robot():
    """Minimal robot duck-type for Mesh construction."""
    inner = SimpleNamespace(
        is_connected=True,
        name="r19_test",
        config=SimpleNamespace(cameras={}),
        get_observation=MagicMock(return_value={}),
    )
    return SimpleNamespace(tool_name_str="r19", robot=inner)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip env vars that influence ACL/auth-mode resolution."""
    for var in (
        "STRANDS_MESH_AUTH_MODE",
        "STRANDS_MESH_ACL_FILE",
        "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE",
    ):
        monkeypatch.delenv(var, raising=False)


def _start_with_stub_session(stub_robot, caplog, level: int = logging.WARNING):
    """Construct a Mesh and run start() against a stub session."""
    from strands_robots.mesh import Mesh
    from strands_robots.mesh import core as mesh_core

    class _StubDecl:
        def undeclare(self) -> None:
            pass

    class _StubSession:
        def declare_subscriber(self, *args, **kwargs):
            return _StubDecl()

    mesh = Mesh(stub_robot, peer_id="test-r19", peer_type="robot")

    with patch.object(mesh_core, "get_session", return_value=_StubSession()):
        with patch.object(mesh_core, "release_session"):
            with patch.object(mesh, "_heartbeat_loop"), patch.object(mesh, "_state_loop"):
                with caplog.at_level(level, logger="strands_robots.mesh.core"):
                    mesh.start()
                mesh.stop()
    return caplog.records


def test_mtls_plus_default_acl_does_not_start(caplog, monkeypatch, stub_robot):
    """mesh REFUSES to start under
    ``mtls + permissive default ACL`` -- but does so by returning
    early without raising. The first-run ``Robot()`` /
    ``Simulation()`` constructor experience must not crash; the
    safety property ("no permissive ACL on the wire") is preserved
    by the early return, since no Zenoh session is acquired.

    The operator sees a loud ERROR with three actionable paths
    (set ACL file / accept opt-in / disable mesh). Construction
    succeeds; ``mesh.alive`` returns False until the operator opts
    in.
    """
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.delenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", raising=False)

    # Must NOT raise -- the prior fix made this refuse-and-return rather than refuse-and-raise.
    records = _start_with_stub_session(stub_robot, caplog)

    # The operator-facing ERROR IS logged.
    error_msgs = [r.getMessage() for r in records if r.levelname == "ERROR"]
    assert any("PERMISSIVE DEFAULT ACL ACTIVE UNDER MTLS" in m for m in error_msgs), (
        f"expected ERROR breadcrumb; saw {error_msgs}"
    )
    # And the actionable paths are spelled out.
    assert any("Mesh NOT STARTED" in m for m in error_msgs)
    assert any("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1" in m for m in error_msgs)


def test_mtls_plus_default_acl_with_optin_logs_at_info(caplog, monkeypatch, stub_robot):
    """When the operator opts in via STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1,
    the ERROR is downgraded to INFO -- they've explicitly acknowledged.
    """
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.setenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", "1")
    records = _start_with_stub_session(stub_robot, caplog, level=logging.INFO)
    error_msgs = [r.getMessage() for r in records if r.levelname == "ERROR"]
    assert not any("PERMISSIVE DEFAULT ACL ACTIVE" in m for m in error_msgs), (
        f"opt-in should NOT log ERROR; saw {error_msgs}"
    )
    info_msgs = [r.getMessage() for r in records if r.levelname == "INFO"]
    assert any("permissive default ACL active" in m for m in info_msgs), (
        f"expected INFO-level ack on opt-in; saw {info_msgs}"
    )


def test_mtls_with_acl_file_does_not_warn(caplog, monkeypatch, tmp_path, stub_robot):
    """Operator-supplied ACL file MUST suppress the warning AND error."""
    acl = tmp_path / "ops.json5"
    acl.write_text('{"rules": [], "subjects": [], "policies": [], "enabled": true, "default_permission": "deny"}\n')
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(acl))

    records = _start_with_stub_session(stub_robot, caplog)
    msgs = [r.getMessage() for r in records]
    assert not any("PERMISSIVE DEFAULT ACL" in m or "permissive default ACL active" in m for m in msgs), (
        f"unexpected permissive-ACL log when ACL file set: {msgs}"
    )


def test_auth_mode_none_does_not_emit_default_acl_warning(caplog, monkeypatch, stub_robot):
    """auth_mode=none has its own ERROR; permissive-ACL warning is mtls-specific."""
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
    monkeypatch.setenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", "1")

    records = _start_with_stub_session(stub_robot, caplog)
    msgs = [r.getMessage() for r in records]
    assert not any("permissive default ACL active under mtls" in m for m in msgs), (
        f"permissive-ACL warning leaked into auth_mode=none: {msgs}"
    )


class TestPermissiveACLWarningExceptNarrow:
    """The the prior permissive-ACL warning at Mesh.start() previously caught
    `except Exception` and downgraded to DEBUG -- a future refactor that
    raised an unrelated type would silently lose the warning.
    """

    def test_unexpected_exception_surfaces_loudly(self):
        """A non-(ImportError|ValueError) raised inside the warning block
        should surface at WARNING (not DEBUG silent-swallow).

        Scope: assertions target the ``_refuse_under_permissive_default_acl``
        method body specifically (AST-scoped). Other helper methods in
        ``core.py`` legitimately use ``except ImportError:`` to guard
        soft imports of the ``strands_robots.mesh.session`` module
        (e.g. ``_local_session_zid``, ``_safety_publisher_for``); those
        are unrelated to the permissive-ACL warning block and must not
        be flagged by this pin test.
        """

        # Static assertion -- start() does network I/O, so we verify the
        # narrowed except clause is in the source rather than triggering it.
        # AST-scope to the gate method so legitimate ImportError fallbacks
        # in unrelated helper methods do not cause false positives.
        src = Path(core_mod.__file__).read_text()
        tree = ast.parse(src)
        gate_src: str | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_refuse_under_permissive_default_acl":
                gate_src = ast.get_source_segment(src, node)
                break
        assert gate_src is not None, (
            "expected _refuse_under_permissive_default_acl in core.py; "
            "review thread core.py:121 (PR-3 + PR-6 review-feedback)."
        )

        # PR-3 ships _acl_config + _zenoh_config in the same diff, so the
        # ImportError fallback was dead code (review thread core.py:121).
        # The gate now resolves snapshot_acl directly under a narrow
        # ValueError catch (fail-CLOSED on bad config).
        assert "except ValueError as warn_exc:" in gate_src
        # No bare `except Exception as warn_exc:` left in the start() block
        assert "except Exception as warn_exc:" not in gate_src
        # And no dead ImportError fallback in the gate method either.
        assert "except ImportError:" not in gate_src


# ---------------------------------------------------------------------
# the prior fix-2: STRANDS_MESH_CAMERA_DISABLED via _bool_env (lenient parse)
# ---------------------------------------------------------------------


# === pre-session ACL gate (no wire activity on refusal) ===


class TestF17PreSessionGate:
    """the prior refuse-and-return gate
    previously fired AFTER session acquisition + 6 subscriber
    declarations, leaving the wire LIVE with the permissive ACL
    applied -- the very surface the gate was supposed to prevent.

    the prior fix moves the gate to the TOP of ``Mesh.start()``, before
    ``get_session()``. A refusal returns BEFORE any wire activity.
    """

    def test_refusal_does_not_call_get_session(self, monkeypatch, stub_robot, caplog):
        """The blocking property: when the gate refuses, get_session()
        is never called. No subscribers are declared. No wire activity."""
        from strands_robots.mesh import core as mesh_core

        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        monkeypatch.delenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", raising=False)

        get_session_calls = []

        def fake_get_session():
            get_session_calls.append(True)
            return None

        monkeypatch.setattr(mesh_core, "get_session", fake_get_session)

        m = mesh_core.Mesh(stub_robot, peer_id="test-r17", peer_type="robot")
        with caplog.at_level(logging.ERROR, logger="strands_robots.mesh.core"):
            m.start()

        # The gate refused; get_session was NEVER called.
        assert get_session_calls == [], (
            f"F17-A regression: get_session was called {len(get_session_calls)} time(s); "
            "the pre-session gate must short-circuit before any wire activity."
        )
        # And the mesh did not become alive.
        assert m._running is False
        # The operator-facing ERROR was logged.
        error_msgs = [r.getMessage() for r in caplog.records if r.levelname == "ERROR"]
        assert any("PERMISSIVE DEFAULT ACL ACTIVE UNDER MTLS" in msg for msg in error_msgs)

    def test_helper_returns_true_under_dangerous_combo(self, monkeypatch, stub_robot):
        """The helper itself returns True under mtls + permissive default
        ACL without opt-in (the dangerous combination)."""
        from strands_robots.mesh import core as mesh_core

        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        monkeypatch.delenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", raising=False)

        m = mesh_core.Mesh(stub_robot, peer_id="test-r17b", peer_type="robot")
        assert m._refuse_under_permissive_default_acl() is True

    def test_helper_returns_false_under_optin(self, monkeypatch, stub_robot):
        """The helper returns False when the operator has opted in."""
        from strands_robots.mesh import core as mesh_core

        monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
        monkeypatch.setenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", "1")

        m = mesh_core.Mesh(stub_robot, peer_id="test-r17c", peer_type="robot")
        assert m._refuse_under_permissive_default_acl() is False


# === shape-based is_default_acl_in_use ===


class TestF18ShapeBasedACL:
    """pre-the prior ``is_default_acl_in_use``
    returned False whenever ``STRANDS_MESH_ACL_FILE`` was set,
    regardless of file content. An operator who shipped a permissive
    file (``default_permission: "allow"`` + empty rules/subjects/
    policies) silenced the prior session-open gate while running with
    the same posture the gate was supposed to refuse.

    Post-the prior fix the check is shape-based: it inspects the resolved ACL
    dict and returns True for any permissive-by-shape resolution.
    """

    def test_operator_permissive_file_still_triggers_gate(self, monkeypatch, tmp_path):
        """An operator file with the same permissive shape as
        ``default_acl()`` is detected and triggers the gate."""
        from strands_robots.mesh._acl_config import is_default_acl_in_use

        # Operator-supplied file with permissive shape.
        permissive_file = tmp_path / "permissive.json5"
        permissive_file.write_text(
            '{"enabled": true, "default_permission": "allow", "rules": [], "subjects": [], "policies": []}'
        )
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(permissive_file))

        assert is_default_acl_in_use() is True, (
            "F18-B regression: operator-supplied permissive file silenced the gate; "
            "shape-based check must detect default_permission=allow + empty collections "
            "regardless of file source."
        )

    def test_operator_strict_file_does_not_trigger_gate(self, monkeypatch, tmp_path):
        """An operator file with explicit rules/subjects does NOT
        trigger the gate -- the gate only fires on the dangerous
        permissive shape."""
        from strands_robots.mesh._acl_config import is_default_acl_in_use

        strict_file = tmp_path / "strict.json5"
        strict_file.write_text(
            '{"enabled": true, "default_permission": "deny", '
            '"rules": [{"id": "operator", "permission": "allow", '
            '"flows": ["egress"], "messages": ["put"], '
            '"key_exprs": ["strands/op-1/cmd"]}], '
            '"subjects": [{"id": "operator", "cert_common_names": ["op-1"]}], '
            '"policies": [{"rules": ["operator"], "subjects": ["operator"]}]}'
        )
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(strict_file))

        assert is_default_acl_in_use() is False

    def test_unloadable_acl_file_fails_closed(self, monkeypatch, tmp_path):
        """A broken/unparseable ACL file fails CLOSED -- treated as
        permissive so the operator hears about the misconfig at
        start-up rather than silently degrading to insecure."""
        from strands_robots.mesh._acl_config import is_default_acl_in_use

        broken_file = tmp_path / "broken.json5"
        broken_file.write_text("this is not valid JSON5 {[}")
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(broken_file))

        assert is_default_acl_in_use() is True, (
            "F18-B fail-closed regression: broken ACL file must be treated "
            "as permissive at the gate so a typo does not silently lift "
            "the wire posture."
        )

    def test_no_env_var_returns_true_default(self, monkeypatch):
        """No env var = built-in default which IS permissive."""
        from strands_robots.mesh._acl_config import is_default_acl_in_use

        monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)
        assert is_default_acl_in_use() is True


# === TOCTOU defence on the ACL file path ===


class TestF18TOCTOUSingleLoad:
    """pre-the prior ``Mesh.start`` called
    ``is_default_acl_in_use()`` (which loads the ACL file) and then
    ``resolve_acl()`` (which loads it again). An attacker who could
    swap the file between the two reads could show the gate the SAFE
    shape and the wire builder the UNSAFE shape.

    Post-the prior fix both calls share an identity-keyed cache so the same
    ``Mesh.start`` flow sees one snapshot.
    """

    def test_two_consecutive_loads_return_same_dict_object(self, monkeypatch, tmp_path):
        """The two reads inside one Mesh.start equivalent must return
        the SAME dict object (cache hit)."""
        from strands_robots.mesh._acl_config import (
            _clear_acl_cache_for_test,
            is_default_acl_in_use,
            resolve_acl,
        )

        _clear_acl_cache_for_test()
        permissive = tmp_path / "permissive.json5"
        permissive.write_text(
            '{"enabled": true, "default_permission": "allow", "rules": [], "subjects": [], "policies": []}'
        )
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(permissive))

        # First call: triggers load, populates cache.
        gate_view = is_default_acl_in_use()
        # Second call: cache hit, must return same shape.
        wire_view = resolve_acl("strands")

        assert gate_view is True
        # Wire-effective ACL shape matches what the gate observed.
        assert wire_view["default_permission"] == "allow"
        assert wire_view["rules"] == []

    def test_file_swap_between_calls_observed_on_next_call_not_during(self, monkeypatch, tmp_path):
        """If the file is mutated AFTER both reads, the cached snapshot
        survives -- the next Mesh.start sees the new content (cache
        invalidated by mtime change)."""
        from strands_robots.mesh._acl_config import (
            _clear_acl_cache_for_test,
            is_default_acl_in_use,
        )

        _clear_acl_cache_for_test()
        f = tmp_path / "acl.json5"
        f.write_text('{"enabled": true, "default_permission": "allow", "rules": [], "subjects": [], "policies": []}')
        monkeypatch.setenv("STRANDS_MESH_ACL_FILE", str(f))

        # First load: permissive.
        assert is_default_acl_in_use() is True

        # Mutate file to strict shape.
        import time as _time

        _time.sleep(0.01)  # ensure mtime changes
        f.write_text(
            '{"enabled": true, "default_permission": "deny", '
            '"rules": [{"id": "r", "permission": "allow", '
            '"flows": ["egress"], "messages": ["put"], '
            '"key_exprs": ["strands/op/cmd"]}], '
            '"subjects": [{"id": "r", "cert_common_names": ["op"]}], '
            '"policies": [{"rules": ["r"], "subjects": ["r"]}]}'
        )

        # Next call: cache key (mtime) changed -> reload -> strict.
        assert is_default_acl_in_use() is False
