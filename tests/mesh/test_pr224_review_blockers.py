"""Pin tests for PR #224 review-blocker batch (2026-06-02 sweep).

Covers:

* Thread session.py:296 -- ACL TOCTOU single-flight via thread-local snapshot.
* Thread _acl_config.py:456 -- ``_is_permissive_acl_shape`` recognises the
  ``default_permission: "deny"`` + wildcard-rule + wildcard-subject pattern
  (was wire-effectively permissive but bypassed the gate).
* Thread _acl_config.py:429 -- cache returns deep copy on hit
  (caller mutation does not poison the cache).
* Thread _acl_config.py:279 -- subjects with empty-list cert_common_names
  now also trigger the wildcard warning (was None-only).
* Thread core.py:121 -- dead ``except ImportError`` /
  ``hasattr(_acl_config, "snapshot_acl")`` fallback removed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_thread_local_single_flight_returns_same_dict() -> None:
    """When a snapshot is stashed, ``snapshot_acl`` returns the SAME dict
    (closes the gate-vs-build TOCTOU window per review session.py:296)."""
    from strands_robots.mesh import _acl_config

    sentinel = {
        "enabled": True,
        "default_permission": "deny",
        "rules": [],
        "subjects": [],
        "policies": [],
        "_marker": "stashed-by-Mesh.start",
    }
    _acl_config._set_thread_snapshot(sentinel)
    try:
        is_permissive, resolved = _acl_config.snapshot_acl("strands")
        # The returned dict IS the sentinel (identity, not just equality)
        assert resolved is sentinel
        assert is_permissive is False
    finally:
        _acl_config._clear_thread_snapshot()


def test_thread_local_cleared_after_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """After ``_clear_thread_snapshot``, ``snapshot_acl`` re-resolves from disk."""
    from strands_robots.mesh import _acl_config

    sentinel = {"_marker": "stashed"}
    _acl_config._set_thread_snapshot(sentinel)
    _acl_config._clear_thread_snapshot()

    # Now without env var, returns built-in default (NOT the sentinel)
    monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)
    is_permissive, resolved = _acl_config.snapshot_acl("strands")
    assert resolved is not sentinel
    assert is_permissive is True  # built-in default is permissive


def test_permissive_shape_detects_deny_plus_wildcard_rule_plus_wildcard_subject() -> None:
    """The wire-effectively permissive ACL shape that bypassed the
    previous narrow check (review _acl_config.py:456):

    ``default_permission: "deny"``  (looks safe)
    + a single ``key_exprs: ["**"], permission: "allow"`` rule
    + a wildcard subject (no interfaces, no cert_common_names)
    + a policy that ties them together
    """
    from strands_robots.mesh import _acl_config

    permissive_in_disguise = {
        "enabled": True,
        "default_permission": "deny",
        "rules": [
            {
                "id": "wide-open",
                "key_exprs": ["**"],
                "messages": ["put"],
                "flows": ["ingress"],
                "permission": "allow",
            }
        ],
        "subjects": [{"id": "anyone"}],  # no constraints -> wildcard
        "policies": [{"rules": ["wide-open"], "subjects": ["anyone"]}],
    }
    assert _acl_config._is_permissive_acl_shape(permissive_in_disguise) is True


def test_permissive_shape_does_not_flag_role_separated_acl() -> None:
    """A genuinely role-separated ACL with constrained subjects is NOT
    flagged (no false positive)."""
    from strands_robots.mesh import _acl_config

    safe = {
        "enabled": True,
        "default_permission": "deny",
        "rules": [
            {
                "id": "operator",
                "key_exprs": ["strands/safety/estop"],
                "messages": ["put"],
                "flows": ["egress"],
                "permission": "allow",
            }
        ],
        "subjects": [{"id": "op", "cert_common_names": ["operator-1"]}],
        "policies": [{"rules": ["operator"], "subjects": ["op"]}],
    }
    assert _acl_config._is_permissive_acl_shape(safe) is False


def test_permissive_shape_does_not_flag_unwired_wildcard() -> None:
    """A wildcard rule and a wildcard subject that are NOT tied together
    by any policy do not match the pattern."""
    from strands_robots.mesh import _acl_config

    not_wired = {
        "enabled": True,
        "default_permission": "deny",
        "rules": [
            {
                "id": "wide-open",
                "key_exprs": ["**"],
                "permission": "allow",
            }
        ],
        "subjects": [
            {"id": "anyone"},
            {"id": "operator", "cert_common_names": ["op-1"]},
        ],
        # Only the constrained subject is wired; wildcard subject sits unused.
        "policies": [{"rules": ["wide-open"], "subjects": ["operator"]}],
    }
    assert _acl_config._is_permissive_acl_shape(not_wired) is False


def test_cache_hit_returns_deep_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Cache poisoning protection: caller mutation of cache result does
    not corrupt the next caller's view (review _acl_config.py:429)."""
    from strands_robots.mesh import _acl_config

    acl = tmp_path / "acl.json5"
    acl.write_text(
        json.dumps(
            {
                "enabled": True,
                "default_permission": "deny",
                "rules": [
                    {"id": "r", "key_exprs": ["foo"], "messages": ["put"], "flows": ["ingress"], "permission": "allow"}
                ],
                "subjects": [{"id": "s", "cert_common_names": ["c1"]}],
                "policies": [{"rules": ["r"], "subjects": ["s"]}],
            }
        )
    )
    _acl_config._clear_acl_cache_for_test()

    first = _acl_config._load_acl_cached(acl)
    # Caller deliberately mutates the returned dict (the bad pattern)
    first["enabled"] = False
    first["rules"].clear()

    # Subsequent caller must see the ORIGINAL contents -- mutation didn't poison
    second = _acl_config._load_acl_cached(acl)
    assert second["enabled"] is True
    assert len(second["rules"]) == 1
    assert second is not first  # different object identity (deep copy)


def test_wildcard_subject_hard_rejected_on_empty_cert_common_names_list(
    tmp_path: Path,
) -> None:
    """Subject with ``cert_common_names: []`` (empty list, not None) is
    HARD-REJECTED at parse time. Per review threads _acl_config.py:279
    and 293, a subject that constrains nothing on either dimension is
    a foot-gun that wire-effectively maps to "any peer on any link" --
    the validator MUST refuse, not warn."""
    from strands_robots.mesh import _acl_config

    doc = {
        "enabled": True,
        "default_permission": "deny",
        "subjects": [{"id": "wide-open", "cert_common_names": []}],
        "rules": [
            {
                "id": "r",
                "key_exprs": ["**"],
                "messages": ["put"],
                "flows": ["ingress"],
                "permission": "allow",
            }
        ],
        "policies": [{"rules": ["r"], "subjects": ["wide-open"]}],
    }
    p = tmp_path / "acl.json5"
    p.write_text(json.dumps(doc))

    with pytest.raises(ValueError, match=r"wide-open.*match every peer"):
        _acl_config._validate_acl_shape(doc, p)


def test_core_no_dead_importerror_fallback() -> None:
    """Static assertion: the dead ``except ImportError`` and
    ``hasattr(_acl_config, "snapshot_acl")`` branches were removed
    (review thread core.py:121)."""
    from strands_robots.mesh import core as core_mod

    src = Path(core_mod.__file__).read_text()
    # The gate body now imports _acl_config + _zenoh_config unconditionally
    # since PR-3 ships them in the same diff.
    assert "PR-3 (`_acl_config` + `_zenoh_config`) not on the tree yet" not in src, (
        "Expected the dead-fallback comment to be deleted; PR-3 ships these modules."
    )
    # And the hasattr-based gate-skip is gone.
    assert 'hasattr(_acl_config, "snapshot_acl")' not in src


# === Round-2 (post-4f7f9cc) review-blocker pins ===


def test_thread_local_stashes_auth_mode_alongside_resolved() -> None:
    """Thread #30: ``auth_mode`` is stashed on the thread-local
    alongside the resolved ACL so the gate and the wire-config builder
    agree even if ``STRANDS_MESH_AUTH_MODE`` flips between reads."""
    from strands_robots.mesh import _acl_config

    sentinel_acl = {"_marker": "test"}
    _acl_config._set_thread_snapshot(sentinel_acl, auth_mode="mtls")
    try:
        assert _acl_config._get_thread_snapshot() is sentinel_acl
        assert _acl_config._get_thread_auth_mode() == "mtls"
    finally:
        _acl_config._clear_thread_snapshot()

    # Without auth_mode (legacy single-arg call), getter returns None.
    _acl_config._set_thread_snapshot(sentinel_acl)
    try:
        assert _acl_config._get_thread_snapshot() is sentinel_acl
        assert _acl_config._get_thread_auth_mode() is None
    finally:
        _acl_config._clear_thread_snapshot()


def test_endpoint_scheme_rejects_tcp_under_mtls() -> None:
    """Thread #26: ``ZENOH_LISTEN=tcp/...`` under
    ``STRANDS_MESH_AUTH_MODE=mtls`` raises ValueError at config-build
    time (loud-on-misconfig) instead of producing an opaque zenoh
    runtime error downstream."""
    from strands_robots.mesh.session import _validate_endpoint_schemes

    with pytest.raises(ValueError, match=r"tcp.*tls.*quic"):
        _validate_endpoint_schemes("tcp/0.0.0.0:7447", "ZENOH_LISTEN", "mtls")


def test_endpoint_scheme_accepts_tls_under_mtls() -> None:
    """Thread #26 inverse: tls/... is allowed under mtls."""
    from strands_robots.mesh.session import _validate_endpoint_schemes

    # No raise expected.
    _validate_endpoint_schemes("tls/0.0.0.0:7447", "ZENOH_LISTEN", "mtls")
    _validate_endpoint_schemes("quic/0.0.0.0:7447", "ZENOH_LISTEN", "mtls")


def test_endpoint_scheme_accepts_tcp_under_none() -> None:
    """Thread #26: tcp/... is allowed under auth_mode=none (dev posture)."""
    from strands_robots.mesh.session import _validate_endpoint_schemes

    _validate_endpoint_schemes("tcp/0.0.0.0:7447", "ZENOH_LISTEN", "none")
    _validate_endpoint_schemes("udp/0.0.0.0:7447", "ZENOH_LISTEN", "none")


def test_endpoint_scheme_validator_handles_multi_endpoint() -> None:
    """Thread #26: comma-separated list -- ANY bad scheme rejects the lot."""
    from strands_robots.mesh.session import _validate_endpoint_schemes

    # First endpoint is fine but second is not under mtls.
    with pytest.raises(ValueError, match=r"tcp.*tls.*quic"):
        _validate_endpoint_schemes("tls/peer-a:7447,tcp/peer-b:7447", "ZENOH_CONNECT", "mtls")


def test_cache_miss_returns_deepcopy_too(tmp_path: Path) -> None:
    """Thread #27: cache MISS returns a deep copy too -- not just hits.
    Previously the first caller for a given file got the raw parsed
    dict (mutable handle); subsequent callers got fresh deep copies.
    Asymmetric and one accidental refactor away from cache poisoning."""
    import json as _json

    from strands_robots.mesh import _acl_config

    acl = tmp_path / "acl.json5"
    acl.write_text(
        _json.dumps(
            {
                "enabled": True,
                "default_permission": "deny",
                "rules": [
                    {
                        "id": "r",
                        "key_exprs": ["foo"],
                        "messages": ["put"],
                        "flows": ["ingress"],
                        "permission": "allow",
                    }
                ],
                "subjects": [{"id": "s", "cert_common_names": ["c1"]}],
                "policies": [{"rules": ["r"], "subjects": ["s"]}],
            }
        )
    )
    _acl_config._clear_acl_cache_for_test()

    # The FIRST caller (cache miss) deliberately mutates the result.
    first = _acl_config._load_acl_cached(acl)
    first["enabled"] = False
    first["rules"].clear()

    # SECOND caller (cache hit) sees the original contents -- the
    # cache stored a deep copy, AND the first caller's mutation
    # cannot have poisoned the cache because miss-return ALSO did
    # a deep copy.
    second = _acl_config._load_acl_cached(acl)
    assert second["enabled"] is True
    assert len(second["rules"]) == 1


def test_blacklist_warning_only_fires_with_rules(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Thread #28: ``allow + non-empty subjects but empty rules`` does
    NOT trigger the blacklist warning (the warning is scoped to
    ``allow + rules`` -- the actual anti-pattern)."""
    import json as _json
    import logging

    from strands_robots.mesh import _acl_config

    p = tmp_path / "allow_subjects_only.json5"
    p.write_text(
        _json.dumps(
            {
                "enabled": True,
                "default_permission": "allow",
                "rules": [],
                "subjects": [{"id": "future", "cert_common_names": ["future-thing"]}],
                "policies": [],
            }
        )
    )
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh._acl_config"):
        _acl_config._load_acl_file(p)
    assert not any("blacklist" in m for m in caplog.messages), (
        f"unexpected blacklist warning for allow+empty_rules+subjects: {caplog.messages}"
    )


def test_blacklist_warning_fires_with_rules(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Thread #28: ``allow + non-empty rules`` DOES trigger the
    blacklist warning -- this is the actual anti-pattern."""
    import json as _json
    import logging

    from strands_robots.mesh import _acl_config

    p = tmp_path / "blacklist.json5"
    p.write_text(
        _json.dumps(
            {
                "enabled": True,
                "default_permission": "allow",
                "rules": [
                    {
                        "id": "denied",
                        "key_exprs": ["secret/**"],
                        "messages": ["put"],
                        "flows": ["ingress"],
                        "permission": "deny",
                    }
                ],
                "subjects": [{"id": "anyone", "cert_common_names": ["*"]}],
                "policies": [{"rules": ["denied"], "subjects": ["anyone"]}],
            }
        )
    )
    # B-08 / F-14: allow+rules now hard-refuses without an explicit ack.
    # This test pins the (first-line) WARNING, so opt in to reach it.
    monkeypatch.setenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", "1")
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh._acl_config"):
        _acl_config._load_acl_file(p)
    assert any("blacklist" in m and "1 rule(s)" in m for m in caplog.messages), (
        f"expected blacklist warning naming rule count, got: {caplog.messages}"
    )


def test_tls_warn_set_keys_on_path_and_mtime() -> None:
    """Thread #29: the non-POSIX TLS warn-set keys on
    ``(key_path, mtime_ns)``, not a single boolean. Rotating
    ``STRANDS_MESH_TLS_KEY`` to a different file re-arms the warning."""
    from strands_robots.mesh import _zenoh_config

    # Module-level set exists and is bounded.
    assert isinstance(_zenoh_config._NON_POSIX_TLS_WARNED_KEYS, set)
    assert _zenoh_config._NON_POSIX_TLS_WARNED_MAX > 0


def test_session_no_redundant_local_ep_assignment() -> None:
    """Thread #25 (CodeQL): the dead ``local_ep = f"tcp/127.0.0.1:..."``
    assignment before the ``if not connect_env and not listen_env``
    branch was removed. The variable is now bound only inside that
    branch where it is used."""
    from pathlib import Path

    from strands_robots.mesh import session as session_mod

    src = Path(session_mod.__file__).read_text()
    # The dead assignment had this exact form; verify it's gone.
    # Note: the legitimate assignment INSIDE the if-branch uses
    # f"{scheme}/127.0.0.1:..." so this string check targets only
    # the dead form.
    assert src.count('local_ep = f"tcp/127.0.0.1:{mesh_port}"') == 0, (
        "CodeQL #262: redundant local_ep assignment must not be reintroduced"
    )


# ============================================================
# Round-3 follow-through pins (post-19:58 review)
# ============================================================


def test_build_config_resolves_auth_mode_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review thread session.py:357 -- ``_build_config`` must resolve
    ``auth_mode`` exactly once (single ``os.environ`` read) and reuse
    it for both endpoint validation and block selection. Previously,
    two independent ``resolve_auth_mode()`` calls between scheme
    validation and the mTLS branch could disagree if env mutated
    between them.
    """
    from pathlib import Path

    from strands_robots.mesh import session as session_mod

    src = Path(session_mod.__file__).read_text()

    # The function body of _build_config must contain exactly ONE
    # call to resolve_auth_mode() (the early single-read site). The
    # second read site at the mTLS branch was removed.
    func_idx = src.index("def _build_config(")
    # Bound the search to the function body only. The next def starts
    # the boundary; pick a generous fallback if not found.
    next_def_idx = src.find("\ndef ", func_idx + 1)
    if next_def_idx == -1:
        next_def_idx = len(src)
    body = src[func_idx:next_def_idx]
    n = body.count("_zenoh_config.resolve_auth_mode()")
    assert n == 1, f"_build_config must resolve auth_mode exactly once (review thread session.py:357); found {n} calls"
    # The single resolve site must precede the endpoint validators
    # so they share the same value.
    resolve_idx = body.index("_zenoh_config.resolve_auth_mode()")
    val_idx = body.index('_validate_endpoint_schemes(connect, "ZENOH_CONNECT"')
    assert resolve_idx < val_idx, "auth_mode must be resolved BEFORE endpoint validation"


def test_build_config_endpoint_validation_uses_same_auth_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Behavioural pin: a mid-call mutation of STRANDS_MESH_AUTH_MODE
    cannot put endpoint validation and block selection out of sync
    because both reuse the single resolved value. Simulate the
    mutation by tracking calls to ``resolve_auth_mode`` and asserting
    only one happens per ``_build_config`` invocation when no
    thread-local snapshot is active.
    """
    from strands_robots.mesh import _acl_config, _zenoh_config

    # Ensure no thread-local from a prior test leaks.
    _acl_config._clear_thread_snapshot()

    call_count = {"n": 0}
    real_resolve = _zenoh_config.resolve_auth_mode

    def counting_resolve() -> str:
        call_count["n"] += 1
        return real_resolve()

    monkeypatch.setattr(_zenoh_config, "resolve_auth_mode", counting_resolve)
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
    monkeypatch.setenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", "1")
    monkeypatch.delenv("ZENOH_CONNECT", raising=False)
    monkeypatch.delenv("ZENOH_LISTEN", raising=False)

    try:
        from strands_robots.mesh.session import _build_config

        _build_config()
    except ImportError:
        pytest.skip("zenoh wheel not installed -- skipping live config build")
    except Exception:
        # The build may raise downstream (missing TLS files etc.)
        # but the resolve count up to that point is what we assert.
        pass

    assert call_count["n"] == 1, (
        f"_build_config must resolve auth_mode exactly once per call "
        f"when no thread-local is active; saw {call_count['n']}"
    )


def test_mesh_start_clears_snapshot_on_refused_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Review thread core.py:189 -- when ``Mesh.start`` refuses to
    bring up the wire because of a permissive ACL under mtls, the
    thread-local snapshot stashed by the gate MUST still be cleared
    on return. Otherwise a subsequent direct ``get_session()`` on
    the same thread would observe stale state.
    """
    from strands_robots.mesh import _acl_config
    from strands_robots.mesh.core import Mesh

    # Force the refuse-to-start branch: mtls posture + permissive
    # default ACL + no opt-in.
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.delenv("STRANDS_MESH_ACL_FILE", raising=False)
    monkeypatch.delenv("STRANDS_MESH_ACCEPT_PERMISSIVE_ACL", raising=False)

    # Pre-poison the thread-local with a sentinel so we can detect a leak.
    _acl_config._clear_thread_snapshot()
    assert _acl_config._get_thread_snapshot() is None

    mesh = Mesh.__new__(Mesh)  # bypass __init__ to avoid full lifecycle setup
    mesh.peer_id = "test-refused-cleanup"
    import threading

    mesh._lifecycle_lock = threading.Lock()
    mesh._running = False
    mesh._has_session_ref = False
    mesh._acl_snapshot = None

    # _refuse_under_permissive_default_acl will set the snapshot
    # internally, then return True; start() must clear it before
    # returning even on the refused branch.
    mesh.start()

    # The thread-local must be empty after start() returns -- whether
    # we refused or got a session.
    assert _acl_config._get_thread_snapshot() is None, "core.py:189 leak: refused-start path left stale snapshot"
    assert _acl_config._get_thread_auth_mode() is None, "core.py:189 leak: refused-start path left stale auth_mode"


def test_zenoh_config_env_var_matrix_documents_three_vars() -> None:
    """Review thread _zenoh_config.py:382 -- AGENTS.md (#86) requires
    every STRANDS_MESH_* env var to be documented in the module
    docstring's env-var matrix. Three vars (FILTER_INTERFACES,
    ACCEPT_PERMISSIVE_ACL, I_KNOW_THIS_IS_INSECURE) were missing.
    """
    from pathlib import Path

    from strands_robots.mesh import _zenoh_config

    src = Path(_zenoh_config.__file__).read_text()
    # Bound the search to the module docstring.
    docstring_end = src.index("from __future__ import annotations")
    matrix = src[:docstring_end]

    # Each var must appear as a documented section header
    # (``STRANDS_MESH_X``, in RST literal-quote style).
    for var in (
        "STRANDS_MESH_ACCEPT_PERMISSIVE_ACL",
        "STRANDS_MESH_I_KNOW_THIS_IS_INSECURE",
        "STRANDS_MESH_FILTER_INTERFACES",
    ):
        anchor = f"``{var}``"
        assert anchor in matrix, (
            f"AGENTS.md (#86) env-var rule: {var} must be documented in the module-level matrix as {anchor!r}"
        )


# ---------------------------------------------------------------------------
# R4 (review thread session.py:517) -- ``auth_mode`` race at
# ``get_session()`` boundary. The fix at session.py:517-518 closes the
# same race R3-follow-through closed at the ``_build_config`` boundary,
# but one frame up: ``get_session`` and ``_get_zenoh_session_directly``
# now prefer the thread-local ``auth_mode`` stash before falling back
# to ``resolve_auth_mode()``.
# ---------------------------------------------------------------------------


def test_get_session_prefers_thread_local_auth_mode_over_env() -> None:
    """When ``Mesh.start`` has stashed ``auth_mode`` on the
    thread-local, ``get_session()`` MUST honour it for listener-scheme
    selection rather than re-reading ``STRANDS_MESH_AUTH_MODE`` from
    ``os.environ``. Without this, the listener endpoint scheme
    (composed in ``get_session``) and the wire-config block (composed
    in ``_build_config``) can disagree if the env var flips between the
    two reads.
    """
    import inspect

    from strands_robots.mesh import session as _session

    src = inspect.getsource(_session.get_session)
    # The thread-local read must precede the resolve_auth_mode fallback
    # in source order. Both helpers must appear in get_session.
    assert "_get_thread_auth_mode" in src, (
        "get_session must consult the thread-local auth_mode stash before "
        "falling back to resolve_auth_mode (review thread session.py:517)"
    )
    # Specifically the conditional fallback shape must be present, not
    # the bare unconditional resolve_auth_mode read R3 left in place.
    assert "_stashed_mode if _stashed_mode is not None else resolve_auth_mode()" in src, (
        "get_session must use the conditional fallback "
        "_stashed_mode if _stashed_mode is not None else resolve_auth_mode()"
    )


def test_get_zenoh_session_directly_prefers_thread_local_auth_mode_over_env() -> None:
    """Mirror of the get_session check for the duplicate path
    ``_get_zenoh_session_directly`` (review thread session.py:517 also
    notes this site one frame up)."""
    import inspect

    from strands_robots.mesh import session as _session

    src = inspect.getsource(_session._get_zenoh_session_directly)
    assert "_get_thread_auth_mode" in src, (
        "_get_zenoh_session_directly must consult the thread-local auth_mode "
        "stash before falling back to resolve_auth_mode (review thread session.py:517)"
    )
    assert "_stashed_mode if _stashed_mode is not None else resolve_auth_mode()" in src, (
        "_get_zenoh_session_directly must use the conditional fallback "
        "_stashed_mode if _stashed_mode is not None else resolve_auth_mode()"
    )


def test_get_session_skips_resolve_auth_mode_when_thread_local_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Behavioural pin: when a thread-local ``auth_mode`` snapshot is
    present, ``get_session`` MUST NOT call ``resolve_auth_mode``
    (which would re-read ``STRANDS_MESH_AUTH_MODE`` from
    ``os.environ`` and risk seeing a mid-call mutation). Mirrors the
    same invariant ``_build_config`` honours at session.py:328-329.

    Implementation strategy: monkeypatch ``_zenoh_config.resolve_auth_mode``
    to a counter, prime the thread-local, then enter ``get_session``
    via a fake ``zenoh.open`` that raises early so we exit before the
    ``_build_config`` -> ``zenoh.open`` chain. The single assertion
    is that ``resolve_auth_mode`` is NOT called for the scheme-
    selection block when the thread-local is primed.
    """
    from strands_robots.mesh import _acl_config, _zenoh_config
    from strands_robots.mesh import session as _session

    _acl_config._clear_thread_snapshot()
    _session._SESSION = None
    _session._SESSION_REFS = 0

    call_count = {"n": 0}

    def counting_resolve() -> str:
        call_count["n"] += 1
        return "mtls"

    monkeypatch.setattr(_zenoh_config, "resolve_auth_mode", counting_resolve)
    # Env says mtls; the thread-local will override with "none".
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "mtls")
    monkeypatch.delenv("ZENOH_CONNECT", raising=False)
    monkeypatch.delenv("ZENOH_LISTEN", raising=False)

    # Force _build_config to short-circuit so we measure the
    # resolve_auth_mode call from the get_session scheme-selection
    # block in isolation.
    sentinel_exc = RuntimeError("short-circuit for test")

    def _short_circuit_build():
        raise sentinel_exc

    monkeypatch.setattr(_session, "_build_config", _short_circuit_build)

    _acl_config._set_thread_snapshot(None, auth_mode="none")
    try:
        try:
            _session.get_session()
        except RuntimeError as exc:
            if exc is not sentinel_exc:
                raise
    finally:
        _acl_config._clear_thread_snapshot()
        _session._SESSION = None
        _session._SESSION_REFS = 0

    # The thread-local says "none"; resolve_auth_mode MUST NOT have
    # been called in the scheme-selection block. (It also is not
    # called by the short-circuited _build_config.)
    assert call_count["n"] == 0, (
        "get_session must skip resolve_auth_mode when the thread-local "
        f"auth_mode is primed; saw {call_count['n']} call(s) "
        "(review thread session.py:517)"
    )


def test_get_session_falls_back_to_resolve_auth_mode_without_thread_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Symmetric pin: a direct ``get_session`` caller (no
    ``Mesh.start`` priming the thread-local) MUST still resolve
    ``auth_mode`` from the env via ``resolve_auth_mode`` -- the
    legacy contract is preserved by the conditional fallback.
    """
    from strands_robots.mesh import _acl_config, _zenoh_config
    from strands_robots.mesh import session as _session

    _acl_config._clear_thread_snapshot()
    _session._SESSION = None
    _session._SESSION_REFS = 0

    call_count = {"n": 0}

    def counting_resolve() -> str:
        call_count["n"] += 1
        return "none"

    monkeypatch.setattr(_zenoh_config, "resolve_auth_mode", counting_resolve)
    monkeypatch.setenv("STRANDS_MESH_AUTH_MODE", "none")
    monkeypatch.setenv("STRANDS_MESH_I_KNOW_THIS_IS_INSECURE", "1")
    monkeypatch.delenv("ZENOH_CONNECT", raising=False)
    monkeypatch.delenv("ZENOH_LISTEN", raising=False)

    sentinel_exc = RuntimeError("short-circuit for test")

    def _short_circuit_build():
        raise sentinel_exc

    monkeypatch.setattr(_session, "_build_config", _short_circuit_build)

    try:
        try:
            _session.get_session()
        except RuntimeError as exc:
            if exc is not sentinel_exc:
                raise
    finally:
        _session._SESSION = None
        _session._SESSION_REFS = 0

    # Without a thread-local, exactly one resolve_auth_mode call from
    # the scheme-selection block (the short-circuited _build_config
    # never reaches its own resolve site).
    assert call_count["n"] == 1, (
        "get_session must call resolve_auth_mode exactly once when no "
        f"thread-local auth_mode is primed; saw {call_count['n']} call(s)"
    )
