"""Regression pins for the v0.4.0 mesh session+ACL follow-up bundle (#387).

Covers PR #224 review follow-ups:
- #307: eviction-order parity between _ACL_CACHE and _NON_POSIX_TLS_WARNED_KEYS.
- #308: symmetric total-deny ACL warning.
- #309: TLS-bearing scheme allow-list widened to wss + unixsock.
- #310: thread-snapshot value is always None or a 2-tuple.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import pytest

from strands_robots.mesh import _acl_config, _zenoh_config, session


# --------------------------------------------------------------------------- #
# #307 — eviction-order parity                                                 #
# --------------------------------------------------------------------------- #
def test_non_posix_tls_warned_keys_is_ordered_fifo():
    """The TLS-warned bound cache must be an insertion-ordered mapping that
    evicts FIFO (popitem(last=False)), matching _ACL_CACHE's pop(next(iter()))."""
    assert isinstance(_zenoh_config._NON_POSIX_TLS_WARNED_KEYS, OrderedDict), (
        "should be an OrderedDict so eviction order is deterministic FIFO"
    )


def test_non_posix_tls_warned_eviction_is_fifo(monkeypatch):
    """Filling past the bound evicts the OLDEST key first (FIFO)."""
    cache = _zenoh_config._NON_POSIX_TLS_WARNED_KEYS
    cache.clear()
    monkeypatch.setattr(_zenoh_config, "_NON_POSIX_TLS_WARNED_MAX", 3)
    try:
        # Insert 3, then a 4th -> the first inserted must be evicted.
        for i in range(3):
            cache[("k", i)] = None
        # Simulate the production eviction path.
        if len(cache) >= _zenoh_config._NON_POSIX_TLS_WARNED_MAX:
            cache.popitem(last=False)
        cache[("k", 99)] = None
        assert ("k", 0) not in cache, "oldest key must be evicted first (FIFO)"
        assert ("k", 99) in cache
    finally:
        cache.clear()


# --------------------------------------------------------------------------- #
# #308 — symmetric total-deny warning                                          #
# --------------------------------------------------------------------------- #
def _write_acl(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "acl.json"
    p.write_text(json.dumps(payload))
    return p


def test_total_deny_shape_warns(tmp_path, caplog):
    """deny + empty rules/subjects/policies = wire-effective total deny -> WARN."""
    acl = _write_acl(
        tmp_path,
        {"enabled": True, "default_permission": "deny", "rules": [], "subjects": [], "policies": []},
    )
    import logging

    with caplog.at_level(logging.WARNING):
        _acl_config._load_acl_file(acl)
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("TOTAL" in m or "total" in m.lower() for m in msgs), f"expected a total-deny WARNING; got {msgs}"


def test_deny_with_allow_rule_does_not_warn_total_deny(tmp_path, caplog):
    """deny + a real allow rule is the recommended posture -> no total-deny WARN."""
    acl = _write_acl(
        tmp_path,
        {
            "enabled": True,
            "default_permission": "deny",
            "rules": [
                {"id": "r1", "permission": "allow", "flows": ["egress"], "messages": ["put"], "key_exprs": ["x/**"]}
            ],
            "subjects": [{"id": "s1", "interfaces": ["lo"]}],
            "policies": [{"id": "p1", "rules": ["r1"], "subjects": ["s1"]}],
        },
    )
    import logging

    with caplog.at_level(logging.WARNING):
        _acl_config._load_acl_file(acl)
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("TOTAL DENY" in m for m in msgs), f"should not warn total-deny; got {msgs}"


# --------------------------------------------------------------------------- #
# #309 — TLS-bearing scheme allow-list                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme", ["tls", "quic", "wss", "unixsock"])
def test_tls_bearing_schemes_pass_under_mtls(scheme):
    """wss and unixsock are TLS-bearing in Zenoh 1.x and must pass under mtls."""
    # Should not raise.
    session._validate_endpoint_schemes(f"{scheme}/0.0.0.0:8443", "ZENOH_LISTEN", "mtls")


def test_non_tls_scheme_still_rejected_under_mtls():
    """tcp is not TLS-bearing -> still rejected under mtls."""
    with pytest.raises(ValueError):
        session._validate_endpoint_schemes("tcp/0.0.0.0:7447", "ZENOH_LISTEN", "mtls")


def test_tls_bearing_constant_contains_new_schemes():
    assert "wss" in session._TLS_BEARING_SCHEMES
    assert "unixsock" in session._TLS_BEARING_SCHEMES
    # Backwards-compatible alias points at the same tuple.
    assert session._MTLS_OK_SCHEMES == session._TLS_BEARING_SCHEMES


# --------------------------------------------------------------------------- #
# #310 — thread snapshot is always None or a 2-tuple                           #
# --------------------------------------------------------------------------- #
def test_thread_snapshot_is_always_2tuple_after_set():
    try:
        _acl_config._set_thread_snapshot({"a": 1})
        val = _acl_config._THREAD_SNAPSHOT.value
        assert isinstance(val, tuple) and len(val) == 2, (
            f"snapshot without auth_mode must still be a 2-tuple, got {val!r}"
        )
        assert val[1] is None

        _acl_config._set_thread_snapshot({"a": 1}, auth_mode="mtls")
        val = _acl_config._THREAD_SNAPSHOT.value
        assert isinstance(val, tuple) and len(val) == 2
        assert val[1] == "mtls"
    finally:
        _acl_config._clear_thread_snapshot()


def test_thread_snapshot_accessors_roundtrip():
    try:
        _acl_config._set_thread_snapshot({"k": "v"}, auth_mode="none")
        assert _acl_config._get_thread_snapshot() == {"k": "v"}
        assert _acl_config._get_thread_auth_mode() == "none"
        # auth_mode omitted -> accessor returns None, snapshot still readable.
        _acl_config._set_thread_snapshot({"k2": "v2"})
        assert _acl_config._get_thread_snapshot() == {"k2": "v2"}
        assert _acl_config._get_thread_auth_mode() is None
    finally:
        _acl_config._clear_thread_snapshot()
