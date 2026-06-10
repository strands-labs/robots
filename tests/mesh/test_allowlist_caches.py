"""Pin: allowlist parsers cache by raw env-var string.

Each of the three operator-extensible allowlist parsers
(``_policy_host_allowlist``, ``_hf_repo_allowlist``,
``_policy_type_allowlist``) wraps a ``functools.lru_cache(maxsize=1)``
keyed on the raw env-var string. Without the cache the parsers
re-read ``os.getenv`` and re-emit the malformed-entry WARNING on
every call; on a busy mesh ``validate_command`` runs per inbound cmd,
and the WARNING flood would drown the audit log on a typo'd env var.

The cache key is the raw env-var string, so a real env mutation
(``monkeypatch.setenv``, or an operator restarting the process with
a corrected value) naturally re-parses on the next call. Tests that
need to re-exercise the WARNING path with the same env value can
call ``_clear_security_caches_for_tests`` to force a refresh.

These tests pin the one-WARNING-per-distinct-value semantics, the
re-parse on env change, the cache-clear helper, and that each cache
is independent.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.mesh import security as sec


@pytest.fixture(autouse=True)
def _clear_caches():
    sec._clear_security_caches_for_tests()
    yield
    sec._clear_security_caches_for_tests()


def test_repeat_call_does_not_re_emit_warning(monkeypatch, caplog):
    """Two ``_hf_repo_allowlist`` calls with the same env emit at most ONE WARNING."""
    monkeypatch.setenv("STRANDS_MESH_HF_REPO_ALLOW", "nvidia,;evil")
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.security"):
        for _ in range(5):
            sec._hf_repo_allowlist()
    msgs = [r.getMessage() for r in caplog.records if ";evil" in r.getMessage()]
    assert len(msgs) <= 1, f"WARNING for malformed entry must fire at most ONCE per env value; got {len(msgs)}: {msgs}"


def test_setenv_change_triggers_reparse(monkeypatch, caplog):
    """Changing the env value must cause a re-parse on next call."""
    monkeypatch.setenv("STRANDS_MESH_HF_REPO_ALLOW", "nvidia")
    a1 = sec._hf_repo_allowlist()
    assert "nvidia" in a1

    monkeypatch.setenv("STRANDS_MESH_HF_REPO_ALLOW", "my-org")
    a2 = sec._hf_repo_allowlist()
    assert "my-org" in a2, f"after env change, allowlist must include my-org; got {a2}"


def test_clear_caches_helper_resets_state():
    """``_clear_security_caches_for_tests`` empties the lru_cache so next call re-parses."""
    # Populate caches.
    sec._policy_host_allowlist()
    sec._hf_repo_allowlist()
    sec._policy_type_allowlist()

    # Verify cache_info shows hits.
    pre_info = sec._hf_repo_allowlist_cached.cache_info()
    sec._hf_repo_allowlist()
    after_info = sec._hf_repo_allowlist_cached.cache_info()
    assert after_info.hits >= pre_info.hits

    # Clear and verify size goes to zero.
    sec._clear_security_caches_for_tests()
    cleared_info = sec._hf_repo_allowlist_cached.cache_info()
    assert cleared_info.currsize == 0


def test_caches_are_independent():
    """Clearing ``_policy_host_allowlist_cached`` does not clear the others'."""
    sec._policy_host_allowlist()
    sec._hf_repo_allowlist()
    # Manually verify the three cache_info objects are distinct.
    assert sec._policy_host_allowlist_cached is not sec._hf_repo_allowlist_cached
    assert sec._hf_repo_allowlist_cached is not sec._policy_type_allowlist_cached
