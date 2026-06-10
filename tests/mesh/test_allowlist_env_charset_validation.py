"""Pin: HF + policy_type env-var allowlists are charset-validated.

All three operator-extensible allowlist parsers
(``STRANDS_MESH_POLICY_HOST_ALLOW``, ``STRANDS_MESH_HF_REPO_ALLOW``,
``STRANDS_MESH_POLICY_TYPE_ALLOW``) share a single charset-validation
helper (``_validate_env_allowlist_entries``). Malformed entries are
dropped with a WARNING that names the env var, the offending entry,
and the expected charset.

The malformed entries are not exploitable downstream (the values are
matched against literal-equality / CIDR / set-membership comparisons,
never subprocess-interpolated), but a typo like
``STRANDS_MESH_HF_REPO_ALLOW="nvidia,;rm -rf /"`` would otherwise
silently produce an allowlist containing ``';rm -rf '`` -- the WARNING
is the operator-visible signal that the env-var content does not match
intent. A uniform fail-loud-on-misconfig posture also means a reviewer
reading ``security.py`` does not have to ask "which env vars are
validated and which are not".

These tests pin the WARNING + drop-malformed behaviour for HF and
policy_type, complementing the existing pin in
``test_policy_host_allowlist.py``.
"""

from __future__ import annotations

import logging

import pytest

from strands_robots.mesh import security as sec


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear allowlist caches around each test so monkeypatch.setenv re-parses."""
    sec._clear_security_caches_for_tests()
    yield
    sec._clear_security_caches_for_tests()


def test_hf_repo_allowlist_drops_malformed_entries(monkeypatch, caplog):
    """An entry containing shell metacharacters is dropped + logged."""
    monkeypatch.setenv("STRANDS_MESH_HF_REPO_ALLOW", "nvidia,;rm -rf /,evil$(whoami)")
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.security"):
        allow = sec._hf_repo_allowlist()
    assert "nvidia" in allow, "valid entry must survive"
    # The legit defaults are always present.
    for builtin in ("huggingface", "lerobot"):
        assert builtin in allow
    # The malformed entries must NOT appear.
    assert ";rm -rf " not in allow
    assert "evil$(whoami)" not in allow
    # WARNING messages must name the env var so an operator can fix it.
    msgs = [r.getMessage() for r in caplog.records if "STRANDS_MESH_HF_REPO_ALLOW" in r.getMessage()]
    assert len(msgs) >= 2, f"expected at least 2 WARNING records, got {msgs}"


def test_policy_type_allowlist_drops_malformed_entries(monkeypatch, caplog):
    """Lowercase-identifier shape is enforced; typos like ``evil;rm`` are dropped."""
    monkeypatch.setenv("STRANDS_MESH_POLICY_TYPE_ALLOW", "evil;rm,my_type,FOO BAR")
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.security"):
        allow = sec._policy_type_allowlist()
    # The whole set is lowercased so "my_type" survives but the typo'd
    # ones are gone.
    assert "my_type" in allow, "valid entry must survive"
    assert "evil;rm" not in allow
    assert "foo bar" not in allow
    # Built-ins always present.
    assert "mock" in allow
    msgs = [r.getMessage() for r in caplog.records if "STRANDS_MESH_POLICY_TYPE_ALLOW" in r.getMessage()]
    assert len(msgs) >= 2, f"expected >=2 WARNING records, got {msgs}"


def test_policy_host_allowlist_warning_still_works(monkeypatch, caplog):
    """Pre-existing F3 behaviour: policy_host charset still enforced."""
    monkeypatch.setenv("STRANDS_MESH_POLICY_HOST_ALLOW", "10.0.0.0/24,;rm -rf /")
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.security"):
        allow = sec._policy_host_allowlist()
    assert "10.0.0.0/24" in allow
    assert ";rm -rf " not in allow
    assert any("STRANDS_MESH_POLICY_HOST_ALLOW" in r.getMessage() for r in caplog.records)


def test_clean_hf_env_does_not_log(monkeypatch, caplog):
    """Sanity: clean entries produce no WARNINGs."""
    monkeypatch.setenv("STRANDS_MESH_HF_REPO_ALLOW", "my-org,my-org/my-repo")
    with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.security"):
        allow = sec._hf_repo_allowlist()
    assert "my-org" in allow
    assert "my-org/my-repo" in allow
    warns = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warns == [], f"clean entries must not warn; got {[w.getMessage() for w in warns]}"
