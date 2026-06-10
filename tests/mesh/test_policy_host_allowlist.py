"""Pin tests for STRANDS_MESH_POLICY_HOST_ALLOW operator entries.

Operator-supplied entries are split on ',' into the policy-host allowlist
that gates ``is_safe_policy_host`` (and therefore the ``policy_host`` /
``server_address`` fields in mesh commands). Entries are validated against
a charset regex; malformed entries are dropped with a WARNING rather than
silently accepted.

This is a fail-loud-on-misconfig posture, not an injection defence -- the
downstream code does literal-equality and CIDR comparisons, not subprocess
interpolation. The point is operator visibility: a typo like
``STRANDS_MESH_POLICY_HOST_ALLOW=10.0.0.0/24,;rm -rf /`` previously
silently parsed into two entries (the malformed one would never match
downstream, but the operator had no signal that their allowlist was not
what they thought).
"""

from __future__ import annotations

import logging

from strands_robots.mesh import security as sec


class TestPolicyHostAllowlistValidation:
    """Operator-supplied entries with shell metacharacters / whitespace
    are dropped with a WARNING (fail-loud-on-misconfig).
    """

    def test_malformed_entry_dropped_with_warning(self, monkeypatch, caplog):
        monkeypatch.setenv(
            "STRANDS_MESH_POLICY_HOST_ALLOW",
            "10.0.0.0/24,;rm -rf /,vla.internal",
        )
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.security"):
            allowlist = sec._policy_host_allowlist()
        # Defaults + valid entries only
        assert "10.0.0.0/24" in allowlist
        assert "vla.internal" in allowlist
        # Malformed entry was dropped
        assert ";rm -rf /" not in allowlist
        assert any("dropping malformed entry" in m for m in caplog.messages), (
            f"expected WARNING about malformed entry; got {caplog.messages}"
        )

    def test_clean_entries_pass_through(self, monkeypatch, caplog):
        monkeypatch.setenv(
            "STRANDS_MESH_POLICY_HOST_ALLOW",
            "10.0.0.5,vla.internal,2001:db8::1",
        )
        with caplog.at_level(logging.WARNING, logger="strands_robots.mesh.security"):
            allowlist = sec._policy_host_allowlist()
        # No warnings on clean input
        assert not any("dropping malformed" in m for m in caplog.messages)
        for entry in ("10.0.0.5", "vla.internal", "2001:db8::1"):
            assert entry in allowlist
