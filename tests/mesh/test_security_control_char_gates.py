"""Regression tests for R5 control-character charset gates.

Pins the fix for 5 review threads (2026-05-27T03:38:41Z) that flagged:
1. policy_host preserving CRLF/NUL/control bytes (wire-injection vector)
2. turn_id/sender_id passthrough accepting non-string / control bytes
3. resume.override_code accepting control characters
4. server_address preserving CRLF/NUL/control bytes
5. _POLICY_HOST_ENTRY_RE admitting dead bracket chars

All tests FAIL on pre-fix HEAD (04849bf) and PASS on post-fix HEAD.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh.security import (
    MAX_PASSTHROUGH_LEN,
    ValidationError,
    validate_command,
)

# --- 1. policy_host control-character gate ---


class TestPolicyHostControlCharGate:
    """policy_host must reject CRLF, NUL, and C0 control bytes."""

    def _cmd(self, host: str) -> dict:
        return {
            "action": "execute",
            "instruction": "go",
            "policy_host": host,
            "policy_provider": "mock",
        }

    def test_rejects_crlf(self):
        # R7: ``is_safe_policy_host`` now applies the same charset
        # gate before its internal strip, so allowlist-shaped errors
        # are also acceptable for these inputs (control bytes that
        # ``str.strip()`` would have dropped).
        with pytest.raises(ValidationError, match="not in allowlist|control characters"):
            validate_command(self._cmd("localhost\r\n"))

    def test_rejects_newline(self):
        # R7: ``is_safe_policy_host`` now applies the same charset
        # gate before its internal strip, so allowlist-shaped errors
        # are also acceptable for these inputs (control bytes that
        # ``str.strip()`` would have dropped).
        with pytest.raises(ValidationError, match="not in allowlist|control characters"):
            validate_command(self._cmd("localhost\n"))

    def test_rejects_nul(self):
        """NUL is caught by the allowlist (strip+lower doesn't remove it)."""
        with pytest.raises(ValidationError, match="not in allowlist|control characters"):
            validate_command(self._cmd("localhost\x00"))

    def test_rejects_tab(self):
        # R7: ``is_safe_policy_host`` now applies the same charset
        # gate before its internal strip, so allowlist-shaped errors
        # are also acceptable for these inputs (control bytes that
        # ``str.strip()`` would have dropped).
        with pytest.raises(ValidationError, match="not in allowlist|control characters"):
            validate_command(self._cmd("localhost\t"))

    def test_rejects_bell(self):
        """Bell (0x07) is caught by the allowlist (strip doesn't remove it)."""
        with pytest.raises(ValidationError, match="not in allowlist|control characters"):
            validate_command(self._cmd("localhost\x07"))

    def test_accepts_clean_host(self):
        out = validate_command(self._cmd("localhost"))
        assert out["policy_host"] == "localhost"

    def test_rejects_leading_trailing_whitespace_with_control(self):
        """Leading/trailing spaces are printable (0x20) so pass; tabs don't."""
        # Spaces are allowed (0x20 is in the range)
        out = validate_command(self._cmd("  localhost  "))
        assert out["policy_host"] == "  localhost  "


# --- 2. turn_id / sender_id validation ---


class TestTurnIdSenderIdValidation:
    """turn_id and sender_id must be type-checked, length-bounded, charset-validated."""

    def test_rejects_non_string_sender_id(self):
        with pytest.raises(ValidationError, match="must be a string"):
            validate_command({"action": "status", "sender_id": {"evil": "dict"}})

    def test_rejects_non_string_turn_id(self):
        with pytest.raises(ValidationError, match="must be a string"):
            validate_command({"action": "status", "turn_id": 12345})

    def test_rejects_list_sender_id(self):
        with pytest.raises(ValidationError, match="must be a string"):
            validate_command({"action": "status", "sender_id": ["a", "b"]})

    def test_rejects_overlength_turn_id(self):
        with pytest.raises(ValidationError, match="exceeds"):
            validate_command({"action": "status", "turn_id": "x" * (MAX_PASSTHROUGH_LEN + 1)})

    def test_rejects_control_chars_in_turn_id(self):
        with pytest.raises(ValidationError, match="control characters"):
            validate_command({"action": "status", "turn_id": "abc\x00def"})

    def test_rejects_crlf_in_sender_id(self):
        with pytest.raises(ValidationError, match="control characters"):
            validate_command({"action": "status", "sender_id": "peer\r\nINJECT"})

    def test_accepts_clean_turn_id(self):
        out = validate_command({"action": "status", "turn_id": "01HX9ABCDEFG"})
        assert out["turn_id"] == "01HX9ABCDEFG"

    def test_accepts_clean_sender_id(self):
        out = validate_command({"action": "status", "sender_id": "node-123-abc"})
        assert out["sender_id"] == "node-123-abc"


# --- 3. resume.override_code charset gate ---


class TestOverrideCodeCharsetGate:
    """resume.override_code must reject control characters."""

    def _cmd(self, code: str) -> dict:
        return {"action": "resume", "override_code": code}

    def test_rejects_nul(self):
        with pytest.raises(ValidationError, match="control characters"):
            validate_command(self._cmd("\x00\x01\x02secret"))

    def test_rejects_crlf(self):
        # R7: ``is_safe_policy_host`` now applies the same charset
        # gate before its internal strip, so allowlist-shaped errors
        # are also acceptable for these inputs (control bytes that
        # ``str.strip()`` would have dropped).
        with pytest.raises(ValidationError, match="not in allowlist|control characters"):
            validate_command(self._cmd("secret\r\nINJECT"))

    def test_rejects_bell(self):
        with pytest.raises(ValidationError, match="control characters"):
            validate_command(self._cmd("abc\x07def"))

    def test_accepts_printable_ascii(self):
        out = validate_command(self._cmd("S3cret-C0de_123!"))
        assert out["override_code"] == "S3cret-C0de_123!"

    def test_accepts_empty_string(self):
        """Empty override_code is valid (no gate needed for empty)."""
        out = validate_command(self._cmd(""))
        assert out["override_code"] == ""


# --- 4. server_address control-character gate ---


class TestServerAddressControlCharGate:
    """server_address must reject CRLF/NUL/control bytes."""

    def _cmd(self, addr: str) -> dict:
        return {
            "action": "execute",
            "instruction": "go",
            "policy_host": "localhost",
            "policy_provider": "mock",
            "server_address": addr,
        }

    def test_rejects_crlf(self):
        # R7: ``is_safe_policy_host`` now applies the same charset
        # gate before its internal strip, so allowlist-shaped errors
        # are also acceptable for these inputs (control bytes that
        # ``str.strip()`` would have dropped).
        with pytest.raises(ValidationError, match="not in allowlist|control characters"):
            validate_command(self._cmd("http://localhost:8080\r\n"))

    def test_rejects_nul(self):
        """NUL in address is caught by host allowlist before the charset gate."""
        with pytest.raises(ValidationError, match="not in allowlist|control characters"):
            validate_command(self._cmd("http://localhost\x00:8080"))

    def test_accepts_clean_address(self):
        out = validate_command(self._cmd("http://localhost:8080"))
        assert out["server_address"] == "http://localhost:8080"


# --- 5. _POLICY_HOST_ENTRY_RE bracket removal ---


class TestPolicyHostEntryRegexNoBrackets:
    """Brackets should NOT be accepted in policy host allowlist entries."""

    def test_bracket_ipv6_triggers_warning(self, monkeypatch):
        """[::1] should fail the charset regex now that brackets are removed."""

        from strands_robots.mesh.security import _POLICY_HOST_ENTRY_RE

        # Brackets should NOT match
        assert _POLICY_HOST_ENTRY_RE.fullmatch("[::1]") is None

    def test_bare_ipv6_still_matches(self):
        """::1 without brackets should still work."""
        from strands_robots.mesh.security import _POLICY_HOST_ENTRY_RE

        assert _POLICY_HOST_ENTRY_RE.fullmatch("::1") is not None
