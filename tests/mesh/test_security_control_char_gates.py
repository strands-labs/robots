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


# --- 6. instruction control-character gate ---


#: The control bytes the sibling wire fields already refuse, paired with the
#: codepoint the refusal must name. ``instruction`` is the field the
#: validator's own comment calls out ("a malicious peer cannot smuggle
#: control-byte instruction strings in") and it was the one admitting them.
_CONTROL_BYTES = [
    ("\n", 0x0A),
    ("\r", 0x0D),
    ("\r\n", 0x0D),
    ("\x00", 0x00),
    ("\t", 0x09),
    ("\x07", 0x07),
    ("\x1b", 0x1B),
    ("\x7f", 0x7F),
    ("\x85", 0x85),
]


class TestInstructionControlCharGate:
    """``instruction`` must refuse the control bytes its siblings refuse.

    ``validate_command`` charset-gates ``policy_host``, ``server_address``,
    ``turn_id``/``sender_id``, ``override_code`` and ``robot_name``, and the
    comment introducing the sim-targeted block names the threat as a peer
    smuggling "control-byte instruction strings". ``instruction`` itself was
    bounded only by type and length, so a remote ``execute`` payload could
    carry CR/LF through to any log record that echoes it - one call emitting
    two records, the second forgeable at an arbitrary level and logger name.
    """

    def _cmd(self, instruction: str, action: str = "execute") -> dict:
        return {
            "action": action,
            "instruction": instruction,
            "policy_provider": "mock",
        }

    @pytest.mark.parametrize("ctl,_codepoint", _CONTROL_BYTES)
    def test_rejects_control_byte(self, ctl: str, _codepoint: int):
        with pytest.raises(ValidationError, match="control character"):
            validate_command(self._cmd(f"pick up the cube{ctl}trailing"))

    @pytest.mark.parametrize("ctl,codepoint", _CONTROL_BYTES)
    def test_refusal_names_the_codepoint_and_the_offset(self, ctl: str, codepoint: int):
        """The refusal locates the byte instead of only asserting one exists.

        A peer sending a 2000-char instruction cannot act on "contains a
        control character"; the codepoint plus offset identifies it exactly.
        """
        prefix = "pick up the cube"
        with pytest.raises(ValidationError) as exc:
            validate_command(self._cmd(f"{prefix}{ctl}trailing"))
        assert f"U+{codepoint:04X}" in str(exc.value)
        assert f"offset {len(prefix)}" in str(exc.value)

    def test_refusal_does_not_echo_the_payload(self):
        """The refusal must not carry the injection into the record reporting it.

        A ``ValidationError`` is logged by the dispatcher, so interpolating the
        instruction would forge the very record that reports the forgery
        attempt. The sibling fields can echo with ``%r`` because they are short
        and ASCII; a natural-language field is neither.
        """
        with pytest.raises(ValidationError) as exc:
            validate_command(self._cmd("pick up cube\nWARNING:audit:E-STOP CLEARED"))
        message = str(exc.value)
        assert "E-STOP" not in message
        assert "\n" not in message
        assert "\r" not in message

    @pytest.mark.parametrize("action", ["execute", "start"])
    def test_the_gate_covers_every_action_that_takes_an_instruction(self, action: str):
        with pytest.raises(ValidationError, match="control character"):
            validate_command(self._cmd("go\nforged", action=action))

    @pytest.mark.parametrize("ctl,_codepoint", _CONTROL_BYTES)
    def test_instruction_and_robot_name_agree_on_control_bytes(self, ctl: str, _codepoint: int):
        """One rule for both, so neither field is the soft way in.

        ``robot_name`` refused these bytes while ``instruction`` - carried in
        the same payload, on the same wire, to the same dispatcher - admitted
        them.
        """
        with pytest.raises(ValidationError):
            validate_command({"action": "execute", "instruction": "go", "robot_name": f"arm{ctl}1"})
        with pytest.raises(ValidationError):
            validate_command(self._cmd(f"go{ctl}1"))

    def test_accepts_a_clean_instruction(self):
        out = validate_command(self._cmd("pick up the red cube and place it in the bin"))
        assert out["instruction"] == "pick up the red cube and place it in the bin"

    def test_accepts_printable_non_ascii(self):
        """A natural-language field bounds the control range, not the charset.

        This is why the gate is not :data:`_SAFE_PASSTHROUGH_RE`: that regex
        admits only 0x20-0x7E, so reusing it for ``instruction`` would refuse
        any instruction needing a non-ASCII letter. ``robot_name`` is an
        identifier and is right to be ASCII-only; an instruction is prose.
        """
        instruction = "pick up the caf\u00e9 cup, then the \u65e5\u672c\u8a9e box"
        out = validate_command(self._cmd(instruction))
        assert out["instruction"] == instruction

    def test_a_whitespace_only_instruction_is_still_refused_as_empty(self):
        """The pre-existing empty check keeps its verdict and its wording.

        ``"\\n\\n"`` strips to nothing, so it must still be refused as empty
        rather than reclassified by the new charset gate - the two guards are
        ordered, not overlapping.
        """
        with pytest.raises(ValidationError, match="non-empty `instruction`"):
            validate_command(self._cmd("\n\n"))

    def test_the_length_bound_still_outranks_the_charset_gate(self):
        """Order is type -> length -> charset, matching the sibling fields.

        An oversize instruction that also carries a control byte reports the
        length, so a peer fixing one problem at a time is not sent in circles.
        """
        from strands_robots.mesh.security import MAX_INSTRUCTION_LEN

        with pytest.raises(ValidationError, match="exceeds"):
            validate_command(self._cmd("x" * MAX_INSTRUCTION_LEN + "\n"))
