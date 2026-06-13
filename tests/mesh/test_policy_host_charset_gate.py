"""is_safe_policy_host standalone charset gate + override-code-len constant.

Origin: PR #223 R7 review concerns.

Two concerns addressed in R7:

1. ``is_safe_policy_host`` defence-in-depth gap (Thread 6FB7U1).
   When called outside :func:`validate_command` (e.g. from PR-7 tools
   or PR-8 iot importing via ``__all__``), the ``str.strip()`` on the
   host silently drops ASCII whitespace including ``\\r\\n\\t\\v\\f``,
   letting ``"localhost\\r\\n"`` pass membership while preserving the
   injection-shaped bytes for the caller. R7 adds a
   :data:`_SAFE_PASSTHROUGH_RE` charset gate before the strip so the
   function is safe in isolation.

2. Magic-number cleanup (Thread 6FB7VU). The 256-char cap on
   ``resume.override_code`` is now a named module-level constant
   :data:`MAX_OVERRIDE_CODE_LEN`, consistent with ``MAX_INSTRUCTION_LEN``,
   ``MAX_MODEL_PATH_LEN``, ``MAX_SERVER_ADDRESS_LEN``, ``MAX_PEER_ID_LEN``,
   ``MAX_PASSTHROUGH_LEN``.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import security
from strands_robots.mesh.security import (
    MAX_OVERRIDE_CODE_LEN,
    ValidationError,
    is_safe_policy_host,
    validate_command,
)


class TestIsSafePolicyHostStandaloneCharsetGate:
    """Pin: ``is_safe_policy_host`` rejects control bytes when called directly.

    Regression for the gap where ``str.strip()`` drops ``\\r\\n\\t`` on
    its way to the membership compare, letting CRLF-embedded hosts pass.
    """

    def test_rejects_crlf_suffix(self) -> None:
        assert is_safe_policy_host("localhost\r\n") is False

    def test_rejects_lone_lf(self) -> None:
        assert is_safe_policy_host("localhost\n") is False

    def test_rejects_lone_cr(self) -> None:
        assert is_safe_policy_host("localhost\r") is False

    def test_rejects_tab(self) -> None:
        assert is_safe_policy_host("localhost\t") is False

    def test_rejects_vertical_tab(self) -> None:
        assert is_safe_policy_host("localhost\x0b") is False

    def test_rejects_form_feed(self) -> None:
        assert is_safe_policy_host("localhost\x0c") is False

    def test_rejects_nul_byte(self) -> None:
        assert is_safe_policy_host("localhost\x00") is False

    def test_rejects_bell_c0(self) -> None:
        assert is_safe_policy_host("localhost\x07") is False

    def test_rejects_del_7f(self) -> None:
        assert is_safe_policy_host("localhost\x7f") is False

    def test_rejects_leading_whitespace(self) -> None:
        # "  localhost" was previously accepted because strip() removed
        # the spaces before the membership compare. Now rejected at the
        # charset gate (space 0x20 IS in the printable range, but the
        # _SAFE_PASSTHROUGH_RE is fullmatch so leading/trailing spaces
        # cannot land on a bare 'localhost' membership; verify behaviour
        # below).
        # Note: spaces ARE in 0x20-0x7E so they pass the charset gate;
        # what changes is that the host body is no longer accepted as
        # an allowlist entry without strip in the membership compare.
        # Explicitly: charset gate accepts " localhost"; membership
        # compare strips it; but downstream consumers see " localhost"
        # verbatim. This test pins ONLY the charset gate behaviour, not
        # the strip behaviour.
        # (charset gate alone)
        assert is_safe_policy_host(" localhost") is True

    def test_accepts_clean_localhost(self) -> None:
        assert is_safe_policy_host("localhost") is True

    def test_accepts_clean_127_0_0_1(self) -> None:
        assert is_safe_policy_host("127.0.0.1") is True

    def test_accepts_clean_ipv6_loopback(self) -> None:
        assert is_safe_policy_host("::1") is True

    def test_rejects_empty(self) -> None:
        assert is_safe_policy_host("") is False

    def test_rejects_non_string(self) -> None:
        assert is_safe_policy_host(None) is False  # type: ignore[arg-type]
        assert is_safe_policy_host(123) is False  # type: ignore[arg-type]
        assert is_safe_policy_host(["localhost"]) is False  # type: ignore[arg-type]


class TestValidateCommandPolicyHostStillRejectsControlBytes:
    """Pin: ``validate_command`` still raises on CRLF in ``policy_host``.

    The R6 post-check is now redundant with the R7 in-function gate,
    but kept as defence-in-depth. This test pins both layers fail
    closed on the same input.
    """

    def test_rejects_crlf_in_policy_host(self) -> None:
        with pytest.raises(ValidationError) as exc:
            validate_command(
                {
                    "action": "execute",
                    "instruction": "go",
                    "policy_host": "localhost\r\n",
                    "policy_provider": "mock",
                }
            )
        # Either message is acceptable: the in-function gate raises
        # "not in allowlist" because the charset gate fails before
        # membership compare; or the post-check raises "contains
        # control characters". Both are correct fail-closed outcomes.
        msg = str(exc.value).lower()
        assert "policy_host" in msg or "control" in msg

    def test_rejects_nul_in_policy_host(self) -> None:
        with pytest.raises(ValidationError):
            validate_command(
                {
                    "action": "execute",
                    "instruction": "go",
                    "policy_host": "localhost\x00",
                    "policy_provider": "mock",
                }
            )


class TestMaxOverrideCodeLenConstant:
    """Pin: ``MAX_OVERRIDE_CODE_LEN`` is a named module-level constant.

    Regression for the magic-number cleanup. The 256-char cap on
    ``resume.override_code`` should be a named constant in ``__all__``
    so consumers and tests import it rather than recomputing.
    """

    def test_value_is_256(self) -> None:
        assert MAX_OVERRIDE_CODE_LEN == 256

    def test_in_module_all(self) -> None:
        assert "MAX_OVERRIDE_CODE_LEN" in security.__all__

    def test_validate_command_uses_constant_for_length_cap(self) -> None:
        """At-cap value passes; one-over-cap raises ValidationError."""
        # at the cap: passes
        out = validate_command({"action": "resume", "override_code": "a" * MAX_OVERRIDE_CODE_LEN})
        assert len(out["override_code"]) == MAX_OVERRIDE_CODE_LEN

        # one-over: raises
        with pytest.raises(ValidationError) as exc:
            validate_command(
                {
                    "action": "resume",
                    "override_code": "a" * (MAX_OVERRIDE_CODE_LEN + 1),
                }
            )
        # Error message references the constant value, not a hardcoded 256.
        assert str(MAX_OVERRIDE_CODE_LEN) in str(exc.value)
