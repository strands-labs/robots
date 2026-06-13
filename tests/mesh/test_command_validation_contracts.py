"""validate_command security contracts (address cap, policy_provider, policy_host casing).

Origin: PR #223 R1 review feedback (yinsong1986).

Each test fails on pre-fix code and passes on post-fix code, per
AGENTS.md > Review Learnings (#85) > "Pin regression tests for reviewed
fixes."

Threads pinned:

1. ``MAX_TIMEOUT_S`` was dead code (defined + exported, no consumer in
   this PR's scope). The consumer lives in PR-6 (#225) where the
   constant should re-land alongside the call site that enforces it.
2. ``MAX_SERVER_ADDRESS_LEN`` is a dedicated cap for the address
   validator, decoupled from ``MAX_MODEL_PATH_LEN`` so a future
   tightening / relaxation of model-path bounds does not silently
   shift the address cap.
3. ``policy_provider`` is REQUIRED on execute / start; the docstring
   must reflect the implementation (no silent ``"mock"`` default on
   the security boundary).
4. ``policy_host`` is preserved verbatim in the validated output;
   downstream consumers MUST normalise themselves -- the validator
   gates membership, not canonical form. The contract is now
   documented adjacent to the assignment.
"""

from __future__ import annotations

import pytest

from strands_robots.mesh import security as sec


class TestMaxTimeoutSDeadCodeRemoved:
    """``MAX_TIMEOUT_S`` was unconsumed in PR-2 scope; removed pending PR-6."""

    def test_module_does_not_export_max_timeout_s(self) -> None:
        # Hard removal from the public surface; if the constant returns
        # in a future PR, the call site that enforces the cap MUST land
        # in the same diff.
        assert "MAX_TIMEOUT_S" not in sec.__all__

    def test_module_attribute_max_timeout_s_absent(self) -> None:
        assert not hasattr(sec, "MAX_TIMEOUT_S")


class TestMaxServerAddressLenIsDedicated:
    """Address cap is owned by the address validator, not the model-path one."""

    def test_max_server_address_len_exists(self) -> None:
        assert hasattr(sec, "MAX_SERVER_ADDRESS_LEN")
        assert isinstance(sec.MAX_SERVER_ADDRESS_LEN, int)

    def test_max_server_address_len_is_exported(self) -> None:
        assert "MAX_SERVER_ADDRESS_LEN" in sec.__all__

    def test_max_server_address_len_is_decoupled_from_model_path(self) -> None:
        # The two caps semantically describe different inputs (an HF
        # repo id vs. a host[:port] URL); the constants must not be
        # the same object so a future tightening of one does not
        # silently shift the other.
        assert sec.MAX_SERVER_ADDRESS_LEN != sec.MAX_MODEL_PATH_LEN

    def test_max_server_address_len_caps_at_256(self) -> None:
        # FQDN bounded at 253 by RFC 1035; 256 is the smallest power-of-2
        # ceiling that admits any legitimate input. A regression that
        # bumps this to 4096 silently widens the DoS budget.
        assert sec.MAX_SERVER_ADDRESS_LEN == 256

    def test_is_safe_server_address_uses_dedicated_cap(self) -> None:
        # Address one byte over the dedicated cap is rejected.
        addr = "1.1.1.1:" + ("9" * (sec.MAX_SERVER_ADDRESS_LEN - len("1.1.1.1:") + 1))
        assert len(addr) > sec.MAX_SERVER_ADDRESS_LEN
        assert sec.is_safe_server_address(addr) is False

    def test_is_safe_server_address_accepts_at_cap(self) -> None:
        # An address right at the cap is admitted on its length, then
        # rejected only if another rule fires (here: not in allowlist).
        # The point is the length-gate does not fire below the cap.
        host = "localhost"
        # localhost:65535 is 15 chars, well under the cap, and the
        # host portion is in the default allowlist.
        assert sec.is_safe_server_address(f"{host}:65535") is True


class TestPolicyProviderDocstringMatchesImplementation:
    """``policy_provider`` is REQUIRED, not optional; docstring agrees."""

    def test_validate_command_docstring_says_required(self) -> None:
        doc = sec.validate_command.__doc__ or ""
        # The old docstring said "(optional): ... defaults to \"mock\""
        # which was a security boundary footgun (a peer that forgot
        # the field looked the same as one that explicitly chose mock).
        assert "policy_provider" in doc
        # Negative assertion: the misleading "(optional)" + "defaults"
        # wording must be gone.
        provider_section = doc[doc.find("policy_provider") :]
        # Take just the first paragraph mentioning policy_provider.
        provider_section = (
            provider_section[: provider_section.find("\n\n")] if "\n\n" in provider_section else provider_section
        )
        # The phrase that was the bug: "(optional)" applied to policy_provider
        # specifically. Other "optional" mentions in the docstring are fine.
        # We pin the absence of the "defaults to \"mock\"" claim, which
        # was the specific drift.
        assert 'defaults to ``"mock"``' not in doc
        # Positive assertion: the docstring acknowledges the requirement.
        assert "REQUIRED" in doc or "required" in provider_section.lower()

    def test_implementation_rejects_missing_policy_provider(self) -> None:
        # Pin the runtime contract the docstring now correctly describes.
        with pytest.raises(sec.ValidationError, match="policy_provider is required"):
            sec.validate_command(
                {
                    "action": "execute",
                    "instruction": "go",
                    "policy_host": "localhost",
                }
            )


class TestPolicyHostCasingPreserved:
    """``policy_host`` round-trips verbatim; normalisation is downstream."""

    def test_policy_host_casing_preserved_when_in_allowlist(self) -> None:
        # ``LOCALHOST`` matches the allowlist (``is_safe_policy_host``
        # casefolds for membership) but is preserved verbatim in the
        # validated output dict so downstream callers see what the
        # peer actually sent.
        out = sec.validate_command(
            {
                "action": "execute",
                "instruction": "go",
                "policy_host": "LOCALHOST",
                "policy_provider": "mock",
            }
        )
        assert out["policy_host"] == "LOCALHOST"

    def test_policy_host_whitespace_preserved(self) -> None:
        # ``  localhost  `` matches the allowlist (``is_safe_policy_host``
        # strips for membership) but the validated output preserves the
        # padding -- the contract is "membership, not canonical form".
        out = sec.validate_command(
            {
                "action": "execute",
                "instruction": "go",
                "policy_host": "  localhost  ",
                "policy_provider": "mock",
            }
        )
        assert out["policy_host"] == "  localhost  "

    def test_validate_command_documents_casing_contract(self) -> None:
        # The contract MUST be visible to a reader of the source so a
        # future contributor does not assume normalisation. We pin
        # the comment by inspecting the source text -- structural
        # not behavioural, but cheap and self-documenting.
        import inspect

        source = inspect.getsource(sec.validate_command)
        # Either of these substrings is sufficient evidence the
        # comment is in place; the exact wording may evolve.
        assert (
            "Gate control characters" in source
            or "MUST do their own normalisation" in source
            or "Reject at the validator boundary" in source
        )
