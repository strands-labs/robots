# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The default policy allowlist admits every provider the registry can build.

``strands_robots/registry/policies.json`` is the shipped set of policy
providers ``create_policy`` can build, and it resolves a provider by its
canonical name *or* by any alias / shorthand the entry declares. The mesh
``execute`` / ``start`` boundary and the Device Connect drivers both gate
``policy_provider`` through one allowlist, whose built-in half is a literal in
:mod:`strands_robots.mesh.security`.

A provider present in the registry but absent from that literal is refused on
every remote path while working locally, and the only recourse an operator has
is a broad ``STRANDS_MESH_POLICY_TYPE_ALLOW`` grant -- which reopens the gate
the allowlist exists to close. The literal is intentional (a new provider must
not silently widen a security allowlist), so this module supplies the
enforcement the "keep this in sync" comment cannot: the omission fails here
instead of shipping as a remote-only availability bug.

The coverage requirement is one-directional. The allowlist legitimately holds
names the provider registry does not: the LeRobot policy *families* accepted as
``policy_type``. So these tests require registry-implies-allowlist and pin the
excess separately, rather than demanding set equality.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import strands_robots.mesh.security as sec
from strands_robots.mesh.security import ValidationError, is_safe_policy_provider, validate_command
from strands_robots.registry.policies import get_policy_provider, list_policy_providers

_REGISTRY_JSON = Path(sec.__file__).resolve().parent.parent / "registry" / "policies.json"

# A registry that has shrunk to a handful of entries would make every
# coverage assertion below pass without proving anything. Pinned well under
# today's count so ordinary provider churn never trips it.
_MINIMUM_PROVIDERS = 10

# The LeRobot policy families the allowlist accepts as ``policy_type``. Not
# provider names, so they are the allowlist's legitimate excess over the
# provider registry.
_LEROBOT_POLICY_FAMILIES = frozenset({"act", "diffusion", "tdmpc", "vqbet", "pi0", "pi0fast", "smolvla", "sac"})


def _registry_spellings() -> dict[str, str]:
    """Map every accepted spelling to the provider it resolves to.

    Reads the shipped JSON directly so the expectation comes from the data the
    registry serves, not from a second copy of it in this file.
    """
    providers = json.loads(_REGISTRY_JSON.read_text(encoding="utf-8"))["providers"]
    spellings: dict[str, str] = {}
    for name, info in providers.items():
        spellings[name] = name
        for spelling in (*info.get("aliases", []), *info.get("shorthands", [])):
            spellings[spelling] = name
    return spellings


@pytest.fixture
def allowlist(monkeypatch: pytest.MonkeyPatch) -> frozenset[str]:
    """The built-in allowlist with no operator extension applied."""
    monkeypatch.delenv("STRANDS_MESH_POLICY_TYPE_ALLOW", raising=False)
    sec._clear_security_caches_for_tests()
    return sec._policy_type_allowlist()


class TestTheAllowlistCoversTheProviderRegistry:
    """Every spelling the registry resolves is accepted by the gate."""

    def test_the_registry_is_big_enough_for_this_to_mean_anything(self) -> None:
        """Premise: a shrunken registry would make the coverage checks vacuous."""
        assert len(list_policy_providers()) >= _MINIMUM_PROVIDERS

    def test_every_registry_spelling_is_allowlisted(self, allowlist: frozenset[str]) -> None:
        """A buildable provider the allowlist omits is refused on every remote path."""
        spellings = _registry_spellings()
        missing = sorted(name for name in spellings if name not in allowlist)
        assert not missing, (
            f"{len(missing)} spelling(s) resolve to a provider create_policy can build but are "
            f"refused by the mesh and Device Connect gate: "
            + ", ".join(f"{name!r} -> {spellings[name]!r}" for name in missing)
            + ". Add them to _REGISTRY_POLICY_PROVIDERS in strands_robots/mesh/security.py."
        )

    def test_every_canonical_provider_passes_the_wire_validator(self) -> None:
        """The gate is reached through validate_command, so assert on that boundary."""
        refused = []
        for provider in list_policy_providers():
            try:
                out = validate_command(
                    {"action": "execute", "instruction": "do the thing", "policy_provider": provider}
                )
            except ValidationError as exc:
                refused.append(f"{provider} ({exc})")
            else:
                assert out["policy_provider"] == provider
        assert not refused, f"validate_command refused registered provider(s): {refused}"

    def test_an_alias_is_accepted_wherever_its_canonical_name_is(self) -> None:
        """An alias resolves to the same policy class, so a split verdict is arbitrary.

        ``mock`` is the provider the validator's own refusal message recommends;
        its aliases naming the same class must not be refused while it passes.
        """
        split = []
        for spelling, canonical in _registry_spellings().items():
            if spelling == canonical:
                continue
            if is_safe_policy_provider(canonical) and not is_safe_policy_provider(spelling):
                split.append(f"{spelling!r} (alias of {canonical!r})")
        assert not split, f"alias spelling(s) refused while their canonical provider is accepted: {split}"

    def test_the_allowlist_excess_is_only_lerobot_policy_families(self, allowlist: frozenset[str]) -> None:
        """Pin the non-provider half so resyncing cannot smuggle in a stranger.

        Held as a literal rather than read back off the module under test: a
        second opinion on the accepted vocabulary is the point, and this is the
        assertion that fails if a name reaches the allowlist without being
        either a registry spelling or a declared LeRobot family.
        """
        excess = allowlist - set(_registry_spellings())
        assert excess == _LEROBOT_POLICY_FAMILIES


class TestWideningTheProviderVocabularyRelaxesNothingElse:
    """The other payload gates are what defend hosts and model downloads."""

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("policy_host", "evil.example.com"),
            ("server_address", "attacker.example.net:9000"),
            ("pretrained_name_or_path", "attacker/backdoor"),
            ("model_path", "../../etc/passwd"),
        ],
    )
    def test_a_hostile_field_is_still_refused_for_every_provider(self, field: str, value: str) -> None:
        """Each gate applies to the payload regardless of which provider it names."""
        for provider in list_policy_providers():
            with pytest.raises(ValidationError):
                validate_command(
                    {
                        "action": "execute",
                        "instruction": "do the thing",
                        "policy_provider": provider,
                        field: value,
                    }
                )

    def test_an_unregistered_provider_is_still_refused(self) -> None:
        """The gate still bounds the vocabulary to what the registry ships.

        ``import_policy_class`` falls back to importing
        ``strands_robots.policies.<name>`` for an unregistered name, so this is
        the check that keeps an arbitrary module name off the wire.
        """
        for bogus in ("definitely_not_a_provider", "base", "factory"):
            assert get_policy_provider(bogus) is None
            assert not is_safe_policy_provider(bogus)

    def test_a_runtime_registered_provider_is_not_auto_allowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``register_policy`` grants a local capability, never a remote one.

        Runtime registration lives outside the shipped registry, so it cannot
        authorise itself on the wire; the operator opts in explicitly.
        """
        from strands_robots.policies import factory

        monkeypatch.setitem(factory._runtime_registry, "in_process_only", lambda: None)
        assert "in_process_only" in factory.list_providers()
        assert not is_safe_policy_provider("in_process_only")
