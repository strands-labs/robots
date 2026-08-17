"""Every ``policy_config`` key the schema attributes to a provider must exist.

``policy_config`` is a free-form dict splatted into
:func:`strands_robots.policies.create_policy`, so a key no constructor declares
is swallowed by that policy's ``**kwargs`` forward-compatibility absorber: the
build reports success and the value is dropped without a warning. The router
binds against the method signature and never inspects these names, so an
advertised-but-absent key is never *refused* - it is simply never honoured,
which reads as the capability being broken rather than as the key being wrong.

``tool_spec.json``'s ``policy_config`` description is the only per-provider key
list an agent driving the schema ever sees, so these tests grade it against the
live constructors rather than against a copied list, and a provider that grows
a knob cannot leave the schema behind.

``TestToolSpecIsClean`` in ``test_tool_spec.py`` already pins the *level* these
keys live at - nested under ``policy_config``, never top-level. It says nothing
about which provider accepts which key, which is what these tests add.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

from strands_robots.policies.factory import _resolve_policy_class
from strands_robots.registry.policies import list_policy_providers

_SPEC_PATH = Path(__file__).resolve().parents[3] / "strands_robots/simulation/mujoco/tool_spec.json"

# A description that stopped naming per-provider key lists would make every
# grading test below vacuous, so the shape the parser depends on is pinned.
_MINIMUM_PROVIDER_GROUPS = 3
_MINIMUM_ADVERTISED_KEYS = 10

# "For 'lerobot_local': a, b, c." - one group per provider the description
# bothers to enumerate keys for.
_GROUP_RE = re.compile(r"For '([a-z0-9_]+)': ([^.]*)\.")


def _description() -> str:
    spec = json.loads(_SPEC_PATH.read_text())
    return str(spec["properties"]["policy_config"]["description"])


def _advertised(description: str) -> dict[str, list[str]]:
    """Map each provider the description enumerates to the keys it claims.

    ``For 'mock': {} is fine.`` advertises no keys, which is a legitimate entry
    and yields an empty list rather than being dropped.
    """
    groups: dict[str, list[str]] = {}
    for provider, blob in _GROUP_RE.findall(description):
        keys = [k.strip() for k in blob.split(",")]
        # Accumulate rather than assign: a provider named twice would otherwise
        # have its first group silently dropped, ungraded.
        groups.setdefault(provider, []).extend(k for k in keys if k and "is fine" not in k)
    return groups


def _constructor_parameters(provider: str) -> tuple[set[str], bool]:
    """Return ``(explicit parameter names, accepts **kwargs)`` for a provider."""
    _, cls, _ = _resolve_policy_class(provider)
    signature = inspect.signature(cls.__init__)
    explicit = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
    }
    accepts_kwargs = any(p.kind is p.VAR_KEYWORD for p in signature.parameters.values())
    return explicit, accepts_kwargs


def _unbacked_keys(description: str) -> dict[str, list[str]]:
    """Advertised keys, per provider, that the provider's constructor lacks."""
    unbacked: dict[str, list[str]] = {}
    for provider, keys in _advertised(description).items():
        explicit, _ = _constructor_parameters(provider)
        absent = [k for k in keys if k not in explicit]
        if absent:
            unbacked[provider] = absent
    return unbacked


class TestTheSchemaOnlyAdvertisesKeysAProviderAccepts:
    """The per-provider key list is graded against the live constructors."""

    def test_every_advertised_key_is_a_constructor_parameter(self) -> None:
        unbacked = _unbacked_keys(_description())
        assert not unbacked, (
            "tool_spec.json's policy_config description attributes keys to providers whose "
            f"constructors do not declare them: {unbacked}. create_policy splats policy_config, "
            "so each of these is absorbed by that policy's **kwargs and silently dropped - the "
            "build reports success and the value never reaches the policy. Name the parameter the "
            "provider really accepts, or drop the key."
        )

    def test_every_named_provider_is_registered(self) -> None:
        registered = set(list_policy_providers())
        named = set(_advertised(_description()))
        assert named <= registered, (
            f"policy_config's description names providers that are not registered: {sorted(named - registered)}"
        )


class TestTheGradingIsNotVacuous:
    """A clean result has to mean the description is right."""

    def test_the_description_still_names_several_provider_key_lists(self) -> None:
        groups = _advertised(_description())
        assert len(groups) >= _MINIMUM_PROVIDER_GROUPS, (
            f"only {len(groups)} provider group(s) parsed out of policy_config's description "
            f"(expected at least {_MINIMUM_PROVIDER_GROUPS}); a reformat that hides the "
            "'For <provider>: <keys>.' shape would make the grading above pass by reaching nothing"
        )
        total = sum(len(keys) for keys in groups.values())
        assert total >= _MINIMUM_ADVERTISED_KEYS, (
            f"only {total} key(s) parsed out of policy_config's description "
            f"(expected at least {_MINIMUM_ADVERTISED_KEYS})"
        )

    def test_a_key_no_constructor_declares_is_reported(self) -> None:
        planted = _description().replace(
            "For 'mock': {} is fine.",
            "For 'mock': definitely_not_a_parameter.",
            1,
        )
        # Scoped to the planted group so this stays a control: it reports whether the
        # grader can see a bogus key at all, independently of any real offender.
        assert _unbacked_keys(planted).get("mock") == ["definitely_not_a_parameter"]

    def test_a_provider_may_advertise_no_keys(self) -> None:
        """``For 'mock': {} is fine.`` is an entry, not a parse failure."""
        assert _advertised(_description())["mock"] == []


class TestWhyAnUnbackedKeyIsSilent:
    """The absorber is what makes a wrong key a dropped value, not an error."""

    @pytest.mark.parametrize("provider", sorted(_advertised(_description())))
    def test_the_constructor_absorbs_unknown_keywords(self, provider: str) -> None:
        _, accepts_kwargs = _constructor_parameters(provider)
        assert accepts_kwargs, (
            f"'{provider}' no longer absorbs unknown constructor keywords, so an unbacked "
            "policy_config key would now raise TypeError instead of being dropped. That is a "
            "stricter contract than the one these tests assume - re-check whether the schema "
            "still needs grading against the signature, or whether the constructor now does it."
        )

    def test_an_unknown_key_reaches_the_absorber_unvalidated(self) -> None:
        """Nothing between the schema and the constructor filters these names."""
        explicit, _ = _constructor_parameters("lerobot_local")
        assert "definitely_not_a_parameter" not in explicit
        policy: Any = _resolve_policy_class("lerobot_local")[1]
        bound = inspect.signature(policy.__init__).bind_partial(
            None, **dict.fromkeys(["definitely_not_a_parameter"], True)
        )
        assert "definitely_not_a_parameter" in bound.kwargs or "definitely_not_a_parameter" in bound.arguments
