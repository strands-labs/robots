"""Every ``policy_config`` key the schema attributes to a provider must exist.

``policy_config`` is a free-form dict splatted into
:func:`strands_robots.policies.create_policy`, which calls
``PolicyClass(**resolved_kwargs)`` without inspecting the names. What a key no
constructor declares does therefore depends on the provider, and the registry
holds both shapes: a provider whose ``__init__`` has a ``**kwargs``
forward-compatibility absorber builds and drops the value without a warning,
and a provider without one raises ``TypeError`` naming the keyword. Neither
answer tells the caller that the key list they read is wrong - the drop is
silent, and the ``TypeError`` names a Python keyword rather than the schema that
advertised it. So an advertised-but-absent key is never refused *as a wrong
key*, which is what these tests grade the description for.

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

import pytest

from strands_robots.policies import create_policy
from strands_robots.policies.factory import _resolve_policy_class
from strands_robots.registry.policies import list_policy_providers

_SPEC_PATH = Path(__file__).resolve().parents[3] / "strands_robots/simulation/mujoco/tool_spec.json"

# A description that stopped naming per-provider key lists would make every
# grading test below vacuous, so the shape the parser depends on is pinned.
_MINIMUM_PROVIDER_GROUPS = 3
_MINIMUM_ADVERTISED_KEYS = 10

# A key no provider declares, and the two things a constructor can do with one.
_UNBACKED_KEY = "definitely_not_a_parameter"
_OUTCOMES = ("dropped", "TypeError")

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


def _constructor_parameters(provider: str) -> set[str]:
    """Return the parameter names a provider's constructor declares by name."""
    _, cls, _ = _resolve_policy_class(provider)
    return {
        name
        for name, parameter in inspect.signature(cls.__init__).parameters.items()
        if name != "self" and parameter.kind not in (parameter.VAR_KEYWORD, parameter.VAR_POSITIONAL)
    }


def _unbacked_keys(description: str) -> dict[str, list[str]]:
    """Advertised keys, per provider, that the provider's constructor lacks."""
    unbacked: dict[str, list[str]] = {}
    for provider, keys in _advertised(description).items():
        explicit = _constructor_parameters(provider)
        absent = [k for k in keys if k not in explicit]
        if absent:
            unbacked[provider] = absent
    return unbacked


def _unknown_keyword_outcome(provider: str) -> str:
    """Return what a provider's constructor does with an unbacked keyword.

    ``"dropped"`` when a ``**kwargs`` absorber takes the name, ``"TypeError"``
    when the signature refuses it. Read by binding against the signature, which
    is the step Python performs at :func:`create_policy`'s
    ``PolicyClass(**resolved_kwargs)`` call, so this needs neither the
    provider's optional dependencies nor a constructed policy.
    """
    cls = _resolve_policy_class(provider)[1]
    try:
        inspect.signature(cls.__init__).bind_partial(None, **{_UNBACKED_KEY: True})
    except TypeError:
        return "TypeError"
    return "dropped"


def _a_provider_that_refuses() -> str:
    """The first registered provider whose constructor refuses an unbacked keyword.

    Derived rather than named: a provider that grows a ``**kwargs`` absorber must
    not leave a cell pinned to a name that no longer refuses anything.
    """
    refusing = sorted(p for p in list_policy_providers() if _unknown_keyword_outcome(p) == "TypeError")
    assert refusing, "no registered provider refuses an unbacked keyword"
    return refusing[0]


class TestTheSchemaOnlyAdvertisesKeysAProviderAccepts:
    """The per-provider key list is graded against the live constructors."""

    def test_every_advertised_key_is_a_constructor_parameter(self) -> None:
        unbacked = _unbacked_keys(_description())
        assert not unbacked, (
            "tool_spec.json's policy_config description attributes keys to providers whose "
            f"constructors do not declare them: {unbacked}. create_policy splats policy_config, so "
            "each of these either lands in that policy's **kwargs and is silently dropped or "
            "raises TypeError at the constructor - and neither answer tells the caller the key "
            "list is wrong. Name the parameter the provider really accepts, or drop the key."
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


class TestWhatAnUnbackedKeyDoesAtTheConstructor:
    """Both outcomes reach the caller, so no cell here may assume one of them."""

    def test_the_registry_holds_both_outcomes(self) -> None:
        """A provider that refuses an unbacked key is a supported shape, not a defect."""
        outcomes = {provider: _unknown_keyword_outcome(provider) for provider in list_policy_providers()}
        assert set(outcomes.values()) == set(_OUTCOMES), (
            f"every registered provider now answers an unbacked policy_config key the same way: "
            f"{outcomes}. The cells below describe both answers, and a cell that assumed one of "
            "them would be green or red by which providers the description happens to enumerate "
            "rather than by whether the key list is right."
        )

    def test_an_absorbing_provider_builds_and_drops_the_key(self) -> None:
        """The silent answer: create_policy reports success and the value is gone."""
        assert _unknown_keyword_outcome("mock") == "dropped", "this cell needs an absorbing provider"
        policy = create_policy("mock", **{_UNBACKED_KEY: True})
        assert not hasattr(policy, _UNBACKED_KEY), (
            f"'mock' kept {_UNBACKED_KEY} rather than absorbing it into **kwargs; if a provider "
            "now stores unknown keys, an unbacked schema key is no longer silently dropped"
        )

    def test_a_refusing_provider_raises_a_typeerror_naming_the_keyword(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The loud answer: a TypeError about a keyword, which names no schema."""
        provider = _a_provider_that_refuses()
        # A remote-code provider can sort first, and its gate runs before the
        # constructor. Keyword binding fails before __init__ executes, so opening
        # the gate here still touches no checkpoint.
        monkeypatch.setenv("STRANDS_TRUST_REMOTE_CODE", "1")
        with pytest.raises(TypeError, match=_UNBACKED_KEY) as refusal:
            create_policy(provider, **{_UNBACKED_KEY: True})
        assert "policy_config" not in str(refusal.value), (
            f"'{provider}' now refuses the key as a schema key: {refusal.value}. That is a better "
            "answer than a bare TypeError, and it means the grading above is no longer the only "
            "thing that catches a wrong key - re-read this file's premise."
        )
