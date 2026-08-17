"""Every provider spelling ``create_policy`` accepts is discoverable.

``create_policy`` resolves a provider by its canonical name *or* by any
``aliases``/``shorthands`` its ``policies.json`` entry declares, so
``create_policy("sonic")`` builds the same policy as
``create_policy("wbc")``. A caller who cannot enumerate those spellings has
to already know them, which is the opposite of what a discovery surface is
for -- and ``docs/policies/overview.md`` points readers at
``list_providers()`` as the way to list what ``create_policy`` accepts.

The headline guard asks the question a caller actually has -- is every
accepted spelling reported by *some* public discovery surface -- rather than
naming one function, so it keeps holding if the surfaces are later renamed
or split.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from strands_robots.policies import create_policy, list_providers, register_policy
from strands_robots.registry import list_policy_providers

POLICIES_JSON = Path(__file__).resolve().parents[2] / "strands_robots" / "registry" / "policies.json"

#: Surfaces a caller can reach to discover provider spellings. Probed by name
#: so this file collects (and the headline guard reports a real coverage gap)
#: whether or not a given surface exists yet.
_SURFACE_NAMES = ("list_providers", "list_aliases", "list_policy_providers", "list_policy_aliases")


@pytest.fixture(autouse=True)
def _isolate_runtime_registry() -> Iterator[None]:
    """Restore the process-global runtime registry after each test.

    ``register_policy`` writes into module-level dicts. Other guards in the
    suite read them, so a test that registers must not leak.
    """
    from strands_robots.policies import factory

    providers = dict(factory._runtime_registry)
    aliases = dict(factory._runtime_aliases)
    yield
    factory._runtime_registry.clear()
    factory._runtime_registry.update(providers)
    factory._runtime_aliases.clear()
    factory._runtime_aliases.update(aliases)


def _accepted_spellings() -> set[str]:
    """Every spelling ``create_policy`` resolves, read from the JSON registry."""
    providers = json.loads(POLICIES_JSON.read_text(encoding="utf-8"))["providers"]
    spellings = set(providers)
    for config in providers.values():
        spellings |= set(config.get("aliases") or [])
        spellings |= set(config.get("shorthands") or [])
    return spellings


def _discovery_surfaces() -> dict[str, Any]:
    """Return the public spelling-discovery surfaces that currently exist."""
    import strands_robots.policies as policies
    import strands_robots.registry as registry

    found: dict[str, Any] = {}
    for module in (policies, registry):
        for name in _SURFACE_NAMES:
            surface = getattr(module, name, None)
            if surface is not None:
                found[f"{module.__name__}.{name}"] = surface
    return found


def _reported_spellings() -> set[str]:
    """Union of every spelling the discovery surfaces report.

    A surface returning a list contributes its elements; one returning an
    alias mapping contributes its keys, which are the spellings a caller
    would pass.
    """
    reported: set[str] = set()
    for surface in _discovery_surfaces().values():
        reported |= set(surface())
    return reported


def test_the_registry_declares_aliases_to_discover() -> None:
    """Premise: without aliases in the registry the coverage guard is vacuous."""
    extra = _accepted_spellings() - set(list_policy_providers())
    assert extra, "policies.json declares no aliases, so alias coverage proves nothing"


def test_every_accepted_spelling_is_reported_by_some_discovery_surface() -> None:
    """No spelling ``create_policy`` accepts is invisible to discovery."""
    missing = sorted(_accepted_spellings() - _reported_spellings())
    assert not missing, (
        f"create_policy accepts {sorted(missing)} but no public discovery surface reports them, "
        f"so a caller has to already know the spelling; surfaces probed: "
        f"{sorted(_discovery_surfaces())}"
    )


def test_each_reported_alias_names_a_registered_provider() -> None:
    """An advertised alias must resolve to a provider that exists.

    Graded against ``list_providers`` rather than the JSON registry alone:
    a runtime alias legitimately names a runtime provider.
    """
    from strands_robots.policies import list_aliases

    known = set(list_providers())
    dangling = {alias: target for alias, target in list_aliases().items() if target not in known}
    assert not dangling, f"aliases point at unregistered providers: {dangling}"


def test_no_alias_is_reported_as_an_alias_of_itself() -> None:
    """A canonical name is not its own alias.

    Thirteen providers redundantly repeat their own name in ``aliases``.
    Reporting those would claim ``mock`` is an alias of ``mock`` and would
    double-count spellings ``list_providers`` already returns, so the
    mapping matches the robot registry's ``list_aliases``, which has no
    such entries.
    """
    from strands_robots.policies import list_aliases

    identity = sorted(alias for alias, target in list_aliases().items() if alias == target)
    assert not identity, f"reported as aliases of themselves: {identity}"


@pytest.mark.parametrize("alias", ["random", "test"])
def test_a_reported_alias_builds_the_same_policy_as_its_canonical_name(alias: str) -> None:
    """The mapping is true: the alias really does build the canonical policy.

    Restricted to the ``mock`` provider's aliases so the check needs no
    optional dependency and no checkpoint.
    """
    from strands_robots.policies import list_aliases

    canonical = list_aliases()[alias]
    assert canonical == "mock", f"fixture assumes {alias!r} is a mock alias, registry says {canonical!r}"
    assert type(create_policy(alias)) is type(create_policy(canonical))


def test_a_runtime_alias_is_reported_too() -> None:
    """``register_policy`` aliases are discoverable, as its provider names are.

    ``list_providers`` already reports runtime aliases, so a mapping that
    covered only the JSON registry would advertise a spelling while leaving
    the provider it resolves to unknowable.
    """
    from strands_robots.policies import list_aliases

    register_policy("alias_probe_provider", lambda: type(create_policy("mock")), aliases=["alias_probe"])
    assert list_aliases().get("alias_probe") == "alias_probe_provider"


def test_a_runtime_alias_shadows_a_json_alias_as_create_policy_does() -> None:
    """Reported precedence matches resolution precedence.

    ``_resolve_policy_class`` consults the runtime registry before
    ``policies.json``, so a runtime alias reusing a JSON alias name wins.
    """
    from strands_robots.policies import list_aliases

    assert list_aliases()["sonic"] == "wbc"
    register_policy("shadowing_provider", lambda: type(create_policy("mock")), aliases=["sonic"])
    assert list_aliases()["sonic"] == "shadowing_provider"


def test_list_providers_does_not_absorb_the_registry_aliases() -> None:
    """Control: the canonical list stays canonical.

    ``docs/policies/overview.md``'s provider table is pinned to exactly the
    canonical set by ``tests/test_docs_policy_coverage.py``, so widening
    this list instead of adding the alias mapping would put alias rows in a
    table of providers. Passes before and after the alias surface exists.

    Asserted as "no JSON alias appears among the reported providers" rather
    than set equality, because other tests in the suite register throwaway
    runtime providers into the process-global registry without tearing them
    down, and those legitimately show up in ``list_providers()``. The alias
    set is read from ``policies.json`` so this control does not depend on
    the surface it is guarding against.
    """
    reported = set(list_providers())
    json_aliases = _accepted_spellings() - set(list_policy_providers())
    absorbed = sorted(json_aliases & reported)
    assert not absorbed, f"list_providers() now reports JSON aliases as providers: {absorbed}"
    assert set(list_policy_providers()) <= reported, "a canonical provider stopped being reported"


def test_the_policy_alias_mapping_has_the_shape_the_robot_one_does() -> None:
    """Both registries answer "what does this alias mean" alike.

    The robot registry already promotes its private alias builder as
    ``list_aliases``; this pins that the policy counterpart reports the same
    kind of mapping rather than a differently-shaped one.
    """
    from strands_robots.policies import list_aliases
    from strands_robots.registry import list_aliases as list_robot_aliases

    robot_aliases = list_robot_aliases()
    policy_aliases = list_aliases()
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in policy_aliases.items())
    assert not [a for a, c in robot_aliases.items() if a == c], "robot registry shape changed"
    assert not [a for a, c in policy_aliases.items() if a == c]
