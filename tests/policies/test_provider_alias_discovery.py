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

That headline grades the *registered* spellings, which is the set the
discovery surfaces own. ``create_policy`` resolves one more kind of spelling:
``import_policy_class`` falls back to auto-discovery, so a module under
``strands_robots.policies`` exporting a ``Policy`` subclass resolves under its
own module name with no registry entry. Those cannot be reported as providers
- one of them (``persistent``) cannot even be built through ``create_policy``,
so listing it would advertise a provider the factory refuses - so the contract
for them is the other half of discovery: the prose that teaches a caller how to
enumerate must name them. The second guard here grades that, deriving the
spellings from the resolution path rather than from a list, so a third such
module is held to the same rule the day it lands.
"""

from __future__ import annotations

import inspect
import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from strands_robots.policies import create_policy, list_providers, register_policy
from strands_robots.registry import list_policy_providers

REPO_ROOT = Path(__file__).resolve().parents[2]
POLICIES_JSON = REPO_ROOT / "strands_robots" / "registry" / "policies.json"

#: Docs that teach a caller how to enumerate accepted provider spellings.
#: Graded as prose: a runnable snippet is a usage example, not a claim about
#: which set is complete.
_ENUMERATION_DOCS = ("docs/api-reference.md", "docs/policies/overview.md")

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


def _resolved_outside_the_registry() -> dict[str, str]:
    """Spellings ``create_policy`` resolves that no discovery surface reports.

    ``import_policy_class`` falls back to auto-discovery, so a module under
    ``strands_robots.policies`` exporting a ``Policy`` subclass resolves under
    its own module name. Only names the surfaces do *not* already report are
    probed, so this needs no optional dependency: every registry provider is
    reported, and it is the unreported remainder that has to be documented.

    Returns:
        Mapping of module-name spelling to the dotted class it resolves to.
    """
    import pkgutil

    import strands_robots.policies as policies_pkg
    from strands_robots.registry.policies import import_policy_class

    reported = _reported_spellings()
    resolved: dict[str, str] = {}
    for module in pkgutil.iter_modules(policies_pkg.__path__):
        if module.name.startswith("_") or module.name in reported:
            continue
        try:
            policy_class = import_policy_class(module.name)
        except (ValueError, ImportError):
            # Not a spelling create_policy resolves: either no Policy subclass
            # to auto-discover, or its optional dependency is absent. Both are
            # "nothing to document" rather than a gap.
            continue
        resolved[module.name] = f"{policy_class.__module__}.{policy_class.__name__}"
    return resolved


def _prose_blocks(text: str) -> list[str]:
    """Split markdown into blank-line-delimited prose blocks.

    Fenced code is dropped: a runnable snippet naming both surfaces is a usage
    example, not a claim about which set is complete. A table row is its own
    block, because a whole table read as one block would let any row answer for
    a claim made in another.
    """
    blocks: list[str] = []
    chunk: list[str] = []
    in_fence = False

    def flush() -> None:
        if not chunk:
            return
        rows = [line for line in chunk if line.lstrip().startswith("|")]
        if rows:
            blocks.extend(" ".join(row.split()) for row in rows)
        else:
            blocks.append(" ".join(" ".join(chunk).split()))
        chunk.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line.strip():
            flush()
            continue
        chunk.append(line)
    flush()
    return blocks


def _names_the_spelling(text: str, spelling: str) -> bool:
    """Whether ``text`` names ``spelling`` as a value a caller would pass.

    Matched as a whole token, so a dotted path does not answer for it:
    ``strands_robots.policies.persistent.PersistentPolicy`` identifies the
    *class* to construct directly, which is a different fact from ``persistent``
    being a spelling the factory resolves. Without the boundary the substring
    inside that path would satisfy the rule.
    """
    return re.search(rf"(?<![\w.]){re.escape(spelling)}(?![\w.])", text) is not None


def _teaches_enumeration(text: str) -> bool:
    """Whether ``text`` tells a caller how to enumerate what the factory takes.

    Both surfaces plus the factory: a block naming only one of them is
    describing that function, not the accepted set.
    """
    return all(token in text for token in ("list_providers", "list_aliases", "create_policy"))


def _enumeration_surfaces() -> dict[str, str]:
    """Every surface that teaches a caller how to enumerate accepted spellings."""
    from strands_robots.policies import list_aliases

    found: dict[str, str] = {}
    docstring = inspect.getdoc(list_aliases) or ""
    if _teaches_enumeration(docstring):
        found["strands_robots/policies/factory.py::list_aliases"] = docstring
    for relative in _ENUMERATION_DOCS:
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for block in _prose_blocks(text):
            if _teaches_enumeration(block):
                found[f"{relative}: {block[:56]}..."] = block
    return found


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


def test_a_spelling_resolves_outside_the_registry() -> None:
    """Premise: with nothing auto-discovered the documentation guard is vacuous."""
    resolved = _resolved_outside_the_registry()
    assert resolved, (
        "no module under strands_robots.policies resolves outside the registry, so grading the "
        "docs for naming them proves nothing"
    )


def test_the_enumeration_surfaces_are_found() -> None:
    """Premise: a reflow that hides a surface must fail, not report clean."""
    surfaces = _enumeration_surfaces()
    assert "strands_robots/policies/factory.py::list_aliases" in surfaces, (
        f"list_aliases' own docstring no longer teaches enumeration, so the guard below grades "
        f"less than it claims; found: {sorted(surfaces)}"
    )
    for relative in _ENUMERATION_DOCS:
        assert any(key.startswith(relative) for key in surfaces), (
            f"{relative} no longer states the enumeration recipe in prose; found: {sorted(surfaces)}"
        )


def test_every_enumeration_surface_names_the_unregistered_spellings() -> None:
    """A spelling the factory resolves is reported, or named where enumeration is taught.

    ``list_providers()`` and ``list_aliases()`` report the registry. A caller
    who takes their union as "every spelling ``create_policy`` accepts" - which
    is what these surfaces told them to do - rejects ``composite``, which
    ``create_policy`` builds. Naming it is the discovery the surfaces cannot
    provide, since neither wrapper can be listed as a provider.
    """
    resolved = _resolved_outside_the_registry()
    silent = {
        label: sorted(name for name in resolved if not _names_the_spelling(text, name))
        for label, text in _enumeration_surfaces().items()
    }
    silent = {label: missing for label, missing in silent.items() if missing}
    assert not silent, (
        f"create_policy resolves {sorted(resolved)} with no registry entry, and these surfaces "
        f"teach a caller to enumerate accepted spellings without naming them: {silent}. A caller "
        f"validating against the documented union refuses a spelling the factory accepts."
    )


def test_the_composite_spelling_builds_through_the_factory() -> None:
    """Control: the spelling the docs now name really is accepted.

    Passes before and after this change - the factory's behaviour is what the
    documentation was already supposed to describe.
    """
    from strands_robots.policies import list_aliases

    assert "composite" not in set(list_providers()) | set(list_aliases())
    policy = create_policy("composite", lower=create_policy("mock"), upper=create_policy("mock"))
    assert type(policy).__name__ == "CompositePolicy"
    assert policy.provider_name == "composite"


def test_the_persistent_spelling_resolves_but_is_not_buildable_here() -> None:
    """Control: why the second wrapper is documented as a direct construction.

    ``PersistentPolicy``'s first parameter is named ``provider``, which
    :func:`create_policy` has already bound, so no keyword can reach it. That is
    what makes reporting it as a provider wrong rather than merely incomplete.
    """
    from strands_robots.registry.policies import import_policy_class

    assert import_policy_class("persistent").__name__ == "PersistentPolicy"
    with pytest.raises(TypeError, match="provider"):
        create_policy("persistent")


@pytest.mark.parametrize("module_name", ["base", "factory"])
def test_a_module_with_no_policy_subclass_is_not_a_spelling(module_name: str) -> None:
    """Control: the auto-discovered set is not "every module in the package".

    Fails if the derivation is widened to module names, which would demand the
    docs name modules ``create_policy`` refuses.
    """
    from strands_robots.registry.policies import import_policy_class

    with pytest.raises(ValueError, match="Unknown policy provider"):
        import_policy_class(module_name)
    assert module_name not in _resolved_outside_the_registry()


def test_a_dotted_path_does_not_answer_for_the_spelling() -> None:
    """Control: the boundary in the token match is load-bearing.

    Naming ``strands_robots.policies.persistent.PersistentPolicy`` tells a
    reader which class to construct; it does not tell them ``persistent`` is a
    spelling the factory resolves. A substring match would treat the two as the
    same statement, so a surface that only linked the class would pass.
    """
    assert not _names_the_spelling("see strands_robots.policies.persistent.PersistentPolicy", "persistent")
    assert _names_the_spelling("the ``persistent`` spelling resolves", "persistent")
    assert not _names_the_spelling("CompositePolicy wraps two policies", "composite")
    assert _names_the_spelling("`composite` builds through the factory", "composite")
