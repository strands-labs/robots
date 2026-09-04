"""Guard against phantom policy providers advertised to agents and users.

An agent-facing tool schema enumerates example policy provider names to guide
the model driving the tool, and carries a ``default`` for the parameter. The
model reads that schema and nothing else, so a name it advertises which is not
actually registered is not a documentation nit -- it is a provider the model
will pass. What happens then depends on which resolver the surface reaches, and
neither outcome names the schema that supplied the name:

* ``create_policy`` raises ``Unknown policy provider``, and on the hardware
  path that happens inside ``Robot._get_policy`` on the executor thread, after
  ``start_task`` has already answered ``Task started`` and the arm has
  connected.
* ``resolve_policy`` redirects any unrecognised string to ``lerobot_local``
  with the string as ``pretrained_name_or_path``, so the name resolves -- to
  the wrong provider, pointed at a checkpoint repository that does not exist.

The refusal for an unregistered provider is also not distinguishable from the
refusal for a registered one. ``Robot._get_policy`` reads the registry's
``requires`` field to decide whether a port is owed and falls back to demanding
one when the provider is unknown, so a schema-advertised phantom name passed
without a port is refused for the *port* -- byte-identical to the same call
naming a real port-dialing provider. Nothing on that path says the provider
does not exist.

**The population is derived from the tree, not listed here.** These tests used
to read one file, ``simulation/mujoco/tool_spec.json``, which is the surface
that prompted them. Measured over the package there are two ``policy_provider``
schema entries, and the second one -- ``Robot.tool_spec``, the schema every
hardware robot presents -- advertised ``openai``, which resolves to
``lerobot_local`` rather than to itself. A guard for this class that names its
own subject cannot see a second surface arrive, so the sweep walks the package
and grades whatever it finds: a schema entry added later is graded on arrival.

Two spellings of the same literal are graded, because a provider name reaches a
caller by two routes:

* a ``policy_provider`` entry in a tool ``inputSchema`` -- its ``e.g.``
  examples and its ``default`` -- which is what an agent reads;
* a ``policy_provider=`` default in a function signature, which is what a
  Python caller gets when the argument is omitted.

The second is the gap #3075 names: nothing links a registry entry to the
defaults that name it, so a provider rename or removal leaves those defaults
naming nothing and the loss is invisible until a caller omits the argument.
``None`` is skipped there -- it is the not-supplied sentinel, not a provider.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
import re
from typing import Any

import pytest

from strands_robots.policies import list_aliases, list_providers
from strands_robots.registry.policies import resolve_policy

# Derived from the defining module of an imported symbol rather than from a
# path literal, which is what lets scripts/check_whole_tree_graders.py resolve
# the walk below and roster this module without a second edit.
_PACKAGE_ROOT = pathlib.Path(inspect.getfile(resolve_policy)).parents[1]

# The two surfaces the sweep is known to reach. Asserted rather than iterated:
# a resolver or layout change that leaves the walk finding nothing would
# otherwise pass every cell below vacuously.
_KNOWN_SCHEMA_FILES = {
    "hardware_robot.py",
    "tool_spec.json",
}


def _registered_spellings() -> set[str]:
    """Every provider spelling either registry holds, canonical or alias."""
    return set(list_providers()) | set(list_aliases())


def _eg_names(text: str) -> list[str]:
    """Extract provider tokens from an ``e.g. a, b, c`` clause in ``text``."""
    match = re.search(r"e\.g\.\s*([^).]*)", text)
    if not match:
        return []
    return [tok.strip() for tok in re.split(r"[,/]", match.group(1)) if tok.strip()]


def _is_schema_entry(value: Any) -> bool:
    """Is ``value`` a JSON-schema property rather than a config mapping?

    A property declares a ``type``. Keying on that keeps a ``policy_provider``
    key that merely holds a configured value out of the population, which is
    what would otherwise make the sweep grade a caller's argument as though it
    were an advertisement.
    """
    return isinstance(value, dict) and "type" in value


def _py_schema_entries(path: pathlib.Path) -> list[tuple[str, dict[str, Any]]]:
    """Every ``{"policy_provider": {...}}`` schema literal in a Python file."""
    found: list[tuple[str, dict[str, Any]]] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and key.value == "policy_provider"):
                continue
            if not isinstance(value, ast.Dict):
                continue
            entry: dict[str, Any] = {}
            for inner_key, inner_value in zip(value.keys, value.values):
                if not (isinstance(inner_key, ast.Constant) and isinstance(inner_key.value, str)):
                    continue
                if isinstance(inner_value, ast.Constant):
                    entry[inner_key.value] = inner_value.value
            if _is_schema_entry(entry):
                found.append((f"{path.name}:{value.lineno}", entry))
    return found


def _json_schema_entries(path: pathlib.Path) -> list[tuple[str, dict[str, Any]]]:
    """Every ``"policy_provider"`` schema property in a JSON file."""
    found: list[tuple[str, dict[str, Any]]] = []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return found
    pending: list[Any] = [doc]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if key == "policy_provider" and _is_schema_entry(value):
                    found.append((path.name, value))
                pending.append(value)
        elif isinstance(current, list):
            pending.extend(current)
    return found


def _provider_schema_entries() -> list[tuple[str, dict[str, Any]]]:
    """Every agent-facing ``policy_provider`` schema entry in the package."""
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        found.extend(_py_schema_entries(path))
    for path in sorted(_PACKAGE_ROOT.rglob("*.json")):
        found.extend(_json_schema_entries(path))
    return found


def _signature_defaults() -> list[tuple[str, str]]:
    """Every ``policy_provider=`` default in the package, as (site, name).

    ``None`` defaults are skipped: that is the not-supplied sentinel the
    port-requiring entry points use, not a provider name.
    """
    found: list[tuple[str, str]] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            args = node.args
            positional = args.posonlyargs + args.args
            padding: list[ast.expr | None] = [None] * (len(positional) - len(args.defaults))
            pairs = list(zip(positional, padding + list(args.defaults)))
            pairs += list(zip(args.kwonlyargs, args.kw_defaults))
            for arg, default in pairs:
                if arg.arg != "policy_provider" or default is None:
                    continue
                if isinstance(default, ast.Constant) and isinstance(default.value, str):
                    found.append((f"{path.name}:{node.lineno} {node.name}", default.value))
    return found


_SCHEMA_ENTRIES = _provider_schema_entries()
_SIGNATURE_DEFAULTS = _signature_defaults()

_ADVERTISED = [(site, name) for site, entry in _SCHEMA_ENTRIES for name in _eg_names(str(entry.get("description", "")))]
_SCHEMA_DEFAULTS = [
    (site, entry["default"]) for site, entry in _SCHEMA_ENTRIES if isinstance(entry.get("default"), str)
]


def test_the_sweep_reaches_every_known_provider_schema() -> None:
    """The walk must find both known surfaces, so nothing below passes vacuously."""
    files = {site.split(":")[0] for site, _ in _SCHEMA_ENTRIES}
    assert _KNOWN_SCHEMA_FILES <= files, (
        f"the policy_provider schema sweep found {sorted(files)}; "
        f"expected at least {sorted(_KNOWN_SCHEMA_FILES)}. A surface moved or the walk broke - "
        "re-read the discovery helpers rather than trimming this expectation."
    )


def test_every_provider_schema_advertises_examples_in_a_readable_form() -> None:
    """Each schema must list its examples as ``e.g. a, b, c``.

    The form is what makes the examples gradable at all. ``Robot.tool_spec``
    advertised ``"Policy provider (groot, openai, etc.)"``, which names two
    providers and matches no ``e.g.`` clause, so the phantom one was invisible
    to a guard that reads the clause.
    """
    silent = [site for site, entry in _SCHEMA_ENTRIES if not _eg_names(str(entry.get("description", "")))]
    assert not silent, (
        f"policy_provider schemas at {silent} state no 'e.g. ...' provider examples, "
        "so the names they advertise cannot be graded against the registry"
    )


@pytest.mark.parametrize(("site", "name"), _ADVERTISED)
def test_every_advertised_provider_example_is_registered(site: str, name: str) -> None:
    """Every provider a schema names as an example must be registered."""
    registered = _registered_spellings()
    assert name in registered, f"{site} advertises unregistered provider {name!r}; registered={sorted(registered)}"


@pytest.mark.parametrize(("site", "name"), _ADVERTISED)
def test_every_advertised_provider_example_resolves_to_itself(site: str, name: str) -> None:
    """An advertised name must resolve to its own provider, not the fallback.

    ``resolve_policy`` redirects an unrecognised string to ``lerobot_local``,
    so a phantom name "resolves" and misdirects rather than refusing. Asserting
    the resolved provider equals the advertised name is what separates the two.
    """
    provider, _ = resolve_policy(name)
    assert provider == name, f"{site} advertises {name!r}, which resolves to {provider!r} (phantom/misdirecting)"


@pytest.mark.parametrize(("site", "name"), _SCHEMA_DEFAULTS)
def test_every_provider_schema_default_resolves_to_itself(site: str, name: str) -> None:
    """A schema ``default`` is what the model sends when it says nothing.

    It is written by hand beside the signature default it mirrors rather than
    derived from it, so the two go stale independently: a Python caller reads
    the signature and an agent reads the schema.
    """
    registered = _registered_spellings()
    assert name in registered, f"{site} defaults to unregistered provider {name!r}; registered={sorted(registered)}"
    provider, _ = resolve_policy(name)
    assert provider == name, f"{site} defaults to {name!r}, which resolves to {provider!r}"


@pytest.mark.parametrize(("site", "name"), _SIGNATURE_DEFAULTS)
def test_every_policy_provider_signature_default_resolves_to_itself(site: str, name: str) -> None:
    """A ``policy_provider=`` default must name a registered provider.

    This is the half #3075 names: the defaults are reached by exactly the calls
    that pass the fewest arguments, so a provider rename or removal leaves them
    naming nothing and the loss stays invisible until a caller omits the
    argument. Deriving the population means a provider-dialing entry point
    added later is held to the rule on arrival.
    """
    registered = _registered_spellings()
    assert name in registered, f"{site} defaults to unregistered provider {name!r}; registered={sorted(registered)}"
    provider, _ = resolve_policy(name)
    assert provider == name, f"{site} defaults to {name!r}, which resolves to {provider!r}"


def test_the_signature_default_sweep_reaches_the_entry_points() -> None:
    """The signature sweep must find defaults, so the cells above are not empty."""
    assert len(_SIGNATURE_DEFAULTS) >= 20, (
        f"the policy_provider signature sweep found {len(_SIGNATURE_DEFAULTS)} defaults; "
        "the entry points that carry one are the driver start_task family and the simulation "
        "rollout helpers, so a count this low means the walk broke"
    )


def test_resolve_policy_docstring_examples_reference_registered_providers() -> None:
    """The ``# -> ("name", ...)`` examples in resolve_policy's docstring must be real."""
    doc = resolve_policy.__doc__ or ""
    names = re.findall(r'#\s*\u2192\s*\("([^"]+)"', doc)
    assert names, "no docstring resolution examples found in resolve_policy"
    registered = _registered_spellings()
    unknown = [name for name in names if name not in registered]
    assert not unknown, f"resolve_policy docstring references unregistered providers {unknown}"
