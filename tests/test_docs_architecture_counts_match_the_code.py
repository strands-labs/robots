"""Repo hygiene: the orientation page's counts are the code's counts.

``docs/architecture.md`` is the map a contributor reads to size each subsystem,
and its "ABCs" paragraph is read as the contract a new implementation conforms
to. Both restate the code in prose, and nothing tied most of that prose to the
code, so it drifted: the module table claimed 4 policy providers for a package
shipping 14 and 8 ``@tool`` helpers for 20, while the ``Policy`` paragraph named
four implementations of fifteen and two of the contract's three declaration
seams. README quoted 67 Simulation actions for a published enum of 77 - a number
someone had already corrected once by hand, which is exactly the drift a guard
prevents and prose does not.

One row of that same table *was* graded, by
``test_docs_robot_catalog_coverage.py``, and it is exact. This module extends
that idea to the rest of the page, deriving every number from the code so a
count cannot be right on the day it is written and wrong a month later:

* The module-table counts come from ``registry/policies.json`` and from an AST
  walk of ``strands_robots/tools/``.
* The Simulation action count comes from the published ``tool_spec.json`` enum,
  and is graded wherever either document states one.
* The ``Policy`` paragraph is graded against ``Policy.__abstractmethods__`` (what
  an implementation must supply) and against the *declaration seams* - the
  "policy declares, runtime supplies" family, itself derived from the base
  class's own docstrings rather than listed here.

The seam half is the one with teeth. A policy conforming to the five members the
paragraph used to name receives no body pose at all: the runtime supplies one
only for a body ``required_bodies`` names, so a whole-body tracker that skips it
reads ``base_quat`` - the pelvis, which diverges from ``torso_link`` by tens of
degrees once the waist turns. Every implementation the paragraph named is one for
which that omission is invisible, and every implementation that uses a seam was
absent, so the enumeration was self-consistently incomplete.
"""

from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path

import pytest

from strands_robots.policies.base import Policy

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "strands_robots"
ARCHITECTURE = REPO_ROOT / "docs" / "architecture.md"
README = REPO_ROOT / "README.md"

#: Documents that restate the Simulation action surface as a number.
ACTION_COUNT_SITES = (ARCHITECTURE, README)

#: The phrase the base class uses to mark a member of the declaration family.
_SEAM_PHRASE = "policy declares, runtime supplies"


def _architecture() -> str:
    return ARCHITECTURE.read_text(encoding="utf-8")


def _policy_paragraph() -> str:
    """The ABCs paragraph describing ``Policy``, whitespace-normalised."""
    for block in _architecture().split("\n\n"):
        if block.lstrip().startswith("**`Policy`**"):
            return " ".join(block.split())
    raise AssertionError("docs/architecture.md has no '**`Policy`**' paragraph in its ABCs section")


def _module_table_row(module: str) -> str:
    """The ``| `<module>` | ... |`` row of the Modules table."""
    prefix = f"| `{module}` |"
    for line in _architecture().splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(f"docs/architecture.md has no Modules row for {module!r}")


def _tool_count() -> int:
    """``@tool``-decorated module-level helpers under ``strands_robots/tools/``."""
    total = 0
    for path in sorted((PACKAGE / "tools").glob("*.py")):
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            if any(ast.unparse(d).split("(")[0].strip() == "tool" for d in node.decorator_list):
                total += 1
    return total


def _shipped_providers() -> list[str]:
    """Providers ``registry/policies.json`` ships.

    The shipped registry rather than ``list_providers()``: the row describes what
    the package contains, and ``list_providers()`` also reports providers a caller
    registered at runtime - which in a test session includes every throwaway
    ``register_policy`` a sibling module left behind.
    """
    registry = json.loads((PACKAGE / "registry" / "policies.json").read_text(encoding="utf-8"))
    return sorted(registry["providers"])


def _published_action_count() -> int:
    """Actions the MuJoCo tool schema publishes to a model."""
    spec = json.loads((PACKAGE / "simulation" / "mujoco" / "tool_spec.json").read_text(encoding="utf-8"))
    return len(spec["properties"]["action"]["enum"])


def _policy_implementations() -> frozenset[str]:
    """Concrete ``Policy`` implementations, resolved transitively without importing.

    An AST walk rather than ``issubclass`` so an optional dependency missing
    locally cannot silently shrink the set and let a stale number pass.
    """
    bases: dict[str, set[str]] = {}
    for path in sorted((PACKAGE / "policies").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ClassDef):
                bases[node.name] = {ast.unparse(b).split(".")[-1] for b in node.bases}
    reached = {"Policy"}
    grew = True
    while grew:
        grew = False
        for name, parents in bases.items():
            if name not in reached and parents & reached:
                reached.add(name)
                grew = True
    return frozenset(reached - {"Policy"})


def _member_doc(name: str) -> str:
    attr = getattr(Policy, name)
    target = attr.fget if isinstance(attr, property) else attr
    return inspect.getdoc(target) or ""


def _declaration_seams() -> frozenset[str]:
    """The "policy declares, runtime supplies" members, derived from the base class.

    A member is in the family when its own docstring names the contract, or when
    a member that does cites it with an ``:attr:`` role - which is how the base
    class records that ``requires_images`` is the precedent the later two follow.
    Deriving it means a fourth seam is graded the day it is added.
    """
    public = {name for name in dir(Policy) if not name.startswith("_")}
    marked: set[str] = set()
    cited: set[str] = set()
    for name in sorted(public):
        doc = _member_doc(name)
        if _SEAM_PHRASE.split(" supplies")[0] in doc:
            marked.add(name)
            cited |= set(re.findall(r":attr:`(?:~[\w.]*\.)?(\w+)`", doc))
    return frozenset(marked | (cited & public))


def _names(paragraph: str, member: str) -> bool:
    """Whether ``paragraph`` names ``member`` as a whole token.

    A dotted path names the thing to import, which is a different fact from the
    member a paragraph is telling an implementer about, so the boundary matters.
    """
    return re.search(rf"(?<![\w.]){re.escape(member)}(?![\w.])", paragraph) is not None


def test_the_module_table_states_the_number_of_providers_that_ship() -> None:
    """The policies row quotes ``list_providers()``, not a number from its past."""
    providers = _shipped_providers()
    row = _module_table_row("strands_robots/policies/")
    stated = re.findall(r"\b(\d+)\s+providers\b", row)
    assert stated == [str(len(providers))], (
        f"docs/architecture.md's policies row states {stated or 'no'} providers; "
        f"registry/policies.json ships {len(providers)}: {providers}"
    )


def test_the_module_table_states_the_number_of_tools_that_ship() -> None:
    """The tools row quotes the ``@tool``-decorated helpers that exist."""
    expected = _tool_count()
    row = _module_table_row("strands_robots/tools/")
    stated = re.findall(r"\b(\d+)\s+`@tool`", row)
    assert stated == [str(expected)], (
        f"docs/architecture.md's tools row states {stated or 'no'} @tool helpers; "
        f"strands_robots/tools/ defines {expected}"
    )


@pytest.mark.parametrize("path", ACTION_COUNT_SITES, ids=lambda p: p.name)
def test_a_documented_simulation_action_count_is_the_published_enum(path: Path) -> None:
    """Every action count either document states is the count a model is offered.

    Graded wherever a number qualifies "actions" rather than at fixed lines, so a
    fresh claim added elsewhere in the file is held to the same enum.
    """
    expected = _published_action_count()
    text = path.read_text(encoding="utf-8")
    stated = {int(n) for n in re.findall(r"\b(\d+)\s+(?:\*\*)?(?:Simulation )?actions\b", text)}
    wrong = sorted(n for n in stated if n != expected)
    assert not wrong, (
        f"{path.relative_to(REPO_ROOT)} states {wrong} Simulation actions; tool_spec.json publishes {expected}"
    )


def test_the_policy_paragraph_names_every_abstract_member() -> None:
    """A member an implementation *must* supply is a member the paragraph names."""
    paragraph = _policy_paragraph()
    missing = sorted(m for m in Policy.__abstractmethods__ if not _names(paragraph, m))
    assert not missing, (
        f"docs/architecture.md's Policy paragraph does not name {missing}, which "
        f"Policy declares abstract, so an implementer reading it learns an incomplete "
        f"must-implement set: {paragraph[:160]}..."
    )


def test_the_policy_paragraph_claims_nothing_abstract_that_is_not() -> None:
    """The converse: a member the paragraph presents as abstract really is one."""
    paragraph = _policy_paragraph()
    claimed = set(re.findall(r"`(\w+)\(", paragraph)) | set(re.findall(r"`(\w+)` property", paragraph))
    optional = {name for name in claimed if hasattr(Policy, name) and name not in Policy.__abstractmethods__}
    intro = paragraph.split(":")[0] if ":" in paragraph else paragraph
    wrong = sorted(name for name in optional if _names(intro, name))
    assert not wrong, (
        f"docs/architecture.md's Policy paragraph introduces {wrong} alongside the "
        f"abstract members, but Policy supplies a default for each: "
        f"{sorted(Policy.__abstractmethods__)} are the abstract ones"
    )


def test_the_policy_paragraph_names_every_declaration_seam() -> None:
    """A seam the runtime reads is a seam the contract paragraph names.

    ``requires_images`` alone was named. ``required_bodies`` is the only route to
    a body pose, and ``children`` the only route into a wrapped policy, so a
    paragraph naming one of the three teaches a contract whose two silent halves
    are the ones a policy cannot recover from by trying harder.
    """
    seams = _declaration_seams()
    paragraph = _policy_paragraph()
    missing = sorted(s for s in seams if not _names(paragraph, s))
    assert not missing, (
        f"docs/architecture.md's Policy paragraph does not name {missing}; the base "
        f"class marks {sorted(seams)} as the 'policy declares, runtime supplies' "
        f"family, and a policy that skips one is not told it exists"
    )


def test_the_policy_paragraph_states_the_number_of_implementations_that_ship() -> None:
    """An implementation count in the paragraph is the number of implementations."""
    expected = len(_policy_implementations())
    paragraph = _policy_paragraph()
    stated = {int(n) for n in re.findall(r"\b(\d+)\s+implementations\b", paragraph)}
    wrong = sorted(n for n in stated if n != expected)
    assert not wrong, (
        f"docs/architecture.md's Policy paragraph states {wrong} implementations; "
        f"strands_robots/policies/ defines {expected}"
    )


def test_the_paragraph_does_not_enumerate_a_strict_subset_as_the_implementations() -> None:
    """Naming implementations is fine; naming some of them as *the* set is not.

    The paragraph previously read "Four implementations: MockPolicy, Gr00tPolicy,
    LerobotLocalPolicy, Cosmos3Policy" - a closed list of four of fifteen, and
    the four for which the omitted seams happen to be invisible.
    """
    paragraph = _policy_paragraph()
    implementations = _policy_implementations()
    named = {n for n in re.findall(r"`(\w+)`", paragraph) if n in implementations}
    total = len(implementations)
    assert not (0 < len(named) < total), (
        f"docs/architecture.md's Policy paragraph names {sorted(named)} - "
        f"{len(named)} of {total} implementations - which reads as the complete set. "
        f"State the count and point at the provider table instead."
    )


def test_the_seam_family_is_derived_from_the_base_class() -> None:
    """Non-vacuity: the derived family is real members, and it spans all three.

    Without this a refactor that stopped matching the base class's phrasing would
    make the seam guard grade an empty set and report a clean page.
    """
    seams = _declaration_seams()
    assert seams == {"requires_images", "required_bodies", "children"}, (
        f"derived seam family is {sorted(seams)}; expected the three the base class "
        f"marks. Update this pin deliberately if a seam is added or removed."
    )
    for name in seams:
        assert hasattr(Policy, name), f"{name} is not a Policy member"


@pytest.mark.parametrize(
    ("paragraph", "expected_missing"),
    [
        ("**`Policy`** - `get_actions()`, `set_robot_state_keys()`, `provider_name` property.", "children"),
        ("**`Policy`** - `requires_images`, `required_bodies`, `children`.", "get_actions"),
    ],
    ids=["seam-omitted", "abstract-omitted"],
)
def test_the_graders_report_a_planted_omission(paragraph: str, expected_missing: str) -> None:
    """The rules fire on prose that omits a member, not merely on the real page."""
    seams = _declaration_seams()
    wanted = set(Policy.__abstractmethods__) | set(seams)
    missing = sorted(m for m in wanted if not _names(paragraph, m))
    assert expected_missing in missing, f"planted omission of {expected_missing} not reported: {missing}"
