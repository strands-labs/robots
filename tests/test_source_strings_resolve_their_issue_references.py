"""Regression: no caller-reachable string may cite an issue in a repository the
reader cannot resolve.

A refusal envelope, a log line or a help string is read by an operator, and the
tracker reference it carries is a *remedy*: the reader follows it to find out
what is missing and when it lands. Two spellings resolve for that reader - a
bare ``#2765`` (an issue in this repository, which every tracker and IDE turns
into a link) and an owner-qualified ``strands-labs/robots-sim#46`` (a complete
coordinate). A bare ``<slug>#361`` resolves to nothing: it names a repository
without an owner, so the reader has no path to open and the message advertises a
remedy that cannot be followed.

The scan walks every module under the ``strands_robots`` package, parses it, and
inspects only string constants a *caller* can receive - runtime literals, with
module, class and function docstrings excluded. That boundary is deliberate and
narrow. Developer-facing prose legitimately cites the sibling repositories this
package grew out of (over twenty ``robots-sim#NN`` references sit in
``simulation/isaac`` comments and docstrings, and ``strands-labs/robots-sim`` is
a real repository), and a maintainer reading a docstring has the git history and
the sibling checkouts to hand. An operator holding a refusal envelope has
neither, so the rule applies where the text reaches them.

:mod:`test_test_module_names_do_not_spell_a_tracker_coordinate` applies this
module's :func:`_unresolvable_references` to the *other* surface a coordinate
gets written on -- a test module's own name -- so the two guards cannot drift to
two definitions of "resolvable".

It would have failed while ``G1Driver._check_motion_gates`` refused every write
with "FSM id unknown - motion-switcher source has not been wired (harness#361
PR-C); see #2765 for the wire-side decision". That refusal is the one every real
driver hits - ``_fsm_id`` has a single assignment, its ``None`` initialiser - so
the unresolvable half was the most-read tracker reference in the package, sitting
beside a bare ``#2765`` that already carried the same information.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import strands_robots

# A tracker reference qualified by a repository name: the slug, ``#``, a number.
# A bare ``#2765`` does not match (no leading slug) and needs no owner - it is
# an issue in this repository. A slug carrying ``/`` is owner-qualified and
# resolves on its own; a slug without one names a repository with no owner.
_QUALIFIED_ISSUE_REF = re.compile(r"([A-Za-z][A-Za-z0-9_.\-/]*)#(\d+)")

_PACKAGE_DIR = Path(strands_robots.__file__).resolve().parent


def _python_sources() -> list[Path]:
    return sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _docstring_constant_ids(tree: ast.Module) -> set[int]:
    """Identify the docstring node of every module, class and function."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            ids.add(id(first.value))
    return ids


def _caller_reachable_literals(path: Path) -> list[tuple[int, str]]:
    """Every string constant a caller can receive, paired with its line number.

    Docstrings are excluded: they are read by a maintainer with the repository
    checked out, not returned to a caller.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_constant_ids(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings
    ]


def _unresolvable_references(text: str) -> list[str]:
    """Tracker references in ``text`` that name a repository with no owner."""
    return [f"{slug}#{number}" for slug, number in _QUALIFIED_ISSUE_REF.findall(text) if "/" not in slug]


def test_package_sources_discovered() -> None:
    """Guard: the scan walked the whole package, not one subtree."""
    sources = _python_sources()
    assert len(sources) > 50
    rel_dirs = {p.relative_to(_PACKAGE_DIR).parts[0] for p in sources if p.parent != _PACKAGE_DIR}
    assert {"drivers", "simulation", "tools", "policies"} <= rel_dirs


def test_caller_reachable_literals_discovered() -> None:
    """Guard: excluding docstrings did not empty the population being scanned."""
    total = sum(len(_caller_reachable_literals(path)) for path in _python_sources())
    assert total > 1000, f"only {total} caller-reachable literals found; the extraction is too narrow"


def test_no_caller_reachable_string_cites_an_unresolvable_repository() -> None:
    """A string a caller receives must not cite an issue in an unowned repository."""
    offenders: list[str] = []
    for path in _python_sources():
        for lineno, literal in _caller_reachable_literals(path):
            for reference in _unresolvable_references(literal):
                offenders.append(f"{path.relative_to(_PACKAGE_DIR.parent)}:{lineno}: {reference}")
    assert not offenders, (
        "A string a caller can receive cites an issue in a repository the reader "
        "cannot resolve. Use a bare '#2765' for an issue in this repository, or "
        "an owner-qualified 'strands-labs/robots-sim#46' for a sibling - an "
        "unowned slug advertises a remedy the reader has no path to open:\n" + "\n".join(offenders)
    )


def test_an_unowned_slug_reference_is_flagged() -> None:
    """The predicate flags the shape this guard exists to keep out."""
    refused = "FSM id unknown - motion-switcher source has not been wired (harness#361 PR-C)"
    assert _unresolvable_references(refused) == ["harness#361"]


def test_bare_and_owner_qualified_references_are_not_flagged() -> None:
    """Both resolvable spellings must pass, or the rule is blanket strictness."""
    accepted = [
        "see issue #2765 for the wire-side decision",  # bare: this repository
        "tracked in strands-labs/robots-sim#46",  # owner-qualified sibling repository
        "see strands-labs/robots#708 for the root-cause analysis",
        "see strands-labs/robots-sim#12 while that lands",
    ]
    for text in accepted:
        assert _unresolvable_references(text) == [], f"should not be flagged: {text!r}"


def test_the_predicate_separates_the_two_shapes() -> None:
    """Non-vacuity: the predicate answers both ways over the exemplars."""
    outcomes = {
        bool(_unresolvable_references(text))
        for text in ("harness#361 PR-C", "issue #2765", "strands-labs/robots-sim#46")
    }
    assert outcomes == {True, False}


def test_developer_facing_docstrings_are_deliberately_out_of_scope() -> None:
    """The scope boundary is a measurement, not an oversight.

    Docstrings and comments in this package do carry unowned sibling-repository
    references today. They are excluded because a maintainer reading them has
    the sibling checkout; an operator holding a refusal envelope does not. If
    that population ever empties, widening the scan becomes a free tightening.
    """
    in_docstrings = 0
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = _docstring_constant_ids(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) in docstrings:
                in_docstrings += len(_unresolvable_references(node.value))
    assert in_docstrings > 0, (
        "no docstring cites an unowned sibling repository any more; the scan can "
        "be widened to docstrings and this boundary test removed"
    )
