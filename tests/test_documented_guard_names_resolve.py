"""Repo hygiene: a guard named in prose is named by the spelling it was defined with.

History: ``strands_robots/utils.py`` defines the traversal guard as ``safe_join``
and 13 call sites across ``assets/manager.py``, ``assets/download.py``,
``simulation/task_objects/catalog.py`` and ``tools/harness_memory.py`` spell it
that way. Two pieces of prose spelled it ``_safe_join`` instead - the
``.. warning:: Security`` block on :func:`strands_robots.registry.register_robot`,
which tells a future implementer to "validate all paths with" it, and the
registry-conventions section of ``AGENTS.md``, which attributed it to the right
file under the wrong name.

The cost is measurable rather than theoretical. A reader who greps the documented
spelling finds no implementation - the only ``_safe_join`` in the tree was a
synthetic stub inside a test fixture string - and concludes the remediation does
not exist. That conclusion was written down: a repository audit reported
``_safe_join`` as "a phantom helper" and "no ``_safe_join`` implementation
anywhere in ``strands_robots/``", on the strength of a grep for the documented
name, while the guard was present and tested the whole time. A security warning
that names a symbol nobody can find reads as though the remediation were still
unwritten, which is the same failure shape as a caller-less table: prose that
looks like it is enforcing something.

The rule below is derived from the tree on both sides, so it needs no allowlist:

* The graded set is every **public** callable ``utils.py`` defines - 39 of them -
  rather than a list of names someone has to remember to extend.
* The exemption is also derived. ``_coerce_rgba`` is a real, distinct private
  function in ``simulation/mujoco/physics.py`` that wraps the public
  ``coerce_rgba``, so the underscored spelling is correct there. Any
  underscore-prefixed name that is genuinely defined somewhere in the tree is
  therefore allowed; only a spelling that resolves to nothing is drift.

That second point is what keeps this from being a blacklist: the sweep asks
whether the name a reader would grep for exists, not whether a particular string
appears.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Where a reader looks for a guard's name: source, tests, docs, and the two
#: root markdown files that carry project rules.
SCAN_DIRS = ("strands_robots", "tests", "tests_integ", "docs", "scripts")
SCAN_ROOT_FILES = ("AGENTS.md", "README.md")

#: Symbol-defining directories. The exemption set is derived from these, so a
#: private helper only excuses an underscored spelling if it is really there.
DEFINITION_DIRS = ("strands_robots", "tests", "tests_integ", "scripts")

SKIP_PARTS = {"__pycache__", ".venv", "node_modules", "build", "dist"}


def _iter_files(suffixes: frozenset[str]) -> list[Path]:
    files: list[Path] = []
    for name in SCAN_ROOT_FILES:
        candidate = REPO_ROOT / name
        if candidate.is_file() and candidate.suffix in suffixes:
            files.append(candidate)
    for directory in SCAN_DIRS:
        root = REPO_ROOT / directory
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.suffix not in suffixes or not path.is_file():
                continue
            if SKIP_PARTS & set(path.parts):
                continue
            files.append(path)
    return files


@functools.cache
def _public_utils_guards() -> tuple[str, ...]:
    """Every public callable ``utils.py`` defines, derived from its AST."""
    tree = ast.parse((REPO_ROOT / "strands_robots" / "utils.py").read_text(encoding="utf-8"))
    return tuple(
        sorted(
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) and not node.name.startswith("_")
        )
    )


@functools.cache
def _defined_names() -> frozenset[str]:
    """Every name bound at any scope across the tree's Python sources.

    Deliberately generous: an import alias, a class attribute and a nested
    function all count. A name that is bound *anywhere* is a name a reader can
    find, which is the property this sweep is about.

    Cached because the tree does not change during a session and seven cells
    ask the same question of it: the walk parses every Python file under the
    definition directories, and what is kept is the set of names, not the
    parsed trees.
    """
    names: set[str] = set()
    for path in _iter_files(frozenset({".py"})):
        if REPO_ROOT / "strands_robots" not in path.parents and not any(
            (REPO_ROOT / d) in path.parents or (REPO_ROOT / d) == path.parent for d in DEFINITION_DIRS
        ):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
    return frozenset(names)


def _unresolvable_private_spellings(guards: tuple[str, ...], defined: frozenset[str]) -> list[str]:
    """Underscored spellings of public guards that resolve to nothing."""
    return [f"_{guard}" for guard in guards if f"_{guard}" not in defined]


def _find(spelling: str, text: str) -> list[int]:
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(spelling) + r"(?![A-Za-z0-9_])")
    return [text[: match.start()].count("\n") + 1 for match in pattern.finditer(text)]


class TestAGuardIsNamedBySpellingThatResolves:
    """A public ``utils.py`` guard is never referred to by an underscored name.

    The graded set is the module's own public surface, so a guard added later is
    held to the rule without anyone editing this file.
    """

    def test_the_graded_set_is_the_public_utils_surface(self) -> None:
        """Guard the sweep's own reach: an empty graded set would pass vacuously."""
        guards = _public_utils_guards()
        assert "safe_join" in guards, "the traversal guard must be in the graded set"
        assert len(guards) > 20, f"expected the full public surface, graded {len(guards)}"
        assert all(not g.startswith("_") for g in guards)

    def test_the_exemption_is_derived_and_not_a_list(self) -> None:
        """``_coerce_rgba`` is a real private wrapper, so it must not be flagged.

        This is the assertion that keeps the rule from being a blacklist. If the
        exemption stopped being derived from the tree, this case is the one that
        would start failing.
        """
        defined = _defined_names()
        assert "_coerce_rgba" in defined, "the private rgba wrapper is a real symbol"
        assert "coerce_rgba" in _public_utils_guards()
        assert "_coerce_rgba" not in _unresolvable_private_spellings(_public_utils_guards(), defined)

    def test_no_prose_names_a_guard_by_an_unresolvable_private_spelling(self) -> None:
        guards = _public_utils_guards()
        unresolvable = _unresolvable_private_spellings(guards, _defined_names())
        assert unresolvable, "expected at least one guard with no private twin to grade"

        offences: list[str] = []
        for path in _iter_files(frozenset({".py", ".md"})):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            for spelling in unresolvable:
                if spelling not in text:
                    continue
                for line in _find(spelling, text):
                    offences.append(
                        f"{path.relative_to(REPO_ROOT)}:{line} names {spelling}, defined nowhere; use {spelling[1:]}"
                    )

        assert not offences, "a guard is referred to by a name that resolves to nothing:\n" + "\n".join(
            sorted(offences)
        )


class TestTheSweepDetectsTheDriftItWasWrittenFor:
    """Shown to reach: the rule fires on the pre-fix spelling and not on a real one."""

    @pytest.mark.parametrize(
        ("prose", "should_flag"),
        [
            ("validate all paths with _safe_join.", True),
            ("validate all paths with safe_join.", False),
            ("The `_safe_join` helper in `strands_robots/utils.py` guards traversal.", True),
            ("The `safe_join` helper in `strands_robots/utils.py` guards traversal.", False),
            ("colour goes through _coerce_rgba first.", False),
        ],
    )
    def test_the_rule_grades_prose_as_expected(self, prose: str, should_flag: bool) -> None:
        unresolvable = _unresolvable_private_spellings(_public_utils_guards(), _defined_names())
        flagged = any(_find(spelling, prose) for spelling in unresolvable)
        assert flagged is should_flag
