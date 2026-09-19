"""A ``del`` of parameters must precede the body it narrows scope for.

``del <unused parameters>`` is this tree's idiom for a signature a caller
supplies and the implementation ignores - a Strands agent-tool ``stream``, a
``HardwareDriver`` verb a platform does not implement, a ``Policy.reset`` that
takes no seed. Twenty sites use it, in ``strands_robots/``, ``tests/`` and
``examples/``.

The idiom has a property nothing was checking: the ``del`` has to come *before*
the statements it narrows scope for. As the last statement of a function it
narrows nothing, because the frame is discarded on return either way - so it is
a genuine no-op statement rather than a scope-narrowing one, and CodeQL's
``py/unnecessary-delete`` reports it.

That report is why this file is a merge concern rather than a style preference.
``.github/codeql/codeql-config.yml`` records that a CodeQL alert is a hard gate
here - ``github-advanced-security`` opens a review thread per new alert and the
``default`` ruleset sets ``required_review_thread_resolution: true`` - and it
also states that the no-op-statement capability given up by excluding
``py/ineffectual-statement`` "is NOT given up -- it moves to ruff, which is
merge-blocking here where CodeQL is advisory". For this shape only the first
half holds. Measured against ruff 0.15: a terminal parameter ``del`` is
reported by no ruff rule, under the repo's ``select`` list *or* under
``--select ALL``. ``B015`` covers a useless comparison and ``B018`` a useless
expression; a ``del`` is neither statement form. So the class reaches the merge
gate without the local ``ruff``/``mypy``/``pytest`` gate having anything to say
about it, and the author learns about it from a review thread after a push.

This file is that missing local half. The rule is deliberately the direct-body
form - the shape the idiom actually takes, a docstring followed by the ``del``.
A ``del`` nested inside a branch is a different construct and is left alone;
:class:`TestTheRuleIsGradedOnConstructedShapes` pins that boundary rather than
leaving it to be inferred.

The tree satisfies the rule today, so the scan finds nothing to report and
cannot exercise its own failing branch. Two things stand in for that: the
non-vacuity class asserts the scan really sees the twenty sites it grades, and
the constructed-shape class drives the same predicate over exemplars written
here, so the rule is graded rather than merely satisfied.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path
from typing import NamedTuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Areas that must be reached however the tree grows. A directory added later
# that ships Python is picked up by _scanned_areas() without an edit here; this
# set only refuses the reverse, an area silently dropping out of the scan.
_REQUIRED_AREAS = frozenset({"strands_robots", "tests", "examples"})


class ParameterDelete(NamedTuple):
    """One ``del`` statement whose targets are all parameters of its function.

    Attributes:
        path: Repository-relative path of the module holding the statement.
        lineno: 1-based line of the ``del``.
        function: Name of the enclosing function.
        names: The parameter names the statement deletes.
        terminal: Whether the ``del`` is the last statement of the enclosing
            function's own body, and so narrows scope for nothing.
    """

    path: str
    lineno: int
    function: str
    names: tuple[str, ...]
    terminal: bool

    def __str__(self) -> str:
        """Return a locator a reader can open, plus the names deleted."""
        return f"{self.path}:{self.lineno} in {self.function}() deletes {', '.join(self.names)}"


def _parameter_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return every name bound by ``function``'s signature.

    Args:
        function: The function whose parameters to collect.

    Returns:
        Positional, positional-only, keyword-only, ``*args`` and ``**kwargs``
        names, which together are exactly the locals a caller supplies.
    """
    args = function.args
    names = {arg.arg for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs)}
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            names.add(extra.arg)
    return names


def parameter_deletes(source: str, path: str) -> list[ParameterDelete]:
    """Find every ``del`` of parameters in ``source``.

    A statement qualifies only when *all* of its targets are plain names bound
    by the enclosing signature. That excludes the unrelated construct of
    dropping a reference to a local to make an object collectable - the
    ``del engine`` and ``del renderer`` probes under ``tests/simulation/`` -
    which is load-bearing where this idiom is decorative.

    Args:
        source: Python source text.
        path: Repository-relative path, used to build the locator.

    Returns:
        One :class:`ParameterDelete` per qualifying statement, in source order.
    """
    tree = ast.parse(source)
    found: list[ParameterDelete] = []
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        params = _parameter_names(function)
        body_ids = {id(statement) for statement in function.body}
        for statement in ast.walk(function):
            if not isinstance(statement, ast.Delete):
                continue
            names = tuple(target.id for target in statement.targets if isinstance(target, ast.Name))
            if len(names) != len(statement.targets) or not names or not set(names) <= params:
                continue
            terminal = bool(function.body) and id(function.body[-1]) == id(statement) and id(statement) in body_ids
            found.append(ParameterDelete(path, statement.lineno, function.name, names, terminal))
    return sorted(found, key=lambda entry: (entry.path, entry.lineno))


def _scanned_areas() -> tuple[str, ...]:
    """Return every top-level directory of the repository that ships Python.

    Deriving the list means a directory added later is graded on arrival. A
    hardcoded tuple equal to today's set would fire on nothing when the tree
    grows, which is the silent hole this avoids.

    Returns:
        Directory names, sorted, excluding dot-directories.
    """
    areas = []
    for entry in sorted(_REPO_ROOT.iterdir()):
        if entry.is_dir() and not entry.name.startswith(".") and any(entry.rglob("*.py")):
            areas.append(entry.name)
    return tuple(areas)


@functools.cache
def _scan_tree() -> tuple[ParameterDelete, ...]:
    """Return every parameter ``del`` in the repository.

    Cached: the tree does not change during a session, and both cells that
    read it want the same walk.

    Returns:
        One entry per qualifying statement across :func:`_scanned_areas`.
    """
    found: list[ParameterDelete] = []
    for area in _scanned_areas():
        for module in sorted((_REPO_ROOT / area).rglob("*.py")):
            try:
                source = module.read_text(encoding="utf-8")
            except OSError:  # pragma: no cover - unreadable file in a work tree
                continue
            try:
                found.extend(parameter_deletes(source, str(module.relative_to(_REPO_ROOT))))
            except SyntaxError:  # pragma: no cover - a fixture of deliberately bad source
                continue
    return tuple(found)


class TestNoParameterDeleteIsTerminal:
    """The rule, over the whole tree."""

    def test_the_tree_carries_no_terminal_parameter_delete(self) -> None:
        offenders = [entry for entry in _scan_tree() if entry.terminal]
        assert offenders == [], (
            "a parameter del as a function's last statement narrows scope for nothing "
            "and CodeQL reports it as py/unnecessary-delete, which opens a merge-blocking "
            "review thread; move it above the body it narrows, or drop it and leave the "
            "docstring as the body: " + "; ".join(str(entry) for entry in offenders)
        )


class TestTheScanSeesTheIdiomItGrades:
    """The tree satisfies the rule, so the scan must be shown to be looking."""

    def test_the_tree_uses_the_idiom_the_rule_is_about(self) -> None:
        found = _scan_tree()
        assert len(found) >= 10, f"the scan found {len(found)} parameter deletes; it has stopped matching the idiom"
        assert len({entry.path for entry in found}) >= 3, f"only {len({e.path for e in found})} file(s) matched"
        areas = {entry.path.split("/", 1)[0] for entry in found}
        assert len(areas) >= 2, f"the idiom was found in one area only ({areas}), so the walk may be truncated"

    def test_every_area_that_ships_python_is_scanned(self) -> None:
        reached = set(_scanned_areas())
        assert _REQUIRED_AREAS <= reached, f"these areas ship Python and are not scanned: {_REQUIRED_AREAS - reached}"
        for area in reached:
            assert any((_REPO_ROOT / area).rglob("*.py")), f"{area} was scanned but ships no Python"


# (source, flagged, why) - the rule and its boundary, written here because the
# tree contains no violation for the scan to grade.
_SHAPES: tuple[tuple[str, bool, str], ...] = (
    (
        'def f(a, b):\n    """Ignore the signature."""\n    del a, b\n',
        True,
        "terminal-parameter-del",
    ),
    (
        'def f(a, b):\n    """Narrow scope, then work."""\n    del a, b\n    return 1\n',
        False,
        "parameter-del-followed-by-a-body",
    ),
    (
        "class C:\n    async def start(self, on_joints, on_imu):\n"
        '        """Accept the callbacks."""\n        del on_joints, on_imu\n',
        True,
        "terminal-parameter-del-in-an-async-method",
    ),
    (
        'def f(*args, **kwargs):\n    """Ignore both packs."""\n    del args, kwargs\n',
        True,
        "terminal-del-of-the-vararg-and-kwarg-packs",
    ),
    (
        'def f():\n    """Drop a reference so the object is collectable."""\n    engine = object()\n    del engine\n',
        False,
        "terminal-del-of-a-local-is-a-refcount-probe",
    ),
    (
        'def f(a):\n    """Delete a parameter and a local together."""\n    local = 1\n    del a, local\n',
        False,
        "terminal-del-mixing-a-parameter-and-a-local",
    ),
    (
        'def f(a):\n    """Delete inside a branch."""\n    if a:\n        del a\n',
        False,
        "parameter-del-nested-in-a-branch-is-out-of-scope",
    ),
    (
        'class C:\n    def f(self):\n        """Delete an attribute."""\n        del self.cached\n',
        False,
        "terminal-del-of-an-attribute-is-a-real-deletion",
    ),
    (
        'class C:\n    def f(self, a):\n        """Delete a parameter and an attribute together."""\n'
        "        del a, self.cached\n",
        False,
        "terminal-del-mixing-a-parameter-and-an-attribute",
    ),
    (
        'def f(mapping, key):\n    """Delete an entry."""\n    del mapping[key]\n',
        False,
        "terminal-del-of-a-subscript-is-a-real-deletion",
    ),
)


class TestTheRuleIsGradedOnConstructedShapes:
    """Drive the predicate the tree-wide rule uses over shapes written here."""

    @pytest.mark.parametrize(("source", "flagged", "why"), _SHAPES, ids=[shape[2] for shape in _SHAPES])
    def test_a_constructed_shape_is_judged_as_documented(self, source: str, flagged: bool, why: str) -> None:
        terminal = [entry for entry in parameter_deletes(source, f"<{why}>") if entry.terminal]
        assert bool(terminal) is flagged, f"{why}: expected flagged={flagged}, got {terminal}"

    def test_the_exemplars_reach_both_verdicts(self) -> None:
        verdicts = {shape[1] for shape in _SHAPES}
        assert verdicts == {True, False}, f"the exemplars only ever expect {verdicts}, so one branch is ungraded"

    def test_the_report_names_the_line_and_the_parameters(self) -> None:
        source = 'class C:\n    async def start(self, on_joints, on_imu):\n        """Accept."""\n        del on_joints, on_imu\n'
        (entry,) = parameter_deletes(source, "tests/example.py")
        assert entry.terminal is True
        rendered = str(entry)
        assert "tests/example.py:4" in rendered
        assert "start()" in rendered
        assert "on_joints" in rendered and "on_imu" in rendered
