"""Regression: no list of literals may hold two adjacent string parts.

Python folds adjacent string literals into a single value, so inside a list
display ``["a" "b", "c"]`` and ``["a", "b", "c"]`` differ by one character and
mean different things - two elements or three. A dropped comma therefore does
not fail: it silently merges two neighbouring elements into one. A parametrize
case disappears from a suite that still reports green, and a table of expected
strings loses a row.

Nothing in the local gate sees it, and ``ruff`` cannot be made to. Its selected
set is ``E``, ``W``, ``F``, ``I``, ``UP``, ``B015`` and ``B018``, none of which
covers implicit concatenation, and neither ``flake8-implicit-str-concat`` rule
reaches this shape at a price the tree can pay. ``ISC002`` is silent because
``allow-multiline`` defaults to true; setting it to false reports 4056 sites,
almost all of them ordinary wrapped prose in docstrings and messages. ``ISC001``
is cheap - 4 sites, all under ``examples`` - but it only sees a single line, and
none of those 4 sits in a display, so it would not have reported the element this
rule was written for.

CodeQL does report it, as ``py/implicit-string-concatenation-in-list`` - but only
after a push, and only once ``github-advanced-security`` has opened a review
thread, which the ``default`` branch ruleset then requires somebody to resolve
before the merge. ``.github/codeql/codeql-config.yml`` states the remedy for
exactly that situation: a capability CodeQL only advises on "moves to ruff, which
is merge-blocking here". Where ruff cannot express the rule, as here, the
merge-blocking local gate is this suite - so an author sees it before pushing
rather than during review.

Scope is a **list** display, matching the rule that gates the merge. Tuples are
deliberately left alone: three tuple elements in
``tests/test_container_refusals_render_elementwise.py`` are written this way
today and CodeQL does not report them, so widening the scope would trade a
zero-exemption rule for three exemptions. An element with a formatted part is
also out of scope - it parses as :class:`ast.JoinedStr` rather than
:class:`ast.Constant`, and a caller who reached for an f-string was building one
string on purpose.

The scan covers this file too, so the rule it states applies to itself.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

# The Python-bearing directories the CodeQL workflow scans. ``docs`` holds none.
_SCAN_ROOTS = ("strands_robots", "tests", "tests_integ", "examples", "scripts")


def _python_sources() -> list[Path]:
    files: list[Path] = []
    for name in _SCAN_ROOTS:
        root = _REPO_ROOT / name
        if root.is_dir():
            files.extend(p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts)
    return files


def _string_token_starts(source: str) -> list[tuple[int, int]]:
    """``(line, column)`` of every plain string-literal token in ``source``."""
    return [
        token.start for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.STRING
    ]


def implicit_concatenations(source: str) -> list[tuple[int, int]]:
    """List elements in ``source`` built from two or more adjacent string parts.

    The parser folds adjacent literals into one :class:`ast.Constant`, which is
    also what a triple-quoted string produces, so the AST alone cannot tell the
    two apart. The number of string tokens inside the element's own extent can:
    a triple-quoted string is one token, an implicit concatenation is several.

    Args:
        source: Python source text.

    Returns:
        One ``(line number, number of string parts)`` pair per offending
        element, where the line number is the element's first line.
    """
    tree = ast.parse(source)
    starts = _string_token_starts(source)
    found: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.List):
            continue
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                continue
            end_line, end_col = element.end_lineno, element.end_col_offset
            if end_line is None or end_col is None:  # pragma: no cover - always set by ast.parse
                continue
            begin = (element.lineno, element.col_offset)
            parts = sum(1 for start in starts if begin <= start < (end_line, end_col))
            if parts >= 2:
                found.append((element.lineno, parts))
    return found


# The element this rule was written for: the own-private-state case in the
# ``clean`` parametrize list of ``tests/test_examples_release_the_mesh.py``,
# reported as CodeQL alert 895 before it was hoisted into a named constant.
_THE_REPORTED_ELEMENT = """\
CASES = [
    "class D:\\n    def __init__(self, mesh):\\n        self._mesh = mesh\\n"
    "    def go(self):\\n        return self._mesh.peers\\n",
]
"""

_PLANTED = {
    "multi-line": _THE_REPORTED_ELEMENT,
    "single-line": 'CASES = ["alpha" "beta", "gamma"]\n',
    "three-parts": 'CASES = [\n    "one "\n    "two "\n    "three",\n]\n',
}

_UNAMBIGUOUS = {
    "separate-elements": 'CASES = ["alpha", "beta"]\n',
    "explicit-concatenation": 'CASES = [\n    "alpha "\n    + "beta",\n]\n',
    "one-triple-quoted-literal": 'CASES = [\n    """alpha\nbeta\n""",\n]\n',
    "formatted-element": 'name = "x"\nCASES = [\n    f"alpha {name} "\n    f"beta",\n]\n',
    "escaped-newline-in-one-literal": 'CASES = ["alpha\\nbeta\\n"]\n',
}


def test_the_scan_covers_every_python_bearing_directory() -> None:
    """Non-vacuity: an empty result must mean clean sources, not an empty scan."""
    sources = _python_sources()
    assert len(sources) > 1000
    roots = {p.relative_to(_REPO_ROOT).parts[0] for p in sources}
    assert set(_SCAN_ROOTS) <= roots
    assert Path(__file__).resolve() in {p.resolve() for p in sources}


def test_no_list_of_literals_holds_two_adjacent_string_parts() -> None:
    """No list display in the repository merges its elements by omitting a comma."""
    offenders: list[str] = []
    for path in _python_sources():
        rel = path.relative_to(_REPO_ROOT)
        for line, parts in implicit_concatenations(path.read_text(encoding="utf-8")):
            offenders.append(f"{rel}:{line} is one element built from {parts} adjacent string parts")
    assert not offenders, (
        "List elements built by implicit string concatenation. A reader cannot tell "
        "these from a dropped comma, so a merged pair looks like a single case:\n  "
        + "\n  ".join(offenders)
        + "\nHoist the text into a named constant, or join the parts with `+` if they "
        "really are one string."
    )


def test_the_scanner_reports_the_element_it_was_written_for() -> None:
    """The rule reproduces the reviewed finding, not merely something like it."""
    assert implicit_concatenations(_THE_REPORTED_ELEMENT) == [(2, 2)]


@pytest.mark.parametrize("planted", list(_PLANTED.values()), ids=list(_PLANTED))
def test_the_scanner_detects_a_planted_concatenation(planted: str) -> None:
    """Meta: the scanner fires on the shape it claims to find."""
    assert implicit_concatenations(planted)


@pytest.mark.parametrize("clean", list(_UNAMBIGUOUS.values()), ids=list(_UNAMBIGUOUS))
def test_the_scanner_accepts_an_unambiguous_element(clean: str) -> None:
    """One literal per element, however it is spelled, is never ambiguous."""
    assert implicit_concatenations(clean) == []


def test_the_scanner_leaves_tuples_alone() -> None:
    """The measured scope boundary: the gating rule reports list displays only.

    Three tuple elements under ``tests`` are written as adjacent parts today and
    CodeQL reports none of them, so this rule stops where that one stops.
    """
    as_tuple = 'CASES = (\n    "alpha "\n    "beta",\n)\n'
    assert implicit_concatenations(as_tuple) == []
    assert implicit_concatenations(as_tuple.replace("(", "[", 1).replace(")", "]", 1))
