"""Regression: no shipped statement may sit directly after an unconditional exit.

A statement whose immediate predecessor in the same block is a ``return``,
``raise``, ``continue`` or ``break`` can never execute. Unlike the unreachable
code a type narrowing or a platform check makes dead - which is deliberate,
defensive and correct - this shape is never intentional. It is what a
mechanical edit leaves behind: an insertion anchored on a line that already
ended a function, or a duplicated tail.

Nothing in the local gate sees it. ``ruff`` selects ``E``, ``W``, ``F``, ``I``
and ``UP``, none of which covers unreachable code, and the ``mypy`` run is
configured without ``--warn-unreachable`` - for a reason this guard is careful
not to undo. Turning that flag on reports every branch narrowing makes dead,
including a dataclass coercing a ``str`` into its declared ``Path`` field and a
``sys.platform != "darwin"`` guard evaluated on Linux: legitimate code that
would have to be silenced one site at a time. This scan instead asks the one
question with no legitimate answer, so it is clean on the package as it stands
and stays quiet about the narrowing cases.

It would have caught a duplicated ``return None`` in
:func:`strands_robots.utils.non_negative_count_error`, left by the edit that
inserted :func:`~strands_robots.utils.tcp_port_error` above it - a shared
numeric-domain helper carrying a line that could never run, through a green
lint and a green test suite.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots

# A statement that transfers control out of its block unconditionally. Anything
# the same block lists after one of these is dead by construction, with no
# dependence on types, values or the platform.
_TERMINATORS = (ast.Return, ast.Raise, ast.Continue, ast.Break)

# The fields of an AST node that hold a statement list. ``handlers`` and the
# bodies nested inside them are reached by ``ast.walk`` on their own nodes.
_BLOCK_FIELDS = ("body", "orelse", "finalbody")

_PACKAGE_DIR = Path(strands_robots.__file__).resolve().parent


def _python_sources() -> list[Path]:
    return sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def unreachable_statements(source: str) -> list[tuple[int, str]]:
    """Statements in ``source`` that directly follow an unconditional exit.

    Args:
        source: Python source text.

    Returns:
        One ``(line number, terminator kind)`` pair per dead statement, where
        the line number is the dead statement's own.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(source)):
        for field in _BLOCK_FIELDS:
            block = getattr(node, field, None)
            if not isinstance(block, list):
                continue
            for statement, following in zip(block, block[1:], strict=False):
                if isinstance(statement, _TERMINATORS):
                    found.append((following.lineno, type(statement).__name__.lower()))
                    break
    return found


def test_package_sources_discovered() -> None:
    """Guard: the scan walked the whole package rather than one subtree."""
    sources = _python_sources()
    assert len(sources) > 50
    rel_dirs = {p.relative_to(_PACKAGE_DIR).parts[0] for p in sources if p.parent != _PACKAGE_DIR}
    assert {"simulation", "tools", "registry", "policies", "mesh"} <= rel_dirs


def test_no_shipped_statement_follows_an_unconditional_exit() -> None:
    """No module under ``strands_robots`` carries a statement that cannot run."""
    offenders: list[str] = []
    for path in _python_sources():
        rel = path.relative_to(_PACKAGE_DIR.parent)
        for line, kind in unreachable_statements(path.read_text(encoding="utf-8")):
            offenders.append(f"{rel}:{line} follows an unconditional {kind}")
    assert not offenders, "Statements that can never execute:\n  " + "\n  ".join(offenders)


def test_scanner_detects_a_planted_unreachable_statement() -> None:
    """Meta: an empty result means clean sources, not a scanner that matches nothing."""
    planted = "def f() -> int:\n    return 1\n    return 2\n"
    assert unreachable_statements(planted) == [(3, "return")]


def test_scanner_accepts_a_terminator_that_ends_its_block() -> None:
    """A terminator followed only by a sibling block's code is not flagged."""
    legitimate = (
        "def f(x: int) -> int:\n"
        "    for i in range(x):\n"
        "        if i:\n"
        "            continue\n"
        "        break\n"
        "    else:\n"
        "        return 0\n"
        "    raise ValueError(x)\n"
    )
    assert unreachable_statements(legitimate) == []
