"""Regression: runtime diagnostic strings in ``strands_robots`` stay ASCII.

AGENTS.md mandates plain ASCII in logs, error messages, and tool-result text.
Typographic glyphs (arrows like ``U+2192`` ``->``, ``U+2194`` ``<->``) render
inconsistently across terminals and log pipelines and are tokenizer noise for
the agents that read these strings programmatically. An ASCII rendering carries
the same meaning everywhere.

Unlike :mod:`tests.test_source_strings_no_emoji` and
:mod:`tests.test_source_strings_no_unicode_dashes` -- which scan every source
byte and therefore also police docstrings/comments -- this guard is deliberately
*surgical*: it parses each module's AST and only inspects the string literals
that are arguments to ``logger.<level>(...)`` / ``warnings.warn(...)`` calls or
the message of a ``raise <Exception>(...)`` statement. That keeps intentional,
semantic Unicode in docstrings (mapping arrows, math symbols) untouched while
enforcing the ASCII rule exactly on the surface AGENTS.md names: what a human or
agent reads out of a log or traceback.

It would have failed when 15 ``logger``/``raise`` strings across 6 modules still
carried ``U+2192``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots

_PACKAGE_DIR = Path(strands_robots.__file__).resolve().parent

# Standard-library ``logging.Logger`` emitters plus ``logging.log``. Matching on
# the method name keeps the guard agnostic to the logger's binding name
# (``logger``, ``LOGGER``, ``self._log``, ...); we additionally require the
# receiver's root identifier to look logger-ish so an unrelated ``obj.info(...)``
# is not swept in.
_LOG_LEVELS = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical", "log"})


def _python_sources() -> list[Path]:
    return sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def _string_constants(node: ast.AST) -> list[str]:
    """All ``str`` constants reachable in an expression (f-strings, concat, ...)."""
    return [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _root_name(expr: ast.expr) -> str:
    """Left-most identifier of an attribute chain (``a.b.c`` -> ``a``)."""
    while isinstance(expr, ast.Attribute):
        expr = expr.value
    return expr.id if isinstance(expr, ast.Name) else ""


def _is_logger_call(call: ast.Call) -> bool:
    func = call.func
    if not isinstance(func, ast.Attribute) or func.attr not in _LOG_LEVELS:
        return False
    if func.attr == "warn" and _root_name(func) in {"warnings", "warning"}:
        return True  # warnings.warn(...)
    return "log" in _root_name(func).lower()


def _diagnostic_strings(tree: ast.AST) -> list[tuple[int, str]]:
    """(lineno, string) for every logger/raise/warn diagnostic literal."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_logger_call(node):
            for arg in node.args:
                found.extend((node.lineno, s) for s in _string_constants(arg))
        elif isinstance(node, ast.Raise) and node.exc is not None:
            found.extend((node.lineno, s) for s in _string_constants(node.exc))
    return found


def test_package_sources_discovered() -> None:
    """Guard the guard: the scan walked the whole package, not one subtree."""
    sources = _python_sources()
    assert len(sources) > 50
    rel_dirs = {p.relative_to(_PACKAGE_DIR).parts[0] for p in sources if p.parent != _PACKAGE_DIR}
    assert {"simulation", "tools", "registry", "policies", "assets"} <= rel_dirs


def test_diagnostic_string_scan_finds_calls() -> None:
    """Sanity: the AST walk actually locates diagnostic strings to inspect."""
    total = sum(len(_diagnostic_strings(ast.parse(p.read_text(encoding="utf-8")))) for p in _python_sources())
    assert total > 100, "AST scan found suspiciously few logger/raise strings"


def test_log_and_error_strings_are_ascii() -> None:
    """No ``logger``/``raise``/``warnings.warn`` string literal may be non-ASCII."""
    offenders: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for lineno, text in _diagnostic_strings(tree):
            if not text.isascii():
                bad = sorted({f"U+{ord(c):04X}" for c in text if ord(c) > 0x7F})
                rel = path.relative_to(_PACKAGE_DIR.parent)
                offenders.append(f"{rel}:{lineno}: {' '.join(bad)} in {text.strip()[:80]!r}")
    assert not offenders, "Non-ASCII in logger/raise/warn strings (use ASCII, e.g. '->' for arrows):\n" + "\n".join(
        offenders
    )
