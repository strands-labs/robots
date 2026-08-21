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
that reach one of the three surfaces AGENTS.md names. That keeps intentional,
semantic Unicode in docstrings (mapping arrows, math symbols) untouched while
enforcing the ASCII rule exactly where a human or an agent reads the string:

1. ``logger.<level>(...)`` / ``warnings.warn(...)`` arguments - a log line.
2. The message of a ``raise <Exception>(...)`` statement - a traceback.
3. The ``text`` / ``message`` a tool result carries - what an agent reads back
   out of a tool call, and the surface AGENTS.md names first.

Surface 3 needs a flow step the other two do not: a handler rarely spells its
report inline. The dominant idiom builds the report up in a local list and joins
that into the returned dict, so grading only the expressions written inside the
``return`` misses the line the glyph is actually in. :func:`_tool_result_strings`
therefore walks the local names that flow into the ``content`` / ``message``
value and grades the strings assigned or appended to them. It stays narrower
than "every literal in a handler": on the tree that first passed it, the
flow-following scan graded 4946 strings and the blanket alternative 10002, both
reporting the same offenders, so nothing here is caught by breadth alone.

It would have failed when 15 ``logger``/``raise`` strings across 6 modules still
carried ``U+2192``, and again when ten glyphs sat in agent-facing tool-result
text: ``U+2194`` in ``get_contact_forces``'s contact list (which
``simulation.mujoco.rendering`` already spelled ``<->`` for the same pair),
``U+2192`` in ``set_geom_properties`` / ``set_body_properties`` / the GR00T
checkpoint download, ``U+00D7`` in ``get_mass_matrix``, ``U+00B7`` in
``apply_force``'s torque unit, ``U+00B1`` in ``randomize``, and ``U+2022`` in
``list_benchmarks``.
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


# Keys whose value a tool result renders for the caller. ``content`` is the
# ``AgentTool`` shape (``[{"text": ...}, {"json": ...}]``); ``message`` is the
# flat shape the ``@tool``-decorated functions in ``strands_robots.tools`` use.
# The ``json`` block is machine-readable payload, not prose, and is not graded.
_RESULT_TEXT_KEYS = frozenset({"content", "message"})

# Mutators through which a local accumulates the report. ``lines.append(...)``
# is the dominant idiom; ``extend`` / ``insert`` / ``add`` appear beside it.
_ACCUMULATORS = frozenset({"append", "extend", "insert", "add"})


def _dict_keys(node: ast.Dict) -> set[str]:
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _returned_result_dicts(fn: ast.AST) -> list[ast.Dict]:
    """Dict literals ``fn`` returns that carry a status plus rendered text."""
    out: list[ast.Dict] = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys = _dict_keys(node.value)
            if "status" in keys and keys & _RESULT_TEXT_KEYS:
                out.append(node.value)
    return out


def _names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _string_nodes(node: ast.AST) -> list[ast.Constant]:
    return [n for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]


def _flow_into(fn: ast.AST, seeds: set[str]) -> list[ast.Constant]:
    """String constants that flow into ``seeds`` through local names in ``fn``.

    A fixed-point walk: a statement that assigns to - or accumulates into - a
    name already known to reach the result contributes both its string literals
    and the further names it reads. It terminates because the tainted set only
    grows and is bounded by the function's identifier count.
    """
    tainted = set(seeds)
    collected: dict[int, ast.Constant] = {}
    while True:
        grew = False
        for node in ast.walk(fn):
            source: ast.AST | None = None
            if isinstance(node, ast.Assign):
                if {t.id for t in node.targets if isinstance(t, ast.Name)} & tainted:
                    source = node.value
            elif isinstance(node, ast.AugAssign):
                if isinstance(node.target, ast.Name) and node.target.id in tainted:
                    source = node.value
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _ACCUMULATORS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in tainted
            ):
                source = node
            if source is None:
                continue
            if new := _names_in(source) - tainted:
                tainted |= new
                grew = True
            for const in _string_nodes(source):
                collected.setdefault(id(const), const)
        if not grew:
            return list(collected.values())


def _tool_result_strings(tree: ast.AST) -> list[tuple[int, str]]:
    """(lineno, string) for every literal a tool result renders as prose."""
    found: list[tuple[int, str]] = []
    seen: set[int] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for result in _returned_result_dicts(fn):
            for key, value in zip(result.keys, result.values, strict=True):
                if not (isinstance(key, ast.Constant) and key.value in _RESULT_TEXT_KEYS):
                    continue
                for const in _string_nodes(value) + _flow_into(fn, _names_in(value)):
                    if id(const) in seen:
                        continue
                    seen.add(id(const))
                    # ``_string_nodes`` only yields ``str`` constants; the check
                    # narrows ``ast.Constant.value`` (typed as the whole literal
                    # union) for the type checker.
                    if isinstance(const.value, str):
                        found.append((const.lineno, const.value))
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


def test_tool_result_scan_finds_text() -> None:
    """Sanity: the flow-following scan actually locates rendered result text."""
    total = sum(len(_tool_result_strings(ast.parse(p.read_text(encoding="utf-8")))) for p in _python_sources())
    assert total > 500, f"AST scan found suspiciously few tool-result strings ({total})"


def test_tool_result_strings_are_ascii() -> None:
    """No string a tool result renders as prose may be non-ASCII.

    This is the surface AGENTS.md names first, and the one whose whole audience
    is programmatic: an agent reads this text back out of its own tool call.
    """
    offenders: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for lineno, text in _tool_result_strings(tree):
            if not text.isascii():
                bad = sorted({f"U+{ord(c):04X}" for c in text if ord(c) > 0x7F})
                rel = path.relative_to(_PACKAGE_DIR.parent)
                offenders.append(f"{rel}:{lineno}: {' '.join(bad)} in {text.strip()[:80]!r}")
    assert not offenders, (
        "Non-ASCII in tool-result text (use ASCII: '->' for arrows, '<->', '+/-', 'x', 'N*m'):\n" + "\n".join(offenders)
    )


def test_the_scan_follows_a_report_built_up_in_a_local() -> None:
    """The dominant idiom is graded: append to a local, join it into the result.

    Without the flow step the glyph below is invisible, because the ``return``
    expression holds no literal of its own. Eight of the ten offenders this
    guard first caught were written exactly this way.
    """
    source = "\n".join(
        [
            "def handler(items):",
            '    lines = ["header"]',
            "    for item in items:",
            '        lines.append(f"{item} \u2192 done")',
            '    return {"status": "success", "content": [{"text": "".join(lines)}]}',
        ]
    )
    graded = [text for _lineno, text in _tool_result_strings(ast.parse(source))]
    assert any("\u2192" in text for text in graded), f"flow step missed the appended line: {graded}"


def test_a_string_that_never_reaches_the_result_is_left_alone() -> None:
    """Precision: a literal the result does not render is not graded.

    This is what keeps the rule narrower than "every literal in a function that
    returns a result dict" - a comparison value or a lookup key is not prose an
    agent reads back.
    """
    source = "\n".join(
        [
            "def handler(mode):",
            '    if mode == "\u00b1special":',
            "        pass",
            '    return {"status": "success", "content": [{"text": "plain"}]}',
        ]
    )
    graded = [text for _lineno, text in _tool_result_strings(ast.parse(source))]
    assert not any("\u00b1" in text for text in graded), f"graded a literal the result never renders: {graded}"


def test_semantic_unicode_in_a_handler_docstring_is_left_alone() -> None:
    """A docstring keeps its mapping arrows even inside a result-returning handler.

    The package carries dozens of such literals deliberately; the rule is about
    what a caller reads back, not about how the source documents itself.
    """
    source = "\n".join(
        [
            "def handler():",
            "    'Maps a \u2192 b.'",
            '    return {"status": "success", "content": [{"text": "plain"}]}',
        ]
    )
    graded = [text for _lineno, text in _tool_result_strings(ast.parse(source))]
    assert not any("\u2192" in text for text in graded), f"graded a docstring: {graded}"
