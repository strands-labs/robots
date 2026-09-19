"""``docs/hooks/tool_reference.py`` publishes the schema an agent receives.

The tool reference is generated at build time from the source, so the page cannot go stale the
way a hand-written table does. What it *can* do is disagree with ``strands.tool``: the hook reads
each signature and Google docstring with :mod:`ast`, because the docs environment installs mkdocs
alone and cannot import the package, while an agent is handed a schema that ``docstring_parser``
and pydantic build from the same function. This grader drives both and compares them tool by tool
-- names, order, which parameters are required, every default, every description.

That comparison is not decorative. On the tree it arrived in it caught two tools whose parameter
description no agent ever received: ``g1_decode_error_code`` and ``g1_get_state`` each began a
docstring line with an RST role (``:func:`` / ``:data:``), which ``docstring_parser`` reads as a
REST field. Its style auto-detection scores REST equal to Google on those docstrings and breaks
the tie in REST's favour, so the Google ``Args:`` block was not read at all and every parameter
description degraded to the literal ``"Parameter <name>"``. The words were in the source and in
no schema, and only a comparison against the built spec could say so.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

import strands_robots

_REPO = Path(strands_robots.__file__).resolve().parents[1]
_PKG = _REPO / "strands_robots"

# A hook that found nothing must not read as a clean sweep.
_MINIMUM_TOOLS = 60


def _hook():
    """The hook module, loaded from the docs tree the build loads it from."""
    name = "docs_tool_reference_hook"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REPO / "docs/hooks/tool_reference.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # the hook's dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def _published() -> tuple:
    """Every tool the hook puts on the page, flattened."""
    return tuple(item for family in _hook().tools().values() for item in family)


def _importable_tools(tree: ast.Module):
    """Every ``def`` a caller can reach as an attribute: module level, or on a class there.

    A ``@tool`` written inside another function is not one of those. The dashboard's agent
    console builds its eight (``strands_robots.dashboard.agent_console.build_tools``) per
    conversation, closed over that conversation's safety object, so nothing can import them
    and no page could tell a reader how to call one - the same reason a per-instance name is
    out of scope below.
    """
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            yield node
        elif isinstance(node, ast.ClassDef):
            yield from (item for item in node.body if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef))


def _statically_named_tools() -> set[tuple[str, str]]:
    """(module, name) for every importable ``@tool`` in the package whose name is a literal.

    Derived here rather than through the hook, so a hook that stops looking in a directory
    fails this grader instead of quietly shortening the page. A tool whose name the decorator
    computes per instance (the mesh robots build one per robot) names nothing a page could
    spell and is out of scope, as is one that is not importable at all (:func:`_importable_tools`).
    """
    found: set[tuple[str, str]] = set()
    for path in sorted(_PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _importable_tools(tree):
            for decorator in node.decorator_list:
                call = decorator.func if isinstance(decorator, ast.Call) else decorator
                if not (isinstance(call, ast.Name) and call.id == "tool"):
                    continue
                name: str | None = node.name
                if isinstance(decorator, ast.Call):
                    for keyword in decorator.keywords:
                        if keyword.arg == "name":
                            value = keyword.value
                            name = str(value.value) if isinstance(value, ast.Constant) else None
                if name is not None:
                    module = ".".join(path.relative_to(_REPO).with_suffix("").parts)
                    found.add((module, name))
    return found


def test_the_page_names_every_statically_named_tool_in_the_package() -> None:
    published = {(item.module, item.name) for item in _published()}
    assert len(published) >= _MINIMUM_TOOLS
    assert published == _statically_named_tools()


def test_every_tool_gets_exactly_one_section() -> None:
    page = _hook().render()
    for item in _published():
        assert page.count(f"### `{item.name}`\n") == 1, item.name


@pytest.mark.parametrize("item", _published(), ids=lambda item: item.name)
def test_the_page_publishes_the_schema_the_agent_receives(item) -> None:  # noqa: ANN001 - the hook's Tool
    schema = getattr(importlib.import_module(item.module), item.name).tool_spec
    published = schema["inputSchema"]["json"]
    properties = published["properties"]

    assert [param.name for param in item.params] == list(properties)
    assert {param.name for param in item.params if param.required} == set(published.get("required", []))
    assert " ".join(schema["description"].split()).startswith(item.summary)
    for param in item.params:
        declared = properties[param.name]
        assert param.description == " ".join(declared.get("description", "").split())
        if param.required:
            assert "default" not in declared
        else:
            assert ast.literal_eval(param.default or "") == declared["default"]


def test_a_default_or_type_carrying_a_pipe_cannot_split_a_table_row() -> None:
    """``str | None`` is the commonest annotation here, and a raw pipe would forge a column."""
    rows = [line for line in _hook().render().splitlines() if line.startswith("| ") and "---" not in line]
    assert rows
    for row in rows:
        assert len(re.split(r"(?<!\\)\|", row.strip("|"))) == 4, row


def test_the_hook_reads_the_tree_without_importing_the_package() -> None:
    """The docs environment installs mkdocs only, so an import of the package fails the build."""
    tree = ast.parse((_REPO / "docs/hooks/tool_reference.py").read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    } | {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert "strands_robots" not in imported
    assert "strands" not in imported


def test_the_page_is_a_token_and_a_provenance_line() -> None:
    """Nothing on the page is typed by hand except the header that says so."""
    source = (_REPO / "docs/reference/tools.md").read_text(encoding="utf-8")
    assert source.count("{{tool_reference}}") == 1
    assert "do not edit" in source
    assert "{{tool_reference}}" not in _hook().substitute(source)
