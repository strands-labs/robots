"""Every parameter an agent tool advertises must reach that tool's body.

``tests/tools/test_agent_tool_parameter_descriptions.py`` grades the two
directions in which a ``@tool`` signature and its ``Args:`` section can
disagree: a parameter the model is shown with no description, and a docstring
entry naming a parameter the function does not have. Neither asks the third
question - whether the function ever *reads* the parameter it advertises.

That third case is the one that costs the model most. An undescribed parameter
is visibly opaque, and an entry naming nothing is invisible to the model
entirely; a described parameter the body discards reads as a working knob. The
model is told which values it accepts, spends a decision choosing one, reports
having set it, and the value reaches nothing. ``lerobot_calibrate`` carried
``format_output: str = "rich"`` described as ``"Output format (rich, simple,
json)"``: the name was declared, the description was specific, and the function
body never mentioned it. Both sibling directions passed it
(``TestTheSiblingDirectionsPassADeadParameter`` below pins that they do).

Such a parameter is also not merely unimplemented. An agent tool returns a
fixed envelope - a ``content`` list of typed blocks, and every
``lerobot_calibrate`` success path already emits both a ``text`` rendering and
a ``json`` one - so the caller selects a rendering by reading the block it
wants. There was no format axis for the parameter to switch, which is why the
remedy is to drop it rather than to implement three formats.

Scope
-----
The population is every ``@tool``-decorated function under
``strands_robots/``, read with ``ast`` rather than by importing. That is wider
than the sibling's, which walks ``strands_robots.tools`` with ``pkgutil`` and
so reaches the 26 top-level verbs only: the ``tools/g1``, ``tools/reachy`` and
``mesh`` tool families are graded here for the first time.
``TestTheTwoScansAgree`` pins that this scan is a superset of the sibling's, so
a walk that silently stopped finding tools fails rather than reporting a clean
sweep.

The rule is a *load*: the parameter's name must appear somewhere in the body in
a context that reads it. Two spellings that mention the name without reading it
do not count, and both are pinned in ``TestTheRuleIsWhatTheModelCanObserve``
- rebinding it (``x = ...``) and deleting it (``del x``) each discard whatever
the caller sent. Reading it anywhere the body reaches counts, including from a
nested function or a comprehension.

No exemption list: all 87 tools pass.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from tests.tools.test_agent_tool_parameter_descriptions import _BOUND_TOOLS, _declared_parameters

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[2] / "strands_robots"


def _is_agent_tool(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Whether ``func`` carries the ``@tool`` decorator, in either spelling.

    Both the bare ``@tool`` and the configured ``@tool(...)`` form bind an
    agent-callable verb, and both are reached through a plain name (``tool``)
    or an attribute (``strands.tool``).
    """
    for decorator in func.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "tool":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "tool":
            return True
    return False


def _agent_tools() -> list[tuple[str, str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Return ``(module_path, func_name, node)`` for every ``@tool`` in the package."""
    found: list[tuple[str, str, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and _is_agent_tool(node):
                found.append((str(path.relative_to(_PACKAGE_ROOT.parent)), node.name, node))
    return found


_AGENT_TOOLS = _agent_tools()
_IDS = [f"{module}::{name}" for module, name, _ in _AGENT_TOOLS]


def _names_the_body_reads(func: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Names the body loads, anywhere it reaches (nested scopes included).

    Only a load counts. A ``Store``-context appearance rebinds the name and a
    ``Del`` one discards it; neither consults the value the caller supplied.
    """
    read: set[str] = set()
    for statement in func.body:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                read.add(node.id)
    return read


def _dead_parameters(func: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Declared parameters of ``func`` that its body never reads."""
    read = _names_the_body_reads(func)
    return sorted(name for name in _declared_parameters(func) if name not in read and name not in {"self", "cls"})


class TestEveryAdvertisedParameterReachesTheBody:
    """The model's knob must be able to act."""

    @pytest.mark.parametrize(("module_path", "func_name", "func"), _AGENT_TOOLS, ids=_IDS)
    def test_no_parameter_is_declared_and_never_read(
        self, module_path: str, func_name: str, func: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        dead = _dead_parameters(func)
        assert dead == [], (
            f"{module_path}:{func.lineno} {func_name} advertises parameters its body never "
            f"reads: {dead}. The model is shown each one in the tool schema and can spend a "
            f"call setting it, and the value reaches nothing. Either read the parameter or "
            f"remove it from the signature and from the Args: section."
        )


class TestTheTwoScansAgree:
    """A walk that stopped finding tools would report a clean sweep."""

    def test_this_scan_reaches_every_tool_the_sibling_binds(self) -> None:
        found = {name for _module, name, _func in _AGENT_TOOLS}
        assert {name for _module, name, _tool in _BOUND_TOOLS} <= found

    def test_this_scan_also_reaches_the_families_the_sibling_cannot(self) -> None:
        """``pkgutil.iter_modules`` does not descend, so subpackages need this scan."""
        directories = {pathlib.PurePosixPath(module).parent.as_posix() for module, _name, _func in _AGENT_TOOLS}
        assert {"strands_robots/tools/g1", "strands_robots/tools/reachy", "strands_robots/mesh"} <= directories


class TestTheRuleIsWhatTheModelCanObserve:
    """Pin which spellings count as reading a parameter, and which do not."""

    @staticmethod
    def _dead(source: str) -> list[str]:
        func = ast.parse(source).body[0]
        assert isinstance(func, ast.FunctionDef)
        return _dead_parameters(func)

    def test_a_parameter_the_body_loads_is_read(self) -> None:
        assert self._dead("@tool\ndef verb(fmt):\n    return {'text': fmt}\n") == []

    def test_a_parameter_read_only_from_a_nested_scope_is_read(self) -> None:
        assert self._dead("@tool\ndef verb(fmt):\n    def inner():\n        return fmt\n    return inner()\n") == []
        assert self._dead("@tool\ndef verb(fmt):\n    return [fmt for _ in range(1)]\n") == []

    def test_a_parameter_only_rebound_is_not_read(self) -> None:
        assert self._dead("@tool\ndef verb(fmt):\n    fmt = 'rich'\n    return {}\n") == ["fmt"]

    def test_a_parameter_rebound_from_its_own_value_is_read(self) -> None:
        """``x = x or default`` consults the caller's value, so it counts."""
        assert self._dead("@tool\ndef verb(fmt):\n    fmt = fmt or 'rich'\n    return {}\n") == []

    def test_a_parameter_only_deleted_is_not_read(self) -> None:
        assert self._dead("@tool\ndef verb(fmt):\n    del fmt\n    return {}\n") == ["fmt"]

    def test_a_parameter_named_only_in_a_docstring_is_not_read(self) -> None:
        assert self._dead(
            '@tool\ndef verb(fmt):\n    """Args:\n        fmt: Output format.\n    """\n    return {}\n'
        ) == ["fmt"]


class TestTheSiblingDirectionsPassADeadParameter:
    """Why this file exists: the described-and-real checks both clear it.

    Reconstructs the parameter this guard was added for - declared, described
    with a specific description, and never read - and shows the sibling's two
    predicates report nothing while this one reports the name.
    """

    SOURCE = (
        "@tool\n"
        "def lerobot_calibrate(action='list', format_output='rich'):\n"
        '    """Manage calibrations.\n'
        "\n"
        "    Args:\n"
        "        action: Action to perform (list, view).\n"
        "        format_output: Output format (rich, simple, json).\n"
        '    """\n'
        "    return {'status': 'success', 'content': [{'text': action}]}\n"
    )

    def _func(self) -> ast.FunctionDef:
        func = ast.parse(self.SOURCE).body[0]
        assert isinstance(func, ast.FunctionDef)
        return func

    def test_the_description_is_specific_so_the_placeholder_direction_is_green(self) -> None:
        import docstring_parser

        entries = {
            p.arg_name: (p.description or "")
            for p in docstring_parser.parse(ast.get_docstring(self._func()) or "").params
        }
        assert entries["format_output"] != "Parameter format_output"
        assert "rich, simple, json" in entries["format_output"]

    def test_the_entry_names_a_real_parameter_so_that_direction_is_green_too(self) -> None:
        import docstring_parser

        func = self._func()
        declared = _declared_parameters(func)
        parsed = docstring_parser.parse(ast.get_docstring(func) or "")
        assert [p.arg_name for p in parsed.params if p.arg_name not in declared] == []

    def test_this_direction_reports_it(self) -> None:
        assert _dead_parameters(self._func()) == ["format_output"]
