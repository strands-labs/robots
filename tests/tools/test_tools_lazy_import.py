"""Behavior tests for the ``strands_robots.tools`` lazy-import contract.

The tools package defers importing each tool module until first attribute
access so that ``import strands_robots.tools`` never pulls in numpy, pyserial,
psutil, and the other heavy per-tool dependencies. These tests pin that public
contract:

- Every name advertised in ``__all__`` is lazily importable as a real attribute.
- The first access materializes the tool via ``__getattr__`` and caches it into
  the package namespace, so a repeat access returns the identical object.
- An unknown attribute raises ``AttributeError`` naming the package and the
  missing attribute, matching the standard module-level ``__getattr__`` protocol.
- Materializing a tool has no effect on the host process: reaching one is an
  import, and an import must not configure logging or write to the filesystem.

That last one needs the lazy table to be graded at the right point. Because the
table defers each module, ``import strands_robots.tools`` runs no tool module at
all, so asserting *that* import is side-effect-free passes without grading
anything. The side effect lands on the first attribute access - the documented
way in (``from strands_robots.tools import <tool>`` resolves through
``__getattr__``) - so the population graded here is the tool modules themselves.
"""

from __future__ import annotations

import importlib
import pathlib
from collections.abc import Iterator

import pytest

import strands_robots.tools as tools_pkg


def test_all_lists_every_lazy_import_name() -> None:
    """``__all__`` advertises exactly the lazily-importable tool names."""
    assert set(tools_pkg.__all__) == set(tools_pkg._LAZY_IMPORTS)
    # The advertised fleet of tools, so a silent drop is caught here.
    assert set(tools_pkg.__all__) == {
        "create_judge_agent",
        "DeciderJudge",
        "download_assets",
        "harness_memory",
        "lerobot_camera",
        "lerobot_teleoperate",
        "lerobot_train",
        "load_episode",
        "pose_tool",
        "read_predicate_verdict",
        "robot_mesh",
        "run_policy",
        "sample_frames",
        "serial_tool",
        "train_policy",
        "use_lerobot",
        "use_ros",
        "use_rosbridge",
        "use_rtps",
        "write_label",
    }


@pytest.fixture
def restored_binding(request: pytest.FixtureRequest) -> Iterator[None]:
    """Put back the package attribute this test re-caches as the tool object.

    ``vars(tools_pkg)[name]`` is process-global and holds either the tool or the
    submodule of the same name, whichever was read first. Re-entering the lazy
    path leaves the TOOL there for the rest of the process, so without this a
    later ``strands_robots.tools.<name>.<attribute>`` target elsewhere in the
    same process resolves to an object that cannot serve it - a failure that
    depends on which file ran first rather than on behavior
    (tests/tools/test_lazy_tool_name_is_not_read_as_a_module.py).
    """
    name = request.getfixturevalue("name")
    absent = object()
    before = vars(tools_pkg).get(name, absent)
    yield
    if before is absent:
        vars(tools_pkg).pop(name, None)
    else:
        vars(tools_pkg)[name] = before


@pytest.mark.parametrize("name", sorted(tools_pkg._LAZY_IMPORTS))
def test_each_tool_is_lazily_importable(name: str, restored_binding: None) -> None:
    """Each advertised name resolves to the matching object via ``__getattr__``.

    The package caches resolved tools into its namespace, and a prior test (or
    import) may already have triggered that cache, so first drop any cached
    binding to force a fresh ``__getattr__`` resolution. Then assert the access
    materializes the documented target and re-caches it for subsequent reads.
    """
    # Force the next access to go through __getattr__ rather than a cached glob.
    vars(tools_pkg).pop(name, None)
    assert name not in vars(tools_pkg)

    value = getattr(tools_pkg, name)
    assert value is not None

    # The lazy target maps to the documented (relative module, attribute) pair.
    rel_module, attr_name = tools_pkg._LAZY_IMPORTS[name]
    submodule = importlib.import_module(rel_module, tools_pkg.__name__)
    assert value is getattr(submodule, attr_name)

    # First access caches into the package namespace; second access is identical.
    assert name in vars(tools_pkg)
    assert getattr(tools_pkg, name) is value


def test_unknown_attribute_raises_attribute_error() -> None:
    """An unknown attribute raises ``AttributeError`` naming package + attr."""
    with pytest.raises(AttributeError) as excinfo:
        tools_pkg.definitely_not_a_tool  # noqa: B018 - trigger __getattr__

    message = str(excinfo.value)
    assert "strands_robots.tools" in message
    assert "definitely_not_a_tool" in message


class TestMaterializingAToolDoesNotConfigureTheHostProcess:
    """A tool module runs at first attribute access, so its top level is an import.

    Reaching a tool is not a call. ``from strands_robots.tools import pose_tool``
    executes that module's top level in whatever process imported it, which may
    be an agent, a notebook, or another library's test suite. Two top-level
    statements are therefore not available to a tool module even though they run
    fine in a script:

    ``logging.basicConfig()`` configures the *root* logger, so a library calling
    it takes over the host application's logging - installing a stderr handler
    and, with ``level=``, lowering the root level for every logger in the
    process. The module needs ``logging.getLogger(__name__)``, which configures
    nothing, and leaves the choice of handlers to whoever runs the program.

    The module-level ``logging.debug()`` / ``.info()`` / ``.warning()`` helpers
    are refused for the same reason and are the easier one to reach by accident:
    each calls ``basicConfig()`` itself when the root logger has no handler, so
    logging one line at import installs the handler without naming it. LeRobot
    does exactly this - ``lerobot.utils.import_utils`` probes its optional
    dependencies at import and reports each with ``logging.debug()``, which is
    why importing ``lerobot.cameras`` installs a root stderr handler. Only calls
    on a named logger are import-safe.

    ``mkdir()`` at the top level writes to the filesystem for an import that may
    never call the tool. The detached-session store did it under ``Path.cwd()``,
    so merely reading a tool's help littered the directory the process happened
    to be in - and raised ``PermissionError`` from the import statement where
    that directory is read-only. Its two write doors make it now
    (``tests/tools/test_the_session_directory_is_made_by_its_writer.py`` holds
    that behaviour), so no tool module is expected to create anything here.

    Graded with ``ast``, not by importing, so each offender is attributed to its
    own module: ``basicConfig`` is a no-op once any handler exists, so the second
    module to call it looks clean at runtime.
    """

    @staticmethod
    def _import_time_calls(source: str) -> list[str]:
        """Return dotted names of calls that run when the module is imported.

        Descends into ``if`` / ``try`` / ``with`` / ``class`` bodies, which all
        execute at import, and stops at ``def`` / ``lambda``, which do not.
        """
        import ast

        found: list[str] = []

        def dotted(node: ast.AST) -> str:
            if isinstance(node, ast.Attribute):
                return f"{dotted(node.value)}.{node.attr}"
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Call):
                return dotted(node.func)
            return "<expr>"

        def visit(node: ast.AST) -> None:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                return
            if isinstance(node, ast.Call):
                found.append(dotted(node.func))
            for child in ast.iter_child_nodes(node):
                visit(child)

        for statement in ast.parse(source).body:
            visit(statement)
        return found

    def _tool_modules(self) -> list[pathlib.Path]:
        root = pathlib.Path(tools_pkg.__file__).parent
        modules = sorted(p for p in root.glob("*.py") if p.name != "__init__.py")
        assert len(modules) > 10, f"read {len(modules)} tool modules under {root}; the glob found nothing"
        return modules

    #: Calls on the ``logging`` module itself that configure the root logger:
    #: ``basicConfig`` directly, and each level helper by delegating to it when
    #: the root logger has no handler yet.
    ROOT_LOGGER_CALLS = frozenset(
        {
            "logging.basicConfig",
            "logging.critical",
            "logging.debug",
            "logging.error",
            "logging.exception",
            "logging.info",
            "logging.log",
            "logging.warning",
        }
    )

    def test_no_tool_module_configures_the_root_logger(self) -> None:
        """No tool module reaches the root logger at import, by either door."""
        offenders = {
            module.name: sorted(reached)
            for module in self._tool_modules()
            if (
                reached := self.ROOT_LOGGER_CALLS.intersection(
                    self._import_time_calls(module.read_text(encoding="utf-8"))
                )
            )
        }
        assert offenders == {}, (
            f"{offenders} configure the root logger at import, taking over logging "
            "for whatever process imported the tool; log through "
            "logging.getLogger(__name__) instead, which configures nothing"
        )

    def test_no_tool_module_creates_a_directory_at_import(self) -> None:
        """A directory a verb writes into is made by the verb, not by the import."""
        offenders = sorted(
            module.name
            for module in self._tool_modules()
            if any(call.endswith(".mkdir") for call in self._import_time_calls(module.read_text(encoding="utf-8")))
        )
        assert offenders == [], (
            f"{offenders} create a directory at import, so a caller who only read a "
            "tool's help pays for it and one whose working directory is read-only "
            "cannot import the tool at all; create it from the function that writes "
            "into it"
        )
