"""The consolidated ``tools/g1`` surface stays complete, and discovery stays honest.

Two properties, both of which a green suite happened not to hold while the
lookup-verb consolidation (refs #2928) was in flight.

**The declared surface accounts for every ``@tool`` on disk.** After the
consolidation the package publishes its verbs through the ``_LAZY_IMPORTS``
table on :mod:`strands_robots.tools.g1`, so that table *is* the surface. It is
hand-written, and two independent things drift it: a verb landing on ``main``
after a consolidation branch forks is kept by a three-way merge while the
branch's rewritten ``__init__`` never learns its name, and a lookup pair
landing the same way survives a removal it was never part of. Neither shows up
as a conflict - git sees an addition on one side and no change on the other -
so the check has to be derived from the tree rather than looked for by hand.
The rule is a biconditional: every ``@tool`` on disk is declared, and every
declared name resolves.

**Discovery degrades only for an absent SDK.** ``list_operations`` and
``describe_operation`` answer from :mod:`inspect` when ``unitree_sdk2py`` is
importable and from an AST read of the SDK source when it is not. That
fallback is the whole reason the dispatcher replaces the lookup modules on a
machine with no robot, and it is reached by catching the *absent-SDK*
condition specifically. Catching more than that turns any defect in the
introspection path into a silently differently-sourced answer: an operation
list read from a possibly-stale on-disk tree, an empty list that reads as
"this service has no operations", or a ``parameters: []`` that reads as "this
operation takes no arguments". An agent then calls the operation with no
arguments. So the narrow set is load-bearing, not stylistic.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
from typing import Any

import pytest

import strands_robots.tools.g1 as g1_pkg

_PKG_DIR = pathlib.Path(g1_pkg.__file__).parent

# A lookup module answers a constant table and nothing else; the dispatcher's
# ``describe_operation`` answers the same question without a ``@tool`` name.
_LOOKUP_SUFFIXES = ("_envelope", "_admits", "_topics", "_ids", "_keys", "_notes")


def _tool_names_on_disk() -> dict[str, str]:
    """Map every ``@tool`` function in the package to the module declaring it."""
    found: dict[str, str] = {}
    for path in sorted(_PKG_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                name = (
                    dec.id
                    if isinstance(dec, ast.Name)
                    else dec.func.id
                    if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name)
                    else None
                )
                if name == "tool":
                    found[node.name] = path.name
    return found


class TestTheDeclaredSurfaceAccountsForEveryToolOnDisk:
    """``_LAZY_IMPORTS`` and the ``@tool`` functions on disk name one set."""

    def test_every_tool_on_disk_is_reachable_through_the_package(self) -> None:
        on_disk = _tool_names_on_disk()
        assert on_disk, "found no @tool in the g1 package - this scan is looking in the wrong place"

        unreachable = {name: mod for name, mod in on_disk.items() if not hasattr(g1_pkg, name)}
        assert not unreachable, (
            "these @tool names exist in strands_robots/tools/g1 but are not reachable through the "
            f"package: {unreachable}. Either add each to _LAZY_IMPORTS (a kept verb), or delete the "
            "module and route its constants through use_unitree's describe_operation (a lookup pair)."
        )

    def test_every_declared_name_resolves(self) -> None:
        declared = [n for n in g1_pkg.__all__ if n.startswith(("g1_", "use_unitree"))]
        assert declared, "the package declares no verbs - this scan is looking in the wrong place"

        for name in declared:
            assert getattr(g1_pkg, name, None) is not None, (
                f"strands_robots.tools.g1 declares {name!r} but it does not resolve"
            )


class TestNoLookupShapedModuleSurvivesTheConsolidation:
    """A constant table does not earn a ``@tool`` name of its own."""

    def test_the_package_carries_no_lookup_shaped_module(self) -> None:
        survivors = sorted(p.name for p in _PKG_DIR.glob("*.py") if any(p.stem.endswith(s) for s in _LOOKUP_SUFFIXES))
        assert not survivors, (
            f"lookup-shaped modules survive in strands_robots/tools/g1: {survivors}. Each costs a "
            "tool-schema slot in every agent's context for a constant table use_unitree's "
            "describe_operation already answers. Delete the module and keep any factual statement "
            "it carried inline in the docstring that cited it."
        )


class _Client:
    """Stand-in SDK client class with one introspectable operation."""

    def Move(self, vx: float, vy: float = 0.0) -> int:  # noqa: N802 - SDK spelling
        return 0


def _unintrospectable_client() -> type:
    def _op(self: Any, x: int) -> int:
        return x

    # inspect.signature refuses a non-Signature __signature__ - the shape a C
    # builtin or a signature-less wrapper presents.
    _op.__signature__ = "nonsense"  # type: ignore[attr-defined]

    return type("_NoSigClient", (), {"Move": _op})


class TestDiscoveryDegradesOnlyForAnAbsentSdk:
    """The AST fallback answers for a missing SDK, never for a defect."""

    # The absent-SDK condition: the SDK is not installed, or it renamed the
    # client class the service table names.
    @pytest.mark.parametrize("absent", [ImportError("no unitree_sdk2py"), AttributeError("LocoClient")])
    def test_an_absent_sdk_falls_back_without_raising(self, monkeypatch: pytest.MonkeyPatch, absent: Exception) -> None:
        mod = importlib.import_module("strands_robots.tools.g1.use_unitree")

        def _raise(_qualname: str) -> Any:
            raise absent

        monkeypatch.setattr(mod, "_import_client_class", _raise)

        # No raise, and the answer comes from the AST reader.
        assert isinstance(mod.list_operations("loco"), list)
        assert isinstance(mod.describe_operation("loco", "Move"), dict)

    @pytest.mark.parametrize("defect", [RuntimeError("bug in the reader"), KeyError("service table")])
    def test_a_defect_in_the_introspection_path_propagates(
        self, monkeypatch: pytest.MonkeyPatch, defect: Exception
    ) -> None:
        mod = importlib.import_module("strands_robots.tools.g1.use_unitree")

        def _raise(_qualname: str) -> Any:
            raise defect

        monkeypatch.setattr(mod, "_import_client_class", _raise)

        # Degrading here would answer from a possibly-stale on-disk source, or
        # return [] for "this service has no operations", with no signal.
        with pytest.raises(type(defect)):
            mod.list_operations("loco")
        with pytest.raises(type(defect)):
            mod.describe_operation("loco", "Move")

    def test_the_tool_boundary_reports_a_discovery_defect_instead_of_hiding_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = importlib.import_module("strands_robots.tools.g1.use_unitree")

        def _raise(_qualname: str) -> Any:
            raise RuntimeError("bug in the reader")

        monkeypatch.setattr(mod, "_import_client_class", _raise)

        # The tool still does not raise past dispatch; it names the defect
        # rather than returning a differently-sourced answer as a success.
        res = mod.use_unitree("meta", "describe_operation", {"service_name": "loco", "operation_name": "Move"})
        assert res["status"] == "error"
        assert "bug in the reader" in res["message"]

    def test_an_unintrospectable_operation_is_not_reported_as_parameterless(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mod = importlib.import_module("strands_robots.tools.g1.use_unitree")
        monkeypatch.setattr(mod, "_import_client_class", lambda _q: _unintrospectable_client())
        monkeypatch.setattr(mod, "_ast_methods_for_class", lambda _q: {"Move": ["x"]})

        out = mod.describe_operation("loco", "Move")

        # Claiming ``parameters: []`` from inspect would tell an agent the
        # operation takes no arguments; the AST reader knows it takes one.
        assert out["source"] == "ast"
        assert [p["name"] for p in out["parameters"]] == ["x"]

    def test_an_introspectable_operation_still_answers_from_inspect(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mod = importlib.import_module("strands_robots.tools.g1.use_unitree")
        monkeypatch.setattr(mod, "_import_client_class", lambda _q: _Client)

        out = mod.describe_operation("loco", "Move")

        assert out["source"] == "inspect"
        assert [p["name"] for p in out["parameters"]] == ["vx", "vy"]
        assert out["signature"].startswith("Move(")
