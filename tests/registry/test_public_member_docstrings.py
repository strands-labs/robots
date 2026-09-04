# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The registry public API must document every public member.

The ``strands_robots.registry`` package is the single source of truth that
callers use to discover and resolve robots and policy providers:
:func:`~strands_robots.registry.robots.get_robot` /
:func:`~strands_robots.registry.robots.resolve_name` /
:func:`~strands_robots.registry.robots.list_robots` answer "which robots exist
and what are their aliases", :func:`~strands_robots.registry.policies.resolve_policy`
and friends answer the same for policy providers,
:mod:`~strands_robots.registry.discovery` auto-discovers installed
``robot_descriptions`` models, :mod:`~strands_robots.registry.loader` owns JSON
load + mtime hot-reload, and :mod:`~strands_robots.registry.user_registry` lets
users register their own robots. Agents and integrators drive this surface
directly off its docstrings, so every public entry point must state its own
behavior rather than leave a caller guessing.

This guard walks the package modules by AST (no import, so it never needs any
optional discovery dependency installed) and fails if any public class, public
method/property, or public module-level function lacks a docstring. It pins the
discovered public surface so a dropped or renamed symbol trips the completeness
guard instead of silently shrinking the scan.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.registry as registry_pkg

_PACKAGE_DIR = Path(registry_pkg.__file__).parent

# The public-API modules of the package (``__init__`` only re-exports). All are
# scanned by AST, so the walk needs no optional dependency installed.
_MODULES = ("discovery.py", "loader.py", "policies.py", "robots.py", "user_registry.py")

# The registry package exposes no public classes -- it is a set of module-level
# functions over JSON-backed state. Pinned empty so a future public class trips
# the completeness guard and gets its own docstring coverage here.
_EXPECTED_CLASSES: set[str] = set()

# Every public module-level function the package exposes, keyed
# ``module.py::func``. Pinned so a refactor that drops or renames a function
# trips the completeness guard instead of silently shrinking the scan.
_EXPECTED_FUNCTIONS = {
    "discovery.py::descriptions_module",
    "discovery.py::discover_robot",
    "discovery.py::discover_urdf_path",
    "discovery.py::invalidate_cache",
    "discovery.py::is_discoverable",
    "discovery.py::is_urdf_discoverable",
    "discovery.py::list_discoverable",
    "discovery.py::list_urdf_discoverable",
    "discovery.py::urdf_descriptions_module",
    "loader.py::invalidate_cache",
    "loader.py::reload",
    "policies.py::build_policy_kwargs",
    "policies.py::get_policy_provider",
    "policies.py::import_policy_class",
    "policies.py::list_policy_aliases",
    "policies.py::list_policy_providers",
    "policies.py::port_reading_providers",
    "policies.py::provider_reads_a_port",
    "policies.py::resolve_policy",
    "robots.py::format_robot_table",
    "robots.py::get_driver",
    "robots.py::get_hardware_type",
    "robots.py::get_robot",
    "robots.py::has_hardware",
    "robots.py::has_sim",
    "robots.py::list_aliases",
    "robots.py::list_robots",
    "robots.py::list_robots_by_category",
    "robots.py::resolve_name",
    "user_registry.py::get_user_robots",
    "user_registry.py::parse_user_robots",
    "user_registry.py::list_user_robots",
    "user_registry.py::register_robot",
    "user_registry.py::unregister_robot",
    "user_registry.py::user_registry_source",
}


def _module_tree(module: str) -> ast.Module:
    """Parse one package module into an AST (no import)."""
    source_file = _PACKAGE_DIR / module
    return ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))


def _public_members_without_docstring(class_node: ast.ClassDef) -> list[str]:
    """Return names of public methods/properties in the class body lacking a docstring.

    Dunder methods (``__init__`` and friends) are out of scope: their contract
    is documented on the class docstring itself.
    """
    offenders: list[str] = []
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        if ast.get_docstring(node) is None:
            offenders.append(node.name)
    return offenders


def _public_classes() -> dict[str, ast.ClassDef]:
    """Map ``module.py::ClassName`` -> ClassDef for every public class in the modules."""
    classes: dict[str, ast.ClassDef] = {}
    for module in _MODULES:
        for node in _module_tree(module).body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                classes[f"{module}::{node.name}"] = node
    return classes


def _public_functions() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    """Map ``module.py::func`` -> FunctionDef for every public module-level function."""
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for module in _MODULES:
        for node in _module_tree(module).body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_"):
                funcs[f"{module}::{node.name}"] = node
    return funcs


def test_modules_define_expected_public_surface() -> None:
    """Guard: the scan actually found the classes and functions it protects."""
    assert set(_public_classes()) == _EXPECTED_CLASSES, set(_public_classes())
    assert set(_public_functions()) == _EXPECTED_FUNCTIONS, set(_public_functions())


def test_public_classes_and_members_have_docstrings() -> None:
    offenders: dict[str, list[str]] = {}
    for qualname, node in _public_classes().items():
        missing = _public_members_without_docstring(node)
        if ast.get_docstring(node) is None:
            missing = ["<class docstring>", *missing]
        if missing:
            offenders[qualname] = missing
    assert not offenders, (
        "Every public class in strands_robots.registry -- and every public "
        "method/property it defines -- must have a docstring describing its "
        "behavior. Undocumented members: " + repr(offenders)
    )


def test_public_module_functions_have_docstrings() -> None:
    offenders = [qualname for qualname, node in _public_functions().items() if ast.get_docstring(node) is None]
    assert not offenders, (
        "Every public module-level function in strands_robots.registry must "
        "have a docstring so callers can drive robot/policy resolution off the "
        "API text. Undocumented functions: " + repr(offenders)
    )
