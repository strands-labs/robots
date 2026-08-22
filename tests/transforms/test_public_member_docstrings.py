# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The transforms package public API must document every public member.

The :mod:`strands_robots.transforms` package is the dataset-transform provider
surface an agent drives to synthesize episode variants: the
:class:`~strands_robots.transforms.base.DatasetTransform` ABC and its
dataclasses (:class:`~strands_robots.transforms.base.TransformSpec` /
:class:`~strands_robots.transforms.base.TransformResult`), the provider factory
in :mod:`~strands_robots.transforms.factory`, the provenance helpers in
:mod:`~strands_robots.transforms.provenance`, and the two built-in backends.
Agents and integrators read these docstrings to drive the surface, so each
public class, method, property, and module-level function must state its own
behavior.

Same shape as the peer guards (``tests/training/test_public_member_docstrings.py``
and friends): the scan walks the package modules by AST (no import) and also
pins the discovered public surface, so a refactor that drops or renames a
class/function trips the guard instead of silently shrinking the scan.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.transforms as transforms_pkg

_PACKAGE_DIR = Path(transforms_pkg.__file__).parent

# Public-API modules, keyed by their path relative to the package dir. The
# re-export-only ``__init__`` is out of scope.
_MODULES = (
    "base.py",
    "cosmos_transfer.py",
    "factory.py",
    "mock.py",
    "provenance.py",
)

# Every public class the package exposes, keyed ``module.py::ClassName``.
_EXPECTED_CLASSES = {
    "base.py::TransformSpec",
    "base.py::TransformResult",
    "base.py::DatasetTransform",
    "cosmos_transfer.py::CosmosTransferTransform",
    "mock.py::MockTransform",
}

# Every public module-level function the package exposes.
_EXPECTED_FUNCTIONS = {
    "base.py::derive_variant_seed",
    "factory.py::register_transform",
    "factory.py::list_transforms",
    "factory.py::import_transform_class",
    "factory.py::create_transform",
    "provenance.py::provenance_path",
    "provenance.py::write_provenance",
    "provenance.py::load_provenance",
    "provenance.py::synthetic_episode_indices",
}


def _module_tree(module: str) -> ast.Module:
    """Parse one package module into an AST (no import)."""
    source_file = _PACKAGE_DIR / module
    return ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))


def _public_members_without_docstring(class_node: ast.ClassDef) -> list[str]:
    """Return names of public methods/properties in the class body lacking a docstring."""
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
    """Every public class and its public methods/properties carry a docstring."""
    offenders: dict[str, list[str]] = {}
    for qualname, node in _public_classes().items():
        missing = _public_members_without_docstring(node)
        if ast.get_docstring(node) is None:
            missing = ["<class docstring>", *missing]
        if missing:
            offenders[qualname] = missing
    assert not offenders, (
        "Every public class in strands_robots.transforms -- and every public "
        "method/property it defines -- must have a docstring describing its "
        "behavior. Undocumented members: " + repr(offenders)
    )


def test_public_module_functions_have_docstrings() -> None:
    """Every public module-level function carries a docstring."""
    offenders = [qualname for qualname, node in _public_functions().items() if ast.get_docstring(node) is None]
    assert not offenders, (
        "Every public module-level function in strands_robots.transforms must "
        "have a docstring. Undocumented functions: " + repr(offenders)
    )
