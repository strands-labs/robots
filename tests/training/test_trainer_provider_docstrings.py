"""Concrete ``Trainer`` providers must document every public method/property.

The :class:`~strands_robots.training.base.Trainer` ABC documents its lifecycle
contract (``validate`` / ``prepare`` / ``train`` / ``export`` / ``status`` /
``provider_name`` / ``hardware_floor``) with rich docstrings, and the reference
:class:`~strands_robots.training.mock.MockTrainer` follows suit. The real
providers (:class:`~strands_robots.training.cosmos3.Cosmos3Trainer`,
:class:`~strands_robots.training.groot.Gr00tTrainer`,
:class:`~strands_robots.training.lerobot.LerobotTrainer`) genuinely differ per
backend - each ``validate`` checks backend-specific inputs and each ``train``
drives a different in-process training function - so every public override needs
its own docstring rather than silently leaning on the inherited one.

This guard walks the provider modules and fails if any concrete ``Trainer``
subclass defines a public method or property with no docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.training as training_pkg

_PACKAGE_DIR = Path(training_pkg.__file__).parent

# Provider modules that define a concrete Trainer subclass (mock is the
# dependency-free reference; the others are the real backends).
_PROVIDER_MODULES = ("mock.py", "cosmos3.py", "groot.py", "lerobot.py", "sagemaker.py")


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


def _concrete_trainer_classes() -> dict[str, ast.ClassDef]:
    """Map ``module.py::ClassName`` -> ClassDef for every ``*Trainer`` subclass."""
    classes: dict[str, ast.ClassDef] = {}
    for module in _PROVIDER_MODULES:
        source_file = _PACKAGE_DIR / module
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Trainer"):
                classes[f"{module}::{node.name}"] = node
    return classes


def test_provider_modules_define_concrete_trainers() -> None:
    """Guard: the scan actually found the five concrete Trainer subclasses."""
    found = set(_concrete_trainer_classes())
    assert found == {
        "mock.py::MockTrainer",
        "cosmos3.py::Cosmos3Trainer",
        "groot.py::Gr00tTrainer",
        "lerobot.py::LerobotTrainer",
        "sagemaker.py::SagemakerTrainer",
    }, found


def test_concrete_trainer_public_members_have_docstrings() -> None:
    offenders = {
        qualname: missing
        for qualname, node in _concrete_trainer_classes().items()
        if (missing := _public_members_without_docstring(node))
    }
    assert not offenders, (
        "Every public method/property of a concrete Trainer provider must have a "
        "docstring describing its provider-specific behavior (the base ABC and "
        "MockTrainer already do). Undocumented members: " + repr(offenders)
    )
