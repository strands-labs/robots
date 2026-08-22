# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MuJoCo backend public API must document every public member.

The ``strands_robots.simulation.mujoco`` package is the default simulation
backend: :class:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine`
is a Strands ``AgentTool`` that hosts one stateful MuJoCo world, composed from
the behaviour mixins (:class:`~strands_robots.simulation.mujoco.physics.PhysicsMixin`,
:class:`~strands_robots.simulation.mujoco.rendering.RenderingMixin`,
:class:`~strands_robots.simulation.mujoco.recording.RecordingMixin`,
:class:`~strands_robots.simulation.mujoco.manipulation.ManipulationMixin`,
:class:`~strands_robots.simulation.mujoco.randomization.RandomizationMixin`) plus
the :class:`~strands_robots.simulation.mujoco.spec_builder.SpecBuilder` and the
``scene_ops`` MJCF surgery helpers. Agents drive this surface through docstrings
alone, so the ``AgentTool`` interface members (``tool_name``, ``tool_type``,
``tool_spec``, ``stream``) must state their OWN behaviour rather than lean on
the base ABC.

This guard walks the package modules by AST (no import, so it never needs the
optional ``sim-mujoco`` extra installed) and fails if any public class, public
method/property, or public module-level function lacks a docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.simulation.mujoco as mujoco_pkg

_PACKAGE_DIR = Path(mujoco_pkg.__file__).parent

# The public-API modules of the package (``__init__`` only re-exports). All are
# scanned by AST, so the walk needs no optional dependency installed.
_MODULES = (
    "backend.py",
    "manipulation.py",
    "physics.py",
    "randomization.py",
    "recording.py",
    "rendering.py",
    "scene_ops.py",
    "simulation.py",
    "spec_builder.py",
)

# Every public class the package exposes, keyed ``module.py::ClassName``. Pinned
# so a refactor that drops or renames a class trips the completeness guard
# instead of silently shrinking the scan.
_EXPECTED_CLASSES = {
    "manipulation.py::ManipulationMixin",
    "physics.py::PhysicsMixin",
    "randomization.py::RandomizationMixin",
    "recording.py::RecordingMixin",
    "rendering.py::RenderingMixin",
    "simulation.py::MuJoCoSimEngine",
    "spec_builder.py::SpecBuilder",
}

# Every public module-level function the package exposes.
_EXPECTED_FUNCTIONS = {
    "backend.py::capture_stderr_fd",
    "backend.py::filter_mujoco_attach_noise",
    "backend.py::mj_name_to_id",
    "rendering.py::no_gl_context_message",
    "scene_ops.py::actuate_robot_in_scene",
    "scene_ops.py::actuator_driven_joint_ids",
    "scene_ops.py::actuator_joint_id",
    "scene_ops.py::actuator_target_body_ids",
    "scene_ops.py::add_weld_constraint",
    "scene_ops.py::eject_body_from_scene",
    "scene_ops.py::eject_camera_from_scene",
    "scene_ops.py::eject_robot_from_scene",
    "scene_ops.py::fromto_fixed_size_components",
    "scene_ops.py::inject_camera_into_scene",
    "scene_ops.py::inject_object_into_scene",
    "scene_ops.py::inject_robot_into_scene",
    "scene_ops.py::install_compiled_model",
    "scene_ops.py::joint_drive_map",
    "scene_ops.py::patch_scene_mjcf",
    "scene_ops.py::persist_body_mass",
    "scene_ops.py::persist_geom_properties",
    "scene_ops.py::persist_world_option",
    "scene_ops.py::refresh_body_inertial_from_geometry",
    "scene_ops.py::remove_equality_constraint",
    "scene_ops.py::replace_scene_mjcf",
    "scene_ops.py::reposition_body_in_scene",
    "scene_ops.py::robot_owned_actuator_ids",
    "scene_ops.py::tendon_joint_ids",
    "spec_builder.py::material_spec_error",
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
        "Every public class in strands_robots.simulation.mujoco -- and every "
        "public method/property it defines -- must have a docstring describing "
        "its behavior (the AgentTool interface members on MuJoCoSimEngine must "
        "not lean on the base ABC's text). Undocumented members: " + repr(offenders)
    )


def test_public_module_functions_have_docstrings() -> None:
    offenders = [qualname for qualname, node in _public_functions().items() if ast.get_docstring(node) is None]
    assert not offenders, (
        "Every public module-level function in strands_robots.simulation.mujoco "
        "must have a docstring. Undocumented functions: " + repr(offenders)
    )
