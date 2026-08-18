# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The hybrid-rendering public API must document every public member.

The ``strands_robots.rendering`` package is the library home of the
Gaussian-splat hybrid-render layer: the :class:`BackgroundRenderer` protocol
with its :class:`PanoramaBackground` (zero-ML-deps) and
:class:`GsplatBackground` (``sim-gs`` extra) implementations, the
:class:`HybridCompositor` depth-compare layer, the :class:`CameraParams`
camera description, and the ``encode_clip`` / ``mjpeg_frames`` media
utilities. Agents and integrators read these docstrings to drive the surface,
so each concrete override needs its own docstring rather than silently leaning
on inherited protocol text -- a ``render`` override, for instance, should state
its own depth convention and optional-dependency contract.

This guard walks the package modules by AST (no import, so it never needs the
optional ``sim-gs`` / ``imageio`` extras installed) and fails if any public
class, public method/property, or public module-level function lacks a
docstring.
"""

from __future__ import annotations

import ast
from pathlib import Path

import strands_robots.rendering as rendering_pkg

_PACKAGE_DIR = Path(rendering_pkg.__file__).parent

# The public-API modules of the package (``__init__`` only re-exports). All are
# scanned by AST, so the walk needs no optional backend installed.
_MODULES = ("backgrounds.py", "camera.py", "color.py", "compositor.py", "ibl.py", "video.py")

# Every public class the package exposes, keyed ``module.py::ClassName``. Pinned
# so a refactor that drops or renames a class trips the completeness guard
# instead of silently shrinking the scan.
_EXPECTED_CLASSES = {
    "backgrounds.py::BackgroundRenderer",
    "backgrounds.py::PanoramaBackground",
    "backgrounds.py::GsplatBackground",
    "camera.py::CameraParams",
    "compositor.py::FrameSource",
    "compositor.py::CompositeFrame",
    "compositor.py::HybridCompositor",
    "ibl.py::KeyLightEstimate",
}

# Every public module-level function the package exposes.
_EXPECTED_FUNCTIONS = {
    "backgrounds.py::gsplat_rasterizer_available",
    "backgrounds.py::gsplat_scene_names",
    "backgrounds.py::gsplat_skybox_scene_names",
    "backgrounds.py::gsplat_skybox_align_for",
    "backgrounds.py::download_gsplat_scene",
    "backgrounds.py::bake_gsplat_panorama",
    "color.py::srgb_to_linear",
    "color.py::linear_to_srgb",
    "color.py::relative_luminance",
    "compositor.py::feather_mask",
    "compositor.py::plane_depth",
    "ibl.py::render_environment_map",
    "ibl.py::bake_environment_map",
    "ibl.py::environment_map_cache_path",
    "ibl.py::derive_key_light",
    "video.py::encode_clip",
    "video.py::mjpeg_frames",
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
        "Every public class in strands_robots.rendering -- and every public "
        "method/property it defines -- must have a docstring describing its "
        "behavior (concrete overrides must not lean on inherited protocol "
        "text). Undocumented members: " + repr(offenders)
    )


def test_public_module_functions_have_docstrings() -> None:
    offenders = [qualname for qualname, node in _public_functions().items() if ast.get_docstring(node) is None]
    assert not offenders, (
        "Every public module-level function in strands_robots.rendering must "
        "have a docstring. Undocumented functions: " + repr(offenders)
    )
