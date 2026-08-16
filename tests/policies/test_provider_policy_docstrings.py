# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend policy *providers* must document every public member they override.

:mod:`~strands_robots.policies.base.Policy` documents its runtime contract, and
:mod:`tests.policies.test_builtin_policy_docstrings` already pins that guard for
the dependency-free built-ins (``MockPolicy`` / ``CompositePolicy`` /
``PersistentPolicy``). The backend providers - GR00T, cuRobo, cosmos3, the two
lerobot providers, MotionBricks, Kimodo, MoveIt2, VERA and the two WBC
controllers - each override public members such as ``provider_name``,
``get_actions``, ``requires_images`` and ``config``. An agent picking a
provider reads those docstrings to learn which one it is holding and what it
needs, so every public override needs its own docstring rather than silently
leaning on the inherited one (a ``provider_name`` override still has to state
the registry key it maps to).

This guard walks the provider policy modules by AST (no import), so it never
needs any optional policy backend (``[groot]`` / ``[cosmos3]`` / ``[vera]`` /
``[moveit2]`` / ``[wbc]`` ...) installed. It descends one level into
module-level ``if`` blocks because a provider class may be defined under an
optional-dependency guard. The pinned provider set is cross-checked against the
``policies.json`` registry so that registering a new ``strands_robots.policies``
provider without documenting it trips this guard instead of shrinking the scan.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import strands_robots.policies as policies_pkg

_PACKAGE_DIR = Path(policies_pkg.__file__).parent
_REGISTRY = Path(policies_pkg.__file__).parents[1] / "registry" / "policies.json"

# Provider policy source file (relative to the policies package) -> class name.
# Pinned so a rename or a dropped provider trips the completeness guard below
# instead of silently narrowing the docstring scan.
_PROVIDER_POLICIES = {
    "groot/policy.py": "Gr00tPolicy",
    "lerobot_local/policy.py": "LerobotLocalPolicy",
    "lerobot_async/policy.py": "LerobotAsyncPolicy",
    "cosmos3/policy.py": "Cosmos3Policy",
    "moveit2/policy.py": "MoveIt2Policy",
    "curobo/policy.py": "CuroboPolicy",
    "wbc/policy.py": "WBCPolicy",
    "wbc/gait.py": "WBCGaitPolicy",
    "vera/provider.py": "VeraPolicy",
    "motionbricks/policy.py": "MotionBricksPolicy",
    "kimodo/policy.py": "KimodoPolicy",
    "protomotions/policy.py": "ProtoMotionsPolicy",
}

# Built-in policy classes documented by test_builtin_policy_docstrings; the
# ``remote`` provider's RemotePolicy lives in strands_robots.inference and is
# guarded by tests/inference. Neither is a backend provider under this scan.
_CORE_CLASSES = {"MockPolicy", "CompositePolicy", "PersistentPolicy", "RemotePolicy"}


def _iter_defs(module: ast.Module) -> list[ast.stmt]:
    """Yield class/function defs at module top level and inside its top-level ``if`` blocks.

    A provider class may sit under an ``if _HAVE_<dep>:`` / ``if TYPE_CHECKING:``
    guard so the module imports without the optional backend; the walk descends
    one level into module-level ``if`` bodies to reach it.
    """
    defs: list[ast.stmt] = []
    for node in module.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            defs.append(node)
        elif isinstance(node, ast.If):
            for inner in [*node.body, *node.orelse]:
                if isinstance(inner, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                    defs.append(inner)
    return defs


def _public_members_without_docstring(class_node: ast.ClassDef) -> list[str]:
    """Return names of public methods/properties in the class body lacking a docstring.

    Dunder and ``_private`` members are out of scope: their contract belongs on
    the class docstring or is internal.
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


def _provider_class_node(rel_path: str, class_name: str) -> ast.ClassDef:
    """Locate the pinned provider class in its source file by AST (no import)."""
    source_file = _PACKAGE_DIR / rel_path
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    for node in _iter_defs(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"{class_name} not found at top level of {rel_path}")


def test_pinned_providers_match_registry() -> None:
    """Guard: the pinned set matches every backend provider in policies.json.

    Any ``strands_robots.policies`` provider registered in ``policies.json`` that
    is not a core built-in must be pinned here, so a newly registered provider
    cannot ship without a documentation guard entry.
    """
    registry = json.loads(_REGISTRY.read_text(encoding="utf-8"))["providers"]
    registered_backend_classes = {
        info["class"]
        for info in registry.values()
        if info.get("module", "").startswith("strands_robots.policies.") and info["class"] not in _CORE_CLASSES
    }
    assert registered_backend_classes == set(_PROVIDER_POLICIES.values()), (
        "The pinned provider set drifted from policies.json. Registered backend "
        f"providers: {sorted(registered_backend_classes)}; pinned: "
        f"{sorted(_PROVIDER_POLICIES.values())}"
    )


def test_provider_policy_public_members_have_docstrings() -> None:
    """Every public method/property of a backend provider policy is documented."""
    offenders: dict[str, list[str]] = {}
    for rel_path, class_name in _PROVIDER_POLICIES.items():
        node = _provider_class_node(rel_path, class_name)
        missing = _public_members_without_docstring(node)
        if ast.get_docstring(node) is None:
            missing = ["<class docstring>", *missing]
        if missing:
            offenders[f"{rel_path}::{class_name}"] = missing
    assert not offenders, (
        "Every public method/property of a backend policy provider must have a "
        "docstring (the base Policy ABC already documents the contract; an "
        "override still states its provider-specific behavior, e.g. the "
        "registry key provider_name maps to). Undocumented members: " + repr(offenders)
    )
