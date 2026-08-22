"""A test may not leave ``sys.modules`` missing a module a sibling patches.

Removing an entry from ``sys.modules`` and not putting it back does not undo
an import - it *orphans* every reference already bound to that module. A test
module that does ``import boto3`` at collection time keeps the original module
object; once the entry is gone, the next ``import boto3`` executes the package
again and returns a **different** object. So::

    monkeypatch.setattr(boto3, "client", fake)   # patches the orphan
    exec(deployed_source)                        # its `import boto3` gets the
                                                 # fresh module - the real SDK

Measured consequence, in the ordering the full suite actually collects:
``tests/mesh/test_iot_camera_offload.py`` used to end
``test_boto3_missing_returns_none`` with a bare ``sys.modules.pop("boto3",
None)``. Three tests in ``tests/mesh/test_mesh_role_attribute_is_reserved.py``
then reached the real SDK instead of their double and failed with
``botocore.exceptions.NoRegionError`` - a unit test attempting a real AWS call,
and one that passes in isolation and on any host where a default region happens
to be configured. The pop was also redundant: the ``builtins.__import__`` block
on the line above already makes the function-scope ``import boto3`` raise
``ImportError``, which is what that test asserts, and ``monkeypatch`` undoes it.

The rule graded here is derived from the tree rather than listed:

* **Protected** is every module a test file binds with a module-level ``import``
  *and* then patches an attribute on (``monkeypatch.setattr(<binding>, ...)``).
  Those are exactly the references a removal can orphan - a file that only
  *reads* ``mod.attr`` is unaffected by getting a fresh copy.
* **Reported** is a removal of a protected module with no restoration in the
  same function.

It is deliberately one-directional and under-reports rather than over-reports:

* Only a **literal** key is graded. A removal whose key is a variable (a loop
  purging ``mujoco*``, say) is out of reach of a static read and is not claimed.
* Every name bound to ``sys`` in the file is followed, so an aliased
  ``import sys as _sys`` is graded on both sides of the rule - four files use
  that spelling, all of them with the restoring idiom.
* Any ``finally``, ``patch.dict``, ``monkeypatch.setitem`` or re-assignment in
  the same function counts as restoring, without checking that it restores the
  same key.
* Purging a module **no test patches** stays legal. That is a deliberate
  cache-invalidation idiom here - ``tests/policies/lerobot_local/
  test_resolution.py`` drops ``lerobot.*`` to force re-registration, and
  ``tests/simulation/test_policy_runner.py`` drops the runner to measure its
  import graph - and neither can orphan a patched reference.

``monkeypatch.setitem(sys.modules, name, None)`` is the idiom for "make
``import name`` raise ``ImportError``": it has the same effect and it restores.
``tests/mesh/test_iot_camera_offload.py`` uses it for ``cv2`` in the same file.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

import strands_robots

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_TEST_TREES = ("tests", "tests_integ")

#: A protected set smaller than this means the scan stopped reaching the tree.
_MINIMUM_PROTECTED = 10


def _module_level_bindings(tree: ast.Module) -> dict[str, str]:
    """Map each name bound by a module-level ``import X`` to its dotted path."""
    bindings: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = alias.name
    return bindings


def _graded_files() -> list[Path]:
    """Every Python file under the test trees."""
    return sorted(p for tree in _TEST_TREES for p in (_REPO_ROOT / tree).rglob("*.py"))


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):  # pragma: no cover - defensive
        return None


def _patched_module_level_imports(tree: ast.Module) -> set[str]:
    """Dotted names this module binds at import time and patches attributes on."""
    bindings = _module_level_bindings(tree)
    if not bindings:
        return set()
    patched: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "setattr"
            and node.args
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in bindings
        ):
            dotted = bindings[node.args[0].id]
            if dotted.split(".")[0] not in sys.stdlib_module_names:
                patched.add(dotted)
    return patched


def protected_modules() -> dict[str, set[str]]:
    """Modules whose identity a removal can orphan, and the files that patch them."""
    protected: dict[str, set[str]] = {}
    for path in _graded_files():
        tree = _parse(path)
        if tree is None:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for dotted in _patched_module_level_imports(tree):
            protected.setdefault(dotted, set()).add(rel)
    return protected


def _sys_aliases(tree: ast.Module) -> set[str]:
    """Every name bound to the ``sys`` module in this file, at any scope."""
    aliases = {"sys"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(alias.asname or "sys" for alias in node.names if alias.name == "sys")
    return aliases


def _own_scope_removals(fn: ast.FunctionDef | ast.AsyncFunctionDef, registries: set[str]) -> list[tuple[int, str]]:
    """``(lineno, key)`` for each literal-key removal in *fn*'s own scope."""
    found: list[tuple[int, str]] = []

    def visit(node: ast.AST, *, top: bool = False) -> None:
        if not top and isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
            return
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pop"
            and ast.unparse(node.func.value) in registries
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            found.append((node.lineno, node.args[0].value))
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and ast.unparse(target.value) in registries
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    found.append((node.lineno, target.slice.value))
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in fn.body:
        visit(statement, top=True)
    return found


def _restores(fn: ast.FunctionDef | ast.AsyncFunctionDef, registries: set[str]) -> bool:
    """Whether *fn* puts something back. Permissive on purpose - see the module docstring."""
    source = ast.unparse(fn)
    if "finally" in source or "patch.dict" in source:
        return True
    return any(
        f"setitem({registry}" in source
        or f"{registry}.update" in source
        or (f"{registry}[" in source and "] =" in source)
        for registry in registries
    )


def unrestored_removals(tree: ast.Module) -> list[tuple[int, str, str]]:
    """``(lineno, function, key)`` for each literal removal *tree* never undoes."""
    registries = {f"{alias}.modules" for alias in _sys_aliases(tree)}
    reported: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) or _restores(node, registries):
            continue
        reported.extend((lineno, node.name, key) for lineno, key in _own_scope_removals(node, registries))
    return reported


def orphaning_removals() -> list[str]:
    """Every removal of a protected module the removing function does not undo."""
    protected = protected_modules()
    offenders: list[str] = []
    for path in _graded_files():
        tree = _parse(path)
        if tree is None:
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for lineno, function, key in unrestored_removals(tree):
            if key in protected:
                holders = ", ".join(sorted(protected[key]))
                offenders.append(f"{rel}:{lineno} in {function}() removes {key!r}, which is patched by {holders}")
    return offenders


class TestNoRemovalOrphansAPatchedModule:
    """The rule."""

    def test_no_protected_module_is_removed_without_being_restored(self) -> None:
        offenders = orphaning_removals()
        assert offenders == [], (
            "a test removes a module a sibling test module patches attributes on, and "
            "does not put it back - the sibling's reference is orphaned, so its patch "
            "is invisible to the next import and the real package is used instead. "
            "Use monkeypatch.setitem(sys.modules, name, None) to make `import name` "
            "raise ImportError with restoration, or restore the entry in a finally:\n  " + "\n  ".join(offenders)
        )

    def test_the_protected_set_is_derived_from_the_test_tree(self) -> None:
        """So a clean result means the scan looked, rather than found nothing to look at."""
        protected = protected_modules()
        assert len(protected) >= _MINIMUM_PROTECTED, (
            f"only {len(protected)} modules read as protected; the scan is no longer "
            f"reaching {_TEST_TREES} under {_REPO_ROOT}"
        )
        assert "boto3" in protected, (
            "boto3 is bound at module level and attribute-patched by the IoT fan-out tests, "
            f"so it must be protected; got {sorted(protected)}"
        )

    def test_purging_a_module_no_test_patches_stays_legal(self) -> None:
        """The cache-invalidation idiom is not reported - it cannot orphan a patch."""
        protected = protected_modules()
        purges = [
            "lerobot.policies.vla_jepa.processor_vla_jepa",
            "strands_robots.simulation.policy_runner",
        ]
        assert [key for key in purges if key in protected] == [], (
            "these are deliberately dropped to force a re-import; if a test starts "
            "patching one through a module-level binding the rule reaches it, but "
            f"today it must not: {sorted(protected)}"
        )


class TestTheScanIsSpecific:
    """Planted sources, so a clean tree means the rule works rather than accepts anything."""

    def test_an_unrestored_literal_removal_is_reported(self) -> None:
        source = "\n".join(
            [
                "import sys",
                "def test_x(monkeypatch):",
                "    sys.modules.pop('boto3', None)",
            ]
        )
        assert unrestored_removals(ast.parse(source)) == [(3, "test_x", "boto3")]

    def test_a_del_statement_is_reported_too(self) -> None:
        source = "\n".join(["import sys", "def test_x():", "    del sys.modules['boto3']"])
        assert unrestored_removals(ast.parse(source)) == [(3, "test_x", "boto3")]

    def test_a_restored_removal_is_accepted(self) -> None:
        source = "\n".join(
            [
                "import sys",
                "def test_x():",
                "    held = sys.modules.pop('boto3')",
                "    try:",
                "        pass",
                "    finally:",
                "        sys.modules['boto3'] = held",
            ]
        )
        assert unrestored_removals(ast.parse(source)) == []

    def test_the_restoring_monkeypatch_idiom_is_accepted(self) -> None:
        source = "\n".join(
            ["import sys", "def test_x(monkeypatch):", "    monkeypatch.setitem(sys.modules, 'boto3', None)"]
        )
        assert unrestored_removals(ast.parse(source)) == []

    def test_an_aliased_sys_is_followed_on_both_sides(self) -> None:
        """Four files spell it ``import sys as _sys``; the rule must not lose them."""
        removal = "\n".join(["import sys as _sys", "def test_x():", "    _sys.modules.pop('boto3', None)"])
        restored = "\n".join(
            [
                "import sys as _sys",
                "def test_x(monkeypatch):",
                "    monkeypatch.setitem(_sys.modules, 'boto3', None)",
            ]
        )
        assert unrestored_removals(ast.parse(removal)) == [(3, "test_x", "boto3")]
        assert unrestored_removals(ast.parse(restored)) == []

    def test_a_dynamic_key_is_not_claimed(self) -> None:
        """A key a static read cannot resolve is out of scope rather than guessed at."""
        source = "\n".join(
            [
                "import sys",
                "def test_x():",
                "    for name in [m for m in sys.modules if m.startswith('mujoco')]:",
                "        del sys.modules[name]",
            ]
        )
        assert unrestored_removals(ast.parse(source)) == []

    def test_a_module_only_read_is_not_protected(self) -> None:
        """Reading ``mod.attr`` survives a fresh import; patching it does not."""
        reader = "\n".join(["import boto3", "def test_x():", "    assert boto3.client is not None"])
        patcher = "\n".join(
            ["import boto3", "def test_x(monkeypatch):", "    monkeypatch.setattr(boto3, 'client', None)"]
        )
        assert _patched_module_level_imports(ast.parse(reader)) == set()
        assert _patched_module_level_imports(ast.parse(patcher)) == {"boto3"}


@pytest.fixture
def probe_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """An importable throwaway module, left out of ``sys.modules`` afterwards."""
    name = "strands_orphan_probe"
    (tmp_path / f"{name}.py").write_text("value = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    try:
        yield name
    finally:
        sys.modules.pop(name, None)


class TestTheOrphaningMechanism:
    """What the rule protects against, driven on a throwaway module."""

    def test_a_removal_orphans_a_reference_taken_before_it(self, probe_module: str) -> None:
        name = probe_module
        held = __import__(name)
        assert sys.modules[name] is held, "premise: the binding is the registered module"

        del sys.modules[name]
        fresh = __import__(name)

        assert fresh is not held, "a fresh import after a removal must be a different module object"

    def test_a_patch_on_the_orphan_is_invisible_to_the_next_import(
        self, probe_module: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The exact failure mode: the double is installed somewhere nothing will look."""
        name = probe_module
        held = __import__(name)
        monkeypatch.setattr(held, "value", "double")

        del sys.modules[name]
        fresh = __import__(name)

        assert held.value == "double", "premise: the patch reached the object it was applied to"
        assert fresh.value == 1, "the fresh module does not carry the patch - the double is orphaned"
