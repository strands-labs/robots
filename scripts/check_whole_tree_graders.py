#!/usr/bin/env python3
"""Run the whole-tree graders a diff-scoped ``pytest`` selector cannot see.

Why this exists
---------------
Several tests in this repository derive their expectation from the whole tree
rather than from any single file under change: every declared dependency is
imported or documented, no source file hard-codes a host path, every
``strands_robots.*`` cross-reference role resolves, no log string carries a
non-ASCII character, and so on.

These have one property in common: their input is the *rest* of the repository,
not the file under change. A path- or ``-k``-scoped pytest run collects none of
them (``tests/test_docstring_xref_roles_resolve.py`` is not under
``tests/drivers/`` and does not match ``-k g1``, and so on), so a narrow local
selector that reads green does not certify anything about the class of check
that would gate a real merge.

Issue strands-labs/robots#2940 documents two consecutive verb ports
(#2934, #2938) that shipped a dead ``:mod:`` role behind exactly this shape of
green narrow run: the qualified role named a sibling module that lived in a
still-open PR, so it was correct in the author's mental model of the port
series and dead on arrival in the branch's tree. Both were caught by a
reviewer, not by CI on the branch (``call-test-lint`` reads the head alone, so
being behind ``main`` does not update the graders' input).

How the roster is built
-----------------------
The roster is *derived from the tree*, not written down. A grader whose input
is the rest of the repository has a structural signature: it walks a directory
that is the repository root or one of its top-level Python areas (including the
installed ``strands_robots`` package root), then grades every file the walk
yields. :func:`derive_graders` finds those by resolving each walk's receiver to
a concrete path and keeping the modules whose walk lands on such an area.

A hand-maintained list was tried first and is what this derivation replaces.
Its failure mode is that a whole-tree grader added later is absent from it *by
default*, and the absence is silent in the reassuring direction: the preflight
passes because it never collected the grader that would have failed. Issue
#3105 records that arriving in review - a branch cited a green preflight over a
roster that did not name the grader the required check then failed on.

Deciding membership by the walk's resolved *value* is what makes the derivation
usable. An earlier attempt keyed on the AST shape alone and was rejected for a
good reason: a subject test globbing its own fixture directory is
shape-identical to one walking the repository. It is not value-identical -
``Path(__file__).parent / "fixtures"`` and ``tmp_path`` are neither the
repository root nor a top-level area - so resolving the path separates them.
Measured over this repository, the derivation selects 60 of 1461 test modules.

A walk rooted *inside* a subpackage (``strands_robots/policies/``) is
deliberately not selected: a path-scoped run over the mirroring test directory
does collect it, so it is not in the class this script exists to rescue.

:data:`UNDERIVABLE_GRADERS` carries the graders whose population is the tree
but whose walk the derivation cannot resolve - one enumerates a package with
``pkgutil.iter_modules`` and so walks no path at all. Each entry states its
reason, because a list that grows without one is the hand-maintained roster
returning under a new name; ``tests/test_whole_tree_graders_roster_is_complete``
refuses an entry the derivation *can* see.

Exit codes
----------
- ``0`` -- every grader collected and passed.
- ``1`` -- at least one grader failed, could not be collected, or the
  ``pytest`` invocation itself errored.
- ``2`` -- the invocation was misconfigured (an argument this script does not
  accept, a working directory without a ``tests/`` subtree).

The script forwards its own stdout/stderr from ``pytest`` unchanged, so the
diagnosis a reviewer would read on a red required check is the diagnosis a
preflight run prints locally, byte-for-byte.

Usage
-----
::

    # Direct
    python scripts/check_whole_tree_graders.py

    # Via hatch (installs the test extras first)
    hatch run whole-tree-check

Neither form takes arguments. This is deliberate: the input set is a property
of the tree, not something a caller composes.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

#: The package whose root counts as a whole-tree walk target.
PACKAGE = "strands_robots"

#: Directory-walking calls. ``ast.walk`` shares the ``walk`` name and is not a
#: false positive: its receiver is the ``ast`` module, which resolves to no
#: path, so it is dropped for want of a target rather than by a name exception.
_WALK_METHODS = frozenset({"rglob", "glob", "iterdir", "walk"})

#: Callables that answer "the file this module was loaded from".
_MODULE_FILE_FUNCS = frozenset({"getfile", "getsourcefile"})

#: Graders whose input is the tree but whose walk :func:`derive_graders`
#: cannot resolve, each with the reason it is invisible. The roster pin refuses
#: an entry here that the derivation can already see, so this stays a list of
#: genuine blind spots rather than growing back into a hand-maintained roster.
UNDERIVABLE_GRADERS: tuple[tuple[str, str], ...] = (
    (
        "tests/tools/test_agent_tool_parameter_descriptions.py",
        "enumerates the tool package with pkgutil.iter_modules, so it walks no path",
    ),
    (
        "tests/test_test_module_names_do_not_spell_a_tracker_coordinate.py",
        "walks (_REPO_ROOT / area) for area in a tuple, so the walked path is a loop variable",
    ),
)


def _repo_root() -> Path:
    """Return the repository root, resolved from this file's location.

    The script lives at ``<root>/scripts/check_whole_tree_graders.py``. Reading
    the parent of ``__file__`` twice reaches the root regardless of the
    caller's working directory, so a preflight invocation from a subshell in
    ``strands_robots/`` behaves the same as one from the root.
    """
    return Path(__file__).resolve().parent.parent


def walk_targets(root: Path) -> frozenset[Path]:
    """Return the directories whose walk means "the rest of the repository".

    The repository root plus each of its top-level Python areas, derived by
    listing the root rather than naming them, so an area added later counts on
    arrival.

    :param root: The repository root.
    :returns: The set of paths a whole-tree grader may walk.
    """
    targets = {root}
    for entry in sorted(root.iterdir()):
        if entry.is_dir() and not entry.name.startswith(".") and any(entry.rglob("*.py")):
            targets.add(entry)
    return frozenset(targets)


def _module_file(dotted: str, root: Path) -> Path | None:
    """Return the file a first-party dotted module name loads from, if any.

    Third-party modules resolve to ``None``: a grader walking somebody else's
    installed package is not grading this tree.
    """
    if dotted != PACKAGE and not dotted.startswith(f"{PACKAGE}."):
        return None
    relative = Path(*dotted.split("."))
    for candidate in (root / relative / "__init__.py", root / relative.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    return None


def _imported_module_files(tree: ast.Module, root: Path) -> dict[str, Path]:
    """Map each locally bound name of a first-party module to its file.

    Both import forms and any ``as`` alias are read, so a package reached under
    a local name (``from strands_robots import tools as tools_pkg``) resolves
    the same as one imported plainly. The alias matters: binding under another
    name is the ordinary way to avoid shadowing, and a scan that keyed on the
    imported spelling would be blind to exactly the modules that took care.
    """
    bindings: dict[str, Path] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # ``import a.b`` binds ``a``; ``import a.b as c`` binds ``c``.
                dotted = alias.name if alias.asname else alias.name.split(".")[0]
                found = _module_file(dotted, root)
                if found is not None:
                    bindings[alias.asname or dotted] = found
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                found = _module_file(f"{node.module}.{alias.name}", root)
                if found is not None:
                    bindings[alias.asname or alias.name] = found
    return bindings


def _resolve(node: ast.AST, module_path: Path, bindings: dict[str, Path], root: Path) -> Path | tuple[str, Path] | None:
    """Resolve a path expression to a concrete path, or ``None`` if unknown.

    Handles the spellings this repository's graders actually use: ``Path`` of a
    module file, ``.resolve()``, ``.parent``, ``.parents[n]``, ``/`` with a
    literal segment, a module-level name, and a zero-argument module-level
    helper. Anything else is unresolved, which is the safe direction - an
    unresolved walk is reported by :data:`UNDERIVABLE_GRADERS`, never silently
    counted as a whole-tree walk.

    :returns: The resolved path, a ``("parents", base)`` marker awaiting a
        subscript, or ``None``.
    """
    if isinstance(node, ast.Name):
        return bindings.get(node.id)
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "Path" and node.args:
            return _resolve_module_file(node.args[0], module_path, bindings, root)
        if name in {"resolve", "absolute"} and isinstance(func, ast.Attribute):
            return _resolve(func.value, module_path, bindings, root)
        if isinstance(func, ast.Name) and not node.args:
            return bindings.get(f"{func.id}()")
        return None
    if isinstance(node, ast.Attribute):
        if node.attr == "parent":
            base = _resolve(node.value, module_path, bindings, root)
            return base.parent if isinstance(base, Path) else None
        if node.attr == "parents":
            base = _resolve(node.value, module_path, bindings, root)
            return ("parents", base) if isinstance(base, Path) else None
        return None
    if isinstance(node, ast.Subscript):
        base = _resolve(node.value, module_path, bindings, root)
        index = node.slice
        if (
            isinstance(base, tuple)
            and isinstance(index, ast.Constant)
            and isinstance(index.value, int)
            and not isinstance(index.value, bool)
        ):
            try:
                return base[1].parents[index.value]
            except IndexError:
                return None
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        base = _resolve(node.left, module_path, bindings, root)
        segment = node.right
        if isinstance(base, Path) and isinstance(segment, ast.Constant) and isinstance(segment.value, str):
            return base / segment.value
        return None
    return None


def _resolve_module_file(node: ast.AST, module_path: Path, bindings: dict[str, Path], root: Path) -> Path | None:
    """Resolve the argument of ``Path(...)`` to a file on disk.

    Three spellings reach the same fact and all are read, because a scan that
    knew only one would report a clean sweep over a tree using another:
    ``__file__`` (this module), ``<module>.__file__``, and
    ``inspect.getfile(<module>)``.
    """
    if isinstance(node, ast.Name) and node.id == "__file__":
        return module_path
    if isinstance(node, ast.Attribute) and node.attr == "__file__" and isinstance(node.value, ast.Name):
        return bindings.get(node.value.id)
    if isinstance(node, ast.Call):
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name in _MODULE_FILE_FUNCS and len(node.args) == 1 and isinstance(node.args[0], ast.Name):
            return bindings.get(node.args[0].id)
    resolved = _resolve(node, module_path, bindings, root)
    return resolved if isinstance(resolved, Path) else None


def _module_bindings(tree: ast.Module, module_path: Path, root: Path) -> dict[str, Path]:
    """Resolve every module-level name and zero-argument helper to a path.

    Graders spell their root as a module constant (``_REPO_ROOT``) or as a
    helper (``def _package_root() -> Path``); both are read. The loop repeats
    so a constant defined in terms of an earlier one resolves regardless of
    the order the module happens to declare them in.
    """
    bindings = _imported_module_files(tree, root)
    for _ in range(3):
        for stmt in tree.body:
            candidates: list[tuple[str, ast.expr]] = []
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                candidates = [(stmt.targets[0].id, stmt.value)]
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                candidates = [(stmt.target.id, stmt.value)]
            elif isinstance(stmt, ast.FunctionDef) and not stmt.args.args:
                returned = [n.value for n in ast.walk(stmt) if isinstance(n, ast.Return) and n.value is not None]
                # One return is the whole helper's answer; several would need
                # a branch analysis this does not attempt.
                if len(returned) == 1:
                    candidates = [(f"{stmt.name}()", returned[0])]
            for name, value in candidates:
                resolved = _resolve(value, module_path, bindings, root)
                if isinstance(resolved, Path):
                    bindings[name] = resolved
    return bindings


def walked_paths(source: str, module_path: Path, root: Path) -> set[Path]:
    """Return every directory ``source`` walks that resolves to a concrete path.

    :param source: The module's source text.
    :param module_path: Where the module lives, so ``__file__`` resolves.
    :param root: The repository root.
    """
    tree = ast.parse(source)
    bindings = _module_bindings(tree, module_path, root)
    targets: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _WALK_METHODS:
            resolved = _resolve(node.func.value, module_path, bindings, root)
            if isinstance(resolved, Path):
                targets.add(resolved)
    return targets


def derive_graders(root: Path) -> tuple[str, ...]:
    """Return the test modules that walk the tree, as repo-relative posix paths.

    :param root: The repository root.
    :returns: Sorted paths of every module whose walk lands on a
        :func:`walk_targets` directory.
    """
    targets = walk_targets(root)
    found: set[str] = set()
    for path in (root / "tests").rglob("test_*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            walked = walked_paths(source, path, root)
        except SyntaxError:
            # A module pytest itself cannot collect is not this script's to
            # report; the run that collects it will say so.
            continue
        if walked & targets:
            found.add(path.relative_to(root).as_posix())
    return tuple(sorted(found))


def roster(root: Path) -> tuple[str, ...]:
    """Return every grader a preflight run collects.

    The derived set plus :data:`UNDERIVABLE_GRADERS`, deduplicated and sorted
    so the invocation is stable between runs on the same tree.

    :param root: The repository root.
    """
    return tuple(sorted({*derive_graders(root), *(path for path, _reason in UNDERIVABLE_GRADERS)}))


def main(argv: list[str] | None = None) -> int:
    """Run every grader :func:`roster` derives as a single pytest invocation.

    :param argv: The command-line arguments this script received. Only the
        program name is honored; any additional argument prints usage on
        ``stderr`` and returns ``2``. This is intentional -- see the module
        docstring's *Usage* section.
    :returns: The exit code described in the module docstring's *Exit codes*
        section.
    """
    args = argv if argv is not None else sys.argv
    if len(args) > 1:
        sys.stderr.write(
            "check_whole_tree_graders.py takes no arguments; the grader roster "
            "is derived from the tree (see module docstring).\n"
        )
        return 2

    root = _repo_root()
    if not (root / "tests").is_dir():
        sys.stderr.write(
            f"check_whole_tree_graders.py: no 'tests/' directory under {root}. "
            "Run this from a clone of strands-labs/robots.\n"
        )
        return 2

    graders = roster(root)
    # An entry naming a file that is gone can only come from
    # UNDERIVABLE_GRADERS -- the derived half is read off the tree. Naming it
    # beats pytest's exit 4 ("usage error"), which reads like this script is
    # broken rather than like one hand-written entry is stale.
    missing = [path for path in graders if not (root / path).is_file()]
    if missing:
        sys.stderr.write("check_whole_tree_graders.py: the following graders do not exist on disk:\n")
        for path in missing:
            sys.stderr.write(f"  - {path}\n")
        sys.stderr.write(
            "Each is named in this script's UNDERIVABLE_GRADERS. Either the "
            "grader was moved or renamed and that entry needs the matching "
            "update, or it is gone and the entry should be dropped.\n"
        )
        return 1

    sys.stderr.write(
        f"check_whole_tree_graders.py: {len(graders)} graders "
        f"({len(graders) - len(UNDERIVABLE_GRADERS)} derived from the tree, "
        f"{len(UNDERIVABLE_GRADERS)} named explicitly)\n"
    )
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        # Disable coverage: the pyproject default runs a full-tree ``--cov``
        # sweep that pulls in every ``strands_robots`` import, which is
        # neither useful for a preflight check nor faithful to what the
        # required ``call-test-lint`` job does (it runs a separate coverage
        # gate).
        "--no-cov",
        # ``-p no:cacheprovider`` keeps a preflight run from writing a
        # ``.pytest_cache`` that a subsequent full run would then respect.
        "-p",
        "no:cacheprovider",
        *graders,
    ]
    result = subprocess.run(cmd, cwd=root, check=False)
    return 0 if result.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
