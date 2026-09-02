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
Measured over this repository, the derivation selects 96 of 1479 test modules.

Resolving the *value* is also why the area a grader walks may be held in a loop
variable. The idiom here is a tuple of area names walked one at a time::

    _TREES = ("strands_robots", "tests", "tests_integ", "examples")
    sorted(p for tree in _TREES for p in (_REPO_ROOT / tree).rglob("*.py"))

The segment reaching ``/`` is an ``ast.Name`` bound per iteration, not a
constant, so reading constants alone resolved this to nothing and skipped the
module. Issue #3111 measured 21 modules on that spelling, 15 of them absent
from the roster; the 5 that were present were rescued incidentally by a
*second*, resolvable walk elsewhere in the same file, which is why the gap read
as healthy from the roster's own pin. #3108 is the live instance - a redundant
``except`` tuple took the required check red while this preflight passed,
never having collected the grader that failed.

:func:`_resolve_area_loop` contributes one candidate path per literal in the
iterated tuple and leaves the membership question where it already was, which
is what makes the change safe rather than merely wider.

The root may also arrive as a *function parameter*. A grader that plants source
for its own predicate factors the walk into a helper so the sweep and the
planted cases share one implementation::

    def _scan(root: Path) -> list[str]: ...       # root.rglob("*.py")
    _SURFACES = _scan(_PACKAGE_ROOT)              # the real sweep
    ... _scan(tmp_path)                           # a planted control

The receiver is then an ``ast.Name`` bound per call rather than at module
scope, so reading module-level bindings alone resolved it to nothing.
:func:`_call_site_paths` resolves it from the module's *own* calls: every
argument a call in the same module passes for that parameter, plus any default
the helper carries. The planted call contributes nothing (``tmp_path`` resolves
to no path) and the real one contributes the package root, which is exactly the
asymmetry that makes the module a grader. Measured over this repository, five
whole-tree graders were unrostered on this spelling - three docstring-
completeness sweeps over the installed package, the package import-cycle graph,
and the render-gating sweep over all of ``tests/``.

The root may also be bound *inside the function that walks it*. A grader that
keeps its sweep beside the assertion it feeds has no reason to hoist the root
to module scope::

    def test_no_publish_loop_still_paces_on_an_inflated_wait() -> None:
        root = Path(__file__).resolve().parent.parent / "strands_robots"
        for path in sorted(root.rglob("*.py")):

The receiver is an ``ast.Name`` again, bound by a statement in a function body
that neither the module-level pass nor the call-site pass reads.
:func:`_enclosing_assignments` reads it from the assignments the functions
enclosing the walk own, outermost first so a name rebound closer to the walk
wins, as Python resolves it. Eight whole-tree graders were unrostered on this
spelling - among them the sweep that grades every pacing loop in the package
and three whose root is the installed package reached through an imported
module's ``__file__``.

Two spellings remained after those, and both name the root through one more
level of indirection than the resolvers above read. The first is a *symbol*
rather than a module - deriving the root from an imported class or function is
what this repository asks for over a path literal, and ``inspect.getfile``
answers the same fact for either::

    from strands_robots.simulation.base import SimEngine
    root = pathlib.Path(inspect.getfile(SimEngine)).parents[1]

The second is a helper declared as a *member of the test class* that uses it,
which is the same no-argument helper already read at module scope, reached one
scope in::

    class TestEveryTimeoutIsBounded:
        @staticmethod
        def _package_root() -> pathlib.Path:
            return pathlib.Path(inspect.getfile(strands_robots)).parent

Ten whole-tree graders were unrostered across the two - four wire-transport
timeout sweeps and a wait-budget sweep on the method spelling, and five package
sweeps on the symbol spelling, one of which needs both. A symbol resolves only
through the module that *defines* it, checked by reading that module's own
top-level names: ``strands_robots.simulation`` re-exports ``Simulation`` from
``simulation/mujoco/simulation.py``, so resolving the import to the
re-exporting file would answer with the package root where the truth is the
``simulation`` subpackage - a wrong answer rather than a rescue, and
``test_a_symbol_the_named_module_does_not_define_is_not_resolved`` is the pin
that keeps it refused.

Scoping is what keeps that from over-selecting, and there is a live measure of
it: ``tests/policies/curobo/test_action_horizon_domain.py`` binds the
repository root in one test method - to *read* a file, not to walk one - and
walks a subpackage in another. Reading every assignment in the module would let
the first lend its root to the second and select the file; reading only the
assignments the enclosing functions own leaves it out, which is correct,
because a path-scoped run over ``tests/policies/`` collects it already.

A walk rooted *inside* a subpackage (``strands_robots/policies/``) is
deliberately not selected: a path-scoped run over the mirroring test directory
does collect it, so it is not in the class this script exists to rescue. The
eight ``tests/simulation`` backend sweeps are the live measure of that rule
holding across the loop-variable spelling - they walk ``_SIM_PACKAGE / backend``,
so they resolve now and are still excluded. A resolver change that pulled them
in would be over-selecting, so they are the control this one was measured
against.

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

#: Parsed top-level name sets, keyed by module file. A first-party module is
#: read at most once per process however many graders import a symbol from it.
_TOP_LEVEL_NAMES: dict[Path, frozenset[str]] = {}

#: Graders whose input is the tree but whose walk :func:`derive_graders`
#: cannot resolve, each with the reason it is invisible. The roster pin refuses
#: an entry here that the derivation can already see, so this stays a list of
#: genuine blind spots rather than growing back into a hand-maintained roster.
UNDERIVABLE_GRADERS: tuple[tuple[str, str], ...] = (
    (
        "tests/tools/test_agent_tool_parameter_descriptions.py",
        "enumerates the tool package with pkgutil.iter_modules, so it walks no path",
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


def _top_level_names(path: Path) -> frozenset[str]:
    """Return the names a module defines at its own top level.

    Read to decide whether an imported name is a member the module *defines*,
    which is when ``inspect.getfile`` of that member answers this file. A
    re-export defines nothing, so it is refused rather than resolved to the
    wrong file.

    :param path: The module file to read.
    """
    cached = _TOP_LEVEL_NAMES.get(path)
    if cached is not None:
        return cached
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        tree = ast.Module(body=[], type_ignores=[])
    names: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(stmt.name)
        elif isinstance(stmt, ast.Assign):
            names.update(target.id for target in stmt.targets if isinstance(target, ast.Name))
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            names.add(stmt.target.id)
    found = frozenset(names)
    _TOP_LEVEL_NAMES[path] = found
    return found


def _module_file(dotted: str, root: Path) -> Path | None:
    """Return the file a first-party dotted name loads from, if any.

    Both a module and a *member* a module defines resolve, because
    ``inspect.getfile`` answers the same fact for either - the file the object
    was loaded from - and a grader deriving its root from an imported symbol
    is the idiom this repository asks for over a path literal::

        from strands_robots.simulation.base import SimEngine
        root = pathlib.Path(inspect.getfile(SimEngine)).parents[1]

    A member resolves only when the import names the module that *defines* it,
    which is exactly when ``inspect.getfile`` agrees. A name re-exported from a
    package ``__init__`` is refused instead of resolving to the re-exporting
    file, since that file is not the one the running grader would walk from.

    Third-party names resolve to ``None``: a grader walking somebody else's
    installed package is not grading this tree.
    """
    if dotted != PACKAGE and not dotted.startswith(f"{PACKAGE}."):
        return None
    relative = Path(*dotted.split("."))
    for candidate in (root / relative / "__init__.py", root / relative.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    package, _, member = dotted.rpartition(".")
    defining = _module_file(package, root) if member else None
    if defining is not None and member in _top_level_names(defining):
        return defining
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
        if name is not None and not node.args and not node.keywords:
            # A zero-argument helper answering with the root, reached either
            # plainly (``_package_root()``) or through its class
            # (``self._package_root()``, ``cls._package_root()``). Both are the
            # same helper, so a resolver reading only the plain spelling would
            # report a clean sweep over a tree using the other.
            return bindings.get(f"{name}()")
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


#: The parameter names a method carries for its receiver rather than for a
#: caller's argument, so a helper spelled as a method still counts as taking
#: none.
_IMPLICIT_RECEIVERS = frozenset({"self", "cls"})


def _root_scope_definitions(tree: ast.Module) -> list[tuple[ast.stmt, bool]]:
    """Return the statements a module-level name or helper can be declared by.

    The module body, plus the members of every class declared there, because a
    grader that keeps its sweep in the test class it feeds declares the helper
    as a method::

        class TestTheSweep:
            @staticmethod
            def _package_root() -> pathlib.Path:
                return pathlib.Path(inspect.getfile(strands_robots)).parent

    Class *bodies* are descended into and function bodies are not: a name bound
    inside a function is local to it, which :func:`_enclosing_assignments`
    reads under the scope rules that apply there.

    :param tree: The parsed module.
    :returns: Each statement paired with whether a class body holds it.
    """
    found: list[tuple[ast.stmt, bool]] = []

    def collect(body: list[ast.stmt], in_class: bool) -> None:
        for stmt in body:
            found.append((stmt, in_class))
            if isinstance(stmt, ast.ClassDef):
                collect(stmt.body, True)

    collect(tree.body, False)
    return found


def _helper_answer(stmt: ast.stmt, in_class: bool) -> ast.expr | None:
    """Return the single expression a no-argument helper answers with, if any.

    A method's receiver parameter is not an argument a caller passes, so
    ``self``/``cls`` is discounted and an instance or class method spelling of
    the helper reads the same as a plain function.

    :param stmt: The statement to read.
    :param in_class: Whether a class body holds it.
    :returns: The returned expression, or ``None`` if this is not a helper that
        answers with one.
    """
    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    positional = [parameter.arg for parameter in [*stmt.args.posonlyargs, *stmt.args.args]]
    if in_class and positional[:1] and positional[0] in _IMPLICIT_RECEIVERS:
        positional = positional[1:]
    if positional or stmt.args.kwonlyargs:
        return None
    returned = [n.value for n in ast.walk(stmt) if isinstance(n, ast.Return) and n.value is not None]
    # One return is the whole helper's answer; several would need a branch
    # analysis this does not attempt.
    return returned[0] if len(returned) == 1 else None


def _module_bindings(tree: ast.Module, module_path: Path, root: Path) -> dict[str, Path]:
    """Resolve every module-level name and no-argument helper to a path.

    Graders spell their root as a module constant (``_REPO_ROOT``) or as a
    helper (``def _package_root() -> Path``); both are read, and the helper
    counts whether it is declared at module scope or as a member of the test
    class that uses it. The loop repeats so a constant defined in terms of an
    earlier one resolves regardless of the order the module happens to declare
    them in.
    """
    bindings = _imported_module_files(tree, root)
    definitions = _root_scope_definitions(tree)
    for _ in range(3):
        for stmt, in_class in definitions:
            candidates: list[tuple[str, ast.expr]] = []
            answer = _helper_answer(stmt, in_class)
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                candidates = [(stmt.targets[0].id, stmt.value)]
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                candidates = [(stmt.target.id, stmt.value)]
            elif answer is not None and isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                candidates = [(f"{stmt.name}()", answer)]
            for name, value in candidates:
                resolved = _resolve(value, module_path, bindings, root)
                if isinstance(resolved, Path):
                    bindings[name] = resolved
    return bindings


def _literal_segments(node: ast.expr, sequences: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """Return the path segments ``node`` enumerates, or ``()`` if it is not a literal set of them.

    A tuple, list or set of string constants answers for itself; a name answers
    from ``sequences``, which carries the module-level ones. A container with a
    single non-string element resolves to ``()`` rather than to its string
    members: a partially understood iterable would contribute *some* of the
    areas walked, and a subset is the one answer worse than none here - it
    reads as a resolved walk while omitting directories the grader covers.

    :param node: The expression a loop iterates.
    :param sequences: Module-level names already known to hold literal segments.
    """
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values = [
            element.value
            for element in node.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        ]
        return tuple(values) if values and len(values) == len(node.elts) else ()
    if isinstance(node, ast.Name):
        return sequences.get(node.id, ())
    return ()


def _segment_bindings(tree: ast.Module) -> dict[str, tuple[str, ...]]:
    """Map every loop variable that iterates literal path segments to those segments.

    Both loop forms are read - a ``for`` statement and a comprehension's
    ``for`` clause - because the graders in this tree use each, and the
    comprehension spelling is the more common of the two. A name bound by
    two different loops accumulates both sets, which is the safe direction:
    the union over-approximates the paths one walk can take, and membership is
    then decided by intersection with :func:`walk_targets`, so an extra
    candidate that is not a walk target changes no verdict.

    :param tree: The parsed module.
    :returns: Loop variable name to the segments it takes.
    """
    sequences: dict[str, tuple[str, ...]] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name, value = stmt.targets[0].id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            name, value = stmt.target.id, stmt.value
        else:
            continue
        found = _literal_segments(value, {})
        if found:
            sequences[name] = found

    bindings: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)) and isinstance(node.target, ast.Name):
            found = _literal_segments(node.iter, sequences)
            if found:
                bindings.setdefault(node.target.id, set()).update(found)
    return {name: tuple(sorted(values)) for name, values in bindings.items()}


def _resolve_area_loop(
    receiver: ast.expr,
    module_path: Path,
    bindings: dict[str, Path],
    segments: dict[str, tuple[str, ...]],
    root: Path,
) -> set[Path]:
    """Return every path a walk of ``base / <loop variable>`` reaches.

    Empty unless the receiver is exactly that shape with a resolvable base and
    a loop variable known to iterate literal segments. Anything else stays
    unresolved, which :data:`UNDERIVABLE_GRADERS` reports rather than this
    silently counting as a whole-tree walk.

    :param receiver: The expression the walk method is called on.
    :param module_path: Where the module lives, so ``__file__`` resolves.
    :param bindings: Module-level names resolved to paths.
    :param segments: Loop variables resolved to literal segments.
    :param root: The repository root.
    """
    if not (
        isinstance(receiver, ast.BinOp) and isinstance(receiver.op, ast.Div) and isinstance(receiver.right, ast.Name)
    ):
        return set()
    base = _resolve(receiver.left, module_path, bindings, root)
    if not isinstance(base, Path):
        return set()
    return {base / segment for segment in segments.get(receiver.right.id, ())}


def _function_owners(tree: ast.Module) -> dict[ast.AST, ast.AST | None]:
    """Map every node to the innermost function definition whose body holds it.

    A walk receiver spelled as a bare name may be a parameter, and *which*
    function's parameter it is decides what the module's own calls bind it to.
    Nested definitions override, so a helper defined inside a test method
    answers for its own parameters rather than for the method's.

    :param tree: The parsed module.
    :returns: Node to its innermost enclosing function, or ``None`` for a node
        at module scope.
    """
    owners: dict[ast.AST, ast.AST | None] = {}

    def visit(node: ast.AST, owner: ast.AST | None) -> None:
        inner = node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) else owner
        for child in ast.iter_child_nodes(node):
            owners[child] = inner
            visit(child, inner)

    visit(tree, None)
    return owners


def _call_site_paths(
    tree: ast.Module, module_path: Path, bindings: dict[str, Path], root: Path
) -> dict[tuple[str, str], set[Path]]:
    """Map ``(function name, parameter name)`` to the paths this module binds it to.

    A grader that plants source for its own predicate factors the walk into a
    helper taking the root as a parameter, then calls it once with the real
    area and once per planted case. The parameter is resolvable from the
    module's own calls, so those are read: every argument a call in this module
    passes for that parameter, plus any default the helper declares.

    A name matched as an attribute (``self._scan(root)``) is the same helper
    reached another way, so both spellings are read - the alternative is being
    blind to whichever spelling a grader happened to choose, which is the
    defect this resolver keeps being widened for. Two definitions sharing a
    name contribute to one entry; the union over-approximates the paths one
    call can pass, and membership is then decided by intersection with
    :func:`walk_targets`, so a candidate that is not a walk target changes no
    verdict. A helper imported from a ``conftest`` stays unresolved, which
    :data:`UNDERIVABLE_GRADERS` reports rather than this guessing.

    :param tree: The parsed module.
    :param module_path: Where the module lives, so ``__file__`` resolves.
    :param bindings: Module-level names already resolved to paths.
    :param root: The repository root.
    """
    definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            definitions.setdefault(node.name, []).append(node)

    found: dict[tuple[str, str], set[Path]] = {}

    def record(name: str, parameter: str, value: ast.expr) -> None:
        resolved = _resolve(value, module_path, bindings, root)
        if isinstance(resolved, Path):
            found.setdefault((name, parameter), set()).add(resolved)

    for name, overloads in definitions.items():
        for definition in overloads:
            args = definition.args
            positional = [*args.posonlyargs, *args.args]
            # Defaults align with the *last* positional parameters.
            for parameter, default in zip(positional[len(positional) - len(args.defaults) :], args.defaults):
                record(name, parameter.arg, default)
            for parameter, keyword_default in zip(args.kwonlyargs, args.kw_defaults):
                if keyword_default is not None:
                    record(name, parameter.arg, keyword_default)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = node.func
        called_name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")
        for definition in definitions.get(called_name, ()):
            signature = definition.args
            names = [parameter.arg for parameter in [*signature.posonlyargs, *signature.args]]
            for index, value in enumerate(node.args):
                if index < len(names):
                    record(called_name, names[index], value)
            accepted = {*names, *(parameter.arg for parameter in signature.kwonlyargs)}
            for keyword in node.keywords:
                if keyword.arg is not None and keyword.arg in accepted:
                    record(called_name, keyword.arg, keyword.value)
    return found


def _parameter_scopes(
    call: ast.Call,
    owners: dict[ast.AST, ast.AST | None],
    parameters: dict[tuple[str, str], set[Path]],
    bindings: dict[str, Path],
) -> list[dict[str, Path]]:
    """Return ``bindings`` extended with one path a parameter in scope can take.

    One scope per ``(parameter, candidate path)`` pair, so a helper called with
    both a real area and a planted directory is read as reaching each in turn.
    Empty for a walk at module scope, and for one inside a ``lambda``, which
    has no name for a call site to name.

    :param call: The walk call whose receiver did not resolve.
    :param owners: Node to its innermost enclosing function.
    :param parameters: What the module's own calls bind each parameter to.
    :param bindings: Module-level names resolved to paths.
    """
    owner = owners.get(call)
    if not isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    args = owner.args
    scopes = []
    for parameter in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
        for candidate in sorted(parameters.get((owner.name, parameter.arg), ())):
            scopes.append({**bindings, parameter.arg: candidate})
    return scopes


def _enclosing_assignments(
    call: ast.Call,
    owners: dict[ast.AST, ast.AST | None],
    scope: dict[str, Path],
    module_path: Path,
    root: Path,
) -> dict[str, Path]:
    """Return ``scope`` extended with the local names the functions around ``call`` bind to paths.

    A grader that keeps its sweep beside the assertion it feeds binds the root
    on a line of the test itself::

        def test_no_loop_still_paces_on_an_inflated_wait() -> None:
            root = Path(__file__).resolve().parent.parent / "strands_robots"
            for path in sorted(root.rglob("*.py")):

    The receiver is an ``ast.Name`` bound inside a function body, so reading
    module-level names and call-site arguments alone resolved it to nothing.

    Only the assignments the enclosing functions *own* are read, walking that
    chain outermost first so a name rebound closer to the walk wins, exactly as
    Python resolves it. A name assigned in some unrelated function is therefore
    not in scope and lends nothing - that separation is the point, since a
    module holding both a fixture-directory glob and a package sweep must not
    have the one borrow the other's root.

    :param call: The walk call whose receiver did not resolve.
    :param owners: Node to its innermost enclosing function.
    :param scope: The names already resolved, which the locals extend.
    :param module_path: Where the module lives, so ``__file__`` resolves.
    :param root: The repository root.
    """
    chain: list[ast.AST] = []
    owner = owners.get(call)
    while owner is not None:
        chain.append(owner)
        owner = owners.get(owner)
    extended = dict(scope)
    for function in reversed(chain):
        # The loop repeats so a local defined in terms of an earlier one
        # resolves regardless of how many steps the grader spells it in.
        for _ in range(3):
            for statement in ast.walk(function):
                if owners.get(statement) is not function:
                    continue
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                ):
                    name, value = statement.targets[0].id, statement.value
                elif (
                    isinstance(statement, ast.AnnAssign)
                    and isinstance(statement.target, ast.Name)
                    and statement.value is not None
                ):
                    name, value = statement.target.id, statement.value
                else:
                    continue
                resolved = _resolve(value, module_path, extended, root)
                if isinstance(resolved, Path):
                    extended[name] = resolved
    return extended


def _walk_scopes(
    call: ast.Call,
    owners: dict[ast.AST, ast.AST | None],
    parameters: dict[tuple[str, str], set[Path]],
    bindings: dict[str, Path],
    module_path: Path,
    root: Path,
) -> list[dict[str, Path]]:
    """Return every scope a walk receiver that did not resolve may be read under.

    The module's own bindings first, then one per ``(parameter, candidate)``
    pair, and each of those extended with the locals in scope at the call - so
    a root assembled from a parameter (``root = base / "strands_robots"``)
    resolves without either resolver needing to know about the other.
    """
    return [
        _enclosing_assignments(call, owners, scope, module_path, root)
        for scope in [bindings, *_parameter_scopes(call, owners, parameters, bindings)]
    ]


def _receiver_targets(
    receiver: ast.expr,
    module_path: Path,
    bindings: dict[str, Path],
    segments: dict[str, tuple[str, ...]],
    root: Path,
) -> set[Path]:
    """Return every directory a walk on ``receiver`` reaches under ``bindings``.

    Both receiver shapes a grader uses: a name or expression resolving straight
    to a path, and ``base / <loop variable>`` over literal area names.
    """
    resolved = _resolve(receiver, module_path, bindings, root)
    if isinstance(resolved, Path):
        return {resolved}
    return _resolve_area_loop(receiver, module_path, bindings, segments, root)


def walked_paths(source: str, module_path: Path, root: Path) -> set[Path]:
    """Return every directory ``source`` walks that resolves to a concrete path.

    :param source: The module's source text.
    :param module_path: Where the module lives, so ``__file__`` resolves.
    :param root: The repository root.
    """
    tree = ast.parse(source)
    bindings = _module_bindings(tree, module_path, root)
    segments = _segment_bindings(tree)
    parameters = _call_site_paths(tree, module_path, bindings, root)
    owners = _function_owners(tree)
    targets: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _WALK_METHODS:
            receiver = node.func.value
            found = _receiver_targets(receiver, module_path, bindings, segments, root)
            if found:
                targets.update(found)
                continue
            for scope in _walk_scopes(node, owners, parameters, bindings, module_path, root):
                targets.update(_receiver_targets(receiver, module_path, scope, segments, root))
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
