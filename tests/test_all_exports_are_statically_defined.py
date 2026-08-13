"""Every name a module promises in ``__all__`` must be defined in that module.

``__all__`` is the module's promise about its own namespace: it is what
``from <module> import *`` binds, what a type-checker resolves an explicit
``from <module> import <name>`` against, and what a documentation build
enumerates. A name listed there but never defined is a promise nothing keeps,
and the failure is silent in the direction that matters -- the module imports
fine, so nothing reports the gap until somebody reaches for the name.

Two ways it stays hidden here. The packages in this tree resolve heavy symbols
through a PEP 562 module-level ``__getattr__`` so that importing them does not
pull mujoco / isaacsim / torch, and a name resolved that way is invisible to
every static reader while working perfectly at runtime. And the remedy is
itself easy to apply incompletely: the convention is to mirror each lazy name
in an ``if TYPE_CHECKING:`` import, which costs nothing at runtime, so a
missing mirror changes no behaviour and shows up only as a silently widened
type.

Both instances this guard was written for were exactly that, and both named the
same class:

    strands_robots/simulation/__init__.py        MuJoCoSimEngine
    strands_robots/simulation/mujoco/__init__.py MuJoCoSimEngine, MuJoCoSimulation

Measured with mypy before the fix, ``from strands_robots.simulation import
MuJoCoSimEngine`` revealed ``Any`` while its two aliases ``Simulation`` and
``MuJoCoSimulation`` -- the same class object at runtime -- revealed the
concrete constructor signature, because those two had the ``TYPE_CHECKING``
mirror and the class's own name did not. The sibling package resolved both of
its exports to a bare ``type``. So the canonical spelling of the class was the
one spelling that carried no type information, and every attribute access,
constructor argument and return value reached through it went unchecked.

CodeQL reports **all three** of those entries as ``py/undefined-export``
(severity ``error``), so this guard is not justified by CodeQL missing any of
them -- read against the code-scanning API on ``main`` rather than inferred:

    alert 718  simulation/__init__.py:120        MuJoCoSimEngine    open 2026-07-09
    alert  15  simulation/mujoco/__init__.py:23  MuJoCoSimulation   open 2026-05-21
    alert  14  simulation/mujoco/__init__.py:22  MuJoCoSimEngine    open 2026-05-21

An earlier draft of this docstring claimed the sibling module went unreported
because it binds through ``globals()["MuJoCoSimEngine"] = _Cls`` and that the
analyzer credits such an assignment as a definition. Both halves are wrong: it
is reported, twice, and the ``globals()`` write does not suppress the rule.

What this guard adds is therefore not coverage of a blind spot but two things
CodeQL structurally cannot give here. It is **local and merge-blocking**, so a
gap reaches the author before the push instead of a reviewer afterwards -- the
same reasoning ``.github/codeql/codeql-config.yml`` gives for preferring a
local gate wherever one can express the rule. And it is a **standing contract
over every module**, not a verdict on the three sites that happen to be
flagged today: the scan below covers all 72 literal-``__all__`` modules, so the
next lazy export that ships without its mirror fails immediately.

The open-since dates are the argument for the first of those. Alerts 14 and 15
predate this guard by nearly three months and 718 by one, all three on ``main``
the whole time. An advisory alert that nobody is obliged to clear is evidently
not the mechanism that gets this class fixed.

Scope, measured on this tree rather than assumed:

- 72 modules under ``strands_robots`` declare a literal ``__all__``; all 72 are
  scanned and none is exempt.
- 1 module declares a non-literal one (``strands_robots/tools/__init__.py``,
  ``__all__ = list(_LAZY_IMPORTS.keys())``) and is out of scope by
  construction: a list derived from the lazy map cannot disagree with it, so
  the drift this guard exists to catch is unrepresentable there. CodeQL cannot
  evaluate that form either.
- 0 modules use a star-import, so no module's namespace is populated by a means
  this scan cannot see. A star-import added later would surface here as a
  finding, which is the safe direction: it asks for an explicit re-export.

A ``TYPE_CHECKING`` import counts as a definition, which is the point -- the
remedy must stay free at runtime, or the guard would argue for eagerly
importing the very modules the lazy loader exists to defer.
"""

import ast
from pathlib import Path

import pytest

import strands_robots

PACKAGE_ROOT = Path(strands_robots.__file__).parent

# The two modules whose gap prompted this guard. Named explicitly so that a
# scan which silently stops covering them fails instead of passing vacuously.
KNOWN_LAZY_EXPORT_MODULES = (
    "simulation/__init__.py",
    "simulation/mujoco/__init__.py",
)

# The sole module whose __all__ is built at runtime rather than written out.
DYNAMIC_ALL_MODULES = ("tools/__init__.py",)


def module_scope_bindings(tree: ast.Module) -> set[str]:
    """Every name bound at module scope, including inside ``if``/``try`` blocks.

    ``if TYPE_CHECKING:`` and ``try: ... except ImportError:`` both bind at
    module scope despite being nested statements, and both are load-bearing
    here: the first is the lazy-export convention and the second is how optional
    dependencies are probed. Recursing into them is what makes those bindings
    count.
    """
    names: set[str] = set()

    def walk_body(body: list[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    # "import a.b" binds "a"; "import a.b as c" binds "c".
                    names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for sub in ast.walk(target):
                        if isinstance(sub, ast.Name):
                            names.add(sub.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    names.add(node.target.id)
            elif isinstance(node, (ast.If, ast.Try)):
                walk_body(node.body)
                walk_body(node.orelse)
                for handler in getattr(node, "handlers", []):
                    walk_body(handler.body)
                walk_body(getattr(node, "finalbody", []))

    walk_body(tree.body)
    return names


def declared_all(tree: ast.Module) -> tuple[bool, list[str] | None]:
    """Return ``(declares_all, literal_names)`` for a module.

    ``literal_names`` is ``None`` when the module builds ``__all__`` from an
    expression, which no static reader -- this one or CodeQL -- can evaluate.
    """
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
            value = node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                try:
                    return True, list(ast.literal_eval(value))
                except ValueError:
                    return True, None
    return False, None


def undefined_exports(source: str) -> list[str]:
    """Names promised by a literal ``__all__`` that the module never binds."""
    tree = ast.parse(source)
    declares, names = declared_all(tree)
    if not declares or names is None:
        return []
    bound = module_scope_bindings(tree)
    return sorted(name for name in names if name not in bound)


def scanned_modules() -> dict[str, list[str]]:
    """Every module with a literal ``__all__``, mapped to its undefined exports."""
    found: dict[str, list[str]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        declares, names = declared_all(ast.parse(source))
        if not declares or names is None:
            continue
        found[path.relative_to(PACKAGE_ROOT).as_posix()] = undefined_exports(source)
    return found


class TestEveryExportIsDefined:
    """The contract itself, over the whole package."""

    def test_no_module_promises_a_name_it_does_not_define(self):
        offenders = {mod: missing for mod, missing in scanned_modules().items() if missing}
        assert not offenders, (
            "__all__ names with no module-scope definition. A lazily resolved "
            "symbol needs a matching 'if TYPE_CHECKING:' import so it is "
            "statically defined (this costs nothing at runtime): "
            f"{offenders}"
        )


class TestTheScanCoversWhatItClaims:
    """A scan that stops finding modules would pass while asserting nothing.

    Every case above reduces to "no offenders in the scanned set", so an empty
    or shrunken set is indistinguishable from a clean tree. These pin the
    population.
    """

    def test_the_scanned_population_is_substantial(self):
        # 72 at the time of writing; asserting a floor rather than the exact
        # count keeps ordinary growth from failing an unrelated pull request.
        assert len(scanned_modules()) >= 60

    def test_the_lazy_export_modules_are_in_the_scanned_set(self):
        scanned = scanned_modules()
        for module in KNOWN_LAZY_EXPORT_MODULES:
            assert module in scanned, f"{module} dropped out of the scan"

    def test_the_dynamic_all_module_is_not_in_the_scanned_set(self):
        # Out of scope by construction, and the reason is worth pinning: if it
        # ever gains a written-out __all__ it must join the scan rather than
        # stay silently excluded.
        scanned = scanned_modules()
        for module in DYNAMIC_ALL_MODULES:
            assert module not in scanned

    def test_no_module_uses_a_star_import(self):
        # A star-import populates a namespace in a way this scan cannot see.
        # There are none today; this fails if one appears, so the exclusion
        # above stays the only one.
        offenders = []
        for path in sorted(PACKAGE_ROOT.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names):
                    offenders.append(path.relative_to(PACKAGE_ROOT).as_posix())
                    break
        assert not offenders, f"star-imports hide module-scope bindings from this scan: {offenders}"


class TestTheDetectorItself:
    """The tree is clean, so these are what fail if the detector is weakened."""

    def test_a_missing_definition_is_reported(self):
        source = '__all__ = ["Present", "Absent"]\n\n\nclass Present:\n    pass\n'
        assert undefined_exports(source) == ["Absent"]

    def test_a_type_checking_import_counts_as_a_definition(self):
        # The whole remedy. If this stopped counting, the guard would demand a
        # runtime import and defeat the lazy loader it is protecting.
        source = (
            "from typing import TYPE_CHECKING\n\n"
            "if TYPE_CHECKING:\n"
            "    from somewhere import Thing\n\n"
            '__all__ = ["Thing"]\n'
        )
        assert undefined_exports(source) == []

    def test_an_aliased_import_binds_the_alias_not_the_original(self):
        source = (
            "from typing import TYPE_CHECKING\n\n"
            "if TYPE_CHECKING:\n"
            "    from somewhere import Engine as Alias\n\n"
            '__all__ = ["Alias", "Engine"]\n'
        )
        assert undefined_exports(source) == ["Engine"]

    def test_a_try_except_import_counts_as_a_definition(self):
        source = (
            "try:\n"
            "    from fast import Loader\n"
            "except ImportError:\n"
            "    from slow import Loader\n\n"
            '__all__ = ["Loader"]\n'
        )
        assert undefined_exports(source) == []

    def test_a_dynamic_all_is_not_inspected(self):
        # Cannot be evaluated statically, so it yields no verdict either way.
        source = '_LAZY = {"A": ("m", "A")}\n__all__ = list(_LAZY.keys())\n'
        assert undefined_exports(source) == []

    def test_a_module_without_all_is_not_inspected(self):
        assert undefined_exports("class Thing:\n    pass\n") == []

    def test_a_getattr_alone_does_not_count_as_a_definition(self):
        # The pre-fix shape of both offending modules: runtime resolution via
        # PEP 562, invisible to every static reader.
        source = (
            '__all__ = ["Thing"]\n\n\n'
            "def __getattr__(name):\n"
            "    if name == 'Thing':\n"
            "        from backing import Thing\n\n"
            "        return Thing\n"
            "    raise AttributeError(name)\n"
        )
        assert undefined_exports(source) == ["Thing"]


class TestTheReviewedFinding:
    """Reproduces the two instances this guard was written for.

    These are the cases that fail on pre-fix code, naming the exact modules and
    symbols rather than only the aggregate, so a revert is attributed rather
    than merely detected.
    """

    @pytest.mark.parametrize(
        ("module", "symbols"),
        (
            ("simulation/__init__.py", ("MuJoCoSimEngine",)),
            ("simulation/mujoco/__init__.py", ("MuJoCoSimEngine", "MuJoCoSimulation")),
        ),
    )
    def test_the_lazily_exported_engine_is_statically_defined(self, module, symbols):
        source = (PACKAGE_ROOT / module).read_text(encoding="utf-8")
        bound = module_scope_bindings(ast.parse(source))
        _, exported = declared_all(ast.parse(source))
        for symbol in symbols:
            assert exported is not None and symbol in exported, f"{module} no longer exports {symbol}"
            assert symbol in bound, (
                f"{module} exports {symbol} without defining it; add it to the "
                "'if TYPE_CHECKING:' block (CodeQL py/undefined-export)"
            )

    def test_the_engine_and_its_aliases_are_one_object_at_runtime(self):
        # The fix is static-only. Pin that it stayed that way: all three names
        # must remain the same class, resolved through the lazy loader.
        #
        # No importorskip: resolving these names imports the module defining the
        # class, whose own mujoco import is deferred to instantiation, so this
        # runs on a box without the [sim-mujoco] extra.
        from strands_robots.simulation import MuJoCoSimEngine, MuJoCoSimulation, Simulation
        from strands_robots.simulation.mujoco import MuJoCoSimEngine as BackendEngine

        assert MuJoCoSimEngine is Simulation is MuJoCoSimulation is BackendEngine
