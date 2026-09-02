"""Pin: the preflight roster is derived from the tree, so a new grader joins it.

Why this exists
---------------
``scripts/check_whole_tree_graders.py`` runs the tests whose input is the
*rest* of the repository rather than the file under change - the class of
check a diff-scoped ``pytest`` selector (a ``-k`` keyword, a
``tests/drivers/`` path) does not collect. Issue #2940 documents the failure
mode: two consecutive verb-port PRs (#2934, #2938) cited a green
``pytest tests/drivers/ -k g1`` in their descriptions and both landed on CI
with ``call-test-lint`` red on the exact class of grader a narrow selector
skipped.

That remedy was first written as a hand-maintained roster, and issue #3105
records the second-order failure: a grader added later is absent from a hand
list *by default*, and its absence is silent in the reassuring direction. A
branch cited a green preflight over a roster of seven while the required check
went red on ``tests/test_mesh_pacing_ticker.py``, which walks the installed
package and was never named. The preflight could not have said otherwise - it
never collected the file that failed.

So the roster is now derived: a grader whose population is the tree is one that
walks the repository root or a top-level Python area. This module grades that
derivation, in both directions, because a derivation that silently selects
nothing reads exactly like a tree with nothing to select:

- the roster covers the graders the two issues name, including #3105's live
  instance;
- the derivation is not vacuous, and each spelling of "the package root" a
  grader in this tree actually uses resolves (a resolver blind to one spelling
  is the defect #3105 describes, one layer down);
- a walk of a fixture directory or a ``tmp_path`` is *not* selected, which is
  the property that makes deriving viable at all - an earlier AST-shape scan
  was rejected because it could not tell those apart;
- every entry in ``UNDERIVABLE_GRADERS`` is genuinely invisible to the
  derivation, so that list cannot quietly grow back into the hand roster this
  replaced;
- an area held in a *loop variable* resolves, which issue #3111 records as the
  third turn of the same screw. The derivation read a ``/`` segment only as a
  string constant, and the idiom here is a tuple of area names walked one at a
  time, so 15 of the 21 modules on that spelling were unrostered - while the
  bullet above passed, because 5 of the remaining 6 were rescued incidentally
  by a second, resolvable walk elsewhere in the same file. A pin that grades
  only the graders an issue names cannot see a resolver gap that its own named
  graders survive by accident, so the spellings are graded directly.
- a root held in a *function parameter* resolves. This is the fourth turn of
  the same screw and the one the previous three make predictable: a grader that
  plants source for its own predicate factors the walk into a helper taking the
  root as an argument, so the receiver is bound per call rather than at module
  scope. Five graders were unrostered on it - three docstring-completeness
  sweeps over the installed package, the package import-cycle graph, and the
  render-gating sweep over all of ``tests/``. The asymmetry that makes such a
  module a grader is visible in its own calls: the planted call passes a
  ``tmp_path`` and resolves to nothing, the real one passes the package root.
- a root bound *inside the function that walks it* resolves. This is the fifth
  turn, and the plainest of the five: a grader that keeps its sweep beside the
  assertion it feeds binds the root where it is used. Eight graders were
  unrostered on it. Both directions are graded, because reading locals is the
  first of these widenings that could over-select - a name assigned in one
  function must not stand in for a receiver in another, and the tree carries a
  live instance of exactly that pair.

A roster only helps a caller who knows to run it. Issue #2940's failure mode
was a *green* narrow run, so the remedy is only reachable if the pre-push
instructions a contributor reads name the command - which is why
``test_agents_md_names_the_preflight_command`` reads the command's name out of
``pyproject.toml`` and requires ``AGENTS.md`` to name it. Renaming the hatch
script then fails here rather than leaving the document pointing at a command
that no longer exists.
"""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path

import pytest

_TESTS_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_ROOT.parent

_SCRIPT_PATH = _REPO_ROOT / "scripts" / "check_whole_tree_graders.py"
_spec = importlib.util.spec_from_file_location("_cwtg", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None, (
    f"pin cannot locate the preflight script at {_SCRIPT_PATH}. Either the "
    "script moved and this pin needs the matching path, or the working tree "
    "is missing scripts/check_whole_tree_graders.py entirely."
)
_cwtg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cwtg)


# The graders the three issues name by filename. #2940 named five; #3105 named
# the sixth, which a hand roster had omitted and which is the reason the roster
# is derived rather than written down; #3111 named the seventh, which the
# derivation itself could not see because it walks an area held in a loop
# variable.
_NAMED_BY_ISSUES: frozenset[str] = frozenset(
    {
        "tests/test_docstring_xref_roles_resolve.py",
        "tests/test_no_host_paths.py",
        "tests/test_dependency_audit.py",
        "tests/tools/test_agent_tool_parameter_descriptions.py",
        "tests/test_parameter_deletes_precede_the_body_they_narrow.py",
        "tests/test_mesh_pacing_ticker.py",
        "tests/test_except_tuples_state_their_real_scope.py",
    }
)


@pytest.fixture(scope="module")
def roster() -> tuple[str, ...]:
    """Every grader a preflight run of this tree collects."""
    return _cwtg.roster(_REPO_ROOT)


@pytest.fixture(scope="module")
def derived() -> tuple[str, ...]:
    """The half of the roster read off the tree."""
    return _cwtg.derive_graders(_REPO_ROOT)


def _selects(source: str, module_path: Path) -> bool:
    """Whether the derivation would call a module with ``source`` a whole-tree grader.

    :param source: Planted module source.
    :param module_path: Where the module would live, so ``__file__`` resolves.
    """
    walked = _cwtg.walked_paths(source, module_path, _REPO_ROOT)
    return bool(walked & _cwtg.walk_targets(_REPO_ROOT))


def test_the_roster_covers_every_grader_the_issues_name(roster: tuple[str, ...]) -> None:
    """The graders #2940 and #3105 name are all collected by a preflight run.

    #3105's instance is the one that matters here: it was absent from the hand
    roster, so a preflight run over that roster passed while the required
    check failed on it.
    """
    missing = _NAMED_BY_ISSUES - set(roster)
    assert not missing, (
        "a preflight run does not collect graders that issues #2940 and #3105 "
        "name by filename. Each walks the tree, so a diff-scoped pytest "
        "selector does not collect it either, and the failure mode both issues "
        "document returns unreported:\n" + "\n".join(f"  - {path}" for path in sorted(missing))
    )


def test_the_derivation_is_not_vacuous(derived: tuple[str, ...]) -> None:
    """A derivation that selected nothing would report a clean sweep.

    The floor is deliberately far below the measured count (68 at the time of
    writing): the point is to fail when the resolver breaks and selects
    almost nothing, not to pin a number that every added grader edits.
    """
    assert len(derived) >= 20, (
        f"the derivation selected only {len(derived)} whole-tree graders. It "
        "resolves each walk's receiver to a concrete path, so a change that "
        "stops resolving the spellings this tree uses makes the preflight "
        "silently collect almost nothing."
    )


@pytest.mark.parametrize(
    ("label", "root_expression"),
    [
        ("__file__ ancestry", "ROOT = pathlib.Path(__file__).resolve().parents[1]"),
        ("module __file__", "ROOT = pathlib.Path(strands_robots.__file__).resolve().parent"),
        ("inspect.getfile", "ROOT = pathlib.Path(inspect.getfile(strands_robots)).parent"),
        ("helper returning the root", "def _root():\n    return pathlib.Path(inspect.getfile(strands_robots)).parent"),
    ],
)
def test_every_spelling_of_the_root_a_grader_uses_resolves(label: str, root_expression: str) -> None:
    """Each way this tree names the repository or package root is selected.

    Graders reach the same directory three ways, and a resolver that knew only
    one would report a clean sweep over a tree using another - which is the
    shape of defect #3105 describes, one layer down. ``label`` names the
    spelling under test so a failure says which one stopped resolving.
    """
    walker = "_root()" if root_expression.startswith("def ") else "ROOT"
    source = f"import inspect\nimport pathlib\n\nimport strands_robots\n\n{root_expression}\n\nfor p in {walker}.rglob('*.py'):\n    pass\n"
    assert _selects(source, _TESTS_ROOT / "test_planted.py"), (
        f"the derivation cannot resolve a root spelled as {label}, so a grader "
        "written that way is invisible to the preflight rather than merely "
        "unrostered."
    )


@pytest.mark.parametrize(
    ("label", "walk"),
    [
        ("a fixture directory beside the test", "(pathlib.Path(__file__).parent / 'fixtures').rglob('*.py')"),
        ("a tmp_path handed in by pytest", "tmp_path.rglob('*.py')"),
        (
            "a subpackage of the installed package",
            "(pathlib.Path(strands_robots.__file__).parent / 'policies').rglob('*.py')",
        ),
    ],
)
def test_a_subject_scoped_walk_is_not_a_whole_tree_grader(label: str, walk: str) -> None:
    """Walking a fixture directory, a tmp_path or a subpackage is not selected.

    This is the property that makes deriving viable. An AST-shape scan was
    tried and rejected because a fixture glob is shape-identical to a
    repository walk; it is not value-identical, and resolving the path is what
    separates them. The subpackage case is excluded on its own merit: a
    path-scoped run over the mirroring test directory does collect it, so it
    is not in the class the preflight exists to rescue.
    """
    source = f"import pathlib\n\nimport strands_robots\n\ndef test_it(tmp_path):\n    for p in {walk}:\n        pass\n"
    assert not _selects(source, _TESTS_ROOT / "subject" / "test_planted.py"), (
        f"the derivation selected a module that only walks {label}. Every "
        "subject test that globs its own fixtures would join the preflight, "
        "which is the reason an earlier shape-based scan was rejected."
    )


@pytest.mark.parametrize(
    ("label", "grader"),
    [
        (
            "a module-level tuple walked in a comprehension",
            "_TREES = ('strands_robots', 'tests')\n\nFILES = [p for tree in _TREES for p in (ROOT / tree).rglob('*.py')]",
        ),
        (
            "a module-level tuple walked in a for statement",
            "_TREES = ('strands_robots', 'tests')\n\nfor tree in _TREES:\n    for p in (ROOT / tree).rglob('*.py'):\n        pass",
        ),
        (
            "a tuple written inline in the loop",
            "for area in ('strands_robots', 'tests'):\n    for p in (ROOT / area).rglob('*.py'):\n        pass",
        ),
        (
            "a module-level list",
            "_AREAS = ['strands_robots']\n\nFILES = [p for a in _AREAS for p in (ROOT / a).rglob('*.py')]",
        ),
    ],
)
def test_an_area_held_in_a_loop_variable_resolves(label: str, grader: str) -> None:
    """A grader that walks ``root / area`` for each name in a tuple is selected.

    This is the spelling most whole-tree graders in this tree use, and #3111
    measured 21 modules on it with 15 unrostered. The segment reaching ``/`` is
    bound per iteration rather than written as a constant, so a resolver that
    reads constants alone resolves the walk to nothing and skips the module -
    silently, and in the reassuring direction. ``label`` names the spelling so a
    failure says which one stopped resolving.
    """
    source = f"import pathlib\n\nimport strands_robots\n\nROOT = pathlib.Path(strands_robots.__file__).resolve().parents[1]\n\n{grader}\n"
    assert _selects(source, _TESTS_ROOT / "test_planted.py"), (
        f"the derivation cannot resolve an area spelled as {label}, so a grader "
        "written that way is invisible to the preflight rather than merely "
        "unrostered - which is the defect #3111 records."
    )


@pytest.mark.parametrize(
    ("label", "grader"),
    [
        (
            "a subpackage walked per backend",
            "_BACKENDS = ('mujoco', 'newton', 'isaac')\n\nFILES = [p for b in _BACKENDS for p in (PKG / 'simulation' / b).rglob('*.py')]",
        ),
        (
            "a loop over paths rather than name segments",
            "_ROOTS = [PKG, PKG / 'policies']\n\nFILES = [p for r in _ROOTS for p in (PKG / r).rglob('*.py')]",
        ),
        (
            "a loop whose iterable is computed at run time",
            "_AREAS = os.environ['AREAS'].split(',')\n\nFILES = [p for a in _AREAS for p in (PKG / a).rglob('*.py')]",
        ),
    ],
)
def test_a_loop_variable_area_that_is_not_a_top_level_area_is_not_selected(label: str, grader: str) -> None:
    """Resolving a loop variable must not widen *which* walks count as whole-tree.

    The resolver contributes candidate paths; :func:`walk_targets` still decides
    membership. The eight ``tests/simulation`` backend sweeps are the live
    control - they walk ``_SIM_PACKAGE / backend``, so they resolve and are
    still excluded, because a path-scoped run over the mirroring test directory
    collects them. A loop over values that are not literal name segments stays
    unresolved instead of contributing a partially understood walk.

    The run-time case deliberately computes its areas from the environment
    rather than from the tree. Spelling it ``sorted(PKG.iterdir())`` would not
    grade this resolver at all: that walks ``PKG`` itself, which is a top-level
    area, so the module is selected on the ``iterdir`` call alone and reads as
    a pass here on ``main`` too.
    """
    source = f"import os\nimport pathlib\n\nimport strands_robots\n\nPKG = pathlib.Path(strands_robots.__file__).resolve().parent\n\n{grader}\n"
    assert not _selects(source, _TESTS_ROOT / "simulation" / "test_planted.py"), (
        f"the derivation selected a module walking {label}. Resolving a loop "
        "variable is meant to read the same walks the tree already has, not to "
        "widen the class of walk that counts as whole-tree."
    )


@pytest.mark.parametrize(
    ("label", "grader"),
    [
        (
            "a helper called at module scope with the package root",
            "def _scan(root):\n    return [p for p in root.rglob('*.py')]\n\nFILES = _scan(PKG)",
        ),
        (
            "a helper called with the root as a keyword",
            "def _scan(root):\n    return [p for p in root.rglob('*.py')]\n\nFILES = _scan(root=PKG)",
        ),
        (
            "a helper declaring the root as its default",
            "def _scan(root=PKG):\n    return [p for p in root.rglob('*.py')]",
        ),
        (
            "a helper reached as an attribute",
            "def _scan(root):\n    return [p for p in root.rglob('*.py')]\n\n\nclass TestIt:\n"
            "    def test_it(self):\n        assert self._scan(PKG)\n",
        ),
        (
            "a parameter root combined with a loop-variable area",
            "def _scan(root):\n    for area in ('strands_robots', 'tests'):\n"
            "        for p in (root / area).rglob('*.py'):\n            pass\n\nFILES = _scan(PKG.parent)",
        ),
        (
            "a helper the module calls with a tmp_path as well as the real root",
            "def _scan(root):\n    return [p for p in root.rglob('*.py')]\n\nFILES = _scan(PKG)\n\n\n"
            "def test_planted(tmp_path):\n    assert not _scan(tmp_path)\n",
        ),
    ],
)
def test_a_root_that_arrives_as_a_parameter_resolves(label: str, grader: str) -> None:
    """A grader whose walk root is a function argument is selected.

    The last case is the shape every real instance takes: the helper exists so
    the sweep and the module's planted controls share one implementation, and
    the real call is what makes the module a grader. Reading module-level
    bindings alone resolved the receiver to nothing and skipped all six - in the
    reassuring direction, exactly as #3105 and #3111 describe one spelling
    each. ``label`` names the spelling so a failure says which one stopped
    resolving.
    """
    source = f"import pathlib\n\nimport strands_robots\n\nPKG = pathlib.Path(strands_robots.__file__).resolve().parent\n\n{grader}\n"
    assert _selects(source, _TESTS_ROOT / "test_planted.py"), (
        f"the derivation cannot resolve a root spelled as {label}, so a grader "
        "written that way is invisible to the preflight rather than merely "
        "unrostered."
    )


@pytest.mark.parametrize(
    ("label", "grader"),
    [
        (
            "a parameter no call in the module binds",
            "def _scan(root):\n    return [p for p in root.rglob('*.py')]",
        ),
        (
            "a parameter bound only to a tmp_path",
            "def _scan(root):\n    return [p for p in root.rglob('*.py')]\n\n\n"
            "def test_it(tmp_path):\n    assert not _scan(tmp_path)\n",
        ),
        (
            "a parameter bound only to a fixture directory beside the test",
            "def _scan(root):\n    return [p for p in root.rglob('*.py')]\n\n"
            "FILES = _scan(pathlib.Path(__file__).parent / 'fixtures')",
        ),
        (
            "a parameter bound only to a subpackage",
            "def _scan(root):\n    return [p for p in root.rglob('*.py')]\n\nFILES = _scan(PKG / 'policies')",
        ),
        (
            "an inner helper's parameter, where only the outer one takes the root",
            "def _outer(root):\n    def _inner(root):\n        return [p for p in root.rglob('*.py')]\n\n"
            "    return _inner(root / 'policies')\n\nFILES = _outer(PKG)",
        ),
    ],
)
def test_a_parameter_the_module_does_not_bind_to_an_area_is_not_selected(label: str, grader: str) -> None:
    """Resolving a parameter must not widen *which* walks count as whole-tree.

    The resolver contributes candidate paths; :func:`walk_targets` still
    decides membership, so a helper the module only ever hands a fixture
    directory, a ``tmp_path`` or a subpackage stays out. The unbound case is
    the safe direction stated directly: a helper whose caller lives in a
    ``conftest`` resolves to nothing rather than to a guess.

    The inner-helper case grades the attribution rather than the resolution.
    ``_outer`` is handed the package root and ``_inner`` is handed a subpackage
    of it; the walk sits in ``_inner``, so reading the *innermost* enclosing
    function is what keeps the outer call from lending its root to a walk that
    never sees it.
    """
    source = f"import pathlib\n\nimport strands_robots\n\nPKG = pathlib.Path(strands_robots.__file__).resolve().parent\n\n{grader}\n"
    assert not _selects(source, _TESTS_ROOT / "subject" / "test_planted.py"), (
        f"the derivation selected a module whose walk root is only ever {label}. "
        "Resolving a parameter is meant to read the same walks the tree already "
        "has, not to widen the class of walk that counts as whole-tree."
    )


@pytest.mark.parametrize(
    ("label", "grader"),
    [
        (
            "a local bound to a module constant",
            "def test_it():\n    root = PKG\n    for p in root.rglob('*.py'):\n        pass\n",
        ),
        (
            "a local assembled from the installed package's file",
            "def test_it():\n    root = pathlib.Path(strands_robots.__file__).parent\n"
            "    for p in root.rglob('*.py'):\n        pass\n",
        ),
        (
            "a local returned by a zero-argument module helper",
            "def _package_root():\n    return PKG\n\n\ndef test_it():\n    root = _package_root()\n"
            "    for p in root.rglob('*.py'):\n        pass\n",
        ),
        (
            "a local defined in terms of an earlier local",
            "def test_it():\n    own = pathlib.Path(strands_robots.__file__).resolve()\n    root = own.parent\n"
            "    for p in root.rglob('*.py'):\n        pass\n",
        ),
        (
            "a local inside a test method on a class",
            "class TestIt:\n    def test_it(self):\n        root = PKG\n"
            "        for p in root.rglob('*.py'):\n            pass\n",
        ),
        (
            "an enclosing function's local read by a nested one",
            "def test_it():\n    root = PKG\n\n    def _sweep():\n        return [p for p in root.rglob('*.py')]\n\n"
            "    assert _sweep() is not None\n",
        ),
        (
            "a local assembled from a parameter the module binds",
            "def _scan(base):\n    root = base / 'strands_robots'\n    return [p for p in root.rglob('*.py')]\n\n"
            "FILES = _scan(PKG.parent)",
        ),
    ],
)
def test_a_root_bound_inside_the_function_resolves(label: str, grader: str) -> None:
    """A grader whose walk root is assigned on a line of the function is selected.

    This is the fifth turn of the same screw, and the plainest: a grader that
    keeps its sweep beside the assertion it feeds has no reason to hoist the
    root to module scope, so it binds it where it is used. Reading module-level
    names and call-site arguments alone resolved the receiver to nothing, and
    eight graders were unrostered on it - among them the sweep in
    ``tests/test_mesh_state_loop_rate.py`` that grades every pacing loop in the
    package, and three sweeps whose walk is the installed package root reached
    through an imported module's ``__file__``.

    The last case is the one neither resolver settles alone: the root is
    *assembled* from a parameter, so the local and the argument have to compose.

    ``label`` names the spelling so a failure says which one stopped resolving.
    """
    source = f"import pathlib\n\nimport strands_robots\n\nPKG = pathlib.Path(strands_robots.__file__).resolve().parent\n\n{grader}\n"
    assert _selects(source, _TESTS_ROOT / "test_planted.py"), (
        f"the derivation cannot resolve a root spelled as {label}, so a grader "
        "written that way is invisible to the preflight rather than merely "
        "unrostered."
    )


@pytest.mark.parametrize(
    ("label", "grader"),
    [
        (
            "a local in a different function than the walk",
            "def test_reads_the_tree():\n    root = PKG\n    assert root.is_dir()\n\n\n"
            "def test_walks_a_fixture(tmp_path):\n    root = tmp_path\n"
            "    for p in root.rglob('*.py'):\n        pass\n",
        ),
        (
            "a local rebound to a subpackage closer to the walk",
            "def test_it():\n    root = PKG\n\n    def _sweep():\n        root = PKG / 'policies'\n"
            "        return [p for p in root.rglob('*.py')]\n\n    assert _sweep() is not None\n",
        ),
        (
            "a local bound to a fixture directory beside the test",
            "def test_it():\n    root = pathlib.Path(__file__).parent / 'fixtures'\n"
            "    for p in root.rglob('*.py'):\n        pass\n",
        ),
        (
            "a local bound to a subpackage",
            "def test_it():\n    root = PKG / 'policies'\n    for p in root.rglob('*.py'):\n        pass\n",
        ),
    ],
)
def test_a_local_the_walk_cannot_see_lends_nothing(label: str, grader: str) -> None:
    """Reading locals must not lend a name across functions that do not share it.

    The first case is the live shape this separation is measured against.
    ``tests/policies/curobo/test_action_horizon_domain.py`` binds the repository
    root in one test method - to *read* a file, not to walk one - and walks a
    subpackage in another. Reading every assignment in the module would let the
    first method's root stand in for the second method's receiver and select the
    file; reading only the assignments the enclosing functions own leaves it
    out, which is correct, because a path-scoped run over
    ``tests/policies/`` does collect it.

    The second case grades the same attribution one scope in: a nested function
    that rebinds the name resolves to its own value, as Python does.
    """
    source = f"import pathlib\n\nimport strands_robots\n\nPKG = pathlib.Path(strands_robots.__file__).resolve().parent\n\n{grader}\n"
    assert not _selects(source, _TESTS_ROOT / "subject" / "test_planted.py"), (
        f"the derivation selected a module whose walk root is only ever {label}. "
        "Resolving a local is meant to read the same walks the tree already has, "
        "not to widen the class of walk that counts as whole-tree."
    )


@pytest.mark.parametrize(
    ("label", "grader"),
    [
        (
            "a class the named module defines",
            "from strands_robots.simulation.base import SimEngine\n\n"
            "ROOT = pathlib.Path(inspect.getfile(SimEngine)).parents[1]\n\n"
            "for p in ROOT.rglob('*.py'):\n    pass\n",
        ),
        (
            "a function the named module defines",
            "from strands_robots.utils import sequence_length\n\n"
            "ROOT = pathlib.Path(inspect.getfile(sequence_length)).parent\n\n"
            "for p in ROOT.rglob('*.py'):\n    pass\n",
        ),
        (
            "a symbol read through getsourcefile",
            "from strands_robots.utils import sequence_length\n\n"
            "ROOT = pathlib.Path(inspect.getsourcefile(sequence_length)).parent\n\n"
            "for p in ROOT.rglob('*.py'):\n    pass\n",
        ),
        (
            "a symbol bound under an alias",
            "from strands_robots.simulation.base import SimEngine as Engine\n\n"
            "ROOT = pathlib.Path(inspect.getfile(Engine)).parents[1]\n\n"
            "for p in ROOT.rglob('*.py'):\n    pass\n",
        ),
    ],
)
def test_a_root_derived_from_an_imported_symbol_resolves(label: str, grader: str) -> None:
    """A root derived from an imported class or function is selected.

    Deriving the root from a symbol rather than from a path literal is what
    this repository asks a grader to do, and ``inspect.getfile`` answers the
    same fact for a class, a function or a module - the file the object was
    loaded from. Reading only the module spelling resolved the symbol ones to
    nothing, which skips the grader in the reassuring direction rather than
    reporting it unrostered. ``label`` names the spelling so a failure says
    which one stopped resolving.
    """
    source = f"import inspect\nimport pathlib\n\n{grader}"
    assert _selects(source, _TESTS_ROOT / "test_planted.py"), (
        f"the derivation cannot resolve a root derived from {label}, so a "
        "grader written that way is invisible to the preflight rather than "
        "merely unrostered."
    )


def test_a_symbol_the_named_module_does_not_define_is_not_resolved() -> None:
    """A re-exported symbol resolves to nothing rather than to the re-exporter.

    ``strands_robots.simulation`` re-exports ``Simulation``, which
    ``strands_robots/simulation/mujoco/simulation.py`` defines. The import
    statement does not say that, so guessing the re-exporting file would name
    the wrong directory: ``parents[1]`` of ``simulation/__init__.py`` is the
    package root - a whole-tree area - while ``parents[1]`` of the file
    ``inspect.getfile`` really answers is the ``simulation`` subpackage, which
    is deliberately not one. Resolving a symbol only through the module that
    *defines* it is what keeps this a rescue rather than a wrong answer.
    """
    source = (
        "import inspect\nimport pathlib\n\n"
        "from strands_robots.simulation import Simulation\n\n"
        "ROOT = pathlib.Path(inspect.getfile(Simulation)).parents[1]\n\n"
        "for p in ROOT.rglob('*.py'):\n    pass\n"
    )
    assert not _selects(source, _TESTS_ROOT / "subject" / "test_planted.py"), (
        "the derivation resolved a re-exported symbol to the file that "
        "re-exports it. That file is not the one the running grader walks "
        "from, so the path it yields is a guess, and here the guess is a "
        "whole-tree area where the truth is a subpackage."
    )


@pytest.mark.parametrize(
    ("label", "grader"),
    [
        (
            "a static method called through the instance",
            "class TestIt:\n    @staticmethod\n    def _package_root():\n        return PKG\n\n"
            "    def test_it(self):\n        for p in self._package_root().rglob('*.py'):\n            pass\n",
        ),
        (
            "a static method called through the class object",
            "class TestIt:\n    @staticmethod\n    def _package_root():\n        return PKG\n\n"
            "    @classmethod\n    def test_it(cls):\n"
            "        for p in cls._package_root().rglob('*.py'):\n            pass\n",
        ),
        (
            "a static method called through its class by name",
            "class TestIt:\n    @staticmethod\n    def _package_root():\n        return PKG\n\n"
            "    def test_it(self):\n        for p in TestIt._package_root().rglob('*.py'):\n            pass\n",
        ),
        (
            "a class method",
            "class TestIt:\n    @classmethod\n    def _package_root(cls):\n        return PKG\n\n"
            "    def test_it(self):\n        for p in self._package_root().rglob('*.py'):\n            pass\n",
        ),
        (
            "an instance method",
            "class TestIt:\n    def _package_root(self):\n        return PKG\n\n"
            "    def test_it(self):\n        for p in self._package_root().rglob('*.py'):\n            pass\n",
        ),
        (
            "a method whose answer a local then holds",
            "class TestIt:\n    @staticmethod\n    def _package_root():\n        return PKG\n\n"
            "    def test_it(self):\n        root = self._package_root()\n"
            "        for p in root.rglob('*.py'):\n            pass\n",
        ),
    ],
)
def test_a_root_answered_by_a_method_resolves(label: str, grader: str) -> None:
    """A grader whose root comes from a method of its test class is selected.

    A no-argument helper answering with the root already resolved at module
    scope; declared as a member of the class that uses it, it is the same
    helper reached one scope in, and reading only the plain spelling resolved
    it to nothing. The last case is the live shape - every instance in this
    tree hands the method's answer to a local and walks that - so the two
    resolvers have to compose. ``label`` names the spelling so a failure says
    which one stopped resolving.
    """
    source = f"import pathlib\n\nimport strands_robots\n\nPKG = pathlib.Path(strands_robots.__file__).resolve().parent\n\n\n{grader}"
    assert _selects(source, _TESTS_ROOT / "test_planted.py"), (
        f"the derivation cannot resolve a root answered by {label}, so a "
        "grader written that way is invisible to the preflight rather than "
        "merely unrostered."
    )


@pytest.mark.parametrize(
    ("label", "grader"),
    [
        (
            "a method that takes the root as an argument",
            "class TestIt:\n    @staticmethod\n    def _package_root(area):\n        return area\n\n"
            "    def test_it(self):\n"
            "        for p in self._package_root(PKG / 'policies').rglob('*.py'):\n            pass\n",
        ),
        (
            "a method answering with a subpackage",
            "class TestIt:\n    @staticmethod\n    def _package_root():\n        return PKG / 'policies'\n\n"
            "    def test_it(self):\n        for p in self._package_root().rglob('*.py'):\n            pass\n",
        ),
        (
            "a method answering with a fixture directory beside the test",
            "class TestIt:\n    @staticmethod\n    def _package_root():\n"
            "        return pathlib.Path(__file__).parent / 'fixtures'\n\n"
            "    def test_it(self):\n        for p in self._package_root().rglob('*.py'):\n            pass\n",
        ),
        (
            "a method with two returns, so its answer needs a branch analysis",
            "class TestIt:\n    @staticmethod\n    def _package_root(flag=True):\n"
            "        if flag:\n            return PKG\n        return PKG / 'policies'\n\n"
            "    def test_it(self):\n        for p in self._package_root().rglob('*.py'):\n            pass\n",
        ),
    ],
)
def test_a_method_that_does_not_answer_with_an_area_is_not_selected(label: str, grader: str) -> None:
    """Reading a method must not widen *which* walks count as whole-tree.

    The resolver contributes candidate paths; :func:`walk_targets` still
    decides membership, so a method answering with a subpackage or with its own
    fixture directory stays out. The first case keeps the no-argument rule
    honest - a method taking the area is not a helper naming one - and the last
    states the refusal directly: two returns is a branch this does not follow,
    so it resolves to nothing rather than to whichever return comes first.
    """
    source = f"import pathlib\n\nimport strands_robots\n\nPKG = pathlib.Path(strands_robots.__file__).resolve().parent\n\n\n{grader}"
    assert not _selects(source, _TESTS_ROOT / "subject" / "test_planted.py"), (
        f"the derivation selected a module whose walk root is only ever {label}. "
        "Reading a method is meant to read the same walks the tree already has, "
        "not to widen the class of walk that counts as whole-tree."
    )


def test_each_explicitly_named_grader_is_invisible_to_the_derivation(derived: tuple[str, ...]) -> None:
    """``UNDERIVABLE_GRADERS`` holds only graders the derivation cannot see.

    An entry the derivation already selects is a hand-maintained roster
    growing back one line at a time, which is the thing #3105 asks to end.
    """
    redundant = [path for path, _reason in _cwtg.UNDERIVABLE_GRADERS if path in set(derived)]
    assert not redundant, (
        "these graders are named explicitly in UNDERIVABLE_GRADERS but the "
        "derivation already selects them; drop the entries so the list stays "
        "a record of genuine blind spots:\n" + "\n".join(f"  - {path}" for path in redundant)
    )


def test_each_explicitly_named_grader_states_why_it_is_invisible() -> None:
    """Every explicit entry carries a reason.

    A path with no reason cannot be re-examined later: a reader cannot tell
    whether the derivation still misses it or whether the entry outlived the
    blind spot it was added for.
    """
    unexplained = [path for path, reason in _cwtg.UNDERIVABLE_GRADERS if not reason.strip()]
    assert not unexplained, (
        "these UNDERIVABLE_GRADERS entries carry no reason, so a later reader "
        "cannot tell whether the derivation still misses them:\n" + "\n".join(f"  - {path}" for path in unexplained)
    )


def test_every_roster_entry_points_at_a_real_grader_file(roster: tuple[str, ...]) -> None:
    """Every entry in the roster is a file that exists.

    Only ``UNDERIVABLE_GRADERS`` can go stale this way - the derived half is
    read off the tree - and the preflight script already refuses at runtime.
    This pin runs under the required check, catching a stale entry at review
    time rather than at the preflight run that would exit 1 on the same tree.
    """
    missing = [entry for entry in roster if not (_REPO_ROOT / entry).is_file()]
    assert not missing, (
        "the preflight roster names files that do not exist on disk; either the "
        "grader was moved or renamed and scripts/check_whole_tree_graders.py's "
        "UNDERIVABLE_GRADERS needs the matching update, or the entry was added "
        "in error:\n" + "\n".join(f"  - {entry!r}" for entry in missing)
    )


def test_the_roster_runs_each_grader_once(roster: tuple[str, ...]) -> None:
    """No grader is collected twice.

    The derived half and the explicit half are separate sources for one list,
    so an entry could appear in both; a duplicate would run the grader twice
    against the same tree for no added signal.
    """
    duplicates = sorted({entry for entry in roster if list(roster).count(entry) > 1})
    assert not duplicates, (
        "the preflight roster carries duplicate entries, so these graders run "
        "twice against the same tree:\n" + "\n".join(f"  - {entry!r}" for entry in duplicates)
    )


def test_script_has_no_arguments_beyond_the_program_name() -> None:
    """The preflight script's ``main`` refuses extra arguments.

    The input set is a property of the tree, not something a caller composes.
    A future edit that accepted an argument would break that guarantee
    silently for any caller who happened not to pass one; this pin makes such
    an edit a required-check failure.
    """
    # A ``main`` that returns 2 on any extra arg is the guarantee; check
    # the behaviour directly instead of an AST scan for signatures.
    assert _cwtg.main(["prog", "--anything"]) == 2, (
        "check_whole_tree_graders.py accepted an argument other than its "
        "program name. The input set is derived from the tree; a "
        "caller-composed one defeats the point of the preflight script."
    )


def _preflight_hatch_script_name() -> str:
    """The hatch script name that runs the preflight, read from ``pyproject.toml``.

    Derived rather than repeated so this pin grades the document against the
    command that exists, not against a string a previous edit happened to
    write down.
    """
    pyproject = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    envs = pyproject.get("tool", {}).get("hatch", {}).get("envs", {})
    script_path = _SCRIPT_PATH.relative_to(_REPO_ROOT).as_posix()
    names = sorted(
        {
            name
            for env in envs.values()
            for name, command in (env.get("scripts") or {}).items()
            if script_path in (command if isinstance(command, str) else " ".join(command))
        }
    )
    assert len(names) == 1, (
        f"expected exactly one hatch script to invoke {script_path!r}, found {names}. "
        "This pin derives the documented command from pyproject.toml; with zero or "
        "several it cannot say which name AGENTS.md should carry."
    )
    return names[0]


def test_agents_md_names_the_preflight_command() -> None:
    """The pre-push instructions name the command that runs these graders.

    The graders in the roster are the ones a diff-scoped selector structurally
    cannot collect, so an author who narrows the run to the area they changed
    gets a green result that reads exactly like a green full run - the failure
    mode issue #2940 documents, observed three times. A preflight command that
    exists but is unnamed in the document a contributor reads before pushing
    does not close it.
    """
    command = f"hatch run {_preflight_hatch_script_name()}"
    agents_md = _REPO_ROOT / "AGENTS.md"
    assert agents_md.is_file(), f"pin cannot locate {agents_md}"
    assert command in agents_md.read_text(encoding="utf-8"), (
        f"AGENTS.md does not name {command!r}. The roster in "
        f"{_SCRIPT_PATH.relative_to(_REPO_ROOT).as_posix()} only helps a caller who "
        "knows to run it, and the graders it collects are exactly the ones a "
        "path-or-keyword selector over the changed area does not. Name it in the "
        "pre-push instructions, or rename the hatch script and update both."
    )
