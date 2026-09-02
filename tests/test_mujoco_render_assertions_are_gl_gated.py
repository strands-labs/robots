"""A MuJoCo render-success assertion must be gated on the shared GL probe.

``Simulation.render`` returns ``{"status": "error"}`` on a host with no usable
offscreen GL context - headless without EGL/OSMesa - for a reason that has
nothing to do with whatever contract the calling test is pinning. Asserting
``render(...)["status"] == "success"`` inline therefore conflates the property
under test with a host graphics capability: it passes only where a GL context
happens to exist, and elsewhere it reports a bare ``'error' != 'success'`` that
names neither GL nor the contract.

:mod:`tests.simulation.mujoco._gl_probe` exists for exactly this and is the
convention in ten sibling modules. This guard keeps it the convention: every
render-success assertion in a module that requires ``mujoco`` must sit behind
that probe, so a test added later cannot re-open the confusion.

Scope is the mujoco requirement, not a directory. A module that asserts a render
succeeded *without* requiring ``mujoco`` renders through some other backend,
whose availability the MuJoCo probe does not describe - ``tests/simulation/newton``
gates its own render tests on a Newton-availability marker instead. Keying on the
requirement rather than on a path excludes those by construction, so this rule
needs no exemption list.

Three gating forms are accepted, all of which stop the assertion from running
without a context: the probe's marker on the test function, the same marker on
its class, or an in-function ``gl_available()`` skip. All three are in use.

The rule's own non-vacuity pin is keyed on modules, never on a count of
assertions: splitting one render call into its own gated case is what the remedy
below asks for, and must not move a number a contributor then has to chase.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

#: Envelope-returning render entry points. ``get_frame`` is excluded: it returns
#: raw arrays rather than a status envelope, so it cannot carry this assertion.
RENDER_ATTRS = frozenset({"render", "render_depth", "render_all"})

#: Modules that require ``mujoco`` and assert a render succeeded. Pinned so a
#: scan rooted somewhere unexpected fails loudly instead of reporting a clean
#: sweep over nothing.
EXPECTED_IN_SCOPE = frozenset(
    {
        "tests/simulation/mujoco/test_entity_name_lookup_type_safety.py",
        "tests/simulation/mujoco/test_remove_camera_refused_recompile.py",
        "tests/simulation/test_unhashable_entity_name_is_reported.py",
    }
)

#: Asserts a render succeeded through another backend, so the MuJoCo probe does
#: not describe its host requirement. Pinned as the discriminator this rule
#: rests on rather than as an exemption.
OTHER_BACKEND = "tests/simulation/newton/test_domain_randomization.py"


def _tests_root() -> pathlib.Path:
    """This module's own directory - the tree the rule covers."""
    return pathlib.Path(__file__).parent


def requires_mujoco(tree: ast.AST) -> bool:
    """True when *tree* calls ``pytest.importorskip("mujoco")`` anywhere."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "importorskip":
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and arg.value == "mujoco":
                return True
    return False


def render_success_assertions(tree: ast.AST) -> list[int]:
    """Line numbers of every ``<x>.render*(...)["status"] == "success"`` compare."""
    lines = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Compare) and len(node.comparators) == 1):
            continue
        right = node.comparators[0]
        if not (isinstance(right, ast.Constant) and right.value == "success"):
            continue
        subscript = node.left
        if not isinstance(subscript, ast.Subscript):
            continue
        key = subscript.slice
        if not (isinstance(key, ast.Constant) and key.value == "status"):
            continue
        call = subscript.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in RENDER_ATTRS:
            lines.append(node.lineno)
    return lines


def probe_names(tree: ast.AST) -> set[str]:
    """Local names bound by importing from the shared GL probe module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("_gl_probe"):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


Decorated = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _decorator_names(node: Decorated) -> set[str]:
    """Every decorator on *node*, as the bare name applied."""
    found: set[str] = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            found.add(target.id)
        elif isinstance(target, ast.Attribute):
            found.add(target.attr)
    return found


def _enclosing(tree: ast.AST, lineno: int) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, ast.ClassDef | None]:
    """The innermost function containing *lineno*, and its enclosing class."""
    function: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    enclosing_class: ast.ClassDef | None = None
    for node in ast.walk(tree):
        if not isinstance(node, Decorated):
            continue
        if not node.lineno <= lineno <= (node.end_lineno or node.lineno):
            continue
        if isinstance(node, ast.ClassDef):
            enclosing_class = node
        elif function is None or node.lineno > function.lineno:
            function = node
    return function, enclosing_class


def is_gated(tree: ast.AST, lineno: int, names: set[str]) -> bool:
    """True when the assertion at *lineno* cannot run without a GL context."""
    function, enclosing_class = _enclosing(tree, lineno)
    if function is None:
        return False
    if _decorator_names(function) & names:
        return True
    if enclosing_class is not None and _decorator_names(enclosing_class) & names:
        return True
    return any(isinstance(node, ast.Name) and node.id in names for node in ast.walk(function))


def survey(root: pathlib.Path) -> tuple[dict[str, list[int]], dict[str, list[int]], list[str]]:
    """Return (ungated, gated, out-of-scope) render-success assertions under *root*."""
    ungated: dict[str, list[int]] = {}
    gated: dict[str, list[int]] = {}
    other_backend: list[str] = []
    for path in sorted(root.rglob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        lines = render_success_assertions(tree)
        if not lines:
            continue
        label = path.relative_to(root.parent).as_posix()
        if not requires_mujoco(tree):
            other_backend.append(label)
            continue
        names = probe_names(tree)
        for lineno in lines:
            bucket = gated if is_gated(tree, lineno, names) else ungated
            bucket.setdefault(label, []).append(lineno)
    return ungated, gated, other_backend


def unaccounted_modules(gated: dict[str, list[int]], expected: frozenset[str]) -> tuple[set[str], set[str]]:
    """Return (in-scope modules with no gated assertion, gated modules not expected).

    Both sides are keyed on modules. A count of gated assertions is a different
    quantity from a count of modules: splitting one render assertion into its own
    case -- exactly what this guard's remedy text asks a contributor to do -- moves
    the first and not the second, so comparing them would fail a diff that did
    precisely the right thing, with no pin left to update.
    """
    return set(expected) - set(gated), set(gated) - set(expected)


class TestEveryMuJoCoRenderAssertionIsGated:
    def test_no_module_asserts_a_render_succeeded_without_the_probe(self) -> None:
        ungated, _, _ = survey(_tests_root())
        assert not ungated, (
            f"these modules assert a MuJoCo render succeeded without gating it on the shared "
            f"GL probe: {ungated}. On a headless host without EGL/OSMesa render reports an "
            f"error for a reason unrelated to the contract under test, so the assertion fails "
            f"there and names neither cause. Import requires_gl from "
            f"tests.simulation.mujoco._gl_probe and split the render assertion into its own case."
        )

    def test_every_module_the_survey_covers_contributes_a_gated_assertion(self) -> None:
        """Non-vacuity: a scan that found nothing must not read as a clean sweep.

        Keyed on modules in both directions, which is also the module-set check
        this replaces: a module whose assertion stopped being gated leaves
        ``gated`` and is reported here by name, and a scan rooted somewhere
        unexpected reports every entry as missing.
        """
        _, gated, _ = survey(_tests_root())
        missing, unexpected = unaccounted_modules(gated, EXPECTED_IN_SCOPE)
        assert not missing and not unexpected, (
            f"the survey no longer accounts for EXPECTED_IN_SCOPE: {sorted(missing)} contribute no "
            f"gated render assertion, and {sorted(unexpected)} are not listed. Add or drop the "
            f"module path. Adding a second gated assertion to a module already listed is not a "
            f"change to this pin."
        )


class TestTheScopeIsTheMujocoRequirement:
    def test_another_backends_render_assertion_is_out_of_scope(self) -> None:
        """The discriminator, pinned: no mujoco requirement, so a different probe."""
        _, _, other_backend = survey(_tests_root())
        assert OTHER_BACKEND in other_backend
        tree = ast.parse((_tests_root().parent / OTHER_BACKEND).read_text(encoding="utf-8"))
        assert render_success_assertions(tree)
        assert not requires_mujoco(tree)


_PLANTED_UNGATED = """
import pytest

mujoco = pytest.importorskip("mujoco")


def test_planted(sim):
    assert sim.render(camera_name="default")["status"] == "success"
"""

_PLANTED_GATED = """
import pytest

mujoco = pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl  # noqa: E402


@requires_gl
def test_planted(sim):
    assert sim.render(camera_name="default")["status"] == "success"
"""


_PLANTED_TWO_GATED = """
import pytest

mujoco = pytest.importorskip("mujoco")

from tests.simulation.mujoco._gl_probe import requires_gl  # noqa: E402


@requires_gl
def test_planted_one(sim):
    assert sim.render(camera_name="default")["status"] == "success"


@requires_gl
def test_planted_two(sim):
    assert sim.render_depth(camera_name="default")["status"] == "success"
"""


class TestTheSurveyDetectsWhatItClaimsTo:
    """A scanner that silently matched nothing would look like a clean tree."""

    def test_a_planted_ungated_assertion_is_reported(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(_PLANTED_UNGATED, encoding="utf-8")
        ungated, gated, _ = survey(root)
        assert list(ungated) == ["tests/test_planted.py"]
        assert not gated

    def test_the_same_assertion_behind_the_probe_is_accepted(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(_PLANTED_GATED, encoding="utf-8")
        ungated, gated, _ = survey(root)
        assert not ungated
        assert list(gated) == ["tests/test_planted.py"]

    @pytest.mark.parametrize("attr", sorted(RENDER_ATTRS))
    def test_each_render_entry_point_is_recognised(self, attr: str) -> None:
        tree = ast.parse(f'assert sim.{attr}(camera_name="c")["status"] == "success"\n')
        assert render_success_assertions(tree) == [1]

    def test_a_non_render_status_assertion_is_not_matched(self) -> None:
        """The rule is about rendering, not about every envelope in the suite."""
        tree = ast.parse('assert sim.step(n_steps=5)["status"] == "success"\n')
        assert render_success_assertions(tree) == []


class TestThePinIsKeyedOnModulesNotOnAnAssertionCount:
    """Splitting a render assertion into its own case must need no pin updated.

    That split is what this guard's remedy text instructs, so a pin that moved
    with the number of assertions would turn the required check red on a diff
    that complied with it -- and report only two bare integers.
    """

    def test_a_second_gated_assertion_in_a_listed_module_is_accounted_for(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(_PLANTED_TWO_GATED, encoding="utf-8")
        ungated, gated, _ = survey(root)
        assert not ungated
        assert unaccounted_modules(gated, frozenset({"tests/test_planted.py"})) == (set(), set())

    def test_the_module_and_assertion_counts_are_different_quantities(self, tmp_path: pathlib.Path) -> None:
        """One module, two gated assertions: why a count cannot stand in for a set."""
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(_PLANTED_TWO_GATED, encoding="utf-8")
        _, gated, _ = survey(root)
        assert set(gated) == {"tests/test_planted.py"}
        assert sum(len(lines) for lines in gated.values()) == 2

    def test_a_listed_module_contributing_nothing_gated_is_reported(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(_PLANTED_UNGATED, encoding="utf-8")
        _, gated, _ = survey(root)
        missing, unexpected = unaccounted_modules(gated, frozenset({"tests/test_planted.py"}))
        assert missing == {"tests/test_planted.py"}
        assert not unexpected

    def test_a_module_that_is_not_listed_is_reported(self, tmp_path: pathlib.Path) -> None:
        root = tmp_path / "tests"
        root.mkdir()
        (root / "test_planted.py").write_text(_PLANTED_GATED, encoding="utf-8")
        _, gated, _ = survey(root)
        missing, unexpected = unaccounted_modules(gated, frozenset())
        assert not missing
        assert unexpected == {"tests/test_planted.py"}
