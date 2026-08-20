"""A MuJoCo ``mjt*`` enum is matched by VALUE, never by operand order.

``mjModel`` / ``mjData`` expose their type fields as numpy integer arrays
(``model.geom_type[i]`` is an ``np.int32``), while the matching vocabulary is a
pybind11 enum (``mujoco.mjtGeom.mjGEOM_HFIELD``). Whether those two compare
equal depends on WHICH SIDE the enum is on:

* ``np.int32(1) == mjGEOM_HFIELD`` - numpy on the left - is ``True`` on every
  build measured.
* ``mjGEOM_HFIELD == np.int32(1)`` - the enum on the left - is ``True`` on
  3.9.0, 3.10.0 and 3.11.0 and ``False`` on 3.12.0 (measured on released
  wheels; the declared range is ``mujoco>=3.5.0,<4.0.0``, so both answers are
  in-range).

That second form is not usually written by hand: it is what ``in`` produces.
CPython's ``tuple.__contains__`` compares ``element == needle``, so
``model.geom_type[i] in (mjGEOM_HFIELD, mjGEOM_PLANE)`` puts the ENUM on the
left and silently stops matching - the membership test just answers ``False``
for every geom, with nothing raised and no warning. A hash-based ``in`` over a
``set`` of enum members degrades the same way: the hashes still collide, and the
equality that confirms the hit is the failing direction.

Measured consequence of one such site (mujoco 3.12.0, the shipped
terrain-seat helper): a heightfield ground geom stopped being recognised as
ground, so the "lowest robot geom" it computes returned the ground's own
``z=0.0`` and three ``create_world(terrain=...)`` regression tests failed with
``assert 0.0 >= 0.16``. Another (``examples/*/scene.py``): the home-pose helper
matched 0 of 3 joints and wrote no ``qpos`` and no ``ctrl``, leaving the arm at
its compiled default pose while logging that the home pose had been set.

So the rule is to compare integers on both sides - ``int(model.geom_type[i]) in
(int(mjGEOM_HFIELD), int(mjGEOM_PLANE))`` - which is what the MuJoCo backend
already does at its own type checks (``int(model.actuator_trntype[act_id]) !=
int(mj.mjtTrn.mjTRN_TENDON)``). This guard is structural rather than
behavioural on purpose: the defect only manifests from mujoco 3.12.0 on, so a
test that merely exercised the installed build would pass on an older one and
pin nothing.
"""

import ast
import pathlib
import re

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TREES = ("strands_robots", "tests", "tests_integ", "examples")

# ``mj.mjtGeom.mjGEOM_HFIELD`` / ``mujoco.mjtJoint.mjJNT_HINGE`` and friends.
_ENUM_MEMBER = re.compile(r"^(?:mj|mujoco)\.mjt[A-Za-z]+\.[A-Za-z_]+$")

_MINIMUM_GRADED_FILES = 40


def _is_enum_member(node: ast.expr) -> bool:
    """Whether ``node`` spells a ``mjt*`` enum member access."""
    return bool(_ENUM_MEMBER.match(ast.unparse(node)))


def _enum_collection_names(tree: ast.Module) -> set[str]:
    """Names bound to a tuple/set/list whose every element is a bare enum member."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Tuple | ast.Set | ast.List):
            continue
        elements = node.value.elts
        if not elements or not all(_is_enum_member(e) for e in elements):
            continue
        names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    return names


def _order_dependent_comparisons(source: str) -> list[tuple[int, str, str]]:
    """Return ``[(lineno, kind, snippet)]`` for every enum-on-the-left comparison."""
    tree = ast.parse(source)
    collections = _enum_collection_names(tree)
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            kind = None
            if isinstance(op, ast.In | ast.NotIn):
                if (
                    isinstance(comparator, ast.Tuple | ast.Set | ast.List)
                    and comparator.elts
                    and all(_is_enum_member(e) for e in comparator.elts)
                ):
                    kind = "membership in a collection of enum members"
                elif isinstance(comparator, ast.Name) and comparator.id in collections:
                    kind = f"membership in {comparator.id}, a collection of enum members"
            elif isinstance(op, ast.Eq | ast.NotEq) and _is_enum_member(node.left):
                kind = "an enum member on the left of =="
            if kind:
                found.append((node.lineno, kind, ast.unparse(node)))
    return found


def _graded_files() -> list[pathlib.Path]:
    """Every python file in the graded trees that names the enum vocabulary."""
    out: list[pathlib.Path] = []
    for tree in _TREES:
        for path in sorted((_REPO_ROOT / tree).rglob("*.py")):
            if "mjt" in path.read_text(encoding="utf-8"):
                out.append(path)
    return out


class TestEveryEnumComparisonIsValueBased:
    """No source compares a ``mjt*`` enum from the left, where numpy stops matching."""

    def test_no_source_puts_a_mujoco_enum_on_the_left_of_a_comparison(self) -> None:
        offenders: list[str] = []
        for path in _graded_files():
            for lineno, kind, snippet in _order_dependent_comparisons(path.read_text(encoding="utf-8")):
                rel = path.relative_to(_REPO_ROOT)
                offenders.append(f"{rel}:{lineno} ({kind}): {snippet}")
        assert not offenders, (
            "these comparisons put a mujoco enum on the left, where a numpy model/data "
            "field stops matching it on mujoco 3.12.0; compare int() to int() instead:\n  " + "\n  ".join(offenders)
        )

    def test_the_scan_reaches_every_tree(self) -> None:
        """A refactor that stops reaching the sources must fail, not report clean."""
        graded = _graded_files()
        assert len(graded) >= _MINIMUM_GRADED_FILES, f"only {len(graded)} files graded"
        reached = {p.relative_to(_REPO_ROOT).parts[0] for p in graded}
        assert reached == set(_TREES), f"trees reached: {sorted(reached)}"


class TestTheScanFindsBothOrderDependentShapes:
    """Planted sources: a clean result means the sources are right, not that anything passes."""

    @pytest.mark.parametrize(
        "snippet",
        [
            "x = model.geom_type[0] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)",
            "x = model.geom_type[0] not in (mj.mjtGeom.mjGEOM_PLANE,)",
            "x = model.geom_type[0] in {mujoco.mjtGeom.mjGEOM_PLANE}",
            "GROUND = (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)\nx = m.geom_type[0] in GROUND",
            "x = mujoco.mjtGeom.mjGEOM_PLANE == model.geom_type[0]",
            "x = mj.mjtJoint.mjJNT_FREE != model.jnt_type[0]",
        ],
        ids=["in-tuple", "not-in-tuple", "in-set", "in-named-collection", "enum-left-eq", "enum-left-neq"],
    )
    def test_an_order_dependent_comparison_is_reported(self, snippet: str) -> None:
        assert _order_dependent_comparisons(snippet), f"not reported: {snippet}"

    @pytest.mark.parametrize(
        "snippet",
        [
            "x = int(model.geom_type[0]) in (int(mujoco.mjtGeom.mjGEOM_PLANE),)",
            "GROUND = (int(mujoco.mjtGeom.mjGEOM_PLANE),)\nx = int(m.geom_type[0]) in GROUND",
            "x = model.geom_type[0] == mujoco.mjtGeom.mjGEOM_PLANE",
            "x = int(model.actuator_trntype[0]) != int(mj.mjtTrn.mjTRN_TENDON)",
            "x = name in ('a', 'b')",
        ],
        ids=["int-in-tuple", "int-in-named", "enum-on-the-right", "int-vs-int", "unrelated"],
    )
    def test_a_value_based_comparison_is_not_reported(self, snippet: str) -> None:
        assert not _order_dependent_comparisons(snippet), f"falsely reported: {snippet}"


class TestTheValueBasedFormMatchesOnThisBuild:
    """The convention selects the right geoms on whichever mujoco is installed."""

    def test_int_membership_selects_the_ground_geom_types(self) -> None:
        mujoco = pytest.importorskip("mujoco")
        model = mujoco.MjModel.from_xml_string(
            "<mujoco>"
            '<asset><hfield name="hf" nrow="4" ncol="4" size="1 1 .2 .1"/></asset>'
            "<worldbody>"
            '<geom name="pl" type="plane" size="1 1 .1"/>'
            '<geom name="hg" type="hfield" hfield="hf"/>'
            '<geom name="bx" type="box" size=".1 .1 .1"/>'
            "</worldbody></mujoco>"
        )
        ground = (int(mujoco.mjtGeom.mjGEOM_PLANE), int(mujoco.mjtGeom.mjGEOM_HFIELD))
        selected = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in range(model.ngeom)
            if int(model.geom_type[g]) in ground
        }
        assert selected == {"pl", "hg"}, f"selected {sorted(selected)}"
