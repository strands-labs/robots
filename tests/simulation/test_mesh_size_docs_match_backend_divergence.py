"""The mesh ``size`` contract is not shared across backends, and prose claiming it is is a defect.

``add_object(shape="mesh", ...)`` reads ``size`` two different ways:

* **Newton consumes it** as a per-axis scale on the loaded geometry. Its
  per-shape default is ``[1, 1, 1]`` rather than the ``[0.05, 0.05, 0.05]`` every
  primitive gets, because a scale of ``0.05`` would be meaningless, and the value
  reaches the solver as ``add_shape_mesh(..., scale=wp.vec3(sx, sy, sz))``. An
  empty ``size`` is refused there like any other unusable extent
  (``test_an_empty_mesh_size_is_refused_too``), because it *is* consumed.
* **MuJoCo does not.** ``_SIZE_LAYOUT["mesh"]`` requires zero components and
  ``_validate_size`` returns early for a mesh, so a full 3-vector and an empty
  one are accepted alike - the mark of a value nothing reads - and
  ``SpecBuilder`` gives the mesh geom a ``meshname`` rather than a size, leaving
  the compiled extent to the asset.

Isaac sits on MuJoCo's side of this: its mesh ``add_object`` (#2459) documents
``size`` as ignored - the asset's own units define the extent - so the
divergence measured here stays a two-party one (Newton vs. that shared read).
That third read is measured by
:mod:`tests.simulation.isaac.test_mesh_size_is_discarded_for_the_asset_extent`,
which imports :data:`MESH_SIZE` from here - so if Isaac ever starts consuming
it, the party count asserted above and the behaviour go out of step in one
place rather than leaving this file describing a divergence it no longer
measures correctly.

So ``size=[2, 3, 4]`` scales the asset on one backend and is ignored on the
other, and **both calls report success** - which is what makes the divergence
expensive to discover. ``docs/simulation/newton.md`` asserted the opposite in as
many words ("at parity with the MuJoCo backend"), so a reader porting a scene
between the two had the one paragraph that would have warned them telling them
not to look.

Why this guard and not a fix
----------------------------
Converging the two is a contract decision with a real cost in either direction -
``size`` would carry an extent for primitives and a multiplier for meshes, or
Newton loses a working capability and its documented default - and it overlaps
two of the three per-shape axes #1858 tracks. #2300 holds that decision. What
does *not* have to wait for it is the prose: until the two agree, each page must
say so and point at the open decision, which is the invariant asserted here.

Why the premise is measured rather than assumed
-----------------------------------------------
A prose guard that hardcodes "these backends disagree" outlives the disagreement:
once #2300 lands, it would keep demanding a caveat the code no longer needs, and
the guard becomes the drift it was written to catch. So each run re-measures both
behaviours - one ``add_object`` on a Newton stub, two ``_validate_size`` calls -
and ``test_the_backends_still_disagree_about_a_mesh_size`` **fails** rather than
skips when they converge. That failure is the prompt to settle both pages and
retire this file, and it arrives in the pull request that converges them rather
than a release later.

Needs no ``newton``/``warp`` install: Newton's ``add_object`` validates and
registers before anything touches a solver, so calling the unbound method with a
stand-in for ``self`` exercises it in every environment - the same technique
``tests/simulation/test_object_size_domain_across_backends.py`` uses, and that
file pins the surrounding ``size`` domain for all three backends.
"""

from __future__ import annotations

import pathlib
import re

import pytest

from strands_robots.simulation.mujoco.spec_builder import _SIZE_LAYOUT, _validate_size
from strands_robots.simulation.newton.simulation import NewtonSimEngine
from tests.simulation.test_pose_vector_domain_across_backends import _newton_stub

_DOCS = pathlib.Path(__file__).parents[2] / "docs" / "simulation"

#: The two pages that document a mesh ``add_object``: the Newton backend page and
#: the backend-agnostic world-building guide whose ``size`` table is MuJoCo's.
MESH_PAGES: tuple[pathlib.Path, ...] = (
    _DOCS / "newton.md",
    _DOCS / "world-building.md",
)

#: The open contract decision each page must point at while the two disagree.
DECISION = "#2300"

#: A scale no per-shape default could be mistaken for, so a backend that stores
#: it verbatim is provably consuming what the caller passed.
MESH_SIZE = [2.0, 3.0, 4.0]

#: The claim removed from the Newton page. Narrow on purpose: it is the sentence
#: that misled, and the positive requirement below is what generalizes.
UNQUALIFIED_PARITY = re.compile(r"at parity\s+with the MuJoCo backend", re.I)

#: Smallest asset ``trimesh`` will read - one triangle.
_ONE_TRIANGLE = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


def mesh_section(page: pathlib.Path) -> str:
    """The ``## Mesh objects`` block of ``page``, up to the next level-2 heading.

    The block is the unit that carries the claim: a reader who jumps to "Mesh
    objects" from the nav never passes the ``size`` table earlier on the page, so
    a caveat parked there would not reach them.
    """
    text = page.read_text(encoding="utf-8")
    match = re.search(r"^## Mesh objects$(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match is not None, f"{page.name} has no '## Mesh objects' section"
    return match.group(1)


def newton_stores_mesh_size(tmp_path: pathlib.Path) -> list[float]:
    """What Newton records as the mesh object's ``size`` when the caller passes one."""
    asset = tmp_path / "part.obj"
    asset.write_text(_ONE_TRIANGLE, encoding="utf-8")
    stub = _newton_stub()
    result = NewtonSimEngine.add_object(stub, "part", shape="mesh", mesh_path=str(asset), size=list(MESH_SIZE))
    assert result["status"] == "success", result
    return [float(component) for component in stub._world.objects["part"].size]


def mujoco_consumes_a_mesh_size() -> bool:
    """Whether MuJoCo's per-shape contract reads any ``size`` component for a mesh.

    A shape that consumes component ``i`` rejects a vector shorter than ``i + 1``,
    so accepting both a full vector and an empty one is the observable form of
    "nothing reads this".
    """
    if _SIZE_LAYOUT["mesh"][0] > 0:
        return True
    accepts_a_full_vector = _validate_size("mesh", list(MESH_SIZE)) is None
    accepts_nothing_at_all = _validate_size("mesh", []) is None
    return not (accepts_a_full_vector and accepts_nothing_at_all)


def backends_disagree(tmp_path: pathlib.Path) -> bool:
    """Whether a mesh ``size`` still means two different things."""
    newton_consumes = newton_stores_mesh_size(tmp_path) == MESH_SIZE
    return newton_consumes and not mujoco_consumes_a_mesh_size()


class TestTheMeasuredDivergence:
    """The behaviour the prose requirement below is conditioned on."""

    def test_newton_consumes_a_mesh_size_as_a_scale(self, tmp_path: pathlib.Path) -> None:
        assert newton_stores_mesh_size(tmp_path) == MESH_SIZE

    def test_mujoco_reads_no_mesh_size_component(self) -> None:
        assert _SIZE_LAYOUT["mesh"][0] == 0
        assert _validate_size("mesh", list(MESH_SIZE)) is None
        assert _validate_size("mesh", []) is None
        assert mujoco_consumes_a_mesh_size() is False

    def test_the_backends_still_disagree_about_a_mesh_size(self, tmp_path: pathlib.Path) -> None:
        """Fails - deliberately, not skips - on the change that converges them."""
        assert backends_disagree(tmp_path), (
            "Newton and MuJoCo now agree on what 'size' means for a mesh. #2300 is "
            "settled, so state the one contract on both pages and delete this file "
            "rather than leaving a caveat the code no longer needs."
        )


class TestTheProseSaysSo:
    """While the two disagree, neither page may present the mesh add as portable."""

    @pytest.mark.parametrize("page", MESH_PAGES, ids=lambda page: page.name)
    def test_the_mesh_section_names_the_open_decision(self, page: pathlib.Path, tmp_path: pathlib.Path) -> None:
        assert backends_disagree(tmp_path), "premise gone - see TestTheMeasuredDivergence"
        assert DECISION in mesh_section(page), (
            f"{page.name}'s mesh section documents a 'size' the other backend reads "
            f"differently without pointing at {DECISION}, so a reader porting a scene "
            "between the two learns nothing about it."
        )

    def test_no_page_claims_unqualified_mesh_parity(self, tmp_path: pathlib.Path) -> None:
        assert backends_disagree(tmp_path), "premise gone - see TestTheMeasuredDivergence"
        for page in MESH_PAGES:
            found = UNQUALIFIED_PARITY.search(mesh_section(page))
            assert found is None, (
                f"{page.name}'s mesh section claims parity with MuJoCo while the two "
                f"disagree about 'size': {found.group(0)!r}"  # type: ignore[union-attr]
            )
