"""Isaac's mesh ``size`` is discarded and the asset's own extent reported (#2459).

``add_object(shape="mesh", ...)`` reads ``size`` differently per backend, and
:mod:`tests.simulation.test_mesh_size_docs_match_backend_divergence` is the gate
that measures the disagreement: **Newton consumes it** as a per-axis scale,
**MuJoCo does not**, and *both calls report success* - which is what makes the
divergence expensive to discover rather than merely inconsistent.

#2498 put Isaac on MuJoCo's side of that split, and that gate's prose now says
so. Nothing measured it. The claim was reachable only through the GPU journey
test, which calls the mesh ``add_object`` **without** a ``size=`` argument - so
it pinned "the reported extent comes from the asset when none was requested",
not "a requested extent is discarded and the asset's reported anyway". The
remaining unit-level mesh cases are all refusals (``mesh_path`` without
``shape='mesh'``, asset missing, ``.dae``, no world), none of which reach the
success path where a ``size`` would be consumed or dropped.

So a future change that starts reading ``size`` as a scale - the Newton reading,
and the intuitive one for anyone arriving from Newton code - would break no
test, and the gate above would then be documenting a third backend's behaviour
falsely. A gate trusted to *describe* a hazard is worse than none once it is
wrong about it.

Both halves are asserted, because the payload alone does not cover the class:

* the **report** is the asset's extent whatever was requested, and is
  byte-identical to the same call with no ``size`` at all - "ignored" in its
  strongest form;
* **no component of the request reaches the prim construction**. Newton's
  reading scales the *geometry*, while ``resolved_size`` here is computed from
  the *unscaled* asset file - so a scale that reached the prim would leave the
  payload honest and the object the wrong size on the stage, invisible to any
  assertion on the report. Every argument the construction leaves are handed is
  searched numerically, so a scale arriving as a ``numpy`` array - which is how
  a real Isaac call would spell it - is not missed.

The request ``[2, 3, 4]`` is imported from that gate rather than restated, so
the claim and its pin are one edit apart in either direction, and the asset's
``0.1 x 0.2 x 0.3`` extent shares no component with it - a per-axis scale, a
uniform scale and a verbatim echo all read differently from a discard. The
placement is chosen for the same reason, and the first test asserts the
disjointness rather than assuming it, so a future change to either value reports
that the fixture stopped being able to see the difference.

Scope, following :mod:`tests.simulation.isaac.test_scene_object_meshes`: the
Kit-only leaves (the stage reference and the prim wrappers) are stood in, so
these pin the ``size`` contract and the report rather than the USD authoring.
``mesh_aabb`` - the function whose output the assertions are about - runs for
real; only the USD *conversion* is stubbed, because the extent is parsed from
the source asset and not from the converted stage.
"""

from __future__ import annotations

import inspect
import math
import pathlib
import sys
import threading
import types
from collections.abc import Iterator
from typing import Any

import pytest

from tests.simulation.test_mesh_size_docs_match_backend_divergence import MESH_SIZE

pytest.importorskip("strands_robots.simulation.isaac")

#: A box whose extent differs from :data:`MESH_SIZE` on every axis and is
#: anisotropic, so a per-axis scale is distinguishable from a uniform one.
_BOX_OBJ = (
    "v 0 0 0\nv 0.1 0 0\nv 0.1 0.2 0\nv 0 0.2 0\n"
    "v 0 0 0.3\nv 0.1 0 0.3\nv 0.1 0.2 0.3\nv 0 0.2 0.3\n"
    "f 1 2 3\nf 1 3 4\nf 5 6 7\nf 5 7 8\n"
)

#: The extent ``mesh_aabb`` measures from :data:`_BOX_OBJ` (full extents).
_ASSET_EXTENT = [0.1, 0.2, 0.3]

#: A primitive extent used by the no-over-reach control, distinct from both.
_BOX_SIZE = [0.21, 0.31, 0.41]

#: The placement. Named so the premise below can assert the request is
#: distinguishable from every other number the call passes - otherwise
#: "the request reached the construction" could be satisfied by a coincidence
#: with the pose rather than by a scale.
_POSITION = [0.41, 0.07, 0.23]

#: ``add_object``'s identity-quaternion default, for the same disjointness check.
_IDENTITY_QUAT = [1.0, 0.0, 0.0, 0.0]


def _numbers(value: Any) -> Iterator[float]:
    """Every float reachable in ``value``, flattening arrays and containers.

    Searched numerically rather than by ``repr``: a scale handed over as a
    ``numpy`` array renders as ``array([2., 3., 4.])``, so a textual search for
    ``"2.0"`` would miss exactly the spelling a real Isaac call would use.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
        return
    if hasattr(value, "tolist"):
        yield from _numbers(value.tolist())
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            yield from _numbers(item)


class _Recorder:
    """Every argument the stood-in construction leaves received."""

    def __init__(self) -> None:
        self.constructed: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.referenced: list[tuple[Any, Any]] = []

    def prim_class(self, name: str) -> type:
        recorder = self

        class _Prim:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                recorder.constructed.append((name, args, kwargs))
                self.prim = object()
                self.name = kwargs.get("name")

            def set_collision_approximation(self, approximation: str) -> None:
                recorder.constructed.append((f"{name}.collision", (approximation,), {}))

            def __getattr__(self, attribute: str) -> Any:
                # A real prim wrapper carries many ``set_*`` mutators (scale
                # among them), so record any that is called rather than only
                # the constructor. Anything else raises, so an attribute probe
                # still gets an honest answer.
                if not attribute.startswith(("set_", "apply_")):
                    raise AttributeError(attribute)

                def _record(*args: Any, **kwargs: Any) -> None:
                    recorder.constructed.append((f"{name}.{attribute}", args, kwargs))

                return _record

        return _Prim

    def mentions(self, value: float) -> bool:
        """Whether ``value`` was handed to the construction path, numerically."""
        seen = list(_numbers(self.constructed)) + list(_numbers(self.referenced))
        return any(math.isclose(value, candidate) for candidate in seen)


@pytest.fixture
def recorder() -> _Recorder:
    return _Recorder()


@pytest.fixture
def fake_isaacsim(monkeypatch, recorder: _Recorder) -> None:
    """Fake ``isaacsim`` tree covering every import the mesh and primitive
    ``add_object`` success paths perform, in the same shape as the URDF branch's
    ``fake_isaacsim``: the modern ``isaacsim.*`` names, wired so dotted imports
    resolve."""
    names = (
        "isaacsim",
        "isaacsim.core",
        "isaacsim.core.api",
        "isaacsim.core.api.objects",
        "isaacsim.core.prims",
        "isaacsim.core.utils",
        "isaacsim.core.utils.stage",
    )
    mods: dict[str, types.ModuleType] = {}
    for name in names:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        mods[name] = module

    mods["isaacsim"].core = mods["isaacsim.core"]  # type: ignore[attr-defined]
    mods["isaacsim.core"].api = mods["isaacsim.core.api"]  # type: ignore[attr-defined]
    mods["isaacsim.core"].prims = mods["isaacsim.core.prims"]  # type: ignore[attr-defined]
    mods["isaacsim.core"].utils = mods["isaacsim.core.utils"]  # type: ignore[attr-defined]
    mods["isaacsim.core.api"].objects = mods["isaacsim.core.api.objects"]  # type: ignore[attr-defined]
    mods["isaacsim.core.utils"].stage = mods["isaacsim.core.utils.stage"]  # type: ignore[attr-defined]

    def add_reference_to_stage(usd_path: Any = None, prim_path: Any = None) -> None:
        recorder.referenced.append((usd_path, prim_path))

    mods["isaacsim.core.utils.stage"].add_reference_to_stage = add_reference_to_stage  # type: ignore[attr-defined]
    for wrapper in ("SingleGeometryPrim", "SingleRigidPrim"):
        setattr(mods["isaacsim.core.prims"], wrapper, recorder.prim_class(wrapper))
    for primitive in (
        "FixedCuboid",
        "DynamicCuboid",
        "FixedSphere",
        "DynamicSphere",
        "FixedCylinder",
        "DynamicCylinder",
        "FixedCapsule",
        "DynamicCapsule",
    ):
        setattr(mods["isaacsim.core.api.objects"], primitive, recorder.prim_class(primitive))


@pytest.fixture
def asset(tmp_path: pathlib.Path, monkeypatch) -> pathlib.Path:
    """A parseable OBJ, with the USD conversion stubbed.

    ``mesh_aabb`` stays real - it is the function under test here - and only
    ``convert_mesh_to_usd`` is stood in, because it needs ``pxr`` and the extent
    is parsed from this source file rather than from the converted stage.
    """
    from strands_robots.simulation.isaac import mesh_assets

    path = tmp_path / "widget.obj"
    path.write_text(_BOX_OBJ, encoding="utf-8")
    monkeypatch.setattr(
        mesh_assets,
        "convert_mesh_to_usd",
        lambda mesh_path, cache_dir=None: f"{mesh_path}.usd",
    )
    return path


def _engine() -> Any:
    """A stand-in engine whose world is present, so ``add_object`` reaches the
    success path instead of the "No world created" gate."""
    from strands_robots.simulation.isaac.simulation import IsaacConfig, IsaacSimulation

    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig()

    class _Scene:
        def add(self, handle: Any) -> None:
            return None

    class _World:
        physics_sim_view = object()
        scene = _Scene()

        def play(self) -> None:
            return None

        def reset(self) -> None:
            return None

        def stop(self) -> None:
            return None

    engine._world = _World()
    engine._world_created = True
    engine._objects = {}
    engine._robots = {}
    engine._scene_objects = set()
    engine._prim_registry = []
    engine._cameras = {}
    return engine


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    assert result["status"] == "success", result
    return next(block["json"] for block in result["content"] if "json" in block)


def _add_mesh(asset: pathlib.Path, **overrides: Any) -> dict[str, Any]:
    return _engine().add_object(
        "widget",
        shape="mesh",
        mesh_path=str(asset),
        position=list(_POSITION),
        is_static=True,
        **overrides,
    )


class TestAMeshSizeIsDiscardedForTheAssetExtent:
    def test_the_request_is_distinguishable_from_every_other_number_in_the_call(self, asset: pathlib.Path) -> None:
        """Premise: a discard is only observable while the request collides with
        nothing else the call carries.

        Guards the fixture rather than the code. If ``MESH_SIZE`` upstream or the
        placement here ever coincide with the asset's extent, the assertions
        below stop distinguishing a discard from an echo, and this says so
        instead of leaving them quietly weaker.
        """
        from strands_robots.simulation.isaac.mesh_assets import mesh_aabb

        _center, extent = mesh_aabb(str(asset))
        assert list(extent) == pytest.approx(_ASSET_EXTENT)
        others = list(_ASSET_EXTENT) + list(_POSITION) + list(_IDENTITY_QUAT)
        for component in MESH_SIZE:
            assert not any(math.isclose(component, other) for other in others), (
                f"{component} from the request also appears elsewhere in the call ({others}), "
                "so it cannot tell a discarded request from a consumed one"
            )

    def test_a_requested_mesh_size_is_discarded_for_the_asset_extent(
        self, fake_isaacsim: None, asset: pathlib.Path
    ) -> None:
        """The reported extent is the asset's, not the request.

        Wrong on every axis, so nothing about the answer could have come from
        ``MESH_SIZE`` - the same value
        ``test_mesh_size_docs_match_backend_divergence`` uses to prove Newton
        *does* consume it.
        """
        payload = _payload(_add_mesh(asset, size=list(MESH_SIZE)))
        assert payload["shape"] == "mesh"
        assert payload["size"] == pytest.approx(_ASSET_EXTENT), (
            f"a requested size={MESH_SIZE} was reported back rather than discarded: {payload['size']}"
        )

    def test_the_report_is_identical_to_the_same_call_with_no_size(
        self, fake_isaacsim: None, asset: pathlib.Path
    ) -> None:
        """ "Ignored" in its strongest form: passing a ``size`` has no
        observable effect on the result at all."""
        requested = _payload(_add_mesh(asset, size=list(MESH_SIZE)))
        omitted = _payload(_add_mesh(asset))
        assert requested == omitted

    def test_no_component_of_the_request_reaches_the_prim_construction(
        self, fake_isaacsim: None, asset: pathlib.Path, recorder: _Recorder
    ) -> None:
        """A scale that reached the prim would leave the report honest.

        ``resolved_size`` is parsed from the source asset, so a Newton-style
        scale applied to the referenced geometry would keep the payload correct
        and put the wrong-sized object on the stage. Assert instead that the
        construction path was handed no component of the request.
        """
        _payload(_add_mesh(asset, size=list(MESH_SIZE)))
        assert recorder.constructed, "premise: the construction path was not reached"
        for component in MESH_SIZE:
            assert not recorder.mentions(component), (
                f"{component} from the requested size reached the prim construction: {recorder.constructed}"
            )

    def test_the_mesh_prim_constructor_declares_no_size_parameter(self) -> None:
        """The discard is structural, and this names the signature that has to
        change for it to stop being.

        A scale-consuming regression arrives in two steps - the parameter, then
        the behaviour - and the assertions above only see the second. This one
        fails on the first, in the diff that introduces it.
        """
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        parameters = inspect.signature(IsaacSimulation._create_mesh_prim).parameters
        assert "mesh_path" in parameters, "premise: this is the mesh construction path"
        assert "size" not in parameters, (
            "_create_mesh_prim now accepts a size; Isaac's documented mesh contract discards it "
            "(see tests/simulation/test_mesh_size_docs_match_backend_divergence.py)"
        )

    def test_a_primitive_size_is_still_consumed(self, fake_isaacsim: None) -> None:
        """No-over-reach control: only ``shape="mesh"`` ignores ``size``.

        Fails for the tempting shortcut of dropping the extent earlier, which
        would take the shapes that do read it with it.
        """
        payload = _payload(
            _engine().add_object("crate", shape="box", position=[0.0, 0.0, 0.5], size=list(_BOX_SIZE), is_static=True)
        )
        assert payload["size"] == pytest.approx(_BOX_SIZE)
