"""Stepping a scene PhysX's tensor view no longer covers is refused, not ticked.

PhysX builds its tensor simulation view at ``world.reset()``, and adding or
deleting a physics-body prim afterwards invalidates it. Isaac says so itself on
the delete path -- "prim '/World/Objects/b1' was deleted while being used by a
shape in a tensor view class. The physics.tensors simulationView was
invalidated" -- and on the add path it says nothing at all: the new body simply
sits outside the view.

Measured on ``nvcr.io/nvidia/isaac-sim:6.0.1`` (A10G), every consequence is
SILENT, which is what makes a refusal the fix rather than a nicety:

* ``add_object`` then ``step(90)`` leaves the body at its spawn height forever
  while ``step`` reports ``"Stepped 90x ... 33 steps/sec"``. A cube spawned at
  ``z=0.600`` was still at ``z=0.600``; the same step after a ``reset()`` took it
  to ``z=0.025``. The caller's only evidence is a cube that never falls.
* the same ``add_object`` empties an already-working robot's
  ``get_observation`` -- 2 keys to 0 on a 2-joint URDF arm, back to 2 after
  ``reset()``.
* ``remove_object`` then reading that robot *raises* out of
  ``SingleArticulation.get_joint_positions`` -- a bare ``Exception`` ("Failed to
  get DOF positions from backend") -- through a method the ``SimEngine`` ABC
  documents as returning a dict.

``reset()`` repairs all three, so the operation is one call away from correct and
the refusal names it.

The observation half was measured on a URDF robot deliberately: the three
procedural builders leave ``_RobotState.articulation`` as ``None`` and report 0
keys at every point in the lifecycle, so neither the drop nor the raise is
observable on one. That is a separate defect, and it is why the GPU run of this
gate reads 0 keys for ``so100`` before any mutation has happened.

These pins are unit-level: the Kit leaves are stood in (the pattern
:mod:`tests.simulation.isaac.test_mesh_size_is_discarded_for_the_asset_extent`
uses), so what is graded is which mutations mark the scene, which lifecycle calls
clear the mark, and what ``step`` / ``get_observation`` do about it. The live-Kit
half is exercised on GPU.

Two of these classes exist because of what the flag would otherwise break rather
than what it fixes:

* :class:`TestLoadSceneRebuildsTheViewItself` -- ``load_scene`` calls the real
  ``add_object`` / ``remove_object``, then rebuilds the view via
  ``SimulationManager.initialize_physics()`` and ``world.play()``, deliberately
  *not* ``world.reset()`` (#1802: a full reset re-applies every registered
  prim's default state and was measured to explode an already-posed articulation
  into non-finite PhysX bounds). So it repairs the view by a route that clears no
  reset-keyed flag, and a mark left standing there would refuse ``step`` for the
  whole LIBERO/scene path with the view perfectly live.
* :class:`TestTheMarkSurvivesAnInstanceBuiltWithoutInit` -- 24 test modules build
  a skeleton engine with ``IsaacSimulation.__new__``, which never runs
  ``__init__``.
"""

from __future__ import annotations

import logging
import sys
import threading
import types
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")

from strands_robots.simulation.isaac.simulation import (  # noqa: E402 - after importorskip
    IsaacConfig,
    IsaacSimulation,
    _ObjectState,
    _RobotState,
)

#: The remedy every refusal has to name. An agent reads the text and nothing
#: else, so a refusal that omits the one call that fixes the state is a dead end.
_REMEDY = "reset()"


class _Scene:
    def add(self, handle: Any) -> None:
        return None

    def remove_object(self, name: str) -> None:
        return None


class _World:
    """A world whose lifecycle calls are observable and whose prims are inert."""

    def __init__(self) -> None:
        self.physics_sim_view = object()
        self.reset_calls = 0
        self.step_calls = 0
        self.play_calls = 0
        self.scene = _Scene()

    def reset(self) -> None:
        self.reset_calls += 1

    def step(self, render: bool = False) -> None:
        self.step_calls += 1

    def play(self) -> None:
        self.play_calls += 1

    def stop(self) -> None:
        return None


def _engine(with_robot: bool = False) -> Any:
    """A skeleton engine whose world is present, so the mutation and step paths
    reach their success bodies rather than the "No world created" gate."""
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._config = IsaacConfig()
    engine._world = _World()
    engine._world_created = True
    engine._objects = {}
    engine._robots = {}
    engine._scene_objects = set()
    engine._prim_registry = []
    engine._cameras = {}
    engine._sim_time = 0.0
    engine._step_count = 0
    # ``reset`` flushes an open recording episode first, and the Isaac backend
    # keeps that state in its own dict rather than in ``_backend_state``.
    engine._recording_state_dict = {}
    # Main-thread affinity (#1896): these pins measure the stale-view gate, not
    # kit-thread marshalling, so the call's own thread is declared the owning
    # one and the genuinely-bound helper stays on its inline path.
    engine._main_tid = threading.get_ident()
    engine._pump_running = False
    if with_robot:
        robot = _RobotState(name="arm", prim_path="/World/Robots/arm", joint_names=["j0"])
        engine._robots["arm"] = robot
    return engine


@pytest.fixture
def fake_isaacsim(monkeypatch) -> None:
    """Fake ``isaacsim`` tree covering the imports ``add_object``'s primitive
    success path performs, in the shape the mesh-size pins use."""
    names = (
        "isaacsim",
        "isaacsim.core",
        "isaacsim.core.api",
        "isaacsim.core.api.objects",
        "isaacsim.core.prims",
        "isaacsim.core.utils",
        "isaacsim.core.utils.prims",
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
    mods["isaacsim.core.utils"].prims = mods["isaacsim.core.utils.prims"]  # type: ignore[attr-defined]
    mods["isaacsim.core.utils"].stage = mods["isaacsim.core.utils.stage"]  # type: ignore[attr-defined]

    class _Prim:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.prim = object()
            self.name = kwargs.get("name")

        def __getattr__(self, attribute: str) -> Any:
            if not attribute.startswith(("set_", "apply_")):
                raise AttributeError(attribute)
            return lambda *args, **kwargs: None

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
        setattr(mods["isaacsim.core.api.objects"], primitive, _Prim)
    for wrapper in ("SingleGeometryPrim", "SingleRigidPrim"):
        setattr(mods["isaacsim.core.prims"], wrapper, _Prim)
    mods["isaacsim.core.utils.prims"].delete_prim = lambda prim_path: None  # type: ignore[attr-defined]
    mods["isaacsim.core.utils.stage"].add_reference_to_stage = (  # type: ignore[attr-defined]
        lambda usd_path=None, prim_path=None: None
    )


def _add_cube(engine: Any, name: str = "cube") -> dict[str, Any]:
    result = engine.add_object(name, shape="cuboid", position=[0.0, 0.0, 0.5], size=[0.05, 0.05, 0.05])
    assert result["status"] == "success", result
    return result


def _text(result: dict[str, Any]) -> str:
    return " ".join(block.get("text", "") for block in result["content"])


class TestABodyMutationMarksTheScene:
    """The two mutations that invalidate the view say so."""

    def test_add_object_marks_the_scene(self, fake_isaacsim: None) -> None:
        engine = _engine()
        assert engine._physics_view_stale is False
        _add_cube(engine)
        assert engine._physics_view_stale is True

    def test_remove_object_marks_the_scene(self, fake_isaacsim: None) -> None:
        engine = _engine()
        _add_cube(engine)
        engine._physics_view_stale = False  # isolate the removal's own effect
        assert engine.remove_object("cube")["status"] == "success"
        assert engine._physics_view_stale is True

    def test_a_refused_add_marks_nothing(self, fake_isaacsim: None) -> None:
        """A mutation that did not happen invalidates nothing.

        The mark gates ``step``, so setting it on a refusal would strand a
        caller behind a reset they have no reason to run - the scene PhysX holds
        is exactly the one it held before the refused call.
        """
        engine = _engine()
        assert engine.add_object("cube", shape="not_a_shape")["status"] == "error"
        assert engine._physics_view_stale is False


class TestSteppingAMarkedSceneIsRefused:
    def test_step_refuses_after_an_add(self, fake_isaacsim: None) -> None:
        engine = _engine()
        _add_cube(engine)
        result = engine.step(90)
        assert result["status"] == "error", result
        assert engine._world.step_calls == 0, "a refused step must not tick the world"

    def test_step_refuses_after_a_removal(self, fake_isaacsim: None) -> None:
        engine = _engine()
        _add_cube(engine)
        assert engine.remove_object("cube")["status"] == "success"
        result = engine.step(90)
        assert result["status"] == "error", result
        assert engine._world.step_calls == 0

    def test_the_refusal_names_the_remedy(self, fake_isaacsim: None) -> None:
        """An agent matches on the text, and the state is one call from correct."""
        engine = _engine()
        _add_cube(engine)
        text = _text(engine.step(90))
        assert _REMEDY in text
        # The two mutations that require it are named, and the ones that do not
        # are named too - so a caller is not sent resetting after every call.
        # The wording is now more precise than "add_object / remove_object",
        # because only the DYNAMIC case invalidates - measured on an A10G, a
        # static add and a static remove both leave a Franka's 9 joint keys intact.
        assert "DYNAMIC" in text
        assert "a static add_object or" in text
        assert "add_camera" in text

    def test_the_refusal_does_not_report_a_rate(self, fake_isaacsim: None) -> None:
        """The measured defect was a success envelope carrying "33 steps/sec"
        for 90 steps that moved nothing. A refusal must carry no such number."""
        engine = _engine()
        _add_cube(engine)
        text = _text(engine.step(90))
        assert "steps/sec" not in text
        assert engine._step_count == 0
        assert engine._sim_time == 0.0


class TestResetClearsTheMark:
    def test_reset_lets_the_next_step_run(self, fake_isaacsim: None) -> None:
        engine = _engine()
        _add_cube(engine)
        assert engine.step(2)["status"] == "error"

        assert engine.reset()["status"] == "success"
        assert engine._physics_view_stale is False

        result = engine.step(2)
        assert result["status"] == "success", result
        assert engine._world.step_calls == 2

    def test_a_reset_that_never_reached_the_rebuild_leaves_the_mark(self, fake_isaacsim: None) -> None:
        """The mark is cleared where the view is rebuilt, not on entry.

        A ``reset`` that never got ``world.reset()`` to land has repaired
        nothing, so clearing on entry would hand the next ``step`` a scene the
        view still does not cover - the silent tick this whole gate exists to
        refuse.

        ``reset`` propagates the failure rather than enveloping it, which is its
        own pre-existing contract and not this gate's to change; what is pinned
        here is that the scene stays marked either way, so ``step`` keeps
        refusing until a reset actually succeeds.
        """
        engine = _engine()
        _add_cube(engine)

        def _explode() -> None:
            raise RuntimeError("kit session torn down")

        engine._world.reset = _explode  # type: ignore[method-assign]
        with pytest.raises(RuntimeError):
            engine.reset()
        assert engine._physics_view_stale is True
        assert engine.step(2)["status"] == "error"
        assert engine._world.step_calls == 0


class TestAMarkedSceneAnswersNoObservation:
    def test_get_observation_is_empty_and_says_why(self, fake_isaacsim: None, caplog) -> None:
        """Empty is this method's documented degraded mode; the WARNING is what
        keeps it from being silent.

        The read is not attempted at all: after a removal
        ``SingleArticulation.get_joint_positions`` raises a bare ``Exception``
        straight out of a method the ABC documents as returning a dict, and the
        narrow handler downstream cannot catch that without widening to
        ``except Exception``.
        """
        engine = _engine(with_robot=True)
        _add_cube(engine)
        with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.isaac.simulation"):
            assert engine.get_observation() == {}
        assert _REMEDY in caplog.text

    def test_the_read_is_not_attempted(self, fake_isaacsim: None) -> None:
        """A handle that raises on every read stands in for the live one."""
        engine = _engine(with_robot=True)

        class _RaisingArticulation:
            def get_joint_positions(self) -> Any:
                raise AssertionError("the articulation must not be read while the view is stale")

        engine._robots["arm"].articulation = _RaisingArticulation()
        _add_cube(engine)
        assert engine.get_observation() == {}

    def test_reset_restores_the_observation(self, fake_isaacsim: None) -> None:
        """Control: the empty answer is the mark's doing and nothing else."""
        engine = _engine(with_robot=True)
        _add_cube(engine)
        assert engine.get_observation() == {}
        assert engine.reset()["status"] == "success"
        # Past the gate now: whatever this skeleton's handleless robot reports is
        # the ordinary path's business, not this gate's. What is pinned is that
        # the gate is no longer the reason.
        assert engine._physics_view_stale is False


class TestOnlyADynamicBodyMarksTheScene:
    """Static adds and removes do not invalidate the view, so they must not mark it.

    Measured on an A10G under Isaac Sim 6.0.1, reading a Franka's joint keys either
    side of each operation:

        add_object(is_static=True)     9 keys -> 9    view intact
        add_object(is_static=False)    9 keys -> 0    view dead
        remove a static prim           9 keys -> 9    view intact
        remove a dynamic prim          PhysX: "prim ... was deleted while being used
                                       by a shape in a tensor view class. The
                                       physics.tensors simulationView was
                                       invalidated." The next joint read HUNG until
                                       a 2-minute timeout - it did not return empty.

    The mechanism is already gated this way inside ``add_object``:
    ``_construct_shape_prim`` stops the timeline - which is what clears the sim view
    - only for a dynamic prim.

    Marking unconditionally was a real regression, not a theoretical one. It disabled
    every ``step`` in ``examples/isaac_gs`` (all three of its adds are
    ``is_static=True``, it never calls ``reset()``, and its six step sites all
    discard the envelope), the same in
    ``tests_integ/simulation/test_isaac_body_state_gpu``, and it made
    ``docs/simulation/isaac.md``'s own usage example refuse. The original evidence
    for the mark was a single DynamicCuboid measurement, generalised one step too far.
    """

    def test_a_static_add_does_not_mark(self, fake_isaacsim: None) -> None:
        engine = _engine()

        assert (
            engine.add_object(name="s", shape="cuboid", position=[0.4, 0.0, 0.03], size=[0.05] * 3, is_static=True)[
                "status"
            ]
            == "success"
        )

        assert engine._physics_view_stale is False

    def test_a_dynamic_add_marks(self, fake_isaacsim: None) -> None:
        engine = _engine()

        assert (
            engine.add_object(
                name="d", shape="cuboid", position=[0.6, 0.0, 0.4], size=[0.05] * 3, is_static=False, mass=0.2
            )["status"]
            == "success"
        )

        assert engine._physics_view_stale is True

    def test_a_static_add_leaves_step_working(self, fake_isaacsim: None) -> None:
        """The property examples/isaac_gs depends on, stated as behaviour."""
        engine = _engine()
        engine.add_object(name="s", shape="cuboid", position=[0.4, 0.0, 0.03], size=[0.05] * 3, is_static=True)

        assert engine.step(20)["status"] == "success"

    def test_the_isaac_gs_sequence_steps(self, fake_isaacsim: None) -> None:
        """examples/isaac_gs/scene.py's exact build order, which has no reset() and
        whose step envelope is discarded - so a refusal there is invisible."""
        engine = _engine()
        engine.add_object(
            name="shadow", shape="cuboid", position=[0.0, 0.0, -0.01], size=[2.0, 2.0, 0.02], is_static=True
        )
        engine.add_object(name="cube", shape="cuboid", position=[0.45, 0.0, 0.03], size=[0.05] * 3, is_static=True)

        assert engine.step(20)["status"] == "success", "the settle step examples/isaac_gs relies on was refused"

    def test_a_static_removal_does_not_mark(self, fake_isaacsim: None) -> None:
        engine = _engine()
        engine.add_object(name="s", shape="cuboid", position=[0.4, 0.0, 0.03], size=[0.05] * 3, is_static=True)
        engine._physics_view_stale = False

        assert engine.remove_object("s")["status"] == "success"

        assert engine._physics_view_stale is False

    def test_a_dynamic_removal_marks(self, fake_isaacsim: None) -> None:
        engine = _engine()
        engine.add_object(
            name="d", shape="cuboid", position=[0.6, 0.0, 0.4], size=[0.05] * 3, is_static=False, mass=0.2
        )
        engine._physics_view_stale = False

        assert engine.remove_object("d")["status"] == "success"

        assert engine._physics_view_stale is True


class TestOnlyABodyMutationMarksTheScene:
    """Which mutations invalidate the view was measured, not inferred.

    On the live runtime ``add_camera``, ``remove_camera``, ``move_object``,
    ``add_robot`` and ``remove_robot`` each left an already-working robot
    reporting all 9 of its observation keys, so marking the scene for any of
    them would refuse ``step`` over a view that is perfectly live - and a gate
    that fires when nothing is wrong is one a caller learns to route around.

    Derived from the module's AST rather than driven per method, for two
    reasons. It grades a *method added later* on arrival, which a fixed list of
    calls cannot; and the alternative is standing in Kit's camera and USD
    stacks (``omni.usd``, ``pxr.Gf``, ``pxr.UsdGeom``) to reach success paths
    whose pixels and xform ops have nothing to do with this gate - a fixture
    heavy enough that its own breakage would read as this gate failing.
    """

    #: The only two methods entitled to mark the scene.
    _MARKERS = {"add_object", "remove_object"}

    def _assignments(self) -> dict[str, list[bool]]:
        """Every ``self._physics_view_stale = <bool>`` in the module, by the
        function that immediately encloses it.

        Attributed to the INNERMOST enclosing function. ``reset`` does its work
        in a nested ``_reset_impl`` so it can be marshalled onto the kit thread,
        and an ``ast.walk`` per function would report that one assignment under
        both names - which reads as two clearers where there is one.
        """
        import ast
        import pathlib

        from strands_robots.simulation.isaac import simulation as module

        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        found: dict[str, list[bool]] = {}

        def visit(node: ast.AST, enclosing: str | None) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(child, child.name)
                    continue
                if isinstance(child, ast.Assign) and enclosing is not None:
                    for target in child.targets:
                        if (
                            isinstance(target, ast.Attribute)
                            and target.attr == "_physics_view_stale"
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and isinstance(child.value, ast.Constant)
                            and isinstance(child.value.value, bool)
                        ):
                            found.setdefault(enclosing, []).append(child.value.value)
                visit(child, enclosing)

        visit(tree, None)
        return found

    def test_the_premise_holds(self) -> None:
        """Guards the scan, not the code: an AST walk that matched nothing would
        satisfy every assertion below by vacuum."""
        assert self._assignments(), "the scan found no assignment at all"

    def test_nothing_but_a_body_mutation_marks_the_scene(self) -> None:
        marks = {method for method, values in self._assignments().items() if True in values}
        assert marks == self._MARKERS

    def test_the_lifecycle_calls_that_clear_it_are_the_ones_that_rebuild_the_view(self) -> None:
        """``create_world`` and ``reset`` call ``world.reset()``, which builds the
        view; ``load_scene`` rebuilds it without one (#1802) and so must clear it
        too. ``__init__`` seeds the field. Any other clearer is a silent re-open
        of the gate."""
        clears = {method for method, values in self._assignments().items() if False in values}
        # ``_reset_impl`` is ``reset``'s nested body, marshalled onto the kit
        # thread; it is where the clear sits, and there is no other clearer.
        assert clears == {"__init__", "create_world", "_reset_impl", "load_scene"}


class TestLoadSceneRebuildsTheViewItself:
    """``load_scene`` mutates bodies and then repairs the view without a reset.

    So the mark it sets through its own ``add_object`` calls has to be cleared
    where that rebuild lands. Otherwise every scene load ends with a live view
    and a ``step`` that refuses to use it.
    """

    _SCENE = """
    <mujoco model="probe">
      <worldbody>
        <body name="table" pos="0 0 0.4"><geom type="box" size="0.5 0.5 0.4"/></body>
      </worldbody>
    </mujoco>
    """

    def _scene_file(self, tmp_path: Any) -> str:
        path = tmp_path / "scene.xml"
        path.write_text(self._SCENE)
        return str(path)

    def test_a_loaded_scene_steps(self, fake_isaacsim: None, tmp_path: Any) -> None:
        engine = _engine()
        result = engine.load_scene(self._scene_file(tmp_path))
        assert result["status"] == "success", result
        assert engine._physics_view_stale is False, "load_scene rebuilt the view; step must not refuse"
        assert engine.step(2)["status"] == "success"
        # Pin the #1802 contract this clearing rides on: had load_scene reached
        # for world.reset() instead, the mark would clear for the wrong reason
        # and this pin would stop measuring anything.
        assert engine._world.reset_calls == 0

    def test_a_load_that_realized_nothing_leaves_the_mark(self, fake_isaacsim: None, tmp_path: Any) -> None:
        """The rebuild is gated on having realized something, so a reload that
        only removes prior objects repairs nothing and must stay marked."""
        engine = _engine()
        assert engine.load_scene(self._scene_file(tmp_path))["status"] == "success"
        assert engine._physics_view_stale is False

        # A prior scene whose objects are removed, with nothing new to realize.
        # The removed object must be DYNAMIC for this to leave the mark: measured on
        # an A10G, removing a static prim leaves a Franka's 9 joint keys intact,
        # while removing a dynamic one makes PhysX log "prim ... was deleted while
        # being used by a shape in a tensor view class. The physics.tensors
        # simulationView was invalidated" - and the next joint read HUNG until a
        # 2-minute timeout. This test previously used a static table and passed only
        # because the mark was unconditional.
        engine._scene_objects = {"crate"}
        engine._objects["crate"] = _ObjectState(
            name="crate", prim_path="/World/Objects/crate", shape="cuboid", is_static=False
        )
        empty = tmp_path / "empty.xml"
        empty.write_text('<mujoco model="empty"><worldbody/></mujoco>')
        assert engine.load_scene(str(empty))["status"] == "success"
        assert engine._physics_view_stale is True
        assert engine.step(2)["status"] == "error"

    def test_a_load_that_removes_only_static_objects_leaves_no_mark(self, fake_isaacsim: None, tmp_path: Any) -> None:
        """The other half of the measured asymmetry, and the case that disabled
        examples/isaac_gs: a static prim is not held by a shape in the tensor view,
        so releasing it invalidates nothing and ``step`` must not be affected."""
        engine = _engine()
        assert engine.load_scene(self._scene_file(tmp_path))["status"] == "success"

        engine._scene_objects = {"table"}
        engine._objects["table"] = _ObjectState(
            name="table", prim_path="/World/Objects/table", shape="cuboid", is_static=True
        )
        empty = tmp_path / "empty.xml"
        empty.write_text('<mujoco model="empty"><worldbody/></mujoco>')

        assert engine.load_scene(str(empty))["status"] == "success"

        assert engine._physics_view_stale is False
        assert engine.step(2)["status"] == "success"


class TestTheMarkSurvivesAnInstanceBuiltWithoutInit:
    """``IsaacSimulation.__new__`` never runs ``__init__``.

    24 test modules build their engine that way and seed only the state they
    exercise, so a field living solely as an instance attribute made ``step``
    and ``get_observation`` raise ``AttributeError`` on all of them. The class
    default is what this pins - and it pins the *value*, because a default of
    ``True`` would refuse every one of those engines instead.
    """

    def test_a_skeleton_engine_reads_not_stale(self) -> None:
        assert IsaacSimulation.__new__(IsaacSimulation)._physics_view_stale is False

    def test_the_class_carries_the_default(self) -> None:
        assert IsaacSimulation._physics_view_stale is False

    def test_a_real_engine_does_not_depend_on_it(self) -> None:
        """``__init__`` assigns the field itself, so the default is a tolerance
        for skeletons rather than the production path's source of truth."""
        import inspect

        source = inspect.getsource(IsaacSimulation.__init__)
        assert "self._physics_view_stale = False" in source


class TestALiveSceneStepsNormally:
    """Control: the gate is inert until a body mutation marks the scene."""

    def test_an_untouched_scene_steps(self, fake_isaacsim: None) -> None:
        engine = _engine()
        result = engine.step(3)
        assert result["status"] == "success", result
        assert engine._world.step_calls == 3

    def test_the_gate_sits_behind_the_existing_refusals(self, fake_isaacsim: None) -> None:
        """A marked scene with no world still reports the world, not the mark:
        the caller cannot reset their way out of a torn-down world, so the
        settled message has to win."""
        engine = _engine()
        _add_cube(engine)
        engine._world_created = False
        assert "No world created." in _text(engine.step(1))

        engine._world_created = True
        engine._world = None
        assert "World not initialized." in _text(engine.step(1))
