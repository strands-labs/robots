"""IsaacSimulation.load_scene physics-view lifecycle regression pins (#1802).

The scene-realization step must rebuild the PhysX simulation view WITHOUT
``world.reset()``: a full reset re-applies every registered prim's default
state on ``post_reset``, which was measured (2026-07-31, Isaac 6.0 / L4) to
destabilize the already-posed Franka articulation into a PhysX
"Illegal BroadPhaseUpdateData - non-finite bounds" explosion within ~2 s of
stepping. These tests drive ``load_scene`` through a ``__new__``-skeleton
``IsaacSimulation`` (same pattern as ``test_dataset_recording.py``) with the
prim-mutation surface mocked, pinning:

* ``world.reset()`` is NEVER called from ``load_scene`` (the explosion pin);
* every robot articulation handle is re-initialized after objects are
  realized (the ``state.gripper``-missing pin: a stale handle makes
  ``get_joint_positions`` return nothing);
* a prior episode's scene objects are removed before the new episode's are
  added (episode-2 ``DynamicCuboid`` constructions crash with "Failed to get
  rigid body velocities from backend" against a stale-but-non-None view).

The live-Kit halves of these behaviours are exercised by
``tests_integ/simulation/test_isaac_body_state_gpu.py`` and the LIBERO GPU
drivers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState

_SCENE_MJCF = """
<mujoco model="scene_probe">
  <worldbody>
    <body name="fixture_table" pos="0.0 0.0 0.4">
      <geom type="box" size="0.5 0.5 0.4"/>
    </body>
    <body name="cube_1_main" pos="0.1 0.0 0.85">
      <joint type="free"/>
      <geom type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


class _RecordingWorld:
    """World stub recording lifecycle calls; ``reset`` is the forbidden one."""

    def __init__(self) -> None:
        self.reset_calls = 0
        self.step_calls = 0
        self.physics_sim_view = object()

    def reset(self) -> None:
        self.reset_calls += 1

    def step(self, render: bool = True) -> None:
        self.step_calls += 1


class _RecordingArticulation:
    def __init__(self) -> None:
        self.initialize_calls: list[Any] = []

    def initialize(self, physics_sim_view: Any = None) -> None:
        self.initialize_calls.append(physics_sim_view)


@dataclass
class _Calls:
    """Names passed to the mocked add_object / remove_object, in order."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


def _make_engine(with_robot: bool = True) -> tuple[IsaacSimulation, _Calls]:
    engine = IsaacSimulation.__new__(IsaacSimulation)
    engine._lock = threading.RLock()
    engine._world = _RecordingWorld()
    engine._world_created = True
    engine._objects = {}
    engine._robots = {}
    engine._scene_objects = set()
    engine._prim_registry = []
    calls = _Calls()
    if with_robot:
        robot = _RobotState(name="robot", prim_path="/World/Robots/robot", joint_names=[])
        robot.articulation = _RecordingArticulation()
        engine._robots["robot"] = robot

    def _fake_add_object(name: str, **kwargs: Any) -> dict[str, Any]:
        calls.added.append(name)
        return {"status": "success", "content": [{"text": f"Object '{name}' added."}]}

    def _fake_remove_object(name: str) -> dict[str, Any]:
        calls.removed.append(name)
        engine._objects.pop(name, None)
        return {"status": "success", "content": [{"text": f"Object '{name}' removed."}]}

    # setattr on the instance shadows the bound methods; the fakes accept the
    # same (name, **kwargs) call shape load_scene uses.
    setattr(engine, "add_object", _fake_add_object)  # noqa: B010 - mypy-safe method shadow
    setattr(engine, "remove_object", _fake_remove_object)  # noqa: B010 - mypy-safe method shadow
    return engine, calls


@pytest.fixture
def scene_file(tmp_path):
    p = tmp_path / "scene.xml"
    p.write_text(_SCENE_MJCF)
    return str(p)


def test_load_scene_never_calls_world_reset(scene_file) -> None:
    """The #1802 explosion pin: scene realization must not world.reset().

    ``World.reset()`` re-applies registered default states on ``post_reset``
    and was observed to blow the Franka articulation into non-finite PhysX
    transforms. If this assertion starts failing, re-read the view-rebuild
    comment in ``load_scene`` before reaching for ``reset()``.
    """
    engine, calls = _make_engine()
    result = engine.load_scene(scene_file)
    assert result["status"] == "success"
    assert calls.added == ["fixture_table", "cube_1_main"]
    assert engine._world.reset_calls == 0


def test_load_scene_reinitializes_robot_articulations(scene_file) -> None:
    """Stale articulation handles -> empty joint obs -> no state.gripper."""
    engine, _ = _make_engine()
    result = engine.load_scene(scene_file)
    assert result["status"] == "success"
    art = engine._robots["robot"].articulation
    assert art is not None
    assert len(art.initialize_calls) == 1


def test_load_scene_skips_reinit_for_handleless_robot(scene_file) -> None:
    """Phase-1 stub robots (no articulation) don't abort the load."""
    engine, _ = _make_engine()
    engine._robots["stub"] = _RobotState(name="stub", prim_path="/World/Robots/stub", joint_names=[])
    result = engine.load_scene(scene_file)
    assert result["status"] == "success"


def test_load_scene_reload_removes_prior_scene_objects_first(scene_file) -> None:
    """Episode-2 reload: prior scene objects are removed before new adds."""
    engine, calls = _make_engine()
    assert engine.load_scene(scene_file)["status"] == "success"
    # Simulate the registry state load_scene leaves behind. Only membership
    # is read on the reload path, so a bare object stands in for the state.
    for name in calls.added:
        engine._objects[name] = object()  # type: ignore[assignment]
    calls.added.clear()

    assert engine.load_scene(scene_file)["status"] == "success"
    # _scene_objects is a set, so removal order is unspecified.
    assert sorted(calls.removed) == ["cube_1_main", "fixture_table"]
    assert calls.added == ["fixture_table", "cube_1_main"]


def test_load_scene_no_world_errors(scene_file) -> None:
    engine, _ = _make_engine()
    engine._world_created = False
    result = engine.load_scene(scene_file)
    assert result["status"] == "error"
    assert "no world" in result["content"][0]["text"].lower()
