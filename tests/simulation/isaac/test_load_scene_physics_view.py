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
        self.play_calls = 0
        self.play_raises: Exception | None = None
        self.events: list[str] = []
        self.physics_sim_view = object()

    def reset(self) -> None:
        self.reset_calls += 1

    def step(self, render: bool = True) -> None:
        self.step_calls += 1

    def play(self) -> None:
        self.play_calls += 1
        self.events.append("play")
        if self.play_raises is not None:
            raise self.play_raises


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


# --- Timeline restart (#1820) ------------------------------------------------
#
# Every dynamic prim realized by load_scene stops the timeline (#159), and
# before #1820 nothing restarted it: the whole episode ran on frozen physics
# (SimulationContext.step only integrates when is_playing(); joint reads went
# stale; send_action targeted a view that never integrates) while every
# envelope still reported success. These pins assert the restart contract:
# ``world.play()`` (NOT a bare ``timeline.play()``, which on 6.0.x only
# queues the state change and never lands on the headless step path) fires
# AFTER the articulation re-init, a failed restart is a fatal error envelope
# (never a silent warn-and-continue), and a fake ``omni.timeline`` that
# reports the play never landed also fails loud.


def test_load_scene_restarts_timeline_after_articulation_reinit(scene_file) -> None:
    """world.play() fires exactly once, and only after the articulation
    re-init. Ordering matters: the stop -> flush -> initialize_physics ->
    articulation.initialize -> play cycle was verified live (Isaac 6.0 / L4)
    to preserve articulation state; playing earlier lets the queued #159
    STOP handlers null the freshly-built physics handles."""
    engine, _ = _make_engine()
    world = engine._world
    art = engine._robots["robot"].articulation
    assert art is not None
    original_initialize = art.initialize

    def _recording_initialize(physics_sim_view=None) -> None:
        world.events.append("articulation_initialize")
        original_initialize(physics_sim_view)

    art.initialize = _recording_initialize  # type: ignore[method-assign]

    result = engine.load_scene(scene_file)
    assert result["status"] == "success"
    assert world.events == ["articulation_initialize", "play"]
    assert world.play_calls == 1


def test_load_scene_play_failure_is_a_fatal_error_envelope(scene_file) -> None:
    """A failed restart must NOT be warn-and-continue: the timeline was
    stopped by the dynamic-prim constructors, so continuing runs the whole
    episode on frozen physics while every action reports success - the
    exact defect #1820 fixes."""
    engine, _ = _make_engine()
    engine._world.play_raises = RuntimeError("kit session torn down")

    result = engine.load_scene(scene_file)
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "FROZEN" in text
    assert "timeline" in text.lower()


def test_load_scene_nonlanding_play_is_a_fatal_error_envelope(scene_file, monkeypatch) -> None:
    """world.play() returning is not proof the play LANDED: on 6.0.x the
    timeline state changes on a kit update tick, so load_scene reads
    ``is_playing()`` back and refuses to hand out a frozen episode."""
    import sys
    import types

    fake_timeline_mod = types.SimpleNamespace(
        get_timeline_interface=lambda: types.SimpleNamespace(is_playing=lambda: False)
    )
    fake_omni = types.ModuleType("omni")
    fake_omni.timeline = fake_timeline_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.timeline", fake_timeline_mod)  # type: ignore[arg-type]

    engine, _ = _make_engine()
    result = engine.load_scene(scene_file)
    assert result["status"] == "error"
    assert "still stopped" in result["content"][0]["text"]


def test_load_scene_playing_verification_passes(scene_file, monkeypatch) -> None:
    """The happy live path: play lands, is_playing() confirms, load succeeds."""
    import sys
    import types

    fake_timeline_mod = types.SimpleNamespace(
        get_timeline_interface=lambda: types.SimpleNamespace(is_playing=lambda: True)
    )
    fake_omni = types.ModuleType("omni")
    fake_omni.timeline = fake_timeline_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    monkeypatch.setitem(sys.modules, "omni.timeline", fake_timeline_mod)  # type: ignore[arg-type]

    engine, calls = _make_engine()
    result = engine.load_scene(scene_file)
    assert result["status"] == "success"
    assert engine._world.play_calls == 1
    assert calls.added == ["fixture_table", "cube_1_main"]


def test_load_scene_missing_omni_timeline_skips_verification(scene_file, monkeypatch) -> None:
    """Without omni.timeline there is no live Kit session (the CPU
    skeleton, where the #159 stop never ran either): world.play() is still
    issued, but the is_playing() read-back is skipped."""
    import sys
    import types

    fake_omni = types.ModuleType("omni")
    monkeypatch.setitem(sys.modules, "omni", fake_omni)
    # A None sys.modules entry makes ``import omni.timeline`` raise ImportError.
    monkeypatch.setitem(sys.modules, "omni.timeline", None)  # type: ignore[arg-type]

    engine, calls = _make_engine()
    result = engine.load_scene(scene_file)
    assert result["status"] == "success"
    assert engine._world.play_calls == 1
    assert calls.added == ["fixture_table", "cube_1_main"]
