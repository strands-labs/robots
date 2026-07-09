"""Scene-supplied Panda discovery + graceful degradation for the LIBERO adapter.

RoboSuite/LIBERO scenes ship their own Panda under the ``robot0_`` /
``gripper0_`` namespaces. ``LiberoAdapter._register_default_robot`` wraps that
existing arm in ``world.robots["robot"]`` WITHOUT recompiling the spec, so the
base :class:`BenchmarkProtocol` skips its unconditional ``add_robot`` call and
the scene never gets a second, camera-occluding Panda injected (#166 / #168).

The heavy end of that path needs robosuite + a compiled LIBERO scene, but the
discovery helper :func:`_build_scene_robot_wrapper` and the best-effort guards
around it are pure model-walking logic. These tests pin them with lightweight
MuJoCo-API fakes (no robosuite, no libero assets, no cached scene XML):

* the wrapper returns ``None`` when the scene has no ``robot0_`` body, so the
  base protocol falls back to adding its own Panda;
* a malformed ``body_parentid`` is swallowed rather than aborting discovery;
* joint AND actuator names from both the arm and gripper namespaces land in
  the wrapper so downstream observation code can surface ``state.gripper``;
* ``_register_default_robot`` degrades to a no-op (never registers a bogus
  ``"robot"``) when there is no world, when mujoco is unimportable, when
  discovery raises, or when no Panda is found.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

from strands_robots.benchmarks.libero import LiberoAdapter
from strands_robots.benchmarks.libero.adapter import _build_scene_robot_wrapper

_BDDL = """
(define (problem libero_discovery_probe)
  (:language "pick up the cube")
  (:objects cube_1 - object)
  (:goal (on cube_1 table_1)))
"""


class _FakeMjtObj:
    mjOBJ_BODY = 1
    mjOBJ_JOINT = 3
    mjOBJ_ACTUATOR = 5


class _FakeMj:
    """Minimal stand-in for the ``mujoco`` module's name-lookup surface."""

    mjtObj = _FakeMjtObj

    def __init__(self, *, bodies, joints, actuators):
        self._names = {
            _FakeMjtObj.mjOBJ_BODY: bodies,
            _FakeMjtObj.mjOBJ_JOINT: joints,
            _FakeMjtObj.mjOBJ_ACTUATOR: actuators,
        }

    def mj_id2name(self, model, obj, i):
        return self._names[obj][i]


class _FakeModel:
    def __init__(self, *, nbody, njnt, nu, body_parentid):
        self.nbody = nbody
        self.njnt = njnt
        self.nu = nu
        self.body_parentid = body_parentid


def _adapter() -> LiberoAdapter:
    return LiberoAdapter.from_text(_BDDL, auto_generate_scene=False, install_cameras=False)


# --- _build_scene_robot_wrapper -------------------------------------------


def test_wrapper_is_none_when_no_arm_body_matches_prefix():
    """No ``robot0_`` body -> return None so the base protocol adds its own Panda."""
    mj = _FakeMj(bodies=["world", "table_1"], joints=["slide"], actuators=[])
    model = _FakeModel(nbody=2, njnt=1, nu=0, body_parentid=[0, 0])

    assert _build_scene_robot_wrapper(mj, model, prefix="robot0_", gripper_prefix="gripper0_") is None


def test_wrapper_collects_arm_and_gripper_names_and_tolerates_bad_parentid():
    """A malformed ``body_parentid`` is swallowed; both namespaces are captured."""
    mj = _FakeMj(
        bodies=["robot0_base"],
        joints=["robot0_joint1", "gripper0_finger_joint1", "table_hinge"],
        actuators=["robot0_act1", "gripper0_act", "world_act"],
    )
    # Empty body_parentid -> indexing raises IndexError inside the parent-is-world
    # probe; discovery must still succeed off the first prefix match.
    model = _FakeModel(nbody=1, njnt=3, nu=3, body_parentid=[])

    wrapper = _build_scene_robot_wrapper(mj, model, prefix="robot0_", gripper_prefix="gripper0_")

    assert wrapper is not None
    assert wrapper.name == "robot"
    assert wrapper.namespace == "robot0_"
    assert wrapper.body_id == 0
    # Arm + gripper joints kept; unrelated scene joint dropped.
    assert wrapper.joint_names == ["robot0_joint1", "gripper0_finger_joint1"]
    assert wrapper.joint_ids == [0, 1]
    # Same prefix filter applied to actuators.
    assert wrapper.actuator_ids == [0, 1]


# --- _register_default_robot best-effort guards ----------------------------


def test_register_default_robot_noop_without_world():
    """No ``_world`` -> nothing registered, no raise."""
    adapter = _adapter()
    sim = SimpleNamespace(_world=None)

    adapter._register_default_robot(sim)  # must not raise


def test_register_default_robot_noop_when_already_registered():
    """An existing ``robot`` key is left untouched (base protocol will find it)."""
    adapter = _adapter()
    sentinel = object()
    world = SimpleNamespace(robots={"robot": sentinel}, _model=object())
    sim = SimpleNamespace(_world=world)

    adapter._register_default_robot(sim)

    assert world.robots["robot"] is sentinel


def test_register_default_robot_noop_when_mujoco_unimportable(monkeypatch):
    """mujoco missing -> skip pre-register (the ``[sim-mujoco]`` extra is absent)."""
    adapter = _adapter()
    world = SimpleNamespace(robots={}, _model=object())
    sim = SimpleNamespace(_world=world)
    monkeypatch.setitem(sys.modules, "mujoco", None)

    adapter._register_default_robot(sim)

    assert world.robots == {}


def test_register_default_robot_swallows_discovery_error(monkeypatch):
    """A discovery exception is logged and swallowed, not propagated."""
    adapter = _adapter()
    world = SimpleNamespace(robots={}, _model=object())
    sim = SimpleNamespace(_world=world)

    def _boom(*args, **kwargs):
        raise RuntimeError("model walk blew up")

    monkeypatch.setattr("strands_robots.benchmarks.libero.adapter._build_scene_robot_wrapper", _boom)

    adapter._register_default_robot(sim)  # must not raise

    assert "robot" not in world.robots


def test_register_default_robot_noop_when_no_panda_discovered(monkeypatch):
    """Wrapper None (scene has no Panda) -> base protocol adds one instead."""
    adapter = _adapter()
    world = SimpleNamespace(robots={}, _model=object())
    sim = SimpleNamespace(_world=world)

    monkeypatch.setattr(
        "strands_robots.benchmarks.libero.adapter._build_scene_robot_wrapper",
        lambda *a, **k: None,
    )

    adapter._register_default_robot(sim)

    assert "robot" not in world.robots
