"""Regression tests for the shipped built-in benchmark specs.

The floating-base predicate/reward DSL (``base_velocity_tracking`` /
``base_height`` / ``base_orientation`` reward terms and the ``base_beyond_x`` /
``base_tipped`` / ``base_below_z`` predicates) was complete but wired into no
runnable benchmark: :func:`list_benchmarks` was empty until a caller
hand-authored a spec. :func:`register_builtin_benchmarks` ships a canonical
velocity-tracking locomotion benchmark (``go2_walk_forward``) composed from
those primitives so a floating-base robot has a discoverable, runnable eval out
of the box; g1_walk_forward and t1_walk_forward are the humanoid counterparts.

These tests (1) pin that ``register_builtin_benchmarks`` puts ``go2_walk_forward``
into the registry with the right metadata, (2) drive the compiled
success/failure/dense-reward functions on a REAL inline floating-base MuJoCo sim
at KNOWN base poses to prove the shipped spec's DSL wiring works on physics, and
(3) pin the opt-in / idempotent / non-mutating contract. They are GL-free
(``get_observation`` with ``skip_images``) so they run in CI without a display,
and they use no downloaded robot asset (the compiled predicates are driven
directly, bypassing the ``supported_robots`` robot-load check that only fires in
``on_episode_start``).
"""

import math
import os
import tempfile

import mujoco
import pytest

from strands_robots.simulation.benchmark import (
    get_benchmark,
    list_benchmarks,
    unregister_benchmark,
)
from strands_robots.simulation.benchmark_spec import DeclarativeBenchmark
from strands_robots.simulation.builtin_benchmarks import (
    builtin_benchmark_specs,
    register_builtin_benchmarks,
)
from strands_robots.simulation.mujoco.simulation import Simulation

# Floating base with a NAMED free joint plus one actuated hinge (mirrors the
# base_tipped / base_below_z regression fixtures). get_observation surfaces
# base_pos / base_quat / base_lin_vel / base_ang_vel for this robot, which is
# all the go2_walk_forward predicates read.
NAMED_BASE_XML = """
<mujoco model="test_named_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="pelvis" pos="0 0 0.8">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.3 0.3 0.8 1"/>
      <body name="thigh" pos="0 0 -0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.3" rgba="0.8 0.3 0.3 1"/>
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_act" joint="hip"/>
  </actuator>
</mujoco>
"""


def _quat_pitch(deg: float) -> list[float]:
    """Unit (w, x, y, z) quaternion for a pitch (rotation about world +y)."""
    h = math.radians(deg) / 2.0
    return [math.cos(h), 0.0, math.sin(h), 0.0]


def _quat_yaw(deg: float) -> list[float]:
    """Unit (w, x, y, z) quaternion for a yaw turn (rotation about world +z)."""
    h = math.radians(deg) / 2.0
    return [math.cos(h), 0.0, 0.0, math.sin(h)]


@pytest.fixture
def sim():
    s = Simulation(tool_name="test_builtin_benchmarks", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    s.cleanup()


@pytest.fixture(autouse=True)
def _clean_registry():
    """Keep the module-global benchmark registry clean around each test."""
    for _n in ("go2_walk_forward", "g1_walk_forward", "t1_walk_forward", "go2_strafe_left", "go2_turn_left"):
        unregister_benchmark(_n)
    yield
    for _n in ("go2_walk_forward", "g1_walk_forward", "t1_walk_forward", "go2_strafe_left", "go2_turn_left"):
        unregister_benchmark(_n)


def _write(xml: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "model.xml")
    with open(p, "w") as f:
        f.write(xml)
    return p


def _set_base_pose(sim, x: float = 0.0, z: float = 0.8, quat_wxyz=None) -> None:
    """Set the robot's (only) free joint to a known world pose (x, z, orientation)."""
    if quat_wxyz is None:
        quat_wxyz = [1.0, 0.0, 0.0, 0.0]
    model, data = sim._world._model, sim._world._data
    jid = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr : qadr + 7] = [float(x), 0.0, float(z), *quat_wxyz]
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# Registration + discovery contract
# ---------------------------------------------------------------------------
def test_register_builtin_benchmarks_registers_go2_walk_forward():
    """The shipped go2_walk_forward benchmark is absent until registered, then
    discoverable via list_benchmarks with the expected metadata."""
    assert "go2_walk_forward" not in list_benchmarks()
    names = register_builtin_benchmarks()
    assert "go2_walk_forward" in names
    snap = list_benchmarks()
    assert "go2_walk_forward" in snap
    meta = snap["go2_walk_forward"]
    assert meta["class"] == "DeclarativeBenchmark"
    assert meta["supported_robots"] == ["unitree_go2"]
    assert meta["default_robot"] == "unitree_go2"
    assert meta["max_steps"] == 1000


def test_register_builtin_benchmarks_is_idempotent():
    """Re-registering overwrites without raising and keeps a single entry."""
    register_builtin_benchmarks()
    first = get_benchmark("go2_walk_forward")
    register_builtin_benchmarks()
    second = get_benchmark("go2_walk_forward")
    assert isinstance(first, DeclarativeBenchmark)
    assert isinstance(second, DeclarativeBenchmark)
    # a fresh instance each call (compiled from the spec), still one registry key
    assert list(list_benchmarks()).count("go2_walk_forward") == 1


def test_builtin_benchmark_specs_returns_defensive_copies():
    """Callers get deep copies; mutating one cannot corrupt the shipped spec."""
    a = builtin_benchmark_specs()
    assert "go2_walk_forward" in a
    a["go2_walk_forward"]["max_steps"] = -999
    a["go2_walk_forward"]["success"]["all"].clear()
    b = builtin_benchmark_specs()
    assert b["go2_walk_forward"]["max_steps"] == 1000
    assert b["go2_walk_forward"]["success"]["all"]


def test_facade_register_builtin_benchmarks(sim):
    """The SimEngine facade registers the built-ins and reports them."""
    res = sim.register_builtin_benchmarks()
    assert res["status"] == "success"
    payload = next(c["json"] for c in res["content"] if "json" in c)
    assert "go2_walk_forward" in payload["registered"]
    assert "go2_walk_forward" in list_benchmarks()


# ---------------------------------------------------------------------------
# The shipped spec compiles into predicates that work on real physics
# ---------------------------------------------------------------------------
def test_go2_walk_forward_success_fires_on_forward_progress(sim):
    """success = base_beyond_x(2.0): False until the base walks past x=2 m."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0)
    assert bench.is_success(sim) is False
    _set_base_pose(sim, x=1.5)
    assert bench.is_success(sim) is False  # short of the 2 m line
    _set_base_pose(sim, x=3.0)
    assert bench.is_success(sim) is True


def test_go2_walk_forward_failure_fires_on_topple(sim):
    """failure includes base_tipped(0.7): a level base continues, a toppled one
    terminates the episode."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, quat_wxyz=[1.0, 0.0, 0.0, 0.0])
    assert bench.is_failure(sim) is False
    _set_base_pose(sim, x=0.0, quat_wxyz=_quat_pitch(90.0))  # on its side
    assert bench.is_failure(sim) is True


def test_go2_walk_forward_failure_fires_on_height_collapse(sim):
    """failure includes base_below_z(0.18): a standing base continues, a
    collapsed one terminates."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, z=0.32)  # nominal Go2 stance
    assert bench.is_failure(sim) is False
    _set_base_pose(sim, x=0.0, z=0.1)  # collapsed below 0.18
    assert bench.is_failure(sim) is True


def test_go2_walk_forward_dense_reward_is_finite(sim):
    """on_step sums base_velocity_tracking + base_height + base_orientation into
    a finite dense reward the RL/BC loop can shape on."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, z=0.32)
    info = bench.on_step(sim, {}, {})
    assert math.isfinite(info.reward)
    assert info.done is False


# ---------------------------------------------------------------------------
# g1_walk_forward: the humanoid counterpart - same DSL, biped thresholds
# ---------------------------------------------------------------------------
def test_register_builtin_benchmarks_ships_both_go2_and_g1():
    """register_builtin_benchmarks ships the quadruped AND humanoid tasks; both
    are absent until registered, then discoverable with the right metadata."""
    assert "g1_walk_forward" not in list_benchmarks()
    names = register_builtin_benchmarks()
    assert set(names) >= {"go2_walk_forward", "g1_walk_forward"}
    assert names == sorted(names)  # returned sorted
    snap = list_benchmarks()
    meta = snap["g1_walk_forward"]
    assert meta["class"] == "DeclarativeBenchmark"
    assert meta["supported_robots"] == ["unitree_g1"]
    assert meta["default_robot"] == "unitree_g1"
    assert meta["max_steps"] == 1000


def test_builtin_specs_include_g1_with_biped_thresholds():
    """The shipped g1 spec uses the humanoid height/collapse thresholds and the
    fuller anti-bounce / anti-wobble regularizer stack (grounded in the G1's
    ~0.79 m standing base height), not the Go2's quadruped thresholds."""
    specs = builtin_benchmark_specs()
    g1 = specs["g1_walk_forward"]
    # height-collapse failure well below the ~0.79 m stance, well above 0
    collapse = next(p for p in g1["failure"]["any"] if p["predicate"] == "base_below_z")
    assert collapse["z"] == 0.4
    height = next(r for r in g1["dense_reward"] if r["predicate"] == "base_height")
    assert height["target"] == 0.78
    # the biped-specific regularizers are present (Go2 stack does not use them)
    reward_terms = {r["predicate"] for r in g1["dense_reward"]}
    assert {"base_lin_vel_z", "base_ang_vel_xy"} <= reward_terms


def test_g1_walk_forward_success_fires_on_forward_progress(sim):
    """success = base_beyond_x(2.0): False until the base walks past x=2 m."""
    register_builtin_benchmarks()
    bench = get_benchmark("g1_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, z=0.78)
    assert bench.is_success(sim) is False
    _set_base_pose(sim, x=1.5, z=0.78)
    assert bench.is_success(sim) is False  # short of the 2 m line
    _set_base_pose(sim, x=3.0, z=0.78)
    assert bench.is_success(sim) is True


def test_g1_walk_forward_failure_fires_on_topple(sim):
    """failure includes base_tipped(0.7): a level humanoid base continues, a
    toppled one terminates the episode."""
    register_builtin_benchmarks()
    bench = get_benchmark("g1_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, z=0.78, quat_wxyz=[1.0, 0.0, 0.0, 0.0])
    assert bench.is_failure(sim) is False
    _set_base_pose(sim, x=0.0, z=0.78, quat_wxyz=_quat_pitch(90.0))  # on its side
    assert bench.is_failure(sim) is True


def test_g1_walk_forward_failure_fires_on_height_collapse(sim):
    """failure includes base_below_z(0.4): the ~0.79 m standing G1 continues, a
    collapsed pelvis (below 0.4 m) terminates. The Go2's 0.18 m threshold would
    NOT catch a humanoid that has folded to ~0.3 m - hence the biped threshold."""
    register_builtin_benchmarks()
    bench = get_benchmark("g1_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, z=0.78)  # nominal G1 stance
    assert bench.is_failure(sim) is False
    _set_base_pose(sim, x=0.0, z=0.3)  # collapsed below 0.4 (Go2's 0.18 would miss this)
    assert bench.is_failure(sim) is True


def test_g1_walk_forward_dense_reward_is_finite(sim):
    """on_step sums the full biped stack (velocity tracking + height +
    orientation + lin_vel_z + ang_vel_xy) into a finite dense reward."""
    register_builtin_benchmarks()
    bench = get_benchmark("g1_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, z=0.78)
    info = bench.on_step(sim, {}, {})
    assert math.isfinite(info.reward)
    assert info.done is False


# ---------------------------------------------------------------------------
# t1_walk_forward: a second humanoid - same biped DSL, T1-scale thresholds
# ---------------------------------------------------------------------------
def test_register_builtin_benchmarks_ships_go2_g1_and_t1():
    """register_builtin_benchmarks ships the quadruped AND both humanoid tasks;
    t1_walk_forward is absent until registered, then discoverable with the T1's
    own metadata."""
    assert "t1_walk_forward" not in list_benchmarks()
    names = register_builtin_benchmarks()
    assert set(names) >= {"go2_walk_forward", "g1_walk_forward", "t1_walk_forward"}
    assert names == sorted(names)  # returned sorted
    snap = list_benchmarks()
    meta = snap["t1_walk_forward"]
    assert meta["class"] == "DeclarativeBenchmark"
    assert meta["supported_robots"] == ["booster_t1"]
    assert meta["default_robot"] == "booster_t1"
    assert meta["max_steps"] == 1000


def test_builtin_specs_include_t1_with_its_own_biped_thresholds():
    """The shipped t1 spec uses the T1's own measured stance thresholds (~0.665 m
    standing), distinct from the taller G1 (~0.79 m), while carrying the same
    biped anti-bounce / anti-wobble regularizer stack. Grounding each biped in
    its own stance is why they are separate specs, not one shared humanoid spec."""
    specs = builtin_benchmark_specs()
    t1 = specs["t1_walk_forward"]
    # height-collapse failure below the ~0.665 m stance, well above 0, and
    # distinct from the G1's 0.4 m line (grounded in the T1's own geometry)
    collapse = next(p for p in t1["failure"]["any"] if p["predicate"] == "base_below_z")
    assert collapse["z"] == 0.35
    height = next(r for r in t1["dense_reward"] if r["predicate"] == "base_height")
    assert height["target"] == 0.66
    # distinct from the G1's taller stance
    g1_height = next(r for r in specs["g1_walk_forward"]["dense_reward"] if r["predicate"] == "base_height")
    assert height["target"] != g1_height["target"]
    # the biped-specific regularizers carry over from the G1 (Go2 lacks them)
    reward_terms = {r["predicate"] for r in t1["dense_reward"]}
    assert {"base_lin_vel_z", "base_ang_vel_xy"} <= reward_terms


def test_t1_walk_forward_success_fires_on_forward_progress(sim):
    """success = base_beyond_x(2.0): False until the base walks past x=2 m."""
    register_builtin_benchmarks()
    bench = get_benchmark("t1_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, z=0.66)
    assert bench.is_success(sim) is False
    _set_base_pose(sim, x=1.5, z=0.66)
    assert bench.is_success(sim) is False  # short of the 2 m line
    _set_base_pose(sim, x=3.0, z=0.66)
    assert bench.is_success(sim) is True


def test_t1_walk_forward_failure_fires_on_topple(sim):
    """failure includes base_tipped(0.7): a level humanoid base continues, a
    toppled one terminates the episode."""
    register_builtin_benchmarks()
    bench = get_benchmark("t1_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, z=0.66, quat_wxyz=[1.0, 0.0, 0.0, 0.0])
    assert bench.is_failure(sim) is False
    _set_base_pose(sim, x=0.0, z=0.66, quat_wxyz=_quat_pitch(90.0))  # on its side
    assert bench.is_failure(sim) is True


def test_t1_walk_forward_failure_fires_on_height_collapse(sim):
    """failure includes base_below_z(0.35): the ~0.665 m standing T1 continues, a
    collapsed pelvis (below 0.35 m) terminates. The threshold is grounded in the
    T1's own stance, distinct from the taller G1's 0.4 m line."""
    register_builtin_benchmarks()
    bench = get_benchmark("t1_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, z=0.66)  # nominal T1 stance
    assert bench.is_failure(sim) is False
    _set_base_pose(sim, x=0.0, z=0.25)  # collapsed below 0.35
    assert bench.is_failure(sim) is True


def test_t1_walk_forward_dense_reward_is_finite(sim):
    """on_step sums the full biped stack (velocity tracking + height +
    orientation + lin_vel_z + ang_vel_xy) into a finite dense reward."""
    register_builtin_benchmarks()
    bench = get_benchmark("t1_walk_forward")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_pose(sim, x=0.0, z=0.66)
    info = bench.on_step(sim, {}, {})
    assert math.isfinite(info.reward)
    assert info.done is False


# ---------------------------------------------------------------------------
# go2_strafe_left: the LATERAL counterpart - same quadruped/thresholds, but a
# pure vy command scored by base_beyond_y (the vx/base_beyond_x tasks never
# exercise the lateral axis).
# ---------------------------------------------------------------------------
def _set_base_y(sim, y: float, x: float = 0.0, z: float = 0.32, quat_wxyz=None) -> None:
    """Set the (only) free joint to world (x, y, z) - a y-aware pose setter for
    the strafe task (the shared _set_base_pose fixes y=0)."""
    if quat_wxyz is None:
        quat_wxyz = [1.0, 0.0, 0.0, 0.0]
    model, data = sim._world._model, sim._world._data
    jid = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr : qadr + 7] = [float(x), float(y), float(z), *quat_wxyz]
    mujoco.mj_forward(model, data)


def test_register_builtin_benchmarks_registers_go2_strafe_left():
    """go2_strafe_left is absent until registered, then discoverable + runnable."""
    assert "go2_strafe_left" not in list_benchmarks()
    names = register_builtin_benchmarks()
    assert "go2_strafe_left" in names
    assert "go2_strafe_left" in list_benchmarks()
    assert get_benchmark("go2_strafe_left") is not None


def test_go2_strafe_left_success_fires_on_lateral_progress(sim):
    """success = base_beyond_y(1.0): False until the base strafes past y=1 m, and
    forward x-progress alone must NOT satisfy it (distinct from go2_walk_forward)."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_strafe_left")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_y(sim, y=0.0)
    assert bench.is_success(sim) is False
    _set_base_y(sim, y=0.5)
    assert bench.is_success(sim) is False  # short of the 1 m lateral line
    _set_base_y(sim, y=0.0, x=3.0)  # walked far FORWARD but zero lateral
    assert bench.is_success(sim) is False, "forward progress must not score a strafe goal"
    _set_base_y(sim, y=2.0)  # strafed well left
    assert bench.is_success(sim) is True


def test_go2_strafe_left_failure_fires_on_topple(sim):
    """failure includes base_tipped(0.7): a level base continues, a toppled one
    terminates - the same fall vocabulary as the forward task."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_strafe_left")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_y(sim, y=0.0, quat_wxyz=[1.0, 0.0, 0.0, 0.0])
    assert bench.is_failure(sim) is False
    _set_base_y(sim, y=0.0, quat_wxyz=_quat_pitch(90.0))  # on its side
    assert bench.is_failure(sim) is True


def test_go2_strafe_left_failure_fires_on_height_collapse(sim):
    """failure includes base_below_z(0.18): a standing base continues, a
    collapsed one terminates."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_strafe_left")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_y(sim, y=0.0, z=0.32)  # nominal Go2 stance
    assert bench.is_failure(sim) is False
    _set_base_y(sim, y=0.0, z=0.1)  # collapsed below 0.18
    assert bench.is_failure(sim) is True


def test_go2_strafe_left_dense_reward_is_finite_and_tracks_vy(sim):
    """on_step sums the lateral base_velocity_tracking + base_height +
    base_orientation into a finite dense reward the RL/BC loop can shape on."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_strafe_left")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_y(sim, y=0.0, z=0.32)
    info = bench.on_step(sim, {}, {})
    assert math.isfinite(info.reward)
    assert info.done is False


# ---------------------------------------------------------------------------
# go2_turn_left: the YAW counterpart - same quadruped/thresholds, a wz command
# and a base_yaw_beyond heading goal (completes the vx/vy/wz vocabulary)
# ---------------------------------------------------------------------------
def _set_base_yaw(sim, deg: float, z: float = 0.32) -> None:
    """Set the free joint to a KNOWN yaw heading at the origin (turn-in-place)."""
    model, data = sim._world._model, sim._world._data
    jid = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr : qadr + 7] = [0.0, 0.0, float(z), *_quat_yaw(deg)]
    mujoco.mj_forward(model, data)


def test_register_builtin_benchmarks_registers_go2_turn_left():
    """go2_turn_left is absent until registered, then discoverable + runnable."""
    assert "go2_turn_left" not in list_benchmarks()
    names = register_builtin_benchmarks()
    assert "go2_turn_left" in names
    assert "go2_turn_left" in list_benchmarks()
    assert get_benchmark("go2_turn_left") is not None


def test_builtin_specs_include_go2_turn_left_with_a_pure_yaw_command():
    """The shipped turn spec commands a PURE yaw twist (vx=vy=0, wz>0) and scores
    a heading goal with base_yaw_beyond - the wz axis the walk/strafe tasks leave
    at zero."""
    spec = builtin_benchmark_specs()["go2_turn_left"]
    succ = spec["success"]["all"][0]
    assert succ["predicate"] == "base_yaw_beyond"
    assert succ["yaw"] == 1.0
    track = next(r for r in spec["dense_reward"] if r["predicate"] == "base_velocity_tracking")
    assert track["vx"] == 0.0 and track["vy"] == 0.0 and track["wz"] == 0.5


def test_go2_turn_left_success_fires_on_yaw_progress(sim):
    """success = base_yaw_beyond(1.0): False until the base turns past ~1 rad left,
    and pure forward displacement (no turn) never satisfies it."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_turn_left")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_yaw(sim, deg=0.0)
    assert bench.is_success(sim) is False
    _set_base_yaw(sim, deg=45.0)  # ~0.79 rad, short of the 1.0 rad line
    assert bench.is_success(sim) is False
    _set_base_pose(sim, x=3.0)  # walked far forward but never turned
    assert bench.is_success(sim) is False
    _set_base_yaw(sim, deg=90.0)  # turned well left
    assert bench.is_success(sim) is True


def test_go2_turn_left_failure_fires_on_topple(sim):
    """failure includes base_tipped(0.7): a level (turned) base continues, a
    toppled one terminates - vetoing a 'turned then fell' rollout."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_turn_left")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_yaw(sim, deg=90.0)  # turned, upright
    assert bench.is_failure(sim) is False
    _set_base_pose(sim, x=0.0, quat_wxyz=_quat_pitch(90.0))  # on its side
    assert bench.is_failure(sim) is True


def test_go2_turn_left_failure_fires_on_height_collapse(sim):
    """failure includes base_below_z(0.18): a standing base continues, a
    collapsed one terminates."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_turn_left")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_yaw(sim, deg=90.0, z=0.32)  # nominal Go2 stance, turned
    assert bench.is_failure(sim) is False
    _set_base_yaw(sim, deg=90.0, z=0.1)  # collapsed below 0.18
    assert bench.is_failure(sim) is True


def test_go2_turn_left_dense_reward_is_finite_and_tracks_wz(sim):
    """on_step sums the yaw base_velocity_tracking + base_height + base_orientation
    into a finite dense reward the RL/BC loop can shape on."""
    register_builtin_benchmarks()
    bench = get_benchmark("go2_turn_left")
    sim.add_robot("floater", urdf_path=_write(NAMED_BASE_XML))
    _set_base_yaw(sim, deg=0.0, z=0.32)
    info = bench.on_step(sim, {}, {})
    assert math.isfinite(info.reward)
    assert info.done is False


# ---------------------------------------------------------------------------
# Cross-spec sanity invariants (GL-free, standard unit gate)
#
# The tests above pin each spec's SPECIFIC threshold values and drive its
# predicates at hand-picked poses. The end-to-end "a freshly-spawned upright
# robot does not already trip failure / satisfy success" (dead-on-arrival)
# contract, however, is asserted only against the REAL robot in
# ``tests_integ/simulation/test_builtin_benchmark_load.py`` - network + MuJoCo +
# GPU, collected only via ``hatch run test-integ``, NOT the PR CI gate. So a
# newly-added shipped spec (the roadmap's next locomotion benchmarks - a Go1
# quadruped, terrain variants) could ship an internally-inconsistent threshold
# (e.g. a ``base_below_z`` collapse line ABOVE the ``base_height`` rewarded
# stance -> the robot AT its own rewarded stance is already "collapsed" ->
# every episode fails at t=0) and the PR CI gate would stay green, since the
# per-spec unit tests only assert the numbers their author wrote and the
# real-robot spawn guard does not run.
#
# These parametrized invariants close that gap in the standard gate. They are
# derived from :func:`builtin_benchmark_specs` so any benchmark added to the
# shipped set is covered automatically - no hardcoded list to drift.
# ---------------------------------------------------------------------------
_SUCCESS_PROGRESS_THRESHOLD_KEYS = {
    "base_beyond_x": "x",
    "base_beyond_y": "y",
    "base_yaw_beyond": "yaw",
}


def _clause_predicates(spec: dict, key: str) -> list[dict]:
    """Flatten a spec's ``success``/``failure`` all/any clause into its predicate
    entry dicts (both connectives, so the invariant is agnostic to which is used)."""
    clause = spec.get(key, {}) or {}
    return list(clause.get("all", [])) + list(clause.get("any", []))


def _find_predicate(entries: list[dict], name: str) -> dict | None:
    return next((e for e in entries if e.get("predicate") == name), None)


@pytest.mark.parametrize("bench_name", sorted(builtin_benchmark_specs()))
def test_builtin_spec_collapse_threshold_below_rewarded_stance(bench_name: str):
    """A loco benchmark's height-collapse FAILURE threshold (``base_below_z``)
    must be strictly below its rewarded nominal stance (``base_height`` target).

    Otherwise the robot AT the stance the dense reward drives it toward is
    already classified as a height-collapse failure, so the episode terminates
    at t=0 (dead-on-arrival). This is the standard-gate counterpart of the
    real-robot standing-spawn contract that runs only in ``tests_integ``."""
    spec = builtin_benchmark_specs()[bench_name]
    below = _find_predicate(_clause_predicates(spec, "failure"), "base_below_z")
    height = _find_predicate(spec.get("dense_reward", []), "base_height")
    if below is None or height is None:
        pytest.skip(f"{bench_name}: no base_below_z / base_height pair to check")
    assert below["z"] < height["target"], (
        f"{bench_name}: base_below_z(z={below['z']}) is NOT below the rewarded "
        f"base_height stance (target={height['target']}) - the nominal stance "
        f"the reward targets is classified as a height-collapse failure, so the "
        f"benchmark fails at t=0 (dead-on-arrival)."
    )


@pytest.mark.parametrize("bench_name", sorted(builtin_benchmark_specs()))
def test_builtin_spec_success_is_positive_progress(bench_name: str):
    """A loco benchmark scores POSITIVE forward/lateral/yaw progress, so a
    standing-still (or non-progressing) policy never trivially satisfies it."""
    spec = builtin_benchmark_specs()[bench_name]
    success = _clause_predicates(spec, "success")
    assert success, f"{bench_name}: empty success clause"
    progress = [
        (e, _SUCCESS_PROGRESS_THRESHOLD_KEYS[e["predicate"]])
        for e in success
        if e.get("predicate") in _SUCCESS_PROGRESS_THRESHOLD_KEYS
    ]
    assert progress, f"{bench_name}: no forward/lateral/yaw progress success predicate"
    for entry, key in progress:
        assert entry[key] > 0, (
            f"{bench_name}: success {entry['predicate']} threshold {key}={entry[key]} "
            f"is not positive - a standing-still policy would satisfy it."
        )


@pytest.mark.parametrize("bench_name", sorted(builtin_benchmark_specs()))
def test_builtin_spec_is_structurally_complete(bench_name: str):
    """Every shipped benchmark defines non-empty success, failure, and
    dense_reward clauses (a runnable eval: a terminal, a fall guard, and a dense
    shaping signal)."""
    spec = builtin_benchmark_specs()[bench_name]
    assert _clause_predicates(spec, "success"), f"{bench_name}: no success predicates"
    assert _clause_predicates(spec, "failure"), f"{bench_name}: no failure predicates"
    assert spec.get("dense_reward"), f"{bench_name}: no dense_reward terms"


@pytest.mark.parametrize("bench_name", sorted(builtin_benchmark_specs()))
def test_builtin_spec_declares_language_instruction(bench_name: str):
    """Every shipped velocity-tracking locomotion benchmark declares a
    natural-language ``instruction`` describing its command, and it compiles
    through to the benchmark's ``instruction`` property.

    A locomotion benchmark's command IS its task ("walk forward", "strafe
    left", "turn left"), so the spec carries it as the instruction. Without it
    a language-conditioned policy (the shipped GR00T ``WBCPolicy``) evaluated on
    the benchmark receives an empty string - the #187 off-task failure mode -
    and ``evaluate_benchmark`` emits a spurious empty-instruction warning on
    every eval. Before ``instruction`` was a spec key these all defaulted to
    ``""``."""
    spec = builtin_benchmark_specs()[bench_name]
    instruction = spec.get("instruction")
    assert isinstance(instruction, str) and instruction.strip(), (
        f"{bench_name}: must declare a non-empty natural-language instruction; got {instruction!r}"
    )
    # The declared instruction compiles through to the benchmark instance.
    bench = DeclarativeBenchmark.from_dict(dict(spec))
    assert bench.instruction == instruction
