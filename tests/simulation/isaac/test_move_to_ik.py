"""IK-backed ``move_to`` on the Isaac backend (GH #2155, child of #2123).

The Isaac ``move_to`` reuses the shared mink damped-least-squares bridge
(:class:`strands_robots.simulation.ik.MinkIKBridge`) on the MuJoCo model the
robot's ``data_config`` resolves to, then drives PD position targets on the
articulation. The contracts pinned here:

* validation rejections come from the shared core
  (:class:`~strands_robots.simulation.motion_primitives_base.MotionPrimitivesCore`),
  so their wording is byte-identical to MuJoCo's by construction;
* the workspace sanity box refuses a unit-mistake target up front;
* the MJCF-side solution is reconciled with the articulation through an
  explicit NAME-KEYED joint map - a solved joint with no articulation
  counterpart is a structured refusal, never a positional/flat-index write
  (AGENTS.md "Per-name state copy, not flat index"), and an articulation
  whose DOF ORDER differs from the MJCF still converges (the mapping, not
  the index, carries the write);
* convergence / timeout / unreachable / mid-run-abort answer with the same
  envelopes as the MuJoCo reference, and a policy running on the robot
  refuses the primitive (``policy_running`` is the per-robot flag every
  Isaac policy-driving loop sets).

These tests deliberately do NOT require NVIDIA Isaac Sim (the pattern of
``tests/simulation/isaac/test_motion_primitives.py``): the articulation and
the world are faked, and the IK side runs on a real inline MJCF arm.
Guard/validation tests run anywhere; joint-map tests ``importorskip`` on
``mujoco``; convergence tests ``importorskip`` on ``mink`` (the dev env
ships both; a clean base install skips them). Real-GPU integration coverage
is the tests child of #2123 and lives in ``tests_integ/``.
"""

from __future__ import annotations

import concurrent.futures
import math
import sys
import types
from typing import Any

import numpy as np
import pytest

from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState
from strands_robots.simulation.motion_primitives_base import MotionPrimitivesCore

# ---------------------------------------------------------------------------
# Fakes: articulation with PD-target servo semantics + a world pose, world
# whose step() advances the articulation, and the one isaacsim type the
# adapter lazily imports. Same shape as test_motion_primitives.py, extended
# with ``get_world_pose`` (move_to maps the world-frame target through the
# articulation's base pose).
# ---------------------------------------------------------------------------


class _FakeArticulationAction:
    """Stand-in for ``isaacsim.core.utils.types.ArticulationAction``."""

    def __init__(self, joint_positions=None, joint_indices=None):
        self.joint_positions = joint_positions
        self.joint_indices = joint_indices


class _FakeArticulation:
    """Articulation with a per-step PD-servo model and a world base pose.

    ``apply_action`` records the commanded targets; each ``advance()`` (wired
    to the fake world's ``step``) moves every targeted DOF toward its target
    by ``servo_rate`` of the remaining distance. ``servo_rate=0.0`` models an
    arm that never converges (the timeout path).
    """

    def __init__(
        self,
        joint_names: list[str],
        positions: list[float] | None = None,
        servo_rate: float = 0.5,
        base_pos: list[float] | None = None,
        base_quat: list[float] | None = None,
    ):
        n = len(joint_names)
        self.positions = np.array(positions if positions is not None else [0.0] * n, dtype=np.float64)
        self.servo_rate = servo_rate
        self.applied: list[Any] = []
        self._targets: dict[int, float] = {}
        self._base_pos = np.array(base_pos if base_pos is not None else [0.0, 0.0, 0.0], dtype=np.float64)
        self._base_quat = np.array(base_quat if base_quat is not None else [1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def get_joint_positions(self):
        return self.positions.copy()

    def get_world_pose(self):
        return self._base_pos.copy(), self._base_quat.copy()

    def apply_action(self, action) -> None:
        self.applied.append(action)
        for idx, value in zip(np.asarray(action.joint_indices), np.asarray(action.joint_positions)):
            self._targets[int(idx)] = float(value)

    def advance(self) -> None:
        for idx, target in self._targets.items():
            self.positions[idx] += self.servo_rate * (target - self.positions[idx])


class _FakeWorld:
    """World whose ``step`` drives the articulation servo and an optional hook."""

    def __init__(self, articulation: _FakeArticulation, on_step=None):
        self.articulation = articulation
        self.on_step = on_step
        self.steps = 0

    def step(self, render: bool = False) -> None:  # noqa: ARG002 - signature parity
        self.steps += 1
        if self.on_step is not None:
            self.on_step()
        self.articulation.advance()


@pytest.fixture(autouse=True)
def fake_articulation_action(monkeypatch):
    """Provide the ``isaacsim.core.utils.types`` module the adapter imports."""
    names = ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.types")
    mods = {}
    for name in names:
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        mods[name] = mod
    mods["isaacsim.core.utils.types"].ArticulationAction = _FakeArticulationAction
    mods["isaacsim"].core = mods["isaacsim.core"]
    mods["isaacsim.core"].utils = mods["isaacsim.core.utils"]
    mods["isaacsim.core.utils"].types = mods["isaacsim.core.utils.types"]
    return mods


# The same stable 4-DOF positioner + jaw the MuJoCo move_to tests solve on
# (tests/simulation/mujoco/test_motion_primitives.py): an ``ee_site`` TCP
# site (EE-frame discovery), conventional joint names, and a ``jaw`` gripper.
# Here it is the IK MODEL only - the "articulation" the targets land on is
# the fake above.
ARM_XML = """
<mujoco model="prim_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.04 0.02"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/>
          <body name="link3" pos="0.15 0 0">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
            <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.018"/>
            <body name="link4" pos="0.15 0 0">
              <joint name="wrist_roll" type="hinge" axis="1 0 0" range="-3.0 3.0"/>
              <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.015"/>
              <site name="ee_site" pos="0.05 0 0"/>
              <body name="jaw_body" pos="0.05 0 0">
                <joint name="jaw" type="hinge" axis="0 0 1" range="-0.2 1.5"/>
                <geom type="box" size="0.01 0.01 0.02" contype="0" conaffinity="0"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

MJCF_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_roll", "jaw"]

# The GH #1658 regression arm (MuJoCo-parity, #2156): same kinematics, but the
# base pan joint is named ``finger_camera_pan`` (contains the "finger" hint -
# the name heuristic misclassifies it as a gripper DOF) and the real gripper
# matches so100's registry metadata vocabulary (``Jaw``). REACHABLE_LOCAL has
# a nonzero y component, so the EE target is only reachable when the pan DOF
# is available to the IK solve: with the heuristic in charge, move_to holds
# the pan at 0 and the target is unreachable.
HINT_COLLIDER_XML = ARM_XML.replace('"prim_arm"', '"hint_collider_arm"').replace("shoulder_pan", "finger_camera_pan")
assert HINT_COLLIDER_XML.count("finger_camera_pan") == 1  # the joint def
HINT_COLLIDER_XML = HINT_COLLIDER_XML.replace('"jaw"', '"Jaw"')
HINT_COLLIDER_JOINTS = ["finger_camera_pan", "shoulder_lift", "elbow", "wrist_roll", "Jaw"]

# Reachable for the arm above in its OWN (model) frame; same point the MuJoCo
# suite verified against the real solver. UNREACHABLE is inside the sanity
# box but outside the workspace.
REACHABLE_LOCAL = [0.2, 0.1, 0.2]
UNREACHABLE_LOCAL = [1.5, 0.0, 0.2]


@pytest.fixture()
def arm_xml_path(tmp_path, monkeypatch):
    """Write the IK model and point the adapter's ``resolve_model`` at it."""
    path = tmp_path / "prim_arm.xml"
    path.write_text(ARM_XML)
    monkeypatch.setattr(
        "strands_robots.simulation.isaac.motion_primitives.resolve_model",
        lambda name: str(path) if name == "prim_arm" else None,
    )
    return str(path)


@pytest.fixture()
def hint_collider_xml_path(tmp_path, monkeypatch):
    """The #1658 hint-collider IK model, registered for data_config 'so100'."""
    path = tmp_path / "hint_collider_arm.xml"
    path.write_text(HINT_COLLIDER_XML)
    monkeypatch.setattr(
        "strands_robots.simulation.isaac.motion_primitives.resolve_model",
        lambda name: str(path) if name == "so100" else None,
    )
    return str(path)


def _make_sim(
    joint_names: list[str] = MJCF_JOINTS,
    robot_name: str = "arm",
    data_config: str | None = "prim_arm",
    positions: list[float] | None = None,
    servo_rate: float = 0.5,
    base_pos: list[float] | None = None,
    base_quat: list[float] | None = None,
    on_step=None,
) -> tuple[IsaacSimulation, _FakeArticulation]:
    sim = IsaacSimulation()
    art = _FakeArticulation(
        joint_names,
        positions=positions,
        servo_rate=servo_rate,
        base_pos=base_pos,
        base_quat=base_quat,
    )
    sim._world = _FakeWorld(art, on_step=on_step)
    sim._world_created = True
    sim._robots[robot_name] = _RobotState(
        name=robot_name,
        prim_path=f"/World/Robots/{robot_name}",
        joint_names=list(joint_names),
        articulation=art,
        data_config=data_config,
    )
    return sim, art


def _json_block(result: dict) -> dict:
    blocks = [c["json"] for c in result["content"] if "json" in c]
    assert blocks, result
    return blocks[0]


def _text(result: dict) -> str:
    return result["content"][0]["text"]


# ---------------------------------------------------------------------------
# Validation: the rejections are the shared core's, byte-identical to MuJoCo.
# No IK stack required - these refuse before any engine or model is touched.
# ---------------------------------------------------------------------------


class TestValidationReusesTheCore:
    """A rejected parameter answers with the core's own envelope."""

    def test_missing_position_matches_core_wording(self):
        sim, art = _make_sim()
        result = sim.move_to(robot_name="arm")
        _, _, _, _, expected = MotionPrimitivesCore()._validate_move_to_args(None, None, 0.01, 200)
        assert result == expected
        assert "position" in _text(result)
        assert art.applied == []

    @pytest.mark.parametrize("position", [[0.1, 0.2], [0.1, 0.2, 0.3, 0.4], 0.3, True, [0.1, math.nan, 0.2]])
    def test_bad_position_shape_matches_core_wording(self, position):
        sim, art = _make_sim()
        result = sim.move_to(robot_name="arm", position=position)
        _, _, _, _, expected = MotionPrimitivesCore()._validate_move_to_args(position, None, 0.01, 200)
        assert result == expected
        assert result["status"] == "error"
        assert art.applied == []

    def test_zero_norm_orientation_matches_core_wording(self):
        sim, _ = _make_sim()
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, orientation=[0.0, 0.0, 0.0, 0.0])
        _, _, _, _, expected = MotionPrimitivesCore()._validate_move_to_args(
            REACHABLE_LOCAL, [0.0, 0.0, 0.0, 0.0], 0.01, 200
        )
        assert result == expected

    @pytest.mark.parametrize("tol", [0.0, -0.1, math.nan, math.inf, True, "0.05"])
    def test_bad_tol_matches_core_wording(self, tol):
        sim, art = _make_sim()
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, tol=tol)
        _, _, _, _, expected = MotionPrimitivesCore()._validate_move_to_args(REACHABLE_LOCAL, None, tol, 200)
        assert result == expected
        assert art.applied == []

    @pytest.mark.parametrize("max_steps", [0, -1, 10_001, 2.7, True, None])
    def test_bad_max_steps_matches_core_wording(self, max_steps):
        sim, art = _make_sim()
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, max_steps=max_steps)
        _, _, _, _, expected = MotionPrimitivesCore()._validate_move_to_args(REACHABLE_LOCAL, None, 0.01, max_steps)
        assert result == expected
        assert art.applied == []


# ---------------------------------------------------------------------------
# Guards: backend-owned world / robot / policy / kinematic-model resolution.
# ---------------------------------------------------------------------------


class TestGuards:
    def test_no_world_errors(self):
        sim = IsaacSimulation()
        result = sim.move_to(position=REACHABLE_LOCAL)
        assert result["status"] == "error"
        assert "No world created." in _text(result)

    def test_unknown_robot_errors(self):
        sim, _ = _make_sim()
        result = sim.move_to(robot_name="nope", position=REACHABLE_LOCAL)
        assert result["status"] == "error"
        assert "not found" in _text(result)

    def test_running_policy_refuses_up_front(self):
        sim, art = _make_sim()
        sim._robots["arm"].policy_running = True
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL)
        assert result["status"] == "error"
        assert "while its policy is running" in _text(result)
        assert art.applied == []

    def test_workspace_sanity_box_rejects_far_target(self):
        # 10 m from the base: a unit mistake (mm vs m), refused before the IK
        # model is even resolved.
        sim, art = _make_sim(data_config=None)  # no model needed - sanity fires first
        result = sim.move_to(robot_name="arm", position=[10.0, 0.0, 0.2])
        assert result["status"] == "error"
        assert "sanity box" in _text(result)
        assert art.applied == []

    def test_sanity_box_is_relative_to_the_base_pose(self):
        # The same world point is fine for a robot whose base sits next to it.
        sim, _ = _make_sim(data_config=None, base_pos=[10.0, 0.0, 0.0])
        result = sim.move_to(robot_name="arm", position=[10.0, 0.0, 0.2])
        # Passes the sanity box, then fails on the missing data_config - which
        # proves the sanity check measured from the base, not the origin.
        assert result["status"] == "error"
        assert "data_config" in _text(result)

    def test_unreadable_base_pose_is_a_loud_error(self):
        sim, art = _make_sim()
        art.get_world_pose = lambda: None  # type: ignore[method-assign]
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL)
        assert result["status"] == "error"
        assert "base pose" in _text(result)

    def test_no_data_config_is_a_loud_error(self):
        sim, _ = _make_sim(data_config=None)
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL)
        assert result["status"] == "error"
        assert "data_config" in _text(result)
        assert "send_action" in _text(result)

    def test_unresolvable_model_is_a_loud_error(self, monkeypatch):
        monkeypatch.setattr(
            "strands_robots.simulation.isaac.motion_primitives.resolve_model",
            lambda name: None,
        )
        sim, _ = _make_sim(data_config="prim_arm")
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL)
        assert result["status"] == "error"
        assert "no MJCF/URDF model resolves" in _text(result)

    def test_worker_thread_without_pump_is_a_structured_error(self):
        sim, art = _make_sim()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(sim.move_to, robot_name="arm", position=REACHABLE_LOCAL).result()
        assert result["status"] == "error"
        assert "run_pump_forever" in _text(result)
        assert art.applied == []

    def test_missing_mink_degrades_to_structured_error(self, arm_xml_path, monkeypatch):
        """A missing IK stack is a structured error through the tool surface, not a raise."""
        pytest.importorskip("mujoco")
        monkeypatch.setitem(sys.modules, "mink", None)  # forces `import mink` to ImportError
        sim, _ = _make_sim()
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL)
        assert result["status"] == "error"
        assert "IK bridge unavailable" in _text(result)
        assert "mink" in _text(result)


# ---------------------------------------------------------------------------
# Joint-name reconciliation: the MJCF-side solution is written to the
# articulation per NAME, and an unmappable solved joint is a refusal.
# ---------------------------------------------------------------------------


class TestJointNameReconciliation:
    def test_unmappable_arm_joint_is_a_structured_error(self, arm_xml_path):
        # The articulation is missing 'elbow': the solve would command a joint
        # the robot cannot name, so the primitive refuses - a flat-index write
        # is forbidden (AGENTS.md "Per-name state copy, not flat index").
        pytest.importorskip("mujoco")
        sim, art = _make_sim(joint_names=["shoulder_pan", "shoulder_lift", "wrist_roll", "jaw"])
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL)
        assert result["status"] == "error"
        text = _text(result)
        assert "elbow" in text
        assert "no articulation counterpart" in text
        assert "flat index" in text
        assert art.applied == []

    def test_namespaced_articulation_names_map_via_stripping(self, arm_xml_path):
        pytest.importorskip("mink")
        sim, _ = _make_sim(joint_names=[f"arm/{n}" for n in MJCF_JOINTS])
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, tol=0.02)
        assert result["status"] == "success", result

    def test_reordered_articulation_dofs_still_converge(self, arm_xml_path):
        # The articulation reports its DOFs in REVERSE of the MJCF order. A
        # positional write would command the wrong joints and never converge;
        # the name-keyed map must not care about order.
        pytest.importorskip("mink")
        sim, art = _make_sim(joint_names=list(reversed(MJCF_JOINTS)))
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, tol=0.02)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["reached"] is True
        # The pan joint lives at articulation index 4 under the reversed
        # order; the commanded targets must have followed the names there.
        commanded = {int(i) for a in art.applied for i in np.asarray(a.joint_indices)}
        assert commanded == {0, 1, 2, 3, 4}

    def test_unmappable_gripper_joint_does_not_refuse(self, arm_xml_path):
        # The MJCF 'jaw' is a gripper joint; the articulation names its jaw
        # differently ('gripper'). The gripper is never commanded from the
        # solve, so the move itself is still legitimate - the articulation
        # jaw is simply held by the articulation-side classification.
        pytest.importorskip("mink")
        sim, _ = _make_sim(joint_names=["shoulder_pan", "shoulder_lift", "elbow", "wrist_roll", "gripper"])
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, tol=0.02)
        assert result["status"] == "success", result


# ---------------------------------------------------------------------------
# Semantics: convergence, base-frame transform, grasp preservation,
# unreachable / timeout / abort envelopes (real mink solver, fake servo).
# ---------------------------------------------------------------------------


class TestMoveTo:
    def test_reaches_reachable_target(self, arm_xml_path):
        pytest.importorskip("mink")
        sim, _ = _make_sim()
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, tol=0.02)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["reached"] is True
        assert payload["position_error_m"] <= 0.02
        assert payload["frame"] == "ee_site"
        assert payload["frame_type"] == "site"
        assert np.linalg.norm(np.array(payload["ee_position"]) - np.array(REACHABLE_LOCAL)) <= 0.02

    def test_world_frame_target_respects_a_translated_base(self, arm_xml_path):
        # The robot's base sits away from the origin: the world-frame target
        # must be mapped through the base pose, not solved as-is.
        pytest.importorskip("mink")
        base = [1.0, -0.5, 0.0]
        target = list(np.array(base) + np.array(REACHABLE_LOCAL))
        sim, _ = _make_sim(base_pos=base)
        result = sim.move_to(robot_name="arm", position=target, tol=0.02)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["reached"] is True
        assert np.linalg.norm(np.array(payload["ee_position"]) - np.array(target)) <= 0.02

    def test_world_frame_target_respects_a_rotated_base(self, arm_xml_path):
        # Base yawed 90 degrees about z: the local-frame reachable point lands
        # at a rotated world position, which is where the target is given.
        pytest.importorskip("mink")
        half = math.pi / 4.0
        quat = [math.cos(half), 0.0, 0.0, math.sin(half)]
        rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        target = list(rot @ np.array(REACHABLE_LOCAL))
        sim, _ = _make_sim(base_quat=quat)
        result = sim.move_to(robot_name="arm", position=target, tol=0.02)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["reached"] is True
        assert np.linalg.norm(np.array(payload["ee_position"]) - np.array(target)) <= 0.02

    def test_gripper_is_held_not_commanded(self, arm_xml_path):
        # Grasp preservation: the jaw DOF (index 4) is written every tick with
        # its LIVE position, never a solver output.
        pytest.importorskip("mink")
        jaw_before = 0.9
        sim, art = _make_sim(positions=[0.0, 0.0, 0.0, 0.0, jaw_before])
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, tol=0.02)
        assert result["status"] == "success", result
        for action in art.applied:
            held = dict(
                zip(
                    (int(i) for i in np.asarray(action.joint_indices)),
                    (float(v) for v in np.asarray(action.joint_positions)),
                )
            )
            assert held[4] == pytest.approx(jaw_before, abs=1e-6)
        assert art.positions[4] == pytest.approx(jaw_before, abs=1e-6)

    def test_unreachable_target_returns_structured_error_with_residual(self, arm_xml_path):
        pytest.importorskip("mink")
        sim, art = _make_sim()
        result = sim.move_to(robot_name="arm", position=UNREACHABLE_LOCAL, tol=0.01)
        assert result["status"] == "error", result
        assert "unreachable" in _text(result)
        payload = _json_block(result)
        assert payload["reached"] is False
        assert payload["ik_residual_m"] > 0.01
        assert art.applied == []  # refused before a single tick

    def test_servo_timeout_reports_the_residual_and_a_budget_that_fixes_it(self, arm_xml_path):
        # A solvable pose the (deliberately slow) servo cannot reach inside
        # the budget: the envelope separates "needs more steps" (small IK
        # residual, large position error) from "unreachable".
        pytest.importorskip("mink")
        sim, _ = _make_sim(servo_rate=0.05)
        short = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, tol=0.02, max_steps=2)
        assert short["status"] == "error", short
        text = _text(short)
        assert "did not reach" in text
        assert "max_steps=2" in text
        payload = _json_block(short)
        assert payload["reached"] is False
        assert payload["steps"] == 2
        assert payload["position_error_m"] > 0.02
        assert payload["ik_residual_m"] <= 0.02

        again = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, tol=0.02, max_steps=400)
        assert again["status"] == "success", again
        assert _json_block(again)["reached"] is True


class TestGraspPreservation:
    """``move_to`` must never command the gripper (review on #1654, MuJoCo parity).

    The gripper DOF is kinematically irrelevant to the EE task, so the solver
    passes its seed value straight through; the pre-fix MuJoCo restart loop
    randomized the jaw over its full range and the stage -> close -> transport
    sequence silently dropped whatever it held. The direct-path pin lives in
    ``TestMoveTo::test_gripper_is_held_not_commanded``; this class pins the
    RESTART branch, whose seeding is the half that randomizes.
    """

    def test_restart_path_move_to_preserves_closed_gripper(self, arm_xml_path):
        # A target the direct solve misses (restart branch engaged) must not
        # move a closed jaw. [-0.3, 0.0, 0.25] has a direct-solve residual of
        # ~0.65 m on this arm from the zero seed - the exact repro from the
        # MuJoCo review, on the same IK model.
        pytest.importorskip("mink")
        jaw_closed = -0.2
        sim, art = _make_sim(positions=[0.0, 0.0, 0.0, 0.0, jaw_closed])
        result = sim.move_to(robot_name="arm", position=[-0.3, 0.0, 0.25], tol=0.03, max_steps=400)
        assert result["status"] == "success", result
        assert _json_block(result)["reached"] is True
        for action in art.applied:
            held = dict(
                zip(
                    (int(i) for i in np.asarray(action.joint_indices)),
                    (float(v) for v in np.asarray(action.joint_positions)),
                )
            )
            assert held[4] == pytest.approx(jaw_closed, abs=1e-6), (
                "the restart path commanded the jaw away from its held position"
            )
        assert art.positions[4] == pytest.approx(jaw_closed, abs=1e-6)


class TestGripperRegistryMetadata:
    """Registry gripper metadata beats the name heuristic in the IK split (GH #1658).

    The move_to half of the MuJoCo parity class: the MJCF-side arm/gripper
    split (:meth:`_mjcf_articulation_joint_map`) and the articulation-side
    hold set (:meth:`_resolve_gripper_dofs`) share the registry-metadata-first
    classification, so a hint-colliding ARM joint stays in the solve and stale
    metadata refuses move_to exactly as it refuses set_gripper. Uses the REAL
    shipped so100 registry entry (``actuators: ["Jaw"]``) - no metadata
    patching.
    """

    def test_move_to_keeps_hint_colliding_arm_dof_in_ik(self, hint_collider_xml_path):
        # The off-axis target NEEDS the base pan DOF (REACHABLE_LOCAL has a
        # nonzero y). Pre-#1658 the heuristic classified 'finger_camera_pan'
        # as a gripper drive, excluded it from the solve and held it at 0 -
        # unreachable. Metadata keeps the DOF usable while the real gripper
        # stays held (grasp preservation intact).
        pytest.importorskip("mink")
        jaw_before = 0.9
        sim, art = _make_sim(
            joint_names=HINT_COLLIDER_JOINTS,
            data_config="so100",
            positions=[0.0, 0.0, 0.0, 0.0, jaw_before],
        )
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, tol=0.03, max_steps=400)
        assert result["status"] == "success", result
        assert _json_block(result)["reached"] is True
        # The pan joint moved (it was solved, not held) ...
        assert abs(art.positions[0]) > 0.05, "the hint-colliding base pan joint never moved - it was excluded from IK"
        # ... and the jaw did not (held at its live position).
        assert art.positions[4] == pytest.approx(jaw_before, abs=1e-6), "move_to moved the gripper"

    def test_stale_metadata_is_a_loud_error_not_a_heuristic_fallback(self, hint_collider_xml_path, monkeypatch):
        # Metadata naming a joint absent from the articulation refuses with
        # the articulation's actual joint list - silently degrading to the
        # heuristic would reintroduce the misclassification the metadata
        # prevents (same contract as set_gripper's, pinned per action because
        # move_to reaches it through the joint-map split).
        pytest.importorskip("mujoco")
        monkeypatch.setattr(
            "strands_robots.simulation.isaac.motion_primitives.get_robot",
            lambda name: {"gripper": {"actuators": ["no_such_actuator"], "closed": "low", "open": "high"}},
        )
        sim, art = _make_sim(joint_names=HINT_COLLIDER_JOINTS, data_config="so100")
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL)
        assert result["status"] == "error", result
        text = _text(result)
        assert "no_such_actuator" in text
        assert "Jaw" in text
        assert art.applied == []


class TestMidRunAbort:
    """The loop releases the lock per tick; teardown / a policy start aborts loudly."""

    def test_world_destroyed_mid_run_aborts(self, arm_xml_path):
        pytest.importorskip("mink")
        sim_box: dict[str, IsaacSimulation] = {}

        def _destroy_world():
            sim_box["sim"]._world_created = False

        sim, _ = _make_sim(servo_rate=0.0, on_step=_destroy_world)
        sim_box["sim"] = sim
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, max_steps=50)
        assert result["status"] == "error"
        assert "world was destroyed mid-run" in _text(result)

    def test_policy_started_mid_run_aborts(self, arm_xml_path):
        pytest.importorskip("mink")
        sim_box: dict[str, IsaacSimulation] = {}

        def _start_policy():
            sim_box["sim"]._robots["arm"].policy_running = True

        sim, _ = _make_sim(servo_rate=0.0, on_step=_start_policy)
        sim_box["sim"] = sim
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, max_steps=50)
        assert result["status"] == "error"
        assert "a policy started on 'arm' mid-run" in _text(result)


class TestDiscoverySurface:
    """describe() advertises move_to like the MuJoCo backend does."""

    def test_describe_advertises_move_to(self):
        sim = IsaacSimulation()
        methods = sim.describe()["methods"]
        assert "move_to" in methods
        assert "world-frame" in methods["move_to"]
