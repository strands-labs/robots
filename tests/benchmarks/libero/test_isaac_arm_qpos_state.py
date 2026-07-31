"""Arm-qpos half of the Isaac init-state apply (#1828).

#1827 (fixing #1820) applies LIBERO init states on model-less backends
(Isaac) as per-object poses plus robot *base* alignment, decoded through a
CPU MuJoCo compile of the scene MJCF. One slice stayed unapplied: the robot
**arm qpos** (LIBERO's Panda ready pose ``[0, -0.161, 0, -2.444, 0, 2.227,
pi/4]`` + gripper). On MuJoCo the full qpos vector lands in one write; on
Isaac the USD Franka articulation started every episode at its USD default
(all-zero, upright), so the policy's first observation was OOD relative to
the LIBERO training distribution.

These tests pin ``LiberoAdapter._apply_scene_arm_qpos`` (reached through
the public ``_apply_canonical_state`` entry when the sim exposes no
compiled MuJoCo model) and the ``_map_scene_joints_to_articulation``
name-mapping helper:

* the decode model names joints with robosuite prefixes
  (``robot0_joint1..7`` / ``gripper0_finger_joint1..2``) while the Isaac
  USD articulation names them ``panda_joint1..7`` /
  ``panda_finger_joint1..2``; the prefix-stripped names map onto
  articulation DOFs by LONGEST whole-token suffix (plain suffix matching
  is ambiguous: ``joint1`` is a suffix of both ``panda_joint1`` and
  ``panda_finger_joint1``);
* ambiguous / unmappable scene joints raise BEFORE anything is written --
  no silent partial writes;
* a failed ``set_joint_positions`` write raises;
* missing engine seams (no ``set_joint_positions`` / ``robot_joint_names``
  / ``list_robots``) and robot-less scenes degrade to a debug-log skip,
  preserving the graceful-degradation contract of the enclosing branch.

CPU-only: the decode is plain ``mujoco`` (no Isaac Sim), the sim is a
recording stub (same pattern as ``test_isaac_object_pose_state.py``).
"""

from __future__ import annotations

import math
import random
from typing import Any, cast

import numpy as np
import pytest

from strands_robots.benchmarks.libero import LiberoAdapter
from strands_robots.benchmarks.libero.adapter import _map_scene_joints_to_articulation
from strands_robots.simulation.base import SimEngine

pytest.importorskip("mujoco")

PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the red cube and place it on the plate")
  (:objects cube_1 plate_1 table_1 - object)
  (:init (on cube_1 table_1))
  (:goal (on cube_1 plate_1)))
"""

# LIBERO's Panda "ready" pose + PandaGripper init qpos -- the init-state
# arm slice a policy's training data assumes at episode start.
ARM_READY = (0.0, -0.161, 0.0, -2.444, 0.0, 2.227, math.pi / 4)
GRIPPER_READY = (0.020833, -0.020833)

# Real USD Franka articulation DOF names -- deliberately including the
# suffix trap: bare ``joint1`` is a suffix of BOTH ``panda_joint1`` and
# ``panda_finger_joint1``.
FRANKA_DOFS = [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"]

_TARGET_POS = (0.1, 0.2, 0.9)
_TARGET_QUAT = (1.0, 0.0, 0.0, 0.0)


def _scene_mjcf(robot_prefix: str = "robot0_", gripper_prefix: str = "gripper0_") -> str:
    """A robosuite-shaped probe scene: a 7-hinge arm chain + 2 slide
    fingers under ``robot0_base``, plus one free-jointed cube.

    Joint declaration order (= qpos order): arm 1..7, fingers 1..2, cube
    free joint. nq = 7 + 2 + 7 = 16, nv = 7 + 2 + 6 = 15 -> init-state
    width 1 + 16 + 15 = 32.
    """
    arm = ""
    for i in range(1, 8):
        axis = "0 0 1" if i % 2 else "0 1 0"
        arm += (
            f'<body name="{robot_prefix}link{i}" pos="0 0 0.1">'
            f'<joint name="{robot_prefix}joint{i}" type="hinge" axis="{axis}"/>'
            f'<geom type="sphere" size="0.02"/>'
        )
    fingers = ""
    for i, x in ((1, 0.03), (2, -0.03)):
        fingers += (
            f'<body name="{gripper_prefix}finger{i}" pos="{x} 0 0.05">'
            f'<joint name="{gripper_prefix}finger_joint{i}" type="slide" axis="1 0 0"/>'
            f'<geom type="box" size="0.005 0.005 0.02"/>'
            f"</body>"
        )
    arm_close = "</body>" * 7
    return f"""
<mujoco model="arm_qpos_probe">
  <worldbody>
    <body name="{robot_prefix}base" pos="-0.66 0.0 0.912">
      <geom type="box" size="0.05 0.05 0.05"/>
      {arm}{fingers}{arm_close}
    </body>
    <body name="cube_1_main" pos="0.0 -0.1 0.02">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02"/>
    </body>
  </worldbody>
</mujoco>
"""


def _init_state_row(arm=ARM_READY, gripper=GRIPPER_READY) -> list[float]:
    return [0.0, *arm, *gripper, *_TARGET_POS, *_TARGET_QUAT, *([0.0] * 15)]


class _ArmRecordingSim:
    """Model-less sim stub recording ``set_joint_positions`` alongside the
    ``move_object`` / ``set_robot_pose`` / ``step`` calls, exposing the
    Isaac joint-write seam (``list_robots`` / ``robot_joint_names`` /
    ``set_joint_positions``) with real Franka DOF names."""

    def __init__(
        self,
        dof_names: list[str] | None = None,
        robots: list[str] | None = None,
        joint_write_status: str = "success",
    ) -> None:
        self._dof_names = FRANKA_DOFS if dof_names is None else dof_names
        self._robots = ["robot"] if robots is None else robots
        self._joint_write_status = joint_write_status
        self.joint_writes: list[tuple[str | None, dict[str, float]]] = []
        self.moves: list[tuple[str, list[float], list[float]]] = []
        self.robot_poses: list[tuple[list[float], list[float]]] = []
        self.step_calls: list[int] = []
        self.events: list[str] = []

    def list_robots(self) -> list[str]:
        return list(self._robots)

    def robot_joint_names(self, robot_name: str) -> list[str]:
        assert robot_name in self._robots
        return list(self._dof_names)

    def set_joint_positions(self, positions: Any = None, robot_name: str | None = None) -> dict[str, Any]:
        self.joint_writes.append((robot_name, dict(positions)))
        self.events.append("set_joint_positions")
        if self._joint_write_status == "error":
            return {"status": "error", "content": [{"text": "Robot 'robot' not initialized."}]}
        return {"status": "success", "content": [{"text": "Set joint positions (main)."}]}

    def move_object(self, *, name: str, position: list[float], orientation: list[float]) -> dict[str, Any]:
        self.moves.append((name, list(position), list(orientation)))
        self.events.append("move_object")
        return {"status": "success", "content": [{"text": f"'{name}' moved."}]}

    def set_robot_pose(self, *, position: list[float], orientation: list[float]) -> dict[str, Any]:
        self.robot_poses.append((list(position), list(orientation)))
        self.events.append("set_robot_pose")
        return {"status": "success", "content": [{"text": "Robot base moved."}]}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        self.step_calls.append(n_steps)
        self.events.append("step")
        return {"status": "success", "content": [{"text": "stepped"}]}


@pytest.fixture
def scene_file(tmp_path):
    p = tmp_path / "scene.xml"
    p.write_text(_scene_mjcf())
    return str(p)


def _adapter(scene_file: str, init_states: np.ndarray | None) -> LiberoAdapter:
    return LiberoAdapter.from_text(
        PICK_CUBE_BDDL,
        scene_path=scene_file,
        auto_generate_scene=False,
        init_states=init_states,
    )


class TestJointNameMapping:
    """The pure name-mapping helper: longest-suffix wins, loud failures."""

    def test_franka_mapping_resolves_the_joint1_suffix_trap(self) -> None:
        scene = {f"joint{i}": float(i) for i in range(1, 8)}
        scene.update({"finger_joint1": 8.0, "finger_joint2": 9.0})
        mapped = _map_scene_joints_to_articulation(scene, FRANKA_DOFS)
        # panda_finger_joint1 claims finger_joint1 (longest suffix), NOT
        # joint1 -- so joint1 lands uniquely on panda_joint1.
        assert mapped == {
            **{f"panda_joint{i}": float(i) for i in range(1, 8)},
            "panda_finger_joint1": 8.0,
            "panda_finger_joint2": 9.0,
        }

    def test_exact_names_map_identically(self) -> None:
        mapped = _map_scene_joints_to_articulation({"joint1": 1.5}, ["joint1"])
        assert mapped == {"joint1": 1.5}

    def test_alphanumeric_boundary_is_not_a_suffix_match(self) -> None:
        # ``arm_pjoint1`` merely shares a tail with ``joint1``; matching it
        # would write the value into a different joint.
        with pytest.raises(ValueError, match="unmappable"):
            _map_scene_joints_to_articulation({"joint1": 0.5}, ["arm_pjoint1"])

    def test_dual_arm_ambiguity_raises(self) -> None:
        with pytest.raises(ValueError, match="ambiguous"):
            _map_scene_joints_to_articulation({"joint1": 0.5}, ["left_joint1", "right_joint1"])

    def test_unmappable_scene_joint_raises(self) -> None:
        with pytest.raises(ValueError, match="joint7"):
            _map_scene_joints_to_articulation(
                {"joint1": 0.1, "joint7": 0.7},
                ["panda_joint1"],
            )


class TestArmQposApply:
    def test_arm_qpos_written_to_mapped_articulation_dofs(self, scene_file) -> None:
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))
        sim = _ArmRecordingSim()

        adapter._apply_canonical_state(cast(SimEngine, sim), random.Random(0))

        # One joint write, addressed to the resolved robot, carrying the
        # decoded arm + gripper qpos on the USD articulation's names.
        assert len(sim.joint_writes) == 1
        robot_name, mapped = sim.joint_writes[0]
        assert robot_name == "robot"
        expected = {
            **{f"panda_joint{i}": ARM_READY[i - 1] for i in range(1, 8)},
            "panda_finger_joint1": GRIPPER_READY[0],
            "panda_finger_joint2": GRIPPER_READY[1],
        }
        assert sorted(mapped) == sorted(expected)
        for dof, value in expected.items():
            assert mapped[dof] == pytest.approx(value, abs=1e-9), dof
        # The write happens BEFORE the settle steps (a pose written after
        # settling would never be integrated against the scene) and the
        # object-pose half still runs.
        assert sim.events.index("set_joint_positions") < sim.events.index("step")
        assert [m[0] for m in sim.moves] == ["cube_1_main"]
        assert len(sim.robot_poses) == 1
        assert sim.step_calls == [5]
        assert adapter._episode_count == 1

    def test_unmappable_dofs_raise_with_no_partial_write(self, scene_file) -> None:
        # An articulation missing panda_joint7 leaves bare joint7 unclaimed:
        # fail loud BEFORE any write.
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))
        sim = _ArmRecordingSim(dof_names=FRANKA_DOFS[:6] + ["panda_finger_joint1", "panda_finger_joint2"])
        with pytest.raises(RuntimeError, match="joint7"):
            adapter._apply_canonical_state(cast(SimEngine, sim), random.Random(0))
        assert sim.joint_writes == []

    def test_ambiguous_dofs_raise_with_no_partial_write(self, scene_file) -> None:
        # A dual-arm articulation: every bare name is claimed twice.
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))
        dual = [f"left_{d}" for d in FRANKA_DOFS] + [f"right_{d}" for d in FRANKA_DOFS]
        sim = _ArmRecordingSim(dof_names=dual)
        with pytest.raises(RuntimeError, match="ambiguous"):
            adapter._apply_canonical_state(cast(SimEngine, sim), random.Random(0))
        assert sim.joint_writes == []

    def test_failed_joint_write_raises(self, scene_file) -> None:
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))
        sim = _ArmRecordingSim(joint_write_status="error")
        with pytest.raises(RuntimeError, match="arm qpos"):
            adapter._apply_canonical_state(cast(SimEngine, sim), random.Random(0))

    def test_multiple_robots_raise(self, scene_file) -> None:
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))
        sim = _ArmRecordingSim(robots=["robot_a", "robot_b"])
        with pytest.raises(RuntimeError, match="ambiguous"):
            adapter._apply_canonical_state(cast(SimEngine, sim), random.Random(0))
        assert sim.joint_writes == []

    def test_empty_dof_list_raises(self, scene_file) -> None:
        # A robot that reports no DOF names cannot absorb the arm qpos --
        # skipping silently would leave the arm at the USD default pose.
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))
        sim = _ArmRecordingSim(dof_names=[])
        with pytest.raises(RuntimeError, match="no articulation DOF names"):
            adapter._apply_canonical_state(cast(SimEngine, sim), random.Random(0))
        assert sim.joint_writes == []


class TestArmQposGracefulDegradation:
    """Missing engine seams / robot-less scenes skip (never raise) so
    arbitrary model-less sims keep hosting the adapter, matching the
    contract of the enclosing object-pose branch (#1820)."""

    def test_sim_without_joint_write_seam_skips_arm_but_applies_objects(self, scene_file) -> None:
        # The #1820-era stub surface (move_object / set_robot_pose / step
        # only): the object-pose half still runs, the arm half skips.
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))

        class _NoJointSeam:
            def __init__(self) -> None:
                self.moves: list[str] = []
                self.step_calls: list[int] = []

            def move_object(self, *, name: str, position: list[float], orientation: list[float]) -> dict[str, Any]:
                self.moves.append(name)
                return {"status": "success", "content": [{"text": "moved"}]}

            def set_robot_pose(self, *, position: list[float], orientation: list[float]) -> dict[str, Any]:
                return {"status": "success", "content": [{"text": "moved"}]}

            def step(self, n_steps: int = 1) -> dict[str, Any]:
                self.step_calls.append(n_steps)
                return {"status": "success", "content": [{"text": "stepped"}]}

        sim = _NoJointSeam()
        adapter._apply_canonical_state(cast(SimEngine, sim), random.Random(0))
        assert sim.moves == ["cube_1_main"]
        assert sim.step_calls == [5]

    def test_no_registered_robot_skips(self, scene_file) -> None:
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))
        sim = _ArmRecordingSim(robots=[])
        adapter._apply_canonical_state(cast(SimEngine, sim), random.Random(0))
        assert sim.joint_writes == []
        assert [m[0] for m in sim.moves] == ["cube_1_main"]

    def test_robotless_scene_skips_joint_write(self, tmp_path) -> None:
        # A scene without prefixed robot joints (the #1820 probe scenes)
        # never reaches the engine seam.
        scene = tmp_path / "scene.xml"
        scene.write_text(_scene_mjcf(robot_prefix="fixture_", gripper_prefix="fixturegrip_"))
        adapter = _adapter(str(scene), np.array([_init_state_row()], dtype=np.float64))
        sim = _ArmRecordingSim()
        adapter._apply_canonical_state(cast(SimEngine, sim), random.Random(0))
        assert sim.joint_writes == []
        assert [m[0] for m in sim.moves] == ["cube_1_main"]
