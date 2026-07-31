"""Object-pose init-state branch of :class:`LiberoAdapter` (#1820, Isaac).

``IsaacSimulation.load_scene`` realizes LIBERO scene objects at their MJCF
*placeholder* body poses - LIBERO encodes the real per-episode poses in BDDL
init states (flat ``[time, qpos, qvel]`` vectors), which the MuJoCo backend
applies by writing ``data.qpos`` (``_apply_init_state_branch``). Isaac has
no engine ``qpos``, so before #1820 the init state was silently skipped and
live physics started from coincident, robot-interpenetrating placeholder
poses - PhysX "Illegal BroadPhaseUpdateData - non-finite bounds" storms the
moment the timeline integrates.

These tests pin ``_apply_object_pose_state`` (reached through the public
``_apply_canonical_state`` entry when the sim exposes no compiled MuJoCo
model): the init state is decoded through a LOCAL CPU MuJoCo compile of the
scene MJCF and applied per named object via ``sim.move_object``:

* dynamic (free-jointed) objects are teleported to ``xpos + R(xquat) @
  offset`` (the realized prim sits at the body's collision-AABB centre,
  not the body origin);
* static fixtures are left alone (no free joint - qpos cannot move them on
  MuJoCo either);
* the robot base is aligned with the scene's ``robot0_base`` body via
  ``sim.set_robot_pose`` (Isaac spawns the USD robot at the origin, inside
  the footprint of the scene's origin-anchored static fixtures);
* row selection matches ``_apply_init_state_branch`` (episode 0 pinned to
  row 0, episodes 1+ RNG-sampled; ``_episode_count`` increments);
* width mismatches and failed teleports raise (an object left at its
  placeholder pose WILL explode live physics - warn-and-continue is the
  silent failure mode this branch removes);
* missing preconditions (no init states / no scene file / no
  ``move_object`` / no ``mujoco``) degrade to a debug-log skip, preserving
  the graceful-degradation contract for arbitrary model-less sims.

CPU-only: the decode is plain ``mujoco`` (no Isaac Sim), the sim is a stub.
"""

from __future__ import annotations

import random
import sys
from typing import Any, cast

import numpy as np
import pytest

from strands_robots.benchmarks.libero import LiberoAdapter
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

# One static fixture + one free-jointed cube whose collision geom is OFFSET
# from the body origin (pos="0 0 0.01"), so the prim-pose composition
# ``xpos + R(xquat) @ offset`` is exercised, not just the translation.
# The robot0_base body carries the scene's robot base pose (skipped by the
# scene parser, but read for ``sim.set_robot_pose`` alignment).
# nq = 7 (one free joint), nv = 6 -> init-state width 1 + 7 + 6 = 14.
_SCENE_MJCF = """
<mujoco model="pose_probe">
  <worldbody>
    <body name="robot0_base" pos="-0.66 0.0 0.912">
      <geom type="box" size="0.05 0.05 0.05"/>
    </body>
    <body name="fixture_table" pos="0.0 0.0 0.4">
      <geom type="box" size="0.5 0.5 0.4"/>
    </body>
    <body name="cube_1_main" pos="0.0 -0.1 0.02">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" pos="0 0 0.01"/>
    </body>
  </worldbody>
</mujoco>
"""

_ROBOT_BASE_POS = (-0.66, 0.0, 0.912)

# Target pose for the cube's free joint: translated to (0.1, 0.2, 0.9) and
# rotated +90 deg about x. Free-joint qpos layout: [x y z qw qx qy qz].
_SQRT_HALF = float(np.sqrt(0.5))
_TARGET_POS = (0.1, 0.2, 0.9)
_TARGET_QUAT = (_SQRT_HALF, _SQRT_HALF, 0.0, 0.0)  # wxyz, +90deg about x
# R(+90deg about x) @ (0, 0, 0.01) = (0, -0.01, 0).
_EXPECTED_PRIM_POS = (0.1, 0.19, 0.9)


def _init_state_row(pos=_TARGET_POS, quat=_TARGET_QUAT) -> list[float]:
    return [0.0, *pos, *quat, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class _PoseRecordingSim:
    """Model-less sim stub recording ``move_object`` / ``set_robot_pose`` /
    ``step`` calls."""

    def __init__(self, move_status: str = "success", robot_pose_status: str = "success") -> None:
        self._move_status = move_status
        self._robot_pose_status = robot_pose_status
        self.moves: list[tuple[str, list[float], list[float]]] = []
        self.robot_poses: list[tuple[list[float], list[float]]] = []
        self.step_calls: list[int] = []

    def move_object(self, *, name: str, position: list[float], orientation: list[float]) -> dict[str, Any]:
        self.moves.append((name, list(position), list(orientation)))
        if self._move_status == "error":
            return {"status": "error", "content": [{"text": "Object not found."}]}
        return {"status": "success", "content": [{"text": f"'{name}' moved."}]}

    def set_robot_pose(self, *, position: list[float], orientation: list[float]) -> dict[str, Any]:
        self.robot_poses.append((list(position), list(orientation)))
        if self._robot_pose_status == "error":
            return {"status": "error", "content": [{"text": "Robot not initialized."}]}
        return {"status": "success", "content": [{"text": "Robot base moved."}]}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        self.step_calls.append(n_steps)
        return {"status": "success", "content": [{"text": "stepped"}]}


@pytest.fixture
def scene_file(tmp_path):
    p = tmp_path / "scene.xml"
    p.write_text(_SCENE_MJCF)
    return str(p)


def _adapter(scene_file: str, init_states: np.ndarray | None) -> LiberoAdapter:
    return LiberoAdapter.from_text(
        PICK_CUBE_BDDL,
        scene_path=scene_file,
        auto_generate_scene=False,
        init_states=init_states,
    )


class TestObjectPoseApply:
    def test_dynamic_object_teleported_to_decoded_pose(self, scene_file) -> None:
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))
        sim = _PoseRecordingSim()

        adapter._apply_canonical_state(cast(SimEngine, sim), random.Random(0))

        # Only the free-jointed cube moves; the static fixture keeps its
        # MJCF pose (qpos cannot move it on MuJoCo either).
        assert [m[0] for m in sim.moves] == ["cube_1_main"]
        _, position, orientation = sim.moves[0]
        np.testing.assert_allclose(position, _EXPECTED_PRIM_POS, atol=1e-9)
        np.testing.assert_allclose(orientation, _TARGET_QUAT, atol=1e-9)
        # The robot base is aligned with the scene's robot0_base body:
        # Isaac spawns the USD robot at the origin, inside the footprint
        # of the scene's origin-anchored static fixtures.
        assert len(sim.robot_poses) == 1
        np.testing.assert_allclose(sim.robot_poses[0][0], _ROBOT_BASE_POS, atol=1e-9)
        np.testing.assert_allclose(sim.robot_poses[0][1], (1.0, 0.0, 0.0, 0.0), atol=1e-9)
        # Settle runs AFTER the teleports (settling at placeholder poses is
        # the part-2 explosion) and the episode counter advances so the
        # next episode gets RNG-sampled selection.
        assert sim.step_calls == [5]
        assert adapter._episode_count == 1

    def test_episode_zero_pins_row_zero_then_rng_samples(self, scene_file) -> None:
        rows = np.array(
            [_init_state_row(), _init_state_row(pos=(0.3, -0.2, 0.7), quat=(1.0, 0.0, 0.0, 0.0))],
            dtype=np.float64,
        )
        adapter = _adapter(scene_file, rows)
        sim = _PoseRecordingSim()

        # Episode 0: deterministic row 0 regardless of rng.
        adapter._apply_canonical_state(cast(SimEngine, sim), random.Random(7))
        np.testing.assert_allclose(sim.moves[0][1], _EXPECTED_PRIM_POS, atol=1e-9)

        # Episode 1: RNG-sampled with the same semantics as the MuJoCo
        # init-state branch (rng.randint(0, n_states - 1)).
        rng = random.Random(7)
        expected_idx = random.Random(7).randint(0, 1)
        adapter._apply_canonical_state(cast(SimEngine, sim), rng)
        expected_pos = _EXPECTED_PRIM_POS if expected_idx == 0 else (0.3, -0.2, 0.7 + 0.01)
        np.testing.assert_allclose(sim.moves[1][1], expected_pos, atol=1e-9)
        assert adapter._episode_count == 2

    def test_width_mismatch_raises(self, scene_file) -> None:
        adapter = _adapter(scene_file, np.zeros((1, 5), dtype=np.float64))
        with pytest.raises(RuntimeError, match="width"):
            adapter._apply_canonical_state(cast(SimEngine, _PoseRecordingSim()))

    def test_failed_teleport_raises(self, scene_file) -> None:
        # An object left at its placeholder pose interpenetrates the robot
        # base and explodes live physics - never warn-and-continue.
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))
        with pytest.raises(RuntimeError, match="cube_1_main"):
            adapter._apply_canonical_state(cast(SimEngine, _PoseRecordingSim(move_status="error")))

    def test_failed_robot_base_alignment_raises(self, scene_file) -> None:
        # A robot left inside the scene's origin-anchored fixtures explodes
        # live physics just like a misplaced object - never warn-and-continue.
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))
        with pytest.raises(RuntimeError, match="robot0_base"):
            adapter._apply_canonical_state(cast(SimEngine, _PoseRecordingSim(robot_pose_status="error")))

    def test_robotless_scene_skips_base_alignment(self, tmp_path) -> None:
        # A scene without a robot0_base body applies object poses without
        # attempting (or requiring) base alignment.
        scene = _SCENE_MJCF.replace("robot0_base", "not_a_robot_base")
        p = tmp_path / "scene.xml"
        p.write_text(scene)
        adapter = _adapter(str(p), np.array([_init_state_row()], dtype=np.float64))
        sim = _PoseRecordingSim()
        adapter._apply_canonical_state(cast(SimEngine, sim), random.Random(0))
        assert sim.robot_poses == []
        assert [m[0] for m in sim.moves] == ["cube_1_main"]


class TestObjectPoseGracefulDegradation:
    """Missing preconditions skip (never raise) so arbitrary model-less
    sims keep hosting the adapter, matching the pre-#1820 contract."""

    def test_skips_without_init_states(self, scene_file) -> None:
        adapter = _adapter(scene_file, None)
        sim = _PoseRecordingSim()
        adapter._apply_canonical_state(cast(SimEngine, sim))
        assert sim.moves == []
        assert sim.step_calls == []

    def test_skips_without_scene_file(self, tmp_path) -> None:
        adapter = _adapter(str(tmp_path / "missing.xml"), np.array([_init_state_row()], dtype=np.float64))
        sim = _PoseRecordingSim()
        adapter._apply_canonical_state(cast(SimEngine, sim))
        assert sim.moves == []

    def test_skips_when_sim_lacks_move_object(self, scene_file) -> None:
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))

        class _NoMove:
            pass

        adapter._apply_canonical_state(cast(SimEngine, _NoMove()))

    def test_skips_when_mujoco_unimportable(self, scene_file, monkeypatch) -> None:
        adapter = _adapter(scene_file, np.array([_init_state_row()], dtype=np.float64))
        sim = _PoseRecordingSim()
        monkeypatch.setitem(sys.modules, "mujoco", None)
        adapter._apply_canonical_state(cast(SimEngine, sim))
        assert sim.moves == []
