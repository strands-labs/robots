"""GPU-gated integration: the init-state arm qpos lands on Isaac (#1828).

#1827 (fixing #1820) applies LIBERO init states on Isaac as per-object
poses plus robot *base* alignment; the arm qpos slice (LIBERO's Panda
ready pose) stayed unapplied, so the USD Franka articulation started every
episode at its USD default (all-zero, upright) and the policy's first
observation was OOD relative to the LIBERO training distribution.

This test drives the full adapter half against a real Isaac articulation:
``LiberoAdapter._apply_canonical_state`` on a model-less sim decodes the
init state through a CPU MuJoCo compile of the scene MJCF, maps the
robosuite-prefixed joint names (``robot0_joint1..7`` /
``gripper0_finger_joint1..2``) onto the REAL USD Franka DOF names
(``panda_joint1..7`` / ``panda_finger_joint1..2``) by longest-suffix
match, and writes them via ``IsaacSimulation.set_joint_positions``. The
assertion is the one a stub cannot fake: after the apply (settle steps
included -- ``set_joint_positions`` writes both state and PD targets, so
the pose must HOLD through the settle), the articulation's observed joint
positions match the decoded init-state arm qpos within tolerance.

Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ \\
        tests_integ/simulation/test_isaac_init_arm_qpos_gpu.py -m gpu -v
"""

from __future__ import annotations

import math
import os
import random

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")
pytest.importorskip("mujoco")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason="Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable.",
    ),
]

PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the red cube and place it on the plate")
  (:objects cube_1 plate_1 table_1 - object)
  (:init (on cube_1 table_1))
  (:goal (on cube_1 plate_1)))
"""

# LIBERO's Panda ready pose + PandaGripper init qpos.
ARM_READY = (0.0, -0.161, 0.0, -2.444, 0.0, 2.227, math.pi / 4)
GRIPPER_READY = (0.020833, -0.020833)

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
GRIPPER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]

# PD tracking during the settle steps and finger drive coupling eat a few
# hundredths of a radian/metre; the USD default pose differs from the
# ready pose by >1 rad on several joints, so 0.05 cleanly separates
# "applied and held" from "never applied".
_TOL = 0.05


def _scene_mjcf() -> str:
    """A robosuite-shaped probe scene: 7-hinge arm + 2 slide fingers under
    ``robot0_base`` (decode-only -- ``load_scene`` skips ``robot0``/
    ``gripper0`` bodies) plus one free cube realized as a prim."""
    arm = ""
    for i in range(1, 8):
        axis = "0 0 1" if i % 2 else "0 1 0"
        arm += (
            f'<body name="robot0_link{i}" pos="0 0 0.1">'
            f'<joint name="robot0_joint{i}" type="hinge" axis="{axis}"/>'
            f'<geom type="sphere" size="0.02"/>'
        )
    fingers = "".join(
        f'<body name="gripper0_finger{i}" pos="{x} 0 0.05">'
        f'<joint name="gripper0_finger_joint{i}" type="slide" axis="1 0 0"/>'
        f'<geom type="box" size="0.005 0.005 0.02"/>'
        f"</body>"
        for i, x in ((1, 0.03), (2, -0.03))
    )
    return f"""
<mujoco model="arm_qpos_probe">
  <worldbody>
    <body name="robot0_base" pos="-0.66 0.0 0.912">
      <geom type="box" size="0.05 0.05 0.05"/>
      {arm}{fingers}{"</body>" * 7}
    </body>
    <body name="fixture_table" pos="0.6 0.0 0.4">
      <geom type="box" size="0.3 0.3 0.4" group="0"/>
    </body>
    <body name="cube_1_main" pos="0.0 -0.1 0.02">
      <freejoint/>
      <geom type="box" size="0.02 0.02 0.02" group="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _init_state_row() -> list[float]:
    # qpos order: arm(7), fingers(2), cube free(7); qvel = 7 + 2 + 6.
    cube_pose = (0.55, 0.0, 0.85, 1.0, 0.0, 0.0, 0.0)
    return [0.0, *ARM_READY, *GRIPPER_READY, *cube_pose, *([0.0] * 15)]


def _skip_if_isaac_unavailable() -> None:
    from strands_robots.simulation.isaac import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")


def _add_franka(sim) -> None:
    try:
        from isaacsim.storage.native import get_assets_root_path  # type: ignore[import-not-found]
    except ImportError:
        from omni.isaac.nucleus import get_assets_root_path  # type: ignore[import-not-found]

    assets_root = get_assets_root_path()
    if not assets_root:
        pytest.skip("No Isaac assets root (Nucleus/CDN) reachable for the Franka USD")
    r = sim.add_robot("robot", usd_path=f"{assets_root}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd")
    if r["status"] != "success":
        r = sim.add_robot("robot", usd_path=f"{assets_root}/Isaac/Robots/Franka/franka.usd")
    assert r["status"] == "success", f"add_robot: {r}"


class TestIsaacInitArmQposGPU:
    def test_apply_canonical_state_lands_the_ready_pose(self, tmp_path):
        from strands_robots.benchmarks.libero import LiberoAdapter
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()
        scene_file = tmp_path / "scene.xml"
        scene_file.write_text(_scene_mjcf())

        adapter = LiberoAdapter.from_text(
            PICK_CUBE_BDDL,
            scene_path=str(scene_file),
            auto_generate_scene=False,
            init_states=np.array([_init_state_row()], dtype=np.float64),
        )

        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            r = sim.create_world()
            assert r["status"] == "success", f"create_world: {r}"
            _add_franka(sim)
            r = sim.load_scene(str(scene_file))
            assert r["status"] == "success", f"load_scene: {r}"
            sim.step(2)

            # Sanity: the USD default is NOT the ready pose, otherwise the
            # assertion below would pass without the apply.
            before = sim.get_observation("robot", skip_images=True)
            drift = sum(abs(float(before[j]) - v) for j, v in zip(ARM_JOINTS, ARM_READY))
            assert drift > 1.0, f"USD default already at the ready pose? total |delta|={drift}"

            # The full model-less apply: decode, base align, ARM QPOS
            # (#1828), object teleports, settle.
            adapter._apply_canonical_state(sim, random.Random(0))

            obs = sim.get_observation("robot", skip_images=True)
            for joint, expected in zip(ARM_JOINTS + GRIPPER_JOINTS, ARM_READY + GRIPPER_READY):
                assert joint in obs, f"{joint} missing from observation keys {sorted(obs)}"
                actual = float(obs[joint])
                assert math.isfinite(actual), f"{joint} is non-finite ({actual})"
                assert abs(actual - expected) < _TOL, (
                    f"{joint} = {actual:.4f}, expected {expected:.4f} (+/- {_TOL}): the init-state "
                    f"arm qpos did not land (or did not HOLD through the settle) on Isaac (#1828)."
                )
        finally:
            sim.destroy()
