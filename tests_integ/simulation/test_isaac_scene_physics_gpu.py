"""GPU-gated integration: load_scene leaves physics LIVE, not frozen (#1820).

The frozen-physics regression gate. ``IsaacSimulation.load_scene`` realizes
LIBERO dynamic objects via constructors that stop the timeline (#159), and
before #1820 nothing restarted it: the whole episode rendered without
integrating physics - joint reads went stale, ``send_action`` targeted a
view that never integrates, and every envelope still reported success, so a
5-episode groot eval read green with a motionless robot. The eval-level
signals (state keys present, no PhysX errors) were all satisfiable without
integration, which is exactly how the defect shipped; this test asserts the
one thing frozen physics cannot fake: a joint-target ``send_action`` after
``load_scene`` measurably changes the joint state.

Also pins part 2 of #1820: the scene MJCF places dynamic objects at
*placeholder* poses (LIBERO's real per-episode poses live in BDDL init
states), so the objects are teleported to legal poses via ``move_object``
(the engine half of ``LiberoAdapter._apply_object_pose_state``) BEFORE the
first integrating step, and the articulation must stay finite afterwards -
no "Illegal BroadPhaseUpdateData - non-finite bounds" NaN storm.

Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ tests_integ/simulation/test_isaac_scene_physics_gpu.py -m gpu -v
"""

from __future__ import annotations

import math
import os

import pytest

# The isaac subpackage import itself is CPU-safe (heavy omni/isaacsim imports
# are lazy, deferred to create_world()); importorskip guards against a
# broken/partial install rather than the Isaac Kit runtime.
pytest.importorskip("strands_robots.simulation.isaac")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason="Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable.",
    ),
]

# A LIBERO-shaped scene: one static fixture plus two dynamic bodies at
# COINCIDENT placeholder poses (the measured libero_spatial pattern - both
# bowls at [0, -0.1, 0.02], inside the robot base footprint). Faithful to
# the defect: integrating from this configuration is what exploded PhysX.
_SCENE_MJCF = """
<mujoco model="frozen_physics_probe">
  <worldbody>
    <body name="fixture_table" pos="0.6 0.0 0.4">
      <geom type="box" size="0.3 0.3 0.4" group="0"/>
    </body>
    <body name="bowl_1_main" pos="0.0 -0.1 0.02">
      <freejoint/>
      <geom type="box" size="0.03 0.03 0.02" group="0"/>
    </body>
    <body name="bowl_2_main" pos="0.0 -0.1 0.02">
      <freejoint/>
      <geom type="box" size="0.03 0.03 0.02" group="0"/>
    </body>
  </worldbody>
</mujoco>
"""

# Where the "init state" puts the dynamic bodies: resting on the fixture,
# clear of the robot and of each other.
_LEGAL_POSES = {
    "bowl_1_main": [0.55, -0.12, 0.83],
    "bowl_2_main": [0.55, 0.12, 0.83],
}

_PROBE_JOINT = "panda_joint2"
_PROBE_DELTA = 0.3
_MIN_MOTION = 0.05


def _skip_if_isaac_unavailable() -> None:
    from strands_robots.simulation.isaac import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")


def _add_franka(sim) -> None:
    try:
        from isaacsim.storage.native import get_assets_root_path
    except ImportError:
        from omni.isaac.nucleus import get_assets_root_path
    assets_root = get_assets_root_path()
    if not assets_root:
        pytest.skip("No Isaac assets root (Nucleus/CDN) reachable for the Franka USD")
    r = sim.add_robot("robot", usd_path=f"{assets_root}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd")
    if r["status"] != "success":
        r = sim.add_robot("robot", usd_path=f"{assets_root}/Isaac/Robots/Franka/franka.usd")
    assert r["status"] == "success", f"add_robot: {r}"


def _joint_positions(sim) -> dict[str, float]:
    obs = sim.get_observation("robot", skip_images=True)
    joints = {k: v for k, v in obs.items() if isinstance(v, float)}
    assert joints, f"observation carries no joint keys: {sorted(obs)}"
    return joints


def _assert_actions_integrate(sim, label: str) -> None:
    """The frozen-physics gate: a joint-target send_action must move the
    articulation. Under a stopped timeline this probe measures ~0 motion
    while every envelope still reports success (the #1820 diagnosis probe).
    """
    q0 = _joint_positions(sim)
    assert _PROBE_JOINT in q0, f"{label}: probe joint missing from {sorted(q0)}"
    target = q0[_PROBE_JOINT] + _PROBE_DELTA
    r = sim.send_action({_PROBE_JOINT: target}, robot_name="robot", n_substeps=30)
    assert r["status"] == "success", f"{label}: send_action: {r}"
    q1 = _joint_positions(sim)
    moved = abs(q1[_PROBE_JOINT] - q0[_PROBE_JOINT])
    assert moved > _MIN_MOTION, (
        f"{label}: joint-target send_action moved {_PROBE_JOINT} by only {moved:.6f} rad "
        f"(> {_MIN_MOTION} required). Physics is FROZEN after load_scene (#1820)."
    )
    for name, value in q1.items():
        assert math.isfinite(value), f"{label}: joint {name} is non-finite ({value}) - PhysX exploded (#1820 part 2)"


class TestIsaacScenePhysicsGPU:
    def test_load_scene_timeline_plays_and_actions_integrate(self, tmp_path):
        """One Kit boot covers the full per-episode cycle twice (SimulationApp
        can only boot once per process, so the journeys share a session):
        load_scene -> timeline playing -> pose teleport -> settle -> probe,
        then the episode-2 reload of the same scene."""
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()
        scene_file = tmp_path / "scene.xml"
        scene_file.write_text(_SCENE_MJCF)

        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            r = sim.create_world()
            assert r["status"] == "success", f"create_world: {r}"
            _add_franka(sim)
            sim.step(2)

            # Sanity: actuation works BEFORE the scene loads (isolates the
            # regression to load_scene rather than the action path).
            _assert_actions_integrate(sim, "pre-scene")

            for episode in (1, 2):
                label = f"episode {episode}"
                r = sim.load_scene(str(scene_file))
                assert r["status"] == "success", f"{label}: load_scene: {r}"

                # The timeline must be PLAYING after load_scene: the dynamic
                # prim constructors stopped it (#159) and load_scene owns the
                # restart (#1820 part 1).
                import omni.timeline

                assert omni.timeline.get_timeline_interface().is_playing(), (
                    f"{label}: timeline stopped after load_scene - the episode would run on frozen physics (#1820)"
                )

                # Part 2: the dynamic bodies sit at coincident placeholder
                # poses inside the robot base. Teleport them to legal poses
                # BEFORE the first integrating step (the engine half of
                # LiberoAdapter._apply_object_pose_state), then settle.
                for name, pos in _LEGAL_POSES.items():
                    r = sim.move_object(name=name, position=pos, orientation=[1.0, 0.0, 0.0, 0.0])
                    assert r["status"] == "success", f"{label}: move_object({name}): {r}"
                r = sim.step(5)
                assert r["status"] == "success", f"{label}: settle step: {r}"

                # The frozen-physics gate + the no-NaN-explosion gate.
                _assert_actions_integrate(sim, label)
        finally:
            sim.destroy()
