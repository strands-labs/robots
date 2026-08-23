"""GPU integration tests for the Isaac delta-EEF action controller (#1812).

Verifies, against a real Isaac Sim articulation, the acceptance criterion
that a scripted delta sequence MOVES the robot: GR00T-style task-space
deltas installed via ``install_action_controller`` +
``IsaacDeltaEEFController`` must displace ``panda_hand`` in the commanded
world direction. Pre-#1812 the same action dicts landed 100% in
``send_action``'s ``unresolved_keys`` and the Franka never moved.

Requirements match ``test_isaac_gpu.py``: NVIDIA GPU, Isaac Sim 6.0+, and
``STRANDS_GPU_TEST=1``.

Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ \\
        tests_integ/simulation/test_isaac_delta_eef_gpu.py -m gpu -v
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("strands_robots.simulation.isaac")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason="Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable.",
    ),
]

ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
GRIPPER_JOINTS = ["panda_finger_joint1", "panda_finger_joint2"]
EEF_LINK = "panda_hand"


def _skip_if_isaac_unavailable() -> None:
    from strands_robots.simulation.isaac import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")


def _assets_root_path() -> str:
    """Resolve the bundled-assets root; call AFTER ``create_world()``.

    The ``isaacsim.storage.native`` extension module only joins ``sys.path``
    once ``SimulationApp`` has booted, which happens inside the first
    ``create_world()`` -- resolving earlier raises ``ModuleNotFoundError``
    on a fresh process.
    """
    try:
        from isaacsim.storage.native import get_assets_root_path  # type: ignore[import-not-found]
    except ImportError:
        from omni.isaac.nucleus import get_assets_root_path  # type: ignore[import-not-found]

    assets_root = get_assets_root_path()
    assert assets_root, "get_assets_root_path() returned empty"
    return assets_root


def _add_franka(sim) -> None:
    """Load the bundled Franka, trying the 6.0 asset layout then the 4.x one.

    Same dual-subpath fallback as ``examples/libero/run.py``'s
    ``_FRANKA_USD_SUBPATHS``: the Franka moved to
    ``Isaac/Robots/FrankaRobotics/FrankaPanda/`` in Isaac Sim 6.0.
    """
    assets_root = _assets_root_path()
    result = None
    for sub in (
        "Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",  # Isaac Sim 6.0+
        "Isaac/Robots/Franka/franka.usd",  # Isaac Sim 4.x and earlier
    ):
        result = sim.add_robot("robot", usd_path=f"{assets_root}/{sub}")
        if result["status"] == "success":
            return
    raise AssertionError(f"add_robot failed for both Franka USD layouts: {result}")


def _panda_hand_world_position(sim) -> np.ndarray:
    """World position of the ``panda_hand`` prim via USD (no get_body_state
    dependency -- that lands separately in #1802/#1811)."""
    import omni.usd  # type: ignore[import-not-found]
    from pxr import Usd, UsdGeom  # type: ignore[import-not-found]

    stage = omni.usd.get_context().get_stage()
    for prim in stage.Traverse():
        if prim.GetName() == EEF_LINK:
            xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            row = xform.ExtractTranslation()
            return np.array([row[0], row[1], row[2]], dtype=np.float64)
    raise AssertionError(f"No prim named {EEF_LINK!r} found on the stage")


def _build_controller(sim, robot_name: str):
    """Mirror LiberoAdapter._try_install_isaac_action_controller's wiring."""
    from strands_robots.simulation.isaac import IsaacDeltaEEFController

    joint_names = sim.robot_joint_names(robot_name)
    arm_indices = [joint_names.index(j) for j in ARM_JOINTS]

    def jacobian_fn() -> np.ndarray:
        result = sim.get_jacobian(body_name=EEF_LINK, robot_name=robot_name)
        assert result["status"] == "success", f"get_jacobian: {result}"
        payload = result["content"][1]["json"]
        jac = np.asarray(payload["jacp"] + payload["jacr"], dtype=np.float64)
        return jac[:, arm_indices]

    def joint_positions_fn() -> np.ndarray:
        obs = sim.get_observation(robot_name, skip_images=True)
        return np.array([float(obs[j]) for j in ARM_JOINTS], dtype=np.float64)

    return IsaacDeltaEEFController(
        arm_joint_names=ARM_JOINTS,
        gripper_joint_names=GRIPPER_JOINTS,
        joint_positions_fn=joint_positions_fn,
        jacobian_fn=jacobian_fn,
    )


class TestDeltaEEFActuationGPU:
    def test_scripted_deltas_move_panda_hand_in_commanded_direction(self):
        """A +x / -z delta sequence must displace panda_hand accordingly.

        This is the #1812 acceptance test at the engine level: the same
        ``{x, y, z, roll, pitch, yaw, gripper}`` dicts that previously
        no-op'd now actuate the articulation. 20 control steps of a
        saturated +x delta (0.05 m/step commanded) must produce clearly
        measurable +x motion; the exact magnitude depends on PD tracking,
        so the assertion is directional with a conservative floor.
        """
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()
        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            r = sim.create_world()
            assert r["status"] == "success", f"create_world: {r}"
            _add_franka(sim)

            # Two render-less steps create/settle the physics simulation
            # view (joint writes and Jacobian reads need it). Deliberately
            # NO sim.reset() here: a world.reset() after add_robot leaves
            # the articulation's cached physics view stale on Isaac 6.0
            # (get_joint_positions returns None) -- verified live on this
            # exact sequence.
            sim.step(2)
            # Move off the fully-folded home pose so the +x direction is not
            # near a workspace boundary, then settle.
            ready_pose = dict(zip(ARM_JOINTS, [0.0, -0.5, 0.0, -2.0, 0.0, 1.6, 0.8]))
            r = sim.set_joint_positions(ready_pose, robot_name="robot")
            assert r["status"] == "success", f"set_joint_positions: {r}"
            sim.step(20)

            # Jacobian must be readable after the world has stepped.
            jac_probe = sim.get_jacobian(body_name=EEF_LINK, robot_name="robot")
            assert jac_probe["status"] == "success", f"get_jacobian probe: {jac_probe}"
            assert jac_probe["content"][1]["json"]["nv"] == 9

            controller = _build_controller(sim, "robot")
            r = sim.install_action_controller("robot", controller)
            assert r["status"] == "success", f"install_action_controller: {r}"

            # physics_dt=1/120 at 20 Hz control -> 6 substeps per action,
            # matching what PolicyRunner derives from physics_timestep().
            n_substeps = round((1.0 / 20.0) / sim.physics_timestep())
            assert n_substeps == 6

            start = _panda_hand_world_position(sim)
            for _ in range(20):
                r = sim.send_action({"x": 1.0, "gripper": 1.0}, robot_name="robot", n_substeps=n_substeps)
                assert r["status"] == "success", f"send_action(+x): {r}"
            after_x = _panda_hand_world_position(sim)

            dx = after_x - start
            # 20 saturated steps command 1.0 m; PD tracking and IK damping
            # eat into that, but anything above 5 cm is unambiguous motion
            # (pre-fix displacement was exactly 0).
            assert dx[0] > 0.05, f"panda_hand did not move in +x: displacement={dx}"
            assert abs(dx[0]) > abs(dx[1]), f"+x command produced dominant off-axis y motion: {dx}"

            for _ in range(20):
                r = sim.send_action({"z": -1.0}, robot_name="robot", n_substeps=n_substeps)
                assert r["status"] == "success", f"send_action(-z): {r}"
            after_z = _panda_hand_world_position(sim)
            dz = after_z - after_x
            assert dz[2] < -0.05, f"panda_hand did not move in -z: displacement={dz}"
        finally:
            sim.destroy()

    def test_gripper_channel_drives_fingers(self):
        """RLDS gripper commands (0=close, 1=open) must move both fingers."""
        from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

        _skip_if_isaac_unavailable()
        sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
        try:
            assert sim.create_world()["status"] == "success"
            _add_franka(sim)
            sim.step(5)

            controller = _build_controller(sim, "robot")
            assert sim.install_action_controller("robot", controller)["status"] == "success"
            n_substeps = round((1.0 / 20.0) / sim.physics_timestep())

            for _ in range(10):
                assert sim.send_action({"gripper": 1.0}, robot_name="robot", n_substeps=n_substeps)["status"] == (
                    "success"
                )
            open_obs = sim.get_observation("robot", skip_images=True)
            for _ in range(10):
                assert sim.send_action({"gripper": 0.0}, robot_name="robot", n_substeps=n_substeps)["status"] == (
                    "success"
                )
            closed_obs = sim.get_observation("robot", skip_images=True)

            for finger in GRIPPER_JOINTS:
                assert open_obs[finger] > 0.03, f"{finger} did not open: {open_obs[finger]}"
                assert closed_obs[finger] < 0.01, f"{finger} did not close: {closed_obs[finger]}"
        finally:
            sim.destroy()
