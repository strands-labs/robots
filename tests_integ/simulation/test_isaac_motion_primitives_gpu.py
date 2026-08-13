"""GPU integration tests for the Isaac motion primitives (#2156, parent #2123).

The unit suites (``tests/simulation/isaac/test_motion_primitives.py`` /
``test_move_to_ik.py``) drive ``IsaacMotionPrimitivesMixin`` against faked
articulations; these tests drive the same entry points against a real
SimulationApp/Kit runtime - real URDF import, real PD position drives, real
physics stepping, real ``dof_properties`` limits - covering the #2156
acceptance criteria:

* ``move_to`` to a known-reachable pose converges within ``tol`` (real mink
  IK on the registered MJCF, real servo descent on the articulation);
* ``move_to`` to an unreachable target (inside the workspace sanity radius,
  beyond the arm's reach) answers with the structured unreachable envelope
  carrying the IK residual - no hang, no raise, and physics does not step;
* ``set_gripper`` open -> close round trip drives the jaw to the range ends
  the shared open=HIGH / close=LOW mapping selects from the articulation's
  own reported DOF limits.

The robot is a self-contained inline URDF (the same 4-DOF positioner + jaw
kinematics the unit suites and the MuJoCo motion-primitive suite solve on -
no asset downloads), imported through the real Isaac URDF importer; its IK
model is the matching inline MJCF registered via
:func:`strands_robots.simulation.model_registry.register_urdf` under a
test-unique ``data_config``. Known-reachable / unreachable targets are the
solver-verified points the unit suites use.

Reset caution (#1895): ``reset()`` kills articulation handles on this
backend (see ``tests/simulation/isaac/test_reset_revives_articulations.py``
for the current state of that behavior), so this module shares ONE sim
session with NO reset between cases - each test leaves only joint-state
changes behind, and no test depends on the pose a previous one ended at.

Requirements match ``test_isaac_gpu.py``: NVIDIA GPU + CUDA, Isaac Sim 6.0+
installed out-of-band, ``pip install 'strands-robots[sim-mujoco]'`` for the
mink/mujoco IK stack, and ``STRANDS_GPU_TEST=1``. Run with::

    STRANDS_GPU_TEST=1 hatch run test-integ \\
        tests_integ/simulation/test_isaac_motion_primitives_gpu.py -m gpu -v
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytest.importorskip("strands_robots.simulation.isaac")
pytest.importorskip("mujoco")
pytest.importorskip("mink")

_GPU_ENABLED = os.environ.get("STRANDS_GPU_TEST", "0") == "1"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not _GPU_ENABLED,
        reason="Requires an NVIDIA GPU + Isaac Sim 6.0. Set STRANDS_GPU_TEST=1 to enable.",
    ),
]

# The same 4-DOF positioner + jaw the unit suites solve on
# (tests/simulation/isaac/test_move_to_ik.py; originally
# tests/simulation/mujoco/test_motion_primitives.py), as a URDF for the real
# Isaac importer. Joint vocabulary matches the MJCF below per NAME - the
# name-keyed reconciliation is the contract move_to writes through. The jaw
# is named 'Jaw' (so100's registry vocabulary) so the shared gripper hint
# matching is exercised against a case-differing real articulation name.
ARM_URDF = """<?xml version="1.0"?>
<robot name="prim_arm_integ">
  <link name="base_link">
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <geometry><cylinder radius="0.04" length="0.04"/></geometry>
    </collision>
  </link>
  <joint name="shoulder_pan" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="10"/>
  </joint>
  <link name="link1">
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <origin xyz="0 0 0.1"/>
      <geometry><cylinder radius="0.02" length="0.2"/></geometry>
    </collision>
  </link>
  <joint name="shoulder_lift" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.2"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="10"/>
  </joint>
  <link name="link2">
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <origin xyz="0.075 0 0"/>
      <geometry><box size="0.15 0.04 0.04"/></geometry>
    </collision>
  </link>
  <joint name="elbow" type="revolute">
    <parent link="link2"/>
    <child link="link3"/>
    <origin xyz="0.15 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="50" velocity="10"/>
  </joint>
  <link name="link3">
    <inertial>
      <mass value="0.15"/>
      <inertia ixx="0.001" iyy="0.001" izz="0.001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <origin xyz="0.075 0 0"/>
      <geometry><box size="0.15 0.036 0.036"/></geometry>
    </collision>
  </link>
  <joint name="wrist_roll" type="revolute">
    <parent link="link3"/>
    <child link="link4"/>
    <origin xyz="0.15 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.0" upper="3.0" effort="50" velocity="10"/>
  </joint>
  <link name="link4">
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0005" iyy="0.0005" izz="0.0005" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <origin xyz="0.025 0 0"/>
      <geometry><box size="0.05 0.03 0.03"/></geometry>
    </collision>
  </link>
  <joint name="Jaw" type="revolute">
    <parent link="link4"/>
    <child link="jaw_link"/>
    <origin xyz="0.05 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.2" upper="1.5" effort="20" velocity="10"/>
  </joint>
  <link name="jaw_link">
    <inertial>
      <mass value="0.02"/>
      <inertia ixx="0.00001" iyy="0.00001" izz="0.00001" ixy="0" ixz="0" iyz="0"/>
    </inertial>
    <collision>
      <geometry><box size="0.02 0.02 0.04"/></geometry>
    </collision>
  </link>
</robot>
"""

# The IK model move_to solves on: same kinematics, same joint names, an
# ``ee_site`` TCP site for EE-frame discovery. Identical to the unit suite's
# ARM_XML with the jaw renamed to the URDF's 'Jaw'.
ARM_MJCF = """
<mujoco model="prim_arm_integ">
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
                <joint name="Jaw" type="hinge" axis="0 0 1" range="-0.2 1.5"/>
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

_DATA_CONFIG = "prim_arm_integ"
_JAW_RANGE = (-0.2, 1.5)  # the URDF's declared limits; open=HIGH, close=LOW

# Solver-verified targets from the unit suites: REACHABLE is inside the
# arm's ~0.35 m workspace shell; UNREACHABLE is far outside it but inside
# the 5 m workspace sanity radius, so it reaches the IK solve and fails
# there (not at the sanity pre-flight).
REACHABLE = [0.2, 0.1, 0.2]
UNREACHABLE = [1.5, 0.0, 0.2]


def _skip_if_isaac_unavailable() -> None:
    from strands_robots.simulation.isaac import IsaacSimulation

    available, reason = IsaacSimulation.is_available()
    if not available:
        pytest.skip(f"Isaac Sim not available: {reason}")


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    for block in result["content"]:
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in result: {result}")


def _jaw_position(sim: Any) -> float:
    obs = sim.get_observation("arm", skip_images=True)
    assert "Jaw" in obs, f"no Jaw joint in observation: {sorted(obs)}"
    return float(obs["Jaw"])


@pytest.fixture(scope="module")
def sim_arm(tmp_path_factory):
    """ONE Kit session holding the inline URDF arm, shared module-wide.

    Kit startup dominates the wall time, so all tests below share this
    simulation. NO ``reset()`` anywhere in the module (the #1895 caution):
    ``sim.step()`` settles physics after the import instead, the same
    sequence ``test_isaac_delta_eef_gpu.py`` documents as live-verified.
    """
    from strands_robots.simulation import model_registry
    from strands_robots.simulation.isaac import IsaacConfig, IsaacSimulation

    _skip_if_isaac_unavailable()

    workdir = tmp_path_factory.mktemp("prim_arm_integ")
    urdf_path = workdir / "prim_arm_integ.urdf"
    urdf_path.write_text(ARM_URDF)
    mjcf_path = workdir / "prim_arm_integ.xml"
    mjcf_path.write_text(ARM_MJCF)

    # Point move_to's IK-model resolution at the matching MJCF. register_urdf
    # writes module state, so the key is removed at teardown.
    model_registry.register_urdf(_DATA_CONFIG, str(mjcf_path))

    sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
    try:
        r = sim.create_world()
        assert r["status"] == "success", f"create_world: {r}"
        r = sim.add_robot("arm", urdf_path=str(urdf_path), data_config=_DATA_CONFIG)
        assert r["status"] == "success", f"add_robot: {r}"
        # Settle the physics view; deliberately no reset() (see module
        # docstring and #1895).
        sim.step(10)
        yield sim
    finally:
        sim.destroy()
        model_registry._URDF_REGISTRY.pop(_DATA_CONFIG, None)


class TestMoveToGPU:
    def test_reaches_known_reachable_pose_within_tol(self, sim_arm):
        """#2156 acceptance: move_to a known-reachable pose converges within tol."""
        import numpy as np

        tol = 0.03
        result = sim_arm.move_to(robot_name="arm", position=REACHABLE, tol=tol, max_steps=600)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["reached"] is True
        assert payload["position_error_m"] <= tol
        assert payload["frame"] == "ee_site"
        assert payload["frame_type"] == "site"
        # The reported EE position is a real FK readback, not the target
        # echoed: it must sit within tol of the target it converged to.
        assert np.linalg.norm(np.array(payload["ee_position"]) - np.array(REACHABLE)) <= tol

    def test_unreachable_target_returns_structured_error_with_residual(self, sim_arm):
        """#2156 acceptance: unreachable -> structured error + IK residual,
        refused before a single control tick (no hang, no raise, no motion)."""
        tol = 0.01
        before = sim_arm.get_observation("arm", skip_images=True)
        result = sim_arm.move_to(robot_name="arm", position=UNREACHABLE, tol=tol, max_steps=100)
        assert result["status"] == "error", result
        assert "unreachable" in result["content"][0]["text"]
        payload = _json_block(result)
        assert payload["reached"] is False
        assert payload["ik_residual_m"] > tol
        assert payload["steps"] == 0
        # Refused pre-tick: the arm did not move toward the impossible target.
        after = sim_arm.get_observation("arm", skip_images=True)
        for joint, value in before.items():
            if isinstance(value, float):
                assert after[joint] == pytest.approx(value, abs=1e-3), (
                    f"{joint} moved on a refused move_to: {value} -> {after[joint]}"
                )


class TestSetGripperGPU:
    def test_open_close_round_trip_reaches_range_ends(self, sim_arm):
        """#2156 acceptance: open -> close round trip lands the jaw at the
        mapped range ends (open=HIGH, close=LOW - the shared registry-first
        mapping's no-metadata convention) of the articulation's own reported
        limits."""
        lo, hi = _JAW_RANGE

        result = sim_arm.set_gripper(robot_name="arm", state="open", steps=60)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["actuators"] == ["Jaw"]
        assert payload["targets"]["Jaw"] == pytest.approx(hi, abs=0.05)
        assert payload["setpoint_sources"]["Jaw"] == "articulation dof limits"
        jaw_open = _jaw_position(sim_arm)
        assert jaw_open == pytest.approx(hi, abs=0.15), f"jaw did not reach the open (HIGH) end: {jaw_open}"

        result = sim_arm.set_gripper(robot_name="arm", state="close", steps=60)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["targets"]["Jaw"] == pytest.approx(lo, abs=0.05)
        jaw_closed = _jaw_position(sim_arm)
        assert jaw_closed == pytest.approx(lo, abs=0.15), f"jaw did not reach the closed (LOW) end: {jaw_closed}"
        # The round trip traveled the range, not a numerical wiggle.
        assert jaw_open - jaw_closed > 0.8 * (hi - lo)
