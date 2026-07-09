"""End-to-end MuJoCo simulation test with Policy ABC.

Tests the full observe → policy → act → step → render pipeline
without requiring strands SDK or lerobot - just mujoco + numpy.

Run: python -m pytest tests/test_mujoco_e2e.py -v
"""

import asyncio
import os
import shutil
import tempfile

import numpy as np
import pytest

# Skip entire module if mujoco not installed
mj = pytest.importorskip("mujoco")


from strands_robots.policies import MockPolicy  # noqa: E402
from strands_robots.simulation.base import SimEngine  # noqa: E402
from strands_robots.simulation.models import SimObject, SimRobot, SimStatus, SimWorld  # noqa: E402
from tests.simulation.mujoco._gl_probe import requires_gl  # noqa: E402

# Fixtures

ROBOT_XML = """
<mujoco model="test_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <camera name="front" pos="1.5 0 1" xyaxes="0 1 0 -0.5 0 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
        <body name="link2" pos="0 0 0.2">
          <geom type="capsule" size="0.025" fromto="0 0 0 0 0 0.15" rgba="0.3 0.8 0.3 1"/>
          <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
          <body name="gripper" pos="0 0 0.15">
            <geom type="sphere" size="0.03" rgba="1 1 0 1"/>
          </body>
        </body>
      </body>
    </body>
    <body name="red_cube" pos="0.3 0 0.05">
      <freejoint name="cube_joint"/>
      <inertial pos="0 0 0" mass="0.05" diaginertia="0.001 0.001 0.001"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" rgba="1 0 0 1" condim="3"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="shoulder_lift_act" joint="shoulder_lift" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim_env():
    """Create a MuJoCo model+data from test XML."""
    tmpdir = tempfile.mkdtemp()
    xml_path = os.path.join(tmpdir, "test_arm.xml")
    with open(xml_path, "w") as f:
        f.write(ROBOT_XML)

    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    yield model, data

    shutil.rmtree(tmpdir, ignore_errors=True)


JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow"]


def read_joints(model, data):
    obs = {}
    for jname in JOINT_NAMES:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jname)
        obs[jname] = float(data.qpos[model.jnt_qposadr[jid]])
    return obs


def apply_action(model, data, action_dict):
    for key, val in action_dict.items():
        act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, f"{key}_act")
        if act_id >= 0:
            data.ctrl[act_id] = val


# Tests


class TestSimulationBase:
    def test_abc_has_required_methods(self):
        required = [
            "create_world",
            "destroy",
            "reset",
            "step",
            "get_state",
            "add_robot",
            "remove_robot",
            "add_object",
            "remove_object",
            "get_observation",
            "send_action",
            "render",
        ]
        for method in required:
            assert hasattr(SimEngine, method)

    def test_shared_dataclasses(self):
        w = SimWorld()
        assert w.timestep == 0.002
        assert w.gravity == [0.0, 0.0, -9.81]
        assert w.status == SimStatus.IDLE

        r = SimRobot(name="test", urdf_path="/tmp/test.urdf")
        assert r.joint_names == []

        o = SimObject(name="cube", shape="box")
        assert o.mass == 0.1


class TestMuJoCoPhysics:
    def test_step_advances_time(self, sim_env):
        model, data = sim_env
        assert data.time == 0.0
        for _ in range(100):
            mj.mj_step(model, data)
        assert data.time == pytest.approx(0.2, abs=1e-6)

    def test_position_actuators_move_joints(self, sim_env):
        model, data = sim_env
        data.ctrl[0] = 1.0  # shoulder_pan target
        for _ in range(1000):
            mj.mj_step(model, data)
        obs = read_joints(model, data)
        assert abs(obs["shoulder_pan"] - 1.0) < 0.15

    def test_contacts_detected(self, sim_env):
        model, data = sim_env
        for _ in range(100):
            mj.mj_step(model, data)
        assert data.ncon > 0  # cube on ground

    def test_reset_zeros_time(self, sim_env):
        model, data = sim_env
        for _ in range(100):
            mj.mj_step(model, data)
        mj.mj_resetData(model, data)
        assert data.time == 0.0


@requires_gl
class TestMuJoCoRendering:
    def test_render_rgb(self, sim_env):
        model, data = sim_env
        mj.mj_forward(model, data)
        renderer = mj.Renderer(model, height=240, width=320)
        cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "front")
        renderer.update_scene(data, camera=cam_id)
        img = renderer.render()
        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8
        assert img.max() > 0
        del renderer

    def test_render_depth(self, sim_env):
        model, data = sim_env
        mj.mj_forward(model, data)
        renderer = mj.Renderer(model, height=120, width=160)
        renderer.update_scene(data)
        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()
        assert depth.shape == (120, 160)
        assert depth.max() > 0
        del renderer


class TestMockPolicyLoop:
    def test_mock_policy_generates_actions(self):
        policy = MockPolicy()
        policy.set_robot_state_keys(JOINT_NAMES)
        obs = {j: 0.0 for j in JOINT_NAMES}
        actions = asyncio.run(policy.get_actions(obs, "test"))
        assert len(actions) == 8
        assert all(j in actions[0] for j in JOINT_NAMES)

    def test_full_observe_act_loop(self, sim_env):
        model, data = sim_env
        policy = MockPolicy()
        policy.set_robot_state_keys(JOINT_NAMES)

        for step in range(20):
            obs = read_joints(model, data)
            actions = asyncio.run(policy.get_actions(obs, "pick up cube"))
            apply_action(model, data, actions[0])
            mj.mj_step(model, data)

        assert data.time > 0
        final_obs = read_joints(model, data)
        # Joints should have moved from 0
        assert any(abs(v) > 0.001 for v in final_obs.values())

    @requires_gl
    def test_loop_with_rendering(self, sim_env):
        """Full loop: observe → policy → act → step → render (10 iterations)."""
        model, data = sim_env
        policy = MockPolicy()
        policy.set_robot_state_keys(JOINT_NAMES)
        renderer = mj.Renderer(model, height=120, width=160)

        frames = []
        for _ in range(10):
            obs = read_joints(model, data)
            actions = asyncio.run(policy.get_actions(obs, "wave"))
            apply_action(model, data, actions[0])
            mj.mj_step(model, data)

            renderer.update_scene(data)
            frames.append(renderer.render().copy())

        assert len(frames) == 10
        assert all(f.shape == (120, 160, 3) for f in frames)
        # Frames should differ (robot is moving)
        assert not np.array_equal(frames[0], frames[-1])
        del renderer


class TestDomainRandomization:
    def test_color_randomization(self, sim_env):
        model, data = sim_env
        orig = model.geom_rgba.copy()
        rng = np.random.default_rng(42)
        for i in range(model.ngeom):
            model.geom_rgba[i, :3] = rng.uniform(0.1, 1.0, size=3)
        assert not np.array_equal(orig, model.geom_rgba)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestToolSpecActionCoverage:
    """Verify every action enum in tool_spec.json maps to a real method on Simulation."""

    def test_all_actions_have_methods(self):
        """Every action in tool_spec.json must resolve to a method on Simulation."""
        import json
        from pathlib import Path

        from strands_robots.simulation.mujoco.simulation import Simulation

        spec_path = Path(__file__).resolve().parents[3] / "strands_robots" / "simulation" / "mujoco" / "tool_spec.json"
        with open(spec_path) as f:
            spec = json.load(f)

        actions = spec["properties"]["action"]["enum"]
        assert len(actions) > 0, "tool_spec.json should have at least one action"

        # Aliases used by _dispatch_action
        aliases = {
            "list_robots": "list_robots_info",
        }

        missing = []
        for action in actions:
            method_name = aliases.get(action, action)
            if not hasattr(Simulation, method_name):
                missing.append(f"{action} (looked for method '{method_name}')")

        assert not missing, "tool_spec.json actions with no matching Simulation method:\n" + "\n".join(
            f"  - {m}" for m in missing
        )

    def test_action_enum_is_not_empty(self):
        """Sanity: tool_spec.json action enum is populated."""
        import json
        from pathlib import Path

        spec_path = Path(__file__).resolve().parents[3] / "strands_robots" / "simulation" / "mujoco" / "tool_spec.json"
        with open(spec_path) as f:
            spec = json.load(f)

        actions = spec["properties"]["action"]["enum"]
        assert len(actions) >= 30, f"Expected ≥30 actions, got {len(actions)}"
