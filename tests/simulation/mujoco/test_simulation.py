"""Integration tests for the MuJoCo Simulation class.

Tests the full Simulation public API through behavioral end-to-end scenarios
- create worlds, add robots/objects/cameras, step physics, render, record,
randomize, dispatch actions, and clean up.

Every test exercises real user-visible behavior. No isinstance checks or
attribute-existence tests.

Run: MUJOCO_GL=osmesa python -m pytest tests/test_mujoco_simulation.py -v
"""

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco import backend as backend_mod  # noqa: E402
from strands_robots.simulation.mujoco.backend import _can_render  # noqa: E402

requires_gl = pytest.mark.skipif(
    not _can_render(),
    reason="No OpenGL context available (headless without EGL/OSMesa)",
)

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# Test robot XML

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
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="shoulder_lift_act" joint="shoulder_lift" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
  </actuator>
</mujoco>
"""


# Robot with a gripper-named end body. list_bodies(robot_name=...) advertises the
# best-guess gripper/EEF mount so an agent can attach a wrist camera without
# guessing a body name. The short name must contain gripper/hand/ee/tool.
GRIPPER_ARM_XML = """
<mujoco model="gripper_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base" pos="0 0 0.1">
      <geom type="cylinder" size="0.05 0.05" rgba="0.3 0.3 0.8 1"/>
      <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
      <body name="link1" pos="0 0 0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="elbow" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
        <body name="gripper" pos="0 0 0.2">
          <geom type="box" size="0.02 0.02 0.02" rgba="0.2 0.2 0.2 1"/>
          <joint name="jaw" type="slide" axis="0 1 0" range="0 0.04"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan_act" joint="shoulder_pan" kp="50"/>
    <position name="elbow_act" joint="elbow" kp="50"/>
    <position name="jaw_act" joint="jaw" kp="20"/>
  </actuator>
</mujoco>
"""


# Free-base (floating) robot - a 1-DoF "leg" on a free joint, like the G1's
# floating base. Used to test that get_observation surfaces base_quat /
# base_ang_vel for locomotion controllers (WBC).
FREE_BASE_XML = """
<mujoco model="test_floating">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="torso" pos="0 0 0.5">
      <freejoint name="floating_base_joint"/>
      <geom type="box" size="0.1 0.1 0.1" rgba="0.3 0.3 0.8 1"/>
      <body name="leg" pos="0 0 -0.1">
        <geom type="capsule" size="0.03" fromto="0 0 0 0 0 -0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="hip" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="hip_act" joint="hip"/>
  </actuator>
</mujoco>
"""


# Scene with a material-backed, UNNAMED geom - mirrors a real robot whose mesh
# geoms are unnamed and draw their colour from a referenced material (so the
# renderer ignores geom_rgba for them).
MATERIAL_SCENE_XML = """
<mujoco model="mat_scene">
  <compiler angle="radian" autolimits="true"/>
  <asset>
    <material name="body_mat" rgba="0.2 0.2 0.2 1"/>
  </asset>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="base" pos="0 0 0.1">
      <joint name="j" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.05 0.05 0.05" material="body_mat"/>
    </body>
  </worldbody>
  <actuator>
    <position name="j_act" joint="j" kp="10"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def sim():
    """Create a fresh Simulation instance."""
    s = Simulation(tool_name="test_sim", mesh=False)
    yield s
    s.cleanup()


@pytest.fixture
def free_base_robot_xml_path():
    """Write the free-base (floating) robot XML to a temp file."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_floating.xml")
    with open(path, "w") as f:
        f.write(FREE_BASE_XML)
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)


def test_get_observation_free_base_has_base_imu(sim_with_world, free_base_robot_xml_path):
    """A floating-base robot surfaces base_quat (w,x,y,z) + base_ang_vel (rad/s)
    from its free joint, and per-joint .vel for the hinge joints - the inputs a
    locomotion controller (WBC) needs to close the loop. The free joint itself
    gets no scalar .vel (it is 6-DoF, not a 1-DoF hinge/slide)."""
    sim_with_world.add_robot("floater", urdf_path=free_base_robot_xml_path)
    obs = sim_with_world.get_observation(robot_name="floater", skip_images=True)

    assert "base_quat" in obs, "floating-base robot must surface base_quat"
    assert len(obs["base_quat"]) == 4
    assert all(isinstance(x, float) for x in obs["base_quat"])
    assert "base_ang_vel" in obs, "floating-base robot must surface base_ang_vel"
    assert len(obs["base_ang_vel"]) == 3

    # The hinge joint has a scalar .vel; the free joint does not.
    assert "hip.vel" in obs and isinstance(obs["hip.vel"], float)
    assert "floating_base_joint.vel" not in obs


@pytest.fixture
def sim_with_world(sim):
    """Simulation with a world already created."""
    result = sim.create_world(gravity=[0, 0, -9.81])
    assert result["status"] == "success"
    return sim


@pytest.fixture
def robot_xml_path():
    """Write test robot XML to a temp file."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "test_arm.xml")
    with open(path, "w") as f:
        f.write(ROBOT_XML)
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def material_scene_path():
    """Write the material-backed scene XML to a temp file."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "mat_scene.xml")
    with open(path, "w") as f:
        f.write(MATERIAL_SCENE_XML)
    yield path
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def sim_with_robot(sim_with_world, robot_xml_path):
    """Simulation with world + robot loaded."""
    result = sim_with_world.add_robot("arm1", urdf_path=robot_xml_path)
    assert result["status"] == "success"
    return sim_with_world


@pytest.fixture
def sim_with_gripper_robot(sim_with_world):
    """Simulation with world + a robot whose end body is named ``gripper``."""
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "gripper_arm.xml")
    with open(path, "w") as f:
        f.write(GRIPPER_ARM_XML)
    try:
        result = sim_with_world.add_robot("griparm", urdf_path=path)
        assert result["status"] == "success"
        yield sim_with_world
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# World Management


class TestConstructorForwardCompatKwargs:
    """The engine constructor accepts and ignores backend-specific kwargs.

    The shared ``create_simulation`` / ``Robot`` factory forwards one superset
    of keyword arguments to whichever backend is selected. MuJoCo tolerates and
    drops kwargs it does not use (e.g. ``num_envs`` / ``device`` meant for GPU
    backends), mirroring ``NewtonSimEngine``, so an identical factory call
    resolves across backends. This pins that documented forward-compatible
    contract: unknown kwargs must neither raise nor disturb recognized params.
    """

    def test_unknown_kwargs_are_ignored_not_raised(self):
        sim = Simulation(num_envs=4, device="cuda", some_future_backend_kwarg=True)
        assert sim._world is None  # construction only; no world yet
        sim.cleanup()

    def test_recognized_params_survive_alongside_unknown_kwargs(self):
        sim = Simulation(
            tool_name="factory_sim",
            default_width=320,
            default_height=240,
            num_envs=8,  # forwarded-but-ignored backend kwarg
        )
        assert sim.tool_name_str == "factory_sim"
        assert sim.default_width == 320
        assert sim.default_height == 240
        # The ignored kwarg leaves no stray attribute behind.
        assert not hasattr(sim, "num_envs")
        sim.cleanup()


class TestWorldLifecycle:
    """Test create_world → get_state → reset → destroy lifecycle."""

    def test_create_world_defaults(self, sim):
        result = sim.create_world()
        assert result["status"] == "success"
        assert "Simulation world created" in result["content"][0]["text"]
        assert sim._world is not None
        assert sim._world.gravity == [0.0, 0.0, -9.81]

    def test_create_world_custom_gravity(self, sim):
        result = sim.create_world(gravity=[0, 0, -5.0])
        assert result["status"] == "success"
        assert sim._world.gravity == [0.0, 0.0, -5.0]

    def test_create_world_scalar_gravity(self, sim):
        result = sim.create_world(gravity=-3.0)
        assert result["status"] == "success"
        assert sim._world.gravity == [0.0, 0.0, -3.0]

    def test_create_world_custom_timestep(self, sim):
        result = sim.create_world(timestep=0.001)
        assert result["status"] == "success"
        assert sim._world.timestep == 0.001

    def test_create_world_no_ground_plane(self, sim):
        result = sim.create_world(ground_plane=False)
        assert result["status"] == "success"

    def test_create_world_duplicate_fails(self, sim_with_world):
        result = sim_with_world.create_world()
        assert result["status"] == "error"
        assert "already exists" in result["content"][0]["text"]

    def test_get_state(self, sim_with_world):
        result = sim_with_world.get_state()
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Simulation State" in text
        assert "t=" in text

    def test_reset(self, sim_with_world):
        # Step forward
        sim_with_world.step(n_steps=100)
        assert sim_with_world._world.sim_time > 0

        # Reset
        result = sim_with_world.reset()
        assert result["status"] == "success"
        assert sim_with_world._world.sim_time == 0.0
        assert sim_with_world._world.step_count == 0

    def test_destroy(self, sim_with_world):
        result = sim_with_world.destroy()
        assert result["status"] == "success"
        assert sim_with_world._world is None

    def test_destroy_no_world(self, sim):
        result = sim.destroy()
        assert result["status"] == "success"

    def test_step_advances_state(self, sim_with_world):
        result = sim_with_world.step(n_steps=50)
        assert result["status"] == "success"
        assert sim_with_world._world.step_count == 50
        assert sim_with_world._world.sim_time > 0

    def test_set_gravity(self, sim_with_world):
        result = sim_with_world.set_gravity([0, 0, -5.0])
        assert result["status"] == "success"
        assert sim_with_world._world.gravity == [0, 0, -5.0]

    def test_set_gravity_scalar(self, sim_with_world):
        result = sim_with_world.set_gravity(-3.0)
        assert result["status"] == "success"
        assert sim_with_world._world.gravity == [0.0, 0.0, -3.0]

    def test_set_timestep(self, sim_with_world):
        result = sim_with_world.set_timestep(0.001)
        assert result["status"] == "success"
        assert sim_with_world._world.timestep == 0.001

    def test_load_scene_from_file(self, sim, robot_xml_path):
        result = sim.load_scene(robot_xml_path)
        assert result["status"] == "success"
        assert "Scene loaded" in result["content"][0]["text"]
        assert sim._world._model.njnt > 0

    def test_load_scene_nonexistent(self, sim):
        result = sim.load_scene("/nonexistent/path.xml")
        assert result["status"] == "error"

    def test_load_scene_malformed_mjcf_returns_error(self, sim, tmp_path):
        """A syntactically-present but semantically-invalid MJCF must resolve to
        a structured error dict, not an escaped exception.

        ``load_scene`` guards the missing-file case up front, but a file that
        exists and passes that guard can still fail deep in the MuJoCo compile
        (bad attribute value, inconsistent joints, unknown element). That failure
        must be converted into the ``{"status": "error"}`` contract every facade
        method upholds so a caller/agent never has to catch a raw compile
        exception mid-dispatch.
        """
        bad = tmp_path / "malformed.xml"
        bad.write_text(
            '<mujoco model="bad"><worldbody><body><geom type="box" size="not-a-number"/></body></worldbody></mujoco>'
        )

        result = sim.load_scene(str(bad))

        assert result["status"] == "error"
        assert "Failed to load scene" in result["content"][0]["text"]


# Object Management


class TestObjectManagement:
    """Test add_object → list_objects → move_object → remove_object."""

    def test_add_object_box(self, sim_with_world):
        result = sim_with_world.add_object("red_cube", shape="box", position=[0.3, 0, 0.1], color=[1, 0, 0, 1])
        assert result["status"] == "success"
        assert "red_cube" in sim_with_world._world.objects

    def test_add_object_sphere(self, sim_with_world):
        result = sim_with_world.add_object("ball", shape="sphere", mass=0.2)
        assert result["status"] == "success"

    def test_add_object_cylinder(self, sim_with_world):
        result = sim_with_world.add_object("can", shape="cylinder", is_static=True)
        assert result["status"] == "success"

    def test_add_duplicate_object_fails(self, sim_with_world):
        sim_with_world.add_object("obj1", shape="box")
        result = sim_with_world.add_object("obj1", shape="sphere")
        assert result["status"] == "error"
        assert "exists" in result["content"][0]["text"]

    def test_add_object_no_world(self, sim):
        result = sim.add_object("obj", shape="box")
        assert result["status"] == "error"

    def test_list_objects_empty(self, sim_with_world):
        result = sim_with_world.list_objects()
        assert result["status"] == "success"
        assert "No objects" in result["content"][0]["text"]

    def test_list_objects_populated(self, sim_with_world):
        sim_with_world.add_object("a", shape="box")
        sim_with_world.add_object("b", shape="sphere")
        result = sim_with_world.list_objects()
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "a" in text
        assert "b" in text

    def test_move_object(self, sim_with_world):
        sim_with_world.add_object("cube", shape="box", position=[0, 0, 0.1])
        result = sim_with_world.move_object("cube", position=[1.0, 0, 0.1])
        assert result["status"] == "success"
        assert sim_with_world._world.objects["cube"].position == [1.0, 0, 0.1]

    def test_move_nonexistent_object(self, sim_with_world):
        result = sim_with_world.move_object("ghost", position=[0, 0, 0])
        assert result["status"] == "error"

    def test_remove_object(self, sim_with_world):
        sim_with_world.add_object("tmp", shape="box")
        assert "tmp" in sim_with_world._world.objects
        result = sim_with_world.remove_object("tmp")
        assert result["status"] == "success"
        assert "tmp" not in sim_with_world._world.objects

    def test_remove_nonexistent_object(self, sim_with_world):
        result = sim_with_world.remove_object("ghost")
        assert result["status"] == "error"


# Robot Management


class TestRobotManagement:
    """Test add_robot → list_robots → get_robot_state → remove_robot."""

    def test_add_robot(self, sim_with_world, robot_xml_path):
        result = sim_with_world.add_robot("arm1", urdf_path=robot_xml_path)
        assert result["status"] == "success"
        assert "arm1" in sim_with_world._world.robots
        robot = sim_with_world._world.robots["arm1"]
        assert len(robot.joint_names) == 3
        assert len(robot.actuator_ids) > 0

    def test_add_robot_no_world(self, sim, robot_xml_path):
        result = sim.add_robot("arm1", urdf_path=robot_xml_path)
        assert result["status"] == "error"

    def test_add_duplicate_robot(self, sim_with_robot, robot_xml_path):
        result = sim_with_robot.add_robot("arm1", urdf_path=robot_xml_path)
        assert result["status"] == "error"

    def test_add_robot_nonexistent_file(self, sim_with_world):
        result = sim_with_world.add_robot("arm", urdf_path="/nonexistent.xml")
        assert result["status"] == "error"

    def test_add_robot_no_path(self, sim_with_world):
        # Neither urdf_path nor data_config, and name doesn't resolve
        result = sim_with_world.add_robot("nonexistent_model_xyz")
        assert result["status"] == "error"

    def test_add_robot_name_optional_derives_from_urdf(self, sim_with_world, robot_xml_path):
        """Friction fix: name= is optional; auto-derived from the URDF filename."""
        result = sim_with_world.add_robot(urdf_path=robot_xml_path)
        assert result["status"] == "success"
        # Label derived from the URDF stem; robot is addressable under it.
        assert len(sim_with_world._world.robots) == 1
        derived = next(iter(sim_with_world._world.robots))
        assert derived  # non-empty label

    def test_add_robot_auto_numbers_on_collision(self, sim_with_world, robot_xml_path):
        """Two no-name adds of the same model auto-dedupe instead of erroring."""
        import os

        stem = os.path.splitext(os.path.basename(robot_xml_path))[0]
        r1 = sim_with_world.add_robot(urdf_path=robot_xml_path)
        r2 = sim_with_world.add_robot(urdf_path=robot_xml_path)
        assert r1["status"] == "success"
        assert r2["status"] == "success"
        names = set(sim_with_world._world.robots)
        assert stem in names
        assert f"{stem}_2" in names

    def test_add_duplicate_explicit_name_lists_existing(self, sim_with_robot, robot_xml_path):
        """Explicit duplicate name still errors, but names the existing robots."""
        result = sim_with_robot.add_robot("arm1", urdf_path=robot_xml_path)
        assert result["status"] == "error"
        assert "arm1" in result["content"][0]["text"]

    def test_list_robots_empty(self, sim_with_world):
        # SimEngine ABC: list[str]
        assert sim_with_world.list_robots() == []
        # Agent-tool action surface: dict
        result = sim_with_world.list_robots_info()
        assert result["status"] == "success"
        assert "No robots" in result["content"][0]["text"]

    def test_list_robots_populated(self, sim_with_robot):
        # SimEngine ABC: list[str]
        assert "arm1" in sim_with_robot.list_robots()
        # Agent-tool action surface: dict
        result = sim_with_robot.list_robots_info()
        assert result["status"] == "success"
        assert "arm1" in result["content"][0]["text"]

    def test_get_robot_state(self, sim_with_robot):
        result = sim_with_robot.get_robot_state("arm1")
        assert result["status"] == "success"
        # Should contain joint position data
        text = result["content"][0]["text"]
        assert "shoulder_pan" in text

    def test_get_robot_state_invalid(self, sim_with_robot):
        result = sim_with_robot.get_robot_state("nonexistent")
        assert result["status"] == "error"

    def test_remove_robot(self, sim_with_robot):
        result = sim_with_robot.remove_robot("arm1")
        assert result["status"] == "success"
        assert "arm1" not in sim_with_robot._world.robots

    def test_remove_nonexistent_robot(self, sim_with_world):
        result = sim_with_world.remove_robot("ghost")
        assert result["status"] == "error"

    def test_robot_compatible_observation(self, sim_with_robot):
        """Robot ABC compatible get_observation should return joint data."""
        obs = sim_with_robot.get_observation(robot_name="arm1")
        assert isinstance(obs, dict)
        # Should have joint positions
        assert len(obs) > 0

    @requires_gl
    def test_get_observation_schema_joints_plus_cameras(self, sim_with_robot):
        """get_observation must return {short_joint: float, camera_name: ndarray}.

        Locks the ABC schema contract for downstream policies/backends.
        """
        import numpy as np

        sim_with_robot.add_camera("wrist", position=[0.2, -0.2, 0.3], target=[0, 0, 0])
        obs = sim_with_robot.get_observation(robot_name="arm1")

        # Joint POSITION entries: keyed by *short* names, values are floats.
        joint_names = set(sim_with_robot._world.robots["arm1"].joint_names)
        joint_entries = {k: v for k, v in obs.items() if k in joint_names}
        assert joint_entries, "expected at least one joint in observation"
        for name, value in joint_entries.items():
            assert isinstance(value, float), f"joint {name} must be float, got {type(value).__name__}"

        # Per-joint VELOCITY entries (`<joint>.vel`, float) - additive, for
        # velocity-feedback controllers (WBC). Each must correspond to a joint.
        vel_entries = {k: v for k, v in obs.items() if k.endswith(".vel")}
        for name, value in vel_entries.items():
            assert name[:-4] in joint_names, f"{name} has no matching joint"
            assert isinstance(value, float), f"{name} must be float, got {type(value).__name__}"

        # Base IMU entries (floating-base robots only): base_quat / base_ang_vel
        # are lists of floats. A fixed-base arm has neither.
        base_keys = {"base_quat", "base_ang_vel"}
        for name in base_keys & set(obs):
            assert isinstance(obs[name], list) and all(isinstance(x, float) for x in obs[name])

        # Camera entries: any remaining non-joint, non-vel, non-base key is an
        # RGB uint8 ndarray.
        camera_entries = {
            k: v for k, v in obs.items() if k not in joint_names and not k.endswith(".vel") and k not in base_keys
        }
        assert "wrist" in camera_entries, "user-added camera must appear in observation"
        for name, frame in camera_entries.items():
            assert isinstance(frame, np.ndarray), f"camera {name} must be ndarray"
            assert frame.ndim == 3 and frame.shape[2] == 3, f"camera {name} must be HxWx3, got shape {frame.shape}"
            assert frame.dtype == np.uint8, f"camera {name} must be uint8, got {frame.dtype}"

    def test_get_observation_fixed_base_has_vel_but_no_base_imu(self, sim_with_robot):
        """A fixed-base arm gets per-joint `.vel` keys (velocity-feedback
        controllers consume them) but NO base_quat / base_ang_vel - it has no
        floating-base free joint."""
        obs = sim_with_robot.get_observation(robot_name="arm1", skip_images=True)
        joint_names = set(sim_with_robot._world.robots["arm1"].joint_names)
        vel_keys = [k for k in obs if k.endswith(".vel")]
        assert vel_keys, "expected per-joint .vel keys"
        for k in vel_keys:
            assert k[:-4] in joint_names and isinstance(obs[k], float)
        # Fixed-base arm: no floating-base IMU signals.
        assert "base_quat" not in obs
        assert "base_ang_vel" not in obs

    def test_get_observation_signature_has_no_camera_name(self):
        """Regression: get_observation must not accept a camera_name param.

        Single-camera render belongs to ``render()``. See base.py schema docs.
        """
        import inspect

        from strands_robots.simulation.base import SimEngine
        from strands_robots.simulation.mujoco.simulation import Simulation

        for cls in (SimEngine, Simulation):
            params = inspect.signature(cls.get_observation).parameters
            assert "camera_name" not in params, (
                f"{cls.__name__}.get_observation must not take camera_name; use render() for single-camera rendering."
            )
            assert "robot_name" in params

    def test_robot_compatible_send_action(self, sim_with_robot):
        """Robot ABC compatible send_action should not crash."""
        sim_with_robot.send_action(
            {"shoulder_pan_act": 0.5, "shoulder_lift_act": 0.1, "elbow_act": -0.2},
            robot_name="arm1",
        )
        # Verify physics advanced
        assert sim_with_robot._world.sim_time > 0

    def test_send_action_unresolved_keys_in_json_content_block(self, sim_with_robot):
        """Unresolved action keys must surface as a ``json`` content block
        (not a top-level sibling of ``status``/``content``), so the result
        conforms to the Strands tool-result schema and agents can read the
        structured payload alongside the human-readable ``text`` block."""
        result = sim_with_robot.send_action(
            {"shoulder_pan_act": 0.3, "nonexistent_joint": 1.0},
            robot_name="arm1",
        )
        assert result["status"] == "error"
        # Must NOT leak the key at the top level.
        assert "unresolved_keys" not in result
        # Find the json content block.
        json_blocks = [c["json"] for c in result["content"] if "json" in c]
        assert len(json_blocks) == 1, f"expected one json block, got {result['content']}"
        payload = json_blocks[0]
        assert payload["unresolved_keys"] == ["nonexistent_joint"]
        assert "shoulder_pan_act" in payload["applied"]
        # A human-readable text block must still be present.
        assert any("text" in c for c in result["content"])


# Camera Management


class TestCameraManagement:
    def test_add_camera(self, sim_with_world):
        result = sim_with_world.add_camera("overhead", position=[0, 0, 3], target=[0, 0, 0])
        assert result["status"] == "success"
        assert "overhead" in sim_with_world._world.cameras

    def test_add_camera_no_world(self, sim):
        result = sim.add_camera("cam")
        assert result["status"] == "error"

    def test_remove_camera(self, sim_with_world):
        sim_with_world.add_camera("tmp_cam")
        result = sim_with_world.remove_camera("tmp_cam")
        assert result["status"] == "success"
        assert "tmp_cam" not in sim_with_world._world.cameras

    def test_remove_nonexistent_camera(self, sim_with_world):
        result = sim_with_world.remove_camera("ghost")
        assert result["status"] == "error"

    def test_add_camera_unknown_parent_body_lists_available_bodies(self, sim_with_robot):
        """Mounting a camera on a non-existent body must fail with a discovery
        error that names the bodies actually present, so an agent can correct
        the call without guessing. Robot bodies are namespaced ``<robot>/<body>``.
        """
        result = sim_with_robot.add_camera(
            "wrist",
            position=[0.2, -0.2, 0.3],
            target=[0, 0, 0],
            parent_body="no_such_body",
        )
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "no_such_body" in text
        assert "arm1/base" in text
        # The failed mount must not leave a dangling registry entry behind.
        assert "wrist" not in sim_with_robot._world.cameras

    def test_add_camera_valid_parent_body_mounts_and_reports(self, sim_with_robot):
        """A camera mounted on an existing namespaced body succeeds and the
        success message names the mount target."""
        result = sim_with_robot.add_camera(
            "wrist",
            position=[0.2, -0.2, 0.3],
            target=[0, 0, 0],
            parent_body="arm1/base",
        )
        assert result["status"] == "success"
        assert "arm1/base" in result["content"][0]["text"]
        assert sim_with_robot._world.cameras["wrist"].parent_body == "arm1/base"


class TestListBodies:
    """list_bodies is the discovery surface for add_camera(parent_body=...).

    Before it existed, the add_camera docstring pointed agents at list_robots
    / get_robot_state to find mount points -- but those return robot names and
    joint names, never body names. An agent had to guess a body (e.g.
    ``Fixed_Jaw``), mount against a non-existent body, read the failure, and
    retry. These tests pin the deterministic discovery path that removes the
    wasted turn.
    """

    def test_list_bodies_no_world_errors(self, sim):
        result = sim.list_bodies()
        assert result["status"] == "error"
        assert "No world" in result["content"][0]["text"]

    def test_list_bodies_global_lists_every_body(self, sim_with_robot):
        result = sim_with_robot.list_bodies()
        assert result["status"] == "success"
        bodies = result["content"][1]["json"]["bodies"]
        # world body + the robot's namespaced bodies are all present.
        assert "world" in bodies
        assert "arm1/base" in bodies
        assert "arm1/link1" in bodies

    def test_list_bodies_scoped_to_robot_excludes_world(self, sim_with_robot):
        result = sim_with_robot.list_bodies(robot_name="arm1")
        assert result["status"] == "success"
        payload = result["content"][1]["json"]
        bodies = payload["bodies"]
        assert "world" not in bodies
        assert all(b.startswith("arm1/") for b in bodies)
        # gripper_body is reported (None here -- the test arm has no gripper-like body).
        assert "gripper_body" in payload

    def test_list_bodies_unknown_robot_errors_with_known_names(self, sim_with_robot):
        result = sim_with_robot.list_bodies(robot_name="ghost")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "ghost" in text
        assert "arm1" in text

    def test_discovered_body_mounts_camera_without_guessing(self, sim_with_robot):
        """End-to-end: list_bodies -> pick a body -> add_camera succeeds on the
        first attempt, with no failed-then-recovered round trip."""
        bodies = sim_with_robot.list_bodies(robot_name="arm1")["content"][1]["json"]["bodies"]
        mount = bodies[0]
        result = sim_with_robot.add_camera(
            "wrist",
            position=[0.0, 0.0, 0.1],
            target=[0.0, 0.0, 0.0],
            parent_body=mount,
        )
        assert result["status"] == "success"
        assert sim_with_robot._world.cameras["wrist"].parent_body == mount

    def test_list_bodies_dispatches_via_action_string(self, sim_with_robot):
        """The action is wired through the agent-tool dispatcher, not just the
        Python method."""
        result = sim_with_robot(action="list_bodies", robot_name="arm1")
        assert result["status"] == "success"
        assert "arm1/base" in result["content"][1]["json"]["bodies"]

    def test_describe_advertises_bodies_and_list_bodies_method(self, sim_with_robot):
        described = sim_with_robot.describe()
        assert "arm1/base" in described["bodies"]
        assert "list_bodies" in described["methods"]

    def test_describe_advertises_get_features_the_action_key_source(self, sim_with_robot):
        """describe() must advertise get_features -- the joint/actuator/camera-
        name discovery method that is the source of truth for the action keys a
        policy must emit. The advertisement is only useful if the method it names
        is real and returns those names, so assert both: the signature is on the
        discovery surface, and calling it yields a non-empty actuator listing."""
        described = sim_with_robot.describe()
        assert "get_features" in described["methods"]
        assert "robot_name" in described["methods"]["get_features"]

        feats = sim_with_robot.get_features(robot_name="arm1")
        assert feats["status"] == "success", feats
        payload = next(c["json"] for c in feats["content"] if isinstance(c, dict) and "json" in c)
        assert payload["features"]["actuator_names"], "get_features must list actuator names"

    def test_fail_fast_recommended_method_is_discoverable_via_describe(self, sim_with_robot):
        """Closed loop: when a policy's action keys resolve to no actuator,
        run_policy's fail-fast error tells the caller to inspect the expected
        keys via get_features(robot_name=...). The method that error names must
        be discoverable on the primary discovery surface (describe), so an agent
        recovering from the error does not have to scrape a method name out of an
        error string only to find describe() never listed it."""
        from strands_robots.policies.base import Policy

        class _WrongKeysPolicy(Policy):
            @property
            def provider_name(self) -> str:
                return "wrong_keys_test"

            @property
            def requires_images(self) -> bool:
                return False

            def set_robot_state_keys(self, robot_state_keys):
                pass

            async def get_actions(self, observation_dict, instruction, **kwargs):
                # A key no actuator can absorb -> 100% unresolved every step.
                return [{"definitely_not_a_joint": 0.5}]

        result = sim_with_robot.run_policy(
            robot_name="arm1",
            policy_object=_WrongKeysPolicy(),
            n_steps=50,
            control_frequency=20.0,
            fast_mode=True,
        )
        assert result["status"] == "error", result
        err_text = result["content"][0]["text"]
        # The diagnostic recommends get_features by name...
        assert "get_features" in err_text
        # ...and that method is discoverable on describe()'s method surface.
        assert "get_features" in sim_with_robot.describe()["methods"]

    def test_list_bodies_reports_gripper_mount_when_present(self, sim_with_gripper_robot):
        """A robot with a gripper-like body gets its wrist/EEF mount resolved
        automatically. This is the payoff of the discovery surface: the agent
        reads ``gripper_body`` and mounts a wrist camera in one turn instead of
        guessing the body name. The detection matches on the short (namespace-
        stripped) body name, so the reported mount stays namespaced."""
        result = sim_with_gripper_robot.list_bodies(robot_name="griparm")
        assert result["status"] == "success"
        payload = result["content"][1]["json"]
        assert payload["gripper_body"] == "griparm/gripper"
        # The human-readable text advertises the same mount point.
        assert "Gripper/EEF mount: 'griparm/gripper'" in result["content"][0]["text"]
        # And the discovered mount actually works as an add_camera parent_body.
        cam = sim_with_gripper_robot.add_camera(
            "wrist",
            position=[0.0, 0.0, 0.05],
            target=[0.0, 0.0, 0.0],
            parent_body=payload["gripper_body"],
        )
        assert cam["status"] == "success"
        assert sim_with_gripper_robot._world.cameras["wrist"].parent_body == "griparm/gripper"


# Scene Injection (XML round-trip)


class TestSceneInjection:
    """Test that objects/cameras injected into a robot scene persist."""

    def test_add_object_to_robot_scene(self, sim_with_robot):
        """Adding an object to a scene with robots uses XML injection."""
        old_nbody = sim_with_robot._world._model.nbody
        result = sim_with_robot.add_object("cube", shape="box", position=[0.3, 0, 0.05])
        assert result["status"] == "success"
        # The model should have more bodies after injection
        assert sim_with_robot._world._model.nbody > old_nbody

    def test_remove_object_from_robot_scene(self, sim_with_robot):
        sim_with_robot.add_object("cube", shape="box", position=[0.3, 0, 0.05])
        nbody_with_cube = sim_with_robot._world._model.nbody
        sim_with_robot.remove_object("cube")
        # After ejection, body count should decrease
        assert sim_with_robot._world._model.nbody < nbody_with_cube

    def test_add_camera_to_robot_scene(self, sim_with_robot):
        """Cameras injected into robot scene via XML round-trip."""
        result = sim_with_robot.add_camera("top", position=[0, 0, 2])
        assert result["status"] == "success"
        assert "top" in sim_with_robot._world.cameras

    def test_robot_joints_survive_object_injection(self, sim_with_robot):
        """Verify robot joint IDs are re-discovered after scene recompile."""
        robot = sim_with_robot._world.robots["arm1"]
        original_joints = list(robot.joint_names)

        sim_with_robot.add_object("box1", shape="box", position=[0.5, 0, 0.1])

        # Joints should still be valid
        assert robot.joint_names == original_joints
        assert len(robot.joint_ids) == len(original_joints)
        assert len(robot.actuator_ids) > 0


# Rendering


@requires_gl
class TestRendering:
    def test_render_default_camera(self, sim_with_world):
        result = sim_with_world.render(camera_name="default")
        assert result["status"] == "success"
        assert any("image" in c for c in result["content"])

    def test_render_custom_size(self, sim_with_world):
        result = sim_with_world.render(width=320, height=240)
        assert result["status"] == "success"

    def test_render_depth(self, sim_with_world):
        result = sim_with_world.render_depth()
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "Depth" in text

    def test_render_no_world(self, sim):
        result = sim.render()
        assert result["status"] == "error"

    def test_get_contacts(self, sim_with_world):
        # Add an object that will contact the ground
        sim_with_world.add_object("ball", shape="sphere", position=[0, 0, 0.5])
        sim_with_world.step(n_steps=500)
        result = sim_with_world.get_contacts()
        assert result["status"] == "success"


# Randomization


class TestRandomization:
    def test_randomize_colors(self, sim_with_world):
        sim_with_world.add_object("cube", shape="box")
        result = sim_with_world.randomize(randomize_colors=True, seed=42)
        assert result["status"] == "success"
        assert "Colors" in result["content"][0]["text"]

    def test_randomize_lighting(self, sim_with_world):
        result = sim_with_world.randomize(randomize_lighting=True, seed=42)
        assert result["status"] == "success"

    def test_randomize_physics(self, sim_with_world):
        sim_with_world.add_object("cube", shape="box")
        result = sim_with_world.randomize(randomize_physics=True, seed=42)
        assert result["status"] == "success"
        assert "Physics" in result["content"][0]["text"]

    def test_randomize_physics_scales_inertia_with_mass(self, sim_with_robot):
        """Scaling a body's mass must scale its inertia by the same factor.

        For a rigid body at fixed geometry a mass scale ``s`` (a uniform
        density change) scales the inertia tensor by the same ``s`` (I is the
        integral of r^2 dm). Pre-fix ``randomize_physics`` scaled ``body_mass``
        but left ``body_inertia`` untouched, producing physically inconsistent
        bodies - heavy in translation but with the light body's rotational
        resistance - which silently corrupts the dynamics the randomization is
        meant to perturb (and diverges from the Newton backend, which scales
        both).
        """
        m = sim_with_robot._world._model
        mass_before = m.body_mass.copy()
        inertia_before = m.body_inertia.copy()

        result = sim_with_robot.randomize(
            randomize_colors=False, randomize_lighting=False, randomize_physics=True, seed=42
        )
        assert result["status"] == "success"

        mass_after = m.body_mass.copy()
        inertia_after = m.body_inertia.copy()

        scaled_bodies = [
            i for i in range(m.nbody) if mass_before[i] > 0 and not np.isclose(mass_before[i], mass_after[i])
        ]
        assert scaled_bodies, "the robot must have bodies whose mass was scaled"

        # inertia must actually move (pre-fix it stayed stale) ...
        assert not np.allclose(inertia_before[scaled_bodies], inertia_after[scaled_bodies])
        # ... and by exactly the per-body mass factor, keeping each body consistent.
        for i in scaled_bodies:
            s = mass_after[i] / mass_before[i]
            assert np.allclose(inertia_after[i], inertia_before[i] * s, rtol=1e-6), (
                f"body {i}: inertia scaled by "
                f"{inertia_after[i] / np.where(inertia_before[i] != 0, inertia_before[i], 1)} "
                f"but mass scaled by {s}"
            )

    def test_randomize_positions(self, sim_with_world):
        sim_with_world.add_object("cube", shape="box", position=[0, 0, 0.1])
        result = sim_with_world.randomize(randomize_positions=True, seed=42)
        assert result["status"] == "success"

    def test_randomize_colors_recolors_unnamed_geoms_and_reports_true_count(self, sim_with_robot):
        """Recoloring must touch every non-ground geom - including the UNNAMED
        mesh geoms a real robot body is built from - and the success message
        must report the count it actually changed, not the total geom count.

        Pre-fix the loop skipped any geom with a falsy (unnamed) name, so a
        robot kept its colours while the call reported every geom randomized.
        """
        m = sim_with_robot._world._model
        before = m.geom_rgba.copy()

        result = sim_with_robot.randomize(randomize_colors=True, randomize_lighting=False, seed=42)
        assert result["status"] == "success"
        after = m.geom_rgba.copy()

        changed = [i for i in range(m.ngeom) if not np.allclose(before[i, :3], after[i, :3])]
        ground = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "ground")
        # the robot's geoms are unnamed; every non-ground geom must recolor
        assert ground not in changed
        assert len(changed) == m.ngeom - 1
        # the reported count is the true number changed, not ngeom
        assert f"Colors: {len(changed)} geoms" in result["content"][0]["text"]

    def test_randomize_colors_propagates_to_materials(self, sim, material_scene_path):
        """A material-backed geom draws its colour from the material, which
        overrides geom_rgba in the renderer, so randomization must recolor the
        material too or the change is visually inert.

        The scene's geom is both unnamed and material-backed, so pre-fix it was
        skipped entirely and no material colour ever changed.
        """
        sim.create_world(ground_plane=False)
        assert sim.add_robot("m", urdf_path=material_scene_path)["status"] == "success"
        m = sim._world._model
        assert m.nmat >= 1

        mat_before = m.mat_rgba.copy()
        result = sim.randomize(randomize_colors=True, randomize_lighting=False, seed=1)
        assert result["status"] == "success"
        mat_after = m.mat_rgba.copy()

        changed_mats = [i for i in range(m.nmat) if not np.allclose(mat_before[i, :3], mat_after[i, :3])]
        assert changed_mats, "material colours must be randomized so the recolor is visible"

    def test_randomize_no_world(self, sim):
        result = sim.randomize()
        assert result["status"] == "error"


# Introspection


class TestIntrospection:
    def test_get_features_with_robot(self, sim_with_robot):
        result = sim_with_robot.get_features()
        assert result["status"] == "success"
        json_content = result["content"][1]
        data = json_content.get("json") or json.loads(json_content.get("text", "{}"))
        features = data["features"]
        assert features["n_joints"] > 0
        assert features["n_actuators"] > 0
        assert "arm1" in features["robots"]

    def test_get_features_no_world(self, sim):
        result = sim.get_features()
        assert result["status"] == "error"


# URDF Registry


class TestURDFRegistry:
    def test_list_urdfs(self, sim):
        result = sim.list_urdfs()
        assert result["status"] == "success"

    def test_register_urdf(self, sim, robot_xml_path):
        result = sim.register_urdf("test_arm", robot_xml_path)
        assert result["status"] == "success"
        assert "test_arm" in result["content"][0]["text"]


# Policy Execution


class TestPolicyExecution:
    """Test run_policy and eval_policy through the Simulation class."""

    def test_run_policy_mock(self, sim_with_robot):
        result = sim_with_robot.run_policy(
            "arm1",
            policy_provider="mock",
            instruction="wave",
            duration=0.1,
            fast_mode=True,
        )
        assert result["status"] == "success"
        assert "Policy complete" in result["content"][0]["text"]
        assert sim_with_robot._world.sim_time > 0

    def test_run_policy_no_world(self, sim):
        result = sim.run_policy("arm1", policy_provider="mock")
        assert result["status"] == "error"

    def test_run_policy_invalid_robot(self, sim_with_world):
        result = sim_with_world.run_policy("nonexistent", policy_provider="mock")
        assert result["status"] == "error"

    def test_eval_policy_mock(self, sim_with_robot):
        result = sim_with_robot.eval_policy(
            robot_name="arm1",
            policy_provider="mock",
            instruction="reach",
            n_episodes=2,
            max_steps=10,
        )
        assert result["status"] == "success"
        # eval_policy returns json in the second content item
        json_content = result["content"][1]
        data = json_content.get("json") or json.loads(json_content.get("text", "{}"))
        assert data["n_episodes"] == 2
        assert "success_rate" in data

    def test_eval_policy_no_world(self, sim):
        result = sim.eval_policy()
        assert result["status"] == "error"

    def test_start_policy_and_stop(self, sim_with_robot):
        result = sim_with_robot.start_policy(
            "arm1",
            policy_provider="mock",
            duration=0.2,
            fast_mode=True,
        )
        assert result["status"] == "success"
        assert "started" in result["content"][0]["text"]

        # Stop it
        result = sim_with_robot.stop_policy("arm1")
        assert result["status"] == "success"

    def test_start_policy_no_world(self, sim):
        result = sim.start_policy("arm1")
        assert result["status"] == "error"

    def test_start_policy_invalid_robot(self, sim_with_world):
        result = sim_with_world.start_policy("ghost")
        assert result["status"] == "error"

    def test_describe_advertises_background_policy_lifecycle(self, sim_with_robot):
        """describe() advertises the whole start/stop/list background-policy
        lifecycle, not just start_policy.

        The MuJoCo backend overrides start_policy to run in a background thread
        (non-blocking), unlike the base engine's synchronous passthrough, and
        provides stop_policy + list_policies_running to manage it. describe()
        already advertised start_policy (inherited from the base surface), but
        omitted its lifecycle siblings -- so an agent that discovered
        start_policy here and launched a rollout could not learn how to stop it
        or inspect what is running without guessing the names, a resource-leak
        trap. Both are first-class actions in the tool spec + dispatcher and
        belong on the discovery surface alongside start_policy.
        """
        methods = sim_with_robot.describe()["methods"]
        for name in ("start_policy", "stop_policy", "list_policies_running"):
            assert name in methods, f"describe() omits policy-lifecycle method {name!r}"
        # Advertised signatures name the real parameters / return shape so a
        # caller can invoke them without reading the source.
        assert "robot_name" in methods["stop_policy"]
        assert "-> dict" in methods["list_policies_running"]

        # The advertisement is only useful if the methods it names are real and
        # invocable: list before start reports none, stop is idempotent.
        listed = sim_with_robot.list_policies_running()
        assert listed["status"] == "success"
        assert "No policies running" in listed["content"][0]["text"]
        idempotent = sim_with_robot.stop_policy("arm1")
        assert idempotent["status"] == "success"
        assert "Was not running" in idempotent["content"][0]["text"]

    def test_describe_advertises_multi_robot_rollout_family(self, sim_with_robot):
        """describe() advertises run_multi_policy and the per-robot action/joint
        introspection a multi-policy caller needs to wire each robot.

        describe() advertises run_policy (drive ONE robot with a created policy)
        and the background start/stop/list lifecycle, but omitted
        run_multi_policy -- the facade that drives SEVERAL robots, each with its
        own Policy, in one synchronized loop (the correct path for bimanual /
        multi-agent data collection). It also omitted the two per-robot
        introspection primitives a caller uses to build that {robot_name:
        Policy} map: robot_action_keys (the actuator short-names a policy must
        emit -- NOT always the joint names) and robot_joint_names (the ordered
        observation.state vector). An agent enumerating the sim from describe()
        alone could drive one robot but had to guess how to drive many, or key a
        policy by joint name and watch tendon/mimic DOFs silently no-op.
        """
        methods = sim_with_robot.describe()["methods"]
        for name in ("run_multi_policy", "robot_action_keys", "robot_joint_names"):
            assert name in methods, f"describe() omits multi-robot method {name!r}"
        # Advertised signatures name the real parameters / return shape so a
        # caller can invoke them without reading the source.
        assert "policies" in methods["run_multi_policy"]
        assert "-> dict" in methods["run_multi_policy"]
        assert "-> list[str]" in methods["robot_action_keys"]
        assert "-> list[str]" in methods["robot_joint_names"]
        # The advertisement names actuator-vs-joint distinction that makes
        # robot_action_keys the correct key source over robot_joint_names.
        assert "send_action" in methods["robot_action_keys"]

        # The advertisement is only useful if the methods it names are real and
        # return the exact per-robot lists a policy is keyed on.
        joints = sim_with_robot.robot_joint_names("arm1")
        keys = sim_with_robot.robot_action_keys("arm1")
        assert isinstance(joints, list) and joints, "robot_joint_names('arm1') is empty"
        assert isinstance(keys, list) and keys, "robot_action_keys('arm1') is empty"
        # Unknown robots return an empty list rather than raising.
        assert sim_with_robot.robot_joint_names("ghost") == []
        assert sim_with_robot.robot_action_keys("ghost") == []


# Action Dispatch


class TestActionDispatch:
    """Test _dispatch_action routes correctly via tool_spec actions."""

    def test_dispatch_create_world(self, sim):
        result = sim._dispatch_action("create_world", {"action": "create_world"})
        assert result["status"] == "success"

    def test_dispatch_get_state(self, sim_with_world):
        result = sim_with_world._dispatch_action("get_state", {"action": "get_state"})
        assert result["status"] == "success"

    def test_dispatch_step(self, sim_with_world):
        result = sim_with_world._dispatch_action("step", {"action": "step", "n_steps": 10})
        assert result["status"] == "success"

    def test_dispatch_add_object(self, sim_with_world):
        result = sim_with_world._dispatch_action(
            "add_object",
            {"action": "add_object", "name": "box1", "shape": "box", "position": [0, 0, 0.1]},
        )
        assert result["status"] == "success"

    def test_dispatch_unknown_action(self, sim):
        result = sim._dispatch_action("nonexistent", {"action": "nonexistent"})
        assert result["status"] == "error"
        assert "Unknown action" in result["content"][0]["text"]

    def test_dispatch_private_action_blocked(self, sim):
        """Actions starting with _ are blocked (security)."""
        result = sim._dispatch_action("_compile_world", {"action": "_compile_world"})
        assert result["status"] == "error"

    def test_dispatch_list_urdfs_alias(self, sim):
        result = sim._dispatch_action("list_urdfs", {"action": "list_urdfs"})
        assert result["status"] == "success"

    def test_dispatch_set_gravity(self, sim_with_world):
        result = sim_with_world._dispatch_action("set_gravity", {"action": "set_gravity", "gravity": [0, 0, -5.0]})
        assert result["status"] == "success"


# Context Manager


class TestContextManager:
    def test_context_manager_cleanup(self):
        with Simulation(tool_name="ctx_test", mesh=False) as sim:
            sim.create_world()
            assert sim._world is not None
        # After exit, world should be cleaned up
        assert sim._world is None


# Tool Spec


class TestToolSpec:
    def test_tool_name(self, sim):
        assert sim.tool_name == "test_sim"

    def test_tool_type(self, sim):
        assert sim.tool_type == "simulation"

    def test_tool_spec_schema(self, sim):
        spec = sim.tool_spec
        assert spec["name"] == "test_sim"
        assert "inputSchema" in spec
        assert "json" in spec["inputSchema"]
        schema = spec["inputSchema"]["json"]
        assert "properties" in schema
        assert "action" in schema["properties"]


# Viewer (headless safe)


class TestViewer:
    def test_open_viewer_no_world(self, sim):
        result = sim.open_viewer()
        assert result["status"] == "error"

    def test_close_viewer_noop(self, sim):
        result = sim.close_viewer()
        assert result["status"] == "success"

    def test_open_viewer_reports_when_mujoco_viewer_unavailable(self, sim_with_world, monkeypatch):
        """With a world but no importable ``mujoco.viewer`` backend, open_viewer
        returns an actionable error instead of raising."""

        monkeypatch.setattr(backend_mod, "_mujoco_viewer", None)
        result = sim_with_world.open_viewer()
        assert result["status"] == "error"
        assert "viewer not available" in result["content"][0]["text"].lower()

    def test_open_viewer_launches_passive_and_close_releases_handle(self, sim_with_world, monkeypatch):
        """open_viewer launches a passive viewer on the live model/data and keeps
        the handle; close_viewer then closes that handle and clears it."""

        launched: dict[str, object] = {}

        class _FakeHandle:
            def __init__(self) -> None:
                self.closed = False

            def close(self) -> None:
                self.closed = True

        class _FakeViewer:
            @staticmethod
            def launch_passive(model, data):
                launched["model"] = model
                launched["data"] = data
                return _FakeHandle()

        monkeypatch.setattr(backend_mod, "_mujoco_viewer", _FakeViewer)

        result = sim_with_world.open_viewer()
        assert result["status"] == "success"
        assert "viewer opened" in result["content"][0]["text"].lower()
        # Launched against the live world's model/data, not copies.
        assert launched["model"] is sim_with_world._world._model
        assert launched["data"] is sim_with_world._world._data

        handle = sim_with_world._viewer_handle
        assert handle is not None

        close_result = sim_with_world.close_viewer()
        assert close_result["status"] == "success"
        assert handle.closed is True
        assert sim_with_world._viewer_handle is None

    def test_open_viewer_does_not_relaunch_when_already_open(self, sim_with_world, monkeypatch):
        """A second open_viewer call reports the existing viewer rather than
        launching a second one (which would leak the first handle)."""

        launches = {"n": 0}

        class _FakeViewer:
            @staticmethod
            def launch_passive(model, data):
                launches["n"] += 1
                return object()

        monkeypatch.setattr(backend_mod, "_mujoco_viewer", _FakeViewer)

        assert sim_with_world.open_viewer()["status"] == "success"
        second = sim_with_world.open_viewer()
        assert second["status"] == "success"
        assert "already open" in second["content"][0]["text"].lower()
        assert launches["n"] == 1

    def test_open_viewer_surfaces_launch_failure_without_retaining_handle(self, sim_with_world, monkeypatch):
        """When launch_passive raises (e.g. no display), open_viewer returns a
        structured error and leaves no half-open handle behind."""

        class _FakeViewer:
            @staticmethod
            def launch_passive(model, data):
                raise RuntimeError("no display available")

        monkeypatch.setattr(backend_mod, "_mujoco_viewer", _FakeViewer)

        result = sim_with_world.open_viewer()
        assert result["status"] == "error"
        assert "viewer failed" in result["content"][0]["text"].lower()
        assert sim_with_world._viewer_handle is None

    def test_close_viewer_swallows_handle_close_error(self, sim_with_world, monkeypatch):
        """close_viewer must not propagate an exception raised by handle.close();
        the handle is cleared regardless so the sim is never wedged open."""

        class _BadHandle:
            def close(self) -> None:
                raise RuntimeError("boom")

        class _FakeViewer:
            @staticmethod
            def launch_passive(model, data):
                return _BadHandle()

        monkeypatch.setattr(backend_mod, "_mujoco_viewer", _FakeViewer)

        assert sim_with_world.open_viewer()["status"] == "success"
        result = sim_with_world.close_viewer()
        assert result["status"] == "success"
        assert sim_with_world._viewer_handle is None


# Error Paths


class TestErrorPaths:
    """Test that error conditions return proper error dicts, not exceptions."""

    def test_get_state_no_world(self, sim):
        result = sim.get_state()
        assert result["status"] == "error"

    def test_step_no_world(self, sim):
        result = sim.step()
        assert result["status"] == "error"

    def test_reset_no_world(self, sim):
        result = sim.reset()
        assert result["status"] == "error"

    def test_add_object_no_world(self, sim):
        result = sim.add_object("x", shape="box")
        assert result["status"] == "error"

    def test_move_object_no_world(self, sim):
        result = sim.move_object("x", position=[0, 0, 0])
        assert result["status"] == "error"

    def test_list_objects_no_world(self, sim):
        result = sim.list_objects()
        assert result["status"] == "error"

    def test_list_robots_no_world(self, sim):
        # ABC returns empty list when no world
        assert sim.list_robots() == []
        # Action-tool surface returns a friendly error dict
        result = sim.list_robots_info()
        assert result["status"] == "error"

    def test_render_no_world(self, sim):
        result = sim.render()
        assert result["status"] == "error"

    def test_render_depth_no_world(self, sim):
        result = sim.render_depth()
        assert result["status"] == "error"

    def test_get_contacts_no_world(self, sim):
        result = sim.get_contacts()
        assert result["status"] == "error"

    def test_get_features_no_world(self, sim):
        result = sim.get_features()
        assert result["status"] == "error"

    def test_set_gravity_no_world(self, sim):
        result = sim.set_gravity([0, 0, -5])
        assert result["status"] == "error"

    def test_set_timestep_no_world(self, sim):
        result = sim.set_timestep(0.001)
        assert result["status"] == "error"

    def test_get_robot_state_no_world(self, sim):
        result = sim.get_robot_state("x")
        assert result["status"] == "error"

    def test_randomize_no_world(self, sim):
        result = sim.randomize()
        assert result["status"] == "error"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# Thread-safety regression


class TestRendererThreadSafety:
    """Regression for SIGSEGV in cgl.free() when renderers cached across threads.

    Bug: renderers were kept in a plain dict on Simulation. Worker threads
    created renderers via `run_policy`, cached them on the instance, and
    `cleanup()` on the main thread then called `renderer.close()` →
    `cgl.free()` on the wrong thread → SIGSEGV.

    Fix: renderers are thread-local; each thread owns its cache.
    """

    def test_renderer_cache_is_thread_local(self, sim_with_world):
        """Different threads must see different renderer dicts."""
        import threading

        sim_with_world.add_object("blk", shape="box", position=[0, 0, 0.1])
        sim_with_world.add_camera("cam", position=[0.3, -0.3, 0.3], target=[0, 0, 0])
        sim_with_world.step(n_steps=1)

        main_renderer = sim_with_world._get_renderer(64, 64)
        if main_renderer is None:
            import pytest

            pytest.skip("rendering unavailable in this environment")
        main_id = id(main_renderer)

        worker_id_box = {}

        def worker():
            r = sim_with_world._get_renderer(64, 64)
            worker_id_box["id"] = id(r) if r is not None else None

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert worker_id_box["id"] is not None, "worker got None renderer"
        assert worker_id_box["id"] != main_id, (
            "worker thread should get its OWN renderer instance, not the "
            "main-thread one - otherwise CGL context mismatch on cleanup."
        )

    def test_cleanup_after_policy_thread_no_segfault(self, sim_with_robot):
        """start_policy+stop+cleanup must not SIGSEGV (was fatal pre-fix)."""
        r = sim_with_robot.start_policy("arm1", policy_provider="mock", duration=0.2, fast_mode=True)
        assert r["status"] == "success"
        sim_with_robot.stop_policy("arm1")
        # Wait for the policy thread to drain so its renderer ref is released.
        future = sim_with_robot._policy_threads.get("arm1")
        if future is not None:
            future.result(timeout=5.0)
        # cleanup() should succeed - pre-fix this segfaulted when the
        # worker-thread renderer was closed on the main thread.
        sim_with_robot.cleanup()


# XML round-trip state poisoning regression


@requires_gl
class TestMjSaveLastXMLGlobalState:
    """Regression: MuJoCo's ``mj_saveLastXML`` is a global-state function
    that always emits the *last loaded* model, ignoring its ``model`` arg.
    Any renderer creation or ancillary model load would poison subsequent
    inject/eject XML round-trips, causing silent "Body not found" warnings
    and skipped ejections.
    """

    def test_remove_object_after_render(self, sim_with_robot):
        """After rendering, remove_object must still find and eject the body."""
        sim_with_robot.add_object("cube", shape="box", size=[0.025, 0.025, 0.025], position=[0.25, 0, 0.05])
        sim_with_robot.add_camera("cam", position=[0.3, -0.3, 0.3], target=[0, 0, 0])
        # Render poisons mj_saveLastXML (loads an ancillary model internally).
        obs = sim_with_robot.get_observation("arm1")
        assert "cam" in obs, "get_observation should include the 'cam' camera frame"

        # This used to silently log "Body 'cube' not found in MJCF XML" and
        # leave the body in the scene.
        result = sim_with_robot.remove_object("cube")
        assert result["status"] == "success"

        # Verify the body is really gone from the live model
        import mujoco as mj

        names = [
            mj.mj_id2name(sim_with_robot._world._model, mj.mjtObj.mjOBJ_BODY, i)
            for i in range(sim_with_robot._world._model.nbody)
        ]
        assert "cube" not in names, "cube should be ejected from the model"

    def test_remove_object_after_run_policy(self, sim_with_robot):
        """After a policy runs (creates renderers + observations), eject still works."""
        sim_with_robot.add_object("cube", shape="box", size=[0.025, 0.025, 0.025], position=[0.25, 0, 0.05])
        sim_with_robot.add_camera("cam", position=[0.3, -0.3, 0.3], target=[0, 0, 0])
        r = sim_with_robot.run_policy("arm1", policy_provider="mock", duration=0.1, fast_mode=True)
        assert r["status"] == "success"

        result = sim_with_robot.remove_object("cube")
        assert result["status"] == "success"

        import mujoco as mj

        names = [
            mj.mj_id2name(sim_with_robot._world._model, mj.mjtObj.mjOBJ_BODY, i)
            for i in range(sim_with_robot._world._model.nbody)
        ]
        assert "cube" not in names


# Multi-robot same-config injection


class TestMultipleSameConfigRobots:
    """Regression: adding multiple robots with the same ``data_config``
    used to fail with "XML Error: repeated default class name" / "repeated
    name 'base' in body".

    Fix: robot bodies/joints/actuators/sensors are namespaced (prefixed
    with the robot instance name) during MJCF injection; <default> and
    <asset> blocks are deduped by name/class. The public API still returns
    short joint names so policies see a config-level schema.
    """

    def _robot_xml(self, tmp_path):
        """Write a tiny 1-DOF arm XML to a temp file."""
        xml = """<mujoco>
  <default>
    <default class="arm">
      <geom rgba="0.8 0.5 0.2 1"/>
    </default>
  </default>
  <worldbody>
    <body name="base">
      <geom type="cylinder" size="0.05 0.05" class="arm"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="shoulder" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" size="0.02 0.1" class="arm"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder" joint="shoulder" kp="50"/>
  </actuator>
</mujoco>
"""
        path = tmp_path / "arm.xml"
        path.write_text(xml)
        return str(path)

    def test_three_same_config_robots(self, sim, tmp_path):
        """Three robots using the same XML should inject without error."""
        xml_path = self._robot_xml(tmp_path)
        sim.create_world()

        for i in range(3):
            r = sim.add_robot(f"arm{i}", urdf_path=xml_path, position=[i * 0.5 - 0.5, 0, 0])
            assert r["status"] == "success", f"add_robot arm{i} failed: {r}"

        assert sim.list_robots() == ["arm0", "arm1", "arm2"]

        # Each robot should have its own joint_ids (no sharing).
        ids = [set(sim._world.robots[f"arm{i}"].joint_ids) for i in range(3)]
        assert all(ids[i] for i in range(3)), f"robots with empty joint_ids: {ids}"
        assert ids[0].isdisjoint(ids[1]) and ids[1].isdisjoint(ids[2]), f"robots share joint IDs: {ids}"

    def test_per_robot_action_isolation(self, sim, tmp_path):
        """send_action must route to the target robot's actuators only."""
        xml_path = self._robot_xml(tmp_path)
        sim.create_world()
        for i in range(3):
            sim.add_robot(f"arm{i}", urdf_path=xml_path, position=[i * 0.5 - 0.5, 0, 0])

        # Action on arm0 should set arm0's ctrl, not arm1 or arm2.
        sim.send_action({"shoulder": 0.7}, robot_name="arm0")

        import numpy as np

        ctrl = np.array(sim._world._data.ctrl)
        r0 = sim._world.robots["arm0"]
        r1 = sim._world.robots["arm1"]
        r2 = sim._world.robots["arm2"]

        assert np.isclose(ctrl[r0.actuator_ids[0]], 0.7)
        assert np.isclose(ctrl[r1.actuator_ids[0]], 0.0)
        assert np.isclose(ctrl[r2.actuator_ids[0]], 0.0)

    def test_observation_returns_short_keys(self, sim, tmp_path):
        """get_observation should return short joint names (e.g. 'shoulder'),
        not the namespaced MuJoCo names ('arm0/shoulder')."""
        xml_path = self._robot_xml(tmp_path)
        sim.create_world()
        for i in range(2):
            sim.add_robot(f"arm{i}", urdf_path=xml_path, position=[i * 0.5 - 0.25, 0, 0])

        obs0 = sim.get_observation("arm0")
        obs1 = sim.get_observation("arm1")

        assert "shoulder" in obs0
        assert "shoulder" in obs1
        # No namespaced keys leak into the observation.
        assert "arm0/shoulder" not in obs0
        assert "arm1/shoulder" not in obs1


# Physics/recording name resolution after namespacing


class TestPhysicsNameResolution:
    """Physics methods (jacobian, body_state, forward_kinematics) accept
    raw body/joint names. After PR #85 multi-robot namespacing, they now
    fall back to namespaced lookups so single-robot code keeps working
    without churn.
    """

    def test_get_body_state_accepts_short_name_single_robot(self, sim_with_robot):
        """In a single-robot scene, ``gripper`` should resolve via the
        namespace fallback (actual body is ``arm1/gripper``)."""
        # ROBOT_XML has bodies: base, link1, link2. After namespacing the
        # real names are arm1/base etc. The short name must resolve.
        r = sim_with_robot._dispatch_action("get_body_state", {"body_name": "link1"})
        assert r["status"] == "success", r

    def test_get_body_state_rejects_unknown(self, sim_with_robot):
        r = sim_with_robot._dispatch_action("get_body_state", {"body_name": "nope"})
        assert r["status"] == "error"


class TestRecordingSafeCameraNames:
    """LeRobot feature names can't contain ``/``. When a robot namespace
    leaks into the camera name (e.g. ``arm0/wrist_cam``), the dataset
    recorder must sanitize the separator before handing off to LeRobot.
    """

    def test_start_recording_sanitizes_namespaced_cameras(self, sim_with_robot, tmp_path):
        pytest.importorskip("lerobot")
        # The sim_with_robot fixture's robot XML injects a camera; for
        # so101 it becomes ``arm1/wrist_cam``. Without sanitization,
        # LeRobot raises: "Feature names should not contain '/'".
        root = str(tmp_path / "ds")
        r = sim_with_robot._dispatch_action(
            "start_recording",
            {"repo_id": "local/test-ns", "root": root},
        )
        assert r["status"] == "success", r
        # cleanup - don't leave a dangling recorder on the fixture
        sim_with_robot._dispatch_action("stop_recording", {})


class TestSimulationOutputIsAscii:
    """Tool output must be ASCII-only (AGENTS.md: no emojis in code/logs/errors).

    Agent-tool ``text`` payloads are surfaced to the model and into logs;
    decorative emoji (robot/world glyphs, status dots, bullets) break the
    project's ASCII-only contract and render inconsistently across terminals.
    These tests drive the common lifecycle, scene-mutation, physics and
    policy-listing methods and assert every returned ``text`` is ``isascii()``.
    """

    @staticmethod
    def _texts(result):
        return "".join(item.get("text", "") for item in result.get("content", []))

    def test_create_world_output_is_ascii(self, sim):
        assert self._texts(sim.create_world()).isascii()

    def test_get_state_output_is_ascii(self, sim_with_world):
        assert self._texts(sim_with_world.get_state()).isascii()

    def test_reset_output_is_ascii(self, sim_with_world):
        assert self._texts(sim_with_world.reset()).isascii()

    def test_set_gravity_and_timestep_output_is_ascii(self, sim_with_world):
        assert self._texts(sim_with_world.set_gravity([0, 0, -3.0])).isascii()
        # Large timestep takes the warning branch - it too must be ASCII.
        large = sim_with_world.set_timestep(0.5)
        text = self._texts(large)
        assert text.isascii()
        assert "Warning:" in text and "unusually" in text

    def test_step_output_is_ascii(self, sim_with_world):
        assert self._texts(sim_with_world.step(0)).isascii()
        assert self._texts(sim_with_world.step(3)).isascii()

    def test_destroy_output_is_ascii(self, sim_with_world):
        assert self._texts(sim_with_world.destroy()).isascii()

    def test_add_robot_and_listing_output_is_ascii(self, sim_with_robot):
        # add_robot text was emitted by the fixture; re-exercise the
        # introspection surfaces that format robot/joint/camera summaries.
        assert self._texts(sim_with_robot.list_robots_info()).isascii()
        assert self._texts(sim_with_robot.get_robot_state("arm1")).isascii()
        assert self._texts(sim_with_robot.get_features()).isascii()

    def test_object_lifecycle_output_is_ascii(self, sim_with_world):
        added = sim_with_world.add_object("box1", shape="box", position=[0.3, 0, 0.1])
        assert self._texts(added).isascii()
        assert self._texts(sim_with_world.list_objects()).isascii()
        assert self._texts(sim_with_world.move_object("box1", position=[0.4, 0, 0.1])).isascii()
        assert self._texts(sim_with_world.remove_object("box1")).isascii()

    def test_list_policies_running_output_is_ascii(self, sim_with_robot):
        assert self._texts(sim_with_robot.list_policies_running()).isascii()


class TestUnknownModelSuggestions:
    """The 'no model found' error names close registry matches (difflib).

    When a caller asks for a robot model that does not exist, the sim should
    not dead-end with a bare "use list_urdfs"; it suggests the closest known
    registry keys so a typo can usually be fixed in-place without a discovery
    round-trip. This pins that ergonomics contract.
    """

    def test_suggests_close_registry_matches_for_typo(self):
        # 'so10x' is one edit away from the so100/so101 family.
        msg = Simulation._unknown_model_msg("so10x")
        assert "No model found for 'so10x'." in msg
        assert "Did you mean:" in msg
        assert "so101" in msg or "so100" in msg
        assert "list_urdfs" in msg

    def test_caps_suggestions_at_three(self):
        # difflib is asked for at most 3 matches; the rendered list must not
        # exceed that even when many registry keys are near.
        msg = Simulation._unknown_model_msg("panda")
        if "Did you mean:" in msg:
            suggestions = msg.split("Did you mean:")[1].split("?")[0]
            assert len([s for s in suggestions.split(",") if s.strip()]) <= 3

    def test_no_suggestions_when_nothing_is_close(self):
        # A string with no near neighbours omits the 'Did you mean' clause but
        # still points at the discovery action.
        msg = Simulation._unknown_model_msg("zzqqxx0000nope")
        assert "No model found for 'zzqqxx0000nope'." in msg
        assert "Did you mean:" not in msg
        assert "list_urdfs" in msg

    def test_falls_back_gracefully_when_registry_unavailable(self, monkeypatch):
        # If the registry lookup raises, suggestions are best-effort: the
        # message degrades to the bare form rather than propagating the error.
        import strands_robots.registry as registry

        def _boom():
            raise RuntimeError("registry exploded")

        monkeypatch.setattr(registry, "list_robots", _boom)
        msg = Simulation._unknown_model_msg("anything")
        assert "No model found for 'anything'." in msg
        assert "Did you mean:" not in msg
        assert "list_urdfs" in msg

    def test_message_is_ascii_only(self):
        # Error strings must stay ASCII (no emojis / smart punctuation).
        assert Simulation._unknown_model_msg("so10x").isascii()

    def test_add_robot_surfaces_suggestions_for_unknown_data_config(self, sim_with_world):
        # End-to-end: an unknown data_config flows the suggestion message out
        # through add_robot's error result.
        result = sim_with_world.add_robot(name="arm", data_config="so10x")
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "No model found for 'so10x'." in text
        assert "list_urdfs" in text
