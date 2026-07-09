"""Regression tests: start_recording preserves a floating base's orientation +
angular velocity in the recorded observation.state.

``get_observation`` surfaces the full floating-base kinematics -- ``base_pos``
(world x,y,z incl. height), ``base_quat`` (orientation, w,x,y,z), ``base_lin_vel``
(m/s) and ``base_ang_vel`` (rad/s) -- for a floating-base robot (a humanoid's
named ``floating_base_joint`` or a mobile base's unnamed ``<freejoint>``). Those
base signals used to be dropped from the LeRobotDataset ``observation.state``
schema (derived from scalar joint names only), leaving a locomotion /
velocity-tracking / whole-body-control policy trained on the dataset base-blind.
They are now written as per-component scalar columns (``base_pos.x``..
``base_ang_vel.z``); a fixed-base arm is unchanged (no base columns, no schema
growth).
"""

import tempfile

import numpy as np
import pytest

from strands_robots.simulation.mujoco.simulation import Simulation

# Humanoid-style floating base: a NAMED free root joint (enumerated in
# robot.joint_names, like a real ``floating_base_joint``) + one actuated hinge.
NAMED_BASE_XML = """
<mujoco model="test_named_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="torso" pos="0 0 0.5">
      <joint name="floating_base_joint" type="free"/>
      <geom type="box" size="0.1 0.1 0.2" rgba="0.3 0.3 0.8 1"/>
      <body name="thigh" pos="0 0 -0.2">
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

# Mobile base: an UNNAMED free joint (not in robot.joint_names) + an actuated
# hinge. Exercises the kinematic-tree fallback detection (LeKiwi-style).
UNNAMED_BASE_XML = """
<mujoco model="test_unnamed_base">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <geom name="ground" type="plane" size="5 5 0.01" rgba="0.9 0.9 0.9 1"/>
    <body name="base_plate" pos="0 0 0.1">
      <freejoint/>
      <geom type="box" size="0.15 0.15 0.03" rgba="0.3 0.3 0.8 1"/>
      <body name="arm" pos="0 0 0.05">
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" rgba="0.8 0.3 0.3 1"/>
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder_act" joint="shoulder"/>
  </actuator>
</mujoco>
"""

# Fixed-base arm: NO free joint anywhere. Must gain NO base columns (guards
# against a false positive that would grow the schema for a plain arm).
FIXED_ARM_XML = """
<mujoco model="test_fixed_arm">
  <compiler angle="radian" autolimits="true"/>
  <worldbody>
    <light name="main" pos="0 0 3" dir="0 0 -1"/>
    <body name="link" pos="0 0 0.1">
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2"/>
      <joint name="j0" type="hinge" axis="0 0 1"/>
    </body>
  </worldbody>
  <actuator>
    <position name="j0_act" joint="j0" kp="10"/>
  </actuator>
</mujoco>
"""

_BASE_COLS = [
    "base_pos.x",
    "base_pos.y",
    "base_pos.z",
    "base_quat.w",
    "base_quat.x",
    "base_quat.y",
    "base_quat.z",
    "base_lin_vel.x",
    "base_lin_vel.y",
    "base_lin_vel.z",
    "base_ang_vel.x",
    "base_ang_vel.y",
    "base_ang_vel.z",
]


def _write(xml: str) -> str:
    import os

    fd, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd, "w") as f:
        f.write(xml)
    return path


@pytest.fixture
def sim():
    pytest.importorskip("lerobot")  # start_recording produces a LeRobotDataset
    s = Simulation(tool_name="test_fb_state", mesh=False)
    s.create_world(ground_plane=False)
    yield s
    try:
        s.cleanup()
    except Exception:
        # Best-effort teardown: cleanup failures must not mask the test result.
        pass


def _start(sim, name, xml, cameras=None):
    sim.add_robot(name, urdf_path=_write(xml))
    root = tempfile.mkdtemp(prefix=f"fbstate_{name}_")
    res = sim.start_recording(repo_id=f"local/{name}_fbstate", task="t", fps=20, root=root, cameras=cameras or [])
    return res, root


def _state_names(sim):
    rec = sim._world._backend_state["dataset_recorder"]
    feat = rec.dataset.features.get("observation.state", {})
    names = feat.get("names", []) if isinstance(feat, dict) else getattr(feat, "names", [])
    shape = feat.get("shape") if isinstance(feat, dict) else getattr(feat, "shape", None)
    return list(names), shape


def _free_joint_addrs(sim):
    import mujoco as mj

    m = sim._world._model
    fid = next(j for j in range(m.njnt) if m.jnt_type[j] == mj.mjtJoint.mjJNT_FREE)
    return int(m.jnt_qposadr[fid]), int(m.jnt_dofadr[fid])


def test_named_floating_base_preserved_in_schema(sim):
    """A humanoid with a NAMED floating_base_joint: observation.state gains the
    seven per-component base columns (was scalar joint positions only)."""
    res, _ = _start(sim, "humanoid", NAMED_BASE_XML)
    assert res["status"] == "success"
    names, shape = _state_names(sim)
    for col in _BASE_COLS:
        assert col in names, f"{col} missing from recorded observation.state schema"
    # shape length must match the flat per-element name count.
    assert tuple(shape)[0] == len(names)
    sim.stop_recording()


def test_unnamed_mobile_base_preserved_in_schema(sim):
    """A mobile base on an UNNAMED freejoint (detected via the kinematic-tree
    fallback) also gains the base columns."""
    res, _ = _start(sim, "mob", UNNAMED_BASE_XML)
    assert res["status"] == "success"
    names, shape = _state_names(sim)
    for col in _BASE_COLS:
        assert col in names, f"{col} missing for unnamed mobile base"
    assert tuple(shape)[0] == len(names)
    sim.stop_recording()


def test_fixed_arm_has_no_base_columns(sim):
    """A fixed-base arm has no floating base -> no base columns (no schema
    growth / false positive)."""
    res, _ = _start(sim, "arm", FIXED_ARM_XML)
    assert res["status"] == "success"
    names, _shape = _state_names(sim)
    assert not any(n.startswith(("base_pos", "base_quat", "base_lin_vel", "base_ang_vel")) for n in names), (
        "fixed-base arm must NOT gain floating-base columns"
    )
    sim.stop_recording()


def test_recorded_base_values_round_trip(sim):
    """End-to-end: a known base orientation + angular velocity is written to the
    dataset and read back byte-for-byte after reopen (the base state is no longer
    silently dropped)."""
    import mujoco as mj
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    res, root = _start(sim, "humanoid", NAMED_BASE_XML)
    assert res["status"] == "success"
    rec = sim._world._backend_state["dataset_recorder"]
    m, d = sim._world._model, sim._world._data
    qadr, vadr = _free_joint_addrs(sim)
    known_pos = [0.11, -0.22, 0.83]  # world x, y, HEIGHT
    known_quat = [0.7071068, 0.0, 0.7071068, 0.0]  # +90deg about Y
    known_linvel = [0.41, -0.52, 0.63]
    known_angvel = [0.13, 0.24, 0.37]
    for _ in range(3):
        d.qpos[qadr : qadr + 3] = known_pos
        d.qpos[qadr + 3 : qadr + 7] = known_quat
        d.qvel[vadr : vadr + 3] = known_linvel
        d.qvel[vadr + 3 : vadr + 6] = known_angvel
        mj.mj_forward(m, d)
        obs = sim.get_observation("humanoid", skip_images=True)
        act = {k: 0.0 for k in sim.robot_action_keys("humanoid")}
        rec.add_frame(observation=obs, action=act, task="t")
    rec.save_episode()
    rec.finalize()
    sim._world._backend_state["recording"] = False

    ds = LeRobotDataset("local/humanoid_fbstate", root=root)
    names = ds.features["observation.state"]["names"]
    idx = {n: i for i, n in enumerate(names)}
    state = np.asarray(ds[0]["observation.state"], dtype=np.float32)
    got_pos = [float(state[idx[f"base_pos.{c}"]]) for c in "xyz"]
    got_quat = [float(state[idx[f"base_quat.{c}"]]) for c in "wxyz"]
    got_linvel = [float(state[idx[f"base_lin_vel.{c}"]]) for c in "xyz"]
    got_angvel = [float(state[idx[f"base_ang_vel.{c}"]]) for c in "xyz"]
    assert np.allclose(got_pos, known_pos, atol=1e-3), f"base_pos not preserved: {got_pos}"
    assert np.allclose(got_quat, known_quat, atol=1e-3), f"base_quat not preserved: {got_quat}"
    assert np.allclose(got_linvel, known_linvel, atol=1e-3), f"base_lin_vel not preserved: {got_linvel}"
    assert np.allclose(got_angvel, known_angvel, atol=1e-3), f"base_ang_vel not preserved: {got_angvel}"


def test_no_drop_warning_for_floating_base(sim, caplog):
    """The base state is now preserved, so start_recording must NOT emit the
    legacy 'have a floating base ... NOT written' drop-warning (superseded)."""
    import logging

    with caplog.at_level(logging.WARNING, logger="strands_robots.simulation.recording"):
        res, _ = _start(sim, "humanoid", NAMED_BASE_XML)
    assert res["status"] == "success"
    assert not any("have a floating base" in r.getMessage() for r in caplog.records), (
        "the base-drop warning must not fire now that base state is preserved"
    )
    sim.stop_recording()
