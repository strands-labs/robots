"""Isaac backend parity + honesty for action/world/robot setup.

These tests pin three behaviours where the Isaac backend previously accepted a
parameter and then silently ignored or misapplied it, diverging from the
MuJoCo/Newton backends. None of them require NVIDIA Isaac Sim to be installed:

  * A partial dict ``send_action`` must command ONLY the named joints and leave
    every unnamed joint at its current PD target (parity with MuJoCo/Newton).
    The pre-fix code built a full zero-filled ``joint_positions`` vector, which
    drove every unnamed joint to 0.0 -- ``send_action({"gripper": 0.04})``
    slammed the whole arm to its home pose.
  * ``create_world`` must reject a non-Z-aligned gravity vector up front instead
    of silently reducing it to its (zero) z-component while echoing the full
    vector as applied.
  * ``add_robot`` must reject an ``mjcf_path`` (no MJCF importer) and a
    non-identity ``orientation`` (never applied) rather than silently spawning a
    procedural stub / dropping the rotation.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState


class _FakeArticulationAction:
    """Stand-in for isaacsim.core.utils.types.ArticulationAction."""

    def __init__(self, joint_positions=None, joint_indices=None):
        self.joint_positions = joint_positions
        self.joint_indices = joint_indices


class _FakeArticulation:
    """Records the last ArticulationAction handed to apply_action."""

    def __init__(self):
        self.last_action = None

    def apply_action(self, action):
        self.last_action = action

    def get_joint_positions(self):
        return None


class _FakeWorld:
    """Minimal Isaac World: send_action only needs step()."""

    def step(self, render=False):  # noqa: ARG002 - signature parity
        return None


@pytest.fixture
def fake_isaacsim_types(monkeypatch):
    """Inject a fake ``isaacsim.core.utils.types`` exposing ArticulationAction.

    ``send_action`` imports ``ArticulationAction`` lazily inside the method, so a
    fake module tree lets the apply-action path run with no Omniverse install.
    """
    mods = {}
    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.types"):
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        mods[name] = mod
    mods["isaacsim.core.utils.types"].ArticulationAction = _FakeArticulationAction
    # Wire the submodule attributes so `import isaacsim.core.utils.types` resolves.
    mods["isaacsim"].core = mods["isaacsim.core"]
    mods["isaacsim.core"].utils = mods["isaacsim.core.utils"]
    mods["isaacsim.core.utils"].types = mods["isaacsim.core.utils.types"]
    return mods


def _seed_running_world(sim, joint_names, articulation):
    sim._world = _FakeWorld()
    sim._world_created = True
    sim._robots = {
        "arm": _RobotState(
            name="arm",
            prim_path="/World/Robots/arm",
            joint_names=list(joint_names),
            articulation=articulation,
        )
    }


class TestSendActionPartialDict:
    JOINTS = ["shoulder", "elbow", "wrist", "gripper"]

    def test_partial_dict_commands_only_named_joints(self, fake_isaacsim_types):
        sim = IsaacSimulation()
        art = _FakeArticulation()
        _seed_running_world(sim, self.JOINTS, art)

        result = sim.send_action({"gripper": 0.04}, robot_name="arm")
        assert result["status"] == "success", result

        act = art.last_action
        assert act is not None, "apply_action was never called"
        # Only the gripper joint (index 3) is commanded; the shoulder/elbow/wrist
        # are left at their current PD targets (joint_indices excludes them).
        assert act.joint_indices is not None, "joint_indices must scope the command"
        assert list(np.asarray(act.joint_indices)) == [3]
        assert list(np.asarray(act.joint_positions)) == pytest.approx([0.04])
        # Regression guard: the command must NOT be a full 4-vector homing the arm.
        assert np.asarray(act.joint_positions).shape == (1,)

    def test_full_dict_commands_all_joints(self, fake_isaacsim_types):
        sim = IsaacSimulation()
        art = _FakeArticulation()
        _seed_running_world(sim, self.JOINTS, art)

        result = sim.send_action({"shoulder": 0.1, "elbow": 0.2, "wrist": 0.3, "gripper": 0.04}, robot_name="arm")
        assert result["status"] == "success", result
        act = art.last_action
        assert list(np.asarray(act.joint_indices)) == [0, 1, 2, 3]
        assert list(np.asarray(act.joint_positions)) == pytest.approx([0.1, 0.2, 0.3, 0.04])

    def test_vector_action_addresses_all_joints(self, fake_isaacsim_types):
        """A full-width vector commands every joint, in order.

        The assertion is on which DOFs are addressed rather than on how that is
        spelled. It used to require ``joint_indices is None`` -- the "all DOFs"
        shorthand the raw vector path happened to use -- which made this and
        ``test_full_dict_commands_all_joints`` pin two different spellings of one
        outcome. The vector is now bound to ``robot_action_keys`` by the shared
        coercion, so both arrive as a mapping over every joint and name the DOFs
        explicitly; ``[0, 1, 2, 3]`` and ``None`` address the same articulation.
        """
        sim = IsaacSimulation()
        art = _FakeArticulation()
        _seed_running_world(sim, self.JOINTS, art)

        result = sim.send_action([0.1, 0.2, 0.3, 0.04], robot_name="arm")
        assert result["status"] == "success", result
        act = art.last_action
        assert list(np.asarray(act.joint_indices)) == list(range(len(self.JOINTS)))
        assert list(np.asarray(act.joint_positions)) == pytest.approx([0.1, 0.2, 0.3, 0.04])


class TestCreateWorldGravity:
    def test_non_z_aligned_vector_rejected(self):
        sim = IsaacSimulation()
        result = sim.create_world(gravity=[0.0, -9.81, 0.0])
        assert result["status"] == "error"
        text = result["content"][0]["text"].lower()
        assert "z-aligned" in text or "z-aligned gravity" in text

    def test_wrong_length_vector_rejected(self):
        sim = IsaacSimulation()
        result = sim.create_world(gravity=[0.0, -9.81])
        assert result["status"] == "error"
        assert "3 components" in result["content"][0]["text"]

    def test_non_finite_gravity_rejected(self):
        sim = IsaacSimulation()
        result = sim.create_world(gravity=[0.0, 0.0, float("nan")])
        assert result["status"] == "error"
        assert "finite" in result["content"][0]["text"].lower()

    def test_z_aligned_vector_passes_validation(self):
        # A Z-aligned vector clears validation and proceeds to the Isaac import
        # (which fails on a host without Isaac Sim) -- NOT the gravity error.
        sim = IsaacSimulation()
        result = sim.create_world(gravity=[0.0, 0.0, -9.81])
        assert result["status"] == "error"
        assert "z-aligned" not in result["content"][0]["text"].lower()
        assert "3 components" not in result["content"][0]["text"]


class TestAddRobotUnsupportedParams:
    def test_mjcf_path_rejected(self):
        sim = IsaacSimulation()
        # so100 matches the procedural registry: pre-fix this silently spawned
        # the procedural stub and ignored mjcf_path.
        result = sim.add_robot(name="so100", mjcf_path="/tmp/so100.xml")
        assert result["status"] == "error"
        assert "mjcf_path" in result["content"][0]["text"]

    def test_non_identity_orientation_rejected(self):
        sim = IsaacSimulation()
        result = sim.add_robot(name="so100", orientation=[0.707, 0.0, 0.707, 0.0])
        assert result["status"] == "error"
        assert "orientation" in result["content"][0]["text"]

    def test_identity_orientation_not_rejected(self):
        # Identity quaternion must pass the guard (it reaches the world-created
        # check, since no world exists here).
        sim = IsaacSimulation()
        result = sim.add_robot(name="so100", orientation=[1.0, 0.0, 0.0, 0.0])
        assert result["status"] == "error"
        assert "orientation" not in result["content"][0]["text"]
        assert "No world created" in result["content"][0]["text"]
