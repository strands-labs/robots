"""Unit tests for the Isaac delta-EEF differential-IK action controller (#1812).

Covers, with a fully mocked articulation (no Isaac Sim install needed):

* The DLS delta -> joint-position-target conversion itself
  (:class:`IsaacDeltaEEFController`): clip-then-scale input semantics
  matching robosuite ``OSC_POSE``, gripper RLDS convention, joint-limit
  clipping, pass-through of unknown keys, and loud failures on unusable
  injected state.
* The engine seam: ``IsaacSimulation.install_action_controller`` routing in
  ``send_action`` (dict actions converted, vector actions untouched,
  conversion failures become error envelopes), ``get_jacobian``'s envelope
  built from a fake articulation view, and the ``physics_timestep``
  override.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from strands_robots.simulation.isaac.delta_eef import (
    DEFAULT_POS_SCALE,
    DEFAULT_ROT_SCALE,
    IsaacDeltaEEFController,
)
from strands_robots.simulation.isaac.simulation import IsaacSimulation, _RobotState

ARM = [f"panda_joint{i}" for i in range(1, 8)]
GRIP = ["panda_finger_joint1", "panda_finger_joint2"]
DAMPING = 0.05


def _controller(
    q: np.ndarray | None = None,
    jac: np.ndarray | None = None,
    **kwargs,
) -> IsaacDeltaEEFController:
    q = np.zeros(7) if q is None else q
    jac = np.eye(6, 7) if jac is None else jac
    return IsaacDeltaEEFController(
        arm_joint_names=ARM,
        gripper_joint_names=GRIP,
        joint_positions_fn=lambda: q,
        jacobian_fn=lambda: jac,
        **kwargs,
    )


class TestDeltaToJointTargets:
    def test_unit_x_delta_moves_first_dof_by_scaled_dls_step(self):
        """For J = eye(6, 7), the DLS solve has the closed form
        dq = twist / (1 + damping^2) padded with a zero -- pin it exactly."""
        controller = _controller()
        targets = controller.compute_joint_targets({"x": 1.0})
        expected_dq0 = DEFAULT_POS_SCALE / (1.0 + DAMPING**2)
        assert set(targets) == set(ARM)
        assert targets["panda_joint1"] == pytest.approx(expected_dq0)
        for name in ARM[1:]:
            assert targets[name] == pytest.approx(0.0)

    def test_rotation_channel_uses_rot_scale(self):
        controller = _controller()
        targets = controller.compute_joint_targets({"yaw": 1.0})
        # yaw is twist row 5 -> J = eye maps it onto dof index 5.
        assert targets["panda_joint6"] == pytest.approx(DEFAULT_ROT_SCALE / (1.0 + DAMPING**2))

    def test_input_clipped_to_unit_range(self):
        """robosuite OSC_POSE clips inputs to [-1, 1] before scaling; a
        saturated and an over-saturated command must produce identical
        targets."""
        controller = _controller()
        assert controller.compute_joint_targets({"x": 5.0}) == controller.compute_joint_targets({"x": 1.0})

    def test_half_input_scales_linearly(self):
        controller = _controller()
        full = controller.compute_joint_targets({"y": 1.0})["panda_joint2"]
        half = controller.compute_joint_targets({"y": 0.5})["panda_joint2"]
        assert half == pytest.approx(full / 2.0)

    def test_achieved_twist_tracks_commanded_twist(self):
        """Behavioral: J @ dq must approximate the commanded twist (small
        damping bias only) on a well-conditioned Jacobian."""
        rng = np.random.default_rng(7)
        # Orthonormal rows -> singular values are all 1, so the DLS bias is
        # exactly damping^2 / (1 + damping^2) ~ 0.25%.
        jac = np.linalg.qr(rng.standard_normal((7, 6)))[0].T
        controller = _controller(jac=jac)
        targets = controller.compute_joint_targets({"x": 0.4, "z": -0.8, "pitch": 0.3})
        dq = np.array([targets[n] for n in ARM])
        twist = np.array([0.4 * DEFAULT_POS_SCALE, 0.0, -0.8 * DEFAULT_POS_SCALE, 0.0, 0.3 * DEFAULT_ROT_SCALE, 0.0])
        assert jac @ dq == pytest.approx(twist, rel=0.01)

    def test_targets_offset_from_current_joint_positions(self):
        q = np.linspace(-0.5, 0.5, 7)
        controller = _controller(q=q)
        targets = controller.compute_joint_targets({"x": 1.0})
        dq0 = DEFAULT_POS_SCALE / (1.0 + DAMPING**2)
        assert targets["panda_joint1"] == pytest.approx(q[0] + dq0)
        assert targets["panda_joint7"] == pytest.approx(q[6])

    def test_zero_delta_returns_no_arm_targets(self):
        """An all-zero delta (e.g. the settle step's ``send_action({})``)
        must hold the current PD targets, not re-command them -- and must
        not even read the Jacobian."""

        def poisoned():
            raise AssertionError("jacobian_fn must not be called for a zero delta")

        controller = IsaacDeltaEEFController(
            arm_joint_names=ARM,
            gripper_joint_names=GRIP,
            joint_positions_fn=poisoned,
            jacobian_fn=poisoned,
        )
        assert controller.compute_joint_targets({}) == {}
        assert controller.compute_joint_targets({"x": 0.0, "roll": 0.0}) == {}

    def test_list_shaped_channels_use_first_element(self):
        """GR00T-LIBERO packs every action channel list-shaped (#168)."""
        controller = _controller()
        scalar = controller.compute_joint_targets({"x": 1.0})
        listed = controller.compute_joint_targets({"x": [1.0, 0.25]})
        assert listed == scalar

    def test_joint_limits_clip_targets(self):
        limits = np.column_stack([np.full(7, -0.01), np.full(7, 0.01)])
        controller = _controller(joint_limits=limits)
        targets = controller.compute_joint_targets({"x": 1.0})
        assert targets["panda_joint1"] == pytest.approx(0.01)

    def test_unknown_keys_pass_through(self):
        """Keys the controller does not consume must survive so the
        engine's unresolved-key reporting still fires for them."""
        controller = _controller()
        targets = controller.compute_joint_targets({"x": 1.0, "unmapped.foo": 3.0})
        assert targets["unmapped.foo"] == 3.0


class TestGripperConvention:
    """RLDS gripper channel: 0 = close, 1 = open, 0.5 = no command."""

    def test_open_command_targets_open_position(self):
        targets = _controller().compute_joint_targets({"gripper": 1.0})
        assert targets == {name: pytest.approx(0.04) for name in GRIP}

    def test_close_command_targets_close_position(self):
        targets = _controller().compute_joint_targets({"gripper": 0.0})
        assert targets == {name: pytest.approx(0.0) for name in GRIP}

    def test_neutral_and_absent_gripper_hold(self):
        controller = _controller()
        assert controller.compute_joint_targets({"gripper": 0.5}) == {}
        assert controller.compute_joint_targets({}) == {}

    def test_list_shaped_gripper(self):
        targets = _controller().compute_joint_targets({"gripper": [1.0, 1.0]})
        assert targets["panda_finger_joint1"] == pytest.approx(0.04)

    def test_custom_open_close_positions(self):
        controller = _controller(gripper_open=0.03, gripper_close=0.001)
        assert _round(controller.compute_joint_targets({"gripper": 1.0})) == {name: 0.03 for name in GRIP}
        assert _round(controller.compute_joint_targets({"gripper": 0.0})) == {name: 0.001 for name in GRIP}


def _round(d: dict) -> dict:
    return {k: round(v, 9) for k, v in d.items()}


class TestControllerErrors:
    def test_non_mapping_action_raises_type_error(self):
        with pytest.raises(TypeError, match="mapping"):
            _controller().compute_joint_targets([0.1] * 7)  # type: ignore[arg-type]

    def test_wrong_jacobian_shape_raises(self):
        controller = _controller(jac=np.eye(5, 7))
        with pytest.raises(RuntimeError, match=r"expected \(6, 7\)"):
            controller.compute_joint_targets({"x": 1.0})

    def test_wrong_joint_count_raises(self):
        controller = _controller(q=np.zeros(6))
        with pytest.raises(RuntimeError, match="arm_joint_names"):
            controller.compute_joint_targets({"x": 1.0})

    def test_non_finite_state_raises(self):
        controller = _controller(q=np.full(7, np.nan))
        with pytest.raises(RuntimeError, match="non-finite"):
            controller.compute_joint_targets({"x": 1.0})

    def test_constructor_rejects_empty_arm(self):
        with pytest.raises(ValueError, match="non-empty"):
            IsaacDeltaEEFController(
                arm_joint_names=[],
                gripper_joint_names=GRIP,
                joint_positions_fn=lambda: [],
                jacobian_fn=lambda: [],
            )

    def test_constructor_rejects_arm_gripper_overlap(self):
        with pytest.raises(ValueError, match="overlap"):
            IsaacDeltaEEFController(
                arm_joint_names=ARM,
                gripper_joint_names=[ARM[0]],
                joint_positions_fn=lambda: np.zeros(7),
                jacobian_fn=lambda: np.eye(6, 7),
            )

    def test_constructor_rejects_bad_damping(self):
        with pytest.raises(ValueError, match="damping"):
            _controller(damping=0.0)

    def test_constructor_rejects_bad_limits_shape(self):
        with pytest.raises(ValueError, match="joint_limits"):
            _controller(joint_limits=np.zeros((3, 2)))


# --- Engine seam: send_action routing -------------------------------------


class _FakeArticulationAction:
    def __init__(self, joint_positions=None, joint_indices=None):
        self.joint_positions = joint_positions
        self.joint_indices = joint_indices


class _FakeArticulation:
    def __init__(self):
        self.last_action = None

    def apply_action(self, action):
        self.last_action = action

    def get_joint_positions(self):
        return None


class _FakeWorld:
    def step(self, render=False):  # noqa: ARG002 - signature parity
        return None


@pytest.fixture
def fake_isaacsim_types(monkeypatch):
    """Inject a fake ``isaacsim.core.utils.types`` exposing ArticulationAction."""
    mods = {}
    for name in ("isaacsim", "isaacsim.core", "isaacsim.core.utils", "isaacsim.core.utils.types"):
        mod = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, mod)
        mods[name] = mod
    mods["isaacsim.core.utils.types"].ArticulationAction = _FakeArticulationAction
    mods["isaacsim"].core = mods["isaacsim.core"]
    mods["isaacsim.core"].utils = mods["isaacsim.core.utils"]
    mods["isaacsim.core.utils"].types = mods["isaacsim.core.utils.types"]
    return mods


JOINTS = ARM + GRIP


def _seed_running_world(sim, articulation, joint_names=None):
    sim._world = _FakeWorld()
    sim._world_created = True
    sim._robots = {
        "arm": _RobotState(
            name="arm",
            prim_path="/World/Robots/arm",
            joint_names=list(joint_names or JOINTS),
            articulation=articulation,
        )
    }


class TestSendActionControllerRouting:
    def test_task_space_dict_is_converted_and_applied(self, fake_isaacsim_types):
        sim = IsaacSimulation()
        art = _FakeArticulation()
        _seed_running_world(sim, art)
        r = sim.install_action_controller("arm", _controller())
        assert r["status"] == "success", r

        result = sim.send_action({"x": 1.0, "gripper": 1.0}, robot_name="arm")
        assert result["status"] == "success", result

        act = art.last_action
        assert act is not None, "apply_action was never called"
        commanded = dict(zip(np.asarray(act.joint_indices).tolist(), np.asarray(act.joint_positions).tolist()))
        # Arm target on dof 0 from the DLS step, fingers open at 0.04.
        assert commanded[0] == pytest.approx(DEFAULT_POS_SCALE / (1.0 + DAMPING**2), abs=1e-6)
        assert commanded[7] == pytest.approx(0.04)
        assert commanded[8] == pytest.approx(0.04)
        assert set(commanded) == set(range(9))

    def test_pre_fix_behaviour_without_controller_stays_unresolved(self, fake_isaacsim_types):
        """Without a controller, task-space keys still surface as unresolved
        (the honest pre-#1812 failure mode, not a silent success)."""
        sim = IsaacSimulation()
        _seed_running_world(sim, _FakeArticulation())
        result = sim.send_action({"x": 1.0, "gripper": 1.0}, robot_name="arm")
        assert result["status"] == "error"
        payload = result["content"][1]["json"]
        assert set(payload["unresolved_keys"]) == {"x", "gripper"}

    def test_controller_failure_is_an_error_envelope(self, fake_isaacsim_types):
        sim = IsaacSimulation()
        _seed_running_world(sim, _FakeArticulation())
        controller = _controller(jac=np.eye(4, 7))  # wrong shape -> RuntimeError
        assert sim.install_action_controller("arm", controller)["status"] == "success"
        result = sim.send_action({"x": 1.0}, robot_name="arm")
        assert result["status"] == "error"
        assert "Action controller" in result["content"][0]["text"]

    def test_vector_action_bypasses_controller(self, fake_isaacsim_types):
        sim = IsaacSimulation()
        art = _FakeArticulation()
        _seed_running_world(sim, art)

        def poisoned(_action):
            raise AssertionError("controller must not see vector actions")

        controller = _controller()
        controller.compute_joint_targets = poisoned  # type: ignore[method-assign]
        sim.install_action_controller("arm", controller)
        result = sim.send_action([0.0] * 9, robot_name="arm")
        assert result["status"] == "success", result
        assert art.last_action is not None

    def test_empty_dict_with_controller_is_a_clean_settle(self, fake_isaacsim_types):
        sim = IsaacSimulation()
        art = _FakeArticulation()
        _seed_running_world(sim, art)
        sim.install_action_controller("arm", _controller())
        result = sim.send_action({}, robot_name="arm")
        assert result["status"] == "success", result
        assert art.last_action is None, "a zero delta must not command any joint"

    def test_uninstall_restores_raw_path(self, fake_isaacsim_types):
        sim = IsaacSimulation()
        _seed_running_world(sim, _FakeArticulation())
        sim.install_action_controller("arm", _controller())
        assert sim.uninstall_action_controller("arm")["status"] == "success"
        result = sim.send_action({"x": 1.0}, robot_name="arm")
        assert result["status"] == "error"
        assert "x" in result["content"][1]["json"]["unresolved_keys"]

    def test_uninstall_is_idempotent(self):
        sim = IsaacSimulation()
        assert sim.uninstall_action_controller("nope")["status"] == "success"

    def test_install_rejects_unknown_robot(self):
        sim = IsaacSimulation()
        result = sim.install_action_controller("ghost", _controller())
        assert result["status"] == "error"
        assert "ghost" in result["content"][0]["text"]

    def test_install_rejects_non_controller(self, fake_isaacsim_types):
        sim = IsaacSimulation()
        _seed_running_world(sim, _FakeArticulation())
        result = sim.install_action_controller("arm", object())
        assert result["status"] == "error"
        assert "compute_joint_targets" in result["content"][0]["text"]

    def test_remove_robot_drops_controller(self, fake_isaacsim_types):
        sim = IsaacSimulation()
        _seed_running_world(sim, _FakeArticulation())
        sim.install_action_controller("arm", _controller())
        assert sim.remove_robot("arm")["status"] == "success"
        assert sim._action_controllers == {}


# --- Engine seam: get_jacobian ---------------------------------------------


class _FakeView:
    def __init__(self, body_names, jacobians):
        self.body_names = body_names
        self._jacobians = jacobians

    def get_jacobians(self):
        return self._jacobians


class _FakeArticulationWithView(_FakeArticulation):
    def __init__(self, view):
        super().__init__()
        self._articulation_view = view


class TestGetJacobian:
    BODIES = ["panda_link0", "panda_link1", "panda_hand"]

    def _sim(self, jacobians):
        sim = IsaacSimulation()
        art = _FakeArticulationWithView(_FakeView(self.BODIES, jacobians))
        _seed_running_world(sim, art)
        return sim

    def test_envelope_carries_link_rows(self):
        # Fixed base: (1 articulation, num_links-1, 6, num_dof).
        jacs = np.arange(1 * 2 * 6 * 9, dtype=np.float64).reshape(1, 2, 6, 9)
        sim = self._sim(jacs)
        result = sim.get_jacobian(body_name="panda_hand", robot_name="arm")
        assert result["status"] == "success", result
        payload = result["content"][1]["json"]
        assert payload["nv"] == 9
        expected = jacs[0, 1]  # body index 2, minus the excluded root
        assert np.asarray(payload["jacp"]) == pytest.approx(expected[:3])
        assert np.asarray(payload["jacr"]) == pytest.approx(expected[3:])

    def test_unknown_link_is_an_error(self):
        sim = self._sim(np.zeros((1, 2, 6, 9)))
        result = sim.get_jacobian(body_name="nonexistent", robot_name="arm")
        assert result["status"] == "error"
        assert "nonexistent" in result["content"][0]["text"]

    def test_root_link_is_an_error(self):
        sim = self._sim(np.zeros((1, 2, 6, 9)))
        result = sim.get_jacobian(body_name="panda_link0", robot_name="arm")
        assert result["status"] == "error"
        assert "root" in result["content"][0]["text"]

    def test_missing_physics_view_is_an_error(self):
        sim = self._sim(None)
        result = sim.get_jacobian(body_name="panda_hand", robot_name="arm")
        assert result["status"] == "error"
        assert "physics" in result["content"][0]["text"].lower()

    def test_unsupported_layout_is_an_error(self):
        # Floating-base layout: num_links rows, num_dof + 6 columns.
        sim = self._sim(np.zeros((1, 3, 6, 15)))
        result = sim.get_jacobian(body_name="panda_hand", robot_name="arm")
        assert result["status"] == "error"
        assert "fixed-base" in result["content"][0]["text"].lower()

    def test_site_and_geom_names_rejected(self):
        sim = self._sim(np.zeros((1, 2, 6, 9)))
        for kwargs in ({"site_name": "grip_site"}, {"geom_name": "g"}):
            result = sim.get_jacobian(body_name="panda_hand", robot_name="arm", **kwargs)
            assert result["status"] == "error"
            assert "unsupported" in result["content"][0]["text"]


class TestPhysicsTimestep:
    def test_reports_config_physics_dt(self):
        sim = IsaacSimulation(physics_dt=0.005)
        assert sim.physics_timestep() == pytest.approx(0.005)
