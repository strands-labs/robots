"""Tests for the WBC sim integration: G1 default-fill + the torque controller.

Two units are covered, both runnable without real SONIC weights:

* ``WBCPolicy._fill_g1_defaults`` - a checkpoint that ships only ONNX weights
  yields a config with empty per-joint vectors; for the 15-DOF G1 the policy
  fills the upstream SONIC gains/stance so a real gait still works (and the
  observation builder sees the same defaults). Non-G1 configs are untouched,
  and explicit values always win.
* ``WBCTorqueController`` - flips the robot's actuators to torque mode (restored
  on uninstall), declares ``owns_stepping``, and on ``apply`` writes PD torques
  to ``data.ctrl`` and advances physics by the decimation count.

The end-to-end "does it actually WALK through run_policy" validation needs real
weights and lives in the gated integration suite.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from strands_robots.policies.wbc import WBC_G1_ALL_JOINTS, WBCConfig, WBCPolicy
from strands_robots.policies.wbc.policy import (
    _G1_SONIC_DEFAULT_ANGLES,
    _G1_SONIC_KDS,
    _G1_SONIC_KPS,
)
from strands_robots.simulation.base import SimEngine


class _StubSession:
    class _In:
        name = "obs"

    def get_inputs(self):  # type: ignore[no-untyped-def]
        return [self._In()]

    def run(self, output_names, feed):  # type: ignore[no-untyped-def]
        return [np.zeros((1, 15), dtype=np.float32)]


def _g1_policy(**cfg_kwargs) -> WBCPolicy:  # type: ignore[no-untyped-def]
    cfg = WBCConfig(policy_path="x.onnx", **cfg_kwargs)
    p = WBCPolicy(config=cfg, walk=False, allow_missing_models=True)
    p.policy_session = _StubSession()
    return p


# ---------------------------------------------------------------------------
# G1 default-fill (config normalisation)
# ---------------------------------------------------------------------------


class TestG1DefaultFill:
    def test_onnx_only_g1_fills_sonic_gains_and_stance(self) -> None:
        # num_actions defaults to 15 (the G1); no per-joint vectors given.
        p = _g1_policy()
        assert p.config.kps == list(_G1_SONIC_KPS)
        assert p.config.kds == list(_G1_SONIC_KDS)
        assert p.config.default_angles == list(_G1_SONIC_DEFAULT_ANGLES)
        # The resolved arrays the controller/PD law use match too.
        assert np.allclose(p._kps, _G1_SONIC_KPS)
        assert np.allclose(p.default_angles, _G1_SONIC_DEFAULT_ANGLES)

    def test_observation_builder_sees_filled_defaults(self) -> None:
        # The qj block subtracts default_angles; with the fill, config.default_angles
        # is non-empty so the offset is applied (was the ONNX-only gait bug).
        p = _g1_policy()
        assert p.config.default_angles  # non-empty
        assert len(p.config.default_angles) == p.config.num_actions

    def test_explicit_values_are_preserved(self) -> None:
        custom_kps = [10.0] * 15
        p = _g1_policy(kps=custom_kps)
        assert p.config.kps == custom_kps  # explicit wins
        # the unspecified vectors still get the G1 fill
        assert p.config.kds == list(_G1_SONIC_KDS)

    def test_non_g1_config_is_not_filled(self) -> None:
        # A 6-DOF embodiment must not receive the G1's 15-length gains.
        p = _g1_policy(num_actions=6, single_obs_dim=200, n_obs_joints=6)
        assert p.config.kps == []
        assert p.config.kds == []
        assert p.config.default_angles == []
        # resolved fallback is the neutral generic one
        assert np.allclose(p._kps, np.ones(6))


# ---------------------------------------------------------------------------
# WBCTorqueController mechanics (needs mujoco + a torque-capable G1 model)
# ---------------------------------------------------------------------------

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")


def _build_min_g1():  # type: ignore[no-untyped-def]
    """Resolve and load the standard unitree_g1 model (position-servo scene)."""
    from strands_robots.simulation.model_registry import resolve_model

    xml = resolve_model("unitree_g1")
    if not xml:
        pytest.skip("unitree_g1 model assets not available")
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    return model, data


class _FakeRobot:
    def __init__(self, namespace: str) -> None:
        self.namespace = namespace


class _FakeWorld:
    def __init__(self, model, data, namespace) -> None:  # type: ignore[no-untyped-def]
        self._model = model
        self._data = data
        self.robots = {"unitree_g1": _FakeRobot(namespace)}
        self._backend_state: dict = {}


class _FakeSim:
    def __init__(self, world) -> None:  # type: ignore[no-untyped-def]
        self._world = world


def _namespace_for(model) -> str:  # type: ignore[no-untyped-def]
    # The Menagerie scene namespaces joints as "unitree_g1/..."; the bare
    # robot_descriptions model does not. Detect which we got.
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 1) or ""
    return "unitree_g1/" if name.startswith("unitree_g1/") else ""


class TestWBCTorqueController:
    def test_install_flips_actuators_to_torque_and_restores(self) -> None:
        from strands_robots.policies.wbc import WBCTorqueController, install_wbc_torque_control

        model, data = _build_min_g1()
        ns = _namespace_for(model)
        sim = _FakeSim(_FakeWorld(model, data, ns))
        policy = _g1_policy()

        # Pre-install: the stock scene uses position-servo actuators (biastype
        # AFFINE). Capture one driven actuator's original gains.
        ctrl = install_wbc_torque_control(cast(SimEngine, sim), policy, "unitree_g1")
        assert isinstance(ctrl, WBCTorqueController)
        assert ctrl.owns_stepping is True
        assert len(ctrl.leg_waist_actuator_ids) == policy.config.num_actions

        # Driven actuators are now torque motors: gaintype FIXED, biastype NONE.
        for ai in ctrl.leg_waist_actuator_ids:
            assert int(model.actuator_gaintype[ai]) == int(mujoco.mjtGain.mjGAIN_FIXED)
            assert int(model.actuator_biastype[ai]) == int(mujoco.mjtBias.mjBIAS_NONE)

        # Physics step matches the SONIC training rate.
        assert abs(float(model.opt.timestep) - 0.005) < 1e-9

        # Uninstall restores the original biastype (position-servo => AFFINE).
        first = ctrl.leg_waist_actuator_ids[0]
        ctrl.uninstall()
        assert int(model.actuator_biastype[first]) == int(mujoco.mjtBias.mjBIAS_AFFINE)

    def test_apply_writes_torques_and_owns_stepping(self) -> None:
        from strands_robots.policies.wbc import install_wbc_torque_control

        model, data = _build_min_g1()
        ns = _namespace_for(model)
        sim = _FakeSim(_FakeWorld(model, data, ns))
        policy = _g1_policy()
        ctrl = install_wbc_torque_control(cast(SimEngine, sim), policy, "unitree_g1")

        t0 = float(data.time)
        # Action dict keyed by the WBC joint names (bare), holding the stance.
        from strands_robots.policies.wbc import WBC_G1_LEG_WAIST_JOINTS

        action = {name: float(policy.default_angles[i]) for i, name in enumerate(WBC_G1_LEG_WAIST_JOINTS)}
        ctrl.apply(action, model, data, "unitree_g1")

        # owns_stepping: apply advanced physics by decimation substeps of dt.
        expected_dt = ctrl.physics_substeps_per_control * float(model.opt.timestep)
        assert abs((float(data.time) - t0) - expected_dt) < 1e-6

        # At least one driven actuator received a finite torque command.
        ctrls = np.array([float(data.ctrl[ai]) for ai in ctrl.leg_waist_actuator_ids])
        assert np.all(np.isfinite(ctrls))

    def test_apply_non_numeric_action_value_holds_previous_target(self) -> None:
        # One malformed action value (e.g. a NaN string leaking from a policy)
        # must degrade to holding that joint's previous target, not abort the
        # whole control step - the remaining joints still track their commands.
        from strands_robots.policies.wbc import (
            WBC_G1_LEG_WAIST_JOINTS,
            install_wbc_torque_control,
        )

        model, data = _build_min_g1()
        ns = _namespace_for(model)
        sim = _FakeSim(_FakeWorld(model, data, ns))
        policy = _g1_policy()
        ctrl = install_wbc_torque_control(cast(SimEngine, sim), policy, "unitree_g1")

        stance = {name: float(policy.default_angles[i]) for i, name in enumerate(WBC_G1_LEG_WAIST_JOINTS)}
        ctrl.apply(stance, model, data, "unitree_g1")
        held = float(ctrl._target_q[0])

        # Poison joint 0 with a non-numeric value; give joint 1 a fresh command.
        corrupt: dict[str, Any] = dict(stance)
        corrupt[WBC_G1_LEG_WAIST_JOINTS[0]] = "not-a-number"
        corrupt[WBC_G1_LEG_WAIST_JOINTS[1]] = 0.123
        ctrl.apply(corrupt, model, data, "unitree_g1")

        # Bad key held its prior target; the good key still updated.
        assert float(ctrl._target_q[0]) == held
        assert abs(float(ctrl._target_q[1]) - 0.123) < 1e-12
        # And the step still produced finite torques (no abort / NaN spill).
        ctrls = np.array([float(data.ctrl[ai]) for ai in ctrl.leg_waist_actuator_ids])
        assert np.all(np.isfinite(ctrls))


# ---------------------------------------------------------------------------
# Fail-fast resolution contracts
# ---------------------------------------------------------------------------
#
# ``from_sim`` and ``install_wbc_torque_control`` flip a robot's actuators to
# torque mode by name. When the sim can't provide what they need, they must
# raise an actionable RuntimeError rather than silently install a controller
# wired to the wrong (or no) actuators - a mis-wired torque shim would drive a
# real G1 with garbage commands. ``wbc_uses_position_servo`` is the opposite
# contract: it is a conservative predicate that returns False (leave the scene
# untouched) when it cannot resolve the driven joints.


def _model_from_xml(xml: str):  # type: ignore[no-untyped-def]
    return mujoco.MjModel.from_xml_string(xml)


# A single hinge that is NOT a WBC joint: from_sim can't resolve any driven
# joint against it.
_XML_NO_WBC_JOINTS = """
<mujoco>
  <worldbody>
    <body name="b">
      <joint name="unrelated_joint" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""

# The first WBC driven joint exists but has no actuator driving it.
_XML_JOINT_WITHOUT_ACTUATOR = f"""
<mujoco>
  <worldbody>
    <body name="b">
      <joint name="{WBC_G1_ALL_JOINTS[0]}" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


class TestWBCTorqueControllerFailFast:
    def test_from_sim_without_world_raises(self) -> None:
        from strands_robots.policies.wbc import WBCTorqueController

        sim = _FakeSim(None)  # no compiled world/model on the sim
        with pytest.raises(RuntimeError, match="no compiled world/model"):
            WBCTorqueController.from_sim(cast(SimEngine, sim), _g1_policy(), "unitree_g1")

    def test_from_sim_unresolvable_joint_raises(self) -> None:
        from strands_robots.policies.wbc import WBCTorqueController

        model = _model_from_xml(_XML_NO_WBC_JOINTS)
        sim = _FakeSim(_FakeWorld(model, mujoco.MjData(model), ""))
        with pytest.raises(RuntimeError, match="not found in the model"):
            WBCTorqueController.from_sim(cast(SimEngine, sim), _g1_policy(), "unitree_g1")

    def test_from_sim_joint_without_actuator_raises(self) -> None:
        from strands_robots.policies.wbc import WBCTorqueController

        model = _model_from_xml(_XML_JOINT_WITHOUT_ACTUATOR)
        sim = _FakeSim(_FakeWorld(model, mujoco.MjData(model), ""))
        with pytest.raises(RuntimeError, match="no driving actuator"):
            WBCTorqueController.from_sim(cast(SimEngine, sim), _g1_policy(), "unitree_g1")

    def test_hold_target_falls_back_to_zeros_on_shape_mismatch(self) -> None:
        # The hold target must have length num_actions so the PD law never sees
        # a ragged array, even if the policy's resolved default_angles disagree.
        from strands_robots.policies.wbc import WBCTorqueController

        class _MismatchCfg:
            num_actions = 5

        class _MismatchPolicy:
            config = _MismatchCfg()
            default_angles = np.array([0.1, 0.2, 0.3], dtype=np.float64)  # len 3 != 5

        ctrl = WBCTorqueController(
            cast(Any, _MismatchPolicy()),
            leg_waist_actuator_ids=[],
            arm_actuator_ids=[],
            leg_waist_qpos_addrs=[],
            leg_waist_dof_addrs=[],
            arm_qpos_addrs=[],
            arm_dof_addrs=[],
            saved_actuator_gains={},
            model=None,
        )
        assert ctrl._target_q.shape == (5,)
        assert np.array_equal(ctrl._target_q, np.zeros(5))


class TestWbcUsesPositionServo:
    def test_unresolvable_scene_reports_false(self) -> None:
        # Conservative predicate: a scene with none of the WBC joints cannot be
        # a position-servo G1, so leave it untouched (no torque conversion).
        from strands_robots.policies.wbc import wbc_uses_position_servo

        model = _model_from_xml(_XML_NO_WBC_JOINTS)
        sim = _FakeSim(_FakeWorld(model, mujoco.MjData(model), ""))
        assert wbc_uses_position_servo(cast(SimEngine, sim), _g1_policy(), "unitree_g1") is False
