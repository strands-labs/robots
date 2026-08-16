"""Tests for the WBC sim integration: G1 default-fill + the torque controller.

Two units are covered, both runnable without real SONIC weights:

* ``WBCPolicy._fill_g1_defaults`` - a checkpoint that ships only ONNX weights
  yields a config with empty per-joint vectors; for the 15-DOF G1 the policy
  fills the upstream SONIC gains/stance so a real gait still works (and the
  observation builder sees the same defaults). Non-G1 configs are untouched,
  and explicit values always win.
* ``WBCTorqueController`` - flips the robot's actuators to torque mode (restored
  on uninstall), declares ``owns_stepping``, and on ``apply`` writes PD torques
  to ``data.ctrl`` and advances physics by the decimation count. The arm joints
  WBC does not drive track whatever target the action dict names for them and
  hold nominal while unnamed, so one controller can carry the upper body of a
  ``CompositePolicy`` (legs from WBC, arms from a manipulation policy) rather
  than overriding it. ``uninstall``
  releases *both* things ``install_wbc_torque_control`` acquires - the
  registration in ``world._backend_state["action_controller"]`` and the gains -
  so the documented manual install/uninstall pair hands the world back as
  completely as ``run_policy``'s auto-install does.

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


class TestArmJointsFollowTheCommandedTarget:
    """The arm joints WBC does not drive must honour a commanded target.

    ``CompositePolicy``'s contract is that each joint is driven by exactly one
    child and a dropped command is an error, never a silent resolution. The
    canonical composition puts WBC on the legs+waist and a manipulation policy
    on the arms - and this controller is what reaches the actuators for BOTH,
    because it owns the control step. An arm PD pinned to nominal would discard
    every upper-body command without a word: the robot walks correctly and the
    arms simply never move.

    The nominal hold is still the default, for the arm joints nothing commands -
    a bare WBC rollout must be unchanged.
    """

    def _installed(self):  # type: ignore[no-untyped-def]
        from strands_robots.policies.wbc import install_wbc_torque_control

        model, data = _build_min_g1()
        sim = _FakeSim(_FakeWorld(model, data, _namespace_for(model)))
        policy = _g1_policy()
        ctrl = install_wbc_torque_control(cast(SimEngine, sim), policy, "unitree_g1")
        if not ctrl.arm_actuator_ids:
            pytest.skip("scene resolves no arm actuators for WBC to hold")
        stance = {
            name: float(policy.default_angles[i])
            for i, name in enumerate(WBC_G1_ALL_JOINTS[: policy.config.num_actions])
        }
        return ctrl, model, data, stance

    def test_a_commanded_arm_joint_is_driven_toward_that_target(self) -> None:
        ctrl, model, data, stance = self._installed()
        arm_joint = ctrl.arm_actuator_ids[0]
        commanded = 0.8
        action: dict[str, Any] = dict(stance)
        action[WBC_G1_ALL_JOINTS[len(ctrl.leg_waist_actuator_ids)]] = commanded

        # One step: the torque must push the joint toward the command, not hold
        # it at nominal (from rest at ~0 a nominal hold produces ~0 torque).
        ctrl.apply(action, model, data, "unitree_g1")
        assert float(data.ctrl[arm_joint]) > 1.0

        # And it converges there over the following control steps.
        for _ in range(30):
            ctrl.apply(action, model, data, "unitree_g1")
        reached = float(data.qpos[ctrl.arm_qpos_addrs[0]])
        assert abs(reached - commanded) < 0.1, f"arm joint reached {reached:.4f}, commanded {commanded}"

    def test_an_uncommanded_arm_joint_holds_its_nominal_pose(self) -> None:
        # The bare-WBC path: the action dict names only leg+waist joints, so
        # every arm joint keeps the nominal hold of the reference deploy loop.
        ctrl, model, data, stance = self._installed()
        for _ in range(30):
            ctrl.apply(dict(stance), model, data, "unitree_g1")
        held = np.array([float(data.qpos[a]) for a in ctrl.arm_qpos_addrs])
        assert np.all(np.abs(held) < 0.15), f"uncommanded arms drifted to {held}"

    def test_commanding_one_arm_joint_leaves_its_neighbours_nominal(self) -> None:
        # Routing check: the command must land on the joint it names, so a
        # partial upper-body action does not disturb the rest of the arm.
        ctrl, model, data, stance = self._installed()
        n_legs = len(ctrl.leg_waist_actuator_ids)
        action: dict[str, Any] = dict(stance)
        action[WBC_G1_ALL_JOINTS[n_legs]] = 0.8
        for _ in range(30):
            ctrl.apply(action, model, data, "unitree_g1")
        assert abs(float(data.qpos[ctrl.arm_qpos_addrs[0]]) - 0.8) < 0.1
        others = np.array([float(data.qpos[a]) for a in ctrl.arm_qpos_addrs[1:]])
        assert np.all(np.abs(others) < 0.15), f"uncommanded neighbours moved to {others}"

    def test_a_non_numeric_arm_command_holds_the_previous_arm_target(self) -> None:
        # Same degradation the leg joints get: one malformed value holds that
        # joint's prior target instead of aborting the control step.
        ctrl, model, data, stance = self._installed()
        n_legs = len(ctrl.leg_waist_actuator_ids)
        name = WBC_G1_ALL_JOINTS[n_legs]
        ctrl.apply({**stance, name: 0.4}, model, data, "unitree_g1")
        held = float(ctrl._arm_target_q[0])
        ctrl.apply({**stance, name: "not-a-number"}, model, data, "unitree_g1")
        assert float(ctrl._arm_target_q[0]) == held
        assert np.all(np.isfinite([float(data.ctrl[a]) for a in ctrl.arm_actuator_ids]))


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


class TestUninstallReleasesBothHalvesOfTheInstall:
    """The documented manual pair: ``install_wbc_torque_control`` then ``uninstall``.

    ``run_policy``'s auto-install path is covered by the hook's own suite; this
    is the public API a caller drives directly, and it has to hand the world
    back just as completely - otherwise the registration outlives the rollout and
    the hook reads it as a manual install that wins.
    """

    def test_uninstall_deregisters_the_controller_it_registered(self) -> None:
        from strands_robots.policies.wbc import install_wbc_torque_control

        model, data = _build_min_g1()
        sim = _FakeSim(_FakeWorld(model, data, _namespace_for(model)))
        ctrl = install_wbc_torque_control(cast(SimEngine, sim), _g1_policy(), "unitree_g1")
        assert sim._world._backend_state["action_controller"] is ctrl

        ctrl.uninstall()

        assert "action_controller" not in sim._world._backend_state, (
            "uninstall left its registration behind; the next run_policy reads it as an "
            "already-installed controller and dispatches through a finished shim"
        )

    def test_uninstall_drops_the_registration_before_restoring_the_gains(self) -> None:
        """Ordering: a failure restoring gains must not leave a live registration.

        The gain restore is the part that can fail, so the registry entry goes
        first - a controller still registered against actuators that are already
        position servos again is the state this whole teardown exists to avoid.
        """
        from strands_robots.policies.wbc import install_wbc_torque_control

        model, data = _build_min_g1()
        sim = _FakeSim(_FakeWorld(model, data, _namespace_for(model)))
        ctrl = install_wbc_torque_control(cast(SimEngine, sim), _g1_policy(), "unitree_g1")

        seen: list[bool] = []
        saved = ctrl._saved_actuator_gains

        class _Boom(dict):  # type: ignore[type-arg]
            def items(self):  # type: ignore[no-untyped-def]
                seen.append("action_controller" in sim._world._backend_state)
                raise RuntimeError("gain restore failed")

        ctrl._saved_actuator_gains = _Boom(saved)  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="gain restore failed"):
            ctrl.uninstall()

        assert seen == [False], "the registration was still live when the gain restore ran"
        assert "action_controller" not in sim._world._backend_state

    def test_uninstall_leaves_a_controller_installed_since_alone(self) -> None:
        """Release only what this install acquired.

        The LIBERO adapter shares the same seam, so a later install has to
        survive an earlier controller's teardown.
        """
        from strands_robots.policies.wbc import install_wbc_torque_control

        model, data = _build_min_g1()
        sim = _FakeSim(_FakeWorld(model, data, _namespace_for(model)))
        ctrl = install_wbc_torque_control(cast(SimEngine, sim), _g1_policy(), "unitree_g1")
        newer = object()
        sim._world._backend_state["action_controller"] = newer

        ctrl.uninstall()

        assert sim._world._backend_state["action_controller"] is newer

    def test_a_never_registered_controller_deregisters_nothing(self) -> None:
        """``from_sim`` builds without registering, so its teardown has nothing to drop."""
        from strands_robots.policies.wbc import WBCTorqueController

        model, data = _build_min_g1()
        sim = _FakeSim(_FakeWorld(model, data, _namespace_for(model)))
        ctrl = WBCTorqueController.from_sim(cast(SimEngine, sim), _g1_policy(), "unitree_g1")
        assert "action_controller" not in sim._world._backend_state

        ctrl.uninstall()  # must not raise

        assert "action_controller" not in sim._world._backend_state
