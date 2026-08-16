"""Regression tests for the WBC auto-torque-control path on ``run_policy``.

:class:`WBCPolicy` emits joint-**position** targets. The stock
``Robot("unitree_g1")`` ships position-servo actuators with a uniform
``kp=500`` gain that overrides SONIC's tuned per-joint PD, so a bare
``sim.run_policy(policy_provider="wbc", robot_name="unitree_g1")`` drove the
servos directly and the gait diverged within a fraction of a second - the
documented quickstart silently fell over.

The fix gives the MuJoCo engine a ``_maybe_install_wbc_torque_control`` hook
that ``run_policy`` invokes after binding the policy: when a WBCPolicy meets a
position-servo scene it auto-installs the torque shim for the duration of the
call and restores the actuators afterwards. The opt-out is the
``wbc_install_torque_control=False`` kwarg.

These run WITHOUT real SONIC weights (stub ONNX session, real config + joint
mapping) on the real torque/position-servo G1 model. The end-to-end "does it
actually WALK" validation needs real weights and lives in the gated
integration suite.

``TestAutoInstallHookThroughWrappers`` pins that the shim is keyed on the WBC
policy driving the joints rather than on the type of object handed to
``run_policy``: the same policy inside a ``CompositePolicy`` (legs from WBC,
arms from a manipulation policy - the composition WBC's own docs recommend) or a
``PersistentPolicy`` needs the identical shim, because the position-servo gain
it corrects is a property of the scene and the policy, not of the wrapper.

``TestAutoInstallHook`` drives the install path and every no-op condition the
hook documents: no ``[wbc]`` extra, a non-WBC policy, no compiled world, a
controller already registered, and ``wbc_uses_position_servo`` reporting no
position-servo actuator - which it does for two distinct scenes (actuators
already flipped to torque, and a scene holding none of the WBC joints). Those
last three matter because a no-op is indistinguishable from a hook that never
ran: each one is the difference between leaving a scene alone and silently
converting somebody else's actuators.
"""

from __future__ import annotations

import ast
import inspect
import logging
import sys
import textwrap
from typing import cast

import numpy as np
import pytest

from strands_robots.policies import MockPolicy
from strands_robots.policies.wbc import (
    WBCConfig,
    WBCPolicy,
    WBCTorqueController,
    wbc_uses_position_servo,
)
from strands_robots.simulation.base import SimEngine

mujoco = pytest.importorskip("mujoco", reason="mujoco not installed")


class _StubSession:
    class _In:
        name = "obs"

    def get_inputs(self):  # type: ignore[no-untyped-def]
        return [self._In()]

    def run(self, output_names, feed):  # type: ignore[no-untyped-def]
        return [np.zeros((1, 15), dtype=np.float32)]


def _g1_policy() -> WBCPolicy:
    cfg = WBCConfig(policy_path="x.onnx")
    p = WBCPolicy(config=cfg, walk=False, allow_missing_models=True)
    p.policy_session = _StubSession()
    return p


def _build_g1_model():  # type: ignore[no-untyped-def]
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


def _namespace_for(model) -> str:  # type: ignore[no-untyped-def]
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, 1) or ""
    return "unitree_g1/" if name.startswith("unitree_g1/") else ""


def _mujoco_sim_with_world(model, data):  # type: ignore[no-untyped-def]
    """A real MuJoCo Simulation engine whose world holds the given G1 model."""
    from strands_robots.simulation.mujoco.simulation import Simulation

    sim = Simulation()
    sim._world = _FakeWorld(model, data, _namespace_for(model))  # type: ignore[assignment]
    return sim


# A scene holding none of the WBC joints: ``wbc_uses_position_servo`` cannot
# resolve a driven joint against it and conservatively reports False. Declared
# here rather than imported so this module needs no G1 assets for that case.
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


def _hook_no_op_guards() -> int:
    """Count the hook's ``return None`` early-outs by AST.

    The hook's only other exit returns the cleanup callable, so this is exactly
    the number of conditions under which it declines to touch the scene.
    """
    from strands_robots.simulation.mujoco.simulation import Simulation

    src = textwrap.dedent(inspect.getsource(Simulation._maybe_install_wbc_torque_control))
    fn = ast.parse(src).body[0]
    return sum(
        1
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant) and node.value.value is None
    )


# ---------------------------------------------------------------------------
# wbc_uses_position_servo predicate
# ---------------------------------------------------------------------------


class TestPositionServoDetection:
    def test_stock_g1_is_position_servo(self) -> None:
        model, data = _build_g1_model()
        sim = _FakeWorld(model, data, _namespace_for(model))
        wrapper = type("S", (), {"_world": sim})()
        policy = _g1_policy()
        assert wbc_uses_position_servo(cast(SimEngine, wrapper), policy, "unitree_g1") is True

    def test_false_after_actuators_flipped_to_torque(self) -> None:
        from strands_robots.policies.wbc import install_wbc_torque_control

        model, data = _build_g1_model()
        world = _FakeWorld(model, data, _namespace_for(model))
        wrapper = type("S", (), {"_world": world})()
        policy = _g1_policy()
        # Flip to torque mode; the predicate must now report "no servo".
        install_wbc_torque_control(cast(SimEngine, wrapper), policy, "unitree_g1")
        assert wbc_uses_position_servo(cast(SimEngine, wrapper), policy, "unitree_g1") is False

    def test_false_without_world(self) -> None:
        wrapper = type("S", (), {"_world": None})()
        assert wbc_uses_position_servo(cast(SimEngine, wrapper), _g1_policy(), "unitree_g1") is False


# ---------------------------------------------------------------------------
# MuJoCo engine auto-install hook
# ---------------------------------------------------------------------------


class TestAutoInstallHook:
    def test_installs_torque_shim_and_cleanup_restores(self, caplog) -> None:  # type: ignore[no-untyped-def]
        model, data = _build_g1_model()
        sim = _mujoco_sim_with_world(model, data)
        policy = _g1_policy()

        driven_before = [int(model.actuator_biastype[ai]) for ai in range(model.nu)]
        assert int(mujoco.mjtBias.mjBIAS_AFFINE) in driven_before  # stock = servo

        with caplog.at_level(logging.INFO):
            cleanup = sim._maybe_install_wbc_torque_control(policy, "unitree_g1")

        assert callable(cleanup), "expected a cleanup callable when shim is installed"
        assert "auto-installed WBC torque control" in caplog.text
        controller = sim._world._backend_state["action_controller"]
        # The driven actuators are now torque motors (biastype NONE).
        for ai in controller.leg_waist_actuator_ids:
            assert int(model.actuator_biastype[ai]) == int(mujoco.mjtBias.mjBIAS_NONE)

        # Cleanup restores the original position-servo gains.
        cleanup()
        first = controller.leg_waist_actuator_ids[0]
        assert int(model.actuator_biastype[first]) == int(mujoco.mjtBias.mjBIAS_AFFINE)

    def test_cleanup_unregisters_so_a_second_rollout_gets_the_shim(self) -> None:
        """The cleanup must undo the registry write as well as the gains.

        ``install_wbc_torque_control`` flips the driven actuators to torque
        *and* registers the controller for ``_apply_sim_action``;
        ``WBCTorqueController.uninstall`` only restores the gains. A cleanup
        that stopped there left the controller registered on a scene whose
        actuators were back to position servos - so it kept dispatching PD
        torques into servos that read them as position targets - and the
        "a manually-installed controller wins" check above then declined to
        install on the next ``run_policy``, leaving the second rollout without
        the shim the first one needed.
        """
        model, data = _build_g1_model()
        sim = _mujoco_sim_with_world(model, data)

        cleanup = sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1")
        assert callable(cleanup)
        controller = sim._world._backend_state["action_controller"]
        driven = controller.leg_waist_actuator_ids[0]
        assert int(model.actuator_biastype[driven]) == int(mujoco.mjtBias.mjBIAS_NONE)

        cleanup()

        # Both halves undone, not just the gains.
        assert int(model.actuator_biastype[driven]) == int(mujoco.mjtBias.mjBIAS_AFFINE)
        assert "action_controller" not in sim._world._backend_state

        # ... so the next rollout on the same sim installs the shim again.
        again = sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1")
        assert callable(again), "the second rollout must get the shim too"
        assert int(model.actuator_biastype[driven]) == int(mujoco.mjtBias.mjBIAS_NONE)
        again()

    def test_cleanup_leaves_a_controller_it_did_not_install(self) -> None:
        """Only the entry this hook wrote is removed.

        A manual installation that happens to be registered while an
        auto-installed cleanup runs must survive it - the cleanup owns exactly
        what its own ``install_wbc_torque_control`` call wrote.
        """
        model, data = _build_g1_model()
        sim = _mujoco_sim_with_world(model, data)

        cleanup = sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1")
        assert callable(cleanup)
        sentinel = object()
        sim._world._backend_state["action_controller"] = sentinel

        cleanup()

        assert sim._world._backend_state["action_controller"] is sentinel

    def test_skips_when_controller_already_installed(self) -> None:
        model, data = _build_g1_model()
        sim = _mujoco_sim_with_world(model, data)
        sim._world._backend_state["action_controller"] = object()  # manual install wins
        assert sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1") is None

    def test_skips_for_non_wbc_policy(self) -> None:
        model, data = _build_g1_model()
        sim = _mujoco_sim_with_world(model, data)
        assert sim._maybe_install_wbc_torque_control(MockPolicy(), "unitree_g1") is None
        assert "action_controller" not in sim._world._backend_state

    def test_skips_when_the_wbc_extra_is_absent(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # A minimal install has no [wbc] extra, so the hook's import fails and
        # run_policy must carry on unchanged rather than raise out of binding.
        premise = _mujoco_sim_with_world(*_build_g1_model())
        assert callable(premise._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1")), (
            "premise: this pair installs the shim while [wbc] is importable"
        )

        sim = _mujoco_sim_with_world(*_build_g1_model())
        policy = _g1_policy()  # built while the extra is still importable
        monkeypatch.setitem(sys.modules, "strands_robots.policies.wbc", None)
        assert sim._maybe_install_wbc_torque_control(policy, "unitree_g1") is None
        assert "action_controller" not in sim._world._backend_state

    def test_skips_without_a_world(self) -> None:
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation()
        assert sim._world is None, "premise: a bare engine has no world yet"
        assert sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1") is None

    def test_skips_when_the_world_has_no_compiled_model(self) -> None:
        from strands_robots.simulation.mujoco.simulation import Simulation

        sim = Simulation()
        # Built directly rather than through _mujoco_sim_with_world, whose
        # namespace probe needs a compiled model. Held in a local so the
        # assertions read the world under test rather than the engine's
        # SimWorld | None attribute.
        world = _FakeWorld(None, None, "")
        sim._world = world  # type: ignore[assignment]
        assert world._model is None
        assert sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1") is None
        assert "action_controller" not in world._backend_state

    def test_skips_when_the_driven_actuators_are_already_torque_motors(self) -> None:
        model, data = _build_g1_model()
        sim = _mujoco_sim_with_world(model, data)
        # from_sim is install_wbc_torque_control minus the registry write, so
        # this is the "already torque mode" scene rather than the "controller
        # already registered" one test_skips_when_controller_already_installed
        # covers - the two conditions are checked separately and in that order.
        controller = WBCTorqueController.from_sim(cast(SimEngine, sim), _g1_policy(), "unitree_g1")
        assert "action_controller" not in sim._world._backend_state
        assert controller.leg_waist_actuator_ids, "premise: driven actuators resolved"
        for ai in controller.leg_waist_actuator_ids:
            assert int(model.actuator_biastype[ai]) == int(mujoco.mjtBias.mjBIAS_NONE)
        assert wbc_uses_position_servo(cast(SimEngine, sim), _g1_policy(), "unitree_g1") is False

        assert sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1") is None
        assert "action_controller" not in sim._world._backend_state

    def test_skips_when_no_wbc_joint_resolves_in_the_scene(self) -> None:
        model = mujoco.MjModel.from_xml_string(_XML_NO_WBC_JOINTS)
        sim = _mujoco_sim_with_world(model, mujoco.MjData(model))
        assert wbc_uses_position_servo(cast(SimEngine, sim), _g1_policy(), "unitree_g1") is False

        assert sim._maybe_install_wbc_torque_control(_g1_policy(), "unitree_g1") is None
        assert "action_controller" not in sim._world._backend_state


class TestEveryNoOpConditionIsDriven:
    def test_the_hook_declines_in_exactly_the_five_ways_this_module_drives(self) -> None:
        """A sixth no-op guard is a condition nothing above exercises.

        The five, in check order: no ``[wbc]`` extra; a non-WBC policy; no
        compiled world; a controller already registered; no position-servo
        actuator. Adding a guard without a test fails here.
        """
        assert _hook_no_op_guards() == 5


class TestAutoInstallHookThroughWrappers:
    """The shim resolves a WBCPolicy declared through ``Policy.children``.

    A wrapper is a different object than the policy it wraps, so the hook's
    ``isinstance`` test saw no WBCPolicy and skipped the install - leaving the
    composition WBC's own documentation recommends (legs+waist from WBC, arms
    from a manipulation policy) driving the stock uniform-gain position servos
    that override SONIC's tuned per-joint PD. The hook now walks the declared
    policy tree, so the shim follows the policy rather than the wrapper's type.
    """

    def _sim(self):  # type: ignore[no-untyped-def]
        model, data = _build_g1_model()
        return _mujoco_sim_with_world(model, data)

    def test_composite_wrapping_wbc_gets_the_shim(self) -> None:
        from strands_robots.policies.composite import CompositePolicy

        sim = self._sim()
        composite = CompositePolicy(lower=_g1_policy(), upper=MockPolicy())
        undo = sim._maybe_install_wbc_torque_control(composite, "unitree_g1")
        assert undo is not None, "a WBCPolicy inside a CompositePolicy still needs the torque shim"
        assert isinstance(sim._world._backend_state["action_controller"], WBCTorqueController)
        undo()

    def test_the_shim_is_built_for_the_wbc_child_not_the_wrapper(self) -> None:
        # The controller runs the child's PD law, so it must hold that child -
        # a controller built around the wrapper could not compute torques at all.
        from strands_robots.policies.composite import CompositePolicy

        sim = self._sim()
        wbc = _g1_policy()
        composite = CompositePolicy(lower=wbc, upper=MockPolicy())
        undo = sim._maybe_install_wbc_torque_control(composite, "unitree_g1")
        assert undo is not None
        controller = cast(WBCTorqueController, sim._world._backend_state["action_controller"])
        assert controller.policy is wbc
        undo()

    def test_persistent_wrapping_wbc_gets_the_shim(self) -> None:
        from strands_robots.policies.persistent import PersistentPolicy

        sim = self._sim()
        wbc = _g1_policy()
        undo = sim._maybe_install_wbc_torque_control(PersistentPolicy("wbc", policy_object=wbc), "unitree_g1")
        assert undo is not None, "a WBCPolicy held warm by a PersistentPolicy still needs the torque shim"
        assert cast(WBCTorqueController, sim._world._backend_state["action_controller"]).policy is wbc
        undo()

    def test_a_wrapper_holding_no_wbc_policy_is_still_a_no_op(self) -> None:
        # The walk must not turn every wrapped policy into a WBC install: a
        # composite of two non-WBC policies leaves the scene alone.
        from strands_robots.policies.composite import CompositePolicy

        sim = self._sim()
        composite = CompositePolicy(lower=MockPolicy(), upper=MockPolicy())
        assert sim._maybe_install_wbc_torque_control(composite, "unitree_g1") is None
        assert "action_controller" not in sim._world._backend_state

    def test_wbc_nested_two_wrappers_deep_is_still_found(self) -> None:
        from strands_robots.policies.composite import CompositePolicy
        from strands_robots.policies.persistent import PersistentPolicy

        sim = self._sim()
        wbc = _g1_policy()
        nested = CompositePolicy(lower=PersistentPolicy("wbc", policy_object=wbc), upper=MockPolicy())
        undo = sim._maybe_install_wbc_torque_control(nested, "unitree_g1")
        assert undo is not None
        assert cast(WBCTorqueController, sim._world._backend_state["action_controller"]).policy is wbc
        undo()
