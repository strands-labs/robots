"""Behavior tests for :meth:`_LiberoOSCController.from_sim`.

``from_sim`` is the factory that binds an OSC_POSE controller to a compiled
LIBERO Panda scene: it imports robosuite lazily, discovers the arm joints,
arm/gripper actuators and end-effector site from the MuJoCo model, builds a
robosuite ``MjSim`` shim + ``OSC_POSE`` controller, and (when the LIBERO home
pose is resolvable) swaps ``data.qpos`` to the canonical Panda ready pose around
controller construction so the controller latches its initial goal on the home
pose rather than the perturbed per-episode pose.

Because robosuite is a heavy transitive optional dependency it is absent on the
standard image, so the whole method is otherwise unexercised. These tests pin
the two dependency-classification contracts and the scene-validation guards
without needing a real robosuite install:

* robosuite genuinely missing -> ``_ControllerDependencyMissing`` (the caller
  degrades gracefully; requiring robosuite as a hard dep would break installs).
* the known numba / coverage>=7 import clash -> the base
  ``_ControllerInstallError`` (a FIXABLE setup bug the caller surfaces strictly,
  never silently dropping every action).
* a fully-formed Panda scene -> a constructed controller with the discovered
  actuator IDs, EEF site and physics substeps, including the #176 home-pose
  swap-and-restore around ``controller_factory``.
* scene-validation guards (no world / no compiled model / wrong arm-joint count
  / no gripper actuators / missing EEF site) each degrade with a clear
  ``_ControllerDependencyMissing``.

robosuite is faked in ``sys.modules`` for the paths that must get past the
import; the MuJoCo model is real so joint/actuator/site discovery runs against
the true binding API.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from strands_robots.benchmarks.libero import adapter as libero_adapter  # noqa: E402
from strands_robots.benchmarks.libero.adapter import (  # noqa: E402
    _ControllerDependencyMissing,
    _ControllerInstallError,
    _LiberoOSCController,
)

# A minimal but structurally-faithful LIBERO Panda: 7 ``robot0_joint*`` arm
# hinges (each driven by one motor), a 2-finger gripper with ``gripper0_*``
# slide joints + position actuators, and the ``gripper0_grip_site`` EEF site.
_PANDA_LIKE_XML = """
<mujoco>
  <worldbody>
    <body name="robot0_link1">
      <joint name="robot0_joint1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.1"/>
      <body name="robot0_link2" pos="0 0 0.1">
        <joint name="robot0_joint2" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.1"/>
        <body name="robot0_link3" pos="0 0 0.1">
          <joint name="robot0_joint3" type="hinge" axis="0 0 1"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.1"/>
          <body name="robot0_link4" pos="0 0 0.1">
            <joint name="robot0_joint4" type="hinge" axis="0 1 0"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.1"/>
            <body name="robot0_link5" pos="0 0 0.1">
              <joint name="robot0_joint5" type="hinge" axis="0 0 1"/>
              <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.1"/>
              <body name="robot0_link6" pos="0 0 0.1">
                <joint name="robot0_joint6" type="hinge" axis="0 1 0"/>
                <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.1"/>
                <body name="robot0_link7" pos="0 0 0.1">
                  <joint name="robot0_joint7" type="hinge" axis="0 0 1"/>
                  <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.05"/>
                  <site name="gripper0_grip_site" pos="0 0 0.05"/>
                  <body name="gripper0_finger1" pos="0.02 0 0.05">
                    <joint name="gripper0_finger_joint1" type="slide" axis="1 0 0" range="0 0.04"/>
                    <geom type="box" size="0.005 0.005 0.02"/>
                  </body>
                  <body name="gripper0_finger2" pos="-0.02 0 0.05">
                    <joint name="gripper0_finger_joint2" type="slide" axis="-1 0 0" range="0 0.04"/>
                    <geom type="box" size="0.005 0.005 0.02"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="robot0_act1" joint="robot0_joint1" ctrlrange="-10 10"/>
    <motor name="robot0_act2" joint="robot0_joint2" ctrlrange="-10 10"/>
    <motor name="robot0_act3" joint="robot0_joint3" ctrlrange="-10 10"/>
    <motor name="robot0_act4" joint="robot0_joint4" ctrlrange="-10 10"/>
    <motor name="robot0_act5" joint="robot0_joint5" ctrlrange="-10 10"/>
    <motor name="robot0_act6" joint="robot0_joint6" ctrlrange="-10 10"/>
    <motor name="robot0_act7" joint="robot0_joint7" ctrlrange="-10 10"/>
    <position name="gripper0_finger1_act" joint="gripper0_finger_joint1" ctrlrange="0 0.04"/>
    <position name="gripper0_finger2_act" joint="gripper0_finger_joint2" ctrlrange="0 0.04"/>
  </actuator>
</mujoco>
"""


class _StubWorld:
    """Duck-typed ``sim._world`` exposing the ``_model`` / ``_data`` the
    factory reads."""

    def __init__(self, model, data):
        self._model = model
        self._data = data


class _StubSim:
    """Duck-typed ``SimEngine`` exposing only ``_world``."""

    def __init__(self, world):
        self._world = world


class _FakeMjSim:
    """Stand-in for robosuite's ``MjSim`` - only ``.data._data`` is touched."""

    def __init__(self, model):
        self.model = model
        self.data = types.SimpleNamespace(_data=None)


def _make_panda_sim():
    model = mujoco.MjModel.from_xml_string(_PANDA_LIKE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return _StubSim(_StubWorld(model, data)), model, data


def _install_fake_robosuite(monkeypatch, *, clash: bool = False):
    """Register a fake ``robosuite`` package so ``from_sim`` gets past its
    lazy import.

    With ``clash=True`` the ``robosuite.controllers`` from-import raises the
    ``coverage.types`` ``Tracer`` ``AttributeError`` that models the #522
    numba/coverage>=7 incompatibility.
    """
    rs = types.ModuleType("robosuite")
    controllers = types.ModuleType("robosuite.controllers")
    utils = types.ModuleType("robosuite.utils")
    binding = types.ModuleType("robosuite.utils.binding_utils")

    if clash:
        # Model the #522 clash: importing the robosuite OSC path pulls in
        # numba, whose coverage-support shim references the ``coverage.types``
        # ``Tracer`` that coverage>=7 removed. We raise on attribute access
        # with that signature so ``_is_numba_coverage_clash`` recognises it.
        # (CPython's IMPORT_FROM swallows an AttributeError and re-wraps it,
        # discarding the message, so we raise an ImportError - the production
        # code catches ImportError and AttributeError alike and classifies
        # purely by message.)
        def _clash_getattr(name):
            raise ImportError("module 'coverage.types' has no attribute 'Tracer'")

        setattr(controllers, "__getattr__", _clash_getattr)  # PEP 562 module __getattr__
    else:
        setattr(
            controllers,
            "controller_factory",
            lambda name, config: types.SimpleNamespace(name=name, config=config),
        )
        setattr(controllers, "load_controller_config", lambda default_controller=None: {})

    setattr(binding, "MjSim", _FakeMjSim)

    monkeypatch.setitem(sys.modules, "robosuite", rs)
    monkeypatch.setitem(sys.modules, "robosuite.controllers", controllers)
    monkeypatch.setitem(sys.modules, "robosuite.utils", utils)
    monkeypatch.setitem(sys.modules, "robosuite.utils.binding_utils", binding)


def _from_sim(sim):
    return _LiberoOSCController.from_sim(
        sim,
        eef_site_name="gripper0_grip_site",
        arm_prefix="robot0_",
        gripper_prefix="gripper0_",
    )


class TestFromSimDependencyClassification:
    """from_sim must classify import failures so the caller can decide whether
    to degrade (missing optional dep) or surface strictly (fixable setup bug)."""

    def test_missing_robosuite_degrades_as_dependency_missing(self, monkeypatch):
        """robosuite genuinely absent -> _ControllerDependencyMissing (the
        caller degrades; robosuite is not a hard dependency)."""
        # Force the robosuite import to fail regardless of what is installed.
        monkeypatch.setitem(sys.modules, "robosuite.controllers", None)
        sim, _model, _data = _make_panda_sim()

        with pytest.raises(_ControllerDependencyMissing):
            _from_sim(sim)

    def test_numba_coverage_clash_surfaces_as_install_error(self, monkeypatch):
        """The numba/coverage>=7 clash is a FIXABLE setup bug: it must raise the
        base _ControllerInstallError, NOT the dependency-missing subclass, so
        the caller surfaces it in strict mode (#522)."""
        _install_fake_robosuite(monkeypatch, clash=True)
        sim, _model, _data = _make_panda_sim()

        with pytest.raises(_ControllerInstallError) as exc:
            _from_sim(sim)
        # Must be the base class, not the degrade-gracefully subclass.
        assert not isinstance(exc.value, _ControllerDependencyMissing)
        assert "coverage" in str(exc.value).lower()


class TestFromSimSceneValidationGuards:
    """With robosuite importable, from_sim validates the compiled scene and
    degrades with a clear _ControllerDependencyMissing when it is not a Panda."""

    def test_no_world_degrades(self, monkeypatch):
        _install_fake_robosuite(monkeypatch)
        with pytest.raises(_ControllerDependencyMissing, match="no _world"):
            _from_sim(_StubSim(None))

    def test_no_compiled_model_degrades(self, monkeypatch):
        _install_fake_robosuite(monkeypatch)
        with pytest.raises(_ControllerDependencyMissing, match="no compiled MuJoCo"):
            _from_sim(_StubSim(_StubWorld(None, None)))

    def test_wrong_arm_joint_count_degrades(self, monkeypatch):
        """A scene whose arm prefix matches a non-7-DoF joint set is not a
        LIBERO Panda -> degrade rather than abort the eval."""
        _install_fake_robosuite(monkeypatch)
        sim, _model, _data = _make_panda_sim()
        with pytest.raises(_ControllerDependencyMissing, match="7 arm joints"):
            _LiberoOSCController.from_sim(
                sim,
                eef_site_name="gripper0_grip_site",
                arm_prefix="nomatch_",
                gripper_prefix="gripper0_",
            )

    def test_missing_gripper_actuators_degrades(self, monkeypatch):
        _install_fake_robosuite(monkeypatch)
        sim, _model, _data = _make_panda_sim()
        with pytest.raises(_ControllerDependencyMissing, match="gripper actuators"):
            _LiberoOSCController.from_sim(
                sim,
                eef_site_name="gripper0_grip_site",
                arm_prefix="robot0_",
                gripper_prefix="nomatch_",
            )

    def test_missing_eef_site_degrades(self, monkeypatch):
        _install_fake_robosuite(monkeypatch)
        sim, _model, _data = _make_panda_sim()
        with pytest.raises(_ControllerDependencyMissing, match="EEF site"):
            _LiberoOSCController.from_sim(
                sim,
                eef_site_name="gripper0_no_such_site",
                arm_prefix="robot0_",
                gripper_prefix="gripper0_",
            )


class TestFromSimHappyPath:
    """A fully-formed Panda scene yields a bound controller."""

    def test_builds_controller_with_discovered_ids_and_substeps(self, monkeypatch):
        """from_sim discovers the 7 arm actuators, both gripper actuators and
        the EEF site, and derives the 20 Hz-control / 500 Hz-physics substep
        count from the model timestep."""
        _install_fake_robosuite(monkeypatch)
        # No LIBERO/robosuite home pose resolvable on this host -> the #176
        # swap branch is skipped; the controller still builds.
        monkeypatch.setattr(libero_adapter, "_resolve_libero_arm_home_qpos", lambda n: None)
        sim, model, _data = _make_panda_sim()

        controller = _from_sim(sim)

        assert isinstance(controller, _LiberoOSCController)
        assert controller.arm_actuator_ids == [0, 1, 2, 3, 4, 5, 6]
        # Two gripper position actuators, discovered by name prefix.
        assert len(controller.gripper_actuator_ids) == 2
        assert controller.eef_site_id == mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        # dt=0.002 default -> (1/20)/0.002 = 25 physics substeps per control.
        assert controller.physics_substeps_per_control == 25
        # The shim's data buffer was hot-patched to the sim's real data (#168).
        assert controller.sim_shim.data._data is _data

    def test_home_pose_swapped_and_restored_around_construction(self, monkeypatch):
        """When the LIBERO home pose resolves, from_sim writes it into
        data.qpos around controller_factory (so the controller latches its goal
        on the ready pose) and restores the per-episode qpos afterwards (#176)."""
        _install_fake_robosuite(monkeypatch)
        sim, model, data = _make_panda_sim()

        # Seed a distinct per-episode canonical arm pose we can assert is
        # restored byte-for-byte after construction.
        arm_qpos_addrs = [
            int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}")])
            for i in range(1, 8)
        ]
        canonical = np.array([0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77], dtype=np.float64)
        for adr, v in zip(arm_qpos_addrs, canonical, strict=True):
            data.qpos[adr] = v
        mujoco.mj_forward(model, data)

        home = np.array([0.0, -0.161, 0.0, -2.4446, 0.0, 2.2268, np.pi / 4], dtype=np.float64)
        captured: dict[str, np.ndarray] = {}

        def _factory(name, config):
            # Snapshot qpos AT construction time - it must be the home pose.
            captured["at_build"] = np.array([data.qpos[adr] for adr in arm_qpos_addrs], dtype=np.float64)
            return types.SimpleNamespace(name=name)

        monkeypatch.setattr(libero_adapter, "_resolve_libero_arm_home_qpos", lambda n: home.copy())
        setattr(sys.modules["robosuite.controllers"], "controller_factory", _factory)

        controller = _from_sim(sim)

        assert isinstance(controller, _LiberoOSCController)
        # At construction time the arm was at the home pose.
        np.testing.assert_allclose(captured["at_build"], home, atol=1e-9)
        # After construction the per-episode canonical pose was restored.
        restored = np.array([data.qpos[adr] for adr in arm_qpos_addrs], dtype=np.float64)
        np.testing.assert_allclose(restored, canonical, atol=1e-9)
