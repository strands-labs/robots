"""LiberoAdapter action-controller install on the Isaac backend (#1812).

The MuJoCo path builds robosuite's OSC_POSE controller against the compiled
MuJoCo model; on Isaac there is no compiled model, so pre-#1812 the install
degraded to a warning and every GR00T task-space action key landed in
``send_action``'s ``unresolved_keys`` -- success_rate pinned at 0.00 while
the run read green. These tests pin the new routing:

* On an engine exposing the Isaac action seam (``install_action_controller``
  / ``get_jacobian`` / ``list_robots`` / ``robot_joint_names`` /
  ``get_observation``), ``_install_action_controller`` installs an
  :class:`IsaacDeltaEEFController` instead of warning.
* Setup breakage on a genuine Isaac engine (missing Franka joints, broken
  Jacobian, ambiguous robots) stays LOUD: strict mode raises
  ``_ControllerInstallError``; non-strict records the failure.
* Engines with neither the MuJoCo nor the Isaac path keep the pre-existing
  warn-and-degrade behaviour, and the warning now names both missing paths.

No Isaac Sim install required -- the fake engine implements only the public
seam the adapter probes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from strands_robots.benchmarks.libero.adapter import (
    LiberoAdapter,
    _ControllerInstallError,
)
from strands_robots.simulation.isaac.delta_eef import IsaacDeltaEEFController

PICK_CUBE_BDDL = """
(define (problem libero_spatial_pick_cube)
  (:domain kitchen)
  (:language "pick up the red cube and place it on the plate")
  (:objects cube_1 plate_1 table_1 - object)
  (:init (on cube_1 table_1))
  (:goal (on cube_1 plate_1)))
"""

ARM = [f"panda_joint{i}" for i in range(1, 8)]
GRIP = ["panda_finger_joint1", "panda_finger_joint2"]
ALL_JOINTS = ARM + GRIP


class _FakeIsaacSim:
    """Duck-typed stand-in for IsaacSimulation's public action seam."""

    def __init__(
        self,
        joint_names: list[str] | None = None,
        robots: list[str] | None = None,
        jacobian_status: str = "success",
    ):
        self.joint_names = list(joint_names or ALL_JOINTS)
        self.robots = list(robots or ["robot"])
        self.jacobian_status = jacobian_status
        self.installed: dict[str, Any] = {}

    def list_robots(self) -> list[str]:
        return list(self.robots)

    def robot_joint_names(self, robot_name: str) -> list[str]:  # noqa: ARG002
        return list(self.joint_names)

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:  # noqa: ARG002
        return {name: 0.1 for name in self.joint_names}

    def get_jacobian(self, body_name=None, site_name=None, geom_name=None, robot_name=None):  # noqa: ARG002
        if self.jacobian_status != "success":
            return {"status": "error", "content": [{"text": "physics simulation view not created yet"}]}
        nv = len(self.joint_names)
        jacp = np.eye(3, nv).tolist()
        jacr = np.eye(3, nv, k=3).tolist()
        return {
            "status": "success",
            "content": [
                {"text": f"Jacobian for link '{body_name}'"},
                {"json": {"jacp": jacp, "jacr": jacr, "nv": nv}},
            ],
        }

    def install_action_controller(self, robot_name: str, controller: Any) -> dict[str, Any]:
        self.installed[robot_name] = controller
        return {"status": "success", "content": [{"text": f"Action controller installed for '{robot_name}'."}]}


def _adapter(**kwargs) -> LiberoAdapter:
    return LiberoAdapter.from_text(PICK_CUBE_BDDL, **kwargs)


class TestIsaacInstallPath:
    def test_installs_delta_eef_controller_on_isaac_seam(self):
        adapter = _adapter()
        sim = _FakeIsaacSim()
        adapter._install_action_controller(sim)
        assert adapter._action_controller_error is None
        assert adapter._isaac_action_controller_robot == "robot"
        controller = sim.installed["robot"]
        assert isinstance(controller, IsaacDeltaEEFController)
        assert controller.arm_joint_names == ARM
        assert controller.gripper_joint_names == GRIP

    def test_installed_controller_converts_a_task_space_action(self):
        """End-to-end wiring: the injected closures read the fake engine's
        Jacobian/observation and produce joint-name targets."""
        adapter = _adapter()
        sim = _FakeIsaacSim()
        adapter._install_action_controller(sim)
        targets = sim.installed["robot"].compute_joint_targets({"x": 1.0, "gripper": 1.0})
        assert set(targets) == set(ALL_JOINTS)
        # q = 0.1 everywhere; identity-block Jacobian moves dof 0 for an x delta.
        assert targets["panda_joint1"] > 0.1
        assert targets["panda_finger_joint1"] == pytest.approx(0.04)

    def test_reinstall_is_idempotent(self):
        adapter = _adapter()
        sim = _FakeIsaacSim()
        adapter._install_action_controller(sim)
        adapter._install_action_controller(sim)
        assert adapter._isaac_action_controller_robot == "robot"
        assert isinstance(sim.installed["robot"], IsaacDeltaEEFController)


class TestIsaacInstallFailuresStayLoud:
    def test_missing_franka_joints_raises_in_strict_mode(self):
        adapter = _adapter()  # strict_action_controller defaults True
        sim = _FakeIsaacSim(joint_names=["shoulder", "elbow"])
        with pytest.raises(_ControllerInstallError, match="panda_joint1"):
            adapter._install_action_controller(sim)
        assert adapter._action_controller_error is not None

    def test_missing_franka_joints_warns_in_non_strict_mode(self, caplog):
        adapter = _adapter(strict_action_controller=False)
        sim = _FakeIsaacSim(joint_names=["shoulder", "elbow"])
        with caplog.at_level("WARNING"):
            adapter._install_action_controller(sim)
        assert adapter._action_controller_error is not None
        assert "panda_joint1" in adapter._action_controller_error
        assert sim.installed == {}

    def test_multiple_robots_raise_in_strict_mode(self):
        adapter = _adapter()
        sim = _FakeIsaacSim(robots=["robot_a", "robot_b"])
        with pytest.raises(_ControllerInstallError, match="exactly one robot"):
            adapter._install_action_controller(sim)

    def test_broken_jacobian_probe_raises_in_strict_mode(self):
        adapter = _adapter()
        sim = _FakeIsaacSim(jacobian_status="error")
        with pytest.raises(_ControllerInstallError, match="Jacobian"):
            adapter._install_action_controller(sim)
        assert sim.installed == {}


class TestNonIsaacEnginesKeepDegradedPath:
    def test_engine_with_no_action_path_warns_and_names_both_paths(self, caplog):
        """A sim with neither a compiled MuJoCo model nor the Isaac seam must
        keep the pre-existing warn-and-degrade behaviour -- and the warning
        must name BOTH unavailable paths so the 0.00 success_rate is
        attributable (#1812 acceptance criterion)."""
        adapter = _adapter()

        class _BareSim:
            pass

        with caplog.at_level("WARNING"):
            adapter._install_action_controller(_BareSim())
        assert adapter._action_controller_error is not None
        assert "robosuite + mujoco" in adapter._action_controller_error
        assert "Isaac" in adapter._action_controller_error
        assert adapter._isaac_action_controller_robot is None

    def test_no_raise_even_in_strict_mode_when_no_engine_seam_exists(self):
        """Missing optional deps / wrong engine is environmental, not a
        fixable setup bug: strict mode must not raise (pre-#1812 contract)."""
        adapter = _adapter()  # strict
        adapter._install_action_controller(object())
        assert adapter._action_controller_error is not None
