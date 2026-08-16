"""Isaac ``move_to`` measures a pose request exactly as the MuJoCo reference does.

The pose-convergence contract lives in the shared core
(:class:`~strands_robots.simulation.motion_primitives_base.MotionPrimitivesCore`)
so the two backends cannot drift on it: an ``orientation`` request is converged
to within ``orientation_tol`` RADIANS as well as ``tol`` METERS, both residuals
are reported, ``orientation_tol`` without an ``orientation`` is refused, and an
unreachable pose names the component that is out of reach instead of blaming the
point. The MuJoCo reference for the same behaviour is
``tests/simulation/mujoco/test_move_to_pose_convergence.py``.

No NVIDIA Isaac Sim required: the articulation and world are faked and the IK
side runs on the real inline MJCF arm, the pattern of ``test_move_to_ik.py``,
whose fakes and fixtures this module reuses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from .test_move_to_ik import (  # noqa: F401 - fake_articulation_action is an autouse fixture
    ARM_XML,
    REACHABLE_LOCAL,
    UNREACHABLE_LOCAL,
    _json_block,
    _make_sim,
    fake_articulation_action,
)


@pytest.fixture()
def arm_xml_path(tmp_path, monkeypatch):
    """Write the IK model and point the adapter's ``resolve_model`` at it.

    Defined here rather than imported so the fixture NAME is not also a
    module-level import that every test signature would shadow.
    """
    path = tmp_path / "prim_arm.xml"
    path.write_text(ARM_XML)
    monkeypatch.setattr(
        "strands_robots.simulation.isaac.motion_primitives.resolve_model",
        lambda name: str(path) if name == "prim_arm" else None,
    )
    return str(path)


# Same documented default the MuJoCo reference pins, restated rather than
# imported so a parity assertion cannot be satisfied by reading the value out
# of the result it is judging.
DOCUMENTED_ORIENTATION_TOL_RAD = 0.1

# An orientation this 4-DOF arm cannot realize at REACHABLE_LOCAL: the solver
# honours the rotation and gives up the position, so the POSITION is what the
# residual reports missing while the point itself is reachable.
INFEASIBLE_ORIENTATION = [0.7071, 0.0, 0.7071, 0.0]


def _orientation_error_rad(requested: list[float], achieved: list[float]) -> float:
    """Angle between two wxyz quaternions, measured independently of the code under test."""
    a = np.asarray(requested, dtype=np.float64)
    b = np.asarray(achieved, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(2.0 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


class TestOrientationTolDomainParity:
    """The parameter domain is the shared one, so the wording matches MuJoCo's."""

    def test_orientation_tol_without_an_orientation_is_refused(self, arm_xml_path):
        sim, _ = _make_sim()
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, orientation_tol=0.05)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "'orientation_tol' only bounds an 'orientation' target" in text
        assert "orientation=[w, x, y, z]" in text

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True, "0.1"])
    def test_orientation_tol_rejects_a_value_that_cannot_bound_an_angle(self, arm_xml_path, bad):
        sim, _ = _make_sim()
        result = sim.move_to(
            robot_name="arm",
            position=REACHABLE_LOCAL,
            orientation=[1.0, 0.0, 0.0, 0.0],
            orientation_tol=bad,
        )
        assert result["status"] == "error"
        assert "'orientation_tol' must be a positive number of radians" in result["content"][0]["text"]


class TestPoseMeasurementParity:
    """A reached pose reports both halves; a position-only call reports neither."""

    def test_a_reached_pose_reports_and_honours_its_orientation_bound(self, arm_xml_path):
        pytest.importorskip("mink")
        sim, _ = _make_sim()
        # The orientation a position-only solve settles on IS realizable here,
        # which isolates convergence from reachability.
        reference = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, tol=0.02)
        assert reference["status"] == "success", reference
        feasible = _json_block(reference)["ee_orientation_wxyz"]

        sim, _ = _make_sim()
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, orientation=feasible, tol=0.05)
        assert result["status"] == "success", result["content"][0]["text"]
        payload = _json_block(result)
        assert payload["reached"] is True
        assert payload["orientation_tol_rad"] == pytest.approx(DOCUMENTED_ORIENTATION_TOL_RAD)
        assert payload["orientation_error_rad"] <= payload["orientation_tol_rad"]
        # A loose POSITION tolerance did not buy a loose orientation.
        assert _orientation_error_rad(feasible, payload["ee_orientation_wxyz"]) <= DOCUMENTED_ORIENTATION_TOL_RAD

    def test_a_position_only_call_reports_no_orientation_measurement(self, arm_xml_path):
        pytest.importorskip("mink")
        sim, _ = _make_sim()
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, tol=0.02)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert "orientation_error_rad" not in payload
        assert "orientation_tol_rad" not in payload
        assert "ik_orientation_residual_rad" not in payload


class TestUnreachableRefusalParity:
    """The refusal names the out-of-reach component, and only for a pose request."""

    def test_an_infeasible_orientation_is_not_reported_as_an_unreachable_point(self, arm_xml_path):
        pytest.importorskip("mink")
        sim, _ = _make_sim()
        result = sim.move_to(robot_name="arm", position=REACHABLE_LOCAL, orientation=INFEASIBLE_ORIENTATION)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        payload = _json_block(result)
        assert payload["position_only_ik_residual_m"] < payload["ik_residual_m"]
        assert "on its own IS reachable" in text
        assert "Omit 'orientation'" in text
        assert "Choose a closer target or loosen tol." not in text

    def test_an_unreachable_point_keeps_the_position_only_refusal(self, arm_xml_path):
        pytest.importorskip("mink")
        sim, _ = _make_sim()
        result = sim.move_to(robot_name="arm", position=UNREACHABLE_LOCAL, tol=0.01)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "is unreachable for 'arm'" in text
        assert "Choose a closer target or loosen tol." in text
        assert "orientation" not in text
