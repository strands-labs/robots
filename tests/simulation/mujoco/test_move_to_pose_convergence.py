"""``move_to`` measures the POSE it was asked for, not just the translation.

A ``move_to(position=..., orientation=...)`` call states a 6-DOF target, but
every gate and every reported number used to collapse the error onto the
translation: ``ik_residual_m`` was the position residual of the solve, and the
servo loop broke the moment ``position_error <= tol``. Two consequences, both
pinned here:

* the servo stopped while the orientation was still descending, so the achieved
  orientation was a function of ``tol`` - a tolerance documented in METERS -
  and the miss was reported nowhere. On the arm below, every ``tol`` from 0.01
  to 0.1 answered ``status="success"`` / ``reached=True`` while the achieved
  orientation drifted from 0.76 to 20.45 degrees off the request;
* when the ORIENTATION was the infeasible half, the refusal blamed the point
  ("target ... is unreachable ... Choose a closer target or loosen tol") even
  though that point solved position-only to 0.0017 m - and following that
  advice landed in the silent case above.

The contract now: an ``orientation`` request is converged to within
``orientation_tol`` radians as well as ``tol`` meters, both residuals are
reported, and an unreachable pose names the component that is out of reach.

Convergence tests ``importorskip`` on ``mink``; the parameter-domain tests run
anywhere.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

from .test_motion_primitives import ARM_XML, REACHABLE  # noqa: E402

# An orientation the 4-DOF arm CAN realize at REACHABLE: the one a
# position-only solve settles on. Using a feasible orientation is what isolates
# the convergence defect from reachability - a request the arm can honour, that
# it nonetheless honoured only as well as `tol` happened to allow.
FEASIBLE_ORIENTATION = [0.8748, -0.0967, 0.4324, 0.1959]

# Orientations this arm cannot realize at REACHABLE. The solver honours the
# rotation and gives up the position, so each of these misses the POSITION
# while the point itself is reachable to 0.0017 m.
INFEASIBLE_ORIENTATION = [0.7071, 0.0, 0.7071, 0.0]

# The documented default orientation tolerance, restated here on purpose: the
# assertions below must fail on BEHAVIOUR (an orientation left outside the
# documented bound) rather than on a payload key, so they may not read the
# bound out of the result they are judging.
DOCUMENTED_ORIENTATION_TOL_RAD = 0.1


@pytest.fixture
def arm_path(tmp_path):
    path = tmp_path / "prim_arm.xml"
    path.write_text(ARM_XML)
    return str(path)


@pytest.fixture
def sim(arm_path):
    s = Simulation(tool_name="test_move_to_pose_convergence", mesh=False)
    assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
    assert s.add_robot("arm", urdf_path=arm_path)["status"] == "success"
    yield s
    s.cleanup(policy_stop_timeout=2.0)


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    for block in result["content"]:
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in result: {result}")


def _orientation_error_rad(requested: list[float], achieved: list[float]) -> float:
    """Angle between two wxyz quaternions, measured independently of the code under test.

    Deliberately recomputed here from the ``ee_orientation_wxyz`` the result has
    always carried, so the strongest assertions in this module do not depend on
    any number the fix itself introduced.
    """
    a = np.asarray(requested, dtype=np.float64)
    b = np.asarray(achieved, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(2.0 * math.acos(min(1.0, abs(float(np.dot(a, b))))))


class TestOrientationIsConverged:
    """An ``orientation`` request is driven to convergence, not just to the solver."""

    @pytest.mark.parametrize("tol", [0.01, 0.03, 0.05, 0.08, 0.1])
    def test_a_loose_position_tolerance_does_not_loosen_the_orientation(self, sim, tol):
        """``tol`` bounds meters only: the orientation stays within its own bound.

        The regression, asserted purely on behaviour: the achieved orientation
        is recomputed from ``ee_orientation_wxyz`` (a key the result has always
        carried) and compared against the documented bound, so nothing here
        depends on a payload key the fix introduced.

        Pre-fix the servo broke the instant the POSITION converged, leaving the
        orientation wherever the transient had reached - so loosening a
        tolerance documented in METERS silently changed the ORIENTATION, and
        every one of these answered ``reached=True`` regardless:

            tol=0.01 -> 0.76 deg    tol=0.05 -> 7.85 deg
            tol=0.03 -> 4.18 deg    tol=0.08 -> 14.90 deg
                                    tol=0.10 -> 20.45 deg

        The last three exceed the documented 0.1 rad (5.73 deg) bound and fail
        pre-fix; the first two were already inside it and pass either way.
        """
        pytest.importorskip("mink")
        result = sim.move_to(robot_name="arm", position=REACHABLE, orientation=FEASIBLE_ORIENTATION, tol=tol)
        assert result["status"] == "success", result["content"][0]["text"]
        payload = _json_block(result)
        assert payload["reached"] is True
        achieved = _orientation_error_rad(FEASIBLE_ORIENTATION, payload["ee_orientation_wxyz"])
        assert achieved <= DOCUMENTED_ORIENTATION_TOL_RAD, (
            f"tol={tol} m converged the position ({payload['position_error_m']:.4f} m) but left the "
            f"orientation {achieved:.4f} rad ({math.degrees(achieved):.2f} deg) off the request, "
            f"outside the documented {DOCUMENTED_ORIENTATION_TOL_RAD} rad bound - yet reached=True"
        )

    def test_a_reached_pose_reports_both_halves_of_its_error(self, sim):
        """The payload carries the rotational error and the bound it was judged against."""
        pytest.importorskip("mink")
        result = sim.move_to(robot_name="arm", position=REACHABLE, orientation=FEASIBLE_ORIENTATION, tol=0.05)
        payload = _json_block(result)
        assert payload["reached"] is True
        assert payload["orientation_error_rad"] <= payload["orientation_tol_rad"]
        # The bound the primitive applies is the one it documents.
        assert payload["orientation_tol_rad"] == pytest.approx(DOCUMENTED_ORIENTATION_TOL_RAD)
        # The reported error agrees with the readback it is derived from.
        assert payload["orientation_error_rad"] == pytest.approx(
            _orientation_error_rad(FEASIBLE_ORIENTATION, payload["ee_orientation_wxyz"]), abs=1e-6
        )
        assert "orientation" in result["content"][0]["text"]

    def test_a_tighter_orientation_tol_is_honoured(self, sim):
        """A caller who asks for a tighter rotation gets it, or gets told."""
        pytest.importorskip("mink")
        result = sim.move_to(
            robot_name="arm",
            position=REACHABLE,
            orientation=FEASIBLE_ORIENTATION,
            tol=0.05,
            orientation_tol=0.02,
        )
        payload = _json_block(result)
        assert payload["orientation_tol_rad"] == pytest.approx(0.02)
        if payload["reached"]:
            assert payload["orientation_error_rad"] <= 0.02
        else:
            # A timeout must name the rotational residual, not just the position.
            assert "orientation residual" in result["content"][0]["text"]


class TestUnreachablePoseNamesTheComponent:
    """A refusal points at the half that is actually out of reach."""

    def test_an_infeasible_orientation_is_not_reported_as_an_unreachable_point(self, sim):
        """The position solves to 0.0017 m, so blaming the point sends the caller nowhere."""
        pytest.importorskip("mink")
        result = sim.move_to(robot_name="arm", position=REACHABLE, orientation=INFEASIBLE_ORIENTATION)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        payload = _json_block(result)
        # The evidence: the same point, solved with the orientation task off.
        assert payload["position_only_ik_residual_m"] < payload["ik_residual_m"]
        assert "on its own IS reachable" in text
        assert "Omit 'orientation'" in text
        # The pre-fix advice was actively harmful here - loosening tol accepted
        # a solve that still pointed the wrong way.
        assert "Choose a closer target or loosen tol." not in text

    def test_an_unreachable_point_keeps_the_position_only_refusal(self, sim):
        """Control: with no orientation requested the wording is unchanged.

        Passes both pre- and post-fix - that is the point. The diagnosis is
        added for pose requests without rewording the position-only refusal.
        """
        pytest.importorskip("mink")
        result = sim.move_to(robot_name="arm", position=[1.5, 0.0, 0.2])
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "is unreachable for 'arm'" in text
        assert "Choose a closer target or loosen tol." in text
        assert "orientation" not in text


class TestPositionOnlyIsUnchanged:
    """A position-only call claims no orientation it never converged."""

    def test_a_position_only_call_reports_no_orientation_measurement(self, sim):
        """Control: no rotational keys appear when no rotation was requested."""
        pytest.importorskip("mink")
        result = sim.move_to(robot_name="arm", position=REACHABLE)
        assert result["status"] == "success"
        payload = _json_block(result)
        assert payload["reached"] is True
        assert "orientation_error_rad" not in payload
        assert "orientation_tol_rad" not in payload
        assert "ik_orientation_residual_rad" not in payload
        # The pre-existing readback key stays.
        assert len(payload["ee_orientation_wxyz"]) == 4


class TestOrientationTolDomain:
    """``orientation_tol`` is validated, and refused where it would do nothing."""

    def test_orientation_tol_without_an_orientation_is_refused(self, sim):
        """Silently dropping a tolerance the caller set is how a miss goes unnoticed."""
        result = sim.move_to(robot_name="arm", position=REACHABLE, orientation_tol=0.05)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "'orientation_tol' only bounds an 'orientation' target" in text
        assert "orientation=[w, x, y, z]" in text

    @pytest.mark.parametrize("bad", [0.0, -1.0, -0.01, float("nan"), float("inf"), True, "0.1", None])
    def test_orientation_tol_rejects_a_value_that_cannot_bound_an_angle(self, sim, bad):
        """Zero/negative/non-finite/non-real are all refused; ``None`` means "use the default"."""
        result = sim.move_to(
            robot_name="arm",
            position=REACHABLE,
            orientation=FEASIBLE_ORIENTATION,
            orientation_tol=bad,
        )
        if bad is None:
            # None is the documented "take the default" sentinel, not an error.
            assert "orientation_tol" not in result["content"][0]["text"]
            return
        assert result["status"] == "error"
        assert "'orientation_tol' must be a positive number of radians" in result["content"][0]["text"]

    def test_orientation_tol_is_validated_before_the_world_is_touched(self, arm_path):
        """The domain refusal does not depend on a world, like every sibling domain."""
        s = Simulation(tool_name="test_move_to_pose_convergence_domain", mesh=False)
        try:
            result = s.move_to(robot_name="arm", position=REACHABLE, orientation=[1, 0, 0, 0], orientation_tol=-1)
            assert result["status"] == "error"
            assert "'orientation_tol' must be a positive number of radians" in result["content"][0]["text"]
        finally:
            s.cleanup(policy_stop_timeout=2.0)
