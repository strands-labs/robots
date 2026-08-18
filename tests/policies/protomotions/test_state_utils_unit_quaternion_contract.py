"""Every rotation in ``state_utils`` reads a quaternion as the rotation it encodes.

``quat_rotate_inverse`` and ``extract_yaw_quat`` implement formulas that are only
valid for a unit quaternion: each mixes a quadratic term in the quaternion
components with a constant, so scaling the quaternion does not cancel. A
quaternion scaled by ``s`` encodes exactly the same rotation, so both helpers
must answer the same thing for it - which means normalising the input rather
than trusting it.

That matters because not every quaternion reaching this module is unit. The
policy's own refusal invites a caller to hand-assemble ``body_rot_xyzw``; a
real-robot IMU reading drifts off unit; and an orientation obtained by linearly
interpolating two samples is short by up to ~8% (linear interpolation is exactly
why the sibling ``motion_utils`` normalises inside its slerp). Read without
normalising, such a quaternion is silently taken to be a *different* rotation:
the tests below measure a lerp'd orientation being read as a heading 6.2 deg off
and an angular velocity 29% short, and an all-zero orientation - the spelling of
one that was never written - being read as a rotation that *negates* the
angular velocity it is supposed to body-frame.

The rest of the package already settled this contract. Three other world-to-body
helpers (``policies/wbc/control.quat_rotate_inverse``,
``simulation/predicates._quat_rotate_inverse_wxyz`` and the Newton backend's
copy) all normalise internally, and the degenerate case is refused by the
same-layer ``policies/wbc/control`` sibling rather than answered with a
made-up rotation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from strands_robots.policies.protomotions.state_utils import (
    apply_heading_offset,
    compute_root_local_ang_vel,
    compute_yaw_offset,
    extract_yaw_quat,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
)

# A unit quaternion encoding a composite 90 deg rotation, in both conventions.
Q_UNIT_XYZW = np.array([0.5, -0.5, 0.5, 0.5])
Q_UNIT_WXYZ = [0.5, 0.5, -0.5, 0.5]
VEC = np.array([1.0, -2.0, 0.5])

# Two orientations 90 deg apart about world +Z, in xyzw.
Q_IDENTITY_XYZW = np.array([0.0, 0.0, 0.0, 1.0])
Q_YAW90_XYZW = np.array([0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)])


def _reference_world_to_body(q_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """``R(q)^T @ vec`` from first principles, normalising ``q`` explicitly.

    An independent oracle: it builds the rotation matrix rather than reusing the
    Rodrigues-form expansion the helpers under test share.
    """
    x, y, z, w = (np.asarray(q_xyzw, dtype=np.float64) / np.linalg.norm(q_xyzw)).tolist()
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return rot.T @ np.asarray(vec, dtype=np.float64)


def _yaw_degrees(yaw_quat_xyzw: np.ndarray) -> float:
    """Recover the yaw angle in degrees from a yaw-only ``xyzw`` quaternion."""
    q = np.asarray(yaw_quat_xyzw, dtype=np.float64)
    return math.degrees(2.0 * math.atan2(float(q[2]), float(q[3])))


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linearly interpolate two quaternions without renormalising."""
    return (1.0 - t) * np.asarray(a, dtype=np.float64) + t * np.asarray(b, dtype=np.float64)


class TestAScaledQuaternionIsTheSameRotation:
    """Scaling a quaternion must not change the rotation a helper reads from it."""

    @pytest.mark.parametrize("scale", [0.5, 0.923879, 1.01, 2.0, 4.0])
    def test_quat_rotate_inverse_is_scale_invariant(self, scale: float) -> None:
        expected = _reference_world_to_body(Q_UNIT_XYZW, VEC)
        got = np.asarray(quat_rotate_inverse(Q_UNIT_XYZW * scale, VEC), dtype=np.float64)
        assert got == pytest.approx(expected, abs=1e-9), (
            f"a quaternion scaled by {scale} encodes the same rotation, but the "
            f"world-frame vector {VEC.tolist()} body-framed to {got.tolist()} "
            f"instead of {expected.tolist()}"
        )

    @pytest.mark.parametrize("scale", [0.5, 0.923879, 1.01, 2.0, 4.0])
    def test_extract_yaw_quat_is_scale_invariant(self, scale: float) -> None:
        got = _yaw_degrees(extract_yaw_quat(Q_YAW90_XYZW * scale))
        assert got == pytest.approx(90.0, abs=1e-4), (
            f"a quaternion scaled by {scale} encodes the same 90 deg heading, but the yaw was read as {got:.3f} deg"
        )

    def test_a_lerped_orientation_reads_as_the_heading_it_encodes(self) -> None:
        # Halfway between 0 deg and 90 deg by linear interpolation: the encoded
        # heading is 45 deg, and the quaternion is ~8% short of unit.
        lerped = _lerp(Q_IDENTITY_XYZW, Q_YAW90_XYZW, 0.5)
        assert float(np.linalg.norm(lerped)) == pytest.approx(0.92388, abs=1e-4), "premise: lerp leaves |q| < 1"
        got = _yaw_degrees(extract_yaw_quat(lerped))
        assert got == pytest.approx(45.0, abs=1e-4), (
            f"an orientation obtained by interpolating two samples (|q| = "
            f"{float(np.linalg.norm(lerped)):.5f}) was read as a heading of "
            f"{got:.3f} deg instead of 45.000 deg"
        )

    def test_a_lerped_orientation_body_frames_an_angular_velocity_at_full_magnitude(self) -> None:
        lerped = _lerp(Q_IDENTITY_XYZW, Q_YAW90_XYZW, 0.5)
        # A yaw-only orientation leaves a world +Z angular velocity unchanged.
        got = np.asarray(quat_rotate_inverse(lerped, np.array([0.0, 0.0, 1.0])), dtype=np.float64)
        assert got == pytest.approx([0.0, 0.0, 1.0], abs=1e-9), (
            f"a 1.0 rad/s world-frame yaw rate body-framed to {got.tolist()} through an interpolated orientation"
        )

    def test_heading_alignment_is_scale_invariant_end_to_end(self) -> None:
        # compute_yaw_offset + apply_heading_offset is the heading aligner: it
        # maps a clip's heading onto the robot's. A robot orientation that is
        # off unit must not move where the clip ends up pointing.
        robot_unit = Q_YAW90_XYZW
        robot_lerped = _lerp(Q_IDENTITY_XYZW, Q_YAW90_XYZW, 0.5)
        clip_first_frame = Q_IDENTITY_XYZW
        aligned_from_unit = apply_heading_offset(
            compute_yaw_offset(robot_unit, clip_first_frame), np.array([Q_IDENTITY_XYZW])
        )
        aligned_from_lerped = apply_heading_offset(
            compute_yaw_offset(robot_lerped / np.linalg.norm(robot_lerped), clip_first_frame),
            np.array([Q_IDENTITY_XYZW]),
        )
        raw = apply_heading_offset(compute_yaw_offset(robot_lerped, clip_first_frame), np.array([Q_IDENTITY_XYZW]))
        assert _yaw_degrees(raw[0]) == pytest.approx(_yaw_degrees(aligned_from_lerped[0]), abs=1e-4), (
            "the same robot orientation, scaled, aligned the clip to a "
            f"different heading: {_yaw_degrees(raw[0]):.3f} deg vs "
            f"{_yaw_degrees(aligned_from_lerped[0]):.3f} deg"
        )
        assert _yaw_degrees(aligned_from_unit[0]) == pytest.approx(90.0, abs=1e-4)


class TestEveryWorldToBodyHelperInThePackageAgrees:
    """The four world-to-body helpers must answer alike for one rotation.

    They live in different layers and take different conventions, but they all
    compute ``R(q)^T @ v``. A caller that reads the same orientation through two
    of them must not get two answers.
    """

    @pytest.mark.parametrize("scale", [1.0, 4.0])
    def test_all_four_helpers_match_first_principles(self, scale: float) -> None:
        from strands_robots.policies.wbc.control import quat_rotate_inverse as wbc_helper
        from strands_robots.simulation.newton.simulation import (
            _quat_rotate_inverse_wxyz as newton_helper,
        )
        from strands_robots.simulation.predicates import (
            _quat_rotate_inverse_wxyz as predicates_helper,
        )

        expected = _reference_world_to_body(Q_UNIT_XYZW, VEC)
        scaled_wxyz = [scale * c for c in Q_UNIT_WXYZ]
        answers = {
            "policies/protomotions/state_utils": np.asarray(
                quat_rotate_inverse(Q_UNIT_XYZW * scale, VEC), dtype=np.float64
            ),
            "policies/wbc/control": np.asarray(wbc_helper(np.array(scaled_wxyz), VEC), dtype=np.float64),
            "simulation/predicates": np.asarray(predicates_helper(scaled_wxyz, VEC.tolist()), dtype=np.float64),
            "simulation/newton/simulation": np.asarray(newton_helper(scaled_wxyz, VEC.tolist()), dtype=np.float64),
        }
        disagreeing = {
            name: value.tolist() for name, value in answers.items() if not np.allclose(value, expected, atol=1e-9)
        }
        assert not disagreeing, (
            f"with |q| scaled by {scale}, these helpers disagreed with R(q)^T v = {expected.tolist()}: {disagreeing}"
        )


class TestAQuaternionThatIsNotARotationIsRefused:
    """An all-zero or non-finite orientation is refused, not answered."""

    def test_a_zero_quaternion_does_not_negate_the_vector(self) -> None:
        # Read without normalising, an all-zero quaternion collapses the
        # Rodrigues form to ``v * (0 - 1)``: the angular velocity a balance
        # controller closes the loop on comes back with its sign flipped.
        with pytest.raises(ValueError, match="cannot define a rotation"):
            quat_rotate_inverse(np.zeros(4), VEC)

    def test_a_zero_quaternion_is_refused_by_extract_yaw_quat(self) -> None:
        with pytest.raises(ValueError, match="cannot define a rotation"):
            extract_yaw_quat(np.zeros(4))

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_quaternion_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="cannot define a rotation"):
            quat_rotate_inverse(np.array([bad, 0.0, 0.0, 1.0]), VEC)

    def test_the_refusal_names_the_helper_the_value_and_a_remedy(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            quat_rotate_inverse(np.zeros(4), VEC)
        message = str(excinfo.value)
        assert "quat_rotate_inverse" in message, message
        assert "[0.0, 0.0, 0.0, 0.0]" in message, message
        assert "mujoco_wxyz_to_xyzw" in message, message


class TestTheDocumentedHandAssembledObservationPath:
    """The policy invites a caller to supply ``body_rot_xyzw`` by hand.

    ``ProtoMotionsPolicy._extract_root_local_ang_vel``'s own refusal names that
    path, and it derives the tracker's root angular-velocity input from it, so a
    non-unit row in that batch reaches the network.
    """

    @staticmethod
    def _extract(root_quat_xyzw: np.ndarray) -> np.ndarray:
        from types import SimpleNamespace
        from typing import Any

        from strands_robots.policies.protomotions.policy import ProtoMotionsPolicy

        rots = np.zeros((3, 4), dtype=np.float32)
        rots[:, 3] = 1.0
        rots[0] = np.asarray(root_quat_xyzw, dtype=np.float32)
        avels = np.zeros((3, 3), dtype=np.float32)
        avels[0] = [0.0, 0.0, 1.0]
        policy: Any = SimpleNamespace(_config=SimpleNamespace(root_body_index=0))
        return np.asarray(
            ProtoMotionsPolicy._extract_root_local_ang_vel(
                policy, {"body_rot_xyzw": rots, "body_ang_vel_world": avels}, {}
            )
        )

    def test_a_dropped_root_orientation_is_refused_rather_than_flipping_the_yaw_rate(self) -> None:
        with pytest.raises(ValueError, match="cannot define a rotation"):
            self._extract(np.zeros(4))

    def test_an_interpolated_root_orientation_keeps_the_yaw_rate_magnitude(self) -> None:
        got = self._extract(_lerp(Q_IDENTITY_XYZW, Q_YAW90_XYZW, 0.5))
        assert got.astype(np.float64) == pytest.approx([0.0, 0.0, 1.0], abs=1e-6), (
            f"a 1.0 rad/s world yaw rate reached the tracker as {got.tolist()}"
        )


class TestTheHelpersThatWereAlreadyRight:
    """Controls: normalising must not disturb what already worked."""

    def test_a_unit_quaternion_answers_exactly_as_before(self) -> None:
        expected = _reference_world_to_body(Q_UNIT_XYZW, VEC)
        assert np.asarray(quat_rotate_inverse(Q_UNIT_XYZW, VEC), dtype=np.float64) == pytest.approx(expected, abs=1e-12)

    def test_an_identity_quaternion_leaves_the_vector_alone(self) -> None:
        assert np.asarray(quat_rotate_inverse(Q_IDENTITY_XYZW, VEC), dtype=np.float64) == pytest.approx(VEC, abs=1e-12)

    def test_float32_input_stays_float32(self) -> None:
        # The tracker feeds these straight into an ONNX session, which types its
        # inputs as float32; silently widening to float64 would break that.
        q32 = Q_UNIT_XYZW.astype(np.float32)
        v32 = VEC.astype(np.float32)
        assert quat_rotate_inverse(q32, v32).dtype == np.float32
        assert extract_yaw_quat(q32).dtype == np.float32

    def test_quat_mul_is_left_as_a_plain_product(self) -> None:
        # A product of unit quaternions is already unit, and normalising a
        # product would change what "Hamilton product" means, so quat_mul is
        # deliberately untouched: scaling an input scales the output uniformly,
        # which encodes the same rotation.
        scaled = quat_mul(Q_UNIT_XYZW * 3.0, Q_YAW90_XYZW)
        plain = quat_mul(Q_UNIT_XYZW, Q_YAW90_XYZW)
        assert np.asarray(scaled, dtype=np.float64) == pytest.approx(
            3.0 * np.asarray(plain, dtype=np.float64), abs=1e-6
        )

    def test_quat_conjugate_is_left_as_a_plain_conjugate(self) -> None:
        assert np.asarray(quat_conjugate(Q_UNIT_XYZW * 2.0), dtype=np.float64) == pytest.approx(
            [-1.0, 1.0, -1.0, 1.0], abs=1e-12
        )

    def test_a_unit_root_orientation_body_frames_as_before(self) -> None:
        rots = np.array([Q_UNIT_XYZW], dtype=np.float64)
        avels = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)
        expected = _reference_world_to_body(Q_UNIT_XYZW, np.array([0.0, 0.0, 1.0]))
        assert np.asarray(compute_root_local_ang_vel(rots, avels, 0), dtype=np.float64) == pytest.approx(
            expected, abs=1e-12
        )
