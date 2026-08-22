"""An end-effector orientation that encodes no rotation is dropped, not read as identity.

``LiberoAdapter`` already refuses a config-supplied ``eef_quat_offset`` that is
not a unit quaternion, and says why in the constructor: "a malformed offset
silently feeding GR00T garbage state is exactly the failure class #168 spent 30
rounds bisecting". The orientation read *from the backend* fed the same
observation and was checked for shape only - four numbers, any four - and the
helper consuming it normalised with ``float(np.linalg.norm(q)) or 1.0``.

That substitution is what made the two indistinguishable. Measured on ``146ace1``
against the ``[0, 0, 0.097]`` wrist offset #1802 added for the Isaac hand body::

    identity  [1, 0, 0, 0]    rotate([0, 0, 0.097]) -> [0.0, 0.0, 0.097]
    all-zero  [0, 0, 0, 0]    rotate([0, 0, 0.097]) -> [0.0, 0.0, 0.097]
    NaN       [nan, 0, 0, 0]  rotate([0, 0, 0.097]) -> [nan, nan, nan]

So an orientation that was never written added the body-frame offset along world
Z whatever the wrist was doing, and ``_read_eef_state`` returned that as a
corrected position; ``_quat_wxyz_multiply`` then reported ``[0, 0, 0, 0]`` as the
orientation. Both land in the observation dict a policy consumes, which is state
that is silently wrong rather than absent.

The tree had already settled the contract elsewhere.
``tests/policies/protomotions/test_state_utils_unit_quaternion_contract.py``
names this exact value - "an all-zero orientation - the spelling of one that was
never written" - and records that three other world-to-body helpers normalise
internally while the degenerate case is *refused* by the
``policies/wbc/control.quat_rotate_inverse`` sibling "rather than answered with a
made-up rotation". ``_quat_wxyz_rotate_vec`` was a fourth helper of the same
shape and the only one that answered.

The two halves below are one rule applied at the two places a quaternion can
arrive:

* ``_extract_pose`` **drops** a degenerate orientation, because the caller wraps
  ``get_body_state`` in ``except Exception`` precisely so a state read cannot
  abort an eval - and it already has an honest branch for a missing orientation
  ("reported no orientation; cannot express the body-frame offset - using the
  uncorrected position"). A degenerate quaternion is the same unreadable
  orientation with the length right, so it takes the same path as the
  wrong-length one already did.
* ``_quat_wxyz_rotate_vec`` **raises**, so the made-up rotation cannot return
  through a caller that has not gone through ``_extract_pose``. With the gate
  above in place this is a backstop rather than a reachable path, which is why it
  is asserted directly rather than through the adapter.

#2588.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.benchmarks.libero.adapter import (
    _extract_pose,
    _quat_wxyz_rotate_vec,
    _rotation_quat_norm,
)

#: The wrist offset #1802 configures for the Isaac hand body, in the body frame.
_HAND_OFFSET = [0.0, 0.0, 0.097]

#: Spellings of an orientation that was never written. Each is four numbers, so
#: each passed the pre-#2588 shape check.
_NON_ROTATIONS = [
    pytest.param([0.0, 0.0, 0.0, 0.0], id="all-zero"),
    pytest.param([0, 0, 0, 0], id="all-zero-ints"),
    pytest.param([1e-12, 0.0, 0.0, 0.0], id="norm-below-threshold"),
    pytest.param([math.nan, 0.0, 0.0, 0.0], id="nan-w"),
    pytest.param([1.0, 0.0, math.nan, 0.0], id="nan-component"),
    pytest.param([math.inf, 0.0, 0.0, 0.0], id="inf-w"),
    pytest.param([-math.inf, 0.0, 0.0, 0.0], id="negative-inf-w"),
]

#: Orientations that do encode a rotation, including ones that are not unit.
#: A non-unit quaternion encodes the same rotation as its normalisation, so it
#: must be accepted and normalised rather than refused.
_ROTATIONS = [
    pytest.param([1.0, 0.0, 0.0, 0.0], id="identity"),
    pytest.param([0.0, 1.0, 0.0, 0.0], id="180-about-x"),
    pytest.param([2.0, 0.0, 0.0, 0.0], id="scaled-identity"),
    pytest.param([0.5, 0.5, 0.5, 0.5], id="unit-mixed"),
    pytest.param([-1.0, 0.0, 0.0, 0.0], id="negated-identity"),
    pytest.param([1e-4, 0.0, 0.0, 0.0], id="small-but-above-threshold"),
    pytest.param([1e200, 1e200, 0.0, 0.0], id="huge-but-normalizable"),
]


def _payload(quat: list[float] | None = None, pos: list[float] | None = None) -> dict:
    """Build a ``get_body_state`` success envelope carrying the given fields."""
    block: dict[str, object] = {}
    if pos is not None:
        block["position"] = pos
    if quat is not None:
        block["quaternion"] = quat
    return {"status": "success", "content": [{"json": block}]}


class TestRotationQuatNorm:
    """The domain predicate both halves of the rule share."""

    @pytest.mark.parametrize("quat", _NON_ROTATIONS)
    def test_a_non_rotation_has_no_norm_to_report(self, quat):
        assert _rotation_quat_norm(quat) is None

    @pytest.mark.parametrize("quat", _ROTATIONS)
    def test_a_rotation_reports_its_norm(self, quat):
        norm = _rotation_quat_norm(quat)
        assert norm is not None
        assert norm == pytest.approx(math.hypot(*(float(c) for c in quat)))

    @pytest.mark.parametrize("quat", [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0], []])
    def test_a_wrong_length_quaternion_is_not_a_rotation(self, quat):
        assert _rotation_quat_norm(quat) is None

    def test_an_uncoercible_component_is_not_a_rotation(self):
        """The predicate answers about the *domain*, and coerces to get there.

        A component that ``float()`` accepts is a number whatever it was spelled
        as, so ``"1.0"`` is deliberately not refused here - ``_extract_pose``
        rejects non-``(int, float)`` components before the predicate is reached,
        and duplicating that dtype rule would give two readers of one field.
        What is refused is a component there is no number in.
        """
        assert _rotation_quat_norm([None, 0.0, 0.0, 0.0]) is None
        assert _rotation_quat_norm(["not-a-number", 0.0, 0.0, 0.0]) is None
        assert _rotation_quat_norm(None) is None
        assert _rotation_quat_norm(_rotation_quat_norm) is None


class TestExtractPoseDropsANonRotation:
    """``_extract_pose`` reports an unreadable orientation as absent."""

    @pytest.mark.parametrize("quat", _NON_ROTATIONS)
    def test_a_non_rotation_is_dropped(self, quat):
        assert _extract_pose(_payload(quat=quat)) == (None, None)

    @pytest.mark.parametrize("quat", _NON_ROTATIONS)
    def test_the_position_beside_it_still_survives(self, quat):
        """The drop is per-field: this is the branch the caller already logs for."""
        pos, orientation = _extract_pose(_payload(quat=quat, pos=[1.0, 2.0, 3.0]))
        assert pos == [1.0, 2.0, 3.0]
        assert orientation is None

    @pytest.mark.parametrize("quat", _ROTATIONS)
    def test_a_rotation_is_still_returned_verbatim(self, quat):
        """Non-unit input is passed through unchanged - normalising is the
        consumer's job, and #1802's offset multiply needs the raw value."""
        assert _extract_pose(_payload(quat=quat)) == (None, [float(c) for c in quat])


class TestRotateVecRefusesANonRotation:
    """``_quat_wxyz_rotate_vec`` refuses rather than substituting a norm of 1."""

    @pytest.mark.parametrize("quat", _NON_ROTATIONS)
    def test_a_non_rotation_is_refused(self, quat):
        with pytest.raises(ValueError, match="encodes no rotation"):
            _quat_wxyz_rotate_vec(quat, _HAND_OFFSET)

    def test_the_all_zero_spelling_no_longer_answers_like_the_identity(self):
        """The measurement in the module docstring, as an assertion.

        Both calls returned ``[0.0, 0.0, 0.097]`` before #2588, which is why the
        wrong frame was invisible.
        """
        assert _quat_wxyz_rotate_vec([1.0, 0.0, 0.0, 0.0], _HAND_OFFSET) == pytest.approx(_HAND_OFFSET)
        with pytest.raises(ValueError):
            _quat_wxyz_rotate_vec([0.0, 0.0, 0.0, 0.0], _HAND_OFFSET)

    @pytest.mark.parametrize("quat", _ROTATIONS)
    def test_a_rotation_is_normalized_rather_than_trusted(self, quat):
        """A quaternion and its scaling encode one rotation, so both must answer
        the same thing - the formula is quadratic in the components and does not
        cancel, so this only holds if the helper normalises internally."""
        scaled = [3.0 * c for c in quat]
        assert _quat_wxyz_rotate_vec(scaled, _HAND_OFFSET) == pytest.approx(_quat_wxyz_rotate_vec(quat, _HAND_OFFSET))

    def test_a_rotation_still_rotates(self):
        """180 degrees about x maps +z to -z; a guard that refused everything
        would pass every test above."""
        rotated = _quat_wxyz_rotate_vec([0.0, 1.0, 0.0, 0.0], _HAND_OFFSET)
        assert rotated == pytest.approx([0.0, 0.0, -0.097])
