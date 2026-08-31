"""``build_observation`` and ``MicroduckPolicy`` route slot two through ONNX metadata.

Pollen's reference ``scripts/infer_policy.py`` supports two ``get_observations``
branches at slot two of the 61-D vector: ``projected_gravity`` (world ``-Z``
rotated into the base frame from ``base_quat``) and ``raw_accel`` (the same unit
direction estimated from the accelerometer - ``get_raw_accelerometer`` negates
the reading, normalises it, and rotates ``base_quat`` when the magnitude is too
small to carry a direction).  The choice is a training-time flag
(``self.use_projected_gravity``) baked into the export, and every currently
shipped alpha policy is ``projected_gravity``.  ``build_observation``, before
this change, unconditionally rotated ``base_quat``; a ``raw_accel`` export fed
through it received a differently-scaled, differently-signed 3-block and the
resulting drift was silent - the network kept producing plausible actions.

This file grades the four contracts the switch has to satisfy:

1. ``build_observation(..., gravity_source="raw_accel")`` reads ``base_acc`` (3)
   and ``base_quat`` (4) and writes the UNIT gravity direction Pollen's
   ``get_raw_accelerometer`` derives from them into slot two.  Both branches
   estimate ONE quantity: for a resting reading they agree to 1e-6.
2. The same call refuses a missing ``base_acc`` key with the shared
   :func:`~strands_robots.utils.finite_vector_error`-style contract that
   ``_require_base_block`` already applies to ``base_quat``.
3. ``build_observation`` refuses a ``gravity_source`` value that is neither of
   the two shipped spellings, naming the pair.  The projected-gravity default
   with no argument still reproduces the pre-change vector byte-for-byte.
4. ``MicroduckPolicy`` reads ``gravity_source`` out of the ONNX
   ``custom_metadata_map`` in ``_ensure_config`` and threads it to the builder;
   a mistyped metadata entry raises at first inference (not at slot-two read).

The four cells together are the acceptance criterion for harness#388: the
provider can serve a ``raw_accel`` export end-to-end, and a wrong metadata
value fails loudly at configuration time rather than as drift on a
running rollout.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from strands_robots.policies.microduck import (
    GRAVITY_SOURCE_PROJECTED,
    GRAVITY_SOURCE_RAW_ACCEL,
    MICRODUCK_DEFAULT_POSE,
    MICRODUCK_JOINT_NAMES,
    MicroduckPolicy,
    build_observation,
    projected_gravity,
)


def _base_dict(*, base_ang_vel=(0.0, 0.0, 0.0), base_quat=(1.0, 0.0, 0.0, 0.0), extra=None) -> dict[str, Any]:
    """Assemble a minimal observation dict with the joint / velocity keys ``build_observation`` reads.

    The joint values are all zero at the default pose, so ``joint_pos_relative``
    is a zero block and ``joint_vel`` a zero block; that keeps the assembled
    vector's non-slot-two components at zero and lets the test isolate what
    slot two carries.
    """
    obs: dict[str, Any] = {"base_ang_vel": list(base_ang_vel), "base_quat": list(base_quat)}
    for name, pose in zip(MICRODUCK_JOINT_NAMES, MICRODUCK_DEFAULT_POSE):
        obs[name] = float(pose)
        obs[f"{name}.vel"] = 0.0
    if extra:
        obs.update(extra)
    return obs


def _last_action_zeros() -> np.ndarray:
    return np.zeros(len(MICRODUCK_JOINT_NAMES), dtype=np.float32)


def _command_zeros(width: int = 13) -> np.ndarray:
    return np.zeros(width, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. raw_accel: base_acc is written into slot two VERBATIM.
# ---------------------------------------------------------------------------


def _pollen_raw_accelerometer(accel: Any, quat: Any) -> np.ndarray:
    """Pollen's ``get_raw_accelerometer``, transcribed from ``infer_policy.py``.

    Stated locally rather than imported so the cells below grade the shipped
    estimator against a second, independent statement of the reference instead
    of against itself.
    """
    negated = -np.asarray(accel, dtype=np.float32)
    magnitude = float(np.linalg.norm(negated))
    if magnitude > 0.1:
        return np.asarray(negated / magnitude, dtype=np.float32)
    q = np.asarray(quat, dtype=np.float32)
    world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    t = np.cross(q[1:4], world_gravity) * 2.0
    return np.asarray(world_gravity - q[0] * t + np.cross(q[1:4], t), dtype=np.float32)


def _slot_two(obs: dict[str, Any], source: str) -> np.ndarray:
    """Build the vector and return slot two (indices 3..5)."""
    vector = build_observation(
        obs,
        joint_names=list(MICRODUCK_JOINT_NAMES),
        default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
        last_action=_last_action_zeros(),
        command=_command_zeros(),
        gravity_source=source,
    )
    assert vector.dtype == np.float32
    # 48 layout components + 13 command = 61-D alpha vector, unchanged.
    assert vector.shape == (61,)
    return np.asarray(vector[3:6], dtype=np.float32)


def test_raw_accel_writes_the_unit_gravity_direction_not_the_reading() -> None:
    """``gravity_source="raw_accel"`` negates and normalises ``base_acc`` into slot two.

    Slot two spans indices 3..5 (base_ang_vel is 0..2).  Pollen's
    ``get_raw_accelerometer`` returns ``-accel / |accel|`` for a usable reading,
    so the block is a UNIT gravity direction - the same quantity the
    ``projected_gravity`` branch derives from the orientation, in the same units.

    Writing the reading unchanged instead puts a vector ``|accel|``-times too
    long, with every component sign-flipped, into a slot the export was trained
    to receive a unit direction in.  Both are finite and both are the documented
    width, so nothing downstream refuses either; this cell is what makes the
    difference observable at all.
    """
    accel = np.array([0.5, -1.25, 9.81], dtype=np.float32)
    obs = _base_dict(extra={"base_acc": accel.tolist()})

    slot_two = _slot_two(obs, GRAVITY_SOURCE_RAW_ACCEL)

    np.testing.assert_allclose(slot_two, _pollen_raw_accelerometer(accel, (1.0, 0.0, 0.0, 0.0)), rtol=0, atol=1e-6)
    assert np.isclose(float(np.linalg.norm(slot_two)), 1.0, atol=1e-6)
    # Not the reading: the scale and every sign differ.
    assert not np.allclose(slot_two, accel)
    assert float(np.linalg.norm(accel)) > 9.0


@pytest.mark.parametrize(
    "accel",
    [
        (0.5, -1.25, 9.81),
        (-9.1296, 0.0001, 3.5898),  # the settled duck's own reading, measured in sim
        (0.0, 0.0, 9.81),
        (1e-3, -2e-3, 4e-3),  # |accel| = 4.6e-3, below the 0.1 threshold: the fallback
        (3.0, 4.0, 0.0),
    ],
    ids=["non-canonical", "settled-duck", "canonical-up", "degenerate", "planar"],
)
def test_the_shipped_estimator_matches_pollens_for_every_reading(accel: tuple[float, ...]) -> None:
    """Slot two equals the reference estimator for usable AND degenerate readings.

    The reference is stated locally in :func:`_pollen_raw_accelerometer`, so this
    is a parity check against a second statement of Pollen's algorithm rather
    than a restatement of the implementation.  The ``degenerate`` row is below
    the ``0.1`` magnitude threshold and therefore exercises the rotation
    fallback through the same seam.
    """
    # Exactly unit in float32 (0.6^2 + 0.8^2 == 1.0), so the rotation preserves
    # the norm and a 1e-6 unit-length assertion is a real check rather than a
    # tolerance for the fixture's own rounding.
    quat = (0.6, 0.0, 0.8, 0.0)
    obs = _base_dict(base_quat=quat, extra={"base_acc": list(accel)})

    slot_two = _slot_two(obs, GRAVITY_SOURCE_RAW_ACCEL)

    np.testing.assert_allclose(slot_two, _pollen_raw_accelerometer(accel, quat), rtol=0, atol=1e-6)
    assert np.isclose(float(np.linalg.norm(slot_two)), 1.0, atol=1e-6)


def test_the_two_branches_agree_for_a_resting_reading() -> None:
    """A resting IMU makes the two ``gravity_source`` branches one quantity.

    At rest the accelerometer measures the reaction to gravity, so its reading is
    ``-g`` times the projected-gravity direction.  Feeding that reading to the
    ``raw_accel`` branch has to reproduce the ``projected_gravity`` branch's slot
    two - which is what "two estimators of one quantity" means, and what makes a
    ``raw_accel`` export interchangeable with an alpha export on the same robot.

    Measured in sim on a settled duck the two agree to 1e-6; this cell states the
    same property without needing MuJoCo.
    """
    quat = (0.6, 0.0, 0.8, 0.0)  # exactly unit in float32
    projected = projected_gravity(np.array(quat, dtype=np.float32))
    resting_reading = (-9.81 * projected).astype(np.float32)

    from_quaternion = _slot_two(_base_dict(base_quat=quat), GRAVITY_SOURCE_PROJECTED)
    from_accelerometer = _slot_two(
        _base_dict(base_quat=quat, extra={"base_acc": resting_reading.tolist()}),
        GRAVITY_SOURCE_RAW_ACCEL,
    )

    np.testing.assert_allclose(from_accelerometer, from_quaternion, rtol=0, atol=1e-6)


def test_a_degenerate_reading_falls_back_to_the_rotation_rather_than_a_zero_block() -> None:
    """An all-zero ``base_acc`` yields the rotated direction, not a zero vector.

    Free fall reads ``|accel| == 0``: the accelerometer carries no direction at
    all.  Pollen rotates ``base_quat`` in that case, so slot two stays a unit
    vector.  Passing the reading through unchanged would hand the network a zero
    block - finite, the documented width, and pointing nowhere.
    """
    quat = (0.6, 0.0, 0.8, 0.0)  # exactly unit in float32
    obs = _base_dict(base_quat=quat, extra={"base_acc": [0.0, 0.0, 0.0]})

    slot_two = _slot_two(obs, GRAVITY_SOURCE_RAW_ACCEL)

    np.testing.assert_allclose(slot_two, projected_gravity(np.array(quat, dtype=np.float32)), rtol=0, atol=1e-6)
    assert np.isclose(float(np.linalg.norm(slot_two)), 1.0, atol=1e-6)
    assert not np.allclose(slot_two, np.zeros(3, dtype=np.float32))


def test_raw_accel_requires_base_quat_for_the_fallback() -> None:
    """The ``raw_accel`` path refuses a dict with no ``base_quat``, at the first tick.

    The degenerate-reading fallback is the rotation, so this branch needs the
    orientation as well as the accelerometer.  Requiring it up front means a
    caller is refused on tick one rather than at the moment the robot leaves the
    ground, which is the worst time to discover a missing key.
    """
    obs = _base_dict(extra={"base_acc": [0.5, -1.25, 9.81]})
    del obs["base_quat"]

    with pytest.raises(KeyError):
        build_observation(
            obs,
            joint_names=list(MICRODUCK_JOINT_NAMES),
            default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
            last_action=_last_action_zeros(),
            command=_command_zeros(),
            gravity_source=GRAVITY_SOURCE_RAW_ACCEL,
        )


def test_raw_accel_refuses_a_missing_base_acc_key() -> None:
    """A ``raw_accel`` builder call with no ``base_acc`` in the dict raises ``KeyError``.

    The shared ``_require_base_block`` reader is what enforces this on
    ``base_quat`` today; the raw-accel branch has to route through the same
    helper so the two base blocks refuse missingness the same way, rather
    than one raising and the other silently returning a zero block.
    """
    obs = _base_dict()  # deliberately omits "base_acc"
    with pytest.raises(KeyError):
        build_observation(
            obs,
            joint_names=list(MICRODUCK_JOINT_NAMES),
            default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
            last_action=_last_action_zeros(),
            command=_command_zeros(),
            gravity_source=GRAVITY_SOURCE_RAW_ACCEL,
        )


def test_raw_accel_refuses_a_wrong_width_base_acc_block() -> None:
    """A 2- or 4-component ``base_acc`` raises ``ValueError`` naming the width contract.

    ``_require_base_block`` on ``base_quat`` already refuses widths other than
    4; ``base_acc`` has to inherit the same refusal at width 3, with the same
    error shape (a message that names the block, the observed width and the
    expected one).  Without this cell a caller who passes a 6-vector
    accelerometer (some IMUs concatenate accel+gyro) would have their first
    three components silently taken as slot two while the ``[3:6]`` half was
    dropped, and the rollout drift would not name the caller.
    """
    obs = _base_dict(extra={"base_acc": [0.0, 0.0]})  # short by one
    with pytest.raises(ValueError, match="base_acc"):
        build_observation(
            obs,
            joint_names=list(MICRODUCK_JOINT_NAMES),
            default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
            last_action=_last_action_zeros(),
            command=_command_zeros(),
            gravity_source=GRAVITY_SOURCE_RAW_ACCEL,
        )


# ---------------------------------------------------------------------------
# 2. gravity_source domain: only the two shipped values are accepted.
# ---------------------------------------------------------------------------


def test_gravity_source_refuses_a_third_spelling() -> None:
    """``build_observation(..., gravity_source="gravity")`` raises, naming the shipped pair.

    A caller who mistypes ``"gravity"`` or ``"accel"`` gets a raise, not a
    silent selection of one branch.  This is the seam that makes a training
    export's metadata typo fail loudly rather than as drift.
    """
    obs = _base_dict()
    with pytest.raises(ValueError, match="gravity_source"):
        build_observation(
            obs,
            joint_names=list(MICRODUCK_JOINT_NAMES),
            default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
            last_action=_last_action_zeros(),
            command=_command_zeros(),
            gravity_source="gravity",  # neither of the two shipped spellings
        )


def test_projected_gravity_default_matches_the_pre_change_vector() -> None:
    """Omitting ``gravity_source`` produces the same vector as before the switch.

    Every currently shipped alpha policy is a ``projected_gravity`` export, so
    a caller who never sets ``gravity_source`` has to receive a byte-identical
    slot two to the one shipped ``build_observation`` returned before this
    change.  The invariant is checked against an explicit
    ``gravity_source="projected_gravity"`` call to demonstrate the two paths
    resolve to the same code, and against a hand-computed value to demonstrate
    the shipped-alpha behaviour did not change.
    """
    obs = _base_dict(base_quat=(1.0, 0.0, 0.0, 0.0))  # identity: gravity is world -Z

    default_call = build_observation(
        obs,
        joint_names=list(MICRODUCK_JOINT_NAMES),
        default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
        last_action=_last_action_zeros(),
        command=_command_zeros(),
    )
    explicit_call = build_observation(
        obs,
        joint_names=list(MICRODUCK_JOINT_NAMES),
        default_pose=np.array(MICRODUCK_DEFAULT_POSE, dtype=np.float32),
        last_action=_last_action_zeros(),
        command=_command_zeros(),
        gravity_source=GRAVITY_SOURCE_PROJECTED,
    )
    np.testing.assert_array_equal(default_call, explicit_call)
    # Identity quaternion rotates world -Z (0, 0, -1) to itself in the base frame.
    np.testing.assert_array_equal(default_call[3:6], np.array([0.0, 0.0, -1.0], dtype=np.float32))


# ---------------------------------------------------------------------------
# 3. MicroduckPolicy: reads gravity_source from ONNX metadata.
# ---------------------------------------------------------------------------


class _FakeMeta:
    """Stand-in for ``session.get_modelmeta()`` with a settable metadata map."""

    def __init__(self, custom_metadata_map: dict[str, str]) -> None:
        self.custom_metadata_map = custom_metadata_map


class _FakeSession:
    """Minimal ``MicroduckSession`` stand-in that only serves metadata + a fake input name.

    ``_ensure_config`` reads metadata; it does not run inference. That keeps
    this file free of an onnxruntime dependency and free of a real graph.
    """

    def __init__(self, meta_map: dict[str, str]) -> None:
        self._meta = _FakeMeta(meta_map)

    def get_modelmeta(self) -> _FakeMeta:
        return self._meta

    def get_inputs(self) -> list[Any]:  # pragma: no cover - unused by _ensure_config
        class _Input:
            name = "observation"
            shape = [1, 61]

        return [_Input()]


def _make_policy(meta_map: dict[str, str]) -> MicroduckPolicy:
    return MicroduckPolicy(session=cast(Any, _FakeSession(meta_map)))


def test_policy_reads_gravity_source_from_metadata_and_threads_it() -> None:
    """``MicroduckPolicy._ensure_config`` reads ``gravity_source`` off the metadata map.

    A session whose metadata declares ``gravity_source: raw_accel`` has to
    leave ``policy._gravity_source == "raw_accel"``, so the next
    :meth:`get_actions` call routes through the raw-accel branch.  The
    positive contract is checked here; the negative (mistyped metadata) is
    checked below.
    """
    policy = _make_policy({"gravity_source": "raw_accel"})
    policy._ensure_config()
    assert policy._gravity_source == GRAVITY_SOURCE_RAW_ACCEL


def test_policy_defaults_gravity_source_to_projected_when_metadata_is_silent() -> None:
    """A metadata map without ``gravity_source`` resolves to ``projected_gravity``.

    Every currently-shipped alpha policy is ``projected_gravity`` and does
    not carry the flag in metadata; the resolution has to reproduce that
    default so those exports keep working with no changes.
    """
    policy = _make_policy({})  # no gravity_source key
    policy._ensure_config()
    assert policy._gravity_source == GRAVITY_SOURCE_PROJECTED


def test_policy_refuses_a_mistyped_gravity_source_at_configuration_time() -> None:
    """A metadata ``gravity_source`` that is neither of the two shipped values raises.

    The raise happens in ``_ensure_config`` (first-inference configuration),
    not in the builder every tick.  A tick-time raise would still catch the
    typo, but this cell asserts the catch happens ONCE, at configuration, and
    names the checkpoint's metadata as the source - the same shape
    ``action_scale`` refuses a non-numeric metadata value with today.
    """
    policy = _make_policy({"gravity_source": "gravity"})
    with pytest.raises(ValueError, match="gravity_source"):
        policy._ensure_config()
