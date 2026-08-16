"""Contract pins for driving :class:`ProtoMotionsPolicy` from the runtime.

The tracker needs two signals that are not joint state: the WORLD orientation
of its anchor link and the floating base's body-frame angular velocity. Both
reach a policy through the observation feed, so the names the policy reads have
to be the names the runtime writes. These tests pin that agreement, and pin the
frame itself: ``base_quat`` is the pelvis, the anchor is ``torso_link``, and on
a G1 sweeping its waist the two differ by up to 42 degrees, so accepting the
base as a stand-in would be a silently wrong input rather than an approximation.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.base import Policy
from strands_robots.policies.protomotions import (
    GTP_G1_JOINT_NAMES,
    ProtoMotionsConfig,
    ProtoMotionsPolicy,
)

_NUM_BODIES = 33
_NUM_DOFS = 29


class _RecordingSession:
    """Stub tracker that records the ONNX inputs it was handed."""

    def __init__(self, targets: np.ndarray | None = None) -> None:
        self.targets = np.zeros(_NUM_DOFS, dtype=np.float32) if targets is None else targets.astype(np.float32)
        self.inputs: list[dict[str, np.ndarray]] = []

    def run(self, output_names: list[str] | None, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.inputs.append({k: np.asarray(v).copy() for k, v in inputs.items()})
        row = self.targets.reshape(1, _NUM_DOFS)
        named = {
            "actions": row,
            "joint_pos_targets": row,
            "stiffness_targets": np.full((1, _NUM_DOFS), 40.0, dtype=np.float32),
            "damping_targets": np.full((1, _NUM_DOFS), 2.5, dtype=np.float32),
        }
        requested = list(output_names or named)
        return [named[name] for name in requested]


def _flat_cache(num_frames: int = 40) -> dict[str, Any]:
    return {
        "dof_pos": np.zeros((num_frames, _NUM_DOFS), dtype=np.float32),
        "dof_vel": np.zeros((num_frames, _NUM_DOFS), dtype=np.float32),
        "body_rot": np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            (num_frames, _NUM_BODIES, 1),
        ),
        "body_pos": np.zeros((num_frames, _NUM_BODIES, 3), dtype=np.float32),
        "body_vel": np.zeros((num_frames, _NUM_BODIES, 3), dtype=np.float32),
        "body_ang_vel": np.zeros((num_frames, _NUM_BODIES, 3), dtype=np.float32),
        "control_dt": 0.02,
        "num_frames": num_frames,
    }


def _runtime_observation(
    *,
    anchor_quat_wxyz: tuple[float, float, float, float],
    base_quat_wxyz: tuple[float, float, float, float],
    anchor_body: str = "torso_link",
    with_velocities: bool = True,
) -> dict[str, Any]:
    """Build an observation shaped exactly like a simulation rollout's."""
    obs: dict[str, Any] = {
        "base_pos": [0.0, 0.0, 0.79],
        "base_quat": list(base_quat_wxyz),
        "base_lin_vel": [0.0, 0.0, 0.0],
        "base_ang_vel": [0.1, -0.2, 0.3],
        f"body.{anchor_body}.quat": list(anchor_quat_wxyz),
    }
    for i, name in enumerate(GTP_G1_JOINT_NAMES):
        obs[name] = 0.01 * i
        if with_velocities:
            obs[f"{name}.vel"] = 0.001 * i
    return obs


def _policy(session: _RecordingSession, **kwargs: Any) -> ProtoMotionsPolicy:
    return ProtoMotionsPolicy(session=session, motion=_flat_cache(), **kwargs)


def _act(policy: ProtoMotionsPolicy, obs: dict[str, Any]) -> dict[str, float]:
    return asyncio.run(policy.get_actions(obs, ""))[0]


# ---------------------------------------------------------------------------
# The anchor link the policy cannot derive
# ---------------------------------------------------------------------------


def test_policy_declares_the_anchor_body_the_runtime_must_supply() -> None:
    """The tracker's anchor link is declared, so the runtime merges its pose."""
    policy = _policy(_RecordingSession())
    assert policy.required_bodies == ("torso_link",)


def test_anchor_rotation_is_read_from_the_anchor_body_not_the_floating_base() -> None:
    """``body.torso_link.quat`` wins over ``base_quat`` when both are present.

    A G1 with a 0.6 rad waist yaw puts 42 degrees between the two, so reading
    the wrong one is a wrong input, not a small error.
    """
    session = _RecordingSession()
    policy = _policy(session)
    # 90 deg about z as the anchor; identity as the floating base.
    anchor_wxyz = (0.7071068, 0.0, 0.0, 0.7071068)
    _act(
        policy,
        _runtime_observation(anchor_quat_wxyz=anchor_wxyz, base_quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
    )
    sent = session.inputs[0]["current_anchor_rot"].reshape(4)
    # The ONNX input convention is xyzw.
    np.testing.assert_allclose(sent, [0.0, 0.0, 0.7071068, 0.7071068], atol=1e-6)


def test_a_rollout_observation_needs_no_hand_assembled_kwargs() -> None:
    """Every ONNX input resolves from the runtime's own observation keys."""
    session = _RecordingSession()
    policy = _policy(session)
    action = _act(
        policy,
        _runtime_observation(
            anchor_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            base_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        ),
    )
    assert set(action) == set(GTP_G1_JOINT_NAMES)
    sent = session.inputs[0]
    # base_ang_vel is the root freejoint's qvel[3:6] - already body-frame.
    np.testing.assert_allclose(sent["current_root_local_ang_vel"].reshape(3), [0.1, -0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(
        sent["current_dof_pos"].reshape(_NUM_DOFS),
        [0.01 * i for i in range(_NUM_DOFS)],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sent["current_dof_vel"].reshape(_NUM_DOFS),
        [0.001 * i for i in range(_NUM_DOFS)],
        atol=1e-6,
    )


def test_the_floating_base_is_refused_as_a_stand_in_for_a_distinct_anchor() -> None:
    """With only ``base_quat`` on the obs, the policy refuses and says why."""
    policy = _policy(_RecordingSession())
    obs = _runtime_observation(anchor_quat_wxyz=(1.0, 0.0, 0.0, 0.0), base_quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    del obs["body.torso_link.quat"]
    with pytest.raises(KeyError) as excinfo:
        _act(policy, obs)
    message = str(excinfo.value)
    assert "torso_link" in message
    assert "body.torso_link.quat" in message
    assert "base_quat" in message


def test_the_floating_base_is_accepted_when_the_config_anchors_on_the_root() -> None:
    """A config whose anchor IS the root can legitimately read ``base_quat``."""
    session = _RecordingSession()
    root_anchored = dataclasses.replace(ProtoMotionsConfig(), anchor_body_index=ProtoMotionsConfig().root_body_index)
    policy = _policy(session)
    policy._config = root_anchored  # noqa: SLF001 - exercising a non-default config
    assert policy.required_bodies == (root_anchored.root_body_name,)
    obs = _runtime_observation(
        anchor_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        base_quat_wxyz=(0.7071068, 0.0, 0.0, 0.7071068),
    )
    del obs["body.torso_link.quat"]
    _act(policy, obs)
    np.testing.assert_allclose(
        session.inputs[0]["current_anchor_rot"].reshape(4),
        [0.0, 0.0, 0.7071068, 0.7071068],
        atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Per-episode reset
# ---------------------------------------------------------------------------


def test_reset_accepts_the_seed_the_runtime_forwards() -> None:
    """``reset(seed=...)`` is how the runtime starts an episode; it must bind.

    A ``reset(self)`` override raises ``TypeError`` there. The runtime catches
    that and continues, so the playhead would silently carry over and every
    episode after the first would resume mid-clip.
    """

    # Compare the parameter shape, not the annotation text: this module and
    # the policy module both use postponed annotations, so a raw Signature
    # comparison would differ on string-vs-object annotations alone.
    def _shape(fn: Any) -> list[tuple[str, Any]]:
        return [(name, param.default) for name, param in inspect.signature(fn).parameters.items()]

    assert _shape(ProtoMotionsPolicy.reset) == _shape(Policy.reset)
    session = _RecordingSession()
    policy = _policy(session)
    obs = _runtime_observation(anchor_quat_wxyz=(1.0, 0.0, 0.0, 0.0), base_quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    for _ in range(5):
        _act(policy, obs)
    policy.reset(seed=1234)
    _act(policy, obs)
    # A reset playhead asks the clip for the same future window as frame 0 did.
    np.testing.assert_allclose(session.inputs[-1]["mimic_future_dof_pos"], session.inputs[0]["mimic_future_dof_pos"])


# ---------------------------------------------------------------------------
# No silently substituted state
# ---------------------------------------------------------------------------


def test_absent_joint_velocities_are_refused_rather_than_zero_filled() -> None:
    """Joint velocity is tracker feedback; zeros would degrade it silently."""
    policy = _policy(_RecordingSession())
    obs = _runtime_observation(
        anchor_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        base_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        with_velocities=False,
    )
    with pytest.raises(KeyError) as excinfo:
        _act(policy, obs)
    # Name the signal that is missing, so the refusal cannot be mistaken for
    # some other absent input.
    assert ".vel" in str(excinfo.value)


def test_one_absent_joint_velocity_does_not_zero_that_joint() -> None:
    """A partial velocity set is refused too, naming the joint that is missing."""
    policy = _policy(_RecordingSession())
    obs = _runtime_observation(anchor_quat_wxyz=(1.0, 0.0, 0.0, 0.0), base_quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    del obs["waist_yaw_joint.vel"]
    with pytest.raises(KeyError) as excinfo:
        _act(policy, obs)
    assert "waist_yaw_joint.vel" in str(excinfo.value)


# ---------------------------------------------------------------------------
# ONNX outputs are named, not positional
# ---------------------------------------------------------------------------


def test_joint_targets_are_read_by_output_name_not_by_position() -> None:
    """A config that declares the outputs in another order still reads targets."""
    targets = np.linspace(-0.3, 0.3, _NUM_DOFS, dtype=np.float32)
    session = _RecordingSession(targets=targets)
    policy = _policy(session)
    policy._config = dataclasses.replace(  # noqa: SLF001 - non-default output order
        ProtoMotionsConfig(),
        onnx_out_names=(
            "stiffness_targets",
            "damping_targets",
            "joint_pos_targets",
            "actions",
        ),
    )
    action = _act(
        policy,
        _runtime_observation(anchor_quat_wxyz=(1.0, 0.0, 0.0, 0.0), base_quat_wxyz=(1.0, 0.0, 0.0, 0.0)),
    )
    np.testing.assert_allclose([action[name] for name in GTP_G1_JOINT_NAMES], targets, atol=1e-6)


def test_a_session_without_joint_pos_targets_is_refused() -> None:
    """An export missing the PD target output is named, not indexed past."""
    session = _RecordingSession()
    policy = _policy(session)
    policy._config = dataclasses.replace(  # noqa: SLF001 - incomplete export
        ProtoMotionsConfig(), onnx_out_names=("actions", "stiffness_targets")
    )
    with pytest.raises(RuntimeError, match="joint_pos_targets"):
        _act(
            policy,
            _runtime_observation(
                anchor_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
                base_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            ),
        )
