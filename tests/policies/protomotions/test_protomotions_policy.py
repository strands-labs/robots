"""Unit tests for :class:`ProtoMotionsPolicy` - no ONNX weights, no CUDA.

Uses the :class:`ProtoMotionsSession` injection seam so the whole test suite
runs in CPU-only CI in single-digit seconds.
"""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from strands_robots.policies.factory import create_policy, import_policy_class
from strands_robots.policies.protomotions import (
    GTP_G1_JOINT_NAMES,
    MotionPlayer,
    ProtoMotionsConfig,
    ProtoMotionsPolicy,
    ProtoMotionsSession,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _StubSession:
    """A deterministic stub tracker: outputs a known joint-target vector."""

    def __init__(self, targets: np.ndarray | None = None) -> None:
        self._targets = targets if targets is not None else np.linspace(-0.1, 0.1, 29, dtype=np.float32)
        self.call_count = 0
        self.last_inputs: dict[str, np.ndarray] | None = None

    def run(self, output_names, inputs):
        self.call_count += 1
        self.last_inputs = {k: v.copy() for k, v in inputs.items()}
        n = 29
        actions = self._targets.reshape(1, n).astype(np.float32)
        joint_pos = self._targets.reshape(1, n).astype(np.float32)
        stiff = np.full((1, n), 40.0, dtype=np.float32)
        damp = np.full((1, n), 2.5, dtype=np.float32)
        return [actions, joint_pos, stiff, damp]


def _make_flat_cache(num_frames: int = 20) -> dict:
    nb, nd = 33, 29
    return {
        "dof_pos": np.zeros((num_frames, nd), dtype=np.float32),
        "dof_vel": np.zeros((num_frames, nd), dtype=np.float32),
        "body_rot": np.tile(
            np.array([0, 0, 0, 1], dtype=np.float32),
            (num_frames, nb, 1),
        ),
        "body_pos": np.zeros((num_frames, nb, 3), dtype=np.float32),
        "body_vel": np.zeros((num_frames, nb, 3), dtype=np.float32),
        "body_ang_vel": np.zeros((num_frames, nb, 3), dtype=np.float32),
        "control_dt": 0.02,
        "num_frames": num_frames,
    }


@pytest.fixture
def stub_session() -> _StubSession:
    return _StubSession()


@pytest.fixture
def flat_motion() -> MotionPlayer:
    return MotionPlayer(_make_flat_cache())


@pytest.fixture
def policy(stub_session: _StubSession, flat_motion: MotionPlayer) -> ProtoMotionsPolicy:
    return ProtoMotionsPolicy(session=stub_session, motion=flat_motion)


def _minimal_obs_kwargs() -> dict:
    return {
        "anchor_rot_xyzw": np.array([0, 0, 0, 1], dtype=np.float32),
        "root_ang_vel_local": np.zeros(3, dtype=np.float32),
        "dof_pos": np.zeros(29, dtype=np.float32),
        "dof_vel": np.zeros(29, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_registry_resolves_class():
    """The factory resolves ``"protomotions"`` to the correct class."""
    cls = import_policy_class("protomotions")
    assert cls is ProtoMotionsPolicy


def test_registry_shorthands_resolve_class():
    """Every shorthand documented in ``policies.json`` resolves too."""
    for shorthand in ("gtp", "gtp_g1", "protomotions_g1"):
        cls = import_policy_class(shorthand)
        assert cls is ProtoMotionsPolicy, f"shorthand {shorthand!r} failed"


def test_factory_creates_policy(stub_session, flat_motion):
    """``create_policy`` builds a working policy with injected knobs."""
    p = create_policy("protomotions", session=stub_session, motion=flat_motion)
    assert isinstance(p, ProtoMotionsPolicy)
    assert p.provider_name == "protomotions"
    assert p.num_dofs == 29


# ---------------------------------------------------------------------------
# Construction contract
# ---------------------------------------------------------------------------


def test_construction_requires_onnx_path_or_session():
    """Building without ``onnx_path`` OR ``session`` is a clean refusal."""
    with pytest.raises(ValueError, match="onnx_path|session"):
        ProtoMotionsPolicy()


def test_construction_history_length_positive():
    """``history_length`` must be at least 1."""
    with pytest.raises(ValueError, match="history_length"):
        ProtoMotionsPolicy(session=_StubSession(), history_length=0)


def test_construction_history_length_shapes_buffer():
    """The history buffer's shape matches ``history_length`` x ``num_dofs``."""
    p = ProtoMotionsPolicy(session=_StubSession(), history_length=4)
    assert p._action_history.shape == (4, 29)


def test_load_motion_accepts_cache_dict(stub_session):
    """Passing a plain dict to ``motion=`` wraps it in a MotionPlayer."""
    p = ProtoMotionsPolicy(session=stub_session, motion=_make_flat_cache())
    assert p._motion_player is not None
    assert p._motion_player.total_frames == 20


def test_load_motion_rejects_bad_type(stub_session):
    """Non-MotionPlayer/dict/str raises ``TypeError`` from ``load_motion``."""
    p = ProtoMotionsPolicy(session=stub_session)
    with pytest.raises(TypeError, match="MotionPlayer, cache dict, or path"):
        p.load_motion(42)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Non-VLA contract
# ---------------------------------------------------------------------------


def test_requires_images_is_false(policy):
    """The tracker is proprioceptive - no cameras."""
    assert policy.requires_images is False


def test_config_defaults_match_gtp_g1():
    """Default config matches the pinned G1 GTP checkpoint."""
    cfg = ProtoMotionsConfig()
    assert cfg.num_dofs == 29
    assert cfg.num_bodies == 33
    assert cfg.anchor_body_index == 16  # torso_link
    assert cfg.root_body_index == 0  # pelvis
    assert cfg.control_dt == 0.02
    assert cfg.future_step_indices == (1, 2, 4, 8)


# ---------------------------------------------------------------------------
# get_actions behaviour
# ---------------------------------------------------------------------------


def test_get_actions_without_motion_raises():
    """Calling ``get_actions`` before a motion is loaded is a loud refusal."""
    p = ProtoMotionsPolicy(session=_StubSession())
    with pytest.raises(RuntimeError, match="no reference motion loaded"):
        asyncio.run(p.get_actions({}, "walk"))


def test_get_actions_returns_29_joint_dict(policy):
    """A successful call returns a single-frame list of 29-joint dict."""
    actions = asyncio.run(policy.get_actions({}, "walk", **_minimal_obs_kwargs()))
    assert isinstance(actions, list) and len(actions) == 1
    d = actions[0]
    assert len(d) == 29
    assert set(d.keys()) == set(GTP_G1_JOINT_NAMES)
    assert all(isinstance(v, float) for v in d.values())


def test_get_actions_advances_frame_cursor(policy):
    """Each call advances the playhead by exactly one frame."""
    for expected in range(1, 4):
        asyncio.run(policy.get_actions({}, "", **_minimal_obs_kwargs()))
        assert policy._frame_cursor == expected


def test_get_actions_fills_action_history(stub_session, flat_motion):
    """The rolling history buffer keeps the last N actions."""
    p = ProtoMotionsPolicy(session=stub_session, motion=flat_motion, history_length=3)
    # Before first call - all zeros.
    assert (p._action_history == 0).all()
    asyncio.run(p.get_actions({}, "", **_minimal_obs_kwargs()))
    # After one call - last row is the stub's action vector, earlier still zero.
    assert (p._action_history[:-1] == 0).all()
    assert not (p._action_history[-1] == 0).all()


def test_get_actions_swaps_motion_via_kwarg(stub_session, flat_motion):
    """Passing ``motion=`` on a per-call kwarg swaps the reference clip."""
    p = ProtoMotionsPolicy(session=stub_session, motion=flat_motion)
    # Advance the cursor a bit.
    for _ in range(3):
        asyncio.run(p.get_actions({}, "", **_minimal_obs_kwargs()))
    assert p._frame_cursor == 3
    # Now swap in a longer motion - cursor resets, history clears.
    new_motion = MotionPlayer(_make_flat_cache(num_frames=50))
    asyncio.run(p.get_actions({}, "", motion=new_motion, **_minimal_obs_kwargs()))
    # After a fresh motion swap and one call, cursor should be exactly 1.
    assert p._frame_cursor == 1


def test_get_actions_uses_kwargs_over_obs(stub_session, flat_motion):
    """Explicit ``dof_pos`` / ``dof_vel`` kwargs win over the obs dict."""
    p = ProtoMotionsPolicy(session=stub_session, motion=flat_motion)
    # Obs dict has malformed per-joint entries; kwargs should be used instead.
    obs = {"garbage": 42}
    asyncio.run(p.get_actions(obs, "", **_minimal_obs_kwargs()))
    assert stub_session.call_count == 1
    dof_pos_in = stub_session.last_inputs["current_dof_pos"]
    assert dof_pos_in.shape == (1, 29)


def test_get_actions_forwards_correct_input_shapes(policy, stub_session):
    """The 8 ONNX inputs arrive in the right shapes."""
    asyncio.run(policy.get_actions({}, "", **_minimal_obs_kwargs()))
    ins = stub_session.last_inputs
    assert ins is not None
    assert ins["current_anchor_rot"].shape == (1, 4)
    assert ins["current_dof_pos"].shape == (1, 29)
    assert ins["current_dof_vel"].shape == (1, 29)
    assert ins["current_root_local_ang_vel"].shape == (1, 3)
    assert ins["historical_processed_actions"].shape == (1, 1, 29)
    assert ins["mimic_future_anchor_rot"].shape == (1, 4, 4)
    assert ins["mimic_future_dof_pos"].shape == (1, 4, 29)
    assert ins["mimic_future_dof_vel"].shape == (1, 4, 29)


def test_get_actions_reset_zeroes_state(policy):
    """``reset()`` returns the policy to a clean per-episode state."""
    for _ in range(5):
        asyncio.run(policy.get_actions({}, "", **_minimal_obs_kwargs()))
    assert policy._frame_cursor > 0
    policy.reset()
    assert policy._frame_cursor == 0
    assert (policy._action_history == 0).all()


# ---------------------------------------------------------------------------
# Robot state-key wiring
# ---------------------------------------------------------------------------


def test_set_robot_state_keys_accepts_full_g1(policy):
    """A full 29-joint G1 key list is accepted."""
    keys = list(GTP_G1_JOINT_NAMES) + ["free_joint"]
    policy.set_robot_state_keys(keys)
    assert policy._robot_state_keys == keys


def test_set_robot_state_keys_refuses_missing_joints(policy):
    """A partial key list is a loud refusal, not a silent fallback."""
    with pytest.raises(ValueError, match="missing expected G1 joints"):
        policy.set_robot_state_keys(list(GTP_G1_JOINT_NAMES[:20]))


# ---------------------------------------------------------------------------
# Session Protocol adherence
# ---------------------------------------------------------------------------


def test_stub_satisfies_protomotions_session_protocol():
    """The test stub is structurally a :class:`ProtoMotionsSession`."""
    assert isinstance(_StubSession(), ProtoMotionsSession)
