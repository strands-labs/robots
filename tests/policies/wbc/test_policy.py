"""Unit tests for :mod:`strands_robots.policies.wbc` - no GPU, no onnxruntime.

These tests exercise :class:`WBCPolicy` against a stubbed ONNX session
(injected via the ``allow_missing_models`` seam), so they run on any developer
machine. The integration test under ``tests_integ/policies/wbc/`` covers the
live ONNX + downloaded-checkpoint path.

Issue #466 acceptance criteria pinned here:

* ``create_policy("wbc", ...)`` / ``create_policy("sonic", ...)`` round-trip via
  the factory + registry.
* Raises ``RuntimeError`` on missing ``onnxruntime`` / checkpoint; never emits
  zero/garbage torques silently.
* ``requires_images is False``.
* Observation builder produces the exact 86-dim layout; PD-control + quat
  helpers match hand-computed values; action shape is 15-dim; history deque
  length honoured; reset clears history.
* Explicit ``unitree_g1`` actuator <-> 15-dim WBC mapping table + validation.
"""

from __future__ import annotations

import asyncio
import json
import math

import numpy as np
import pytest

from strands_robots.policies import Policy, create_policy, list_providers
from strands_robots.policies.wbc import WBC_G1_LEG_WAIST_JOINTS, WBCConfig, WBCPolicy
from strands_robots.policies.wbc.control import (
    compute_targets,
    pd_control,
    projected_gravity,
    quat_rotate_inverse,
)
from strands_robots.policies.wbc.observation import ObservationHistory, build_single_frame

# ---------------------------------------------------------------------------
# Stub ONNX session
# ---------------------------------------------------------------------------

_N = 15  # leg + waist DOFs


class _StubInput:
    name = "obs"


class _StubSession:
    """Minimal stand-in for ``onnxruntime.InferenceSession``.

    Records the observation width it was fed and returns a fixed-shape
    ``(1, num_actions)`` output so the policy's unpack path is exercised
    without onnxruntime installed.
    """

    def __init__(self, num_actions: int = _N, fill: float = 0.04) -> None:
        self.num_actions = num_actions
        self.fill = fill
        self.calls: list[np.ndarray] = []

    def get_inputs(self) -> list[_StubInput]:
        return [_StubInput()]

    def run(self, output_names, feed):  # type: ignore[no-untyped-def]
        (arr,) = feed.values()
        self.calls.append(np.asarray(arr))
        return [np.full((1, self.num_actions), self.fill, dtype=np.float32)]


def _g1_keys() -> list[str]:
    """Full 29-DOF G1 key order: 15 leg+waist then 14 arm joints."""
    return list(WBC_G1_LEG_WAIST_JOINTS) + [f"arm_{i}" for i in range(14)]


def _make_config(**overrides) -> WBCConfig:  # type: ignore[no-untyped-def]
    base = dict(
        policy_path="policy.onnx",
        num_actions=_N,
        command_dim=7,
        single_obs_dim=86,
        obs_history_len=1,
        default_angles=[0.1] * _N,
        kps=[100.0] * _N,
        kds=[2.0] * _N,
        action_scale=0.25,
    )
    base.update(overrides)
    return WBCConfig(**base)  # type: ignore[arg-type]


def _make_policy(walk: bool = True, **cfg_overrides) -> WBCPolicy:  # type: ignore[no-untyped-def]
    p = WBCPolicy(config=_make_config(**cfg_overrides), walk=walk, allow_missing_models=True)
    p.policy_session = _StubSession()
    if walk:
        p.walk_session = _StubSession()
    p.set_robot_state_keys(_g1_keys())
    return p


# ---------------------------------------------------------------------------
# Control / quaternion math (hand-computed)
# ---------------------------------------------------------------------------


class TestControlMath:
    def test_pd_control_hand_value(self) -> None:
        tau = pd_control(
            np.array([1.0, 2.0]),
            np.array([0.0, 0.0]),
            np.array([10.0, 10.0]),
            np.array([0.0, 0.0]),
            np.array([0.5, 0.0]),
            np.array([2.0, 2.0]),
        )
        # (1-0)*10 + (0-0.5)*2 = 10 - 1 = 9 ; (2-0)*10 + 0 = 20
        assert np.allclose(tau, [9.0, 20.0])

    def test_pd_control_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="share one shape"):
            pd_control(
                np.array([1.0, 2.0]),
                np.array([0.0]),
                np.array([1.0, 1.0]),
                np.array([0.0, 0.0]),
                np.array([0.0, 0.0]),
                np.array([1.0, 1.0]),
            )

    def test_compute_targets_hand_value(self) -> None:
        out = compute_targets(np.array([0.1, 0.2]), np.array([0.04, -0.04]), 0.25)
        # 0.1 + 0.25*0.04 = 0.11 ; 0.2 + 0.25*-0.04 = 0.19
        assert np.allclose(out, [0.11, 0.19])

    def test_quat_identity_is_noop(self) -> None:
        out = quat_rotate_inverse(np.array([1.0, 0, 0, 0]), np.array([1.0, 2.0, 3.0]))
        assert np.allclose(out, [1.0, 2.0, 3.0])

    def test_quat_yaw90_inverse(self) -> None:
        c, s = math.cos(math.pi / 4), math.sin(math.pi / 4)
        out = quat_rotate_inverse(np.array([c, 0, 0, s]), np.array([1.0, 0.0, 0.0]))
        # inverse of a +90deg yaw maps world +x -> body -y
        assert np.allclose(out, [0.0, -1.0, 0.0], atol=1e-9)

    def test_projected_gravity_upright(self) -> None:
        assert np.allclose(projected_gravity(np.array([1.0, 0, 0, 0])), [0.0, 0.0, -1.0])

    def test_projected_gravity_is_unit(self) -> None:
        c, s = math.cos(0.3), math.sin(0.3)
        pg = projected_gravity(np.array([c, 0, s, 0]))
        assert np.isclose(np.linalg.norm(pg), 1.0)

    def test_quat_zero_norm_raises(self) -> None:
        with pytest.raises(ValueError, match="zero norm"):
            quat_rotate_inverse(np.array([0.0, 0, 0, 0]), np.array([1.0, 0, 0]))


# ---------------------------------------------------------------------------
# Observation layout
# ---------------------------------------------------------------------------


class TestObservationLayout:
    def test_exact_86_dim_layout(self) -> None:
        cfg = _make_config()
        frame = build_single_frame(
            cfg,
            command=np.array([0.5, 0.0, 0.0]),  # short -> zero-padded to 7
            base_ang_vel=np.array([1.0, 2.0, 3.0]),
            proj_gravity=np.array([0.0, 0.0, -1.0]),
            qj=np.array([0.2] * _N),
            dqj=np.array([0.0] * _N),
            prev_action=np.array([0.0] * _N),
        )
        assert frame.shape == (86,)
        # command head: vx at 0, padded zeros at 3..7
        assert frame[0] == 0.5
        assert np.allclose(frame[3:7], 0.0)
        # base ang vel scaled by obs_scales.ang_vel (0.25): index 7 = 1.0*0.25
        assert np.isclose(frame[7], 0.25)
        # projected gravity unscaled at [10:13]
        assert np.allclose(frame[10:13], [0.0, 0.0, -1.0])
        # qj - defaults scaled by dof_pos (1.0): (0.2-0.1) at index 13
        assert np.isclose(frame[13], 0.1)
        # tail padding (gait/style) is zero: end = 7+6+45 = 58
        assert np.allclose(frame[58:], 0.0)

    def test_command_overflow_raises(self) -> None:
        cfg = _make_config(command_dim=3)
        with pytest.raises(ValueError, match="exceeds command_dim"):
            build_single_frame(
                cfg,
                command=np.array([0.1, 0.2, 0.3, 0.4]),
                base_ang_vel=np.zeros(3),
                proj_gravity=np.zeros(3),
                qj=np.zeros(_N),
                dqj=np.zeros(_N),
                prev_action=np.zeros(_N),
            )

    def test_history_warm_start_and_width(self) -> None:
        cfg = _make_config(obs_history_len=3)
        assert cfg.num_obs == 86 * 3
        hist = ObservationHistory(cfg)
        frame = np.arange(86, dtype=np.float64)
        stacked = hist.push(frame)
        assert stacked.shape == (258,)
        assert len(hist) == 3  # warm-filled with copies of the first frame
        assert np.allclose(stacked[:86], stacked[86:172])

    def test_history_reset_clears(self) -> None:
        hist = ObservationHistory(_make_config(obs_history_len=2))
        hist.push(np.zeros(86))
        assert len(hist) == 2
        hist.reset()
        assert len(hist) == 0


# ---------------------------------------------------------------------------
# WBCConfig
# ---------------------------------------------------------------------------


class TestWBCConfig:
    def test_num_obs(self) -> None:
        assert _make_config(single_obs_dim=86, obs_history_len=4).num_obs == 344

    def test_wrong_vector_length_raises(self) -> None:
        with pytest.raises(ValueError, match="kps has length"):
            WBCConfig(policy_path="p.onnx", num_actions=15, kps=[1.0, 2.0])

    def test_from_dict_requires_policy_path(self) -> None:
        with pytest.raises(ValueError, match="policy_path"):
            WBCConfig.from_dict({"num_actions": 15})

    def test_from_file_round_trip(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        p = tmp_path / "wbc.json"
        p.write_text(json.dumps({"policy_path": "policy.onnx", "num_actions": 15, "obs_history_len": 2}))
        cfg = WBCConfig.from_file(str(p))
        assert cfg.num_actions == 15 and cfg.obs_history_len == 2

    def test_from_file_missing_raises(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(FileNotFoundError):
            WBCConfig.from_file(str(tmp_path / "nope.json"))

    def test_command_dim_floor(self) -> None:
        with pytest.raises(ValueError, match="command_dim must be >= 3"):
            WBCConfig(policy_path="p.onnx", command_dim=2)


# ---------------------------------------------------------------------------
# WBCPolicy behaviour
# ---------------------------------------------------------------------------


class TestWBCPolicy:
    def test_requires_images_false(self) -> None:
        assert _make_policy().requires_images is False

    def test_provider_name(self) -> None:
        assert _make_policy().provider_name == "wbc"

    def test_get_actions_returns_single_15dim_dict(self) -> None:
        p = _make_policy()
        obs = {k: 0.2 for k in _g1_keys()}
        actions = asyncio.run(p.get_actions(obs, "", target_velocity=[0.5, 0.0, 0.0]))
        assert len(actions) == 1  # closed-loop per-tick, not a chunk
        a = actions[0]
        assert set(a.keys()) == set(WBC_G1_LEG_WAIST_JOINTS)
        # target = default(0.1) + action_scale(0.25)*stub_fill(0.04) = 0.11
        assert np.isclose(a["left_hip_pitch_joint"], 0.11)

    def test_action_keys_in_wbc_order(self) -> None:
        p = _make_policy()
        obs = {k: 0.0 for k in _g1_keys()}
        actions = asyncio.run(p.get_actions(obs, "", target_velocity=[0.1, 0.0, 0.0]))
        assert list(actions[0].keys()) == list(WBC_G1_LEG_WAIST_JOINTS)

    def test_walk_session_used_for_nonzero_command(self) -> None:
        p = _make_policy(walk=True)
        obs = {k: 0.0 for k in _g1_keys()}
        asyncio.run(p.get_actions(obs, "", target_velocity=[0.5, 0.0, 0.0]))
        assert p.walk_session.calls, "walk session should run for a nonzero command"
        assert not p.policy_session.calls, "main session should not run when walking"

    def test_main_session_used_for_zero_command(self) -> None:
        p = _make_policy(walk=True)
        obs = {k: 0.0 for k in _g1_keys()}
        asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))
        assert p.policy_session.calls, "main session should run when standing"
        assert not p.walk_session.calls

    def test_constructor_default_velocity_used_when_no_kwarg(self) -> None:
        p = WBCPolicy(config=_make_config(), walk=True, target_velocity=[0.3, 0.0, 0.0], allow_missing_models=True)
        p.policy_session = _StubSession()
        p.walk_session = _StubSession()
        p.set_robot_state_keys(_g1_keys())
        obs = {k: 0.0 for k in _g1_keys()}
        asyncio.run(p.get_actions(obs, ""))  # no per-call velocity
        assert p.walk_session.calls, "constructor default_command should drive the walk session"

    def test_observation_width_fed_to_session(self) -> None:
        p = _make_policy(walk=False, obs_history_len=2)
        obs = {k: 0.0 for k in _g1_keys()}
        asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))
        fed = p.policy_session.calls[0]
        assert fed.shape == (1, 86 * 2)
        assert fed.dtype == np.float32

    def test_reset_clears_history_and_prev_action(self) -> None:
        p = _make_policy()
        obs = {k: 0.2 for k in _g1_keys()}
        asyncio.run(p.get_actions(obs, "", target_velocity=[0.5, 0.0, 0.0]))
        assert not np.allclose(p._prev_action, 0.0)
        p.reset()
        assert np.allclose(p._prev_action, 0.0)
        assert len(p._history) == 0

    def test_prev_action_feeds_back(self) -> None:
        """The previous raw action lands in the next frame's prev-action slot."""
        p = _make_policy(walk=False)
        obs = {k: 0.0 for k in _g1_keys()}
        asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))
        # second call: the frame's prev-action block should equal the stub's
        # raw output (0.04), proving feedback. The block sits at
        # [c+6+2n : c+6+3n] = [7+6+30 : 7+6+45] = [43:58] for c=7, n=15.
        asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))
        fed = p.policy_session.calls[1][0]  # (num_obs,)
        prev_block = fed[43 : 43 + 15]
        assert np.allclose(prev_block, 0.04)

    def test_no_session_raises_not_silent_zeros(self) -> None:
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)
        p.set_robot_state_keys(_g1_keys())
        obs = {k: 0.0 for k in _g1_keys()}
        with pytest.raises(RuntimeError, match="no ONNX session"):
            asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# Actuator <-> WBC mapping table + validation
# ---------------------------------------------------------------------------


class TestActuatorMapping:
    def test_mapping_table_is_15_leg_waist(self) -> None:
        assert len(WBC_G1_LEG_WAIST_JOINTS) == 15
        assert WBC_G1_LEG_WAIST_JOINTS[0] == "left_hip_pitch_joint"
        assert WBC_G1_LEG_WAIST_JOINTS[12:] == ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")

    def test_bad_ordering_raises(self) -> None:
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)
        with pytest.raises(ValueError, match="expected G1 leg\\+waist order"):
            p.set_robot_state_keys([f"wrong_{i}" for i in range(20)])

    def test_too_few_keys_raises(self) -> None:
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)
        with pytest.raises(ValueError, match="at least 15"):
            p.set_robot_state_keys(list(WBC_G1_LEG_WAIST_JOINTS[:10]))

    def test_exact_15_keys_accepted(self) -> None:
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)
        p.set_robot_state_keys(list(WBC_G1_LEG_WAIST_JOINTS))  # exactly 15 is fine


# ---------------------------------------------------------------------------
# compute_torques public helper
# ---------------------------------------------------------------------------


class TestComputeTorques:
    def test_pd_law(self) -> None:
        p = _make_policy()
        tau = p.compute_torques(
            np.array([0.11] * _N),
            np.array([0.2] * _N),
            np.zeros(_N),
        )
        # (0.11 - 0.2) * kp(100) + (0 - 0) * kd = -9.0
        assert np.allclose(tau, [(0.11 - 0.2) * 100.0] * _N)


# ---------------------------------------------------------------------------
# Error paths: missing dep / checkpoint
# ---------------------------------------------------------------------------


class TestErrorPaths:
    def test_bad_target_velocity_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            WBCPolicy(config=_make_config(), target_velocity=[float("nan"), 0, 0], allow_missing_models=True)

    def test_short_target_velocity_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            WBCPolicy(config=_make_config(), target_velocity=[0.5], allow_missing_models=True)

    def test_missing_checkpoint_or_dep_raises_runtime_error(self) -> None:
        """Either onnxruntime is absent (ImportError->RuntimeError) or the
        checkpoint file is missing; both must surface as RuntimeError, never a
        silent zero-torque policy."""
        with pytest.raises(RuntimeError):
            WBCPolicy(
                checkpoint="/nonexistent/checkpoint/dir",
                config=_make_config(policy_path="/nonexistent/policy.onnx"),
                allow_missing_models=False,
            )


class TestCheckpointResolution:
    """The local-path | HF-download | cache resolution (issue #466)."""

    def test_existing_local_path_returned_unchanged(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        d = tmp_path / "ckpt"
        d.mkdir()
        assert WBCPolicy._maybe_download_checkpoint(str(d)) == str(d)

    def test_none_passes_through(self) -> None:
        assert WBCPolicy._maybe_download_checkpoint(None) is None

    def test_onnx_file_path_not_treated_as_hf_id(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        # A non-existent .onnx path is not an HF id; returned unchanged so the
        # path resolver surfaces a clear not-found error (not a download attempt).
        bogus = str(tmp_path / "missing" / "policy.onnx")
        assert WBCPolicy._maybe_download_checkpoint(bogus) == bogus

    def test_hf_id_without_hub_raises_runtime_error(self, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        # An org/repo id with huggingface_hub absent must raise RuntimeError
        # (not silently proceed). Simulate the missing dep via require_optional.
        import strands_robots.policies.wbc.policy as wbc_policy

        def _boom(*a, **k):  # type: ignore[no-untyped-def]
            raise ImportError("no huggingface_hub")

        monkeypatch.setattr(wbc_policy, "require_optional", _boom)
        with pytest.raises(RuntimeError, match="HuggingFace model id"):
            WBCPolicy._maybe_download_checkpoint("nvidia/GEAR-SONIC")


# ---------------------------------------------------------------------------
# Factory / registry resolution
# ---------------------------------------------------------------------------


class TestFactoryResolution:
    def test_create_policy_by_canonical_name(self) -> None:
        p = create_policy("wbc", config=_make_config(), allow_missing_models=True)
        assert isinstance(p, WBCPolicy)
        assert isinstance(p, Policy)

    def test_create_policy_by_sonic_shorthand(self) -> None:
        p = create_policy("sonic", config=_make_config(), allow_missing_models=True)
        assert isinstance(p, WBCPolicy)

    def test_wbc_in_list_providers(self) -> None:
        assert "wbc" in list_providers()

    def test_requires_images_false_via_factory(self) -> None:
        p = create_policy("wbc", config=_make_config(), allow_missing_models=True)
        assert p.requires_images is False
