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
from typing import Any

import numpy as np
import pytest

from strands_robots.policies import Policy, create_policy, list_providers
from strands_robots.policies.wbc import WBC_G1_ALL_JOINTS, WBC_G1_LEG_WAIST_JOINTS, WBCConfig, WBCPolicy
from strands_robots.policies.wbc import policy as wbc_policy
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

_N = 15  # leg + waist DOFs (controlled / action dim)
_NO = 29  # observed joints (legs + waist + arms) - the qj/dqj block width


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
    """Real MuJoCo G1 joint key order, as ``robot_joint_names`` returns it.

    Verified against the actual robot_descriptions ``g1_mj_description`` model:
    MuJoCo prepends the free/floating-base joint, so the list is
    ``["floating_base_joint", <15 leg+waist>, <14 arm>]`` (the real arm joint
    NAMES, since set_robot_state_keys now resolves the whole-body observed set
    by name). The leg+waist joints are at indices [1:16], NOT [0:15].
    """
    return ["floating_base_joint", *WBC_G1_ALL_JOINTS]


def _make_config(**overrides) -> WBCConfig:  # type: ignore[no-untyped-def]
    # Default to the real G1 layout: 29 observed joints, 15 controlled, 86-wide
    # frame (7 cmd + 3 omega + 3 grav + 29 qj + 29 dqj + 15 action). Tests that
    # want a faster/smaller layout override n_obs_joints + single_obs_dim.
    base = dict(
        policy_path="policy.onnx",
        num_actions=_N,
        n_obs_joints=_NO,
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
        cfg = _make_config()  # n_obs_joints=29, num_actions=15
        frame = build_single_frame(
            cfg,
            command=np.array([0.5, 0.0, 0.0]),  # short -> zero-padded to 7
            base_ang_vel=np.array([1.0, 2.0, 3.0]),
            proj_gravity=np.array([0.0, 0.0, -1.0]),
            qj=np.array([0.2] * _NO),  # 29 observed joints
            dqj=np.array([0.0] * _NO),
            prev_action=np.array([0.0] * _N),  # 15 controlled
        )
        assert frame.shape == (86,)
        # build_single_frame zero-pads a short command: vx at 0, slots 3..7 zero.
        assert frame[0] == 0.5
        assert np.allclose(frame[3:7], 0.0)
        # base ang vel scaled by obs_scales.ang_vel (upstream 0.5): index 7 = 1.0*0.5
        assert np.isclose(frame[7], 1.0 * cfg.obs_scales["ang_vel"])
        assert np.isclose(frame[7], 0.5)
        # projected gravity unscaled at [10:13]
        assert np.allclose(frame[10:13], [0.0, 0.0, -1.0])
        # qj block at [13 : 13+29]; default_angles only covers the first 15
        # (legs+waist), arms get zero default. qj[0]=(0.2-0.1)=0.1 at index 13;
        # qj[15] (first arm) = (0.2-0.0)=0.2 at index 13+15=28.
        assert np.isclose(frame[13], 0.1)
        assert np.isclose(frame[28], 0.2)
        # dqj block at [13+29 : 13+58] = [42:71]; action block at [71:86].
        assert np.allclose(frame[42:71], 0.0)  # dqj all zero here
        assert np.allclose(frame[71:86], 0.0)  # prev_action all zero
        # populated end = 7 + 3 + 3 + 29 + 29 + 15 = 86; no reserved tail remains.
        assert frame.shape[0] == 86

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

    def test_history_zero_warm_start_and_width(self) -> None:
        """Upstream warm-start: the deque is pre-filled with ZERO frames, so the
        first push yields [zeros .. zeros, frame] (oldest-first), NOT copies of
        the first frame. Matches run_mujoco_gear_wbc.py:47-50."""
        cfg = _make_config(obs_history_len=3)
        assert cfg.num_obs == 86 * 3
        hist = ObservationHistory(cfg)
        # Buffer is always full (zero-warm-started) even before any push.
        assert len(hist) == 3
        frame = np.arange(1.0, 87.0, dtype=np.float64)  # all-nonzero, distinct
        stacked = hist.push(frame)
        assert stacked.shape == (258,)
        assert len(hist) == 3
        # Oldest two blocks are the zero warm-start; newest block is `frame`.
        assert np.allclose(stacked[0:86], 0.0)
        assert np.allclose(stacked[86:172], 0.0)
        assert np.allclose(stacked[172:258], frame)

    def test_history_reset_restores_zero_warm_start(self) -> None:
        hist = ObservationHistory(_make_config(obs_history_len=2))
        # Always full (zero-warm-started) - never empties.
        assert len(hist) == 2
        hist.push(np.full(86, 5.0))
        assert len(hist) == 2
        hist.reset()
        assert len(hist) == 2  # re-seeded with zero frames, not emptied
        # After reset, a push again yields [zeros, frame].
        frame = np.arange(1.0, 87.0, dtype=np.float64)
        stacked = hist.push(frame)
        assert np.allclose(stacked[0:86], 0.0)
        assert np.allclose(stacked[86:172], frame)

    def test_history_len3_zero_warm_start_then_rolling_window(self) -> None:
        """End-to-end: with obs_history_len=3, the network input is a 258-wide
        stack [oldest .. newest]. The deque is ZERO-warm-started (upstream), so
        early ticks show zeros in the older slots - NOT copies of the first
        frame. Distinguishes the two by using a NONZERO tick-0 qj."""
        # Zero default_angles so the frame's qj block equals the raw qj (no
        # offset), isolating the warm-start/rolling-window from default subtraction.
        p = _make_policy(walk=False, obs_history_len=3, default_angles=[0.0] * _N)
        obs = {k: 0.0 for k in _g1_keys()}
        for t in range(4):
            for nm in WBC_G1_LEG_WAIST_JOINTS:
                obs[nm] = float(t + 1)  # tick0 qj=1 (NONZERO, so != zero warm-start)
            asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))
        fed = [c[0] for c in p.policy_session.calls]
        assert all(f.shape[0] == 86 * 3 for f in fed)

        def qj0(frame: np.ndarray, block: int) -> float:
            # qj[0] sits at index 13 within each 86-wide block (c=7 + angvel3 + grav3).
            return float(frame[block * 86 + 13])

        # tick 0: ZERO warm-start in the two older blocks; newest is the tick-0
        # frame (qj=1). If warm-fill copied the first frame, all three would be 1.
        assert [qj0(fed[0], b) for b in range(3)] == [0.0, 0.0, 1.0]
        # tick 3: rolling window of the last 3 frames -> oldest=2, mid=3, newest=4.
        assert [qj0(fed[3], b) for b in range(3)] == [2.0, 3.0, 4.0]


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

    def test_from_file_yaml_with_flat_scale_keys(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        """The upstream g1_gear_wbc.yaml uses flat ang_vel_scale/dof_pos_scale/
        dof_vel_scale keys and YAML format. WBCConfig.from_file must load it and
        normalise the flat scales into obs_scales."""
        pytest.importorskip("yaml", reason="pyyaml not installed")
        y = tmp_path / "g1_gear_wbc.yaml"
        y.write_text(
            "policy_path: policy/ft92.onnx\n"
            "walk_policy_path: policy/ft109.onnx\n"
            "num_actions: 15\n"
            "num_obs: 516\n"
            "obs_history_len: 6\n"
            "action_scale: 0.25\n"
            "ang_vel_scale: 0.5\n"
            "dof_pos_scale: 1.0\n"
            "dof_vel_scale: 0.05\n"
            "cmd_scale: [2.0, 2.0, 0.5]\n"
            "height_cmd: 0.74\n"
            "default_angles: [-0.1, 0, 0, 0.3, -0.2, 0, -0.1, 0, 0, 0.3, -0.2, 0, 0, 0, 0]\n"
            "kps: [150,150,150,200,40,40,150,150,150,200,40,40,250,250,250]\n"
            "kds: [2,2,2,4,2,2,2,2,2,4,2,2,5,5,5]\n"
            "simulation_dt: 0.005\n"  # an unknown key the loader must ignore
            "cmd_init: [0.0, 0.0, 0.0]\n"  # another unknown key
        )
        cfg = WBCConfig.from_file(str(y))
        assert cfg.num_obs == 516 and cfg.obs_history_len == 6
        assert cfg.obs_scales == {"ang_vel": 0.5, "dof_pos": 1.0, "dof_vel": 0.05}
        assert cfg.cmd_scale == [2.0, 2.0, 0.5] and cfg.height_cmd == 0.74
        assert len(cfg.default_angles) == 15 and len(cfg.kps) == 15 and len(cfg.kds) == 15

    def test_from_file_unsupported_extension_raises(self, tmp_path) -> None:  # type: ignore[no-untyped-def]
        p = tmp_path / "config.txt"
        p.write_text("policy_path: x")
        with pytest.raises(ValueError, match="unsupported extension"):
            WBCConfig.from_file(str(p))

    def test_explicit_obs_scales_wins_over_flat_keys(self) -> None:
        c = WBCConfig.from_dict(
            {"policy_path": "x", "ang_vel_scale": 0.5, "obs_scales": {"ang_vel": 0.9, "dof_pos": 1.0, "dof_vel": 0.05}}
        )
        assert c.obs_scales["ang_vel"] == 0.9  # explicit map overrides the flat key

    def test_cmd_scale_wrong_length_rejected(self) -> None:
        with pytest.raises(ValueError, match="cmd_scale must have exactly 3"):
            WBCConfig(policy_path="p.onnx", cmd_scale=[2.0, 2.0])


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
        # History is re-seeded to the zero warm-start (always full = maxlen), and
        # the next push must yield the zero warm-start transient again.
        assert len(p._history) == p.config.obs_history_len
        if p.config.obs_history_len == 1:
            # With history_len=1 the single slot is the live frame; verify the
            # zero warm-start by checking a fresh push after reset starts clean.
            stacked = p._history.push(np.full(p.config.single_obs_dim, 7.0))
            assert np.allclose(stacked, 7.0)

    def test_prev_action_feeds_back(self) -> None:
        """The previous raw action lands in the next frame's prev-action slot."""
        p = _make_policy(walk=False)
        obs = {k: 0.0 for k in _g1_keys()}
        asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))
        # second call: the frame's prev-action block should equal the stub's
        # raw output (0.04), proving feedback. The block sits at
        # [c+6+2*no : c+6+2*no+na] = [7+6+58 : 7+6+58+15] = [71:86] for c=7,
        # no=29, na=15.
        asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))
        fed = p.policy_session.calls[1][0]  # (num_obs,)
        prev_block = fed[71 : 71 + 15]
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

    def test_resolves_leg_waist_by_name_with_floating_base_prefix(self) -> None:
        """REGRESSION: the real sim joint list leads with 'floating_base_joint',
        so the leg+waist joints are at [1:16], not [0:15]. set_robot_state_keys
        must resolve them by name and still accept the list."""
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)
        p.set_robot_state_keys(_g1_keys())  # ["floating_base_joint", <15>, <14 arm>]
        assert tuple(p._wbc_joint_names) == WBC_G1_LEG_WAIST_JOINTS
        # The free-base joint and arm joints are excluded from what WBC drives.
        assert "floating_base_joint" not in p._wbc_joint_names

    def test_missing_joint_raises_listing_missing(self) -> None:
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)
        with pytest.raises(ValueError, match="missing expected G1"):
            p.set_robot_state_keys([f"wrong_{i}" for i in range(20)])

    def test_partial_leg_waist_raises(self) -> None:
        # Only the first 10 leg+waist joints present -> the other 5 are reported.
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)
        with pytest.raises(ValueError, match="missing expected G1"):
            p.set_robot_state_keys(list(WBC_G1_LEG_WAIST_JOINTS[:10]))

    def test_all_29_joints_accepted(self) -> None:
        # The default config observes 29 joints, so the whole-body set is required.
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)
        p.set_robot_state_keys(list(WBC_G1_ALL_JOINTS))  # exactly the 29, any surrounding order

    def test_missing_arm_joints_raises_for_observed_set(self) -> None:
        # Only the 15 leg+waist joints (no arms) is insufficient when n_obs_joints=29.
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)
        with pytest.raises(ValueError, match="missing observed G1 joints"):
            p.set_robot_state_keys(list(WBC_G1_LEG_WAIST_JOINTS))

    def test_reads_state_by_name_not_position(self) -> None:
        """qj is read from the named joints even when a free-base joint precedes
        them in the observation (so a positional slice would be wrong). qj spans
        the 29 observed joints in WBC_G1_ALL_JOINTS order."""
        p = _make_policy()  # uses _g1_keys() ordering with floating_base first
        obs = {k: 0.0 for k in _g1_keys()}
        # Give each observed joint a distinct value; floating_base gets a
        # sentinel that must NOT appear in qj.
        for i, name in enumerate(WBC_G1_ALL_JOINTS):
            obs[name] = 0.1 * (i + 1)
        obs["floating_base_joint"] = 99.0
        qj = p._read_joint_vector(obs, "position", p._obs_joint_names)
        assert qj.shape[0] == 29
        assert np.allclose(qj, [0.1 * (i + 1) for i in range(29)])
        assert 99.0 not in qj  # the free-base value never leaks in


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
        def _boom(*a, **k):  # type: ignore[no-untyped-def]
            raise ImportError("no huggingface_hub")

        monkeypatch.setattr(wbc_policy, "require_optional", _boom)
        with pytest.raises(RuntimeError, match="HuggingFace model id"):
            WBCPolicy._maybe_download_checkpoint("nvidia/GEAR-SONIC")

    def test_hf_id_heuristic_accepts_org_repo(self) -> None:
        assert wbc_policy._looks_like_hf_repo_id("nvidia/GEAR-SONIC")
        assert wbc_policy._looks_like_hf_repo_id("org-name/repo.name_1")

    def test_hf_id_heuristic_rejects_path_like(self) -> None:
        # Path-like strings must NOT be treated as HF ids (no surprise downloads).
        assert not wbc_policy._looks_like_hf_repo_id("./models/policy")
        assert not wbc_policy._looks_like_hf_repo_id("../ckpt/sonic")
        assert not wbc_policy._looks_like_hf_repo_id("/abs/path/sonic")
        assert not wbc_policy._looks_like_hf_repo_id("~/sonic")
        assert not wbc_policy._looks_like_hf_repo_id("a/b/c")  # more than one slash
        assert not wbc_policy._looks_like_hf_repo_id("dir/policy.onnx")  # .onnx file
        assert not wbc_policy._looks_like_hf_repo_id("win\\path")  # backslash


# ---------------------------------------------------------------------------
# Regression tests for bugs found in exhaustive review (against real deps)
# ---------------------------------------------------------------------------


class TestRegressionFixes:
    def test_target_orientation_does_not_crash_get_actions(self) -> None:
        """REGRESSION #1: a long target_orientation overflowed the 7-wide
        command block and raised on every tick. The command is now truncated
        to command_dim instead of crashing."""
        p = _make_policy(walk=False)
        obs = {k: 0.0 for k in _g1_keys()}
        # velocity(3) + orientation(6) = 9 > command_dim(7): previously crashed.
        actions = asyncio.run(
            p.get_actions(
                obs,
                "",
                target_velocity=[0.5, 0.0, 0.0],
                target_orientation=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
        )
        assert len(actions[0]) == 15
        # The fed observation's command block is exactly command_dim wide.
        fed = p.policy_session.calls[0][0]
        assert fed.shape[0] == 86  # single frame, history_len=1

    def test_target_orientation_fits_within_command_dim(self) -> None:
        """A velocity(3) + orientation(4) = 7 command fits exactly and is used."""
        p = _make_policy(walk=False)
        obs = {k: 0.0 for k in _g1_keys()}
        actions = asyncio.run(
            p.get_actions(obs, "", target_velocity=[0.5, 0.0, 0.0], target_orientation=[0.0, 0.0, 0.0, 1.0])
        )
        assert len(actions[0]) == 15

    def test_warns_once_when_no_velocity_in_observation(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """REGRESSION #4: the real sim observation has no joint/base velocity,
        so WBC runs open-loop on velocity. The policy must WARN (once), not
        silently pretend dqj converges."""
        import logging

        p = _make_policy(walk=False)
        obs = {k: 0.0 for k in _g1_keys()}  # positions only, no .vel / base_ang_vel
        with caplog.at_level(logging.WARNING):
            asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))
            asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))
        warnings = [r for r in caplog.records if "no joint velocities" in r.message]
        assert len(warnings) == 1, "velocity-absent warning should fire exactly once"

    def test_no_velocity_warning_when_base_ang_vel_present(self, caplog) -> None:  # type: ignore[no-untyped-def]
        import logging

        p = _make_policy(walk=False)
        obs: dict[str, Any] = {k: 0.0 for k in _g1_keys()}
        obs["base_ang_vel"] = [0.0, 0.0, 0.1]  # a real velocity signal
        with caplog.at_level(logging.WARNING):
            asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))
        assert not [r for r in caplog.records if "no joint velocities" in r.message]

    def test_per_joint_velocity_read_by_name(self) -> None:
        """When the observation DOES expose '<name>.vel' keys, dqj reads them
        for all 29 observed joints in WBC_G1_ALL_JOINTS order."""
        p = _make_policy(walk=False)
        obs = {k: 0.0 for k in _g1_keys()}
        for i, name in enumerate(WBC_G1_ALL_JOINTS):
            obs[f"{name}.vel"] = 0.01 * (i + 1)
        dqj = p._read_joint_vector(obs, "velocity", p._obs_joint_names)
        assert dqj.shape[0] == 29
        assert np.allclose(dqj, [0.01 * (i + 1) for i in range(29)])

    def test_num_actions_exceeding_mapping_table_rejected(self) -> None:
        """REGRESSION #8: num_actions > 15 silently truncated the 15-entry
        mapping table everywhere and failed late. Now rejected at construction."""
        cfg = _make_config(
            num_actions=20, single_obs_dim=200, default_angles=[0.0] * 20, kps=[1.0] * 20, kds=[0.0] * 20
        )
        with pytest.raises(ValueError, match="exceeds the 15-entry"):
            WBCPolicy(config=cfg, allow_missing_models=True)

    def test_flat_state_used_without_set_robot_state_keys(self) -> None:
        """REGRESSION (review #1): a provided observation.state must be USED even
        when set_robot_state_keys was never called - not silently zeroed. Matches
        the positional observation.state contract of cuRobo / MoveIt2. Consumed
        in the observed-joint order (29 entries by default)."""
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)  # no set_robot_state_keys
        state = [0.1 * (i + 1) for i in range(29)]
        qj = p._read_joint_vector({"observation.state": state}, "position", p._obs_joint_names)
        assert np.allclose(qj, state), "flat observation.state must be consumed positionally without keys"

    def test_flat_state_name_resolved_first_occurrence_wins(self) -> None:
        """REGRESSION (review #8): a duplicated joint name in the key list must
        not shift the resolved slot - first occurrence wins."""
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)
        # floating_base first, then all 29 observed joints, then a DUP of the
        # first joint at the end (which must NOT win the slot).
        keys = ["floating_base_joint", *WBC_G1_ALL_JOINTS, "left_hip_pitch_joint"]
        p.set_robot_state_keys(keys)
        arr = [float(i) for i in range(len(keys))]  # value at index i == i
        qj = p._read_joint_vector({"observation.state": arr}, "position", p._obs_joint_names)
        assert qj[0] == 1.0, "left_hip_pitch_joint must resolve to its FIRST occurrence (index 1), not the dup"


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

    def test_policy_config_path_sets_static_walk_command(self) -> None:
        """The mesh tell() / policy_config path forwards constructor kwargs to
        create_policy(provider, **policy_config). A target_velocity supplied that
        way becomes the static-walk default command (used when no per-call kwarg
        is given). This is how a command reaches WBC without policy_kwargs."""
        p = create_policy(
            "wbc",
            config=_make_config(),
            walk=True,
            target_velocity=[0.4, 0.0, 0.0],
            allow_missing_models=True,
        )
        assert isinstance(p, WBCPolicy)
        assert p._default_command is not None
        assert np.allclose(p._default_command, [0.4, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Upstream-fidelity pins (verified against NVlabs/GR00T-WholeBodyControl
# decoupled_wbc/sim2mujoco: run_mujoco_gear_wbc.py + g1_gear_wbc.yaml)
# ---------------------------------------------------------------------------


class TestUpstreamFidelity:
    """Pin the exact contract of the upstream reference runner so a refactor
    can't silently drift from it. Values are transcribed from g1_gear_wbc.yaml
    and compute_observation()/run() in run_mujoco_gear_wbc.py."""

    def test_default_config_matches_upstream_yaml(self) -> None:
        c = WBCConfig(policy_path="p.onnx")  # defaults only
        assert c.single_obs_dim == 86
        assert c.obs_history_len == 6  # g1_gear_wbc.yaml
        assert c.num_obs == 516  # 86 * 6
        assert c.num_actions == 15
        assert c.n_obs_joints == 29  # qj/dqj observe the whole body, not just 15
        assert c.command_dim == 7
        assert c.action_scale == 0.25
        assert c.obs_scales == {"ang_vel": 0.5, "dof_pos": 1.0, "dof_vel": 0.05}
        assert c.cmd_scale == [2.0, 2.0, 0.5]
        assert c.height_cmd == 0.74

    def test_obs_layout_observes_29_joints_not_15(self) -> None:
        """CRITICAL pin: qj/dqj blocks are n_obs_joints (29) wide, NOT num_actions
        (15). 7+3+3+29+29+15 = 86 = single_obs_dim. A 15-wide qj/dqj would only
        populate 58 and misplace the data - the network would see garbage even
        though the 516 total still loads. Verified against upstream
        compute_observation (qj=qpos[7:7+n_joints], n_joints=29) and the real
        GR00T-WholeBodyControl-Balance.onnx (input width 516)."""
        c = WBCConfig(policy_path="p.onnx")
        populated = c.command_dim + 3 + 3 + 2 * c.n_obs_joints + c.num_actions
        assert populated == c.single_obs_dim == 86
        # The whole-body mapping must be long enough for n_obs_joints, and its
        # first num_actions names must be exactly the controlled leg+waist set.
        assert len(WBC_G1_ALL_JOINTS) >= c.n_obs_joints
        assert WBC_G1_ALL_JOINTS[: c.num_actions] == WBC_G1_LEG_WAIST_JOINTS

    def test_n_obs_joints_below_num_actions_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_obs_joints .* must be >= num_actions"):
            WBCConfig(policy_path="p.onnx", n_obs_joints=10, num_actions=15)

    def test_command_layout_matches_compute_observation(self) -> None:
        """command[0:3] = vel*cmd_scale; command[3] = height; command[4:7] = rpy."""
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)
        cmd, raw_vel = p._resolve_command(
            {"target_velocity": [0.5, -0.25, 2.0], "target_orientation": [0.1, 0.2, 0.3], "height": 0.8}
        )
        # cmd_scale default [2,2,0.5]: [0.5*2, -0.25*2, 2.0*0.5] = [1.0, -0.5, 1.0]
        assert np.allclose(cmd[:3], [1.0, -0.5, 1.0])
        assert np.isclose(cmd[3], 0.8)  # per-call height overrides config
        assert np.allclose(cmd[4:7], [0.1, 0.2, 0.3])
        # raw velocity is UNSCALED (used for walk selection)
        assert np.allclose(raw_vel, [0.5, -0.25, 2.0])

    def test_height_defaults_to_config_when_not_supplied(self) -> None:
        p = WBCPolicy(config=_make_config(), allow_missing_models=True)
        cmd, _ = p._resolve_command({"target_velocity": [0.0, 0.0, 0.0]})
        assert np.isclose(cmd[3], 0.74)  # upstream default height_cmd

    def test_walk_threshold_is_0_05_on_raw_velocity(self) -> None:
        """Upstream: norm(loco_cmd) <= 0.05 -> main (standing) policy; above -> walk.
        The threshold is tested on the RAW (unscaled) velocity, not the
        cmd_scale'd command block."""
        obs = {k: 0.0 for k in _g1_keys()}
        # vel norm 0.04 < 0.05 -> main policy (standing)
        p1 = _make_policy(walk=True)
        asyncio.run(p1.get_actions(obs, "", target_velocity=[0.04, 0.0, 0.0]))
        assert p1.policy_session.calls and not p1.walk_session.calls
        # vel norm 0.06 > 0.05 -> walk policy
        p2 = _make_policy(walk=True)
        asyncio.run(p2.get_actions(obs, "", target_velocity=[0.06, 0.0, 0.0]))
        assert p2.walk_session.calls and not p2.policy_session.calls

    def test_walk_selection_uses_raw_not_scaled_velocity(self) -> None:
        """A velocity of 0.03 scales to 0.06 under cmd_scale=2.0, but the walk
        decision must use the RAW 0.03 (< 0.05 -> standing), not the scaled 0.06.
        Guards against regressing to testing the scaled command block."""
        p = _make_policy(walk=True)
        obs = {k: 0.0 for k in _g1_keys()}
        asyncio.run(p.get_actions(obs, "", target_velocity=[0.03, 0.0, 0.0]))
        assert p.policy_session.calls, "raw 0.03 < 0.05 -> standing/main, despite scaling to 0.06"
        assert not p.walk_session.calls

    def test_quat_rotate_inverse_matches_upstream_formula(self) -> None:
        """My quat helper must be numerically identical to the upstream
        conjugate-based quat_rotate_inverse (run_mujoco_gear_wbc.py:126-141)."""

        def upstream(q: np.ndarray, v: np.ndarray) -> np.ndarray:
            w, x, y, z = q
            qc = np.array([w, -x, -y, -z])
            return np.array(
                [
                    v[0] * (qc[0] ** 2 + qc[1] ** 2 - qc[2] ** 2 - qc[3] ** 2)
                    + v[1] * 2 * (qc[1] * qc[2] - qc[0] * qc[3])
                    + v[2] * 2 * (qc[1] * qc[3] + qc[0] * qc[2]),
                    v[0] * 2 * (qc[1] * qc[2] + qc[0] * qc[3])
                    + v[1] * (qc[0] ** 2 - qc[1] ** 2 + qc[2] ** 2 - qc[3] ** 2)
                    + v[2] * 2 * (qc[2] * qc[3] - qc[0] * qc[1]),
                    v[0] * 2 * (qc[1] * qc[3] - qc[0] * qc[2])
                    + v[1] * 2 * (qc[2] * qc[3] + qc[0] * qc[1])
                    + v[2] * (qc[0] ** 2 - qc[1] ** 2 - qc[2] ** 2 + qc[3] ** 2),
                ]
            )

        rng = np.random.RandomState(0)
        for _ in range(200):
            q = rng.randn(4)
            q /= np.linalg.norm(q)
            v = rng.randn(3)
            assert np.allclose(quat_rotate_inverse(q, v), upstream(q, v), atol=1e-10)


# ---------------------------------------------------------------------------
# Mesh / Device Connect security allowlist
# ---------------------------------------------------------------------------


class TestMeshSecurityAllowlist:
    """WBC must pass the mesh / Device Connect policy-provider allowlist, or it
    can't be driven over tell() / Device Connect (validate_command and the DC
    drivers reject an un-allowlisted provider). Regression for the gap where the
    allowlist wasn't updated when the provider was added."""

    def test_wbc_and_sonic_in_policy_provider_allowlist(self) -> None:
        from strands_robots.mesh.security import is_safe_policy_provider

        assert is_safe_policy_provider("wbc")
        assert is_safe_policy_provider("sonic")
        assert is_safe_policy_provider("WBC")  # case-insensitive

    def test_mesh_validate_command_accepts_wbc_execute(self) -> None:
        from strands_robots.mesh.security import validate_command

        # A tell()-shaped execute command with policy_provider=wbc must validate
        # (it previously failed the policy_provider allowlist check).
        cmd = {
            "action": "execute",
            "instruction": "walk forward",
            "policy_provider": "wbc",
            "duration": 5.0,
        }
        out = validate_command(cmd)
        assert out["policy_provider"] == "wbc"
