"""Unit tests for the gait-clock WBC variant - no GPU, no onnxruntime.

Pins the clean-room port of NVIDIA's gait-clock reference controller
(``decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc_gait.py``):

* :class:`GaitClock` reproduces the upstream bipedal phase-clock block
  (walk-entry reseed, warm-up pin, static freeze) and its periodicity.
* :func:`build_gait_frame` produces the exact 95-dim layout (8-wide command
  with ``freq`` at slot [4], two reserved zero torso blocks, a 2-dim clock tail).
* :class:`WBCGaitPolicy` defaults to the gait layout (single_obs_dim=95,
  command_dim=8), runs a SINGLE ONNX policy, emits the 15 leg+waist targets,
  rejects a non-gait config, and round-trips through ``create_policy``.

All values are hand-computed so the tests run on any machine via the
``allow_missing_models`` stub-session seam.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

from strands_robots.policies import create_policy, list_providers
from strands_robots.policies.wbc import (
    GAIT_COMMAND_DIM,
    GAIT_SINGLE_OBS_DIM,
    GaitClock,
    WBCConfig,
    WBCGaitPolicy,
    build_gait_frame,
)
from strands_robots.policies.wbc.policy import WBC_G1_ALL_JOINTS, WBC_G1_LEG_WAIST_JOINTS

_N = 15  # leg + waist DOFs (action dim)
_NO = 29  # observed whole-body joints (qj/dqj block width)


class _StubInput:
    name = "obs"


class _StubSession:
    """Minimal onnxruntime.InferenceSession stand-in returning a fixed action."""

    def __init__(self, num_actions: int = _N, fill: float = 0.03) -> None:
        self.num_actions = num_actions
        self.fill = fill
        self.calls: list[np.ndarray] = []

    def get_inputs(self) -> list[_StubInput]:
        return [_StubInput()]

    def run(self, output_names, feed):  # type: ignore[no-untyped-def]
        (arr,) = feed.values()
        self.calls.append(np.asarray(arr))
        return [np.full((1, self.num_actions), self.fill, dtype=np.float32)]


def _gait_config(**overrides) -> WBCConfig:
    base: dict[str, Any] = dict(
        policy_path="policy.onnx",
        single_obs_dim=GAIT_SINGLE_OBS_DIM,
        command_dim=GAIT_COMMAND_DIM,
        num_actions=_N,
        n_obs_joints=_NO,
        default_angles=[0.0] * _N,
        obs_scales={"ang_vel": 0.5, "dof_pos": 1.0, "dof_vel": 0.05},
    )
    base.update(overrides)
    return WBCConfig(**base)


# ---------------------------------------------------------------------------
# build_gait_frame - exact 95-dim layout
# ---------------------------------------------------------------------------


def test_gait_frame_width_and_offsets_hand_computed():
    cfg = _gait_config()
    command = np.arange(8, dtype=np.float64)  # [0,1,2,3,4,5,6,7]
    base_ang_vel = np.array([1.0, 2.0, 3.0])
    proj_gravity = np.array([0.0, 0.0, -1.0])
    qj = np.arange(_NO, dtype=np.float64)
    dqj = np.full(_NO, 10.0)
    prev_action = np.full(_N, 0.5)
    clock = np.array([0.7, -0.4])

    frame = build_gait_frame(
        cfg,
        command=command,
        base_ang_vel=base_ang_vel,
        proj_gravity=proj_gravity,
        qj=qj,
        dqj=dqj,
        prev_action=prev_action,
        clock=clock,
    )

    assert frame.shape == (95,)
    c = 8
    # command verbatim
    np.testing.assert_array_equal(frame[0:c], command)
    # base ang vel scaled by ang_vel=0.5
    np.testing.assert_allclose(frame[c : c + 3], base_ang_vel * 0.5)
    # projected gravity unscaled
    np.testing.assert_allclose(frame[c + 3 : c + 6], proj_gravity)
    # two reserved torso blocks are zero
    np.testing.assert_array_equal(frame[c + 6 : c + 12], np.zeros(6))
    # qj (defaults 0) * dof_pos=1.0
    np.testing.assert_allclose(frame[c + 12 : c + 12 + _NO], qj)
    # dqj * dof_vel=0.05
    np.testing.assert_allclose(frame[c + 12 + _NO : c + 12 + 2 * _NO], dqj * 0.05)
    # prev action
    np.testing.assert_allclose(frame[c + 12 + 2 * _NO : c + 12 + 2 * _NO + _N], prev_action)
    # clock at the very tail
    np.testing.assert_allclose(frame[-2:], clock)


def test_gait_frame_subtracts_default_angles_padded():
    cfg = _gait_config(default_angles=[0.1] * _N)
    qj = np.zeros(_NO)
    frame = build_gait_frame(
        cfg,
        command=np.zeros(8),
        base_ang_vel=np.zeros(3),
        proj_gravity=np.zeros(3),
        qj=qj,
        dqj=np.zeros(_NO),
        prev_action=np.zeros(_N),
        clock=np.zeros(2),
    )
    qj_block = frame[8 + 12 : 8 + 12 + _NO]
    # first 15 joints had 0.1 default subtracted; arm joints (15..29) padded with 0.
    np.testing.assert_allclose(qj_block[:_N], np.full(_N, -0.1))
    np.testing.assert_allclose(qj_block[_N:], np.zeros(_NO - _N))


@pytest.mark.parametrize(
    "name,kwargs",
    [
        ("base_ang_vel", dict(base_ang_vel=np.zeros(2))),
        ("proj_gravity", dict(proj_gravity=np.zeros(4))),
        ("qj", dict(qj=np.zeros(_NO + 1))),
        ("dqj", dict(dqj=np.zeros(_NO - 1))),
        ("prev_action", dict(prev_action=np.zeros(_N + 1))),
        ("clock", dict(clock=np.zeros(3))),
    ],
)
def test_gait_frame_rejects_wrong_lengths(name, kwargs):
    cfg = _gait_config()
    good = dict(
        command=np.zeros(8),
        base_ang_vel=np.zeros(3),
        proj_gravity=np.zeros(3),
        qj=np.zeros(_NO),
        dqj=np.zeros(_NO),
        prev_action=np.zeros(_N),
        clock=np.zeros(2),
    )
    good.update(kwargs)
    with pytest.raises(ValueError, match=name):
        build_gait_frame(cfg, **good)


def test_gait_frame_rejects_single_obs_dim_too_small():
    cfg = _gait_config(single_obs_dim=94)
    with pytest.raises(ValueError, match="gait observation layout needs 95"):
        build_gait_frame(
            cfg,
            command=np.zeros(8),
            base_ang_vel=np.zeros(3),
            proj_gravity=np.zeros(3),
            qj=np.zeros(_NO),
            dqj=np.zeros(_NO),
            prev_action=np.zeros(_N),
            clock=np.zeros(2),
        )


# ---------------------------------------------------------------------------
# GaitClock - bipedal phase generator
# ---------------------------------------------------------------------------


def test_gait_clock_static_then_freezes_both_feet():
    clk = GaitClock()
    last = None
    for _ in range(200):
        last = clk.update(np.zeros(3), freq=0.75)
    # Held static long enough that both feet freeze at +1.0 (held stance).
    np.testing.assert_allclose(last, np.array([1.0, 1.0]))
    assert clk.frozen_FL and clk.frozen_FR


def test_gait_clock_walk_entry_reseeds_indices():
    clk = GaitClock()
    clk.update(np.zeros(3), freq=1.0)  # static
    assert clk.walking_mask is False
    # First moving tick: gait_indices reseeded to -0.25 then advanced by dt*freq.
    clk.update(np.array([1.0, 0.0, 0.0]), freq=1.0)
    assert clk.walking_mask is True
    expected = (-0.25 + 0.02 * 1.0) % 1.0
    assert clk.gait_indices == pytest.approx(expected)
    # Freeze flags cleared on entering walk.
    assert clk.frozen_FL is False and clk.frozen_FR is False


def test_gait_clock_warmup_pins_right_foot_at_peak():
    # Warm-up ramp (upstream ``just_started`` window): for the first
    # ``0.5 / freq`` seconds of walking the right-foot phase is pinned to 0.25,
    # which the [0, 1) stretch (0.25 < DURATION=0.5 -> 0.25 * 0.5/0.5 = 0.25)
    # and the sine map (sin(2*pi*0.25) = 1.0) turn into clock_FR == +1.0 exactly.
    # This eases the robot into the cycle; the left foot meanwhile tracks the
    # true reseeded phase, so the two channels are NOT identical during warm-up.
    # freq=1.0, dt=0.02 -> window = 0.5 s = ticks 1..24 (just_started 0.02..0.48,
    # each < 0.5); tick 25 (just_started == 0.50) exits the ramp.
    clk = GaitClock()
    clk.update(np.zeros(3), freq=1.0)  # static priming tick
    for tick in range(1, 25):
        sig = clk.update(np.array([1.0, 0.0, 0.0]), freq=1.0)
        assert clk.just_started == pytest.approx(0.02 * tick)
        # Right foot held at the sine peak for the whole warm-up window.
        assert sig[1] == pytest.approx(1.0), f"clock_FR at tick {tick}"
        # Left foot is on its own trajectory (not frozen with the right).
        assert abs(sig[0] - sig[1]) > 1e-6
    # Tick 25 leaves the window (just_started == 0.50, not < 0.50); the right
    # foot is released from the pin and rejoins the phase clock on the next tick.
    clk.update(np.array([1.0, 0.0, 0.0]), freq=1.0)
    assert clk.just_started == pytest.approx(0.50)
    sig26 = clk.update(np.array([1.0, 0.0, 0.0]), freq=1.0)
    assert sig26[1] < 1.0 - 1e-6, "clock_FR should leave the +1.0 pin after warm-up"


def test_gait_clock_signal_bounded_and_periodic_while_walking():
    clk = GaitClock()
    sigs = [clk.update(np.array([1.5, 0.0, 0.0]), freq=1.0) for _ in range(400)]
    arr = np.array(sigs)
    assert np.all(arr >= -1.0001) and np.all(arr <= 1.0001)
    # gait_indices advances by dt*freq=0.02 per tick -> full cycle in 50 ticks.
    # After the warm-up window the two channels are offset (not identical).
    tail = arr[100:]
    assert np.max(np.abs(tail[:, 0] - tail[:, 1])) > 0.1


def test_gait_clock_rejects_nonpositive_freq():
    clk = GaitClock()
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="freq"):
            clk.update(np.zeros(3), freq=bad)


def test_gait_clock_reset_restores_initial_state():
    clk = GaitClock()
    for _ in range(50):
        clk.update(np.array([1.0, 0.0, 0.0]), freq=1.2)
    clk.reset()
    assert clk.gait_indices == 0.0
    assert clk.walking_mask is False
    assert clk.just_started == 0.0
    assert not clk.frozen_FL and not clk.frozen_FR
    np.testing.assert_array_equal(clk.clock_inputs, np.zeros(2))


def test_gait_clock_deterministic():
    a, b = GaitClock(), GaitClock()
    for t in range(120):
        v = np.array([0.4 if 10 < t < 90 else 0.0, 0.0, 0.0]) * 2.0
        ra = a.update(v, freq=0.9)
        rb = b.update(v, freq=0.9)
        np.testing.assert_array_equal(ra, rb)


# ---------------------------------------------------------------------------
# WBCGaitPolicy
# ---------------------------------------------------------------------------


def _full_g1_obs() -> dict[str, Any]:
    keys = list(WBC_G1_ALL_JOINTS)
    obs: dict[str, Any] = {k: 0.0 for k in keys}
    obs.update({f"{k}.vel": 0.0 for k in keys})
    obs["base_ang_vel"] = [0.0, 0.0, 0.0]
    obs["base_quat"] = [1.0, 0.0, 0.0, 0.0]
    return obs


def test_gait_policy_defaults_to_gait_layout():
    p = WBCGaitPolicy(allow_missing_models=True)
    assert p.config.single_obs_dim == GAIT_SINGLE_OBS_DIM
    assert p.config.command_dim == GAIT_COMMAND_DIM
    assert p.config.num_obs == 95 * 6
    assert p.provider_name == "wbc_gait"
    # Single-policy variant: no walk session loaded/used.
    assert p._walk is False


def test_gait_policy_emits_leg_waist_targets_and_feeds_570_obs():
    p = WBCGaitPolicy(allow_missing_models=True)
    stub = _StubSession()
    p.policy_session = stub
    obs = _full_g1_obs()
    p.set_robot_state_keys(list(WBC_G1_ALL_JOINTS))
    out = asyncio.run(p.get_actions(obs, "", target_velocity=[0.5, 0.0, 0.0], gait_frequency=1.5))
    assert list(out[0].keys()) == list(WBC_G1_LEG_WAIST_JOINTS)
    assert stub.calls[-1].shape == (1, 570)
    # target_q = default_angles + action_scale(0.25) * raw(0.03). The G1 SONIC
    # stance is auto-filled (num_actions == 15), so each target is that joint's
    # nominal angle plus the scaled offset.
    expected = p.default_angles + 0.25 * 0.03
    np.testing.assert_allclose(list(out[0].values()), expected)


def test_gait_policy_command_block_layout():
    """freq lands at slot [4]; rpy at [5:8] (vs the non-gait 7-wide command)."""
    p = WBCGaitPolicy(allow_missing_models=True)
    command, raw = p._resolve_command(
        {
            "target_velocity": [1.0, 0.0, 0.0],
            "gait_frequency": 2.0,
            "height": 0.8,
            "target_orientation": [0.1, 0.2, 0.3],
        }
    )
    assert command.shape == (8,)
    assert command[0] == pytest.approx(1.0 * 2.0)  # vx * cmd_scale[0]
    assert command[3] == pytest.approx(0.8)  # height
    assert command[4] == pytest.approx(2.0)  # freq at slot 4
    np.testing.assert_allclose(command[5:8], [0.1, 0.2, 0.3])  # rpy at 5:8
    np.testing.assert_array_equal(raw, [1.0, 0.0, 0.0])


def test_gait_policy_frequency_precedence():
    # constructor default used when no per-call kwarg.
    p = WBCGaitPolicy(allow_missing_models=True, gait_frequency=1.25)
    cmd, _ = p._resolve_command({"target_velocity": [0.5, 0.0, 0.0]})
    assert cmd[4] == pytest.approx(1.25)
    # per-call overrides constructor default.
    cmd2, _ = p._resolve_command({"target_velocity": [0.5, 0.0, 0.0], "gait_frequency": 3.0})
    assert cmd2[4] == pytest.approx(3.0)
    # config default when neither supplied.
    p2 = WBCGaitPolicy(allow_missing_models=True)
    cmd3, _ = p2._resolve_command({"target_velocity": [0.5, 0.0, 0.0]})
    assert cmd3[4] == pytest.approx(0.75)


def test_gait_policy_reset_clears_clock():
    p = WBCGaitPolicy(allow_missing_models=True)
    p.policy_session = _StubSession()
    obs = _full_g1_obs()
    p.set_robot_state_keys(list(WBC_G1_ALL_JOINTS))
    for _ in range(20):
        asyncio.run(p.get_actions(obs, "", target_velocity=[0.6, 0.0, 0.0], gait_frequency=1.5))
    assert p._gait_clock.walking_mask is True
    p.reset()
    assert p._gait_clock.gait_indices == 0.0
    assert p._gait_clock.walking_mask is False
    np.testing.assert_array_equal(p._prev_action, np.zeros(_N))


def test_gait_policy_rejects_nongait_config():
    # An explicit non-gait config (single_obs_dim=86, command_dim=7) is rejected.
    bad = WBCConfig(policy_path="policy.onnx", single_obs_dim=86, command_dim=7)
    with pytest.raises(ValueError, match="gait observation layout"):
        WBCGaitPolicy(config=bad, allow_missing_models=True)


def test_gait_policy_registered_and_round_trips():
    assert "wbc_gait" in list_providers()
    p = create_policy("wbc_gait", allow_missing_models=True)
    assert isinstance(p, WBCGaitPolicy)
    # shorthand
    p2 = create_policy("sonic_gait", allow_missing_models=True)
    assert p2.provider_name == "wbc_gait"
