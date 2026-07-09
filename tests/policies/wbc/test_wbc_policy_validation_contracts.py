"""Input-validation and observation-robustness contracts for :class:`WBCPolicy`.

WBC is a velocity-feedback balance controller for the Unitree G1: a bad command
or a malformed observation can destabilise a walking humanoid, so the policy
validates its inputs loudly and degrades a partial observation to a defined
neutral rather than fabricating motion or aborting mid-rollout. These tests pin
those contracts through the public surface (the constructor and
:meth:`WBCPolicy.get_actions`), using the ``allow_missing_models`` seam + a
stubbed ONNX session so they run with no onnxruntime, GPU, or checkpoint:

* a non-numeric ``target_velocity`` is a loud ``ValueError`` (numeric-sequence
  contract), distinct from the finite/length checks;
* ``n_obs_joints`` beyond the 29-entry G1 whole-body joint map is rejected at
  construction rather than silently reading past the mapping;
* unknown constructor kwargs are ignored (forward-compatible), not fatal;
* a call with no per-call velocity and no constructor default drives a
  zero/standing command through the main (non-walk) session;
* an ONNX output whose width disagrees with ``num_actions`` is a loud
  ``RuntimeError`` (never a truncated/garbage action);
* a session lacking ``get_inputs`` falls back to a plain ``input_name`` attr;
* a single non-numeric per-joint observation entry degrades to its neutral
  default instead of aborting the whole observation build;
* supplying ``base_ang_vel`` satisfies the velocity channel, so the
  degraded-gait warning is NOT emitted;
* a flat ``observation.state`` shorter than the observed-joint set is
  zero-padded when no robot state keys have been resolved (the direct-API
  positional contract).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import numpy as np
import pytest

from strands_robots.policies.wbc import WBC_G1_ALL_JOINTS, WBC_G1_LEG_WAIST_JOINTS, WBCConfig, WBCPolicy

_N = 15  # controlled leg + waist DOFs (action dim)
_NO = 29  # observed joints (legs + waist + arms)


class _StubInput:
    name = "obs"


class _StubSession:
    """Minimal onnxruntime.InferenceSession stand-in returning ``(1, num_actions)``."""

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


class _NoGetInputsSession:
    """Session exposing a plain ``input_name`` attribute but no ``get_inputs``.

    Exercises the duck-typed fallback in ``_session_input_name`` used for stub
    sessions that don't implement the full onnxruntime input-introspection API.
    """

    def __init__(self) -> None:
        self.input_name = "custom_in"
        self.calls: list[np.ndarray] = []

    def run(self, output_names, feed):  # type: ignore[no-untyped-def]
        (arr,) = feed.values()
        self.calls.append(np.asarray(arr))
        return [np.full((1, _N), 0.04, dtype=np.float32)]


def _make_config(**overrides) -> WBCConfig:  # type: ignore[no-untyped-def]
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


def _g1_keys() -> list[str]:
    """Real MuJoCo G1 key order: free base joint prepended to the 29 whole-body joints."""
    return ["floating_base_joint", *WBC_G1_ALL_JOINTS]


def _make_policy(walk: bool = False, **cfg_overrides) -> WBCPolicy:  # type: ignore[no-untyped-def]
    p = WBCPolicy(config=_make_config(**cfg_overrides), walk=walk, allow_missing_models=True)
    p.policy_session = _StubSession()
    if walk:
        p.walk_session = _StubSession()
    p.set_robot_state_keys(_g1_keys())
    return p


class TestCommandValidation:
    def test_non_numeric_target_velocity_rejected(self) -> None:
        """A non-numeric velocity is a numeric-sequence ValueError, not a crash later."""
        # A caller passing non-numeric junk (the typed API says list[float]); the
        # policy must still reject it at runtime rather than crash unpredictably later.
        bad_velocity: Any = ["a", "b", "c"]
        with pytest.raises(ValueError, match="numeric sequence"):
            WBCPolicy(config=_make_config(), target_velocity=bad_velocity, allow_missing_models=True)

    def test_n_obs_joints_exceeding_g1_map_rejected(self) -> None:
        """Observing more joints than the 29-entry G1 whole-body map is rejected up front."""
        cfg = _make_config(n_obs_joints=len(WBC_G1_ALL_JOINTS) + 1, single_obs_dim=88)
        with pytest.raises(ValueError, match="whole-body joint mapping"):
            WBCPolicy(config=cfg, allow_missing_models=True)

    def test_unknown_constructor_kwargs_ignored(self) -> None:
        """Unknown kwargs are forward-compatibly ignored; the policy is still usable."""
        p = WBCPolicy(config=_make_config(), allow_missing_models=True, some_future_kwarg=123)
        p.policy_session = _StubSession()
        p.set_robot_state_keys(_g1_keys())
        actions = asyncio.run(p.get_actions({k: 0.0 for k in _g1_keys()}, "", target_velocity=[0.0, 0.0, 0.0]))
        assert set(actions[0].keys()) == set(WBC_G1_LEG_WAIST_JOINTS)

    def test_zero_command_when_no_velocity_supplied(self) -> None:
        """No per-call velocity and no constructor default -> standing (main session)."""
        p = _make_policy(walk=True)
        asyncio.run(p.get_actions({k: 0.0 for k in _g1_keys()}, ""))
        assert p.policy_session.calls, "the main (standing) session should drive a zero command"
        assert not p.walk_session.calls, "the walk session must not run for a zero command"


class TestSessionContract:
    def test_onnx_output_width_mismatch_raises(self) -> None:
        """An ONNX output width != num_actions is a loud error, never a garbage action."""
        p = WBCPolicy(config=_make_config(), walk=False, allow_missing_models=True)
        p.policy_session = _StubSession(num_actions=_N + 3)
        p.set_robot_state_keys(_g1_keys())
        with pytest.raises(RuntimeError, match="output width"):
            asyncio.run(p.get_actions({k: 0.0 for k in _g1_keys()}, "", target_velocity=[0.0, 0.0, 0.0]))

    def test_session_input_name_fallback_attribute(self) -> None:
        """A session without get_inputs falls back to its plain input_name attr."""
        p = WBCPolicy(config=_make_config(), walk=False, allow_missing_models=True)
        session = _NoGetInputsSession()
        p.policy_session = session
        p.set_robot_state_keys(_g1_keys())
        actions = asyncio.run(p.get_actions({k: 0.0 for k in _g1_keys()}, "", target_velocity=[0.0, 0.0, 0.0]))
        assert session.calls, "session should have been fed despite lacking get_inputs"
        assert set(actions[0].keys()) == set(WBC_G1_LEG_WAIST_JOINTS)


class TestObservationRobustness:
    def test_non_numeric_joint_observation_degrades_to_neutral(self) -> None:
        """One unparseable per-joint entry degrades to neutral, not an aborted build."""
        p = _make_policy(walk=False)
        obs: dict[str, object] = {k: 0.0 for k in _g1_keys()}
        obs[WBC_G1_ALL_JOINTS[0]] = "not-a-number"
        actions = asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))
        assert len(actions) == 1
        assert set(actions[0].keys()) == set(WBC_G1_LEG_WAIST_JOINTS)

    def test_flat_observation_velocity_satisfies_velocity_channel(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """A flat observation.velocity counts as a live velocity channel: no degraded-gait warning."""
        p = _make_policy(walk=False)
        obs: dict[str, object] = {k: 0.0 for k in _g1_keys()}
        obs["observation.velocity"] = [0.01] * len(_g1_keys())
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.wbc.policy"):
            asyncio.run(p.get_actions(obs, "", target_velocity=[0.0, 0.0, 0.0]))
        assert not any("no joint velocities" in r.getMessage() for r in caplog.records)

    def test_short_flat_state_without_keys_zero_pads(self) -> None:
        """A flat observation.state shorter than the observed set is zero-padded (no keys)."""
        p = WBCPolicy(config=_make_config(), walk=False, allow_missing_models=True)
        p.policy_session = _StubSession()
        # Deliberately DO NOT call set_robot_state_keys: the flat vector is then
        # consumed positionally (the direct-API / replay contract).
        short_state = [0.5] * 10  # shorter than the 29 observed joints
        actions = asyncio.run(p.get_actions({"observation.state": short_state}, "", target_velocity=[0.0, 0.0, 0.0]))
        assert len(actions) == 1
        assert set(actions[0].keys()) == set(WBC_G1_LEG_WAIST_JOINTS)
