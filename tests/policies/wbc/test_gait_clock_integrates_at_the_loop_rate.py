"""The gait phase advances by the period the executing loop actually runs at.

``GaitClock`` advances by ``dt * freq`` per tick, so a ``dt`` that is not the
control period the policy is queried at makes the commanded ``gait_frequency``
mean something other than steps per second: the realised cadence comes out
scaled by ``control_frequency / 50``, and the robot walks at a rhythm nobody
commanded while every reported number looks right. The runtime states its rate
via ``Policy.set_control_frequency`` before the loop starts, which is what these
pin the clock to.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np
import pytest

from strands_robots.policies.wbc import GaitClock, WBCGaitPolicy
from strands_robots.policies.wbc.gait import _GAIT_CONTROL_DT
from strands_robots.policies.wbc.policy import WBC_G1_ALL_JOINTS

_COMMANDED_STEPS_PER_S = 1.5
_WALL_SECONDS = 4.0


class _StubInput:
    name = "obs"


class _StubSession:
    """Minimal onnxruntime.InferenceSession stand-in returning a fixed action."""

    def get_inputs(self) -> list[_StubInput]:
        return [_StubInput()]

    def run(self, output_names, feed):  # type: ignore[no-untyped-def]
        return [np.full((1, 15), 0.03, dtype=np.float32)]


def _observation() -> dict[str, object]:
    keys = list(WBC_G1_ALL_JOINTS)
    obs: dict[str, object] = {k: 0.0 for k in keys}
    obs.update({f"{k}.vel": 0.0 for k in keys})
    obs["base_ang_vel"] = [0.0, 0.0, 0.0]
    obs["base_quat"] = [1.0, 0.0, 0.0, 0.0]
    return obs


def _walk(control_frequency: float | None, wall_seconds: float = _WALL_SECONDS) -> list[float]:
    """Drive a gait policy for a wall-clock span; return the gait-phase trace."""
    policy = WBCGaitPolicy(allow_missing_models=True)
    policy.policy_session = _StubSession()
    policy.set_robot_state_keys(list(WBC_G1_ALL_JOINTS))
    if control_frequency is not None:
        policy.set_control_frequency(control_frequency)
    rate = control_frequency if control_frequency is not None else 1.0 / _GAIT_CONTROL_DT
    obs = _observation()
    phases: list[float] = []
    for _ in range(int(wall_seconds * rate)):
        asyncio.run(policy.get_actions(obs, "", target_velocity=[0.5, 0.0, 0.0], gait_frequency=_COMMANDED_STEPS_PER_S))
        phases.append(policy._gait_clock.gait_indices)
    return phases


def _reference_phases(wall_seconds: float = _WALL_SECONDS) -> list[float]:
    """Phase trace of a clock stepped explicitly at the upstream reference period."""
    clock = GaitClock()
    phases: list[float] = []
    for _ in range(int(wall_seconds / _GAIT_CONTROL_DT)):
        clock.update(np.array([1.0, 0.0, 0.0]), freq=_COMMANDED_STEPS_PER_S, dt=_GAIT_CONTROL_DT)
        phases.append(clock.gait_indices)
    return phases


def _cycles(phases: list[float]) -> int:
    """Completed gait cycles in a phase trace (the phase wraps at 1.0)."""
    return sum(1 for before, after in zip(phases, phases[1:], strict=False) if after < before)


@pytest.mark.parametrize("control_frequency", [12.5, 25.0, 50.0, 100.0, 200.0])
def test_the_commanded_step_frequency_is_realised_at_every_control_rate(control_frequency):
    """A cadence command is steps per second, whatever rate the loop runs at.

    Pre-fix the phase advanced by a hardcoded 0.02 s per tick however fast the
    loop queried the policy, so this walk realised
    ``1.5 * control_frequency / 50`` steps/s - 4x too fast at 200 Hz and 3x too
    slow at 12.5 Hz.
    """
    realised = _cycles(_walk(control_frequency)) / _WALL_SECONDS
    assert realised == pytest.approx(_COMMANDED_STEPS_PER_S, abs=0.25), (
        f"at {control_frequency} Hz the gait realised {realised} steps/s for a commanded {_COMMANDED_STEPS_PER_S}"
    )


def test_the_upstream_reference_rate_is_unchanged():
    """50 Hz is the period the upstream runner uses, and it still integrates the same.

    The control: reading the loop rate must not move the one rate that was
    already right, so the 50 Hz trace matches a clock stepped explicitly at the
    upstream reference period.
    """
    assert _walk(1.0 / _GAIT_CONTROL_DT) == pytest.approx(_reference_phases())


def test_an_unstated_control_rate_warns_once_and_uses_the_reference_period(caplog):
    """No rate stated is reported, not silently assumed.

    ``Policy.control_frequency`` documents the contract: a provider that needs
    the rate warns loudly and falls back rather than assuming one. The warning
    is latched per policy - at control rate a per-tick line is a flood.
    """
    with caplog.at_level(logging.WARNING, logger="strands_robots.policies.wbc.gait"):
        phases = _walk(None, wall_seconds=0.5)

    warnings = [r for r in caplog.records if "control frequency" in r.message]
    assert len(warnings) == 1, f"expected one latched warning, got {len(warnings)}"
    assert "set_control_frequency" in warnings[0].getMessage()
    # Fallback is the upstream reference period, so a direct caller is unaffected.
    assert phases == pytest.approx(_reference_phases(wall_seconds=0.5))
