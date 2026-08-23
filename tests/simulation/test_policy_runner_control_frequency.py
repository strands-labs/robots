"""``PolicyRunner`` tells the policy the loop's control rate before running.

Latency-sensitive providers (Real-Time Chunking) convert measured wall-clock
inference latency into a count of consumed action steps using the control rate.
The runner is the only component that knows that rate, so it MUST call
``policy.set_control_frequency`` before the rollout loop - otherwise the policy
falls back to a hardcoded rate and mis-blends the chunk seam at every other
frequency. These tests pin that the rate is propagated on every rollout entry
point (``run`` / ``evaluate``).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.policy_runner import PolicyRunner
from tests.simulation.test_policy_runner import FakeSim


def test_run_sets_control_frequency_on_policy():
    sim = FakeSim()
    policy = MockPolicy()
    assert policy.control_frequency is None
    PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=0.04,
        control_frequency=120.0,
        fast_mode=True,
    )
    assert policy.control_frequency == 120.0


def test_run_default_control_frequency_propagated():
    sim = FakeSim()
    policy = MockPolicy()
    PolicyRunner(sim).run("fake_robot", policy, duration=0.04, fast_mode=True)
    # PolicyRunner.run default control_frequency is 50.0.
    assert policy.control_frequency == 50.0


def test_evaluate_sets_control_frequency_on_policy():
    sim = FakeSim()
    policy = MockPolicy()
    PolicyRunner(sim).evaluate(
        "fake_robot",
        policy,
        n_episodes=1,
        max_steps=3,
        control_frequency=200.0,
    )
    assert policy.control_frequency == 200.0
