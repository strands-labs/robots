"""Fail-fast contract for the ``SimEngine`` policy-preflight seam.

``SimEngine.run_policy`` / ``eval_policy`` run a provider's cheap class-level
:meth:`~strands_robots.policies.base.Policy.preflight` hook (via
``_preflight_policy_config`` in ``strands_robots.simulation.base``) BEFORE
``create_policy`` builds the policy - and therefore before any model-weight
download. This lets a provider fail fast on a misconfiguration (e.g. a camera
name that cannot be routed to a VLA's declared image inputs) instead of
crashing deep inside the first inference after a multi-minute download.

The contract has three observable rules, pinned here through the public MuJoCo
simulation surface with a runtime-registered provider whose ``preflight``
rejects and whose ``__init__`` raises if it is ever constructed:

1. A ``preflight`` ``ValueError`` is surfaced as a ``status=error`` result and
   the policy is never built (both ``run_policy`` and ``eval_policy``).
2. When the runtime observation is not yet available (empty), the preflight is
   skipped rather than blocking the run.
3. A provider with the default no-op preflight is never blocked.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies import factory as policy_factory
from strands_robots.policies import register_policy
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import Simulation

_REJECT_MESSAGE = "camera 'wrist' cannot be routed to declared image inputs"
_PROVIDER = "preflight_reject_probe"


class _PreflightRejectPolicy(MockPolicy):
    """Mock policy whose preflight always rejects and which must never build."""

    @classmethod
    def preflight(cls, observation_keys, **policy_config):
        raise ValueError(_REJECT_MESSAGE)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise AssertionError("policy must not be constructed when preflight rejects")


_BUILDABLE_PROVIDER = "preflight_reject_buildable_probe"


class _RejectingButBuildablePolicy(MockPolicy):
    """Rejects in preflight but constructs normally - used to prove that when
    the preflight seam is skipped, the run proceeds to build and execute.
    """

    @classmethod
    def preflight(cls, observation_keys, **policy_config):
        raise ValueError(_REJECT_MESSAGE)


@pytest.fixture
def reject_provider():
    """Register the rejecting provider and remove it again after the test."""
    register_policy(_PROVIDER, lambda: _PreflightRejectPolicy)
    try:
        yield _PROVIDER
    finally:
        policy_factory._runtime_registry.pop(_PROVIDER, None)


@pytest.fixture
def sim_with_robot():
    s = Simulation(tool_name="preflight_probe", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    yield s
    s.cleanup()


def test_run_policy_surfaces_preflight_rejection(sim_with_robot, reject_provider):
    result = sim_with_robot.run_policy(
        robot_name="alice",
        policy_provider=reject_provider,
        duration=0.05,
        control_frequency=50,
        fast_mode=True,
    )
    assert result["status"] == "error"
    assert _REJECT_MESSAGE in result["content"][0]["text"]


def test_eval_policy_surfaces_preflight_rejection(sim_with_robot, reject_provider):
    result = sim_with_robot.eval_policy(
        robot_name="alice",
        policy_provider=reject_provider,
        n_episodes=1,
        max_steps=5,
    )
    assert result["status"] == "error"
    assert _REJECT_MESSAGE in result["content"][0]["text"]


def test_preflight_skipped_when_observation_unavailable(sim_with_robot, monkeypatch):
    """When the runtime observation is empty (not yet available), the preflight
    seam must be skipped so it cannot block a run on a transient state. The
    provider rejects in preflight but builds normally, so a success result
    proves the check was bypassed rather than run.
    """
    register_policy(_BUILDABLE_PROVIDER, lambda: _RejectingButBuildablePolicy)
    try:
        monkeypatch.setattr(sim_with_robot, "get_observation", lambda *a, **k: {})
        result = sim_with_robot.run_policy(
            robot_name="alice",
            policy_provider=_BUILDABLE_PROVIDER,
            duration=0.05,
            control_frequency=50,
            fast_mode=True,
        )
        assert result["status"] == "success"
    finally:
        policy_factory._runtime_registry.pop(_BUILDABLE_PROVIDER, None)


def test_valid_provider_is_not_blocked_by_preflight(sim_with_robot):
    """The default no-op preflight (mock provider) must not fail-fast: a
    correctly configured provider still runs and returns success. Guards
    against a preflight seam that rejects everything.
    """
    result = sim_with_robot.run_policy(
        robot_name="alice",
        policy_provider="mock",
        duration=0.05,
        control_frequency=50,
        fast_mode=True,
    )
    assert result["status"] == "success"
