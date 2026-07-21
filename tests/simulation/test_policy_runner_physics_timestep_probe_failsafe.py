"""A backend whose ``physics_timestep()`` probe raises must not abort a rollout.

``PolicyRunner`` derives the number of physics substeps per applied action from
the backend's ``physics_timestep()`` so a position-servo arm advances for the
full control period (``round(1 / control_frequency / physics_timestep)``) rather
than a single physics dt. That probe is best-effort: a backend that *raises*
while reporting its timestep must degrade to a single substep and keep running,
never propagate the error and abort the rollout.

Backends that simply cannot report a fixed timestep return ``None`` (the base
``SimEngine.physics_timestep`` default), which is already exercised. These tests
pin the harder failure mode - the probe itself raising - on the shared
``_control_substeps`` helper and on both rollout entry points (``run`` /
``evaluate``).
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "glfw")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.policy_runner import PolicyRunner
from tests.simulation.test_policy_runner import FakeSim


class _RaisingProbeSim(FakeSim):
    """FakeSim whose timestep probe raises; records substeps passed to send_action."""

    def __init__(self, joint_names: tuple[str, ...] = ("j0", "j1", "j2")):
        super().__init__(joint_names)
        self.substeps_seen: list[int] = []

    def physics_timestep(self) -> float | None:
        raise RuntimeError("backend cannot report a fixed timestep")

    def send_action(self, action, robot_name=None, n_substeps=1):
        self.substeps_seen.append(n_substeps)
        return super().send_action(action, robot_name=robot_name)


def test_control_substeps_falls_back_to_one_when_probe_raises():
    """A raising ``physics_timestep`` degrades to a single substep, never raises."""
    substeps = PolicyRunner(_RaisingProbeSim())._control_substeps(control_frequency=50.0)
    assert substeps == 1


def test_run_completes_when_physics_timestep_probe_raises():
    """``run`` finishes a rollout - stepping at 1 substep/action - despite the probe raising."""
    sim = _RaisingProbeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    result = PolicyRunner(sim).run(
        "fake_robot",
        policy,
        duration=0.06,
        control_frequency=50.0,  # -> 3 control steps
        fast_mode=True,
    )

    assert result["status"] == "success"
    assert sim.substeps_seen, "rollout must apply at least one action"
    # The probe failure forces the single-substep fallback on every applied action.
    assert set(sim.substeps_seen) == {1}


def test_evaluate_completes_when_physics_timestep_probe_raises():
    """``evaluate`` runs to completion when the backend's timestep probe raises."""
    sim = _RaisingProbeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    result = PolicyRunner(sim).evaluate(
        "fake_robot",
        policy,
        n_episodes=1,
        max_steps=3,
        control_frequency=50.0,
    )

    assert result["status"] == "success"
    assert sim.substeps_seen, "eval rollout must apply at least one action"
    assert set(sim.substeps_seen) == {1}
