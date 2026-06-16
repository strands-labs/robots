"""Regression tests for per-call ``policy_kwargs`` forwarding through the run loop.

Non-VLA providers (cuRobo, MoveIt2, WBC) read their goal from the well-known
``**kwargs`` keys on :meth:`Policy.get_actions` (issue #300:
``target_pose`` / ``target_joints`` / ``target_velocity`` / ``world_update``).
The mesh ``tell()`` path already forwards those keys, but the local-sim path
(``sim.run_policy``/``start_policy`` -> ``PolicyRunner.run`` ->
``policy.get_actions``) historically dropped them: ``get_actions`` was called
positionally as ``get_actions(observation, instruction)`` with no kwargs.

These tests pin the fix: a ``policy_kwargs`` dict passed to ``run_policy`` /
``PolicyRunner.run`` must arrive verbatim at every ``get_actions`` call, and the
default (``None`` / omitted) must reproduce the historical no-kwargs behaviour.

A future refactor that drops the forwarding will fail here.
"""

from __future__ import annotations

from typing import Any

from strands_robots.policies.base import Policy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner


class FakeSim(SimEngine):
    """Minimal ``SimEngine`` stub - no physics, just enough for the run loop.

    Self-contained (does not import the heavier ``test_policy_runner.FakeSim``)
    so this regression file stays decoupled from that module's optional-dep
    imports.
    """

    def __init__(self, joint_names: tuple[str, ...] = ("j0", "j1", "j2")) -> None:
        self._joint_names = list(joint_names)
        self._robots = {"fake_robot": self._joint_names}

    def create_world(self, timestep=None, gravity=None, ground_plane=True):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def destroy(self):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def reset(self):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def step(self, n_steps: int = 1):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def get_state(self):  # type: ignore[no-untyped-def]
        return {"sim_time": 0.0, "step_count": 0}

    def add_robot(self, name, **kw):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def remove_robot(self, name):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._robots.get(robot_name, []))

    def add_object(self, name, **kw):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def remove_object(self, name):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):  # type: ignore[no-untyped-def]
        return {n: 0.0 for n in self._joint_names}

    def send_action(self, action, robot_name=None, n_substeps=1):  # type: ignore[no-untyped-def]
        return {"status": "success"}

    def render(self, camera_name="default", width=None, height=None):  # type: ignore[no-untyped-def]
        return {"status": "success", "content": []}


class _GoalRecordingPolicy(Policy):
    """Non-VLA policy that records the ``**kwargs`` of every ``get_actions`` call.

    Mirrors the cuRobo / WBC shape: ``requires_images = False`` and the goal
    arrives through kwargs rather than the instruction string. Emits a single
    zero action per call so the runner advances normally.
    """

    def __init__(self) -> None:
        self._keys: list[str] = []
        self.kwargs_seen: list[dict[str, Any]] = []

    @property
    def provider_name(self) -> str:
        return "goal_recording"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = list(robot_state_keys)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        # Record exactly what the run loop forwarded this tick.
        self.kwargs_seen.append(dict(kwargs))
        return [{k: 0.0 for k in self._keys}]


def _run(policy: Policy, *, policy_kwargs: dict[str, Any] | None, n_steps: int = 3) -> dict[str, Any]:
    sim = FakeSim()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
    return PolicyRunner(sim).run(
        "fake_robot",
        policy,
        instruction="walk",
        duration=float(n_steps) / 50.0,
        control_frequency=50.0,
        action_horizon=1,
        fast_mode=True,
        policy_kwargs=policy_kwargs,
    )


def test_policy_kwargs_reach_get_actions() -> None:
    """A goal payload passed to PolicyRunner.run arrives at every get_actions call."""
    policy = _GoalRecordingPolicy()
    goal = {"target_velocity": [0.5, 0.0, 0.0]}

    result = _run(policy, policy_kwargs=goal)

    assert result["status"] == "success"
    assert policy.kwargs_seen, "policy.get_actions was never called"
    # Every tick saw the forwarded goal verbatim.
    for seen in policy.kwargs_seen:
        assert seen == goal
    # Forwarding must not alias the caller's dict (defensive copy on read).
    assert all(seen is not goal for seen in policy.kwargs_seen)


def test_none_policy_kwargs_reproduces_no_kwargs_behaviour() -> None:
    """Omitting policy_kwargs (the historical path) forwards no extra kwargs."""
    policy = _GoalRecordingPolicy()

    result = _run(policy, policy_kwargs=None)

    assert result["status"] == "success"
    assert policy.kwargs_seen, "policy.get_actions was never called"
    assert all(seen == {} for seen in policy.kwargs_seen)


def test_multiple_goal_keys_forwarded_together() -> None:
    """All well-known #300 keys ride through in a single payload."""
    policy = _GoalRecordingPolicy()
    goal = {
        "target_pose": [0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
        "target_joints": {"j0": 0.1},
        "target_velocity": [0.5, 0.0, 0.2],
        "world_update": {"cuboid": {}},
    }

    result = _run(policy, policy_kwargs=goal)

    assert result["status"] == "success"
    assert policy.kwargs_seen[0] == goal


def test_run_policy_threads_policy_kwargs_through_base() -> None:
    """SimEngine.run_policy(policy_kwargs=...) reaches the policy via policy_object.

    Exercises the full public entry point (not just PolicyRunner directly) so
    the base-class signature + forwarding are both pinned.
    """
    sim = FakeSim()
    policy = _GoalRecordingPolicy()
    goal = {"target_velocity": [0.25, 0.0, 0.0]}

    result = sim.run_policy(
        robot_name="fake_robot",
        policy_object=policy,
        instruction="walk forward",
        n_steps=2,
        control_frequency=50.0,
        action_horizon=1,
        fast_mode=True,
        policy_kwargs=goal,
    )

    assert result["status"] == "success"
    assert policy.kwargs_seen, "policy.get_actions was never called"
    assert all(seen == goal for seen in policy.kwargs_seen)
