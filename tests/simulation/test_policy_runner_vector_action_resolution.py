"""Per-actuator resolution stats must credit numeric-vector policy actions.

``Simulation.send_action`` accepts an ordered numeric vector (list / tuple /
1-D array) and binds it positionally to ``robot_action_keys(robot_name)``, so a
policy may emit a raw action vector per tick instead of a ``{joint: value}``
dict. ``PolicyRunner`` tracks per-actuator resolution (issue #165): the fraction
of steps each actuator was actually driven, surfaced as ``action_resolution_rate``
and aggregated into ``partial_action_failure_rate``.

For a dict action the runner credits the emitted keys; for a numeric vector there
are no keys, so it must credit EVERY actuator positionally (the same convention
``send_action`` uses to apply the vector). If that crediting regressed, a
vector-returning policy would be falsely reported as 100% under-actuated every
step (``action_resolution_rate`` all 0.0, ``partial_action_failure_rate`` 1.0)
despite driving the robot every tick. This test locks the numeric-vector
accounting in place.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.base import Policy
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner


class _VectorActionPolicy(Policy):
    """Minimal policy that emits one positional action vector per tick."""

    def __init__(self, n_actuators: int) -> None:
        self._n = n_actuators

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any) -> list[Any]:
        # A single-action chunk whose element is a numeric vector, not a dict.
        return [[0.0] * self._n]

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = robot_state_keys

    @property
    def requires_images(self) -> bool:
        return False

    @property
    def provider_name(self) -> str:
        return "vector_test"


@pytest.fixture
def sim_with_robot():
    s = Simulation(tool_name="vector_action_test", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    yield s
    s.cleanup()


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    for block in result.get("content", []):
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError("result has no json payload block")


class TestVectorActionResolutionStats:
    def test_vector_action_credits_every_actuator(self, sim_with_robot):
        """Every actuator resolves each step -> resolution 1.0, no under-actuation."""
        actuators = sim_with_robot.robot_action_keys("alice")
        policy = _VectorActionPolicy(len(actuators))
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        result = PolicyRunner(sim_with_robot).run(
            "alice",
            policy,
            duration=0.2,
            control_frequency=50,
            fast_mode=True,
        )

        assert result["status"] == "success"
        payload = _payload(result)
        resolution = payload["action_resolution_rate"]
        # A vector binds positionally to every actuator, so each is driven every step.
        assert set(resolution) == set(actuators)
        assert all(rate == 1.0 for rate in resolution.values()), resolution
        # Aggregate: nothing under-driven, no step-level send_action error.
        assert payload["partial_action_failure_rate"] == 0.0
        assert payload["action_errors"] == 0
