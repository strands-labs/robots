"""Fail-soft contract: a raising ``robot_action_keys`` must not mask fail-fast.

``PolicyRunner`` fails fast when EVERY step of the opening probe window drives
zero actuators (100% unresolved keys -- the robot cannot move). While counting
that failure and while building the diagnostic, the runner calls
``SimEngine.robot_action_keys`` twice:

* once up front to seed the per-actuator resolution counters, and
* once again when the probe trips, to name the robot's valid actuator/joint
  names in the raised error.

Both calls are best-effort: the resolution stats and the diagnostic hint are a
convenience, never a correctness invariant. If ``robot_action_keys`` itself
raises (a backend quirk, a mid-rollout world teardown), that secondary failure
must NOT replace the primary, actionable "the robot has not moved" signal with
an opaque traceback. This pins that fail-soft guarantee end to end through the
public MuJoCo ``run_policy`` surface: with ``robot_action_keys`` forced to raise,
a wrong-embodiment policy still surfaces the fail-fast diagnostic (with an empty
valid-keys list, the cached fallback) rather than the backend's raised error.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.base import Policy
from strands_robots.simulation.mujoco.simulation import Simulation


class _WrongEmbodimentPolicy(Policy):
    """Emits a fixed key that resolves to no actuator on the target robot."""

    @property
    def provider_name(self) -> str:
        return "wrong_embodiment_test"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        pass

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return [{"joint_that_does_not_exist": 0.5}]


@pytest.fixture
def sim():
    s = Simulation(mesh=False)
    s.create_world()
    s.add_robot("so101")
    yield s
    s.cleanup()


def test_raising_robot_action_keys_does_not_mask_fail_fast(sim, monkeypatch):
    """robot_action_keys raising -> still the fail-fast signal, not its error."""

    def _boom(robot_name: str) -> list[str]:
        raise RuntimeError("robot_action_keys probe unavailable")

    # Patch both the stats-seeding call and the diagnostic-building call.
    monkeypatch.setattr(sim, "robot_action_keys", _boom)

    result = sim.run_policy(
        robot_name="so101",
        policy_object=_WrongEmbodimentPolicy(),
        n_steps=50,  # would run 50 steps if the probe never tripped
        control_frequency=20.0,
        fast_mode=True,
    )

    assert result["status"] == "error", result
    text = result["content"][0]["text"]
    # The PRIMARY fail-fast diagnostic is what surfaces, not the secondary
    # RuntimeError from the raising robot_action_keys.
    assert "first 3 action steps" in text
    assert "has not moved" in text
    assert "get_features" in text
    assert "probe unavailable" not in text
    # The offending emitted key is still named (fail-soft did not lose it).
    assert "joint_that_does_not_exist" in text
    # Bailed in the probe window; did NOT run the full 50-step episode.
    assert "50 steps" not in text
