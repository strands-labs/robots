"""Regression tests for the legacy ``success_fn`` evaluation path.

The legacy ``PolicyRunner.evaluate(..., success_fn=...)`` route must consume an
action chunk and check success identically to the benchmark (``spec=``) and
single-rollout (``run()``) routes. Historically it diverged on two points:

* It applied only ``actions[0]`` per ``get_actions`` call and re-queried the
  policy every step, silently ignoring ``action_horizon`` and forcing a
  chunk-predicting VLA out of its trained control regime.
* It evaluated the success predicate against the STALE pre-action observation,
  so success was detected one step late and a task completing on the final step
  was never recorded - under-reporting ``success_rate`` and inflating
  ``avg_steps``.

These tests pin chunk consumption, post-action success checking, and agreement
with the benchmark path so the three rollout routes never drift again.
"""

from __future__ import annotations

import os
import sys
from typing import Any

os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

from strands_robots.policies.base import Policy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.benchmark import BenchmarkProtocol, StepInfo
from strands_robots.simulation.policy_runner import PolicyRunner


class _ClockSim(SimEngine):
    """Minimal sim with a monotonic clock advanced once per ``send_action``.

    ``get_observation`` exposes the clock so a success predicate can be a pure
    function of how many actions have been applied. The clock is the live
    post-action world state - exactly what the benchmark path checks via
    ``is_success(sim)``.
    """

    def __init__(self, joint_names: tuple[str, ...] = ("j0", "j1")):
        self._joint_names = list(joint_names)
        self.clock = 0
        self.send_count = 0
        self.reset_count = 0

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        self.clock = 0
        self.reset_count += 1
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        self.clock += n_steps
        return {"status": "success"}

    def get_state(self):
        return {"step_count": self.clock}

    def add_robot(self, name, **kw):
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return ["fake_robot"]

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._joint_names)

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {"clock": self.clock, **{n: 0.0 for n in self._joint_names}}

    def send_action(self, action, robot_name=None, n_substeps=1):
        self.send_count += 1
        self.clock += 1

    def render(self, camera_name="default", width=None, height=None):
        return {"status": "success", "content": [{"text": "render"}]}


class _ChunkPolicy(Policy):
    """Deterministic policy that emits a fixed-length action chunk per call.

    Counts ``get_actions`` invocations so a test can assert the consumer
    re-queries once per chunk (not once per step). ``actions_per_step`` drives
    ``execution_horizon`` so ``resolve_chunk_length`` sizes the chunk.
    """

    def __init__(self, chunk_size: int = 8):
        super().__init__()
        self.actions_per_step = chunk_size
        self._chunk_size = chunk_size
        self._keys: list[str] = []
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return "chunk_test"

    @property
    def requires_images(self) -> bool:
        return False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self._keys = list(robot_state_keys)

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any):
        self.call_count += 1
        return [{k: 0.0 for k in self._keys} for _ in range(self._chunk_size)]


def test_legacy_eval_consumes_full_action_chunk():
    """A success_fn rollout must apply the whole chunk and re-query once per chunk.

    Pre-fix the loop applied only ``actions[0]`` and re-queried every step, so
    ``get_actions`` was called ``max_steps`` times and ``action_horizon`` was
    ignored. Post-fix it consumes ``max(action_horizon, execution_horizon)``
    actions per inference, so the call count is ``max_steps / chunk``.
    """
    sim = _ClockSim()
    policy = _ChunkPolicy(chunk_size=8)
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    max_steps = 16
    result = PolicyRunner(sim).evaluate(
        "fake_robot",
        policy,
        n_episodes=1,
        max_steps=max_steps,
        action_horizon=8,
    )

    assert result["status"] == "success"
    # 16 steps applied via 8-action chunks -> 2 inference calls, not 16.
    assert policy.call_count == 2, f"expected 2 get_actions calls, got {policy.call_count}"
    assert sim.send_count == max_steps


def test_legacy_eval_records_success_on_final_step():
    """A predicate that turns true exactly on the last applied action reports success.

    Pre-fix the predicate was checked against the pre-action observation, so a
    task completing on the final step exited the loop with ``success=False``.
    """
    sim = _ClockSim()
    policy = _ChunkPolicy(chunk_size=8)
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    max_steps = 8  # one chunk; success lands on the very last applied action

    def succeed_on_final(obs: dict[str, Any]) -> bool:
        return obs["clock"] >= max_steps

    result = PolicyRunner(sim).evaluate(
        "fake_robot",
        policy,
        n_episodes=1,
        max_steps=max_steps,
        success_fn=succeed_on_final,
        action_horizon=8,
    )

    payload = result["content"][1]["json"]
    assert payload["success_rate"] == 1.0, "success on the final step must be recorded"
    assert payload["episodes"][0]["success"] is True
    assert payload["episodes"][0]["steps"] == max_steps


def test_legacy_eval_does_not_succeed_on_stale_preaction_obs():
    """Success must reflect post-action state, never the stale pre-action obs.

    With a predicate keyed to a clock that only advances when an action is
    applied, the success step must equal the clock value at success - proving
    the predicate sees the live post-action observation.
    """
    sim = _ClockSim()
    policy = _ChunkPolicy(chunk_size=4)
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

    def succeed_at_three(obs: dict[str, Any]) -> bool:
        return obs["clock"] >= 3

    result = PolicyRunner(sim).evaluate(
        "fake_robot",
        policy,
        n_episodes=1,
        max_steps=20,
        success_fn=succeed_at_three,
        action_horizon=4,
    )

    payload = result["content"][1]["json"]
    assert payload["episodes"][0]["success"] is True
    # clock hits 3 on the 3rd applied action -> exactly 3 steps, no off-by-one.
    assert payload["episodes"][0]["steps"] == 3


class _ClockBenchmark(BenchmarkProtocol):
    """Benchmark whose success is a pure function of the sim clock.

    ``supported_robots`` is empty so the compat check is skipped, letting the
    same ``_ClockSim`` drive both the legacy and spec routes for a head-to-head
    success-rate comparison.
    """

    max_steps = 20

    def __init__(self, success_at: int):
        self._success_at = success_at

    @property
    def supported_robots(self) -> list[str]:
        return []

    @property
    def default_robot(self) -> str:
        return "fake_robot"

    def on_step(self, sim, obs, action):
        return StepInfo(reward=0.0, done=False)

    def is_success(self, sim) -> bool:
        return sim.clock >= self._success_at

    def is_failure(self, sim) -> bool:
        return False


def test_legacy_and_benchmark_paths_agree_on_success_rate():
    """eval_policy(success_fn=...) and evaluate(spec=...) must agree on the same rollout.

    Same deterministic policy + same deterministic success condition (clock >=
    N) on the same sim. Pre-fix the legacy path's stale-obs + single-action
    consumption made it disagree with the benchmark path on identical rollouts.
    """
    success_at = 5
    max_steps = 20

    sim_a = _ClockSim()
    policy_a = _ChunkPolicy(chunk_size=8)
    policy_a.set_robot_state_keys(sim_a.robot_joint_names("fake_robot"))
    legacy = PolicyRunner(sim_a).evaluate(
        "fake_robot",
        policy_a,
        n_episodes=3,
        max_steps=max_steps,
        success_fn=lambda obs: obs["clock"] >= success_at,
        action_horizon=8,
    )

    sim_b = _ClockSim()
    policy_b = _ChunkPolicy(chunk_size=8)
    policy_b.set_robot_state_keys(sim_b.robot_joint_names("fake_robot"))
    benchmark = PolicyRunner(sim_b).evaluate(
        "fake_robot",
        policy_b,
        n_episodes=3,
        spec=_ClockBenchmark(success_at=success_at),
        action_horizon=8,
    )

    legacy_json = legacy["content"][1]["json"]
    bench_json = benchmark["content"][1]["json"]
    assert legacy_json["success_rate"] == bench_json["success_rate"] == 1.0
    # Both detect success on the same applied action -> identical step counts.
    assert legacy_json["avg_steps"] == bench_json["avg_steps"]
