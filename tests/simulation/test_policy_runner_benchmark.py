"""Tests for ``PolicyRunner.evaluate`` with a :class:`BenchmarkProtocol` spec.

Covers the new spec-driven evaluation path:

* Cumulative reward accounting across an episode.
* Early termination on ``is_success`` / ``is_failure`` / ``StepInfo.done``.
* Per-episode seeded RNG so identical seeds produce identical rollouts.
* Robot-compatibility error surfaces as a structured error dict (not raises).
* Legacy ``success_fn`` path still works.
* Passing both ``spec`` and ``success_fn`` is an error.
* :meth:`SimEngine.evaluate_benchmark` facade end-to-end.
* :meth:`SimEngine.list_benchmarks` / :meth:`register_benchmark_from_file`
  facades return structured dicts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.benchmark import (
    _BENCHMARK_REGISTRY,
    BenchmarkProtocol,
    StepInfo,
    register_benchmark,
)
from strands_robots.simulation.policy_runner import PolicyRunner

# Fixtures


@pytest.fixture(autouse=True)
def _clean_registry():
    snapshot = dict(_BENCHMARK_REGISTRY)
    _BENCHMARK_REGISTRY.clear()
    yield
    _BENCHMARK_REGISTRY.clear()
    _BENCHMARK_REGISTRY.update(snapshot)


class _FakeRobot:
    """Simple object with a data_config attr so the base compat check passes."""

    def __init__(self, data_config: str):
        self.data_config = data_config


class _FakeWorld:
    """Minimal world with a ``robots`` dict - enough for the base compat check."""

    def __init__(self, robots: dict[str, _FakeRobot]):
        self.robots = dict(robots)


class FakeSim(SimEngine):
    """Minimal ``SimEngine`` that records a few per-episode counters.

    Deliberately stripped-down: no cameras, no physics - just enough for
    :class:`PolicyRunner` to step through.
    """

    def __init__(self, joint_names: tuple[str, ...] = ("j0", "j1"), data_config: str = "so100"):
        self._joint_names = list(joint_names)
        self._data_config = data_config
        self._step_count = 0
        self._reset_count = 0
        self._world = _FakeWorld({"fake_robot": _FakeRobot(data_config)})

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        self._step_count = 0
        self._reset_count += 1
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        self._step_count += n_steps
        return {"status": "success"}

    def get_state(self):
        return {"step_count": self._step_count}

    def add_robot(self, name, **kw):
        data_config = kw.get("data_config") or self._data_config
        self._world.robots[name] = _FakeRobot(data_config)
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return list(self._world.robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._joint_names)

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {n: 0.0 for n in self._joint_names}

    def send_action(self, action, robot_name=None, n_substeps=1):
        self._step_count += 1

    def render(self, camera_name="default", width=None, height=None):
        return {"status": "success", "content": [{"text": "render"}]}


# Test benchmarks


class _CountingBenchmark(BenchmarkProtocol):
    """Benchmark that tracks how many times each hook was called and rewards +1/step."""

    max_steps = 20

    def __init__(self, *, success_after: int = 10**9, fail_after: int = 10**9):
        self.success_after = success_after
        self.fail_after = fail_after
        self.on_episode_start_calls = 0
        self.on_step_calls = 0
        self.rng_seeds_seen: list[int] = []

    @property
    def supported_robots(self) -> list[str]:
        return ["so100"]

    @property
    def default_robot(self) -> str:
        return "so100"

    def on_episode_start(self, sim, rng):
        self.on_episode_start_calls += 1
        # Record the first draw for reproducibility tests.
        self.rng_seeds_seen.append(rng.randint(0, 1000000))
        super().on_episode_start(sim, rng)

    def on_step(self, sim, obs, action):
        self.on_step_calls += 1
        return StepInfo(reward=1.0)

    def is_success(self, sim):
        return self.on_step_calls >= self.success_after

    def is_failure(self, sim):
        return self.on_step_calls >= self.fail_after


class _DoneAfterBenchmark(BenchmarkProtocol):
    """Returns StepInfo(done=True) on the Nth step."""

    max_steps = 50

    def __init__(self, done_after: int):
        self._done_after = done_after
        self._step = 0

    @property
    def supported_robots(self) -> list[str]:
        return []

    @property
    def default_robot(self) -> str:
        return "so100"

    def on_step(self, sim, obs, action):
        self._step += 1
        return StepInfo(reward=0.5, done=self._step >= self._done_after)

    def is_success(self, sim):
        return False

    def is_failure(self, sim):
        return False


# Cumulative reward


class TestCumulativeReward:
    def test_sums_reward_across_steps(self):
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        spec = _CountingBenchmark()

        result = PolicyRunner(sim).evaluate(
            "fake_robot",
            policy,
            spec=spec,
            n_episodes=1,
        )
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        # _CountingBenchmark rewards +1/step; max_steps=20 → 20.0 cumulative.
        assert payload["episodes"][0]["cumulative_reward"] == pytest.approx(20.0)
        assert payload["avg_reward"] == pytest.approx(20.0)

    def test_success_terminates_and_stops_reward_accumulation(self):
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        spec = _CountingBenchmark(success_after=5)

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=spec, n_episodes=1)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_success"] == 1
        # Reward per step = 1; terminates on the 5th step.
        assert payload["episodes"][0]["cumulative_reward"] == pytest.approx(5.0)
        assert payload["episodes"][0]["steps"] == 5

    def test_failure_marks_episode_unsuccessful(self):
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        spec = _CountingBenchmark(fail_after=3)

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=spec, n_episodes=1)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_failure"] == 1
        assert payload["n_success"] == 0
        assert payload["episodes"][0]["failure"] is True
        assert payload["episodes"][0]["success"] is False

    def test_done_flag_terminates_episode(self):
        """StepInfo.done=True ends the episode even without is_success/is_failure."""
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        spec = _DoneAfterBenchmark(done_after=4)

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=spec, n_episodes=1)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["episodes"][0]["steps"] == 4


# Seed reproducibility


class TestSeedReproducibility:
    def test_same_seed_same_rng_draws(self):
        """Two evaluations with the same seed must produce identical per-episode RNG draws."""
        sim1 = FakeSim()
        sim2 = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim1.robot_joint_names("fake_robot"))

        spec1 = _CountingBenchmark()
        spec2 = _CountingBenchmark()

        PolicyRunner(sim1).evaluate("fake_robot", policy, spec=spec1, n_episodes=3, seed=42)
        PolicyRunner(sim2).evaluate("fake_robot", policy, spec=spec2, n_episodes=3, seed=42)

        assert spec1.rng_seeds_seen == spec2.rng_seeds_seen

    def test_different_seed_different_rng_draws(self):
        sim1 = FakeSim()
        sim2 = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim1.robot_joint_names("fake_robot"))

        spec1 = _CountingBenchmark()
        spec2 = _CountingBenchmark()

        PolicyRunner(sim1).evaluate("fake_robot", policy, spec=spec1, n_episodes=3, seed=42)
        PolicyRunner(sim2).evaluate("fake_robot", policy, spec=spec2, n_episodes=3, seed=999)

        assert spec1.rng_seeds_seen != spec2.rng_seeds_seen

    def test_seed_recorded_in_per_episode_results(self):
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=_CountingBenchmark(), n_episodes=3, seed=7)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        # Every episode has a distinct seed derived from the master seed.
        seeds = [e["seed"] for e in payload["episodes"]]
        assert len(set(seeds)) == 3
        assert payload["seed"] == 7


# Robot compatibility via the spec path


class TestRobotCompatibility:
    def test_mismatched_robot_returns_structured_error(self):
        """Spec with supported=['panda'] vs sim loaded with 'so100' → structured error."""

        class _PandaOnly(_CountingBenchmark):
            @property
            def supported_robots(self) -> list[str]:
                return ["panda"]

            @property
            def default_robot(self) -> str:
                return "panda"

        sim = FakeSim(data_config="so100")  # loaded with so100
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=_PandaOnly(), n_episodes=1)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "compatibility" in text.lower() or "supported" in text.lower()
        assert "so100" in text  # shows the offending data_config
        assert "panda" in text  # shows the allowed list


# augment_observation hook integration (#156)


class TestAugmentObservationHook:
    """The eval loop must call ``spec.augment_observation`` between
    ``sim.get_observation()`` and ``policy.get_actions()`` and feed the
    augmented obs to the policy.
    """

    def test_hook_output_reaches_policy(self):
        """The augmented observation - not the raw sim obs - is what
        ``policy.get_actions(observation, instruction)`` sees."""

        captured: list[dict[str, Any]] = []

        class _CapturePolicy:
            requires_images = False

            def __init__(self):
                self.robot_state_keys: list[str] = []

            def set_robot_state_keys(self, keys):
                self.robot_state_keys = list(keys)

            def get_actions(self, obs, instruction):
                captured.append(dict(obs))
                return [{"j0": 0.0, "j1": 0.0}]

        class _AugSpec(_CountingBenchmark):
            def augment_observation(self, sim, obs):  # type: ignore[override]
                merged = dict(obs)
                merged["x"] = 0.42  # injected key
                return merged

        sim = FakeSim()
        policy = _CapturePolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=_AugSpec(), n_episodes=1)
        assert result["status"] == "success"
        # Every step in the episode ran augment_observation.
        assert all(o.get("x") == 0.42 for o in captured)
        # The raw sim obs (joints only) is still in the dict too.
        assert all("j0" in o for o in captured)

    def test_hook_failure_aborts_with_structured_error(self):
        """If the spec's augment_observation raises, the eval loop returns
        a structured error rather than letting the exception bubble out."""

        class _BoomSpec(_CountingBenchmark):
            def augment_observation(self, sim, obs):  # type: ignore[override]
                raise RuntimeError("intentional failure")

        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=_BoomSpec(), n_episodes=1)
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "augment_observation failed" in text
        assert "intentional failure" in text


# Legacy success_fn path still works


class TestBackwardCompatibility:
    def test_legacy_success_fn_callable_still_works(self):
        """The pre-PR success_fn=callable path must be unchanged."""
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

        result = PolicyRunner(sim).evaluate(
            "fake_robot",
            policy,
            n_episodes=2,
            max_steps=5,
            success_fn=lambda _obs: True,
        )
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["success_rate"] == 1.0
        # Legacy path doesn't emit cumulative_reward; just the pre-PR schema.
        assert "cumulative_reward" not in payload

    def test_legacy_success_fn_none_returns_zero_success(self):
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

        result = PolicyRunner(sim).evaluate("fake_robot", policy, n_episodes=1, max_steps=3, success_fn=None)
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["n_success"] == 0

    def test_cannot_pass_both_spec_and_success_fn(self):
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))

        result = PolicyRunner(sim).evaluate(
            "fake_robot",
            policy,
            spec=_CountingBenchmark(),
            success_fn=lambda _obs: True,
        )
        assert result["status"] == "error"
        assert "both" in result["content"][0]["text"].lower()


# SimEngine facade


class TestSimEngineFacades:
    def test_evaluate_benchmark_dispatches_to_runner(self):
        sim = FakeSim()
        spec = _CountingBenchmark()
        register_benchmark("eval-test", spec)

        result = sim.evaluate_benchmark(
            benchmark_name="eval-test",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=1,
            seed=3,
        )
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert payload["benchmark_class"] == "_CountingBenchmark"
        # Full per-step traversal (nothing terminates early).
        assert payload["episodes"][0]["steps"] == 20

    def test_evaluate_benchmark_unknown_name(self):
        sim = FakeSim()
        result = sim.evaluate_benchmark(benchmark_name="never-registered")
        assert result["status"] == "error"
        assert "no benchmark registered" in result["content"][0]["text"].lower()

    def test_evaluate_benchmark_auto_picks_sole_robot(self):
        """Single-robot scene: robot_name can be omitted."""
        sim = FakeSim()
        register_benchmark("auto-robot", _CountingBenchmark())
        result = sim.evaluate_benchmark(benchmark_name="auto-robot", n_episodes=1)
        assert result["status"] == "success"

    def test_evaluate_benchmark_requires_robot_name_in_multi_robot(self):
        sim = FakeSim()
        sim.add_robot("second", data_config="so100")
        register_benchmark("multi", _CountingBenchmark())
        result = sim.evaluate_benchmark(benchmark_name="multi")
        assert result["status"] == "error"
        assert "robot_name" in result["content"][0]["text"]

    def test_list_benchmarks_returns_snapshot(self):
        sim = FakeSim()
        register_benchmark("a", _CountingBenchmark())
        register_benchmark("b", _CountingBenchmark())
        result = sim.list_benchmarks()
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert set(payload["benchmarks"].keys()) == {"a", "b"}

    def test_list_benchmarks_empty(self):
        sim = FakeSim()
        result = sim.list_benchmarks()
        assert result["status"] == "success"
        assert "No benchmarks" in result["content"][0]["text"]

    def test_register_benchmark_from_file_success(self, tmp_path: Path):
        sim = FakeSim()
        spec_path = tmp_path / "s.json"
        spec_path.write_text(
            json.dumps(
                {
                    "name": "file-bench",
                    "default_robot": "so100",
                    "supported_robots": ["so100"],
                    "max_steps": 7,
                }
            )
        )
        result = sim.register_benchmark_from_file(benchmark_name="file-bench", spec_path=str(spec_path))
        assert result["status"] == "success"
        assert "Registered benchmark" in result["content"][0]["text"]

    def test_register_benchmark_from_file_missing_file(self, tmp_path: Path):
        sim = FakeSim()
        result = sim.register_benchmark_from_file(benchmark_name="missing", spec_path=str(tmp_path / "nope.json"))
        assert result["status"] == "error"
        assert "not found" in result["content"][0]["text"].lower()

    def test_register_benchmark_from_file_empty_name(self):
        sim = FakeSim()
        result = sim.register_benchmark_from_file(benchmark_name="", spec_path="/tmp/x.json")
        assert result["status"] == "error"
        assert "benchmark_name" in result["content"][0]["text"]

    def test_register_benchmark_from_file_bad_schema(self, tmp_path: Path):
        sim = FakeSim()
        spec_path = tmp_path / "bad.json"
        spec_path.write_text('{"name": "x"}')  # missing default_robot
        result = sim.register_benchmark_from_file(benchmark_name="bad", spec_path=str(spec_path))
        assert result["status"] == "error"
        assert "default_robot" in result["content"][0]["text"]


class TestActionHorizon:
    """Round 34 (#168): ``evaluate_benchmark`` accepts ``action_horizon``
    to control how many actions are consumed per ``policy.get_actions``
    inference call.

    Round 36 (#168): default flipped from ``1`` to ``8`` to match
    NVIDIA's upstream GR00T LIBERO eval (``MultiStepWrapper`` with
    ``n_action_steps=8``). GR00T-N1.7-LIBERO checkpoints were trained
    against 8-step open-loop chunk replay, so an ``action_horizon=1``
    default put eval out-of-distribution from training. Set
    ``action_horizon=1`` explicitly for closed-loop receding-horizon
    control (OpenVLA convention).
    """

    def test_default_action_horizon_is_eight_chunk_replay(self):
        """Round 36 (#168): default ``action_horizon=8`` consumes up to
        eight actions per ``policy.get_actions`` call before re-querying.

        ``MockPolicy`` returns 8 actions per call. With
        ``action_horizon=8`` all eight are applied per inference.
        Pin the chunk-replay default so users running GR00T-N1.7-LIBERO
        match NVIDIA's reference eval setup."""
        sim = FakeSim()
        spec = _CountingBenchmark()
        register_benchmark("default-horizon", spec)

        result = sim.evaluate_benchmark(
            benchmark_name="default-horizon",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=1,
            seed=3,
        )
        assert result["status"] == "success", result
        payload = next(c["json"] for c in result["content"] if "json" in c)
        # max_steps=20 ⇒ 2 chunks of 8 (16) + 4 from the 3rd chunk
        # (mid-chunk cap) = 20. on_step still called per APPLIED
        # action.
        assert spec.on_step_calls == 20
        assert payload["episodes"][0]["steps"] == 20

    def test_action_horizon_greater_than_one_consumes_chunk(self):
        """Round 34 (#168): ``action_horizon=4`` consumes 4 actions per
        policy.get_actions call. With ``MockPolicy`` returning 8 actions
        per call, 4 of them get applied, then we re-query.

        ``max_steps=20`` ⇒ 5 inferences × 4 actions = 20 step total.
        ``on_step_calls`` matches because the loop calls on_step per
        applied action, not per inference."""
        sim = FakeSim()
        spec = _CountingBenchmark()
        register_benchmark("h-equals-4", spec)

        result = sim.evaluate_benchmark(
            benchmark_name="h-equals-4",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=1,
            seed=3,
            action_horizon=4,
        )
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        # max_steps=20 ⇒ 5 inferences × 4 actions per chunk = 20 steps
        # (on_step still called per APPLIED action, which is 20).
        assert spec.on_step_calls == 20
        assert payload["episodes"][0]["steps"] == 20

    def test_action_horizon_caps_at_max_steps_mid_chunk(self):
        """Round 34 (#168): when ``max_steps`` is reached mid-chunk, the
        remaining actions in the chunk are NOT applied. Pin so the
        chunk-replay logic respects the spec's ``max_steps`` bound and
        doesn't run physics past the episode budget."""
        sim = FakeSim()
        # max_steps=15, so 3 full chunks of 4 (12 actions) then 3 of
        # the 4 from the 4th chunk (15 total). The 4th action of the
        # 4th chunk should NOT fire.
        spec = _CountingBenchmark()
        spec.max_steps = 15  # type: ignore[misc]
        register_benchmark("max-steps-mid-chunk", spec)

        result = sim.evaluate_benchmark(
            benchmark_name="max-steps-mid-chunk",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=1,
            seed=3,
            action_horizon=4,
        )
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert spec.on_step_calls == 15
        assert payload["episodes"][0]["steps"] == 15

    def test_action_horizon_zero_rejected(self):
        """Round 34 (#168): ``action_horizon=0`` is invalid and rejected
        with a structured error. Pin so a typo doesn't silently produce
        an episode that never applies any action and exits at
        ``max_steps`` with success_rate=0."""
        sim = FakeSim()
        register_benchmark("h-zero", _CountingBenchmark())
        result = sim.evaluate_benchmark(
            benchmark_name="h-zero",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=1,
            action_horizon=0,
        )
        assert result["status"] == "error"
        assert "action_horizon" in result["content"][0]["text"]

    def test_action_horizon_negative_rejected(self):
        """Round 34 (#168): negative ``action_horizon`` is rejected."""
        sim = FakeSim()
        register_benchmark("h-neg", _CountingBenchmark())
        result = sim.evaluate_benchmark(
            benchmark_name="h-neg",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=1,
            action_horizon=-1,
        )
        assert result["status"] == "error"
        assert "action_horizon" in result["content"][0]["text"]

    def test_early_success_terminates_within_chunk(self):
        """Round 34 (#168): when ``is_success`` flips mid-chunk, the
        episode terminates and remaining chunk actions are NOT applied.

        Pin so reward / step accounting reflects the actual applied
        actions, not the full chunk that was queued."""
        sim = FakeSim()
        # Success after exactly 5 steps ⇒ should NOT consume the 6th.
        spec = _CountingBenchmark(success_after=5)
        register_benchmark("early-success-chunk", spec)

        result = sim.evaluate_benchmark(
            benchmark_name="early-success-chunk",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=1,
            seed=3,
            action_horizon=8,
        )
        assert result["status"] == "success"
        payload = next(c["json"] for c in result["content"] if "json" in c)
        assert spec.on_step_calls == 5
        assert payload["episodes"][0]["steps"] == 5
        assert payload["episodes"][0]["success"] is True


class TestEvalSeeding:
    """Round 38 (#168): ``_evaluate_with_spec`` calls ``_set_eval_seed``
    once before the episode loop to seed Python / NumPy / torch / cuDNN
    so policy stochastic ops (e.g. sampling, dropout) are reproducible
    across re-runs.

    Mirrors NVIDIA's upstream ``set_seed`` in
    ``Isaac-GR00T/scripts/deployment/standalone_inference_script.py:81``,
    minus the global ``CUBLAS_WORKSPACE_CONFIG`` env var and
    ``torch.use_deterministic_algorithms(...)`` flag that would persist
    after the eval. Tests target the seeding helper directly + verify
    the per-episode RNG path still works via ``episode_rng``.
    """

    def test_set_eval_seed_seeds_python_random(self):
        """Round 38 (#168): ``_set_eval_seed`` seeds the Python ``random``
        module so two calls with the same seed produce the same draw.

        Pin the basic contract in case future refactors move the seeding
        out of the helper."""
        import random as _stdlib_random

        from strands_robots.simulation.policy_runner import _set_eval_seed

        _set_eval_seed(42)
        first = [_stdlib_random.random() for _ in range(5)]
        _set_eval_seed(42)
        second = [_stdlib_random.random() for _ in range(5)]
        assert first == second

    def test_set_eval_seed_seeds_numpy(self):
        """Round 38 (#168): ``_set_eval_seed`` seeds NumPy's legacy
        global RNG (``np.random.seed``). Pin so policies that use
        ``np.random.rand`` etc. are reproducible across re-runs."""
        import numpy as np

        from strands_robots.simulation.policy_runner import _set_eval_seed

        _set_eval_seed(42)
        first = np.random.rand(5).tolist()
        _set_eval_seed(42)
        second = np.random.rand(5).tolist()
        assert first == second

    def test_set_eval_seed_tolerates_missing_torch(self, monkeypatch):
        """Round 38 (#168): ``_set_eval_seed`` no-ops the torch branch
        when torch isn't importable (mock-policy / minimal-CI installs).

        Pin so the function works on installs without torch — the
        ``ImportError`` should be swallowed silently."""
        import builtins
        import sys

        from strands_robots.simulation.policy_runner import _set_eval_seed

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise ImportError(f"simulated missing torch ({name})")
            return real_import(name, *args, **kwargs)

        # Drop any cached torch modules so the lazy import inside
        # _set_eval_seed actually goes through fake_import.
        for mod in list(sys.modules):
            if mod == "torch" or mod.startswith("torch."):
                monkeypatch.delitem(sys.modules, mod, raising=False)
        monkeypatch.setattr(builtins, "__import__", fake_import)

        # Should not raise.
        _set_eval_seed(42)

    def test_evaluate_benchmark_re_run_yields_same_episode_count(self):
        """Round 38 (#168): re-running ``evaluate_benchmark`` with the
        same seed and a deterministic spec produces the same outcome.

        ``MockPolicy`` is deterministic so the per-step trajectory is
        identical regardless of seed; this test pins that the seeding
        path doesn't accidentally introduce non-determinism (e.g. by
        re-seeding mid-episode)."""
        sim_a = FakeSim()
        sim_b = FakeSim()
        spec_a = _CountingBenchmark()
        spec_b = _CountingBenchmark()
        register_benchmark("seed-rerun-a", spec_a)
        register_benchmark("seed-rerun-b", spec_b)

        result_a = sim_a.evaluate_benchmark(
            benchmark_name="seed-rerun-a",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=2,
            seed=12345,
        )
        result_b = sim_b.evaluate_benchmark(
            benchmark_name="seed-rerun-b",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=2,
            seed=12345,
        )
        assert result_a["status"] == "success"
        assert result_b["status"] == "success"
        payload_a = next(c["json"] for c in result_a["content"] if "json" in c)
        payload_b = next(c["json"] for c in result_b["content"] if "json" in c)
        assert payload_a["success_rate"] == payload_b["success_rate"]
        assert payload_a["n_episodes"] == payload_b["n_episodes"]
        assert spec_a.on_step_calls == spec_b.on_step_calls

    def test_set_eval_seed_is_public(self):
        """#179 — ``set_eval_seed`` is part of the public API surface
        (no leading underscore, exported via ``__all__``).

        Pre-#179 the function was named ``_set_eval_seed`` and only
        invoked from ``_evaluate_with_spec``. Standalone integration
        tests in ``tests_integ/.../test_libero_10_scene5_mujoco_engine_success_rate``
        bypass ``evaluate_benchmark`` and need to call this directly to
        get reproducible policy rollouts. Pin both the public name and
        the backward-compat alias so neither rename breaks consumers
        silently.
        """
        from strands_robots.simulation import policy_runner
        from strands_robots.simulation.policy_runner import (
            _set_eval_seed,
            set_eval_seed,
        )

        # Public function exists.
        assert callable(set_eval_seed)
        # Backward-compat alias points at the same function.
        assert _set_eval_seed is set_eval_seed
        # Exposed in __all__.
        assert "set_eval_seed" in policy_runner.__all__

    def test_set_eval_seed_torch_reproducibility(self):
        """#179 — ``set_eval_seed`` seeds torch's CPU + CUDA RNGs and
        cuDNN flags so the GR00T diffusion sampler (and any other
        ``torch.randn``-driven policy) draws identical sequences across
        runs.

        Pin the contract on torch CPU draws (CUDA path requires a GPU
        and is exercised by the integration tests).
        """
        torch = pytest.importorskip("torch")
        from strands_robots.simulation.policy_runner import set_eval_seed

        set_eval_seed(42)
        first = [float(x) for x in torch.randn(5)]
        set_eval_seed(42)
        second = [float(x) for x in torch.randn(5)]
        assert first == second, (
            f"set_eval_seed(42) should make torch.randn draws bit-identical; got first={first}, second={second}"
        )
        # cuDNN flags pinned to deterministic regardless of CUDA availability.
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_evaluate_benchmark_reseeds_per_episode(self):
        """#179 — ``_evaluate_with_spec`` calls ``set_eval_seed`` at the
        start of EACH episode, not just once before the loop.

        Without per-episode reseeding, every torch op draws from a
        global RNG state that mutates across episodes — so a
        diffusion-based policy produces different action chunks per
        re-run even at the same ``seed=42`` because torch's RNG state
        depends on the cumulative number of draws across all preceding
        episodes.

        Mechanism: capture every ``random.random()`` call inside the
        spec's ``on_episode_start`` (which fires AFTER the per-episode
        ``set_eval_seed`` call). On re-run with the same master seed,
        episode N's draws must match episode N's draws from the first
        run; AND consecutive episodes must NOT be identical (proves we
        seed with episode_seed, not master seed).
        """
        # Closure-captured list that the spec writes to.
        run_a_draws: list[list[float]] = []

        class _RunACapture(_CountingBenchmark):
            def on_episode_start(self, sim, episode_rng):  # noqa: ARG002
                import random as _r

                run_a_draws.append([_r.random() for _ in range(5)])

        sim_a = FakeSim()
        spec_a = _RunACapture()
        register_benchmark("per-ep-reseed-a", spec_a)
        sim_a.evaluate_benchmark(
            benchmark_name="per-ep-reseed-a",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=3,
            seed=42,
        )
        assert len(run_a_draws) == 3

        # Re-run with same master seed.
        run_b_draws: list[list[float]] = []

        class _RunBCapture(_CountingBenchmark):
            def on_episode_start(self, sim, episode_rng):  # noqa: ARG002
                import random as _r

                run_b_draws.append([_r.random() for _ in range(5)])

        sim_b = FakeSim()
        spec_b = _RunBCapture()
        register_benchmark("per-ep-reseed-b", spec_b)
        sim_b.evaluate_benchmark(
            benchmark_name="per-ep-reseed-b",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=3,
            seed=42,
        )
        assert len(run_b_draws) == 3

        # Per-episode reproducibility: ep N in run A == ep N in run B.
        for ep_idx in range(3):
            assert run_b_draws[ep_idx] == run_a_draws[ep_idx], (
                f"episode {ep_idx} drew different random values on re-run; "
                f"per-episode reseed regressed. Run A: {run_a_draws[ep_idx]}, "
                f"Run B: {run_b_draws[ep_idx]}"
            )

        # Cross-episode distinctness: episodes 0 and 1 should NOT have
        # identical draws (they're seeded with different episode_seeds
        # derived from the master seed via ``master_rng.randint``).
        assert run_a_draws[0] != run_a_draws[1], (
            "consecutive episodes drew identical random values — "
            "per-episode seeding may have used a constant instead of "
            "the per-episode-derived seed"
        )


class TestPolicyResetIntegration:
    """#187: ``_evaluate_with_spec`` calls ``policy.reset(seed=episode_seed)``
    at the top of every episode so SERVICE-mode policies (e.g. Gr00tPolicy
    over ZMQ) can forward the seed to a remote inference server.

    Without this hook the server's diffusion sampler RNG drifts across
    calls and breaks reproducibility. The ``Policy.reset`` default is a
    no-op so existing in-process policies (LocalLeRobot, MockPolicy) are
    unaffected; concrete policies override to apply per-episode state
    reset (RNG seeding, action-cache flush, server-side reset endpoint
    call, etc.).
    """

    def test_reset_called_once_per_episode_with_episode_seed(self):
        """``policy.reset(seed=N)`` is invoked exactly once per episode,
        with ``N == episode_seed`` (the deterministic per-episode seed
        derived from the master seed via ``master_rng.randint``)."""

        # Build a MockPolicy with a recording reset spy. We attach the
        # MagicMock to the bound instance method so the existing
        # MockPolicy.get_actions path keeps working.
        policy = MockPolicy()
        policy.set_robot_state_keys(FakeSim().robot_joint_names("fake_robot"))
        reset_calls: list[dict] = []

        def _record_reset(seed: int | None = None) -> None:
            reset_calls.append({"seed": seed})

        policy.reset = _record_reset  # type: ignore[assignment]

        sim = FakeSim()
        spec = _CountingBenchmark()
        PolicyRunner(sim).evaluate("fake_robot", policy, spec=spec, n_episodes=3, seed=42)

        # Three episodes → three reset calls, one per episode.
        assert len(reset_calls) == 3, f"expected 3 reset calls, got {len(reset_calls)}: {reset_calls}"

        # Each reset receives an int seed (the per-episode seed). The
        # actual seed value is deterministic given the master seed but
        # opaque (derived via random.Random(42).randint(0, 2**31-1));
        # we only assert it's a sane int and the three are distinct.
        for c in reset_calls:
            assert isinstance(c["seed"], int), f"reset seed must be int, got {type(c['seed']).__name__}: {c}"
            assert 0 <= c["seed"] < 2**31, f"reset seed out of range: {c['seed']}"
        assert len({c["seed"] for c in reset_calls}) == 3, (
            f"per-episode seeds should be distinct, got dupes: {[c['seed'] for c in reset_calls]}"
        )

    def test_reset_seed_reproducible_across_runs(self):
        """Same master seed → same per-episode reset seeds across re-runs.
        Pin so the seed-forwarding contract is bit-stable: two runs of the
        same eval will hit the server with identical seed sequences, which
        is what makes reproducibility possible end-to-end."""
        seeds_a: list[int] = []
        seeds_b: list[int] = []

        def _capture_a(seed: int | None = None) -> None:
            seeds_a.append(int(seed) if seed is not None else -1)

        def _capture_b(seed: int | None = None) -> None:
            seeds_b.append(int(seed) if seed is not None else -1)

        sim_a = FakeSim()
        policy_a = MockPolicy()
        policy_a.set_robot_state_keys(sim_a.robot_joint_names("fake_robot"))
        policy_a.reset = _capture_a  # type: ignore[assignment]
        PolicyRunner(sim_a).evaluate("fake_robot", policy_a, spec=_CountingBenchmark(), n_episodes=3, seed=42)

        sim_b = FakeSim()
        policy_b = MockPolicy()
        policy_b.set_robot_state_keys(sim_b.robot_joint_names("fake_robot"))
        policy_b.reset = _capture_b  # type: ignore[assignment]
        PolicyRunner(sim_b).evaluate("fake_robot", policy_b, spec=_CountingBenchmark(), n_episodes=3, seed=42)

        assert seeds_a == seeds_b, (
            f"per-episode reset seeds must be reproducible across runs; got {seeds_a} vs {seeds_b}"
        )

    def test_reset_failure_is_swallowed(self, caplog):
        """If ``policy.reset`` raises (e.g. server timeout, client lost
        connection), the eval continues. The exception is logged as a
        WARNING but the rollout proceeds — eval correctness is preserved
        even if per-episode reseed fails."""
        import logging as _logging

        def _raising_reset(seed: int | None = None) -> None:
            raise RuntimeError("server unreachable")

        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        policy.reset = _raising_reset  # type: ignore[assignment]

        with caplog.at_level(_logging.WARNING, logger="strands_robots.simulation.policy_runner"):
            result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=_CountingBenchmark(), n_episodes=2, seed=42)

        # Eval completed despite reset failures.
        assert result["status"] == "success", f"eval should succeed even when reset fails; got {result}"
        # We logged the failure (one per episode = two warnings).
        warnings = [r for r in caplog.records if "policy.reset" in r.getMessage() and "raised" in r.getMessage()]
        assert len(warnings) == 2, f"expected 2 reset warnings, got {len(warnings)}"


class TestSpecInstructionFallback:
    """#187: ``policy_runner._evaluate_with_spec`` falls back to
    ``spec.instruction`` (default ``""``) when the user-supplied
    ``instruction`` argument to ``evaluate_benchmark`` is empty.

    Without this, language-conditioned policies (GR00T, OpenVLA) receive
    an empty string as the task description and produce off-task actions.
    GPU bisection on libero-10/SCENE5 confirmed this was the dominant
    cause of the success_rate gap (0.40 ZMQ vs 1.00 in-process — same
    model, same wire format, same engine, but `policy_obs[\"annotation.\"]`
    was `[\"\"]` on the failing path because robots-sim passes
    `instruction=\"\"` to evaluate_benchmark).
    """

    def test_user_instruction_wins_over_spec(self):
        """When the caller passes ``instruction="explicit task"`` and the
        spec also has a non-empty ``instruction`` property, the user value
        wins. Pin so we don't silently override caller intent."""
        captured: list[str] = []

        class _LangPolicy(MockPolicy):
            async def get_actions(self, observation_dict, instruction, **kwargs):
                captured.append(instruction)
                return [{}]

        class _SpecWithInstruction(_CountingBenchmark):
            @property
            def instruction(self) -> str:
                return "spec said do X"

        sim = FakeSim()
        policy = _LangPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        PolicyRunner(sim).evaluate(
            "fake_robot",
            policy,
            spec=_SpecWithInstruction(),
            n_episodes=1,
            seed=42,
            instruction="user said do Y",
        )
        # Every call sees the user-supplied instruction.
        assert captured, "expected at least one get_actions call"
        assert all(c == "user said do Y" for c in captured), f"got {captured!r}"

    def test_falls_back_to_spec_when_user_instruction_empty(self):
        """Caller passes nothing (or ""), spec has non-empty instruction →
        spec wins. This is the #187 fix."""
        captured: list[str] = []

        class _LangPolicy(MockPolicy):
            async def get_actions(self, observation_dict, instruction, **kwargs):
                captured.append(instruction)
                return [{}]

        class _SpecWithInstruction(_CountingBenchmark):
            @property
            def instruction(self) -> str:
                return "put the white mug on the left plate"

        sim = FakeSim()
        policy = _LangPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        # No instruction= argument — falls back to spec.instruction.
        PolicyRunner(sim).evaluate("fake_robot", policy, spec=_SpecWithInstruction(), n_episodes=1, seed=42)
        assert captured
        assert all(c == "put the white mug on the left plate" for c in captured), f"got {captured!r}"

    def test_warns_when_both_empty(self, caplog):
        """If neither caller nor spec provides an instruction, log a
        WARNING so the user knows the policy is running blind. Pin so a
        future refactor doesn't silently drop the warning."""
        import logging as _logging

        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        with caplog.at_level(_logging.WARNING, logger="strands_robots.simulation.policy_runner"):
            PolicyRunner(sim).evaluate("fake_robot", policy, spec=_CountingBenchmark(), n_episodes=1, seed=42)

        warnings = [r for r in caplog.records if "instruction is empty" in r.getMessage()]
        assert warnings, "expected a warning about empty instruction"


class TestOnFrameHookForSpec:
    """#191: ``policy_runner._evaluate_with_spec`` invokes ``on_frame``
    after each ``sim.send_action`` on the eval thread, before the spec's
    ``on_step`` reward bookkeeping. Plumbed through from
    ``Simulation.evaluate_benchmark`` and ``PolicyRunner.evaluate``.

    Used by ``Simulation.start_cameras_recording_synchronous`` to capture
    frames on the eval thread instead of a daemon thread, eliminating the
    cross-thread ``mjData`` race that produced 2-3% frame-capture rates
    under Strands ``Agent`` tool dispatch.
    """

    def test_on_frame_called_per_applied_step(self):
        """``on_frame`` fires once per applied control step.

        With ``max_steps=20`` and ``success_after=10**9`` (never succeeds),
        the rollout runs to truncation: 20 steps per episode × 1 episode
        = 20 invocations.
        """
        captured: list[tuple[int, dict, dict]] = []

        def on_frame(step, obs, action):
            captured.append(
                (
                    step,
                    dict(obs) if isinstance(obs, dict) else obs,
                    dict(action) if isinstance(action, dict) else action,
                )
            )

        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        spec = _CountingBenchmark()

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=spec, n_episodes=1, seed=42, on_frame=on_frame)

        assert result["status"] == "success", result
        assert len(captured) == 20, f"expected 20 on_frame calls (max_steps=20), got {len(captured)}"
        # Step indices are monotonic global counter starting at 0.
        steps = [c[0] for c in captured]
        assert steps == list(range(20)), f"expected step counter 0..19, got {steps}"

    def test_on_frame_global_counter_crosses_episode_boundaries(self):
        """The counter is *global*, not per-episode. With 3 episodes ×
        max_steps=20 = 60 calls, indices run 0..59.

        Pinned so callers know they can't assume the counter resets per
        episode — they must track ep boundaries via the per-episode
        results dict if needed.
        """
        captured_steps: list[int] = []

        def on_frame(step, obs, action):
            captured_steps.append(step)

        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        spec = _CountingBenchmark()

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=spec, n_episodes=3, seed=42, on_frame=on_frame)

        assert result["status"] == "success", result
        assert len(captured_steps) == 60, f"expected 60 calls (3 ep × 20 steps), got {len(captured_steps)}"
        assert captured_steps == list(range(60)), "global counter should be monotonic across episodes"

    def test_on_frame_failures_logged_not_fatal(self, caplog):
        """A raising ``on_frame`` is logged WARNING but the rollout
        continues. The hook is opt-in telemetry — a broken recorder
        shouldn't crash a 5-episode eval.

        Pinned so a future refactor doesn't accidentally promote hook
        failures to fatal errors.
        """
        import logging as _logging

        def bad_on_frame(step, obs, action):
            raise RuntimeError("camera offline")

        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        spec = _CountingBenchmark()

        with caplog.at_level(_logging.WARNING, logger="strands_robots.simulation.policy_runner"):
            result = PolicyRunner(sim).evaluate(
                "fake_robot", policy, spec=spec, n_episodes=1, seed=42, on_frame=bad_on_frame
            )

        assert result["status"] == "success", f"hook failure must not fail the rollout: {result}"
        warnings = [r for r in caplog.records if "on_frame hook failed" in r.getMessage()]
        # 20 steps → 20 hook invocations → 20 warnings.
        assert len(warnings) == 20, f"expected 20 warnings, got {len(warnings)}"

    def test_on_frame_default_none_is_noop(self):
        """The default ``on_frame=None`` doesn't change behaviour — the
        eval loop runs identically without the hook. Pinned so the
        plumbing addition is a strict superset.
        """
        sim = FakeSim()
        policy = MockPolicy()
        policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
        spec = _CountingBenchmark()

        result = PolicyRunner(sim).evaluate("fake_robot", policy, spec=spec, n_episodes=1, seed=42)
        assert result["status"] == "success"
        assert spec.on_step_calls == 20, "spec hook count unchanged when on_frame is absent"

    def test_evaluate_benchmark_forwards_on_frame(self):
        """``Simulation.evaluate_benchmark`` plumbs ``on_frame`` to
        ``PolicyRunner.evaluate``. End-to-end pin so the full chain
        (Simulation → PolicyRunner.evaluate → _evaluate_with_spec → loop)
        works without callers having to drop down to ``PolicyRunner``
        directly.
        """
        from strands_robots.simulation.benchmark import register_benchmark

        captured: list[int] = []

        def on_frame(step, obs, action):
            captured.append(step)

        sim = FakeSim()
        spec = _CountingBenchmark()
        register_benchmark("on-frame-plumbing-eb", spec)
        result = sim.evaluate_benchmark(
            benchmark_name="on-frame-plumbing-eb",
            robot_name="fake_robot",
            policy_provider="mock",
            n_episodes=1,
            seed=42,
            on_frame=on_frame,
        )
        assert result["status"] == "success", result
        assert len(captured) == 20, f"expected 20 calls via evaluate_benchmark plumbing, got {len(captured)}"
