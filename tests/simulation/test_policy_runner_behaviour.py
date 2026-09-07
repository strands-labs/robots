"""Behavioural tests for PolicyRunner - run/replay/evaluate with a mock policy."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("mujoco")

from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner, VideoConfig, _resolve_coroutine


@pytest.fixture
def sim_with_robot():
    s = Simulation(tool_name="pr_test", mesh=False)
    s.create_world()
    s.add_robot(name="alice", data_config="so100")
    yield s
    s.cleanup()


class TestPolicyRunnerRun:
    def test_run_returns_success(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)
        result = runner.run(
            "alice",
            policy,
            duration=0.1,
            control_frequency=50,
            fast_mode=True,
        )
        assert result["status"] == "success"
        text = result["content"][0]["text"]
        assert "alice" in text

    def test_run_invokes_on_frame_hook(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        calls: list[int] = []

        def on_frame(step: int, obs: dict, action: dict) -> None:
            calls.append(step)

        runner = PolicyRunner(sim_with_robot)
        runner.run(
            "alice",
            policy,
            duration=0.04,
            control_frequency=50,
            fast_mode=True,
            on_frame=on_frame,
        )
        assert calls, "on_frame should fire at least once"
        # Step indices must be non-decreasing.
        assert calls == sorted(calls)


class TestOnFrameFailureCounter:
    """GH #117: on_frame exceptions must abort the episode after N consecutive
    failures so a broken recording hook can't silently corrupt a dataset."""

    def test_single_onframe_failure_is_tolerated(self, sim_with_robot):
        """One failure then success must NOT abort the episode."""
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        calls = {"count": 0}

        def flaky(step: int, obs: dict, action: dict) -> None:
            calls["count"] += 1
            if calls["count"] == 2:
                raise ValueError("transient")

        runner = PolicyRunner(sim_with_robot)
        result = runner.run(
            "alice",
            policy,
            duration=0.2,
            control_frequency=50,
            fast_mode=True,
            on_frame=flaky,
            max_onframe_failures=3,
        )
        # Single failure in a sea of successes: episode completes.
        assert result["status"] == "success", result

    def test_consecutive_onframe_failures_abort_episode(self, sim_with_robot):
        """N consecutive on_frame failures must make run() return an error,
        preventing the silent-empty-dataset footgun described in GH #117."""
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        call_count = {"n": 0}

        def always_fails(step: int, obs: dict, action: dict) -> None:
            call_count["n"] += 1
            raise ValueError(f"boom-{step}")

        runner = PolicyRunner(sim_with_robot)
        result = runner.run(
            "alice",
            policy,
            duration=5.0,  # plenty of time - early-abort is the point
            control_frequency=50,
            fast_mode=True,
            on_frame=always_fails,
            max_onframe_failures=3,
        )
        assert result["status"] == "error", result
        text = result["content"][0]["text"]
        assert "3 times in a row" in text
        # Hook was called exactly the threshold number of times, not more.
        # (Third raise aborts.)
        assert call_count["n"] == 3

    def test_consecutive_counter_resets_on_success(self, sim_with_robot):
        """Two failures then a success then two more failures must NOT abort
        at threshold=3 - the counter resets on a successful call."""
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        calls = {"n": 0}

        def mixed(step: int, obs: dict, action: dict) -> None:
            calls["n"] += 1
            # Fail on calls 1,2, succeed on 3, fail on 4,5, succeed on 6+
            if calls["n"] in (1, 2, 4, 5):
                raise RuntimeError(f"bad-{calls['n']}")

        runner = PolicyRunner(sim_with_robot)
        result = runner.run(
            "alice",
            policy,
            duration=0.3,
            control_frequency=50,
            fast_mode=True,
            on_frame=mixed,
            max_onframe_failures=3,
        )
        assert result["status"] == "success", result

    def test_default_threshold_is_5(self, sim_with_robot):
        """Without explicit max_onframe_failures, default kicks in at 5."""
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        calls = {"n": 0}

        def always_fails(step: int, obs: dict, action: dict) -> None:
            calls["n"] += 1
            raise ValueError(f"boom-{calls['n']}")

        runner = PolicyRunner(sim_with_robot)
        result = runner.run(
            "alice",
            policy,
            duration=5.0,
            control_frequency=50,
            fast_mode=True,
            on_frame=always_fails,
            # max_onframe_failures omitted - default is 5
        )
        assert result["status"] == "error"
        assert "5 times in a row" in result["content"][0]["text"]
        assert calls["n"] == 5


class TestPolicyRunnerEvaluate:
    def test_evaluate_default_success_fn(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)

        result = runner.evaluate(
            "alice",
            policy,
            n_episodes=2,
            max_steps=5,
            success_fn=None,
        )
        assert result["status"] == "success"
        payload = result["content"][-1]["json"]
        assert payload["n_episodes"] == 2
        assert payload["max_steps"] == 5
        assert 0 <= payload["success_rate"] <= 1
        assert len(payload["episodes"]) == 2

    def test_evaluate_unknown_success_fn_errors(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)
        result = runner.evaluate(
            "alice",
            policy,
            n_episodes=1,
            max_steps=2,
            success_fn="__nope__",
        )
        assert result["status"] == "error"

    def test_evaluate_degenerate_policy_advances_and_terminates(self, sim_with_robot):
        """An empty-chunk policy must not hang: each query advances exactly one
        physics step, the episode ends at max_steps, and with no success
        predicate the run reports zero successes instead of spinning forever."""
        policy = _EmptyActionPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)

        result = runner.evaluate(
            "alice",
            policy,
            n_episodes=1,
            max_steps=4,
            success_fn=None,
        )
        assert result["status"] == "success"
        payload = result["content"][-1]["json"]
        assert payload["success_rate"] == 0.0
        assert payload["episodes"][0]["steps"] == 4
        assert payload["episodes"][0]["success"] is False

    def test_evaluate_degenerate_policy_succeeds_on_post_step_obs(self, sim_with_robot):
        """Even with an empty action chunk, a success predicate that holds on
        the post-step observation marks the episode solved on the first query
        and stops early (steps == 1)."""
        policy = _EmptyActionPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)

        seen: list[dict] = []

        def always_success(obs: dict) -> bool:
            seen.append(obs)
            return True

        result = runner.evaluate(
            "alice",
            policy,
            n_episodes=1,
            max_steps=5,
            success_fn=always_success,
        )
        assert result["status"] == "success"
        payload = result["content"][-1]["json"]
        assert payload["success_rate"] == 1.0
        assert payload["episodes"][0]["steps"] == 1
        assert payload["episodes"][0]["success"] is True
        assert seen, "success_fn must be evaluated on the post-step observation"


# require_default_robot / _maybe_sim_time


class TestHelpers:
    def test_maybe_sim_time_reads_cached_world_clock(self, sim_with_robot):
        runner = PolicyRunner(sim_with_robot)
        t = runner._maybe_sim_time()
        # Empty sim at t=0 should return 0.0.
        assert t == pytest.approx(0.0, abs=1e-9)

    def test_cached_sim_time_never_calls_get_state(self):
        fake = MagicMock()
        fake._world = None
        fake._sim_time = None
        fake.get_state.side_effect = AssertionError("must not be called")

        assert PolicyRunner(fake)._cached_sim_time() is None
        fake.get_state.assert_not_called()

    def test_cached_sim_time_reads_isaac_like_engine_cache(self):
        fake = MagicMock()
        fake._world = None
        fake._sim_time = 1.25
        fake.get_state.side_effect = AssertionError("must not be called")

        assert PolicyRunner(fake)._cached_sim_time() == 1.25
        fake.get_state.assert_not_called()

    @pytest.mark.parametrize("value", [True, False, float("nan"), float("inf"), "1.0"])
    def test_cached_sim_time_rejects_non_finite_or_non_real_cache(self, value):
        fake = MagicMock()
        fake._world = None
        fake._sim_time = value
        fake.get_state.side_effect = AssertionError("must not be called")

        assert PolicyRunner(fake)._cached_sim_time() is None
        fake.get_state.assert_not_called()

    def test_cached_sim_time_without_cached_clock_returns_none(self):
        fake = object()
        runner = PolicyRunner(fake)  # type: ignore[arg-type]
        assert runner._cached_sim_time() is None

    def test_maybe_sim_time_on_broken_state_fallback_returns_none(self):
        fake = MagicMock()
        fake._world = None
        fake._sim_time = None
        fake.get_state.side_effect = RuntimeError("boom")

        assert PolicyRunner(fake)._maybe_sim_time() is None

    def test_require_default_robot_empty_raises(self):
        fake = MagicMock()
        fake.list_robots.return_value = []
        runner = PolicyRunner(fake)
        with pytest.raises(ValueError, match="No robots"):
            runner._require_default_robot()

    def test_require_default_robot_returns_first(self):
        fake = MagicMock()
        fake.list_robots.return_value = ["alpha", "beta"]
        runner = PolicyRunner(fake)
        assert runner._require_default_robot() == "alpha"


# replay() error paths (no lerobot -> clean error)


class TestReplayErrorPaths:
    def test_replay_missing_lerobot_clean_error(self, sim_with_robot, monkeypatch):
        """When lerobot isn't importable, replay returns a friendly error
        instead of propagating ImportError to the caller."""

        def _boom(*a, **kw):
            raise ImportError("no lerobot")

        # Patch the lazy import inside replay().
        import builtins

        real_import = builtins.__import__

        def _patched_import(name, *args, **kwargs):
            if name.startswith("strands_robots.dataset_recorder"):
                raise ImportError("no lerobot (test-forced)")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _patched_import)

        runner = PolicyRunner(sim_with_robot)
        result = runner.replay(
            repo_id="fake/ds",
            robot_name="alice",
            episode=0,
        )
        assert result["status"] == "error"
        assert "lerobot" in result["content"][0]["text"].lower()


class TestResolveCoroutine:
    def test_passthrough_for_plain_list(self):
        assert _resolve_coroutine([{"j": 0.1}]) == [{"j": 0.1}]

    def test_awaits_coroutine(self):
        async def inner():
            return [{"j": 0.2}]

        assert _resolve_coroutine(inner()) == [{"j": 0.2}]


class TestVideoConfig:
    def test_enabled_with_path(self):
        v = VideoConfig(path="/tmp/x.mp4", fps=30)
        assert v.enabled is True

    def test_disabled_without_path(self):
        v = VideoConfig()
        assert v.enabled is False


class TestT26PerfBudget:
    """T26: mock-policy rollouts must meet the <2s/500-step budget.

    The optimisation: policies that don't consume images expose
    ``requires_images=False`` and PolicyRunner propagates that to
    ``SimEngine.get_observation(skip_images=True)`` so the per-step
    camera render is skipped.
    """

    def test_mock_policy_500_steps_under_budget(self, sim_with_robot):
        import time

        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        # Warmup so renderer / JIT are hot.
        PolicyRunner(sim_with_robot).run("alice", policy, duration=0.02, control_frequency=50.0, fast_mode=True)

        t0 = time.time()
        result = PolicyRunner(sim_with_robot).run(
            "alice",
            policy,
            duration=1.0,
            control_frequency=500.0,  # → 500 steps
            fast_mode=True,
        )
        wall = time.time() - t0

        assert result["status"] == "success"
        # The T26 budget is < 2s. Local measurements land ~0.02s with
        # skip_images, ~0.38s without. We pin to 2.0 so CI runners with
        # slower renderers don't flake while still catching regressions.
        assert wall < 2.0, f"mock-policy 500 steps took {wall:.2f}s (T26 budget: <2.0s)"

    def test_requires_images_propagates_to_observation(self, sim_with_robot, monkeypatch):
        """PolicyRunner reads policy.requires_images once and passes
        skip_images= to every get_observation call."""
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        captured: list[bool] = []
        original = sim_with_robot.get_observation

        def spy(**kwargs):
            captured.append(bool(kwargs.get("skip_images", False)))
            return original(**kwargs)

        monkeypatch.setattr(sim_with_robot, "get_observation", spy)
        PolicyRunner(sim_with_robot).run(
            "alice",
            policy,
            duration=0.05,
            control_frequency=50.0,  # → a few steps
            fast_mode=True,
        )
        assert captured, "get_observation was never called"
        # Mock policy has requires_images=False → every call skip_images=True.
        assert all(captured), f"skip_images should be True every step; got {captured}"


class _ConstantTargetPolicy(MockPolicy):
    """Commands every joint to a fixed target every step (no sinusoid).

    A position-servo arm only tracks a constant target if
    physics is stepped for the full control period per action. With the old
    n_substeps=1 default, the eval paths integrated a single ~2 ms mj_step and
    the arm crawled.
    """

    def __init__(self, target: float = 0.6) -> None:
        super().__init__()
        self._target = target

    async def get_actions(self, observation_dict, instruction, **kwargs):
        if not self.robot_state_keys:
            self.robot_state_keys = list(observation_dict.keys())
        return [{k: self._target for k in self.robot_state_keys}]


class _EmptyActionPolicy(MockPolicy):
    """Degenerate policy that always returns an empty action chunk.

    Exercises the no-hang guard in :meth:`PolicyRunner.evaluate`: an empty
    chunk must still advance exactly one physics step per query (so episodes
    terminate at ``max_steps`` instead of spinning forever), and the success
    predicate is evaluated against the post-step observation.
    """

    @property
    def provider_name(self) -> str:
        return "empty"

    async def get_actions(self, observation_dict, instruction, **kwargs):
        return []


class TestControlSubsteps:
    """Eval paths must step physics for the full control period per action,
    identical to run()."""

    def test_control_substeps_derivation(self, sim_with_robot):
        runner = PolicyRunner(sim_with_robot)
        dt = sim_with_robot.physics_timestep()
        assert dt and dt > 0
        # 50 Hz control over a 2 ms physics dt -> 10 substeps per action.
        assert runner._control_substeps(50.0) == round((1.0 / 50.0) / dt)
        # Explicit override wins. A non-positive override is NOT floored to 1:
        # clamping it silently reinstated the single-physics-step
        # under-integration this helper exists to prevent, so it raises (the
        # public entry points reject it with a structured error first).
        assert runner._control_substeps(50.0, override=7) == 7
        with pytest.raises(ValueError, match="control_substeps"):
            runner._control_substeps(50.0, override=0)

    def test_evaluate_steps_full_control_period(self, sim_with_robot):
        """A constant-target policy must actually move the arm in evaluate().

        Pre-fix (n_substeps=1) the arm integrated ~10% of the way to target;
        this asserts a meaningful joint delta so the bug can't silently return.
        """
        joints = sim_with_robot.robot_joint_names("alice")
        policy = _ConstantTargetPolicy(target=0.6)
        policy.set_robot_state_keys(joints)
        runner = PolicyRunner(sim_with_robot)

        sim_with_robot.reset()
        q0 = sim_with_robot.get_observation(robot_name="alice", skip_images=True)
        runner.evaluate(
            "alice",
            policy,
            n_episodes=1,
            max_steps=80,
            control_frequency=50.0,
        )
        q1 = sim_with_robot.get_observation(robot_name="alice", skip_images=True)
        max_dq = max(abs(q1[k] - q0[k]) for k in joints)
        assert max_dq > 0.1, f"arm barely moved in evaluate (max|dq|={max_dq:.4f})"

    def test_evaluate_forwards_substeps_to_send_action(self, sim_with_robot, monkeypatch):
        """evaluate() must pass n_substeps>1 (not the default 1) to send_action."""
        seen: list[int] = []
        orig = sim_with_robot.send_action

        def spy(action, robot_name=None, n_substeps=1):
            seen.append(n_substeps)
            return orig(action, robot_name=robot_name, n_substeps=n_substeps)

        monkeypatch.setattr(sim_with_robot, "send_action", spy)
        policy = _ConstantTargetPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        PolicyRunner(sim_with_robot).evaluate("alice", policy, n_episodes=1, max_steps=5, control_frequency=50.0)
        assert seen, "send_action was never called"
        assert all(n > 1 for n in seen), f"eval still under-stepping: n_substeps={seen}"


class TestVideoFinalization:
    """PolicyRunner.run() video-writer finalization: the success summary when
    frames are captured vs. the loud 0-frame warning when they are not.

    Both branches live in the post-loop ``writer is not None`` block and depend
    only on whether ``_extract_frame_ndarray`` decoded at least one frame from
    ``sim.render()``. We stub ``sim.render`` so the up-front camera probe passes
    (status="success") while controlling whether in-loop frames decode - this
    exercises the finalization logic without an OpenGL context, so it runs on
    headless CI where the GL-backed video test is skipped.
    """

    @staticmethod
    def _png_render_result() -> dict:
        """A render() success dict whose image block decodes to a real ndarray."""
        import base64
        import io

        import numpy as np
        from PIL import Image

        arr = np.random.default_rng(0).integers(0, 255, (16, 16, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        encoded = base64.b64encode(buf.getvalue()).decode()
        return {
            "status": "success",
            "content": [
                {"text": "frame"},
                {"image": {"format": "png", "source": {"bytes": encoded}}},
            ],
        }

    def test_zero_frames_captured_warns_and_stays_success(self, sim_with_robot, tmp_path, monkeypatch):
        """Camera passes the up-front probe but every in-loop frame fails to
        decode (e.g. camera removed mid-rollout): run() must NOT crash, must
        flag the empty recording in the result text, and must not leave a
        non-empty MP4 behind."""
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        # render() returns success (probe passes) but carries no image block,
        # so _extract_frame_ndarray() returns None for every in-loop frame.
        monkeypatch.setattr(
            sim_with_robot,
            "render",
            lambda *a, **k: {"status": "success", "content": [{"text": "no image block"}]},
        )

        video_path = tmp_path / "empty.mp4"
        runner = PolicyRunner(sim_with_robot)
        result = runner.run(
            "alice",
            policy,
            duration=0.1,
            control_frequency=50,
            fast_mode=True,
            video=VideoConfig(path=str(video_path), fps=30),
        )
        # Rollout still completes - a dead camera doesn't kill the run.
        assert result["status"] == "success", result
        text = result["content"][0]["text"]
        assert "0 frames captured" in text
        assert str(video_path) in text
        # No real video content was produced.
        assert not video_path.exists() or video_path.stat().st_size == 0

    def test_captured_frames_report_video_summary(self, sim_with_robot, tmp_path, monkeypatch):
        """When frames decode, run() finalizes the writer and reports the video
        path, frame count, fps and resolution in the result text - and the MP4
        actually exists on disk with content."""
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        monkeypatch.setattr(sim_with_robot, "render", lambda *a, **k: self._png_render_result())

        video_path = tmp_path / "ok.mp4"
        runner = PolicyRunner(sim_with_robot)
        result = runner.run(
            "alice",
            policy,
            duration=0.2,
            control_frequency=50,
            fast_mode=True,
            video=VideoConfig(path=str(video_path), fps=30, width=16, height=16),
        )
        assert result["status"] == "success", result
        text = result["content"][0]["text"]
        assert "Video:" in text
        assert "frames" in text
        assert "30fps" in text
        assert video_path.exists() and video_path.stat().st_size > 0


def _json_block(result: dict) -> dict:
    """Extract the agent-consumable ``{"json": {...}}`` content block."""
    return next(c["json"] for c in result["content"] if isinstance(c, dict) and "json" in c)


class TestRunPolicyStructuredResult:
    """run_policy / PolicyRunner.run must return an agent-consumable
    ``{"json": {...}}`` content block alongside the human-readable text, so a
    caller can read n_steps / video_path / action_errors as typed fields
    instead of regex-parsing prose. Mirrors eval_policy()'s json block.
    """

    def test_run_emits_structured_json_block(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)
        result = runner.run(
            "alice",
            policy,
            instruction="pick up cube",
            duration=0.1,
            control_frequency=50,
            fast_mode=True,
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["robot_name"] == "alice"
        assert payload["policy"] == "MockPolicy"
        assert payload["instruction"] == "pick up cube"
        # 0.1s @ 50 Hz -> 5 control steps.
        assert payload["n_steps"] == 5
        assert payload["action_errors"] == 0
        assert payload["stopped_early"] is False
        assert payload["elapsed_s"] >= 0.0
        # No video requested -> path is None, zero frames.
        assert payload["video_path"] is None
        assert payload["video_frames"] == 0

    def test_json_reports_written_video_path(self, sim_with_robot, tmp_path, monkeypatch):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        monkeypatch.setattr(
            sim_with_robot,
            "render",
            lambda *a, **k: TestVideoFinalization._png_render_result(),
        )
        video_path = tmp_path / "rollout.mp4"
        runner = PolicyRunner(sim_with_robot)
        result = runner.run(
            "alice",
            policy,
            duration=0.2,
            control_frequency=50,
            fast_mode=True,
            video=VideoConfig(path=str(video_path), fps=30, width=16, height=16),
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["video_path"] == str(video_path)
        assert payload["video_frames"] > 0

    def test_json_video_path_none_when_no_frames(self, sim_with_robot, tmp_path, monkeypatch):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        # Probe passes but in-loop frames never decode -> 0 frames written.
        calls = {"n": 0}

        def render(*a, **k):
            calls["n"] += 1
            if calls["n"] == 1:
                return TestVideoFinalization._png_render_result()
            return {"status": "success", "content": [{"text": "no image"}]}

        monkeypatch.setattr(sim_with_robot, "render", render)
        video_path = tmp_path / "empty.mp4"
        runner = PolicyRunner(sim_with_robot)
        result = runner.run(
            "alice",
            policy,
            duration=0.1,
            control_frequency=50,
            fast_mode=True,
            video=VideoConfig(path=str(video_path), fps=30, width=16, height=16),
        )
        assert result["status"] == "success", result
        payload = _json_block(result)
        # Recording was requested but produced no frames -> path None.
        assert payload["video_path"] is None
        assert payload["video_frames"] == 0


class TestEvaluateOnFrameSuccessFnPath:
    """The ``on_frame`` hook must fire on the legacy ``success_fn`` eval path.

    Regression for the hook being accepted by ``PolicyRunner.evaluate`` (and
    surfaced via ``Simulation.eval_policy``) yet silently dropped on the
    non-benchmark path - the path ``eval_policy`` actually drives by default.
    """

    def test_success_fn_path_fires_on_frame_once_per_applied_step(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)

        steps_seen: list[int] = []

        def on_frame(step: int, obs: dict, action: dict) -> None:
            steps_seen.append(step)

        result = runner.evaluate(
            "alice",
            policy,
            n_episodes=2,
            max_steps=3,
            control_frequency=50,
            on_frame=on_frame,
        )
        assert result["status"] == "success"

        # Hook fired at least once and crossed the episode boundary (proves the
        # global counter is continuous, not reset per episode).
        assert steps_seen, "on_frame never fired on the success_fn eval path"
        assert steps_seen == sorted(steps_seen)
        assert steps_seen == list(range(len(steps_seen)))
        assert len(steps_seen) > 3, "hook did not continue into the second episode"

        # Exactly one hook call per applied control step (MockPolicy always
        # emits a non-empty chunk, so applied steps == sum of per-episode steps).
        json_block = next(b["json"] for b in result["content"] if "json" in b)
        applied = sum(ep["steps"] for ep in json_block["episodes"])
        assert len(steps_seen) == applied

    def test_default_on_frame_none_is_noop(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)
        result = runner.evaluate("alice", policy, n_episodes=1, max_steps=2, control_frequency=50)
        assert result["status"] == "success"

    def test_on_frame_exception_is_logged_not_fatal(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)

        def boom(step: int, obs: dict, action: dict) -> None:
            raise ValueError("hook blew up")

        # A best-effort telemetry hook must never abort the eval.
        result = runner.evaluate("alice", policy, n_episodes=1, max_steps=2, control_frequency=50, on_frame=boom)
        assert result["status"] == "success"

    def test_eval_policy_public_surface_forwards_on_frame(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))

        seen: list[int] = []

        def on_frame(step: int, obs: dict, action: dict) -> None:
            seen.append(step)

        result = sim_with_robot.eval_policy(
            robot_name="alice",
            policy_object=policy,
            n_episodes=1,
            max_steps=2,
            control_frequency=50,
            on_frame=on_frame,
        )
        assert result["status"] == "success"
        assert seen, "eval_policy did not forward on_frame to the success_fn path"
        assert seen == sorted(seen)


class TestEvaluateSuccessMeasured:
    """Guard against a silently-meaningless success_rate.

    With no success criterion (``success_fn=None`` and no benchmark spec) an
    episode can never be marked successful, so ``success_rate`` is a hard
    ``0.0`` regardless of policy behaviour - indistinguishable from a policy
    that genuinely failed every episode. The payload must flag this
    (``success_measured=False``), the text must say so, and a warning must be
    logged, so callers do not mistake the fabricated 0.0 for a measurement.
    """

    def test_no_success_criterion_flags_unmeasured_and_warns(self, sim_with_robot, caplog):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)

        with caplog.at_level("WARNING", logger="strands_robots.simulation.policy_runner"):
            result = runner.evaluate("alice", policy, n_episodes=1, max_steps=3, success_fn=None)

        assert result["status"] == "success"
        payload = result["content"][-1]["json"]
        # success_rate stays numeric (0.0) for backward compatibility ...
        assert payload["success_rate"] == 0.0
        # ... but the payload must flag that nothing was measured.
        assert payload["success_measured"] is False
        # The human-readable text must not present 0.0% as a real result.
        text = result["content"][0]["text"]
        assert "not measured" in text
        # And a warning must be logged so interactive callers are not misled.
        assert any("without a success criterion" in r.message for r in caplog.records), (
            "expected a warning when eval runs with no success criterion"
        )

    def test_success_fn_marks_measured(self, sim_with_robot):
        policy = MockPolicy()
        policy.set_robot_state_keys(sim_with_robot.robot_joint_names("alice"))
        runner = PolicyRunner(sim_with_robot)

        result = runner.evaluate(
            "alice",
            policy,
            n_episodes=1,
            max_steps=3,
            success_fn=lambda obs: True,
        )
        payload = result["content"][-1]["json"]
        assert payload["success_measured"] is True
        assert payload["success_rate"] == 1.0
        assert "not measured" not in result["content"][0]["text"]
