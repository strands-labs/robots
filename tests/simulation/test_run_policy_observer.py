"""Contract pins for the ``observer`` lane on :meth:`PolicyRunner.run`.

``on_frame`` is not a general observation seam and cannot be made one. It is
*owned* by the backend for the duration of a rollout: MuJoCo's hook raises
:class:`CooperativeStop` for ``stop_policy``, appends the trajectory mirror,
publishes mesh telemetry and drives the LeRobot dataset recorder, and Isaac's
and Newton's do the recording half of that. A caller that supplies its own
``on_frame`` to watch a rollout therefore *replaces* all of it, and
``SimEngine.run_policy`` does not even accept one - it calls
``_make_run_policy_hook`` and passes the backend's.

So the observer is a SECOND lane beside that hook rather than a reinterpretation
of it, and the whole point of these tests is the word "beside": every assertion
below is about the legacy hook continuing to behave exactly as it did, while the
new lane reports what it could not.

Three properties are load-bearing and none of them is obvious:

* **Every physically applied action is reported, including the last one.** The
  legacy hook runs *after* ``send_action``, so an action that a cancelling or
  recording-failing hook aborts on has already advanced the world - and is
  excluded from ``steps_used``, from the video cadence and from the resolution
  denominator, because ``step_count += 1`` sits after the hook. An observability
  lane that inherited that boundary would be silent about the one step a user
  debugging a cancellation most needs to see. The emission is therefore in a
  ``finally``, which is also what keeps the legacy exception byte-for-byte
  unchanged: same type, same traceback, same ``__cause__``.

* **The observation is BORROWED, and is pre-action.** It is the same object the
  legacy hook received - not a copy - and under open-loop chunk replay it is the
  chunk-start observation, so it is stale for every action after the first unless
  a recording forced a refresh. ``observation_is_chunk_reused`` states which of
  those it is, per step, rather than leaving a consumer to assume freshness it
  does not have.

* **An observer failure is not a rollout failure, and is not an ``on_frame``
  failure either.** It must not reach the consecutive-failure watchdog that
  exists to catch a silently-empty dataset (GH #117), because a visualiser that
  cannot draw is not a recorder that cannot write.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import numpy as np
import pytest

from strands_robots.dataset_recorder import RecordingFrameError
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.observers import (
    SCHEMA_VERSION,
    RunPolicyEnded,
    RunPolicyStarted,
    RunPolicyStep,
)
from strands_robots.simulation.policy_runner import CooperativeStop, PolicyRunner


class _FakeSim(SimEngine):
    """Pure-Python engine: no physics, records every public call it receives."""

    def __init__(self, joint_names: tuple[str, ...] = ("j0", "j1", "j2")):
        self._joint_names = list(joint_names)
        self.calls: list[tuple] = []
        self._step_count = 0
        self._sim_time = 0.0
        self._robots = {"fake_robot": self._joint_names}
        self._world: Any = None

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        self._step_count = 0
        self._sim_time = 0.0
        return {"status": "success"}

    def step(self, n_steps: int = 1):
        self._step_count += n_steps
        self._sim_time += 0.002 * n_steps
        return {"status": "success"}

    def get_state(self):
        return {"sim_time": self._sim_time, "step_count": self._step_count}

    def add_robot(self, name, **kw):
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def list_robots(self) -> list[str]:
        return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return list(self._robots.get(robot_name, []))

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        self.calls.append(("get_observation", robot_name))
        return {n: 0.0 for n in self._joint_names}

    def send_action(self, action, robot_name=None, n_substeps=1):
        self.calls.append(("send_action", robot_name))
        self._step_count += 1
        self._sim_time += 0.002

    def render(self, camera_name="default", width=None, height=None):
        return {"image": np.zeros((height or 48, width or 64, 3), dtype=np.uint8)}


class _PartialResolutionSim(_FakeSim):
    """``send_action`` that resolves some keys and rejects the rest."""

    def send_action(self, action, robot_name=None, n_substeps=1):
        self.calls.append(("send_action", robot_name))
        self._step_count += 1
        self._sim_time += 0.002
        keys = list(action) if isinstance(action, dict) else []
        return {
            "status": "error",
            "content": [
                {"text": "partial"},
                {"json": {"applied": keys[:1], "unresolved_keys": keys[1:]}},
            ],
        }


class _CoarseErrorSim(_FakeSim):
    """Backend refusal with no complete per-key evidence."""

    def send_action(self, action, robot_name=None, n_substeps=1):
        self.calls.append(("send_action", robot_name))
        return {"status": "error", "content": [{"text": "atomic refusal"}]}


class _CoarseThenFullSim(_FakeSim):
    """One unknown refusal followed by a fully applied action."""

    def send_action(self, action, robot_name=None, n_substeps=1):
        self.calls.append(("send_action", robot_name))
        attempts = sum(name == "send_action" for name, _ in self.calls)
        if attempts == 1:
            return {"status": "error", "content": [{"text": "atomic refusal"}]}
        self._step_count += 1
        self._sim_time += 0.002
        return None


class _StateOnlyClockSim(_FakeSim):
    """Backend whose terminal clock is available only through ``get_state``."""

    def __init__(self):
        super().__init__()
        del self._sim_time
        self.state_calls = 0

    def send_action(self, action, robot_name=None, n_substeps=1):
        self.calls.append(("send_action", robot_name))
        self._step_count += 1

    def get_state(self):
        self.state_calls += 1
        return {"content": [{"json": {"sim_time": 1.25}}]}


class _RecordingSim(_FakeSim):
    def _is_recording(self) -> bool:
        return True


def _runner_and_policy(sim: _FakeSim | None = None) -> tuple[PolicyRunner, MockPolicy, _FakeSim]:
    sim = sim if sim is not None else _FakeSim()
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("fake_robot"))
    return PolicyRunner(sim), policy, sim


def _run(runner: PolicyRunner, policy: MockPolicy, **kw: Any) -> dict[str, Any]:
    params: dict[str, Any] = {
        "duration": 0.3,
        "control_frequency": 10.0,
        "fast_mode": True,
    }
    params.update(kw)
    return runner.run("fake_robot", policy, **params)


class TestSignature:
    """``observer`` is keyword-only, optional, and defaults to the old behaviour."""

    def test_observer_is_keyword_only_and_optional(self):
        sig = inspect.signature(PolicyRunner.run)
        assert "observer" in sig.parameters, (
            "PolicyRunner.run must accept an ``observer``. Without it the only per-step seam is "
            "``on_frame``, which the backend owns for cancellation and recording."
        )
        param = sig.parameters["observer"]
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is None, "the default must be the historical no-observer rollout"

    def test_omitting_the_observer_leaves_the_payload_shape_untouched(self):
        """``observer_failures`` appears only for a rollout that had an observer.

        A key that materialises on every rollout would change the documented
        payload for callers that never asked for the lane.
        """
        runner, policy, _ = _runner_and_policy()
        payload = _json(_run(runner, policy))
        assert "observer_failures" not in payload

        runner2, policy2, _ = _runner_and_policy()
        payload2 = _json(_run(runner2, policy2, observer=lambda _e: None))
        assert payload2["observer_failures"] == 0


def _json(result: dict[str, Any]) -> dict[str, Any]:
    """Pull the structured block out of a tool-envelope result.

    Scans ``content`` rather than indexing it: an early-refusal envelope carries
    only a text block, so ``content[1]`` is not a safe assumption.
    """
    for block in result.get("content", []):
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in result: {result}")


class TestEventSequence:
    """Exactly one ``Started``, one ``Step`` per applied action, one ``Ended``."""

    def test_lifecycle_is_started_then_steps_then_ended(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()

        result = _run(runner, policy, observer=events.append)

        assert result["status"] == "success"
        assert isinstance(events[0], RunPolicyStarted), "the first event must open the rollout"
        assert isinstance(events[-1], RunPolicyEnded), "the last event must close it"
        assert sum(isinstance(e, RunPolicyStarted) for e in events) == 1
        assert sum(isinstance(e, RunPolicyEnded) for e in events) == 1
        assert SCHEMA_VERSION == 2
        assert {e.schema_version for e in events} == {2}

        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        assert len(steps) == _json(result)["steps_used"] == 3

    def test_event_seq_is_dense_and_monotonic(self):
        """A consumer must be able to detect a gap, so the sequence is dense."""
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        _run(runner, policy, observer=events.append)

        seqs = [e.event_seq for e in events]
        assert seqs == list(range(len(events)))
        assert len({e.run_id for e in events}) == 1, "one rollout is one run_id"

    def test_monotonic_timestamps_never_go_backwards(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        _run(runner, policy, observer=events.append)

        stamps = [e.monotonic_ns for e in events]
        assert stamps == sorted(stamps)

    def test_started_describes_the_rollout_it_opens(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        _run(runner, policy, instruction="pick the cube", observer=events.append)

        started = events[0]
        assert started.robot_name == "fake_robot"
        assert started.policy == "MockPolicy"
        assert started.instruction == "pick the cube"
        assert started.control_frequency == 10.0
        assert started.total_steps == 3
        assert started.async_rtc is False

    def test_ended_reports_the_terminal_facts(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        result = _run(runner, policy, observer=events.append)

        ended = events[-1]
        payload = _json(result)
        assert ended.outcome == "success"
        assert ended.stopped_reason == "budget" == payload["stopped_reason"]
        assert ended.legacy_steps_used == payload["steps_used"]
        assert ended.applied_actions == 3
        assert ended.error_type is None
        assert ended.observer_failures == 0


class TestLegacyHookIsUnchanged:
    """The whole contract: adding an observer changes nothing about ``on_frame``."""

    def test_on_frame_receives_identical_arguments_with_and_without_observer(self):
        seen_without: list[tuple] = []
        runner, policy, _ = _runner_and_policy()
        _run(runner, policy, on_frame=lambda s, o, a: seen_without.append((s, dict(o), dict(a))))

        seen_with: list[tuple] = []
        runner2, policy2, _ = _runner_and_policy()
        _run(
            runner2,
            policy2,
            on_frame=lambda s, o, a: seen_with.append((s, dict(o), dict(a))),
            observer=lambda _e: None,
        )

        assert seen_without == seen_with, (
            "the observer lane must not change the index, the observation or the action the "
            "legacy hook is handed - backends decide cancellation and dataset content from these."
        )

    def test_the_observer_sees_the_same_borrowed_objects_as_on_frame(self):
        """Borrowed, not copied: identity is the contract a consumer must respect."""
        hook_payloads: list[tuple[int, Any, Any]] = []
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()

        _run(
            runner,
            policy,
            on_frame=lambda s, o, a: hook_payloads.append((s, o, a)),
            observer=events.append,
        )

        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        assert len(steps) == len(hook_payloads)
        for step, (idx, obs, action) in zip(steps, hook_payloads, strict=True):
            assert step.legacy_step_index == idx
            assert step.observation is obs, "the observation is borrowed, not copied"
            assert step.action is action, "the action is borrowed, not copied"

    def test_observer_adds_no_observation_or_render_call(self):
        """The lane reports what the rollout already did; it must not sample more."""
        runner, policy, sim = _runner_and_policy()
        _run(runner, policy)
        without = list(sim.calls)

        runner2, policy2, sim2 = _runner_and_policy()
        _run(runner2, policy2, observer=lambda _e: None)

        assert sim2.calls == without

    def test_observer_failure_does_not_arm_the_on_frame_watchdog(self):
        """An observer that fails every step is not a recorder losing frames.

        The consecutive-failure abort exists so a broken recording hook cannot
        produce a silently empty dataset (GH #117). Routing observer failures
        into it would abort healthy rollouts because a visualiser is down.
        """
        hook_calls: list[int] = []
        runner, policy, _ = _runner_and_policy()

        def exploding_observer(_event: Any) -> None:
            raise RuntimeError("observer is down")

        result = _run(
            runner,
            policy,
            duration=1.0,  # 10 steps: well past the 5-failure legacy threshold
            on_frame=lambda s, o, a: hook_calls.append(s),
            observer=exploding_observer,
        )

        assert result["status"] == "success", "a failing observer must not fail the rollout"
        assert len(hook_calls) == 10, "the legacy hook must still run every step"
        payload = _json(result)
        assert payload["steps_used"] == 10
        assert payload["observer_failures"] >= 10, "every failure must be counted, not swallowed"


class TestObserverFailureIsolation:
    """A failing observer is invisible to the rollout's outcome."""

    def test_result_is_identical_whether_the_observer_throws(self):
        runner, policy, _ = _runner_and_policy()
        healthy = _json(_run(runner, policy, observer=lambda _e: None))

        runner2, policy2, _ = _runner_and_policy()

        def boom(_event: Any) -> None:
            raise ValueError("nope")

        broken = _json(_run(runner2, policy2, observer=boom))

        for key in ("steps_used", "stopped_reason", "action_errors", "partial_action_failure_rate"):
            assert healthy[key] == broken[key], f"{key} diverged when the observer threw"
        # Started + one Step per applied action + Ended = 5 dispatches for a
        # 3-step rollout, and an observer that raises unconditionally fails all
        # of them. The count is of DISPATCHES, not of steps.
        assert broken["observer_failures"] == 5
        assert healthy["observer_failures"] == 0

    def test_an_observer_failure_on_started_still_runs_the_rollout(self):
        """Failing on the first event must not prevent the rollout or the rest."""
        seen: list[str] = []
        runner, policy, _ = _runner_and_policy()

        def fail_started(event: Any) -> None:
            seen.append(type(event).__name__)
            if isinstance(event, RunPolicyStarted):
                raise RuntimeError("cannot open")

        result = _run(runner, policy, observer=fail_started)

        assert result["status"] == "success"
        assert seen[0] == "RunPolicyStarted"
        assert "RunPolicyStep" in seen
        assert seen[-1] == "RunPolicyEnded", "a failed Started must not suppress Ended"

    def test_a_base_exception_from_the_observer_is_also_contained(self):
        """``CooperativeStop`` is a ``BaseException``; an observer must not fire it.

        The graceful-stop signal belongs to ``on_frame``. If the observer lane
        let a ``BaseException`` through it would hand every visualiser the
        ability to cancel a rollout.
        """
        runner, policy, _ = _runner_and_policy()

        def rogue(_event: Any) -> None:
            raise CooperativeStop("observer tried to cancel")

        result = _run(runner, policy, observer=rogue)

        assert result["status"] == "success"
        payload = _json(result)
        assert payload["stopped_reason"] == "budget", "the observer must not be able to cancel"
        assert payload["steps_used"] == 3


class TestAppliedActionsIncludesTheAbortingStep:
    """The step a cancelling hook aborts on physically happened. Report it."""

    def test_cooperative_stop_still_reports_the_applied_action(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()

        def hook(step: int, _obs: Any, _action: Any) -> None:
            if step >= 2:
                raise CooperativeStop("user stopped")

        result = _run(runner, policy, duration=10.0, on_frame=hook, observer=events.append)

        assert result["status"] == "success"
        payload = _json(result)
        assert payload["stopped_reason"] == "cancelled"

        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        # Steps 0 and 1 completed; step 2 was applied and then cancelled.
        assert len(steps) == 3
        assert payload["steps_used"] == 2, "legacy accounting excludes the cancelling step"
        assert steps[-1].applied_action_index == 2
        assert steps[-1].legacy_step_index == steps[-1].applied_action_index
        assert steps[-1].legacy_hook_outcome == "cancelled"

        ended = events[-1]
        assert ended.stopped_reason == "cancelled"
        assert ended.applied_actions == 3
        assert ended.legacy_steps_used == 2
        assert ended.applied_actions > ended.legacy_steps_used

    def test_recording_frame_loss_still_reports_the_applied_action(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()

        def hook(step: int, _obs: Any, _action: Any) -> None:
            if step >= 1:
                raise RecordingFrameError("dataset write failed")

        result = _run(runner, policy, duration=10.0, on_frame=hook, observer=events.append)

        assert result["status"] == "error", "lost dataset frames stay fatal on the first occurrence"
        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        assert steps[-1].legacy_hook_outcome == "recording_error"
        assert steps[-1].applied_action_index == 1
        assert steps[-1].legacy_step_index == steps[-1].applied_action_index

        ended = events[-1]
        assert isinstance(ended, RunPolicyEnded)
        assert ended.outcome == "error"
        assert ended.stopped_reason == "error"
        assert ended.error_type == "RecordingFrameError"

    def test_legacy_hook_outcome_is_ok_when_nothing_raised(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        _run(runner, policy, on_frame=lambda s, o, a: None, observer=events.append)

        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        assert {s.legacy_hook_outcome for s in steps} == {"ok"}

    def test_legacy_hook_outcome_is_absent_when_no_hook_was_given(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        _run(runner, policy, observer=events.append)

        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        assert {s.legacy_hook_outcome for s in steps} == {"absent"}


class TestActionResolution:
    """Resolution is normalised, so a consumer never parses a backend blob."""

    def test_full_resolution_on_the_success_path(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        _run(runner, policy, observer=events.append)

        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        assert {s.action_resolution for s in steps} == {"full"}
        for step in steps:
            assert step.unresolved_action_keys == ()
            assert set(step.applied_action_keys) == {"j0", "j1", "j2"}

    def test_partial_resolution_is_reported_as_partial(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy(_PartialResolutionSim())
        result = _run(runner, policy, observer=events.append)

        assert result["status"] == "success", "a partial resolution runs to completion"
        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        assert {s.action_resolution for s in steps} == {"partial"}
        for step in steps:
            assert len(step.applied_action_keys) == 1
            assert len(step.unresolved_action_keys) == 2

    def test_coarse_error_is_unknown_without_fabricated_keys(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy(_CoarseErrorSim())

        result = _run(runner, policy, duration=0.1, observer=events.append)

        assert result["status"] == "error"
        step = next(e for e in events if isinstance(e, RunPolicyStep))
        assert step.action_resolution == "unknown"
        assert step.applied_action_keys == ()
        assert step.unresolved_action_keys == ()
        assert "physical application is unknown" in result["content"][0]["text"]

    def test_coarse_step_is_excluded_from_aggregate_resolution_rates(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy(_CoarseThenFullSim())

        result = _run(runner, policy, duration=0.2, observer=events.append)

        assert result["status"] == "success"
        payload = _json(result)
        assert payload["action_errors"] == 1
        assert payload["action_resolution_rate"] == {"j0": 1.0, "j1": 1.0, "j2": 1.0}
        assert payload["partial_action_failure_rate"] == 0.0
        assert "coarse backend errors" in result["content"][0]["text"]
        steps = [event for event in events if isinstance(event, RunPolicyStep)]
        assert [step.action_resolution for step in steps] == ["unknown", "full"]

    def test_elapsed_is_measured_and_sim_time_is_carried_when_available(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        _run(runner, policy, observer=events.append)

        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        elapsed = [s.elapsed_s for s in steps]
        assert elapsed == sorted(elapsed)
        assert all(e >= 0.0 for e in elapsed)
        # ``_FakeSim._sim_time`` is a cheap cached engine clock.
        sim_times = [s.sim_time_s for s in steps]
        assert all(t is not None for t in sim_times)
        assert sim_times == sorted(sim_times)

    @pytest.mark.parametrize("with_observer", [False, True])
    def test_terminal_result_retains_one_time_get_state_clock_fallback(self, with_observer: bool):
        events: list[Any] = []
        runner, policy, sim = _runner_and_policy(_StateOnlyClockSim())

        result = _run(runner, policy, observer=events.append if with_observer else None)

        assert result["status"] == "success"
        assert _json(result)["sim_time_s"] == 1.25
        assert sim.state_calls == 1, "Step events must not add get_state calls"
        if with_observer:
            steps = [event for event in events if isinstance(event, RunPolicyStep)]
            assert steps
            assert {step.sim_time_s for step in steps} == {None}


class TestObservationFreshness:
    """Chunk reuse is stated per step rather than assumed away."""

    def test_chunk_reuse_is_flagged_truthfully(self):
        """Without a recording, one observation feeds a whole chunk."""
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        _run(runner, policy, duration=0.4, action_horizon=4, observer=events.append)

        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        assert steps, "expected applied actions"
        assert steps[0].observation_is_chunk_reused is False, "the chunk-start action is fresh"
        # MockPolicy emits a chunk, so at least one later action reuses it.
        assert any(s.observation_is_chunk_reused for s in steps[1:]), (
            "an action replayed from a chunk must not claim a fresh observation"
        )
        assert [s.observation_age_steps for s in steps] == [0, 1, 2, 3]

    def test_async_cross_boundary_ages_are_exact(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()

        _run(
            runner,
            policy,
            duration=0.8,
            action_horizon=4,
            async_rtc=True,
            observer=events.append,
        )

        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        assert [s.observation_age_steps for s in steps] == [0, 1, 2, 3, 2, 3, 4, 5]
        assert [s.observation_is_chunk_reused for s in steps] == [
            False,
            True,
            True,
            True,
            False,
            True,
            True,
            True,
        ]

    @pytest.mark.parametrize("async_rtc", [False, True])
    def test_recording_refresh_reports_zero_age_throughout(self, async_rtc: bool):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy(_RecordingSim())

        _run(
            runner,
            policy,
            duration=0.8,
            action_horizon=4,
            async_rtc=async_rtc,
            observer=events.append,
        )

        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        assert len(steps) == 8
        assert [s.observation_age_steps for s in steps] == [0] * 8
        assert not any(s.observation_is_chunk_reused for s in steps)


class TestSyncAsyncParity:
    """Both acquisition strategies report the same number of applied actions."""

    @pytest.mark.parametrize("async_rtc", [False, True])
    def test_step_event_count_matches_applied_actions(self, async_rtc: bool):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()

        result = _run(runner, policy, duration=0.6, async_rtc=async_rtc, observer=events.append)

        assert result["status"] == "success"
        steps = [e for e in events if isinstance(e, RunPolicyStep)]
        ended = events[-1]
        assert len(steps) == ended.applied_actions == 6
        assert events[0].async_rtc is async_rtc
        indices = [s.applied_action_index for s in steps]
        assert indices == list(range(len(steps))), "applied indices are dense and 0-based"


class TestErrorPaths:
    """``Ended`` is attempted on every path that emitted ``Started``."""

    def test_a_failing_policy_still_closes_the_lifecycle(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()

        def explode(*_a: Any, **_k: Any):
            raise RuntimeError("inference exploded")

        policy.get_actions = explode  # type: ignore[method-assign]

        result = _run(runner, policy, observer=events.append)

        assert result["status"] == "error"
        assert isinstance(events[0], RunPolicyStarted)
        ended = events[-1]
        assert isinstance(ended, RunPolicyEnded)
        assert ended.outcome == "error"
        assert ended.stopped_reason == "error"
        assert ended.error_type == "RuntimeError"
        assert "inference exploded" in ended.error_message

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(KeyboardInterrupt("stop"), id="keyboard-interrupt"),
            pytest.param(SystemExit("stop"), id="system-exit"),
            pytest.param(GeneratorExit("stop"), id="generator-exit"),
            pytest.param(asyncio.CancelledError("stop"), id="asyncio-cancelled-error"),
        ],
    )
    def test_non_cooperative_base_exception_propagates_and_closes_once(self, error: BaseException):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()

        def explode(*_a: Any, **_k: Any):
            raise error

        policy.get_actions = explode  # type: ignore[method-assign]

        with pytest.raises(type(error)) as caught:
            _run(runner, policy, observer=events.append)

        assert caught.value is error
        ended = [event for event in events if isinstance(event, RunPolicyEnded)]
        assert len(ended) == 1
        assert ended[0].outcome == "error"
        assert ended[0].error_type == type(error).__name__

    def test_terminal_observer_base_exception_cannot_mask_original(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        original = KeyboardInterrupt("policy interrupted")
        terminal = SystemExit("observer exit during Ended")

        def explode(*_a: Any, **_k: Any):
            raise original

        def observer(event: Any) -> None:
            events.append(event)
            if isinstance(event, RunPolicyEnded):
                raise terminal

        policy.get_actions = explode  # type: ignore[method-assign]

        with pytest.raises(KeyboardInterrupt) as caught:
            _run(runner, policy, observer=observer)

        assert caught.value is original
        assert sum(isinstance(event, RunPolicyEnded) for event in events) == 1
        assert any("SystemExit" in note for note in getattr(original, "__notes__", ()))
        traceback_names: list[str] = []
        current = caught.value.__traceback__
        while current is not None:
            traceback_names.append(current.tb_frame.f_code.co_name)
            current = current.tb_next
        assert "explode" in traceback_names

    def test_step_observer_base_exception_cannot_mask_legacy_hook_exception(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        original = KeyboardInterrupt("legacy hook interrupted")
        secondary = SystemExit("observer exit during Step")

        def on_frame(*_a: Any, **_k: Any) -> None:
            raise original

        def observer(event: Any) -> None:
            events.append(event)
            if isinstance(event, RunPolicyStep):
                raise secondary

        with pytest.raises(KeyboardInterrupt) as caught:
            _run(runner, policy, on_frame=on_frame, observer=observer)

        assert caught.value is original
        assert sum(isinstance(event, RunPolicyStep) for event in events) == 1
        assert sum(isinstance(event, RunPolicyEnded) for event in events) == 1
        assert any("SystemExit" in note for note in getattr(original, "__notes__", ()))
        traceback_names: list[str] = []
        current = caught.value.__traceback__
        while current is not None:
            traceback_names.append(current.tb_frame.f_code.co_name)
            current = current.tb_next
        assert "on_frame" in traceback_names

    def test_outer_handled_exception_cannot_suppress_step_observer_escape(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        secondary = SystemExit("observer exit during Step")
        handled = ValueError("already handled by caller")

        def observer(event: Any) -> None:
            events.append(event)
            if isinstance(event, RunPolicyStep):
                raise secondary

        try:
            raise handled
        except ValueError:
            with pytest.raises(SystemExit) as caught:
                _run(runner, policy, observer=observer)

        assert caught.value is secondary
        assert not getattr(handled, "__notes__", ())
        assert sum(isinstance(event, RunPolicyStep) for event in events) == 1
        assert sum(isinstance(event, RunPolicyEnded) for event in events) == 1

    def test_post_loop_result_assembly_error_closes_once(self, monkeypatch: pytest.MonkeyPatch):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()
        error = RuntimeError("result assembly exploded")

        def explode() -> float:
            raise error

        # Another simulation test deliberately evicts and re-imports the module,
        # so patch the globals of the exact function object this collected class
        # will execute rather than whichever module object is currently cached.
        run_impl = PolicyRunner.run.__wrapped__
        monkeypatch.setitem(run_impl.__globals__, "process_rss_mb", explode)

        with pytest.raises(RuntimeError) as caught:
            _run(runner, policy, observer=events.append)

        assert caught.value is error
        ended = [event for event in events if isinstance(event, RunPolicyEnded)]
        assert len(ended) == 1
        assert ended[0].outcome == "error"
        assert ended[0].error_message == "result assembly exploded"

    def test_non_callable_observer_is_refused_before_direct_runner_side_effects(self):
        runner, policy, sim = _runner_and_policy()

        with pytest.raises(ValueError, match=r"PolicyRunner\.run: observer must be callable or None"):
            _run(runner, policy, observer=42)  # type: ignore[arg-type]

        assert sim.calls == []

    def test_non_callable_observer_is_refused_by_facade_before_side_effects(self):
        runner, policy, sim = _runner_and_policy()
        del runner

        result = sim.run_policy(
            robot_name="fake_robot",
            policy_object=policy,
            n_steps=1,
            control_frequency=10.0,
            fast_mode=True,
            observer=42,  # type: ignore[arg-type]
        )

        assert result == {
            "status": "error",
            "content": [{"text": "run_policy: observer must be callable or None, got 42."}],
        }
        assert sim.calls == []

    def test_a_preflight_refusal_emits_no_events_at_all(self):
        """No ``Started`` means no ``Ended``: the rollout never began."""
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()

        with pytest.raises(ValueError):
            _run(runner, policy, action_horizon=0, observer=events.append)

        assert events == [], "a refused request must not open a rollout lifecycle"

    def test_stop_when_predicate_is_reported_as_the_reason(self):
        events: list[Any] = []
        runner, policy, _ = _runner_and_policy()

        result = _run(
            runner,
            policy,
            duration=10.0,
            stop_when=lambda _sim: True,
            observer=events.append,
        )

        assert result["status"] == "success"
        ended = events[-1]
        assert ended.stopped_reason == "predicate"
        assert ended.applied_actions == 1
