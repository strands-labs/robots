"""A dataset episode the recorder could not flush must stop a recorded evaluation.

``tests/simulation/test_recording_frame_loss_is_not_tolerated.py`` pins the rule
one level down: a lost recording *frame* is fatal, because continuing past it
leaves a short episode whose surviving frames are re-timestamped from the
declared ``fps`` while the rollout still reports success. Its
``test_the_eval_loop_does_not_swallow_it_either`` pins that the eval loop
specifically must surface a lost frame rather than absorb it as telemetry.

The *episode* flush had the opposite rule, and it sits on the worse failure. A
failed ``save_episode`` closes the recorder - the LeRobot episode buffer is in
an undefined state after a partial write - and ``add_frame`` then returns on a
closed recorder without writing a frame, without raising
``RecordingFrameError``, and without counting a ``dropped_frame_count``. So one
failed flush discards every remaining episode of the evaluation in total
silence, and the eval reported a ``success_rate`` over all of them under
``status="success"``. A lost frame truncates an episode; a lost episode
truncated the whole run and took the recorder's own accounting with it.

These tests pin the episode-level rule: the evaluation stops at the episode
whose flush failed and reports the reason in ``recording_save_error``, matching
what every sibling flush already does - ``stop_recording`` and
``SimEngine.save_episode`` drop the poisoned recorder and return an error,
``run_policy(n_episodes=...)`` aborts its remaining episodes, and the MuJoCo
backend's ``reset`` surfaces the failure rather than resetting into an
undefined state. :class:`TestEveryFlushVerdictIsRead` derives that population
from the source, so a new flush cannot go back to discarding its verdict.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import random
from typing import Any

import pytest

pytest.importorskip("mujoco")

from strands_robots.dataset_recorder import DatasetRecorder
from strands_robots.policies.mock import MockPolicy
from strands_robots.simulation.benchmark import BenchmarkProtocol, StepInfo
from strands_robots.simulation.mujoco.backend import _can_render
from strands_robots.simulation.mujoco.simulation import Simulation
from strands_robots.simulation.policy_runner import PolicyRunner

_FEATURES: dict[str, Any] = {
    "observation.state": {"dtype": "float32", "names": ["1", "2", "3", "4", "5", "6"]},
    "action": {"dtype": "float32", "names": ["1", "2", "3", "4", "5", "6"]},
}


class _Dataset:
    """Stand-in ``LeRobotDataset`` whose flush fails from a chosen episode on.

    Stands in for the dataset, not for the recorder: the real
    :class:`~strands_robots.dataset_recorder.DatasetRecorder` runs unmodified,
    so the poisoning this suite is about is the production one.
    """

    def __init__(self, *, fail_from_episode: int | None = 0) -> None:
        self.repo_id = "local/episode-loss"
        self.root = "/tmp/local-episode-loss"
        self.features = _FEATURES
        self.fail_from_episode = fail_from_episode
        self.written = 0
        self.saved = 0
        self.save_attempts = 0

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.written += 1

    def save_episode(self) -> None:
        attempt = self.save_attempts
        self.save_attempts += 1
        if self.fail_from_episode is not None and attempt >= self.fail_from_episode:
            # Deliberately does NOT say "episode": the episode the reason names
            # has to come from the loop, not from the underlying message.
            raise RuntimeError(f"simulated parquet write failure (flush attempt {attempt})")
        self.saved += 1


class _NoopSpec(BenchmarkProtocol):
    """Minimal always-running spec: fixed horizon, never succeeds or fails."""

    max_steps = 3

    @property
    def supported_robots(self) -> list[str]:
        return ["so100"]

    @property
    def default_robot(self) -> str:
        return "arm"

    def on_episode_start(self, sim: Any, rng: random.Random) -> None:
        return None

    def on_step(self, sim: Any, obs: dict[str, Any], action: dict[str, Any]) -> StepInfo:
        return StepInfo(reward=0.0)

    def is_success(self, sim: Any) -> bool:
        return False

    def is_failure(self, sim: Any) -> bool:
        return False


def _recording_sim(recorder: Any) -> Simulation:
    """A one-robot sim with ``recorder`` attached as the live session's writer."""
    sim = Simulation(tool_name="episode_loss", mesh=False)
    sim.create_world()
    sim.add_robot(name="arm", data_config="so100")
    assert sim._world is not None
    sim._world._backend_state["recording"] = True
    sim._world._backend_state["trajectory"] = []
    sim._world._backend_state["dataset_recorder"] = recorder
    return sim


requires_gl = pytest.mark.skipif(not _can_render(), reason="No OpenGL context (EGL/OSMesa) for offscreen rendering")


def _policy(sim: Simulation) -> MockPolicy:
    policy = MockPolicy()
    policy.set_robot_state_keys(sim.robot_joint_names("arm"))
    return policy


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    """The json block, scanned for rather than indexed (the documented read)."""
    return next(block["json"] for block in result["content"] if "json" in block)


def _feed(recorder: Any) -> Any:
    """An ``on_frame`` hook that records - the only way an eval feeds a recorder."""

    def hook(_step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
        recorder.add_frame(observation, action, camera_keys=[])

    return hook


class TestALostRecordingEpisodeStopsTheEvaluation:
    """The regression: the run stops at the lost episode and says so."""

    def test_evaluate_stops_at_the_episode_whose_flush_failed(self) -> None:
        """Pre-fix all 3 episodes ran, 2 of them into a closed recorder."""
        ds = _Dataset(fail_from_episode=0)
        recorder = DatasetRecorder(dataset=ds, task="t")
        sim = _recording_sim(recorder)
        try:
            result = sim.eval_policy(
                robot_name="arm",
                policy_object=_policy(sim),
                n_episodes=3,
                max_steps=2,
                control_frequency=50.0,
                on_frame=_feed(recorder),
            )
        finally:
            sim.cleanup()

        payload = _payload(result)
        assert result["status"] == "error", result
        assert payload["episodes_completed"] == 1, payload
        assert payload["n_episodes"] == 3
        # Only episode 0's frames were ever offered to the dataset: the two
        # later episodes did not run into the closed recorder at all.
        assert ds.written == 2
        assert ds.save_attempts == 1

    def test_the_report_names_the_episode_and_the_underlying_cause(self) -> None:
        """A reason the caller can act on, not a log line they never see."""
        ds = _Dataset(fail_from_episode=0)
        recorder = DatasetRecorder(dataset=ds, task="t")
        sim = _recording_sim(recorder)
        try:
            result = sim.eval_policy(
                robot_name="arm",
                policy_object=_policy(sim),
                n_episodes=2,
                max_steps=2,
                control_frequency=50.0,
                on_frame=_feed(recorder),
            )
        finally:
            sim.cleanup()

        payload = _payload(result)
        reason = payload["recording_save_error"]
        assert reason is not None
        assert reason.startswith("episode 0: "), reason
        assert "simulated parquet write failure" in reason
        assert "recorder closed" in reason
        # The same fact reaches a reader of the text block.
        assert "lost recording episode" in result["content"][0]["text"]

    def test_a_later_episode_is_the_one_that_stops_it(self) -> None:
        """The first episodes are kept; the aggregate covers exactly those."""
        ds = _Dataset(fail_from_episode=2)
        recorder = DatasetRecorder(dataset=ds, task="t")
        sim = _recording_sim(recorder)
        try:
            result = sim.eval_policy(
                robot_name="arm",
                policy_object=_policy(sim),
                n_episodes=5,
                max_steps=2,
                control_frequency=50.0,
                on_frame=_feed(recorder),
            )
        finally:
            sim.cleanup()

        payload = _payload(result)
        assert result["status"] == "error"
        assert payload["episodes_completed"] == 3
        assert len(payload["episodes"]) == 3
        assert payload["recording_save_error"].startswith("episode 2: ")
        assert ds.saved == 2

    @requires_gl
    def test_the_lost_episode_keeps_the_video_that_shows_what_it_did(self, tmp_path) -> None:
        """Stopping must not also discard the rollout footage.

        The dataset episode is gone; the MP4 is the only remaining record of
        what the policy did on it, so the video is closed and collected before
        the loop stops rather than after.
        """
        ds = _Dataset(fail_from_episode=1)
        recorder = DatasetRecorder(dataset=ds, task="t")
        sim = _recording_sim(recorder)
        try:
            result = sim.eval_policy(
                robot_name="arm",
                policy_object=_policy(sim),
                n_episodes=4,
                max_steps=2,
                control_frequency=50.0,
                on_frame=_feed(recorder),
                video={"path": str(tmp_path / "rollout.mp4"), "fps": 25, "width": 64, "height": 64},
            )
        finally:
            sim.cleanup()

        payload = _payload(result)
        assert result["status"] == "error"
        assert payload["episodes_completed"] == 2
        # One MP4 per episode that ran, including the one whose flush failed.
        assert len(payload["video_paths"]) == 2, payload["video_paths"]
        assert all(pathlib.Path(v).is_file() for v in payload["video_paths"])

    def test_the_benchmark_path_stops_and_reports_the_same_way(self) -> None:
        """``_evaluate_with_spec`` shares the rule, not just the schema."""
        ds = _Dataset(fail_from_episode=0)
        recorder = DatasetRecorder(dataset=ds, task="t")
        sim = _recording_sim(recorder)
        try:
            result = PolicyRunner(sim).evaluate(
                robot_name="arm",
                policy=_policy(sim),
                spec=_NoopSpec(),
                n_episodes=3,
                control_frequency=50.0,
                on_frame=_feed(recorder),
            )
        finally:
            sim.cleanup()

        payload = _payload(result)
        assert result["status"] == "error", result
        assert payload["episodes_completed"] == 1
        assert payload["recording_save_error"].startswith("episode 0: ")
        assert ds.save_attempts == 1


class TestAHealthyEvaluationIsNotRefused:
    """No evaluation reports a lost episode where there is not one.

    These fail pre-fix only because the key did not exist. Their point is the
    other direction: they are what a fix cannot pass by simply reporting a
    reason unconditionally, and they pin the key as PRESENT and ``None`` on a
    healthy run rather than absent - an absent key read with ``.get()`` is not
    the same fact as a key that says nothing went wrong.
    """

    def test_a_flushed_episode_reports_the_key_as_none(self) -> None:
        """Present and ``None``, so an absent key is never read as "fine"."""
        ds = _Dataset(fail_from_episode=None)
        recorder = DatasetRecorder(dataset=ds, task="t")
        sim = _recording_sim(recorder)
        try:
            result = sim.eval_policy(
                robot_name="arm",
                policy_object=_policy(sim),
                n_episodes=3,
                max_steps=2,
                control_frequency=50.0,
                on_frame=_feed(recorder),
            )
        finally:
            sim.cleanup()

        payload = _payload(result)
        assert result["status"] == "success", result
        assert "recording_save_error" in payload
        assert payload["recording_save_error"] is None
        assert payload["episodes_completed"] == 3
        assert ds.saved == 3

    def test_an_evaluation_with_no_recorder_runs_every_episode(self) -> None:
        """The common case: no recording session, nothing to flush."""
        sim = Simulation(tool_name="episode_loss_norec", mesh=False)
        sim.create_world()
        sim.add_robot(name="arm", data_config="so100")
        try:
            result = sim.eval_policy(
                robot_name="arm",
                policy_object=_policy(sim),
                n_episodes=3,
                max_steps=2,
                control_frequency=50.0,
            )
        finally:
            sim.cleanup()

        payload = _payload(result)
        assert result["status"] == "success", result
        assert payload["recording_save_error"] is None
        assert payload["episodes_completed"] == 3

    def test_an_empty_buffer_is_not_a_lost_episode(self) -> None:
        """A recorder nothing fed has no episode to lose, so no flush is tried."""
        ds = _Dataset(fail_from_episode=0)
        recorder = DatasetRecorder(dataset=ds, task="t")
        sim = _recording_sim(recorder)
        try:
            result = sim.eval_policy(
                robot_name="arm",
                policy_object=_policy(sim),
                n_episodes=3,
                max_steps=2,
                control_frequency=50.0,
            )
        finally:
            sim.cleanup()

        payload = _payload(result)
        assert result["status"] == "success", result
        assert payload["recording_save_error"] is None
        assert payload["episodes_completed"] == 3
        assert ds.save_attempts == 0


class TestTheClosedRecorderIsWhyItStops:
    """The premise, measured on the real recorder rather than asserted.

    Both cells hold before and after the fix: they describe why continuing is
    not a milder outcome than stopping, which is the reason the evaluation
    stops rather than averaging over the episodes that follow.
    """

    def test_a_failed_flush_closes_the_recorder_and_reports_it(self) -> None:
        ds = _Dataset(fail_from_episode=0)
        recorder = DatasetRecorder(dataset=ds, task="t")
        recorder.add_frame({"1": 0.0}, {"1": 0.0}, camera_keys=[])
        assert recorder.episode_frame_count == 1

        verdict = recorder.save_episode()

        assert verdict["status"] == "error"
        assert recorder._closed is True

    def test_add_frame_on_a_closed_recorder_writes_nothing_and_counts_no_drop(self) -> None:
        """The silence: no frame, no ``RecordingFrameError``, no counted drop."""
        ds = _Dataset(fail_from_episode=0)
        recorder = DatasetRecorder(dataset=ds, task="t")
        recorder.add_frame({"1": 0.0}, {"1": 0.0}, camera_keys=[])
        recorder.save_episode()
        assert recorder._closed is True

        before = (recorder.frame_count, recorder.dropped_frame_count, ds.written)
        for _ in range(5):
            recorder.add_frame({"1": 0.0}, {"1": 0.0}, camera_keys=[])

        assert (recorder.frame_count, recorder.dropped_frame_count, ds.written) == before
        assert recorder.strict is True  # even fail-fast mode raises nothing here


def _discarded_flush_verdicts(source: str) -> list[int]:
    """Line numbers of ``save_episode()`` calls whose verdict is thrown away."""
    discarded: list[int] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr == "save_episode":
            discarded.append(node.lineno)
    return discarded


def _flush_call_sites(source: str) -> list[int]:
    """Line numbers of every ``save_episode()`` call, discarded or not."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "save_episode"
    ]


def _simulation_modules() -> list[pathlib.Path]:
    root = pathlib.Path(inspect.getfile(Simulation)).parent.parent
    assert root.name == "simulation", root
    return sorted(root.rglob("*.py"))


class TestEveryFlushVerdictIsRead:
    """No flush in ``strands_robots/simulation`` discards its verdict.

    This class passes before and after the fix, and that is the point: the
    verdict was never thrown away at the call, it was bound and then only
    logged - which no source scan can see. It is a guard for the next flush
    rather than a pin of this one, derived from the source so a call site added
    later is held to the same rule without being listed here.

    Scoped to the simulation package: ``transforms`` has its own caller chain
    and its own reporting, and is not what this change is about.
    """

    def test_no_save_episode_call_discards_its_verdict(self) -> None:
        offenders = [
            f"{path}:{line}"
            for path in _simulation_modules()
            for line in _discarded_flush_verdicts(path.read_text(encoding="utf-8"))
        ]
        assert not offenders, (
            f"these flushes throw away a verdict that reports whether the episode reached the dataset: {offenders}"
        )

    def test_the_scan_reaches_the_call_sites_it_grades(self) -> None:
        """Non-vacuity: an empty population would report clean forever."""
        sites = [
            f"{path.name}:{line}"
            for path in _simulation_modules()
            for line in _flush_call_sites(path.read_text(encoding="utf-8"))
        ]
        assert len(sites) >= 4, sites
        assert {path.name for path in _simulation_modules()} >= {
            "recording.py",
            "base.py",
            "policy_runner.py",
        }

    def test_a_discarded_verdict_is_reported(self) -> None:
        """Non-vacuity: the scan rejects the shape this change replaced."""
        assert _discarded_flush_verdicts("recorder.save_episode()\n") == [1]
        assert _discarded_flush_verdicts("verdict = recorder.save_episode()\n") == []
        assert _discarded_flush_verdicts("if recorder.save_episode()['status']: pass\n") == []
