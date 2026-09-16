"""``step`` feeds an open dataset recording: a scripted motion is a demonstration.

Asked to "record a small demonstration dataset - move through 3 poses", an
agent's natural plan is ``start_recording`` -> ``set_joint_positions`` +
``step`` x3 -> ``stop_recording``. Only ``run_policy``'s per-step hook called
``add_frame``, so that plan recorded nothing and learned so at
``stop_recording``. Replayed twice with a note on the ``step`` reply, the agent
read past the note (it acts on ``status``, not on prose inside a success),
interleaved ``run_policy(mock)`` calls to make frames appear, and reported the
mock policy's random motion as the poses it set.

Now ``step`` records while a recording is open: one frame per ``1/fps``
seconds of sim time, observation = every robot's state and cameras exactly as
the rollout hook supplies them, action = the position-servo targets in force
(``data.ctrl``) keyed as :meth:`robot_action_keys` keys a policy's output, task
= the session's task. A ``step`` call covering less sim time than one frame
period records nothing and *says so* in its reply. A running rollout owns the
recorder, so ``step`` stays out of its way. Motion primitives are unchanged
(their own test pins that they do not record).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

mj = pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402


def _text(result) -> str:
    for block in result.get("content", []):
        if "text" in block:
            return block["text"]
    return ""


@pytest.fixture
def sim():
    s = Simulation(tool_name="step_records", mesh=False)
    s.create_world()
    assert s.add_robot("so101")["status"] == "success"
    yield s
    s.cleanup()


def _open_fake_recording(sim, *, fps: int = 10, task: str = "three poses") -> MagicMock:
    """Open a recording session around a fake recorder - the shape start_recording leaves."""
    recorder = MagicMock()
    recorder.episode_frame_count = 0

    def _count(**_kw):
        recorder.episode_frame_count += 1

    recorder.add_frame.side_effect = _count
    state = sim._world._backend_state
    state["recording"] = True
    state["trajectory"] = []
    state["dataset_recorder"] = recorder
    state["recording_fps"] = fps
    state["recording_task"] = task
    state.pop("step_recording_due", None)
    return recorder


def _dt(sim) -> float:
    return float(sim._world._model.opt.timestep)


class TestFramesAreCapturedAtTheDatasetRate:
    def test_the_first_step_records_the_opening_frame(self, sim) -> None:
        recorder = _open_fake_recording(sim, fps=10)
        result = sim.step(n_steps=1)
        assert result["status"] == "success", _text(result)
        assert recorder.add_frame.call_count == 1
        assert "recorded 1 frame" in _text(result)

    def test_one_frame_per_period_over_a_long_step(self, sim) -> None:
        recorder = _open_fake_recording(sim, fps=10)
        steps_per_second = round(1.0 / _dt(sim))
        result = sim.step(n_steps=steps_per_second)  # 1.0 s of sim time
        assert result["status"] == "success", _text(result)
        # The opening frame at t=dt plus one per 0.1 s after it: 11 in a second.
        assert recorder.add_frame.call_count == 11, _text(result)
        assert "recorded 11 frames" in _text(result)

    def test_the_rate_follows_the_recording_fps(self, sim) -> None:
        recorder = _open_fake_recording(sim, fps=30)
        steps_per_second = round(1.0 / _dt(sim))
        sim.step(n_steps=steps_per_second)
        assert recorder.add_frame.call_count == 31

    def test_the_clock_persists_across_step_calls(self, sim) -> None:
        """Ten calls of 0.1 s each record what one call of 1 s records."""
        recorder = _open_fake_recording(sim, fps=10)
        tenth = round(0.1 / _dt(sim))
        for _ in range(10):
            assert sim.step(n_steps=tenth)["status"] == "success"
        assert recorder.add_frame.call_count == 11

    def test_a_short_step_records_nothing_and_says_when_the_next_frame_is_due(self, sim) -> None:
        recorder = _open_fake_recording(sim, fps=10)
        sim.step(n_steps=1)  # opening frame
        result = sim.step(n_steps=5)  # 0.01 s: no frame due
        assert result["status"] == "success"
        assert recorder.add_frame.call_count == 1
        text = _text(result)
        assert "recorded 0 frames" in text
        assert "one frame per 0.1000s of sim time (10 fps)" in text
        assert "next is due at t=" in text

    def test_a_zero_step_call_is_the_same_no_op_as_before(self, sim) -> None:
        recorder = _open_fake_recording(sim)
        result = sim.step(n_steps=0)
        assert "no-op" in _text(result)
        recorder.add_frame.assert_not_called()

    def test_without_a_recording_the_reply_is_unchanged(self, sim) -> None:
        result = sim.step(n_steps=10)
        assert _text(result) == f"+10 steps | t={sim._world.sim_time:.4f}s | total={sim._world.step_count}"

    def test_a_trajectory_only_session_is_not_fed(self, sim) -> None:
        """``recording`` without a dataset recorder is the in-memory trajectory path."""
        state = sim._world._backend_state
        state["recording"] = True
        state["trajectory"] = []
        state.pop("dataset_recorder", None)
        result = sim.step(n_steps=10)
        assert result["status"] == "success"
        assert "recorded" not in _text(result)


class TestTheFrameIsTheSceneAsCommanded:
    def test_action_is_the_servo_target_in_force_and_state_follows_it(self, sim) -> None:
        recorder = _open_fake_recording(sim, fps=10, task="raise the shoulder")
        keys = sim.robot_action_keys("so101")
        target = {keys[1]: 0.6}
        assert sim.set_joint_positions(positions=target, robot_name="so101", hold=True)["status"] == "success"
        sim.step(n_steps=round(0.5 / _dt(sim)))
        assert recorder.add_frame.call_count >= 2
        kwargs = recorder.add_frame.call_args.kwargs
        assert set(kwargs["action"]) == set(keys), kwargs["action"]
        assert kwargs["action"][keys[1]] == pytest.approx(0.6)
        assert kwargs["required_action_keys"] == keys
        assert kwargs["task"] == "raise the shoulder"
        # The servo has had half a second: the recorded state is at the target.
        assert kwargs["observation"][keys[1]] == pytest.approx(0.6, abs=0.05)

    def test_observation_is_what_the_rollout_hook_supplies(self, sim) -> None:
        recorder = _open_fake_recording(sim, fps=10)
        sim.step(n_steps=1)
        obs = recorder.add_frame.call_args.kwargs["observation"]
        expected = sim.get_observation("so101")
        assert set(obs) == set(expected)
        images = [k for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim >= 2]
        assert "default" in images, "the overview camera the schema declares is missing"

    def test_camera_scope_from_start_recording_is_honoured(self, sim) -> None:
        recorder = _open_fake_recording(sim, fps=10)
        sim._world._backend_state["recording_cameras"] = set()  # record no camera
        sim.step(n_steps=1)
        obs = recorder.add_frame.call_args.kwargs["observation"]
        assert not any(isinstance(v, np.ndarray) and v.ndim >= 2 for v in obs.values())

    def test_the_in_memory_trajectory_gets_one_step_per_robot_per_frame(self, sim) -> None:
        _open_fake_recording(sim, fps=10, task="three poses")
        sim.step(n_steps=round(0.3 / _dt(sim)))
        trajectory = sim._world._backend_state["trajectory"]
        assert len(trajectory) == 4
        assert {t.robot_name for t in trajectory} == {"so101"}
        assert all(t.instruction == "three poses" for t in trajectory)
        assert not any(isinstance(v, np.ndarray) for t in trajectory for v in t.observation.values())

    def test_a_multi_robot_frame_is_prefixed_like_the_schema(self, sim) -> None:
        assert sim.add_robot(name="bob", data_config="so101", position=[0.6, 0.0, 0.0])["status"] == "success"
        recorder = _open_fake_recording(sim, fps=10)
        sim.step(n_steps=1)
        kwargs = recorder.add_frame.call_args.kwargs
        names = list(sim._world.robots)
        assert len(names) == 2
        scalar_keys = {k for k, v in kwargs["observation"].items() if not isinstance(v, np.ndarray)}
        assert scalar_keys, "no scalar state recorded"
        assert all(k.split("__", 1)[0] in names for k in scalar_keys), sorted(scalar_keys)[:5]
        assert all(k.split("__", 1)[0] in names for k in kwargs["action"]), sorted(kwargs["action"])
        assert len(kwargs["action"]) == sum(len(sim.robot_action_keys(n)) for n in names)
        assert kwargs["required_action_keys"] == list(kwargs["action"])


class TestTheRolloutOwnsTheRecorder:
    def test_step_does_not_double_record_while_a_policy_runs(self, sim) -> None:
        recorder = _open_fake_recording(sim, fps=10)
        robot = sim._world.robots["so101"]
        robot.policy_running = True
        try:
            result = sim.step(n_steps=round(0.5 / _dt(sim)))
        finally:
            robot.policy_running = False
        assert result["status"] == "success"
        recorder.add_frame.assert_not_called()
        assert "recorded" not in _text(result)


class TestAFailedFrameIsReportedNotCounted:
    def test_a_recorder_error_becomes_the_step_error(self, sim) -> None:
        recorder = _open_fake_recording(sim, fps=10)
        recorder.add_frame.side_effect = ValueError("declared column 'x' absent")
        result = sim.step(n_steps=50)
        assert result["status"] == "error"
        text = _text(result)
        assert "recording frame at t=" in text
        assert "declared column 'x' absent" in text
        assert "advanced 1 of 50 steps" in text

    def test_the_step_count_is_exact_when_a_later_batch_fails(self, sim) -> None:
        """Batches already counted are not counted twice on the error path."""
        recorder = _open_fake_recording(sim, fps=10)
        frames = {"n": 0}

        def _fail_on_third(**_kw):
            frames["n"] += 1
            if frames["n"] == 3:
                raise ValueError("boom")

        recorder.add_frame.side_effect = _fail_on_third
        before = sim._world.step_count
        # The third frame is due at t=0.2 s -> step 100 (dt=0.002), inside a later batch
        # when the batch size is smaller than that.
        sim._STEPS_PER_BATCH = 30
        result = sim.step(n_steps=500)
        assert result["status"] == "error", _text(result)
        advanced = sim._world.step_count - before
        expected = round(0.2 / _dt(sim))
        assert advanced == expected, (advanced, expected, _text(result))
        assert f"advanced {expected} of 500 steps" in _text(result)

    def test_the_clock_is_dropped_with_the_session(self, sim) -> None:
        _open_fake_recording(sim, fps=10)
        sim.step(n_steps=1)
        assert "step_recording_due" in sim._world._backend_state
        sim._world._backend_state["recording"] = False
        sim._world._backend_state["dataset_recorder"] = None
        # A new session (start_recording resets the clock) starts from its first step.
        recorder = _open_fake_recording(sim, fps=10)
        sim.step(n_steps=1)
        assert recorder.add_frame.call_count == 1


class TestTheRealRecorderEndToEnd:
    def test_a_scripted_three_pose_demonstration_is_a_saved_episode(self, sim, tmp_path) -> None:
        pytest.importorskip("lerobot")
        opened = sim.start_recording(repo_id="local/step_records_e2e", fps=10, root=str(tmp_path), task="three poses")
        assert opened["status"] == "success", _text(opened)
        assert "step records one frame per 1/10s of sim time" in _text(opened)
        keys = sim.robot_action_keys("so101")
        frames = 0
        for pose in (0.3, -0.3, 0.0):
            assert (
                sim.set_joint_positions(positions={keys[0]: pose}, robot_name="so101", hold=True)["status"] == "success"
            )
            result = sim.step(n_steps=round(0.5 / _dt(sim)))
            assert result["status"] == "success", _text(result)
            assert "recorded" in _text(result)
        recorder = sim._world._backend_state["dataset_recorder"]
        frames = recorder.episode_frame_count
        assert frames == 16, frames  # opening frame + 15 over 1.5 s at 10 fps
        closed = sim.stop_recording()
        assert closed["status"] == "success", _text(closed)
        assert "16 frames, 1 episode(s)" in _text(closed)
