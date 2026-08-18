# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A sim rollout's frame pacing and reported duration survive a wall-clock step.

``PolicyRunner.replay`` paces recorded frames to the dataset's rate: it reads a
base at the top of each frame, applies the recorded action, and sleeps for
whatever is left of ``frame_interval``. That base used to be ``time.time()``,
which is not a clock but the current opinion about the date - an NTP correction,
a ``date -s``, a resume from suspend moves it by an arbitrary amount - and the
sleep was computed from two readings of it, so a step landing mid-episode
changed how fast the recorded trajectory was sent to the robot:

    30 fps replay, 40 frames, ~5 ms of work per frame

    clock event               achieved frame interval after the event   reported
    no step (control)                             33.3 ms                1.3s
    wall clock steps +30s               5.0 ms (pacing skipped)         31.3s
    wall clock steps -2s                    2033.3 ms (one stall)        -0.7s

Forward, ``frame_interval - (time.time() - step_start)`` goes negative, the
sleep is skipped and the remaining frames are pushed at whatever rate the loop
can turn over - a position-servo arm is commanded through the recorded targets
6x faster than they were recorded. Backward, the same subtraction goes negative
in the other direction and the sleep becomes ``frame_interval + step``, so the
replay stalls mid-episode with the arm parked at a recorded pose. Neither is
reported, because nothing raises: both finish ``Frames: N/N`` with
``status="success"``, and the ``duration_s`` in that result moved with the step
too, so the record does not show what happened either.

The same clock backed the rollout duration ``run()`` reports as ``elapsed_s``.

These tests pin the contract on behaviour rather than on which clock is called:
each drives the real ``replay()`` loop through a clock double whose *wall* clock
takes a known step mid-episode while its monotonic clock does not, and asserts
the *achieved* frame interval - the pacing the robot actually experiences -
stayed at the dataset's rate, and that the reported duration is the time that
actually elapsed. Asserting on the achieved interval rather than on the value of
``sleep_time`` is deliberate: it cannot be satisfied by renaming a call.

This is the boundary the library settled for its agent-callable tools (#2404),
its hardware control loops (#2406) and its mesh (#2408). No simulator, dataset
or robot is touched: the engine is an in-memory fake and the dataset is a list
of dicts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from strands_robots import dataset_recorder as dataset_recorder_module
from strands_robots.simulation import policy_runner as policy_runner_module
from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.policy_runner import PolicyRunner

#: Recorded rate of the fake dataset, and therefore the rate replay must pace at.
FPS = 30.0
#: Frames in the fake episode. Enough that a step lands mid-episode with many
#: frames on either side of it.
FRAMES = 40
#: Simulated cost of applying one frame (physics + actuator write), in seconds.
#: Well under one frame interval, so a loop that paces correctly always sleeps.
WORK_PER_FRAME = 0.005
#: Frame index the wall-clock step lands on.
STEP_AT_FRAME = 20


class _SteppingClock:
    """A ``time`` double whose wall clock takes one step and whose monotonic does not.

    Both clocks advance together for simulated work and for sleeps, which is
    what makes the *virtual* monotonic reading a faithful record of the pacing
    the robot experienced. ``apply_step`` moves only the wall clock, which is
    exactly what an NTP correction does. Sleeps are recorded and advance the
    clocks instead of blocking, so the suite stays fast.
    """

    def __init__(self, wall_step: float) -> None:
        self._wall_step = wall_step
        # Distinct, unrelated epochs: a test that accidentally reads the wrong
        # clock cannot coincidentally pass.
        self._mono = 10_000.0
        self._wall = 1_781_000_000.0
        self.sleeps: list[float] = []

    # -- the time module surface replay() uses --
    def monotonic(self) -> float:
        return self._mono

    def time(self) -> float:
        return self._wall

    def perf_counter(self) -> float:
        return self._mono

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self._advance(seconds)

    # -- test-side drivers --
    def _advance(self, seconds: float) -> None:
        self._mono += seconds
        self._wall += seconds

    def do_work(self, seconds: float) -> None:
        """Time passes on both clocks, as it does while a frame is applied."""
        self._advance(seconds)

    def apply_step(self) -> None:
        """The date changes. Nothing else does."""
        self._wall += self._wall_step


class _FakeEngine(SimEngine):
    """The smallest engine ``replay()`` drives, plus a per-frame clock hook.

    Records the monotonic reading at the top of every applied frame; the
    differences between those readings are the achieved frame intervals.
    """

    def __init__(self, clock: _SteppingClock) -> None:
        self._clock = clock
        self.frame_starts_mono: list[float] = []
        self.substeps: list[int] = []

    # -- engine surface --
    def list_robots(self) -> list[str]:
        return ["arm"]

    def robot_action_keys(self, robot_name: str) -> list[str]:
        return ["j1", "j2"]

    def physics_timestep(self) -> float:
        return 0.002

    def send_action(
        self,
        action: dict[str, Any] | Sequence[float],
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> dict[str, Any]:
        self.frame_starts_mono.append(self._clock.monotonic())
        self.substeps.append(n_substeps)
        # Applying a frame costs time on every clock in the process.
        self._clock.do_work(WORK_PER_FRAME)
        if len(self.frame_starts_mono) == STEP_AT_FRAME:
            self._clock.apply_step()
        return {"status": "success"}

    def step(self, n_steps: int = 1) -> dict[str, Any]:  # pragma: no cover - no actionless frames here
        raise AssertionError("every fake frame carries an action")

    # -- SimEngine abstract boilerplate: replay() reaches none of it -- #

    def create_world(self, timestep=None, gravity=None, ground_plane=True):
        return {"status": "success"}

    def destroy(self):
        return {"status": "success"}

    def reset(self):
        return {"status": "success"}

    def get_state(self):
        return {"sim_time": 0.0, "step_count": 0}

    def add_robot(self, name, **kw):
        return {"status": "success"}

    def remove_robot(self, name):
        return {"status": "success"}

    def robot_joint_names(self, robot_name: str) -> list[str]:
        return ["j1", "j2"]

    def add_object(self, name, **kw):
        return {"status": "success"}

    def remove_object(self, name):
        return {"status": "success"}

    def get_observation(self, robot_name=None, *, skip_images=False):
        return {}

    def render(self, camera_name="default", width=None, height=None):
        return {"status": "success"}


class _FakeDataset:
    """A recorded episode as a column store of ``{"action": [...]}`` frames."""

    def __init__(self, frames: int, fps: float) -> None:
        self.fps = fps
        self._frames = [{"action": [float(i), float(-i)]} for i in range(frames)]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._frames[idx]


@pytest.fixture
def replay_harness(monkeypatch):
    """Return a factory that runs the real ``replay()`` under a stepping clock."""

    def _run(wall_step: float) -> tuple[dict, _FakeEngine, _SteppingClock]:
        clock = _SteppingClock(wall_step)
        engine = _FakeEngine(clock)
        dataset = _FakeDataset(FRAMES, FPS)

        monkeypatch.setattr(policy_runner_module, "time", clock)
        monkeypatch.setattr(
            dataset_recorder_module,
            "load_lerobot_episode",
            lambda repo_id, episode=0, root=None: (dataset, 0, FRAMES),
        )

        result = PolicyRunner(engine).replay("local/fake", robot_name="arm")
        assert result["status"] == "success", result["content"][0]["text"]
        return result, engine, clock

    return _run


def _achieved_intervals(engine: _FakeEngine) -> list[float]:
    starts = engine.frame_starts_mono
    return [b - a for a, b in zip(starts, starts[1:], strict=False)]


#: Every step a wall clock realistically takes, in both directions: a leap-second
#: smear, an NTP correction after a long offline stretch, a resume from suspend,
#: and a correction backwards.
WALL_STEPS = [
    pytest.param(0.0, id="no-step-control"),
    pytest.param(30.0, id="forward-30s"),
    pytest.param(3600.0, id="forward-1h"),
    pytest.param(-2.0, id="backward-2s"),
    pytest.param(-30.0, id="backward-30s"),
]


@pytest.mark.parametrize("wall_step", WALL_STEPS)
def test_replay_paces_every_frame_at_the_recorded_rate(replay_harness, wall_step):
    """The achieved frame interval is the dataset's, on both sides of the step.

    This is what the robot experiences. Pre-fix, a forward step dropped the
    pacing for the rest of the episode (interval collapses to the work time) and
    a backward step inserted a stall of the step's size at the frame it landed
    on.
    """
    _, engine, _ = replay_harness(wall_step)

    frame_interval = 1.0 / FPS
    intervals = _achieved_intervals(engine)
    assert len(intervals) == FRAMES - 1

    for idx, achieved in enumerate(intervals):
        assert achieved == pytest.approx(frame_interval, rel=1e-9), (
            f"frame {idx} was paced at {achieved * 1e3:.1f} ms instead of "
            f"{frame_interval * 1e3:.1f} ms (wall clock stepped {wall_step:+g}s "
            f"at frame {STEP_AT_FRAME})"
        )


@pytest.mark.parametrize("wall_step", WALL_STEPS)
def test_replay_reports_the_duration_that_actually_elapsed(replay_harness, wall_step):
    """``duration_s`` is the time that passed, not the date's opinion of it.

    Pre-fix a +30s step reported a 1.3s replay as 31.3s and a -2s step reported
    it as negative, both under ``status="success"``.
    """
    result, engine, _ = replay_harness(wall_step)

    # The pacing sleep sits at the bottom of the frame loop, so a correctly
    # paced episode occupies exactly one frame interval per recorded frame.
    expected = FRAMES * (1.0 / FPS)
    duration_s = result["content"][1]["json"]["duration_s"]
    assert duration_s == pytest.approx(expected, abs=0.01), (
        f"replay reported {duration_s}s for a {expected:.2f}s episode (wall clock stepped {wall_step:+g}s)"
    )
    assert duration_s > 0


@pytest.mark.parametrize("wall_step", WALL_STEPS)
def test_replay_never_sleeps_longer_than_one_frame(replay_harness, wall_step):
    """No single sleep exceeds a frame interval, so no step can stall the arm.

    A backward step of S used to make one sleep ``frame_interval + S`` while the
    arm held the pose of the frame it had just been sent.
    """
    _, _, clock = replay_harness(wall_step)

    frame_interval = 1.0 / FPS
    assert clock.sleeps, "a paced replay sleeps between frames"
    assert max(clock.sleeps) <= frame_interval + 1e-12, (
        f"longest sleep was {max(clock.sleeps):.3f}s for a {frame_interval:.3f}s "
        f"frame (wall clock stepped {wall_step:+g}s)"
    )
