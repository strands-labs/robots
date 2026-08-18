# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""The RTC chunk seam survives a wall-clock step during inference.

Real-Time Chunking needs to know how many control steps the executor committed
while an inference was in flight - the RTC paper's ``d``. When the runtime does
not supply that count (true-async hardware driven without a runner, where the arm
really does keep moving during inference) it is derived from measured inference
latency: ``_estimate_inference_delay`` takes the p95 of the recent latency window
and returns ``int(p95 * fps)``. That number is handed to lerobot's denoiser as
``get_prefix_weights(start=d, ...)``, which freezes the first ``d`` actions of the
new chunk to the already-committed prefix, and it is also the offset the
chunk-seam slice skips - so it decides which actions reach the arm.

Those latencies used to be measured with ``time.time()``, which is not a clock
but the current opinion about the date. A wall-clock step - an NTP correction, a
``date -s``, a resume from suspend - is recorded as a single inference that took
the size of the step, and the damage outlives the correction because the window
is a window:

    50-step chunks, 30 Hz, true inference latency 117 ms (delay = 3 steps),
    correction lands during an episode's 2nd inference

    clock event              delay used after it      actions reaching the arm
    no step (control)                       3            47 of the new chunk
    wall clock steps +30s                 903             1 (chunk discarded)
    wall clock steps +1h                >100k             1 (chunk discarded)

At ``d >= execution_horizon`` the whole prefix is frozen: the freshly computed
chunk is discarded at the seam and the arm keeps executing the stale one, with
nothing raised and nothing logged above debug.

The persistence is measured, not assumed. A lone outlier is the p95 of a window
holding 2..20 samples and is excluded from 21 samples on, so one correction
corrupts roughly the next 19 seams and the estimate then recovers by itself -
which is what makes it hard to attribute later. ``reset()`` clears the window at
every episode, so an episode's first inferences are always in that range, and
they are also when a process that has been idle between episodes is most likely
to receive the correction. A flapping clock is worse: 5 poisoned samples in the
full 100-deep window make the outlier the p95 outright.

These tests drive the real ``_predict_with_rtc`` path through a clock double
whose wall clock takes a step mid-inference, and assert on what comes out of the
seam - the number of usable actions - rather than on which clock is called, so
they cannot be satisfied by renaming a call.

Companion to tests/simulation/test_rollout_durations_survive_a_clock_step.py;
same defect class as #2404 (agent tools), #2406 (hardware loops), #2408 (mesh).
"""

from __future__ import annotations

import pytest
import torch  # real or conftest mock - both work

from strands_robots.policies.lerobot_local import policy as lerobot_local_policy_module
from tests.policies.lerobot_local.test_policy import _make_loaded_policy

#: Trained chunk length the fake policy returns.
CHUNK_STEPS = 50
#: Control rate of the executing loop.
FPS = 30.0
#: The policy's real inference latency. At 30 Hz this is a 3-step delay, so the
#: expectation is a non-trivial number rather than a degenerate zero. The value
#: is deliberately not a round 0.1: ``0.117 * 30 == 3.51`` sits away from an
#: integer boundary, so ``int()`` returns 3 whether the subtraction was done on a
#: base of ~1.78e9 (a wall clock, which loses enough precision at that magnitude
#: to turn 0.1 into 0.0999999 and a 3-step delay into 2) or on a small monotonic
#: base. Each test then fails for the clock step it names and nothing else.
TRUE_LATENCY = 0.117
#: Steps the seam skips when the latency is measured honestly.
TRUE_DELAY = int(TRUE_LATENCY * FPS)
#: Inferences to drive per test. Comfortably inside the window range where a
#: lone outlier is the p95 (2..20 samples), which is where a fresh episode sits.
INFERENCES = 12
#: The inference during which the wall clock takes its step.
STEP_ON_CALL = 2


class _SteppingClock:
    """A ``time`` double whose wall clock steps once and whose monotonic never does."""

    def __init__(self, wall_step: float) -> None:
        self._wall_step = wall_step
        # Unrelated epochs so a reader of the wrong clock cannot pass by luck.
        self._mono = 5_000.0
        self._wall = 1_781_000_000.0

    def monotonic(self) -> float:
        return self._mono

    def time(self) -> float:
        return self._wall

    def perf_counter(self) -> float:
        return self._mono

    def sleep(self, seconds: float) -> None:  # pragma: no cover - the RTC path never sleeps
        self.do_work(seconds)

    def do_work(self, seconds: float) -> None:
        """Time passes, as it does while the model computes a chunk."""
        self._mono += seconds
        self._wall += seconds

    def apply_step(self) -> None:
        """The date changes mid-inference. Elapsed time does not."""
        self._wall += self._wall_step


def _make_clock_driven_rtc_policy(clock: _SteppingClock):
    """A loaded RTC policy whose inference costs ``TRUE_LATENCY`` of real time.

    The latency window starts empty, which is what it is at the start of every
    episode: ``reset()`` clears it.
    """
    policy = _make_loaded_policy(include_images=False)
    policy._rtc_enabled = True
    policy._rtc_execution_horizon = 10
    policy._rtc_prev_chunk = None
    policy.actions_per_step = 1
    policy.set_control_frequency(FPS)
    assert not policy._rtc_latency_history, "an episode starts with no measured latencies"

    calls = {"n": 0}

    def _predict_action_chunk(batch, **kwargs):
        calls["n"] += 1
        clock.do_work(TRUE_LATENCY)
        if calls["n"] == STEP_ON_CALL:
            clock.apply_step()
        return torch.zeros((1, CHUNK_STEPS, 6))

    policy._policy.predict_action_chunk = _predict_action_chunk
    return policy


WALL_STEPS = [
    pytest.param(0.0, id="no-step-control"),
    pytest.param(30.0, id="forward-30s"),
    pytest.param(3600.0, id="forward-1h"),
    pytest.param(-2.0, id="backward-2s"),
]


@pytest.mark.parametrize("wall_step", WALL_STEPS)
def test_rtc_seam_keeps_delivering_the_chunk_across_a_clock_step(monkeypatch, wall_step):
    """Every inference after the step still delivers the same usable chunk.

    Pre-fix, the inference the step landed in was recorded as a
    ``wall_step``-long latency, became the p95 of the young window, and every
    seam after it froze its whole prefix - one action instead of 47, for as long
    as the outlier stayed in the p95.
    """
    clock = _SteppingClock(wall_step)
    monkeypatch.setattr(lerobot_local_policy_module, "time", clock)
    policy = _make_clock_driven_rtc_policy(clock)

    usable_lengths = [policy._predict_with_rtc({}).shape[0] for _ in range(INFERENCES)]

    expected = CHUNK_STEPS - TRUE_DELAY
    # The first inference has no measured latency yet, so its delay is 0 by
    # definition and it delivers the untrimmed chunk. Every one after it has the
    # policy's honest latency to work from.
    assert usable_lengths[0] == CHUNK_STEPS
    assert usable_lengths[1:] == [expected] * (INFERENCES - 1), (
        f"the seam delivered {usable_lengths} usable actions per inference "
        f"instead of {expected} from the second on (wall clock stepped "
        f"{wall_step:+g}s during inference {STEP_ON_CALL})"
    )


@pytest.mark.parametrize("wall_step", WALL_STEPS)
def test_rtc_latency_window_records_no_impossible_inference(monkeypatch, wall_step):
    """No latency in the window is negative or longer than the inference took.

    The window feeds a p95 that decides the seam, so a single impossible sample
    is not a cosmetic blemish in a log.
    """
    clock = _SteppingClock(wall_step)
    monkeypatch.setattr(lerobot_local_policy_module, "time", clock)
    policy = _make_clock_driven_rtc_policy(clock)

    for _ in range(INFERENCES):
        policy._predict_with_rtc({})

    measured = list(policy._rtc_latency_history)
    assert all(latency > 0 for latency in measured), f"a negative inference latency was recorded: {measured}"
    assert max(measured) == pytest.approx(TRUE_LATENCY, rel=1e-6), (
        f"longest recorded inference was {max(measured):.3f}s for a "
        f"{TRUE_LATENCY:.3f}s inference (wall clock stepped {wall_step:+g}s)"
    )


@pytest.mark.parametrize("wall_step", WALL_STEPS)
def test_rtc_delay_estimate_is_unmoved_by_a_clock_step(monkeypatch, wall_step):
    """The estimated delay stays the true one, in steps, after the correction."""
    clock = _SteppingClock(wall_step)
    monkeypatch.setattr(lerobot_local_policy_module, "time", clock)
    policy = _make_clock_driven_rtc_policy(clock)

    for _ in range(INFERENCES):
        policy._predict_with_rtc({})

    assert policy._estimate_inference_delay(fps=FPS) == TRUE_DELAY
