"""A cooperative stop issued before a rollout's first frame halts the rollout.

``start_policy`` submits the rollout to an executor and returns; the worker only
reaches the first frame some time later. ``policy_running`` -- the flag
``stop_policy`` lowers and the ``on_frame`` hook reads -- used to be raised by
``_make_run_policy_hook``, which runs on that worker. So for the whole launch
window the flag read ``False`` while ``_active_policy_robots`` already listed the
robot, and a stop in the window was answered ``Was not running`` and then
overwritten by the worker's own raise: the rollout ran to its full duration
having reported that it stopped (#2833).

The launching thread now claims the robot (``_announce_rollout``), so these pin
the two ends of the contract:

* a stop issued anywhere after ``start_policy`` returns is honoured, whether the
  worker is already inside ``run_policy`` or the rollout is still queued, and
* ``stop_policy`` and ``list_policies_running`` do not report opposite facts
  about the same robot at the same instant.

Both windows are forced open deterministically rather than raced: the running
case wraps the hook factory so the worker parks just before the frame the flag
gated, and the queued case saturates the executor so the Future has not started.
The rollout is 100 frames, so "ran anyway" and "was halted" are far apart.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from strands_robots.simulation import create_simulation
from strands_robots.simulation.model_registry import resolve_model_path

DURATION = 2.0
CONTROL_HZ = 50.0
FULL_ROLLOUT_STEPS = int(DURATION * CONTROL_HZ)  # 100


@pytest.fixture
def sim() -> Any:
    """A MuJoCo sim holding one arm, torn down after the test."""
    engine = create_simulation("mujoco")
    engine.create_world()
    engine.add_robot(name="arm", urdf_path=str(resolve_model_path("so101")))
    try:
        yield engine
    finally:
        engine.cleanup()


def _start(sim: Any) -> Any:
    """Launch the background rollout and return its Future."""
    started = sim.start_policy("arm", policy_provider="mock", duration=DURATION, control_frequency=CONTROL_HZ)
    assert started["status"] == "success", started
    return sim._policy_threads["arm"]


def test_a_stop_lands_on_a_running_worker_that_has_not_taken_a_frame(sim: Any) -> None:
    """The worker is inside ``run_policy`` but before the first frame."""
    at_the_hook = threading.Event()
    release = threading.Event()
    real_factory = sim._make_run_policy_hook

    def parked_factory(robot_name: str, instruction: str) -> Any:
        at_the_hook.set()
        assert release.wait(30), "test never released the parked worker"
        return real_factory(robot_name, instruction)

    sim._make_run_policy_hook = parked_factory

    future = _start(sim)
    assert at_the_hook.wait(30), "worker never reached the hook factory"

    # The window is open: the worker has taken no frame. Both readers of "is a
    # rollout in flight" must agree, and the stop must be reported as a stop.
    assert sim._active_policy_robots() == ["arm"]
    assert "arm" in sim.list_policies_running()["content"][0]["text"]
    stopped = sim.stop_policy("arm")
    assert stopped["status"] == "success"
    assert stopped["content"][0]["text"] == "Stopped on 'arm'"

    release.set()
    future.result(timeout=60)
    assert sim._world.robots["arm"].policy_steps == 0
    assert sim._world.robots["arm"].policy_running is False


def test_a_stop_lands_on_a_rollout_that_is_still_queued(sim: Any) -> None:
    """Every worker is busy, so the rollout's Future has not started at all."""
    release = threading.Event()
    for _ in range(sim._executor._max_workers):
        sim._executor.submit(release.wait, 30)

    future = _start(sim)
    try:
        assert not future.running(), "executor was not saturated; the rollout started"
        assert sim._active_policy_robots() == ["arm"]
        stopped = sim.stop_policy("arm")
        assert stopped["content"][0]["text"] == "Stopped on 'arm'"
    finally:
        release.set()

    future.result(timeout=60)
    assert sim._world.robots["arm"].policy_steps == 0


def test_the_shared_stop_seam_halts_a_rollout_in_the_launch_window(sim: Any) -> None:
    """Every stop path reaches ``SimRobot.request_policy_stop``, so pin it directly.

    ``stop_policy`` is one caller; ``remove_robot``, teardown and the Device
    Connect stop / emergency-stop handlers are the others, and they report no
    envelope of their own. Halting a launch-window rollout through the seam they
    share covers all of them without standing up a transport.
    """
    at_the_hook = threading.Event()
    release = threading.Event()
    real_factory = sim._make_run_policy_hook

    def parked_factory(robot_name: str, instruction: str) -> Any:
        at_the_hook.set()
        assert release.wait(30), "test never released the parked worker"
        return real_factory(robot_name, instruction)

    sim._make_run_policy_hook = parked_factory
    future = _start(sim)
    assert at_the_hook.wait(30), "worker never reached the hook factory"

    robot = sim._world.robots["arm"]
    assert robot.request_policy_stop() is True
    release.set()
    future.result(timeout=60)
    assert robot.policy_steps == 0


def test_a_stop_with_nothing_running_is_still_idempotent(sim: Any) -> None:
    """The honest ``Was not running`` case is untouched: no flag, no Future."""
    assert sim._active_policy_robots() == []
    stopped = sim.stop_policy("arm")
    assert stopped["status"] == "success"
    assert stopped["content"][0]["text"] == "Was not running on 'arm'"


def test_a_rollout_left_alone_runs_every_frame(sim: Any) -> None:
    """The control: claiming the robot early does not curtail a rollout."""
    future = _start(sim)
    future.result(timeout=120)
    assert sim._world.robots["arm"].policy_steps == FULL_ROLLOUT_STEPS
