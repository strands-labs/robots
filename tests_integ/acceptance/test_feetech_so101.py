"""Acceptance check 9: a feetech Driver on a physical SO-101, through the v1 surface (#4749).

Plug the follower arm in and point ``STRANDS_SO101_PORT`` at its serial port, then run
``pytest tests_integ/acceptance``. Without an arm every test here skips and says why. The only motion is a
hold at the measured pose, written once and only after the operator gate allowed it; torque is off at exit.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from strands_robots import Robot, Simulation
from strands_robots.robot.drivers import Calibration
from strands_robots.robot.drivers import feetech as ft

PORT = os.environ.get("STRANDS_SO101_PORT", "")
HOLD_COUNTS = 20  # a loaded servo holds within its deadband; 20 counts is 1.8 degrees

pytestmark = pytest.mark.skipif(
    not (PORT and Path(PORT).exists()),
    reason="no SO-101 on the bus: set STRANDS_SO101_PORT to the follower's serial port",
)


@pytest.fixture(autouse=True)
def _fresh_default_world() -> Iterator[None]:
    yield
    if Simulation._default is not None:
        Simulation._default.close()
    Simulation._default = None


@pytest.fixture
def arm() -> Iterator[Callable[..., Robot]]:
    opened: list[Robot] = []

    def open_arm(confirm: Callable[[np.ndarray], bool]) -> Robot:
        opened.append(Robot("so101", port=PORT, confirm=confirm))
        return opened[-1]

    yield open_arm
    for robot in opened:
        driver = robot.body
        robot.close()  # torque off, port closed, twin removed
        assert isinstance(driver, ft.FeetechDriver) and driver.extras()["torque"] is False


def counts(robot: Robot) -> np.ndarray:
    """Present_Position of every servo, straight off the wire."""
    driver = robot.body
    assert isinstance(driver, ft.FeetechDriver)
    raw = driver.bus.sync_read(ft.REG_PRESENT_POSITION, 2, driver.ids)
    return np.array([ft.decode_sm(raw[i], 15) for i in driver.ids])


def test_observe_reads_six_measured_joints(arm):
    real = arm(lambda action: False)
    obs = real.observe(images=False)
    assert obs.joints.names == real.action_keys and len(obs.joints.names) == 6
    assert np.isfinite(obs.joints.positions).all()


def test_a_refused_hold_writes_nothing(arm):
    real = arm(lambda action: False)
    before = counts(real)
    with pytest.raises(PermissionError, match="refused"):
        real.confirm_act(real.observe(images=False).joints)
    assert real.body.extras()["torque"] is False
    assert np.abs(counts(real) - before).max() <= 1


def test_an_allowed_hold_stays_put_and_the_twin_mirrors_it(arm):
    asked: list[np.ndarray] = []

    def operator(action: np.ndarray) -> bool:  # a hold at the measured pose, or nothing
        asked.append(np.array(action))
        return bool(np.allclose(action, measured.positions, atol=1e-6))

    real = arm(operator)
    before, measured = counts(real), real.observe(images=False).joints
    try:
        real.confirm_act(measured)
    except PermissionError:  # a joint rests outside the model's limits, so the clipped target is not a hold
        assert real.body.extras()["torque"] is False and np.abs(counts(real) - before).max() <= 1
        outside = [k for k, a, m in zip(real.action_keys, asked[0], measured.positions, strict=True) if a != m]
        pytest.skip(f"{outside} rest outside the model's limits: the hold was refused, nothing written")
    assert len(asked) == 1 and real.body.extras()["torque"] is True
    assert np.abs(counts(real) - before).max() <= HOLD_COUNTS
    twin = real.twin
    assert twin is not None and twin.sim is not None
    twin.sim.step(200)
    np.testing.assert_allclose(twin.observe().joints.positions, real.coerce(measured), atol=0.05)


def test_calibrate_returns_lerobots_file_unchanged(arm):
    real = arm(lambda action: False)
    driver = real.body
    assert isinstance(driver, ft.FeetechDriver)
    if driver.calibration_source is None:  # the file the driver loaded, if it found one
        pytest.skip(f"no LeRobot calibration in {driver.calibration_looked}: run lerobot-calibrate for this arm first")
    on_disk = json.loads(Path(driver.calibration_source).read_text())
    cal = real.calibrate(interactive=False)
    assert cal == Calibration.from_json(on_disk) and cal.to_json() == on_disk


def test_a_read_costs_the_reply_not_the_timeout(arm):
    """Six servos answer in about a millisecond; a bus that waits out its read timeout reads at 50 ms."""
    driver = arm(lambda action: False).body
    for _ in range(10):
        driver.read()
    ms = []
    for _ in range(100):
        t = time.perf_counter()
        driver.read()
        ms.append((time.perf_counter() - t) * 1e3)
    assert float(np.median(ms)) < 3.0, f"Driver.read() p50 {np.median(ms):.1f} ms"
