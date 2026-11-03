"""A rate or a goal that is not a finite number never reaches the bus.

``json.loads`` gives ``inf`` and ``nan`` for ``Infinity`` and ``NaN``, and both passed ``hz <= 0``: at such a rate
no tick sleeps, so a hardware run or a state stream free-runs the bus. Every bus here is the suite's ``FakeSerial``.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest
from test_drivers import make_driver, writes
from test_operator_gate import goal_writes

from strands_robots import Robot


class _Loads:
    """A policy whose ``attach`` stands for loading a checkpoint."""

    attached = False

    def attach(self, robot):
        self.attached = True

    def act(self, obs, *, instruction=None):
        return {"shoulder_pan": 0.1}

    def reset(self):
        pass


@pytest.mark.parametrize("hz", [math.inf, math.nan], ids=repr)
def test_a_run_at_a_non_finite_rate_is_refused_before_its_policy_loads(hz):
    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False)
    pol = _Loads()
    with pytest.raises(ValueError, match="hz must be a finite number > 0"):
        real.run(pol, hz=hz, steps=50)
    assert not pol.attached and goal_writes(s) == []  # dev: 50 steps of goal writes in under 10 ms
    real.close()


@pytest.mark.parametrize("hz", [math.inf, math.nan], ids=repr)
def test_a_state_stream_at_a_non_finite_rate_is_refused_before_it_reads_the_bus(hz):
    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False)
    try:
        sent = len(s.frames)
        with pytest.raises(ValueError, match="hz must be a finite number > 0"):
            real.subscribe("state", lambda payload: None, hz=hz)
        time.sleep(0.05)
        assert len(s.frames) == sent  # dev: about 6,000 bus reads in 0.2 s, where hz=50 makes 11
    finally:
        real.close()


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf], ids=repr)
def test_a_non_finite_goal_written_to_the_driver_leaves_torque_off(bad):
    d, s = make_driver()
    with pytest.raises(ValueError, match="non-finite goal"):
        d.write(np.array([bad, 0, 0, 0, 0, 0]))
    assert writes(s) == [] and d.extras()["torque"] is False  # dev: Torque_Enable=1 went out first
    d.close()
