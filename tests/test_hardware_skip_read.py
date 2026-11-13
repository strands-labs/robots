"""A hardware run reads the cameras only on a step whose policy looks (``needs_observation``), as a world's run does.

Recording and ``stop_when`` still read every step: their frames need the cameras whatever the policy looks at.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_drivers import make_driver

from strands_robots import Robot
from strands_robots.policy import Policy


class _Camera:
    reads = 0

    def images(self):
        _Camera.reads += 1
        return {"front": np.zeros((48, 64, 3), np.uint8)}


class _Blind(Policy):
    def __init__(self, every=False):
        self.every, self.got = every, []

    def act(self, obs, *, instruction=None):
        self.got.append(set(obs.images))
        return obs.joints.positions

    def needs_observation(self, *, instruction=None):
        return self.every


@pytest.fixture
def arm(monkeypatch):
    _Camera.reads = 0
    d, _ = make_driver()
    monkeypatch.setattr(d, "images", _Camera().images)
    real = Robot("so101", driver=d, twin=False)
    yield real
    real.close()


def test_a_policy_that_does_not_look_gets_no_camera_read_on_hardware(arm):
    blind = _Blind()
    arm.run(blind, hz=200, steps=20)
    assert _Camera.reads <= 1  # the requirements check before step one, nothing per step
    assert blind.got == [set()] * 20


def test_a_policy_that_looks_gets_a_frame_every_step(arm):
    arm.run(_Blind(every=True), hz=200, steps=20)
    assert _Camera.reads >= 20


def test_stop_when_still_sees_the_cameras(arm):
    seen: list = []
    arm.run(_Blind(), hz=200, steps=20, stop_when=lambda obs: seen.append(set(obs.images)) or False)
    assert _Camera.reads >= 20 and seen and all(s == {"front"} for s in seen)
