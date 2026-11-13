"""A tick clips its targets into the joint limits once per coercion, and the clip is the per-joint clamp it was."""

from dataclasses import replace

import numpy as np
import pytest

from strands_robots import Description, Robot
from strands_robots.robot import targets as T
from tests.test_drivers import FakeSerial, make_driver


def _per_joint(description, targets):
    out = np.array(targets, dtype=float)
    for i, k in enumerate(description.action_keys):
        lim = description.limits.get(k)
        if lim is not None:
            out[i] = min(max(out[i], lim[0]), lim[1])
    return out


@pytest.mark.parametrize("name", ["so101", "unitree_g1"])
def test_clip_is_the_per_joint_clamp(name):
    desc = Description.load(name)
    rng = np.random.default_rng(0)
    n = len(desc.action_keys)
    for _ in range(200):
        t = rng.normal(0, 5, n)
        assert np.array_equal(T.clip_to_limits(desc, t), _per_joint(desc, t))


def test_a_joint_without_limits_or_with_a_nan_end_is_unbounded_there_and_nan_stays_nan():
    desc = Description.load("so101")
    k0, k1, *_ = desc.action_keys
    limits = {k: v for k, v in desc.limits.items() if k != k0}
    limits[k1] = (float("nan"), 0.5)
    odd = replace(desc, limits=limits)
    t = np.full(len(desc.action_keys), 1e6)
    t[1] = -1e6
    out = T.clip_to_limits(odd, t)
    assert out[0] == 1e6 and out[1] == -1e6 and np.array_equal(out, _per_joint(odd, t))
    assert np.isnan(T.clip_to_limits(desc, np.full(len(desc.action_keys), np.nan))).all()


def test_new_limits_are_read_not_a_stale_copy():
    desc = Description.load("so101")
    k = desc.action_keys[0]
    T.clip_to_limits(desc, np.zeros(len(desc.action_keys)))
    tight = replace(desc, limits={**desc.limits, k: (-0.1, 0.1)})
    assert T.clip_to_limits(tight, np.ones(len(desc.action_keys)))[0] == 0.1


def test_a_mapping_tick_clips_twice_not_three_times(monkeypatch):
    d, _ = make_driver(serial=FakeSerial())
    robot = Robot("so101", driver=d, twin=False)
    calls = []
    real = T.clip_to_limits
    monkeypatch.setattr(T, "clip_to_limits", lambda desc, t: calls.append(1) or real(desc, t))
    robot.run(lambda obs: {"shoulder_pan": 0.1}, hz=100, steps=10)
    assert len(calls) == 20  # the tick's coercion and the write's guard: two per tick


def test_an_agent_run_still_clips_the_joints_it_leaves_out_but_not_its_own():
    d, _ = make_driver(serial=FakeSerial())
    robot = Robot("so101", driver=d, twin=False)
    robot._held = np.full(len(robot.description.action_keys), 99.0)  # a held target past every limit
    out = robot.coerce({"shoulder_pan": 99.0}, clip=False)
    lo, hi = robot.description.limits["shoulder_lift"]
    assert out[0] == 99.0 and lo <= out[1] <= hi
