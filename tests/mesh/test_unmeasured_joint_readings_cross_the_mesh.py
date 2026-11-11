"""A joint with no effort or velocity reading still answers over the mesh.

ros2_control publishes NaN effort for every joint without an effort state interface (mock hardware, and many
position-only drivers). The wire refuses NaN, so the reply to a move that happened came back as
"cannot send NaN or infinity over the mesh" and a caller could retry a move that already ran (measured on a
ros2_control mock-hardware arm over AWS IoT Core: the arm was at the target while the caller saw an error). An
unmeasured reading now travels as null and decodes back to NaN; a non-finite command is still refused.
"""

from __future__ import annotations

import dataclasses
import threading

import numpy as np
import pytest

from strands_robots import Robot


def _without_effort(obs):
    js = obs.joints
    return dataclasses.replace(obs, joints=dataclasses.replace(js, efforts=np.full(len(js.names), np.nan)))


@pytest.fixture
def served(mesh_pair, monkeypatch):
    a, b = mesh_pair
    arm = Robot("so101")
    observe, confirm_act = arm.observe, arm.confirm_act
    monkeypatch.setattr(arm, "observe", lambda *x, **k: _without_effort(observe(*x, **k)))
    monkeypatch.setattr(arm, "confirm_act", lambda *x, **k: _without_effort(confirm_act(*x, **k)))
    a.mount(arm, device_id="so101-a", labels={"role": "arm"})
    assert b.discover("so101-a", min_count=1, timeout=3.0)
    yield arm, a, b
    a.close()
    arm.close()


def test_a_move_reports_success_when_a_joint_has_no_effort_reading(served):
    arm, _, b = served
    obs = b.invoke("so101-a", "act", action={"shoulder_pan": 0.3}, steps=50)
    assert obs.joints["shoulder_pan"] == pytest.approx(arm.observe().joints["shoulder_pan"])
    assert np.isnan(obs.joints.efforts).all()  # absent stays absent: null on the wire, NaN again on arrival
    assert np.isfinite(obs.joints.positions).all()


def test_observe_returns_the_positions_of_a_joint_with_no_effort_reading(served):
    arm, _, b = served
    obs = b.invoke("so101-a", "observe")
    np.testing.assert_allclose(obs.joints.positions, arm.observe().joints.positions)
    assert np.isnan(obs.joints.efforts).all()


def test_the_state_stream_keeps_flowing_with_an_unmeasured_effort(served):
    _, _, b = served
    got, seen = [], threading.Event()
    unsubscribe = b.subscribe("so101-a", "state", lambda p: (got.append(p), seen.set()))
    try:
        assert seen.wait(5.0), "no state sample arrived"
    finally:
        unsubscribe()
    sample = got[-1]
    assert sample["shoulder_pan.effort"] is None
    assert isinstance(sample["shoulder_pan.pos"], float)


def test_a_non_finite_command_is_still_refused(served):
    _, _, b = served
    with pytest.raises(ValueError, match="cannot send NaN or infinity over the mesh"):
        b.invoke("so101-a", "act", action={"shoulder_pan": float("nan")})
