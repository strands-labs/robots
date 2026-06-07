"""Tests for teleop input-frame validation (pentest B-04 / F-02).

InputReceiver._on_input must validate frames via
security.validate_input_frame before applying them to the robot, so a
LAN-adjacent peer cannot drive joints with unbounded / non-finite /
malformed values.
"""

from __future__ import annotations

import math

import pytest

from strands_robots.mesh import security
from strands_robots.mesh.input import InputReceiver

# --- validate_input_frame unit tests -------------------------------------


def test_valid_frame_passes_through():
    frame = {"motor.pos": 0.5, "shoulder_pan": -1.25, "j0": 10}
    out = security.validate_input_frame(frame)
    assert out == {"motor.pos": 0.5, "shoulder_pan": -1.25, "j0": 10.0}
    assert all(isinstance(v, float) for v in out.values())


def test_non_dict_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame([1, 2, 3])


def test_too_many_keys_rejected():
    frame = {f"j{i}": 0.0 for i in range(security.MAX_INPUT_FRAME_KEYS + 1)}
    with pytest.raises(security.ValidationError):
        security.validate_input_frame(frame)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_non_finite_rejected(bad):
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"j0": bad})


def test_out_of_range_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"j0": security.MAX_INPUT_VALUE_ABS * 2})


def test_bad_key_charset_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"../etc/passwd": 0.0})


def test_bool_value_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"j0": True})


def test_non_numeric_value_rejected():
    with pytest.raises(security.ValidationError):
        security.validate_input_frame({"j0": "0.5"})


# --- InputReceiver wiring tests ------------------------------------------


class _FakeMesh:
    peer_id = "follower-1"

    def subscribe(self, *a, **k):
        return "sub"

    def unsubscribe(self, *a, **k):
        pass


def _make_receiver():
    applied: list[dict] = []
    recv = InputReceiver(
        mesh=_FakeMesh(),
        robot=object(),
        source_peer_id="leader-1",
        apply_fn=lambda robot, action: applied.append(action),
    )
    recv._running = True
    return recv, applied


def test_on_input_applies_valid_frame():
    recv, applied = _make_receiver()
    recv._on_input(recv.topic, {"action": {"j0": 0.1, "j1": 0.2}, "seq": 0})
    assert applied == [{"j0": 0.1, "j1": 0.2}]
    assert recv._frame_count == 1
    assert recv._rejected == 0


def test_on_input_rejects_malicious_frame():
    recv, applied = _make_receiver()
    # non-finite value would otherwise reach send_action()
    recv._on_input(recv.topic, {"action": {"j0": math.inf}, "seq": 0})
    assert applied == []  # never applied
    assert recv._frame_count == 0
    assert recv._rejected == 1


def test_on_input_rejects_giant_frame():
    recv, applied = _make_receiver()
    giant = {f"j{i}": 0.0 for i in range(security.MAX_INPUT_FRAME_KEYS + 5)}
    recv._on_input(recv.topic, {"action": giant, "seq": 0})
    assert applied == []
    assert recv._rejected == 1
