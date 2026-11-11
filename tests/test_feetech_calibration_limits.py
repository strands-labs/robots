"""A Feetech calibration must fit the joint it maps: its span may not cover much more than the joint's limits.

The driver maps each joint's limits onto its ``range_min..range_max``; a span wider than the limits take on the servo
would put a goal inside them past the end stop. Every bus here is the suite's ``FakeSerial``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from test_drivers import CAL, FakeSerial, SweepSerial, make_driver
from test_feetech_calibration import feetech, homing, on_disk

from strands_robots import Description
from strands_robots.robot.drivers import Calibration

FIXTURE = Path(__file__).parent / "fixtures" / "so101_follower_calibration.json"
REAL = json.loads(FIXTURE.read_text())  # a real SO-101's lerobot-calibrate file


def _with(cal, **joints):
    return cal | {name: dict(cal[name], **fields) for name, fields in joints.items()}


@pytest.mark.parametrize(
    "cal, joint",
    [
        (_with(REAL, shoulder_lift=dict(range_min=0, range_max=4095)), "shoulder_lift"),  # a full turn for +-100 deg
        (
            _with(
                REAL,
                gripper=dict(range_min=REAL["shoulder_pan"]["range_min"], range_max=REAL["shoulder_pan"]["range_max"]),
            ),
            "gripper",
        ),  # another joint's span
        (_with(CAL, gripper=dict(range_min=1024, range_max=3072)), "gripper"),  # half a turn for a third of one
    ],
    ids=["full-turn", "another-joints-span", "too-wide-gripper"],
)
def test_a_span_much_wider_than_the_joints_limits_is_refused_at_connect(tmp_path, cal, joint):
    path = tmp_path / "arm.json"
    path.write_text(json.dumps(cal))
    for given, source in ((path, str(path)), (cal, "caller")):
        with pytest.raises(ValueError, match=re.escape(f"calibration {source}: joint {joint!r} spans")):
            make_driver(cal=given)


@pytest.mark.parametrize(
    "cal",
    [REAL, CAL, _with(REAL, wrist_roll=dict(range_min=0, range_max=4095)), _with(REAL, gripper=dict(range_max=2100))],
    ids=["a-real-so101", "the-suites", "lerobots-full-turn-wrist-roll", "a-narrow-span"],
)
def test_a_calibration_that_fits_its_joints_loads(cal):
    drv, _ = make_driver(cal=cal)
    assert drv.calibration == Calibration.from_json(cal)


def test_the_check_scales_with_the_limits():
    limits = Description.load("so101").limits
    cal = Calibration.from_json(_with(REAL, shoulder_lift=dict(range_min=0, range_max=4095)))
    with pytest.raises(ValueError, match=r"1\.80x"):
        cal.check_limits(limits, resolution=4096, source="x")
    assert cal.check_limits(limits, resolution=4096, source="x", max_scale=2.0) is cal


def test_extras_names_the_calibration_by_its_values(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path / "empty"))  # none: no file on this host is looked at
    path = tmp_path / "arm.json"
    path.write_text(json.dumps(REAL, indent=4))
    from_file, _ = make_driver(cal=path)
    given, _ = make_driver(cal=dict(reversed(REAL.items())))
    other, _ = make_driver(cal=_with(REAL, gripper=dict(range_max=3445)))
    none, _ = make_driver(cal=None, serial=FakeSerial())
    digest = from_file.extras()["calibration_sha256"]
    assert len(digest) == 64 and digest == given.extras()["calibration_sha256"]  # key order and whitespace aside
    assert other.extras()["calibration_sha256"] != digest  # one count changes it
    assert none.extras()["calibration_sha256"] is None


def test_calibrate_refuses_a_sweep_wider_than_the_limits_and_saves_nothing(tmp_path, monkeypatch):
    robots = on_disk(tmp_path, monkeypatch, {"so_follower/my_arm.json": REAL})
    serial = SweepSerial()
    homing(serial, 120)
    serial.side = 1500  # 3000 counts each sweep: inside the shoulder's limits, two and a half times the gripper's
    d = feetech(serial, robot_id="my_arm")
    answers = iter(["c", "", ""])  # re-record, middle pose, sweep
    with pytest.raises(ValueError, match=re.escape("calibration calibrate(): joint 'gripper' spans")):
        d.calibrate(interactive=True, prompt=lambda msg: next(answers), sweep_seconds=0.1)
    assert json.loads((robots / "so_follower/my_arm.json").read_text()) == REAL  # the file is as it was
    assert d.calibration == Calibration.from_json(REAL)
    assert [serial._homing(i) for i in serial.regs] == [120] * 6  # the servos hold their offsets again
