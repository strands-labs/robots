"""Units schemes are registry data (part 3.5): gripper keys, LeRobot norm modes, counts round-trip.

The 'lerobot' scheme reads its per-joint modes from ``hardware.lerobot_norm`` and its gripper keys from
``gripper.keys`` -- both registry data now, not Python constants. A robot LeRobot does not define is
refused by name, and the counts the feetech driver maps match LeRobot's own normalisation to 1e-9.
"""

from __future__ import annotations

import numpy as np
import pytest

from strands_robots.description import Description, registry
from strands_robots.robot.drivers.calibration import Calibration, JointCalibration
from strands_robots.robot.drivers.feetech import FeetechDriver
from strands_robots.units import SCHEMES, Units
from strands_robots.units import lerobot_units as lerobot_scheme_units


def _gripper_robots():
    seen, out = set(), []
    for entry in registry().values():
        if entry["name"] in seen or not entry.get("gripper"):
            continue
        seen.add(entry["name"])
        out.append(entry["name"])
    return out


@pytest.mark.parametrize("name", _gripper_robots())
def test_gripper_keys_are_a_subset_of_action_keys(name):
    d = Description.load(name)
    assert d.gripper is not None
    assert set(d.gripper.keys) <= set(d.action_keys), f"{name}: {d.gripper.keys} not in {d.action_keys}"


def test_an_unknown_scheme_is_refused_naming_the_known_ones():
    d = Description.load("so101")
    with pytest.raises(ValueError, match=r"unknown units 'furlongs'; known: lerobot, rad"):
        Units.for_robot(d, "furlongs")
    assert set(SCHEMES) == {"rad", "lerobot"}


def test_a_robot_without_lerobot_norm_is_refused_by_name():
    d = Description.load("panda")  # driver 'strands', no hardware.lerobot_norm
    assert d.hardware.lerobot_norm is None
    with pytest.raises(ValueError, match=r"panda declares no hardware.lerobot_norm"):
        Units.for_robot(d, "lerobot")


class _FakeBus:
    port = "fake"

    def ping(self, _id):
        return True

    def close(self):
        pass


def test_counts_to_rad_to_lerobot_equals_lerobots_own_normalisation():
    d = Description.load("so101")
    cmin, cmax = 100, 3900
    cal = Calibration({k: JointCalibration(i + 1, 0, 0, cmin, cmax) for i, k in enumerate(d.action_keys)})
    drv = FeetechDriver(d, port=None, bus=_FakeBus(), calibration=cal)
    u = Units.for_robot(d, "lerobot")
    for counts in (cmin, 1000, 2000, cmax):
        rad = np.array([drv.counts_to_rad(k, counts) for k in d.action_keys])
        got = u.from_rad(rad)
        frac = (counts - cmin) / (cmax - cmin)
        for i, k in enumerate(d.action_keys):
            expected = frac * 100.0 if k in d.gripper.keys else frac * 200.0 - 100.0
            assert abs(got[i] - expected) < 1e-9, f"{k} at {counts}: {got[i]} vs {expected}"


def test_range_modes_clamp_outside_the_range_but_degrees_extrapolate():
    """K9: a bounded ``range_*`` key clamps to its ends outside the calibrated travel, as LeRobot's
    ``_normalize`` does; a write is never clamped; a ``degrees`` key is unbounded and keeps extrapolating."""
    d = Description.load("so101")
    body = d.action_keys.index("shoulder_pan")
    grip = d.action_keys.index("gripper")
    ranged = Units.for_robot(d, "lerobot")  # registry default: range_m100_100 body, range_0_100 gripper
    lo, hi = ranged.lo, ranged.hi
    below = lo - (hi - lo)  # one full range below every lower limit
    above = hi + (hi - lo)
    assert ranged.from_rad(below)[body] == pytest.approx(-100.0)  # clamped, not -300
    assert ranged.from_rad(below)[grip] == pytest.approx(0.0)
    assert ranged.from_rad(above)[body] == pytest.approx(100.0)
    # to_rad does not clamp: a write past the end stays out of range for the safety chain to refuse (K6)
    past = np.full(len(d.action_keys), 400.0)
    assert ranged.to_rad(past)[body] > hi[body] + 1.0
    # degrees extrapolate: no clamp
    degrees = lerobot_scheme_units(d, {"*": "degrees", "gripper": "range_0_100"})
    assert not bool(degrees.bounded[body])
    assert abs(degrees.from_rad(below)[body]) > abs(degrees.from_rad(lo)[body]) + 1.0
