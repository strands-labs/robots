"""The 'lerobot' scheme's three modes, and a checkpoint's semantics overriding the registry default."""

import numpy as np
import pytest
from test_drivers import make_driver

from strands_robots import Description, Policy, Robot
from strands_robots.units import lerobot_units, semantics_for, units_for_semantics

DEG = {"*": "degrees", "gripper": "range_0_100"}


def test_degrees_are_physical_degrees_about_the_joint_range_midpoint():
    d = Description.load("so101")
    u = lerobot_units(d, DEG)
    lo = np.array([d.limits[k][0] for k in d.action_keys])
    hi = np.array([d.limits[k][1] for k in d.action_keys])
    mid = (lo + hi) / 2
    q = lo + 0.3 * (hi - lo)
    out = u.from_rad(q)
    body = [i for i, k in enumerate(d.action_keys) if k != "gripper"]
    np.testing.assert_allclose(out[body], np.rad2deg(q - mid)[body], atol=1e-9)
    np.testing.assert_allclose(out[d.action_keys.index("gripper")], 30.0, atol=1e-9)  # range_0_100 unchanged
    np.testing.assert_allclose(u.to_rad(out), q, atol=1e-12)  # round trip


def test_range_and_degrees_round_trip_through_radians():
    d = Description.load("so101")
    q = np.array([(d.limits[k][0] + d.limits[k][1]) / 2 + 0.1 for k in d.action_keys])
    for norm in (DEG, {"*": "range_m100_100", "gripper": "range_0_100"}):
        u = lerobot_units(d, norm)
        np.testing.assert_allclose(u.to_rad(u.from_rad(q)), q, atol=1e-12)


def test_semantics_labels_are_data():
    d = Description.load("so101")
    assert units_for_semantics("radians-v1", d) is None
    deg = units_for_semantics("so101-degrees-gripper-percent-v1", d)
    np.testing.assert_allclose(deg.hi_u, lerobot_units(d, DEG).hi_u)
    assert semantics_for(d) == "so101-range-gripper-percent-v1"  # registry default
    assert semantics_for(d, DEG) == "so101-degrees-gripper-percent-v1"


def test_an_unknown_mode_is_refused_by_name():
    d = Description.load("so101")
    with pytest.raises(ValueError, match=r"unknown lerobot_norm mode 'furlongs'"):
        lerobot_units(d, {"*": "furlongs"})


@pytest.mark.parametrize(
    ("label", "norm"),
    [
        ("so101-v2", {"*": "range_m100_100", "gripper": "range_0_100"}),  # says nothing: the robot's own convention
        ("x-v9", {"*": "range_m100_100", "gripper": "range_0_100"}),  # names another thing: still this robot's joints
        ("so101-degrees-v3", {"*": "degrees", "gripper": "range_0_100"}),  # the body's mode, the gripper as calibrated
        ("so101_normalized", {"*": "range_m100_100", "gripper": "range_0_100"}),
        ("lab.so101.deg.gripper-percent", {"*": "degrees", "gripper": "range_0_100"}),
    ],
)
def test_an_unknown_label_is_derived_from_its_words_and_the_robots_convention(label, norm):
    d = Description.load("so101")
    with pytest.warns(
        UserWarning, match=rf"{label!r} is not in strands_robots/units/semantics.json; read as LeRobot modes"
    ):
        got = units_for_semantics(label, d)
    want = lerobot_units(d, norm)
    np.testing.assert_allclose(got.lo_u, want.lo_u)
    np.testing.assert_allclose(got.hi_u, want.hi_u)
    assert got.modes == want.modes


def test_an_unknown_label_that_says_radians_is_radians():
    d = Description.load("so101")
    with pytest.warns(UserWarning, match="read as radians"):
        assert units_for_semantics("lab-radians-v2", d) is None


def test_an_unknown_label_on_a_robot_with_no_convention_is_refused_naming_both():
    d = Description.load("ur5e")  # no hardware.lerobot_norm in the registry
    with pytest.raises(
        ValueError, match=r"unknown LeRobot semantics 'x-v9', and ur5e declares no hardware.lerobot_norm"
    ):
        units_for_semantics("x-v9", d)
    with pytest.warns(UserWarning):  # the label itself says degrees: derivable without a convention
        assert units_for_semantics("ur5e-degrees-v1", d).modes == ("degrees",) * len(d.action_keys)


@pytest.mark.parametrize(
    ("label", "same_as"),
    [
        ("so101-normalized-v1", "so101-range-gripper-percent-v1"),
        ("so100-normalized-v1", "so100-range-gripper-percent-v1"),
    ],
)
def test_a_lerobot_normalized_label_is_an_alias_that_never_becomes_the_default(label, same_as):
    d = Description.load("so101")
    got, want = units_for_semantics(label, d), units_for_semantics(same_as, d)
    np.testing.assert_allclose(got.hi_u, want.hi_u)
    np.testing.assert_allclose(got.lo_u, want.lo_u)
    assert semantics_for(d) == "so101-range-gripper-percent-v1"  # strands still writes its own label


@pytest.mark.parametrize("said", [{}, {"semantics": ""}, {"norm": {}}], ids=["nothing", "empty-label", "empty-norm"])
def test_a_real_arm_refuses_a_checkpoint_that_does_not_say_its_norm_mode(tmp_path, said):
    (tmp_path / "config.json").write_text("{}")  # never read: the refusal comes before LeRobot is imported
    d, s = make_driver()
    real = Robot("so101", driver=d, twin=False)
    s.frames.clear()
    names = r"so101 is a real body: pass norm=\{'\*': 'degrees' \| 'range_m100_100', \.\.\.\}.* or semantics="
    with pytest.raises(ValueError, match=names):  # an empty label falls back to the registry's, an empty norm to rad
        real.run(Policy(str(tmp_path), **said), steps=1)
    assert s.frames == []  # nothing on the wire, not even a read
    real.close()
