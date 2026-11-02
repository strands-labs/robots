"""A replay rebuilds the units its dataset was recorded in: LeRobot's norm mode per joint, never the registry's guess.

The parquet reader is LeRobot's (``tests/recording`` runs it end to end); here one recorded row stands in for it,
so the required check exercises what a replay does with ``meta/units.json`` on the repo's fake Feetech bus.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from test_drivers import make_driver

from strands_robots import Description, Policy, Robot
from strands_robots.policy import dataset
from strands_robots.recording.features import units_meta
from strands_robots.robot.drivers import feetech as ft
from strands_robots.units import lerobot_units

DEG = {"*": "degrees", "gripper": "range_0_100"}
ROW = [45.0] * 5 + [50.0]  # 45 degrees on every arm joint, the gripper half open
AT_ROW = [2560, 2560, 2560, 2560, 2560, 2048]  # ROW in DEG on the fake bus: what the degrees run itself writes
WHOLE = "norm={'*': 'degrees', 'gripper': 'range_0_100'} or norm={'*': 'range_m100_100', 'gripper': 'range_0_100'}"


class Degrees:
    """What a LeRobot checkpoint run with norm=DEG declares, without loading one."""

    def units_for(self, description):
        return lerobot_units(description, DEG)

    def act(self, obs, *, instruction=None):
        return np.array(ROW)

    def reset(self):
        pass


def _arm():
    driver, serial = make_driver()
    arm = Robot("so101", driver=driver, twin=False)
    serial.frames.clear()
    return arm, serial


def _goals(policy, units="auto"):
    """The Goal_Position counts one step of ``policy`` writes to a hardware so101 on the fake bus."""
    arm, serial = _arm()
    try:
        arm.run(policy, steps=1, units=units)
    finally:
        arm.close()
    at = slice(ft.REG_GOAL_POSITION, ft.REG_GOAL_POSITION + 2)
    return [int.from_bytes(serial.regs[i][at], "little") for i in sorted(serial.regs)]


def _units_json(norm, *, modes=True):
    d = Description.load("so101")
    said = units_meta(lerobot_units(d, norm), d)
    for j in said["joints"].values():
        if not modes:
            j.pop("mode", None)  # a units.json written before it named the modes: the scheme and the ranges only
    return said


@pytest.fixture
def recorded(tmp_path, monkeypatch):
    """A dataset directory with ``meta/info.json``, and ``meta/units.json`` when given; its one row is ``ROW``."""
    monkeypatch.setattr(dataset, "_episode_actions", lambda root, episode: np.array([ROW]))
    keys = Description.load("so101").action_keys

    def make(name, units_json):
        (tmp_path / name / "meta").mkdir(parents=True)
        info = {"fps": 30, "features": {"action": {"names": [f"{k}.pos" for k in keys]}}}
        (tmp_path / name / "meta" / "info.json").write_text(json.dumps(info))
        if units_json is not None:
            (tmp_path / name / "meta" / "units.json").write_text(json.dumps(units_json))
        return str(tmp_path / name)

    return make


def test_units_json_names_the_norm_mode_of_every_joint():
    keys = Description.load("so101").action_keys
    said = _units_json(DEG)
    modes = {k: j.get("mode") for k, j in said["joints"].items()}
    assert modes == {f"{k}.pos": "range_0_100" if k == "gripper" else "degrees" for k in keys}
    assert units_meta(None, Description.load("so101")) == {"units": "rad", "joints": {}}  # radians have no modes


def test_a_degrees_recording_replays_on_a_real_arm_at_the_counts_the_run_wrote(recorded):
    direct = _goals(Degrees())
    replayed = _goals(Policy(recorded("deg", _units_json(DEG))))
    print(f"[measured] Goal_Position counts: the degrees run {direct}, its replay {replayed}")
    assert replayed == direct


def test_a_real_arm_refuses_a_replay_that_does_not_name_its_norm_modes(recorded):
    for spec, kw in (
        (recorded("old", _units_json(DEG, modes=False)), {}),
        (recorded("bare", None), {"units": "lerobot"}),
    ):
        arm, serial = _arm()
        with pytest.raises(ValueError, match="pass norm=") as refused:
            arm.run(Policy(spec, **kw), steps=1)
        arm.close()
        assert "real body" in str(refused.value) and serial.frames == []  # refused before the bus
        assert WHOLE in str(refused.value)  # the hint names the gripper: a "*" alone would give it the body's mode
        assert _goals(Policy(spec, norm=DEG)) == _goals(Degrees())  # the caller names the modes: exact
        assert Robot("so101").run(Policy(spec, **kw), steps=1).steps == 1  # a simulated arm: the registry's modes


def test_a_callers_norm_never_overrides_the_modes_the_dataset_names(recorded):
    spec = recorded("deg", _units_json(DEG))
    with pytest.raises(ValueError, match="says the joints are in"):
        Policy(spec, norm={"*": "range_m100_100", "gripper": "range_0_100"})
    assert Policy(spec, norm=DEG).units == "lerobot"  # agreeing is fine; so is saying nothing


def test_a_norm_key_that_names_no_recorded_joint_is_refused(recorded):
    # a misspelt joint would take the "*" mode: the gripper's 50 % would be written as 50 degrees
    with pytest.raises(ValueError, match="norm= names griper, which the dataset does not record"):
        Policy(recorded("bare", None), norm={"*": "degrees", "griper": "range_0_100"})


@pytest.mark.parametrize("policy", ["replay", "checkpoint"])
def test_run_units_lerobot_keeps_the_norm_modes_the_policy_declares(recorded, policy):
    # units="lerobot" took the registry's range modes over the policy's degrees: 4.5 degrees off on four joints,
    # 85.5 on wrist_roll ([2509, 2509, 2509, 2509, 1587, 2048])
    spec = recorded("deg", _units_json(DEG))
    make = (lambda: Policy(spec)) if policy == "replay" else Degrees
    assert _goals(make(), units="lerobot") == _goals(make()) == AT_ROW


@pytest.mark.parametrize("units", ["rad", None])
def test_run_units_that_contradict_the_policy_are_refused_before_the_bus(recorded, units):
    # 45 degrees read as 45 rad drove every joint to its stop ([3072, 3072, 3072, 3072, 1024, 3072]), the hazard
    # Policy(dataset, units=) already refuses
    arm, serial = _arm()
    with pytest.raises(ValueError, match=f"declares units 'lerobot', but units={units!r} was passed"):
        arm.run(Policy(recorded("deg", _units_json(DEG))), steps=1, units=units)
    arm.close()
    assert serial.frames == []


def test_a_norm_that_leaves_the_gripper_to_the_star_mode_is_refused(recorded):
    # The real-body hint read norm={'*': 'degrees' | 'range_m100_100', ...}; norm={'*': 'degrees'} replayed the
    # gripper's 50 % as 50 degrees (+569 counts). LeRobot records an SO gripper in range_0_100 in both modes, as the
    # registry's hardware.lerobot_norm says: a joint it names may not fall to a different "*" unnamed.
    spec = recorded("bare", None)
    arm, serial = _arm()
    with pytest.raises(ValueError, match=r"norm= leaves gripper to its '\*' mode 'degrees'.*'range_0_100'"):
        arm.run(Policy(spec, norm={"*": "degrees"}), steps=1)
    arm.close()
    assert serial.frames == []  # refused before the bus
    with pytest.raises(ValueError, match=r"leaves gripper to its '\*' mode 'range_m100_100'"):
        Robot("so101").run(Policy(spec, norm={"*": "range_m100_100"}), steps=1)  # a simulated arm too
    assert _goals(Policy(spec, norm=DEG)) == AT_ROW
    assert _goals(Policy(spec, norm={"*": "degrees", "gripper": "degrees"}))[5] != AT_ROW[5]  # named: the caller's
