"""Tests for Cosmos 3 embodiment specs."""

import pytest

from strands_robots.policies.cosmos3.embodiments import (
    Cosmos3Embodiment,
    get_embodiment,
    list_embodiments,
)


def test_known_embodiments_present():
    names = list_embodiments()
    assert {"droid", "umi", "av", "bridge"} <= set(names)


def test_droid_spec_matches_released_policy():
    e = get_embodiment("droid")
    assert e.domain_name == "droid_lerobot"
    assert e.raw_action_dim == 10
    assert e.action_chunk_size == 32
    assert e.fps == 15
    assert e.default_action_space == "joint_pos"
    # joint_pos layout = 7 joints + gripper = 8 columns
    assert e.action_layouts["joint_pos"] == [
        "joint_0",
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    ]


def test_aliases_resolve():
    assert get_embodiment("droid_lerobot").name == "droid"
    assert get_embodiment("franka").name == "droid"
    assert get_embodiment("bridge_orig_lerobot").name == "bridge"
    assert get_embodiment("autonomous_vehicle").name == "av"


def test_unknown_embodiment_raises():
    with pytest.raises(ValueError, match="Unknown Cosmos 3 embodiment"):
        get_embodiment("totally_not_a_robot")


def test_av_has_no_gripper_and_9d():
    e = get_embodiment("av")
    assert e.raw_action_dim == 9
    assert "grasp" not in e.action_layouts["midtrain"]
    assert len(e.action_layouts["midtrain"]) == 9


def test_embodiment_is_frozen():
    e = get_embodiment("droid")
    assert isinstance(e, Cosmos3Embodiment)
    with pytest.raises(Exception):
        e.fps = 99  # frozen dataclass
