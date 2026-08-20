"""Tests for Cosmos 3 embodiment specs."""

import pytest

from strands_robots.policies.cosmos3.embodiments import (
    Cosmos3Embodiment,
    get_embodiment,
    list_embodiments,
)


def test_known_embodiments_present():
    names = list_embodiments()
    assert {"droid", "umi", "av", "bridge", "openarm"} <= set(names)


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


class TestOpenArmEmbodiment:
    """The OpenArm entry (#2461): single-arm, post-training-only domain."""

    def test_spec(self):
        e = get_embodiment("openarm")
        assert e.name == "openarm"
        assert e.domain_name == "openarm_lerobot"
        # Single-arm unified action: 9D EE pose (3D translation + 6D rot) + grasp.
        assert e.raw_action_dim == 10
        assert e.normalization == "quantile"
        # Recorded OpenArm episodes carry front + wrist views (openarm_real).
        assert e.camera_keys == ["observation/image", "observation/wrist_image"]

    def test_action_dim_and_mapping_round_trip(self):
        """Every named layout matches its width; midtrain IS the raw action."""
        e = get_embodiment("openarm")
        assert e.default_action_space == "midtrain"
        assert e.default_action_space in e.action_layouts
        layout = e.action_layouts["midtrain"]
        assert len(layout) == e.raw_action_dim
        assert layout == e.raw_action_layout
        assert layout == ["tx", "ty", "tz", "r0", "r1", "r2", "r3", "r4", "r5", "grasp"]
        # eef-space with a trailing grasp column - the sim_ik decode contract.
        assert layout[-1] == "grasp"

    def test_no_fabricated_joint_pos_layout(self):
        """The RoboLab server's joint conversion is DROID-only - an openarm
        ``joint_pos`` layout would promise a server post-process that does not
        exist for this domain."""
        e = get_embodiment("openarm")
        assert "joint_pos" not in e.action_layouts

    def test_aliases_resolve(self):
        for alias in ("openarm_lerobot", "openarm_follower", "enactic_openarm", "OpenArm"):
            assert get_embodiment(alias).name == "openarm", alias

    def test_no_bundled_stats_fails_loudly_naming_the_domain(self):
        """Post-training produces the domain's own q01/q99; until then the
        stats lookup refuses by name (no silent substitute)."""
        from strands_robots.policies.cosmos3.action_decode import load_action_stats

        with pytest.raises(FileNotFoundError, match="openarm_lerobot") as excinfo:
            load_action_stats(get_embodiment("openarm").domain_name)
        # The refusal advises the safe explicit-stats route.
        assert "stats_domain='openarm_lerobot'" in str(excinfo.value)
