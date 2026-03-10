#!/usr/bin/env python3
"""Tests for AgiBot World dual-arm data configurations.

Validates the 3 new AgiBot data configs for GR00T N1.6 training:
- agibot_dual_arm / agibot_dual_arm_gripper: 14-DOF arms + 2-DOF grippers (16 total)
- agibot_dual_arm_dexhand: 14-DOF arms + 12-DOF dexterous hands (26 total)
- agibot_dual_arm_full: 14-DOF arms + 2-DOF grippers + 2 head + 2 waist + 2 base (22 total)
"""

import pytest

# CROSS_PR_SKIP: Tests depend on go1 policy with full data configs
try:
    from strands_robots.policies.go1 import Go1Policy

    if not hasattr(Go1Policy, "_BUILTIN_DATA_CONFIGS"):
        raise AttributeError
except (ImportError, AttributeError):
    import pytest as _xfail_pytest

    _xfail_pytest.skip("Tests require Go1Policy with data configs from PR #8", allow_module_level=True)


class TestAgibotDualArmGripper:
    """Tests for AgibotDualArmGripperDataConfig."""

    def test_load_by_name(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_gripper")
        assert cfg is not None

    def test_alias(self):
        """agibot_dual_arm is an alias for the gripper variant."""
        from strands_robots.policies.groot.data_config import load_data_config

        cfg_alias = load_data_config("agibot_dual_arm")
        cfg_explicit = load_data_config("agibot_dual_arm_gripper")
        assert cfg_alias.video_keys == cfg_explicit.video_keys
        assert cfg_alias.state_keys == cfg_explicit.state_keys

    def test_video_keys(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_gripper")
        assert cfg.video_keys == ["video.head", "video.left_hand", "video.right_hand"]

    def test_state_keys(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_gripper")
        assert cfg.state_keys == [
            "state.left_arm",
            "state.right_arm",
            "state.left_gripper",
            "state.right_gripper",
        ]

    def test_action_keys(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_gripper")
        assert cfg.action_keys == [
            "action.left_arm",
            "action.right_arm",
            "action.left_gripper",
            "action.right_gripper",
        ]

    def test_action_indices_30_steps(self):
        """GO-1 uses 30-step action chunks (30Hz × 1s)."""
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_gripper")
        assert cfg.action_indices == list(range(30))

    def test_language_keys(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_gripper")
        assert cfg.language_keys == ["annotation.language.action_text"]

    def test_observation_indices(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_gripper")
        assert cfg.observation_indices == [0]

    def test_modality_config(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_gripper")
        mc = cfg.modality_config()
        assert "video" in mc
        assert "state" in mc
        assert "action" in mc
        assert "language" in mc
        assert mc["video"].modality_keys == ["video.head", "video.left_hand", "video.right_hand"]


class TestAgibotDualArmDexHand:
    """Tests for AgibotDualArmDexHandDataConfig."""

    def test_load_by_name(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_dexhand")
        assert cfg is not None

    def test_video_keys(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_dexhand")
        assert cfg.video_keys == ["video.head", "video.left_hand", "video.right_hand"]

    def test_state_keys_use_hand_not_gripper(self):
        """DexHand variant uses 'state.left_hand'/'state.right_hand' for 6-DOF hands."""
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_dexhand")
        assert "state.left_hand" in cfg.state_keys
        assert "state.right_hand" in cfg.state_keys
        assert "state.left_gripper" not in cfg.state_keys

    def test_action_keys_use_hand(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_dexhand")
        assert "action.left_hand" in cfg.action_keys
        assert "action.right_hand" in cfg.action_keys

    def test_action_indices_30_steps(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_dexhand")
        assert cfg.action_indices == list(range(30))


class TestAgibotDualArmFull:
    """Tests for AgibotDualArmFullDataConfig."""

    def test_load_by_name(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_full")
        assert cfg is not None

    def test_state_includes_head_and_waist(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_full")
        assert "state.head" in cfg.state_keys
        assert "state.waist" in cfg.state_keys

    def test_action_includes_base_velocity(self):
        """Full config includes mobile base velocity for navigation tasks."""
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_full")
        assert "action.base_velocity" in cfg.action_keys

    def test_action_keys_complete(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_full")
        expected = [
            "action.left_arm",
            "action.right_arm",
            "action.left_gripper",
            "action.right_gripper",
            "action.head",
            "action.waist",
            "action.base_velocity",
        ]
        assert cfg.action_keys == expected

    def test_state_keys_complete(self):
        from strands_robots.policies.groot.data_config import load_data_config

        cfg = load_data_config("agibot_dual_arm_full")
        expected = [
            "state.left_arm",
            "state.right_arm",
            "state.left_gripper",
            "state.right_gripper",
            "state.head",
            "state.waist",
        ]
        assert cfg.state_keys == expected


class TestDataConfigRegistry:
    """Test data config registry integration."""

    def test_total_config_count(self):
        from strands_robots.policies.groot.data_config import DATA_CONFIG_MAP

        # 21 original + 4 new (3 unique + 1 alias)
        assert len(DATA_CONFIG_MAP) >= 25

    def test_all_agibot_configs_in_registry(self):
        from strands_robots.policies.groot.data_config import DATA_CONFIG_MAP

        for name in [
            "agibot_genie1",
            "agibot_dual_arm",
            "agibot_dual_arm_gripper",
            "agibot_dual_arm_dexhand",
            "agibot_dual_arm_full",
        ]:
            assert name in DATA_CONFIG_MAP, f"'{name}' not in DATA_CONFIG_MAP"

    def test_configs_differ(self):
        """The three AgiBot World configs should have different state keys."""
        from strands_robots.policies.groot.data_config import load_data_config

        gripper = load_data_config("agibot_dual_arm_gripper")
        dexhand = load_data_config("agibot_dual_arm_dexhand")
        full = load_data_config("agibot_dual_arm_full")

        assert gripper.state_keys != dexhand.state_keys
        assert gripper.state_keys != full.state_keys
        assert dexhand.state_keys != full.state_keys

    def test_configs_share_video_keys(self):
        """All AgiBot World configs use the same 3-camera setup."""
        from strands_robots.policies.groot.data_config import load_data_config

        gripper = load_data_config("agibot_dual_arm_gripper")
        dexhand = load_data_config("agibot_dual_arm_dexhand")
        full = load_data_config("agibot_dual_arm_full")

        assert gripper.video_keys == dexhand.video_keys == full.video_keys
        assert gripper.video_keys == ["video.head", "video.left_hand", "video.right_hand"]

    def test_unknown_config_raises(self):
        from strands_robots.policies.groot.data_config import load_data_config

        with pytest.raises(ValueError, match="Unknown data_config"):
            load_data_config("nonexistent_config")
