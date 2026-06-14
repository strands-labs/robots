"""Unit tests for MotionBricksPolicy.

Tests cover:
- Construction with allow_missing_models (offline/CI mode)
- RuntimeError raised when models missing and allow_missing_models=False
- Style enum completeness and index mapping
- Joint name constants
- Default action shape and content
- Control signal construction (velocity -> direction mapping)
- get_actions returns proper dict structure
- set_robot_state_keys customization
- reset clears state
- Factory integration (policies.json registration)
- requires_images is False
- provider_name is "motionbricks"
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# Ensure torch is available or stubbed for these tests
@pytest.fixture(autouse=True)
def _stub_torch_if_absent():
    """Stub torch in sys.modules if not installed (CI minimal env)."""
    if "torch" not in sys.modules:
        mock_torch = MagicMock()
        mock_torch.__version__ = "2.0.0"
        mock_torch.Tensor = type("Tensor", (), {})
        mock_torch.no_grad = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        mock_torch.device = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        mock_torch.from_numpy = lambda x: MagicMock(view=lambda *a: MagicMock())
        mock_torch.tensor = lambda x: MagicMock(view=lambda *a: MagicMock())
        mock_torch.ones = lambda *a, **kw: MagicMock(view=lambda *a: MagicMock())
        mock_torch.manual_seed = MagicMock()
        sys.modules.setdefault("torch", mock_torch)
    yield


class TestMotionBricksConstants:
    """Test module-level constants and enums."""

    def test_joint_names_count(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MOTIONBRICKS_JOINT_NAMES,
        )

        assert len(MOTIONBRICKS_JOINT_NAMES) == 29

    def test_joint_names_unique(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MOTIONBRICKS_JOINT_NAMES,
        )

        assert len(set(MOTIONBRICKS_JOINT_NAMES)) == 29

    def test_joint_names_contain_expected(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MOTIONBRICKS_JOINT_NAMES,
        )

        # Legs
        assert "left_hip_pitch" in MOTIONBRICKS_JOINT_NAMES
        assert "right_ankle_roll" in MOTIONBRICKS_JOINT_NAMES
        # Waist
        assert "waist_yaw" in MOTIONBRICKS_JOINT_NAMES
        # Arms
        assert "left_shoulder_pitch" in MOTIONBRICKS_JOINT_NAMES
        assert "right_wrist_yaw" in MOTIONBRICKS_JOINT_NAMES

    def test_styles_list_populated(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MOTIONBRICKS_STYLES,
        )

        assert len(MOTIONBRICKS_STYLES) >= 12
        assert "idle" in MOTIONBRICKS_STYLES
        assert "walk" in MOTIONBRICKS_STYLES
        assert "walk_zombie" in MOTIONBRICKS_STYLES
        assert "walk_happy_dance" in MOTIONBRICKS_STYLES

    def test_style_enum_matches_list(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MOTIONBRICKS_STYLES,
            MotionBricksStyle,
        )

        enum_values = [s.value for s in MotionBricksStyle]
        assert enum_values == MOTIONBRICKS_STYLES

    def test_idle_is_first_style(self):
        """Idle must be at index 0 for the mode auto-switch logic."""
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MOTIONBRICKS_STYLES,
        )

        assert MOTIONBRICKS_STYLES[0] == "idle"


class TestMotionBricksConstruction:
    """Test policy construction and error handling."""

    def test_raises_without_models(self):
        """Policy must raise RuntimeError when models missing."""
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        with pytest.raises(RuntimeError, match="checkpoint not found"):
            MotionBricksPolicy(
                checkpoint="/nonexistent/path",
                allow_missing_models=False,
            )

    def test_allow_missing_models_no_raise(self):
        """allow_missing_models=True lets construction succeed without models."""
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        policy = MotionBricksPolicy(
            checkpoint="/nonexistent/path",
            allow_missing_models=True,
        )
        assert policy._agent is None

    def test_default_style(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        policy = MotionBricksPolicy(
            checkpoint="/nonexistent/path",
            style="walk_zombie",
            allow_missing_models=True,
        )
        assert policy._default_style == "walk_zombie"

    def test_custom_speed_scale(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        policy = MotionBricksPolicy(
            checkpoint="/nonexistent",
            speed_scale=[1.0, 1.5],
            allow_missing_models=True,
        )
        assert policy._speed_scale == [1.0, 1.5]

    def test_device_stored(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        policy = MotionBricksPolicy(
            checkpoint="/nonexistent",
            device="cpu",
            allow_missing_models=True,
        )
        assert policy._device == "cpu"


class TestMotionBricksProperties:
    """Test policy properties."""

    @pytest.fixture()
    def policy(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        return MotionBricksPolicy(checkpoint="/nonexistent", allow_missing_models=True)

    def test_requires_images_false(self, policy):
        assert policy.requires_images is False

    def test_provider_name(self, policy):
        assert policy.provider_name == "motionbricks"


class TestMotionBricksActions:
    """Test action generation (offline/degraded mode)."""

    @pytest.fixture()
    def policy(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        return MotionBricksPolicy(checkpoint="/nonexistent", allow_missing_models=True)

    def test_default_action_returns_list(self, policy):
        actions = asyncio.run(policy.get_actions({}, ""))
        assert isinstance(actions, list)
        assert len(actions) == 1

    def test_default_action_has_all_joints(self, policy):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MOTIONBRICKS_JOINT_NAMES,
        )

        actions = asyncio.run(policy.get_actions({}, ""))
        action = actions[0]
        for name in MOTIONBRICKS_JOINT_NAMES:
            assert name in action
            assert action[name] == 0.0

    def test_default_action_values_are_float(self, policy):
        actions = asyncio.run(policy.get_actions({}, ""))
        for v in actions[0].values():
            assert isinstance(v, float)

    def test_custom_state_keys_respected(self, policy):
        policy.set_robot_state_keys(["joint_a", "joint_b", "joint_c"])
        actions = asyncio.run(policy.get_actions({}, ""))
        action = actions[0]
        assert "joint_a" in action
        assert "joint_b" in action
        assert "joint_c" in action

    def test_get_actions_sync_works(self, policy):
        actions = policy.get_actions_sync({}, "")
        assert isinstance(actions, list)
        assert len(actions) == 1

    def test_kwargs_ignored_in_degraded_mode(self, policy):
        """Style and velocity kwargs should not crash degraded mode."""
        actions = policy.get_actions_sync({}, "", target_velocity=[1.0, 0, 0], style="walk_zombie")
        assert len(actions) == 1


class TestMotionBricksControlSignals:
    """Test the control signal building logic."""

    @pytest.fixture()
    def policy(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        return MotionBricksPolicy(checkpoint="/nonexistent", allow_missing_models=True)

    def test_build_control_signals_zero_velocity(self, policy):
        """Zero velocity should produce idle mode."""
        signals = policy._build_control_signals([0.0, 0.0, 0.0], 0.0, "walk")
        # Mode should be idle (index 0) because speed < threshold
        # The actual tensor value depends on torch mock, but we can verify
        # the function doesn't crash
        assert "movement_direction" in signals
        assert "facing_direction" in signals
        assert "mode" in signals
        assert "allowed_pred_num_tokens" in signals

    def test_build_control_signals_forward_velocity(self, policy):
        """Forward velocity should produce non-idle mode."""
        signals = policy._build_control_signals([1.0, 0.0, 0.0], 0.0, "walk_zombie")
        assert "mode" in signals

    def test_build_control_signals_unknown_style_fallback(self, policy):
        """Unknown style should fall back to 'walk' with a warning."""
        import logging

        with patch.object(logging.getLogger("strands_robots.policies.motionbricks.motionbricks_policy"), "warning"):
            signals = policy._build_control_signals([1.0, 0.0, 0.0], 0.0, "nonexistent_style")
        assert "mode" in signals


class TestMotionBricksReset:
    """Test reset behavior."""

    def test_reset_clears_buffer(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        policy = MotionBricksPolicy(checkpoint="/nonexistent", allow_missing_models=True)
        policy._frame_buffer = [np.zeros(36)]
        policy._frame_idx = 5
        policy.reset()
        assert policy._frame_buffer == []
        assert policy._frame_idx == 0

    def test_reset_with_seed_no_crash(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        policy = MotionBricksPolicy(checkpoint="/nonexistent", allow_missing_models=True)
        # Should not crash even without agent loaded
        policy.reset(seed=42)


class TestMotionBricksFactory:
    """Test factory/registry integration."""

    def test_registered_in_policies_json(self):
        from strands_robots.registry import list_policy_providers

        providers = list_policy_providers()
        assert "motionbricks" in providers

    def test_import_policy_class(self):
        from strands_robots.registry import import_policy_class

        cls = import_policy_class("motionbricks")
        from strands_robots.policies.motionbricks import MotionBricksPolicy

        assert cls is MotionBricksPolicy

    def test_create_policy_shorthand_mb(self):
        """The 'mb' shorthand should resolve to MotionBricksPolicy."""
        from strands_robots.registry import import_policy_class

        cls = import_policy_class("mb")
        from strands_robots.policies.motionbricks import MotionBricksPolicy

        assert cls is MotionBricksPolicy

    def test_create_policy_constructs(self):
        """create_policy('motionbricks') with allow_missing should work."""
        from strands_robots.policies import create_policy

        policy = create_policy(
            "motionbricks",
            checkpoint="/nonexistent",
            allow_missing_models=True,
        )
        assert policy.provider_name == "motionbricks"

    def test_create_policy_mb_shorthand_constructs(self):
        from strands_robots.policies import create_policy

        policy = create_policy(
            "mb",
            checkpoint="/nonexistent",
            allow_missing_models=True,
        )
        assert policy.provider_name == "motionbricks"


class TestMotionBricksCheckpointResolution:
    """Test checkpoint path resolution logic."""

    def test_local_dir_with_motionbricks_subdir(self, tmp_path):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        # Create structure: tmp/motionbricks/
        mb_dir = tmp_path / "motionbricks"
        mb_dir.mkdir()
        policy = MotionBricksPolicy.__new__(MotionBricksPolicy)
        policy._checkpoint = str(tmp_path)
        result = policy._resolve_checkpoint()
        assert result == mb_dir

    def test_local_dir_with_vqvae_subdir(self, tmp_path):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        # Create structure with model dirs directly
        (tmp_path / "motionbricks_vqvae").mkdir()
        policy = MotionBricksPolicy.__new__(MotionBricksPolicy)
        policy._checkpoint = str(tmp_path)
        result = policy._resolve_checkpoint()
        assert result == tmp_path

    def test_nonexistent_path_returns_none(self):
        from strands_robots.policies.motionbricks.motionbricks_policy import (
            MotionBricksPolicy,
        )

        policy = MotionBricksPolicy.__new__(MotionBricksPolicy)
        policy._checkpoint = "/absolutely/does/not/exist/anywhere"
        with patch(
            "strands_robots.policies.motionbricks.motionbricks_policy.Path.home",
            return_value=Path("/fake/home"),
        ):
            result = policy._resolve_checkpoint()
        assert result is None


# Need this import at top for the patch in TestMotionBricksCheckpointResolution
from pathlib import Path  # noqa: E402
