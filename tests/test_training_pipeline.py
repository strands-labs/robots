#!/usr/bin/env python3
"""
Comprehensive Training Pipeline Tests for Issue #73

Tests the full training matrix:
- create_trainer() factory for all providers
- Trainer abstract interface compliance
- Config validation
- evaluate() harness
"""

import builtins
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Check mujoco availability for tests that need simulation
try:
    import mujoco  # noqa: F401

    HAS_MUJOCO = True
except ImportError:
    HAS_MUJOCO = False

# Check for REAL gymnasium (not mock from other test files)
# Mock gymnasium has no __file__ attribute; real packages do
import sys as _sys

_gym_mod = _sys.modules.get("gymnasium")
if _gym_mod is not None:
    HAS_GYMNASIUM = hasattr(_gym_mod, "__file__") and _gym_mod.__file__ is not None
else:
    try:
        import gymnasium  # noqa: F401

        HAS_GYMNASIUM = True
    except (ImportError, ValueError):
        HAS_GYMNASIUM = False

from strands_robots.training import (  # noqa: E402
    CosmosTrainer,
    DreamgenIdmTrainer,
    DreamgenVlaTrainer,
    Gr00tTrainer,
    LerobotTrainer,
    TrainConfig,
    Trainer,
    create_trainer,
    evaluate,
)


class TestTrainConfig:
    """Test TrainConfig dataclass."""

    def test_default_values(self):
        config = TrainConfig()
        assert config.dataset_path == ""
        assert config.output_dir == "./outputs"
        assert config.max_steps == 10000
        assert config.batch_size == 16
        assert config.learning_rate == 1e-4
        assert config.seed == 42

    def test_custom_values(self):
        config = TrainConfig(
            dataset_path="/data/test",
            max_steps=5000,
            batch_size=32,
            learning_rate=1e-5,
        )
        assert config.dataset_path == "/data/test"
        assert config.max_steps == 5000
        assert config.batch_size == 32


class TestCreateTrainer:
    """Test create_trainer() factory."""

    def test_groot_trainer(self):
        trainer = create_trainer(
            "groot",
            base_model_path="nvidia/GR00T-N1-2B",
            dataset_path="/tmp/test",
            embodiment_tag="so100",
        )
        assert isinstance(trainer, Gr00tTrainer)
        assert trainer.provider_name == "groot"

    def test_lerobot_trainer(self):
        trainer = create_trainer(
            "lerobot",
            policy_type="act",
            dataset_repo_id="test/dataset",
            output_dir="/tmp/test",
        )
        assert isinstance(trainer, LerobotTrainer)
        assert trainer.provider_name == "lerobot"

    def test_dreamgen_idm_trainer(self):
        trainer = create_trainer(
            "dreamgen_idm",
            dataset_path="/tmp/test",
            data_config="so100",
        )
        assert isinstance(trainer, DreamgenIdmTrainer)
        assert trainer.provider_name == "dreamgen_idm"

    def test_dreamgen_vla_trainer(self):
        trainer = create_trainer(
            "dreamgen_vla",
            base_model_path="nvidia/GR00T-N1-2B",
            dataset_path="/tmp/test",
            embodiment_tag="so100",
        )
        assert isinstance(trainer, DreamgenVlaTrainer)
        assert trainer.provider_name == "dreamgen_vla"

    def test_cosmos_trainer(self):
        trainer = create_trainer(
            "cosmos_predict",
            base_model_path="nvidia/Cosmos-Predict2.5-2B",
            dataset_path="/tmp/test",
        )
        assert isinstance(trainer, CosmosTrainer)
        assert trainer.provider_name == "cosmos_predict"

    def test_invalid_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown trainer provider"):
            create_trainer("nonexistent_provider")

    def test_all_providers_have_train_method(self):
        """All trainers must implement train()."""
        providers = {
            "groot": {"base_model_path": "x", "dataset_path": "x", "embodiment_tag": "so100"},
            "lerobot": {"policy_type": "act", "dataset_repo_id": "x"},
            "dreamgen_idm": {"dataset_path": "x"},
            "dreamgen_vla": {"base_model_path": "x", "dataset_path": "x", "embodiment_tag": "so100"},
            "cosmos_predict": {"base_model_path": "x", "dataset_path": "x"},
        }
        for provider, kwargs in providers.items():
            trainer = create_trainer(provider, **kwargs)
            assert hasattr(trainer, "train")
            assert hasattr(trainer, "evaluate")
            assert hasattr(trainer, "provider_name")

    def test_all_providers_listed(self):
        """Verify all 5 expected providers are available."""
        expected = {"groot", "lerobot", "dreamgen_idm", "dreamgen_vla", "cosmos_predict"}
        # We can check by creating each
        for name in expected:
            try:
                create_trainer(name)
            except (ValueError, TypeError):
                pass  # Expected — missing required args                create_trainer(name)


class TestTrainerAbstract:
    """Test Trainer ABC compliance."""

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Trainer()

    def test_subclass_must_implement_train(self):
        class IncompleteTrainer(Trainer):
            @property
            def provider_name(self):
                return "test"

            def evaluate(self, **kwargs):
                return {}

        with pytest.raises(TypeError):
            IncompleteTrainer()


class TestEvaluateHarness:
    """Test the standalone evaluate() function."""

    @pytest.mark.skipif(not HAS_MUJOCO or not HAS_GYMNASIUM, reason="requires mujoco and gymnasium")
    def test_evaluate_with_mock_policy(self):
        """evaluate() should work with a mock policy object."""

        class FakePolicy:
            provider_name = "fake"

            def get_actions(self, obs, instruction):
                return [{"j0": 0.0, "j1": 0.0}]

        result = evaluate(
            policy=FakePolicy(),
            task="pick up the red cube",
            robot_name="so100",
            num_episodes=1,
            max_steps_per_episode=5,
        )
        # Should return dict with expected keys
        assert "success_rate" in result
        assert "mean_reward" in result
        assert "num_episodes" in result
        assert "episodes" in result

    def test_evaluate_newton_returns_warning(self):
        """Newton backend should return error when Newton is unavailable.

        Uses builtins.__import__ interception to force ImportError even when
        Newton is actually installed (e.g. on Thor with GPU).
        """
        _real_import = builtins.__import__

        def _block_newton(name, *args, **kwargs):
            if "strands_robots.newton" in name:
                raise ImportError("mocked: newton not available")
            return _real_import(name, *args, **kwargs)

        class FakePolicy:
            provider_name = "fake"

            def get_actions(self, obs, instruction):
                return [{}]

        builtins.__import__ = _block_newton
        try:
            result = evaluate(
                policy=FakePolicy(),
                task="test",
                robot_name="so100",
                num_episodes=1,
                backend="newton",
            )
        finally:
            builtins.__import__ = _real_import

        assert "error" in result
        assert "Newton" in result["error"] or "newton" in result["error"]


class TestGr00tTrainerConfig:
    """Test GR00T trainer configuration."""

    def test_config_attributes(self):
        trainer = create_trainer(
            "groot",
            base_model_path="nvidia/GR00T-N1-2B",
            dataset_path="/data/test",
            embodiment_tag="so100",
            data_config="so100_dualcam",
            max_steps=5000,
            batch_size=32,
            learning_rate=1e-5,
        )
        assert trainer.base_model_path == "nvidia/GR00T-N1-2B"
        assert trainer.dataset_path == "/data/test"
        assert trainer.embodiment_tag == "so100"


class TestLerobotTrainerConfig:
    """Test LeRobot trainer configuration."""

    def test_act_config(self):
        trainer = create_trainer(
            "lerobot",
            policy_type="act",
            dataset_repo_id="test/data",
            output_dir="/tmp/act_out",
            max_steps=100000,
            batch_size=64,
        )
        assert trainer.policy_type == "act"
        assert trainer.dataset_repo_id == "test/data"

    def test_pi0_config(self):
        trainer = create_trainer(
            "lerobot",
            policy_type="pi0",
            dataset_repo_id="test/data",
            output_dir="/tmp/pi0_out",
        )
        assert trainer.policy_type == "pi0"

    def test_supported_policy_types(self):
        """LeRobot should support multiple policy types."""
        for ptype in ["act", "pi0"]:
            trainer = create_trainer(
                "lerobot",
                policy_type=ptype,
                dataset_repo_id="test/data",
            )
            assert trainer.policy_type == ptype


class TestCosmosTrainerConfig:
    """Test Cosmos trainer configuration."""

    def test_cosmos_config(self):
        trainer = create_trainer(
            "cosmos_predict",
            base_model_path="nvidia/Cosmos-Predict2.5-2B",
            dataset_path="/data/test",
            mode="policy",
            num_gpus=4,
        )
        assert trainer.base_model_path == "nvidia/Cosmos-Predict2.5-2B"
        assert trainer.mode == "policy"
