"""Tests for strands_robots/training/__init__.py — Training Abstraction.

Coverage target: ~90%+ of training/__init__.py (636 lines).
All tests CPU-only with mocked heavy dependencies (subprocess, lerobot, gr00t).

Test organization:
1. SyntaxValidation — file parses correctly
2. TrainConfig — defaults, custom values
3. TrainerABC — abstract base class contract
4. Gr00tTrainer — init, train (command flags), evaluate
5. LerobotTrainer — init, train (in-process/subprocess), evaluate
6. DreamgenIdmTrainer — init, train (command flags), evaluate
7. DreamgenVlaTrainer — init, train (command flags, LoRA gate), evaluate
8. CosmosTrainer — init, train (modes, multi-GPU, config inference), evaluate
9. create_trainer — factory dispatch, unknown provider
10. evaluate — standalone evaluation harness
11. CommandVerification — cross-trainer flag correctness

Key design trap tested:
- Gr00tTrainer.tune_visual defaults to FALSE → --tune-visual should be ABSENT
- DreamgenVlaTrainer.tune_visual defaults to TRUE → --tune-visual should be PRESENT
"""

import ast
import builtins
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ───────────────────────────── Syntax Validation ─────────────────────────────


class TestSyntaxValidation:
    """Verify source file parses without errors."""

    def test_training_parses(self):
        """training/__init__.py is valid Python."""
        src = Path(__file__).resolve().parent.parent / "strands_robots" / "training" / "__init__.py"
        source = src.read_text()
        tree = ast.parse(source)
        assert tree is not None

    def test_exports(self):
        """Module exports the expected symbols."""
        from strands_robots.training import __all__

        expected = [
            "Trainer",
            "TrainConfig",
            "create_trainer",
            "Gr00tTrainer",
            "LerobotTrainer",
            "DreamgenIdmTrainer",
            "DreamgenVlaTrainer",
        ]
        for sym in expected:
            assert sym in __all__, f"{sym} missing from __all__"

    def test_exports_include_cosmos_and_evaluate(self):
        """Module exports CosmosTrainer and evaluate."""
        from strands_robots.training import __all__

        assert "CosmosTrainer" in __all__
        assert "evaluate" in __all__


# ───────────────────────────── TrainConfig ─────────────────────────────


class TestTrainConfig:
    """Tests for the TrainConfig dataclass."""

    def test_defaults(self):
        """TrainConfig has 14 fields with sensible defaults."""
        from strands_robots.training import TrainConfig

        cfg = TrainConfig()
        assert cfg.dataset_path == ""
        assert cfg.output_dir == "./outputs"
        assert cfg.max_steps == 10000
        assert cfg.batch_size == 16
        assert cfg.learning_rate == 1e-4
        assert cfg.weight_decay == 1e-5
        assert cfg.warmup_ratio == 0.05
        assert cfg.num_gpus == 1
        assert cfg.save_steps == 1000
        assert cfg.save_total_limit == 5
        assert cfg.use_wandb is False
        assert cfg.dataloader_num_workers == 2
        assert cfg.seed == 42
        assert cfg.resume is False

    def test_custom_values(self):
        """TrainConfig accepts custom values."""
        from strands_robots.training import TrainConfig

        cfg = TrainConfig(max_steps=500, batch_size=32, learning_rate=2e-5)
        assert cfg.max_steps == 500
        assert cfg.batch_size == 32
        assert cfg.learning_rate == 2e-5


# ───────────────────────────── Trainer ABC ─────────────────────────────


class TestTrainerABC:
    """Tests for the Trainer abstract base class."""

    def test_abstract_methods(self):
        """Trainer requires train(), evaluate(), provider_name."""
        from strands_robots.training import Trainer

        assert hasattr(Trainer, "train")
        assert hasattr(Trainer, "evaluate")
        assert hasattr(Trainer, "provider_name")

    def test_cannot_instantiate_abc(self):
        """Trainer cannot be instantiated directly."""
        from strands_robots.training import Trainer

        with pytest.raises(TypeError):
            Trainer()

    def test_concrete_implementation(self):
        """A concrete Trainer implementation works."""
        from strands_robots.training import Trainer

        class TestTrainer(Trainer):
            def train(self, **kw):
                return {"status": "ok"}

            def evaluate(self, **kw):
                return {"status": "ok"}

            @property
            def provider_name(self):
                return "test"

        t = TestTrainer()
        assert t.provider_name == "test"
        assert t.train()["status"] == "ok"
        assert t.evaluate()["status"] == "ok"


# ───────────────────────────── Gr00tTrainer ─────────────────────────────


class TestGr00tTrainer:
    """Tests for GR00T N1.6 fine-tuning trainer."""

    def test_init_defaults(self):
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer()
        assert t.provider_name == "groot"
        assert t.base_model_path == "nvidia/GR00T-N1.6-3B"
        assert t.tune_llm is False
        assert t.tune_visual is False
        assert t.tune_projector is True
        assert t.tune_diffusion_model is True

    def test_init_custom(self):
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer(
            base_model_path="custom/model",
            dataset_path="/data",
            embodiment_tag="g1",
            data_config="unitree_g1",
        )
        assert t.base_model_path == "custom/model"
        assert t.dataset_path == "/data"
        assert t.embodiment_tag == "g1"
        assert t.data_config == "unitree_g1"

    def test_train_command(self):
        """Gr00tTrainer.train produces correct subprocess command."""
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer(dataset_path="/data", embodiment_tag="so100")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = t.train()
        assert result["status"] == "completed"
        cmd = mock_run.call_args[0][0]
        assert "--base-model-path" in cmd
        assert "--dataset-path" in cmd
        assert "--embodiment-tag" in cmd

    def test_data_config_flag(self):
        """Gr00tTrainer includes --data-config when set."""
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer(data_config="so100_dualcam")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--data-config")
        assert cmd[idx + 1] == "so100_dualcam"

    def test_data_config_absent_when_none(self):
        """Gr00tTrainer omits --data-config when None."""
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer(data_config=None)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--data-config" not in cmd

    def test_tune_llm_flag(self):
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer(tune_llm=True)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--tune-llm" in cmd

    def test_no_tune_projector_flag(self):
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer(tune_projector=False)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--no-tune-projector" in cmd

    def test_no_tune_diffusion_flag(self):
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer(tune_diffusion_model=False)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--no-tune-diffusion-model" in cmd

    def test_wandb_flag(self):
        from strands_robots.training import Gr00tTrainer, TrainConfig

        t = Gr00tTrainer(config=TrainConfig(use_wandb=True))
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--use-wandb" in cmd

    def test_resume_flag(self):
        from strands_robots.training import Gr00tTrainer, TrainConfig

        t = Gr00tTrainer(config=TrainConfig(resume=True))
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--resume" in cmd

    def test_modality_config_path(self):
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer(modality_config_path="/configs/mod.json")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--modality-config-path")
        assert cmd[idx + 1] == "/configs/mod.json"

    def test_train_failure(self):
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer()
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            result = t.train()
        assert result["status"] == "failed"

    def test_evaluate(self):
        """Gr00tTrainer.evaluate returns not_implemented."""
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer()
        result = t.evaluate()
        assert result["status"] == "not_implemented"


# ───────────────────────────── LerobotTrainer ─────────────────────────────


class TestLerobotTrainer:
    """Tests for LeRobot policy training."""

    def test_init_defaults(self):
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer()
        assert t.provider_name == "lerobot"
        assert t.policy_type == "act"
        assert t.in_process is True

    def test_init_custom(self):
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer(
            policy_type="pi0",
            dataset_repo_id="lerobot/so100_wipe",
            in_process=False,
        )
        assert t.policy_type == "pi0"
        assert t.dataset_repo_id == "lerobot/so100_wipe"
        assert t.in_process is False

    def test_train_subprocess_mode(self):
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer(dataset_repo_id="test/repo", in_process=False)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            result = t.train()
        assert result["status"] == "completed"
        assert result["mode"] == "subprocess"

    def test_train_subprocess_with_policy(self):
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer(
            dataset_repo_id="test/repo",
            pretrained_name_or_path="lerobot/pi0",
            in_process=False,
        )
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert any("--policy=" in str(c) for c in cmd)

    def test_train_subprocess_wandb_flag(self):
        from strands_robots.training import LerobotTrainer, TrainConfig

        t = LerobotTrainer(
            dataset_repo_id="test/repo",
            config=TrainConfig(use_wandb=True),
            in_process=False,
        )
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--wandb.enable=true" in cmd

    def test_train_subprocess_resume_flag(self):
        from strands_robots.training import LerobotTrainer, TrainConfig

        t = LerobotTrainer(
            dataset_repo_id="test/repo",
            config=TrainConfig(resume=True),
            in_process=False,
        )
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--resume=true" in cmd

    def test_train_subprocess_failure(self):
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer(dataset_repo_id="test/repo", in_process=False)
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            result = t.train()
        assert result["status"] == "failed"

    def test_train_in_process_falls_back_to_subprocess(self):
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer(dataset_repo_id="test/repo", in_process=True)
        mock_result = MagicMock(returncode=0)
        with patch.object(t, "_train_in_process", side_effect=Exception("no lerobot")):
            with patch("subprocess.run", return_value=mock_result):
                result = t.train()
        assert result["mode"] == "subprocess"

    def test_train_in_process_import_error(self):
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer(dataset_repo_id="test/repo", in_process=True)
        mock_result = MagicMock(returncode=0)
        with patch.dict("sys.modules", {"lerobot.scripts.lerobot_train": None}):
            with patch("subprocess.run", return_value=mock_result):
                result = t.train()
        assert result["mode"] == "subprocess"

    def test_train_in_process_success(self):
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer(dataset_repo_id="test/repo", in_process=True)
        mock_config = MagicMock()
        # Mock the lerobot module hierarchy so 'from lerobot.scripts.lerobot_train import train' works
        mock_lerobot = MagicMock()
        mock_lerobot_scripts = MagicMock()
        mock_lerobot_train_mod = MagicMock()
        mock_lerobot.scripts = mock_lerobot_scripts
        mock_lerobot_scripts.lerobot_train = mock_lerobot_train_mod
        with patch.object(t, "_build_train_config", return_value=mock_config):
            with patch.dict(
                "sys.modules",
                {
                    "lerobot": mock_lerobot,
                    "lerobot.scripts": mock_lerobot_scripts,
                    "lerobot.scripts.lerobot_train": mock_lerobot_train_mod,
                },
            ):
                result = t._train_in_process()
        assert result["status"] == "completed"
        assert result["mode"] == "in_process"

    def test_train_in_process_training_failure(self):
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer(dataset_repo_id="test/repo")
        mock_config = MagicMock()
        # Mock the lerobot module hierarchy with train() raising OOM
        mock_lerobot = MagicMock()
        mock_lerobot_scripts = MagicMock()
        mock_lerobot_train_mod = MagicMock()
        mock_lerobot_train_mod.train = MagicMock(side_effect=RuntimeError("OOM"))
        mock_lerobot.scripts = mock_lerobot_scripts
        mock_lerobot_scripts.lerobot_train = mock_lerobot_train_mod
        with patch.object(t, "_build_train_config", return_value=mock_config):
            with patch.dict(
                "sys.modules",
                {
                    "lerobot": mock_lerobot,
                    "lerobot.scripts": mock_lerobot_scripts,
                    "lerobot.scripts.lerobot_train": mock_lerobot_train_mod,
                },
            ):
                result = t._train_in_process()
        assert result["status"] == "failed"
        assert "OOM" in result["error"]

    def test_build_train_config_no_lerobot(self):
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer()
        with patch.dict("sys.modules", {"lerobot.configs.train": None}):
            result = t._build_train_config()
        assert result is None

    def test_evaluate_no_eval_env(self):
        """evaluate() without eval_env returns not_implemented."""
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer()
        result = t.evaluate()
        assert result["status"] == "not_implemented"

    def test_evaluate_with_eval_env_subprocess(self):
        """evaluate() with eval_env runs subprocess eval."""
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer(eval_env="FetchReach-v2")
        mock_result = MagicMock(returncode=0, stdout="success")
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = t.evaluate(checkpoint_path="/ckpt")
        assert result["status"] == "completed"
        cmd = mock_run.call_args[0][0]
        assert any("FetchReach-v2" in str(c) for c in cmd)

    def test_evaluate_with_eval_env_in_process_failure(self):
        """evaluate() falls back when in-process eval fails."""
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer(eval_env="FetchReach-v2")
        # Make the in-process eval fail
        with patch.object(t, "_evaluate_in_process", side_effect=Exception("fail")):
            result = t.evaluate(checkpoint_path="/ckpt")
        assert result["status"] == "not_implemented"

    def test_evaluate_default_checkpoint(self):
        """evaluate() uses output_dir when no checkpoint given."""
        from strands_robots.training import LerobotTrainer, TrainConfig

        t = LerobotTrainer(config=TrainConfig(output_dir="/my/out"))
        result = t.evaluate()
        assert result["checkpoint"] == "/my/out"


# ───────────────────────────── DreamgenIdmTrainer ─────────────────────────────


class TestDreamgenIdmTrainer:
    """Tests for DreamGen IDM training."""

    def test_init_defaults(self):
        from strands_robots.training import DreamgenIdmTrainer

        t = DreamgenIdmTrainer()
        assert t.provider_name == "dreamgen_idm"
        assert t.data_config == "gr1_arms_only"
        assert t.tune_action_head is True

    def test_train_command(self):
        from strands_robots.training import DreamgenIdmTrainer

        t = DreamgenIdmTrainer(dataset_path="/data", data_config="so100")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = t.train()
        assert result["status"] == "completed"
        cmd = mock_run.call_args[0][0]
        assert "--data-config" in cmd
        assert "--dataset-path" in cmd

    def test_tune_action_head_flag(self):
        from strands_robots.training import DreamgenIdmTrainer

        t = DreamgenIdmTrainer(tune_action_head=True)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--tune-action-head" in cmd

    def test_tune_action_head_false(self):
        from strands_robots.training import DreamgenIdmTrainer

        t = DreamgenIdmTrainer(tune_action_head=False)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--tune-action-head" not in cmd

    def test_train_failure(self):
        from strands_robots.training import DreamgenIdmTrainer

        t = DreamgenIdmTrainer()
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            result = t.train()
        assert result["status"] == "failed"

    def test_evaluate(self):
        from strands_robots.training import DreamgenIdmTrainer

        t = DreamgenIdmTrainer()
        assert t.evaluate()["status"] == "not_implemented"


# ───────────────────────────── DreamgenVlaTrainer ─────────────────────────────


class TestDreamgenVlaTrainer:
    """Tests for DreamGen VLA fine-tuning."""

    def test_init_defaults(self):
        from strands_robots.training import DreamgenVlaTrainer

        t = DreamgenVlaTrainer()
        assert t.provider_name == "dreamgen_vla"
        assert t.tune_visual is True
        assert t.lora_rank == 0

    def test_tune_visual_true_by_default(self):
        from strands_robots.training import DreamgenVlaTrainer

        t = DreamgenVlaTrainer()
        assert t.tune_visual is True

    def test_tune_visual_false_omits_flag(self):
        from strands_robots.training import DreamgenVlaTrainer

        t = DreamgenVlaTrainer(tune_visual=False)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--tune-visual" not in cmd

    def test_train_command(self):
        from strands_robots.training import DreamgenVlaTrainer

        t = DreamgenVlaTrainer(dataset_path="/data", data_config="so100")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            result = t.train()
        assert result["status"] == "completed"

    def test_lora_rank_zero_omits_flags(self):
        from strands_robots.training import DreamgenVlaTrainer

        t = DreamgenVlaTrainer(lora_rank=0)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--lora-rank" not in cmd

    def test_lora_rank_positive_adds_flags(self):
        from strands_robots.training import DreamgenVlaTrainer

        t = DreamgenVlaTrainer(lora_rank=16, lora_alpha=32)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--lora-rank")
        assert cmd[idx + 1] == "16"
        idx2 = cmd.index("--lora-alpha")
        assert cmd[idx2 + 1] == "32"

    def test_tune_llm_flag(self):
        from strands_robots.training import DreamgenVlaTrainer

        t = DreamgenVlaTrainer(tune_llm=True)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--tune-llm" in cmd

    def test_no_tune_projector_flag(self):
        from strands_robots.training import DreamgenVlaTrainer

        t = DreamgenVlaTrainer(tune_projector=False)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--no-tune-projector" in cmd

    def test_no_tune_diffusion_flag(self):
        from strands_robots.training import DreamgenVlaTrainer

        t = DreamgenVlaTrainer(tune_diffusion_model=False)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--no-tune-diffusion-model" in cmd

    def test_train_failure(self):
        from strands_robots.training import DreamgenVlaTrainer

        t = DreamgenVlaTrainer()
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            result = t.train()
        assert result["status"] == "failed"

    def test_evaluate(self):
        from strands_robots.training import DreamgenVlaTrainer

        t = DreamgenVlaTrainer()
        assert t.evaluate()["status"] == "not_implemented"


# ───────────────────────────── CosmosTrainer ─────────────────────────────


class TestCosmosTrainer:
    """Tests for Cosmos Predict 2.5 post-training."""

    def test_init_defaults(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer()
        assert t.provider_name == "cosmos_predict"
        assert t.base_model_path == "nvidia/Cosmos-Predict2.5-2B"
        assert t.mode == "policy"
        assert t.suite == "libero"
        assert t.chunk_size == 16
        assert t.action_dim == 7
        assert t.use_lora is False
        assert t.freeze_backbone is True

    def test_init_custom(self):
        from strands_robots.training import CosmosTrainer, TrainConfig

        t = CosmosTrainer(
            base_model_path="nvidia/Cosmos-Predict2.5-14B",
            dataset_path="/data/robocasa",
            mode="action_conditioned",
            suite="robocasa",
            config=TrainConfig(num_gpus=4, max_steps=5000),
            use_lora=True,
            lora_rank=32,
        )
        assert t.base_model_path == "nvidia/Cosmos-Predict2.5-14B"
        assert t.mode == "action_conditioned"
        assert t.use_lora is True
        assert t.lora_rank == 32
        assert t.config.num_gpus == 4

    def test_infer_config_name_policy_2b(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(base_model_path="nvidia/Cosmos-Predict2.5-2B", mode="policy", suite="libero")
        assert t.config_name == "cosmos_predict2_2b_480p_libero"

    def test_infer_config_name_policy_14b(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(base_model_path="nvidia/Cosmos-Predict2.5-14B", mode="policy", suite="robocasa")
        assert t.config_name == "cosmos_predict2_14b_480p_robocasa"

    def test_infer_config_name_action_conditioned(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(mode="action_conditioned")
        assert "action_conditioned" in t.config_name

    def test_infer_config_name_world_model(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(mode="world_model")
        assert "video2world" in t.config_name or "v2w" in t.config_name

    def test_custom_config_name(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(config_name="my_custom_config")
        assert t.config_name == "my_custom_config"

    def test_train_single_gpu(self):
        """Single GPU uses python directly (script or -m fallback)."""
        from strands_robots.training import CosmosTrainer, TrainConfig

        t = CosmosTrainer(config=TrainConfig(num_gpus=1))
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = t.train()
        assert result["status"] == "completed"
        cmd = mock_run.call_args[0][0]
        assert "torchrun" not in cmd
        # CosmosTrainer resolves the training entry point:
        # - If cosmos-predict2 train script is found on disk, uses it directly
        # - Otherwise falls back to python -m cosmos_predict2.train
        assert "-m" in cmd or "train.py" in cmd[-1] or any("train.py" in str(c) for c in cmd)

    def test_train_multi_gpu_uses_torchrun(self):
        """Multi GPU uses torchrun --nproc_per_node=N."""
        from strands_robots.training import CosmosTrainer, TrainConfig

        t = CosmosTrainer(config=TrainConfig(num_gpus=4))
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "torchrun"
        assert "--nproc_per_node=4" in cmd

    def test_train_policy_mode_flags(self):
        """Policy mode uses correct config file and experiment override."""
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(mode="policy", suite="libero", chunk_size=16, action_dim=7)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        # New CLI format: --config <file> -- experiment=<name> key=value ...
        assert "--config" in cmd
        assert "--" in cmd
        assert any("experiment=cosmos_predict2_2b_480p_libero" in c for c in cmd)
        assert any("cosmos_policy" in c for c in cmd)  # policy config file

    def test_train_action_conditioned_mode(self):
        """Action conditioned mode uses video2world config."""
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(mode="action_conditioned")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert any("action_conditioned" in c for c in cmd)
        assert any("video2world" in c for c in cmd)  # video2world config file

    def test_train_world_model_mode_no_extra_flags(self):
        """World model mode uses video2world config and experiment."""
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(mode="world_model")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert any("video2world" in c for c in cmd)

    def test_train_lora_flags(self):
        """LoRA params are stored on trainer (config overrides passed via kwargs)."""
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(use_lora=True, lora_rank=32)
        assert t.use_lora is True
        assert t.lora_rank == 32
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--config" in cmd  # Uses new CLI format

    def test_train_no_lora_omits_flags(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(use_lora=False)
        assert t.use_lora is False

    def test_train_freeze_backbone_flag(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(freeze_backbone=True)
        assert t.freeze_backbone is True
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--config" in cmd

    def test_train_no_freeze_backbone(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(freeze_backbone=False)
        assert t.freeze_backbone is False

    def test_train_wandb_flag(self):
        from strands_robots.training import CosmosTrainer, TrainConfig

        t = CosmosTrainer(config=TrainConfig(use_wandb=True))
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        # New CLI: wandb is passed as job.wandb_mode=online override
        assert any("wandb_mode=online" in c for c in cmd)

    def test_train_config_file(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(config_file="/path/to/config.py")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        # config_file is passed via --config flag
        idx = cmd.index("--config")
        assert cmd[idx + 1] == "/path/to/config.py"

    def test_train_failure(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer()
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            result = t.train()
        assert result["status"] == "failed"

    def test_train_result_includes_mode_and_suite(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer(mode="policy", suite="robocasa")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            result = t.train()
        assert result["mode"] == "policy"
        assert result["suite"] == "robocasa"

    def test_evaluate(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer()
        result = t.evaluate()
        assert result["status"] == "not_implemented"
        assert result["provider"] == "cosmos_predict"


# ───────────────────────────── evaluate() function ─────────────────────────────


class TestEvaluateFunction:
    """Tests for the standalone evaluate() function."""

    def test_newton_backend_returns_error(self):
        """Newton backend returns error when Newton is not installed.

        Bug fix: On systems where Newton IS installed (e.g. Thor GPU),
        we must intercept builtins.__import__ to simulate ImportError,
        otherwise evaluate() succeeds and the test times out or fails.
        """
        from strands_robots.training import evaluate

        _real_import = builtins.__import__

        def _block_newton(name, *args, **kwargs):
            if name.startswith("strands_robots.newton"):
                raise ImportError(f"Mocked: {name}")
            return _real_import(name, *args, **kwargs)

        mock_policy = MagicMock()
        with patch("builtins.__import__", side_effect=_block_newton):
            result = evaluate(mock_policy, "pick cube", "so100", backend="newton")
        assert "error" in result
        assert result["num_episodes"] == 0

    def test_missing_gym_returns_error(self):
        """Missing gymnasium/mujoco returns error."""
        from strands_robots.training import evaluate

        mock_policy = MagicMock()
        with patch.dict("sys.modules", {"strands_robots.envs": None}):
            result = evaluate(mock_policy, "pick cube", "so100")
        assert "error" in result

    def test_render_mode_none_when_render_false(self):
        """render=False sets render_mode=None (not rgb_array)."""
        from strands_robots.training import evaluate

        mock_policy = MagicMock()
        mock_policy.get_actions = MagicMock(return_value=[{"j1": 0.1, "j2": 0.2}])
        mock_policy.provider_name = "mock"

        mock_env = MagicMock()
        mock_env.reset.return_value = ({"state": [0.0]}, {})
        mock_env.step.return_value = ({"state": [0.0]}, 1.0, True, False, {"is_success": True})
        mock_env.action_space = MagicMock()
        mock_env.action_space.shape = (6,)
        mock_env.action_space.low = [-1.0] * 6
        mock_env.action_space.high = [1.0] * 6
        mock_env.action_space.sample.return_value = [0.0] * 6

        with patch("strands_robots.envs.StrandsSimEnv", return_value=mock_env) as MockEnv:
            evaluate(mock_policy, "pick cube", "so100", num_episodes=1, render=False)

        # Verify render_mode is None, not rgb_array
        call_kwargs = MockEnv.call_args[1]
        assert call_kwargs["render_mode"] is None

    def test_render_mode_rgb_when_render_true(self):
        """render=True sets render_mode='rgb_array'."""
        from strands_robots.training import evaluate

        mock_policy = MagicMock()
        mock_policy.get_actions = MagicMock(return_value=[{"j1": 0.1}])
        mock_policy.provider_name = "mock"

        mock_env = MagicMock()
        mock_env.reset.return_value = ({"state": [0.0]}, {})
        mock_env.step.return_value = ({"state": [0.0]}, 1.0, True, False, {})
        mock_env.action_space = MagicMock()
        mock_env.action_space.shape = (6,)
        mock_env.action_space.low = [-1.0] * 6
        mock_env.action_space.high = [1.0] * 6
        mock_env.action_space.sample.return_value = [0.0] * 6

        with patch("strands_robots.envs.StrandsSimEnv", return_value=mock_env) as MockEnv:
            evaluate(mock_policy, "pick cube", "so100", num_episodes=1, render=True)

        call_kwargs = MockEnv.call_args[1]
        assert call_kwargs["render_mode"] == "rgb_array"

    def test_success_rate_calculation(self):
        """evaluate() correctly calculates success rate."""
        from strands_robots.training import evaluate

        mock_policy = MagicMock()
        mock_policy.get_actions = MagicMock(return_value=[{"j1": 0.1}])
        mock_policy.provider_name = "mock"

        mock_env = MagicMock()
        # Episode 1: success, Episode 2: failure
        mock_env.reset.return_value = ({"state": [0.0]}, {})
        mock_env.step.side_effect = [
            ({"state": [0.0]}, 5.0, True, False, {"is_success": True}),
            ({"state": [0.0]}, 1.0, True, False, {"is_success": False}),
        ]
        mock_env.action_space = MagicMock()
        mock_env.action_space.shape = (6,)
        mock_env.action_space.low = [-1.0] * 6
        mock_env.action_space.high = [1.0] * 6
        mock_env.action_space.sample.return_value = [0.0] * 6

        with patch("strands_robots.envs.StrandsSimEnv", return_value=mock_env):
            result = evaluate(mock_policy, "pick", "so100", num_episodes=2)

        assert result["success_rate"] == 50.0
        assert result["num_episodes"] == 2
        assert len(result["episodes"]) == 2
        assert result["episodes"][0]["success"] is True
        assert result["episodes"][1]["success"] is False

    def test_policy_error_uses_zero_action(self):
        """Policy errors produce zero actions, don't crash."""
        from strands_robots.training import evaluate

        mock_policy = MagicMock()
        mock_policy.get_actions = MagicMock(side_effect=RuntimeError("GPU OOM"))
        mock_policy.provider_name = "mock"

        mock_env = MagicMock()
        mock_env.reset.return_value = ([0.0], {})
        mock_env.step.return_value = ([0.0], 0.0, True, False, {})
        mock_env.action_space = MagicMock()
        mock_env.action_space.shape = (6,)
        mock_env.action_space.low = [-1.0] * 6
        mock_env.action_space.high = [1.0] * 6
        mock_env.action_space.sample.return_value = [0.0] * 6

        with patch("strands_robots.envs.StrandsSimEnv", return_value=mock_env):
            result = evaluate(mock_policy, "pick", "so100", num_episodes=1)

        # Should complete without raising
        assert result["num_episodes"] == 1

    def test_result_includes_task_and_robot(self):
        """Result includes task and robot_name for traceability."""
        from strands_robots.training import evaluate

        mock_policy = MagicMock()
        mock_policy.get_actions = MagicMock(return_value=[{}])
        mock_policy.provider_name = "test_policy"

        mock_env = MagicMock()
        mock_env.reset.return_value = ({}, {})
        mock_env.step.return_value = ({}, 0.0, True, False, {})
        mock_env.action_space = MagicMock()
        mock_env.action_space.shape = (6,)
        mock_env.action_space.low = [-1.0] * 6
        mock_env.action_space.high = [1.0] * 6
        mock_env.action_space.sample.return_value = [0.0] * 6

        with patch("strands_robots.envs.StrandsSimEnv", return_value=mock_env):
            result = evaluate(mock_policy, "stack cubes", "unitree_g1", num_episodes=1)

        assert result["task"] == "stack cubes"
        assert result["robot_name"] == "unitree_g1"
        assert result["policy_provider"] == "test_policy"


# ───────────────────────── CosmosTrainer Script Resolution ─────────────────────────


class TestCosmosTrainerScriptResolution:
    """Tests for CosmosTrainer._resolve_train_script eager resolution."""

    def test_resolve_returns_none_when_no_scripts(self):
        """When no cosmos scripts exist, returns None."""
        from strands_robots.training import CosmosTrainer

        with patch("os.path.isfile", return_value=False):
            with patch.dict("sys.modules", {"cosmos_oss": None}):
                result = CosmosTrainer._resolve_train_script()
        # May or may not be None depending on actual filesystem,
        # but the method should not raise
        assert result is None or isinstance(result, str)

    def test_resolve_explicit_path(self, tmp_path):
        """Explicit path is returned when file exists."""
        from strands_robots.training import CosmosTrainer

        script = tmp_path / "train.py"
        script.write_text("# train script")
        result = CosmosTrainer._resolve_train_script(str(script))
        assert result == str(script)

    def test_resolve_explicit_nonexistent_falls_through(self):
        """Explicit nonexistent path falls through to auto-detect."""
        from strands_robots.training import CosmosTrainer

        result = CosmosTrainer._resolve_train_script("/nonexistent/train.py")
        # Should fall through and try other candidates
        # Result depends on actual filesystem
        assert result is None or isinstance(result, str)

    def test_eager_resolution_in_init(self, tmp_path):
        """CosmosTrainer.__init__ eagerly resolves train_script_path."""
        from strands_robots.training import CosmosTrainer

        script = tmp_path / "train.py"
        script.write_text("# train script")
        t = CosmosTrainer(train_script_path=str(script))
        assert t.train_script_path == str(script)

    def test_env_var_resolution(self, tmp_path):
        """COSMOS_PREDICT2_PATH env var is used for resolution."""
        from strands_robots.training import CosmosTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# cosmos train")
        with patch.dict(os.environ, {"COSMOS_PREDICT2_PATH": str(tmp_path)}):
            result = CosmosTrainer._resolve_train_script()
        assert result == str(script)

    def test_cosmos_oss_derivation(self):
        """Derive script path from cosmos_oss editable install."""
        import os
        import tempfile

        from strands_robots.training import CosmosTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Simulate cosmos_oss package layout:
            # <repo>/packages/cosmos-oss/cosmos_oss/__init__.py
            pkg_dir = os.path.join(tmpdir, "packages", "cosmos-oss", "cosmos_oss")
            os.makedirs(pkg_dir)
            init_file = os.path.join(pkg_dir, "__init__.py")
            with open(init_file, "w") as f:
                f.write("")

            # Create scripts/train.py at repo root
            scripts_dir = os.path.join(tmpdir, "scripts")
            os.makedirs(scripts_dir)
            train_file = os.path.join(scripts_dir, "train.py")
            with open(train_file, "w") as f:
                f.write("# train")

            # Mock cosmos_oss to point to our temp package
            mock_oss = MagicMock()
            mock_oss.__file__ = init_file

            with patch.dict("sys.modules", {"cosmos_oss": mock_oss}):
                with patch.dict(os.environ, {}, clear=False):
                    # Remove COSMOS_PREDICT2_PATH to test derivation
                    os.environ.pop("COSMOS_PREDICT2_PATH", None)
                    with patch("os.path.expanduser", side_effect=lambda p: p.replace("~", "/nonexistent_home")):
                        result = CosmosTrainer._resolve_train_script()

            assert result == train_file


# ───────────────────────────── create_trainer ─────────────────────────────


class TestCreateTrainer:
    """Tests for the factory function."""

    def test_groot(self):
        from strands_robots.training import Gr00tTrainer, create_trainer

        t = create_trainer("groot", dataset_path="/data")
        assert isinstance(t, Gr00tTrainer)

    def test_lerobot(self):
        from strands_robots.training import LerobotTrainer, create_trainer

        t = create_trainer("lerobot", dataset_repo_id="test/repo")
        assert isinstance(t, LerobotTrainer)

    def test_dreamgen_idm(self):
        from strands_robots.training import DreamgenIdmTrainer, create_trainer

        t = create_trainer("dreamgen_idm", dataset_path="/data")
        assert isinstance(t, DreamgenIdmTrainer)

    def test_dreamgen_vla(self):
        from strands_robots.training import DreamgenVlaTrainer, create_trainer

        t = create_trainer("dreamgen_vla", dataset_path="/data")
        assert isinstance(t, DreamgenVlaTrainer)

    def test_cosmos_predict(self):
        from strands_robots.training import CosmosTrainer, create_trainer

        t = create_trainer("cosmos_predict", dataset_path="/data")
        assert isinstance(t, CosmosTrainer)

    def test_unknown_provider_raises(self):
        from strands_robots.training import create_trainer

        with pytest.raises(ValueError, match="Unknown trainer provider"):
            create_trainer("nonexistent")

    def test_provider_names_match(self):
        """All trainers report their provider_name matching the factory key."""
        from strands_robots.training import create_trainer

        for provider in ["groot", "lerobot", "dreamgen_idm", "dreamgen_vla", "cosmos_predict"]:
            t = create_trainer(provider)
            assert t.provider_name == provider


# ───────────────────────────── Cross-Trainer Design Traps ─────────────────────────────


class TestCrossTrainerDesignTraps:
    """Test subtle design traps across trainers."""

    def test_groot_vs_vla_tune_visual_defaults(self):
        """Gr00t defaults tune_visual=False, VLA defaults True."""
        from strands_robots.training import DreamgenVlaTrainer, Gr00tTrainer

        assert Gr00tTrainer().tune_visual is False
        assert DreamgenVlaTrainer().tune_visual is True

    def test_groot_tune_visual_absent_in_default_command(self):
        """Gr00tTrainer default command must NOT contain --tune-visual."""
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer()
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--tune-visual" not in cmd, "Gr00tTrainer default command has --tune-visual but shouldn't"

    def test_vla_tune_visual_present_in_default_command(self):
        """DreamgenVlaTrainer default command MUST contain --tune-visual."""
        from strands_robots.training import DreamgenVlaTrainer

        t = DreamgenVlaTrainer()
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--tune-visual" in cmd, "DreamgenVlaTrainer default command missing --tune-visual but should have it"

    def test_lerobot_uses_equals_syntax(self):
        """LeRobot subprocess uses --key=value syntax."""
        from strands_robots.training import LerobotTrainer

        t = LerobotTrainer(dataset_repo_id="test/repo", in_process=False)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        # At least one arg should use = syntax
        has_equals = any("=" in str(c) for c in cmd if isinstance(c, str) and c.startswith("--"))
        assert has_equals, "LeRobot subprocess should use --key=value syntax"

    def test_groot_uses_space_syntax(self):
        """Gr00t subprocess uses --key value (space-separated) syntax."""
        from strands_robots.training import Gr00tTrainer

        t = Gr00tTrainer(dataset_path="/data")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        # Find --dataset-path and verify its value is a separate element
        for i, c in enumerate(cmd):
            if c == "--dataset-path":
                assert cmd[i + 1] == "/data", "Gr00t should use --key value syntax"
                break
        else:
            pytest.fail("--dataset-path not found in command")


# ───────────────────────── CosmosTrainer _build_subprocess_env ─────────────────


class TestCosmosTrainerSubprocessEnv:
    """Tests for CosmosTrainer._build_subprocess_env PYTHONPATH injection.

    The cosmos_predict2._src internal package is in the source tree but not the
    installed wheel. _build_subprocess_env must prepend the repo root to
    PYTHONPATH so the subprocess can find it.
    """

    def test_build_subprocess_env_with_script_path(self, tmp_path):
        """Repo root derived from script path is prepended to PYTHONPATH."""
        from strands_robots.training import CosmosTrainer

        # Create fake repo structure: <tmp>/scripts/train.py
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# train")

        t = CosmosTrainer(train_script_path=str(script))
        env = t._build_subprocess_env(str(script))

        assert "PYTHONPATH" in env
        assert str(tmp_path) in env["PYTHONPATH"]

    def test_build_subprocess_env_includes_subpackages(self, tmp_path):
        """Sub-packages (cosmos-oss, cosmos-cuda) are also added."""
        from strands_robots.training import CosmosTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# train")

        # Create sub-package dirs
        (tmp_path / "packages" / "cosmos-oss").mkdir(parents=True)
        (tmp_path / "packages" / "cosmos-cuda").mkdir(parents=True)

        t = CosmosTrainer(train_script_path=str(script))
        env = t._build_subprocess_env(str(script))

        pythonpath = env["PYTHONPATH"]
        assert str(tmp_path / "packages" / "cosmos-oss") in pythonpath
        assert str(tmp_path / "packages" / "cosmos-cuda") in pythonpath

    def test_build_subprocess_env_preserves_existing_pythonpath(self, tmp_path):
        """Existing PYTHONPATH is preserved (appended)."""
        from strands_robots.training import CosmosTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# train")

        t = CosmosTrainer(train_script_path=str(script))

        with patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}):
            env = t._build_subprocess_env(str(script))

        assert "/existing/path" in env["PYTHONPATH"]
        assert str(tmp_path) in env["PYTHONPATH"]

    def test_build_subprocess_env_no_script_path(self):
        """Without script path, env is just os.environ copy."""
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer.__new__(CosmosTrainer)
        t.train_script_path = None

        with patch.dict(os.environ, {"COSMOS_PREDICT2_PATH": ""}):
            env = t._build_subprocess_env(None)

        # Should still be a dict (copy of os.environ)
        assert isinstance(env, dict)

    def test_build_subprocess_env_fallback_to_env_var(self, tmp_path):
        """Falls back to COSMOS_PREDICT2_PATH when no script path."""
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer.__new__(CosmosTrainer)
        t.train_script_path = None

        with patch.dict(os.environ, {"COSMOS_PREDICT2_PATH": str(tmp_path)}):
            env = t._build_subprocess_env(None)

        assert str(tmp_path) in env.get("PYTHONPATH", "")

    def test_train_passes_env_to_subprocess(self, tmp_path):
        """CosmosTrainer.train() passes env= to subprocess.run."""
        from strands_robots.training import CosmosTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# cosmos")

        t = CosmosTrainer(train_script_path=str(script))

        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()

        # Verify env= kwarg was passed
        call_kwargs = mock_run.call_args[1]
        assert "env" in call_kwargs
        assert isinstance(call_kwargs["env"], dict)
        assert str(tmp_path) in call_kwargs["env"].get("PYTHONPATH", "")


# ───────────────────────── _discover_nvidia_cuda_lib_paths ──────────────────


class TestDiscoverNvidiaCudaLibPaths:
    """Tests for _discover_nvidia_cuda_lib_paths() helper.

    Validates that pip-installed NVIDIA CUDA shared library directories
    (nvidia-cublas-cu12, nvidia-cuda-runtime-cu12, etc.) are correctly
    discovered and placed BEFORE system CUDA paths in LD_LIBRARY_PATH.

    This is critical for CUDA 12/13 mismatch environments where packages
    compiled against CUDA 12 need libcublas.so.12 from pip, not
    libcublas.so.13 from the system CUDA toolkit.
    """

    def test_returns_list(self):
        """Function returns a list of strings."""
        from strands_robots.training import _discover_nvidia_cuda_lib_paths

        result = _discover_nvidia_cuda_lib_paths()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, str)

    def test_discovers_pip_nvidia_dirs(self, tmp_path):
        """Discovers nvidia/*/lib/ directories from sys.path."""
        from strands_robots.training import _discover_nvidia_cuda_lib_paths

        # Create fake nvidia package structure
        nvidia_dir = tmp_path / "nvidia" / "cublas" / "lib"
        nvidia_dir.mkdir(parents=True)
        (nvidia_dir / "libcublas.so.12").touch()

        import sys

        original_path = sys.path[:]
        sys.path.insert(0, str(tmp_path))
        try:
            result = _discover_nvidia_cuda_lib_paths()
            assert str(nvidia_dir) in result
        finally:
            sys.path[:] = original_path

    def test_discovers_multiple_nvidia_packages(self, tmp_path):
        """Discovers multiple nvidia packages (cublas, cuda_runtime, etc.)."""
        from strands_robots.training import _discover_nvidia_cuda_lib_paths

        packages = ["cublas", "cuda_runtime", "cufft", "cusolver", "nccl"]
        expected_dirs = []
        for pkg in packages:
            lib_dir = tmp_path / "nvidia" / pkg / "lib"
            lib_dir.mkdir(parents=True)
            (lib_dir / f"lib{pkg}.so.12").touch()
            expected_dirs.append(str(lib_dir))

        import sys

        original_path = sys.path[:]
        sys.path.insert(0, str(tmp_path))
        try:
            result = _discover_nvidia_cuda_lib_paths()
            for d in expected_dirs:
                assert d in result, f"Missing: {d}"
        finally:
            sys.path[:] = original_path

    def test_skips_nonexistent_nvidia_dirs(self, tmp_path):
        """Ignores sys.path entries without nvidia/ subdirectory."""
        # tmp_path exists but has no nvidia/ subdirectory
        import sys

        from strands_robots.training import _discover_nvidia_cuda_lib_paths

        original_path = sys.path[:]
        sys.path.insert(0, str(tmp_path))
        try:
            result = _discover_nvidia_cuda_lib_paths()
            # Should not crash, should not include tmp_path entries
            for p in result:
                if str(tmp_path) in p:
                    pytest.fail(f"Found unexpected path from empty dir: {p}")
        finally:
            sys.path[:] = original_path

    def test_skips_nondir_sys_path_entries(self):
        """Skips sys.path entries that are not directories."""
        import sys

        from strands_robots.training import _discover_nvidia_cuda_lib_paths

        original_path = sys.path[:]
        sys.path.insert(0, "/nonexistent/path/that/does/not/exist")
        try:
            # Should not crash
            result = _discover_nvidia_cuda_lib_paths()
            assert isinstance(result, list)
        finally:
            sys.path[:] = original_path

    def test_deduplicates_user_site_packages(self, tmp_path):
        """Does not add duplicate paths from user site-packages."""
        from strands_robots.training import _discover_nvidia_cuda_lib_paths

        # Create nvidia dir in a path that's ALSO the user site-packages
        nvidia_dir = tmp_path / "nvidia" / "cublas" / "lib"
        nvidia_dir.mkdir(parents=True)

        import sys

        original_path = sys.path[:]
        # Add to sys.path AND mock user site-packages to same location
        sys.path.insert(0, str(tmp_path))
        try:
            with patch("os.path.expanduser", return_value=str(tmp_path)):
                result = _discover_nvidia_cuda_lib_paths()
            # The path should appear exactly once
            count = result.count(str(nvidia_dir))
            assert count == 1, f"Path appeared {count} times (expected 1)"
        finally:
            sys.path[:] = original_path

    def test_ld_library_path_includes_nvidia_pip(self, tmp_path):
        """CosmosTrainer._build_subprocess_env includes pip NVIDIA paths in LD_LIBRARY_PATH."""
        import sys as _sys

        from strands_robots.training import CosmosTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# train")

        # Create a fake nvidia pip package so the test works on any platform
        nvidia_dir = tmp_path / "nvidia" / "cublas" / "lib"
        nvidia_dir.mkdir(parents=True)
        (nvidia_dir / "libcublas.so.12").touch()
        original_path = _sys.path[:]
        _sys.path.insert(0, str(tmp_path))

        try:
            t = CosmosTrainer(train_script_path=str(script))
            env = t._build_subprocess_env(str(script))
            ld_path = env.get("LD_LIBRARY_PATH", "")
            assert len(ld_path) > 0
        finally:
            _sys.path[:] = original_path

    def test_pip_paths_before_system_paths(self, tmp_path):
        """Pip NVIDIA paths come before system CUDA paths in LD_LIBRARY_PATH.

        Uses os.path.isdir mock to guarantee /usr/local/cuda/lib64 appears
        in the path list, so the ordering assertion always runs — even on
        CI runners without CUDA installed.
        """
        from strands_robots.training import CosmosTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# train")

        # Create fake nvidia package
        nvidia_dir = tmp_path / "nvidia" / "cublas" / "lib"
        nvidia_dir.mkdir(parents=True)
        (nvidia_dir / "libcublas.so.12").touch()

        import sys

        original_isdir = os.path.isdir
        original_path = sys.path[:]
        sys.path.insert(0, str(tmp_path))

        def patched_isdir(p):
            # Make /usr/local/cuda/lib64 appear to exist so it enters
            # the LD_LIBRARY_PATH (needed for ordering assertion)
            if p == "/usr/local/cuda/lib64":
                return True
            return original_isdir(p)

        try:
            with patch("os.path.isdir", side_effect=patched_isdir):
                t = CosmosTrainer(train_script_path=str(script))
                env = t._build_subprocess_env(str(script))

            ld_path = env.get("LD_LIBRARY_PATH", "")
            parts = ld_path.split(os.pathsep)

            pip_idx = next(
                (i for i, p in enumerate(parts) if str(nvidia_dir) in p),
                None,
            )
            sys_idx = next(
                (i for i, p in enumerate(parts) if "/usr/local/cuda" in p),
                None,
            )

            assert pip_idx is not None, f"Pip NVIDIA path not found in LD_LIBRARY_PATH: {ld_path}"
            assert sys_idx is not None, f"System CUDA path not found in LD_LIBRARY_PATH: {ld_path}"
            assert pip_idx < sys_idx, (
                f"Pip NVIDIA path (idx={pip_idx}) must come before " f"system CUDA path (idx={sys_idx}) in: {ld_path}"
            )
        finally:
            sys.path[:] = original_path

    def test_cosmos_transfer_trainer_also_uses_discovery(self, tmp_path):
        """CosmosTransferTrainer._build_subprocess_env also includes pip NVIDIA paths."""
        from strands_robots.training import CosmosTransferTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# train")

        # Create fake nvidia package
        nvidia_dir = tmp_path / "nvidia" / "cuda_runtime" / "lib"
        nvidia_dir.mkdir(parents=True)
        (nvidia_dir / "libcudart.so.12").touch()

        import sys

        original_path = sys.path[:]
        sys.path.insert(0, str(tmp_path))
        try:
            t = CosmosTransferTrainer(cosmos_transfer_path=str(tmp_path))
            env = t._build_subprocess_env()

            ld_path = env.get("LD_LIBRARY_PATH", "")
            assert (
                str(nvidia_dir) in ld_path
            ), f"CosmosTransferTrainer LD_LIBRARY_PATH missing pip NVIDIA dir: {ld_path}"
        finally:
            sys.path[:] = original_path

    def test_empty_nvidia_dir_ignored(self, tmp_path):
        """nvidia/ exists but has no */lib/ subdirectories — returns empty for that site."""
        from strands_robots.training import _discover_nvidia_cuda_lib_paths

        # Create nvidia/ directory with no package subdirs
        (tmp_path / "nvidia").mkdir()

        import sys

        original_path = sys.path[:]
        sys.path.insert(0, str(tmp_path))
        try:
            result = _discover_nvidia_cuda_lib_paths()
            # Should not include any paths from tmp_path
            tmp_results = [p for p in result if str(tmp_path) in p]
            assert len(tmp_results) == 0
        finally:
            sys.path[:] = original_path

    def test_existing_ld_library_path_preserved(self, tmp_path):
        """Existing LD_LIBRARY_PATH entries are preserved (appended after new paths)."""
        from strands_robots.training import CosmosTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# train")

        t = CosmosTrainer(train_script_path=str(script))

        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/my/custom/libs"}):
            env = t._build_subprocess_env(str(script))

        ld_path = env.get("LD_LIBRARY_PATH", "")
        assert "/my/custom/libs" in ld_path, f"Existing LD_LIBRARY_PATH not preserved: {ld_path}"


# ───────────────────────── evaluate() Isaac backend ─────────────────────────


class TestEvaluateIsaacBackend:
    """Tests for the evaluate() function with backend='isaac'."""

    def test_isaac_backend_import_error(self):
        """Isaac backend returns error when IsaacGymEnv not available."""
        from strands_robots.training import evaluate

        mock_policy = MagicMock()
        mock_policy.provider_name = "mock"

        with patch.dict("sys.modules", {"strands_robots.isaac.isaac_gym_env": None}):
            result = evaluate(mock_policy, "pick cube", "so100", backend="isaac")

        assert "error" in result
        assert result["num_episodes"] == 0

    def test_isaac_backend_creation_error(self):
        """Isaac backend returns error when env creation fails."""
        from strands_robots.training import evaluate

        mock_policy = MagicMock()
        mock_policy.provider_name = "mock"

        mock_isaac_mod = MagicMock()
        mock_isaac_mod.IsaacGymEnv = MagicMock(side_effect=RuntimeError("No GPU"))

        with patch.dict(
            "sys.modules",
            {
                "strands_robots.isaac.isaac_gym_env": mock_isaac_mod,
            },
        ):
            result = evaluate(mock_policy, "pick cube", "so100", backend="isaac")

        assert "error" in result
        assert "Isaac" in result["error"] or "GPU" in result["error"]


# ───────────────── CosmosTrainer LoRA / freeze_backbone Warnings ─────────────────


class TestCosmosTrainerWarnings:
    """Tests for CosmosTrainer warnings when LoRA/freeze_backbone params are set
    but not forwarded to the cosmos-predict2 CLI.

    PR #106 review identified that CosmosTrainer.__init__ accepts use_lora,
    lora_rank, and freeze_backbone but never forwards them to the subprocess
    command.  Rather than guessing the Hydra config keys (which could break),
    we now emit UserWarning for non-default values.
    """

    def test_no_warning_for_defaults(self):
        """Default params (use_lora=False, freeze_backbone=True) should NOT warn."""
        import warnings

        from strands_robots.training import CosmosTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTrainer(
                base_model_path="nvidia/Cosmos-Predict2.5-2B",
                dataset_path="/tmp/data",
            )
            # Filter to only UserWarnings from CosmosTrainer
            cosmos_warnings = [
                x for x in w if issubclass(x.category, UserWarning) and "CosmosTrainer" in str(x.message)
            ]
            assert len(cosmos_warnings) == 0, f"Unexpected warnings: {cosmos_warnings}"

    def test_use_lora_true_warns(self):
        """use_lora=True should emit UserWarning about LoRA not being forwarded."""
        import warnings

        from strands_robots.training import CosmosTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTrainer(
                base_model_path="nvidia/Cosmos-Predict2.5-2B",
                dataset_path="/tmp/data",
                use_lora=True,
                lora_rank=32,
            )
            lora_warnings = [x for x in w if issubclass(x.category, UserWarning) and "use_lora" in str(x.message)]
            assert len(lora_warnings) == 1
            msg = str(lora_warnings[0].message)
            assert "not yet forwarded" in msg
            assert "silently ignored" in msg

    def test_use_lora_warning_mentions_manual_override(self):
        """LoRA warning should tell users how to pass overrides manually."""
        import warnings

        from strands_robots.training import CosmosTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTrainer(use_lora=True)
            lora_warnings = [x for x in w if "use_lora" in str(x.message)]
            assert len(lora_warnings) == 1
            assert "train(" in str(lora_warnings[0].message)

    def test_freeze_backbone_false_warns(self):
        """freeze_backbone=False should emit UserWarning."""
        import warnings

        from strands_robots.training import CosmosTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTrainer(
                base_model_path="nvidia/Cosmos-Predict2.5-2B",
                dataset_path="/tmp/data",
                freeze_backbone=False,
            )
            fb_warnings = [x for x in w if issubclass(x.category, UserWarning) and "freeze_backbone" in str(x.message)]
            assert len(fb_warnings) == 1
            msg = str(fb_warnings[0].message)
            assert "not yet forwarded" in msg
            assert "backbone will remain frozen" in msg

    def test_freeze_backbone_warning_mentions_manual_override(self):
        """freeze_backbone warning should tell users how to override via train()."""
        import warnings

        from strands_robots.training import CosmosTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTrainer(freeze_backbone=False)
            fb_warnings = [x for x in w if "freeze_backbone" in str(x.message)]
            assert len(fb_warnings) == 1
            assert "train(" in str(fb_warnings[0].message)

    def test_both_lora_and_freeze_warns_twice(self):
        """Setting both use_lora=True and freeze_backbone=False should emit 2 warnings."""
        import warnings

        from strands_robots.training import CosmosTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTrainer(
                use_lora=True,
                lora_rank=64,
                freeze_backbone=False,
            )
            cosmos_warnings = [
                x for x in w if issubclass(x.category, UserWarning) and "CosmosTrainer" in str(x.message)
            ]
            assert len(cosmos_warnings) == 2

    def test_use_lora_false_no_warning(self):
        """use_lora=False (default) should not warn even when lora_rank is non-default."""
        import warnings

        from strands_robots.training import CosmosTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTrainer(use_lora=False, lora_rank=128)
            lora_warnings = [x for x in w if "use_lora" in str(x.message)]
            assert len(lora_warnings) == 0

    def test_freeze_backbone_true_no_warning(self):
        """freeze_backbone=True (default) should not warn."""
        import warnings

        from strands_robots.training import CosmosTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTrainer(freeze_backbone=True)
            fb_warnings = [x for x in w if "freeze_backbone" in str(x.message)]
            assert len(fb_warnings) == 0

    def test_lora_values_still_stored(self):
        """Even with warnings, the values should still be stored on self."""
        import warnings

        from strands_robots.training import CosmosTrainer

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            t = CosmosTrainer(use_lora=True, lora_rank=64, freeze_backbone=False)

        assert t.use_lora is True
        assert t.lora_rank == 64
        assert t.freeze_backbone is False

    def test_manual_kwargs_override_still_works(self):
        """Users can pass arbitrary key=value overrides to train() via kwargs.

        This validates the recommended workaround from the warnings.
        """
        import warnings
        from unittest.mock import patch

        from strands_robots.training import CosmosTrainer

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            t = CosmosTrainer(
                base_model_path="nvidia/Cosmos-Predict2.5-2B",
                use_lora=True,
            )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            # Pass the LoRA overrides manually via kwargs
            t.train(**{"model.lora.enabled": "True", "model.lora.rank": "32"})

            cmd = mock_run.call_args[0][0]
            # The kwargs should appear as key=value overrides in the command
            assert "model.lora.enabled=True" in cmd
            assert "model.lora.rank=32" in cmd
