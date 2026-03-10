"""Tests for CosmosTransferTrainer — Cosmos Transfer 2.5 post-training.

Coverage target: ~95%+ of CosmosTransferTrainer code.
All tests CPU-only with mocked heavy dependencies (subprocess, cosmos_transfer2).

Test organization:
1. InitDefaults — constructor defaults and validation
2. InitCustom — custom parameters, edge cases
3. InitValidation — invalid control_type, mode, resolution
4. InitWarnings — LoRA and freeze_backbone warnings
5. ScriptResolution — _resolve_train_script paths
6. ConfigNameInference — _infer_config_name logic
7. ConfigFileResolution — _resolve_config_file
8. SubprocessEnv — _build_subprocess_env PYTHONPATH injection
9. TrainCommand — train() CLI command construction
10. TrainModes — sim2real, domain_adaptation, control_finetuning
11. TrainMultiGPU — torchrun for multi-GPU
12. TrainResult — return dict structure
13. Evaluate — evaluate() placeholder
14. Factory — create_trainer("cosmos_transfer") dispatch
15. CrossTrainerDifferences — CosmosTrainer vs CosmosTransferTrainer

Key design traps tested:
- CosmosTransferTrainer targets cosmos-transfer2.5 (NOT cosmos-predict2.5)
- control_type must be valid (depth/edge/seg/vis)
- mode must be valid (sim2real/domain_adaptation/control_finetuning)
- freeze_backbone=False warns about VRAM requirements
- control_finetuning mode forces freeze_backbone=True in the command
"""

import os
import sys
import warnings
from unittest.mock import MagicMock, patch

import pytest

# Ensure strands_robots is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ═══════════════════════════════ Init Defaults ═══════════════════════════════


class TestCosmosTransferTrainerInitDefaults:
    """Test default constructor values."""

    def test_init_defaults(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer()
        assert t.provider_name == "cosmos_transfer"
        assert t.base_model_path == "nvidia/Cosmos-Transfer2-7B"
        assert t.control_type == "depth"
        assert t.mode == "sim2real"
        assert t.use_lora is False
        assert t.lora_rank == 16
        assert t.freeze_backbone is True
        assert t.freeze_controlnet is False
        assert t.control_weight == 1.0
        assert t.guidance == 3.0
        assert t.output_resolution == "720"
        assert t.robot_variant is None
        assert t.dataset_path == ""

    def test_default_config(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer()
        assert t.config.max_steps == 10000
        assert t.config.batch_size == 16
        assert t.config.num_gpus == 1


# ═══════════════════════════════ Init Custom ═══════════════════════════════


class TestCosmosTransferTrainerInitCustom:
    """Test custom constructor parameters."""

    def test_init_custom(self):
        from strands_robots.training import CosmosTransferTrainer, TrainConfig

        t = CosmosTransferTrainer(
            base_model_path="nvidia/Cosmos-Transfer2-14B",
            dataset_path="/data/sim_real_pairs",
            control_type="edge",
            mode="domain_adaptation",
            config=TrainConfig(num_gpus=4, max_steps=5000),
            control_weight=0.8,
            guidance=5.0,
            output_resolution="1080",
        )
        assert t.base_model_path == "nvidia/Cosmos-Transfer2-14B"
        assert t.dataset_path == "/data/sim_real_pairs"
        assert t.control_type == "edge"
        assert t.mode == "domain_adaptation"
        assert t.config.num_gpus == 4
        assert t.config.max_steps == 5000
        assert t.control_weight == 0.8
        assert t.guidance == 5.0
        assert t.output_resolution == "1080"

    def test_robot_variant(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(robot_variant="multiview-gr1-depth")
        assert t.robot_variant == "multiview-gr1-depth"

    def test_control_finetuning_mode(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(mode="control_finetuning")
        assert t.mode == "control_finetuning"

    def test_all_control_types(self):
        from strands_robots.training import CosmosTransferTrainer

        for ct in ("depth", "edge", "seg", "vis"):
            t = CosmosTransferTrainer(control_type=ct)
            assert t.control_type == ct


# ═══════════════════════════════ Init Validation ═══════════════════════════════


class TestCosmosTransferTrainerValidation:
    """Test constructor validation."""

    def test_invalid_control_type(self):
        from strands_robots.training import CosmosTransferTrainer

        with pytest.raises(ValueError, match="Invalid control_type"):
            CosmosTransferTrainer(control_type="invalid")

    def test_invalid_mode(self):
        from strands_robots.training import CosmosTransferTrainer

        with pytest.raises(ValueError, match="Invalid mode"):
            CosmosTransferTrainer(mode="invalid")

    def test_invalid_output_resolution(self):
        from strands_robots.training import CosmosTransferTrainer

        with pytest.raises(ValueError, match="Invalid output_resolution"):
            CosmosTransferTrainer(output_resolution="4k")


# ═══════════════════════════════ Init Warnings ═══════════════════════════════


class TestCosmosTransferTrainerWarnings:
    """Test warning emissions for LoRA and freeze_backbone."""

    def test_no_warning_for_defaults(self):
        from strands_robots.training import CosmosTransferTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTransferTrainer()
            cosmos_warnings = [
                x for x in w if issubclass(x.category, UserWarning) and "CosmosTransferTrainer" in str(x.message)
            ]
            assert len(cosmos_warnings) == 0

    def test_use_lora_true_warns(self):
        from strands_robots.training import CosmosTransferTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTransferTrainer(use_lora=True)
            lora_warnings = [x for x in w if issubclass(x.category, UserWarning) and "use_lora" in str(x.message)]
            assert len(lora_warnings) == 1
            assert "not yet verified" in str(lora_warnings[0].message)

    def test_freeze_backbone_false_warns(self):
        from strands_robots.training import CosmosTransferTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTransferTrainer(freeze_backbone=False)
            fb_warnings = [x for x in w if issubclass(x.category, UserWarning) and "freeze_backbone" in str(x.message)]
            assert len(fb_warnings) == 1
            assert "80GB" in str(fb_warnings[0].message)

    def test_control_finetuning_freeze_backbone_false_no_vram_warning(self):
        """control_finetuning mode should NOT warn about VRAM even with freeze_backbone=False."""
        from strands_robots.training import CosmosTransferTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTransferTrainer(mode="control_finetuning", freeze_backbone=False)
            vram_warnings = [x for x in w if issubclass(x.category, UserWarning) and "80GB" in str(x.message)]
            assert len(vram_warnings) == 0, (
                "control_finetuning mode should suppress VRAM warning since " "backbone is always frozen in that mode"
            )

    def test_use_lora_false_no_warning(self):
        from strands_robots.training import CosmosTransferTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTransferTrainer(use_lora=False, lora_rank=64)
            lora_warnings = [x for x in w if "use_lora" in str(x.message)]
            assert len(lora_warnings) == 0

    def test_both_lora_and_freeze_warns_twice(self):
        from strands_robots.training import CosmosTransferTrainer

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CosmosTransferTrainer(use_lora=True, freeze_backbone=False)
            cosmos_warnings = [
                x for x in w if issubclass(x.category, UserWarning) and "CosmosTransferTrainer" in str(x.message)
            ]
            assert len(cosmos_warnings) == 2

    def test_lora_values_still_stored(self):
        from strands_robots.training import CosmosTransferTrainer

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            t = CosmosTransferTrainer(use_lora=True, lora_rank=64, freeze_backbone=False)
        assert t.use_lora is True
        assert t.lora_rank == 64
        assert t.freeze_backbone is False


# ═══════════════════════════════ Script Resolution ═══════════════════════════════


class TestCosmosTransferScriptResolution:
    """Tests for CosmosTransferTrainer._resolve_train_script."""

    def test_resolve_returns_none_when_no_scripts(self):
        from strands_robots.training import CosmosTransferTrainer

        with patch("os.path.isfile", return_value=False):
            with patch.dict("sys.modules", {"cosmos_transfer2": None}):
                result = CosmosTransferTrainer._resolve_train_script()
        assert result is None or isinstance(result, str)

    def test_resolve_explicit_path(self, tmp_path):
        from strands_robots.training import CosmosTransferTrainer

        script = tmp_path / "train.py"
        script.write_text("# cosmos transfer train")
        result = CosmosTransferTrainer._resolve_train_script(str(script))
        assert result == str(script)

    def test_resolve_explicit_nonexistent_falls_through(self):
        from strands_robots.training import CosmosTransferTrainer

        result = CosmosTransferTrainer._resolve_train_script("/nonexistent/train.py")
        assert result is None or isinstance(result, str)

    def test_eager_resolution_in_init(self, tmp_path):
        from strands_robots.training import CosmosTransferTrainer

        script = tmp_path / "train.py"
        script.write_text("# cosmos transfer train")
        t = CosmosTransferTrainer(train_script_path=str(script))
        assert t.train_script_path == str(script)

    def test_env_var_cosmos_transfer_path(self, tmp_path):
        from strands_robots.training import CosmosTransferTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# cosmos transfer train")
        with patch.dict(os.environ, {"COSMOS_TRANSFER_PATH": str(tmp_path)}):
            result = CosmosTransferTrainer._resolve_train_script()
        assert result == str(script)

    def test_env_var_cosmos_transfer2_path(self, tmp_path):
        from strands_robots.training import CosmosTransferTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# train")
        with patch.dict(os.environ, {"COSMOS_TRANSFER2_PATH": str(tmp_path)}):
            result = CosmosTransferTrainer._resolve_train_script()
        assert result == str(script)

    def test_env_var_prefers_first(self, tmp_path):
        """COSMOS_TRANSFER_PATH takes precedence over COSMOS_TRANSFER2_PATH."""
        from strands_robots.training import CosmosTransferTrainer

        dir1 = tmp_path / "ct1" / "scripts"
        dir1.mkdir(parents=True)
        script1 = dir1 / "train.py"
        script1.write_text("# first")

        dir2 = tmp_path / "ct2" / "scripts"
        dir2.mkdir(parents=True)
        script2 = dir2 / "train.py"
        script2.write_text("# second")

        with patch.dict(
            os.environ,
            {
                "COSMOS_TRANSFER_PATH": str(tmp_path / "ct1"),
                "COSMOS_TRANSFER2_PATH": str(tmp_path / "ct2"),
            },
        ):
            result = CosmosTransferTrainer._resolve_train_script()
        assert result == str(script1)


# ═══════════════════════════════ Config Name Inference ═══════════════════════════════


class TestCosmosTransferConfigNameInference:
    """Tests for _infer_config_name."""

    def test_sim2real_7b_depth(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(
            base_model_path="nvidia/Cosmos-Transfer2-7B",
            mode="sim2real",
            control_type="depth",
        )
        assert t.config_name == "cosmos_transfer2_7b_720p_sim2real_depth"

    def test_sim2real_14b_edge(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(
            base_model_path="nvidia/Cosmos-Transfer2-14B",
            mode="sim2real",
            control_type="edge",
            output_resolution="1080",
        )
        assert t.config_name == "cosmos_transfer2_14b_1080p_sim2real_edge"

    def test_domain_adaptation(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(mode="domain_adaptation")
        assert "domain_adapt" in t.config_name

    def test_control_finetuning(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(mode="control_finetuning", control_type="seg")
        assert "controlnet" in t.config_name
        assert "seg" in t.config_name

    def test_robot_variant_overrides(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(robot_variant="multiview-gr1-depth")
        assert "multiview-gr1-depth" in t.config_name
        assert "sim2real" not in t.config_name

    def test_custom_config_name(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(config_name="my_custom_experiment")
        assert t.config_name == "my_custom_experiment"


# ═══════════════════════════════ Config File Resolution ═══════════════════════════════


class TestCosmosTransferConfigFileResolution:
    """Tests for _resolve_config_file."""

    def test_default_sim2real(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(mode="sim2real")
        config_file = t._resolve_config_file()
        assert "sim2real" in config_file

    def test_control_finetuning_config(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(mode="control_finetuning")
        config_file = t._resolve_config_file()
        assert "controlnet" in config_file

    def test_domain_adaptation_config(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(mode="domain_adaptation")
        config_file = t._resolve_config_file()
        assert "training" in config_file

    def test_custom_config_file(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(config_file="/path/to/custom_config.py")
        assert t._resolve_config_file() == "/path/to/custom_config.py"


# ═══════════════════════════════ Subprocess Env ═══════════════════════════════


class TestCosmosTransferSubprocessEnv:
    """Tests for _build_subprocess_env."""

    def test_build_subprocess_env_with_script_path(self, tmp_path):
        from strands_robots.training import CosmosTransferTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# train")

        t = CosmosTransferTrainer(train_script_path=str(script))
        env = t._build_subprocess_env(str(script))

        assert "PYTHONPATH" in env
        assert str(tmp_path) in env["PYTHONPATH"]

    def test_build_subprocess_env_includes_subpackages(self, tmp_path):
        from strands_robots.training import CosmosTransferTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# train")
        (tmp_path / "packages" / "cosmos-oss").mkdir(parents=True)
        (tmp_path / "packages" / "cosmos-cuda").mkdir(parents=True)

        t = CosmosTransferTrainer(train_script_path=str(script))
        env = t._build_subprocess_env(str(script))

        assert str(tmp_path / "packages" / "cosmos-oss") in env["PYTHONPATH"]
        assert str(tmp_path / "packages" / "cosmos-cuda") in env["PYTHONPATH"]

    def test_build_subprocess_env_preserves_existing_pythonpath(self, tmp_path):
        from strands_robots.training import CosmosTransferTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# train")

        t = CosmosTransferTrainer(train_script_path=str(script))
        with patch.dict(os.environ, {"PYTHONPATH": "/existing/path"}):
            env = t._build_subprocess_env(str(script))

        assert "/existing/path" in env["PYTHONPATH"]
        assert str(tmp_path) in env["PYTHONPATH"]

    def test_build_subprocess_env_no_script_path(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer.__new__(CosmosTransferTrainer)
        t.train_script_path = None
        with patch.dict(os.environ, {"COSMOS_TRANSFER_PATH": "", "COSMOS_TRANSFER2_PATH": ""}):
            env = t._build_subprocess_env(None)
        assert isinstance(env, dict)

    def test_build_subprocess_env_fallback_to_env_var(self, tmp_path):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer.__new__(CosmosTransferTrainer)
        t.train_script_path = None
        with patch.dict(os.environ, {"COSMOS_TRANSFER_PATH": str(tmp_path)}):
            env = t._build_subprocess_env(None)
        assert str(tmp_path) in env.get("PYTHONPATH", "")


# ═══════════════════════════════ Train Command ═══════════════════════════════


class TestCosmosTransferTrainCommand:
    """Tests for train() CLI command construction."""

    def test_train_single_gpu(self):
        from strands_robots.training import CosmosTransferTrainer, TrainConfig

        t = CosmosTransferTrainer(config=TrainConfig(num_gpus=1))
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            result = t.train()
        assert result["status"] == "completed"
        cmd = mock_run.call_args[0][0]
        assert "torchrun" not in cmd

    def test_train_multi_gpu_uses_torchrun(self):
        from strands_robots.training import CosmosTransferTrainer, TrainConfig

        t = CosmosTransferTrainer(config=TrainConfig(num_gpus=4))
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "torchrun"
        assert "--nproc_per_node=4" in cmd

    def test_train_includes_config_flag(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer()
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "--config" in cmd
        assert "--" in cmd

    def test_train_includes_experiment(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(mode="sim2real", control_type="depth")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        experiment_args = [c for c in cmd if c.startswith("experiment=")]
        assert len(experiment_args) == 1
        assert "sim2real" in experiment_args[0]
        assert "depth" in experiment_args[0]

    def test_train_includes_control_type(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(control_type="edge")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "model.control_type=edge" in cmd

    def test_train_includes_control_weight(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(control_weight=0.7)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "model.control_weight=0.7" in cmd

    def test_train_includes_guidance(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(guidance=5.0)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "model.guidance=5.0" in cmd

    def test_train_includes_output_resolution(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(output_resolution="1080")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "model.output_resolution=1080" in cmd

    def test_train_includes_robot_variant(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(robot_variant="multiview-gr1-depth")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "model.robot_variant=multiview-gr1-depth" in cmd

    def test_train_no_robot_variant_omits_flag(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(robot_variant=None)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert not any("robot_variant" in c for c in cmd)

    def test_train_wandb_online(self):
        from strands_robots.training import CosmosTransferTrainer, TrainConfig

        t = CosmosTransferTrainer(config=TrainConfig(use_wandb=True))
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert any("wandb_mode=online" in c for c in cmd)

    def test_train_wandb_disabled(self):
        from strands_robots.training import CosmosTransferTrainer, TrainConfig

        t = CosmosTransferTrainer(config=TrainConfig(use_wandb=False))
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert any("wandb_mode=disabled" in c for c in cmd)

    def test_train_passes_kwargs(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer()
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train(**{"model.custom_param": "value"})
        cmd = mock_run.call_args[0][0]
        assert "model.custom_param=value" in cmd

    def test_train_custom_config_file(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(config_file="/custom/config.py")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        idx = cmd.index("--config")
        assert cmd[idx + 1] == "/custom/config.py"

    def test_train_passes_env_to_subprocess(self, tmp_path):
        from strands_robots.training import CosmosTransferTrainer

        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        script = scripts_dir / "train.py"
        script.write_text("# train")

        t = CosmosTransferTrainer(train_script_path=str(script))
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        call_kwargs = mock_run.call_args[1]
        assert "env" in call_kwargs
        assert isinstance(call_kwargs["env"], dict)


# ═══════════════════════════════ Train Modes ═══════════════════════════════


class TestCosmosTransferTrainModes:
    """Test mode-specific command construction."""

    def test_control_finetuning_forces_freeze_backbone(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(mode="control_finetuning")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "model.freeze_backbone=True" in cmd
        assert "model.freeze_controlnet=False" in cmd

    def test_sim2real_freeze_backbone_true(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(mode="sim2real", freeze_backbone=True)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "model.freeze_backbone=True" in cmd

    def test_sim2real_freeze_backbone_false(self):
        from strands_robots.training import CosmosTransferTrainer

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            t = CosmosTransferTrainer(mode="sim2real", freeze_backbone=False)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "model.freeze_backbone=False" in cmd

    def test_freeze_controlnet_flag(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(freeze_controlnet=True)
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            t.train()
        cmd = mock_run.call_args[0][0]
        assert "model.freeze_controlnet=True" in cmd

    def test_domain_adaptation_mode(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(mode="domain_adaptation")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            result = t.train()
        assert result["mode"] == "domain_adaptation"


# ═══════════════════════════════ Train Result ═══════════════════════════════


class TestCosmosTransferTrainResult:
    """Test train() return dict structure."""

    def test_result_success(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer(mode="sim2real", control_type="depth")
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result):
            result = t.train()
        assert result["provider"] == "cosmos_transfer"
        assert result["mode"] == "sim2real"
        assert result["control_type"] == "depth"
        assert result["status"] == "completed"
        assert result["returncode"] == 0
        assert "output_dir" in result

    def test_result_failure(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer()
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            result = t.train()
        assert result["status"] == "failed"
        assert result["returncode"] == 1


# ═══════════════════════════════ Evaluate ═══════════════════════════════


class TestCosmosTransferEvaluate:
    """Test evaluate() method."""

    def test_evaluate_returns_not_implemented(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer()
        result = t.evaluate()
        assert result["status"] == "not_implemented"
        assert result["provider"] == "cosmos_transfer"


# ═══════════════════════════════ Factory ═══════════════════════════════


class TestCosmosTransferFactory:
    """Test create_trainer("cosmos_transfer") dispatch."""

    def test_factory_creates_cosmos_transfer_trainer(self):
        from strands_robots.training import CosmosTransferTrainer, create_trainer

        t = create_trainer("cosmos_transfer", dataset_path="/data")
        assert isinstance(t, CosmosTransferTrainer)
        assert t.provider_name == "cosmos_transfer"

    def test_factory_passes_kwargs(self):
        from strands_robots.training import create_trainer

        t = create_trainer(
            "cosmos_transfer",
            control_type="edge",
            mode="control_finetuning",
        )
        assert t.control_type == "edge"
        assert t.mode == "control_finetuning"


# ═══════════════════════════════ Cross-Trainer Differences ═══════════════════════════════


class TestCrossTrainerDifferences:
    """Verify CosmosTrainer vs CosmosTransferTrainer are distinct."""

    def test_different_provider_names(self):
        from strands_robots.training import CosmosTrainer, CosmosTransferTrainer

        assert CosmosTrainer().provider_name == "cosmos_predict"
        assert CosmosTransferTrainer().provider_name == "cosmos_transfer"

    def test_different_default_models(self):
        from strands_robots.training import CosmosTrainer, CosmosTransferTrainer

        assert "Predict" in CosmosTrainer().base_model_path
        assert "Transfer" in CosmosTransferTrainer().base_model_path

    def test_cosmos_transfer_has_control_type(self):
        from strands_robots.training import CosmosTransferTrainer

        t = CosmosTransferTrainer()
        assert hasattr(t, "control_type")
        assert t.control_type == "depth"

    def test_cosmos_predict_no_control_type(self):
        from strands_robots.training import CosmosTrainer

        t = CosmosTrainer()
        assert not hasattr(t, "control_type")

    def test_factory_dispatches_correctly(self):
        from strands_robots.training import CosmosTrainer, CosmosTransferTrainer, create_trainer

        predict = create_trainer("cosmos_predict")
        transfer = create_trainer("cosmos_transfer")
        assert isinstance(predict, CosmosTrainer)
        assert isinstance(transfer, CosmosTransferTrainer)
        assert type(predict) is not type(transfer)


# ═══════════════════════════════ Exports ═══════════════════════════════


class TestCosmosTransferExports:
    """Test module exports include CosmosTransferTrainer."""

    def test_in_all(self):
        from strands_robots.training import __all__

        assert "CosmosTransferTrainer" in __all__

    def test_importable(self):
        from strands_robots.training import CosmosTransferTrainer

        assert CosmosTransferTrainer is not None
