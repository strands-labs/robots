#!/usr/bin/env python3
"""
Training Abstraction for Universal VLA Training

Provides a unified interface for training any VLA model:
- groot: Fine-tune GR00T N1.6 (Isaac-GR00T's launch_finetune)
- lerobot: Train LeRobot policies (ACT, Pi0, SmolVLA, Diffusion, etc.)
- dreamgen_idm: Train DreamGen IDM (inverse dynamics model)
- dreamgen_vla: Fine-tune GR00T VLA from GR00T-Dreams
- cosmos_predict: Post-train Cosmos Predict 2.5 for robot policy

Usage:
    from strands_robots.training import create_trainer

    # GR00T N1.6 fine-tuning
    trainer = create_trainer("groot",
        base_model_path="nvidia/GR00T-N1.6-3B",
        dataset_path="/data/my_trajectories",
        embodiment_tag="so100",
    )
    trainer.train()

    # LeRobot (Pi0, ACT, etc.)
    trainer = create_trainer("lerobot",
        policy_type="pi0",
        dataset_repo_id="lerobot/so100_wipe",
        output_dir="./outputs",
    )
    trainer.train()

    # Cosmos Predict 2.5 post-training
    trainer = create_trainer("cosmos_predict",
        base_model_path="nvidia/Cosmos-Predict2.5-2B",
        dataset_path="/data/robot_trajectories",
        mode="policy",
    )
    trainer.train()
"""

import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _discover_nvidia_cuda_lib_paths() -> list:
    """Discover pip-installed NVIDIA CUDA shared library directories.

    On systems where the system CUDA toolkit version differs from what
    packages like ``transformer_engine`` or ``megatron-core`` were compiled
    against, the required shared libraries (e.g. ``libcublas.so.12``) are
    often pip-installed under ``site-packages/nvidia/*/lib/``.

    This function discovers those directories so they can be added to
    ``LD_LIBRARY_PATH`` for subprocesses.

    Returns:
        List of directory paths containing NVIDIA CUDA shared libraries.
    """
    import glob
    import os
    import sys

    nvidia_lib_dirs = []

    # Check all site-packages directories (system, user, venv)
    for site_dir in sys.path:
        if not os.path.isdir(site_dir):
            continue
        nvidia_base = os.path.join(site_dir, "nvidia")
        if os.path.isdir(nvidia_base):
            # Pattern: nvidia/*/lib/ (e.g. nvidia/cublas/lib/, nvidia/cuda_runtime/lib/)
            for lib_dir in glob.glob(os.path.join(nvidia_base, "*", "lib")):
                if os.path.isdir(lib_dir):
                    nvidia_lib_dirs.append(lib_dir)

    # Also check ~/.local/lib/pythonX.Y/site-packages/nvidia/*/lib/
    user_sp = os.path.expanduser(
        f"~/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}" f"/site-packages/nvidia"
    )
    if os.path.isdir(user_sp):
        for lib_dir in glob.glob(os.path.join(user_sp, "*", "lib")):
            if os.path.isdir(lib_dir) and lib_dir not in nvidia_lib_dirs:
                nvidia_lib_dirs.append(lib_dir)

    return nvidia_lib_dirs


@dataclass
class TrainConfig:
    """Universal training configuration."""

    dataset_path: str = ""
    output_dir: str = "./outputs"
    max_steps: int = 10000
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    num_gpus: int = 1
    save_steps: int = 1000
    save_total_limit: int = 5
    use_wandb: bool = False
    dataloader_num_workers: int = 2
    seed: int = 42
    resume: bool = False


class Trainer(ABC):
    """Abstract base class for VLA model trainers."""

    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        """Run the training loop.

        Returns:
            Dict with training results (loss, steps, checkpoint_path, etc.)
        """
        pass

    @abstractmethod
    def evaluate(self, **kwargs) -> Dict[str, Any]:
        """Run evaluation.

        Returns:
            Dict with evaluation metrics
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get trainer provider name."""
        pass


class Gr00tTrainer(Trainer):
    """Fine-tune GR00T N1.6 VLA models using Isaac-GR00T's training pipeline.

    Wraps Isaac-GR00T's `launch_finetune.py` with the FinetuneConfig interface.
    Supports single-node fine-tuning with configurable model components
    (LLM, visual encoder, projector, diffusion model).

    Example:
        trainer = Gr00tTrainer(
            base_model_path="nvidia/GR00T-N1.6-3B",
            dataset_path="/data/trajectories",
            embodiment_tag="so100",
            data_config="so100_dualcam",
        )
        result = trainer.train()
    """

    def __init__(
        self,
        base_model_path: str = "nvidia/GR00T-N1.6-3B",
        dataset_path: str = "",
        embodiment_tag: str = "new_embodiment",
        data_config: Optional[str] = None,
        modality_config_path: Optional[str] = None,
        tune_llm: bool = False,
        tune_visual: bool = False,
        tune_projector: bool = True,
        tune_diffusion_model: bool = True,
        state_dropout_prob: float = 0.0,
        config: Optional[TrainConfig] = None,
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.dataset_path = dataset_path
        self.embodiment_tag = embodiment_tag
        self.data_config = data_config
        self.modality_config_path = modality_config_path
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.state_dropout_prob = state_dropout_prob
        self.config = config or TrainConfig(dataset_path=dataset_path)
        self.extra_kwargs = kwargs

        logger.info(f"🧠 GR00T Trainer: {base_model_path}")
        logger.info(f"📁 Dataset: {dataset_path}")
        logger.info(f"🤖 Embodiment: {embodiment_tag}")

    @property
    def provider_name(self) -> str:
        return "groot"

    def train(self, **kwargs) -> Dict[str, Any]:
        """Launch GR00T N1.6 fine-tuning via Isaac-GR00T pipeline."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "-m",
            "gr00t.experiment.launch_finetune",
            "--base-model-path",
            self.base_model_path,
            "--dataset-path",
            self.dataset_path,
            "--embodiment-tag",
            self.embodiment_tag,
            "--output-dir",
            self.config.output_dir,
            "--max-steps",
            str(self.config.max_steps),
            "--global-batch-size",
            str(self.config.batch_size),
            "--learning-rate",
            str(self.config.learning_rate),
            "--weight-decay",
            str(self.config.weight_decay),
            "--warmup-ratio",
            str(self.config.warmup_ratio),
            "--num-gpus",
            str(self.config.num_gpus),
            "--save-steps",
            str(self.config.save_steps),
            "--save-total-limit",
            str(self.config.save_total_limit),
            "--dataloader-num-workers",
            str(self.config.dataloader_num_workers),
        ]

        if self.tune_llm:
            cmd.append("--tune-llm")
        if self.tune_visual:
            cmd.append("--tune-visual")
        if not self.tune_projector:
            cmd.append("--no-tune-projector")
        if not self.tune_diffusion_model:
            cmd.append("--no-tune-diffusion-model")
        if self.config.use_wandb:
            cmd.append("--use-wandb")
        if self.config.resume:
            cmd.append("--resume")
        if self.modality_config_path:
            cmd.extend(["--modality-config-path", self.modality_config_path])
        if self.data_config:
            cmd.extend(["--data-config", self.data_config])

        logger.info(f"🚀 Launching GR00T training: {' '.join(cmd[:6])}...")

        result = subprocess.run(cmd, capture_output=False)
        return {
            "provider": "groot",
            "returncode": result.returncode,
            "output_dir": self.config.output_dir,
            "status": "completed" if result.returncode == 0 else "failed",
        }

    def evaluate(self, checkpoint_path: str = None, **kwargs) -> Dict[str, Any]:
        """Evaluate a GR00T checkpoint (open-loop)."""
        logger.info("📊 GR00T evaluation not yet integrated — use Isaac-GR00T eval scripts directly")
        return {"provider": "groot", "status": "not_implemented"}


class LerobotTrainer(Trainer):
    """Train LeRobot policies (ACT, Pi0, SmolVLA, Diffusion, etc.).

    Supports two modes:
    1. **In-process** (default): Uses LeRobot's training internals directly.
       Full access to model, optimizer, dataloader — supports callbacks,
       custom eval, and programmatic checkpointing.
    2. **Subprocess**: Wraps `lerobot_train` CLI for isolation.

    Example:
        trainer = LerobotTrainer(
            policy_type="pi0",
            pretrained_name_or_path="lerobot/pi0",
            dataset_repo_id="lerobot/so100_wipe",
        )
        result = trainer.train()
    """

    def __init__(
        self,
        policy_type: str = "act",
        pretrained_name_or_path: Optional[str] = None,
        dataset_repo_id: str = "",
        config: Optional[TrainConfig] = None,
        in_process: bool = True,
        eval_env: Optional[str] = None,
        eval_freq: int = 20000,
        save_freq: int = 20000,
        log_freq: int = 200,
        rename_map: Optional[Dict[str, str]] = None,
        **kwargs,
    ):
        self.policy_type = policy_type
        self.pretrained_name_or_path = pretrained_name_or_path
        self.dataset_repo_id = dataset_repo_id
        self.config = config or TrainConfig()
        self.in_process = in_process
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_freq = save_freq
        self.log_freq = log_freq
        self.rename_map = rename_map or {}
        self.extra_kwargs = kwargs

        # In-process training state
        self._train_config = None
        self._policy = None
        self._dataset = None
        self._optimizer = None

        logger.info(f"🤗 LeRobot Trainer: {policy_type} ({'in-process' if in_process else 'subprocess'})")
        logger.info(f"📁 Dataset: {dataset_repo_id}")

    @property
    def provider_name(self) -> str:
        return "lerobot"

    def _build_train_config(self) -> Any:
        """Build a LeRobot TrainPipelineConfig from our TrainConfig.

        Bridges strands-robots TrainConfig → LeRobot TrainPipelineConfig.
        """
        try:
            from pathlib import Path

            from lerobot.configs.default import DatasetConfig, WandBConfig
            from lerobot.configs.policies import PreTrainedConfig
            from lerobot.configs.train import TrainPipelineConfig

            # Build dataset config
            dataset_cfg = DatasetConfig(repo_id=self.dataset_repo_id)

            # Build policy config (from pretrained or type)
            policy_cfg = None
            if self.pretrained_name_or_path:
                policy_cfg = PreTrainedConfig.from_pretrained(self.pretrained_name_or_path)
                policy_cfg.pretrained_path = Path(self.pretrained_name_or_path)

            # Build wandb config
            wandb_cfg = WandBConfig(enable=self.config.use_wandb)

            # Construct full training config
            train_cfg = TrainPipelineConfig(
                dataset=dataset_cfg,
                policy=policy_cfg,
                output_dir=Path(self.config.output_dir),
                seed=self.config.seed,
                batch_size=self.config.batch_size,
                steps=self.config.max_steps,
                num_workers=self.config.dataloader_num_workers,
                eval_freq=self.eval_freq,
                save_freq=self.save_freq,
                log_freq=self.log_freq,
                wandb=wandb_cfg,
                rename_map=self.rename_map,
            )

            self._train_config = train_cfg
            return train_cfg

        except ImportError as e:
            logger.warning(f"LeRobot training config not available: {e}")
            return None

    def train(self, **kwargs) -> Dict[str, Any]:
        """Launch LeRobot training.

        Uses in-process training by default for full control.
        Falls back to subprocess if in_process=False or if internal API fails.
        """
        if self.in_process:
            try:
                return self._train_in_process(**kwargs)
            except Exception as e:
                logger.warning(f"In-process training failed: {e}, falling back to subprocess")
                return self._train_subprocess(**kwargs)
        else:
            return self._train_subprocess(**kwargs)

    def _train_in_process(self, **kwargs) -> Dict[str, Any]:
        """Run training using LeRobot's internal training loop.

        Uses lerobot.scripts.lerobot_train.train() directly.
        Provides full access to the training loop, model, optimizer, etc.
        """
        try:
            from lerobot.scripts.lerobot_train import train as lerobot_train_fn
        except ImportError:
            raise ImportError("LeRobot training script not available. " "Install with: pip install lerobot")

        # Build the training config
        train_cfg = self._build_train_config()
        if train_cfg is None:
            raise RuntimeError("Failed to build LeRobot training config")

        logger.info(f"🚀 Starting in-process LeRobot training: {self.policy_type}")
        logger.info(f"   Steps: {self.config.max_steps}, Batch: {self.config.batch_size}")
        logger.info(f"   Output: {self.config.output_dir}")

        # Run the actual LeRobot training function
        try:
            lerobot_train_fn(train_cfg)
            return {
                "provider": "lerobot",
                "mode": "in_process",
                "policy_type": self.policy_type,
                "output_dir": self.config.output_dir,
                "steps": self.config.max_steps,
                "status": "completed",
            }
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {
                "provider": "lerobot",
                "mode": "in_process",
                "policy_type": self.policy_type,
                "status": "failed",
                "error": str(e),
            }

    def _train_subprocess(self, **kwargs) -> Dict[str, Any]:
        """Launch LeRobot training via CLI subprocess."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--dataset.repo_id={self.dataset_repo_id}",
            f"--output_dir={self.config.output_dir}",
            f"--steps={self.config.max_steps}",
            f"--batch_size={self.config.batch_size}",
            f"--seed={self.config.seed}",
            f"--num_workers={self.config.dataloader_num_workers}",
            f"--save_freq={self.save_freq}",
            f"--eval_freq={self.eval_freq}",
            f"--log_freq={self.log_freq}",
        ]

        if self.pretrained_name_or_path:
            cmd.append(f"--policy={self.pretrained_name_or_path}")

        if self.config.use_wandb:
            cmd.append("--wandb.enable=true")

        if self.config.resume:
            cmd.append("--resume=true")

        logger.info(f"🚀 Launching LeRobot training (subprocess): {self.policy_type}...")

        result = subprocess.run(cmd, capture_output=False)
        return {
            "provider": "lerobot",
            "mode": "subprocess",
            "policy_type": self.policy_type,
            "returncode": result.returncode,
            "output_dir": self.config.output_dir,
            "status": "completed" if result.returncode == 0 else "failed",
        }

    def evaluate(self, checkpoint_path: str = None, **kwargs) -> Dict[str, Any]:
        """Run evaluation on a trained checkpoint.

        Uses LeRobot's eval infrastructure if a Gymnasium env is configured.
        """
        if not checkpoint_path:
            checkpoint_path = self.config.output_dir

        if self.eval_env:
            try:
                return self._evaluate_in_process(checkpoint_path, **kwargs)
            except Exception as e:
                logger.warning(f"In-process eval failed: {e}")

        logger.info("📊 Use lerobot eval scripts for evaluation")
        return {"provider": "lerobot", "checkpoint": checkpoint_path, "status": "not_implemented"}

    def _evaluate_in_process(self, checkpoint_path: str, **kwargs) -> Dict[str, Any]:
        """Run evaluation using LeRobot's eval scripts."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_eval",
            f"--policy={checkpoint_path}",
            f"--env={self.eval_env}",
            f"--seed={self.config.seed}",
        ]

        logger.info(f"📊 Evaluating: {checkpoint_path} on {self.eval_env}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1h timeout for eval

        return {
            "provider": "lerobot",
            "checkpoint": checkpoint_path,
            "env": self.eval_env,
            "returncode": result.returncode,
            "output": result.stdout[:2000],
            "status": "completed" if result.returncode == 0 else "failed",
        }


class DreamgenIdmTrainer(Trainer):
    """Train DreamGen IDM (Inverse Dynamics Model) from GR00T-Dreams.

    The IDM learns to predict action chunks from consecutive video frame pairs.
    Uses SigLIP-2 vision encoder + flow matching DiT architecture.

    Example:
        trainer = DreamgenIdmTrainer(
            dataset_path="/data/robot_trajectories",
            data_config="so100",
            embodiment_tag="so100",
        )
        result = trainer.train()
    """

    def __init__(
        self,
        dataset_path: str = "",
        data_config: str = "gr1_arms_only",
        embodiment_tag: str = "new_embodiment",
        config: Optional[TrainConfig] = None,
        tune_action_head: bool = True,
        video_backend: str = "decord",
        **kwargs,
    ):
        self.dataset_path = dataset_path
        self.data_config = data_config
        self.embodiment_tag = embodiment_tag
        self.config = config or TrainConfig(dataset_path=dataset_path)
        self.tune_action_head = tune_action_head
        self.video_backend = video_backend
        self.extra_kwargs = kwargs

        # Warn about commonly misused parameters from Issue #73
        _unknown_params = {"idm_architecture", "action_dim"} & set(kwargs.keys())
        if _unknown_params:
            logger.warning(
                f"⚠️ DreamgenIdmTrainer received unknown parameters: {_unknown_params}. "
                f"These are NOT supported and will be silently ignored. "
                f"Supported params: dataset_path, data_config, embodiment_tag, "
                f"tune_action_head, video_backend. "
                f"The IDM architecture is controlled by the underlying training scripts."
            )

        logger.info("🎬 DreamGen IDM Trainer")
        logger.info(f"📁 Dataset: {dataset_path}")
        logger.info(f"🤖 Config: {data_config}")

    @property
    def provider_name(self) -> str:
        return "dreamgen_idm"

    def train(self, **kwargs) -> Dict[str, Any]:
        """Launch IDM training via GR00T-Dreams scripts."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "-m",
            "scripts.idm_training",
            "--dataset-path",
            self.dataset_path,
            "--data-config",
            self.data_config,
            "--embodiment-tag",
            self.embodiment_tag,
            "--output-dir",
            self.config.output_dir,
            "--max-steps",
            str(self.config.max_steps),
            "--batch-size",
            str(self.config.batch_size),
            "--learning-rate",
            str(self.config.learning_rate),
            "--num-gpus",
            str(self.config.num_gpus),
            "--save-steps",
            str(self.config.save_steps),
            "--video-backend",
            self.video_backend,
        ]

        if self.tune_action_head:
            cmd.append("--tune-action-head")

        logger.info("🚀 Launching DreamGen IDM training...")

        result = subprocess.run(cmd, capture_output=False)
        return {
            "provider": "dreamgen_idm",
            "returncode": result.returncode,
            "output_dir": self.config.output_dir,
            "status": "completed" if result.returncode == 0 else "failed",
        }

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        return {"provider": "dreamgen_idm", "status": "not_implemented"}


class DreamgenVlaTrainer(Trainer):
    """Fine-tune GR00T VLA using GR00T-Dreams pipeline (with LoRA support).

    This is the GR00T-Dreams variant which supports training on
    neural trajectories (synthetic data from video world models).

    Example:
        trainer = DreamgenVlaTrainer(
            base_model_path="nvidia/GR00T-N1-2B",
            dataset_path="/data/neural_trajectories",
            data_config="so100",
        )
        result = trainer.train()
    """

    def __init__(
        self,
        base_model_path: str = "nvidia/GR00T-N1-2B",
        dataset_path: str = "",
        data_config: str = "gr1_arms_only",
        embodiment_tag: str = "new_embodiment",
        tune_llm: bool = False,
        tune_visual: bool = True,
        tune_projector: bool = True,
        tune_diffusion_model: bool = True,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        config: Optional[TrainConfig] = None,
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.dataset_path = dataset_path
        self.data_config = data_config
        self.embodiment_tag = embodiment_tag
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.config = config or TrainConfig(dataset_path=dataset_path)
        self.extra_kwargs = kwargs

        logger.info(f"🎬 DreamGen VLA Trainer: {base_model_path}")
        logger.info(f"📁 Dataset: {dataset_path}")

    @property
    def provider_name(self) -> str:
        return "dreamgen_vla"

    def train(self, **kwargs) -> Dict[str, Any]:
        """Launch GR00T-Dreams VLA fine-tuning."""
        import subprocess
        import sys

        cmd = [
            sys.executable,
            "-m",
            "scripts.gr00t_finetune",
            "--base-model-path",
            self.base_model_path,
            "--dataset-path",
            self.dataset_path,
            "--data-config",
            self.data_config,
            "--embodiment-tag",
            self.embodiment_tag,
            "--output-dir",
            self.config.output_dir,
            "--max-steps",
            str(self.config.max_steps),
            "--batch-size",
            str(self.config.batch_size),
            "--learning-rate",
            str(self.config.learning_rate),
            "--num-gpus",
            str(self.config.num_gpus),
            "--save-steps",
            str(self.config.save_steps),
        ]

        if self.tune_llm:
            cmd.append("--tune-llm")
        if self.tune_visual:
            cmd.append("--tune-visual")
        if not self.tune_projector:
            cmd.append("--no-tune-projector")
        if not self.tune_diffusion_model:
            cmd.append("--no-tune-diffusion-model")
        if self.lora_rank > 0:
            cmd.extend(["--lora-rank", str(self.lora_rank), "--lora-alpha", str(self.lora_alpha)])

        logger.info("🚀 Launching DreamGen VLA training...")

        result = subprocess.run(cmd, capture_output=False)
        return {
            "provider": "dreamgen_vla",
            "returncode": result.returncode,
            "output_dir": self.config.output_dir,
            "status": "completed" if result.returncode == 0 else "failed",
        }

    def evaluate(self, **kwargs) -> Dict[str, Any]:
        return {"provider": "dreamgen_vla", "status": "not_implemented"}


class CosmosTrainer(Trainer):
    """Post-train Cosmos Predict 2.5 for robot policy.

    Wraps the cosmos-predict2 post-training pipeline for fine-tuning
    Cosmos world models into robot action policies.

    Supports three modes:
    - "policy": Post-train for direct action prediction (LIBERO/RoboCasa/custom)
    - "action_conditioned": Post-train for action-conditioned video generation
    - "world_model": Post-train base world model on domain-specific video

    Example:
        trainer = CosmosTrainer(
            base_model_path="nvidia/Cosmos-Predict2.5-2B",
            dataset_path="/data/robot_trajectories",
            mode="policy",
            suite="libero",
        )
        result = trainer.train()
    """

    @staticmethod
    def _resolve_train_script(explicit_path: Optional[str] = None) -> Optional[str]:
        """Resolve the cosmos-predict2 training script path.

        Resolution order:
            1. Explicit ``train_script_path`` if provided
            2. ``COSMOS_PREDICT2_PATH`` environment variable
            3. Well-known filesystem locations (``~/cosmos-predict2.5/``, etc.)
            4. Derive from ``cosmos_oss`` editable-install location
               (``cosmos_oss`` lives at ``<repo>/packages/cosmos-oss/cosmos_oss/``
               so the repo root is three directories up)
            5. ``None`` (caller falls back to ``python -m`` invocation)

        Returns:
            Absolute path to ``scripts/train.py``, or ``None`` if not found.
        """
        import os

        if explicit_path and os.path.isfile(explicit_path):
            return os.path.abspath(explicit_path)

        candidate_paths: list[str] = []

        # 2. Environment variable
        cosmos_root = os.environ.get("COSMOS_PREDICT2_PATH")
        if cosmos_root:
            candidate_paths.append(os.path.join(cosmos_root, "scripts", "train.py"))

        # 3. Well-known filesystem locations
        candidate_paths.extend(
            [
                os.path.expanduser("~/cosmos-predict2.5/scripts/train.py"),
                os.path.expanduser("~/cosmos-predict2/scripts/train.py"),
                "/opt/cosmos-predict2/scripts/train.py",
            ]
        )

        # 4. Derive from cosmos_oss editable-install location
        try:
            import cosmos_oss as _oss

            # cosmos_oss.__file__ is <repo>/packages/cosmos-oss/cosmos_oss/__init__.py
            # repo root is 3 parents up
            oss_init = os.path.abspath(_oss.__file__)
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(oss_init))))
            candidate_paths.append(os.path.join(repo_root, "scripts", "train.py"))
        except (ImportError, AttributeError, TypeError):
            pass

        for candidate in candidate_paths:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

        return None

    def __init__(
        self,
        base_model_path: str = "nvidia/Cosmos-Predict2.5-2B",
        dataset_path: str = "",
        mode: str = "policy",
        suite: str = "libero",
        config: Optional[TrainConfig] = None,
        config_file: Optional[str] = None,
        config_name: Optional[str] = None,
        use_lora: bool = False,
        lora_rank: int = 16,
        freeze_backbone: bool = True,
        chunk_size: int = 16,
        action_dim: int = 7,
        train_script_path: Optional[str] = None,
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.dataset_path = dataset_path
        self.mode = mode
        self.suite = suite
        self.config = config or TrainConfig(dataset_path=dataset_path)
        self.config_file = config_file
        self.config_name = config_name or self._infer_config_name()
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.freeze_backbone = freeze_backbone
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.extra_kwargs = kwargs

        # Warn about parameters that are accepted but not yet forwarded to
        # the cosmos-predict2 CLI.  The exact Hydra config keys for LoRA and
        # backbone freezing have not been verified against the cosmos-predict2
        # schema, so we warn instead of silently dropping them.
        if self.use_lora:
            warnings.warn(
                "CosmosTrainer: use_lora=True is set but LoRA configuration "
                "is not yet forwarded to the cosmos-predict2 CLI.  The LoRA "
                "settings will be silently ignored.  To pass LoRA overrides "
                "manually, use train(model_lora_enabled=True, "
                "model_lora_rank=<rank>).",
                UserWarning,
                stacklevel=2,
            )
        if not self.freeze_backbone:
            warnings.warn(
                "CosmosTrainer: freeze_backbone=False is set but this "
                "parameter is not yet forwarded to the cosmos-predict2 CLI.  "
                "The backbone will remain frozen (default).  To override, "
                "pass the config key directly: train(model_freeze_backbone=False).",
                UserWarning,
                stacklevel=2,
            )

        # Eagerly resolve training script path so callers can inspect it
        # before calling train() and get a clear error early.
        self.train_script_path = self._resolve_train_script(train_script_path)
        if self.train_script_path:
            logger.info(f"✅ Cosmos training script: {self.train_script_path}")
        else:
            logger.warning(
                "⚠️ Cosmos training script not found on disk. "
                "Set train_script_path, COSMOS_PREDICT2_PATH env var, "
                "or install cosmos-predict2 source to ~/cosmos-predict2.5/"
            )

        logger.info(f"🌌 Cosmos Trainer: {base_model_path}")
        logger.info(f"📁 Dataset: {dataset_path}")
        logger.info(f"🎯 Mode: {mode}, Suite: {suite}")

    @property
    def provider_name(self) -> str:
        return "cosmos_predict"

    def _build_subprocess_env(self, script_path: Optional[str] = None) -> dict:
        """Build environment variables for the Cosmos training subprocess.

        The ``cosmos_predict2._src`` internal package lives inside the
        cosmos-predict2 repo source tree but is **not** included in the
        installed wheel.  When the training script is launched as a
        subprocess it therefore fails with ``ModuleNotFoundError:
        No module named 'cosmos_predict2._src'``.

        This method derives the repo root from the resolved training-script
        path and prepends it to ``PYTHONPATH`` so the subprocess can find
        the full source tree (including ``_src``).
        """
        import os

        env = os.environ.copy()

        # Derive repo root from script path: <repo>/scripts/train.py → <repo>
        repo_root = None
        sp = script_path or self.train_script_path
        if sp and os.path.isfile(sp):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(sp)))

        if repo_root is None:
            # Fallback: try COSMOS_PREDICT2_PATH env var
            repo_root = os.environ.get("COSMOS_PREDICT2_PATH")

        if repo_root and os.path.isdir(repo_root):
            existing = env.get("PYTHONPATH", "")
            # Also add sub-packages that cosmos-predict2 expects on the path
            extra_paths = [repo_root]
            for subpkg in ("packages/cosmos-oss", "packages/cosmos-cuda"):
                subpkg_path = os.path.join(repo_root, subpkg)
                if os.path.isdir(subpkg_path):
                    extra_paths.append(subpkg_path)
            new_pythonpath = os.pathsep.join(extra_paths)
            if existing:
                new_pythonpath = new_pythonpath + os.pathsep + existing
            env["PYTHONPATH"] = new_pythonpath
            logger.info(f"✅ Cosmos subprocess PYTHONPATH prepended: {repo_root}")

        # Ensure CUDA shared libraries are findable in the subprocess.
        # On systems with CUDA 13+ installed, packages compiled against
        # CUDA 12 (e.g. transformer_engine) look for libcublas.so.12 which
        # may not be on the default linker path.  We add standard CUDA
        # library directories AND pip-installed nvidia/*/lib/ paths to
        # LD_LIBRARY_PATH.
        # Pip-installed NVIDIA CUDA library directories come FIRST.
        # Packages like nvidia-cublas-cu12 install libcublas.so.12 under
        # site-packages/nvidia/cublas/lib/.  When the system CUDA toolkit
        # is a different major version (e.g. CUDA 13 vs compiled CUDA 12),
        # the pip libraries must take priority to provide correct versioned
        # symbols (e.g. libcudart.so.12 with `libcudart.so.12` version tag).
        nvidia_pip_paths = _discover_nvidia_cuda_lib_paths()

        cuda_lib_candidates = [
            os.environ.get("CUDA_HOME", "/usr/local/cuda") + "/lib64",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/lib/x86_64-linux-gnu",
        ]
        system_cuda_paths = [p for p in cuda_lib_candidates if os.path.isdir(p)]

        # Pip NVIDIA paths first (correct versioned CUDA 12 libs),
        # then system CUDA paths (fallback for other libs)
        cuda_paths = nvidia_pip_paths + system_cuda_paths

        if cuda_paths:
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            new_ld = os.pathsep.join(cuda_paths)
            if existing_ld:
                new_ld = new_ld + os.pathsep + existing_ld
            env["LD_LIBRARY_PATH"] = new_ld

        return env

    def _infer_config_name(self) -> str:
        """Infer Cosmos experiment name from model and mode.

        The experiment name selects a registered configuration inside the
        cosmos-predict2 config system (Hydra ConfigStore).
        """
        model_lower = (self.base_model_path or "2b").lower()
        size = "14b" if "14b" in model_lower else "2b"

        if self.mode == "policy":
            return f"cosmos_predict2_{size}_480p_{self.suite}"
        elif self.mode == "action_conditioned":
            return f"cosmos_predict2_{size}_action_conditioned"
        else:
            # video2world base training
            return f"predict2_video2world_training_{size}_groot_gr1_480"

    def _resolve_config_file(self) -> str:
        """Return the config Python file path relative to the cosmos-predict2 repo root.

        The ``scripts/train.py`` CLI takes ``--config=<path>`` where ``<path>``
        is a Python file path relative to the repo root, e.g.:
          ``cosmos_predict2/_src/predict2/cosmos_policy/config/config.py``

        Returns:
            Relative config file path string.
        """
        import os

        if self.config_file:
            return self.config_file

        if self.mode == "policy":
            rel = "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py"
        else:
            # video2world / action_conditioned
            rel = "cosmos_predict2/_src/predict2/configs/video2world/config.py"

        # Verify the file exists when we have a repo root
        script_path = self.train_script_path
        if script_path and os.path.isfile(script_path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(script_path)))
            full = os.path.join(repo_root, rel)
            if not os.path.isfile(full):
                logger.warning(f"Config file not found at {full}, using path anyway")

        return rel

    def train(self, **kwargs) -> Dict[str, Any]:
        """Launch Cosmos Predict 2.5 post-training.

        Uses cosmos-predict2 ``scripts/train.py`` via subprocess.
        Supports multi-GPU training via torchrun.

        The CLI interface is::

            python scripts/train.py --config=<config.py> -- \
                experiment=<name> trainer.max_iter=<N> ...

        where ``--config`` is a Python config file path relative to the repo
        root, and positional arguments after ``--`` are Imaginaire4-style
        ``key=value`` config overrides.
        """
        import os
        import subprocess
        import sys

        num_gpus = self.config.num_gpus

        # Re-resolve at train time in case env/filesystem changed since init.
        script_path = self._resolve_train_script(self.train_script_path)

        # ── Build the base command (torchrun or python) ──
        if script_path and os.path.isfile(script_path):
            if num_gpus > 1:
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={num_gpus}",
                    script_path,
                ]
            else:
                cmd = [sys.executable, script_path]
        else:
            logger.warning(
                "Could not resolve cosmos-predict2 training script path. "
                "Falling back to 'python -m scripts.train' which may fail. "
                "Set train_script_path or COSMOS_PREDICT2_PATH to fix."
            )
            if num_gpus > 1:
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={num_gpus}",
                    "-m",
                    "scripts.train",
                ]
            else:
                cmd = [sys.executable, "-m", "scripts.train"]

        # ── --config flag: path to the Python config file ──
        config_file = self._resolve_config_file()
        cmd.extend(["--config", config_file])

        # ── Separator for positional config overrides ──
        cmd.append("--")

        # ── Positional key=value config overrides ──
        cmd.append(f"experiment={self.config_name}")
        cmd.append(f"trainer.max_iter={self.config.max_steps}")
        cmd.append(f"checkpoint.save_iter={self.config.save_steps}")
        cmd.append("job.wandb_mode=disabled" if not self.config.use_wandb else "job.wandb_mode=online")

        # Set checkpoint load_path for the base model (HuggingFace model ID)
        if self.base_model_path:
            cmd.append(f"checkpoint.load_path={self.base_model_path}")

        # Set dataset path override if provided
        if self.dataset_path:
            cmd.append(f"data_train.dataset_path={self.dataset_path}")

        # Set output directory
        if self.config.output_dir:
            cmd.append(f"job.path_local={self.config.output_dir}")

        # Pass through any extra key=value overrides from kwargs
        for key, value in kwargs.items():
            cmd.append(f"{key}={value}")

        logger.info(f"🚀 Launching Cosmos post-training ({self.mode})...")
        logger.info(f"   Config: {config_file}")
        logger.info(f"   Experiment: {self.config_name}")
        logger.info(f"   GPUs: {num_gpus}, Steps: {self.config.max_steps}")
        logger.info(f"   Command: {' '.join(cmd[:8])}...")

        # ── Derive working directory and subprocess env ──
        sub_env = self._build_subprocess_env(script_path)
        cwd = None
        if script_path and os.path.isfile(script_path):
            cwd = os.path.dirname(os.path.dirname(os.path.abspath(script_path)))

        result = subprocess.run(cmd, capture_output=False, env=sub_env, cwd=cwd)
        return {
            "provider": "cosmos_predict",
            "mode": self.mode,
            "suite": self.suite,
            "returncode": result.returncode,
            "output_dir": self.config.output_dir,
            "status": "completed" if result.returncode == 0 else "failed",
        }

    def evaluate(self, checkpoint_path: str = None, **kwargs) -> Dict[str, Any]:
        """Evaluate a post-trained Cosmos checkpoint."""
        logger.info("📊 Cosmos evaluation — use cosmos_predict2 eval scripts")
        return {"provider": "cosmos_predict", "status": "not_implemented"}


class CosmosTransferTrainer(Trainer):
    """Post-train Cosmos Transfer 2.5 for sim-to-real ControlNet conditioning.

    Wraps the cosmos-transfer2.5 post-training pipeline for fine-tuning
    Cosmos Transfer models on domain-specific simulation-to-reality data.
    Uses ControlNet conditioning (depth, edge, segmentation) to learn
    domain-specific visual transformations.

    Supports three modes:
    - "sim2real": Post-train on sim/real video pairs with control conditioning
    - "domain_adaptation": Adapt the base model to a new visual domain
    - "control_finetuning": Fine-tune only the ControlNet branch weights

    Example:
        trainer = CosmosTransferTrainer(
            base_model_path="nvidia/Cosmos-Transfer2-7B",
            dataset_path="/data/sim_real_pairs",
            control_type="depth",
            mode="sim2real",
        )
        result = trainer.train()

    Hardware Requirements:
        - Training: ~80GB+ VRAM (Thor 132GB preferred)
        - Fine-tuning ControlNet only: ~50GB VRAM (L40S 46GB marginal)
        - Training dataset: paired sim/real videos with control signals
    """

    # Valid control types matching CosmosTransferConfig.VALID_MODEL_VARIANTS
    VALID_CONTROL_TYPES = ("depth", "edge", "seg", "vis")
    VALID_MODES = ("sim2real", "domain_adaptation", "control_finetuning")

    @staticmethod
    def _resolve_train_script(explicit_path: Optional[str] = None) -> Optional[str]:
        """Resolve the cosmos-transfer2.5 training script path.

        Resolution order:
            1. Explicit ``train_script_path`` if provided
            2. ``COSMOS_TRANSFER_PATH`` / ``COSMOS_TRANSFER2_PATH`` env var
            3. Well-known filesystem locations
            4. Derive from ``cosmos_transfer2`` package install location
            5. ``None`` (caller falls back to ``python -m`` invocation)

        Returns:
            Absolute path to the training script, or ``None`` if not found.
        """
        import os

        if explicit_path and os.path.isfile(explicit_path):
            return os.path.abspath(explicit_path)

        candidate_paths: list[str] = []

        # 2. Environment variables (check both naming conventions)
        for env_var in ("COSMOS_TRANSFER_PATH", "COSMOS_TRANSFER2_PATH"):
            cosmos_root = os.environ.get(env_var)
            if cosmos_root:
                candidate_paths.append(os.path.join(cosmos_root, "scripts", "train.py"))
                candidate_paths.append(os.path.join(cosmos_root, "scripts", "finetune.py"))
                candidate_paths.append(os.path.join(cosmos_root, "train.py"))

        # 3. Well-known filesystem locations
        candidate_paths.extend(
            [
                os.path.expanduser("~/cosmos-transfer2.5/scripts/train.py"),
                os.path.expanduser("~/cosmos-transfer2.5/scripts/finetune.py"),
                os.path.expanduser("~/cosmos-transfer2/scripts/train.py"),
                "/opt/cosmos-transfer2/scripts/train.py",
                "/opt/cosmos-transfer2.5/scripts/train.py",
            ]
        )

        # 4. Derive from cosmos_transfer2 package install location
        try:
            import cosmos_transfer2 as _ct2

            ct2_init = os.path.abspath(_ct2.__file__)
            # cosmos_transfer2.__file__ may be <repo>/cosmos_transfer2/__init__.py
            repo_root = os.path.dirname(os.path.dirname(ct2_init))
            candidate_paths.append(os.path.join(repo_root, "scripts", "train.py"))
            candidate_paths.append(os.path.join(repo_root, "scripts", "finetune.py"))
        except (ImportError, AttributeError, TypeError):
            pass

        for candidate in candidate_paths:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)

        return None

    def __init__(
        self,
        base_model_path: str = "nvidia/Cosmos-Transfer2-7B",
        dataset_path: str = "",
        control_type: str = "depth",
        mode: str = "sim2real",
        config: Optional[TrainConfig] = None,
        config_file: Optional[str] = None,
        config_name: Optional[str] = None,
        use_lora: bool = False,
        lora_rank: int = 16,
        freeze_backbone: bool = True,
        freeze_controlnet: bool = False,
        control_weight: float = 1.0,
        guidance: float = 3.0,
        output_resolution: str = "720",
        robot_variant: Optional[str] = None,
        train_script_path: Optional[str] = None,
        **kwargs,
    ):
        self.base_model_path = base_model_path
        self.dataset_path = dataset_path
        self.control_type = control_type
        self.mode = mode
        self.config = config or TrainConfig(dataset_path=dataset_path)
        self.config_file = config_file
        self.output_resolution = output_resolution
        self.robot_variant = robot_variant
        self.config_name = config_name or self._infer_config_name()
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.freeze_backbone = freeze_backbone
        self.freeze_controlnet = freeze_controlnet
        self.control_weight = control_weight
        self.guidance = guidance
        self.extra_kwargs = kwargs

        # Validate control_type
        if self.control_type not in self.VALID_CONTROL_TYPES:
            raise ValueError(
                f"Invalid control_type '{self.control_type}'. " f"Must be one of: {self.VALID_CONTROL_TYPES}"
            )

        # Validate mode
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{self.mode}'. " f"Must be one of: {self.VALID_MODES}")

        # Validate output_resolution
        if self.output_resolution not in ("480", "720", "1080"):
            raise ValueError(
                f"Invalid output_resolution '{self.output_resolution}'. " f"Must be one of: '480', '720', '1080'"
            )

        # Warn about LoRA (not yet verified against cosmos-transfer2 CLI)
        if self.use_lora:
            warnings.warn(
                "CosmosTransferTrainer: use_lora=True is set but LoRA "
                "configuration is not yet verified against the "
                "cosmos-transfer2.5 training CLI. The LoRA settings may "
                "be silently ignored. To pass LoRA overrides manually, "
                "use train(model_lora_enabled=True, model_lora_rank=<rank>).",
                UserWarning,
                stacklevel=2,
            )

        # Warn about unfreezing backbone (very high VRAM)
        # Skip warning for control_finetuning — backbone is always frozen in that mode
        if not self.freeze_backbone and self.mode != "control_finetuning":
            warnings.warn(
                "CosmosTransferTrainer: freeze_backbone=False requires "
                "~80GB+ VRAM. Ensure you are running on Thor (132GB) or "
                "equivalent hardware. L40S (46GB) will likely OOM.",
                UserWarning,
                stacklevel=2,
            )

        # Eagerly resolve training script path
        self.train_script_path = self._resolve_train_script(train_script_path)
        if self.train_script_path:
            logger.info(f"✅ Cosmos Transfer training script: {self.train_script_path}")
        else:
            logger.warning(
                "⚠️ Cosmos Transfer training script not found on disk. "
                "Set train_script_path, COSMOS_TRANSFER_PATH env var, "
                "or install cosmos-transfer2.5 source."
            )

        logger.info(f"🔄 Cosmos Transfer Trainer: {base_model_path}")
        logger.info(f"📁 Dataset: {dataset_path}")
        logger.info(f"🎯 Mode: {mode}, Control: {control_type}")

    @property
    def provider_name(self) -> str:
        return "cosmos_transfer"

    def _infer_config_name(self) -> str:
        """Infer experiment name from model variant, mode, and control type.

        Returns a config/experiment name that matches cosmos-transfer2.5's
        config system.
        """
        model_lower = (self.base_model_path or "7b").lower()
        size = "14b" if "14b" in model_lower else "7b"

        if self.robot_variant:
            # Robot-specific multiview training
            return f"cosmos_transfer2_{size}_{self.robot_variant}_{self.control_type}"
        elif self.mode == "sim2real":
            return f"cosmos_transfer2_{size}_{self.output_resolution}p_sim2real_{self.control_type}"
        elif self.mode == "domain_adaptation":
            return f"cosmos_transfer2_{size}_{self.output_resolution}p_domain_adapt"
        else:
            # control_finetuning
            return f"cosmos_transfer2_{size}_{self.output_resolution}p_controlnet_{self.control_type}"

    def _resolve_config_file(self) -> str:
        """Return the config file path for cosmos-transfer2.5 training.

        Returns:
            Config file path string.
        """
        import os

        if self.config_file:
            return self.config_file

        if self.mode == "control_finetuning":
            rel = "cosmos_transfer2/configs/controlnet/config.py"
        elif self.mode == "sim2real":
            rel = "cosmos_transfer2/configs/sim2real/config.py"
        else:
            rel = "cosmos_transfer2/configs/training/config.py"

        # Verify the file exists when we have a repo root
        script_path = self.train_script_path
        if script_path and os.path.isfile(script_path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(script_path)))
            full = os.path.join(repo_root, rel)
            if not os.path.isfile(full):
                logger.warning(f"Config file not found at {full}, using path anyway")

        return rel

    def _build_subprocess_env(self, script_path: Optional[str] = None) -> dict:
        """Build environment variables for the Cosmos Transfer training subprocess.

        Ensures the cosmos-transfer2.5 repo root and sub-packages are on
        PYTHONPATH, and CUDA shared libraries are findable.
        """
        import os

        env = os.environ.copy()

        # Derive repo root from script path
        repo_root = None
        sp = script_path or self.train_script_path
        if sp and os.path.isfile(sp):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(sp)))

        if repo_root is None:
            for env_var in ("COSMOS_TRANSFER_PATH", "COSMOS_TRANSFER2_PATH"):
                repo_root = os.environ.get(env_var)
                if repo_root:
                    break

        if repo_root and os.path.isdir(repo_root):
            existing = env.get("PYTHONPATH", "")
            extra_paths = [repo_root]
            for subpkg in ("packages/cosmos-oss", "packages/cosmos-cuda"):
                subpkg_path = os.path.join(repo_root, subpkg)
                if os.path.isdir(subpkg_path):
                    extra_paths.append(subpkg_path)
            new_pythonpath = os.pathsep.join(extra_paths)
            if existing:
                new_pythonpath = new_pythonpath + os.pathsep + existing
            env["PYTHONPATH"] = new_pythonpath
            logger.info(f"✅ Cosmos Transfer subprocess PYTHONPATH prepended: {repo_root}")

        # Ensure CUDA shared libraries are findable
        # Pip-installed NVIDIA CUDA library directories come FIRST
        nvidia_pip_paths = _discover_nvidia_cuda_lib_paths()

        cuda_lib_candidates = [
            os.environ.get("CUDA_HOME", "/usr/local/cuda") + "/lib64",
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/targets/x86_64-linux/lib",
            "/usr/lib/x86_64-linux-gnu",
        ]
        system_cuda_paths = [p for p in cuda_lib_candidates if os.path.isdir(p)]
        cuda_paths = nvidia_pip_paths + system_cuda_paths

        if cuda_paths:
            existing_ld = env.get("LD_LIBRARY_PATH", "")
            new_ld = os.pathsep.join(cuda_paths)
            if existing_ld:
                new_ld = new_ld + os.pathsep + existing_ld
            env["LD_LIBRARY_PATH"] = new_ld

        return env

    def train(self, **kwargs) -> Dict[str, Any]:
        """Launch Cosmos Transfer 2.5 post-training.

        Uses cosmos-transfer2.5 training scripts via subprocess.
        Supports multi-GPU training via torchrun.

        The CLI interface follows the same pattern as cosmos-predict2::

            python scripts/train.py --config=<config.py> -- \\
                experiment=<name> trainer.max_iter=<N> ...

        Additional kwargs are passed as key=value config overrides.
        """
        import os
        import subprocess
        import sys

        num_gpus = self.config.num_gpus

        # Re-resolve at train time in case env/filesystem changed
        script_path = self._resolve_train_script(self.train_script_path)

        # Build the base command (torchrun or python)
        if script_path and os.path.isfile(script_path):
            if num_gpus > 1:
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={num_gpus}",
                    script_path,
                ]
            else:
                cmd = [sys.executable, script_path]
        else:
            logger.warning(
                "Could not resolve cosmos-transfer2.5 training script path. "
                "Falling back to 'python -m scripts.train' which may fail. "
                "Set train_script_path or COSMOS_TRANSFER_PATH to fix."
            )
            if num_gpus > 1:
                cmd = [
                    "torchrun",
                    f"--nproc_per_node={num_gpus}",
                    "-m",
                    "scripts.train",
                ]
            else:
                cmd = [sys.executable, "-m", "scripts.train"]

        # --config flag
        config_file = self._resolve_config_file()
        cmd.extend(["--config", config_file])

        # Separator for positional config overrides
        cmd.append("--")

        # Positional key=value config overrides
        cmd.append(f"experiment={self.config_name}")
        cmd.append(f"trainer.max_iter={self.config.max_steps}")
        cmd.append(f"checkpoint.save_iter={self.config.save_steps}")
        cmd.append("job.wandb_mode=disabled" if not self.config.use_wandb else "job.wandb_mode=online")

        # Model / checkpoint
        if self.base_model_path:
            cmd.append(f"checkpoint.load_path={self.base_model_path}")

        # Dataset path
        if self.dataset_path:
            cmd.append(f"data_train.dataset_path={self.dataset_path}")

        # Output directory
        if self.config.output_dir:
            cmd.append(f"job.path_local={self.config.output_dir}")

        # Control-type specific overrides
        cmd.append(f"model.control_type={self.control_type}")
        cmd.append(f"model.control_weight={self.control_weight}")
        cmd.append(f"model.guidance={self.guidance}")

        # Training mode specific
        if self.mode == "control_finetuning":
            if not self.freeze_backbone:
                logger.warning(
                    "control_finetuning mode requires freeze_backbone=True. " "Overriding freeze_backbone=False."
                )
            cmd.append("model.freeze_backbone=True")
            cmd.append("model.freeze_controlnet=False")
        elif self.freeze_backbone:
            cmd.append("model.freeze_backbone=True")
        else:
            cmd.append("model.freeze_backbone=False")

        if self.freeze_controlnet:
            cmd.append("model.freeze_controlnet=True")

        # Robot variant (multiview)
        if self.robot_variant:
            cmd.append(f"model.robot_variant={self.robot_variant}")

        # Resolution
        cmd.append(f"model.output_resolution={self.output_resolution}")

        # Pass through any extra key=value overrides from kwargs
        for key, value in kwargs.items():
            cmd.append(f"{key}={value}")

        logger.info(f"🚀 Launching Cosmos Transfer post-training ({self.mode})...")
        logger.info(f"   Config: {config_file}")
        logger.info(f"   Experiment: {self.config_name}")
        logger.info(f"   Control: {self.control_type}, Weight: {self.control_weight}")
        logger.info(f"   GPUs: {num_gpus}, Steps: {self.config.max_steps}")
        logger.info(f"   Command: {' '.join(cmd[:8])}...")

        # Build working directory and subprocess env
        sub_env = self._build_subprocess_env(script_path)
        cwd = None
        if script_path and os.path.isfile(script_path):
            cwd = os.path.dirname(os.path.dirname(os.path.abspath(script_path)))

        result = subprocess.run(cmd, capture_output=False, env=sub_env, cwd=cwd)
        return {
            "provider": "cosmos_transfer",
            "mode": self.mode,
            "control_type": self.control_type,
            "returncode": result.returncode,
            "output_dir": self.config.output_dir,
            "status": "completed" if result.returncode == 0 else "failed",
        }

    def evaluate(self, checkpoint_path: str = None, **kwargs) -> Dict[str, Any]:
        """Evaluate a post-trained Cosmos Transfer checkpoint.

        Runs inference with the fine-tuned model on a test video to assess
        visual quality of sim-to-real transfer.
        """
        logger.info("📊 Cosmos Transfer evaluation — use cosmos-transfer2.5 eval scripts")
        return {"provider": "cosmos_transfer", "status": "not_implemented"}


def evaluate(
    policy,
    task: str,
    robot_name: str,
    num_episodes: int = 50,
    max_steps_per_episode: int = 1000,
    backend: str = "mujoco",
    render: bool = False,
    seed: int = 42,
    **kwargs,
) -> Dict[str, Any]:
    """Standalone evaluation harness for any policy on any task.

    Runs N episodes of a policy in simulation and reports success rate,
    mean reward, and per-episode statistics.

    Args:
        policy: A Policy instance (from create_policy) or any object with
                get_actions(observation_dict, instruction) method.
        task: Natural language task description.
        robot_name: Robot model name (e.g. "so100", "unitree_g1").
        num_episodes: Number of evaluation episodes.
        max_steps_per_episode: Maximum steps per episode before truncation.
        backend: Simulation backend — "mujoco" (default), "newton", or "isaac".
        render: Whether to render frames during evaluation.
        seed: Random seed for reproducibility.
        **kwargs: Additional kwargs passed to the environment.

    Returns:
        Dict with:
            - success_rate: float (0-100)
            - mean_reward: float
            - num_episodes: int
            - episodes: List[Dict] with per-episode stats
            - policy_provider: str
    """
    import asyncio
    import inspect

    import numpy as np

    episodes = []
    successes = 0
    total_reward = 0.0

    # Create environment based on backend
    if backend == "newton":
        try:
            from strands_robots.newton import NewtonConfig
            from strands_robots.newton.newton_gym_env import NewtonGymEnv

            newton_kwargs = {}
            # Extract Newton-specific config from kwargs
            if "newton_config" in kwargs:
                newton_kwargs["config"] = kwargs.pop("newton_config")
            elif "num_envs" in kwargs or "solver" in kwargs:
                newton_kwargs["config"] = NewtonConfig(
                    num_envs=kwargs.pop("num_envs", 1),
                    solver=kwargs.pop("solver", "mujoco"),
                    device=kwargs.pop("device", "cuda:0"),
                )

            env = NewtonGymEnv(
                robot_name=robot_name,
                task=task,
                render_mode="rgb_array" if render else None,
                max_episode_steps=max_steps_per_episode,
                **newton_kwargs,
            )
            logger.info(f"🚀 Using Newton GPU backend for evaluation (solver={env._config.solver})")
        except ImportError as e:
            logger.warning(f"Newton backend not available: {e}")
            return {
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "num_episodes": 0,
                "episodes": [],
                "policy_provider": getattr(policy, "provider_name", "unknown"),
                "error": f"Newton backend requires newton-sim and warp-lang: {e}",
            }
        except Exception as e:
            logger.warning(f"Newton env creation failed: {e}")
            return {
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "num_episodes": 0,
                "episodes": [],
                "policy_provider": getattr(policy, "provider_name", "unknown"),
                "error": f"Newton environment creation failed: {e}",
            }
    elif backend == "isaac":
        try:
            from strands_robots.isaac.isaac_gym_env import IsaacGymEnv

            isaac_kwargs = {}
            if "num_envs" in kwargs:
                isaac_kwargs["num_envs"] = kwargs.pop("num_envs")
            if "device" in kwargs:
                isaac_kwargs["device"] = kwargs.pop("device")

            env = IsaacGymEnv(
                robot_name=robot_name,
                task=task,
                render_mode="rgb_array" if render else None,
                max_episode_steps=max_steps_per_episode,
                **isaac_kwargs,
            )
            logger.info("🚀 Using Isaac Sim GPU backend for evaluation")
        except ImportError as e:
            logger.warning(f"Isaac backend not available: {e}")
            return {
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "num_episodes": 0,
                "episodes": [],
                "policy_provider": getattr(policy, "provider_name", "unknown"),
                "error": f"Isaac backend requires Isaac Sim: {e}",
            }
        except Exception as e:
            logger.warning(f"Isaac env creation failed: {e}")
            return {
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "num_episodes": 0,
                "episodes": [],
                "policy_provider": getattr(policy, "provider_name", "unknown"),
                "error": f"Isaac environment creation failed: {e}",
            }
    else:
        # MuJoCo backend via StrandsSimEnv
        try:
            from strands_robots.envs import StrandsSimEnv

            env_kwargs = {
                "robot_name": robot_name,
                "task": task,
                "max_episode_steps": max_steps_per_episode,
                "render_mode": "rgb_array" if render else None,
            }
            env_kwargs.update(kwargs)

            env = StrandsSimEnv(**env_kwargs)
        except ImportError:
            return {
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "num_episodes": 0,
                "episodes": [],
                "policy_provider": getattr(policy, "provider_name", "unknown"),
                "error": "gymnasium and mujoco required for evaluation",
            }

    # Create a single event loop for async policies (reused across all steps)
    _async_loop = None
    _owns_loop = False
    if inspect.iscoroutinefunction(getattr(policy, "get_actions", None)):
        try:
            _async_loop = asyncio.get_running_loop()
        except RuntimeError:
            _async_loop = asyncio.new_event_loop()
            _owns_loop = True

    try:
        for ep in range(num_episodes):
            obs, info = env.reset(seed=seed + ep)
            episode_reward = 0.0
            episode_steps = 0
            done = False

            while not done:
                # Build observation dict for the policy
                if isinstance(obs, dict):
                    obs_dict = obs
                else:
                    obs_dict = {"observation.state": obs}

                # Get actions from policy
                try:
                    if _async_loop is not None:
                        if _owns_loop:
                            actions = _async_loop.run_until_complete(policy.get_actions(obs_dict, task))
                        else:
                            # Running inside an existing event loop (e.g. Jupyter, async agent)
                            import concurrent.futures

                            with concurrent.futures.ThreadPoolExecutor() as pool:
                                actions = pool.submit(asyncio.run, policy.get_actions(obs_dict, task)).result()
                    else:
                        actions = policy.get_actions(obs_dict, task)
                except Exception as e:
                    logger.debug(f"Policy error at step {episode_steps}: {e}")
                    # Use zero action on error
                    actions = [{}]

                # Extract raw action vector from action dict
                if isinstance(actions, list) and len(actions) > 0:
                    action_dict = actions[0]
                    if isinstance(action_dict, dict):
                        action_vec = list(action_dict.values())
                        # Filter out metadata keys
                        action_vec = [v for v in action_vec if isinstance(v, (int, float))]
                        action = np.array(action_vec, dtype=np.float32)
                    else:
                        action = np.array(action_dict, dtype=np.float32)
                else:
                    action = env.action_space.sample() * 0  # Zero action

                # Clip to action space
                action = (
                    np.clip(
                        action[: env.action_space.shape[0]],
                        env.action_space.low,
                        env.action_space.high,
                    )
                    if len(action) >= env.action_space.shape[0]
                    else env.action_space.sample() * 0
                )

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                episode_steps += 1
                done = terminated or truncated

            is_success = info.get("is_success", False)
            if is_success:
                successes += 1
            total_reward += episode_reward

            episodes.append(
                {
                    "episode": ep,
                    "steps": episode_steps,
                    "reward": episode_reward,
                    "success": is_success,
                }
            )
    finally:
        if _async_loop is not None and _owns_loop:
            _async_loop.close()
        env.close()

    success_rate = (successes / num_episodes * 100) if num_episodes > 0 else 0.0
    mean_reward = total_reward / num_episodes if num_episodes > 0 else 0.0

    logger.info(f"📊 Evaluation: {num_episodes} episodes, " f"success={success_rate:.1f}%, reward={mean_reward:.2f}")

    return {
        "success_rate": success_rate,
        "mean_reward": mean_reward,
        "num_episodes": num_episodes,
        "episodes": episodes,
        "policy_provider": getattr(policy, "provider_name", "unknown"),
        "task": task,
        "robot_name": robot_name,
    }


def create_trainer(provider: str, **kwargs) -> Trainer:
    """Create a trainer instance based on provider name.

    Convenience kwargs such as ``max_steps``, ``output_dir``, ``batch_size``,
    ``learning_rate``, etc. are automatically extracted and forwarded to a
    :class:`TrainConfig` instance so callers do not need to construct one
    manually.

    Args:
        provider: Trainer provider name
        **kwargs: Provider-specific parameters **and** TrainConfig fields.
            TrainConfig fields (max_steps, output_dir, batch_size, …) are
            extracted and used to build a ``config`` object that is passed
            to the trainer constructor.  If an explicit ``config`` kwarg is
            provided, TrainConfig fields from kwargs are merged into it
            (kwargs take precedence).

    Supported providers:
        - "groot": GR00T N1.6 fine-tuning (Isaac-GR00T)
        - "lerobot": LeRobot policy training (ACT, Pi0, SmolVLA, etc.)
        - "dreamgen_idm": DreamGen IDM training (inverse dynamics)
        - "dreamgen_vla": DreamGen VLA fine-tuning (GR00T-Dreams)
        - "cosmos_predict": Cosmos Predict 2.5 post-training
        - "cosmos_transfer": Cosmos Transfer 2.5 ControlNet post-training (sim-to-real)
        - "robometer": Robometer VLM reward-head fine-tuning

    Returns:
        Trainer instance

    Examples:
        trainer = create_trainer("groot",
            base_model_path="nvidia/GR00T-N1.6-3B",
            dataset_path="/data/trajectories",
            embodiment_tag="so100")

        trainer = create_trainer("lerobot",
            policy_type="pi0",
            dataset_repo_id="lerobot/so100_wipe")

        trainer = create_trainer("dreamgen_idm",
            dataset_path="/data/trajectories",
            data_config="so100")

        trainer = create_trainer("cosmos_predict",
            base_model_path="nvidia/Cosmos-Predict2.5-2B",
            dataset_path="/data/trajectories",
            mode="policy",
            max_steps=5000,
            output_dir="./my_output")

        trainer = create_trainer("cosmos_transfer",
            base_model_path="nvidia/Cosmos-Transfer2-7B",
            dataset_path="/data/sim_real_pairs",
            control_type="depth",
            mode="sim2real")
    """
    import dataclasses as _dc

    providers = {
        "groot": Gr00tTrainer,
        "lerobot": LerobotTrainer,
        "dreamgen_idm": DreamgenIdmTrainer,
        "dreamgen_vla": DreamgenVlaTrainer,
        "cosmos_predict": CosmosTrainer,
        "cosmos_transfer": CosmosTransferTrainer,
    }

    # Lazy-load optional providers that have heavy deps
    if provider == "robometer" and provider not in providers:
        try:
            from strands_robots.robometer import RobometerTrainer

            providers["robometer"] = RobometerTrainer
        except ImportError:
            pass

    if provider not in providers:
        raise ValueError(f"Unknown trainer provider: {provider}. Available: {list(providers.keys())}")

    # ── Extract TrainConfig fields from kwargs ──
    # This allows callers to write:
    #   create_trainer("cosmos_predict", max_steps=10, output_dir="./out")
    # instead of needing:
    #   create_trainer("cosmos_predict", config=TrainConfig(max_steps=10, output_dir="./out"))
    config_field_names = {f.name for f in _dc.fields(TrainConfig)}
    config_overrides = {}
    remaining_kwargs = {}
    for k, v in kwargs.items():
        if k in config_field_names and k != "config":
            config_overrides[k] = v
        else:
            remaining_kwargs[k] = v

    if config_overrides:
        # Merge into existing config if one was provided, otherwise create new
        explicit_config = remaining_kwargs.get("config")
        if explicit_config is not None and isinstance(explicit_config, TrainConfig):
            # Override fields on the existing config
            for k, v in config_overrides.items():
                setattr(explicit_config, k, v)
        else:
            # Build a new TrainConfig from the overrides
            # Start with defaults, then apply dataset_path from kwargs if present
            if "dataset_path" in remaining_kwargs and "dataset_path" not in config_overrides:
                config_overrides["dataset_path"] = remaining_kwargs["dataset_path"]
            remaining_kwargs["config"] = TrainConfig(**config_overrides)

    return providers[provider](**remaining_kwargs)


__all__ = [
    "Trainer",
    "TrainConfig",
    "create_trainer",
    "evaluate",
    "Gr00tTrainer",
    "LerobotTrainer",
    "DreamgenIdmTrainer",
    "DreamgenVlaTrainer",
    "CosmosTrainer",
    "CosmosTransferTrainer",
]
