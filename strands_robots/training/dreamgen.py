"""DreamGen IDM and VLA trainers."""

import logging
from typing import Any, Dict, Optional

from ._base import TrainConfig, Trainer

logger = logging.getLogger(__name__)


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
        logger.info("📁 Dataset: %s", dataset_path)
        logger.info("🤖 Config: %s", data_config)

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

        logger.info("🎬 DreamGen VLA Trainer: %s", base_model_path)
        logger.info("📁 Dataset: %s", dataset_path)

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
            cmd.extend(
                [
                    "--lora-rank",
                    str(self.lora_rank),
                    "--lora-alpha",
                    str(self.lora_alpha),
                ]
            )

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
