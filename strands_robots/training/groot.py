"""GR00T N1.6 fine-tuning trainer."""

import logging
from typing import Any, Dict, Optional

from ._base import TrainConfig, Trainer

logger = logging.getLogger(__name__)


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

        logger.info("🧠 GR00T Trainer: %s", base_model_path)
        logger.info("📁 Dataset: %s", dataset_path)
        logger.info("🤖 Embodiment: %s", embodiment_tag)

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

        logger.info("🚀 Launching GR00T training: %s...", ' '.join(cmd[:6]))

        result = subprocess.run(cmd, capture_output=False)
        return {
            "provider": "groot",
            "returncode": result.returncode,
            "output_dir": self.config.output_dir,
            "status": "completed" if result.returncode == 0 else "failed",
        }

    def evaluate(self, checkpoint_path: str = None, **kwargs) -> Dict[str, Any]:
        """Evaluate a GR00T checkpoint (open-loop)."""
        logger.info(
            "📊 GR00T evaluation not yet integrated — use Isaac-GR00T eval scripts directly"
        )
        return {"provider": "groot", "status": "not_implemented"}
