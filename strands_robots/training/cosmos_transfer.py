"""Cosmos Transfer 2.5 ControlNet post-training trainer."""

import logging
import warnings
from typing import Any, Dict, Optional

from ._base import TrainConfig, Trainer, _build_cosmos_subprocess_env

logger = logging.getLogger(__name__)


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

        for env_var in ("COSMOS_TRANSFER_PATH", "COSMOS_TRANSFER2_PATH"):
            cosmos_root = os.environ.get(env_var)
            if cosmos_root:
                candidate_paths.append(os.path.join(cosmos_root, "scripts", "train.py"))
                candidate_paths.append(
                    os.path.join(cosmos_root, "scripts", "finetune.py")
                )
                candidate_paths.append(os.path.join(cosmos_root, "train.py"))

        candidate_paths.extend(
            [
                os.path.expanduser("~/cosmos-transfer2.5/scripts/train.py"),
                os.path.expanduser("~/cosmos-transfer2.5/scripts/finetune.py"),
                os.path.expanduser("~/cosmos-transfer2/scripts/train.py"),
                "/opt/cosmos-transfer2/scripts/train.py",
                "/opt/cosmos-transfer2.5/scripts/train.py",
            ]
        )

        try:
            import cosmos_transfer2 as _ct2

            ct2_init = os.path.abspath(_ct2.__file__)
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

        if self.control_type not in self.VALID_CONTROL_TYPES:
            raise ValueError(
                f"Invalid control_type '{self.control_type}'. "
                f"Must be one of: {self.VALID_CONTROL_TYPES}"
            )

        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{self.mode}'. " f"Must be one of: {self.VALID_MODES}"
            )

        if self.output_resolution not in ("480", "720", "1080"):
            raise ValueError(
                f"Invalid output_resolution '{self.output_resolution}'. "
                f"Must be one of: '480', '720', '1080'"
            )

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

        if not self.freeze_backbone and self.mode != "control_finetuning":
            warnings.warn(
                "CosmosTransferTrainer: freeze_backbone=False requires "
                "~80GB+ VRAM. Ensure you are running on Thor (132GB) or "
                "equivalent hardware. L40S (46GB) will likely OOM.",
                UserWarning,
                stacklevel=2,
            )

        self.train_script_path = self._resolve_train_script(train_script_path)
        if self.train_script_path:
            logger.info("✅ Cosmos Transfer training script: %s", self.train_script_path)
        else:
            logger.warning(
                "⚠️ Cosmos Transfer training script not found on disk. "
                "Set train_script_path, COSMOS_TRANSFER_PATH env var, "
                "or install cosmos-transfer2.5 source."
            )

        logger.info("🔄 Cosmos Transfer Trainer: %s", base_model_path)
        logger.info("📁 Dataset: %s", dataset_path)
        logger.info("🎯 Mode: %s, Control: %s", mode, control_type)

    @property
    def provider_name(self) -> str:
        return "cosmos_transfer"

    def _infer_config_name(self) -> str:
        """Infer experiment name from model variant, mode, and control type."""
        model_lower = (self.base_model_path or "7b").lower()
        size = "14b" if "14b" in model_lower else "7b"

        if self.robot_variant:
            return f"cosmos_transfer2_{size}_{self.robot_variant}_{self.control_type}"
        elif self.mode == "sim2real":
            return f"cosmos_transfer2_{size}_{self.output_resolution}p_sim2real_{self.control_type}"
        elif self.mode == "domain_adaptation":
            return f"cosmos_transfer2_{size}_{self.output_resolution}p_domain_adapt"
        else:
            return f"cosmos_transfer2_{size}_{self.output_resolution}p_controlnet_{self.control_type}"

    def _resolve_config_file(self) -> str:
        """Return the config file path for cosmos-transfer2.5 training."""
        import os

        if self.config_file:
            return self.config_file

        if self.mode == "control_finetuning":
            rel = "cosmos_transfer2/configs/controlnet/config.py"
        elif self.mode == "sim2real":
            rel = "cosmos_transfer2/configs/sim2real/config.py"
        else:
            rel = "cosmos_transfer2/configs/training/config.py"

        script_path = self.train_script_path
        if script_path and os.path.isfile(script_path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(script_path)))
            full = os.path.join(repo_root, rel)
            if not os.path.isfile(full):
                logger.warning("Config file not found at %s, using path anyway", full)

        return rel

    def train(self, **kwargs) -> Dict[str, Any]:
        """Launch Cosmos Transfer 2.5 post-training.

        Uses cosmos-transfer2.5 training scripts via subprocess.
        Supports multi-GPU training via torchrun.
        """
        import os
        import subprocess
        import sys

        num_gpus = self.config.num_gpus
        script_path = self._resolve_train_script(self.train_script_path)

        if script_path and os.path.isfile(script_path):
            if num_gpus > 1:
                cmd = ["torchrun", f"--nproc_per_node={num_gpus}", script_path]
            else:
                cmd = [sys.executable, script_path]
        else:
            logger.warning(
                "Could not resolve cosmos-transfer2.5 training script path. "
                "Falling back to 'python -m scripts.train' which may fail. "
                "Set train_script_path or COSMOS_TRANSFER_PATH to fix."
            )
            if num_gpus > 1:
                cmd = ["torchrun", f"--nproc_per_node={num_gpus}", "-m", "scripts.train"]
            else:
                cmd = [sys.executable, "-m", "scripts.train"]

        config_file = self._resolve_config_file()
        cmd.extend(["--config", config_file])
        cmd.append("--")

        cmd.append(f"experiment={self.config_name}")
        cmd.append(f"trainer.max_iter={self.config.max_steps}")
        cmd.append(f"checkpoint.save_iter={self.config.save_steps}")
        cmd.append(
            "job.wandb_mode=disabled"
            if not self.config.use_wandb
            else "job.wandb_mode=online"
        )

        if self.base_model_path:
            cmd.append(f"checkpoint.load_path={self.base_model_path}")
        if self.dataset_path:
            cmd.append(f"data_train.dataset_path={self.dataset_path}")
        if self.config.output_dir:
            cmd.append(f"job.path_local={self.config.output_dir}")

        cmd.append(f"model.control_type={self.control_type}")
        cmd.append(f"model.control_weight={self.control_weight}")
        cmd.append(f"model.guidance={self.guidance}")

        if self.mode == "control_finetuning":
            if not self.freeze_backbone:
                logger.warning(
                    "control_finetuning mode requires freeze_backbone=True. "
                    "Overriding freeze_backbone=False."
                )
            cmd.append("model.freeze_backbone=True")
            cmd.append("model.freeze_controlnet=False")
        elif self.freeze_backbone:
            cmd.append("model.freeze_backbone=True")
        else:
            cmd.append("model.freeze_backbone=False")

        if self.freeze_controlnet:
            cmd.append("model.freeze_controlnet=True")

        if self.robot_variant:
            cmd.append(f"model.robot_variant={self.robot_variant}")

        cmd.append(f"model.output_resolution={self.output_resolution}")

        for key, value in kwargs.items():
            cmd.append(f"{key}={value}")

        logger.info("🚀 Launching Cosmos Transfer post-training (%s)...", self.mode)
        logger.info("   Config: %s", config_file)
        logger.info("   Experiment: %s", self.config_name)
        logger.info("   Control: %s, Weight: %s", self.control_type, self.control_weight)
        logger.info("   GPUs: %s, Steps: %s", num_gpus, self.config.max_steps)
        logger.info("   Command: %s...", ' '.join(cmd[:8]))

        sub_env = _build_cosmos_subprocess_env(
            script_path,
            extra_env_vars=["COSMOS_TRANSFER_PATH", "COSMOS_TRANSFER2_PATH"],
        )
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
        """Evaluate a post-trained Cosmos Transfer checkpoint."""
        logger.info(
            "📊 Cosmos Transfer evaluation — use cosmos-transfer2.5 eval scripts"
        )
        return {"provider": "cosmos_transfer", "status": "not_implemented"}
