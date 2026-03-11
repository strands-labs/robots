"""Cosmos Predict 2.5 post-training trainer."""

import logging
import warnings
from typing import Any, Dict, Optional

from ._base import TrainConfig, Trainer, _build_cosmos_subprocess_env

logger = logging.getLogger(__name__)


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
            repo_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(oss_init)))
            )
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

        self.train_script_path = self._resolve_train_script(train_script_path)
        if self.train_script_path:
            logger.info("✅ Cosmos training script: %s", self.train_script_path)
        else:
            logger.warning(
                "⚠️ Cosmos training script not found on disk. "
                "Set train_script_path, COSMOS_PREDICT2_PATH env var, "
                "or install cosmos-predict2 source to ~/cosmos-predict2.5/"
            )

        logger.info("🌌 Cosmos Trainer: %s", base_model_path)
        logger.info("📁 Dataset: %s", dataset_path)
        logger.info("🎯 Mode: %s, Suite: %s", mode, suite)

    @property
    def provider_name(self) -> str:
        return "cosmos_predict"

    def _infer_config_name(self) -> str:
        """Infer Cosmos experiment name from model and mode."""
        model_lower = (self.base_model_path or "2b").lower()
        size = "14b" if "14b" in model_lower else "2b"

        if self.mode == "policy":
            return f"cosmos_predict2_{size}_480p_{self.suite}"
        elif self.mode == "action_conditioned":
            return f"cosmos_predict2_{size}_action_conditioned"
        else:
            return f"predict2_video2world_training_{size}_groot_gr1_480"

    def _resolve_config_file(self) -> str:
        """Return the config Python file path relative to the cosmos-predict2 repo root."""
        import os

        if self.config_file:
            return self.config_file

        if self.mode == "policy":
            rel = "cosmos_predict2/_src/predict2/cosmos_policy/config/config.py"
        else:
            rel = "cosmos_predict2/_src/predict2/configs/video2world/config.py"

        script_path = self.train_script_path
        if script_path and os.path.isfile(script_path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(script_path)))
            full = os.path.join(repo_root, rel)
            if not os.path.isfile(full):
                logger.warning("Config file not found at %s, using path anyway", full)

        return rel

    def train(self, **kwargs) -> Dict[str, Any]:
        """Launch Cosmos Predict 2.5 post-training.

        Uses cosmos-predict2 ``scripts/train.py`` via subprocess.
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
                "Could not resolve cosmos-predict2 training script path. "
                "Falling back to 'python -m scripts.train' which may fail. "
                "Set train_script_path or COSMOS_PREDICT2_PATH to fix."
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

        for key, value in kwargs.items():
            cmd.append(f"{key}={value}")

        logger.info("🚀 Launching Cosmos post-training (%s)...", self.mode)
        logger.info("   Config: %s", config_file)
        logger.info("   Experiment: %s", self.config_name)
        logger.info("   GPUs: %s, Steps: %s", num_gpus, self.config.max_steps)
        logger.info("   Command: %s...", ' '.join(cmd[:8]))

        sub_env = _build_cosmos_subprocess_env(
            script_path, extra_env_vars=["COSMOS_PREDICT2_PATH"]
        )
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
