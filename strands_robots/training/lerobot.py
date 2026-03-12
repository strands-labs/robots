"""LeRobot policy training (ACT, Pi0, Pi0-FAST, SmolVLA, Wall-X, X-VLA, SARM, Diffusion, etc.)."""

import inspect
import logging
from typing import Any, Dict, Optional

from ._base import TrainConfig, Trainer

logger = logging.getLogger(__name__)


class LerobotTrainer(Trainer):
    """Train LeRobot policies (ACT, Pi0, Pi0-FAST, SmolVLA, Wall-X, X-VLA, SARM, Diffusion, etc.).

    Supports two modes:
    1. **In-process** (default): Uses LeRobot's training internals directly.
       Full access to model, optimizer, dataloader — supports callbacks,
       custom eval, and programmatic checkpointing.
    2. **Subprocess**: Wraps `lerobot_train` CLI for isolation.

    Features:
    - PEFT/LoRA support: --policy.peft_config.use_peft=true
    - Real-Time Chunking (RTC): --policy.rtc_config.enabled=true
    - EnvHub environments: --env.type=hub --env.hub_path=username/env
    - 3rd-party policy plugins: pip install lerobot_policy_mypolicy

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
        logger.info("📁 Dataset: %s", dataset_repo_id)

    @property
    def provider_name(self) -> str:
        return "lerobot"

    def _build_train_config(self) -> Any:
        """Build a LeRobot TrainPipelineConfig from our TrainConfig.

        Uses lerobot.configs.train.TrainPipelineConfig directly
        with DatasetConfig and WandBConfig from lerobot.configs.default.
        """
        try:
            from pathlib import Path

            from lerobot.configs.default import DatasetConfig, WandBConfig
            from lerobot.configs.train import TrainPipelineConfig

            # Build dataset config
            dataset_cfg = DatasetConfig(repo_id=self.dataset_repo_id)

            # Build policy config (from pretrained or type)
            policy_cfg = None
            if self.pretrained_name_or_path:
                try:
                    from lerobot.configs.policies import PreTrainedConfig

                    policy_cfg = PreTrainedConfig.from_pretrained(self.pretrained_name_or_path)
                except Exception as e:
                    logger.warning("Could not load pretrained config: %s", e)

            # Build wandb config
            wandb_cfg = WandBConfig(enable=self.config.use_wandb)

            # Build training config kwargs — only pass params that exist
            train_sig = inspect.signature(TrainPipelineConfig)
            train_kwargs = {
                "dataset": dataset_cfg,
                "policy": policy_cfg,
                "seed": self.config.seed,
                "batch_size": self.config.batch_size,
                "steps": self.config.max_steps,
                "num_workers": self.config.dataloader_num_workers,
                "eval_freq": self.eval_freq,
                "save_freq": self.save_freq,
                "log_freq": self.log_freq,
                "wandb": wandb_cfg,
            }

            # output_dir needs Path
            if "output_dir" in train_sig.parameters:
                train_kwargs["output_dir"] = Path(self.config.output_dir)

            # rename_map support
            if "rename_map" in train_sig.parameters and self.rename_map:
                train_kwargs["rename_map"] = self.rename_map

            # Filter to only valid params
            valid_params = set(train_sig.parameters.keys())
            train_kwargs = {k: v for k, v in train_kwargs.items() if k in valid_params}

            train_cfg = TrainPipelineConfig(**train_kwargs)

            self._train_config = train_cfg
            return train_cfg

        except ImportError as e:
            logger.warning("LeRobot training config not available: %s", e)
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
        """Run training using LeRobot's internal training loop."""
        try:
            from lerobot.scripts.lerobot_train import train as lerobot_train_fn
        except ImportError:
            raise ImportError("LeRobot training script not available. Install with: pip install lerobot")

        # Build the training config
        train_cfg = self._build_train_config()
        if train_cfg is None:
            raise RuntimeError("Failed to build LeRobot training config")

        logger.info("🚀 Starting in-process LeRobot training: %s", self.policy_type)
        logger.info(f"   Steps: {self.config.max_steps}, Batch: {self.config.batch_size}")
        logger.info("   Output: %s", self.config.output_dir)

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
            logger.error("Training failed: %s", e)
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
                logger.warning("In-process eval failed: %s", e)

        logger.info("📊 Use lerobot eval scripts for evaluation")
        return {
            "provider": "lerobot",
            "checkpoint": checkpoint_path,
            "status": "not_implemented",
        }

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

        logger.info("📊 Evaluating: %s on %s", checkpoint_path, self.eval_env)
        result = subprocess.run(cmd, capture_output=True, text=True)

        return {
            "provider": "lerobot",
            "checkpoint": checkpoint_path,
            "env": self.eval_env,
            "returncode": result.returncode,
            "output": result.stdout[:2000],
            "status": "completed" if result.returncode == 0 else "failed",
        }
