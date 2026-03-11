"""Base classes and utilities for training providers."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


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
