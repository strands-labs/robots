"""Training Abstraction for Universal VLA Training

Provides a unified interface for training any VLA model:
- groot: Fine-tune GR00T N1.6 (Isaac-GR00T's launch_finetune)
- lerobot: Train LeRobot policies (ACT, Pi0, SmolVLA, Diffusion, etc.)
- dreamgen_idm: Train DreamGen IDM (inverse dynamics model)
- dreamgen_vla: Fine-tune GR00T VLA from GR00T-Dreams
- cosmos_predict: Post-train Cosmos Predict 2.5 for robot policy
- cosmos_transfer: Post-train Cosmos Transfer 2.5 ControlNet (sim-to-real)

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

import dataclasses as _dc
from typing import Any, Dict

from ._base import TrainConfig, Trainer
from .cosmos_predict import CosmosTrainer
from .cosmos_transfer import CosmosTransferTrainer
from .dreamgen import DreamgenIdmTrainer, DreamgenVlaTrainer
from .evaluate import evaluate
from .groot import Gr00tTrainer
from .lerobot import LerobotTrainer


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
    providers = {
        "groot": Gr00tTrainer,
        "lerobot": LerobotTrainer,
        "dreamgen_idm": DreamgenIdmTrainer,
        "dreamgen_vla": DreamgenVlaTrainer,
        "cosmos_predict": CosmosTrainer,
        "cosmos_transfer": CosmosTransferTrainer,
    }

    if provider not in providers:
        raise ValueError(
            f"Unknown trainer provider: {provider}. Available: {list(providers.keys())}"
        )

    # ── Extract TrainConfig fields from kwargs ──
    config_field_names = {f.name for f in _dc.fields(TrainConfig)}
    config_overrides = {}
    remaining_kwargs: Dict[str, Any] = {}
    for k, v in kwargs.items():
        if k in config_field_names and k != "config":
            config_overrides[k] = v
        else:
            remaining_kwargs[k] = v

    if config_overrides:
        explicit_config = remaining_kwargs.get("config")
        if explicit_config is not None and isinstance(explicit_config, TrainConfig):
            for k, v in config_overrides.items():
                setattr(explicit_config, k, v)
        else:
            if (
                "dataset_path" in remaining_kwargs
                and "dataset_path" not in config_overrides
            ):
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
