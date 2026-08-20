"""Training abstraction - post-tune any policy provider natively.

Peer of ``strands_robots.policies``: where ``Policy`` is inference, ``Trainer``
is post-tuning. Selected by the SAME provider name via ``create_trainer``.

Usage::

    from strands_robots.training import create_trainer, TrainSpec

    trainer = create_trainer("lerobot_local")
    spec = TrainSpec(
        dataset_root="/tmp/my_dataset",
        base_model="lerobot/act_aloha_sim",
        output_dir="/tmp/ft_out",
        steps=20000,
    )
    problems = trainer.validate(spec)
    if not problems:
        result = trainer.train(spec)
"""

from strands_robots.training.base import Trainer, TrainResult, TrainSpec
from strands_robots.training.factory import (
    create_trainer,
    import_trainer_class,
    list_trainers,
    register_trainer,
)
from strands_robots.training.reward import (
    compute_rabc_weights,
    load_reward_model,
    reward_progress,
)

__all__ = [
    "Trainer",
    "TrainSpec",
    "TrainResult",
    "create_trainer",
    "register_trainer",
    "list_trainers",
    "import_trainer_class",
    "compute_rabc_weights",
    "load_reward_model",
    "reward_progress",
]


# Register the from-scratch RL trainers (strands_robots.training.rl). These live
# in a torch-importing subpackage, so they are wired through the factory's lazy
# loader rather than imported here - keeping ``import strands_robots.training``
# torch-free. ``create_trainer("ppo")`` resolves the loader on first use.
def _load_ppo_trainer() -> type[Trainer]:
    from strands_robots.training.rl.ppo import PpoTrainer

    return PpoTrainer


register_trainer("ppo", _load_ppo_trainer)


def _load_fast_sac_trainer() -> type[Trainer]:
    from strands_robots.training.rl.fast_sac import FastSacTrainer

    return FastSacTrainer


register_trainer("fast_sac", _load_fast_sac_trainer)


# Register the SageMaker managed-job transport. Auto-discovery would resolve
# ``create_trainer("sagemaker")`` from the module name alone, but registration
# is what puts the provider in ``list_trainers()`` (there is no policy-side
# ``"trainer"`` block to list it - "sagemaker" is a training transport with no
# paired inference provider). The loader keeps the import deferred; the module
# itself defers boto3 to first use via ``require_optional``.
def _load_sagemaker_trainer() -> type[Trainer]:
    from strands_robots.training.sagemaker import SagemakerTrainer

    return SagemakerTrainer


register_trainer("sagemaker", _load_sagemaker_trainer)
