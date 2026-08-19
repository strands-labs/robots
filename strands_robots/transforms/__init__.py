"""Dataset transforms - episode augmentation as a provider surface.

Peer of :mod:`strands_robots.policies` (inference) and
:mod:`strands_robots.training` (post-tuning): where those produce actions and
checkpoints, a :class:`~strands_robots.transforms.base.DatasetTransform`
produces data - LeRobotDataset in, provenance-marked augmented LeRobotDataset
out, with a re-validation gate over every generated episode.

Usage::

    from strands_robots.transforms import create_transform, TransformSpec

    transform = create_transform("mock")
    spec = TransformSpec(
        source_root="/data/recorded",
        output_root="/data/augmented",
        variants_per_episode=4,
        seed=7,
    )
    problems = transform.validate(spec)
    if not problems:
        result = transform.transform(spec)
"""

from strands_robots.transforms.base import (
    DatasetTransform,
    TransformResult,
    TransformSpec,
    derive_variant_seed,
)
from strands_robots.transforms.factory import (
    create_transform,
    import_transform_class,
    list_transforms,
    register_transform,
)
from strands_robots.transforms.provenance import (
    load_provenance,
    provenance_path,
    synthetic_episode_indices,
    write_provenance,
)

__all__ = [
    "DatasetTransform",
    "TransformSpec",
    "TransformResult",
    "derive_variant_seed",
    "create_transform",
    "register_transform",
    "list_transforms",
    "import_transform_class",
    "load_provenance",
    "provenance_path",
    "synthetic_episode_indices",
    "write_provenance",
]


# Built-in transforms, wired through the factory's lazy loaders (the same
# pattern training.__init__ uses for the RL trainers) rather than declared in
# policies.json: ``cosmos_transfer`` is a generation model with no policy
# identity to hang a JSON block on, and ``mock`` stays beside it so both
# resolve the same way.
def _load_mock_transform() -> type[DatasetTransform]:
    from strands_robots.transforms.mock import MockTransform

    return MockTransform


register_transform("mock", _load_mock_transform)


def _load_cosmos_transfer_transform() -> type[DatasetTransform]:
    from strands_robots.transforms.cosmos_transfer import CosmosTransferTransform

    return CosmosTransferTransform


register_transform("cosmos_transfer", _load_cosmos_transfer_transform)
