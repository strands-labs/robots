"""SARM reward-model production helpers - the *producing* half of RA-BC.

Closes the SARM -> RA-BC -> policy loop on top of
:class:`~strands_robots.training.lerobot.LerobotTrainer`:

1. **Train** a SARM (Stage-Aware Reward Model) reward model - ``LerobotTrainer``
   with ``extra['reward_model'] = {'type': 'sarm',
   'annotation_mode': 'single_stage'}`` (``single_stage`` needs no annotations:
   progress is linear over the episode).
2. **Compute** per-frame progress weights from the trained SARM checkpoint -
   :func:`compute_rabc_weights`, which produces the ``sarm_progress.parquet``
   that lerobot's RA-BC sample weighter consumes.
3. **Weight** a policy run with those weights - ``LerobotTrainer`` with
   ``extra['sample_weighting'] = {'type': 'rabc', 'progress_path': <parquet>}``.

:func:`load_reward_model` / :func:`reward_progress` expose a *trained* SARM model
for inference - a dense task-progress score in ``[0, 1]`` - usable as an
eval-time success/score signal.

All of this requires lerobot >= 0.5.2 (the ``lerobot.rewards`` package, where
SARM moved from ``lerobot.policies.sarm`` in earlier 0.5.x). The progress
computation additionally needs ``matplotlib`` (imported by lerobot's
``compute_rabc_weights`` module).
"""

from __future__ import annotations

import importlib.util
import logging
from typing import TYPE_CHECKING, Any

from strands_robots.training.lerobot import _auto_device
from strands_robots.utils import require_optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

# Valid SARM reward heads (compute_rabc_weights ``--head-mode`` choices /
# calculate_rewards ``head_mode``). ``sparse`` is the single-stage default;
# ``dense`` / ``both`` require a dense-annotated SARM (annotation_mode in
# {dense_only, dual}).
_HEAD_MODES = {"sparse", "dense", "both"}


def _require_sarm_progress() -> Any:
    """Import lerobot's in-process SARM progress function, with clear errors.

    lerobot's ``compute_rabc_weights`` module imports ``matplotlib`` at module
    top (for its visualization helpers), so that optional dependency must be
    present even though :func:`compute_rabc_weights` disables visualizations.
    """
    require_optional(
        "matplotlib",
        purpose="SARM RA-BC progress computation (lerobot.rewards.sarm.compute_rabc_weights)",
    )
    try:
        from lerobot.rewards.sarm.compute_rabc_weights import compute_sarm_progress
    except ImportError as e:
        raise ImportError(
            "SARM RA-BC progress computation requires lerobot >= 0.6.0 (the "
            "'lerobot.rewards.sarm' package). Reinstall the extra -- it pulls "
            "lerobot from PyPI:\n"
            "  uv pip install 'strands-robots[lerobot]'"
        ) from e
    return compute_sarm_progress


def compute_rabc_weights(
    reward_model_path: str,
    *,
    dataset_repo_id: str | None = None,
    dataset_root: str | None = None,
    output_path: str | None = None,
    head_mode: str = "sparse",
    device: str | None = None,
    stride: int = 1,
) -> str:
    """Produce the SARM progress parquet that RA-BC policy training consumes.

    Runs a trained SARM reward model over every frame of a dataset and writes a
    ``sarm_progress.parquet`` of per-frame task-progress values in ``[0, 1]``.
    Feed the returned path back into a policy run as
    ``extra['sample_weighting']['progress_path']`` (``type='rabc'``) to weight
    behavior cloning toward high-progress frames - the third step of the
    SARM -> RA-BC -> policy loop.

    Runs **in-process** (calls lerobot's ``compute_sarm_progress`` directly), so
    it never triggers the always-on Hub upload of lerobot's CLI ``main()`` - the
    parquet stays local unless you upload it yourself.

    Args:
        reward_model_path: Path or Hub id of the trained SARM checkpoint (e.g.
            ``LerobotTrainer.latest_checkpoint(output_dir)`` from a reward-model
            run).
        dataset_repo_id: Hugging Face Hub dataset id to score. Mutually
            exclusive with ``dataset_root``; supply exactly one.
        dataset_root: Local LeRobotDataset v3 root to score. Mutually exclusive
            with ``dataset_repo_id``.
        output_path: Destination parquet path. When ``None``, lerobot saves to
            the dataset's local cache directory as ``sarm_progress.parquet``.
        head_mode: SARM head to read - ``"sparse"`` (default), ``"dense"``, or
            ``"both"``. ``dense``/``both`` need a dense-annotated SARM.
        device: Torch device (default auto: cuda > mps > cpu).
        stride: Compute progress every N frames and interpolate the rest
            (``1`` = every frame). Higher values trade resolution for speed.

    Returns:
        Absolute path to the written ``sarm_progress.parquet``.

    Raises:
        ValueError: If neither or both data sources are given, a path/flag-bound
            value is unsafe, or ``head_mode``/``stride`` is invalid.
        ImportError: If lerobot is too old (no ``lerobot.rewards.sarm``) or
            ``matplotlib`` is missing.
    """
    if bool(dataset_repo_id) == bool(dataset_root):
        raise ValueError("pass exactly one data source: dataset_repo_id (Hub) or dataset_root (local v3 root)")
    for label, val in (
        ("reward_model_path", reward_model_path),
        ("output_path", output_path),
        ("dataset_root", dataset_root),
        ("dataset_repo_id", dataset_repo_id),
    ):
        if isinstance(val, str) and val.startswith("-"):
            raise ValueError(f"{label} must not start with '-' (would parse as a stray flag)")
    if head_mode not in _HEAD_MODES:
        raise ValueError(f"head_mode must be one of {sorted(_HEAD_MODES)}, got {head_mode!r}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    compute_sarm_progress = _require_sarm_progress()
    dataset_arg = dataset_repo_id or dataset_root
    dev = device or _auto_device()
    logger.info(
        "Computing SARM progress: dataset=%s model=%s head=%s stride=%d device=%s",
        dataset_arg,
        reward_model_path,
        head_mode,
        stride,
        dev,
    )
    out = compute_sarm_progress(
        dataset_repo_id=dataset_arg,
        reward_model_path=reward_model_path,
        output_path=output_path,
        head_mode=head_mode,
        device=dev,
        num_visualizations=0,
        stride=stride,
    )
    return str(out)


def load_reward_model(
    model_path: str,
    *,
    reward_type: str = "sarm",
    device: str | None = None,
) -> Any:
    """Load a trained reward model (e.g. SARM) for inference.

    Returns lerobot's reward-model object (a ``torch.nn.Module`` subclass of
    ``PreTrainedRewardModel``) with weights restored from ``model_path``. Query
    it with :func:`reward_progress` for a dense task-progress score.

    Args:
        model_path: Path or Hub id of the trained reward-model checkpoint.
        reward_type: Reward-model family (default ``"sarm"``).
        device: Torch device (default auto: cuda > mps > cpu).

    Returns:
        The loaded, device-placed reward model.

    Raises:
        ImportError: If lerobot is too old (no ``lerobot.rewards``).
    """
    if importlib.util.find_spec("lerobot.rewards") is None:
        raise ImportError(
            "reward-model inference requires lerobot >= 0.6.0 (the 'lerobot.rewards' "
            "package). Reinstall 'strands-robots[lerobot]'."
        )
    from lerobot.rewards import make_reward_model, make_reward_model_config

    dev = device or _auto_device()
    cfg = make_reward_model_config(reward_type, pretrained_path=str(model_path), device=dev)
    return make_reward_model(cfg)


def reward_progress(model: Any, batch: Mapping[str, Any]) -> list[float]:
    """Query a loaded reward model for dense task-progress scores in ``[0, 1]``.

    Thin wrapper over the reward model's ``compute_reward(batch)`` that returns
    plain Python floats (one per batch element), so callers don't have to import
    torch or handle tensor/ndarray return types. Useful as an eval-time
    success/score signal alongside the binary/predicate success checks in
    ``PolicyRunner.evaluate``.

    Args:
        model: A reward model from :func:`load_reward_model`.
        batch: The observation batch ``compute_reward`` expects (image/state
            features and a language embedding; see the model's ``compute_reward``
            docstring).

    Returns:
        Progress scores as a flat list of floats, one per batch element.
    """
    rewards = model.compute_reward(batch)
    if hasattr(rewards, "detach"):
        rewards = rewards.detach().to("cpu").flatten().tolist()
    elif hasattr(rewards, "flatten"):
        rewards = rewards.flatten().tolist()
    if not isinstance(rewards, list):
        rewards = [rewards]
    return [float(r) for r in rewards]
