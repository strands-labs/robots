"""LeRobot trainer - drives ``lerobot.scripts.lerobot_train.train`` AS A LIBRARY.

Builds a typed :class:`lerobot.configs.train.TrainPipelineConfig` and calls
lerobot's ``train(cfg)`` **directly in this interpreter** for any LeRobot-native
policy type (act, diffusion, smolvla, pi0, pi05, ...) OR reward-model type
(sarm, ...). The training *logic* is entirely lerobot's; this adapter only
translates a provider-agnostic
:class:`~strands_robots.training.base.TrainSpec` into the config object, manages
resume, and parses the run for a status verdict.

Why in-process (no ``subprocess``)
----------------------------------
lerobot's entry point is a plain function ``train(cfg)`` whose ``@parser.wrap()``
decorator (lerobot ``configs/parser.py``) short-circuits when the first
positional arg is **already** a ``TrainPipelineConfig`` instance - it uses that
object verbatim and never reads ``sys.argv``. So we build the config as typed
Python objects (``make_policy_config`` / ``make_reward_model_config`` +
``DatasetConfig`` + ``PeftConfig``) and hand it straight to ``train(cfg)``. No
shell, no argv, no second interpreter.

Reward models vs policies
--------------------------
A reward model - e.g. SARM (Stage-Aware Reward Model), the model behind RA-BC -
trains through the SAME ``train(cfg)`` entry point as a policy, but populates
``cfg.reward_model`` instead of ``cfg.policy``; lerobot then follows its
``TrainPipelineConfig.is_reward_model_training`` path. Request it via
``TrainSpec.extra['reward_model']`` (a dict of friendly fields). Requires
lerobot >= 0.6.0 (the ``lerobot.rewards`` package).

Launcher selection (still no shell):
    * 1 GPU / CPU    -> call ``train(cfg)`` directly in-process.
    * >1 GPU, 1 node -> ``elastic_launch`` (torch's programmatic launcher, the
      engine behind ``torchrun``); each worker builds the cfg and calls
      ``train(cfg)``. lerobot creates its own ``Accelerator`` inside, which picks
      up the worker's distributed env. No ``accelerate``/``torchrun`` binary.
    * multi-node     -> rejected in ``validate()`` (needs a per-node launcher).

:meth:`build_command` is retained as a PURE argv-parity helper - it documents
the exact draccus CLI the typed config corresponds to and powers the
``test_native_parity`` drift check. It is NOT used to launch anything.

Grounded against lerobot 0.5.x ``TrainPipelineConfig`` / ``DatasetConfig`` /
``PeftConfig`` / ``SampleWeightingConfig`` / ``RewardModelConfig``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import re
import shutil
import time
import types
import typing
from typing import TYPE_CHECKING, Any

from strands_robots.training._inproc import call_callable, elastic_launch_callable, resume_argv
from strands_robots.training.base import Trainer, TrainResult, TrainSpec
from strands_robots.utils import lerobot_version, validation_split_error, validation_split_fraction

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lerobot.configs.train import TrainPipelineConfig

logger = logging.getLogger(__name__)

# LeRobot-native policy types. Parity with lerobot is DYNAMIC, not a hardcoded
# list: the live set is read from lerobot's ``PreTrainedConfig`` draccus
# ChoiceRegistry (see :func:`_lerobot_policy_types`) - the same zero-maintenance
# discovery reward models / robots / teleops / cameras already use. Any policy
# lerobot ships (act, smolvla, the pi0 family, groot, xvla, and newer additions
# such as eo1, evo1, lingbot_va, molmoact2, wall_x, ...) or a plugin registers is
# reachable with no change here. The static set below is the FALLBACK used ONLY
# when lerobot's registry is unavailable (lerobot not importable), where training
# cannot run anyway but ``validate()`` should still produce a useful offline
# message.
_LEROBOT_POLICY_TYPES_FALLBACK = frozenset(
    {"act", "diffusion", "vqbet", "tdmpc", "smolvla", "pi0", "pi05", "pi0_fast", "groot", "xvla"}
)

_SUPPORTED_METHODS = {"full", "lora", "expert_only"}

# LeRobot policy types whose config exposes ``use_relative_actions`` (the
# relative-action processor pair: a RelativeActionsProcessorStep on the input
# side and the matching AbsoluteActionsProcessorStep on the output side, both
# built from ``config.use_relative_actions`` and saved into the checkpoint's
# pre/post processors). Discovered live per policy type off the config class
# (see :func:`_policy_supports_relative_actions`); the static set is the offline
# FALLBACK. Currently the pi0 family and groot expose the field.
_RELATIVE_ACTION_POLICY_TYPES_FALLBACK = frozenset({"pi0", "pi05", "pi0_fast", "groot"})

# LeRobot policy types whose config exposes ``train_expert_only`` (freeze the
# (V)LM backbone, train only the action expert - the cheap VLA finetune recipe).
# Discovered live per policy type off the config class (see
# :func:`_policy_supports_expert_only`); the static set is the offline FALLBACK.
# Currently pi0, pi05, and smolvla expose the field (pi0_fast does NOT).
_EXPERT_ONLY_POLICY_TYPES_FALLBACK = frozenset({"pi0", "pi05", "smolvla"})

# LeRobot policy types whose config normalizes STATE/ACTION with QUANTILES
# (``NormalizationMode.QUANTILES``). Such a policy needs the dataset's stats to
# carry the quantile keys (q01, q10, q50, q90, q99); a dataset recorded before
# quantile stats existed only has mean/std/min/max, so training either fails or
# mis-normalizes deep inside lerobot's stats plumbing. Discovered live per
# policy type off the config class's ``normalization_mapping`` default
# (see :func:`_policy_uses_quantile_norm`); the static set is the offline
# FALLBACK. Currently molmoact2 and the pi05 family normalize with quantiles.
_QUANTILE_NORM_POLICY_TYPES_FALLBACK = frozenset({"molmoact2", "pi05"})

# Quantile stat keys lerobot writes for ``DEFAULT_QUANTILES`` = [0.01, 0.10,
# 0.50, 0.90, 0.99] (``qNN`` where NN = int(q * 100)). A dataset carries quantile
# stats when ANY feature's stats dict holds one of these keys (mirrors lerobot's
# own ``augment_dataset_quantile_stats.has_quantile_stats``).
_QUANTILE_STAT_KEYS = ("q01", "q10", "q50", "q90", "q99")

#: The LeRobotDataset format version the installed lerobot reads, mirrored for
#: the offline case.
#:
#: lerobot REFUSES to load a dataset whose declared ``codebase_version`` has an
#: older MAJOR than this one: ``check_version_compatibility`` raises
#: ``BackwardCompatibilityError``, and that class only builds a message for
#: exactly v2.1 - for any other older major its constructor itself raises
#: ``NotImplementedError("Contact the maintainer on [Discord](...)")``, naming
#: neither the dataset nor the version. An older MINOR only logs a warning and
#: still loads, so only the major is a refusal.
#:
#: Read live off the installed lerobot (:func:`_lerobot_codebase_version`), which
#: reads it from the same module the loader enforces it in; this is the offline
#: FALLBACK for the ``validate()``-without-lerobot case.
_LEROBOT_CODEBASE_VERSION_FALLBACK = "v3.0"

#: The one dataset format version lerobot's converter accepts as its source.
#: ``BackwardCompatibilityError`` builds a message for exactly this version and
#: raises ``NotImplementedError`` for every other older one, so it is also the
#: boundary between the two remedies the preflight below can honestly offer.
_V21_TO_V30_SOURCE_VERSION = "2.1"

#: lerobot's own converter that upgrades a v2.1 dataset in place to the v3
#: format, quoted by the preflight below so the advice names lerobot's remedy
#: rather than describing one. It is the only conversion lerobot ships; a root
#: older than v2.1 has no automated path forward.
_V21_TO_V30_CONVERTER = "lerobot.scripts.convert_dataset_v21_to_v30"

# RA-BC (Reward-Aligned Behavior Cloning) is surfaced to the agent through the
# ``extra['sample_weighting']`` dict. lerobot >= 0.6.0 configures sample
# weighting via a NESTED ``SampleWeightingConfig`` on ``TrainPipelineConfig``
# (``cfg.sample_weighting``), replacing the flat ``use_rabc`` / ``rabc_*``
# fields of earlier 0.5.x. The friendly keys map 1:1 onto that config's fields,
# so the validated dict is forwarded to ``SampleWeightingConfig(**dict)``.
# ``type`` selects the scheme: lerobot ships ``rabc`` and ``uniform``.
_SAMPLE_WEIGHTING_KEYS = {"type", "progress_path", "head_mode", "kappa", "epsilon"}
_SAMPLE_WEIGHTING_TYPES = {"rabc", "uniform"}

# LeRobot reward-model types (``--reward_model.type`` / make_reward_model_config
# keys). A reward model - e.g. SARM (Stage-Aware Reward Model), the model behind
# RA-BC - trains through the SAME ``lerobot_train.train(cfg)`` entry point as a
# policy, but populates ``cfg.reward_model`` instead of ``cfg.policy``
# (``TrainPipelineConfig.is_reward_model_training``). Requires lerobot >= 0.6.0
# (the ``lerobot.rewards`` package).
#
# Parity with lerobot is DYNAMIC, not a hardcoded list: both the set of reward
# types and each type's configurable fields are read live from lerobot's
# ``RewardModelConfig`` draccus ChoiceRegistry (see :func:`_reward_registry`) -
# the same zero-maintenance discovery Robot / Teleop / Camera / Policy already
# use. Any reward model lerobot ships (sarm, robometer, topreward,
# reward_classifier, ...) or a third-party plugin registers is reachable with no
# change here. The static fallbacks below are used ONLY when ``lerobot.rewards``
# is absent (lerobot < 0.6.0), where reward-model training cannot run anyway but
# ``validate()`` should still produce a useful message offline.
_REWARD_MODEL_TYPES_FALLBACK = frozenset({"sarm", "reward_classifier", "robometer", "topreward"})

# Friendly ``extra['reward_model']`` field names to fall back on when the live
# registry is unavailable. These are SARM's (the offline default type)
# configurable keys; ``type`` is the registry selector, handled separately.
_REWARD_MODEL_FIELDS_FALLBACK = frozenset({"annotation_mode", "image_key", "state_key"})

# SARM annotation modes (configuration_sarm.SARMConfig.annotation_mode):
# ``single_stage`` needs NO annotations (linear progress over the episode).
_SARM_ANNOTATION_MODES = {"single_stage", "dense_only", "dual"}


def _policy_registry() -> dict[str, type] | None:
    """Live ``PreTrainedConfig`` ChoiceRegistry (policy_type -> config class), or
    ``None`` when lerobot is unavailable.

    Importing ``lerobot.policies`` runs each config module's
    ``@PreTrainedConfig.register_subclass`` decorator, which is what populates
    the draccus ChoiceRegistry - querying it before that import yields an empty
    mapping, so the import is the load-bearing step (mirrors
    :func:`_reward_registry`). Returns ``None`` when lerobot is not importable.
    """
    try:
        import lerobot.policies  # noqa: F401  (import for register_subclass side effect)
        from lerobot.configs.policies import PreTrainedConfig
    except ImportError:
        return None
    return dict(PreTrainedConfig.get_known_choices())


def _lerobot_policy_types() -> set[str]:
    """LeRobot-native ``policy.type`` names (live registry, else static fallback)."""
    reg = _policy_registry()
    if reg is None:
        return set(_LEROBOT_POLICY_TYPES_FALLBACK)
    return set(reg)


def _policy_supports_relative_actions(ptype: str) -> bool:
    """Whether ``ptype``'s lerobot config exposes ``use_relative_actions``.

    Probed live off the registry's config *class* (a dataclass field lookup, no
    instantiation - so no device warnings or construction cost), so any policy
    lerobot adds with relative-action support is recognized with zero per-type
    maintenance. Falls back to the documented static set when lerobot's registry
    is unavailable offline.
    """
    reg = _policy_registry()
    if reg is not None and ptype in reg:
        return any(f.name == "use_relative_actions" for f in dataclasses.fields(reg[ptype]))
    return ptype in _RELATIVE_ACTION_POLICY_TYPES_FALLBACK


def _policy_supports_expert_only(ptype: str) -> bool:
    """Whether ``ptype``'s lerobot config exposes ``train_expert_only``.

    ``method="expert_only"`` freezes the VLM and trains only the action expert -
    the standard cheap VLA finetune. lerobot implements it as a per-policy
    ``config.train_expert_only`` field; requesting it on a policy whose config
    lacks the field is either a silent no-op (in-process ``build_config`` guards
    on ``hasattr`` and never flips the flag, so the run silently full-finetunes
    the backbone while reporting success) or a hard draccus error (the CLI
    ``--policy.train_expert_only`` flag), so it must be gated by the policy's
    actual capability. Probed live off the registry's config *class* (a
    dataclass field lookup, no instantiation - so no device warnings or
    construction cost), so any policy lerobot adds with expert-only support is
    recognized with zero per-type maintenance. Falls back to the documented
    static set when lerobot's registry is unavailable offline.
    """
    reg = _policy_registry()
    if reg is not None and ptype in reg:
        return any(f.name == "train_expert_only" for f in dataclasses.fields(reg[ptype]))
    return ptype in _EXPERT_ONLY_POLICY_TYPES_FALLBACK


def _policy_uses_quantile_norm(ptype: str) -> bool:
    """Whether ``ptype``'s lerobot config normalizes any feature with QUANTILES.

    A quantile-normalizing policy (molmoact2, pi05, ...) requires the training
    dataset's stats to carry quantile keys (q01..q99); without them lerobot
    either fails or silently mis-normalizes. Probed live off the registry config
    *class*'s ``normalization_mapping`` field default (a ``dataclasses.field``
    ``default_factory`` call - no config instantiation, so no device warnings or
    construction cost), so any policy lerobot adds with quantile normalization is
    recognized with zero per-type maintenance. Falls back to the documented
    static set when lerobot's registry is unavailable offline.
    """
    reg = _policy_registry()
    if reg is None or ptype not in reg:
        return ptype in _QUANTILE_NORM_POLICY_TYPES_FALLBACK
    for f in dataclasses.fields(reg[ptype]):
        if f.name == "normalization_mapping" and f.default_factory is not dataclasses.MISSING:
            try:
                mapping = f.default_factory()
            except Exception:  # noqa: BLE001 - a broken default falls back to the static set
                return ptype in _QUANTILE_NORM_POLICY_TYPES_FALLBACK
            return any(getattr(mode, "value", mode) == "QUANTILES" for mode in mapping.values())
    return False


def _stats_have_quantiles(stats: dict[str, Any] | None) -> bool:
    """Whether a LeRobotDataset stats dict carries quantile keys for any feature.

    Mirrors lerobot's ``augment_dataset_quantile_stats.has_quantile_stats``: the
    stats are ``{feature: {stat: value}}`` and quantiles are present when ANY
    feature holds one of ``q01..q99``.
    """
    if not isinstance(stats, dict):
        return False
    return any(isinstance(feat, dict) and any(q in feat for q in _QUANTILE_STAT_KEYS) for feat in stats.values())


def _dataset_quantile_stats_present(dataset_root: str) -> bool | None:
    """Tri-state quantile-stats probe for a local LeRobotDataset v3 root.

    Reads ``meta/stats.json`` (lerobot's ``STATS_PATH``, the aggregate stats
    ``load_stats`` feeds to normalization). Returns ``True`` when quantile keys
    are present, ``False`` when the file exists but lacks them, and ``None`` when
    the file is absent or unreadable (unknown - e.g. a Hub dataset with no
    materialized local cache), so a definite miss can be flagged without false
    positives on the unknown case.
    """
    stats_path = os.path.join(dataset_root, "meta", "stats.json")
    try:
        with open(stats_path, encoding="utf-8") as fh:
            stats = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return _stats_have_quantiles(stats)


def _format_version_major(version: str) -> int | None:
    """MAJOR component of a ``vN.M`` / ``N.M`` dataset format version, or None.

    Returns ``None`` for anything this cannot read a leading major out of, so an
    unrecognized version string fails OPEN (no problem reported) rather than
    blocking a possibly-loadable dataset on a cosmetic format - the same posture
    as :func:`~strands_robots.dataset_recorder._huggingface_hub_version_error`.
    """
    match = re.match(r"v?(\d+)", version.strip())
    return int(match.group(1)) if match else None


def _lerobot_codebase_version() -> str:
    """Dataset format version the installed lerobot reads.

    Read off ``lerobot.datasets.dataset_metadata``, the module whose
    ``_load_metadata`` enforces it, so the preflight compares against the very
    constant the loader will compare against. Falls back to the documented
    :data:`_LEROBOT_CODEBASE_VERSION_FALLBACK` when lerobot is unavailable, so
    ``validate()`` still produces a useful offline verdict.
    """
    try:
        from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    except ImportError:
        return _LEROBOT_CODEBASE_VERSION_FALLBACK
    return CODEBASE_VERSION if isinstance(CODEBASE_VERSION, str) else _LEROBOT_CODEBASE_VERSION_FALLBACK


def _dataset_codebase_version(dataset_root: str) -> str | None:
    """Declared ``codebase_version`` of a local LeRobotDataset root, or None.

    Reads ``meta/info.json`` (the file lerobot's ``load_info`` reads, and the one
    :meth:`LerobotTrainer._dataset_total_episodes` already reads for the episode
    count). Returns ``None`` when the file is absent, unreadable, or carries no
    string ``codebase_version`` - the unknown case, e.g. a Hub dataset with no
    materialized local cache - so a DEFINITE mismatch can be reported without
    false positives on the unknown one.
    """
    info_path = os.path.join(dataset_root, "meta", "info.json")
    try:
        with open(info_path, encoding="utf-8") as fh:
            declared = json.load(fh).get("codebase_version")
    except (OSError, json.JSONDecodeError):
        return None
    return declared if isinstance(declared, str) else None


def _reward_registry() -> dict[str, type] | None:
    """Live ``RewardModelConfig`` ChoiceRegistry, or ``None`` when unavailable.

    Importing ``lerobot.rewards`` runs each config module's
    ``@RewardModelConfig.register_subclass`` decorator, which is what populates
    the draccus ChoiceRegistry - querying it before that import yields an empty
    mapping, so the import is the load-bearing step. Returns ``None`` when the
    installed lerobot has no ``lerobot.rewards`` (lerobot < 0.6.0).
    """
    try:
        import lerobot.rewards  # noqa: F401  (import for register_subclass side effect)
        from lerobot.configs.rewards import RewardModelConfig
    except ImportError:
        return None
    return dict(RewardModelConfig.get_known_choices())


def _reward_model_types() -> set[str]:
    """LeRobot-native reward-model type names (live registry, else fallback)."""
    reg = _reward_registry()
    if reg is None:
        return set(_REWARD_MODEL_TYPES_FALLBACK)
    return set(reg)


def _reward_friendly_fields(rtype: str) -> set[str]:
    """Configurable ``extra['reward_model']`` keys for a reward type.

    Dynamic when ``lerobot.rewards`` is importable: the resolved config
    dataclass's OWN (subclass-declared) constructor fields - the per-type
    training knobs. The shared ``RewardModelConfig`` base fields are excluded:
    ``device`` is auto-selected, ``push_to_hub`` is forced off, and
    ``pretrained_path`` is set from ``TrainSpec.base_model``, while the rest
    (Hub metadata, feature specs) are plumbing lerobot derives - none belong in
    the friendly surface. This gives every reward type - not just SARM - full
    knob reach with zero per-type maintenance. Falls back to SARM's documented
    friendly keys when the registry is unavailable (offline / lerobot < 0.6.0),
    where reward-model training cannot run anyway.

    ``type`` (the registry selector) is never a config field and is handled by
    the caller, so it is not part of the returned set.
    """
    reg = _reward_registry()
    if reg is None or rtype not in reg:
        return set(_REWARD_MODEL_FIELDS_FALLBACK)
    from lerobot.configs.rewards import RewardModelConfig

    # Only constructor (init=True) fields are valid make_reward_model_config
    # kwargs; subtracting the base class's fields leaves the per-type knobs.
    base = {f.name for f in dataclasses.fields(RewardModelConfig)}
    own = {f.name for f in dataclasses.fields(reg[rtype]) if f.init}
    return own - base


# Hugging Face Hub dataset id: ``org/name`` (each segment alnum plus ._-). Used
# to gate the agent-supplied ``dataset_repo_id`` before it becomes lerobot's
# ``DatasetConfig.repo_id`` (which load_dataset/HfApi feed to a Hub URL).
_HUB_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class LerobotTrainer(Trainer):
    """Post-tune a LeRobot-native policy or reward model by calling ``lerobot`` train in-process.

    Args:
        policy_type: LeRobot policy type (default ``"act"``). Resolved from
            ``TrainSpec.extra['policy_type']`` if present, else this. Ignored for
            reward-model runs (``TrainSpec.extra['reward_model']`` is set).
        device: Torch device string (default auto: cuda > mps > cpu).
    """

    def __init__(
        self,
        policy_type: str = "act",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.policy_type = policy_type
        self.device = device or _auto_device()

    @property
    def provider_name(self) -> str:
        """Provider identity - pairs with the ``lerobot_local`` inference policy."""
        return "lerobot_local"

    @property
    def hardware_floor(self) -> dict[str, Any]:
        """Advisory floor: ACT fits a consumer 8 GB GPU; large VLAs (pi05) want an L40S."""
        return {"min_gpus": 1, "min_vram_gb": 8, "multinode": False}

    # ---- helpers -----------------------------------------------------------

    def _resolve_policy_type(self, spec: TrainSpec) -> str:
        return str(spec.extra.get("policy_type", self.policy_type))

    def _reward_model_dict(self, spec: TrainSpec) -> dict[str, Any] | None:
        """Resolve the reward-model spec from ``extra['reward_model']``.

        When present, this run trains a lerobot *reward model* (e.g. SARM) rather
        than a policy: :meth:`build_config` populates ``cfg.reward_model`` and
        leaves ``cfg.policy`` unset, and ``lerobot_train`` follows its
        ``is_reward_model_training`` path. The dict carries friendly keys
        (``type``, ``annotation_mode``, ``image_key``, ``state_key``) forwarded to
        ``make_reward_model_config``. Returns the dict unchanged, or ``None`` when
        not requested. Raises ``ValueError`` if present but not a dict (caught by
        ``train`` and surfaced as an error result).
        """
        rm = spec.extra.get("reward_model")
        if rm is None:
            return None
        if not isinstance(rm, dict):
            raise ValueError(
                "extra['reward_model'] must be a dict of reward-model fields, "
                "e.g. {'type': 'sarm', 'annotation_mode': 'single_stage'}"
            )
        return rm

    def _reward_model_type(self, rm: dict[str, Any]) -> str:
        return str(rm.get("type", "sarm"))

    def _run_type_label(self, spec: TrainSpec) -> str:
        """Human-readable description of what this run trains (for logs)."""
        rm = spec.extra.get("reward_model")
        if isinstance(rm, dict):
            return f"reward_model:{self._reward_model_type(rm)}"
        return f"policy:{self._resolve_policy_type(spec)}"

    def _dataset_total_episodes(self, dataset_root: str) -> int | None:
        info = os.path.join(dataset_root, "meta", "info.json")
        try:
            with open(info, encoding="utf-8") as f:
                return int(json.load(f).get("total_episodes"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return None

    def _resume_config_path(self, output_dir: str) -> str | None:
        """Return the resumable ``train_config.json`` FILE path, or None.

        lerobot writes checkpoints to ``<output_dir>/checkpoints/<step|last>/
        pretrained_model/train_config.json``; resume needs the FILE path (it
        derives policy_dir/checkpoint_path from it). This is the resume-wiring
        counterpart of the public :meth:`latest_checkpoint` (which returns the
        loadable DIRECTORY for ``export``/``create_policy``).
        """
        ckpts = os.path.join(output_dir, "checkpoints")
        if not os.path.isdir(ckpts):
            return None
        last = os.path.join(ckpts, "last", "pretrained_model", "train_config.json")
        if os.path.isfile(last):
            return last
        candidates = []
        for name in sorted(os.listdir(ckpts)):
            cfg = os.path.join(ckpts, name, "pretrained_model", "train_config.json")
            if os.path.isfile(cfg):
                candidates.append(cfg)
        return candidates[-1] if candidates else None

    def latest_checkpoint(self, output_dir: str) -> str | None:
        """Return the newest loadable ``pretrained_model`` directory, or None.

        ABC contract: a directory ``create_policy``/``export`` can consume.
        lerobot's loadable artifact is the ``pretrained_model`` dir that holds
        ``model.safetensors`` + ``train_config.json``; we locate it from the
        resume config file's parent. For reward-model runs this is the directory
        :func:`~strands_robots.training.reward.compute_rabc_weights` consumes as
        ``reward_model_path``.
        """
        cfg_file = self._resume_config_path(output_dir)
        return os.path.dirname(cfg_file) if cfg_file else None

    def _dataset_source(self, spec: TrainSpec) -> tuple[str, str | None]:
        """Resolve (repo_id, root) for lerobot's ``DatasetConfig``.

        Two mutually sufficient data sources, mirroring lerobot's own model:

        * Hub dataset (``spec.dataset_repo_id`` set) -> ``repo_id`` is the Hub
          id; ``root`` is the optional local cache dir (``spec.dataset_root`` or
          ``None``). With ``streaming=True`` lerobot streams shards from the Hub
          without a full download - the 50-500 GB disk-blowup fix.
        * Local v3 root (``spec.dataset_root`` only) -> ``repo_id="local"`` and
          ``root`` is the dataset path, unchanged from the record->train loop.
        """
        if spec.dataset_repo_id:
            return spec.dataset_repo_id, (spec.dataset_root or None)
        return "local", spec.dataset_root

    def _val_eval_split(self, spec: TrainSpec) -> float | None:
        """``dataset.eval_split`` reserving the LAST ``val_episodes`` episodes.

        Returns the fraction lerobot needs to hold the tail out of TRAINING and
        run an evaluation pass over it, so ``val_episodes`` yields a validation
        loss rather than only a smaller training set. Paired with a non-zero
        ``eval_steps`` in :meth:`_apply_common_config`; lerobot refuses an
        ``eval_steps`` that has no ``eval_split`` to draw held-out data from.

        Requires a local ``meta/info.json`` to know the episode count, so it is
        a no-op for a Hub dataset with no local cache (``dataset_repo_id`` set,
        ``dataset_root`` empty) - the full Hub dataset is used in that case.
        """
        if spec.val_episodes is None:
            return None
        if not spec.dataset_root:
            return None
        total = self._dataset_total_episodes(spec.dataset_root)
        if total is not None and 0 < spec.val_episodes < total:
            return validation_split_fraction(spec.val_episodes, total)
        return None

    def _unreadable_episode_count_problem(self, spec: TrainSpec) -> str:
        """Refusal text for a ``val_episodes`` whose episode count cannot be read.

        :meth:`_val_eval_split` turns ``val_episodes`` into lerobot's
        ``dataset.eval_split`` FRACTION, which needs the dataset's
        ``total_episodes``. That count is only available from a local
        ``meta/info.json``, so it is unreadable for a Hub dataset with no local
        copy (``dataset_repo_id`` set, ``dataset_root`` empty) and for a
        ``dataset_root`` that is a Hub cache directory nothing has been
        downloaded into yet.

        Both remedies named here are honored by this backend: pointing
        ``dataset_root`` at a populated local copy makes the count readable, and
        the raw ``extra`` passthrough reaches lerobot's own two knobs directly.
        """
        where = (
            f"no readable 'meta/info.json' under dataset_root={spec.dataset_root!r}"
            if spec.dataset_root
            else "no dataset_root was given to read one from"
        )
        return (
            f"{self.provider_name}: val_episodes={spec.val_episodes} cannot be reserved because the "
            f"dataset's episode count is unavailable ({where}). A held-out split is a FRACTION in "
            "lerobot (it holds out ceil(episodes_in_task * eval_split)), so the count is what turns "
            "an episode number into that fraction. Either point dataset_root at a local copy of the "
            "dataset - for a Hub dataset that is its local cache directory - or pass the split "
            "directly with extra={'dataset.eval_split': <fraction>, 'eval_steps': <steps>}."
        )

    def _streaming_validation_split_problem(self, spec: TrainSpec) -> str:
        """Refusal text for ``streaming`` and ``val_episodes`` asked for together.

        Each field is honored on its own, and neither survives the pair.
        :meth:`_val_eval_split` turns ``val_episodes`` into lerobot's
        ``dataset.eval_split``, and a non-zero ``eval_split`` sends lerobot down
        ``make_train_eval_datasets``, which rebuilds BOTH splits as map-style
        ``LeRobotDataset`` objects without consulting ``dataset.streaming`` - the
        ``StreamingLeRobotDataset`` it built first is discarded. So the run
        materializes the whole dataset, which is the outcome ``streaming``
        exists to avoid, and it does so while reporting nothing: an annulled
        stream is indistinguishable from ``streaming=False``.

        Refusing here mirrors :meth:`_unreadable_episode_count_problem`, which
        refuses rather than let a requested validation split be silently
        dropped. Both remedies named here are honored by this backend: dropping
        either field delivers the other one whole, and the raw ``extra``
        passthrough still reaches lerobot's own knobs for a caller who wants the
        combination anyway.
        """
        return (
            f"{self.provider_name}: streaming=True cannot be combined with "
            f"val_episodes={spec.val_episodes}. A held-out split makes lerobot rebuild both "
            "splits as map-style datasets, so the whole dataset is materialized and the stream "
            "is dropped - the disk/RAM blowup streaming exists to avoid. Either set "
            "streaming=False to keep the validation split, or val_episodes=None to keep the "
            "stream."
        )

    def _dataset_total_tasks(self, dataset_root: str) -> int:
        """``total_tasks`` from ``meta/info.json``, or 0 when not recorded."""
        from pathlib import Path

        info_path = Path(dataset_root) / "meta" / "info.json"
        if not info_path.exists():
            return 0
        try:
            with open(info_path) as f:
                total = json.load(f).get("total_tasks")
        except (OSError, ValueError):
            return 0
        return total if isinstance(total, int) and not isinstance(total, bool) else 0

    def _relative_actions(self, spec: TrainSpec) -> bool:
        """Whether to train with relative (delta) actions (``extra['relative_actions']``).

        Relative-action training predicts deltas from the current robot state
        instead of absolute targets - part of the strongest manipulation
        ablations. lerobot implements it as a matched processor pair built from
        ``config.use_relative_actions``: a ``RelativeActionsProcessorStep`` on
        the input side (encode target->delta at train time) and the inverse
        ``AbsoluteActionsProcessorStep`` on the output side (decode delta->target
        at inference). Both are saved into the checkpoint's pre/post processors,
        so deployment via ``lerobot_local`` (which loads the saved processor
        pipeline) restores the inverse decode automatically - no separate
        inference-side wiring is needed.

        Only some policy configs expose ``use_relative_actions`` (currently the
        ``pi0`` family and ``groot``); the supported set is discovered live from
        lerobot's registry (:func:`_policy_supports_relative_actions`), and
        :meth:`validate` rejects the flag for any policy type whose config lacks
        the field rather than letting it become a silent no-op.
        """
        return bool(spec.extra.get("relative_actions", False))

    def _sample_weighting_dict(self, spec: TrainSpec) -> dict[str, Any] | None:
        """Resolve the RA-BC sample-weighting spec from ``extra['sample_weighting']``.

        RA-BC (Reward-Aligned Behavior Cloning) per-sample loss weighting is
        surfaced through the ``extra`` escape hatch as a single
        ``sample_weighting`` dict with friendly keys (``type``,
        ``progress_path``, ``head_mode``, ``kappa``, ``epsilon``). lerobot
        >= 0.6.0 configures it via a nested ``SampleWeightingConfig`` on
        ``TrainPipelineConfig`` (``cfg.sample_weighting``); the friendly keys map
        1:1 onto that config's fields. Example::

            extra={"sample_weighting": {"type": "rabc", "kappa": 0.01,
                                        "head_mode": "sparse",
                                        "progress_path": "/path/sarm_progress.parquet"}}

        Returns the dict unchanged, or ``None`` when not requested. Raises
        ``ValueError`` if the value is present but not a dict (caught by
        ``train`` and surfaced as an error result).
        """
        sw = spec.extra.get("sample_weighting")
        if sw is None:
            return None
        if not isinstance(sw, dict):
            raise ValueError(
                "extra['sample_weighting'] must be a dict of RA-BC fields, e.g. {'type': 'rabc', 'kappa': 0.01}"
            )
        return sw

    # ---- ABC ---------------------------------------------------------------

    def validate(self, spec: TrainSpec) -> list[str]:
        """Pure preflight for a LeRobot policy- or reward-model run.

        Runs the shared input-safety gate, then checks a data source -
        exactly one of a local LeRobotDataset v3 ``dataset_root`` or a Hub
        ``dataset_repo_id`` (for streaming) - an ``output_dir``, a usable run
        size (``steps`` / ``global_batch_size``), single-node only
        (``num_nodes == 1``), a ``val_episodes``
        split below the dataset total and not asked for alongside ``streaming``
        (lerobot's split path is map-style only), a local dataset format version
        the installed lerobot can read, usable LoRA hyperparameters when
        ``method == "lora"``, and that ``lerobot.scripts.lerobot_train``
        is importable. ``extra['reward_model']`` switches to reward-model
        preflight; otherwise the default policy path is checked. Returns the
        problem list; empty means launchable. Read-only.
        """
        problems: list[str] = self._security_problems(spec)

        # Data source: either a local v3 root, or a Hub repo id (streaming the
        # 50-500 GB case without a full download). Exactly one must be present.
        if spec.dataset_repo_id:
            if not _HUB_REPO_ID_RE.match(spec.dataset_repo_id):
                problems.append(
                    f"dataset_repo_id '{spec.dataset_repo_id}' is not a valid Hub id "
                    "(expected 'org/name', alnum/._- segments)"
                )
            # dataset_root is optional here (local cache root); if given, it need
            # not yet contain meta/info.json (the Hub provides metadata).
        elif not spec.dataset_root:
            problems.append("a data source is required: set dataset_root (local v3) or dataset_repo_id (Hub)")
        elif not os.path.isfile(os.path.join(spec.dataset_root, "meta", "info.json")):
            problems.append(
                f"dataset_root is not a LeRobotDataset v3 root "
                f"(missing {os.path.join(spec.dataset_root, 'meta', 'info.json')})"
            )

        if not spec.output_dir:
            problems.append("output_dir is required")

        # A run trains EITHER a policy or a reward model (SARM et al.); the two
        # paths validate differently. extra['reward_model'] selects reward-model
        # training (cfg.reward_model) over the default policy path (cfg.policy).
        rm = spec.extra.get("reward_model")
        if rm is not None and not isinstance(rm, dict):
            problems.append(
                "extra['reward_model'] must be a dict of reward-model fields, "
                "e.g. {'type': 'sarm', 'annotation_mode': 'single_stage'}"
            )
            rm = None
        if isinstance(rm, dict):
            problems.extend(self._validate_reward_model(spec, rm))
        else:
            problems.extend(self._validate_policy(spec))

        problems.extend(self._run_size_problems(spec))
        problems.extend(self._learning_rate_problems(spec))
        problems.extend(self._seed_problems(spec))
        # Captured rather than extended blind: the multi-node refusal below
        # compares num_nodes, which is only a meaningful comparison once this
        # gate has established it IS a count - a string or None would raise out
        # of the comparison instead of being reported.
        topology_problems = self._launch_topology_problems(spec)
        problems.extend(topology_problems)

        if not topology_problems and spec.num_nodes > 1:
            problems.append(
                f"num_nodes={spec.num_nodes}: multi-node lerobot needs a per-node "
                "launcher and cannot run in-process; use num_nodes=1."
            )

        # Captured rather than extended blind, for the same reason as the
        # topology gate above: the two dataset-dependent checks below compare
        # val_episodes and interpolate it into a split fraction, and both are
        # only meaningful once this gate has established that it IS a count.
        val_problems = self._validation_episodes_problems(spec)
        problems.extend(val_problems)

        if not val_problems and spec.val_episodes is not None:
            total = self._dataset_total_episodes(spec.dataset_root) if spec.dataset_root else None
            if total is None:
                # The split is a FRACTION derived from the episode count, so with
                # no count to divide by there is nothing to emit - and a missing
                # eval_split is indistinguishable from "no validation asked for".
                # Refuse instead, naming both remedies, rather than launch a run
                # that trains on every episode and records no validation loss.
                problems.append(self._unreadable_episode_count_problem(spec))
            else:
                if spec.val_episodes >= total:
                    problems.append(f"val_episodes={spec.val_episodes} >= total_episodes={total}")
                split_err = validation_split_error(
                    spec.val_episodes,
                    self._dataset_total_tasks(spec.dataset_root),
                    self.provider_name,
                    passthrough_param="extra",
                )
                if split_err:
                    problems.append(split_err)
                if spec.streaming and self._val_eval_split(spec) is not None:
                    # Both fields were honored in isolation and neither survives
                    # the pair: lerobot's split path rebuilds the dataset
                    # map-style, so the stream is dropped. Refuse rather than
                    # emit a config that silently delivers one of the two.
                    problems.append(self._streaming_validation_split_problem(spec))

        # Shared by BOTH run types: policy and reward-model training load the
        # same dataset, so an unreadable format version refuses either one.
        problems.extend(self._dataset_codebase_version_problems(spec))
        problems.extend(self._lora_hyperparameter_problems(spec))

        # lerobot must be importable to actually train.
        try:
            import importlib.util

            if importlib.util.find_spec("lerobot.scripts.lerobot_train") is None:
                problems.append("lerobot is not installed (no lerobot.scripts.lerobot_train)")
        except Exception:  # noqa: BLE001
            problems.append("lerobot is not installed")

        return problems

    def _validate_policy(self, spec: TrainSpec) -> list[str]:
        """Policy-training preflight (the default, ``cfg.policy`` path)."""
        problems: list[str] = []
        ptype = self._resolve_policy_type(spec)
        if ptype not in _lerobot_policy_types():
            problems.append(
                f"policy_type '{ptype}' is not LeRobot-native (expected one of {sorted(_lerobot_policy_types())})"
            )

        if spec.method not in _SUPPORTED_METHODS:
            problems.append(f"unsupported method '{spec.method}' (expected one of {sorted(_SUPPORTED_METHODS)})")
        if spec.method == "lora" and spec.tune.get("expert_only"):
            problems.append("lora and expert_only are mutually exclusive (both freeze the VLM)")

        if self._relative_actions(spec) and not _policy_supports_relative_actions(ptype):
            supported = sorted(t for t in _lerobot_policy_types() if _policy_supports_relative_actions(t))
            problems.append(
                f"relative_actions is not supported by policy_type '{ptype}' "
                f"(only {supported} expose use_relative_actions); "
                "drop extra['relative_actions'] or pick a supporting policy"
            )

        if spec.method == "expert_only" and not _policy_supports_expert_only(ptype):
            supported = sorted(t for t in _lerobot_policy_types() if _policy_supports_expert_only(t))
            problems.append(
                f"method 'expert_only' is not supported by policy_type '{ptype}' "
                f"(only {supported} expose train_expert_only); "
                "use method='full' or pick a supporting policy"
            )

        sw = spec.extra.get("sample_weighting")
        if sw is not None and not isinstance(sw, dict):
            problems.append(
                "extra['sample_weighting'] must be a dict of RA-BC fields, e.g. {'type': 'rabc', 'kappa': 0.01}"
            )
        elif isinstance(sw, dict):
            for k, v in sw.items():
                if isinstance(v, str) and v.startswith("-"):
                    problems.append(f"sample_weighting['{k}'] must not start with '-' (would parse as a stray flag)")

        problems.extend(self._quantile_stats_problems(spec, ptype))
        return problems

    def _quantile_stats_problems(self, spec: TrainSpec, ptype: str) -> list[str]:
        """Preflight the dataset's quantile stats for a QUANTILES-normalizing policy.

        A policy that normalizes STATE/ACTION with ``NormalizationMode.QUANTILES``
        (molmoact2, pi05, ...) reads the dataset stats' quantile keys (q01..q99)
        to normalize. A dataset recorded before quantile stats existed carries
        only mean/std/min/max, so lerobot either raises or silently
        mis-normalizes deep inside its stats plumbing at train time. This lifts
        that failure to spec-validation time with an actionable message naming
        lerobot's ``augment_dataset_quantile_stats`` remedy.

        Read-only and conservative: it only flags a DEFINITE miss (a local
        ``meta/stats.json`` present but lacking quantile keys). A Hub dataset with
        no materialized local cache is left unflagged (stats unknown without a
        download); its quantiles are validated by lerobot when the shards load.
        Datasets recorded by the current DatasetRecorder already include quantile
        stats (lerobot's ``compute_episode_stats`` computes them by default), so
        they pass cleanly.
        """
        if not (spec.dataset_root and _policy_uses_quantile_norm(ptype)):
            return []
        if _dataset_quantile_stats_present(spec.dataset_root) is not False:
            return []
        repo_id = spec.dataset_repo_id or "<your-dataset-repo-id>"
        return [
            f"policy_type '{ptype}' normalizes STATE/ACTION with QUANTILES but the dataset at "
            f"'{spec.dataset_root}' has no quantile stats (missing q01/q99 in meta/stats.json); "
            "lerobot would mis-normalize or fail at train time. Add quantile stats first: "
            f"python -m lerobot.scripts.augment_dataset_quantile_stats --repo-id={repo_id} "
            f"--root={spec.dataset_root} (datasets recorded with current lerobot already include them)."
        ]

    def _dataset_codebase_version_problems(self, spec: TrainSpec) -> list[str]:
        """Preflight a local dataset root's format version against lerobot's.

        lerobot reads ``meta/info.json``'s ``codebase_version`` and refuses a root
        whose MAJOR is older than its own ``CODEBASE_VERSION``. That refusal is a
        poor place to learn this: only a v2.1 root gets a message naming the
        dataset and the converter, and every OTHER older major dies inside
        ``BackwardCompatibilityError.__init__`` with a bare
        ``NotImplementedError: Contact the maintainer on [Discord](...)`` that
        names neither the dataset nor the version nor the problem. This lifts the
        refusal to spec-validation time with a message that names the root, both
        versions, and the remedy - the same job :meth:`_quantile_stats_problems`
        does for the dataset's quantile stats.

        Read-only and conservative, mirroring that sibling: it flags only a
        DEFINITE mismatch (a local ``meta/info.json`` declaring an older major).
        A Hub dataset with no materialized local cache is left unflagged (its
        metadata is not knowable without a download - ``validate()`` does not
        reach the network), and an unparseable version on either side fails open.
        Only the major is checked, because an older minor loads with a warning.
        """
        if not spec.dataset_root:
            return []
        declared = _dataset_codebase_version(spec.dataset_root)
        if declared is None:
            return []
        current = _lerobot_codebase_version()
        declared_major = _format_version_major(declared)
        current_major = _format_version_major(current)
        if declared_major is None or current_major is None or declared_major >= current_major:
            return []

        # lerobot ships exactly one converter (v2.1 -> v3.0), so an older root
        # gets an honest "no automated path" rather than a command that would
        # fail. The repo id is the CONVERTER's argument, and it is a Hub id: a
        # local-only root has none (``_dataset_source`` calls it "local", which
        # is not a repo anyone can convert), so name the placeholder instead -
        # the same substitution ``_quantile_stats_problems`` makes.
        repo_id = spec.dataset_repo_id or "<your-dataset-repo-id>"
        if declared.strip().lstrip("v") == _V21_TO_V30_SOURCE_VERSION:
            remedy = f"Convert it with lerobot's own converter: python -m {_V21_TO_V30_CONVERTER} --repo-id={repo_id}"
        else:
            remedy = (
                f"lerobot ships no converter for {declared}; its only dataset conversion is "
                f"v2.1 -> v3.0 ({_V21_TO_V30_CONVERTER}). Re-record the dataset, or bring it to "
                "v2.1 first."
            )
        return [
            f"dataset_root '{spec.dataset_root}' declares codebase_version '{declared}' in "
            f"meta/info.json, which lerobot {lerobot_version()} cannot read (it loads "
            f"'{current}' and refuses an older major). {remedy}"
        ]

    def _validate_reward_model(self, spec: TrainSpec, rm: dict[str, Any]) -> list[str]:
        """Reward-model training preflight (the ``cfg.reward_model`` path).

        A reward-model run is fresh, full-parameter training of e.g. SARM; the
        policy-only knobs (RA-BC sample weighting, relative actions, LoRA /
        expert-only) are meaningless for it and are rejected rather than silently
        ignored. RA-BC in particular is the *downstream consumer* of a trained
        SARM (its progress parquet weights POLICY training), so combining it with
        reward-model training is a pipeline-ordering mistake worth naming.
        """
        problems: list[str] = []
        rtype = self._reward_model_type(rm)
        valid_types = _reward_model_types()
        if rtype not in valid_types:
            problems.append(
                f"reward_model type '{rtype}' is not LeRobot-native (expected one of {sorted(valid_types)})"
            )
        # Validate friendly keys against the resolved config's OWN fields (live
        # registry), so each reward type is configurable with its own knobs and
        # cross-type fields (e.g. SARM's annotation_mode on robometer) are
        # rejected with a clear message. Falls back to SARM's keys offline.
        friendly = _reward_friendly_fields(rtype)
        unknown = sorted(k for k in rm if k != "type" and k not in friendly)
        if unknown:
            problems.append(
                f"reward_model type '{rtype}' does not support field(s) {unknown}; "
                f"its configurable fields are {sorted(friendly)}."
            )
        if rtype == "sarm":
            am = rm.get("annotation_mode")
            if am is not None and am not in _SARM_ANNOTATION_MODES:
                problems.append(
                    f"reward_model annotation_mode '{am}' is invalid (expected one of {sorted(_SARM_ANNOTATION_MODES)})"
                )
        for k, v in rm.items():
            if isinstance(v, str) and v.startswith("-"):
                problems.append(f"reward_model['{k}'] must not start with '-' (would parse as a stray flag)")

        if spec.extra.get("sample_weighting") is not None:
            problems.append(
                "extra['sample_weighting'] (RA-BC) weights POLICY training; it does not apply to "
                "reward-model training. Train the reward model first, then feed its progress parquet "
                "to a policy run via extra['sample_weighting']['progress_path']."
            )
        if self._relative_actions(spec):
            problems.append("relative_actions applies to policy training, not reward-model training")
        if spec.method != "full":
            problems.append(
                f"method '{spec.method}' applies to policy training; reward-model training uses method='full'"
            )

        import importlib.util

        if importlib.util.find_spec("lerobot.rewards") is None:
            problems.append(
                "the installed lerobot has no reward-model support (no 'lerobot.rewards'); "
                "requires lerobot >= 0.6.0 -- reinstall 'strands-robots[lerobot]'"
            )
        return problems

    def build_command(self, spec: TrainSpec) -> list[str]:
        """PURE argv-parity helper - the draccus CLI the typed config maps to.

        NOT used to launch training (``train()`` builds a typed config and calls
        lerobot's ``train(cfg)`` directly). Retained so ``test_native_parity``
        can assert our field mapping matches lerobot's real CLI, and as a
        human-readable description of the equivalent command.
        """
        rm = self._reward_model_dict(spec)
        repo_id, root = self._dataset_source(spec)
        cmd = ["lerobot.scripts.lerobot_train", f"--dataset.repo_id={repo_id}"]
        if root:
            cmd.append(f"--dataset.root={root}")
        if rm is not None:
            cmd.extend(self._reward_model_command_flags(rm, spec.base_model))
        else:
            ptype = self._resolve_policy_type(spec)
            cmd.append(f"--policy.type={ptype}")
            cmd.append(f"--policy.device={self.device}")
            cmd.append("--policy.push_to_hub=false")
            if spec.learning_rate is not None:
                cmd.append(f"--policy.optimizer_lr={spec.learning_rate}")
        cmd.extend(
            [
                f"--output_dir={spec.output_dir}",
                f"--job_name={spec.extra.get('job_name', 'strands_ft')}",
                f"--steps={spec.steps}",
                f"--batch_size={spec.global_batch_size}",
                f"--save_freq={spec.save_freq}",
                "--wandb.enable=false",
            ]
        )
        if spec.streaming:
            cmd.append("--dataset.streaming=true")
        if spec.seed is not None:
            cmd.append(f"--seed={spec.seed}")
        split = self._val_eval_split(spec)
        if split is not None:
            cmd.append(f"--dataset.eval_split={split}")
            cmd.append(f"--eval_steps={spec.save_freq if spec.save_freq > 0 else spec.steps}")
        if rm is None:
            if spec.base_model:
                cmd.append(f"--policy.pretrained_path={spec.base_model}")
            if spec.method == "lora":
                cmd.append("--peft.method_type=LORA")
                if spec.lora_r is not None:
                    cmd.append(f"--peft.r={spec.lora_r}")
                if spec.lora_alpha is not None:
                    cmd.append(f"--peft.lora_alpha={spec.lora_alpha}")
                if spec.lora_target_modules is not None:
                    cmd.append(f"--peft.target_modules={spec.lora_target_modules}")
            elif spec.method == "expert_only":
                cmd.append("--policy.train_expert_only=true")
            if self._relative_actions(spec):
                cmd.append("--policy.use_relative_actions=true")
            sw = self._sample_weighting_dict(spec)
            if sw is not None:
                for key in ("type", "progress_path", "head_mode", "kappa", "epsilon"):
                    if key in sw:
                        cmd.append(f"--sample_weighting.{key}={sw[key]}")
        if spec.resume:
            ckpt_cfg = self._resume_config_path(spec.output_dir)
            if ckpt_cfg:
                cmd.append("--resume=true")
                cmd.append(f"--config_path={ckpt_cfg}")
        _consumed = {"policy_type", "job_name", "relative_actions", "sample_weighting", "reward_model"}
        for key, value in spec.extra.items():
            if key in _consumed:
                continue
            cmd.append(f"--{key}={value}")
        return cmd

    def _reward_model_command_flags(self, rm: dict[str, Any], base_model: str = "") -> list[str]:
        """argv-parity flags for a reward-model run (``--reward_model.*``)."""
        rtype = self._reward_model_type(rm)
        friendly = _reward_friendly_fields(rtype)
        flags = [f"--reward_model.type={rtype}", f"--reward_model.device={self.device}"]
        # Warm-start checkpoint: build_config sets reward_cfg.pretrained_path from
        # spec.base_model, so the equivalent CLI must set it too (mirrors the policy
        # path's --policy.pretrained_path). Omitting it left the documented
        # reward-model CLI training from scratch (pretrained_path defaults to None)
        # instead of warm-starting from base_model.
        if base_model:
            flags.append(f"--reward_model.pretrained_path={base_model}")
        for key, value in rm.items():
            if key != "type" and key in friendly:
                flags.append(f"--reward_model.{key}={value}")
        return flags

    def build_config(self, spec: TrainSpec) -> TrainPipelineConfig:
        """Build lerobot's typed ``TrainPipelineConfig`` from a TrainSpec (pure).

        The in-process equivalent of :meth:`build_command`: constructs the
        dataclass tree ``train(cfg)`` consumes directly (no argv). Dispatches to
        the reward-model path when ``extra['reward_model']`` is set, else the
        default policy path.

        On ``resume``, the config is rebuilt FROM THE CHECKPOINT rather than
        from the spec (see :meth:`_build_resume_config`): a spec-built resume
        config leaves ``optimizer``/``scheduler`` unset (lerobot's ``validate()``
        applies the preset only when NOT resuming), so lerobot's factory raises
        before the train loop.
        """
        if spec.resume:
            resume_cfg = self._build_resume_config(spec)
            if resume_cfg is not None:
                return resume_cfg
            # No resumable checkpoint on disk: fall through to a fresh build so
            # validate() surfaces lerobot's own resume error, not an AttributeError.
        rm = self._reward_model_dict(spec)
        if rm is not None:
            return self._build_reward_model_config(spec, rm)
        return self._build_policy_config(spec)

    def _build_resume_config(self, spec: TrainSpec) -> TrainPipelineConfig | None:
        """Rebuild the ``TrainPipelineConfig`` from the checkpoint's own config.

        Mirrors lerobot's CLI resume: the CLI passes ``--config_path`` and draccus
        deserializes the checkpoint's ``train_config.json`` as the BASE config, so
        the resumed run inherits the checkpoint's serialized ``optimizer`` and
        ``scheduler``. lerobot's ``validate()`` applies the optimizer preset only
        when NOT resuming (``elif self.use_policy_training_preset and not
        self.resume``), so a fresh spec-built resume cfg keeps ``optimizer=None``
        and ``make_optimizer_and_scheduler`` raises ``ValueError("Optimizer config
        is required ...")`` before the train loop. The checkpoint's policy config
        is likewise ignored on a fresh build (defaults-only -- the same silent
        mismatch class as the ``base_model`` warm-start path).

        Loading via ``TrainPipelineConfig.from_pretrained`` restores
        ``optimizer``/``scheduler``/``policy`` verbatim; only the managed
        run-control overrides (output_dir, steps, save_freq, wandb off, seed) are
        reapplied on top. Returns ``None`` when no resumable checkpoint exists on
        disk, so the caller falls back to a fresh build and lerobot reports its own
        resume error rather than this method raising.

        Raises:
            ValueError: the checkpoint's ``train_config.json`` exists but cannot
                be deserialized into a ``TrainPipelineConfig`` (truncated,
                hand-edited, or written by an incompatible lerobot version).
                Raised in place of draccus' bare ``DecodingError``, which names
                neither the offending file nor a way forward.
        """
        from pathlib import Path

        from lerobot.configs.train import TrainPipelineConfig

        cfg_file = self._resume_config_path(spec.output_dir)
        if cfg_file is None:
            return None

        # from_pretrained takes the pretrained_model DIRECTORY holding
        # train_config.json (the dir latest_checkpoint returns). A checkpoint
        # config that exists but will not decode is fatal for resume: falling
        # back to a fresh build would re-enter the optimizer=None crash this
        # method exists to prevent, so fail loudly with the file path and the
        # two ways out instead of leaking draccus' pathless DecodingError.
        # The broad catch is a translate-and-reraise (nothing is swallowed): the
        # decoder's error taxonomy spans draccus DraccusException, json/ValueError
        # and OSError, and it is lerobot's to change.
        try:
            cfg = TrainPipelineConfig.from_pretrained(os.path.dirname(cfg_file))
        except Exception as e:
            raise ValueError(
                f"Cannot resume: the checkpoint config at '{cfg_file}' did not "
                f"deserialize into a lerobot TrainPipelineConfig ({type(e).__name__}: {e}). "
                "The file is truncated, hand-edited, or was written by an incompatible "
                "lerobot version. Either point output_dir at an intact checkpoint or "
                "set resume=False to start a fresh run."
            ) from e

        cfg.resume = True
        # output_dir MUST stay the resumed run's dir; validate() derives
        # checkpoint_path from it (mirrors _apply_common_config on the fresh path).
        if spec.output_dir:
            cfg.output_dir = Path(spec.output_dir)
        cfg.checkpoint_path = Path(cfg_file).parent.parent
        # Run-control fields the caller legitimately re-specifies on resume.
        cfg.steps = spec.steps
        cfg.save_freq = spec.save_freq
        if spec.seed is not None:
            cfg.seed = spec.seed
        if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "enable"):
            cfg.wandb.enable = False
        self._apply_extra_passthrough(cfg, spec)
        return cfg

    def _build_dataset_config(self, spec: TrainSpec) -> Any:
        """Shared ``DatasetConfig`` for both the policy and reward-model paths."""
        from lerobot.configs.default import DatasetConfig

        repo_id, root = self._dataset_source(spec)
        dataset_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "root": root,
        }
        split = self._val_eval_split(spec)
        if split is not None:
            dataset_kwargs["eval_split"] = split
        if spec.streaming:
            dataset_kwargs["streaming"] = True
        return DatasetConfig(**dataset_kwargs)

    def _apply_common_config(self, cfg: TrainPipelineConfig, spec: TrainSpec) -> None:
        """Wire seed / wandb / resume - identical for policy and reward runs."""
        from pathlib import Path

        if spec.seed is not None:
            cfg.seed = spec.seed
        if hasattr(cfg, "wandb") and hasattr(cfg.wandb, "enable"):
            cfg.wandb.enable = False
        if self._val_eval_split(spec) is not None:
            # Validate on the caller's own checkpoint cadence so every saved
            # checkpoint has a validation loss beside it; a non-positive
            # save_freq disables periodic saving, so evaluate once at the end.
            cfg.eval_steps = spec.save_freq if spec.save_freq > 0 else spec.steps
        if spec.resume:
            ckpt_cfg = self._resume_config_path(spec.output_dir)
            if ckpt_cfg:
                cfg.checkpoint_path = Path(ckpt_cfg).parent.parent

    def _populate_resume_optimizer(self, cfg: Any, spec: TrainSpec) -> None:
        """Populate optimizer/scheduler for a resumed in-process run.

        On resume, lerobot's validate() skips the optimizer preset application
        (``not self.resume`` guard), expecting the config was deserialized from
        the checkpoint's train_config.json (which carries the serialized
        optimizer/scheduler). The in-process path builds the config fresh from
        the spec, so cfg.optimizer stays None and make_optimizer_and_scheduler
        raises before the training loop.

        Resolution: apply the policy's optimizer/scheduler preset ourselves,
        mirroring what validate() does on the non-resume (fresh-train) path.
        On the CLI resume path the preset comes from the deserialized checkpoint
        config; since we build cfg.policy from the checkpoint's saved config
        (via PreTrainedConfig.from_pretrained in the base_model path, or
        make_policy_config for fresh resume), the preset is already correct.
        """
        if cfg.optimizer is None and cfg.policy is not None:
            try:
                cfg.optimizer = cfg.policy.get_optimizer_preset()
                logger.debug("Resume: populated optimizer from policy preset.")
            except Exception as e:  # noqa: BLE001
                logger.warning("Resume: failed to get optimizer preset: %s", e)

        if getattr(cfg, "scheduler", None) is None and cfg.policy is not None:
            try:
                cfg.scheduler = cfg.policy.get_scheduler_preset()
                logger.debug("Resume: populated scheduler from policy preset.")
            except Exception as e:  # noqa: BLE001
                logger.warning("Resume: failed to get scheduler preset: %s", e)

    def _apply_extra_passthrough(self, cfg: TrainPipelineConfig, spec: TrainSpec) -> None:
        """Typed passthrough for remaining ``extra.*`` keys (validate()-gated).

        Only sets attributes that exist on the typed config tree; unknown keys
        are ignored (never become an arbitrary flag). A *text* value is decoded
        to the field's declared type by :func:`_decode_extra_value`, so the same
        spelling means the same thing here as on the ``--flag=value`` CLI that
        :meth:`build_command` documents this config as corresponding to.
        """
        _consumed = {"policy_type", "job_name", "relative_actions", "sample_weighting", "reward_model"}
        for key, value in spec.extra.items():
            if key in _consumed:
                continue
            target, attr = _resolve_dotted(cfg, key)
            if target is not None and hasattr(target, attr):
                setattr(target, attr, _decode_extra_value(target, attr, key, value))
            else:
                logger.warning("LerobotTrainer: ignoring extra '%s' (no matching config field).", key)

    def _build_reward_model_config(self, spec: TrainSpec, rm: dict[str, Any]) -> TrainPipelineConfig:
        """Build a reward-model ``TrainPipelineConfig`` (``cfg.reward_model`` set).

        SARM and the other reward models share lerobot's ``train(cfg)`` loop; the
        config sets ``reward_model`` (and leaves ``policy`` unset) so lerobot
        follows its ``is_reward_model_training`` branch.
        """
        from pathlib import Path

        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.rewards import make_reward_model_config

        rtype = self._reward_model_type(rm)
        # Forward every friendly key that is a real field of the resolved config
        # dataclass (read supported fields, ignore the rest) - the dynamic
        # passthrough that reaches all reward types, not just SARM. ``device``
        # and the other managed base fields are set by the trainer below.
        friendly = _reward_friendly_fields(rtype)
        reward_kwargs: dict[str, Any] = {"device": self.device}
        for key, value in rm.items():
            if key != "type" and key in friendly:
                reward_kwargs[key] = value
        try:
            reward_cfg = make_reward_model_config(rtype, **reward_kwargs)
        except TypeError as e:
            raise ValueError(f"reward_model type '{rtype}' rejected field(s) {sorted(reward_kwargs)}: {e}") from e
        except OSError as e:
            # A reward config may derive a field from a pretrained asset inside
            # its own __post_init__ (robometer reads its backbone's config and
            # tokenizer to size ``vlm_config``), so merely CONSTRUCTING it can
            # need a download. Every huggingface_hub failure class for that is
            # an OSError subclass (LocalEntryNotFoundError, HfHubHTTPError,
            # GatedRepoError, OfflineModeIsEnabled), and transformers re-raises
            # a plain OSError, so this is the narrowest superset that covers
            # "the asset could not be obtained" without swallowing the
            # ValueErrors a config raises for a bad field value - those are
            # already actionable and name the field.
            raise ValueError(
                f"reward_model type '{rtype}' could not be constructed: building its config "
                f"needed a pretrained asset this host could not obtain ({e}). The spec itself "
                f"is fine - validate() cannot reach the network to see this. Either make the "
                f"asset available (a warm Hugging Face cache, or network access with "
                f"HF_HUB_OFFLINE unset), or pass the field the config derives from it in "
                f"extra['reward_model'] so its constructor fetches nothing. Fields this type "
                f"accepts: {', '.join(sorted(friendly))}."
            ) from e
        if hasattr(reward_cfg, "push_to_hub"):
            reward_cfg.push_to_hub = False
        if spec.base_model and hasattr(reward_cfg, "pretrained_path"):
            reward_cfg.pretrained_path = spec.base_model

        cfg = TrainPipelineConfig(
            dataset=self._build_dataset_config(spec),
            policy=None,
            reward_model=reward_cfg,
            output_dir=Path(spec.output_dir) if spec.output_dir else None,
            job_name=str(spec.extra.get("job_name", "strands_ft")),
            steps=spec.steps,
            batch_size=spec.global_batch_size,
            save_freq=spec.save_freq,
            resume=spec.resume,
        )
        self._apply_common_config(cfg, spec)
        self._apply_extra_passthrough(cfg, spec)
        return cfg

    def _build_policy_config(self, spec: TrainSpec) -> TrainPipelineConfig:
        """Build a policy ``TrainPipelineConfig`` (``cfg.policy`` set)."""
        import dataclasses
        from pathlib import Path

        from lerobot.configs.default import PeftConfig
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.policies.factory import make_policy_config

        ptype = self._resolve_policy_type(spec)

        if spec.base_model:
            # Warm start: load the checkpoint's OWN saved config (architecture
            # hyperparameters - chunk_size, vision backbone, hidden dims, ...)
            # rather than make_policy_config's all-defaults. lerobot's make_policy
            # feeds this config verbatim to from_pretrained (strict=False), so a
            # defaults-only config silently loads mismatched/partial weights from a
            # base trained with non-default hyperparameters. Mirror lerobot's own
            # --policy.path flow, which builds PreTrainedConfig.from_pretrained
            # first and then loads the weights against it.
            from lerobot.configs.policies import PreTrainedConfig

            policy_cfg = PreTrainedConfig.from_pretrained(spec.base_model)
            policy_cfg.pretrained_path = Path(spec.base_model)
        else:
            policy_cfg = make_policy_config(ptype)
        if hasattr(policy_cfg, "device"):
            policy_cfg.device = self.device
        if hasattr(policy_cfg, "push_to_hub"):
            policy_cfg.push_to_hub = False
        if spec.method == "expert_only" and hasattr(policy_cfg, "train_expert_only"):
            policy_cfg.train_expert_only = True
        if spec.learning_rate is not None:
            if not hasattr(policy_cfg, "optimizer_lr"):
                raise ValueError(
                    f"learning_rate={spec.learning_rate} was requested but policy_type "
                    f"'{ptype}' has no 'optimizer_lr' field (its optimizer preset is not "
                    "an Adam-style LR). Drop learning_rate to use the policy preset, or "
                    "override the specific optimizer field via extra['policy.<field>']."
                )
            policy_cfg.optimizer_lr = spec.learning_rate
        if self._relative_actions(spec):
            if not hasattr(policy_cfg, "use_relative_actions"):
                rel_supported = sorted(t for t in _lerobot_policy_types() if _policy_supports_relative_actions(t))
                raise ValueError(
                    f"relative_actions requested but policy_type '{ptype}' has no "
                    f"use_relative_actions field (supported: {rel_supported})"
                )
            policy_cfg.use_relative_actions = True

        peft_cfg = None
        if spec.method == "lora":
            peft_kwargs: dict[str, Any] = {"method_type": "LORA"}
            if spec.lora_r is not None:
                peft_kwargs["r"] = spec.lora_r
            if spec.lora_alpha is not None:
                peft_kwargs["lora_alpha"] = spec.lora_alpha
            if spec.lora_target_modules is not None:
                peft_kwargs["target_modules"] = spec.lora_target_modules
            supported = {f.name for f in dataclasses.fields(PeftConfig)}
            unsupported = sorted(k for k in peft_kwargs if k not in supported)
            if unsupported:
                raise ValueError(
                    f"The installed lerobot's PeftConfig does not support LoRA "
                    f"option(s) {unsupported}; it accepts {sorted(supported)}. "
                    "These options were requested via TrainSpec. Upgrade lerobot "
                    "to a version that supports them, or drop them from the spec."
                )
            peft_cfg = PeftConfig(**peft_kwargs)
            # Do NOT set policy_cfg.use_peft here. lerobot never sets use_peft
            # before training; make_policy reads use_peft=True as "load
            # pretrained_path as a PEFT ADAPTER repo" (PeftConfig.from_pretrained),
            # which crashes on a plain base checkpoint, and rejects use_peft=True
            # with no checkpoint outright. Setting cfg.peft alone is what triggers
            # lerobot_train's wrap_with_peft on the freshly loaded base policy;
            # use_peft is flipped by lerobot itself only after wrapping.

        cfg = TrainPipelineConfig(
            dataset=self._build_dataset_config(spec),
            policy=policy_cfg,
            output_dir=Path(spec.output_dir) if spec.output_dir else None,
            job_name=str(spec.extra.get("job_name", "strands_ft")),
            steps=spec.steps,
            batch_size=spec.global_batch_size,
            save_freq=spec.save_freq,
            resume=spec.resume,
            peft=peft_cfg,
        )
        self._apply_common_config(cfg, spec)
        if spec.resume:
            self._populate_resume_optimizer(cfg, spec)

        # RA-BC sample weighting: lerobot >= 0.6.0 configures it via a NESTED
        # SampleWeightingConfig on TrainPipelineConfig (cfg.sample_weighting),
        # which its train loop turns into a per-sample loss reweighting. The
        # friendly extra['sample_weighting'] keys map 1:1 onto that config's
        # fields, so the validated dict is forwarded verbatim. Fail fast on
        # unknown keys, an unsupported scheme, or a lerobot too old to expose
        # sample weighting.
        sw = self._sample_weighting_dict(spec)
        if sw is not None:
            # RA-BC sample weighting is a lerobot >= 0.6.0 surface (the nested
            # SampleWeightingConfig on TrainPipelineConfig). Gate on its presence
            # FIRST so an older lerobot yields an actionable ValueError instead of
            # a raw ModuleNotFoundError from the import below.
            if not hasattr(cfg, "sample_weighting"):
                raise ValueError(
                    "The installed lerobot does not expose sample weighting (no "
                    "'sample_weighting' on TrainPipelineConfig); requires lerobot "
                    ">= 0.6.0, or drop extra['sample_weighting']."
                )
            try:
                from lerobot.utils.sample_weighting import SampleWeightingConfig
            except ImportError as exc:
                raise ValueError(
                    "The installed lerobot does not expose sample weighting (no "
                    "'lerobot.utils.sample_weighting'); requires lerobot >= 0.6.0, "
                    "or drop extra['sample_weighting']."
                ) from exc

            unsupported = sorted(k for k in sw if k not in _SAMPLE_WEIGHTING_KEYS)
            if unsupported:
                raise ValueError(
                    f"extra['sample_weighting'] does not support field(s) "
                    f"{unsupported}; accepted keys are {sorted(_SAMPLE_WEIGHTING_KEYS)}."
                )
            sw_type = sw.get("type", "rabc")
            if sw_type not in _SAMPLE_WEIGHTING_TYPES:
                raise ValueError(
                    f"extra['sample_weighting']['type'] must be one of "
                    f"{sorted(_SAMPLE_WEIGHTING_TYPES)} (the schemes lerobot ships), got {sw_type!r}."
                )
            cfg.sample_weighting = SampleWeightingConfig(**sw)

        self._apply_extra_passthrough(cfg, spec)
        return cfg

    def train(self, spec: TrainSpec) -> TrainResult:
        """Run LeRobot training in-process via ``lerobot_train.train(cfg)``.

        Fails closed on any :meth:`validate` problem, builds the
        ``TrainPipelineConfig`` from the spec (policy or reward-model path,
        resume, learning rate), and calls lerobot's own training function
        directly. ``num_gpus > 1`` spawns workers via torch
        ``elastic_launch``. Blocks until the run terminates and returns a
        terminal ``TrainResult`` with the checkpoint dir + metrics verdict.
        """
        problems = self.validate(spec)
        if problems:
            return TrainResult(
                status="error",
                job_id="",
                message="validation failed: " + "; ".join(problems),
            )

        self.prepare(spec)

        # lerobot's validate() REFUSES a pre-existing output_dir unless
        # resume=True. Don't pre-create output_dir; write our log NEXT TO it.
        parent = os.path.dirname(os.path.abspath(spec.output_dir)) or "."
        os.makedirs(parent, exist_ok=True)

        # Fresh-start hygiene: clear a stale output_dir with no resumable ckpt.
        if not spec.resume and os.path.isdir(spec.output_dir):
            if self.latest_checkpoint(spec.output_dir) is None:
                shutil.rmtree(spec.output_dir, ignore_errors=True)

        job_id = f"lerobot-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            cfg = self.build_config(spec)
        except Exception as e:  # noqa: BLE001 - config build is the typed boundary
            return TrainResult(
                status="error",
                job_id=job_id,
                message=f"failed to build lerobot TrainPipelineConfig: {e}",
            )

        logger.info(
            "LerobotTrainer launching in-process: %s device=%s steps=%d num_gpus=%d",
            self._run_type_label(spec),
            self.device,
            spec.steps,
            spec.num_gpus,
        )

        train_error: Exception | None = None
        try:
            if spec.num_gpus and spec.num_gpus > 1:
                elastic_launch_callable(
                    _lerobot_worker,
                    nproc_per_node=spec.num_gpus,
                    nnodes=1,
                    run_id=job_id,
                    fn_args=(self.policy_type, self.device, spec, log_path),
                )
            else:
                from lerobot.scripts.lerobot_train import train as lerobot_train

                # On resume, lerobot's validate() reads the checkpoint's
                # train_config.json path back off sys.argv (--config_path); the
                # in-process call has no argv, so inject it for the call.
                resume_cfg = self._resume_config_path(spec.output_dir) if spec.resume else None
                with resume_argv(resume_cfg):
                    call_callable(lerobot_train, cfg, log_path=log_path)
        except Exception as e:  # noqa: BLE001 - convert ANY failure to a result
            train_error = e
            logger.error("LerobotTrainer in-process train failed: %s", e)

        ckpt_model_dir = self.latest_checkpoint(spec.output_dir)  # loadable pretrained_model dir
        metrics = self._parse_log(log_path)

        if train_error is not None:
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=ckpt_model_dir,
                metrics=metrics,
                message=f"lerobot train raised {type(train_error).__name__}: {train_error}; see {log_path}",
            )

        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=ckpt_model_dir,
            metrics=metrics,
            message=f"lerobot train complete (in-process); log: {log_path}",
        )

    def _parse_log(self, log_path: str) -> dict[str, Any]:
        """Extract a 'RUNNING != learning' verdict from the captured train log.

        Parses lerobot's MetricsTracker line (verified vs lerobot 0.5.x
        ``utils/logging_utils.py::MetricsTracker.__str__``)::

            step:1.2K smpl:4.9K ep:8 epch:2.00 loss:0.123 ...
        """
        latest_step: int | None = None
        latest_loss: float | None = None
        latest_epoch: float | None = None
        try:
            with open(log_path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "step:" not in line:
                        continue
                    for tok in line.split():
                        key, _, val = tok.partition(":")
                        if not val:
                            continue
                        if key == "step":
                            n = _expand_big_number(val)
                            if n is not None:
                                latest_step = int(n)
                        elif key == "loss":
                            with contextlib.suppress(ValueError):
                                latest_loss = float(val)
                        elif key == "epch":
                            with contextlib.suppress(ValueError):
                                latest_epoch = float(val)
        except OSError:
            return {}

        metrics: dict[str, Any] = {}
        if latest_step is not None:
            metrics["latest_step"] = latest_step
        if latest_epoch is not None:
            metrics["latest_epoch"] = latest_epoch
        if latest_loss is not None:
            import math

            metrics["latest_loss"] = latest_loss
            metrics["learning"] = math.isfinite(latest_loss)
        metrics["liveness_ok"] = latest_step is not None
        return metrics


def _resolve_dotted(cfg: Any, key: str) -> tuple[Any, str]:
    """Map a (optionally dotted) extra key to (obj, attr) on the config tree."""
    if "." not in key:
        return cfg, key
    head, _, tail = key.partition(".")
    sub = getattr(cfg, head, None)
    if sub is None or "." in tail:
        return None, tail
    return sub, tail


def _extra_field_type(target: Any, attr: str) -> Any:
    """Resolved annotation of ``target.attr``, or ``None`` when unavailable.

    ``dataclasses.fields()`` reports ``f.type`` as the *source text* of the
    annotation for a module compiled with ``from __future__ import
    annotations`` - 45 ``bool`` fields across 6 lerobot policy configs
    (``eo1``, ``evo1``, ``fastwam``, ``molmoact2``, ``vla_jepa``, ``xvla``)
    read back as the string ``"bool"`` there - so a caller that compared
    ``f.type is bool`` would silently skip exactly those. The annotation is
    resolved through :func:`typing.get_type_hints` instead, which evaluates
    the string form.

    Returns ``None`` when the annotation cannot be resolved (an unresolvable
    forward reference, or a target that is not annotated at all), which the
    caller reads as "leave the value alone".
    """
    try:
        hints = typing.get_type_hints(type(target))
    except (NameError, TypeError):
        return None
    return hints.get(attr)


def _annotation_admits_text(hint: Any) -> bool:
    """Whether ``hint`` already accepts a ``str`` as-is.

    ``str``, ``str | None`` and ``Any`` need no decoding: the value the caller
    passed is already the field's own type. A generic whose *parameters*
    happen to include ``str`` (``dict[str, int]``) does not qualify - only a
    union is searched - because the field itself is not a string there.
    """
    if hint is str or hint is Any:
        return True
    origin = typing.get_origin(hint)
    if origin is typing.Union or origin is types.UnionType:
        return any(_annotation_admits_text(arg) for arg in typing.get_args(hint))
    return False


def _decode_extra_value(target: Any, attr: str, key: str, value: Any) -> Any:
    """Decode a text ``extra`` value the way lerobot's own CLI decodes it.

    ``build_command`` renders an ``extra`` entry as ``--key=value``, where
    lerobot's draccus parser reads the text with ``cfgparsing.parse_string``
    (YAML scalar rules) and then decodes it to the field's declared type. The
    in-process path assigns to the same dataclass field directly, so a text
    value has to travel the same two stages or the two paths stop agreeing:
    ``extra={"policy.freeze_vision_encoder": "false"}`` otherwise stores the
    *string* ``"false"``, which is truthy, so the encoder stays frozen while
    ``--policy.freeze_vision_encoder=false`` unfreezes it.

    The decoder is borrowed from draccus rather than reimplemented so the
    accepted spellings cannot drift from the CLI's: ``false``/``no``/``off``
    and ``true``/``yes``/``on`` (any case) decode to bools, while ``0``/``1``
    are *not* bools to draccus and are refused on both paths.

    Args:
        target: Config object owning the field (``cfg`` or a sub-config).
        attr: Attribute name on ``target``.
        key: Original ``extra`` key, quoted in the refusal so the caller can
            find it in their spec.
        value: Value from ``spec.extra``. A non-``str`` is returned unchanged -
            a caller who already passed the field's own type never went through
            text, so there is nothing to decode.

    Returns:
        The value to assign to ``target.attr``.

    Raises:
        ValueError: The text does not decode to the field's declared type.
            The CLI refuses the same spelling, and assigning it raw is how a
            wrong type reaches training silently, so it is refused here too.
    """
    if not isinstance(value, str):
        return value
    hint = _extra_field_type(target, attr)
    if hint is None or _annotation_admits_text(hint):
        return value

    import draccus
    from draccus import cfgparsing

    try:
        return draccus.decode(hint, cfgparsing.parse_string(value))
    except Exception as e:
        # Both stages are third-party and raise from disjoint hierarchies
        # (draccus.ParsingError for the decode, yaml.YAMLError for the scalar
        # parse), so the breadth here translates every decoder failure into one
        # actionable refusal. Nothing is swallowed - the cause is chained.
        raise ValueError(
            f"extra['{key}']={value!r} does not decode to {_format_annotation(hint)}, "
            f"the declared type of '{attr}' on {type(target).__name__}. "
            f"Values are read with lerobot's own CLI decoder, so the accepted spellings are "
            f"the ones '--{key}={value}' accepts"
            + (
                " - for a boolean: false/no/off or true/yes/on (any case); note 0 and 1 are not booleans."
                if hint is bool
                else "."
            )
            + f" Pass a {_format_annotation(hint)} value directly to skip decoding."
        ) from e


def _format_annotation(hint: Any) -> str:
    """Human-readable name for an annotation, for use in a refusal."""
    return getattr(hint, "__name__", None) or str(hint)


def _lerobot_worker(policy_type: str, device: str, spec: TrainSpec, log_path: str) -> None:
    """elastic_launch worker: build the cfg and call lerobot train() in this worker.

    Runs in a torch-spawned worker (one per GPU). torch sets RANK / LOCAL_RANK /
    WORLD_SIZE; lerobot's Accelerator picks them up. Only local rank 0 tees to
    the shared log to avoid interleaved writes.
    """
    import os as _os

    trainer = LerobotTrainer(policy_type=policy_type, device=device)
    cfg = trainer.build_config(spec)
    from lerobot.scripts.lerobot_train import train as lerobot_train

    is_rank0 = _os.environ.get("LOCAL_RANK", "0") == "0"
    # Resume needs --config_path on sys.argv for lerobot's validate() (see
    # resume_argv); each spawned worker builds its own cfg and must inject it too.
    resume_cfg = trainer._resume_config_path(spec.output_dir) if spec.resume else None
    with resume_argv(resume_cfg):
        call_callable(lerobot_train, cfg, log_path=log_path if is_rank0 else None)


def _expand_big_number(token: str) -> float | None:
    """Invert lerobot's ``format_big_number`` (e.g. ``"1.2K" -> 1200``)."""
    suffixes = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "Q": 1e15}
    token = token.strip()
    if not token:
        return None
    suffix = token[-1].upper()
    if suffix in suffixes and suffix != "" and not token[-1].isdigit():
        body, mult = token[:-1], suffixes[suffix]
    else:
        body, mult = token, 1
    try:
        return float(body) * mult
    except ValueError:
        return None


def _auto_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"
