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
lerobot >= 0.5.2 (the ``lerobot.rewards`` package).

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
from typing import TYPE_CHECKING, Any

from strands_robots.training._inproc import call_callable, elastic_launch_callable
from strands_robots.training.base import Trainer, TrainResult, TrainSpec

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

# RA-BC (Reward-Aligned Behavior Cloning) is surfaced to the agent through the
# ``extra['sample_weighting']`` dict. lerobot >= 0.5.2 configures sample
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
# (``TrainPipelineConfig.is_reward_model_training``). Requires lerobot >= 0.5.2
# (the ``lerobot.rewards`` package).
#
# Parity with lerobot is DYNAMIC, not a hardcoded list: both the set of reward
# types and each type's configurable fields are read live from lerobot's
# ``RewardModelConfig`` draccus ChoiceRegistry (see :func:`_reward_registry`) -
# the same zero-maintenance discovery Robot / Teleop / Camera / Policy already
# use. Any reward model lerobot ships (sarm, robometer, topreward,
# reward_classifier, ...) or a third-party plugin registers is reachable with no
# change here. The static fallbacks below are used ONLY when ``lerobot.rewards``
# is absent (lerobot < 0.5.2), where reward-model training cannot run anyway but
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


def _reward_registry() -> dict[str, type] | None:
    """Live ``RewardModelConfig`` ChoiceRegistry, or ``None`` when unavailable.

    Importing ``lerobot.rewards`` runs each config module's
    ``@RewardModelConfig.register_subclass`` decorator, which is what populates
    the draccus ChoiceRegistry - querying it before that import yields an empty
    mapping, so the import is the load-bearing step. Returns ``None`` when the
    installed lerobot has no ``lerobot.rewards`` (lerobot < 0.5.2).
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
    friendly keys when the registry is unavailable (offline / lerobot < 0.5.2),
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
        return "lerobot_local"

    @property
    def hardware_floor(self) -> dict[str, Any]:
        # ACT fits a consumer GPU; large VLAs (pi05) want an L40S. Advisory.
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

    def _val_split_episodes(self, spec: TrainSpec) -> list[int] | None:
        """Held-out validation split: train on the FIRST (total - N) episodes.

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
            return list(range(0, total - spec.val_episodes))
        return None

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
        >= 0.5.2 configures it via a nested ``SampleWeightingConfig`` on
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

        if spec.steps <= 0:
            problems.append(f"steps must be > 0, got {spec.steps}")

        if spec.num_nodes > 1:
            problems.append(
                f"num_nodes={spec.num_nodes}: multi-node lerobot needs a per-node "
                "launcher and cannot run in-process; use num_nodes=1."
            )

        if spec.val_episodes is not None and spec.dataset_root:
            total = self._dataset_total_episodes(spec.dataset_root)
            if total is not None and spec.val_episodes >= total:
                problems.append(f"val_episodes={spec.val_episodes} >= total_episodes={total}")

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

        sw = spec.extra.get("sample_weighting")
        if sw is not None and not isinstance(sw, dict):
            problems.append(
                "extra['sample_weighting'] must be a dict of RA-BC fields, e.g. {'type': 'rabc', 'kappa': 0.01}"
            )
        elif isinstance(sw, dict):
            for k, v in sw.items():
                if isinstance(v, str) and v.startswith("-"):
                    problems.append(f"sample_weighting['{k}'] must not start with '-' (would parse as a stray flag)")
        return problems

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
                "requires lerobot >= 0.5.2 (install from source)"
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
            cmd.extend(self._reward_model_command_flags(rm))
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
        eps = self._val_split_episodes(spec)
        if eps is not None:
            cmd.append(f"--dataset.episodes=[{', '.join(map(str, eps))}]")
        if rm is None:
            if spec.base_model:
                cmd.append(f"--policy.pretrained_path={spec.base_model}")
            if spec.method == "lora":
                cmd.append("--peft.method_type=LORA")
                if spec.lora_r is not None:
                    cmd.append(f"--peft.r={spec.lora_r}")
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

    def _reward_model_command_flags(self, rm: dict[str, Any]) -> list[str]:
        """argv-parity flags for a reward-model run (``--reward_model.*``)."""
        rtype = self._reward_model_type(rm)
        friendly = _reward_friendly_fields(rtype)
        flags = [f"--reward_model.type={rtype}", f"--reward_model.device={self.device}"]
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
        """
        rm = self._reward_model_dict(spec)
        if rm is not None:
            return self._build_reward_model_config(spec, rm)
        return self._build_policy_config(spec)

    def _build_dataset_config(self, spec: TrainSpec) -> Any:
        """Shared ``DatasetConfig`` for both the policy and reward-model paths."""
        from lerobot.configs.default import DatasetConfig

        repo_id, root = self._dataset_source(spec)
        dataset_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "root": root,
            "episodes": self._val_split_episodes(spec),
        }
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
        if spec.resume:
            ckpt_cfg = self._resume_config_path(spec.output_dir)
            if ckpt_cfg:
                cfg.checkpoint_path = Path(ckpt_cfg).parent.parent

    def _apply_extra_passthrough(self, cfg: TrainPipelineConfig, spec: TrainSpec) -> None:
        """Typed passthrough for remaining ``extra.*`` keys (validate()-gated).

        Only sets attributes that exist on the typed config tree; unknown keys
        are ignored (never become an arbitrary flag).
        """
        _consumed = {"policy_type", "job_name", "relative_actions", "sample_weighting", "reward_model"}
        for key, value in spec.extra.items():
            if key in _consumed:
                continue
            target, attr = _resolve_dotted(cfg, key)
            if target is not None and hasattr(target, attr):
                setattr(target, attr, value)
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

        policy_cfg = make_policy_config(ptype)
        if hasattr(policy_cfg, "device"):
            policy_cfg.device = self.device
        if hasattr(policy_cfg, "push_to_hub"):
            policy_cfg.push_to_hub = False
        if spec.base_model:
            policy_cfg.pretrained_path = Path(spec.base_model)
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
            if hasattr(policy_cfg, "use_peft"):
                policy_cfg.use_peft = True

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

        # RA-BC sample weighting: lerobot >= 0.5.2 configures it via a NESTED
        # SampleWeightingConfig on TrainPipelineConfig (cfg.sample_weighting),
        # which its train loop turns into a per-sample loss reweighting. The
        # friendly extra['sample_weighting'] keys map 1:1 onto that config's
        # fields, so the validated dict is forwarded verbatim. Fail fast on
        # unknown keys, an unsupported scheme, or a lerobot too old to expose
        # sample weighting.
        sw = self._sample_weighting_dict(spec)
        if sw is not None:
            # RA-BC sample weighting is a lerobot >= 0.5.2 surface (the nested
            # SampleWeightingConfig on TrainPipelineConfig). Gate on its presence
            # FIRST so an older lerobot yields an actionable ValueError instead of
            # a raw ModuleNotFoundError from the import below.
            if not hasattr(cfg, "sample_weighting"):
                raise ValueError(
                    "The installed lerobot does not expose sample weighting (no "
                    "'sample_weighting' on TrainPipelineConfig); requires lerobot "
                    ">= 0.5.2, or drop extra['sample_weighting']."
                )
            try:
                from lerobot.utils.sample_weighting import SampleWeightingConfig
            except ImportError as exc:
                raise ValueError(
                    "The installed lerobot does not expose sample weighting (no "
                    "'lerobot.utils.sample_weighting'); requires lerobot >= 0.5.2, "
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
