"""Trainer abstraction - post-tune ANY policy provider natively.

The :class:`Trainer` ABC is the training-side peer of
:class:`~strands_robots.policies.base.Policy` (inference). Where ``Policy``
hides *how a model produces actions*, ``Trainer`` hides *how a model is
post-tuned* - and those pipelines genuinely differ per provider:

* **LeRobot** - build a ``TrainPipelineConfig`` and call
  ``lerobot.scripts.lerobot_train.train(cfg)`` in-process. HF-native checkpoints.
* **GR00T N1.7** - build a ``FinetuneConfig`` -> ``Config`` and call
  ``gr00t.experiment.experiment.run(config)``; ``tune_llm/visual/projector/
  diffusion`` knobs + a modality-config ``.py``.
* **Cosmos3** - build the SFT ``Config`` via ``load_experiment_from_toml`` and
  call ``cosmos_framework.scripts.train.launch(config, args)``; with an explicit
  **DCP checkpoint conversion** prepare step and a **DCP -> safetensors** export
  step. 8xH100 floor.

All three run **in-process** (imported and called as libraries, no subprocess);
multi-GPU goes through torch's programmatic ``elastic_launch``.

All three nonetheless converge on:

1. the same **dataset format** - LeRobotDataset v3 (what
   :class:`~strands_robots.dataset_recorder.DatasetRecorder` already writes), and
2. the same **lifecycle** - ``validate -> prepare -> train -> export``.

A ``Trainer`` is selected by the SAME provider name as its ``Policy``
(``groot`` / ``lerobot_local`` / ``cosmos3``), so a single registry identity
owns both the inference class and the training class. Adding a new policy =
add a ``Policy`` + a ``Trainer`` under one provider entry.

See :class:`~strands_robots.training.mock.MockTrainer` for the canonical
no-dependency reference implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainSpec:
    """Provider-agnostic post-tuning specification.

    Concrete trainers read the fields they support and **ignore the rest** -
    the same tolerance rule that :meth:`Policy.get_actions` applies to its
    ``**kwargs``. Backends MUST NOT raise on a field they don't use; new,
    backend-specific knobs live in :attr:`extra` until >=2 backends share
    them and they graduate to a first-class field.

    Attributes:
        dataset_root: Path to a LeRobotDataset v3 root (must contain
            ``meta/info.json``). This is exactly what
            :class:`~strands_robots.dataset_recorder.DatasetRecorder` /
            ``Robot.stop_recording`` produce, so the ``record -> train`` loop
            needs no conversion layer. When :attr:`dataset_repo_id` is set this
            is optional and, if given, acts as a local cache root for the Hub
            dataset (lerobot ``DatasetConfig.root``).
        dataset_repo_id: Hugging Face Hub dataset id (``org/name``) to train
            *from the Hub* instead of a local root. Required for
            :attr:`streaming` of a Hub dataset (the 50-500 GB case that would
            otherwise have to download in full first). When set, the backend
            uses it as lerobot's ``DatasetConfig.repo_id`` and leaves
            ``dataset_root`` as an optional local cache. Mutually sufficient
            with ``dataset_root`` - supply exactly one as the data source.
        streaming: Stream frames from the dataset instead of materializing it
            (lerobot ``DatasetConfig.streaming`` -> ``StreamingLeRobotDataset``).
            With :attr:`dataset_repo_id` this streams shards from the Hub with no
            full download (bounded disk); with a local ``dataset_root`` it
            streams from disk (bounded RAM). LeRobot-only; other backends ignore
            it (tolerance rule).
        base_model: HF model id or local checkpoint path to post-tune *from*.
        output_dir: Directory for checkpoints, logs, and the final artifact.
        embodiment: Embodiment tag / robot id. Required by GR00T
            (``--embodiment_tag``); LeRobot infers it from dataset features;
            optional elsewhere.
        steps: Total optimizer steps (maps to lerobot ``--steps`` /
            GR00T ``max_steps`` / Cosmos ``trainer.max_iter``).
        global_batch_size: Batch summed across GPUs before grad accumulation.
        learning_rate: Optimizer learning rate. ``None`` (default) uses the
            backend's own default -- the policy training preset for LeRobot
            (``policy.optimizer_lr``), GR00T's ``FinetuneConfig`` default, or
            Cosmos's TOML default. An explicit value is honored by every
            backend (same opt-in shape as :attr:`seed`); LeRobot maps it to
            ``policy.optimizer_lr`` and rejects it loudly if the policy has no
            such field. RL trainers (PPO/FastSAC) have no preset to defer to,
            so :class:`~strands_robots.training.rl.base_algo.RLTrainSpec`
            overrides this default with a concrete value.
        save_freq: Checkpoint cadence in steps.
        num_gpus: GPUs on this node. ``>1`` runs the backend under torch's
            in-process ``elastic_launch`` (the engine behind ``torchrun``).
        num_nodes: Nodes for multi-node training (Cosmos HSDP /
            ``torchrun --nnodes``).
        resume: Resume from the latest checkpoint under ``output_dir`` when
            one exists.
        seed: Master seed (best-effort; not all backends expose it).
        method: Tuning strategy, mapped per-backend:
            ``"full"`` | ``"lora"`` | ``"expert_only"`` | ``"frozen_backbone"``.
            ``lora`` and ``expert_only`` are mutually exclusive (both freeze
            the VLM); a backend MUST reject the combination in
            :meth:`Trainer.validate`.
        lora_r / lora_alpha / lora_target_modules: LoRA hyperparameters
            (used only when ``method == "lora"``). ``lora_target_modules=None``
            means "use the policy's built-in default targets".
        tune: Fine-grained component toggles for backends that expose them
            (GR00T: ``{"llm": bool, "visual": bool, "projector": bool,
            "diffusion": bool}``). Ignored by backends that don't.
        val_episodes: Hold out the LAST N episodes as a validation set
            (deterministic split; lerobot ``--dataset.episodes=[0..total-N-1]``).
        augmentation: Backend-specific data augmentation (GR00T
            ``color_jitter_params`` / ``random_rotation_angle``; Cosmos
            dataset filter dict).
        fps: Dataset control rate, when a backend needs it explicitly.
        extra: Raw passthrough. Keys become backend-native flags / overrides
            (lerobot ``--key=value``; Cosmos Hydra ``key.path=value``). The
            escape hatch that keeps the ABC stable as backends evolve.
    """

    # --- universal ---
    dataset_root: str = ""
    base_model: str = ""
    output_dir: str = ""
    dataset_repo_id: str | None = None
    embodiment: str | None = None
    steps: int = 10_000
    global_batch_size: int = 32
    learning_rate: float | None = None
    save_freq: int = 1_000
    num_gpus: int = 1
    num_nodes: int = 1
    resume: bool = False
    seed: int | None = None
    # --- tuning strategy ---
    method: str = "full"
    lora_r: int | None = None
    lora_alpha: int | None = None
    lora_target_modules: str | None = None
    tune: dict[str, bool] = field(default_factory=dict)
    # --- data ---
    val_episodes: int | None = None
    augmentation: dict[str, Any] | None = None
    fps: int | None = None
    streaming: bool = False
    # --- escape hatch ---
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainResult:
    """Outcome of a training lifecycle call.

    Attributes:
        status: ``"success"`` | ``"running"`` | ``"error"``.
        job_id: Stable id for this run (used by :meth:`Trainer.status`).
        checkpoint_dir: Where checkpoints are written (``None`` before any
            save / on validation failure).
        exported_model: Final loadable artifact path - a value that
            ``create_policy(...)`` can consume - once :meth:`Trainer.export`
            has run. ``None`` otherwise.
        metrics: Free-form metrics for the "RUNNING != learning" verdict
            (e.g. ``latest_step``, ``latest_loss``, ``learning``,
            ``liveness_ok``).
        message: Human-readable status / error detail.
    """

    status: str
    job_id: str
    checkpoint_dir: str | None = None
    exported_model: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    message: str = ""


class Trainer(ABC):
    """Abstract base class for post-tuning a policy of one provider family.

    Lifecycle: :meth:`validate` (pure preflight) -> :meth:`prepare` (optional
    one-time setup) -> :meth:`train` (run + collect verdict) -> :meth:`export`
    (produce a loadable artifact). :meth:`latest_checkpoint` discovers the
    loadable artifact a run produced; :meth:`status` is an optional best-effort
    verdict for backends that can poll a still-running job.

    Concrete trainers are thin adapters that **import the backend package and
    call its own training function in-process** (LeRobot ``train(cfg)``, GR00T
    ``experiment.run(config)``, Cosmos ``train.launch(config, args)``) - they do
    NOT reimplement training and do NOT shell out to a subprocess. Multi-GPU is
    driven via torch's programmatic ``elastic_launch`` (the engine behind
    ``torchrun``), still in-process.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identity - MUST match the paired ``Policy.provider_name``."""

    @abstractmethod
    def validate(self, spec: TrainSpec) -> list[str]:
        """Pure, side-effect-free preflight.

        Return a list of human-readable problems; an empty list means the spec
        is launchable. Implementations SHOULD check: dataset_root has
        ``meta/info.json``; :attr:`TrainSpec.method` is supported and not a
        contradictory combination (``lora`` + ``expert_only``); any
        backend-required input is present (e.g. a DCP base for Cosmos); and
        rough hardware feasibility against :attr:`hardware_floor`.

        MUST NOT touch the filesystem beyond read-only stat / config reads,
        spawn processes, or allocate GPUs - it powers a ``plan`` advisor that
        runs *before* anything expensive starts.
        """

    def _security_problems(self, spec: TrainSpec) -> list[str]:
        """Input-safety preflight shared by every backend (defense-in-depth).

        Returns problems for any agent-supplied value that would be unsafe to
        feed into the backend's config (path traversal / protected directories,
        a leading ``-`` that a backend's argv-parity helper would read as a
        flag, or an ``extra`` key that would set an arbitrary config attribute /
        Hydra override). Concrete :meth:`validate` implementations MUST call
        this first so untrusted ``TrainSpec`` input is checked before any config
        is built. (Training itself is in-process now - no subprocess argv - but
        the ``extra`` escape hatch and path fields still reach backend internals,
        so the gate remains.)

        Imported lazily here (not at module top) to break the
        ``base ↔ _validate`` cyclic import that CodeQL flagged: ``_validate``
        references :class:`TrainSpec` only under ``TYPE_CHECKING``, so the
        runtime cycle is closed by deferring this import until first call.
        """
        from strands_robots.training._validate import validate_train_inputs

        return validate_train_inputs(spec)

    def prepare(self, spec: TrainSpec) -> None:
        """Optional one-time setup before :meth:`train`. Default no-op.

        Overridden by backends that need it: Cosmos converts the base
        checkpoint to PyTorch DCP; GR00T registers a modality-config ``.py``.
        LeRobot needs nothing here.
        """
        return None

    @abstractmethod
    def train(self, spec: TrainSpec) -> TrainResult:
        """Run the backend's training in-process and return the final result.

        Responsible for: building the backend's typed config from the
        :class:`TrainSpec`, wiring resume, selecting single- vs multi-GPU
        (``elastic_launch`` for ``num_gpus > 1``), invoking the backend's own
        training function, and surfacing the checkpoint dir + metrics verdict.

        Training is **synchronous**: this call blocks until the run finishes (or
        raises) and returns a terminal ``TrainResult`` (``success``/``error``)
        with ``metrics`` already populated - there is no detached job to poll.
        ``status()`` exists only for backends that CAN report on a separately
        launched, still-running job; the default returns an informative error.
        Every implementation MUST call :meth:`validate` first and fail closed.
        """

    def status(self, job_id: str) -> TrainResult:
        """Optional "RUNNING != learning" verdict for a separately launched job.

        Because :meth:`train` is synchronous and already returns the full
        ``metrics`` verdict, this is only meaningful for a job launched OUT of
        band (e.g. a long cosmos run started under an external launcher) that a
        caller wants to poll by id. Most backends do not track detached jobs, so
        the default returns an informative ``error``. Backends that DO override
        parse their training logs for ``latest_step`` / ``latest_loss`` / a
        ``learning`` boolean.
        """
        return TrainResult(
            status="error",
            job_id=job_id,
            message=(
                f"{self.provider_name}: status() polling is not supported - "
                "train() runs synchronously and already returns the metrics verdict."
            ),
        )

    def export(self, spec: TrainSpec, checkpoint_dir: str) -> str:
        """Produce a loadable artifact from a checkpoint.

        Default returns ``checkpoint_dir`` unchanged - correct for HF-native
        backends (LeRobot, GR00T) whose checkpoints are directly loadable by
        ``create_policy(checkpoint_dir)``. Cosmos overrides to convert DCP ->
        safetensors. The returned path MUST be something ``create_policy``
        accepts.
        """
        return checkpoint_dir

    def latest_checkpoint(self, output_dir: str) -> str | None:
        """Return the newest loadable checkpoint directory under ``output_dir``.

        A loadable directory is one that ``export``/``create_policy`` can consume
        (for HF-native backends, the saved model dir). Returns ``None`` when no
        checkpoint exists yet, or when the backend writes no discoverable
        checkpoint tree. Powers the ``export`` action (which needs a checkpoint
        to convert) and resume logic.

        Default returns ``None`` (no discovery). Backends that write a
        predictable checkpoint layout override this. Pure / read-only (stat only).
        """
        return None

    @property
    def hardware_floor(self) -> dict[str, Any]:
        """Advisory minimum hardware, for the ``plan`` advisor.

        Keys: ``min_gpus`` (int), ``min_vram_gb`` (int),
        ``multinode`` (bool). Defaults to a single 24 GB GPU; backends with a
        higher floor (e.g. Cosmos: 8x80 GB) override.
        """
        return {"min_gpus": 1, "min_vram_gb": 24, "multinode": False}
