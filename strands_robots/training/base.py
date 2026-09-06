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

Those three are **local** backends: they run in-process (imported and called as
libraries, no subprocess) and multi-GPU goes through torch's programmatic
``elastic_launch``. A provider may instead be pure **transport**:

* **SageMaker** - submit the same spec as one managed AWS training job whose
  container image packages one of the local paths, so this provider imports no
  training library and the run outlives the submitting process.

Which shape a provider is decides its :meth:`Trainer.train` return contract - a
local run always finishes inside the call, a submitted one need not.

All of them nonetheless converge on:

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
            it (tolerance rule). Mutually exclusive with :attr:`val_episodes` on
            the lerobot backend: a held-out split sends lerobot down a map-style
            path that rebuilds both splits as ``LeRobotDataset`` and drops the
            stream, so asking for both materializes the whole dataset. The
            backend refuses the pair in :meth:`Trainer.validate` rather than
            deliver one of the two silently.
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
            overrides this default with a concrete value. An explicit value
            must be a positive finite number: ``0`` trains for the full run
            without updating a weight and ``inf`` writes a checkpoint of
            ``NaN``, neither of which any backend can report, so both are
            refused by :meth:`Trainer.validate` before a run starts.
        save_freq: Checkpoint cadence in steps (lerobot ``--save_freq`` /
            ``cfg.save_freq``, GR00T ``--save_steps``, Cosmos
            ``checkpoint.save_iter``). A whole number: it is consumed as the
            modulus of a ``step % save_freq`` test, so ``True`` is a cadence of
            one that checkpoints every step, a fractional or non-finite value
            never satisfies the test and silently disables periodic saving, and
            a string raises out of the comparison inside the training loop -
            none of which any backend reports, which is why a backend that
            reads it MUST check it through
            :meth:`Trainer._checkpoint_cadence_problems` rather than forward it.
            A non-positive value is a capability rather than an unusable one: it
            disables periodic saving so only the final checkpoint is written
            (lerobot's ``should_save_checkpoint``), and the backends' own
            ``eval_steps`` fallback is written for that case.
        num_gpus: GPUs on this node. ``>1`` runs the backend under torch's
            in-process ``elastic_launch`` (the engine behind ``torchrun``).
            A positive integer; a non-positive, fractional, non-finite or
            ``bool`` process count cannot be honored - the ``>1`` selector
            would silently route it to a single-process run, or hand it to
            ``elastic_launch`` as the worker count - so it is refused by
            :meth:`Trainer.validate` before a run starts.
        num_nodes: Nodes for multi-node training (Cosmos HSDP /
            ``torchrun --nnodes``). Same positive-integer domain as
            ``num_gpus``, and for the same reason.
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
            means "use the policy's built-in default targets". ``lora_r`` is the
            adapter rank and ``lora_alpha`` the numerator of its ``lora_alpha /
            lora_r`` scaling, so each must be a positive integer or ``None``
            (keep peft's own default); a backend that reads them MUST check them
            through :meth:`Trainer._lora_hyperparameter_problems` rather than
            leave them to peft, which judges only the rank and only once the base
            model is loaded.
        tune: Fine-grained component toggles for backends that expose them
            (GR00T: ``{"llm": bool, "visual": bool, "projector": bool,
            "diffusion": bool}``). Ignored by backends that don't.
        val_episodes: Hold out the LAST N episodes as a validation set
            (deterministic split). A backend MUST make the reserved episodes
            produce a validation signal, not merely shrink the training set;
            the lerobot backend maps it onto ``--dataset.eval_split`` plus a
            non-zero ``--eval_steps`` so an eval loss is logged periodically.
            Must be a positive integer below the dataset's episode count, or
            ``None`` to train on every episode. That upper bound is also a SOURCE
            requirement: the count comes from the dataset's local
            ``meta/info.json``, so a backend that cannot read one (a Hub source
            with no populated :attr:`dataset_root`) MUST refuse rather than emit
            no split - an absent split is indistinguishable from ``None``, and
            reporting no problem would launch exactly the validation-less run
            this field exists to avoid. Because the count is converted
            into a real-valued split fraction whose ceiling lerobot takes, a
            backend MUST check it with
            :meth:`Trainer._validation_episodes_problems` rather than compare it
            itself: a non-positive value produces no split at all, and ``True``
            / ``2.7`` / ``0.5`` reserve 1 / 3 / 0 episodes respectively. On the
            lerobot backend this is mutually exclusive with :attr:`streaming`
            (see that field).
        augmentation: Backend-specific data augmentation (GR00T
            ``color_jitter_params`` / ``random_rotation_angle``; Cosmos
            dataset filter dict).
        fps: Dataset control rate, when a backend needs it explicitly.
        extra: Raw passthrough. Keys become backend-native flags / overrides
            (lerobot ``--key=value``; Cosmos Hydra ``key.path=value``). The
            escape hatch that keeps the ABC stable as backends evolve. A value
            may be given either as text or as the destination field's own Python
            type; a backend that assigns into a typed config MUST decode text
            with the same decoder its ``--key=value`` form uses, so one spec
            means one run whichever path the backend takes. The lerobot backend
            reads text through lerobot's own draccus decoder: ``false``/``no``/
            ``off`` and ``true``/``yes``/``on`` (any case) are booleans, ``0``
            and ``1`` are not, and text that does not decode to the field's type
            is refused rather than stored. For example, unfreezing a SmolVLA
            vision tower takes ``extra={"policy.freeze_vision_encoder": False,
            "policy.train_expert_only": False}`` - or the same values as
            ``"false"`` - because both default to ``True`` in that policy's own
            config.
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

    Concrete trainers come in two shapes, and neither reimplements training. A
    **local** trainer is a thin adapter that **imports the backend package and
    calls its own training function in-process** (LeRobot ``train(cfg)``, GR00T
    ``experiment.run(config)``, Cosmos ``train.launch(config, args)``) - it does
    NOT shell out to a subprocess, and multi-GPU is driven via torch's
    programmatic ``elastic_launch`` (the engine behind ``torchrun``), still
    in-process. A **transport** trainer imports no training library at all: it
    submits the same :class:`TrainSpec` to a managed runner whose image packages
    a local trainer (``sagemaker`` -> one SageMaker training job). Only the
    local shape necessarily finishes inside the :meth:`train` call; see that
    method for the return contract that follows from this.

    The field-scoped shared domains below (:meth:`_seed_problems` and its
    siblings) are each obliged of "a backend that reads the field", and a
    backend reads a field by any means: naming it (``spec.seed``) or forwarding
    it through a table (``getattr(spec, field)`` over a tuple of field names,
    which is how a provider that passes a spec on serializes every field
    without naming any). The second form is a read for this rule exactly as the
    first is - what obliges the gate is that the value reaches the run, not the
    syntax that fetched it.
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

    def _run_size_problems(self, spec: TrainSpec) -> list[str]:
        """Run-size preflight shared by every backend that consumes it.

        Returns problems for :attr:`TrainSpec.steps` /
        :attr:`TrainSpec.global_batch_size` - the two factors of how much
        training the spec asks for - against the one shared positive-count
        domain. A :meth:`validate` implementation that reads either field MUST
        call this instead of comparing the value itself: a local ``<= 0`` test
        admits a ``bool`` as a silent run of one step, admits a fractional or
        non-finite value that then raises inside the backend's ``range()``
        after the dataset and model are already loaded, and raises out of the
        comparison itself for a non-numeric value - from a method documented to
        *return* problems.

        A backend that drives training from other fields (the RL trainers, on
        ``total_timesteps`` / ``batch_size``) MUST NOT call this: per
        :class:`TrainSpec`, a backend ignores the fields it does not support,
        so reporting on one it never reads would be a false rejection.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import run_size_problems

        return run_size_problems(spec, context=self.provider_name)

    def _rl_run_size_problems(self, spec: TrainSpec) -> list[str]:
        """Run-size preflight shared by every RL backend, the peer of the above.

        Returns problems for :attr:`RLTrainSpec.total_timesteps` /
        :attr:`RLTrainSpec.rollout_steps` - the two caller-supplied factors of
        the training-loop bound every RL backend derives,
        ``max(1, total_timesteps // (rollout_steps * num_envs))`` iterated as
        ``range(...)`` - against the same shared positive-count domain
        :meth:`_run_size_problems` uses for the supervised pair.

        This exists as a second gate rather than as part of that one because the
        two field sets are disjoint: per :class:`TrainSpec` a backend ignores the
        fields it does not support, so an RL trainer must not be refused for a
        ``steps`` it never reads, and a supervised backend must not be refused
        for a ``total_timesteps`` it never reads. The *domain* is shared; only the
        fields differ.

        A :meth:`validate` implementation that sizes its loop from either field
        MUST call this instead of comparing the value itself. The local ``<= 0``
        test is weaker here than for the supervised pair, because the bound is
        derived: the ``max(1, ...)`` clamp reads ``True``, a fraction below one
        iteration, ``nan`` and ``inf`` as a **single iteration** and the run then
        reports success and writes a checkpoint, so the run the caller asked for
        is simply not the one that happened. See
        :func:`~strands_robots.training._validate.rl_run_size_problems` for the
        measured table.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import rl_run_size_problems

        return rl_run_size_problems(spec, context=self.provider_name)

    def _rl_replay_problems(self, spec: TrainSpec) -> list[str]:
        """Replay-loop count preflight for the off-policy (SAC / TD3) backends.

        Returns problems for :attr:`RLTrainSpec.buffer_size` /
        :attr:`RLTrainSpec.batch_size` / :attr:`RLTrainSpec.gradient_steps` - the
        three caller-supplied counts of an off-policy replay loop (the buffer
        capacity, the transitions sampled per gradient step, and the updates per
        iteration) - against the same shared positive-count domain the run-size
        and launch-topology gates use.

        A :meth:`validate` that reads any of the three MUST call this instead of
        comparing the value itself. Each is consumed directly as a count (a
        tensor capacity, a sample size, a ``range()`` bound), so a local
        ``<= 0`` test is weaker: it reads ``True`` as a degenerate one-slot
        buffer / batch of one (a run that learns nothing and reports success),
        lets a fraction or non-finite value raise deep inside ``setup`` or the
        update loop, and raises ``TypeError`` itself on a string or ``None``. See
        :func:`~strands_robots.training._validate.rl_replay_problems` for the
        measured table.

        Only the off-policy backends (FastSAC and FastTD3) read these fields;
        PPO sizes its minibatches from ``num_mini_batches`` and never reads
        them, so it must not report on them.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import rl_replay_problems

        return rl_replay_problems(spec, context=self.provider_name)

    def _learning_rate_problems(self, spec: TrainSpec) -> list[str]:
        """Learning-rate preflight shared by EVERY backend.

        Returns a problem when :attr:`TrainSpec.learning_rate` is supplied and
        is not a usable positive finite number, against the one shared
        continuous domain. Unlike :meth:`_run_size_problems` there is no
        backend that may skip this: the supervised backends assign the value to
        their config's optimizer field and the RL trainers pass it to
        ``torch.optim.Adam``, so every concrete :meth:`validate` MUST call it.

        Checking it here rather than leaving it to the optimizer matters
        because the silent ends of the domain never reach an exception: ``0``
        does the full run and updates nothing, and ``inf`` writes a checkpoint
        of ``NaN`` - both under a successful result. The values the optimizer
        *does* reject only reach it after the dataset and model are loaded,
        which is what this preflight exists to precede.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import learning_rate_problems

        return learning_rate_problems(spec, context=self.provider_name)

    def _launch_topology_problems(self, spec: TrainSpec) -> list[str]:
        """Launch-topology preflight shared by every backend that consumes it.

        Returns problems for :attr:`TrainSpec.num_gpus` /
        :attr:`TrainSpec.num_nodes` - the two process counts a distributed
        launch is sized from - against the same shared positive-count domain
        :meth:`_run_size_problems` uses. A :meth:`validate` implementation that
        reads either field MUST call this instead of comparing the value
        itself: the ``> 1`` test that selects the multi-process launch path
        reads ``0``, a negative, ``nan`` and ``True`` as "not more than one" and
        silently runs on a single process, passes ``2.7`` and ``inf`` through to
        ``elastic_launch`` (which accepts them), and raises ``TypeError`` for a
        string or ``None`` - from a method documented to *return* problems.

        A backend that launches from neither field MUST NOT call this: per
        :class:`TrainSpec`, a backend ignores the fields it does not support, so
        reporting on one it never reads would be a false rejection. That is why
        this is a separate gate from :meth:`_learning_rate_problems`, which
        every backend does call.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import launch_topology_problems

        return launch_topology_problems(spec, context=self.provider_name)

    def _seed_problems(self, spec: TrainSpec) -> list[str]:
        """Reproducibility-seed preflight shared by every backend that reads it.

        Returns a problem when :attr:`TrainSpec.seed` is supplied and is not a
        usable non-negative integer, against the one shared non-negative-count
        domain. A :meth:`validate` implementation that reads the field MUST call
        this rather than pass the value straight to its applier: the appliers do
        not agree about it. ``torch.manual_seed`` takes the value modulo
        ``2**64``, so a negative seed silently becomes a large positive one and
        collides with a seed a caller could legitimately have named, while
        lerobot's ``set_seed`` reseeds Python's ``random`` and only then hands
        the value to NumPy, which refuses a negative - leaving the process RNG
        reseeded by a call that failed.

        A backend that does not read the field MUST NOT call this: per
        :class:`TrainSpec`, a backend ignores the fields it does not support, so
        reporting on one it never reads would be a false rejection. That is why
        this is a separate gate from :meth:`_learning_rate_problems`, which
        every backend does call.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import seed_problems

        return seed_problems(spec, context=self.provider_name)

    def _checkpoint_cadence_problems(self, spec: TrainSpec) -> list[str]:
        """Checkpoint-cadence preflight shared by every backend that reads it.

        Returns a problem when :attr:`TrainSpec.save_freq` is not a whole number
        of steps, against the one shared step-cadence domain the
        ``lerobot_train`` tool holds the same field to. A :meth:`validate`
        implementation that reads the field MUST call this rather than forward
        the value: every destination requires a genuine ``int`` and none of them
        says so. LeRobot in-process hands it straight to lerobot's
        ``should_save_checkpoint`` (``save_freq > 0 and step % save_freq == 0``),
        where ``True`` is a modulus of one and checkpoints every step while a
        fractional or non-finite cadence silently becomes the *disabled* mode
        and a string raises ``TypeError`` from inside the training loop; the
        argv, Hydra and hyperparameter routes each fail differently, or not at
        all, after the run has started. Only the type is graded - a non-positive
        cadence is the documented "disable periodic saving" mode.

        A backend that does not read the field MUST NOT call this: per
        :class:`TrainSpec`, a backend ignores the fields it does not support, so
        reporting on one it never reads would be a false rejection. That is why
        this is a separate gate from :meth:`_learning_rate_problems`, which
        every backend does call.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import checkpoint_cadence_problems

        return checkpoint_cadence_problems(spec, context=self.provider_name)

    def _rl_checkpoint_interval_problems(self, spec: TrainSpec) -> list[str]:
        """Checkpoint-cadence preflight for the RL loop, on its own field.

        Returns a problem when :attr:`RLTrainSpec.log_interval` is not a whole
        number of iterations, against the same shared step-cadence domain
        :meth:`_checkpoint_cadence_problems` holds ``save_freq`` to. A
        :meth:`validate` implementation whose loop paces ``save_checkpoint`` on
        ``it % log_interval`` MUST call this: the field is the RL run's
        checkpoint cadence, and the modulus judges it no more than lerobot's
        does. ``nan`` satisfies the truthiness guard and never the modulus, so a
        run that asked to checkpoint every few iterations silently keeps only
        its final one and still reports ``status="success"`` - the reading that
        matters most for RL, where return is non-monotonic and the deployable
        policy is often an earlier checkpoint. ``True`` is a cadence of one, a
        fraction is a silently different cadence, and a string raises
        ``TypeError`` out of the loop after ``setup`` has built the env, the
        networks and the optimizers. Only the type is graded: ``0`` is the
        documented "no intermediate checkpoints" mode.

        Scoped like :meth:`_network_width_problems` rather than like
        :meth:`_gae_lambda_problems`: all three RL backends run the same loop
        over the same field, so there is no RL backend for which reporting on it
        would be a false rejection. A supervised backend does not read it and
        MUST NOT report on it - it has ``save_freq`` for the same question.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.

        Args:
            spec: The spec to preflight.

        Returns:
            A single problem when the cadence cannot be honored; empty
            otherwise.
        """
        from strands_robots.training._validate import rl_checkpoint_interval_problems

        return rl_checkpoint_interval_problems(spec, context=self.provider_name)

    def _validation_episodes_problems(self, spec: TrainSpec) -> list[str]:
        """Held-out-validation preflight shared by every backend that reads it.

        Returns a problem when :attr:`TrainSpec.val_episodes` is supplied and is
        not a usable positive integer, against the same shared positive-count
        domain :meth:`_run_size_problems` uses. A :meth:`validate`
        implementation that reads the field MUST call this instead of comparing
        the value itself, because the count is converted into a real-valued
        split fraction and the comparison is wrong at both ends: a non-positive
        value produces no split and no evaluation cadence at all - the run
        trains on the whole dataset and records no validation loss, with nothing
        reported - while ``True`` reserves one episode, ``2.7`` reserves three,
        and ``0.5`` emits an evaluation cadence over a held-out set of zero
        episodes. A non-numeric value raises out of the comparison from a method
        documented to *return* problems.

        The dataset-dependent upper bound (``val_episodes`` must leave training
        data behind) stays with the backend, which is the side that reads
        ``total_episodes`` from the dataset metadata.

        A backend that does not read the field MUST NOT call this: per
        :class:`TrainSpec`, a backend ignores the fields it does not support, so
        reporting on one it never reads would be a false rejection. That is why
        this is a separate gate from :meth:`_learning_rate_problems`, which
        every backend does call.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import validation_episodes_problems

        return validation_episodes_problems(spec, context=self.provider_name)

    def _lora_hyperparameter_problems(self, spec: TrainSpec) -> list[str]:
        """LoRA adapter preflight shared by every backend that reads it.

        Returns a problem per supplied-and-unusable :attr:`TrainSpec.lora_r` /
        :attr:`TrainSpec.lora_alpha`, against the same shared positive-count
        domain :meth:`_run_size_problems` uses. A :meth:`validate`
        implementation that reads either field MUST call this instead of leaving
        the values to peft, because peft only judges one of them: it refuses a
        non-positive ``lora_r`` from inside ``get_peft_model``, after the base
        model is already loaded, while ``lora_alpha`` is a bare numerator that
        nothing compares - ``lora_alpha=0`` trains an adapter whose scaling is
        ``0.0`` and which therefore cannot change the model's output, and a
        negative value applies the negation of what it learned.

        A backend that does not read the fields MUST NOT call this: per
        :class:`TrainSpec`, a backend ignores the fields it does not support, so
        reporting on ones it never reads would be a false rejection. That is why
        this is a separate gate from :meth:`_learning_rate_problems`, which every
        backend does call.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import lora_hyperparameter_problems

        return lora_hyperparameter_problems(spec, context=self.provider_name)

    def _discount_factor_problems(self, spec: TrainSpec) -> list[str]:
        """Discount-factor preflight shared by every RL backend.

        Returns a problem when :attr:`RLTrainSpec.gamma` is not a real number in
        the closed interval ``[0, 1]``. A :meth:`validate` implementation that
        discounts a return with the field MUST call this instead of leaving the
        value to the arithmetic that consumes it, because that arithmetic never
        judges it: ``gamma > 1`` makes the discounted return diverge in the
        rollout horizon and the run still reports success and writes a
        checkpoint, ``gamma < 0`` alternates the sign of each future reward so
        the trace stops accumulating return at all, and a non-finite value
        surfaces only once the update samples the action distribution - as a
        torch constraint error naming neither the field nor the run, after the
        env, the networks and a full rollout have been built.

        Both interval endpoints are inside the domain (``gamma=1`` is the
        undiscounted episodic return, ``gamma=0`` a myopic agent), so the check
        is a closed interval rather than a positivity test - and it rejects
        ``bool``, which a bare comparison against the bounds accepts as a silent
        ``gamma`` of one.

        Every RL backend reads the field, so unlike
        :meth:`_lora_hyperparameter_problems` there is no backend for which
        reporting on it would be a false rejection.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import discount_factor_problems

        return discount_factor_problems(spec, context=self.provider_name)

    def _gae_lambda_problems(self, spec: TrainSpec) -> list[str]:
        """GAE-lambda preflight shared by every backend that estimates a trace.

        Returns a problem when :attr:`RLTrainSpec.lam` is not a real number in
        the closed interval ``[0, 1]``. A :meth:`validate` implementation that
        decays an advantage trace with the field MUST call this **in addition
        to** :meth:`_discount_factor_problems`, because the two fields are one
        contract: the trace decays by the product ``gamma * lam``, so bounding
        ``gamma`` alone leaves the divergence that gate exists to refuse
        reachable through the other factor - a ``gamma`` of ``0.99`` with a
        ``lam`` of ``1.5`` decays by ``1.485`` and the largest advantage grows
        without bound in the rollout horizon, under a successful run that writes
        a checkpoint.

        Both interval endpoints are inside the domain (``lam=1`` is the
        Monte-Carlo advantage, ``lam=0`` the one-step TD advantage), so the check
        is a closed interval rather than a positivity test - and it rejects
        ``bool``, which a bare comparison against the bounds accepts as a silent
        ``lam`` of one, i.e. a different estimator from the requested one.

        Only the on-policy backend estimates an advantage trace, so unlike
        :meth:`_discount_factor_problems` a backend that does not read the field
        MUST NOT call this: per :class:`TrainSpec` a backend ignores the fields
        it does not support, so reporting on one it never reads would be a false
        rejection.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import gae_lambda_problems

        return gae_lambda_problems(spec, context=self.provider_name)

    def _optimization_epochs_problems(self, spec: TrainSpec) -> list[str]:
        """Optimization-epoch preflight for a backend with an epoch loop.

        Returns a problem when :attr:`RLTrainSpec.num_learning_epochs` is not a
        positive integer. A :meth:`validate` implementation whose update makes
        ``num_learning_epochs`` passes over each rollout batch MUST call this,
        because the field is the loop bound of the whole optimizer step: a
        non-positive value takes no gradient step at all, and the run still
        collects its rollouts, writes a checkpoint and reports success with
        losses of ``0.0`` (the update averages through ``max(1, n_updates)``).
        ``True`` is likewise a silent single epoch, and a non-integer raises a
        bare ``TypeError`` out of ``range()`` after the environment and the
        networks have been built.

        Only a backend that loops over a rollout batch reads the field - an
        off-policy backend optimizes per gradient step from a replay buffer - so
        like :meth:`_gae_lambda_problems`, and unlike
        :meth:`_discount_factor_problems`, a backend that does not read it MUST
        NOT call this: per :class:`TrainSpec` a backend ignores the fields it
        does not support, so reporting on one it never reads would be a false
        rejection.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import optimization_epochs_problems

        return optimization_epochs_problems(spec, context=self.provider_name)

    def _temperature_learning_rate_problems(self, spec: TrainSpec) -> list[str]:
        """Entropy-temperature learning-rate preflight, for backends that tune one.

        Returns a problem when a spec asking for an automatically tuned
        temperature carries an :attr:`RLTrainSpec.alpha_lr` that is not a positive
        finite number. A :meth:`validate` implementation that builds a temperature
        optimizer MUST call this **in addition to**
        :meth:`_learning_rate_problems`: the two fields are separate learning
        rates on separate optimizers, so guarding the one that drives the actor
        and critics leaves the temperature's own rate unchecked. ``alpha_lr=0``
        freezes the temperature at ``init_alpha`` for the whole run - the
        requested automatic tuning silently does not happen - and ``alpha_lr=inf``
        writes a checkpoint holding non-finite parameters, both under a
        successful result.

        Only the SAC backend tunes a temperature, so unlike
        :meth:`_learning_rate_problems` a backend that does not read the field
        MUST NOT call this: per :class:`TrainSpec` a backend ignores the fields it
        does not support, so reporting on one it never reads would be a false
        rejection. For the same reason the check is inert unless the spec's
        ``autotune_alpha`` is set, since that is the only branch that constructs a
        temperature optimizer.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.
        """
        from strands_robots.training._validate import temperature_learning_rate_problems

        return temperature_learning_rate_problems(spec, context=self.provider_name)

    def _initial_temperature_problems(self, spec: TrainSpec) -> list[str]:
        """Entropy-temperature starting-value preflight, for backends that hold one.

        Returns a problem when :attr:`RLTrainSpec.init_alpha` is not a positive
        finite number. FastSAC stores the temperature's logarithm, so the field
        reaches ``torch.log`` and only a positive finite value has a finite one.
        A :meth:`validate` implementation that builds a temperature MUST call
        this **in addition to** :meth:`_temperature_learning_rate_problems`: the
        two fields are the temperature's starting value and the rate that moves
        it, so guarding the rate leaves the value it starts from unchecked -
        which that gate's own reasoning depends on, since it refuses
        ``alpha_lr=0`` on the grounds that "the temperature stays at
        ``init_alpha`` for the whole run".

        ``init_alpha=0`` makes ``log(0) == -inf`` and the temperature exactly
        zero, so the entropy term is absent from both losses and automatic
        tuning cannot lift it back - no finite update moves an infinity - while
        the run reports success and checkpoints the non-finite value. A negative
        value, ``nan`` or ``inf`` poisons the actor loss instead and raises from
        inside ``torch.distributions.Normal``, naming that distribution's
        parameter rather than the field.

        Unlike :meth:`_temperature_learning_rate_problems` this is not scoped to
        ``autotune_alpha``: that gate guards an optimizer only the tuning branch
        constructs, while ``init_alpha`` is read on both branches, and with
        tuning off it is the temperature for the entire run.

        Only a backend that holds an entropy temperature may call this: like
        :meth:`_gae_lambda_problems`, and unlike :meth:`_learning_rate_problems`,
        a backend that does not read the field MUST NOT report on it, because
        per :class:`TrainSpec` a backend ignores the fields it does not support.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.

        Args:
            spec: The spec to preflight.

        Returns:
            A single-element list when ``init_alpha`` has no finite logarithm;
            empty when it is usable.
        """
        from strands_robots.training._validate import initial_temperature_problems

        return initial_temperature_problems(spec, context=self.provider_name)

    def _target_entropy_problems(self, spec: TrainSpec) -> list[str]:
        """Target-entropy preflight, for backends that tune a temperature.

        Returns a problem when :attr:`RLTrainSpec.target_entropy` is neither the
        ``None`` sentinel nor a finite real of either sign. It is the third field
        of FastSAC's temperature block, and a backend that builds that block MUST
        call this **alongside** :meth:`_initial_temperature_problems` and
        :meth:`_temperature_learning_rate_problems`: those two guard the
        temperature's starting value and the rate that moves it, and this one the
        constant it is moved *toward*, so guarding two of the three leaves the
        third to the arithmetic that spends it.

        The domain is signed, which is why it is
        :func:`~strands_robots.utils.finite_number_error` rather than the
        positive-finite domain its two neighbours read: the field defaults to
        ``-num_actions``, so every reading of it is a negative entropy in nats and
        no endpoint is decidable. ``target_entropy=True`` is therefore not merely
        a flag read as a number but a silent sign flip - a target of ``+1.0`` -
        and a run that took it reported success while checkpointing a different
        temperature. ``nan`` poisons ``alpha``, which scales the entropy term of
        both the critic target and the actor loss, and the next rollout raises
        from inside ``torch.distributions.Normal`` about ``nan`` policy means; a
        list or a dict raises ``TypeError`` out of the ``float()`` coercion in
        ``setup``.

        ``None`` is exempt rather than refused: unlike ``init_alpha`` and
        ``alpha_lr`` this field is annotated ``float | None``, and ``None`` is the
        documented request for the ``-num_actions`` heuristic.

        Like :meth:`_initial_temperature_problems` this is not scoped to
        ``autotune_alpha``: the coercion in ``setup`` is unconditional, so a
        non-real value raises on either branch.

        Only a backend that optimizes a temperature against a target entropy may
        call this: like :meth:`_gae_lambda_problems`, and unlike
        :meth:`_learning_rate_problems`, a backend that does not read the field
        MUST NOT report on it, because per :class:`TrainSpec` a backend ignores
        the fields it does not support.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.

        Args:
            spec: The spec to preflight.

        Returns:
            A single-element list when ``target_entropy`` cannot be honored;
            empty when it can.
        """
        from strands_robots.training._validate import target_entropy_problems

        return target_entropy_problems(spec, context=self.provider_name)

    def _polyak_coefficient_problems(self, spec: TrainSpec) -> list[str]:
        """Polyak-coefficient preflight, for a backend that keeps a target network.

        Returns a problem when :attr:`RLTrainSpec.tau` is not a real number in
        ``(0, 1]``. It is the rate at which a target network tracks its online
        network, spent in one expression per mirrored critic pair,
        ``tp.mul_(1.0 - spec.tau).add_(spec.tau * p)``, so it decides whether a
        separate target network exists at all rather than merely how fast it
        moves.

        The interval is the one the two on-policy interval gates cite as their
        precedent - :meth:`_discount_factor_problems` and
        :meth:`_gae_lambda_problems` both generalize "``tau`` must be in
        ``(0, 1]``" - and it is half-open where theirs is closed because zero is
        a degenerate spelling here: it freezes the target parameters at their
        initialization for the whole run. The upper endpoint stays inside, being
        the deliberate hard update ``tp = p``.

        The two backends each carried a bare local interval comparison against
        those bounds instead, which admitted ``True`` as a silent ``tau`` of one -
        a target network that is a copy of the online network, measured as an
        exactly zero online-to-target gap in the exported checkpoint of a run
        that reported success - and raised ``TypeError`` out of the comparison
        itself on a numeric string, ``None`` or a list, from a :meth:`validate`
        documented to *return* its problems.

        Only a backend that maintains a target network may call this: like
        :meth:`_gae_lambda_problems`, and unlike :meth:`_learning_rate_problems`,
        a backend that does not read the field MUST NOT report on it, because
        per :class:`TrainSpec` a backend ignores the fields it does not support.
        PPO has no target network and never reads ``tau``.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.

        Args:
            spec: The spec to preflight.

        Returns:
            A single-element list when ``tau`` cannot be honored; empty when it
            can.
        """
        from strands_robots.training._validate import polyak_coefficient_problems

        return polyak_coefficient_problems(spec, context=self.provider_name)

    def _spec_device_problems(self, spec: TrainSpec) -> list[str]:
        """Device preflight for a backend that places its tensors from the spec.

        Returns a problem when :attr:`RLTrainSpec.device` is not a device string
        torch can parse. Distinct from
        :meth:`~strands_robots.training.lerobot.LerobotTrainer._device_problems`,
        which grades that trainer's ``device`` *constructor* knob: this one grades
        the field on the spec, which is where the from-scratch RL backends carry
        it. Both consult one domain,
        :func:`~strands_robots.utils.torch_device_error`.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the ``base -> _validate`` import one-way at runtime.

        Args:
            spec: The spec to preflight.

        Returns:
            A single-element list when ``device`` cannot be honored; empty when
            it can or when it is unstated.
        """
        from strands_robots.training._validate import torch_device_problems

        return torch_device_problems(spec, context=self.provider_name)

    def _gradient_clip_problems(self, spec: TrainSpec) -> list[str]:
        """Gradient-clip preflight for a backend that clips before it steps.

        Returns a problem when :attr:`RLTrainSpec.max_grad_norm` is not a
        positive real number. Positive infinity is inside the domain: it is the
        field's only spelling of "do not clip", and ``clip_grad_norm_`` honors
        it by leaving every gradient untouched.

        Zero and negative values are not degenerate spellings of that, which is
        why this is a positivity test rather than a non-negative one. Zero
        scales every gradient to zero, so the optimizer steps with no
        information and the run writes a checkpoint bit-identical to a
        never-trained one; a negative bound negates the scaling ratio, so the
        update becomes gradient *ascent* on the loss. Both report success.

        Only a backend that clips may call this: like
        :meth:`_gae_lambda_problems`, and unlike
        :meth:`_learning_rate_problems`, a backend that does not read the field
        MUST NOT report on it, because per :class:`TrainSpec` a backend ignores
        the fields it does not support.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the module import graph one-way.

        Args:
            spec: The spec to preflight.

        Returns:
            A single-element list when the field cannot be honored; empty
            otherwise.
        """
        from strands_robots.training._validate import gradient_clip_problems

        return gradient_clip_problems(spec, context=self.provider_name)

    def _loss_weight_problems(self, spec: TrainSpec) -> list[str]:
        """Loss-weight preflight for a backend that composes a weighted objective.

        Returns a problem per weight when :attr:`RLTrainSpec.value_loss_coef` or
        :attr:`RLTrainSpec.entropy_coef` is not a finite real number. Both are
        multiplied into the loss the update descends, and the multiplication
        judges nothing: a ``nan`` weight makes the loss ``nan``, the optimizer
        writes ``nan`` into every parameter, and the next rollout raises from
        inside ``torch`` naming neither field nor value - after the env, the
        networks and a full rollout have been built.

        The floor is deliberately *not* tested. Zero and negative are inside the
        domain for both: ``entropy_coef`` ships defaulting to ``0.0``, a negative
        entropy weight is a determinism penalty, and ``value_loss_coef=0`` stops
        training the critic. That is what distinguishes this from
        :meth:`_gradient_clip_problems`, whose endpoint is settled by
        ``clip_grad_norm_`` and which therefore tests positivity. A ``bool`` is
        refused here, because it reads as a flag and lands as a coefficient of
        one - turning the entropy bonus on at full weight where the field's
        default is off.

        Only a backend that composes this objective may call this: like
        :meth:`_gradient_clip_problems`, and unlike
        :meth:`_learning_rate_problems`, a backend that does not read the fields
        MUST NOT report on them, because per :class:`TrainSpec` a backend ignores
        the fields it does not support.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the module import graph one-way.

        Args:
            spec: The spec to preflight.

        Returns:
            One problem per weight that cannot be honored; empty when both can.
        """
        from strands_robots.training._validate import loss_weight_problems

        return loss_weight_problems(spec, context=self.provider_name)

    def _clip_range_problems(self, spec: TrainSpec) -> list[str]:
        """Trust-region preflight for a backend that clips a policy ratio.

        Returns a problem when :attr:`RLTrainSpec.clip_param` is not a positive
        real number. It is the half-width of the trust region the on-policy
        surrogate is clipped to, read twice per mini-batch - once for the policy
        ratio and once for the value loss - and ``torch.clamp`` judges it not at
        all, so every unusable value below produced a finite, successful,
        deployable run whose objective was not the configured one. A ``nan``
        half-width is the sharpest: both clipped terms become ``nan``, the
        gradient of ``torch.max`` flows to the *unclipped* branch because every
        comparison against ``nan`` is false, and the resulting checkpoint is
        bit-identical to an unclipped run while every reported loss is ``nan`` -
        the trust region silently gone with no signal a caller can act on. A
        negative half-width inverts the clamp bounds so they return a constant,
        a ``bool`` is a silent half-width of one, and a string, ``None`` or a
        list raises ``TypeError`` mid-update.

        Positive infinity is inside the domain: it is the field's only spelling
        of *do not clip*, and ``clamp(ratio, -inf, inf)`` honors it by returning
        the ratio unchanged. That is the same endpoint, for the same reason, as
        the sibling bound :meth:`_gradient_clip_problems`, so the two share one
        domain helper rather than carrying a copy each.

        Only a backend that clips a policy ratio may call this: like
        :meth:`_gradient_clip_problems`, and unlike
        :meth:`_learning_rate_problems`, a backend that does not read the field
        MUST NOT report on it, because per :class:`TrainSpec` a backend ignores
        the fields it does not support.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the module import graph one-way.

        Args:
            spec: The spec to preflight.

        Returns:
            A single-element list when ``clip_param`` cannot be honored; empty
            otherwise.
        """
        from strands_robots.training._validate import clip_range_problems

        return clip_range_problems(spec, context=self.provider_name)

    def _policy_delay_problems(self, spec: TrainSpec) -> list[str]:
        """Policy-delay preflight for a backend that delays its actor updates.

        Returns a problem when :attr:`RLTrainSpec.policy_delay` is not a
        positive integer. A :meth:`validate` implementation whose update gates
        the actor / target step on ``update_count % policy_delay == 0`` MUST
        call this, because the modulus judges nothing and its silent reading is
        the worst one: a value the test can never satisfy (``nan``, since
        ``n % nan`` is ``nan`` and compares unequal to everything) trains the
        critics for the whole run while the deployable actor never takes a
        gradient step - the run reports success and checkpoints an untrained
        policy. ``True`` is a silent delay of one, a fraction a silently
        different cadence, ``0`` a ``ZeroDivisionError`` and a string a
        ``TypeError`` - each from inside the update loop, after the env, the
        networks, the optimizers and the replay buffer are built.

        ``1`` is inside the domain: a delay of one is TD3 with the delay
        disabled, a configuration rather than a defect.

        Only a backend that delays its policy may call this: like
        :meth:`_gae_lambda_problems`, and unlike
        :meth:`_learning_rate_problems`, a backend that does not read the field
        MUST NOT report on it, because per :class:`TrainSpec` a backend ignores
        the fields it does not support.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the module import graph one-way.

        Args:
            spec: The spec to preflight.

        Returns:
            A single-element list when ``policy_delay`` cannot be honored;
            empty otherwise.
        """
        from strands_robots.training._validate import policy_delay_problems

        return policy_delay_problems(spec, context=self.provider_name)

    def _td3_noise_problems(self, spec: TrainSpec) -> list[str]:
        """Noise-scale preflight for a backend built on a deterministic actor.

        Returns a problem per unusable :attr:`RLTrainSpec.exploration_noise_std`
        / :attr:`RLTrainSpec.target_noise_std` /
        :attr:`RLTrainSpec.target_noise_clip` - the three scalars of TD3's two
        noise mechanisms: the exploration noise that is a deterministic
        policy's only exploration once the random warmup ends, and the target
        policy smoothing that keeps the critic from exploiting its own sharp
        errors. A :meth:`validate` implementation that reads any of the three
        MUST call this, because the multiplications that consume them judge
        nothing: zero silently removes the mechanism (a collection that never
        explores; plain clipped double-Q reported as the smoothed algorithm), a
        negative scale is silently the identical distribution (Gaussian noise
        is symmetric) while a negative clip inverts the clamp into a constant
        bias, and a non-finite value poisons the actions or the TD target under
        a run that keeps stepping. Positive infinity has no "disable" reading
        here, unlike the clip bounds of :meth:`_gradient_clip_problems` - an
        infinite std is a coin-flip between the action bounds, not a large
        noise - so the domain is the plain positive-finite one.

        Only a backend that explores and smooths this way may call this: like
        :meth:`_gae_lambda_problems`, and unlike
        :meth:`_learning_rate_problems`, a backend that does not read the
        fields MUST NOT report on them, because per :class:`TrainSpec` a
        backend ignores the fields it does not support.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the module import graph one-way.

        Args:
            spec: The spec to preflight.

        Returns:
            One problem per noise scalar that cannot be honored; empty when all
            three can.
        """
        from strands_robots.training._validate import td3_noise_problems

        return td3_noise_problems(spec, context=self.provider_name)

    def _network_width_problems(self, spec: TrainSpec) -> list[str]:
        """Hidden-layer-width preflight for a from-scratch RL backend.

        Returns a problem per unusable width in
        :attr:`RLTrainSpec.hidden_dims`, named by index. A :meth:`validate`
        implementation that builds its networks from the field MUST call this,
        because the loop that expands it judges nothing and neither does
        ``nn.Linear``: a width of zero is a legal layer whose activation is
        empty, so the layer after it emits its bias alone and the network's
        output stops depending on the observation at all. The run still
        collects, still trains its critics against that constant, still returns
        ``status="success"``, and still exports a deployable checkpoint - one
        whose actor commands a single fixed action in every state. The empty
        sequence is a genuine linear policy and stays accepted, so the domain
        is per element rather than on the length.

        Scoped like :meth:`_learning_rate_problems` rather than like
        :meth:`_gae_lambda_problems`: every from-scratch RL backend builds its
        actor and critics from this field, so there is no RL backend for which
        reporting on it would be a false rejection. A supervised backend, which
        fine-tunes a pretrained policy whose architecture comes from the
        checkpoint rather than from the spec, does not read it and must not
        report on it.

        Imported lazily for the same reason as :meth:`_security_problems` - to
        keep the module import graph one-way.

        Args:
            spec: The spec to preflight.

        Returns:
            One problem per width that cannot be honored, or a single problem
            when the field is not a sequence of widths; empty when it is usable.
        """
        from strands_robots.training._validate import network_width_problems

        return network_width_problems(spec, context=self.provider_name)

    def prepare(self, spec: TrainSpec) -> None:
        """Optional one-time setup before :meth:`train`. Default no-op.

        Overridden by backends that need it: Cosmos converts the base
        checkpoint to PyTorch DCP; GR00T registers a modality-config ``.py``.
        LeRobot needs nothing here.
        """
        return None

    @abstractmethod
    def train(self, spec: TrainSpec) -> TrainResult:
        """Run the backend's training and return the result of that run.

        Responsible for: building the backend's typed config from the
        :class:`TrainSpec`, wiring resume, selecting single- vs multi-GPU
        (``elastic_launch`` for ``num_gpus > 1``), invoking the backend's own
        training function, and surfacing the checkpoint dir + metrics verdict.

        A **local** trainer runs the training in-process, so the call is
        **synchronous**: it blocks until the run finishes (or raises) and
        returns a terminal ``TrainResult`` (``success``/``error``) with
        ``metrics`` already populated, and there is no detached job to poll. A
        **transport** trainer submits a run that outlives this process, so its
        result MAY be non-terminal: ``running`` with a ``job_id`` that
        :meth:`status` polls, and no ``checkpoint_dir`` yet.

        A caller therefore has to branch on all three
        :attr:`TrainResult.status` values rather than read "not ``error``" as
        finished: a completed-run report rendered for a ``running`` result names
        an artifact that does not exist yet.
        Every implementation MUST call :meth:`validate` first and fail closed.
        """

    def status(self, job_id: str) -> TrainResult:
        """Optional "RUNNING != learning" verdict for a job still in flight.

        Two kinds of job reach here: one launched OUT of band (e.g. a long
        cosmos run started under an external launcher) that a caller wants to
        poll by id, and one a **transport** :meth:`train` submitted and handed
        back as ``running`` because it outlives the submitting process. A local
        trainer produces neither - its ``train`` already returned the full
        ``metrics`` verdict - so most backends track no job at all and inherit
        this default, which returns an informative ``error``. Backends that DO
        override either read the runner's own job API (``sagemaker`` ->
        ``DescribeTrainingJob``) or parse their training logs for
        ``latest_step`` / ``latest_loss`` / a ``learning`` boolean.
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
