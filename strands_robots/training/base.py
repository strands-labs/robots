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
        save_freq: Checkpoint cadence in steps.
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

        Only the off-policy backend tunes a temperature, so unlike
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
