"""Shared, defense-in-depth input validation for the training backends.

Every concrete :class:`~strands_robots.training.base.Trainer` translates a
:class:`~strands_robots.training.base.TrainSpec` into its backend's native
config object and runs it IN-PROCESS (imported and called as a library - no
subprocess). The ``train_policy`` ``@tool`` lets an agent (LLM) populate that
``TrainSpec`` directly, so the path fields and the free-form ``extra`` dict are
*untrusted input that reaches backend internals*. Per ``AGENTS.md`` > Review
Learnings (#92) > "LLM Input Safety", those values MUST be validated before they
can become a config field, a Hydra override, or a token in a backend's
argv-parity helper: a value beginning with ``-`` could read as a *new flag*, and
an arbitrary ``extra`` key could set an arbitrary config attribute / override.

:func:`validate_train_inputs` is the single source of that check. It is invoked
from every backend's :meth:`Trainer.validate`, which each backend's
:meth:`Trainer.train` calls (fail-closed) before building any config - so no
run can start with unvalidated input regardless of the call path.

:func:`run_size_problems` is the second shared gate, on a different axis: the
*run size* numerics. ``steps`` and ``global_batch_size`` are the two factors of
how much training a spec asks for, and both are read straight into a backend's
loop bound / dataloader. They live in their own gate rather than in
:func:`validate_train_inputs` because :class:`TrainSpec` documents that a
backend "reads the fields it supports and ignores the rest": the RL trainers
drive training from ``total_timesteps`` / ``batch_size`` and never read either
field, so reporting a problem for them there would be a false rejection of a
field that backend does not use.

:func:`learning_rate_problems` is the third, on the optimization axis. It lives
in its own gate for the opposite reason to :func:`run_size_problems`: *every*
backend reads ``learning_rate`` -- the three supervised ones map it onto their
config's optimizer field, and the RL trainers hand it straight to
``torch.optim.Adam`` -- so there is no backend for which reporting on it would
be a false rejection, and :class:`~strands_robots.training.rl.base_algo.RLTrainSpec`
documents the field as one of the "universal" ones. It is separate from
:func:`validate_train_inputs` because that gate answers a different question
(is this value safe to interpolate into a config or an argv token) from this one
(can this value be honored at all).

:func:`launch_topology_problems` is the fourth, on the *launch topology* axis:
``num_gpus`` and ``num_nodes``, the two process counts every distributed launch
is sized from. It is scoped like :func:`run_size_problems` rather than like
:func:`learning_rate_problems` - only the three supervised backends read either
field (they become a ``torchrun``/``elastic_launch`` ``nproc_per_node`` /
``nnodes``), so a backend that ignores them must not report on them.

:func:`seed_problems` is the fifth, on the reproducibility axis, and
:func:`validation_episodes_problems` the sixth, on the *evaluation* axis:
``val_episodes``, the episode count a caller reserves as a held-out validation
set. It is scoped like :func:`run_size_problems` - only the LeRobot backend
reads the field (GR00T, Cosmos and the RL trainers never do), so a backend that
ignores it must not report on it. What makes a shared gate the right home
rather than a local test is the conversion: the count becomes a real-valued
split fraction whose ceiling lerobot takes, so a comparison admits values that
reserve a different number of episodes than the one asked for.

:func:`lora_hyperparameter_problems` is the seventh, on the *adapter* axis:
``lora_r`` and ``lora_alpha``, the rank and the scaling numerator of a LoRA
fine-tune. It is scoped like :func:`run_size_problems` and narrowed once more -
only the LeRobot backend reads either field, and only on its ``method == "lora"``
branch, so a value a run's own strategy never reads must not be reported.

:func:`discount_factor_problems` is the eighth, on the *return* axis:
``gamma``, the discount factor of the return the algorithm optimizes. It is
scoped like :func:`learning_rate_problems` rather than like
:func:`run_size_problems` - it is the one
:class:`~strands_robots.training.rl.base_algo.RLTrainSpec` coefficient that
*every* RL backend reads (PPO discounts the GAE recursion with it, FastSAC
discounts its target-Q bootstrap), so there is no RL backend for which
reporting on it would be a false rejection.

:func:`gae_lambda_problems` is the ninth, on the same *return* axis and for the
sibling factor: ``lam``, the GAE trace-decay coefficient. It is a separate gate
from :func:`discount_factor_problems` because the two fields are scoped
differently - every RL backend reads ``gamma``, but only the on-policy backend
estimates an advantage trace, so per :class:`TrainSpec` FastSAC must not report
on a field it never reads. They are nonetheless one contract: the trace decays
by the *product* ``gamma * lam``, so bounding one factor does not bound the
trace.

:func:`optimization_epochs_problems` is the tenth, on the *optimization* axis:
``num_learning_epochs``, the number of passes the on-policy update makes over
each rollout batch. It is the loop bound of the entire optimizer step
(``for _ in range(spec.num_learning_epochs)`` wraps every ``optimizer.step()``),
so a non-positive value takes no gradient step at all while the run still
collects its rollouts, writes a deployable checkpoint and reports success. It is
scoped like :func:`gae_lambda_problems`: only the on-policy backend has an epoch
loop, so FastSAC must not report on a field it never reads.

:func:`temperature_learning_rate_problems` is the eleventh, on the same
*optimization* axis as :func:`learning_rate_problems` and for its sibling
field. FastSAC builds *two* optimizers from two separate learning-rate fields -
``learning_rate`` for the actor and both critics, and ``alpha_lr`` for the
entropy temperature - and passes each straight to
``torch.optim.Adam(..., lr=...)``, so the failure modes
:func:`learning_rate_problems` documents apply to the second field verbatim.
It is a separate gate for the same reason as :func:`gae_lambda_problems`: only
the off-policy backend tunes a temperature.

:func:`gradient_clip_problems` is the twelfth, on the same *optimization* axis:
``max_grad_norm``, the norm the on-policy update clips every gradient to before
stepping. It is scoped like :func:`gae_lambda_problems` - only the on-policy
backend clips, so FastSAC must not report on a field it never reads - and it is
the one coefficient of that group whose zero reading *is* settled, by
``torch.nn.utils.clip_grad_norm_`` itself: see that gate for the measurement.

:func:`loss_weight_problems` is the thirteenth, on the same *optimization* axis
and the last of that group whose endpoints are not the question: ``value_loss_coef``
and ``entropy_coef``, the two scalars that weight the terms of the composed
on-policy objective. It is scoped like :func:`gradient_clip_problems` - only the
on-policy backend composes that objective - and it bounds the *domain* rather than
the floor: zero and negative are real configurations for both fields, while a
non-finite or non-numeric weight is a value no reading makes usable.

:func:`clip_range_problems` is the fourteenth, and the second of the two clip
bounds on that same *optimization* axis: ``clip_param``, the half-width of the
trust region the on-policy surrogate is clipped to, which also clips the value
loss. It shares :func:`_clip_bound_error` with :func:`gradient_clip_problems`
because the two bounds have one domain for one reason - a clip bound is a
positive width, and positive infinity is each field's only spelling of "do not
clip". It is scoped like :func:`gae_lambda_problems`: ``spec.clip_param`` is read
in ``rl/ppo.py`` and nowhere else.

:func:`policy_delay_problems` is the fifteenth, on the *optimization* axis:
``policy_delay``, the number of critic updates FastTD3 runs per delayed actor /
target update. It is consumed as the modulus of an ``update_count %
policy_delay`` test, so a value that never satisfies the test silently trains
the critics for the whole run while the deployable actor never takes a
gradient step - under a successful result. It is scoped like
:func:`gae_lambda_problems`: only the TD3 backend delays its policy, so a
backend that does not read the field must not report on it.

:func:`td3_noise_problems` is the sixteenth, on the *exploration/target* axis:
``exploration_noise_std``, ``target_noise_std`` and ``target_noise_clip``, the
three scalars that shape FastTD3's two noise mechanisms - exploration noise on
collection and target policy smoothing in the critic's TD target. Gaussian
noise is symmetric, so a negative scale is silently the identical
distribution, zero silently removes the mechanism the field configures, and a
non-finite value poisons the actions or the TD target. It is scoped like
:func:`policy_delay_problems`: only the TD3 backend reads the three fields.

:func:`network_width_problems` is the seventeenth, on the *architecture* axis:
``hidden_dims``, the hidden layer widths every from-scratch RL backend builds
its actor and its critics from. It is scoped like
:func:`learning_rate_problems` rather than like :func:`gae_lambda_problems` -
all three RL backends read the field, for every network they construct - and it
is the only one of these gates whose field is a *sequence*, so the domain is
asked of each element in turn and the message names the offending index.

:func:`rl_checkpoint_interval_problems` is the eighteenth, and the second gate on
the *cadence* axis :func:`checkpoint_cadence_problems` opened: ``log_interval``,
the field the RL loop paces ``save_checkpoint`` by. It shares that gate's
:func:`~strands_robots.utils.step_cadence_error` domain because the two fields
are consumed identically - as the modulus of a periodic-checkpoint test - and
differ only in which spec names them, so a cadence refused for a supervised run
cannot be accepted for an RL one. It is scoped like
:func:`network_width_problems`: all three RL backends run that loop.
"""

from __future__ import annotations

import math
import numbers
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from strands_robots.tools._path_validation import validate_save_path
from strands_robots.utils import (
    finite_number_error,
    non_negative_count_error,
    positive_count_error,
    positive_finite_number_error,
    step_cadence_error,
    torch_device_error,
)

if TYPE_CHECKING:
    from strands_robots.training.base import TrainSpec

# ``extra`` keys are interpolated into argv as ``--{key}=...`` (lerobot/groot)
# or ``{key}=...`` (cosmos hydra). Allowlist the key FORMAT only: lowercase,
# dotted (lerobot ``dataset.episodes`` / cosmos ``model.x.y``), no leading dash,
# no ``=``, no whitespace or shell metacharacters. We deliberately do NOT try to
# enumerate every valid backend flag - that allowlist is impossible to keep
# current and would break the documented ``extra`` escape hatch.
_EXTRA_KEY_RE = re.compile(r"^[a-z][a-z0-9_.]*\Z")

# Scalars that are interpolated as the value of a single argv flag
# (e.g. ``--dataset.root={dataset_root}``). A leading ``-`` is the injection
# vector: ``base_model="--config_path=/etc/passwd"`` would otherwise parse as a
# separate flag. An interior ``=`` is harmless (the token stays single, no
# shell) and is legitimate for HF revision refs, so it is NOT rejected.
_FLAG_BOUND_FIELDS = ("dataset_root", "output_dir", "base_model", "embodiment", "dataset_repo_id")

# Path-like fields additionally get the audited filesystem check (null bytes,
# ``..`` traversal, protected system directories).
_PATH_FIELDS = ("dataset_root", "output_dir")


def validate_train_inputs(spec: TrainSpec) -> list[str]:
    """Return a list of input-safety problems for a :class:`TrainSpec`.

    An empty list means every agent-supplied value is safe to interpolate into
    a backend config / argv-parity helper. Pure and side-effect-free
    (read-only ``realpath`` only),
    so it is safe to call from :meth:`Trainer.validate`.
    """
    problems: list[str] = []

    # Path fields: reuse the audited validator used by the other write-path tools.
    for label in _PATH_FIELDS:
        val = getattr(spec, label, None)
        if val:
            try:
                validate_save_path(str(val), label=label)
            except ValueError as e:
                problems.append(str(e))

    # Flag-bound scalars must not smuggle an argv flag via a leading dash.
    for label in _FLAG_BOUND_FIELDS:
        val = getattr(spec, label, None)
        if isinstance(val, str) and val.startswith("-"):
            problems.append(f"{label} must not start with '-' (would parse as a stray flag)")

    # ``extra`` keys become backend-native flags - allowlist the key format.
    for key in spec.extra or {}:
        if not _EXTRA_KEY_RE.match(str(key)):
            problems.append(
                f"extra key {key!r} is not allowed "
                f"(must match {_EXTRA_KEY_RE.pattern}: lowercase, "
                f"no leading dash, no '=', no whitespace)"
            )

    return problems


def run_size_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return run-size problems for a :class:`TrainSpec`.

    ``steps`` and ``global_batch_size`` are the two factors of the amount of
    training a spec asks for, and each is consumed directly as a discrete
    count: ``steps`` bounds the backend's optimizer loop (lerobot iterates
    ``range(step, cfg.steps)``) and ``global_batch_size`` becomes a
    ``DataLoader`` batch size / a ``--global_batch_size`` flag. Only a positive
    integer can be honored, which is why both are checked against the one
    shared :func:`~strands_robots.utils.positive_count_error` domain rather
    than a local comparison: a bare ``value <= 0`` test admits every value that
    is not comparably non-positive, so ``True`` reads as a silent run of one
    step, a fractional or non-finite value reaches ``range()`` and raises
    there, and a string raises out of the comparison itself - inside a
    :meth:`Trainer.validate` that is documented to *return* problems.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        One problem per unusable field; empty when both are usable counts.
    """
    problems: list[str] = []
    for param, value in (("steps", spec.steps), ("global_batch_size", spec.global_batch_size)):
        error = positive_count_error(value, param, context)
        if error is not None:
            problems.append(error)
    return problems


def rl_run_size_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return run-size problems for a reinforcement-learning :class:`TrainSpec`.

    The RL peer of :func:`run_size_problems`. The supervised backends size a run
    from ``steps`` / ``global_batch_size``; the RL trainers size theirs from
    ``total_timesteps`` / ``rollout_steps``, the two caller-supplied factors of
    the one loop bound both of them derive::

        steps_per_iter = rollout_steps * num_envs
        num_iters = max(1, total_timesteps // steps_per_iter)
        for it in range(num_iters):  # collect, update

    Because that bound is *derived* rather than read straight off the spec, a
    bare ``value <= 0`` test on either factor is weaker here than the same test
    would be on a field consumed directly: the ``max(1, ...)`` clamp turns every
    value that survives the comparison but cannot divide into a **silent single
    iteration** instead of an error. Measured on both trainers over a 16-step
    run with ``rollout_steps=4``, before this gate existed:

    =========================  =========  ===========  ====================
    ``total_timesteps``        verdict    iterations   reported
    =========================  =========  ===========  ====================
    ``16`` (control)           success    4            ``latest_step=16``
    ``True``                   success    **1**        ``latest_step=4``
    ``0.5``                    success    **1**        ``latest_step=4``
    ``nan``                    success    **1**        ``latest_step=4``
    ``inf``                    success    **1**        ``latest_step=4``
    ``100.5``                  TypeError  --           from ``range()``
    ``"16"`` / ``None``        TypeError  --           from ``validate``
    =========================  =========  ===========  ====================

    ``inf`` lands in the silent column rather than the raising one because
    ``inf // 4`` is ``nan`` and ``max(1, nan)`` is ``1`` - ``nan`` compares false
    against everything. So four of the five values that pass a ``<= 0`` test
    report ``status="success"``, write a checkpoint, and announce
    ``"1 iterations x 4 steps complete"`` for a run the caller asked to be tens
    of thousands of steps long. The one value that does raise
    (``100.5 // 4 == 25.0``, a float ``range()`` bound) raises only after
    ``setup`` has built the environment, the networks, the optimizers and - for
    FastSAC - the replay buffer, which is the cost a read-only preflight exists
    to precede. A string or ``None`` raises out of the comparison itself, from a
    :meth:`Trainer.validate` documented to *return* problems.

    ``rollout_steps`` fails the same three ways through the other factor, and its
    silent case is worse than a short run because it changes the *shape* of the
    run rather than its length: ``True`` makes ``steps_per_iter`` one, so FastSAC
    ran 16 single-step iterations instead of 4 of 4 (reported as success), and
    PPO normalized advantages over a length-one batch - the standard deviation of
    one sample is ``nan`` - and failed inside torch's ``Normal`` constraint with
    a message naming neither the field nor the run.

    Only a positive integer can be honored, so both factors are checked against
    the one shared :func:`~strands_robots.utils.positive_count_error` domain: the
    same domain :func:`run_size_problems` uses for the supervised pair, and the
    domain whose own contract is that its values are consumed directly as
    ``range()`` bounds.

    ``num_envs``, the third factor of ``steps_per_iter``, is deliberately not
    here, and the reason is narrower than the whole field: which *counts* are
    usable differs between the backends - PPO parallelizes and accepts any
    positive count, while the MuJoCo-backed FastSAC is single-env and requires
    exactly ``1`` - so that half is not one shared rule and each backend keeps it.
    That the value must be a count *at all* is not per-backend, and it is this
    same domain, so both backends consult it before asking their own count rule.
    Excluding the whole field would have left the third factor of this very
    product outside the domain its two siblings are held to, which is what a bare
    per-backend comparison could not carry: it read ``nan`` and ``inf`` as usable
    and the ``max(1, ...)`` clamp turned them into a 1000-iteration and a
    one-iteration run under ``status="success"``, and ``True`` into a count of one
    from a value that reads as a flag.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        One problem per unusable factor; empty when both are usable counts.
    """
    problems: list[str] = []
    for param in ("total_timesteps", "rollout_steps"):
        error = positive_count_error(getattr(spec, param, 1), param, context)
        if error is not None:
            problems.append(error)
    return problems


def torch_device_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return device problems for a from-scratch RL :class:`RLTrainSpec`.

    All three RL backends resolve the device the same way in :meth:`setup`::

        self.device = torch.device(spec.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    so ``device`` is spent by ``torch.device`` itself, which judges nothing on
    this spec's behalf, and every network, buffer and rollout tensor the run
    allocates is placed on the result. ``device`` was the one caller-supplied
    knob on this spec with no domain - the sentence the sibling supervised
    preflight already uses about its own surface - while the counts, the
    interval coefficients, the loss weights, the learning rate, the seed, the
    launch topology and the network width beside it are all held to one.

    Measured on a 6-DoF SO-101 MuJoCo env with PPO, ``validate()`` returning
    ``[]`` - which :meth:`~strands_robots.training.base.Trainer.validate`
    documents as meaning the spec IS launchable - for each of these:

    * ``"gpu"`` (the ordinary mistake: the word the rest of the world uses)
      and ``"cuda:abc"`` raise ``RuntimeError`` out of ``setup`` from the
      ``torch.device`` line itself, after the preflight passed.
    * ``1`` is worse, and is why a non-``str`` is refused before torch is
      consulted: ``torch.device(1)`` constructs on ANY host, so the run reaches
      the first ``.to()`` and dies with ``CUDA error: invalid device ordinal``
      raised from a ``torch/nn/modules/module.py`` frame that names neither the
      field nor the run - and the same spec would train on a host with more
      GPUs. One spec, two answers, decided by the machine's inventory.

    The domain is :func:`~strands_robots.utils.torch_device_error`, the one owner
    the ``lerobot_train`` tool and :class:`~strands_robots.training.lerobot.LerobotTrainer`
    also consult, so a device the supervised trainer refuses is not accepted by
    the RL backend beside it.

    A falsy ``device`` is NOT refused: the ``spec.device or ...`` above documents
    it as "resolve the default", exactly as the supervised trainer's constructor
    does, so it never reaches ``torch.device`` as written and is not this gate's
    to judge. Availability is not graded either - ``"cuda"`` on a CPU-only box is
    a valid spec, because a queued run is written on one host and executed on
    another.

    Args:
        spec: The spec to preflight.
        context: Provider name, prefixed to the message.

    Returns:
        A single-element list when ``device`` cannot be honored; empty when it
        can, when it is unstated, or when torch is not importable.
    """
    device = getattr(spec, "device", None)
    if not device:
        return []
    problem = torch_device_error(device, "device", context)
    return [] if problem is None else [problem]


def rl_replay_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return replay-loop problems for an off-policy :class:`RLTrainSpec`.

    The three caller-supplied counts of an off-policy (SAC / TD3) run's replay
    loop:

    * ``buffer_size`` - the replay buffer's capacity, a tensor dimension built
      in :meth:`~strands_robots.training.rl.fast_sac.FastSacTrainer.setup`.
    * ``batch_size`` - the transitions sampled per gradient step, passed to
      ``ReplayBuffer.sample``.
    * ``gradient_steps`` - the off-policy updates run per iteration, a
      ``range()`` bound.

    Each is consumed directly as a count - a capacity, a sample size, a
    ``range()`` bound - so the same strict-``int``
    :func:`~strands_robots.utils.positive_count_error` domain applies that
    :func:`run_size_problems` uses, and for the same reason: a value that is not
    a positive ``int`` cannot be a tensor dimension or a ``range()`` argument and
    raises ``TypeError`` there rather than being coerced.

    A local ``value <= 0`` test is weaker than that domain, and both of its
    failure modes were measured on the MuJoCo reach env before this gate existed
    (an otherwise-valid run, one field mutated):

    =====================  ==========  =========================================
    value                  verdict     what happened
    =====================  ==========  =========================================
    ``buffer_size=True``   success     a one-slot buffer that never reaches
                                       ``learning_starts``, so **zero** gradient
                                       updates ran, yet the run reported success
                                       and "10 iterations x 4 steps complete"
    ``buffer_size=0.5``    IndexError  ``int(0.5) == 0``: a zero-capacity buffer,
                                       raised from ``ReplayBuffer.add`` after setup
    ``batch_size=0.5``     TypeError   raised from ``torch.randint`` in
                                       ``ReplayBuffer.sample`` after setup
    ``batch_size=True``    TypeError   the same, from a batch of ``True``
    ``gradient_steps=0.5`` TypeError   raised from ``range()`` in the update loop
    ``"256"`` / ``None``   TypeError   raised from the ``<= 0`` comparison itself,
                                       out of a ``validate`` documented to return
    =====================  ==========  =========================================

    So a ``bool`` reads as a silent degenerate size - the ``buffer_size`` case
    runs a whole training loop that learns nothing and reports success - a
    fraction or a non-finite value passes the comparison and raises deep inside
    the update loop after the environment, the networks, the optimizers and the
    replay buffer have been built (the cost a read-only preflight exists to
    precede), and a string or ``None`` raises out of the comparison itself, from
    a :meth:`~strands_robots.training.base.Trainer.validate` documented to
    *return* its problems.

    Only the off-policy backends (FastSAC and FastTD3) read these three fields;
    PPO sizes its minibatches from ``num_mini_batches`` and never reads them, so
    a backend that ignores them must not report on them - which is why this is a
    gate scoped to the field rather than part of :func:`validate_train_inputs`.

    ``learning_starts`` stays in the backend's own ``validate``: it is one side
    of a relation (``>= batch_size``) rather than a bare count, so it does not
    share this domain. ``tau`` does not either - it is a coefficient in
    ``(0, 1]`` rather than a count - but it no longer stays local for that
    reason: it has its own gate on its own interval, in
    :func:`polyak_coefficient_problems`.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`.

    Returns:
        One problem per unusable count; empty when all three are usable counts.
    """
    problems: list[str] = []
    for param in ("buffer_size", "batch_size", "gradient_steps"):
        error = positive_count_error(getattr(spec, param, 1), param, context)
        if error is not None:
            problems.append(error)
    return problems


def launch_topology_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return launch-topology problems for a :class:`TrainSpec`.

    ``num_gpus`` and ``num_nodes`` are the two process counts a distributed run
    is sized from. Each is consumed as a discrete count in three places: a
    ``spec.num_gpus > 1`` / ``spec.num_nodes > 1`` test that selects between the
    single-process and the multi-process launch path, a ``nproc_per_node`` /
    ``nnodes`` argument to torch's ``elastic_launch``, and a
    ``--nproc_per_node=`` / ``--nnodes=`` / ``--num_gpus=`` argv token. Only a
    positive integer can be honored, and each of the three ways a bad value
    fails is silent or late:

    * ``0``, a negative, ``nan`` and ``True`` all read as *not* greater than one
      -- ``nan`` compares false against everything -- so the selector routes
      them to the single-process path and the run proceeds on one process under
      a successful result. The topology the caller asked for is simply not the
      one that ran, and for ``num_nodes`` that also slips past the multi-node
      refusal the backends raise for an unsupported topology.
    * ``2.7`` and ``inf`` *are* greater than one, so they select the
      multi-process path and reach ``elastic_launch`` as the worker count.
      ``LaunchConfig`` accepts both without complaint, so nothing downstream
      rejects them either.
    * A string, ``None`` or a list raises ``TypeError`` out of the comparison
      itself -- from inside a :meth:`Trainer.validate` that is documented to
      *return* problems.

    Both are therefore checked against the one shared
    :func:`~strands_robots.utils.positive_count_error` domain, the same one
    :func:`run_size_problems` uses, rather than by a local comparison.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        One problem per unusable field; empty when both are usable counts.
    """
    problems: list[str] = []
    for param, value in (("num_gpus", spec.num_gpus), ("num_nodes", spec.num_nodes)):
        error = positive_count_error(value, param, context)
        if error is not None:
            problems.append(error)
    return problems


def learning_rate_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return optimizer learning-rate problems for a :class:`TrainSpec`.

    ``learning_rate`` is the one numeric on a :class:`TrainSpec` that decides
    whether a run *learns* rather than how much work it does, and every backend
    reads it: the supervised three assign it to their config's optimizer field
    (LeRobot ``policy.optimizer_lr``, GR00T ``FinetuneConfig.learning_rate``,
    Cosmos ``optimizer.lr``) and the RL trainers pass it directly to
    ``torch.optim.Adam(..., lr=...)``.

    Only a positive finite value can be honored, and the two ends of the domain
    fail *silently* rather than loudly, which is why this is a preflight rather
    than something the backend can be left to notice:

    * ``0`` (and ``False``, which is ``0`` to every consumer) runs the full
      ``steps`` x ``global_batch_size`` of work and updates no weight, so the
      run reports success and writes a checkpoint identical to its
      initialisation. That is the pathology :func:`run_size_problems` exists to
      prevent, reached by a different route and at full cost.
    * ``inf`` diverges on the first optimizer step, so the checkpoint is all
      ``NaN`` -- again under a successful result.
    * ``True`` is a silent learning rate of ``1.0``, four orders of magnitude
      above a typical fine-tuning preset.

    A negative or ``nan`` value *is* refused by ``torch.optim.Adam``
    (``ValueError: Invalid learning rate``), but only once the dataset and model
    are already loaded -- after the point :meth:`Trainer.validate` documents
    itself as running before ("it powers a ``plan`` advisor that runs *before*
    anything expensive starts").

    ``None`` is the documented sentinel for "use the backend's own default" and
    is therefore not a problem. It is checked against the shared
    :func:`~strands_robots.utils.positive_finite_number_error` domain rather
    than a local comparison because a bare ``value <= 0`` test admits ``nan``
    (every comparison against it is ``False``), admits a ``bool``, and raises
    out of the comparison itself for a non-numeric value -- inside a method
    documented to *return* problems.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single problem when ``learning_rate`` is supplied and unusable;
        empty when it is usable or left at ``None``.
    """
    if spec.learning_rate is None:
        return []
    error = positive_finite_number_error(spec.learning_rate, "learning_rate", context)
    return [error] if error is not None else []


def seed_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return reproducibility-seed problems for a :class:`TrainSpec`.

    ``seed`` is the field a caller sets to make a run reproducible, and the four
    backends that read it apply it through appliers that disagree about what a
    single value means:

    * The RL trainers hand it to ``torch.manual_seed``, which reduces it modulo
      ``2**64``. A negative seed is therefore *silently a different seed*:
      ``manual_seed(-1)`` and ``manual_seed(2**64 - 1)`` draw the identical
      stream, so two seeds the caller means to be distinct collapse onto one and
      the run is reproducible under a number nobody asked for. ``True`` is
      likewise a silent seed of ``1`` and ``2.7`` a silent seed of ``2``.
    * LeRobot assigns it to ``cfg.seed``, which reaches lerobot's ``set_seed``:
      ``random.seed`` first, then ``numpy.random.seed``. NumPy is far narrower
      than torch - it refuses a negative value and a float or string outright -
      but only *after* ``random.seed`` has run, so a refused seed leaves the
      process RNG reseeded by a call that failed.
    * Cosmos interpolates it into a ``trainer.seed=`` Hydra override, and
      LeRobot's argv-parity path into a ``--seed=`` token. There every value
      renders - ``nan``, ``2.7``, ``[7]`` - and fails, if at all, inside the
      run after the dataset and model are already loaded.

    So the same ``seed=-1`` is silently rewritten by one backend and refused with
    a bare third-party message by the next. Only a non-negative integer can be
    honored by all of them, so it is checked against the one shared
    :func:`~strands_robots.utils.non_negative_count_error` domain: the same
    non-negative-integer rule, whose ``0`` is first-class here too (seed ``0`` is
    a seed), and which rejects ``bool`` explicitly because a bare ``value < 0``
    test lets ``True`` through as a silent seed of one.

    ``None`` is the documented sentinel for "use the backend's own default"
    (LeRobot's is ``1000``) and is therefore not a problem, exactly as it is not
    one for :func:`learning_rate_problems`.

    One boundary this does not decide: the appliers also disagree about the
    upper end - torch accepts up to ``2**64 - 1`` while NumPy's legacy seeder
    stops at ``2**32 - 1`` - so a per-backend ceiling is a separate question from
    the floor and type checked here.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single problem when ``seed`` is supplied and unusable; empty otherwise.
    """
    if spec.seed is None:
        return []
    error = non_negative_count_error(spec.seed, "seed", context)
    return [] if error is None else [error]


def checkpoint_cadence_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return checkpoint-cadence problems for a :class:`TrainSpec`.

    ``save_freq`` is how often a run writes a checkpoint, and the four backends
    that read it deliver the value to a destination that requires a genuine
    ``int`` - by four different routes, each of which mis-handles every other
    spelling differently and none of which reports it:

    * LeRobot in-process assigns it to ``cfg.save_freq``, which reaches
      lerobot's own ``should_save_checkpoint(step, save_freq, total_steps)`` -
      ``(save_freq > 0 and step % save_freq == 0) or step == total_steps``. No
      parser stands between the spec and that expression, so ``True`` is a
      modulus of one and writes a **full checkpoint every single step** (9999
      of them in a 10 000-step run that asked for 9), a fractional or
      non-finite cadence never satisfies ``step % cadence == 0`` for an
      integral step and so silently becomes the *disabled* mode, and a ``str``
      raises ``TypeError`` out of the comparison - inside the training loop,
      after the dataset and the model are loaded.
    * LeRobot's argv-parity path renders ``--save_freq={value}``, which lerobot
      decodes into the same ``int`` field with draccus: ``True``, ``2.7``,
      ``5000.0``, ``nan`` and ``inf`` all raise ``DecodingError`` there.
    * GR00T renders ``--save_steps={value}`` and Cosmos a
      ``checkpoint.save_iter={value}`` Hydra override, where an unusable value
      fails - if at all - inside the launched run.
    * SageMaker forwards it as a hyperparameter string via ``json.dumps``, so
      ``nan`` and ``inf`` travel as ``NaN`` / ``Infinity``, which only a
      permissive JSON decoder accepts.

    The two LeRobot routes disagree about the *same* spec, which is what makes a
    shared gate the only fix that holds: a ``"5000"`` renders the perfectly
    decodable token ``--save_freq=5000`` and raises ``TypeError`` in-process,
    while ``2.7`` is refused on the argv path and silently disables periodic
    saving in-process. One spec has to mean one run whichever path the backend
    takes - the rule :attr:`TrainSpec.extra` already states for its own values -
    so the cadence is checked against the one shared
    :func:`~strands_robots.utils.step_cadence_error` domain, which the
    ``lerobot_train`` tool holds the same field to when it builds the argv
    itself.

    Only the *type* is graded. A non-positive cadence is a documented
    capability - lerobot's ``should_save_checkpoint`` reads "a non-positive
    ``save_freq`` disables periodic saving (only the final checkpoint is
    written)" - and the ``eval_steps`` fallback in the LeRobot backend
    (``spec.save_freq if spec.save_freq > 0 else spec.steps``) is written for
    exactly that case, so ``0`` and a negative are first-class here. Whether a
    given backend's own trainer accepts the disabled mode is a per-backend
    question, like the per-backend seed ceiling :func:`seed_problems` leaves
    open, and separate from the type checked here.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single problem when ``save_freq`` is not a whole number of steps;
        empty otherwise.
    """
    error = step_cadence_error(spec.save_freq, "save_freq", context)
    return [] if error is None else [error]


def validation_episodes_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return held-out-validation-set problems for a :class:`TrainSpec`.

    ``val_episodes`` is the count of episodes a caller reserves from the tail of
    the dataset to validate on. It is not read straight into a loop bound like
    :func:`run_size_problems`' fields: the LeRobot backend converts it into
    lerobot's ``dataset.eval_split`` FRACTION via
    :func:`~strands_robots.utils.validation_split_fraction`, and lerobot then
    holds out ``ceil(episodes_in_task * eval_split)``. That conversion is what
    makes a local comparison unsafe in both directions at once:

    * A non-positive value is *silently dropped*. The fraction is only computed
      for a count in ``(0, total)``, so ``val_episodes=0`` (or a negative)
      produces no ``eval_split`` and no ``eval_steps`` at all: the run trains on
      the whole dataset, records no validation loss, and reports no problem. The
      caller asked for a validation set and got a run without one.
    * A value that merely *compares* as positive is silently rewritten, because
      the fraction is real-valued and lerobot takes its ceiling: ``True``
      reserves 1 episode and ``2.7`` reserves 3 - a whole number the caller never
      named. ``0.5`` is the sharpest of these: it clears the ``0 < count <
      total`` test, so it emits ``eval_split=0.0`` - a held-out set of zero
      episodes - *together with* an ``eval_steps`` cadence, asking lerobot to
      validate periodically on nothing.
    * A non-numeric value raises out of the comparison itself, from a
      :meth:`~strands_robots.training.base.Trainer.validate` documented to
      *return* problems.

    Only a positive integer strictly below the dataset's episode count can be
    honored, so the type and floor are checked here against the same shared
    :func:`~strands_robots.utils.positive_count_error` domain that
    :func:`run_size_problems` uses. The upper bound is dataset-dependent (it needs
    ``total_episodes`` from ``meta/info.json``) and stays with the backend that
    reads the metadata, which also owns the per-task-fraction refusal in
    :func:`~strands_robots.utils.validation_split_error`.

    ``None`` is the documented sentinel for "train on every episode, no held-out
    set" and is therefore not a problem, exactly as it is not one for
    :func:`seed_problems` or :func:`learning_rate_problems`.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single problem when ``val_episodes`` is supplied and unusable as a
        count; empty otherwise.
    """
    if spec.val_episodes is None:
        return []
    error = positive_count_error(spec.val_episodes, "val_episodes", context)
    return [] if error is None else [error]


def lora_hyperparameter_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return LoRA adapter-hyperparameter problems for a :class:`TrainSpec`.

    ``lora_r`` and ``lora_alpha`` are the rank and the scaling numerator of a
    LoRA fine-tune: peft builds a rank-``r`` adapter and applies its update
    scaled by ``lora_alpha / r``. The two fields fail in opposite ways, and only
    one of them fails loudly:

    * ``lora_r`` is refused by peft, but only from inside
      ``get_peft_model`` - after the base model has been downloaded and loaded.
      A non-positive rank raises ``ValueError: `r` should be a positive integer
      value``, and a ``bool``/float/string one raises out of torch's tensor
      allocation with a message naming neither the field nor the run.
    * ``lora_alpha`` is **accepted for every unusable value**. It is only ever a
      numerator, so nothing downstream compares it: ``lora_alpha=0`` builds the
      adapter, reports its trainable parameters and trains them with a scaling
      of ``0.0``, so the adapter provably cannot change the model's output - the
      fine-tune runs to completion, writes checkpoints, and has learned nothing
      that can ever be applied. A negative value applies the negation of what
      the adapter learned, and ``True`` is a silent alpha of one.

    The two paths that carry these fields also disagree about a fractional
    value. In-process, peft accepts ``lora_alpha=2.7`` and scales by
    ``2.7 / r``; on the argv-parity path the same value reaches lerobot's
    ``PeftConfig``, whose ``r`` and ``lora_alpha`` are declared ``int``, and
    draccus refuses it. So one spelling of one run honors a value the other
    rejects.

    A positive integer is therefore the only thing both paths can honor, and it
    is checked against the same shared
    :func:`~strands_robots.utils.positive_count_error` domain
    :func:`run_size_problems` uses - the domain that also rejects ``bool``,
    which a bare ``value < 1`` test would let through as a silent rank or alpha
    of one.

    ``None`` is the documented sentinel for "omit the option and keep peft's own
    default" and is therefore not a problem, exactly as it is not one for
    :func:`seed_problems` or :func:`validation_episodes_problems`.

    Both fields are read only on the ``method == "lora"`` branch, so a spec that
    carries them under another strategy reports nothing: the fields are inert
    there, and refusing a value the run never reads would be a false rejection -
    the same reason this is a separate gate from :func:`learning_rate_problems`
    rather than part of it.

    ``lora_target_modules`` is out of scope: it is a module-name string rather
    than a count, and :func:`validate_train_inputs` already owns what may be
    interpolated into a config field or an argv token.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        One problem per supplied-and-unusable adapter hyperparameter; empty when
        the spec does not request LoRA or both values are usable.
    """
    if spec.method != "lora":
        return []
    problems: list[str] = []
    for param, value in (("lora_r", spec.lora_r), ("lora_alpha", spec.lora_alpha)):
        if value is None:
            continue
        error = positive_count_error(value, param, context)
        if error is not None:
            problems.append(error)
    return problems


def _closed_unit_interval_error(value: Any, param: str, context: str) -> str | None:
    """Error text when *value* is not a real number in the closed range [0, 1].

    Numeric-ness, ``bool`` rejection and finiteness are delegated to the shared
    :func:`~strands_robots.utils.finite_number_error` domain, so those refusals
    read identically to every other numeric field's. The only thing decided here
    is the interval, which no shared domain expresses: ``utils`` carries
    open-ended families (positive, non-negative) rather than a bounded one.

    Both endpoints are inside the domain and neither is a degenerate spelling of
    "disabled", which is why the interval is closed rather than half-open.

    Args:
        value: The caller-supplied value.
        param: Field name for the message.
        context: Caller label the message is prefixed with.

    Returns:
        The error text, or None when *value* is a real number in [0, 1].
    """
    error = finite_number_error(value, param, context)
    if error is not None:
        return error
    if not 0.0 <= float(value) <= 1.0:
        return f"{context}: {param} must be in [0, 1], got {value!r}."
    return None


def discount_factor_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return discount-factor problems for an RL :class:`TrainSpec`.

    ``gamma`` weights every future reward in the return the algorithm optimizes,
    and it is the one coefficient both RL backends read: PPO discounts the GAE
    recursion with it (twice - the single-env and vectorized rollout paths), and
    FastSAC discounts its target-Q bootstrap. A discounted return is a geometric
    series, so the domain is not a matter of taste:

    * ``gamma > 1`` makes that series **diverge**. The advantages grow without
      bound in the rollout horizon rather than being merely large - over a
      24-step rollout of unit rewards, ``gamma=1.5`` inflates the largest
      advantage from 12.9 to 1.2e4, and ``gamma=5`` to 4.6e15. Nothing refuses
      it: the run trains on those advantages, reports success, and writes a
      checkpoint.
    * ``gamma < 0`` alternates the sign of each successive reward, so the trace
      no longer accumulates future return at all - the same rollout collapses
      the largest advantage to the immediate reward, 1.0.
    * ``nan``/``inf`` make every advantage non-finite, which surfaces only once
      the update samples the action distribution: ``ValueError: Expected
      parameter loc ... of distribution Normal ... to satisfy the constraint
      Real()``, a torch message that names neither the field nor the run, raised
      after the env, the networks and a full rollout have been built. That is
      exactly the "deep stack trace" a read-only preflight exists to replace.
    * ``True`` is a silent ``gamma`` of one, because a bare comparison against
      the interval bounds accepts it - ``bool`` is an ``int`` subclass.

    Both endpoints are legitimate and standard: ``gamma=1`` is the undiscounted
    episodic return, ``gamma=0`` a myopic agent that optimizes the immediate
    reward only. So the domain is the *closed* interval [0, 1], checked through
    :func:`_closed_unit_interval_error`.

    The off-policy backends bound their own interval coefficient this way
    (``tau`` must be in ``(0, 1]``), which is the shape this gate generalizes: an
    interval coefficient is checked against its interval rather than left to the
    arithmetic that consumes it. That precedent is now a shared gate too rather
    than a bare comparison inside each backend - see
    :func:`polyak_coefficient_problems`, which is half-open where this one is
    closed because zero freezes a target network instead of reading as a
    myopic agent.

    ``lam``, the other factor of the trace-decay product, has its own gate for
    that same scoping reason - see :func:`gae_lambda_problems`,
    ``num_learning_epochs`` likewise in :func:`optimization_epochs_problems`,
    and ``max_grad_norm`` in :func:`gradient_clip_problems`.
    The two *loss weights* named there - ``entropy_coef`` and ``value_loss_coef`` -
    now have their own gate too, in :func:`loss_weight_problems`, on the domain
    rather than on the endpoint this docstring called undecided: their zero
    readings are still unsettled and still accepted, but a non-finite weight has
    no reading at all. ``clip_param`` now has its own gate too, in
    :func:`clip_range_problems`: this docstring left it out because it needed the
    endpoint decision :func:`gradient_clip_problems` records for the sibling clip
    bound, and that decision is now shared rather than duplicated - both read
    :func:`_clip_bound_error`. ``init_noise_std`` remains out of scope in all
    six, for a measured reason rather than by omission: every non-finite value is
    refused by ``torch``, which rejects a ``Normal`` of non-positive or
    non-finite scale.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single-element list when ``gamma`` cannot be honored; empty otherwise.
    """
    error = _closed_unit_interval_error(getattr(spec, "gamma", 0.0), "gamma", context)
    return [error] if error is not None else []


def gae_lambda_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return GAE-lambda problems for an on-policy RL :class:`TrainSpec`.

    ``lam`` is the second factor of the advantage trace's decay. The GAE
    recursion carries it forward as ``last_adv = delta + gamma * lam *
    (1 - done) * last_adv``, so the trace decays by the **product**
    ``gamma * lam`` and :func:`discount_factor_problems` bounding ``gamma``
    alone does not bound it: with a ``gamma`` of ``0.99`` - comfortably inside
    that gate's closed interval - a ``lam`` of ``1.5`` gives a decay factor of
    ``1.485`` and the same divergence, measured on this backend's own
    ``compute_gae`` over a rollout of unit rewards:

    ======  =========  =========  =========  =========
    ``lam``  ``T=12``   ``T=24``   ``T=48``   ``T=96``
    ======  =========  =========  =========  =========
    0.95      8.8        13.0       15.9       16.8
    1.5     235.1      2.7e+04    3.6e+08    6.3e+16
    1e6       inf        inf        inf        inf
    ======  =========  =========  =========  =========

    The largest advantage grows without bound in the rollout horizon rather than
    being merely large, and nothing refuses it: the run trains on those
    advantages, reports success, and writes a checkpoint.

    The remaining values outside the interval fail in three further ways:

    * ``lam < -1 / gamma`` diverges as well, because the trace decays by
      ``|gamma * lam|`` - ``lam=-2`` reaches ``1.0e+28`` by ``T=96`` - while a
      ``lam`` merely below zero (``-0.5``) collapses the trace to the immediate
      reward, so the estimator stops accumulating future advantage at all.
    * ``nan``/``inf`` make every advantage non-finite, which surfaces only once
      the update samples the action distribution - a torch constraint error
      naming neither the field nor the run, after the env, the networks and a
      full rollout have been built.
    * ``True`` is a silent ``lam`` of one, because a bare comparison against the
      interval bounds accepts it: ``bool`` is an ``int`` subclass. That is a
      different estimator from the one the caller asked for - Monte-Carlo return
      rather than a bootstrapped trace.

    Both endpoints are legitimate and standard, which is why the domain is the
    *closed* interval [0, 1]: ``lam=1`` is the Monte-Carlo advantage (no
    bootstrapping, 61.9 at ``T=96`` above) and ``lam=0`` is TD(0), the
    one-step advantage.

    Unlike ``gamma`` this is read by the on-policy backend only, so it is scoped
    like :func:`run_size_problems`: FastSAC has no advantage trace and must not
    report on a field it never reads.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single-element list when ``lam`` cannot be honored; empty otherwise.
    """
    error = _closed_unit_interval_error(getattr(spec, "lam", 0.0), "lam", context)
    return [error] if error is not None else []


def optimization_epochs_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return optimization-epoch problems for an on-policy RL :class:`TrainSpec`.

    ``num_learning_epochs`` is the number of passes the update makes over each
    rollout batch, and it is consumed as a bare loop bound around the whole
    optimizer step - ``for _ in range(spec.num_learning_epochs)`` encloses every
    ``optimizer.step()`` in the PPO update. So the field does not merely scale
    how much optimization happens; a non-positive value removes *all* of it, and
    nothing downstream notices:

    ==========================  =========  ==============  ==============
    ``num_learning_epochs``     verdict    optimizer       reported
                                           steps taken     losses
    ==========================  =========  ==============  ==============
    5 (the shipped default)     honored    24              real values
    0                           accepted   **0**           all ``0.0``
    -3                          accepted   **0**           all ``0.0``
    ==========================  =========  ==============  ==============

    Measured on this backend over a 60-step run: ``0`` and ``-3`` both report
    ``status="success"``, take **zero** gradient steps, and write a checkpoint
    whose parameters are bit-identical to each other - the untrained
    initialisation. The losses read ``0.0`` rather than blank because the update
    averages its accumulators through ``max(1, n_updates)``, so an epoch count
    that ran no minibatch reports plausible metrics for a run that learned
    nothing. A caller therefore gets a deployable-looking checkpoint, a
    successful result and a metrics dict, with no signal anywhere that the
    optimizer never ran.

    The remaining values outside the domain fail in two further ways:

    * ``True`` is a silent single epoch, because ``range(True)`` is
      ``range(1)`` - the same run takes 12 optimizer steps instead of 24. That
      is a different amount of optimization from the one requested, reported as
      success.
    * ``2.7``/``nan``/``inf``/``"5"``/``None`` raise a bare ``TypeError:
      'float' object cannot be interpreted as an integer`` out of ``range()``,
      naming neither the field nor the run, and only after the environment, the
      networks and a full rollout have been built - exactly the deep stack trace
      a read-only preflight exists to replace. No checkpoint is written at all.

    The domain is therefore a positive integer, checked by
    :func:`~strands_robots.utils.positive_count_error`, which is already the
    domain this repository uses for a value consumed as a ``range()`` bound: an
    integral float is not usable there (``range(2.0)`` raises) and ``bool`` must
    be rejected rather than silently read as one.

    Unlike ``gamma`` this is read by the on-policy backend only - FastSAC
    optimizes per gradient step from a replay buffer and has no epoch loop over a
    rollout batch - so it is scoped like :func:`gae_lambda_problems`: per
    :class:`TrainSpec` a backend ignores the fields it does not support, so
    reporting on one it never reads would be a false rejection.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single-element list when ``num_learning_epochs`` cannot be honored;
        empty otherwise.
    """
    error = positive_count_error(getattr(spec, "num_learning_epochs", 1), "num_learning_epochs", context)
    return [error] if error is not None else []


def temperature_learning_rate_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return entropy-temperature learning-rate problems for a :class:`TrainSpec`.

    ``alpha_lr`` is documented on :class:`~strands_robots.training.rl.RLTrainSpec`
    as the "Learning rate for the temperature optimizer (SAC)", and FastSAC hands
    it to the same constructor as the already-guarded ``learning_rate`` two lines
    above it::

        self.actor_optimizer = torch.optim.Adam(actor_params, lr=spec.learning_rate)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=spec.learning_rate)
        ...
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=spec.alpha_lr)

    So every failure mode :func:`learning_rate_problems` documents applies to it
    unchanged, and each was measured on a 40-timestep FastSAC run whose
    temperature starts at ``init_alpha=1.0``:

    * ``0`` (and ``0.0``, and ``False``, which is ``0`` to the optimizer) builds
      the optimizer and moves ``log_alpha`` by nothing, so the temperature stays
      at ``init_alpha`` for the whole run. ``autotune_alpha=True`` then behaves
      exactly like ``autotune_alpha=False`` while reporting the automatic
      temperature it was asked for - the "runs the full work and updates no
      weight" pathology, on the one parameter whose job is to adapt.
    * ``inf`` also builds, and the first step sends ``log_alpha`` to an infinity.
      Because ``alpha`` multiplies the log-probability in the *actor* loss, the
      damage is not confined to the temperature: the run finished with
      ``status="success"`` and a checkpoint whose largest parameter magnitude was
      ``inf``.
    * ``True`` is a silent learning rate of ``1.0``, over three thousand times
      the ``3e-4`` default, and moved the temperature 407x further in the same
      40 steps.
    * A negative value and ``nan`` *are* refused, by ``torch.optim.Adam``
      (``ValueError: Invalid learning rate``), and a ``str`` / ``None`` /
      ``list`` raises a bare ``TypeError: '<=' not supported between instances of
      'float' and 'str'`` naming neither the field nor the value. Both arrive in
      :meth:`~strands_robots.training.rl.base_algo.BaseRLAlgo.setup`, after the
      env and both networks are built - past the point :meth:`Trainer.validate`
      documents itself as running before.

    Unlike ``learning_rate`` there is no ``None`` sentinel to exempt: the field is
    annotated ``float`` with a concrete ``3e-4`` default, so ``None`` is a value
    the temperature optimizer cannot take rather than a request for a default.

    The value is read only when ``autotune_alpha`` is set, which is the only
    branch that constructs a temperature optimizer, so a spec that tunes no
    temperature is not reported on - refusing a field that call path never reads
    would be a false rejection. A plain :class:`TrainSpec` has no
    ``autotune_alpha`` at all and is likewise silent.

    Only the SAC backend tunes a temperature, so this is scoped like
    :func:`gae_lambda_problems` rather than :func:`learning_rate_problems`: a
    backend that does not read the field MUST NOT call this.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single-element list when a tuned temperature's ``alpha_lr`` cannot be
        honored; empty when it is usable or no temperature is being tuned.
    """
    if not getattr(spec, "autotune_alpha", False):
        return []
    error = positive_finite_number_error(getattr(spec, "alpha_lr", 3e-4), "alpha_lr", context)
    return [error] if error is not None else []


def initial_temperature_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return entropy-temperature starting-value problems for a :class:`TrainSpec`.

    ``init_alpha`` is documented on :class:`~strands_robots.training.rl.RLTrainSpec`
    as the "Initial entropy temperature (SAC)". FastSAC does not hold it directly:
    it stores the temperature's *logarithm*, so the field reaches ``torch.log``
    on both temperature branches::

        self.log_alpha = torch.tensor(float(torch.log(torch.tensor(spec.init_alpha))), ...)

    and ``alpha`` - which scales the entropy term in the critic's TD target and
    in the actor loss - is ``log_alpha.exp()`` from there on. Only a positive
    finite value has a finite logarithm, so this is the same domain as
    :func:`temperature_learning_rate_problems` applies to the temperature's
    learning rate. That gate's own reasoning already names this field: an
    ``alpha_lr`` of ``0`` is refused because "the temperature stays at
    ``init_alpha`` for the whole run", which is only a usable statement if
    ``init_alpha`` is itself usable.

    Each failure mode below was measured on a 40-timestep FastSAC run:

    * ``0`` (and ``0.0``, and ``False``, which is ``0`` to ``torch.log``) makes
      ``log(0) == -inf``, so ``alpha`` is exactly ``0`` and the entropy term is
      gone from both losses - the run is no longer the maximum-entropy algorithm
      that was asked for. Automatic tuning cannot recover it: ``log_alpha``
      remained ``-inf`` after further gradient steps, because no finite update
      moves an infinity. The run reported ``status="success"`` and saved a
      checkpoint holding ``log_alpha == -inf``, so the unusable temperature
      outlives the run that produced it.
    * ``True`` is a silent temperature of exactly ``1.0`` - the default - rather
      than a value the caller chose.
    * A negative value and ``nan`` make the logarithm ``nan``, and ``inf`` makes
      it ``inf``; either way the temperature poisons the actor loss, and the
      first update raises ``ValueError`` from ``torch.distributions.Normal``
      reporting a tensor of ``nan`` policy means. That message names the
      distribution's ``loc`` parameter, not the field or the value that produced
      it, and it arrives inside ``train`` - after the env and both networks are
      built, past the point :meth:`Trainer.validate` documents itself as running
      before.

    Unlike :func:`temperature_learning_rate_problems` this is **not** scoped to
    ``autotune_alpha``: that gate guards an optimizer only the tuning branch
    constructs, whereas ``init_alpha`` is read on both branches. With tuning off
    it is the temperature for the whole run and nothing can move it afterwards,
    so a spec that tunes nothing needs the check more, not less.

    There is no ``None`` sentinel to exempt: the field is annotated ``float``
    with a concrete ``1.0`` default, so ``None`` is a value ``torch.log`` cannot
    take rather than a request for a default.

    Only the SAC backend holds an entropy temperature, so this is scoped
    like :func:`gae_lambda_problems` rather than :func:`learning_rate_problems`:
    a backend that does not read the field MUST NOT call this. A plain
    :class:`TrainSpec` has no ``init_alpha`` at all and is likewise silent.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single-element list when ``init_alpha`` has no finite logarithm; empty
        when it is usable.
    """
    error = positive_finite_number_error(getattr(spec, "init_alpha", 1.0), "init_alpha", context)
    return [error] if error is not None else []


def target_entropy_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return target-entropy problems for a :class:`TrainSpec`.

    ``target_entropy`` is the third caller-supplied field of FastSAC's
    temperature block, and the only one nothing judged. It is the constant the
    temperature is optimized *against* - the one term of the temperature loss a
    caller supplies::

        alpha_loss = -(self.log_alpha * (logp + self.target_entropy).detach()).mean()

    and it reaches that expression through an unconditional coercion in
    :meth:`~strands_robots.training.rl.fast_sac.FastSacTrainer.setup`::

        self.target_entropy = (
            float(spec.target_entropy) if spec.target_entropy is not None else -float(self.env.num_actions)
        )

    Its two neighbours in the same block - the temperature's starting value in
    :func:`initial_temperature_problems` and the rate that moves it in
    :func:`temperature_learning_rate_problems` - are both held to a *positive*
    finite domain, and this field was explicitly left out of both: it is signed
    by construction, defaulting to ``-num_actions``, so it needs a domain neither
    of them can express. That domain is
    :func:`~strands_robots.utils.finite_number_error`, the signed counterpart the
    two loss weights of :func:`loss_weight_problems` already read - the same
    "finite real, either sign" shape, for the same reason: every reading of this
    field is a signed entropy in nats, so no endpoint is decidable, but a value
    that is not a finite real has no reading at all.

    Each row below was measured on a 40-timestep FastSAC run, with ``validate()``
    returning ``[]`` for every one of them:

    ============================  ===========================================
    ``target_entropy``            Outcome
    ============================  ===========================================
    ``None`` (the default)        trains; ``log_alpha == -0.001801646314``
    ``-6.0`` (``-num_actions``)   trains; identical to the default
    ``True``                      **trains against a target entropy of +1.0**
    ``'-6'``                      trains; ``float()`` coerces it, so the run
                                  is identical to ``-6.0``
    ``nan`` / ``inf`` / ``-inf``  raises mid-update, from inside ``torch``
    ``[-6.0]`` / ``{}``           raises in ``setup``, from ``float()``
    ============================  ===========================================

    Both failing shapes are what make this a gate rather than a lint:

    * **A boolean is a silent sign flip.** ``bool`` is an ``int`` subclass, so
      ``target_entropy=True`` is a target entropy of ``+1.0`` where every
      documented reading of the field is negative - the field's own default is
      ``-num_actions``, which is ``-6.0`` for this env. The run reported
      ``success`` and checkpointed ``log_alpha == -0.0018031001091003418``
      against the honored run's ``-0.001801646314561367``, so it demonstrably
      drove the temperature somewhere else rather than harmlessly.
    * **A non-finite value poisons the temperature; a non-real one raises in
      ``setup``.** ``nan`` makes ``alpha_loss`` ``nan``, the temperature
      optimizer writes ``nan`` into ``log_alpha``, and ``alpha`` scales the
      entropy term of *both* the critic's TD target and the actor loss - so the
      next rollout samples the action distribution from ``nan`` policy means:
      ``ValueError: Expected parameter loc ... of distribution Normal ... to
      satisfy the constraint Real()``, a torch message that names that
      distribution's parameter rather than the field, raised after the env, both
      networks and a full rollout have been built. A list or a dict raises
      ``TypeError: float() argument must be a string or a real number`` out of
      the coercion instead. That is exactly the "deep stack trace" a read-only
      preflight exists to replace.

    **A numeric string is refused even though ``float()`` coerces it.** That is the
    one accepting-to-refusing change this gate makes, and it is a consistency one:
    the field is annotated ``float | None``, so a ``str`` reaches the coercion only
    by accident of ``float()`` accepting one, and both neighbouring fields of the
    same temperature block already refuse it - a spec whose ``alpha_lr`` is
    ``"3e-4"`` is rejected by name while ``target_entropy="-6"`` was not. Accepting
    it here alone would leave the three fields of one block disagreeing about which
    spellings a config may use.

    ``None`` is the one value in scope that is not a value: unlike ``init_alpha``
    and ``alpha_lr``, which are annotated ``float`` with concrete defaults, this
    field is annotated ``float | None`` and its ``None`` is the documented
    request for the ``-num_actions`` heuristic, which the coercion above honors.
    So the sentinel is exempt rather than refused - the only difference in domain
    between this gate and its two neighbours.

    Like :func:`initial_temperature_problems`, and unlike
    :func:`temperature_learning_rate_problems`, this is **not** scoped to
    ``autotune_alpha``: the coercion is unconditional, so a non-real value raises
    in ``setup`` on either branch - measured with tuning off, ``[-6.0]`` raised
    the same ``TypeError`` while ``nan`` reached a successful run that simply
    never spent it. A check conditioned on the tuning branch would therefore let
    the raising shape through for the untuned one.

    Only the SAC backend optimizes a temperature against a target entropy -
    ``spec.target_entropy`` appears in ``rl/fast_sac.py`` and nowhere else - so
    this is scoped like :func:`gae_lambda_problems` rather than
    :func:`learning_rate_problems`: a backend that does not read the field MUST
    NOT call this, because per :class:`TrainSpec` a backend ignores the fields it
    does not support and reporting on one would be a false rejection.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single-element list when ``target_entropy`` is neither the ``None``
        sentinel nor a finite real; empty when it can be honored.
    """
    value = getattr(spec, "target_entropy", None)
    if value is None:
        return []
    error = finite_number_error(value, "target_entropy", context)
    return [error] if error is not None else []


def _clip_bound_error(value: Any, param: str, context: str) -> str | None:
    """Error text when *value* is not a clip bound its consumer honors.

    Shared by the two on-policy clip bounds, which have the same domain for the
    same reason: ``max_grad_norm`` (see :func:`gradient_clip_problems`) and
    ``clip_param`` (see :func:`clip_range_problems`). One rule, one home - a
    second copy of it would be free to drift from this one.

    The whole of this function's own contribution is that positive **infinity is
    also accepted**; every other decision - numeric-ness, ``bool`` rejection,
    the positivity floor and the message text - is delegated to the shared
    :func:`~strands_robots.utils.positive_finite_number_error` domain, so those
    refusals read identically to every other positive-scalar field's.

    Infinity is carved out because both consumers honor it, and it is the only
    spelling of "do not clip" either field has:

    * ``clip_grad_norm_`` scales a gradient by ``max_norm / total_norm`` only
      when that ratio is below one, so an infinite bound leaves every gradient
      untouched - measured on a parameter whose gradient norm is 5, ``inf``
      returns it as ``[3.0, 4.0]`` unchanged.
    * ``torch.clamp(ratio, 1 - clip_param, 1 + clip_param)`` becomes
      ``clamp(ratio, -inf, inf)``, which returns ``ratio`` unchanged, so the
      surrogate and value clips both fall away and the update descends the
      unclipped objective - a coherent, finite run.

    A guard that refused a value its own consumer applies coherently would be
    narrower than the code it protects.

    Nothing raises out of here, for any input. The carve-out asks the
    *conversion* - that is what the consumer performs, ``clip_grad_norm_``
    reading its bound through ``float()`` - and it wraps that conversion, so a
    real past the float64 range and a :class:`numbers.Real` registration with
    no working ``__float__`` are delegated rather than raised on. ``10**400``
    is a registered real that is neither a ``bool`` nor convertible, and
    ``Fraction(10**400, 3)`` overflows identically; the shared domain already
    names that boundary and answers each with a reason of its own. Raising
    here would fail on the one path that exists to answer an unusable value
    with a message rather than an exception - the contract every caller of
    this gate documents.

    Args:
        value: The caller-supplied value.
        param: Field name for the message.
        context: Caller label the message is prefixed with.

    Returns:
        The error text, or None when *value* is a positive real number (finite
        or positive infinity).
    """
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        try:
            is_no_clip = float(value) == math.inf
        except Exception:
            # Decline to answer rather than raise. A ``numbers.Real``
            # registration owes this function no working ``__float__``, and a
            # real past the float64 range raises ``OverflowError`` from the
            # conversion - ``10**400`` and ``Fraction(10**400, 3)`` both do.
            # Neither is the no-clip spelling, and the shared domain below
            # answers each with a reason of its own.
            is_no_clip = False
        if is_no_clip:
            return None
    return positive_finite_number_error(value, param, context)


def gradient_clip_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return gradient-clip problems for an on-policy RL :class:`TrainSpec`.

    ``max_grad_norm`` is the last thing that touches a gradient before the
    optimizer steps - the on-policy update ends every mini-batch with
    ``clip_grad_norm_(self.actor_critic.parameters(), spec.max_grad_norm)`` -
    and nothing judged it. ``clip_grad_norm_`` does not either: it multiplies
    every gradient by ``max_norm / total_norm`` whenever that ratio is below
    one, and that expression is defined for values no caller can have meant.
    Measured on this backend, over a seeded 60-step run whose parameter sum
    starts at ``17.9251941755865118``:

    ======================  =========================================
    ``max_grad_norm``       Outcome
    ======================  =========================================
    ``1.0`` (the default)   trains; parameter sum ``17.9833114612``
    ``inf``                 trains, unclipped; sum ``17.9604155714``
    ``0`` / ``0.0``         **succeeds having learned nothing**
    ``-1.0`` / ``-0.5``     **trains in the opposite direction**
    ``True``               a silent clip of one
    ``"1.0"``              silently accepted
    ``nan``                 raises mid-update, from inside ``torch``
    ``None`` / ``[1.0]``    raises mid-update, from inside ``torch``
    ======================  =========================================

    The two silent rows are the reason this is a gate rather than a lint:

    * **Zero scales every gradient to zero**, so the optimizer steps with no
      information. The run collects its rollouts, reports ``success`` and writes
      a deployable checkpoint whose parameters are *bit-identical* to a
      never-trained control - the parameter delta is exactly ``0.0000000000``.
      This is the same shape as :func:`optimization_epochs_problems`, reached
      through a different field.
    * **A negative bound negates the ratio**, so every gradient is flipped and
      scaled: the same parameter whose gradient is ``[3.0, 4.0]`` comes out of
      ``clip_grad_norm_(-1.0)`` as ``[-0.6, -0.8]``. The update is therefore
      gradient *ascent* on the loss - the seeded run above moves its parameter
      sum to ``17.8211606460`` while the honored run moves it to
      ``17.9833114612``, i.e. away from the objective, silently, under a
      successful run.

    Zero is **not** the "no clipping" spelling, which is the one reading that
    might have made it a contract question rather than a defect: infinity is,
    and it is accepted (see :func:`_clip_bound_error`). So the domain is a
    positive real, finite or infinite, with no undecided endpoint.

    Only the on-policy backend clips gradients - ``grep`` finds
    ``clip_grad_norm_`` in ``rl/ppo.py`` and nowhere else - so this is scoped
    like :func:`gae_lambda_problems` rather than
    :func:`learning_rate_problems`: a backend that does not clip MUST NOT call
    it, because per :class:`TrainSpec` a backend ignores the fields it does not
    support and reporting on one would be a false rejection.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single-element list when ``max_grad_norm`` cannot be honored; empty
        otherwise.
    """
    error = _clip_bound_error(getattr(spec, "max_grad_norm", 1.0), "max_grad_norm", context)
    return [error] if error is not None else []


def loss_weight_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return loss-weight problems for an on-policy RL :class:`TrainSpec`.

    ``value_loss_coef`` and ``entropy_coef`` are the two scalars that weight the
    terms of the objective the on-policy update descends. They are read in
    exactly one place - the single expression that composes it::

        loss = surrogate_loss + spec.value_loss_coef * value_loss - spec.entropy_coef * entropy

    Nothing judged either of them, and the multiplication cannot: it is defined
    for values no caller can have meant, and every one of them reaches the
    backward pass. Measured on this backend, over a seeded 60-step run whose
    checkpoint parameter sum is ``140.6023186540351162`` when both weights are
    honored:

    =========================  =========================================
    Weight                     Outcome
    =========================  =========================================
    defaults (``1.0`` / ``0.0``)  trains; sum ``140.6023186540``
    ``True``                   **trains with a different coefficient**
    ``nan``                    raises mid-update, from inside ``torch``
    ``inf`` / ``-inf``         raises mid-update, from inside ``torch``
    ``"1.0"``                  raises mid-update, from inside ``torch``
    ``None`` / ``[1.0]``       raises mid-update, from inside ``torch``
    =========================  =========================================

    Both rows are what make this a gate rather than a lint:

    * **A boolean is a silently different coefficient.** ``bool`` is an ``int``
      subclass, so ``entropy_coef=True`` is an entropy bonus at full weight where
      the field ships defaulting to ``0.0`` - exploration is turned on by a value
      that reads as a flag. The run reports ``success`` and writes a checkpoint
      whose parameter sum is ``140.6158002523716277`` against the honored run's
      ``140.6023186540351162``, so it demonstrably trained differently rather
      than harmlessly.
    * **Everything non-finite or non-numeric raises out of ``train()``**, which
      is documented to return a terminal ``TrainResult`` and to fail closed on
      :meth:`~strands_robots.training.base.Trainer.validate` first. A ``nan``
      weight makes the loss ``nan``, the optimizer writes ``nan`` into every
      parameter, and the *next* rollout samples the action distribution from
      them: ``ValueError: Expected parameter loc ... of distribution Normal ...
      to satisfy the constraint Real()`` - a torch message that names neither the
      field nor the value, raised after the env, the networks and a full rollout
      have been built. A string raises ``TypeError: only integer tensors of a
      single element can be converted to an index`` from the same depth. That is
      exactly the "deep stack trace" a read-only preflight exists to replace.

    **The floor is deliberately not decided here.** Zero and negative are inside
    the domain for both fields, because both have a real reading:
    ``entropy_coef=0.0`` is the shipped default, a negative entropy weight is a
    penalty that drives the policy deterministic, and ``value_loss_coef=0`` stops
    training the critic. So the domain is a *finite real* - the one property no
    reading of either field can want without - checked through
    :func:`~strands_robots.utils.finite_number_error`, which also refuses the
    ``bool`` a bare comparison against zero would accept. This is the narrower
    counterpart of :func:`gradient_clip_problems`, whose endpoint *is* settled by
    ``clip_grad_norm_`` and which therefore tests positivity.

    Only the on-policy backend composes this objective - ``grep`` finds
    ``spec.value_loss_coef`` and ``spec.entropy_coef`` in ``rl/ppo.py`` and
    nowhere else - so this is scoped like :func:`gae_lambda_problems` rather than
    :func:`learning_rate_problems`: a backend that does not compose it MUST NOT
    call this, because per :class:`TrainSpec` a backend ignores the fields it does
    not support and reporting on one would be a false rejection.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        One problem per weight that cannot be honored; empty when both can.
    """
    defaults = {"value_loss_coef": 1.0, "entropy_coef": 0.0}
    problems = []
    for param, default in defaults.items():
        error = finite_number_error(getattr(spec, param, default), param, context)
        if error is not None:
            problems.append(error)
    return problems


def clip_range_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return trust-region problems for an on-policy RL :class:`TrainSpec`.

    ``clip_param`` is the half-width of the trust region PPO is named for. The
    on-policy update reads it twice, in the two expressions that clip::

        surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - spec.clip_param, 1.0 + spec.clip_param)
        value_clipped = old_values + (value - old_values).clamp(-spec.clip_param, spec.clip_param)

    Nothing judged it, and ``torch.clamp`` cannot: it is defined for every value
    below, and each one produces a *finite, successful, deployable* run whose
    objective is not the one the caller configured. Measured on this backend over
    a seeded 60-step run, against a never-trained control whose checkpoint
    parameter sum is ``139.8929914773252676``:

    ==========================  =====================================================
    ``clip_param``              Outcome (checkpoint parameter sum)
    ==========================  =====================================================
    ``0.2`` (shipped default)   clips; ``140.1741519418580992``
    ``inf``                     no clip, finite losses; ``140.1735330768706262``
    ``nan``                     **bit-identical to the unclipped run**, all losses ``nan``
    ``True``                    silent half-width of one; ``140.1735330768706262``
    ``-0.2``                    inverted bounds; ``140.1913412402318500``
    ``0``                       degenerate window; ``140.2282075245283863``
    ``-inf``                    trains on ``inf`` losses; ``140.0613218537641842``
    ``"0.2"`` / ``None`` / ``[0.2]``  raise ``TypeError`` mid-update, from ``rl/ppo.py``
    ==========================  =====================================================

    Three of those rows are why this is a gate rather than a lint:

    * **A ``nan`` half-width silently removes the trust region.** Both clipped
      terms become ``nan``, so ``torch.max(surrogate, surrogate_clipped)``
      returns ``nan`` - but its gradient flows to the *unclipped* branch, because
      every comparison against ``nan`` is false. The run therefore descends the
      unclipped objective and its checkpoint is bit-identical to the ``inf`` run,
      while ``surrogate_loss``, ``value_loss`` and ``latest_loss`` are all
      reported as ``nan``. PPO's defining mechanism is off and the only signal
      that anything happened is a metric a caller cannot act on.
    * **A negative half-width is not a window.** ``1 - c`` exceeds ``1 + c``, so
      the clamp bounds are inverted and it returns a constant regardless of the
      ratio - measured, ``clamp([0.7, 1.0, 1.4], 1.2, 0.8)`` is
      ``[0.8, 0.8, 0.8]`` - and the reported surrogate loss changes sign, from
      ``-0.008662`` to ``+0.081992``. Zero is the same failure at the boundary:
      the value clip becomes ``clamp(-0, 0)``, so ``value_clipped`` is exactly
      ``old_values`` and the critic's clipped branch is a constant.
    * **A ``bool`` is a silently different trust region.** ``bool`` is an ``int``
      subclass, so ``clip_param=True`` is a half-width of one - five times the
      shipped ``0.2`` - written by a value that reads as a flag.

    So the domain is a *positive* real, with positive infinity accepted as the
    field's only spelling of "do not clip". That is the same rule and the same
    reason as the sibling bound ``max_grad_norm``, so both read
    :func:`_clip_bound_error` rather than carrying a copy each.

    Only the on-policy backend clips a policy ratio - ``grep`` finds
    ``spec.clip_param`` in ``rl/ppo.py`` and nowhere else - so this is scoped
    like :func:`gae_lambda_problems` rather than :func:`learning_rate_problems`:
    a backend that does not clip MUST NOT call this, because per
    :class:`TrainSpec` a backend ignores the fields it does not support and
    reporting on one would be a false rejection.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single-element list when ``clip_param`` cannot be honored; empty
        otherwise.
    """
    error = _clip_bound_error(getattr(spec, "clip_param", 0.2), "clip_param", context)
    return [error] if error is not None else []


def policy_delay_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return policy-delay problems for an off-policy TD3 :class:`RLTrainSpec`.

    ``policy_delay`` is the number of critic updates FastTD3 runs per delayed
    actor / target update - the "delayed" of Twin Delayed DDPG. It is consumed
    as the modulus of the one test that decides whether the actor and the
    target networks move at all::

        if self._update_count % spec.policy_delay == 0:  # actor + Polyak step

    Nothing downstream judges it, and the modulus mis-handles every unusable
    spelling differently - measured on a CPU FastTD3 run of 6 gradient updates
    (an otherwise-valid spec, one field mutated):

    ======================  ==========  ===============  ======================
    ``policy_delay``        verdict     actor updates    what happened
    ======================  ==========  ===============  ======================
    ``2`` (the default)     honored     3                the asked-for delay
    ``float("nan")``        success     **0**            ``n % nan`` is ``nan``,
                                                         never ``== 0``
    ``2.5``                 success     **1** (not 3)    only multiples of 2.5
    ``True``                success     6                a silent delay of one
    ``0``                   raises      --               ``ZeroDivisionError``
                                                         mid-update
    ``"2"`` / ``None``      raises      --               ``TypeError`` from ``%``
    ======================  ==========  ===============  ======================

    The ``nan`` row is the reason this is a gate rather than a lint: the
    critics train for the whole run, the losses are real numbers, the run
    reports success and writes a checkpoint - but the actor inside it never
    took a gradient step, so the deployable half of the policy is its untrained
    initialisation with nothing anywhere reporting that. ``True`` and a
    fraction are silently a *different* delay from the one the caller named,
    and a non-positive or non-numeric value raises from inside the update loop
    after the env, the networks, the optimizers and the replay buffer are
    built - the cost a read-only preflight exists to precede.

    The domain is a positive integer, checked by
    :func:`~strands_robots.utils.positive_count_error` - the same rule this
    package applies to every count consumed as a modulus or ``range()`` bound -
    with ``1`` first-class: a delay of one is TD3 with the delay disabled,
    which is a configuration, not a defect.

    Only the TD3 backend delays its policy - PPO has no target networks and
    SAC updates its actor every gradient step - so this is scoped like
    :func:`gae_lambda_problems` rather than :func:`learning_rate_problems`: a
    backend that does not read the field MUST NOT call this, because per
    :class:`TrainSpec` a backend ignores the fields it does not support and
    reporting on one would be a false rejection.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single-element list when ``policy_delay`` cannot be honored; empty
        otherwise.
    """
    error = positive_count_error(getattr(spec, "policy_delay", 1), "policy_delay", context)
    return [error] if error is not None else []


def td3_noise_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return noise-scale problems for an off-policy TD3 :class:`RLTrainSpec`.

    ``exploration_noise_std``, ``target_noise_std`` and ``target_noise_clip``
    are the three scalars that shape FastTD3's two noise mechanisms, read in
    exactly two expressions::

        action = (action + torch.randn_like(action) * spec.exploration_noise_std).clamp(-1.0, 1.0)
        noise = (torch.randn_like(a) * spec.target_noise_std).clamp(-spec.target_noise_clip, spec.target_noise_clip)

    The first is the only exploration a deterministic policy has once the
    random warmup ends; the second is target policy smoothing, the mechanism
    that keeps the critic from exploiting its own sharp errors. Neither
    expression judges its scalars, and each unusable value produces a finite,
    successful run that is not the configured one:

    * **Zero silently removes the mechanism.** ``randn * 0`` is exactly ``0``,
      so ``exploration_noise_std=0`` collects with the deterministic action
      alone - the policy revisits the identical trajectory instead of
      exploring around it - and ``target_noise_std=0`` (or a clip of ``0``,
      which clamps every sample to ``[0, 0]``) trains plain clipped double-Q
      while reporting the smoothing knobs it was asked for. Neither run can be
      told apart from an honored one by its result object.
    * **A negative scale is silently the identical run.** Gaussian noise is
      symmetric, so ``randn * (-s)`` draws from the same distribution as
      ``randn * s`` - a value that reads as different and changes nothing. A
      negative *clip* is worse than identical: ``clamp`` with inverted bounds
      returns the constant upper bound, so every smoothing sample becomes the
      same negative offset and the target is biased rather than smoothed.
    * **A non-finite value poisons what the expression feeds.** A ``nan`` std
      makes every collected action (or every TD target, and from there every
      critic parameter) ``nan`` under a run that keeps stepping; an ``inf``
      exploration std saturates the clamp so every action is a coin-flip
      between the bounds ``-1`` and ``1`` - bang-bang control, not a large
      noise. ``True`` is a silent scale of ``1.0``, ten times the shipped
      exploration default, from a value that reads as a flag.

    Only a positive finite number can be honored, so all three are checked
    against the shared
    :func:`~strands_robots.utils.positive_finite_number_error` domain - the
    family that already owns the dimensionless-multiplier scalars - which also
    refuses the ``bool`` a bare comparison would accept. There is no "disable"
    spelling to carve out, unlike :func:`_clip_bound_error`'s infinity: a run
    that wants no smoothing is a different algorithm, not a boundary value of
    this one.

    Only the TD3 backend reads the three fields - SAC explores through its
    stochastic actor and smooths nothing - so this is scoped like
    :func:`policy_delay_problems`: a backend that does not read them MUST NOT
    call this, because per :class:`TrainSpec` a backend ignores the fields it
    does not support and reporting on one would be a false rejection.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        One problem per unusable noise scalar; empty when all three are usable.
    """
    problems: list[str] = []
    for param, default in (
        ("exploration_noise_std", 0.1),
        ("target_noise_std", 0.2),
        ("target_noise_clip", 0.5),
    ):
        error = positive_finite_number_error(getattr(spec, param, default), param, context)
        if error is not None:
            problems.append(error)
    return problems


def network_width_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return hidden-layer-width problems for a from-scratch RL :class:`TrainSpec`.

    ``hidden_dims`` is the only spec field that decides the *shape* of the
    networks a run trains. Every RL backend expands it the same way, once per
    network it builds - the on-policy actor and critic, and off-policy the actor,
    its Polyak target and all four Q heads::

        for h in spec.hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, out_dim))

    Nothing judged the widths, and ``nn.Linear`` does not either: a width of
    zero is a legal layer. It builds a ``(n, 0)`` weight - ``torch`` only warns
    "Initializing zero-element tensors is a no-op" - and every forward pass
    through it produces a ``(batch, 0)`` activation, so the next layer sees no
    input and emits its bias alone. **The output stops being a function of the
    observation.** Measured on the fake one-joint env, over a full ``train()``
    on each backend, comparing the trained actor's action for an all-zero
    observation against an all-``50`` one:

    ======================  ==============  ==================================
    ``hidden_dims``         Verdict         Outcome
    ======================  ==============  ==================================
    ``(16,)`` (a control)   accepted        actions differ: ``0.109`` / ``-0.869``
    ``()``                  accepted        a linear policy; actions differ
    ``(0,)``               **accepted**    every action bit-identical
    ``(16, 0)``            **accepted**    every action bit-identical
    ``(-1,)``               accepted        raises inside ``setup``
    ``(16.0,)`` / ``(True,)`` accepted      raises inside ``setup``
    ``np.int64(16)``        accepted        trains, then the checkpoint raises
    ``16`` / ``None``       accepted        raises inside ``setup``
    a generator             accepted        a different shape per network
    ======================  ==============  ==================================

    The two bold rows are why this is a gate rather than a lint. A zero width
    anywhere in the sequence severs the policy from the robot: the run collects
    its rollouts, the critics train against a constant, ``train()`` returns
    ``status="success"`` with a real ``actor_loss`` and a real ``actor_updates``
    count, and it exports ``policy.pt`` + ``policy_meta.json`` - a *deployable*
    checkpoint whose actor emits one fixed action for every state the robot can
    ever be in. On hardware that is an arm driving to a single pose and staying
    there, reported as a trained policy. This is the same shape as
    :func:`gradient_clip_problems` (a run that reports success having learned
    nothing), reached through the one field that can make learning *impossible*
    rather than merely ineffective.

    The empty sequence is **not** in that class and stays accepted: it is the
    honest spelling of a linear policy (input straight to output, no hidden
    layer), and its action still varies with the observation. So the domain is
    per element, not on the length.

    Each width is asked of the shared :func:`~strands_robots.utils.positive_count_error`
    domain, the same one the loop-bound counts use, because a width is consumed
    directly as a tensor dimension: an integral float raises ``TypeError``
    inside ``torch`` rather than being coerced, and ``bool`` would otherwise
    pass a bare ``< 1`` test as a silent width of one. That domain also refuses
    ``np.int64``, which is not an over-refusal here even though ``nn.Linear``
    accepts it - ``save_checkpoint`` writes ``list(spec.hidden_dims)`` to
    ``policy_meta.json`` and ``json.dump`` raises ``TypeError: Object of type
    int64 is not JSON serializable``, so a run configured that way trains to
    completion and then loses the whole run at the save.

    The container is checked before its elements, since a field that is not a
    sequence of widths has no elements to name: a bare ``int`` or ``None`` is
    not iterable, a ``str`` iterates into characters, and a one-shot iterator is
    worse than either - a generator is consumed by the first network built, so
    the actor gets the requested architecture and every later critic silently
    gets none (measured: 452 parameters for the first, 28 for the second).

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        One problem per unusable width (named by index), or a single problem
        when ``hidden_dims`` is not a sequence of widths at all; empty when
        every width can be honored.
    """
    widths = getattr(spec, "hidden_dims", ())
    if isinstance(widths, str) or not isinstance(widths, Sequence):
        return [f"{context}: hidden_dims must be a sequence of positive int layer widths, got {widths!r}"]
    return [
        error
        for index, width in enumerate(widths)
        if (error := positive_count_error(width, f"hidden_dims[{index}]", context)) is not None
    ]


def rl_checkpoint_interval_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return checkpoint-cadence problems for an RL :class:`RLTrainSpec`.

    ``log_interval`` is how often an RL run writes a checkpoint. Its name and
    its :class:`RLTrainSpec` entry both said "iterations between progress logs",
    but no RL module emits a log line at all - the field is read in exactly one
    expression, the one that decides whether an intermediate checkpoint is
    written, and all three backends share it::

        if spec.log_interval and (it % spec.log_interval == 0 or it == num_iters - 1):
            ckpt_dir = self.save_checkpoint(spec.output_dir, iteration=it + 1)

    So it is the same *kind* of value as
    :attr:`~strands_robots.training.base.TrainSpec.save_freq` - a periodic
    checkpoint cadence consumed as a modulus - and it reached that modulus with
    no domain at all, while the supervised field beside it on the same spec has
    one. Measured on the inherited loop over a 20-iteration run, against the
    ``[1, 6, 11, 16, 20]`` an ``int`` cadence of 5 writes:

    ====================  ==============  ====================================
    ``log_interval``      Verdict         Outcome
    ====================  ==============  ====================================
    ``5`` (a control)     accepted        5 checkpoints, at 1, 6, 11, 16, 20
    ``True``             **accepted**     20 checkpoints - one every iteration
    ``2.5``              **accepted**     5, at 1, 6, 11, 16, 20 - the schedule
                                          of ``5``, not of ``2.5``
    ``0.3``              **accepted**     2, at 1 and 20
    ``nan``              **accepted**     1, at 20 - the *disabled* mode
    ``inf``              **accepted**     2, at 1 and 20
    ``"5"``              **accepted**     ``TypeError`` out of ``train()``
    ``0``                 accepted        1, at 20 - the documented disable
    ``-5``                accepted        5, at 1, 6, 11, 16, 20
    ====================  ==============  ====================================

    The bold rows are why this is a gate. ``nan`` is the worst of them: it
    satisfies the truthiness guard, never satisfies the modulus, and so silently
    becomes the disabled mode - a long run that asked to checkpoint every 5
    iterations keeps only the final one, under ``status="success"``. For RL the
    intermediate checkpoints are not a convenience: return is non-monotonic in
    training, so the deployable policy is often an earlier one, and a run that
    kept only its last iteration cannot be recovered without training again.
    ``True`` is the opposite failure at the same seam (a modulus of one, a full
    checkpoint every iteration), ``2.5`` is indistinguishable from the ``5`` a
    caller who wanted twice as many checkpoints did not write, and ``"5"``
    raises out of the training loop *after* ``setup`` has built the env, the
    networks and the optimizers - from a lifecycle whose whole contract is to
    report a failure as :class:`~strands_robots.training.base.TrainResult`, and
    past a :meth:`~strands_robots.training.base.Trainer.validate` documented to
    return every problem a run has.

    Only the *type* is graded, exactly as for :func:`checkpoint_cadence_problems`,
    and the two disabling spellings are first-class: ``0`` disables periodic
    saving and leaves the end-of-run fallback in
    :meth:`~strands_robots.training.rl.base_algo.BaseRLAlgo.train` to write the
    single final checkpoint, a configuration that loop's own contract tests
    already pin. A negative is accepted for the same
    no-floor reason and is *not* the disabled mode here - unlike lerobot's
    ``save_freq > 0`` test, this loop's guard is bare truthiness, so ``-5`` is
    the cadence of its magnitude (measured above). Which spelling disables is
    therefore the loop's own business; the domain's business is that the value
    is a whole number of iterations.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`, so a
            problem names the backend that refused the value.

    Returns:
        A single problem when ``log_interval`` is not a whole number of
        iterations; empty otherwise.
    """
    error = step_cadence_error(getattr(spec, "log_interval", 0), "log_interval", context)
    return [] if error is None else [error]


def _half_open_unit_interval_error(value: Any, param: str, context: str) -> str | None:
    """Error text when *value* is not a real number in the half-open range (0, 1].

    Sibling of :func:`_closed_unit_interval_error`, and the two differ in exactly
    one decision: whether zero is inside the interval. Numeric-ness, ``bool``
    rejection, finiteness and the float64 range are delegated to the shared
    :func:`~strands_robots.utils.finite_number_error` domain by both, so those
    refusals read identically to every other numeric field's.

    Zero is **outside** this interval because it is a degenerate spelling rather
    than a reading: the one consumer is a Polyak update, ``tp.mul_(1 -
    x).add_(x * p)``, which at zero leaves the target parameters at their
    initialization for the whole run. The upper endpoint is inside, because at
    one the same expression is the hard update ``tp = p`` that TD3-family
    algorithms take deliberately.

    Args:
        value: The caller-supplied value.
        param: Field name for the message.
        context: Caller label the message is prefixed with.

    Returns:
        The error text, or None when *value* is a real number in (0, 1].
    """
    error = finite_number_error(value, param, context)
    if error is not None:
        return error
    if not 0.0 < float(value) <= 1.0:
        return f"{context}: {param} must be in (0, 1], got {value!r}."
    return None


def polyak_coefficient_problems(spec: TrainSpec, *, context: str) -> list[str]:
    """Return Polyak-coefficient problems for an off-policy RL :class:`TrainSpec`.

    ``tau`` is the rate at which a target network tracks its online network. The
    two off-policy backends spend it in one expression each, per mirrored critic
    pair::

        tp.mul_(1.0 - spec.tau).add_(spec.tau * p)

    so the field does not merely tune the update - it decides whether a separate
    target network exists at all, and a target network is the mechanism that
    makes an off-policy critic's bootstrap target stationary enough to regress
    onto.

    This interval was the *precedent* the two on-policy interval gates were
    written against - :func:`discount_factor_problems` and
    :func:`gae_lambda_problems` both cite "the sibling FastSAC preflight already
    bounds its own interval coefficient this way (``tau`` must be in ``(0, 1]``)"
    as the shape they generalize - while the check they cited was a bare local
    comparison, ``if not 0.0 < spec.tau <= 1.0``, duplicated verbatim in both
    backends. A bare comparison against the interval bounds cannot carry the
    domain, and it left two holes the generalization had already closed:

    * ``True`` is a silent ``tau`` of **one**, because ``bool`` is an ``int``
      subclass. That is the maximum of the interval, so the Polyak average
      degenerates to ``tp.mul_(0.0).add_(p)`` - the target network becomes a
      copy of the online network on every update, which is the same thing as
      having no target network. Measured on a 60-timestep FastSAC run,
      ``validate()`` returning ``[]`` and the run reporting success: the largest
      online-to-target parameter gap in the exported checkpoint was ``0.0``
      exactly against the default ``tau``'s ``9.9e-04``, and the checkpoint was
      byte-identical to a run that asked for ``tau=1.0``. So a flag landed as a
      request to switch off target networks, and nothing in the run said so.
    * A numeric **string**, ``None`` or a list raises ``TypeError: '<' not
      supported between instances of 'float' and 'str'`` out of the comparison
      itself - from a
      :meth:`~strands_robots.training.base.Trainer.validate` documented to
      *return* its problems, which is the contract every other field on this
      spec is now checked against.

    ``nan`` and the two infinities were already refused, by an accident of
    Python's chained comparison rather than by a finiteness test: every
    comparison against ``nan`` is False and ``inf`` is above the upper bound, so
    the bare test happened to answer them. They stay refused here, on the
    shared domain, with the reason naming finiteness rather than the interval.

    Both endpoints keep the reading the bare test gave them, so no value that
    worked becomes an error: ``1.0`` is accepted as the deliberate hard update,
    and ``0.0`` is refused because it freezes the target network at its
    initialization for the whole run - see
    :func:`_half_open_unit_interval_error`, which owns that one decision and is
    what makes this interval half-open where the on-policy pair's is closed.

    Only a backend that maintains a target network may call this: like
    :func:`gae_lambda_problems`, and unlike :func:`learning_rate_problems`, a
    backend that does not read the field MUST NOT report on it - PPO has no
    target network and never reads ``tau``.

    Args:
        spec: The spec to check.
        context: Caller identity for the message prefix - the backend's
            :attr:`~strands_robots.training.base.Trainer.provider_name`.

    Returns:
        A single-element list when ``tau`` cannot be honored; empty otherwise.
    """
    error = _half_open_unit_interval_error(getattr(spec, "tau", 0.005), "tau", context)
    return [error] if error is not None else []
