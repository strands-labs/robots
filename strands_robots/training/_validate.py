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
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from strands_robots.tools._path_validation import validate_save_path
from strands_robots.utils import positive_count_error, positive_finite_number_error

if TYPE_CHECKING:
    from strands_robots.training.base import TrainSpec

# ``extra`` keys are interpolated into argv as ``--{key}=...`` (lerobot/groot)
# or ``{key}=...`` (cosmos hydra). Allowlist the key FORMAT only: lowercase,
# dotted (lerobot ``dataset.episodes`` / cosmos ``model.x.y``), no leading dash,
# no ``=``, no whitespace or shell metacharacters. We deliberately do NOT try to
# enumerate every valid backend flag - that allowlist is impossible to keep
# current and would break the documented ``extra`` escape hatch.
_EXTRA_KEY_RE = re.compile(r"^[a-z][a-z0-9_.]*$")

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
