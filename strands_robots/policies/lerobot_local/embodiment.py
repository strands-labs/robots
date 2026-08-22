"""Declarative robot/sim ↔ LeRobot-model key mapping for the local policy.

This module replaces the per-step imperative observation/action remapping
heuristics in :mod:`policy` with a **declarative, validated, build-once**
mapping that plugs straight into LeRobot's own processor pipeline.

* :class:`EmbodimentMap` is a frozen dataclass describing how a robot/sim's
  native observation keys map onto the model's declared LeRobot feature keys
  (``observation.images.*`` / ``observation.state``) and how the model's action
  tensor maps back onto named robot actuators. It mirrors the GR00T
  ``Gr00tDataConfig`` pattern that already works in this codebase.

* :class:`PackStateProcessorStep` is the ONE new registered pipeline step: it
  composes the robot's scalar joint observations into ``observation.state`` in a
  declared order, with an **explicit** dim-reconciliation policy (no silent
  truncate/pad). It runs inside LeRobot's pipeline, right after the rename step.

* The map is built and **validated against the model's declared features once at
  load time** (fail-fast), then the pipeline owns every per-step transform.

Embodiment definitions live in ``embodiments.json`` next to this module and
support ``_extends`` inheritance + ``aliases`` (same loader shape as
``groot/data_configs.json``).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.utils import finite_number_error, positive_whole_number_error

logger = logging.getLogger(__name__)


# Dim reconciliation


def reconcile_dim(values: list[float], expected_dim: int, dim_policy: str, *, label: str = "state") -> list[float]:
    """Reconcile a value vector to ``expected_dim`` per an explicit policy.

    Unlike the old hot-path heuristics, this is called ONCE per step inside a
    single registered pipeline step and the policy is **declared** by the
    embodiment, never guessed.

    Args:
        values: The collected scalar values (ordered).
        expected_dim: The dimension the model declares for this feature.
        dim_policy: One of ``"strict"`` | ``"pad"`` | ``"truncate"``.
        label: Human label for error/log messages.

    Returns:
        A list of length ``expected_dim``.

    Raises:
        ValueError: If ``dim_policy == "strict"`` and lengths differ.
    """
    n = len(values)
    if n == expected_dim:
        return values
    if dim_policy == "strict":
        raise ValueError(
            f"{label} dim {n} != model expected {expected_dim}. "
            f"Set dim_policy='pad' or 'truncate' on the embodiment to opt in to adaptation."
        )
    if dim_policy == "truncate":
        if n < expected_dim:
            raise ValueError(f"{label} dim {n} < model expected {expected_dim}; cannot truncate. Use dim_policy='pad'.")
        return values[:expected_dim]
    if dim_policy == "pad":
        if n > expected_dim:
            raise ValueError(f"{label} dim {n} > model expected {expected_dim}; cannot pad. Use dim_policy='truncate'.")
        return values + [0.0] * (expected_dim - n)
    raise ValueError(f"Unknown dim_policy {dim_policy!r}; expected 'strict'|'pad'|'truncate'.")


def _convert_joint_vector(
    values: list[float],
    *,
    to_model: bool,
    gripper_index: int = -1,
    gripper_joint_range: list[float] | None = None,
    joint_mids: list[float] | None = None,
) -> list[float]:
    """Convert an ordered joint vector between sim units (radians + gripper joint
    range) and the LeRobot SO-arm training units (arm degrees, gripper 0..100).

    Shared by :class:`EmbodimentMap` (action side) and ``PackStateProcessorStep``
    (state side) so both directions use one implementation.

    * ``to_model=True``  sim -> model: arm radians -> degrees; gripper joint
      radians -> 0..100.
    * ``to_model=False`` model -> sim: arm degrees -> radians; gripper 0..100 ->
      joint radians.

    The gripper column (``gripper_index``) maps against ``gripper_joint_range``
    because the SO-arm gripper uses ``MotorNormMode.RANGE_0_100`` (0..100), not
    degrees - see ``lerobot/robots/so_follower/so_follower.py``.

    The arm-degrees direction assumes the checkpoint was recorded with the SO
    driver's ``use_degrees=True`` (its default, but opt-out). With
    ``use_degrees=False`` the arm is ``MotorNormMode.RANGE_M100_100`` (-100..100),
    so this degree conversion must not be applied to such a checkpoint.

    LeRobot's ``MotorNormMode.DEGREES`` is **mid-point-centered**: the value a
    checkpoint trains on is the angular displacement from each motor's
    calibration mid-point, not the absolute joint angle (ground truth:
    ``lerobot/motors/motors_bus.py`` ``_normalize`` / ``_unnormalize`` ->
    ``mid = (range_min + range_max) / 2``; reported degrees = ``(val - mid) *
    360 / max_res``). When ``joint_mids`` is supplied (per-joint mid offsets in
    degrees, aligned to ``values``), the arm conversion subtracts the mid going
    to the model and adds it back coming from the model, so the packed
    ``observation.state`` matches the distribution the checkpoint was trained on
    rather than being offset by each joint's mid. When ``joint_mids`` is empty
    (the default), the mid is treated as zero -- i.e. the sim ``qpos = 0`` is
    assumed to coincide with the calibration mid (absolute ``deg = rad *
    180/pi``), preserving the prior behavior.

    Args:
        values: Ordered joint values.
        to_model: Conversion direction (see above).
        gripper_index: Index of the gripper column, or -1 for none.
        gripper_joint_range: ``[min, max]`` radians of the sim gripper joint;
            empty/None treats the gripper like an arm joint (deg<->rad).
        joint_mids: Per-joint calibration mid-points in DEGREES, aligned to
            ``values``. Subtracted from arm columns when ``to_model`` and added
            back otherwise, matching ``motors_bus`` DEGREES mid-centering. The
            gripper column (``gripper_index``) is exempt (RANGE_0_100 has no
            mid). Empty/None / out-of-range indices use a mid of ``0.0``.

    Returns:
        A new list of converted values (input is not mutated).
    """
    out = list(values)
    rad_per_deg = float(np.pi) / 180.0
    grange = gripper_joint_range or []
    mids = joint_mids or []
    for i, v in enumerate(out):
        if i == gripper_index and len(grange) == 2:
            lo, hi = float(grange[0]), float(grange[1])
            span = hi - lo
            if span == 0.0:
                continue
            if to_model:
                out[i] = (float(v) - lo) / span * 100.0  # joint rad -> 0..100
            else:
                out[i] = lo + (float(v) / 100.0) * span  # 0..100 -> joint rad
        else:
            mid = float(mids[i]) if i < len(mids) else 0.0
            if to_model:
                out[i] = float(v) / rad_per_deg - mid  # radians -> mid-centered degrees
            else:
                out[i] = (float(v) + mid) * rad_per_deg  # mid-centered degrees -> radians
    return out


# Hardware observation detection


def _is_boolean_flag(value: object) -> bool:
    """Whether ``value`` carries a boolean, in any of the flavours an observation uses.

    A driver status flag is not a joint reading, and the exclusion has to cover
    every spelling of "boolean" because none of them is caught by a check for
    another one:

    ==========================  ============================================
    ``True``                    an ``int`` subclass, so a numeric check takes it
    ``np.bool_(True)``          NOT a ``bool`` subclass, and NOT an
                                ``np.integer`` - it reaches a duck-typed 0-d
                                check instead (``ndim`` is 0, ``item`` exists)
    ``np.array(True)``          an ``ndarray`` whose ``ndim`` is 0
    ``torch.tensor(True)``      0-d with an ``item``, dtype ``torch.bool``
    ==========================  ============================================

    So the flag test is done once here, ahead of every accept branch, rather
    than per branch: a branch added later cannot reopen the hole.

    Dtype is matched by kind rather than identity so the check does not import
    torch (this module stays light) and covers array libraries generally. An
    unrecognised dtype spelling therefore reads as boolean if it says "bool",
    which is the safe direction: a wrongly-rejected reading yields a SHORT key
    list, and a short list refuses the hardware override (see
    ``hardware_pos_keys`` callers) instead of binding a misaligned one.
    """
    if isinstance(value, (bool, np.bool_)):
        return True
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return False
    # numpy dtypes expose ``.kind`` ('b' for boolean); torch's do not, but every
    # dtype spelling that means boolean says so in its string form.
    return getattr(dtype, "kind", None) == "b" or "bool" in str(dtype).lower()


def _is_joint_scalar(value: object) -> bool:
    """Whether ``value`` is a single numeric joint reading.

    Accepts python and numpy scalars plus 0-d arrays/tensors; rejects booleans
    in every flavour (see :func:`_is_boolean_flag`) and anything with more than
    one element.

    The dtype coverage is the point: ``isinstance(np.float64(1.0), float)`` is
    True but ``isinstance(np.float32(1.0), float)`` is False, so a predicate
    written as "plain floats, and numpy only if that found nothing" answers
    differently for a float32 observation than for a float64 one - and
    differently again for an observation that mixes the two.
    """
    if _is_boolean_flag(value):
        return False
    if isinstance(value, (int, float, np.floating, np.integer)):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim == 0
    # 0-d torch tensor, without importing torch here (this module stays light).
    return getattr(value, "ndim", None) == 0 and hasattr(value, "item")


def hardware_pos_keys(observation: dict[str, Any]) -> list[str]:
    """Ordered ``'<motor>.pos'`` keys of ``observation`` carrying a joint reading.

    The single source of truth for "does this observation come from real
    hardware?", shared by the state side (:class:`PackStateProcessorStep`) and
    the action side (``LerobotLocalPolicy._hardware_action_keys``).

    Both sides bind their vectors positionally, so the returned order - the
    observation's own insertion order, i.e. lerobot motor order - is part of the
    contract.

    Args:
        observation: A raw robot/sim observation dict.

    Returns:
        The matching keys, in observation order.
    """
    return [
        key
        for key, value in observation.items()
        if isinstance(key, str) and key.endswith(".pos") and _is_joint_scalar(value)
    ]


# Above this many observation keys the remedy points at the list the diagnostic
# already printed instead of repeating it - a 29-joint humanoid would otherwise
# print the same long list twice in one message.
_REMEDY_KEYS_INLINE_MAX = 8


def matching_embodiments(observation_keys: Iterable[Any]) -> list[str]:
    """Shipped embodiment names whose entire ``state_keys`` set the observation carries.

    The registry is the only place that knows which joint namings the library
    can already bind, so a diagnostic that wants to recommend an ``embodiment=``
    must ask it rather than hardcode an example. An embodiment qualifies only
    when EVERY one of its declared ``state_keys`` is present, because a partial
    match would reproduce the very mismatch the caller is trying to escape.

    Several embodiments can qualify at once and that is not an error: the real
    SO, Koch and OMX arms all report the same six ``'<motor>.pos'`` keys, so an
    observation cannot distinguish them. Callers should present all of them.

    Aliases are excluded so the result names one spelling per configuration.

    Args:
        observation_keys: Keys of the observation being diagnosed. Non-string
            entries are ignored rather than rejected, since an observation is
            not guaranteed to be string-keyed.

    Returns:
        Sorted matching configuration names; empty when none match.
    """
    present = {key for key in observation_keys if isinstance(key, str)}
    if not present:
        return []
    return sorted(
        name
        for name, embodiment in EMBODIMENT_MAP.items()
        if name in _CONFIG_NAMES and embodiment.state_keys and present.issuperset(embodiment.state_keys)
    )


def state_key_remedy(observation_keys: Iterable[Any]) -> str:
    """Advice for a state-key mismatch, chosen from what the observation contains.

    A fixed example cannot be right for every caller. Recommending
    ``embodiment='so101'`` to a real SO arm is not merely unhelpful: that
    configuration declares the MuJoCo asset's numeric joints (``'1'..'6'``),
    none of which a ``'<motor>.pos'`` hardware observation carries, so following
    the advice lands back on the same all-missing mismatch - and its
    ``state_units='degrees'`` would convert units the hardware reports natively.

    So the embodiment is named only when the registry confirms it binds THIS
    observation (see :func:`matching_embodiments`), and when nothing matches no
    embodiment is offered at all. ``set_robot_state_keys`` is always offered as
    the unambiguous alternative, quoting the observed keys verbatim when the
    list is short enough to paste.

    Args:
        observation_keys: Keys of the observation being diagnosed, in the order
            they should be bound.

    Returns:
        One to three sentences of remedy, plain ASCII, ending in a period. An
        observation with no string keys at all gets no remedy to follow, only a
        statement that nothing can bind it.
    """
    keys = [key for key in observation_keys if isinstance(key, str)]
    if not keys:
        return (
            "This observation carries no scalar state keys at all, so no embodiment or "
            "set_robot_state_keys([...]) ordering can bind it - check that the robot/sim "
            "is reporting joint positions."
        )
    if len(keys) <= _REMEDY_KEYS_INLINE_MAX:
        set_keys = f"call set_robot_state_keys({keys!r})"
    else:
        set_keys = "call set_robot_state_keys([...]) with the observed keys above"

    candidates = matching_embodiments(keys)
    if not candidates:
        return (
            f"No shipped embodiment declares state_keys this observation carries, so {set_keys}. "
            "Passing an embodiment chosen by robot name instead would re-declare keys the "
            "observation does not have and land back here."
        )
    if len(candidates) == 1:
        return f"Pass embodiment='{candidates[0]}', whose state_keys this observation carries, or {set_keys}."
    listed = " / ".join(f"'{name}'" for name in candidates)
    return (
        f"Pass embodiment= one of {listed} - each declares state_keys this observation "
        f"carries, so pick the one matching your robot - or {set_keys}."
    )


# Action diagnostics


def diagnose_action_dim(
    n_action_values: int, n_action_keys: int, *, name: str = "", pad_short: bool = False
) -> str | None:
    """Return a warning message when a model action vector mis-matches the
    embodiment's declared actuator count, else ``None``.

    The local policy maps a model's action tensor onto robot actuators by index
    (``LerobotLocalPolicy._tensor_to_action_dicts``). When the model emits FEWER
    values than the embodiment declares actuator keys, the unmatched actuators
    get no value from the model, and when it emits MORE the extra trailing values
    are dropped. Either case is almost always an embodiment/checkpoint mismatch
    the operator wants surfaced, not swallowed.

    What happens to those unmatched actuators is the caller's choice, so the
    message has to report the behaviour actually in effect: by default they are
    omitted from the action dict and hold position, while
    ``pad_short_actions=True`` sends them an explicit ``0.0``, which on an
    absolute-position action space travels them to zero.

    Args:
        n_action_values: Length of the model's per-step action vector.
        n_action_keys: Number of declared actuator keys (``robot_state_keys``).
        name: Embodiment name for the message (optional).
        pad_short: Whether the caller pads the unmatched actuators with ``0.0``
            (see :func:`strands_robots.policies.base.align_action_values`).
            Selects which consequence the message describes.

    Returns:
        A human-readable warning string, or ``None`` when the dims match.
    """
    if n_action_values == n_action_keys:
        return None
    label = f" '{name}'" if name else ""
    if n_action_values < n_action_keys:
        missing = n_action_keys - n_action_values
        consequence = (
            f"the {missing} unmatched actuator(s) are commanded to 0.0 "
            f"(pad_short_actions=True), which on an absolute-position action space travels "
            f"them to zero rather than holding them"
            if pad_short
            else f"the {missing} unmatched actuator(s) receive no command and hold their current position"
        )
        return (
            f"Policy action dim {n_action_values} < embodiment{label} actuator count "
            f"{n_action_keys}: {consequence}. Check the embodiment's action_keys "
            f"order/count against the checkpoint's action dimension."
        )
    extra = n_action_values - n_action_keys
    return (
        f"Policy action dim {n_action_values} > embodiment{label} actuator count "
        f"{n_action_keys}: {extra} trailing action value(s) are dropped. Check the "
        f"embodiment's action_keys against the checkpoint's action dimension."
    )


class ZeroActionMonitor:
    """Detect a policy that keeps emitting near-zero actions (no robot motion).

    Even with correct action dims and units, a misconfigured obs/rename pipeline
    (a dropped camera key, an all-zero ``observation.state``) makes a VLA emit
    effectively-zero actions every step: the robot "runs the policy" but never
    moves. This monitor watches the per-step action magnitude and emits ONE
    warning when it stays below ``threshold`` for ``patience`` consecutive steps,
    pointing the operator at the embodiment / rename config.

    A NON-FINITE magnitude is reported separately, because it is a different
    fault with a different cause. ``nan`` compares ``False`` against every
    threshold, so it would otherwise advance the near-zero streak and be
    reported as a near-zero stream -- naming the obs/rename pipeline for an
    action that is not near zero but not a number, while five real commands in
    the same vector are ignored. ``inf`` compares ``True`` and would instead
    clear the streak, leaving an action the backends refuse outright entirely
    unreported. Neither value is evidence about the observation pipeline.

    Stateful but dependency-free (no torch/lerobot) so it is unit-testable in
    isolation. Call :meth:`update` once per inference step and :meth:`reset` on
    episode reset.

    Attributes:
        threshold: Max-abs action magnitude below which a step counts as
            near-zero. Finite and ``>= 0``. The comparison is ``>=``, so a
            threshold of ``0`` accepts every magnitude as motion and thereby
            disables the near-zero report; it is permitted for compatibility.
        patience: Consecutive near-zero steps required before warning. A
            positive whole number.
    """

    def __init__(self, threshold: float = 1e-3, patience: int = 10) -> None:
        # The floor is decided here and everything else -- numeric-ness, bool,
        # finiteness -- is delegated to the shared numeric rule, the same
        # division of labour as WBCConfig's gain domain. A bare ``threshold < 0``
        # comparison cannot express it: ``nan < 0`` and ``inf < 0`` are both
        # False, so both were stored, and a threshold of ``nan`` or ``inf``
        # compares False against EVERY magnitude -- the watchdog then fires on a
        # healthy policy. ``True`` is an ``int`` subclass, so it was stored as a
        # threshold of 1.0: on an SO-arm that reads every real action as
        # near-zero. Symmetrically ``patience`` of ``nan``/``inf`` made
        # ``streak >= patience`` False forever, silently disabling the warning
        # this class exists to emit.
        if error := finite_number_error(threshold, "threshold", "ZeroActionMonitor"):
            raise ValueError(error)
        if float(threshold) < 0.0:
            raise ValueError(f"ZeroActionMonitor: threshold must be >= 0, got {threshold!r}.")
        if error := positive_whole_number_error(patience, "patience", "ZeroActionMonitor"):
            raise ValueError(error)
        # Normalized so the two public attributes match their declared types:
        # ``patience`` is read as a step count by callers (``range(mon.patience)``),
        # which an integral float the guard accepts would break.
        self.threshold = float(threshold)
        self.patience = int(patience)
        self._streak = 0
        self._warned = False
        self._nonfinite_warned = False

    def update(self, max_abs_action: float) -> str | None:
        """Record one step's max-abs action magnitude.

        Args:
            max_abs_action: ``max(abs(action))`` for this inference step.

        Returns:
            A warning string exactly once per fault -- on the step where the
            near-zero streak first reaches ``patience``, or on the first step
            whose magnitude is not finite -- and ``None`` otherwise. A single
            above-threshold step clears the near-zero streak and re-arms that
            warning; the two faults are tracked independently.

        Raises:
            TypeError: If ``max_abs_action`` is not a real number, as before.
        """
        if not math.isfinite(max_abs_action):
            # Neither motion nor near-zero: report the fault that was measured
            # and leave the near-zero streak untouched, so a stream that is
            # genuinely both still gets both warnings.
            if self._nonfinite_warned:
                return None
            self._nonfinite_warned = True
            return (
                f"Policy emitted a non-finite action (max abs = {max_abs_action:g}): the robot "
                f"will not move. This is not the near-zero case -- a non-finite action is refused "
                f"by the simulation and hardware backends rather than applied, so the obs_rename / "
                f"camera keys are not implicated. Check the checkpoint's normalization statistics "
                f"(a zero divisor yields inf/nan) and whether any observation value is itself "
                f"non-finite."
            )
        if max_abs_action >= self.threshold:
            self._streak = 0
            self._warned = False
            return None
        self._streak += 1
        if self._streak >= self.patience and not self._warned:
            self._warned = True
            return (
                f"Policy emitted near-zero actions (max abs < {self.threshold:g}) for "
                f"{self._streak} consecutive steps: the robot will not move. This usually "
                f"means the observation never reached the model -- check the embodiment's "
                f"obs_rename / camera keys and that observation.state is populated."
            )
        return None

    def reset(self) -> None:
        """Reset streak + warned state for both faults (call on episode reset)."""
        self._streak = 0
        self._warned = False
        self._nonfinite_warned = False


# Registered pipeline step: pack scalar joint obs -> observation.state


# Imported lazily so this module is importable without lerobot (e.g. for unit
# testing EmbodimentMap loading/validation in a minimal env).
# Warn-once dedup for a declared-vs-observed state_keys mismatch, keyed by the
# (missing keys, observed keys) pair so the message repeats at most once per
# distinct mismatch rather than at every tick of the 50Hz control loop.
_WARNED_STATE_KEY_MISMATCH: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()


def observed_state_keys(observation: Mapping[str, Any]) -> list[str]:
    """Observation keys that can carry a joint/state scalar, in observation order.

    Excludes ``task`` (the instruction string) and any array with 2+ dimensions
    (a camera frame). Both observation-to-batch paths derive their state-ordering
    fallback from this, and every state-key diagnostic quotes it back to the
    caller, so all of them must agree on what counts as a state key - which is
    why this is one function rather than a rule restated per call site.

    Args:
        observation: Raw strands/sim observation for this step.

    Returns:
        The candidate state keys, in the observation's own insertion order.
    """
    return [k for k, v in observation.items() if k != "task" and not (isinstance(v, np.ndarray) and v.ndim >= 2)]


def _warn_state_key_mismatch(missing: list[str], observation: Mapping[str, Any], *, total: bool) -> None:
    """Warn once that declared ``state_keys`` are absent from the observation.

    Reports the two degradations the declarative state path can hit with the
    same registry-checked remedy the generic ``robot_state_keys`` path uses
    (:func:`state_key_remedy`), because both answer the same caller question and
    a remedy invented per call site drifts from the one the registry can prove.

    The remedy is chosen from what the observation carries rather than from the
    declared keys, so it can never name an embodiment that would land back on
    this same mismatch - the reasoning :func:`state_key_remedy` documents.

    Args:
        missing: Declared ``state_keys`` absent from ``observation``, in
            declared order.
        observation: The observation being packed, read for the keys it does
            carry.
        total: Whether NO declared key was present. A total miss leaves the
            observation unpacked for the caller's own handling, so it is
            reported as an unbindable configuration rather than as a
            zero-filled dimension.
    """
    observed = observed_state_keys(observation)
    sig = (tuple(missing), tuple(observed))
    if sig in _WARNED_STATE_KEY_MISMATCH:
        return
    _WARNED_STATE_KEY_MISMATCH.add(sig)
    shown = missing[:_REMEDY_KEYS_INLINE_MAX]
    ellipsis = "..." if len(missing) > _REMEDY_KEYS_INLINE_MAX else ""
    if total:
        detail = (
            f"None of the {len(missing)} declared state_keys {shown}{ellipsis} are present in the "
            f"observation. Observed joint/state keys: {observed}. No observation.state was packed, "
            "so the model receives no proprioceptive input and the failure surfaces downstream. "
            "The embodiment's declared keys describe a different robot/sim - or a different naming "
            "convention for the same one - than the observation reporting them."
        )
    else:
        detail = (
            f"{len(missing)} declared state_keys are not present in the observation: "
            f"{shown}{ellipsis}. Observed joint/state keys: {observed}. Present joints keep their "
            "model index and the missing dims are zero-filled in place, but the sim/robot does not "
            "report those joints - commonly a mimic/tendon gripper actuator whose name differs from "
            "the observation's finger-joint names."
        )
    logger.warning("lerobot_local: %s %s", detail, state_key_remedy(observed))


def register_pack_state_step() -> type | None:
    """Define + register :class:`PackStateProcessorStep` against lerobot.

    Returns the step class, or ``None`` if lerobot's processor framework is
    unavailable. Idempotent: returns the already-registered class on re-call.
    """
    try:
        import torch
        from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
    except ImportError:
        logger.debug("lerobot processor framework unavailable; PackStateProcessorStep not registered")
        return None

    # Idempotent re-registration via the PUBLIC lookup. Reading the internal
    # ``ProcessorStepRegistry._registry`` dict couples us to a private attribute
    # LeRobot can rename/restructure any release (cf. TransitionKey moving
    # between 0.5.1 and 0.5.2). ``get(name)`` is the documented lookup; it
    # raises (KeyError/ValueError) or returns None when the name is unregistered
    # depending on the LeRobot version, so treat any miss as "not yet registered"
    # and fall through to the register decorator below.
    try:
        existing = ProcessorStepRegistry.get("strands_pack_state")
    except (KeyError, ValueError, AttributeError):
        existing = None
    if existing is not None:
        return existing

    @ProcessorStepRegistry.register(name="strands_pack_state")
    @dataclass
    class PackStateProcessorStep(ObservationProcessorStep):  # type: ignore[misc]
        """Compose declared scalar joint keys into ``observation.state``.

        Runs after the rename step and before normalization. If the observation
        already carries ``observation.state`` (e.g. a benchmark adapter or a
        natively-LeRobot obs), it passes through untouched (idempotent).

        Attributes:
            state_keys: Ordered robot/sim scalar keys composing the state vector.
            expected_dim: Model's declared ``observation.state`` dimension.
            dim_policy: ``"strict"`` | ``"pad"`` | ``"truncate"``.
            state_units: ``"native"`` (the default - pack the sim's own values)
                or ``"degrees"`` (convert the sim's radian joints to the model's
                training units before packing). Mirrors
                :attr:`EmbodimentMap.state_units`, which is where this step's
                value comes from.
            gripper_index: Column of the gripper inside ``state_keys``, which
                speaks ``RANGE_0_100`` rather than degrees. ``-1`` (the default)
                = no distinct gripper column.
            gripper_joint_range: The sim gripper joint's ``[min, max]`` radians,
                used to map that column onto 0..100. Empty (the default) =
                convert the gripper like an arm joint.
            joint_mids: Per-joint calibration mid-points in degrees, aligned to
                ``state_keys``, subtracted from the arm columns so the packed
                state is mid-centered like LeRobot's ``DEGREES`` mode. Empty
                (the default) = mid 0.
        """

        state_keys: list[str] = field(default_factory=list)
        expected_dim: int = 0
        dim_policy: str = "strict"
        # Sim->model unit conversion (see EmbodimentMap). "degrees" => the sim's
        # radian joints are converted to the model's training units (arm
        # degrees, gripper 0..100) before packing observation.state.
        state_units: str = "native"
        gripper_index: int = -1
        gripper_joint_range: list[float] = field(default_factory=list)
        # Per-joint calibration mid-points in DEGREES (aligned to state_keys);
        # subtracted from arm columns so observation.state is mid-centered like
        # lerobot motors_bus DEGREES mode. Empty = mid 0 (prior behavior).
        joint_mids: list[float] = field(default_factory=list)

        def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
            """Compose the declared scalar joint keys into ``observation.state`` (passthrough when already packed)."""
            if "observation.state" in observation:
                return observation  # already packed -> passthrough

            vals: list[float] = []
            missing: list[str] = []
            present = 0
            for k in self.state_keys:
                if k in observation:
                    present += 1
                    v = observation[k]
                    if isinstance(v, np.ndarray):
                        if v.ndim == 0:
                            vals.append(float(v))
                        else:
                            vals.extend(float(x) for x in v.ravel())
                    elif isinstance(v, (list, tuple)):
                        vals.extend(float(x) for x in v)
                    else:
                        vals.append(float(v))
                else:
                    # Zero-fill the missing key IN PLACE so it holds its slot.
                    # Skipping it (the previous behavior) collapsed the vector and
                    # shifted every following joint out of its model index -- the
                    # embodiment-path analog of the generic-path fix in
                    # LerobotLocalPolicy._collect_state_values.
                    vals.append(0.0)
                    missing.append(k)

            if present == 0:
                # None of the DECLARED state_keys are present. The canonical
                # trigger is a SIM embodiment (e.g. so101: state_keys ['1'..'6'])
                # driven from REAL hardware, whose observation carries
                # '<motor>.pos' scalar keys instead (lerobot SOFollower). Rather
                # than dead-end with "requires observation.state", fall back to
                # the hardware '.pos' keys so ``embodiment="so101"`` works on the
                # physical arm too - not only in sim.
                #
                # Hardware '.pos' values are ALREADY in the model's training
                # units (so_follower MotorNormMode: arm DEGREES, gripper
                # RANGE_0_100), so we pack them RAW and DO NOT apply the
                # sim-radian -> model-degree conversion below (that would
                # double-convert). Observation insertion order is lerobot motor
                # order (shoulder_pan..gripper), which matches the positional
                # sim state_keys, so a straight collection is index-aligned.
                pos_keys = hardware_pos_keys(observation)
                if len(pos_keys) >= len(self.state_keys) and self.state_keys:
                    n = len(self.state_keys)
                    hw_vals = [float(observation[k]) for k in pos_keys[:n]]
                    target = self.expected_dim or len(hw_vals)
                    hw_vals = reconcile_dim(hw_vals, target, self.dim_policy, label="observation.state")
                    out = {k: v for k, v in observation.items() if k not in pos_keys[:n]}
                    out["observation.state"] = torch.as_tensor(hw_vals, dtype=torch.float32)
                    return out
                # No declared state key AND no usable '.pos' fallback; leave obs
                # alone so a clearer downstream error (or a state-less policy)
                # can handle it, rather than emitting an all-zero state vector.
                #
                # Say so first. Returning silently made this the one degradation
                # on either state path that reported nothing: the generic path
                # names the observed keys and the remedy for both an all-missing
                # and a partly-missing binding, while here the caller learned
                # only that something downstream wanted observation.state - after
                # the weight download, and without the one fact that resolves it
                # (which embodiment DOES bind this observation).
                _warn_state_key_mismatch(list(self.state_keys), observation, total=True)
                return observation

            if missing:
                _warn_state_key_mismatch(missing, observation, total=False)

            # Convert sim units (radians + gripper joint range) to the model's
            # training units (arm degrees, gripper 0..100) BEFORE packing, so the
            # model conditions on state in the space it was trained on. No-op
            # unless state_units == "degrees". See so_follower.py MotorNormMode.
            if self.state_units == "degrees":
                vals = _convert_joint_vector(
                    vals,
                    to_model=True,
                    gripper_index=self.gripper_index,
                    gripper_joint_range=self.gripper_joint_range,
                    joint_mids=self.joint_mids,
                )

            target = self.expected_dim or len(vals)
            vals = reconcile_dim(vals, target, self.dim_policy, label="observation.state")

            out = {k: v for k, v in observation.items() if k not in self.state_keys}
            # Emit a 1-D float32 torch Tensor (not a numpy array) so the
            # pipeline's own AddBatchDimensionObservationStep batches it to
            # (1, D) exactly as it batches image tensors. A numpy state is left
            # unbatched by that step, which on the declarative path stranded
            # observation.state at (D,) while images were (1, C, H, W) -> a
            # torch.stack rank mismatch inside the model. A real LeRobot dataset
            # feeds observation.state as a tensor too, so this matches the
            # convention the downstream steps (normalizer, batcher) expect.
            out["observation.state"] = torch.as_tensor(vals, dtype=torch.float32)
            return out

        def get_config(self) -> dict[str, Any]:
            """Return the JSON-serializable config (``state_keys``, ``expected_dim``, ``dim_policy``) for checkpoint round-trip."""
            return {
                "state_keys": list(self.state_keys),
                "expected_dim": self.expected_dim,
                "dim_policy": self.dim_policy,
            }

        def transform_features(self, features):  # type: ignore[no-untyped-def]
            """Return ``features`` unchanged: packing reshapes only the runtime obs, not the model's declared feature set."""
            # State vector composition doesn't change the model's declared
            # feature set (the normalizer already knows observation.state);
            # we only reshape the runtime obs. Pass features through.
            return features

    return PackStateProcessorStep


# Embodiment map


@dataclass(frozen=True)
class EmbodimentMap:
    """Declarative robot/sim ↔ model key mapping. Built + validated once.

    Attributes:
        name: Config identifier.
        obs_rename: ``{robot_obs_key: model_feature_key}`` for cameras (and any
            other direct passthroughs), e.g.
            ``{"image": "observation.images.image"}``. Fed into LeRobot's
            ``RenameObservationsProcessorStep.rename_map``.
        state_keys: Ordered scalar robot keys composing ``observation.state``.
        action_keys: Ordered robot actuator names for the action tensor's
            index→name mapping (output side).
        dim_policy: ``"strict"`` | ``"pad"`` | ``"truncate"`` for state dim.
        state_units: Unit convention of the sim state vector this map packs:
            ``"native"`` (the default - no conversion) or ``"degrees"`` (arm
            columns in degrees, gripper column in ``RANGE_0_100``), which is
            what :meth:`sim_state_to_model` converts from.
        action_units: Unit convention of the model's action vector, same
            vocabulary as ``state_units``. On ``"degrees"``
            :meth:`model_action_to_sim` converts the model's degrees back to sim
            radians; left ``"native"`` for a degrees-trained SO-arm checkpoint,
            the raw degree values reach the sim unconverted and saturate its
            radian joint limits.
        gripper_index: Column of the gripper inside ``state_keys`` /
            ``action_keys``, which speaks ``RANGE_0_100`` rather than degrees.
            ``-1`` (the default) = no distinct gripper column; SO arms use ``5``.
        gripper_joint_range: The sim gripper joint's ``[min, max]`` radians, used
            to map that column's 0..100 command onto the joint and back. Empty
            (the default) = convert the gripper like an arm joint.
        joint_mids: Per-joint calibration mid-points in degrees, aligned to
            ``state_keys`` / ``action_keys``, because LeRobot's ``DEGREES`` mode
            is mid-point-centered. Empty (the default) = mid 0, i.e. sim
            ``qpos=0`` is assumed to be the calibration mid.
    """

    name: str = ""
    obs_rename: dict[str, str] = field(default_factory=dict)
    state_keys: list[str] = field(default_factory=list)
    action_keys: list[str] = field(default_factory=list)
    dim_policy: str = "strict"
    # Unit conventions for state/action vectors. The MuJoCo sim expresses
    # revolute joints in RADIANS, but LeRobot SO-arm checkpoints (so100/so101,
    # MolmoAct2 etc.) trained on data recorded with the driver's DEGREES mode
    # speak the driver's MotorNormMode: arm joints in DEGREES and the gripper in
    # RANGE_0_100. "native" = no conversion (the default; real-hardware *_real
    # maps already speak the driver units). "degrees" = arm columns are degrees
    # + the gripper column is 0..100; the policy converts deg<->rad and
    # 0..100<->the gripper joint range when packing state (model<-sim) and
    # emitting actions (model->sim). See so_follower.py.
    #
    # CAVEAT: the arm-in-DEGREES convention holds only for checkpoints recorded
    # with the SO driver's use_degrees=True. That is the lerobot so_follower /
    # so_leader DEFAULT (kept "for backward compatibility with previous
    # policies/dataset"), but it is opt-out: with use_degrees=False the arm uses
    # MotorNormMode.RANGE_M100_100 (-100..100), NOT degrees (so_follower.py:50
    # -> RANGE_M100_100). state_units/action_units="degrees" would mis-convert a
    # RANGE_M100_100-recorded checkpoint - leave those on "native".
    state_units: str = "native"
    action_units: str = "native"
    # Index of the gripper column in state_keys/action_keys (RANGE_0_100, not a
    # degree joint). -1 = no special gripper column. SO arms = 5 (the 6th key).
    gripper_index: int = -1
    # The sim gripper joint's [min, max] radians, used to map the model's
    # 0..100 gripper command onto the joint range (and back). Empty = treat the
    # gripper like an arm joint (deg<->rad). SO arms: [-0.175, 1.745].
    gripper_joint_range: list[float] = field(default_factory=list)
    # Per-joint calibration mid-points in DEGREES, aligned to state_keys /
    # action_keys. LeRobot's MotorNormMode.DEGREES is mid-point-centered: a
    # checkpoint conditions on (joint_angle - calibration_mid), not the absolute
    # angle (ground truth: lerobot/motors/motors_bus.py mid = (min + max) / 2).
    # The sim expresses absolute angles, so without the mid the packed
    # observation.state is offset per joint from the training distribution and
    # can fall outside the dataset MIN_MAX range after normalization -> OOD.
    # When set, the "degrees" conversion subtracts the mid (sim -> model) and
    # adds it back (model -> sim). The gripper column (gripper_index) is exempt
    # (RANGE_0_100). Empty (default) = mid 0, i.e. sim qpos=0 is assumed to be
    # the calibration mid (the prior absolute-degrees behavior).
    joint_mids: list[float] = field(default_factory=list)

    def validate(self, input_features: dict[str, Any], output_features: dict[str, Any]) -> None:
        """Fail-fast validation against the model's declared features.

        Args:
            input_features: ``config.input_features`` from the loaded policy.
            output_features: ``config.output_features`` from the loaded policy.

        Raises:
            ValueError: On any mismatch (unknown rename target, wrong state/action dim).
        """
        # 1. Every rename target must be a declared model input feature.
        for src, dst in self.obs_rename.items():
            if dst not in input_features:
                raise ValueError(
                    f"Embodiment '{self.name}': obs_rename {src!r}->{dst!r} targets a feature "
                    f"the model doesn't declare. Model input_features: {sorted(input_features)}"
                )

        # 2. State dim check (only when the model declares observation.state).
        state_feat = input_features.get("observation.state")
        if state_feat is not None and getattr(state_feat, "shape", None):
            sdim = state_feat.shape[0]
            if self.state_keys and self.dim_policy == "strict" and len(self.state_keys) != sdim:
                raise ValueError(
                    f"Embodiment '{self.name}': {len(self.state_keys)} state_keys but model "
                    f"expects observation.state dim {sdim}. Fix state_keys or set "
                    f"dim_policy='pad'/'truncate' to opt in to adaptation."
                )

        # 3. Action dim check (only when the model declares an action feature).
        action_feat = output_features.get("action")
        if action_feat is not None and getattr(action_feat, "shape", None) and self.action_keys:
            adim = action_feat.shape[0]
            if len(self.action_keys) != adim:
                raise ValueError(
                    f"Embodiment '{self.name}': {len(self.action_keys)} action_keys but model "
                    f"action dim is {adim}. Action mapping would mis-index."
                )

    def _convert_vector(self, values: list[float], *, to_model: bool) -> list[float]:
        """Convert an ordered joint vector between sim (radians / sim units) and
        the model's training units (degrees + gripper RANGE_0_100).

        Applies only when ``units == "degrees"``; otherwise returns ``values``
        unchanged. Direction:

        * ``to_model=True``  sim -> model: arm radians -> degrees; gripper joint
          radians -> 0..100.
        * ``to_model=False`` model -> sim: arm degrees -> radians; gripper 0..100
          -> joint radians.

        The gripper column (``gripper_index``) is mapped against
        ``gripper_joint_range`` because the SO-arm gripper uses
        ``MotorNormMode.RANGE_0_100`` (0..100), not degrees - see
        ``lerobot/robots/so_follower/so_follower.py``.

        Args:
            values: Ordered joint values (length matches state_keys/action_keys).
            to_model: Conversion direction (see above).

        Returns:
            A new list of converted values (input is not mutated).
        """
        return _convert_joint_vector(
            values,
            to_model=to_model,
            gripper_index=self.gripper_index,
            gripper_joint_range=self.gripper_joint_range,
            joint_mids=self.joint_mids,
        )

    def sim_state_to_model(self, values: list[float]) -> list[float]:
        """Convert a sim state vector into the model's training units.

        No-op unless ``state_units == "degrees"``.
        """
        if self.state_units != "degrees":
            return list(values)
        return self._convert_vector(values, to_model=True)

    def model_action_to_sim(self, values: list[float]) -> list[float]:
        """Convert a model action vector into sim (radian) units.

        No-op unless ``action_units == "degrees"``.
        """
        if self.action_units != "degrees":
            return list(values)
        return self._convert_vector(values, to_model=False)

    def expected_state_dim(self, input_features: dict[str, Any]) -> int:
        """Return the model's declared state dim, or len(state_keys) if absent."""
        state_feat = input_features.get("observation.state")
        if state_feat is not None and getattr(state_feat, "shape", None):
            return state_feat.shape[0]
        return len(self.state_keys)


# JSON registry loader (with _extends inheritance + aliases)

_CONFIG_FILE = Path(__file__).parent / "embodiments.json"


def _resolve(name: str, definitions: dict) -> EmbodimentMap:
    """Resolve a definition name to an :class:`EmbodimentMap`, following ``_extends``.

    Keys beginning with a double underscore (e.g. ``__note__``, ``__doc__``) are
    treated as human-facing documentation/metadata and are stripped before
    constructing the dataclass, so the JSON can carry inline provenance notes
    (ground-truth source per robot) without breaking the loader.
    """
    definition = definitions[name]
    if "_extends" in definition:
        parent = _resolve(definition["_extends"], definitions)
        merged: dict[str, Any] = {
            "obs_rename": dict(parent.obs_rename),
            "state_keys": list(parent.state_keys),
            "action_keys": list(parent.action_keys),
            "dim_policy": parent.dim_policy,
        }
        for k, v in definition.items():
            if k != "_extends" and not k.startswith("__"):
                merged[k] = v
    else:
        merged = {k: v for k, v in definition.items() if not k.startswith("__")}
    merged["name"] = name
    return EmbodimentMap(**merged)


def _load_defs() -> tuple[dict, dict]:
    if not _CONFIG_FILE.exists():
        return {}, {}
    with open(_CONFIG_FILE) as fh:
        raw = json.load(fh)
    return raw.get("configs", {}), raw.get("aliases", {})


EMBODIMENT_MAP: dict[str, EmbodimentMap] = {}
_defs, _aliases = _load_defs()
for _cfg_name in _defs:
    EMBODIMENT_MAP[_cfg_name] = _resolve(_cfg_name, _defs)
# Configuration names only. EMBODIMENT_MAP also holds every alias pointing at
# the same object, so a diagnostic listing candidates must filter to these or it
# offers the caller several spellings of one configuration.
_CONFIG_NAMES: frozenset[str] = frozenset(_defs)
for _alias, _target in _aliases.items():
    if _target in EMBODIMENT_MAP:
        EMBODIMENT_MAP[_alias] = EMBODIMENT_MAP[_target]
del _defs, _aliases


def load_embodiment(embodiment: str | EmbodimentMap | dict) -> EmbodimentMap:
    """Load an embodiment map by name, dict, or pass through an instance.

    Args:
        embodiment: Registry name (e.g. ``"panda_libero"``), an inline dict
            (``{"obs_rename": ..., "state_keys": ...}``), or an
            :class:`EmbodimentMap`.

    Returns:
        Resolved :class:`EmbodimentMap`.

    Raises:
        ValueError: If a string name is unknown.
    """
    if isinstance(embodiment, EmbodimentMap):
        return embodiment
    if isinstance(embodiment, dict):
        data = dict(embodiment)
        data.setdefault("name", "<inline>")
        return EmbodimentMap(**data)
    if isinstance(embodiment, str):
        if embodiment in EMBODIMENT_MAP:
            return EMBODIMENT_MAP[embodiment]
        raise ValueError(f"Unknown embodiment '{embodiment}'. Available: {sorted(EMBODIMENT_MAP)}")
    raise ValueError(f"embodiment must be str | dict | EmbodimentMap, got {type(embodiment)}")


__all__ = [
    "EmbodimentMap",
    "EMBODIMENT_MAP",
    "ZeroActionMonitor",
    "diagnose_action_dim",
    "load_embodiment",
    "matching_embodiments",
    "observed_state_keys",
    "reconcile_dim",
    "register_pack_state_step",
    "state_key_remedy",
]
